#!/usr/bin/env python3
# finetune_hf.py -- correct, CPU-resident GPT-2 finetune on the transition dataset, using
# HuggingFace GPT2LMHeadModel (a KNOWN-GOOD forward pass -- unlike dds_trainer, whose custom
# reconstruction produced a 1.8M near-random model with loss pinned at ln(vocab)).
#
# Full finetune (no LoRA/peft needed). Runs on CPU in system RAM, so the ~2GB GPU budget that
# blocks the D3D11 trainers does NOT apply: mini(124M)~2GB, medium(355M)~5.7GB of the 15.9GB.
# Loads a local GPT-2 .safetensors (any h.N / transformer.h.N naming, small or medium) by
# detecting dims + remapping keys into GPT2LMHeadModel. Saves HF-format safetensors that
# tools/gpt2_safetensors_to_gguf.py converts for khanary-server.
#
# Usage: python tools/finetune_hf.py --base <in.safetensors> --data <transitions.jsonl>
#          --out <dir> [--epochs 1] [--lr 5e-5] [--batch 4] [--seq 128] [--limit N] [--steps N]

import argparse, json, struct, math, os, time
import torch
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

def detect_and_load(path):
    """Read a GPT-2 safetensors, detect dims, remap keys -> GPT2LMHeadModel state_dict."""
    from safetensors.torch import load_file
    sd = load_file(path)
    norm = lambda k: k[len("transformer."):] if k.startswith("transformer.") else k
    keys = {norm(k): k for k in sd}
    wte = keys.get("wte.weight")
    n_embd = sd[wte].shape[1]
    vocab  = sd[wte].shape[0]
    n_layer = max(int(k.split(".")[1]) for k in keys if k.startswith("h.")) + 1
    n_ctx = sd[keys["wpe.weight"]].shape[0]
    cfg = GPT2Config(n_layer=n_layer, n_embd=n_embd, n_head=n_embd // 64,
                     vocab_size=vocab, n_positions=n_ctx, n_ctx=n_ctx)
    print(f"[base] {os.path.basename(path)}: n_layer={n_layer} n_embd={n_embd} "
          f"n_head={n_embd//64} vocab={vocab}  (~{sum(v.numel() for v in sd.values())/1e6:.0f}M)")
    model = GPT2LMHeadModel(cfg)
    # remap: strip/keep transformer. prefix; skip HF mask buffers; tie lm_head<-wte
    tgt = {}
    for k, kk in keys.items():
        if k.endswith("attn.bias") or k.endswith("attn.masked_bias"): continue
        tgt["transformer." + k] = sd[kk]
    tgt["lm_head.weight"] = sd[wte]
    missing, unexpected = model.load_state_dict(tgt, strict=False)
    miss = [m for m in missing if not (m.endswith("attn.bias") or m.endswith("attn.masked_bias"))]
    if miss: print(f"[warn] missing keys: {miss[:4]}{'...' if len(miss)>4 else ''}")
    return model, cfg

def load_batches(jsonl, tok, seq, limit, field="text"):
    seqs, n = [], 0
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            t = r.get(field)
            if not t: continue
            ids = tok.encode(t)[: seq]
            if len(ids) < 8: continue
            ids = ids + [tok.eos_token_id] * (seq - len(ids))
            seqs.append(ids); n += 1
            if limit and n >= limit: break
    print(f"[data] {n} sequences x {seq} tokens")
    return torch.tensor(seqs, dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (overnight safety)")
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    print(f"[cfg] cpu threads={torch.get_num_threads()} lr={a.lr} batch={a.batch} seq={a.seq}")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model, cfg = detect_and_load(a.base)
    model.train()
    data = load_batches(a.data, tok, a.seq, a.limit)
    opt = AdamW(model.parameters(), lr=a.lr)

    nb = (len(data) + a.batch - 1) // a.batch
    step = 0; t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(data))
        for i in range(0, len(data), a.batch):
            ids = data[perm[i:i+a.batch]]
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); step += 1
            if step % 10 == 0:
                dt = time.time() - t0
                print(f"  ep{ep+1} step {step}/{nb*a.epochs} loss {out.loss.item():.4f}  "
                      f"({step/dt:.2f} it/s)", flush=True)
            if a.save_every and step % a.save_every == 0:
                model.save_pretrained(a.out, safe_serialization=True)
                print(f"  [ckpt] step {step} -> {a.out}", flush=True)
            if a.steps and step >= a.steps: break
        if a.steps and step >= a.steps: break

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    print(f"[ok] saved finetuned model -> {a.out}")

if __name__ == "__main__":
    main()
