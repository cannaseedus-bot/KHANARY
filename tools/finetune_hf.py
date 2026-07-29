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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D
sys_path = os.path.dirname(os.path.abspath(__file__))
if sys_path not in __import__("sys").path: __import__("sys").path.insert(0, sys_path)

# --- manual LoRA (no peft): freeze base, train rank-r adapters, merge back for serving ---
class LoRAConv1D(nn.Module):
    """Wraps a frozen GPT-2 Conv1D (weight [nx,nf]) with a rank-r adapter: y = base(x) + (x@A@B)*s."""
    def __init__(self, base: Conv1D, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        nx, nf = base.weight.shape          # in, out
        self.A = nn.Parameter(torch.zeros(nx, r))
        self.B = nn.Parameter(torch.zeros(r, nf))
        nn.init.normal_(self.A, std=0.02)   # B=0 -> initial delta is exactly 0 (safe start)
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale

# GPT-2 attention/MLP projections are Conv1D; these are the standard LoRA targets.
LORA_TARGETS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")

def apply_lora(model, r, alpha):
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Conv1D) and any(name.endswith(t) for t in LORA_TARGETS):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1], LoRAConv1D(mod, r, alpha)); n += 1
    for p in model.parameters():
        pass  # base already frozen inside LoRAConv1D; embeddings/norms stay frozen below
    for name, p in model.named_parameters():
        p.requires_grad = (".A" in name or ".B" in name)  # only adapters train
    return n

def save_servable(model, cfg, out):
    """Write a STANDARD GPT-2 safetensors (LoRA merged into base non-destructively) that
    tools/gpt2_safetensors_to_gguf.py can convert. Safe to call mid-run for checkpoints."""
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    sd = {}
    lora_names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, LoRAConv1D):
            lora_names.add(name)
            sd[name + ".weight"] = mod.base.weight.data + (mod.A.data @ mod.B.data) * mod.scale  # fresh tensor
            sd[name + ".bias"]   = mod.base.bias.data
    for k, v in model.state_dict().items():
        if k == "lm_head.weight": continue        # tied to wte; converter derives output from wte -> drop (no shared-storage copy)
        if any(k.startswith(n + ".") for n in lora_names): continue   # replaced above
        sd[k] = v
    sd = {k: v.detach().contiguous() for k, v in sd.items()}          # cheap: no copy if already contiguous
    save_file(sd, os.path.join(out, "model.safetensors"))
    cfg.save_pretrained(out)

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

def field_endorsed_ids(tfield, preserve, tok, topk=12):
    """The Trinity field's steer for this example: token ids of the Delta-concepts it endorses
    given the Preserve-state -> these targets get upweighted in the loss."""
    ids = set()
    for concept, _w in tfield.guidance(preserve, topk):
        ids.update(tok.encode(" " + concept))
    return ids

def load_batches(jsonl, tok, seq, limit, tfield=None, textfield="text"):
    seqs, endorsed, n = [], [], 0
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            t = r.get(textfield)
            if not t: continue
            ids = tok.encode(t)[: seq]
            if len(ids) < 8: continue
            ids = ids + [tok.eos_token_id] * (seq - len(ids))
            seqs.append(ids); n += 1
            endorsed.append(field_endorsed_ids(tfield, r.get("preserve", []), tok) if tfield else None)
            if limit and n >= limit: break
    print(f"[data] {n} sequences x {seq} tokens" + ("  (+Trinity field guidance)" if tfield else ""))
    return torch.tensor(seqs, dtype=torch.long), endorsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (overnight safety)")
    ap.add_argument("--lora", action="store_true", help="LoRA: freeze base, train rank-r adapters (fits >full-finetune ceiling)")
    ap.add_argument("--lora-rank", type=int, default=8); ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--field", default="", help="Trinity field.json: upweight CE on field-endorsed tokens (guided finetune)")
    ap.add_argument("--field-weight", type=float, default=2.0, help="loss weight for field-endorsed target tokens")
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    if a.seed: torch.manual_seed(a.seed)   # fair A/B: same init + same batch order
    print(f"[cfg] cpu threads={torch.get_num_threads()} lr={a.lr} batch={a.batch} seq={a.seq}")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model, cfg = detect_and_load(a.base)
    if a.lora:
        n = apply_lora(model, a.lora_rank, a.lora_alpha)
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"[lora] rank={a.lora_rank} alpha={a.lora_alpha} wrapped {n} Conv1D -> "
              f"{tr/1e6:.2f}M trainable / {tot/1e6:.0f}M ({100*tr/tot:.2f}%)")
    tfield = None
    if a.field:
        from trinity_field import TrinityField
        tfield = TrinityField().load(a.field)
        print(f"[field] Trinity guidance {a.field} {tfield.meta}  (endorsed-token weight={a.field_weight})")
    model.train()
    data, endorsed = load_batches(a.data, tok, a.seq, a.limit, tfield)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    nb = (len(data) + a.batch - 1) // a.batch
    step = 0; t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(data))
        for i in range(0, len(data), a.batch):
            bi = perm[i:i+a.batch]
            ids = data[bi]
            if tfield is None:
                loss = model(input_ids=ids, labels=ids).loss
            else:
                # field-guided: per-token CE, upweighted where the target token is field-endorsed
                logits = model(input_ids=ids).logits
                sl, lbl = logits[:, :-1, :], ids[:, 1:]
                ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1),
                                     reduction="none").view(lbl.shape)
                w = torch.ones_like(ce)
                for b, ridx in enumerate(bi.tolist()):
                    es = endorsed[ridx]
                    if es:
                        mask = torch.tensor([tid in es for tid in lbl[b].tolist()])
                        w[b][mask] = a.field_weight
                loss = (ce * w).sum() / w.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); step += 1
            if step % 10 == 0:
                dt = time.time() - t0
                print(f"  ep{ep+1} step {step}/{nb*a.epochs} loss {loss.item():.4f}  "
                      f"({step/dt:.2f} it/s)", flush=True)
            if a.save_every and step % a.save_every == 0:
                save_servable(model, cfg, a.out)
                print(f"  [ckpt] step {step} -> {a.out}", flush=True)
            if a.steps and step >= a.steps: break
        if a.steps and step >= a.steps: break

    save_servable(model, cfg, a.out)
    print(f"[ok] saved finetuned model -> {a.out}")

if __name__ == "__main__":
    main()
