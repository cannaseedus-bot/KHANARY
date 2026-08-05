"""oss_distillation.py — GPT-OSS teacher -> from_zero LoRA distillation

Uses kuhul_engine HTTP API (port 17474) as the GPT-OSS teacher.
Trains low-rank adapter weights (LoRA) on from_zero_v0.6 to match teacher outputs.
No PEFT / HuggingFace transformers required — pure PyTorch + safetensors.

Strategy: response distillation
  1. Send prompts to GPT-OSS teacher -> get completion text
  2. Tokenize the (prompt + completion) as a sequence
  3. Run from_zero student forward pass
  4. Cross-entropy loss on the completion tokens (prompt tokens masked)
  5. Backprop through LoRA adapter params only
  6. Save LoRA.safetensors

LoRA: adds W_delta = A @ B to each frozen projection matrix.
  - A: [r, in_dim]  initialized N(0, 0.02/r)
  - B: [out_dim, r] initialized zeros
  - W_eff = W + B @ A  (scale omitted; alpha=r gives unit scaling)
  - Only A, B are in optimizer; W stays frozen

Usage:
    python tools/oss_distillation.py \
        --student  models/from_zero/from_zero_v0.6_merged.safetensors \
        --out      models/from_zero/from_zero_v0.6_lora.safetensors \
        --rank     8 \
        --steps    500 \
        --lr       1e-4 \
        --engine   http://127.0.0.1:17474 \
        --prompts  tools/distill_prompts.txt

If --engine is unreachable, the script falls back to self-distillation
(student teaches itself — useful for adapter shape validation without the engine running).
"""

import argparse
import json
import math
import os
import urllib.request
import urllib.error
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# GPT-2 config (small / medium — auto-detected from wte shape)
# ---------------------------------------------------------------------------
GPT2_SMALL  = dict(n_layer=6,  n_head=6,  n_embd=768,  vocab_size=50270, block_size=1024)
GPT2_MEDIUM = dict(n_layer=12, n_head=12, n_embd=768,  vocab_size=50270, block_size=1024)

# ---------------------------------------------------------------------------
# Weight shape map — flat storage → proper 2D shapes
# ---------------------------------------------------------------------------
def _gpt2_shapes(cfg):
    E, H, L = cfg['n_embd'], cfg['n_head'], cfg['n_layer']
    V, S    = cfg['vocab_size'], cfg['block_size']
    head    = E // H
    shapes  = {}
    shapes['transformer.wte.weight']  = [V, E]
    shapes['transformer.wpe.weight']  = [S, E]
    shapes['transformer.ln_f.weight'] = [E]
    shapes['transformer.ln_f.bias']   = [E]
    for i in range(L):
        p = f'transformer.h.{i}'
        shapes[f'{p}.ln_1.weight']           = [E]
        shapes[f'{p}.ln_1.bias']             = [E]
        shapes[f'{p}.ln_2.weight']           = [E]
        shapes[f'{p}.ln_2.bias']             = [E]
        shapes[f'{p}.attn.c_attn.weight']    = [E, 3*E]
        shapes[f'{p}.attn.c_attn.bias']      = [3*E]
        shapes[f'{p}.attn.c_proj.weight']    = [E, E]
        shapes[f'{p}.attn.c_proj.bias']      = [E]
        shapes[f'{p}.mlp.c_fc.weight']       = [E, 4*E]
        shapes[f'{p}.mlp.c_fc.bias']         = [4*E]
        shapes[f'{p}.mlp.c_proj.weight']     = [4*E, E]
        shapes[f'{p}.mlp.c_proj.bias']       = [E]
    return shapes

# ---------------------------------------------------------------------------
# Load safetensors into reshaped tensors
# ---------------------------------------------------------------------------
def load_model(path: str, cfg: dict) -> dict:
    shapes = _gpt2_shapes(cfg)
    state  = {}
    with safe_open(path, framework='pt') as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k in shapes:
                expected_numel = 1
                for d in shapes[k]:
                    expected_numel *= d
                if t.numel() == expected_numel and list(t.shape) != shapes[k]:
                    t = t.reshape(shapes[k])
            state[k] = t
    print(f'[student] loaded {len(state)} tensors from {path}')
    return state

# ---------------------------------------------------------------------------
# Minimal GPT-2 forward pass (returns logits, no KV-cache)
# ---------------------------------------------------------------------------
def gpt2_forward(state: dict, lora_deltas: dict, input_ids: torch.Tensor, cfg: dict):
    B, T     = input_ids.shape
    E        = cfg['n_embd']
    n_head   = cfg['n_head']
    n_layer  = cfg['n_layer']
    head_dim = E // n_head

    def get(key):
        base = state[key].float()
        if key in lora_deltas:
            A, B_ = lora_deltas[key]
            base = base + (B_.float() @ A.float())
        return base

    # Embeddings
    wte = get('transformer.wte.weight')
    wpe = get('transformer.wpe.weight')
    pos = torch.arange(T, device=input_ids.device)
    x = wte[input_ids] + wpe[pos].unsqueeze(0)  # [B, T, E]

    for i in range(n_layer):
        p = f'transformer.h.{i}'

        # LayerNorm 1
        ln1_w = get(f'{p}.ln_1.weight')
        ln1_b = get(f'{p}.ln_1.bias')
        x_ln  = F.layer_norm(x, [E], ln1_w, ln1_b)

        # Self-attention
        W_qkv = get(f'{p}.attn.c_attn.weight')   # [E, 3E]
        b_qkv = get(f'{p}.attn.c_attn.bias')     # [3E]
        qkv   = x_ln @ W_qkv + b_qkv             # [B, T, 3E]
        q, k, v = qkv.chunk(3, dim=-1)            # [B, T, E] each

        # Reshape to heads
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale
        # Causal mask
        mask  = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att   = att.masked_fill(mask == 0, float('-inf'))
        att   = F.softmax(att, dim=-1)

        attn_out = att @ v                                          # [B, nh, T, hd]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)

        W_proj = get(f'{p}.attn.c_proj.weight')
        b_proj = get(f'{p}.attn.c_proj.bias')
        attn_out = attn_out @ W_proj + b_proj
        x = x + attn_out

        # LayerNorm 2
        ln2_w = get(f'{p}.ln_2.weight')
        ln2_b = get(f'{p}.ln_2.bias')
        x_ln2 = F.layer_norm(x, [E], ln2_w, ln2_b)

        # MLP
        W_fc  = get(f'{p}.mlp.c_fc.weight')
        b_fc  = get(f'{p}.mlp.c_fc.bias')
        W_cp  = get(f'{p}.mlp.c_proj.weight')
        b_cp  = get(f'{p}.mlp.c_proj.bias')
        h_mlp = x_ln2 @ W_fc + b_fc
        h_mlp = h_mlp * torch.sigmoid(1.702 * h_mlp)   # GELU approx
        h_mlp = h_mlp @ W_cp + b_cp
        x     = x + h_mlp

    # Final LN + LM head
    lnf_w  = get('transformer.ln_f.weight')
    lnf_b  = get('transformer.ln_f.bias')
    x      = F.layer_norm(x, [E], lnf_w, lnf_b)
    lm_w   = state.get('transformer.lm_head.weight', state['transformer.wte.weight'])
    logits = x @ lm_w.float().T                  # [B, T, V]
    return logits

# ---------------------------------------------------------------------------
# LoRA adapter factory
# ---------------------------------------------------------------------------
def make_lora_adapters(state: dict, rank: int, lora_keys: list) -> dict:
    adapters = {}
    for key in lora_keys:
        if key not in state:
            continue
        W = state[key]
        if W.dim() < 2:
            continue
        out_dim, in_dim = W.shape[0], W.shape[1]
        A = nn.Parameter(torch.randn(rank, in_dim) * (0.02 / rank))
        B = nn.Parameter(torch.zeros(out_dim, rank))
        adapters[key] = (A, B)
    return adapters

# ---------------------------------------------------------------------------
# GPT-2 tokenizer (byte-pair encoding via tiktoken if available, else simple)
# ---------------------------------------------------------------------------
def get_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        return lambda text: enc.encode(text)
    except ImportError:
        pass
    # Fallback: very naive character-level (won't give GPT-2 tokens but allows testing)
    def char_encode(text):
        return [ord(c) % 50270 for c in text]
    print('[tokenizer] tiktoken not found, using char fallback — install with: pip install tiktoken')
    return char_encode

# ---------------------------------------------------------------------------
# Teacher: call kuhul_engine (GPT-OSS) for completion
# ---------------------------------------------------------------------------
def teacher_complete(engine_base: str, prompt: str, max_tokens: int = 256) -> str:
    payload = json.dumps({
        'model': 'gpt-oss-20b-MXFP4.gguf',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.7,
        'stream': False
    }).encode()
    req = urllib.request.Request(
        f'{engine_base}/v1/chat/completions',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data['choices'][0]['message']['content']
    except Exception as e:
        return None

# ---------------------------------------------------------------------------
# Default prompts (kuhul domain)
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Explain the K'UHUL fold system and how it organizes semantic execution phases.",
    "Describe the role of the BOSS orchestrator in the WebX compute engine.",
    "How does the micronaut factory create sampling-profile micronauts?",
    "What is the difference between a fold micronaut and a chat micronaut?",
    "Explain the π-nary arc weighting in the KuhulPhysics gradient controller.",
    "How does the DirectML GEMM bridge accelerate matrix operations on Intel HD 4600?",
    "What is SCXQ2 IR and how does it represent the K'UHUL execution graph?",
    "Describe the KUHUL APPS app generation studio and its kuhul_engine backend.",
    "How do semantic micronauts influence llama-server sampling parameters?",
    "Explain the SLERP merge strategy for combining two training phase checkpoints.",
    "What are atomic blocks in the K'UHUL semantic field system?",
    "How does the grammar validator use kuhul.ebnf for constrained generation?",
    "Describe the LoRA distillation pipeline from GPT-OSS to from_zero.",
    "What is the role of the MicrosoftSDK.ps1 in the NNC-K stack?",
    "How does the Phase Pop-Wo-Sek-Chen-Xul loop work in K'UHUL execution?",
]

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    # Auto-detect config from actual weight keys + wte shape
    with safe_open(args.student, framework='pt') as f:
        keys = list(f.keys())
        wte  = f.get_tensor('transformer.wte.weight')
    n_embd  = wte.shape[1] if wte.dim() == 2 else 768
    vocab_size = wte.shape[0] if wte.dim() == 2 else 50270
    layer_idxs = set()
    for k in keys:
        parts = k.split('.')
        if len(parts) > 2 and parts[0] == 'transformer' and parts[1] == 'h' and parts[2].isdigit():
            layer_idxs.add(int(parts[2]))
    n_layer = max(layer_idxs) + 1 if layer_idxs else 12
    n_head  = n_embd // 64
    cfg = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, vocab_size=vocab_size, block_size=1024)
    print(f'[config] n_layer={n_layer} n_embd={n_embd} n_head={n_head} vocab={vocab_size}')

    # Load student
    state = load_model(args.student, cfg)

    # LoRA targets: c_attn and c_proj projection weights for every layer
    lora_keys = []
    for i in range(cfg['n_layer']):
        p = f'transformer.h.{i}'
        lora_keys += [
            f'{p}.attn.c_attn.weight',
            f'{p}.attn.c_proj.weight',
            f'{p}.mlp.c_fc.weight',
            f'{p}.mlp.c_proj.weight',
        ]

    adapters = make_lora_adapters(state, args.rank, lora_keys)
    params   = [p for ab in adapters.values() for p in ab]
    optim    = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    tokenize = get_tokenizer()

    # Load prompts
    prompts = DEFAULT_PROMPTS[:]
    if args.prompts and os.path.isfile(args.prompts):
        with open(args.prompts) as fh:
            prompts = [l.strip() for l in fh if l.strip()]
        print(f'[prompts] loaded {len(prompts)} from {args.prompts}')
    else:
        print(f'[prompts] using {len(prompts)} built-in kuhul prompts')

    # Check engine
    engine_ok = False
    if args.engine:
        test = teacher_complete(args.engine, 'ping', max_tokens=4)
        engine_ok = test is not None
    print(f'[engine] {args.engine} -> {"OK" if engine_ok else "unreachable, using self-distillation"}')

    best_loss = float('inf')
    step      = 0

    while step < args.steps:
        prompt = prompts[step % len(prompts)]

        # Get teacher completion
        if engine_ok:
            completion = teacher_complete(args.engine, prompt, max_tokens=args.teacher_tokens)
            if completion is None:
                engine_ok = False
                completion = prompt   # self-distill fallback
        else:
            completion = prompt

        # Tokenize
        full_text  = prompt + '\n' + completion
        token_ids  = tokenize(full_text)[:cfg['block_size']]
        prompt_len = len(tokenize(prompt))

        if len(token_ids) < 4:
            step += 1
            continue

        input_ids  = torch.tensor([token_ids[:-1]], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:],    dtype=torch.long)

        # Forward pass (student)
        logits = gpt2_forward(state, adapters, input_ids, cfg)   # [1, T-1, V]
        logits = logits[0]                                         # [T-1, V]

        # Mask prompt tokens (only compute loss on completion)
        mask_start = max(0, prompt_len - 1)
        if mask_start >= len(target_ids):
            mask_start = 0

        loss_logits = logits[mask_start:]
        loss_targets = target_ids[mask_start:]

        loss = F.cross_entropy(
            loss_logits.float(),
            loss_targets.to(logits.device),
            ignore_index=-1
        )

        optim.zero_grad()
        loss.backward()

        # Gradient clip
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()

        step += 1

        if step % 10 == 0:
            print(f'  step {step:4d}/{args.steps}  loss={loss.item():.4f}  '
                  f'engine={"GPT-OSS" if engine_ok else "self"}')
        if loss.item() < best_loss:
            best_loss = loss.item()

    print(f'[distill] done. best_loss={best_loss:.4f}')

    # Save LoRA weights
    lora_state = {}
    for key, (A, B_) in adapters.items():
        safe_key = key.replace('.', '__')
        lora_state[f'lora_A__{safe_key}'] = A.detach().cpu().float()
        lora_state[f'lora_B__{safe_key}'] = B_.detach().cpu().float()

    out_path = args.out
    save_file(lora_state, out_path)
    size_mb = os.path.getsize(out_path) // (1024 * 1024)
    print(f'[distill] LoRA saved: {out_path}  ({size_mb} MB, {len(lora_state)} tensors)')

    # Verify
    with safe_open(out_path, framework='pt') as f:
        keys = list(f.keys())
    print(f'[verify] {len(keys)} LoRA tensors, e.g.: {keys[:2]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student',       default='models/from_zero/from_zero_v0.6_merged.safetensors')
    ap.add_argument('--out',           default='models/from_zero/from_zero_v0.6_lora.safetensors')
    ap.add_argument('--rank',    type=int,   default=8,     help='LoRA rank (default 8)')
    ap.add_argument('--steps',   type=int,   default=500,   help='Training steps')
    ap.add_argument('--lr',      type=float, default=1e-4,  help='Learning rate')
    ap.add_argument('--engine',  default='http://127.0.0.1:17474', help='kuhul_engine base URL')
    ap.add_argument('--prompts', default=None, help='Path to prompts file (one per line)')
    ap.add_argument('--teacher-tokens', type=int, default=200, help='Max tokens from teacher per step')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
