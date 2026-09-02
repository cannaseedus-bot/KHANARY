#!/usr/bin/env python3
"""
gguf_to_safetensors.py -- general GGUF -> safetensors converter using gguf-py.

Supports GPT-2 and Phi-2 (e.g. dolphin-2_6-phi-2). Dequantizes any quant type
(F32/F16/BF16/Q4/Q5/Q8...) via gguf-py's quants, maps GGUF tensor names to HF
safetensors names, and writes a .safetensors.

Usage:
  python tools/gguf_to_safetensors.py <in.gguf> <out.safetensors> [--out-f32] [--expand-vocab N]

  --expand-vocab N   Expand wte and lm_head embedding rows to exactly N tokens.
                     New rows are small-random init (stddev=0.02) so KUHUL special
                     tokens (50260-50281) can be looked up without OOB crash.
                     Use N=50282 for the full KUHUL vocab.
"""
import sys, json
from pathlib import Path
import numpy as np

GGUF_PY = Path(__file__).resolve().parent.parent / "khanary-llama-build" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))

from gguf.gguf_reader import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize
from safetensors.torch import save_file
import torch


# ── GGUF -> HF name maps (per architecture) ───────────────────────────────────
def map_name(arch, name):
    if arch in ("gpt2",):
        base = {
            # trainer expects transformer.* prefix on top-level tensors
            "token_embd.weight":   "transformer.wte.weight",
            "position_embd.weight":"transformer.wpe.weight",
            "output_norm.weight":  "transformer.ln_f.weight",
            "output_norm.bias":    "transformer.ln_f.bias",
            "output.weight":       "lm_head.weight",
        }
        if name in base:
            return base[name]
        p = name.split(".")
        if p[0] == "blk" and len(p) >= 3:
            i, s = p[1], ".".join(p[2:])
            m = {
                "attn_norm.weight": "ln_1.weight", "attn_norm.bias": "ln_1.bias",
                "attn_qkv.weight": "attn.c_attn.weight", "attn_qkv.bias": "attn.c_attn.bias",
                "attn_output.weight": "attn.c_proj.weight", "attn_output.bias": "attn.c_proj.bias",
                "ffn_norm.weight": "ln_2.weight", "ffn_norm.bias": "ln_2.bias",
                "ffn_up.weight": "mlp.c_fc.weight", "ffn_up.bias": "mlp.c_fc.bias",
                "ffn_down.weight": "mlp.c_proj.weight", "ffn_down.bias": "mlp.c_proj.bias",
            }
            if s in m:
                return f"transformer.h.{i}.{m[s]}"
    elif arch in ("phi2", "phi-2"):
        base = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.final_layernorm.weight",
            "output_norm.bias": "model.final_layernorm.bias",
        }
        if name == "output.weight":
            return "lm_head.weight"
        if name in base:
            return base[name]
        p = name.split(".")
        if p[0] == "blk" and len(p) >= 3:
            i, s = p[1], ".".join(p[2:])
            L = f"model.layers.{i}."
            m = {
                "attn_norm.weight": L + "input_layernorm.weight",
                "attn_norm.bias": L + "input_layernorm.bias",
                "attn_qkv.weight": L + "self_attn.qkv_proj.weight",
                "attn_qkv.bias": L + "self_attn.qkv_proj.bias",
                "attn_output.weight": L + "self_attn.dense.weight",
                "attn_output.bias": L + "self_attn.dense.bias",
                "ffn_norm.weight": L + "post_attention_layernorm.weight",
                "ffn_norm.bias": L + "post_attention_layernorm.bias",
                "ffn_up.weight": L + "mlp.fc1.weight",
                "ffn_up.bias": L + "mlp.fc1.bias",
                "ffn_down.weight": L + "mlp.fc2.weight",
                "ffn_down.bias": L + "mlp.fc2.bias",
            }
            if s in m:
                return m[s]
    return None


# GGUF stores linear weights as [out, in]; HF stores [in, out]. Transpose.
# trainer expects transformer.* prefix; CONV1D check strips that prefix for matching
CONV1D = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}  # gpt2
PHI2_LINEAR_SUFFIX = ("self_attn.qkv_proj.weight", "self_attn.dense.weight", "mlp.fc1.weight", "mlp.fc2.weight")


def needs_transpose(arch, hf_name):
    if arch in ("gpt2",):
        return any(hf_name.endswith("." + c) or hf_name.endswith(c) for c in CONV1D)
    if arch in ("phi2", "phi-2"):
        return any(hf_name.endswith("." + s) for s in PHI2_LINEAR_SUFFIX)
    return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/gguf_to_safetensors.py <in.gguf> <out.safetensors> [--out-f32] [--expand-vocab N]")
        sys.exit(1)
    in_gguf, out_path = sys.argv[1], sys.argv[2]
    out_f32 = "--out-f32" in sys.argv
    expand_vocab = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--expand-vocab" and i + 1 < len(sys.argv):
            expand_vocab = int(sys.argv[i + 1])

    reader = GGUFReader(in_gguf)
    arch = None
    af = reader.fields.get("general.architecture")
    if af:
        try:
            arch = bytes(af.parts[af.data[0]]).decode("utf-8", "replace")
        except Exception:
            arch = None
    print(f"[gguf] arch={arch}  tensors={len(reader.tensors)}")

    state = {}
    skipped = 0
    for t in reader.tensors:
        hf = map_name(arch, t.name)
        if hf is None:
            skipped += 1
            continue
        # dequantize — dequant() returns the natural logical (memory-layout)
        # shape [out, in, ...], so do NOT reshape to t.shape (GGUF dims order).
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
            arr = np.asarray(t.data).astype(np.float32)
        else:
            arr = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
        # transpose weight matrices (GGUF [out,in] -> HF [in,out])
        if arr.ndim >= 2 and needs_transpose(arch, hf):
            arr = arr.T
        if not out_f32:
            arr = arr.astype(np.float16)
        state[hf] = torch.from_numpy(np.ascontiguousarray(arr))
        print(f"  {t.name:<34} -> {hf:<52} {tuple(arr.shape)}")

    if skipped:
        print(f"[warn] skipped {skipped} unmapped tensors")

    # Expand embedding tables to expand_vocab rows so KUHUL special tokens
    # (50260-50281) can be looked up without OOB crash.  New rows get small-random
    # init (stddev=0.02, matching GPT-2 original init) so the model can learn them.
    if expand_vocab > 0:
        rng = np.random.default_rng(42)
        for key in ("transformer.wte.weight", "lm_head.weight"):
            if key not in state:
                continue
            t = state[key]
            cur_vocab = t.shape[0]
            n_embd    = t.shape[1]
            if cur_vocab >= expand_vocab:
                print(f"[vocab] {key}: already {cur_vocab} >= {expand_vocab}, skip")
                continue
            new_rows = expand_vocab - cur_vocab
            pad = rng.normal(0.0, 0.02, (new_rows, n_embd)).astype(np.float32)
            if not out_f32:
                pad = pad.astype(np.float16)
            pad_t = torch.from_numpy(pad)
            base_t = t.float() if out_f32 else t.half()
            state[key] = torch.cat([base_t, pad_t], dim=0)
            print(f"[vocab] {key}: {cur_vocab} -> {expand_vocab} (+{new_rows} random rows)")

    save_file(state, out_path)
    print(f"[ok] wrote {len(state)} tensors -> {out_path}")


if __name__ == "__main__":
    main()
