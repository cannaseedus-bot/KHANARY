#!/usr/bin/env python3
# gguf_to_safetensors.py -- convert GPT-2 GGUF back to safetensors
#
# Inverse of gpt2_safetensors_to_gguf.py. Reads a GPT-2 GGUF, transposes
# Conv1D weights back to safetensors layout, writes safetensors.
#
# Usage: python gguf_to_safetensors.py <in.gguf> <out.safetensors>

import sys, os, json, struct
import numpy as np
from pathlib import Path

# GGUF reader (from workspace gguf-py)
GGUF_PY = Path(__file__).resolve().parent.parent / "khanary-llama-build" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))
from gguf import GGUFReader

# GPT-2 tensor name mappings: GGUF → safetensors
GGUF_TO_ST = {
    "token_embd.weight":        "wte.weight",
    "position_embd.weight":     "wpe.weight",
    "output_norm.weight":       "ln_f.weight",
    "output_norm.bias":         "ln_f.bias",
    "output.weight":            "lm_head.weight",
}
# Per-layer: blk.{i}.attn_norm.weight → h.{i}.ln_1.weight etc.
LAYER_MAP = {
    "attn_norm.weight":     "ln_1.weight",
    "attn_norm.bias":       "ln_1.bias",
    "attn_qkv.weight":      "attn.c_attn.weight",
    "attn_qkv.bias":        "attn.c_attn.bias",
    "attn_output.weight":   "attn.c_proj.weight",
    "attn_output.bias":     "attn.c_proj.bias",
    "ffn_norm.weight":      "ln_2.weight",
    "ffn_norm.bias":        "ln_2.bias",
    "ffn_up.weight":        "mlp.c_fc.weight",
    "ffn_up.bias":          "mlp.c_fc.bias",
    "ffn_down.weight":      "mlp.c_proj.weight",
    "ffn_down.bias":        "mlp.c_proj.bias",
}

# Conv1D weights need transpose back: GGUF stores [out,in], safetensors stores [in,out]
CONV1D = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}

def gguf_name_to_st(name):
    """Map GGUF tensor name to safetensors name."""
    if name in GGUF_TO_ST:
        return GGUF_TO_ST[name]
    # blk.{i}.{suffix}
    parts = name.split(".")
    if parts[0] == "blk" and len(parts) >= 3:
        i = parts[1]
        suffix = ".".join(parts[2:])
        if suffix in LAYER_MAP:
            return f"h.{i}.{LAYER_MAP[suffix]}"
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python gguf_to_safetensors.py <in.gguf> <out.safetensors>")
        sys.exit(1)

    gguf_path = sys.argv[1]
    out_path = sys.argv[2]

    reader = GGUFReader(gguf_path)
    tensors = {}

    # Extract metadata
    arch = reader.fields.get("general.architecture")
    arch_val = str(arch.parts[-1]) if arch else "?"
    print(f"[read] arch={arch_val} tensors={len(reader.tensors)}")

    n_layers = 0
    for name, tensor in reader.tensors.items():
        st_name = gguf_name_to_st(name)
        if st_name is None:
            print(f"  SKIP: {name}")
            continue

        data = tensor.data
        shape = list(tensor.shape)

        # Transpose Conv1D weights
        if st_name.split(".")[-1] == "weight" and any(
            st_name.endswith(c) for c in [".attn.c_attn.weight", ".attn.c_proj.weight",
                                           ".mlp.c_fc.weight", ".mlp.c_proj.weight"]):
            data = data.T.copy()
            shape = list(data.shape)

        # Convert to float32 if quantized
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        tensors[st_name] = data
        # Track layer count
        if st_name.startswith("h."):
            n_layers = max(n_layers, int(st_name.split(".")[1]) + 1)

    print(f"[convert] {len(tensors)} tensors, {n_layers} layers")

    # Build safetensors header
    header = {}
    offset = 0
    for name in sorted(tensors.keys()):
        arr = tensors[name]
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes]
        }
        offset += nbytes

    # Add metadata
    header["__metadata__"] = {
        "converted_from": os.path.basename(gguf_path),
        "converter": "gguf_to_safetensors.py",
        "n_layers": str(n_layers),
    }

    header_json = json.dumps(header)
    while len(header_json) % 8 != 0:
        header_json += " "

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json.encode("utf-8"))
        for name in sorted(tensors.keys()):
            f.write(tensors[name].tobytes())

    mb = os.path.getsize(out_path) / (1 << 20)
    print(f"[ok] {out_path} ({mb:.1f} MB, {len(tensors)} tensors)")

if __name__ == "__main__":
    main()
