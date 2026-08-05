"""repair_safetensors.py — Fix safetensors files with empty shape [] tensors.

Reads shape map from a valid reference checkpoint (same architecture),
then rebuilds target files using the correct shapes and existing data.

Usage:
    python tools/repair_safetensors.py \
        --ref   models/from_zero/from_zero_v0.1_folded.safetensors \
        --targets models/from_zero/from_zero_v0.4_phase1.safetensors \
                  models/from_zero/from_zero_v0.5_phase2.safetensors
"""

import argparse
import json
import struct
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path


def load_shape_map(ref_path: str) -> dict:
    f = safe_open(ref_path, framework="pt")
    return {k: list(f.get_tensor(k).shape) for k in f.keys()}


def repair(target_path: str, shape_map: dict):
    path = Path(target_path)
    raw = path.read_bytes()

    # Parse existing header
    (hdr_len,) = struct.unpack_from("<Q", raw, 0)
    hdr_json   = raw[8 : 8 + hdr_len].decode("utf-8")
    hdr        = json.loads(hdr_json)
    data_start = 8 + hdr_len

    missing_in_ref = [k for k in hdr if k != "__metadata__" and k not in shape_map]
    if missing_in_ref:
        print(f"  WARN: {len(missing_in_ref)} keys missing from ref (e.g. {missing_in_ref[:3]})")

    tensors = {}
    for key, meta in hdr.items():
        if key == "__metadata__":
            continue
        if key not in shape_map:
            print(f"  SKIP (not in ref): {key}")
            continue

        shape = shape_map[key]
        dtype_str = meta["dtype"]          # "F32", "BF16", etc.
        off_s, off_e = meta["data_offsets"]

        dtype_map = {
            "F32":  (np.float32,  torch.float32),
            "F16":  (np.float16,  torch.float16),
            "BF16": (np.uint16,   torch.bfloat16),
            "I32":  (np.int32,    torch.int32),
            "I64":  (np.int64,    torch.int64),
            "U8":   (np.uint8,    torch.uint8),
        }
        np_dtype, torch_dtype = dtype_map.get(dtype_str, (np.float32, torch.float32))

        raw_slice = raw[data_start + off_s : data_start + off_e]
        n_elements = len(raw_slice) // np.dtype(np_dtype).itemsize

        # Sanity: shape must match data size
        shape_numel = 1
        for d in shape:
            shape_numel *= d
        if shape_numel != n_elements:
            # Fall back to flat 1D
            print(f"  RESHAPE MISMATCH for {key}: ref shape {shape} = {shape_numel} but data has {n_elements} elements → using flat [{n_elements}]")
            shape = [n_elements]

        arr = np.frombuffer(raw_slice, dtype=np_dtype).reshape(shape).copy()
        if dtype_str == "BF16":
            t = torch.from_numpy(arr).view(torch.bfloat16)
        else:
            t = torch.from_numpy(arr)
        tensors[key] = t

    out_path = path.with_suffix(".repaired.safetensors")
    save_file(tensors, str(out_path))
    print(f"  Saved: {out_path}  ({out_path.stat().st_size // 1024 // 1024} MB)")

    # Validate
    try:
        fv = safe_open(str(out_path), framework="pt")
        keys = list(fv.keys())
        print(f"  Validated: {len(keys)} tensors, first 3: {keys[:3]}")
    except Exception as e:
        print(f"  VALIDATION FAILED: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",     required=True, help="Valid reference .safetensors")
    ap.add_argument("--targets", nargs="+",     required=True, help="Broken .safetensors to repair")
    args = ap.parse_args()

    print(f"[repair] loading shape map from: {args.ref}")
    shape_map = load_shape_map(args.ref)
    print(f"[repair] {len(shape_map)} shapes loaded")

    for t in args.targets:
        print(f"\n[repair] target: {t}")
        repair(t, shape_map)


if __name__ == "__main__":
    main()
