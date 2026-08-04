#!/usr/bin/env python3
"""
pack_tokens.py -- Convert flat int32 token binary (from tokenize_transitions.py)
                  into the headed format that gpt2_trainer.exe expects:

  [uint32 n_seq][uint32 seq_len][int32 * n_seq * seq_len]

Each input token is a flat stream; this script chunks it into non-overlapping
sequences of seq_len tokens and discards the final partial sequence.

Usage:
  python tools/pack_tokens.py <flat.bin> <packed.bin> [--seq-len 128]
  python tools/pack_tokens.py E:/data/chunks/v2_corpus/chunk_000_flat.bin \
                               E:/data/chunks/v2_corpus/chunk_000_tokens.bin
"""

import argparse, struct, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flat",   help="Input flat int32 binary (from tokenize_transitions.py)")
    ap.add_argument("packed", help="Output headed binary for gpt2_trainer.exe")
    ap.add_argument("--seq-len", type=int, default=128,
                    help="Sequence length in tokens (default 128, use 256 for medium)")
    a = ap.parse_args()

    S   = a.seq_len
    src = Path(a.flat)
    dst = Path(a.packed)

    file_bytes = src.stat().st_size
    n_tokens   = file_bytes // 4          # flat int32
    n_seq      = n_tokens // S            # complete sequences only

    if n_seq == 0:
        print(f"ERROR: {n_tokens} tokens < seq_len {S} — no complete sequences")
        sys.exit(1)

    print(f"{src.name}: {n_tokens:,} tokens -> {n_seq:,} sequences x {S}")

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<II", n_seq, S))
        data = fin.read(n_seq * S * 4)     # read only the complete sequences
        fout.write(data)

    out_mb = dst.stat().st_size / 1e6
    print(f"Written: {dst}  ({out_mb:.1f} MB)")
    print(f"\nTrain command:")
    print(f'  gpt2_trainer.exe \\')
    print(f'    --model "C:\\Users\\canna\\.ASX.cpp\\trainer\\from_zero_v0.1.safetensors" \\')
    print(f'    --data  "{dst.resolve()}" \\')
    print(f'    --out   "C:\\Users\\canna\\.ASX.cpp\\trainer\\from_zero_v0.2_chunk.safetensors" \\')
    print(f'    --steps 5000 --batch 4 --block {S} --lr 3e-5')

if __name__ == "__main__":
    main()
