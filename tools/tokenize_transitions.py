#!/usr/bin/env python3
# tokenize_transitions.py -- GPT-2 BPE tokenize the transition `text` fields into the flat int32
# token binary the D3D11 trainer eats (gpt2_trainer.exe --data tokens.bin: reads block-length
# int32 sequences).  Each example is separated by <|endoftext|> (50256).
#
# Usage: python tools/tokenize_transitions.py <in_transitions.jsonl> <out_tokens.bin> [--limit N]

import sys, json, argparse, numpy as np, tiktoken

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--limit", type=int, default=0, help="only tokenize first N examples (probe)")
    ap.add_argument("--field", default="text")
    a = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token  # 50256
    ids = []
    n = 0
    with open(a.infile, encoding="utf-8") as f:
        for line in f:
            try: rec = json.loads(line)
            except Exception: continue
            t = rec.get(a.field)
            if not t: continue
            ids.extend(enc.encode(t, allowed_special=set()))
            ids.append(EOT)
            n += 1
            if a.limit and n >= a.limit: break

    arr = np.asarray(ids, dtype=np.int32)
    assert arr.min() >= 0 and arr.max() <= 50256, f"token OOB: {arr.min()}..{arr.max()}"
    arr.tofile(a.outfile)
    mb = arr.nbytes / 1e6
    print(f"[ok] {a.outfile}: {len(arr):,} tokens from {n:,} examples ({mb:.1f} MB int32)")
    for blk in (64, 128):
        print(f"     block={blk} -> {len(arr)//blk:,} sequences")

if __name__ == "__main__":
    main()
