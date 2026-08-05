#!/usr/bin/env python3
# tokenize_transitions.py -- GPT-2 BPE tokenize the transition `text` fields into the flat int32
# token binary the D3D11 trainer eats (gpt2_trainer.exe --data tokens.bin: reads block-length
# int32 sequences).  Each example is separated by <|endoftext|> (50256).
#
# KUHUL special tokens (50260-50269) are inserted by detecting the literal tag strings
# before tiktoken encoding -- tiktoken never sees them, so disallowed_special=() is still used
# for the remainder of each segment.
#
# Usage: python tools/tokenize_transitions.py <in_transitions.jsonl> <out_tokens.bin> [--limit N]

import sys, re, json, argparse, numpy as np, tiktoken

# KUHUL special token map (see tokenizer_config.json)
KUHUL_TOKENS = {
    "<AGENT>":      50260,
    "</AGENT>":     50261,
    "<THINK>":      50262,
    "</THINK>":     50263,
    "<TOOL_CALL>":  50264,
    "</TOOL_CALL>": 50265,
    "<INSTRUCT>":   50266,
    "</INSTRUCT>":  50267,
    "<USER>":       50268,
    "</USER>":      50269,
}

# Sort longer tags first so </TOOL_CALL> matches before </TOOL>-like partial
_KUHUL_TAGS_SORTED = sorted(KUHUL_TOKENS.keys(), key=len, reverse=True)
KUHUL_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in _KUHUL_TAGS_SORTED) + ")")

EXTENDED_VOCAB = max(KUHUL_TOKENS.values()) + 1  # 50270


def encode_with_kuhul(enc, text: str) -> list[int]:
    """Encode text, inserting KUHUL token IDs for special tags."""
    ids = []
    for part in KUHUL_SPLIT_RE.split(text):
        if not part:
            continue
        if part in KUHUL_TOKENS:
            ids.append(KUHUL_TOKENS[part])
        else:
            ids.extend(enc.encode(part, disallowed_special=()))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--limit", type=int, default=0, help="only tokenize first N examples (probe)")
    ap.add_argument("--field", default="text")
    ap.add_argument("--no-kuhul", action="store_true", help="disable KUHUL tag injection (base vocab only)")
    a = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token  # 50256
    ids = []
    n = 0
    kuhul_count = 0

    with open(a.infile, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = rec.get(a.field)
            if not t:
                continue

            if a.no_kuhul:
                toks = enc.encode(t, disallowed_special=())
            else:
                toks = encode_with_kuhul(enc, t)
                kuhul_count += sum(1 for tok in toks if tok >= 50260)

            ids.extend(toks)
            ids.append(EOT)
            n += 1
            if a.limit and n >= a.limit:
                break

    arr = np.asarray(ids, dtype=np.int32)
    max_valid = EXTENDED_VOCAB - 1  # 50269
    assert arr.min() >= 0 and arr.max() <= max_valid, \
        f"token OOB: min={arr.min()} max={arr.max()} (expected 0..{max_valid})"

    arr.tofile(a.outfile)
    mb = arr.nbytes / 1e6
    print(f"[ok] {a.outfile}: {len(arr):,} tokens from {n:,} examples ({mb:.1f} MB int32)")
    if not a.no_kuhul:
        print(f"     KUHUL tokens injected: {kuhul_count:,} ({100*kuhul_count/len(arr):.2f}% of stream)")
    for blk in (64, 128):
        print(f"     block={blk} -> {len(arr)//blk:,} sequences")


if __name__ == "__main__":
    main()
