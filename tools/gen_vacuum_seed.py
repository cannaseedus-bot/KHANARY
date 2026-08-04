#!/usr/bin/env python3
"""
gen_vacuum_seed.py — Generate structural vacuum templates for KUHUL gravity-well seeding.

Produces a binary token file containing only syntactic structural frames:
  <AGENT> <THINK> … </THINK> <TOOL_CALL> … </TOOL_CALL> </AGENT>

with all "payload" slots filled by <|endoftext|> (50256).  No real data, no facts.

The model trained on this corpus first learns:
  1. Where KUHUL structural tokens sit in the embedding manifold
  2. Their pairwise geodesic relationships (used by buildThinkDepth)
  3. The open/close pairing geometry (<THINK>→</THINK>, etc.)

This "dents" the latent space before Phase 1 training floods it with real data.
After this pass, buildThinkDepth() computes meaningful arc depths (not uniform π/2).

Usage:
  python tools/gen_vacuum_seed.py --out E:/data/vacuum_seed.bin --count 50000 --block 64

Output: packed flat int32 binary, same format as tokenize_transitions.py output.
"""

import argparse, random, struct, sys
import numpy as np

# KUHUL token IDs (must match tokenize_transitions.py)
T_AGENT      = 50260
T_AGENT_END  = 50261
T_THINK      = 50262
T_THINK_END  = 50263
T_TOOL_CALL  = 50264
T_TOOL_END   = 50265
T_INSTRUCT   = 50266
T_INSTRUCT_END = 50267
T_USER       = 50268
T_USER_END   = 50269
T_EOT        = 50256   # <|endoftext|>
T_PAD        = 50256   # padding = EOT

# Structural frame templates.
# PAD slots (value=0 here, replaced below) represent payload positions the model
# will learn to "accept" inside structural boundaries.
_PAD = 0  # placeholder — replaced with T_PAD at generation time

TEMPLATES = [
    # Agent wrapping a think block
    [T_AGENT, T_THINK, _PAD, _PAD, _PAD, _PAD, T_THINK_END, T_AGENT_END],
    # Agent wrapping a tool call
    [T_AGENT, T_TOOL_CALL, _PAD, _PAD, _PAD, T_TOOL_END, T_AGENT_END],
    # Full round-trip: instruct → agent think → tool call
    [T_INSTRUCT, _PAD, _PAD, T_INSTRUCT_END,
     T_AGENT, T_THINK, _PAD, _PAD, T_THINK_END,
     T_TOOL_CALL, _PAD, T_TOOL_END, T_AGENT_END],
    # User turn → agent response
    [T_USER, _PAD, _PAD, T_USER_END,
     T_AGENT, T_THINK, _PAD, T_THINK_END, T_AGENT_END],
    # Nested: think inside think (depth test for geodesic arc)
    [T_THINK, T_THINK, _PAD, T_THINK_END, _PAD, T_THINK_END],
    # Open/close pairs alone (teach pairing geometry)
    [T_AGENT, T_AGENT_END],
    [T_THINK, T_THINK_END],
    [T_TOOL_CALL, T_TOOL_END],
    [T_INSTRUCT, T_INSTRUCT_END],
    [T_USER, T_USER_END],
]


def fill_template(tpl: list, block: int) -> list:
    """Replace _PAD placeholders with T_PAD, then pad/truncate to block tokens."""
    seq = [T_PAD if x == _PAD else x for x in tpl]
    if len(seq) < block:
        seq += [T_PAD] * (block - len(seq))
    return seq[:block]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",   default="E:/data/vacuum_seed.bin")
    ap.add_argument("--count", type=int, default=50000, help="number of sequences")
    ap.add_argument("--block", type=int, default=64,    help="tokens per sequence")
    ap.add_argument("--seed",  type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    ids = []
    for _ in range(a.count):
        tpl = rng.choice(TEMPLATES)
        ids.extend(fill_template(tpl, a.block))

    arr = np.asarray(ids, dtype=np.int32)
    arr.tofile(a.out)
    mb = arr.nbytes / 1e6
    print(f"[ok] {a.out}: {a.count:,} sequences × {a.block} tokens = {len(arr):,} tokens ({mb:.1f} MB)")
    kuhul = sum(1 for x in ids if 50260 <= x <= 50269)
    print(f"     KUHUL tokens: {kuhul:,} ({100*kuhul/len(ids):.1f}% of stream)")


if __name__ == "__main__":
    main()
