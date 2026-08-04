#!/usr/bin/env python3
"""
extend_vocab.py -- Extend a GPT-2 safetensors checkpoint with KUHUL special tokens.

Appends 10 zero-initialized embedding rows for the KUHUL token set (IDs 50260-50269)
to the wte (token embedding) tensor and, if present and untied, to lm_head.weight.

Input:  from_zero_v0.1.safetensors  (vocab_size=50260, wte shape [50260, 768])
Output: from_zero_v0.1_kuhul.safetensors (vocab_size=50270, wte shape [50270, 768])

Usage:
  python tools/extend_vocab.py ^
    C:/Users/canna/.ASX.cpp/trainer/from_zero_v0.1.safetensors ^
    C:/Users/canna/_khanary_inspect/models/from_zero/from_zero_v0.1_kuhul.safetensors

KUHUL token mapping (see tokenizer_config.json):
  50260 <AGENT>      50261 </AGENT>
  50262 <THINK>      50263 </THINK>
  50264 <TOOL_CALL>  50265 </TOOL_CALL>
  50266 <INSTRUCT>   50267 </INSTRUCT>
  50268 <USER>       50269 </USER>
"""

import sys, argparse, struct, json
from pathlib import Path
import numpy as np

KUHUL_TOKENS = [
    "<AGENT>", "</AGENT>",
    "<THINK>", "</THINK>",
    "<TOOL_CALL>", "</TOOL_CALL>",
    "<INSTRUCT>", "</INSTRUCT>",
    "<USER>", "</USER>",
]
N_NEW = len(KUHUL_TOKENS)    # 10
KUHUL_START = 50260          # first KUHUL token ID (matches tokenizer_config.json)
TARGET_VOCAB = KUHUL_START + N_NEW  # 50270
N_EMBD = 768                 # GPT-2 small


def st_read(path: str):
    """Read a safetensors file; handles shape=[] (custom trainer format: read as flat 1D)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr  = json.loads(f.read(hlen))
        data_start = 8 + hlen
        meta = hdr.get("__metadata__", {})
        dtype_map = {"F32": np.float32, "F16": np.float16,
                     "BF16": np.uint16,  "I32": np.int32,
                     "I64": np.int64,    "U8":  np.uint8}
        tensors = {}
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            dtype  = dtype_map.get(info["dtype"], np.float32)
            shape  = info["shape"]
            off_s, off_e = info["data_offsets"]
            f.seek(data_start + off_s)
            raw = f.read(off_e - off_s)
            flat = np.frombuffer(raw, dtype=dtype).copy()
            tensors[name] = flat.reshape(shape) if shape else flat  # shape=[] -> stay flat
    return meta, tensors


def st_write(path: str, tensors: dict, metadata: dict = None):
    """Write tensors to a safetensors file (float32, with correct shape metadata)."""
    offset = 0
    records = {}
    blobs   = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        raw = arr.tobytes()
        records[name] = {"dtype": "F32", "shape": list(arr.shape),
                         "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        records["__metadata__"] = metadata
    hdr_bytes = json.dumps(records, separators=(",", ":")).encode()
    pad = (8 - len(hdr_bytes) % 8) % 8
    hdr_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for b in blobs:
            f.write(b)


def extend_vocab(src: str, dst: str, init_std: float = 0.0):
    meta, tensors = st_read(src)

    # ── locate wte ────────────────────────────────────────────────────────────
    wte_key = next((k for k in tensors if "wte" in k), None)
    if wte_key is None:
        sys.exit(f"ERROR: no wte tensor found in {src}")

    wte = tensors[wte_key].astype(np.float32)

    # Handle flat (shape=[]) tensors written by the custom trainer serializer
    if wte.ndim == 1:
        wte = wte.reshape(-1, N_EMBD)
        print(f"  (reshaped flat wte -> [{wte.shape[0]}, {N_EMBD}])")

    V, E = wte.shape
    print(f"wte: [{V}, {E}]  dtype=float32  key='{wte_key}'")

    if V >= TARGET_VOCAB:
        sys.exit(f"ERROR: wte already has {V} rows >= target {TARGET_VOCAB}; already extended?")

    # Pad any gap between current vocab and KUHUL_START with zero rows
    if V < KUHUL_START:
        pad_rows = np.zeros((KUHUL_START - V, E), dtype=np.float32)
        wte = np.concatenate([wte, pad_rows], axis=0)
        print(f"  padded: [{V}, {E}] -> [{KUHUL_START}, {E}] (slots {V}-{KUHUL_START-1} zeroed)")
        V = KUHUL_START

    # ── build KUHUL rows — semantic anchor initialization ─────────────────────
    # Each KUHUL special token is initialized as the mean of semantically proximate
    # base-vocab tokens. This "dents" the embedding manifold before any training:
    # <THINK> lands near {think, reason, consider, understand},
    # <AGENT> lands near {agent, actor, system, assistant}, etc.
    # buildThinkDepth() computes geodesic arcs from these anchors — zero-init
    # gives every arc = π/2 (no structure); semantic anchors give meaningful geometry.
    # Target scale = mean of individual token vector norms in base vocab.
    # DO NOT use norm(mean(wte)) — that nearly cancels to zero because
    # wte rows are roughly zero-centered and average to ~0 vector.
    base_norm_scale = float(np.linalg.norm(wte[:min(V, 50257)], axis=1).mean())

    def anchor(enc, words):
        ids = [enc.encode(w) for w in words]
        ids = [i[0] for i in ids if len(i) == 1 and i[0] < len(wte)]
        if not ids:
            return np.zeros(E, dtype=np.float32)
        # Normalized sum of unit vectors (convex combination on unit sphere)
        acc = np.zeros(E, dtype=np.float64)
        for i in ids:
            v = wte[i].astype(np.float64)
            n = np.linalg.norm(v)
            if n > 1e-9: acc += v / n
        acc /= max(len(ids), 1)
        norm = np.linalg.norm(acc)
        # Re-scale to match typical base vocab vector magnitude
        return (acc / norm * base_norm_scale if norm > 1e-9 else acc).astype(np.float32)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None

    def _anc(words):
        return anchor(enc, words) if enc else np.zeros(E, dtype=np.float32)

    # Semantic anchors: OPEN and CLOSE tags have DIFFERENT anchor directions.
    # Open tags anchor the entry point of a reasoning/execution span;
    # close tags anchor the completion/boundary — tilted toward "end"/"result" concepts.
    # This creates a non-trivial geodesic arc between open and close, so the model
    # learns a geometric span rather than a single collapsed point.
    #
    # buildThinkDepth() computes arc = arccos(dot(wte[open_tok], wte[t]) / (|a||b|))
    # With symmetric anchors: arc(open,close) = 0  →  no span geometry at all.
    # With complementary anchors: arc(open,close) ∈ (0, π)  →  real funnel boundary.
    kuhul_anchors = [
        # Open tags: concept entry — WHERE reasoning/action begins
        _anc(["agent", "actor", "system", "assistant"]),                   # <AGENT>
        _anc(["end", "complete", "done", "agent", "system"]),             # </AGENT>
        _anc(["think", "reason", "analyze", "reflect", "consider"]),      # <THINK>
        _anc(["end", "conclude", "result", "therefore", "finally"]),      # </THINK>
        _anc(["call", "invoke", "execute", "function", "request"]),       # <TOOL_CALL>
        _anc(["end", "return", "output", "result", "response"]),          # </TOOL_CALL>
        _anc(["instruction", "command", "directive", "goal", "task"]),    # <INSTRUCT>
        _anc(["end", "done", "complete", "instruction", "finished"]),     # </INSTRUCT>
        _anc(["user", "human", "person", "input", "query"]),              # <USER>
        _anc(["end", "user", "done", "human", "submitted"]),              # </USER>
    ]
    new_rows = np.stack(kuhul_anchors, axis=0).astype(np.float32)

    if init_std > 0.0:
        rng = np.random.default_rng(0)
        new_rows = new_rows + rng.normal(0.0, init_std, (N_NEW, E)).astype(np.float32)

    print("  KUHUL semantic anchors computed:")
    for i, tok in enumerate(KUHUL_TOKENS):
        n = np.linalg.norm(new_rows[i])
        print(f"    {KUHUL_START+i} {tok:15s} norm={n:.3f}")

    tensors[wte_key] = np.concatenate([wte, new_rows], axis=0)
    print(f"wte extended: [{V}, {E}] -> [{TARGET_VOCAB}, {E}]")

    # ── extend lm_head if untied ──────────────────────────────────────────────
    lmh_key = next((k for k in tensors if "lm_head" in k and k != wte_key), None)
    if lmh_key is not None:
        lmh = tensors[lmh_key].astype(np.float32)
        if lmh.ndim == 1:
            lmh = lmh.reshape(-1, N_EMBD)
        if lmh.shape == (V, E):
            tensors[lmh_key] = np.concatenate([lmh, new_rows], axis=0)
            print(f"lm_head extended: [{V}, {E}] -> [{V + N_NEW}, {E}]  key='{lmh_key}'")
        else:
            print(f"lm_head shape {lmh.shape} != [{V}, {E}] — skipped")
    else:
        print("lm_head: tied to wte (or absent) — no separate extension needed")

    # ── update metadata ───────────────────────────────────────────────────────
    meta["kuhul_vocab_extension"] = json.dumps({
        "base_vocab": wte.shape[0] if wte.shape[0] < KUHUL_START else KUHUL_START,
        "extended_vocab": TARGET_VOCAB,
        "kuhul_start": KUHUL_START,
        "kuhul_tokens": {t: KUHUL_START + i for i, t in enumerate(KUHUL_TOKENS)},
    })

    # ── ensure all tensors are float32 contiguous ─────────────────────────────
    tensors = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()}

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {dst} ...")
    st_write(dst, tensors, metadata=meta)
    size_mb = Path(dst).stat().st_size / 1e6
    print(f"Done: {dst}  ({size_mb:.1f} MB)")
    print()
    print("KUHUL token IDs:")
    for i, tok in enumerate(KUHUL_TOKENS):
        print(f"  {KUHUL_START + i:6d}  {tok}")


def main():
    ap = argparse.ArgumentParser(
        description="Extend GPT-2 safetensors vocab with 10 KUHUL special tokens"
    )
    ap.add_argument("src", help="Source safetensors (from_zero_v0.1.safetensors)")
    ap.add_argument("dst", help="Output safetensors (from_zero_v0.1_kuhul.safetensors)")
    ap.add_argument("--init-std", type=float, default=0.0,
                    help="Init std for new embedding rows (0=zeros, 0.02=small normal)")
    a = ap.parse_args()
    print(f"Reading {a.src} ...")
    extend_vocab(a.src, a.dst, init_std=a.init_std)


if __name__ == "__main__":
    main()
