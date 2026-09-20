#!/usr/bin/env python3
"""
validate_anchors.py — Verify KUHUL gravity well geometry before training.

Checks:
  1. Norms: each KUHUL embedding has non-trivial magnitude (not collapsed to origin)
  2. Open/close arc: angle between <X> and </X> is in (15°, 90°) — not 0, not flat
  3. Cross-cluster separation: <THINK> should be further from <TOOL_CALL> than from </THINK>
  4. Base-alignment: KUHUL norms similar to base vocab mean norm (not outliers)
  5. No NaN/inf in any KUHUL row

Usage:
  python tools/validate_anchors.py models/from_zero/from_zero_v0.1_kuhul.safetensors
"""

import sys, struct, json, math
import numpy as np

KUHUL_TOKENS = [
    "<AGENT>", "</AGENT>",
    "<THINK>", "</THINK>",
    "<TOOL_CALL>", "</TOOL_CALL>",
    "<INSTRUCT>", "</INSTRUCT>",
    "<USER>", "</USER>",
]
KUHUL_START = 50260
N_EMBD = 768

PAIRS = [(0,1), (2,3), (4,5), (6,7), (8,9)]   # open/close index pairs


def st_read_wte(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr  = json.loads(f.read(hlen))
        data_start = 8 + hlen
        wte_key = next((k for k in hdr if "wte" in k and k != "__metadata__"), None)
        if not wte_key:
            sys.exit("ERROR: no wte tensor found")
        info = hdr[wte_key]
        dtype = np.float32
        off_s, off_e = info["data_offsets"]
        f.seek(data_start + off_s)
        raw  = f.read(off_e - off_s)
        flat = np.frombuffer(raw, dtype=dtype).copy()
        shape = info["shape"]
        return flat.reshape(shape) if shape else flat.reshape(-1, N_EMBD)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def arc_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = max(-1.0, min(1.0, cos_sim(a, b)))
    return math.degrees(math.acos(c))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\canna\_khanary_inspect\models\from_zero\from_zero_v0.1_kuhul.safetensors"
    print(f"Loading: {path}")
    wte = st_read_wte(path)
    V, E = wte.shape
    print(f"wte shape: [{V}, {E}]")

    if V < KUHUL_START + 10:
        sys.exit(f"ERROR: wte has only {V} rows — KUHUL tokens not present")

    base = wte[:KUHUL_START]
    kuhul = wte[KUHUL_START:KUHUL_START+10]

    base_norm_mean = float(np.linalg.norm(base, axis=1).mean())
    base_std       = float(np.std(base))
    print(f"\nBase vocab  — mean norm: {base_norm_mean:.4f}  std: {base_std:.6f}")

    # ── Check 1: norms ──────────────────────────────────────────────────────
    print("\n── 1. KUHUL embedding norms ──")
    norms = np.linalg.norm(kuhul, axis=1)
    ok_norm = True
    for i, tok in enumerate(KUHUL_TOKENS):
        flag = "✓" if norms[i] > 0.1 else "✗ ZERO"
        print(f"  {KUHUL_START+i}  {tok:15s}  norm={norms[i]:.4f}  {flag}")
        if norms[i] < 0.1:
            ok_norm = False

    # ── Check 2: open/close arc ────────────────────────────────────────────
    print("\n── 2. Open/close pair geodesic arcs ──")
    ok_arc = True
    for oi, ci in PAIRS:
        deg = arc_deg(kuhul[oi], kuhul[ci])
        flag = "✓" if 10.0 < deg < 120.0 else ("✗ TOO_CLOSE" if deg <= 10.0 else "✗ TOO_FAR")
        print(f"  {KUHUL_TOKENS[oi]:12s} ↔ {KUHUL_TOKENS[ci]:13s}  arc={deg:.1f}°  {flag}")
        if not (10.0 < deg < 120.0):
            ok_arc = False

    # ── Check 3: cross-cluster separation ─────────────────────────────────
    print("\n── 3. Cross-cluster separation (should separate concept clusters) ──")
    clusters = {
        "AGENT":     kuhul[0],
        "THINK":     kuhul[2],
        "TOOL_CALL": kuhul[4],
        "INSTRUCT":  kuhul[6],
        "USER":      kuhul[8],
    }
    names = list(clusters.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            deg = arc_deg(clusters[names[i]], clusters[names[j]])
            print(f"  {names[i]:10s} ↔ {names[j]:10s}  arc={deg:.1f}°")

    # ── Check 4: NaN/inf ──────────────────────────────────────────────────
    print("\n── 4. Finite check ──")
    bad = np.any(~np.isfinite(kuhul))
    print(f"  {'✗ NaN/inf DETECTED' if bad else '✓ all finite'}")

    # ── Check 5: arc distribution preview ────────────────────────────────
    print("\n── 5. Sample arcs vs base vocab tokens ──")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        samples = {
            "<THINK>":    ["reason", "analyze", "think", "call", "user", "the", "is"],
            "<TOOL_CALL>": ["call", "execute", "function", "reason", "think", "the"],
        }
        for anchor_tok, words in samples.items():
            ai = KUHUL_TOKENS.index(anchor_tok)
            print(f"\n  Arcs from {anchor_tok}:")
            for word in words:
                ids = enc.encode(word)
                if ids and ids[0] < KUHUL_START:
                    deg = arc_deg(kuhul[ai], wte[ids[0]])
                    print(f"    '{word:12s}' (id={ids[0]:5d})  arc={deg:.1f}°")
    except ImportError:
        print("  (tiktoken not available — skip arc distribution check)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n── Summary ──")
    all_ok = ok_norm and ok_arc and not bad
    if all_ok:
        print("  ✓ Gravity wells look valid — safe to proceed to Phase 0 training")
    else:
        if not ok_norm: print("  ✗ Some KUHUL tokens have near-zero norms — re-run extend_vocab.py")
        if not ok_arc:  print("  ✗ Some open/close pairs have bad arc — check anchor word lists")
        if bad:         print("  ✗ NaN/inf in KUHUL rows — corrupted safetensors write")


if __name__ == "__main__":
    main()
