#!/usr/bin/env python3
"""
fold_kuhul_to_brain.py — Fold KUHUL special tokens into brain expert clusters.

What this does:
  1. Reads brain2/experts.bin  (30628 int32 expert IDs 0-60)
  2. Reads wte from a safetensors checkpoint
  3. For each of the 61 expert clusters, computes its wte-space centroid
     (mean of wte rows for ALL base-vocab tokens that hash into each cluster)
  4. For each KUHUL token, finds the cluster whose centroid is closest (cosine sim)
     to the token's semantic anchor word list
  5. Overwrites brain_experts_[kuhul_id % N_BRAIN] = matched_cluster_id
  6. Initializes wte[kuhul_id] = cluster centroid (scaled to base vocab norm)
  7. Writes:  <dst>.safetensors  (updated wte)
              brain2/experts_kuhul.bin  (updated expert assignments)

After this, KUHUL tokens are:
  - Embedded AT the cluster centroid in wte-space  (meaningful geometry)
  - Routed to that same cluster by brain_experts_   (consistent expert ID)
  - Linked via buildThinkDepth arcs to same-cluster tokens  (real arc structure)

Usage:
  python tools/fold_kuhul_to_brain.py ^
    models/from_zero/from_zero_v0.1_kuhul.safetensors ^
    models/from_zero/from_zero_v0.1_folded.safetensors ^
    --brain brain2/experts.bin ^
    --out-brain brain2/experts_kuhul.bin
"""

import sys, struct, json, argparse
from pathlib import Path
import numpy as np

KUHUL_TOKENS = [
    "<AGENT>",     "</AGENT>",
    "<THINK>",     "</THINK>",
    "<TOOL_CALL>", "</TOOL_CALL>",
    "<INSTRUCT>",  "</INSTRUCT>",
    "<USER>",      "</USER>",
]
KUHUL_START  = 50260
N_EMBD       = 768
BASE_VOCAB   = 50257

# Semantic anchor words for each KUHUL token (open + close differ)
KUHUL_ANCHORS = {
    "<AGENT>":      (["agent", "system", "assistant", "operator"],
                     "open"),
    "</AGENT>":     (["end", "complete", "agent", "done"],
                     "close"),
    "<THINK>":      (["think", "reason", "analyze", "reflect", "consider"],
                     "open"),
    "</THINK>":     (["conclude", "result", "therefore", "finally", "end"],
                     "close"),
    "<TOOL_CALL>":  (["call", "invoke", "execute", "function", "request"],
                     "open"),
    "</TOOL_CALL>": (["return", "output", "result", "response", "end"],
                     "close"),
    "<INSTRUCT>":   (["instruction", "command", "directive", "goal", "task"],
                     "open"),
    "</INSTRUCT>":  (["done", "complete", "finished", "instruction", "end"],
                     "close"),
    "<USER>":       (["user", "human", "person", "input", "query"],
                     "open"),
    "</USER>":      (["done", "submitted", "user", "end", "human"],
                     "close"),
}

# Preferred cluster for close tags: same cluster as their open partner but
# we also enforce minimum arc by blending in a "completion" direction.
CLOSE_BLEND_ALPHA = 0.4   # 40% from close-word anchor, 60% from open-cluster centroid


def st_read(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr  = json.loads(f.read(hlen))
        data_start = 8 + hlen
        meta = hdr.get("__metadata__", {})
        dtype_map = {"F32": np.float32, "F16": np.float16,
                     "BF16": np.uint16,  "I32": np.int32}
        tensors = {}
        for name, info in hdr.items():
            if name == "__metadata__": continue
            dtype  = dtype_map.get(info["dtype"], np.float32)
            shape  = info["shape"]
            off_s, off_e = info["data_offsets"]
            f.seek(data_start + off_s)
            raw  = f.read(off_e - off_s)
            flat = np.frombuffer(raw, dtype=dtype).copy()
            tensors[name] = flat.reshape(shape) if shape else flat
    return meta, tensors


def st_write(path, tensors, metadata=None):
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
        for b in blobs: f.write(b)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))


def semantic_vec(wte, enc, words):
    """Normalized weighted mean of unit vectors for each word (first subword token)."""
    acc = np.zeros(wte.shape[1], dtype=np.float64)
    n   = 0
    for w in words:
        try:
            ids = enc.encode(w)
        except Exception:
            continue
        tid = ids[0]
        if tid >= BASE_VOCAB: continue
        v = wte[tid].astype(np.float64)
        nrm = np.linalg.norm(v)
        if nrm > 1e-9:
            acc += v / nrm
            n   += 1
    if n == 0: return np.zeros(wte.shape[1], dtype=np.float32)
    return (acc / n).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src",  help="Input safetensors (with zeros or random KUHUL rows)")
    ap.add_argument("dst",  help="Output safetensors (KUHUL rows folded into brain nodes)")
    ap.add_argument("--brain",     default="../../brain2/experts.bin",
                    help="Path to brain2/experts.bin  (30628 int32)")
    ap.add_argument("--out-brain", default="../../brain2/experts_kuhul.bin",
                    help="Output path for updated experts.bin")
    ap.add_argument("--n-brain",   type=int, default=30628)
    ap.add_argument("--k-experts", type=int, default=61)
    a = ap.parse_args()

    # ── Load brain experts ────────────────────────────────────────────────────
    brain_path = Path(a.brain)
    if not brain_path.exists():
        sys.exit(f"brain experts not found: {brain_path}")
    experts = np.fromfile(str(brain_path), dtype=np.int32)
    N = len(experts)
    K = a.k_experts
    print(f"Brain: {N} nodes, {K} clusters")

    # ── Load wte ──────────────────────────────────────────────────────────────
    print(f"Loading wte from {a.src} ...")
    meta, tensors = st_read(a.src)
    wte_key = next((k for k in tensors if "wte" in k), None)
    if not wte_key: sys.exit("No wte tensor found")
    wte = tensors[wte_key].astype(np.float32)
    if wte.ndim == 1: wte = wte.reshape(-1, N_EMBD)
    V, E = wte.shape
    print(f"wte: [{V}, {E}]")

    base_wte = wte[:BASE_VOCAB]   # [50257, 768] — original GPT-2 base vocab
    base_norm = float(np.linalg.norm(base_wte, axis=1).mean())

    # ── Compute per-cluster wte centroids ────────────────────────────────────
    # For each base-vocab token t, its brain hash is t % N.
    # centroid[k] = mean of wte[t] for all t where experts[t % N] == k
    print(f"Computing {K} cluster centroids in wte-space ...")
    centroids = np.zeros((K, E), dtype=np.float64)
    counts    = np.zeros(K, dtype=np.int64)
    for t in range(BASE_VOCAB):
        k = int(experts[t % N])
        centroids[k] += base_wte[t].astype(np.float64)
        counts[k]    += 1
    for k in range(K):
        if counts[k] > 0:
            centroids[k] /= counts[k]
    centroids = centroids.astype(np.float32)

    # Re-scale each centroid to match the base vocab mean norm
    for k in range(K):
        nrm = np.linalg.norm(centroids[k])
        if nrm > 1e-9:
            centroids[k] = centroids[k] * (base_norm / nrm)

    print(f"  Cluster centroid norms: mean={np.linalg.norm(centroids, axis=1).mean():.4f}")

    # ── Load tiktoken for anchor word resolution ──────────────────────────────
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None
        print("WARNING: tiktoken not found — using zero anchors for unresolved words")

    # ── Find best cluster for each KUHUL token ────────────────────────────────
    print("\nFolding KUHUL tokens into brain nodes:")
    new_rows    = np.zeros((10, E), dtype=np.float32)
    assignments = {}   # tok_name -> cluster_id

    # Track open-token cluster assignments for close-tag blending
    open_clusters = {}

    for i, tok in enumerate(KUHUL_TOKENS):
        words, role = KUHUL_ANCHORS[tok]
        anchor = semantic_vec(base_wte, enc, words) if enc else np.zeros(E, np.float32)

        if role == "close":
            # Close tag: blend open-cluster centroid + close-word anchor
            open_tok = KUHUL_TOKENS[i - 1]   # paired opener (always i-1 by construction)
            open_k   = assignments.get(open_tok, 0)
            open_cen = centroids[open_k]
            # Blend: 60% open centroid, 40% close anchor
            blended  = (1.0 - CLOSE_BLEND_ALPHA) * open_cen + CLOSE_BLEND_ALPHA * anchor
            nrm = np.linalg.norm(blended)
            if nrm > 1e-9: blended = blended * (base_norm / nrm)
            # Find closest cluster to the blended direction
            sims = np.array([cosine(blended, centroids[k]) for k in range(K)])
            # Prefer a different cluster than opener to create arc; fall back if needed
            sims_no_open = sims.copy(); sims_no_open[open_k] = -1.0
            best_k = int(np.argmax(sims_no_open)) if sims_no_open.max() > 0.0 else int(np.argmax(sims))
            new_rows[i] = blended
        else:
            # Open tag: find cluster closest to semantic anchor
            if np.linalg.norm(anchor) < 1e-9:
                best_k = 0
                new_rows[i] = centroids[0]
            else:
                sims = np.array([cosine(anchor, centroids[k]) for k in range(K)])
                best_k = int(np.argmax(sims))
                new_rows[i] = centroids[best_k]
            open_clusters[tok] = best_k

        assignments[tok] = best_k
        arc_to_centroid = np.degrees(np.arccos(max(-1.0, min(1.0, cosine(new_rows[i], centroids[best_k])))))
        print(f"  {KUHUL_START+i}  {tok:15s}  -> cluster {best_k:2d}  "
              f"norm={np.linalg.norm(new_rows[i]):.3f}  "
              f"arc_to_centroid={arc_to_centroid:.1f}deg")

    # ── Verify open/close arcs ────────────────────────────────────────────────
    print("\nOpen/close arc verification:")
    for oi in range(0, 10, 2):
        ci = oi + 1
        arc = np.degrees(np.arccos(max(-1.0, min(1.0, cosine(new_rows[oi], new_rows[ci])))))
        print(f"  {KUHUL_TOKENS[oi]:12s} <-> {KUHUL_TOKENS[ci]:12s}  arc={arc:.1f}deg")

    # ── Update brain experts for KUHUL hash slots ─────────────────────────────
    experts_new = experts.copy()
    print("\nUpdating brain expert assignments for KUHUL hash slots:")
    for i, tok in enumerate(KUHUL_TOKENS):
        tid      = KUHUL_START + i
        hash_pos = tid % N
        old_k    = int(experts_new[hash_pos])
        new_k    = assignments[tok]
        experts_new[hash_pos] = new_k
        print(f"  experts[{hash_pos}]  ({tok:15s})  {old_k} -> {new_k}")

    experts_new.tofile(a.out_brain)
    print(f"\nUpdated experts written: {a.out_brain}")

    # ── Write updated safetensors ─────────────────────────────────────────────
    wte_new = wte.copy()
    wte_new[KUHUL_START:KUHUL_START+10] = new_rows

    tensors[wte_key] = wte_new
    meta["kuhul_brain_fold"] = json.dumps({
        "assignments": {tok: int(assignments[tok]) for tok in KUHUL_TOKENS},
        "n_brain_nodes": N,
        "k_experts": K,
    })
    tensors = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()}
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    st_write(a.dst, tensors, metadata=meta)
    mb = Path(a.dst).stat().st_size / 1e6
    print(f"Done: {a.dst}  ({mb:.1f} MB)")

    # ── Print assignment map for gpt2_trainer.cpp reference ──────────────────
    print("\nKUHUL -> brain expert cluster mapping:")
    print("  (paste into loadBrainExperts or a kuhul_experts.h header)")
    for i, tok in enumerate(KUHUL_TOKENS):
        print(f"  // {KUHUL_START+i}  {tok}  -> cluster {assignments[tok]}")


if __name__ == "__main__":
    main()
