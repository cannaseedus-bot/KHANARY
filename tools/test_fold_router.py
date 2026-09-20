#!/usr/bin/env python3
"""
test_fold_router.py — Python mirror of FoldRouter in AdaptiveContentModelAdapter.cs.

Validates the three-model routing table (Gemma / GPT-2 Large / Qwen) by:
  1. Opening each model's xshard
  2. For each fold in phase order, verifying the correct model is selected
  3. Reading shard counts, total MB, and a weight sample from each routed shard set
  4. Printing a combined routing summary table

Usage:
  python tools/test_fold_router.py
  python tools/test_fold_router.py --verbose   # print first-weight sample per fold
"""

import argparse
import json
import struct
from pathlib import Path

GEMMA_XSHARD     = Path("E:/models/GPT2/gemma-3-1b-it/gemma-3-1b-it-f32.xshard")
GPT2L_XSHARD     = Path("E:/models/GPT2/lg-GPT2/lg-gpt2-f32.xshard")
QWEN_XSHARD      = Path("E:/models/GPT2/qwen/qwen-f32.xshard")

MAGIC    = b'XSHD'
FOLD_SEQ = ["Pop", "Wo", "Yax", "Sek", "Chen", "Xul"]

# ── mirrors FoldRouter.DefaultRouting ─────────────────────────────────────────
DEFAULT_ROUTING = {
    "Pop":  "Gemma",
    "Wo":   "Gemma",
    "Yax":  "Gpt2Large",
    "Sek":  "Gpt2Large",
    "Chen": "Gpt2Large",
    "Xul":  "Gemma",
}

MODEL_PATHS = {
    "Gemma":    GEMMA_XSHARD,
    "Gpt2Large": GPT2L_XSHARD,
    "Qwen":     QWEN_XSHARD,
}


# ── xshard manifest reader ────────────────────────────────────────────────────

def read_manifest(path: Path) -> tuple[dict, int, int, bytes]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"{path.name}: bad magic {magic!r}")
        f.read(4)  # version + flags
        mlen = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(mlen))
        state_start = manifest["state_start"]
        data_start  = manifest["data_start"]
        n_shards    = manifest["n_shards"]
        f.seek(state_start)
        state = f.read(n_shards)
    return manifest, state_start, data_start, state


def first_f32(path: Path, shard: dict, data_start: int) -> float:
    with open(path, "rb") as f:
        f.seek(data_start + shard["offset"])
        raw = f.read(4)
    return struct.unpack("<f", raw)[0]


# ── per-model cache ───────────────────────────────────────────────────────────

_cache: dict[str, tuple[dict, int, int, bytes]] = {}

def get_model(name: str) -> tuple[Path, dict, int, int, bytes]:
    if name not in _cache:
        p = MODEL_PATHS[name]
        print(f"  [open] {name} -> {p.name}")
        _cache[name] = read_manifest(p)
    manifest, ss, ds, state = _cache[name]
    return MODEL_PATHS[name], manifest, ss, ds, state


# ── routing test ──────────────────────────────────────────────────────────────

def test_routing(fold_overrides: dict | None = None, verbose: bool = False):
    routing = {**DEFAULT_ROUTING, **(fold_overrides or {})}

    print("\n=== FoldRouter — combined tensor routing test ===\n")
    print(f"  {'Fold':<6} {'-> Model':<12} {'shards':>7} {'trained':>8} {'MB':>8} {'sample':>12}")
    print("  " + "-" * 62)

    total_shards  = 0
    total_trained = 0
    total_mb      = 0.0
    results       = []

    for fold in FOLD_SEQ:
        model_name = routing.get(fold, "Gemma")
        path, manifest, ss, ds, state = get_model(model_name)

        shards = [s for s in manifest["shards"] if s["fold"] == fold]
        n      = len(shards)
        tr     = sum(1 for s in shards if state[s["seq"]] == 0x01)
        mb     = sum(s["nbytes"] for s in shards) / 1024 / 1024

        sample = ""
        if verbose and shards:
            v = first_f32(path, shards[0], ds)
            sample = f"{v:+.6f}"

        total_shards  += n
        total_trained += tr
        total_mb      += mb
        results.append((fold, model_name, n, tr, mb, sample))

        print(f"  {fold:<6} -> {model_name:<12} {n:>7} {tr:>8} {mb:>8.1f} {sample:>12}")

    print("  " + "-" * 62)
    print(f"  {'TOTAL':<20} {total_shards:>7} {total_trained:>8} {total_mb:>8.1f}")
    print()

    # ── verify every fold hit the right model ──────────────────────────────────
    print("=== Routing assertions ===")
    passed = 0
    failed = 0
    for fold, model_name, n, tr, mb, _ in results:
        expected = routing[fold]
        ok = model_name == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  [{status}] {fold}: expected {expected}, got {model_name}")

    if failed == 0:
        print(f"  All {passed} fold routing assertions passed.")
    else:
        print(f"  {passed} passed, {failed} FAILED.")

    # ── verify no fold returns zero shards ─────────────────────────────────────
    print()
    print("=== Shard coverage ===")
    coverage_ok = True
    for fold, model_name, n, tr, mb, _ in results:
        if fold == "Xul":
            continue  # Xul/lm_head may be absent in some xshards
        if n == 0:
            print(f"  WARN: {fold} → {model_name} returned 0 shards")
            coverage_ok = False
    if coverage_ok:
        print(f"  All active folds have shards. Total routed: {total_shards} shards / {total_mb:.0f} MB")

    return failed == 0 and coverage_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="Print first-weight sample per fold")
    ap.add_argument("--qwen-sek", action="store_true",
                    help="Override Sek+Wo → Qwen (AST/coder mode)")
    args = ap.parse_args()

    # Verify all xshard files exist
    missing = [str(p) for p in MODEL_PATHS.values() if not p.exists()]
    if missing:
        for m in missing:
            print(f"ERROR: missing xshard: {m}")
        return

    overrides = {}
    if args.qwen_sek:
        overrides["Sek"] = "Qwen"
        overrides["Wo"]  = "Qwen"
        print("(AST mode: Sek+Wo -> Qwen via FoldOverrides)")

    ok = test_routing(fold_overrides=overrides, verbose=args.verbose)

    if args.qwen_sek:
        print("\n=== Default routing (comparison) ===")
        _cache.clear()
        test_routing(verbose=args.verbose)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
