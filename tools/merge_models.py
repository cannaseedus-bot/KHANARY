"""merge_models.py -- SLERP / linear merge of two same-arch safetensors checkpoints.

Usage:
    python tools/merge_models.py \
        models/from_zero/from_zero_v0.4_phase1.safetensors \
        models/from_zero/from_zero_v0.5_phase2.safetensors \
        models/from_zero/from_zero_v0.6_merged.safetensors \
        --alpha 0.6 --method slerp

    alpha = 0.0 -> pure model_a
    alpha = 1.0 -> pure model_b

Vocab mismatch handling:
    If the two models differ in wte / lm_head vocab dimension (e.g. 50257 vs 50270),
    only the shared rows are interpolated. The extra rows from model_b are kept as-is.
    This lets you merge a pretrained 50257-vocab GPT-2 with a KUHUL-extended checkpoint.
"""

import argparse
import sys
import math
import torch
from safetensors.torch import load_file, save_file


# --- tensor merge primitives ---

def slerp_tensor(t: float, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors, treating each as a
    flat vector on the unit hypersphere.  Falls back to linear when the angle
    is negligible (parallel / anti-parallel vectors).
    """
    af = a.float().flatten()
    bf = b.float().flatten()

    norm_a = af.norm()
    norm_b = bf.norm()

    if norm_a < 1e-8 or norm_b < 1e-8:
        # One tensor is zero -- linear fallback
        result = (1.0 - t) * a.float() + t * b.float()
        return result.to(a.dtype).reshape(a.shape)

    # Interpolate directions, then rescale by interpolated magnitude
    ua = af / norm_a
    ub = bf / norm_b

    dot = (ua * ub).sum().clamp(-1.0, 1.0)
    theta = dot.acos()

    if theta.abs() < 1e-6:
        # Nearly parallel -- linear on the direction, slerp magnitude
        direction = (1.0 - t) * ua + t * ub
    else:
        s0 = math.sin((1.0 - t) * theta.item()) / math.sin(theta.item())
        s1 = math.sin(t          * theta.item()) / math.sin(theta.item())
        direction = s0 * ua + s1 * ub

    mag = (1.0 - t) * norm_a + t * norm_b
    result = (direction * mag).reshape(a.shape)
    return result.to(a.dtype)


def linear_tensor(t: float, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((1.0 - t) * a.float() + t * b.float()).to(a.dtype)


# --- vocab-aware merge for embedding / lm_head ---

VOCAB_KEYS = {"model.embed_tokens.weight", "lm_head.weight",
              "wte.weight", "transformer.wte.weight"}

def merge_tensor(key: str, t: float, a: torch.Tensor, b: torch.Tensor,
                 method: str) -> torch.Tensor:
    """
    Merge two tensors.  For embedding / lm_head keys, handle vocab-size
    differences by merging only shared rows and appending extra rows from b.
    """
    fn = slerp_tensor if method == "slerp" else linear_tensor

    # Check if this is a vocab-dimension tensor that might differ
    vocab_key = any(k in key for k in VOCAB_KEYS)
    if vocab_key and a.shape != b.shape:
        # Assume dim-0 is vocab
        shared = min(a.shape[0], b.shape[0])
        merged_shared = fn(t, a[:shared], b[:shared])
        if b.shape[0] > shared:
            # Keep extra rows from b (the extended KUHUL tokens)
            extra = b[shared:]
            return torch.cat([merged_shared, extra], dim=0)
        else:
            return merged_shared

    if a.shape != b.shape:
        print(f"  [warn] shape mismatch for {key}: {a.shape} vs {b.shape} -- keeping a")
        return a

    # Scalars and 1-element tensors: linear only
    if a.numel() <= 1:
        return linear_tensor(t, a, b)

    return fn(t, a, b)


# --- main ---

def main():
    parser = argparse.ArgumentParser(description="Merge two safetensors checkpoints")
    parser.add_argument("model_a", help="Path to model A (alpha=0)")
    parser.add_argument("model_b", help="Path to model B (alpha=1)")
    parser.add_argument("output",  help="Output safetensors path")
    parser.add_argument("--alpha",  type=float, default=0.5,
                        help="Blend weight toward model_b (0.0=pure A, 1.0=pure B). Default 0.5")
    parser.add_argument("--method", choices=["slerp", "linear"], default="slerp",
                        help="Interpolation method (default: slerp)")
    parser.add_argument("--keys",   nargs="*",
                        help="Restrict merge to specific tensor key(s); others are copied from A")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        sys.exit("--alpha must be in [0.0, 1.0]")

    print(f"[merge] loading A: {args.model_a}")
    sd_a = load_file(args.model_a)
    print(f"[merge] loading B: {args.model_b}")
    sd_b = load_file(args.model_b)

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = keys_a & keys_b

    if only_a:
        print(f"[merge] {len(only_a)} keys only in A -- kept as-is: {sorted(only_a)[:5]}...")
    if only_b:
        print(f"[merge] {len(only_b)} keys only in B -- appended: {sorted(only_b)[:5]}...")

    restrict = set(args.keys) if args.keys else None

    merged = {}
    total = len(common)
    for i, key in enumerate(sorted(common), 1):
        a_t = sd_a[key]
        b_t = sd_b[key]

        if restrict is not None and key not in restrict:
            merged[key] = a_t
            continue

        m = merge_tensor(key, args.alpha, a_t, b_t, args.method)
        merged[key] = m

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] {key}  {tuple(a_t.shape)} alpha={args.alpha:.2f}")

    # Keys only in A or B
    for key in only_a:
        merged[key] = sd_a[key]
    for key in only_b:
        merged[key] = sd_b[key]

    print(f"[merge] saving {len(merged)} tensors -> {args.output}")
    save_file(merged, args.output)

    # Quick sanity: param norm delta
    sample_keys = [k for k in sorted(common) if "weight" in k][:5]
    print("\n[merge] sanity check -- weight norm comparison:")
    print(f"  {'key':<45} {'norm_a':>10} {'norm_b':>10} {'norm_m':>10}")
    for k in sample_keys:
        na = sd_a[k].float().norm().item()
        nb = sd_b[k].float().norm().item()
        nm = merged[k].float().norm().item()
        print(f"  {k:<45} {na:>10.4f} {nb:>10.4f} {nm:>10.4f}")

    print("\n[merge] done.")


if __name__ == "__main__":
    main()
