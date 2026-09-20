#!/usr/bin/env python3
"""
test_qwen.py — Verify Qwen xshard training and run inference.

Steps:
  1. Read xshard state block → confirm trained shard count
  2. Spot-check weight deltas vs original safetensors
  3. Load Qwen2 via HF transformers, patch weights from trained xshard
  4. Generate response to an AST-structured prompt

Usage:
  python tools/test_qwen.py
  python tools/test_qwen.py --no-patch   # use original weights (delta only)
  python tools/test_qwen.py --prompt "Parse this JSON: {\"type\": \"Program\"}"
"""

import argparse
import json
import struct
import sys
from pathlib import Path

XSHARD_PATH   = Path("E:/models/GPT2/qwen/qwen-f32.xshard")
ST_PATH       = Path("E:/models/GPT2/qwen/model.safetensors")
MODEL_DIR     = Path("E:/models/GPT2/qwen")
MAGIC         = b'XSHD'
ALIGNMENT     = 64

DEFAULT_PROMPT = (
    "Generate a JSON AST node for a function declaration named 'add' "
    "with two integer parameters and a return statement."
)


# ── xshard reader ─────────────────────────────────────────────────────────────

def read_xshard_manifest(path: Path) -> tuple[dict, int, int, bytes]:
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f'Not an xshard file: {magic!r}')
        version, flags = struct.unpack('<HH', f.read(4))
        manifest_len   = struct.unpack('<Q', f.read(8))[0]
        manifest_bytes = f.read(manifest_len)
        manifest       = json.loads(manifest_bytes)

        state_start = manifest['state_start']
        data_start  = manifest['data_start']
        n_shards    = manifest['n_shards']

        f.seek(state_start)
        state_block = f.read(n_shards)

    return manifest, state_start, data_start, state_block


def read_shard_tensor(f, shard: dict, data_start: int) -> bytes:
    offset = data_start + shard['offset']
    f.seek(offset)
    return f.read(shard['nbytes'])


# ── safetensors reader (first-element only) ───────────────────────────────────

def read_st_first_f32(path: Path, tensor_name: str) -> float | None:
    import struct as st
    with open(path, 'rb') as f:
        hlen = st.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen))
        if tensor_name not in header:
            return None
        meta = header[tensor_name]
        if meta.get('dtype') != 'F32':
            return None
        data_base = 8 + hlen
        start = meta['data_offsets'][0]
        f.seek(data_base + start)
        raw = f.read(4)
        return st.unpack('<f', raw)[0]


# ── weight delta spot-check ───────────────────────────────────────────────────

def spot_check_deltas(manifest: dict, data_start: int, xshard_path: Path,
                      st_path: Path, n_sample: int = 8):
    shards = manifest['shards']
    # Pick a spread: first, last, and a few from the middle
    indices = sorted(set([
        0, len(shards) // 4, len(shards) // 2, 3 * len(shards) // 4, len(shards) - 1,
        *range(10, min(40, len(shards)), 6)
    ]))[:n_sample]

    print(f"\n{'tensor':<50} {'orig':>12} {'trained':>12} {'delta':>12}")
    print('-' * 90)

    with open(xshard_path, 'rb') as xf:
        for idx in indices:
            s = shards[idx]
            if s['dtype'] != 'F32':
                continue
            raw = read_shard_tensor(xf, s, data_start)
            xs_val = struct.unpack('<f', raw[:4])[0]
            st_val = read_st_first_f32(st_path, s['tensor_name'])
            delta = (xs_val - st_val) if st_val is not None else float('nan')
            name_short = s['tensor_name'][-48:] if len(s['tensor_name']) > 48 else s['tensor_name']
            print(f"  {name_short:<50} {st_val or 0:>12.6f} {xs_val:>12.6f} {delta:>+12.6f}")


# ── patch HF model with xshard weights ───────────────────────────────────────

def patch_model_from_xshard(model, manifest: dict, data_start: int, xshard_path: Path):
    import torch

    shards = manifest['shards']
    # Group sub-shards by tensor_name
    from collections import defaultdict
    groups = defaultdict(list)
    for s in shards:
        groups[s['tensor_name']].append(s)

    patched = 0
    skipped = 0

    with open(xshard_path, 'rb') as xf:
        with torch.no_grad():
            for tensor_name, group in groups.items():
                group.sort(key=lambda s: s.get('shard_index', 0))

                # Reconstruct full tensor from sub-shards
                parts = []
                for s in group:
                    raw = read_shard_tensor(xf, s, data_start)
                    if s['dtype'] != 'F32':
                        skipped += 1
                        continue
                    shape = s['shape']
                    t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
                    parts.append((s, t))

                if not parts:
                    continue

                if len(parts) == 1:
                    full_tensor = parts[0][1]
                else:
                    axis = parts[0][0].get('shard_axis', 0)
                    full_tensor = torch.cat([p[1] for p in parts], dim=axis)

                # Navigate to the parameter
                try:
                    param = model.get_parameter(tensor_name)
                    param.copy_(full_tensor.to(param.dtype))
                    patched += 1
                except AttributeError:
                    # try as buffer (e.g. attention masks)
                    try:
                        buf = model.get_buffer(tensor_name)
                        buf.copy_(full_tensor.to(buf.dtype))
                        patched += 1
                    except Exception:
                        skipped += 1
                except Exception:
                    skipped += 1

    print(f"  patched={patched}  skipped={skipped}")
    return patched


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(model_dir: Path, prompt: str, patch: bool,
                  manifest: dict, data_start: int, xshard_path: Path,
                  max_new_tokens: int = 200):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("Installing transformers + torch ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "transformers", "torch", "-q"])
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

    print(f"\n[load] tokenizer from {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print(f"[load] model (cpu / float32) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if patch:
        print("[patch] applying trained xshard weights ...")
        patched = patch_model_from_xshard(model, manifest, data_start, xshard_path)
        print(f"  {patched} tensors updated from xshard")
    else:
        print("[patch] skipped — using original safetensors weights")

    print(f"\n[prompt] {prompt[:80]}{'...' if len(prompt)>80 else ''}")
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )
    response = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print("\n[response]")
    print(response)
    return response


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xshard",    default=str(XSHARD_PATH))
    ap.add_argument("--safetensors", default=str(ST_PATH))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--prompt",    default=DEFAULT_PROMPT)
    ap.add_argument("--no-patch",  action="store_true",
                    help="Run inference with original weights (skip xshard patch)")
    ap.add_argument("--no-infer",  action="store_true",
                    help="Only run weight checks, skip inference")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    xshard_path = Path(args.xshard)
    st_path     = Path(args.safetensors)
    model_dir   = Path(args.model_dir)

    # ── 1. Read xshard state ──────────────────────────────────────────────────
    print(f"[xshard] reading {xshard_path.name} ...")
    manifest, state_start, data_start, state_block = read_xshard_manifest(xshard_path)

    n_shards  = manifest['n_shards']
    n_trained = sum(1 for b in state_block if b == 0x01)
    n_pending = sum(1 for b in state_block if b == 0x00)
    total_mb  = xshard_path.stat().st_size / 1024 / 1024

    print(f"  shards={n_shards}  trained={n_trained}  pending={n_pending}  file={total_mb:.1f} MB")

    from collections import Counter
    fold_counts = Counter(s['fold'] for s in manifest['shards'])
    print("  fold distribution:", dict(fold_counts))

    # ── 2. Spot-check weight deltas ───────────────────────────────────────────
    if st_path.exists():
        print(f"\n[delta] comparing xshard weights vs {st_path.name} ...")
        spot_check_deltas(manifest, data_start, xshard_path, st_path)
    else:
        print(f"[delta] {st_path} not found — skipping delta check")

    # ── 3. Inference ──────────────────────────────────────────────────────────
    if not args.no_infer:
        run_inference(
            model_dir=model_dir,
            prompt=args.prompt,
            patch=not args.no_patch,
            manifest=manifest,
            data_start=data_start,
            xshard_path=xshard_path,
            max_new_tokens=args.max_tokens,
        )
    else:
        print("\n[infer] skipped (--no-infer)")


if __name__ == "__main__":
    main()
