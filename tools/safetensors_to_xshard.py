#!/usr/bin/env python3
"""
safetensors_to_xshard.py -- Convert a .safetensors model to XSHARD/1 format.

XSHARD/1 is a fold-tagged, streaming-native binary weight format where each
shard is a self-contained training unit.  The streaming pattern is baked into
the format rather than bolted on top of SafeTensors at runtime.

Usage:
  python tools/safetensors_to_xshard.py \\
      --input  model.safetensors \\
      --output model.xshard \\
      [--max-shard-mb 256]   # sub-shard tensors larger than this
      [--shard-axis 0]       # axis to split along (default: 0, keeps data contiguous)
      [--arch gpt2]          # architecture hint for fold assignment
      [--model <name>]       # model identifier written into manifest
      [--dry-run]            # print plan without writing
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

MAGIC       = b'XSHD'
FOOTER_MAGIC = 0x44485358   # "XSHD" bytes (58 53 48 44) read as LE u32
VERSION     = 1
ALIGNMENT   = 64

DTYPE_SIZES = {
    'F32':  4,
    'BF16': 2,
    'F16':  2,
    'I8':   1,
    'INT8': 1,
    'U8':   1,
    'I16':  2,
    'I32':  4,
    'I64':  8,
    'F64':  8,
}

# SafeTensors dtype → XSHARD dtype
DTYPE_MAP = {
    'F32':  'F32',
    'BF16': 'BF16',
    'F16':  'F16',
    'I8':   'INT8',
    'U8':   'INT8',
}

# ── Fold assignment ────────────────────────────────────────────────────────────
# Matched in order; first hit wins.

import re

_FOLD_PATTERNS = [
    # (compiled_regex, fold, phase_angle)
    # ── HuggingFace / GPT-2 naming ────────────────────────────────────────────
    (re.compile(r'(^|\.)(wte|wpe|embed|token_embed|pos_embed)(\.|\b)'),
     'Pop', 0.0),
    (re.compile(r'(^|\.)mlp\.(c_fc|fc_1|gate_proj|gate|up_proj)(\.|\b)'),
     'Wo', math.pi / 3),
    (re.compile(r'(^|\.)mlp\.(c_proj|fc_2|down|down_proj)(\.|\b)'),
     'Yax', 2 * math.pi / 3),
    # HF self_attn: Gemma, Qwen, LLaMA, Mistral — must precede plain attn pattern
    (re.compile(r'(^|\.)self_attn\.(q_proj|k_proj|v_proj|o_proj|out_proj)(\.|\b)'),
     'Sek', math.pi),
    (re.compile(r'(^|\.)attn\.(c_attn|c_proj|q|k|v|q_proj|k_proj|v_proj|o_proj|out_proj)(\.|\b)'),
     'Sek', math.pi),
    # ── GGUF / llama.cpp naming (blk.N.*) ─────────────────────────────────────
    (re.compile(r'(^|\.)token_embd(\.|\b)'),
     'Pop', 0.0),
    (re.compile(r'(^|\.)blk\.\d+\.(ffn_up|ffn_gate)(\.|\b)'),
     'Wo', math.pi / 3),
    (re.compile(r'(^|\.)blk\.\d+\.ffn_down(\.|\b)'),
     'Yax', 2 * math.pi / 3),
    (re.compile(r'(^|\.)blk\.\d+\.(attn_q|attn_k|attn_v|attn_output)(\.|\b)'),
     'Sek', math.pi),
    # ── Normalisation (both naming conventions) ────────────────────────────────
    # HF: .norm. .ln_f. etc.  GGUF: attn_norm, ffn_norm, output_norm (*_norm)
    # Also compound names: post_attention_layernorm, post_feedforward_layernorm
    (re.compile(r'(^|\.)(ln_[0-9f]|ln_f|layernorm|layer_norm|norm)(\.|\b)'),
     'Chen', 4 * math.pi / 3),
    (re.compile(r'layernorm(\.|\b)'),
     'Chen', 4 * math.pi / 3),
    (re.compile(r'_norm(\.|\b)'),
     'Chen', 4 * math.pi / 3),
    (re.compile(r'(^|\.)bias(\.|\b)'),
     'Chen', 4 * math.pi / 3),
    # ── Output head ────────────────────────────────────────────────────────────
    (re.compile(r'(^|\.)(lm_head|output\.weight)(\.|\b)'),
     'Xul', 5 * math.pi / 3),
]

_PHASE_ANGLES = {
    'Pop':  0.0,
    'Wo':   math.pi / 3,
    'Yax':  2 * math.pi / 3,
    'Sek':  math.pi,
    'Chen': 4 * math.pi / 3,
    'Xul':  5 * math.pi / 3,
}


def assign_fold(tensor_name: str) -> tuple[str, float]:
    name = tensor_name.lower()
    for pattern, fold, angle in _FOLD_PATTERNS:
        if pattern.search(name):
            return fold, round(angle, 6)
    return 'Pop', 0.0   # fallback: observe


# ── SafeTensors reader ─────────────────────────────────────────────────────────

def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Returns (header_dict, data_base_offset)."""
    with open(path, 'rb') as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError('Not a safetensors file (too short)')
        hlen = struct.unpack('<Q', raw_len)[0]
        header_bytes = f.read(hlen)
    header = json.loads(header_bytes)
    return header, 8 + hlen


def nbytes_for_shape(shape: list[int], dtype: str) -> int:
    n = DTYPE_SIZES.get(dtype, 4)
    for d in shape:
        n *= d
    return n


def xshard_dtype(st_dtype: str) -> str:
    mapped = DTYPE_MAP.get(st_dtype)
    if mapped is None:
        raise ValueError(f'Unsupported dtype for XSHARD: {st_dtype!r}')
    return mapped


# ── Padding helpers ────────────────────────────────────────────────────────────

def pad_up(n: int, align: int = ALIGNMENT) -> int:
    return (n + align - 1) // align * align


# ── Shard planning ─────────────────────────────────────────────────────────────

def plan_shards(header: dict, max_shard_bytes: int, shard_axis: int) -> list[dict]:
    """
    Returns a list of shard plan dicts (no offset/sha256 yet — those need the
    data).  Skips __metadata__ and any unsupported dtypes with a warning.
    """
    shards = []
    seq = 0

    for tensor_name, meta in sorted(header.items()):
        if tensor_name == '__metadata__':
            continue

        st_dtype = meta.get('dtype', 'F32')
        try:
            dtype = xshard_dtype(st_dtype)
        except ValueError as e:
            print(f'  [skip] {tensor_name}: {e}', file=sys.stderr)
            continue

        shape   = meta['shape']
        offsets = meta['data_offsets']   # [start, end] relative to data area
        total_nbytes = offsets[1] - offsets[0]
        fold, angle = assign_fold(tensor_name)

        if total_nbytes <= max_shard_bytes or len(shape) < 2:
            # Single shard — whole tensor
            shards.append({
                'id':           _shard_id(tensor_name),
                'seq':          seq,
                'tensor_name':  tensor_name,
                'fold':         fold,
                'phase_angle':  angle,
                'shape':        shape,
                'dtype':        dtype,
                'st_offsets':   offsets,      # ephemeral — used during write
                'shard_of':     tensor_name,
                'shard_index':  0,
                'shard_count':  1,
                'shard_axis':   None,
                'shard_slice':  None,
            })
            seq += 1
        else:
            # Sub-shard along shard_axis
            axis   = min(shard_axis, len(shape) - 1)
            n_rows = shape[axis]
            elem_size = DTYPE_SIZES.get(dtype, 4)
            row_bytes = total_nbytes // n_rows
            rows_per_shard = max(1, max_shard_bytes // row_bytes)
            n_splits = math.ceil(n_rows / rows_per_shard)

            for split_i in range(n_splits):
                row_start = split_i * rows_per_shard
                row_end   = min(row_start + rows_per_shard, n_rows)
                sub_shape = list(shape)
                sub_shape[axis] = row_end - row_start

                shards.append({
                    'id':           f'{_shard_id(tensor_name)}.s{split_i}',
                    'seq':          seq,
                    'tensor_name':  tensor_name,
                    'fold':         fold,
                    'phase_angle':  angle,
                    'shape':        sub_shape,
                    'dtype':        dtype,
                    'st_offsets':   offsets,
                    'shard_of':     tensor_name,
                    'shard_index':  split_i,
                    'shard_count':  n_splits,
                    'shard_axis':   axis,
                    'shard_slice':  [row_start, row_end],
                    '_row_bytes':   row_bytes,   # ephemeral
                })
                seq += 1

    return shards


def _shard_id(tensor_name: str) -> str:
    """e.g. transformer.h.0.attn.c_attn.weight → h.0.attn.c_attn"""
    parts = tensor_name.split('.')
    # strip common prefixes and trailing .weight/.bias
    skip = {'transformer', 'model', 'weight', 'bias'}
    kept = [p for p in parts if p not in skip]
    return '.'.join(kept) if kept else tensor_name


# ── Manifest construction ──────────────────────────────────────────────────────

def build_manifest(
    shards: list[dict],
    model_name: str,
    arch: str,
    header: dict,
    state_start: int = 0,
    data_start: int = 0,
) -> dict:
    meta = header.get('__metadata__', {})

    shard_records = []
    for s in shards:
        rec = {
            'id':           s['id'],
            'seq':          s['seq'],
            'tensor_name':  s['tensor_name'],
            'fold':         s['fold'],
            'phase_angle':  s['phase_angle'],
            'shape':        s['shape'],
            'dtype':        s['dtype'],
            'offset':       s.get('offset', 0),
            'nbytes':       s.get('nbytes', 0),
            'sha256':       s.get('sha256', '0' * 64),
            'shard_of':     s['shard_of'],
            'shard_index':  s['shard_index'],
            'shard_count':  s['shard_count'],
        }
        if s['shard_axis'] is not None:
            rec['shard_axis']  = s['shard_axis']
            rec['shard_slice'] = s['shard_slice']
        shard_records.append(rec)

    has_sub = any(s['shard_count'] > 1 for s in shards)
    flags = 0x01 | 0x04    # FOLD_TAGGED | STATE_MUTABLE
    if has_sub:
        flags |= 0x02       # SUB_SHARDED

    n_layer  = int(meta.get('n_layer', 0))
    n_embd   = int(meta.get('n_embd', 0))
    n_head   = int(meta.get('n_head', 0))
    vocab    = int(meta.get('vocab_size', 0))

    manifest = {
        '@kind':        'xshard/1',
        'version':      1,
        'created':      datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model':        model_name,
        'arch':         arch,
        'flags':        flags,
        'alignment':    ALIGNMENT,
        'n_shards':     len(shards),
        'state_start':  state_start,
        'data_start':   data_start,
        'shards':       shard_records,
    }
    if n_layer: manifest['n_layer']    = n_layer
    if n_embd:  manifest['n_embd']     = n_embd
    if n_head:  manifest['n_head']     = n_head
    if vocab:   manifest['vocab_size'] = vocab

    return manifest


def serialize_manifest(manifest: dict) -> bytes:
    return json.dumps(manifest, separators=(',', ':')).encode('utf-8')


def resolve_offsets(manifest: dict, shards: list[dict]) -> tuple[int, int]:
    """
    Given a serialized manifest size, compute state_start and data_start.
    Iterates until convergence (≤2 passes in practice).
    """
    for _ in range(3):
        mbytes = serialize_manifest(manifest)
        state_start = pad_up(16 + len(mbytes))
        data_start  = pad_up(state_start + len(shards))
        if manifest['state_start'] == state_start and manifest['data_start'] == data_start:
            break
        manifest['state_start'] = state_start
        manifest['data_start']  = data_start
    return state_start, data_start


# ── Binary writer ──────────────────────────────────────────────────────────────

def write_xshard(
    output_path: Path,
    input_path: Path,
    shards: list[dict],
    manifest: dict,
    data_start: int,
    state_start: int,
    dry_run: bool = False,
) -> None:
    manifest_bytes = serialize_manifest(manifest)
    n_shards = len(shards)

    if dry_run:
        print(f'  manifest_len  : {len(manifest_bytes):,} bytes')
        print(f'  state_start   : {state_start:,}')
        print(f'  data_start    : {data_start:,}')
        total = data_start + sum(pad_up(s['nbytes']) for s in shards) + 8
        print(f'  estimated size: {total:,} bytes ({total/1024/1024:.1f} MB)')
        return

    with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
        # ── Header ────────────────────────────────────────────────────────────
        dst.write(MAGIC)
        dst.write(struct.pack('<H', VERSION))
        dst.write(struct.pack('<H', manifest['flags']))
        dst.write(struct.pack('<Q', len(manifest_bytes)))
        dst.write(manifest_bytes)

        # pad to state_start
        cur = 16 + len(manifest_bytes)
        dst.write(b'\x00' * (state_start - cur))

        # ── State block ───────────────────────────────────────────────────────
        dst.write(bytes([0x00] * n_shards))
        cur = state_start + n_shards

        # pad to data_start
        dst.write(b'\x00' * (data_start - cur))

        # ── Shard data ────────────────────────────────────────────────────────
        data_base = 8 + (struct.unpack('<Q', open(input_path, 'rb').read(8))[0])
        running_offset = 0

        for i, s in enumerate(shards):
            st_start, st_end = s['st_offsets']

            if s['shard_slice'] is None:
                # whole tensor
                src.seek(data_base + st_start)
                raw = src.read(st_end - st_start)
            else:
                # sub-shard: contiguous slice along shard_axis=0
                row_bytes  = s['_row_bytes']
                slice_start, slice_end = s['shard_slice']
                byte_off   = slice_start * row_bytes
                byte_len   = (slice_end - slice_start) * row_bytes
                src.seek(data_base + st_start + byte_off)
                raw = src.read(byte_len)

            sha256 = hashlib.sha256(raw).hexdigest()

            # patch manifest shard record in memory for meta.json
            s['sha256'] = sha256
            s['offset'] = running_offset
            s['nbytes'] = len(raw)
            manifest['shards'][i]['sha256'] = sha256
            manifest['shards'][i]['offset'] = running_offset
            manifest['shards'][i]['nbytes'] = len(raw)

            dst.write(raw)
            padded = pad_up(len(raw))
            dst.write(b'\x00' * (padded - len(raw)))
            running_offset += padded

            if (i + 1) % 10 == 0 or i + 1 == n_shards:
                pct = (i + 1) / n_shards * 100
                print(f'  [{i+1:4d}/{n_shards}] {pct:5.1f}%  {s["id"]}  fold={s["fold"]}',
                      flush=True)

        # ── Rewrite manifest with real sha256s ────────────────────────────────
        # SHA-256 hex is always 64 chars; placeholder '0'*64 is also 64 chars,
        # so manifest byte-length is invariant — safe to seek back and overwrite.
        final_manifest_bytes = serialize_manifest(manifest)
        assert len(final_manifest_bytes) == len(manifest_bytes), \
            f'Manifest size changed during write ({len(manifest_bytes)} → {len(final_manifest_bytes)}) — bug'
        dst.seek(16)
        dst.write(final_manifest_bytes)
        dst.seek(0, 2)   # back to EOF for footer

        # ── Footer ────────────────────────────────────────────────────────────
        dst.write(struct.pack('<I', n_shards))
        dst.write(struct.pack('<I', FOOTER_MAGIC))

    print(f'  wrote {output_path}')


def write_meta_json(output_path: Path, manifest: dict) -> None:
    meta_path = output_path.with_suffix(output_path.suffix + '.meta.json')
    meta = {
        '@kind':  'xshard.meta/1',
        'model':  manifest['model'],
        'source': manifest.get('arch', ''),
        'passes': [],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'  wrote {meta_path}')


# ── Fold summary ───────────────────────────────────────────────────────────────

def print_fold_summary(shards: list[dict]) -> None:
    from collections import Counter
    counts  = Counter(s['fold'] for s in shards)
    mb      = {f: sum(nbytes_for_shape(s['shape'], s['dtype'])
                      for s in shards if s['fold'] == f) / 1024 / 1024
               for f in counts}
    print('  fold distribution:')
    for fold in ['Pop', 'Wo', 'Yax', 'Sek', 'Chen', 'Xul']:
        if fold in counts:
            angle = _PHASE_ANGLES[fold]
            print(f'    {fold:<5} angle={angle:.4f}rad  '
                  f'shards={counts[fold]:3d}  '
                  f'data={mb[fold]:7.1f} MB')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Convert .safetensors → XSHARD/1 streaming shard format')
    ap.add_argument('--input',        required=True,  help='Input .safetensors file')
    ap.add_argument('--output',       required=True,  help='Output .xshard file')
    ap.add_argument('--max-shard-mb', type=float, default=256.0,
                    help='Sub-shard tensors larger than this (MB). Default: 256')
    ap.add_argument('--shard-axis',   type=int,   default=0,
                    help='Axis to split along when sub-sharding. Default: 0')
    ap.add_argument('--arch',         default='gpt2',
                    help='Architecture hint for fold assignment. Default: gpt2')
    ap.add_argument('--model',        default='',
                    help='Model identifier written into manifest')
    ap.add_argument('--dry-run',      action='store_true',
                    help='Print plan without writing')
    args = ap.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    max_bytes   = int(args.max_shard_mb * 1024 * 1024)
    model_name  = args.model or input_path.stem

    if not input_path.exists():
        print(f'error: {input_path} not found', file=sys.stderr)
        sys.exit(1)

    print(f'[xshard] reading {input_path}')
    header, data_base_unused = read_safetensors_header(input_path)
    n_tensors = sum(1 for k in header if k != '__metadata__')
    print(f'  {n_tensors} tensors in header')

    print('[xshard] planning shards ...')
    shards = plan_shards(header, max_bytes, args.shard_axis)
    n_shards = len(shards)
    sub_count = sum(1 for s in shards if s['shard_count'] > 1)
    print(f'  {n_shards} shards ({n_tensors} tensors'
          + (f', {sub_count} sub-sharded' if sub_count else '') + ')')
    print_fold_summary(shards)

    # ── Provisional manifest (placeholders for offsets) ────────────────────────
    manifest = build_manifest(shards, model_name, args.arch, header, 0, 0)

    # ── Assign shard data sizes (needed for offset computation) ───────────────
    for s in shards:
        st_start, st_end = s['st_offsets']
        if s['shard_slice'] is None:
            nbytes = st_end - st_start
        else:
            row_bytes = s.get('_row_bytes')
            if row_bytes is None:
                # compute from shape
                elem = DTYPE_SIZES.get(s['dtype'], 4)
                row_bytes = elem
                for d in s['shape'][1:]:
                    row_bytes *= d
                s['_row_bytes'] = row_bytes
            slice_start, slice_end = s['shard_slice']
            nbytes = (slice_end - slice_start) * row_bytes
        s['nbytes'] = nbytes

    # ── Assign shard offsets (relative to data_start) ─────────────────────────
    running = 0
    for s in shards:
        s['offset'] = running
        running += pad_up(s['nbytes'])

    # ── Populate manifest with sizes (sha256 patched during write) ────────────
    for i, s in enumerate(shards):
        manifest['shards'][i]['offset'] = s['offset']
        manifest['shards'][i]['nbytes'] = s['nbytes']

    # ── Resolve state_start / data_start (convergence loop) ──────────────────
    state_start, data_start = resolve_offsets(manifest, shards)

    print(f'[xshard] layout: '
          f'manifest={len(serialize_manifest(manifest)):,}B  '
          f'state_start={state_start:,}  '
          f'data_start={data_start:,}')

    if args.dry_run:
        print('[xshard] dry-run — no files written')
        write_xshard(output_path, input_path, shards, manifest,
                     data_start, state_start, dry_run=True)
        return

    print(f'[xshard] writing {output_path} ...')
    write_xshard(output_path, input_path, shards, manifest,
                 data_start, state_start, dry_run=False)

    # ── Rewrite manifest with final sha256s ───────────────────────────────────
    # (offsets + sha256s were patched in-place on the manifest dict during write)
    # The binary manifest is already written; meta.json is the authoritative log.
    write_meta_json(output_path, manifest)

    total_bytes = output_path.stat().st_size
    print(f'[xshard] done — {total_bytes:,} bytes ({total_bytes/1024/1024:.1f} MB)')
    print(f'         {n_shards} shards, {sub_count} sub-sharded tensors')


if __name__ == '__main__':
    main()
