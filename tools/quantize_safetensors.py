# quantize_safetensors.py — portable dual-quant of ANY safetensors model into lean, mmap-able
# KHANARY quant containers (Q8 per-channel INT8 + Q4 group-wise 4-bit).
#
# Deterministic: same source -> byte-identical artifacts -> same sha256, on any machine (local or
# cloud). That is what makes "build in the cloud" and "ship build instructions" the SAME thing:
# the sha256 in the manifest is the verification gate for both.
#
#   python tools/quantize_safetensors.py --src <model.safetensors | HF model dir> --out <dir>
#   (no args -> defaults reproduce this machine's Qwen-1.8B artifacts byte-for-byte)
#
# Handles: single-file OR sharded (*.safetensors + model.safetensors.index.json) HF layouts, and
# F16 / BF16 / F32 source tensors. Scheme:
#   Q8  per-output-channel symmetric INT8 (scale = max|row|/127)
#   Q4  group-wise 4-bit (scale = max|group|/8, 2 nibbles/byte); 1D tensors kept F16.
# Resident-tier design + honest scope: see docs/QUANT_BUILD.md and proof/qwen_quant_v1/.
import os, sys, json, struct, mmap, argparse
import numpy as np

DEF_SRC = r"C:\Users\canna\.lmstudio\models\Qwen-1_8B-Chat-f16\model.safetensors"
DEF_OUT = r"E:\models\Qwen1.8B-quant"
DEF_NAME = "qwen1_8b"

def st_header(mm):
    n = struct.unpack('<Q', mm[:8])[0]
    return json.loads(mm[8:8+n].decode('utf-8')), 8 + n

def to_f32(buf, dtype, shape):
    if dtype == 'F16':  a = np.frombuffer(buf, np.float16).astype(np.float32)
    elif dtype == 'F32': a = np.frombuffer(buf, np.float32).copy()
    elif dtype == 'BF16':                                   # top 16 bits of f32
        a = (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else: raise ValueError(f"unsupported dtype {dtype}")
    return a.reshape(shape)

class Source:
    """Streams tensors (dtype, shape, f32 array) from a single file or a sharded HF dir."""
    def __init__(self, src):
        self.handles = {}   # path -> (fh, mm, header, base)
        if os.path.isdir(src):
            idx = os.path.join(src, "model.safetensors.index.json")
            if os.path.exists(idx):
                wm = json.load(open(idx))["weight_map"]
                self.tmap = {name: os.path.join(src, shard) for name, shard in wm.items()}
            else:
                one = os.path.join(src, "model.safetensors")
                if not os.path.exists(one): raise SystemExit(f"[err] no safetensors in {src}")
                self.tmap = None; self.single = one
        else:
            self.tmap = None; self.single = src
        if self.tmap is None:
            self._open(self.single)
            hdr = self.handles[self.single][2]
            self.tmap = {k: self.single for k in hdr if k != '__metadata__'}
    def _open(self, path):
        if path in self.handles: return
        fh = open(path, 'rb'); mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        hdr, base = st_header(mm); self.handles[path] = (fh, mm, hdr, base)
    def names(self):
        return sorted(self.tmap.keys())
    def get(self, name):
        path = self.tmap[name]; self._open(path)
        _, mm, hdr, base = self.handles[path]
        meta = hdr[name]; s, e = meta['data_offsets']
        return meta['dtype'], meta['shape'], to_f32(mm[base+s:base+e], meta['dtype'], meta['shape'])
    def close(self):
        for fh, mm, _, _ in self.handles.values(): mm.close(); fh.close()

def q8_rows(x):
    amax = np.max(np.abs(x), axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(x / scale[:, None]), -127, 127).astype(np.int8)
    return q.tobytes(), scale.astype(np.float16).tobytes()

def q4_rows(x, group):
    r, inn = x.shape
    inp = ((inn + group - 1) // group) * group
    if inp != inn: x = np.concatenate([x, np.zeros((r, inp - inn), np.float32)], axis=1)
    xg = x.reshape(r, inp // group, group)
    amax = np.max(np.abs(xg), axis=2)
    scale = np.where(amax > 0, amax / 8.0, 1.0).astype(np.float32)
    q = np.clip(np.round(xg / scale[:, :, None]), -8, 7).astype(np.int8)
    nib = (q + 8).astype(np.uint8).reshape(r, -1)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    return packed.tobytes(), scale.astype(np.float16).tobytes()

def main():
    ap = argparse.ArgumentParser(description="Dual-quant (Q8+Q4) any safetensors model into KHANARY quant containers.")
    ap.add_argument('--src', default=DEF_SRC, help="safetensors file, or HF model dir (single or sharded)")
    ap.add_argument('--out', default=DEF_OUT, help="output dir for *.kqz + *.manifest.json")
    ap.add_argument('--name', default=DEF_NAME, help="artifact basename (default qwen1_8b)")
    ap.add_argument('--group', type=int, default=64, help="Q4 group size (default 64)")
    ap.add_argument('--rowchunk', type=int, default=8192, help="row-chunk for big tensors")
    ap.add_argument('--schemes', default="q8,q4", help="comma list, subset of q8,q4")
    args = ap.parse_args()
    schemes = [s.strip() for s in args.schemes.split(',') if s.strip()]
    assert all(s in ('q8', 'q4') for s in schemes), "schemes must be from q8,q4"
    os.makedirs(args.out, exist_ok=True)
    src = Source(args.src); names = src.names()
    print(f"[src] {args.src}\n[src] {len(names)} tensors  schemes={schemes} group={args.group}")

    outs = {}
    for tag in schemes:
        p = os.path.join(args.out, f"{args.name}.{tag}.kqz")
        outs[tag] = {'fh': open(p, 'wb'), 'off': 0, 'man': [], 'path': p}

    for name in names:
        dtype, shape, raw = src.get(name)
        if len(shape) == 1:                                  # 1D kept F16 in every scheme
            payload = raw.astype(np.float16).tobytes()
            for tag in schemes:
                o = outs[tag]; o['fh'].write(payload)
                o['man'].append({'name': name, 'shape': shape, 'quant': 'f16',
                                 'data_off': o['off'], 'data_size': len(payload),
                                 'scale_off': 0, 'scale_size': 0})
                o['off'] += len(payload)
            continue
        assert len(shape) == 2, f"{name} rank {len(shape)}"
        rows, cols = shape
        acc = {t: [b'', b''] for t in schemes}
        for r0 in range(0, rows, args.rowchunk):
            blk = raw[r0:min(r0+args.rowchunk, rows)].astype(np.float32, copy=False)
            if 'q8' in schemes: d, s = q8_rows(blk); acc['q8'][0] += d; acc['q8'][1] += s
            if 'q4' in schemes: d, s = q4_rows(blk, args.group); acc['q4'][0] += d; acc['q4'][1] += s
        for tag in schemes:
            quant = 'q8_perchannel' if tag == 'q8' else f'q4_group{args.group}'
            o = outs[tag]; data, sc = acc[tag]
            o['fh'].write(data); o['fh'].write(sc)
            o['man'].append({'name': name, 'shape': shape, 'quant': quant,
                             'data_off': o['off'], 'data_size': len(data),
                             'scale_off': o['off'] + len(data), 'scale_size': len(sc),
                             'group': args.group if tag == 'q4' else None})
            o['off'] += len(data) + len(sc)

    GiB = 1024**3
    for tag in schemes:
        o = outs[tag]; o['fh'].close()
        json.dump({'source': args.src, 'scheme': tag, 'group': args.group if tag == 'q4' else None,
                   'tensors': o['man'], 'payload_file': os.path.basename(o['path']),
                   'total_bytes': o['off']}, open(os.path.join(args.out, f"{args.name}.{tag}.manifest.json"), 'w'), indent=1)
        actual = os.path.getsize(o['path'])
        assert actual == o['off'], f"{tag} filesize {actual} != manifest {o['off']}"
        print(f"[out] {o['path']}  {actual/GiB:.3f} GiB  ({len(o['man'])} tensors)  manifest OK")
    src.close()
    print("[done]")

if __name__ == '__main__':
    main()
