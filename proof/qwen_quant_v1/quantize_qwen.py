# quantize_qwen.py — stream Qwen-1.8B FP16 safetensors -> two lean, mmap-able quant artifacts.
#
# Produces REAL quantized weight bytes for the residency/hot-swap probe to mount as actual model
# data (replacing the anonymous 1707 MB allocation in gpu_q8_hotswap_v1). Two schemes, both a flat
# payload file + a JSON offset-manifest (same style as the smgm-16 DDS folds), so a D3D12 loader can
# mmap + MakeResident directly:
#
#   Q8  per-output-channel symmetric INT8   (scale = max|row|/127)            ~1.71 GiB
#   Q4  group-wise 4-bit, groups of 64      (scale = max|group|/8, 2 nib/byte) ~0.91 GiB
#   1D tensors (biases / RMSNorm gains) kept F16 (negligible size, keeps fidelity)
#
# SCOPE (honest): these are WEIGHTS, not a runnable model. This stack has no Qwen forward path
# (the #001 DirectML driver is GPT-2-only: LayerNorm/GELU/50257, not RMSNorm/SwiGLU/RoPE/151936;
# vendored llama is CPU-only, off the GPU budget). "Q8 fits resident" is a property of THIS lean
# per-channel scheme (~1.71 GiB < the measured 1.75 GiB stable ceiling); standard GGUF Q8_0 is
# ~1.82 GiB and would NOT fit. Fidelity here is verified at dequant-error + container round-trip,
# not end-to-end perplexity.
import os, sys, json, struct, mmap
import numpy as np

SRC = r"C:\Users\canna\.lmstudio\models\Qwen-1_8B-Chat-f16\model.safetensors"
OUT = r"E:\models\Qwen1.8B-quant"
ROWCHUNK = 8192          # cap transient memory on big tensors (embed/lm_head)
GROUP = 64               # Q4 group size along the input dim

def read_st_header(mm):
    n = struct.unpack('<Q', mm[:8])[0]
    hdr = json.loads(mm[8:8+n].decode('utf-8'))
    return hdr, 8 + n

def q8_rows(x):          # x: [r, in] float32 -> (int8 bytes, f16 scale bytes)
    amax = np.max(np.abs(x), axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(x / scale[:, None]), -127, 127).astype(np.int8)
    return q.tobytes(), scale.astype(np.float16).tobytes()

def q4_rows(x, group=GROUP):   # x: [r, in] float32 -> (packed nibble bytes, f16 scale bytes)
    r, inn = x.shape
    inp = ((inn + group - 1) // group) * group
    if inp != inn:
        x = np.concatenate([x, np.zeros((r, inp - inn), np.float32)], axis=1)
    xg = x.reshape(r, inp // group, group)
    amax = np.max(np.abs(xg), axis=2)
    scale = np.where(amax > 0, amax / 8.0, 1.0).astype(np.float32)
    q = np.clip(np.round(xg / scale[:, :, None]), -8, 7).astype(np.int8)
    nib = (q + 8).astype(np.uint8).reshape(r, -1)          # [r, inp] in 0..15
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    return packed.tobytes(), scale.astype(np.float16).tobytes()

def main():
    os.makedirs(OUT, exist_ok=True)
    f = open(SRC, 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    hdr, base = read_st_header(mm)
    names = [k for k in hdr if k != '__metadata__']
    names.sort()
    print(f"[src] {SRC}\n[src] {len(names)} tensors, data base offset {base}")

    outs = {}
    for tag in ('q8', 'q4'):
        p = os.path.join(OUT, f"qwen1_8b.{tag}.kqz")
        outs[tag] = {'fh': open(p, 'wb'), 'off': 0, 'man': [], 'path': p}

    for name in names:
        meta = hdr[name]; shape = meta['shape']; s, e = meta['data_offsets']
        assert meta['dtype'] == 'F16', f"{name} dtype {meta['dtype']}"
        raw = np.frombuffer(mm[base + s: base + e], dtype=np.float16)
        if len(shape) == 1:      # 1D: keep F16 in both artifacts
            payload = raw.tobytes()
            for tag in ('q8', 'q4'):
                o = outs[tag]; o['fh'].write(payload)
                o['man'].append({'name': name, 'shape': shape, 'quant': 'f16',
                                 'data_off': o['off'], 'data_size': len(payload),
                                 'scale_off': 0, 'scale_size': 0})
                o['off'] += len(payload)
            continue
        assert len(shape) == 2, f"{name} rank {len(shape)}"
        rows, cols = shape
        # accumulate per-scheme data + scales across row-chunks, then write contiguously
        acc = {'q8': [b'', b''], 'q4': [b'', b'']}   # [payload, scales]
        for r0 in range(0, rows, ROWCHUNK):
            blk = raw[r0*cols:(min(r0+ROWCHUNK, rows))*cols].reshape(-1, cols).astype(np.float32)
            d8, s8 = q8_rows(blk); acc['q8'][0] += d8; acc['q8'][1] += s8
            d4, s4 = q4_rows(blk); acc['q4'][0] += d4; acc['q4'][1] += s4
        for tag, quant in (('q8', 'q8_perchannel'), ('q4', f'q4_group{GROUP}')):
            o = outs[tag]; data, sc = acc[tag]
            o['fh'].write(data); o['fh'].write(sc)
            o['man'].append({'name': name, 'shape': shape, 'quant': quant,
                             'data_off': o['off'], 'data_size': len(data),
                             'scale_off': o['off'] + len(data), 'scale_size': len(sc),
                             'group': GROUP if tag == 'q4' else None})
            o['off'] += len(data) + len(sc)

    GiB = 1024**3
    for tag in ('q8', 'q4'):
        o = outs[tag]; o['fh'].close()
        mpath = os.path.join(OUT, f"qwen1_8b.{tag}.manifest.json")
        json.dump({'source': SRC, 'scheme': tag, 'group': GROUP if tag == 'q4' else None,
                   'tensors': o['man'], 'payload_file': os.path.basename(o['path']),
                   'total_bytes': o['off']}, open(mpath, 'w'), indent=1)
        actual = os.path.getsize(o['path'])
        assert actual == o['off'], f"{tag} filesize {actual} != manifest {o['off']}"
        print(f"[out] {o['path']}  {actual/GiB:.3f} GiB  ({len(o['man'])} tensors)  manifest OK")
    mm.close(); f.close()
    print("[done] wrote Q8 + Q4 artifacts")

if __name__ == '__main__':
    main()
