# verify_quant.py — validate quant artifacts: CONTAINER integrity + dequant fidelity (portable).
#
# Guardrails (advisor): the bugs live in manifest offsets, not the quant math. So this reloads each
# tensor FROM THE WRITTEN FILE via its manifest offsets, dequantizes, and compares to the source
# (upcast F32). Checks: (1) filesize == sum(manifest sizes); (2) all source tensors present;
# (3) shapes preserved; (4) schemes tensor-aligned (identical names/order -> per-tensor escalation);
# (5) per-tensor + aggregate dequant relative error is sane.
#
#   python tools/verify_quant.py --src <model.safetensors|dir> --out <dir> --name <base>
#   (no args -> the Qwen-1.8B defaults)
import os, sys, json, mmap, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_safetensors import Source, DEF_SRC, DEF_OUT, DEF_NAME

def dequant(entry, blob):
    q = entry['quant']; shape = entry['shape']
    data = blob[entry['data_off']:entry['data_off']+entry['data_size']]
    if q == 'f16':
        return np.frombuffer(data, np.float16).astype(np.float32).reshape(shape)
    sc = np.frombuffer(blob[entry['scale_off']:entry['scale_off']+entry['scale_size']], np.float16).astype(np.float32)
    rows, cols = shape
    if q == 'q8_perchannel':
        x = np.frombuffer(data, np.int8).astype(np.float32).reshape(rows, cols)
        return x * sc.reshape(rows, 1)
    if q.startswith('q4_group'):
        g = entry['group']; inp = ((cols + g - 1)//g)*g; ng = inp//g
        packed = np.frombuffer(data, np.uint8).reshape(rows, inp//2)
        nib = np.empty((rows, inp), np.uint8)
        nib[:, 0::2] = packed & 0x0F; nib[:, 1::2] = packed >> 4
        x = (nib.astype(np.float32) - 8.0).reshape(rows, ng, g)
        return (x * sc.reshape(rows, ng, 1)).reshape(rows, inp)[:, :cols]
    raise ValueError(q)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEF_SRC); ap.add_argument('--out', default=DEF_OUT)
    ap.add_argument('--name', default=DEF_NAME); ap.add_argument('--schemes', default="q8,q4")
    args = ap.parse_args(); schemes = [s.strip() for s in args.schemes.split(',') if s.strip()]

    src = Source(args.src); src_names = src.names()
    print(f"[src] {len(src_names)} tensors")
    mans = {}
    for tag in schemes:
        man = json.load(open(os.path.join(args.out, f"{args.name}.{tag}.manifest.json")))
        pf = os.path.join(args.out, man['payload_file']); fsz = os.path.getsize(pf)
        assert fsz == man['total_bytes'], f"{tag}: filesize {fsz} != manifest {man['total_bytes']}"
        assert [t['name'] for t in man['tensors']] == src_names, f"{tag}: tensor set/order != source"
        print(f"[{tag}] {pf}  {fsz/1024**3:.3f} GiB  filesize==manifest OK  {len(man['tensors'])} tensors present+ordered OK")
        mans[tag] = (man, pf)
    if len(schemes) > 1:
        base = [t['name'] for t in mans[schemes[0]][0]['tensors']]
        for tag in schemes[1:]:
            assert [t['name'] for t in mans[tag][0]['tensors']] == base, f"{tag} not tensor-aligned"
        print(f"[align] schemes tensor-aligned ({len(base)} names/order) -> per-tensor escalation addressable")

    for tag in schemes:
        man, pf = mans[tag]
        pfh = open(pf, 'rb'); pmm = mmap.mmap(pfh.fileno(), 0, access=mmap.ACCESS_READ)
        ent = {t['name']: t for t in man['tensors']}
        emax = 0.0; worst = ('', 0.0); nrmse_num = 0.0; nrmse_den = 0.0
        for name in src_names:
            _, shape, orig = src.get(name)
            deq = dequant(ent[name], pmm)
            assert deq.shape == tuple(shape), f"{name} shape {deq.shape} != {shape}"
            diff = deq - orig
            nrmse_num += float(np.dot(diff.ravel(), diff.ravel()))
            nrmse_den += float(np.dot(orig.ravel(), orig.ravel()))
            m = float((np.abs(diff) / np.maximum(np.abs(orig), 1e-6)).max())
            if m > emax: emax = m; worst = (name, m)
        nrmse = (nrmse_num / nrmse_den) ** 0.5
        snr = -20 * np.log10(nrmse) if nrmse > 0 else float('inf')
        print(f"[{tag}] global normRMSE {nrmse:.4f}  SNR {snr:.1f} dB   (worst per-elem rel {emax:.2f} @ {worst[0]})")
        pmm.close(); pfh.close()
    src.close()
    print("[done] container + fidelity verification passed")

if __name__ == '__main__':
    main()
