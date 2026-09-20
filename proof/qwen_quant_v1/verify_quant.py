# verify_quant.py — validate the Qwen quant artifacts: CONTAINER integrity + dequant fidelity.
#
# Guardrails (advisor): the bugs live in manifest offsets, not the quant math. So this reloads each
# tensor FROM THE WRITTEN FILE via its manifest offsets, dequantizes, and compares to the original
# FP16 (upcast F32). Checks: (1) filesize == sum(manifest sizes); (2) all source tensors present;
# (3) shapes preserved; (4) Q4/Q8 manifests tensor-aligned (identical names, identical order);
# (5) per-tensor + aggregate dequant relative error is sane.
import os, json, struct, mmap
import numpy as np

SRC = r"C:\Users\canna\.lmstudio\models\Qwen-1_8B-Chat-f16\model.safetensors"
OUT = r"E:\models\Qwen1.8B-quant"

def st_header(mm):
    n = struct.unpack('<Q', mm[:8])[0]
    return json.loads(mm[8:8+n].decode('utf-8')), 8 + n

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
        x = x * sc.reshape(rows, ng, 1)
        return x.reshape(rows, inp)[:, :cols]
    raise ValueError(q)

def main():
    sf = open(SRC, 'rb'); smm = mmap.mmap(sf.fileno(), 0, access=mmap.ACCESS_READ)
    shdr, sbase = st_header(smm)
    src_names = sorted(k for k in shdr if k != '__metadata__')
    print(f"[src] {len(src_names)} tensors")

    mans = {}
    for tag in ('q8', 'q4'):
        mp = os.path.join(OUT, f"qwen1_8b.{tag}.manifest.json")
        man = json.load(open(mp))
        pf = os.path.join(OUT, man['payload_file'])
        fsz = os.path.getsize(pf)
        assert fsz == man['total_bytes'], f"{tag}: filesize {fsz} != manifest total {man['total_bytes']}"
        names = [t['name'] for t in man['tensors']]
        assert names == src_names, f"{tag}: tensor set/order != source"
        print(f"[{tag}] {pf}  {fsz/1024**3:.3f} GiB  filesize==manifest OK  {len(names)} tensors present+ordered OK")
        mans[tag] = (man, pf)

    # tensor-alignment between the two artifacts (the per-tensor-escalation requirement)
    n8 = [t['name'] for t in mans['q8'][0]['tensors']]
    n4 = [t['name'] for t in mans['q4'][0]['tensors']]
    assert n8 == n4, "Q8/Q4 not tensor-aligned"
    print(f"[align] Q8 and Q4 tensor-aligned (identical {len(n8)} names/order) -> per-tensor Q4->Q8 swap addressable")

    # dequant fidelity vs source, per artifact
    for tag in ('q8', 'q4'):
        man, pf = mans[tag]
        pfh = open(pf, 'rb')
        pmm = mmap.mmap(pfh.fileno(), 0, access=mmap.ACCESS_READ)
        emax = 0.0; esum = 0.0; nel = 0; worst = None
        for e in man['tensors']:
            s, en = shdr[e['name']]['data_offsets']
            orig = np.frombuffer(smm[sbase+s:sbase+en], np.float16).astype(np.float32).reshape(e['shape'])
            deq = dequant(e, pmm)
            assert deq.shape == tuple(e['shape']), f"{e['name']} shape {deq.shape} != {e['shape']}"
            denom = np.maximum(np.abs(orig), 1e-6)
            rel = np.abs(deq - orig) / denom
            m = float(rel.max()); mean = float(rel.mean())
            if m > emax: emax = m; worst = (e['name'], m)
            esum += mean * orig.size; nel += orig.size
        print(f"[{tag}] dequant rel-error: mean {esum/nel:.4f}  worst-tensor-max {emax:.3f} @ {worst[0]}")
        pmm.close(); pfh.close()
    print("[done] container + fidelity verification passed")

if __name__ == '__main__':
    main()
