# brain_to_stb.py — bridge: birdsong brain compiler output -> KHANARY .stb
#
# Accepts either:
#   brain.brain  binary file  (BRN1 magic, brain2 format)
#   directory    containing tensor_header.bin + *.bin files (brain1 format)
#
# Run: python tools/brain_to_stb.py [brain.brain | brain_dir] [out.stb]

import os, sys, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stb import write_stb, read_stb

arg1 = sys.argv[1] if len(sys.argv) > 1 else r"C:\ffmpeg\bin\brain2\brain.brain"
arg2 = sys.argv[2] if len(sys.argv) > 2 else ""

# ---- detect input format ----

if os.path.isdir(arg1):
    # Directory mode: read individual .bin files (brain1 format)
    d = arg1
    OUT = arg2 if arg2 else os.path.join(d, "khanary_brain.stb")

    hdr_path = os.path.join(d, "tensor_header.bin")
    with open(hdr_path, "rb") as f:
        nodes, edges, csr, K, features = struct.unpack("5I", f.read(20))
    print(f"[brain-dir] nodes={nodes} edges={edges} csr={csr} experts(K)={K}")

    time_a   = np.fromfile(os.path.join(d, "nodes_time.bin"),    dtype=np.float32)
    freq_a   = np.fromfile(os.path.join(d, "nodes_freq.bin"),    dtype=np.float32)
    energy_a = np.fromfile(os.path.join(d, "nodes_energy.bin"),  dtype=np.float32)
    edge_a   = np.fromfile(os.path.join(d, "edges.bin"),         dtype=np.uint32).reshape(-1, 2)
    csr_idx  = np.fromfile(os.path.join(d, "csr_index.bin"),     dtype=np.uint32)
    csr_nb   = np.fromfile(os.path.join(d, "csr_neighbors.bin"), dtype=np.uint32)
    experts  = np.fromfile(os.path.join(d, "experts.bin"),       dtype=np.uint32)

else:
    # Binary mode: read brain.brain (brain2 format)
    BRAIN = arg1
    OUT   = arg2 if arg2 else r"C:\ffmpeg\bin\brain2\khanary_brain.stb"

    raw = open(BRAIN, "rb").read()
    magic, ver, nodes, edges, csr, K = struct.unpack_from("<IIIIII", raw, 0)
    assert magic == 0x42524E31, f"not a BRN1 brain (magic=0x{magic:08X})"
    print(f"[brain] nodes={nodes} edges={edges} csr={csr} experts(K)={K}")

    off = 24
    def take(dtype, count):
        global off
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=off).copy()
        off += a.nbytes
        return a

    time_a   = take(np.float32, nodes)
    freq_a   = take(np.float32, nodes)
    energy_a = take(np.float32, nodes)
    edge_a   = take(np.uint32, edges * 2).reshape(edges, 2)
    csr_idx  = take(np.uint32, nodes + 1)
    csr_nb   = take(np.uint32, csr)
    experts  = take(np.uint32, nodes)

# STB dtypes: float32 + int32 (indices << 2^31, so int32 view is exact)
tensors = [
    {"tensor_id": 0, "array": time_a},                    # node time  (SVG x / ridge point)
    {"tensor_id": 1, "array": freq_a},                    # node freq  (SVG y)
    {"tensor_id": 2, "array": energy_a},                  # node energy (spectrogram intensity)
    {"tensor_id": 3, "array": edge_a.astype(np.int32)},   # Delaunay mesh edges [E,2]
    {"tensor_id": 4, "array": csr_idx.astype(np.int32)},  # CSR index [N+1]
    {"tensor_id": 5, "array": csr_nb.astype(np.int32)},   # CSR neighbours [csr]
    {"tensor_id": 6, "array": experts.astype(np.int32)},  # geometric MoE expert id per node [N]
]
write_stb(OUT, tensors)
print(f"[stb]   wrote {OUT}  ({len(tensors)} tensors, {os.path.getsize(OUT)} bytes)")

# --- round-trip verify with KHANARY's own reader ---
back = read_stb(OUT)
src = {0: time_a, 1: freq_a, 2: energy_a, 3: edge_a.astype(np.int32),
       4: csr_idx.astype(np.int32), 5: csr_nb.astype(np.int32), 6: experts.astype(np.int32)}
names = {0:"time",1:"freq",2:"energy",3:"edges",4:"csr_index",5:"csr_neighbors",6:"experts"}
ok = True
print("\n[verify] KHANARY read_stb round-trip:")
for tid in range(7):
    a = np.asarray(back[tid]["array"]); b = src[tid]
    match = a.shape == b.shape and np.array_equal(a, b)
    ok = ok and match
    print(f"  {'PASS' if match else 'FAIL'} tid={tid} {names[tid]:<14} dims={list(back[tid]['dims'])} dtype={back[tid]['dtype'].__name__}")

# sanity: graph consistency (CSR sums to 2*edges, experts in [0,K))
csr_ok = int(csr_idx[-1]) == csr and int(experts.max()) + 1 <= K
print(f"\n[graph]  CSR_index[-1]={csr_idx[-1]} (==2*edges={2*edges}? {int(csr_idx[-1])==2*edges}) ; experts in [0,{K}): {int(experts.max())+1<=K}")
print(f"\n=== {'PASS' if ok and csr_ok else 'FAIL'}: brain2 birdsong graph -> real KHANARY .stb, round-trip via KHANARY read_stb ===")
sys.exit(0 if (ok and csr_ok) else 1)
