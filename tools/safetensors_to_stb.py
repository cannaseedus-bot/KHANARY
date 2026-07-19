# safetensors_to_stb.py — bridge: safetensors weights -> a real KHANARY .stb.
#
# The weight-side counterpart of brain_to_stb.py. Makes safetensors tensors consumable by
# KHANARY's own .stb container (and therefore by the G_MATMUL cs_5_0 GEMM path). This is the
# first honest step of "adapting to GGUF/safetensors": get real trained weights into KHANARY's
# tensor format. (GGUF would need an extra dequant step; safetensors is already dense float32.)
#
# Run: python tools/safetensors_to_stb.py <model.safetensors> <out.stb> [key1 key2 ...]
#      (no keys -> writes every 2-D float32 tensor, tensor_id = order of appearance)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from safetensors import safe_open
from stb import write_stb, read_stb


def safetensors_to_stb(src, out, keys=None):
    tensors, manifest = [], []
    with safe_open(src, framework="numpy") as f:
        avail = list(f.keys())
        if not keys:
            keys = [k for k in avail if f.get_tensor(k).ndim == 2 and f.get_tensor(k).dtype == np.float32]
        for tid, k in enumerate(keys):
            if k not in avail:
                raise KeyError(f"{k} not in {src} ({len(avail)} tensors)")
            arr = np.ascontiguousarray(f.get_tensor(k))
            if arr.dtype not in (np.float32, np.int32):
                arr = arr.astype(np.float32)   # .stb dtypes: float32 / int32
            tensors.append({"tensor_id": tid, "array": arr})
            manifest.append((tid, k, tuple(arr.shape), str(arr.dtype)))
    write_stb(out, tensors)
    return manifest


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    keys = sys.argv[3:] or None
    man = safetensors_to_stb(src, out, keys)
    print(f"[stb] wrote {out} ({len(man)} tensors, {os.path.getsize(out)} bytes)")
    back = read_stb(out)
    ok = True
    for tid, name, shape, dtype in man:
        got = tuple(back[tid]["dims"])
        match = got == shape
        ok = ok and match
        print(f"  {'PASS' if match else 'FAIL'} tid={tid} {name:38} {shape} {dtype}")
    print(f"=== {'PASS' if ok else 'FAIL'}: safetensors -> KHANARY .stb, round-trip via read_stb ===")
    sys.exit(0 if ok else 1)
