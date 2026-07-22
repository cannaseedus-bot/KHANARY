# run_kuhul_matmul_tick.py — reproducible launcher for the K'UHUL-3D MATMUL-tick proof.
#
# Resolves the target tensor's byte offsets across the Q4/Q8 manifests + the F16 safetensors, then
# runs scratch/dml/kuhul_matmul_tick.exe (the shared-device D3D12+DML executor). The exe walks the
# MATMUL node's recursive PhaseTick, driving REAL residency + dequant + DirectML GEMM + eviction.
import os, sys, json, struct, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "models", "khanary-qwen1_8b-v0.1.0")
SF = r"C:\Users\canna\.lmstudio\models\Qwen-1_8B-Chat-f16\model.safetensors"
EXE = os.path.join(ROOT, "scratch", "dml", "kuhul_matmul_tick.exe")
TENSOR = "transformer.h.0.attn.c_proj.weight"
M, THRESHOLD = 8, 0.02

def man(tag):
    return {t["name"]: t for t in json.load(open(os.path.join(PKG, f"qwen1_8b.{tag}.manifest.json")))["tensors"]}

def main():
    q4 = man("q4")[TENSOR]; q8 = man("q8")[TENSOR]
    with open(SF, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
    base = 8 + n; s, e = hdr[TENSOR]["data_offsets"]
    rows, cols = q4["shape"]
    args = [EXE,
            os.path.join(PKG, "qwen1_8b.q4.kqz"), q4["data_off"], q4["data_size"], q4["scale_off"], q4["scale_size"], q4["group"],
            os.path.join(PKG, "qwen1_8b.q8.kqz"), q8["data_off"], q8["data_size"], q8["scale_off"], q8["scale_size"],
            SF, base + s, e - s,
            rows, cols, M, THRESHOLD]
    print("[run]", TENSOR, "shape", [rows, cols], "M", M, "thr", THRESHOLD)
    sys.exit(subprocess.call([str(a) for a in args]))

if __name__ == "__main__":
    main()
