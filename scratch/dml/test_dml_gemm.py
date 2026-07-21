# Direct unit test of both dml_gemm.dll entry points vs numpy — validates the exact call the
# ggml-xcfe backend makes (dml_gemm_bt_f32 = ggml MUL_MAT: src1 @ src0^T), which can't be
# link-tested here (no ggml build). Run from repo root: python scratch/dml/test_dml_gemm.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from dml_gemm import dml_matmul, dml_matmul_bt

rng = np.random.default_rng(1)
A = rng.standard_normal((17, 320)).astype(np.float32); B = rng.standard_normal((320, 96)).astype(np.float32)
e1 = np.abs(dml_matmul(A, B) - A @ B).max() / np.abs(A @ B).max()

src1 = rng.standard_normal((17, 320)).astype(np.float32); src0 = rng.standard_normal((96, 320)).astype(np.float32)
e2 = np.abs(dml_matmul_bt(src1, src0) - src1 @ src0.T).max() / np.abs(src1 @ src0.T).max()

print(f"[dml_gemm_f32]    C=A@B                         scale-norm err {e1:.2e}")
print(f"[dml_gemm_bt_f32] C=src1@src0^T (ggml MUL_MAT)  scale-norm err {e2:.2e}")
ok = e1 < 1e-4 and e2 < 1e-4
print(f"=== {'PASS' if ok else 'FAIL'}: DirectML GEMM entry points match numpy ===")
sys.exit(0 if ok else 1)
