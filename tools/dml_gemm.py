"""DirectML GEMM via ctypes — the D3D12 DirectML matmul path for the KHANARY inference driver.

Loads scratch/dml/dml_gemm.dll (built from dml_gemm_dll.cpp), which runs C[M,N]=A[M,K]@B[K,N]
on DirectML (D3D12, HD 4600). Used by tools/kxml_inference_driver.op_matmul when KXML_DML=1, to
prototype routing G_MATMUL onto DirectML (~4.9x faster than the tiled cs_5_0 kernel on this rig).
"""
import os
import ctypes
import numpy as np

_DLL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "dml")
_lib = None


def _load():
    global _lib
    if _lib is None:
        # DirectML.dll sits next to dml_gemm.dll — make it resolvable.
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_DLL_DIR)
        _lib = ctypes.CDLL(os.path.join(_DLL_DIR, "dml_gemm.dll"))
        _lib.dml_gemm_f32.restype = ctypes.c_int
        _lib.dml_gemm_f32.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ]
    return _lib


def dml_matmul(A, B):
    """C[M,N] = A[M,K] @ B[K,N] on DirectML. A, B 2-D; returns float32 [M,N]."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"inner dim mismatch {A.shape} @ {B.shape}"
    C = np.empty((M, N), dtype=np.float32)
    rc = _load().dml_gemm_f32(A, B, C, M, N, K)
    if rc != 0:
        raise RuntimeError(f"dml_gemm_f32 failed rc={rc} for M={M} N={N} K={K}")
    return C
