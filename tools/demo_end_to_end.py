"""Minimal vertical slice: Python -> .stb -> KHΛNARY -> CUDA skeleton source.

This script is illustrative and intentionally small.
"""

from __future__ import annotations

import importlib
import importlib.util

from tools.khlnary_cuda import lower_khlnary_to_cuda
from tools.khlnary_encoder import compile_to_knu
from tools.stb import write_stb

_np_spec = importlib.util.find_spec("numpy")
np = importlib.import_module("numpy") if _np_spec is not None else None


def build_tiny_model_stb():
    if np is None:
        raise RuntimeError("NumPy is required to run demo_end_to_end.py")

    W = np.random.randn(4, 4).astype(np.float16)
    b = np.random.randn(4).astype(np.float16)
    write_stb(
        "tiny.stb",
        [
            {"tensor_id": 0, "array": W},
            {"tensor_id": 1, "array": b},
        ],
    )
    return {0: {"path": "tiny.stb", "alias": "tiny"}}


def build_tiny_program():
    src = "1 + 2"
    return compile_to_knu(src)


def main():
    bin_file_table = build_tiny_model_stb()
    knus = build_tiny_program()
    cuda_src = lower_khlnary_to_cuda(knus, bin_file_table)

    with open("khlnary_kernel.cu", "w", encoding="utf-8") as f:
        f.write(cuda_src)

    print("Generated: tiny.stb + khlnary_kernel.cu")


if __name__ == "__main__":
    main()
