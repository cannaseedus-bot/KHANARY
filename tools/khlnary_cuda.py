"""KHΛNARY -> CUDA lowering skeleton.

Generates host-side C++ initialization snippets and a kernel template by scanning
KNU words for G_LOAD_BIN_TENSOR glyphs.
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Mapping

GLYPH_IDS = {
    "G_LOAD_BIN_TENSOR": 0x30,
    "G_ADD_I32": 0x02,
    "G_RET": 0x03,
}


def decode_knu(word: int) -> Dict[str, int]:
    return {
        "ver": (word >> 28) & 0xF,
        "glyph_id": (word >> 20) & 0xFF,
        "arity": (word >> 16) & 0xF,
        "flags": (word >> 12) & 0xF,
        "payload": (word >> 4) & 0xFF,
        "auth_class": (word >> 1) & 0x7,
        "parity": word & 0x1,
    }


def lower_khlnary_to_cuda(knus: List[int], bin_file_table: Mapping[int, Mapping[str, str]]) -> str:
    """Lower KHΛNARY words into CUDA host/kernel skeleton code."""

    tensors = []  # (handle_name, bin_file_id, tensor_id)

    for w in knus:
        k = decode_knu(w)
        if k["glyph_id"] == GLYPH_IDS["G_LOAD_BIN_TENSOR"]:
            bin_file_id = (k["payload"] >> 4) & 0xF
            tensor_id = (k["payload"] >> 0) & 0xF
            handle_name = f"t_{bin_file_id}_{tensor_id}"
            tensors.append((handle_name, bin_file_id, tensor_id))

    host_code = [
        "#include <cuda.h>",
        "#include <cuda_runtime.h>",
        "#include <fcntl.h>",
        "#include <sys/mman.h>",
        "",
        "",
    ]

    for handle, bin_id, tid in tensors:
        if bin_id not in bin_file_table:
            raise KeyError(f"Missing bin_file_id in table: {bin_id}")
        path = bin_file_table[bin_id]["path"]
        host_code += [
            f"// {handle} from {path} (tensor_id={tid})",
            f"void* {handle}_host = nullptr;",
            f"half* {handle}_dev = nullptr;",
            "",
            f"void init_{handle}() {{",
            f"  int fd = open(\"{path}\", O_RDONLY);",
            f"  // TODO: read header, locate tensor_id={tid}, mmap region",
            f"  // mmap -> {handle}_host",
            f"  cudaMalloc(&{handle}_dev, /*size_bytes*/);",
            f"  cudaMemcpy({handle}_dev, {handle}_host, /*size_bytes*/, cudaMemcpyHostToDevice);",
            "}",
            "",
        ]

    kernel = textwrap.dedent(
        r"""
        __global__ void khlnary_kernel(const half* __restrict__ input,
                                       half* __restrict__ output) {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          // TODO: use mapped tensors, e.g. t_0_0_dev, t_0_1_dev, ...
        }
        """
    ).strip("\n")

    return "\n".join(host_code) + "\n" + kernel + "\n"


__all__ = ["GLYPH_IDS", "decode_knu", "lower_khlnary_to_cuda"]
