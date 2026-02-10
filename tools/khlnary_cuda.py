"""KHΛNARY -> CUDA lowering helpers and module-oriented backend."""

from __future__ import annotations

import textwrap
from typing import Dict, List, Mapping

from tools.khlnary_compiler import KhlnaryModule

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


class CudaBackend:
    def __init__(self, module: KhlnaryModule):
        self.module = module

    def generate_host_code(self) -> str:
        code = [
            "#include <cuda_runtime.h>",
            "#include <cuda_fp16.h>",
            "#include <sys/mman.h>",
            "#include <sys/stat.h>",
            "#include <fcntl.h>",
            "#include <unistd.h>",
            "",
        ]

        for tensor in self.module.tensors:
            code.append(f"static half* {tensor.cuda_ptr_name} = nullptr;")
        code += ["", "void init_stb_tensors() {"]

        for file_id, path in self.module.bin_files.items():
            code += [
                f"  // File {file_id}: {path}",
                f"  int fd_{file_id} = open(\"{path}\", O_RDONLY);",
                f"  struct stat st_{file_id};",
                f"  fstat(fd_{file_id}, &st_{file_id});",
                f"  void* host_ptr_{file_id} = mmap(NULL, st_{file_id}.st_size, PROT_READ, MAP_PRIVATE, fd_{file_id}, 0);",
            ]
            for tensor in self.module.tensors:
                if tensor.file_id == file_id:
                    code += [
                        f"  cudaMalloc((void**)&{tensor.cuda_ptr_name}, {tensor.size_bytes});",
                        f"  cudaMemcpy({tensor.cuda_ptr_name}, (char*)host_ptr_{file_id} + {tensor.offset}, {tensor.size_bytes}, cudaMemcpyHostToDevice);",
                    ]
            code += [f"  munmap(host_ptr_{file_id}, st_{file_id}.st_size);", f"  close(fd_{file_id});", ""]

        code.append("}")
        return "\n".join(code)

    @staticmethod
    def generate_kernel_code() -> str:
        return textwrap.dedent(
            """
            extern "C" __global__ void khlnary_kernel(
                const half* __restrict__ input,
                half* __restrict__ output,
                int batch_size,
                int seq_len,
                int hidden_size) {
              int tid = blockIdx.x * blockDim.x + threadIdx.x;
              int n = batch_size * seq_len * hidden_size;
              if (tid >= n) return;
              output[tid] = input[tid];
            }
            """
        ).strip()

    @staticmethod
    def generate_makefile() -> str:
        return textwrap.dedent(
            """
            CC = nvcc
            CFLAGS = -arch=sm_80 -O3 --use_fast_math
            TARGET = khlnary_module.so

            all: $(TARGET)

            $(TARGET): generated_kernel.cu
            \t$(CC) $(CFLAGS) -shared -Xcompiler -fPIC -o $@ $<

            clean:
            \trm -f $(TARGET) *.o
            """
        ).strip() + "\n"


def lower_khlnary_to_cuda(knus: List[int], bin_file_table: Mapping[int, Mapping[str, str]]) -> str:
    """Lower KHΛNARY words into CUDA host/kernel skeleton code."""
    tensors = []
    for w in knus:
        k = decode_knu(w)
        if k["glyph_id"] == GLYPH_IDS["G_LOAD_BIN_TENSOR"]:
            bin_file_id = (k["payload"] >> 4) & 0xF
            tensor_id = k["payload"] & 0xF
            tensors.append((f"t_{bin_file_id}_{tensor_id}", bin_file_id, tensor_id))

    host_code = ["#include <cuda.h>", "#include <cuda_runtime.h>", "#include <fcntl.h>", "#include <sys/mman.h>", ""]
    for handle, bin_id, tid in tensors:
        if bin_id not in bin_file_table:
            raise KeyError(f"Missing bin_file_id in table: {bin_id}")
        path = bin_file_table[bin_id]["path"]
        host_code += [
            f"void* {handle}_host = nullptr;",
            f"half* {handle}_dev = nullptr;",
            f"void init_{handle}() {{",
            f"  int fd = open(\"{path}\", O_RDONLY);",
            f"  // locate tensor_id={tid} and mmap file range",
            f"  cudaMalloc(&{handle}_dev, /*size_bytes*/);",
            f"  cudaMemcpy({handle}_dev, {handle}_host, /*size_bytes*/, cudaMemcpyHostToDevice);",
            "}",
            "",
        ]

    kernel = textwrap.dedent(
        """
        __global__ void khlnary_kernel(const half* __restrict__ input,
                                       half* __restrict__ output) {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          output[idx] = input[idx];
        }
        """
    ).strip("\n")
    return "\n".join(host_code) + "\n" + kernel + "\n"


__all__ = ["GLYPH_IDS", "decode_knu", "lower_khlnary_to_cuda", "CudaBackend"]
