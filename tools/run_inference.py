"""Python bridge for invoking a compiled KHΛNARY CUDA module."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util

_np_spec = importlib.util.find_spec("numpy")
np = importlib.import_module("numpy") if _np_spec is not None else None


class KhlnaryRuntime:
    def __init__(self, module_path: str = "./khlnary_module.so"):
        if np is None:
            raise RuntimeError("NumPy is required for runtime host buffers")
        self.lib = ctypes.CDLL(module_path)
        self.lib.init_stb_tensors.restype = None
        self.lib.khlnary_kernel.restype = None
        self.lib.khlnary_kernel.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.init_stb_tensors()

    def forward(self, input_tensor):
        batch_size, seq_len, hidden_size = input_tensor.shape
        input_flat = np.asarray(input_tensor, dtype=np.float16).reshape(-1)
        output_flat = np.zeros_like(input_flat)
        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        self.lib.khlnary_kernel(input_ptr, output_ptr, batch_size, seq_len, hidden_size)
        return output_flat.reshape(batch_size, seq_len, hidden_size)


if __name__ == "__main__":
    runtime = KhlnaryRuntime()
    x = np.random.randn(2, 4, 8).astype(np.float32)
    y = runtime.forward(x)
    print("input", x.shape, "output", y.shape, y.dtype)
