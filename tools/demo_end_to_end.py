"""Demo: Python -> .stb -> KHΛNARY module -> CUDA/WGSL source artifacts."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import struct

from tools.khlnary_compiler import KhlnaryCompiler
from tools.khlnary_cuda import CudaBackend
from tools.khlnary_webgpu import WebGpuBackend
from tools.stb import write_stb

_np_spec = importlib.util.find_spec("numpy")
np = importlib.import_module("numpy") if _np_spec is not None else None


def _require_numpy():
    if np is None:
        raise RuntimeError("NumPy is required to run demo_end_to_end.py")


def create_demo_weights(weights_dir: Path) -> None:
    _require_numpy()
    weights_dir.mkdir(parents=True, exist_ok=True)

    linear1_w = np.random.randn(8, 16).astype(np.float16)
    linear1_b = np.random.randn(16).astype(np.float16)
    linear2_w = np.random.randn(16, 8).astype(np.float16)
    linear2_b = np.random.randn(8).astype(np.float16)
    attn_q = np.random.randn(8, 8).astype(np.float16)
    attn_k = np.random.randn(8, 8).astype(np.float16)
    attn_v = np.random.randn(8, 8).astype(np.float16)

    write_stb(weights_dir / "linear1.stb", [{"tensor_id": 0, "array": linear1_w}, {"tensor_id": 1, "array": linear1_b}])
    write_stb(weights_dir / "linear2.stb", [{"tensor_id": 0, "array": linear2_w}, {"tensor_id": 1, "array": linear2_b}])
    write_stb(
        weights_dir / "attention.stb",
        [{"tensor_id": 0, "array": attn_q}, {"tensor_id": 1, "array": attn_k}, {"tensor_id": 2, "array": attn_v}],
    )


def compile_module() -> object:
    compiler = KhlnaryCompiler()
    compiler.compile_attention_layer(hidden_size=8, num_heads=2, file_path="weights/attention.stb")
    compiler.compile_linear_layer(
        weight_file="weights/linear1.stb",
        weight_id=0,
        bias_file="weights/linear1.stb",
        bias_id=1,
        weight_shape=(8, 16),
    )
    compiler.knus.append(compiler.encode_glyph("G_GELU"))
    compiler.compile_linear_layer(
        weight_file="weights/linear2.stb",
        weight_id=0,
        bias_file="weights/linear2.stb",
        bias_id=1,
        weight_shape=(16, 8),
    )
    compiler.knus.append(compiler.encode_glyph("G_TENSOR_ADD"))
    return compiler.build_module()


def generate_artifacts(module) -> None:
    cuda = CudaBackend(module)
    Path("generated_kernel.cu").write_text(
        cuda.generate_host_code() + "\n\n" + cuda.generate_kernel_code(),
        encoding="utf-8",
    )
    Path("Makefile").write_text(cuda.generate_makefile(), encoding="utf-8")

    webgpu = WebGpuBackend()
    Path("khlnary.wgsl").write_text(webgpu.generate_wgsl_shader(module), encoding="utf-8")
    Path("khlnary.js").write_text(webgpu.generate_javascript_loader(), encoding="utf-8")

    with Path("transformer_layer.khn").open("wb") as f:
        f.write(struct.pack(f"<{len(module.knus)}I", *module.knus))


def main() -> None:
    create_demo_weights(Path("weights"))
    module = compile_module()
    generate_artifacts(module)
    print(f"Generated {len(module.knus)} KNU words")
    print("Artifacts: weights/*.stb, transformer_layer.khn, generated_kernel.cu, khlnary.wgsl, khlnary.js, Makefile")


if __name__ == "__main__":
    main()
