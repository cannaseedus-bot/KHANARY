"""KHΛNARY v0.2 compiler helpers for .stb-backed neural programs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Tuple

from tools.kuhul_glyphs import FLAG_BITS, KUHUL_GLYPHS
from tools import stb


DTYPE_BY_STB_ENUM = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
}
LAYOUT_BY_STB_ENUM = {
    0: "row_major",
    1: "col_major",
    2: "channels_last",
}


def _dtype_size(dtype: str) -> int:
    return {
        "float16": 2,
        "float32": 4,
        "int8": 1,
        "int32": 4,
    }[dtype]


@dataclass
class StbTensor:
    file_id: int
    tensor_id: int
    dtype: str
    shape: Tuple[int, ...]
    offset: int = 0
    layout: str = "row_major"

    @property
    def size_bytes(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size * _dtype_size(self.dtype)

    @property
    def cuda_ptr_name(self) -> str:
        return f"stb_{self.file_id}_{self.tensor_id}_dev"


@dataclass
class KhlnaryModule:
    knus: List[int]
    bin_files: Dict[int, str]
    tensors: List[StbTensor]
    functions: Dict[int, int] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=lambda: {"version": "KHΛNARY-2"})


class KhlnaryCompiler:
    """Compiler from KUHUL glyph names into KHΛNARY words + .stb metadata."""

    def __init__(self) -> None:
        self.knus: List[int] = []
        self.bin_files: Dict[int, str] = {}
        self.file_ids_by_path: Dict[str, int] = {}
        self.tensors: List[StbTensor] = []
        self.functions: Dict[int, int] = {}

    @staticmethod
    def _compute_parity(word: int) -> int:
        return ((word & 0xFFFFFFFE) >> 1).bit_count() & 0x1

    def encode_glyph(self, glyph_name: str, payload: int = 0) -> int:
        glyph = KUHUL_GLYPHS[glyph_name]
        flags = 0
        for flag_name in glyph["encoding"]["flags"]:
            flags |= FLAG_BITS[flag_name]

        word = 0
        word |= 0x2 << 28  # KHΛNARY v0.2
        word |= (glyph["id"] & 0xFF) << 20
        word |= (glyph["arity"] & 0xF) << 16
        word |= (flags & 0xF) << 12
        word |= (payload & 0xFF) << 4
        word |= 0x1 << 1  # auth=user
        word |= self._compute_parity(word)
        return word

    def _register_file(self, file_path: str) -> int:
        norm = str(Path(file_path))
        if norm not in self.file_ids_by_path:
            file_id = len(self.file_ids_by_path)
            if file_id > 15:
                raise ValueError("KHΛNARY v0.2 payload supports only 16 .stb files per module")
            self.file_ids_by_path[norm] = file_id
            self.bin_files[file_id] = norm
        return self.file_ids_by_path[norm]

    def add_stb_tensor(self, file_path: str, tensor_id: int, dtype: str, shape: Tuple[int, ...]) -> int:
        """Register tensor; if file exists, read exact offset/shape/dtype from .stb."""
        file_id = self._register_file(file_path)
        tensor = StbTensor(file_id=file_id, tensor_id=tensor_id, dtype=dtype, shape=shape)

        path = Path(file_path)
        if path.exists() and stb.np is not None:
            try:
                tensors = stb.read_stb(path)
                if tensor_id in tensors:
                    meta = tensors[tensor_id]
                    tensor.offset = int(meta["offset"])
                    tensor.shape = tuple(int(d) for d in meta["dims"])
                    tensor.dtype = DTYPE_BY_STB_ENUM.get(stb.DTYPE_ENUM.get(meta["dtype"], -1), dtype)
                    tensor.layout = LAYOUT_BY_STB_ENUM.get(int(meta["layout"]), "row_major")
            except (ValueError, KeyError, stb.StbDependencyError):
                pass

        self.tensors.append(tensor)
        return file_id

    def compile_linear_layer(
        self,
        *,
        weight_file: str,
        weight_id: int,
        bias_file: str,
        bias_id: int,
        weight_shape: Tuple[int, ...],
    ) -> None:
        w_file_id = self.add_stb_tensor(weight_file, weight_id, "float16", weight_shape)
        b_file_id = self.add_stb_tensor(bias_file, bias_id, "float16", (weight_shape[1],))
        self.knus.append(self.encode_glyph("G_LOAD_BIN_TENSOR", payload=(w_file_id << 4) | weight_id))
        self.knus.append(self.encode_glyph("G_LOAD_BIN_TENSOR", payload=(b_file_id << 4) | bias_id))
        self.knus.append(self.encode_glyph("G_TENSOR_MATMUL"))
        self.knus.append(self.encode_glyph("G_TENSOR_ADD"))

    def compile_attention_layer(self, *, hidden_size: int, num_heads: int, file_path: str) -> None:
        for tensor_id in (0, 1, 2):
            file_id = self.add_stb_tensor(file_path, tensor_id, "float16", (hidden_size, hidden_size))
            self.knus.append(self.encode_glyph("G_LOAD_BIN_TENSOR", payload=(file_id << 4) | tensor_id))
        scale = int((1.0 / math.sqrt(hidden_size // num_heads)) * 256)
        self.knus.append(self.encode_glyph("G_SCALED_DOT_PRODUCT", payload=max(0, min(scale, 255))))

    def build_module(self) -> KhlnaryModule:
        return KhlnaryModule(
            knus=self.knus.copy(),
            bin_files=self.bin_files.copy(),
            tensors=self.tensors.copy(),
            functions=self.functions.copy(),
            metadata={
                "version": "KHΛNARY-2",
                "knu_count": len(self.knus),
                "tensor_count": len(self.tensors),
            },
        )


__all__ = ["StbTensor", "KhlnaryModule", "KhlnaryCompiler"]
