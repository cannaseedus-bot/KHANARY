"""Minimal SVG-Tensor Binary Format (.stb) writer/reader for KHΛNARY.

This intentionally mirrors the compact reference implementation from the spec draft.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import struct
from typing import Dict, Mapping, MutableMapping, Sequence

_np_spec = importlib.util.find_spec("numpy")
np = importlib.import_module("numpy") if _np_spec is not None else None

STB_MAGIC = b"STB0"
STB_VERSION = 0x01

# dtype -> enum
DTYPE_ENUM = (
    {
        np.float32: 0,
        np.float16: 1,
        np.int8: 2,
        np.int32: 3,
    }
    if np is not None
    else {}
)

# enum -> dtype
ENUM_DTYPE = {v: k for k, v in DTYPE_ENUM.items()}


class StbDependencyError(RuntimeError):
    """Raised when NumPy is not available for .stb read/write operations."""


def _require_numpy():
    if np is None:
        raise StbDependencyError("NumPy is required for .stb read/write operations")
    return np


# ------------------------------------------------------------
# Writer
# ------------------------------------------------------------

def write_stb(path, tensors: Sequence[Mapping[str, object]]):
    """
    Write an .stb file.

    tensors: list of dicts:
      {
        "tensor_id": int,
        "array": numpy array,
        "layout": 0 (row-major),
      }
    """

    np_mod = _require_numpy()

    path = Path(path)
    f = path.open("wb")

    tensor_count = len(tensors)

    # Placeholder header (32 bytes)
    f.write(b"\x00" * 32)

    # Tensor table offset = 32
    tensor_table_offset = f.tell()

    # Write tensor descriptors (32 bytes each)
    descriptors = []
    for t in tensors:
        arr = np_mod.asarray(t["array"])
        tid = int(t["tensor_id"])
        dtype_enum = DTYPE_ENUM[arr.dtype.type]
        rank = arr.ndim
        layout = int(t.get("layout", 0))

        # Raw data offset will be filled later
        descriptors.append(
            {
                "tensor_id": tid,
                "dtype": dtype_enum,
                "rank": rank,
                "layout": layout,
                "dims": arr.shape,
                "offset": None,
                "size_bytes": arr.nbytes,
            }
        )

        # Write placeholder descriptor
        f.write(b"\x00" * 32)

    # Align to 64 bytes for data region
    def align64(x):
        return (x + 63) & ~63

    data_offset = align64(f.tell())
    f.write(b"\x00" * (data_offset - f.tell()))

    # Write raw tensor data
    for d, t in zip(descriptors, tensors):
        arr = np_mod.asarray(t["array"])
        d["offset"] = f.tell()
        f.write(arr.tobytes(order="C"))

    file_size = f.tell()

    # ------------------------------------------------------------
    # Rewrite header
    # ------------------------------------------------------------
    f.seek(0)
    header = struct.pack(
        "<4sBBHIIQQ",
        STB_MAGIC,
        STB_VERSION,
        0,  # flags
        tensor_count,
        0,
        0,  # reserved
        data_offset,
        file_size,
    )
    f.write(header)

    # ------------------------------------------------------------
    # Rewrite tensor descriptors
    # ------------------------------------------------------------
    f.seek(tensor_table_offset)

    for d in descriptors:
        dims = list(d["dims"])[:3]
        dims += [0] * (3 - len(dims))

        entry = struct.pack(
            "<BBBBQQLLL",
            d["tensor_id"],
            d["dtype"],
            d["rank"],
            d["layout"],
            d["offset"],
            d["size_bytes"],
            dims[0],
            dims[1],
            dims[2],
        )
        f.write(entry)

    f.close()
    return path


# ------------------------------------------------------------
# Reader
# ------------------------------------------------------------

def read_stb(path):
    np_mod = _require_numpy()

    path = Path(path)
    f = path.open("rb")

    # Header
    header = f.read(32)
    magic, version, flags, tensor_count, r0, r1, data_offset, file_size = struct.unpack("<4sBBHIIQQ", header)

    if magic != STB_MAGIC:
        raise ValueError("Invalid STB magic")
    if version != STB_VERSION:
        raise ValueError(f"Unsupported STB version: {version}")
    if flags != 0:
        raise ValueError(f"Unsupported STB flags: {flags}")

    # Tensor table
    tensors: Dict[int, MutableMapping[str, object]] = {}
    for _ in range(tensor_count):
        entry = f.read(32)
        (tid, dtype_enum, rank, layout, offset, size_bytes, d0, d1, d2) = struct.unpack("<BBBBQQLLL", entry)

        if dtype_enum not in ENUM_DTYPE:
            raise ValueError(f"Unsupported dtype enum: {dtype_enum}")

        dims = [d0, d1, d2][:rank]
        dtype = ENUM_DTYPE[dtype_enum]

        tensors[tid] = {
            "dtype": dtype,
            "rank": rank,
            "layout": layout,
            "offset": offset,
            "size_bytes": size_bytes,
            "dims": dims,
        }

    # Raw data region
    raw = f.read()
    f.close()

    # Materialize arrays
    for tid, meta in tensors.items():
        start = meta["offset"] - data_offset
        end = start + meta["size_bytes"]
        buf = raw[start:end]
        arr = np_mod.frombuffer(buf, dtype=meta["dtype"])
        if meta["rank"] <= 3:
            arr = arr.reshape(meta["dims"])
        meta["array"] = arr

    return tensors


# ------------------------------------------------------------
# KHΛNARY payload wiring helpers
# ------------------------------------------------------------

def decode_load_bin_tensor_payload(payload: int) -> tuple[int, int]:
    """Decode KHΛNARY v0.2 payload into (bin_file_id, shape_id)."""

    if not 0 <= payload <= 255:
        raise ValueError("payload must be an 8-bit value")
    return ((payload >> 4) & 0xF, payload & 0xF)


def resolve_khlnary_tensor(bin_file_table: Mapping[int, str | Path], payload: int) -> MutableMapping[str, object]:
    """Resolve a KHΛNARY payload to a tensor descriptor/array from .stb files."""

    bin_file_id, shape_id = decode_load_bin_tensor_payload(payload)
    if bin_file_id not in bin_file_table:
        raise KeyError(f"Unknown bin_file_id: {bin_file_id}")

    tensors = read_stb(bin_file_table[bin_file_id])
    if shape_id not in tensors:
        raise KeyError(f"Tensor id {shape_id} not found in bin_file_id={bin_file_id}")
    return tensors[shape_id]


__all__ = [
    "StbDependencyError",
    "write_stb",
    "read_stb",
    "decode_load_bin_tensor_payload",
    "resolve_khlnary_tensor",
]
