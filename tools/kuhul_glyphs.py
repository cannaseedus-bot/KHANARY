"""KUHUL glyph catalog used by KHΛNARY v0.2 compiler flows."""

from __future__ import annotations

KUHUL_GLYPHS = {
    # Tensor ops
    "G_LOAD_BIN_TENSOR": {
        "id": 0x30,
        "arity": 0,
        "encoding": {"flags": ["BIN_REF"], "payload": "bin_file_id(4)|tensor_id(4)"},
    },
    "G_TENSOR_MATMUL": {
        "id": 0x40,
        "arity": 2,
        "encoding": {"flags": ["SHAPE_DESC"], "payload": "matmul_flags(8)"},
    },
    "G_TENSOR_ADD": {"id": 0x41, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_TENSOR_CONV2D": {
        "id": 0x42,
        "arity": 2,
        "encoding": {"flags": ["BIN_REF"], "payload": "kernel_id(8)"},
    },
    # Activations
    "G_RELU": {"id": 0x50, "arity": 1, "encoding": {"flags": [], "payload": 0}},
    "G_GELU": {"id": 0x51, "arity": 1, "encoding": {"flags": [], "payload": 0}},
    "G_SOFTMAX": {"id": 0x52, "arity": 1, "encoding": {"flags": ["IMM"], "payload": "axis(8)"}},
    # Attention
    "G_QKV_PROJECTION": {
        "id": 0x60,
        "arity": 3,
        "encoding": {"flags": ["BIN_REF"], "payload": "proj_config(8)"},
    },
    "G_SCALED_DOT_PRODUCT": {
        "id": 0x61,
        "arity": 3,
        "encoding": {"flags": ["IMM"], "payload": "scale_factor(8)"},
    },
    # Control flow
    "G_FORWARD_PASS": {
        "id": 0x70,
        "arity": 1,
        "encoding": {"flags": ["IMM"], "payload": "subgraph_id(8)"},
    },
    "G_BACKWARD_PASS": {
        "id": 0x71,
        "arity": 1,
        "encoding": {"flags": ["IMM"], "payload": "subgraph_id(8)"},
    },
}

FLAG_BITS = {
    "IMM": 0x1,
    "BIN_REF": 0x2,
    "SHAPE_DESC": 0x4,
}

__all__ = ["KUHUL_GLYPHS", "FLAG_BITS"]
