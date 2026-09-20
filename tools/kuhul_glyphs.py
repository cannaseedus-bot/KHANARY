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
    # Fixed-size native buffer block operations. The layout is ABI-compatible
    # with Win2D Matrix5x4. These are extension glyphs in the tensor space.
    "G_MATRIX5X4_MUL": {"id": 0x80, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_MATRIX5X4_MATMUL": {"id": 0x81, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_MATRIX5X4_ADD": {"id": 0x82, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_MATRIX5X4_SWIGLU": {"id": 0x83, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_SOBEL_X": {"id": 0x100, "arity": 1, "encoding": {"flags": [], "payload": 0}},
    "G_SOBEL_Y": {"id": 0x101, "arity": 1, "encoding": {"flags": [], "payload": 0}},
    "G_SOBEL_MAGNITUDE": {"id": 0x102, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_ELEMENTWISE_MUL": {"id": 0x103, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_REDUCE_PAIR_SUM": {"id": 0x104, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_SOBEL_THRESHOLD": {"id": 0x105, "arity": 1, "encoding": {"flags": ["IMM"], "payload": "thresholds(16|16)"}},
    "G_SWIGLU": {"id": 0x110, "arity": 3, "encoding": {"flags": [], "payload": 0}},
    "G_SILU": {"id": 0x111, "arity": 1, "encoding": {"flags": [], "payload": 0}},
    "G_SOFTMAX_BACK": {"id": 0x112, "arity": 1, "encoding": {"flags": [], "payload": 0}},
    "G_RMS_NORM_BACK": {"id": 0x113, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_SILU_BACK": {"id": 0x114, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_GELU_BACK": {"id": 0x115, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_ADAMW": {"id": 0x116, "arity": 4, "encoding": {"flags": ["IMM"], "payload": "optimizer_config"}},
    "G_SGD": {"id": 0x117, "arity": 2, "encoding": {"flags": ["IMM"], "payload": "optimizer_config"}},
    "G_RMS_NORM": {"id": 0x118, "arity": 1, "encoding": {"flags": ["IMM"], "payload": "eps"}},
    "G_CROSS_ENTROPY_BACK": {"id": 0x119, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_HF_LOAD": {"id": 0x200, "arity": 0, "encoding": {"flags": ["BIN_REF"], "payload": "model_uri"}},
    "G_HF_MATMUL": {"id": 0x201, "arity": 2, "encoding": {"flags": [], "payload": 0}},
    "G_HF_ATTENTION": {"id": 0x202, "arity": 4, "encoding": {"flags": [], "payload": 0}},
    "G_HF_SWIGLU": {"id": 0x203, "arity": 3, "encoding": {"flags": [], "payload": 0}},
    "G_HF_RMS_NORM": {"id": 0x204, "arity": 1, "encoding": {"flags": ["IMM"], "payload": "eps"}},
}

FLAG_BITS = {
    "IMM": 0x1,
    "BIN_REF": 0x2,
    "SHAPE_DESC": 0x4,
}

__all__ = ["KUHUL_GLYPHS", "FLAG_BITS"]
