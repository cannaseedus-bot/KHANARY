"""Tests for the KXML inference driver (tools/kxml_inference_driver.py).

Self-contained: builds a tiny GPT-2-shaped model, emits a .stb + manifest via the real graph
builder, and checks the driver's graph-walk equals an independent manual wiring of the same ops.
(The op kernels themselves are separately verified on the iGPU; the HF cross-check runs in __main__.)
"""

import json
import os
import tempfile

import numpy as np

from tools.safetensors_to_model_stb import forward_graph
from tools.stb import write_stb
from tools.kxml_inference_driver import (
    KxmlModel, op_embed, op_layernorm, op_matmul, op_gelu, op_attention)

E, H, L, V, CTX = 32, 2, 2, 40, 16
D = E // H


def _tiny_model(tmp):
    rng = np.random.default_rng(3)
    W = {}
    def add(name, shape): W[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    add("transformer.wte.weight", (V, E)); add("transformer.wpe.weight", (CTX, E))
    for l in range(L):
        p = f"transformer.h.{l}."
        add(p + "ln_1.weight", (E,)); add(p + "ln_1.bias", (E,))
        add(p + "attn.c_attn.weight", (E, 3 * E)); add(p + "attn.c_attn.bias", (3 * E,))
        add(p + "attn.c_proj.weight", (E, E)); add(p + "attn.c_proj.bias", (E,))
        add(p + "ln_2.weight", (E,)); add(p + "ln_2.bias", (E,))
        add(p + "mlp.c_fc.weight", (E, 4 * E)); add(p + "mlp.c_fc.bias", (4 * E,))
        add(p + "mlp.c_proj.weight", (4 * E, E)); add(p + "mlp.c_proj.bias", (E,))
    add("transformer.ln_f.weight", (E,)); add("transformer.ln_f.bias", (E,))
    names = list(W); n2i = {k: i for i, k in enumerate(names)}
    cfg = {"arch": "gpt2", "n_layer": L, "n_embd": E, "n_head": H, "n_ctx": CTX,
           "vocab": V, "ln_eps": 1e-5, "lm_head_tied_to": "transformer.wte.weight"}
    manifest = {"config": cfg,
                "tensors": {k: {"id": n2i[k], "dims": list(W[k].shape), "dtype": "float32"} for k in names},
                "forward_graph": forward_graph(cfg, n2i)}
    stb_path = os.path.join(tmp, "tiny.stb")
    write_stb(stb_path, [{"tensor_id": n2i[k], "array": W[k]} for k in names])
    man_path = os.path.join(tmp, "tiny.stb.json")
    json.dump(manifest, open(man_path, "w", encoding="utf-8"))
    return W, cfg, stb_path, man_path


def _manual_forward(W, cfg, tokens):
    x = op_embed(np.asarray(tokens), W["transformer.wte.weight"], W["transformer.wpe.weight"])
    for l in range(cfg["n_layer"]):
        p = f"transformer.h.{l}."
        r = x; h = op_layernorm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = op_matmul(h, W[p + "attn.c_attn.weight"], W[p + "attn.c_attn.bias"])
        a = op_matmul(op_attention(qkv, cfg["n_head"], D), W[p + "attn.c_proj.weight"], W[p + "attn.c_proj.bias"])
        x = r + a
        r = x; h = op_layernorm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        f = op_gelu(op_matmul(h, W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"]))
        x = r + op_matmul(f, W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"])
    x = op_layernorm(x, W["transformer.ln_f.weight"], W["transformer.ln_f.bias"])
    return op_matmul(x, W["transformer.wte.weight"], transpose_B=True)


def test_graph_walk_equals_manual_wiring():
    with tempfile.TemporaryDirectory() as tmp:
        W, cfg, stb_path, man_path = _tiny_model(tmp)
        m = KxmlModel(stb_path, man_path)
        assert len(m.graph) == 1 + L * 8 + 2      # embed + blocks + ln_f + lm_head
        tokens = [3, 1, 4, 1, 5, 9]
        ours = m.forward(tokens)
        ref = _manual_forward(W, cfg, tokens)
        assert ours.shape == (len(tokens), V)
        assert np.max(np.abs(ours - ref)) < 1e-4


def test_generate_is_deterministic_and_in_vocab():
    with tempfile.TemporaryDirectory() as tmp:
        _, cfg, stb_path, man_path = _tiny_model(tmp)
        m = KxmlModel(stb_path, man_path)
        out = m.generate([7], n=5)
        assert len(out) == 6 and all(0 <= t < V for t in out)
        assert m.generate([7], n=5) == out          # greedy = deterministic


def test_op_kernels():
    assert abs(op_gelu(np.array([0.0]))[0]) < 1e-9
    assert op_gelu(np.array([8.0]))[0] > 7.9        # ~identity for large x
    y = op_layernorm(np.array([[1.0, 2, 3, 4]]), np.ones(4), np.zeros(4))
    assert abs(y.mean()) < 1e-5 and abs(y.std() - 1) < 1e-3
    # causal: query 0 attends only to key 0
    qkv = np.zeros((3, 3), dtype=np.float32); qkv[:, 2] = [10, 20, 30]  # 1 head, D=1, V in col 2
    out = op_attention(qkv, 1, 1)
    assert abs(out[0, 0] - 10) < 1e-4
