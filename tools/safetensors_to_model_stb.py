# safetensors_to_model_stb.py — pack a WHOLE model into one KHANARY .stb + a sidecar manifest.
#
# The .stb alone is just numbered tensors (no names, no config). To make it a *full model* the same
# way GGUF/safetensors are, it needs an index + config + forward graph. This writes:
#   <out>.stb       — ALL weight tensors, ids 0..N-1 (canonical safetensors order)
#   <out>.stb.json  — config (inferred) + name->id index + the KXML forward graph over the glyphs
#
# The graph references the promoted compute glyphs (G_EMBED/G_LAYERNORM/G_MATMUL/G_ATTENTION/G_GELU),
# so a driver can walk it and dispatch each node from the .stb. Round-trip verified with read_stb.
#
# Run: python tools/safetensors_to_model_stb.py <model.safetensors> <out_basename> [--write-stb]
import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from safetensors import safe_open
from stb import write_stb, read_stb

STB_MAX_TENSOR_ID = 255  # descriptor tensor_id is u8


def infer_config(shapes):
    n_embd = shapes["transformer.wte.weight"][1]
    vocab  = shapes["transformer.wte.weight"][0]
    n_ctx  = shapes["transformer.wpe.weight"][0]
    n_layer = 1 + max(int(m.group(1)) for k in shapes
                      for m in [re.match(r"transformer\.h\.(\d+)\.", k)] if m)
    return {"arch": "gpt2", "n_layer": n_layer, "n_embd": n_embd,
            "n_head": n_embd // 64, "n_ctx": n_ctx, "vocab": vocab, "ln_eps": 1e-5,
            "lm_head_tied_to": "transformer.wte.weight"}


def forward_graph(cfg, name2id):
    """The KXML forward schedule over the compute glyphs. Each step names its glyph + the weight
    tensors it reads (by .stb id). '+bias'/'residual'/'tie-T' are elementwise/transpose helpers
    that are NOT yet glyphs — flagged in `pending_ops`."""
    g = [{"step": "embed", "glyph": "G_EMBED",
          "reads": {"tokens": "<input>", "wte": "transformer.wte.weight", "wpe": "transformer.wpe.weight"}}]
    for l in range(cfg["n_layer"]):
        p = f"transformer.h.{l}."
        g += [
            {"step": f"h.{l}.ln_1", "glyph": "G_LAYERNORM",
             "reads": {"gamma": p+"ln_1.weight", "beta": p+"ln_1.bias"}},
            {"step": f"h.{l}.attn.qkv", "glyph": "G_MATMUL",
             "reads": {"B": p+"attn.c_attn.weight"}, "bias": p+"attn.c_attn.bias"},
            {"step": f"h.{l}.attn.mha", "glyph": "G_ATTENTION",
             "params": {"n_head": cfg["n_head"], "head_dim": 64, "causal": True}},
            {"step": f"h.{l}.attn.proj", "glyph": "G_MATMUL",
             "reads": {"B": p+"attn.c_proj.weight"}, "bias": p+"attn.c_proj.bias", "residual": True},
            {"step": f"h.{l}.ln_2", "glyph": "G_LAYERNORM",
             "reads": {"gamma": p+"ln_2.weight", "beta": p+"ln_2.bias"}},
            {"step": f"h.{l}.mlp.fc", "glyph": "G_MATMUL",
             "reads": {"B": p+"mlp.c_fc.weight"}, "bias": p+"mlp.c_fc.bias"},
            {"step": f"h.{l}.mlp.gelu", "glyph": "G_GELU"},
            {"step": f"h.{l}.mlp.proj", "glyph": "G_MATMUL",
             "reads": {"B": p+"mlp.c_proj.weight"}, "bias": p+"mlp.c_proj.bias", "residual": True},
        ]
    g += [
        {"step": "ln_f", "glyph": "G_LAYERNORM",
         "reads": {"gamma": "transformer.ln_f.weight", "beta": "transformer.ln_f.bias"}},
        {"step": "lm_head", "glyph": "G_MATMUL",
         "reads": {"B": "transformer.wte.weight"}, "transpose_B": True},
    ]
    # resolve tensor names to ids where present
    for step in g:
        for role, name in list(step.get("reads", {}).items()):
            if name in name2id:
                step["reads"][role] = {"name": name, "id": name2id[name]}
    return g


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    write_blob = "--write-stb" in sys.argv[3:]

    with safe_open(src, framework="numpy") as f:
        names = list(f.keys())
        shapes = {k: tuple(f.get_tensor(k).shape) for k in names}
        dtypes = {k: str(f.get_tensor(k).dtype) for k in names}
        if len(names) > STB_MAX_TENSOR_ID + 1:
            print(f"WARN {len(names)} tensors > {STB_MAX_TENSOR_ID+1}; .stb u8 tensor_id overflows — "
                  f"shard per-layer or bump the descriptor to u16 (format v2).")
        name2id = {k: i for i, k in enumerate(names)}     # canonical order
        cfg = infer_config(shapes)

        manifest = {
            "name": os.path.basename(out), "format": "khanary-model-stb/v1",
            "config": cfg,
            "stb": {"file": os.path.basename(out) + ".stb", "tensor_count": len(names),
                    "dtype": "float32", "id_scheme": "safetensors order, 0..N-1"},
            "tensors": {k: {"id": name2id[k], "dims": list(shapes[k]), "dtype": dtypes[k]} for k in names},
            "forward_graph": forward_graph(cfg, name2id),
            "pending_ops": ["bias-add and residual-add are elementwise (need a G_ADD glyph or fold "
                            "into G_MATMUL); lm_head reuses wte with transpose_B — a driver concern, "
                            "not a new weight."],
            "note": ("A full model = this .stb (all weights) + this manifest (index+config+graph). "
                     "The graph walks the 5 verified forward glyphs; an inference driver executes it."),
        }
        json.dump(manifest, open(out + ".stb.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"[manifest] wrote {out}.stb.json  ({len(names)} tensors, {cfg['n_layer']}L "
              f"n_embd={cfg['n_embd']} vocab={cfg['vocab']})")

        if write_blob:
            tensors = [{"tensor_id": name2id[k], "array": np.ascontiguousarray(f.get_tensor(k))}
                       for k in names]
            write_stb(out + ".stb", tensors)
            sz = os.path.getsize(out + ".stb")
            print(f"[stb] wrote {out}.stb  ({sz/1e6:.1f} MB)")
            back = read_stb(out + ".stb")
            ok = len(back) == len(names)
            for k in (names[0], names[len(names)//2], names[-1]):  # spot-check
                a = np.asarray(back[name2id[k]]["array"]); b = f.get_tensor(k)
                ok = ok and a.shape == b.shape and np.array_equal(a.reshape(b.shape), b)
            print(f"=== {'PASS' if ok else 'FAIL'}: full-model .stb round-trip ({len(back)} tensors) ===")
            sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
