# KHΛNARY gpt2 compute model — v0.4.0

The **compute** counterpart to the geometry model: this version packages KHΛNARY's first
compute glyph (`G_MATMUL`) together with the native gpt2 trainer that produces the weights and
the glyph tokenizer that feeds it.

## Contents
- **`kernels/matmul.{hlsl,wgsl}`** — dense GEMM `C[M,N] = A[M,K] @ B[K,N]`, emitted from the
  `G_MATMUL` glyph (`0x50`). The HLSL was **dispatched on an Intel HD 4600**
  with a real gpt2 QKV weight: scale-normalized error `1.01e-06` vs numpy. See `MODEL.json`.
- **`tokenizer/`** — `glyph_tokenizer.py` (`GlyphTokenizer`) + `glyph_contract.json` (the GPT-2
  token↔glyph mapping).
- **`trainer/`** — the vendored native D3D11 `cs_5_0` gpt2 trainer (`gpt2_trainer.cpp` + backward
  `gpu_fwdbwd_new.cpp` + `shaders/gpt2_*.hlsl`), its physics header, and the engine headers
  (`engine/d3d11_engine.h`, `engine/xvm_core.h`). **Reference-only** — links against a prebuilt
  external xvm engine object, so it does not build standalone here (see `MODEL.json` →
  `trainer.build_requirements`).
- **`data/gpt2_c_proj.stb`** — a real gpt2 weight in KHΛNARY `.stb` form via
  `tools/safetensors_to_stb.py` (the weight-side bridge, sibling to `brain_to_stb.py`).

## Honest scope
`G_MATMUL` is a naive GEMM (correctness-first). A full LLM run still needs attention/softmax/
layernorm/gelu **glyphs** — the trainer already implements these as HLSL shaders, but they are
not yet exposed as KNU glyphs. GGUF would additionally need a dequant → `.stb` step; safetensors
is already dense float32. See `MODEL.json` → `honest_scope`.

## Reproduce
```
python tools/build_gpt2_model.py          # regenerate this folder (needs the .ASX.cpp/.KUHUL_V2 sources)
python tools/safetensors_to_stb.py <model.safetensors> out.stb <keys...>
python -m pytest tests/ -q
```
