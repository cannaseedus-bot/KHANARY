<p align="center"><img src=https://github.com/cannaseedus-bot/KHANARY/blob/main/khanary.png style="width:350px;"></p>

## Multi-alphabet Semantic Encoding and Execution Substrate for Deterministic Neural Compute Pipelines

```
KHΛNARY encodes tensor operations and control flow into 32-bit **Knowledge Numeric Unit** (KNU) words using the `KHΛ-2-DENSE-32`
profile, enabling deterministic replay of neural compute workloads on CPU with optional iGPU acceleration via WebGPU.
```
## Architecture

```
Python source / weights
        │
        ▼
┌──────────────────┐
│   KUHUL Layer    │  Semantic glyph definitions (tensor, attention, control flow)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  KHΛNARY Layer   │  32-bit KNU encoding (KHΛ-2-DENSE-32 profile)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Backend Runtime │  Code generation → CPU / WebGPU / D3D11 cs_5_0 (iGPU)
└──────────────────┘
```

**KNU word layout** (32 bits):

| Bits | Field | Purpose |
|------|-------|---------|
| 31–28 | `VER` | Version tag |
| 27–20 | `GLYPH_ID` | Semantic operation |
| 19–16 | `ARITY` | Operand count |
| 15–12 | `FLAGS` | Immediate / bin_ref / shape_desc |
| 11–4 | `PAYLOAD` | Immediate data or descriptor |
| 3–1 | `AUTH_CLASS` | Authority level |
| 0 | `PARITY` | Even parity validation |

**Design philosophy**: CPU-first with lightweight iGPU offload. SVG clusters handle the structural graph layer, keeping the runtime lean — no heavyweight GPU toolkit dependencies.

---

## Geometry backend (D3D11 `cs_5_0` + WGSL)

Geometry is a first-class KHΛNARY opcode: a geometry glyph in the KNU stream drives which kernel the lowering emits. Two additive glyphs — `G_VERTEX_TRANSFORM` (`0x40`) and `G_VERTEX_SKIN` (`0x41`) — lower to **two co-equal backends** from the same stream:

- **D3D11 `cs_5_0`** (`tools/khlnary_dx11.py`) — a byte-addressable vertex transform and weighted-joint skinning (position + normal). **Hardware-verified** on an Intel HD 4600 (feature level 11_1) bit-exact vs CPU (`max abs err 0.00e+00`): transform (256 verts), skinning (128 verts), and the real **30,628-vertex `brain2` birdsong mesh**.
- **WebGPU / WGSL** (`tools/khlnary_webgpu.py`) — structural mirror emitted by the same glyph-driven lowering, for rigs where WebGPU is available. (Not executed on the HD 4600, where WebGPU is blocklisted — the reason the D3D11 backend exists.)

The `brain2` → `.stb` bridge (`tools/brain_to_stb.py`) turns a birdsong spectrogram graph into an SVG-Tensor `.stb`, closing the chain **audio → ridges → graph → `.stb` → KNU glyph → GPU geometry**. The capability is packaged as a versioned model under `models/khanary-geometry-v0.3.0/` (manifest, both backends' kernels, KNU streams, and the mesh data), reproducible via `python tools/build_geometry_model.py`. See its `MODEL.json` for honest scope (HLSL hardware-verified vs WGSL structural-parity; diagonal-matrix test coverage).

---

## Compute glyphs — the five GPT-2 forward ops

Beyond geometry, the same glyph-driven lowering emits **compute** kernels. **All five forward ops of a GPT-2 block are now glyphs**, each promoted from the trainer's HLSL and lowered to both `cs_5_0` HLSL and WGSL:

| Glyph | Op | HD 4600 verification (vs NumPy f64) |
|---|---|---|
| `G_MATMUL` (`0x50`) | dense GEMM `A@B` | real `768×2304` weight → `1.01e-06` |
| `G_ATTENTION` (`0x51`) | causal MHA (softmax + mask) | real QKV, `S=16,E=768,H=12` → `6.39e-08` |
| `G_LAYERNORM` (`0x52`) | LayerNorm (groupshared reduction) | real gpt2 `ln_1` γ/β → `1.27e-07` |
| `G_GELU` (`0x53`) | GELU (tanh approx) | `3.31e-08` |
| `G_EMBED` (`0x54`) | token + positional embedding | `0.00e+00` (exact) |

All dispatched on the iGPU against real GPT-2 tensors from a safetensors checkpoint. What remains for an end-to-end inference *driver* is a per-block schedule (`embed → [ln, attn, ln, ffn+gelu]×N → ln → lm_head`) + a KV cache — not more ops.

`tools/safetensors_to_stb.py` bridges safetensors weights into KHΛNARY `.stb` tensors (the weight-side sibling of `brain_to_stb.py`), so trained weights flow **safetensors → `.stb` → KNU `G_MATMUL` → GPU GEMM**. This is packaged as the compute model `models/khanary-gpt2-v0.4.0/` — matmul kernels, the glyph tokenizer, a real weight `.stb`, and the vendored native D3D11 GPT-2 trainer (reference-only). *Honest scope:* `G_MATMUL` is a naive GEMM, and a full LLM run still needs attention/softmax/layernorm/gelu **glyphs** (the trainer has them as shaders, not yet glyphs) plus GGUF dequant. See its `MODEL.json`.

---

## KXML — the trainable chat-template + tool-call layer

Where llama.cpp uses a prompt-time **chat template** to structure tool calls, KXML structures them as a declarative node stream that lowers to the **glyph tokens the model is trained on**, and whose tool nodes the **runtime dispatches** (the semantic kernel). `models/khanary-kxml-v0.5.0/` registers *all* of it:

- **12 tool calls** (`read_file`, `write_file`, `exec`, `shell`, `tool`, `agent`, `micronaut`, `skill`, `action`, `verb`, `bot`, `http`) emitted as `kuhul.tools.jsonl` — the runtime-loadable registry `kuhul_tool_runtime.h` expects (previously missing).
- **7 compute node ops** (Attention / FFN / LayerNorm / Embed / LmHead / Loss / FieldOptimizer) with the `Pop→Wo→Sek→Chen→Xul` phase machine.
- **Alignment**: each tool → the glyph tokenizer's tool tier (8/12 have a trained token today), each node → a KNU compute glyph (`ATTENTION_NODE→G_ATTENTION`, `FFN_NODE→G_MATMUL`; layernorm/embed/loss/optimizer are trainer shaders, not yet glyphs).

This is the alignment point between the compute glyphs, the glyph tokenizer, and the K'UHUL semantic kernel — the tool-augmented runtime chat, trained in rather than templated at inference.

**Chat template** (`kxml_chat_template.{json,jinja}`): where llama.cpp uses a prompt-time chat template, KXML's is **trained in** — `render_tokens(messages)` emits glyph tokens (roles → `I_EXPLAIN`/`I_QUESTION`/`I_ANSWER`, tool calls → `TOOL_CALL·T_xxx·TOOL_RESULT`, turns wrapped `BOS…SEP…EOS`), with a **llama-compatible `.jinja` surface** for interop.

**Full-model `.stb`**: `tools/safetensors_to_model_stb.py` packs a whole checkpoint into one `.stb` (all weights) + a `.stb.json` manifest = config + name→id index + the forward graph over the five glyphs — the same layering as GGUF/safetensors. Verified on gpt2-small (148 tensors, 124M, 497.8 MB); fits the `.stb` u8 tensor-id (larger models need per-layer sharding or a u16 v2).

---

## Novel innovation — birdsong as *executable geometry*

Today's birdsong AI maps audio to **labels, embeddings, or generated sequences** through learned neural inference:

| System | Role | Method |
|---|---|---|
| **BirdNET** (Cornell) | global species identification | CNN inference, audio → label |
| **TweetyBERT** | self-supervised syntax segmentation | transformer, syllables → latent clusters |
| **FinchGPT** | song-syntax language model | attention over textualized sequences |
| **eBirdNet-Nano** | edge species monitoring | quantized inference on microcontrollers/NPU |

KHΛNARY sits on a **different axis**. It does not classify, embed, or generate — it **compiles the song's spectrogram structure into a deterministic, replayable geometric-compute artifact**. The spectrogram's ridge graph (`brain2`) becomes a Delaunay mesh; the mesh becomes an SVG-Tensor `.stb`; the `.stb` becomes a **KNU glyph stream**; the glyphs drive **real GPU geometry kernels** (vertex transform / skinning). The song is manipulated with GPU *geometry* ops — not convolution or attention.

What is genuinely new here:

- **Deterministic, not inferential.** Same input → bit-identical output (`max abs err 0.00e+00`), fully replayable from the KNU words — the opposite of weight-dependent, stochastic NN inference.
- **Geometry-native representation.** A song is carried as *executable vertex geometry*, so downstream work is transform / skinning / mesh algebra rather than a black-box embedding.
- **Runs where modern GPU stacks don't.** Verified on a frozen **2015 Intel HD 4600 iGPU** (D3D11 feature level 11_1) — no CUDA, no WebGPU (blocklisted on this rig), no heavyweight toolkit. A reach/edge angle that complements the CPU-bound analyzers above.
- **One stream, co-equal backends.** A single KNU glyph stream lowers to both HLSL `cs_5_0` and WGSL.

**Complementary, not competitive.** KHΛNARY makes no species-ID or accuracy claim and is not a benchmark rival to BirdNET et al. — it could *consume* their detections or embeddings. Its contribution is the **deterministic geometric encoding + commodity-iGPU execution substrate** that turns a birdsong graph into hardware-verified, replayable geometry.

---

## Roadmap

### Phase 1 — Specification & Core Encoding ✅

Foundational spec work and the reference encoder/decoder.

- [x] KHΛNARY v0.1 mapping law and `KHΛ-2-DENSE-32` draft layout
- [x] KHΛNARY v0.2 concrete 32-bit profile with tensor/attention glyphs
- [x] KNU encoder/decoder with parity validation
- [x] Lane-bundle format (128-bit, 4 × KNU)
- [x] Python AST → KNU lowering for a compact Python subset

### Phase 1.1 — SVG-Tensor Binary Format ✅

Memory-mapped tensor interchange via `.stb` files.

- [x] STB format v0.1 spec with glyph-to-header wiring
- [x] Deterministic `.stb` writer/reader (Python)
- [x] KHΛNARY payload wiring helpers

### Phase 2 — End-to-End Neural Compute Pipeline ✅

Full compiler toolchain: Python → KHΛNARY KNUs → backend artifacts.

- [x] KUHUL v0.2 glyph catalog (tensor, activation, attention, control flow)
- [x] Formal EBNF grammar for KHΛNARY v0.2
- [x] AST schemas (JSON Schema + Protobuf)
- [x] Backend-lowering rules (CPU / WebGPU contract)
- [x] WebGPU/WGSL skeleton emitter with binding generation
- [x] End-to-end demo pipeline (weights → `.stb` → KNUs → WGSL/JS artifacts)
- [x] Vertical-stack integration tests

### Phase 3 — Validation & Hardening (current)

Strengthen the toolchain for real workloads.

- [ ] Expand test coverage (edge cases, malformed input, round-trip fidelity)
- [ ] CPU native execution backend
- [ ] WebGPU in-browser execution validation (iGPU offload)
- [ ] Parity and authority-class enforcement across all backends
- [ ] Performance profiling of encode/decode and lowering passes

### Phase 3.1 — Geometry ops & native iGPU execution ✅

Real geometry kernels, glyph-driven, executed on hardware.

- [x] Geometry glyphs `G_VERTEX_TRANSFORM` / `G_VERTEX_SKIN` (additive within `KHΛ-2-DENSE-32`)
- [x] Glyph-driven kernel selection in both HLSL and WGSL lowering (co-equal backends)
- [x] D3D11 `cs_5_0` backend: vertex transform + weighted-joint skinning (position + normal)
- [x] Hardware-verified bit-exact vs CPU on Intel HD 4600 (incl. a 30,628-vertex real mesh)
- [x] `brain2` birdsong graph → `.stb` bridge (SVG-Tensor link made real)
- [x] Versioned model folder `models/khanary-geometry-v0.3.0/` with reproducible generator

### Phase 3.2 — Compute glyphs (GEMM) ✅

The first compute op beyond the copy skeleton.

- [x] `G_MATMUL` glyph → dense `cs_5_0` GEMM (HLSL) + WGSL mirror, glyph-driven
- [x] `G_ATTENTION` glyph → causal multi-head attention (softmax + mask), verified on HD 4600
- [x] Verified on Intel HD 4600 with a real GPT-2 weight (scale-normalized err `1.0e-06` vs NumPy)
- [x] `safetensors → .stb` bridge (`tools/safetensors_to_stb.py`)
- [x] Compute model folder `models/khanary-gpt2-v0.4.0/` with glyph tokenizer + vendored trainer
- [x] Layernorm / gelu / embedding compute glyphs (verified on HD 4600) — all 5 fwd ops done
- [ ] End-to-end inference driver: per-block schedule + KV cache wiring the 5 glyphs
- [ ] GGUF → `.stb` dequant path (safetensors is already dense float32)

### Phase 3.3 — KXML tool/op registry (semantic-kernel layer) ✅

The trainable chat-template + tool-call layer, fully enumerated.

- [x] All 12 KXML tool calls registered as `kuhul.tools.jsonl` (runtime-loadable)
- [x] All 7 compute node ops + phase machine (`kxml_nodes.json`)
- [x] Tool→glyph-token and node→KNU-glyph alignment (`kxml_alignment.json`)
- [x] `models/khanary-kxml-v0.5.0/` registry with vendored semantic-kernel source
- [ ] Trained tokens for the 4 unmapped tools (micronaut / action / verb / bot)

### Phase 4 — Extended Glyph Support

Broaden the semantic surface beyond basic tensor ops.

- [ ] Conv2D / pooling operation compilation
- [ ] Full attention-mechanism compilation (QKV projection → scaled dot-product)
- [ ] Normalization glyphs (LayerNorm, BatchNorm)
- [ ] Backward-pass / autograd glyph lowering
- [ ] Custom glyph registration API

### Phase 4B — Ecosystem & Enterprise Features (blueprint)

- [x] Package registry architecture blueprint (templates, search, cache/store, publish/install)
- [x] Community glyph marketplace blueprint (verification, reviews, analytics)
- [x] Multi-tenant authority management blueprint (isolation, quotas, billing hooks)
- [x] Audit/compliance blueprint (structured audit, reports, DSAR workflows, dashboard)
- [x] Integration and rollout plan documented for Python-first KHANARY repository

See `docs/phase4-ecosystem-enterprise.md`.

### Phase 5 — Packaging & Ecosystem

Make KHΛNARY usable as a standalone tool.

- [ ] `pyproject.toml` / installable package
- [ ] CLI entry point (`khanary compile`, `khanary run`)
- [ ] CI pipeline (lint, test, coverage)
- [ ] Documentation site or expanded docs
- [ ] Example model zoo (small networks compiled through the full stack)

---

## Project Structure

```
KHANARY/
├── docs/                          Specifications
│   ├── khlnary-v0.1.md           v0.1 foundational mapping law
│   ├── khlnary-v2.md             v0.2 concrete 32-bit profile
│   ├── stb-format.md             SVG-Tensor Binary format spec
│   ├── lowering-rules.md         Backend-lowering contract (CPU / WebGPU)
│   ├── grammar.ebnf              Formal KHΛNARY v0.2 grammar
│   ├── khlnary-ast.schema.json   JSON Schema for AST nodes
│   └── khlnary-ast.proto         Protobuf AST interchange schema
├── tools/                         Reference implementations
│   ├── khlnary_encoder.py        KNU encoder/decoder + Python AST lowering
│   ├── khlnary_compiler.py       Compiler (KUHUL encoding + .stb registration)
│   ├── kuhul_glyphs.py           KUHUL v0.2 glyph catalog
│   ├── stb.py                    .stb writer/reader
│   ├── khlnary_webgpu.py         KHΛNARY → WGSL emitter (+ geometry kernels)
│   ├── khlnary_dx11.py           KHΛNARY → D3D11 cs_5_0 HLSL backend (+ geometry kernels)
│   ├── brain_to_stb.py           brain2 birdsong graph → SVG-Tensor .stb bridge
│   ├── build_geometry_model.py   generator for the geometry model version folder
│   ├── safetensors_to_stb.py     safetensors weights → SVG-Tensor .stb bridge
│   ├── build_gpt2_model.py       generator for the gpt2 compute model version folder
│   ├── kxml_ops.py               registry of all KXML tool calls + compute node ops
│   ├── build_kxml_registry.py    generator for the KXML tool/op registry folder
│   ├── kxml_chat_template.py     KXML chat template (trained-in tokens + llama .jinja)
│   ├── safetensors_to_model_stb.py  pack a whole checkpoint into one full-model .stb
│   └── demo_end_to_end.py        Full pipeline demo
├── models/                        Versioned KHΛNARY models
│   ├── khanary-geometry-v0.3.0/  Geometry ops: manifest, HLSL+WGSL kernels, KNU, mesh
│   ├── khanary-gpt2-v0.4.0/      Compute: G_MATMUL + G_ATTENTION, glyph tokenizer, vendored trainer
│   └── khanary-kxml-v0.5.0/      KXML tool/op registry: kuhul.tools.jsonl, node ops, alignment
└── tests/                         Test suite
    ├── test_khlnary_encoder.py   KNU codec + parity tests
    ├── test_stb_minimal.py       .stb format tests
    ├── test_lowering_skeletons.py Backend lowering + WGSL geometry-glyph parity tests
    ├── test_dx11_backend.py      D3D11 HLSL backend + geometry-glyph selection tests
    └── test_vertical_stack.py    Full-stack integration tests
```

## Quick Checks

```bash
# Compile-check all modules
python -m compileall tools/kuhul_glyphs.py tools/khlnary_compiler.py tools/khlnary_encoder.py tools/stb.py tools/khlnary_webgpu.py tools/khlnary_dx11.py tools/brain_to_stb.py tools/build_geometry_model.py tools/safetensors_to_stb.py tools/build_gpt2_model.py tools/demo_end_to_end.py

# Run test suite
python -m pytest tests/ -q

# Build the model version folders (kernels emitted from the real lowering)
python tools/build_geometry_model.py
python tools/build_gpt2_model.py
```

## License

See repository for license details.
<img style="text-align:center;" src="https://github.com/cannaseedus-bot/KHANARY/blob/main/khanary.svg" alt="KHΛNARY">
