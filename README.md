# KHΛNARY

<img src="https://github.com/cannaseedus-bot/KHANARY/blob/main/khanary.svg" alt="KHΛNARY">

Multi-alphabet semantic encoding and execution substrate for deterministic neural compute pipelines.

KHΛNARY encodes tensor operations and control flow into 32-bit **Knowledge Numeric Unit** (KNU) words using the `KHΛ-2-DENSE-32` profile, enabling deterministic replay of neural compute workloads across CPU, CUDA, and WebGPU backends.

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
│  Backend Runtime │  Code generation → CUDA C++ / WGSL / CPU
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
- [x] Backend-lowering rules (CPU / CUDA / WebGPU contract)
- [x] CUDA skeleton emitter with tensor loading and host init
- [x] WebGPU/WGSL skeleton emitter with binding generation
- [x] End-to-end demo pipeline (weights → `.stb` → KNUs → artifacts)
- [x] ctypes runtime bridge for CUDA module invocation
- [x] Vertical-stack integration tests

### Phase 3 — Validation & Hardening (current)

Strengthen the toolchain for real workloads.

- [ ] Expand test coverage (edge cases, malformed input, round-trip fidelity)
- [ ] GPU-native execution tests (CUDA device validation)
- [ ] WebGPU in-browser execution validation
- [ ] Parity and authority-class enforcement across all backends
- [ ] Performance profiling of encode/decode and lowering passes

### Phase 4 — Extended Glyph Support

Broaden the semantic surface beyond basic tensor ops.

- [ ] Conv2D / pooling operation compilation
- [ ] Full attention-mechanism compilation (QKV projection → scaled dot-product)
- [ ] Normalization glyphs (LayerNorm, BatchNorm)
- [ ] Backward-pass / autograd glyph lowering
- [ ] Custom glyph registration API

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
│   ├── lowering-rules.md         Backend-lowering contract
│   ├── grammar.ebnf              Formal KHΛNARY v0.2 grammar
│   ├── khlnary-ast.schema.json   JSON Schema for AST nodes
│   └── khlnary-ast.proto         Protobuf AST interchange schema
├── tools/                         Reference implementations
│   ├── khlnary_encoder.py        KNU encoder/decoder + Python AST lowering
│   ├── khlnary_compiler.py       Compiler (KUHUL encoding + .stb registration)
│   ├── kuhul_glyphs.py           KUHUL v0.2 glyph catalog
│   ├── stb.py                    .stb writer/reader
│   ├── khlnary_cuda.py           KHΛNARY → CUDA skeleton emitter
│   ├── khlnary_webgpu.py         KHΛNARY → WGSL/JS skeleton emitter
│   ├── demo_end_to_end.py        Full pipeline demo
│   └── run_inference.py          ctypes CUDA runtime bridge
└── tests/                         Test suite
    ├── test_khlnary_encoder.py   KNU codec + parity tests
    ├── test_stb_minimal.py       .stb format tests
    ├── test_lowering_skeletons.py Backend lowering tests
    └── test_vertical_stack.py    Full-stack integration tests
```

## Quick Checks

```bash
# Compile-check all modules
python -m compileall tools/kuhul_glyphs.py tools/khlnary_compiler.py tools/khlnary_encoder.py tools/stb.py tools/khlnary_cuda.py tools/khlnary_webgpu.py tools/demo_end_to_end.py tools/run_inference.py

# Run test suite
python -m unittest tests/test_khlnary_encoder.py tests/test_stb_minimal.py tests/test_lowering_skeletons.py tests/test_vertical_stack.py

# End-to-end demo
python tools/demo_end_to_end.py
```

## License

See repository for license details.
