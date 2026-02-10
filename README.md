# KHΛNARY

<img src="https://github.com/cannaseedus-bot/KHANARY/blob/main/khanary.svg" alt="KHΛNARY">

Multi-alphabet semantic encoding and execution substrate for deterministic neural compute pipelines.

- `docs/khlnary-v0.1.md`: foundational KHΛNARY mapping law, concrete `KHΛ-2-DENSE-32` draft layout, extended control-flow/function glyphs, and lane-bundle format.
- `docs/khlnary-v2.md`: concrete 32-bit (`KHΛ-2-DENSE-32`) profile and glyph details for the v0.2 stream.
- `docs/stb-format.md`: SVG-Tensor Binary Format (`.stb`) v0.1 with glyph-to-header wiring.
- `docs/lowering-rules.md`: backend-lowering contract from KHΛNARY glyphs to CPU/CUDA/WebGPU runtime behavior.
- `docs/grammar.ebnf`: formal KHΛNARY v0.2 grammar (KUHUL symbolic layer + glyph-level forms).
- `docs/khlnary-ast.schema.json`: JSON Schema for KHΛNARY AST nodes and statements.
- `docs/khlnary-ast.proto`: protobuf schema for KHΛNARY AST interchange.

## Architecture

- `tools/khlnary_encoder.py`: reference KHΛNARY encoder/decoder, Python AST lowering, and lane-bundle helpers.
- `tools/stb.py`: minimal deterministic `.stb` writer/reader and KHΛNARY payload wiring helpers.
- `tools/kuhul_glyphs.py`: KUHUL v0.2 glyph catalog for tensor/attention/control ops.
- `tools/khlnary_compiler.py`: compiler helpers for KUHUL glyph encoding + `.stb` tensor registration.
- `tools/khlnary_cuda.py`: KHΛNARY -> CUDA skeleton emitter plus module-oriented backend class.
- `tools/khlnary_webgpu.py`: KHΛNARY -> WGSL/JS skeleton emitter plus module-oriented backend class.
- `tools/demo_end_to_end.py`: end-to-end demo (`.stb` + KHΛNARY + CUDA/WebGPU artifact generation).
- `tools/run_inference.py`: ctypes runtime bridge for invoking generated CUDA module entry points.

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
python -m compileall tools/kuhul_glyphs.py tools/khlnary_compiler.py tools/khlnary_encoder.py tools/stb.py tools/khlnary_cuda.py tools/khlnary_webgpu.py tools/demo_end_to_end.py tools/run_inference.py
python -m unittest tests/test_khlnary_encoder.py tests/test_stb_minimal.py tests/test_lowering_skeletons.py tests/test_vertical_stack.py
```

## License

See repository for license details.
