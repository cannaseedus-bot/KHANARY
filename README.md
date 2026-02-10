# KHΛNARY

<img src="https://github.com/cannaseedus-bot/KHANARY/blob/main/khanary.svg" alt="KHΛNARY">

## Specs

- `docs/khlnary-v0.1.md`: foundational KHΛNARY mapping law, concrete `KHΛ-2-DENSE-32` draft layout, extended control-flow/function glyphs, and lane-bundle format.
- `docs/khlnary-v2.md`: concrete 32-bit (`KHΛ-2-DENSE-32`) profile and glyph details for the v0.2 stream.
- `docs/stb-format.md`: SVG-Tensor Binary Format (`.stb`) v0.1 with glyph-to-header wiring.
- `docs/lowering-rules.md`: backend-lowering contract from KHΛNARY glyphs to CPU/CUDA/WebGPU runtime behavior.

## Utilities

- `tools/khlnary_encoder.py`: reference KHΛNARY encoder/decoder, Python AST lowering, and lane-bundle helpers.
- `tools/stb.py`: minimal deterministic `.stb` writer/reader and KHΛNARY payload wiring helpers.
- `tools/khlnary_cuda.py`: KHΛNARY -> CUDA skeleton emitter.
- `tools/khlnary_webgpu.py`: KHΛNARY -> WGSL/JS skeleton emitter.
- `tools/demo_end_to_end.py`: tiny vertical slice demo (`.stb` + KHΛNARY + CUDA source generation).

## Quick checks

```bash
python -m compileall tools/khlnary_encoder.py tools/stb.py tools/khlnary_cuda.py tools/khlnary_webgpu.py tools/demo_end_to_end.py
python -m unittest tests/test_khlnary_encoder.py tests/test_stb_minimal.py tests/test_lowering_skeletons.py
```
