# KHΛNARY geometry model — v0.3.0

A versioned KHΛNARY **model**: geometry operations expressed as KNU glyph streams and lowered
to two co-equal backends. Geometry is a first-class KHΛNARY opcode — the KNU word selects the
kernel (see `MODEL.json` -> `ops`).

## What this version adds
- Glyphs `G_VERTEX_TRANSFORM` (0x40) and
  `G_VERTEX_SKIN` (0x41) — additive within the `KHΛ-2-DENSE-32`
  v0.2 wire format (KNU VER stays 0x2; this is a capability release).
- Glyph-driven kernel selection in **both** `lower_khlnary_to_hlsl` and `lower_khlnary_to_wgsl`.

## Backends
- **D3D11 `cs_5_0`** — hardware-verified on Intel HD 4600 (FL 11_1). Transform, skinning, and the
  real 30628-vertex birdsong mesh all bit-exact vs CPU (`max abs err 0.00e+00`).
- **WebGPU WGSL** — structural parity (same glyph-driven lowering); not executed on this rig
  (WebGPU blocklisted here).

## Data
`data/birdsong_mesh.stb` — the brain2 canary-song graph as an SVG-Tensor `.stb` (nodes = spectrogram
ridge points). This is the mesh transformed bit-exact on the iGPU in the verification harnesses.

## Reproduce
```
python tools/build_geometry_model.py     # regenerate this folder
python -m pytest tests/ -q               # glyph-selection + parity tests
```
See `MODEL.json` -> `coverage_caveats` for honest scope.
