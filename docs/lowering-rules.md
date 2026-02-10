# KHΛNARY Lowering Rules (Draft)

This document captures the mechanical bridge from structure (`SVG`), to encoded execution (`KHΛNARY`), to concrete backends (`CPU`, `WebGPU`).

## 1. Pipeline

```text
SVG graph + CSS compile directives
  -> KUHUL glyph sequence
  -> KHΛNARY KNU stream
  -> backend lowering (CPU/WebGPU)
```

### 1.1 Layer roles

- **Semantic layer:** KUHUL glyph meaning
- **Encoding layer:** KHΛNARY KNU packing, authority/parity checks
- **Numeric substrate:** `.stb` tensor data (mmap-ready)
- **Execution layer:** backend kernels / dispatch

## 2. Required module metadata

- function table (`func_id -> entry_pc`)
- bin file table (`bin_file_id -> path, alignment, size`)
- shape table (`shape_id -> rank, dims, layout`)

## 3. Glyph-to-runtime rules

## 3.1 `G_LOAD_BIN_TENSOR (0x30)`

Input fields:

- `PAYLOAD[7:4] = bin_file_id`
- `PAYLOAD[3:0] = shape_id`

Runtime steps:

1. Resolve `.stb` path from bin table.
2. Validate `.stb` header/table.
3. Resolve tensor descriptor (`tensor_id == shape_id`) for this profile.
4. Compute `base_ptr = file_base + offset`.
5. Build backend tensor handle (`ptr, dtype, dims, layout, size`).

## 3.2 `G_MMAP_BIN_REGION (0x31)`

Runtime steps:

1. Resolve file by `bin_file_id` in `PAYLOAD`.
2. Map `[data_offset, file_size)` or whole file.
3. Cache mapping and return success if already mapped.

## 3.3 `G_PREFETCH_BIN (0x32)`

Runtime steps:

1. Resolve target file/region from `PAYLOAD`.
2. Issue non-blocking prefetch hint (`madvise` or no-op).
3. Must not alter program behavior.

## 4. Backend projections

### 4.1 CPU

- `G_MMAP_BIN_REGION` -> `mmap`
- `G_LOAD_BIN_TENSOR` -> pointer + typed view creation
- `G_PREFETCH_BIN` -> `madvise(..., MADV_WILLNEED)` when available

### 4.2 WebGPU

- read `.stb` into `ArrayBuffer`
- create storage buffers for selected tensor regions
- bind by deterministic binding index from KNU stream order / metadata

## 5. Determinism constraints

For fixed module bytes + metadata + profile:

- KNU decode must be deterministic
- glyph execution order must be deterministic
- all bin/shape references must be resolvable or produce typed errors

Any decode, authority, parity, or bounds failure must stop execution with typed diagnostics.


## 6. Reference skeleton modules

- `tools/khlnary_webgpu.py`: scans KNUs and emits WGSL binding stubs plus JS loader glue.
- `tools/demo_end_to_end.py`: writes a tiny `.stb`, compiles toy KNUs, and emits WGSL/JS artifacts.
