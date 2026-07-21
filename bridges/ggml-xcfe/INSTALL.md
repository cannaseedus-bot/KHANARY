# Installing the KHΛNARY `ggml-xcfe` backend into llama.cpp

This is a **real, distinct** ggml backend (not the orphan `ggml-xcfe` copy of `ggml-webgpu`
that ships in some llama.cpp trees — delete that one first if present). It is modelled on the
BLAS backend: host-memory buffers, claims only `GGML_OP_MUL_MAT`, everything else falls back to
the CPU backend.

## What it does today (honest scope)

- Registers as backend **`XCFE`** with its own GUID via `ggml_backend_xcfe_reg()`.
- The ggml scheduler routes **F32 matmuls ≥ 32 in each dim** to it; all other ops go to CPU.
- `graph_compute` runs those matmuls with a **portable F32 reference GEMM** — the placeholder
  for the KHΛNARY glyph dispatch. The single function `ggml_backend_xcfe_gemm_f32` in
  `ggml-xcfe.cpp` is the seam where the verified **D3D11 `cs_5_0` `G_MATMUL`** kernel — or
  **DirectML** (`DML_OPERATOR_GEMM`, ~4.9× faster than our tiled kernel on the HD 4600) — plugs
  in. Until then results are correct but computed on the CPU.
- Device type is reported as `ACCEL` (honest while compute is on CPU); flip to `IGPU` when the
  GPU dispatch is wired.

## Three steps to wire it in

1. **Copy the sources** into your llama.cpp checkout:
   ```
   cp ggml-xcfe.h                 llama.cpp/ggml/include/ggml-xcfe.h
   mkdir -p                       llama.cpp/ggml/src/ggml-xcfe
   cp ggml-xcfe.cpp CMakeLists.txt llama.cpp/ggml/src/ggml-xcfe/
   ```

2. **Register the backend** in `llama.cpp/ggml/src/CMakeLists.txt` — add next to the others:
   ```cmake
   ggml_add_backend(XCFE)
   ```
   and add an option in `llama.cpp/ggml/CMakeLists.txt` (near the other `GGML_*` backend opts):
   ```cmake
   option(GGML_XCFE "ggml: use KHANARY XCFE backend" OFF)
   ```
   The `ggml_add_backend(XCFE)` macro reads `GGML_XCFE`, adds the `ggml-xcfe` subdirectory, and
   links the target through the `ggml` umbrella + registry.

3. **Configure & build** with the backend on:
   ```
   cmake -B build -DGGML_XCFE=ON
   cmake --build build --config Release
   ```

## Verify it registered

```
./build/bin/llama-cli --list-devices
# expect an "XCFE" device in the list
```
Or run any model — matmuls will be scheduled onto XCFE. To confirm it's actually taking the
matmuls, put a print/breakpoint in `ggml_backend_xcfe_gemm_f32`.

## Notes / limits

- **F32 only.** Quantized `src0` (the usual GGUF case) is rejected by `supports_op` — it would
  need a dequant step first (the same GGUF→`.stb` dequant gap tracked on the KHΛNARY side).
- The reference GEMM is single-threaded and unoptimized on purpose — it exists to prove the
  backend wiring, not to be fast. Speed comes from wiring the glyph/DirectML dispatch into the
  seam function.
- Built against ggml's `GGML_BACKEND_API_VERSION 2` (`ggml-backend-impl.h`). If your llama.cpp
  is much newer/older and the backend struct layout changed, re-check the vtable field order.
- Authored against the real ggml headers but **not link-tested in this repo** (no ggml build
  here); compile it inside your llama.cpp tree.
