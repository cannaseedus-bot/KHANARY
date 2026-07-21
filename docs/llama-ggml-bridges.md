# KHΛNARY ↔ llama.cpp `ggml` runtime bridges

KHΛNARY's tiered backend story (CPU floor → reach GPU → browser GPU → native glyph
runtime) maps directly onto llama.cpp's `ggml` **backend registry**. This note explains the
four backend folders that live under **`llama.cpp/ggml/src/`**, how `ggml` discovers a
backend, and — honestly — the current state of the KHΛNARY-native `ggml-xcfe` slot.

> These are not shipped inside KHΛNARY. They are directories in *your* `llama.cpp` checkout
> (`llama.cpp/ggml/src/<backend>/`). This doc is the map: which backend plays which role in
> KHΛNARY, and what it takes to add a KHΛNARY-native one.

## The four backends and their KHΛNARY role

| `ggml/src/` folder | KHΛNARY tier | Status | Notes |
|---|---|---|---|
| `ggml-cpu` | **CPU-first floor** — the portable, always-works baseline | Stock upstream | The tier KHΛNARY treats as the correctness floor everywhere. |
| `ggml-opencl` | **REACH backend** — runs where WebGPU is blocklisted | Stock upstream | On this rig (Intel HD 4600, Dawn/WebGPU blocklisted) OpenCL is the GPU path that still runs. It's a *reach* play (runs broadly), not a *speed* play. |
| `ggml-webgpu` | **Browser / WGSL tier** — WebGPU → WebGL2(ANGLE→D3D11) → CPU | Stock upstream | Ties to KHΛNARY's own WGSL glyph lowering (`tools/khlnary_webgpu.py`): both target WGSL. |
| `ggml-xcfe` | **KHΛNARY-native slot** (intended) | ⚠️ **Not a real backend yet — see below** | The place a KHΛNARY-native `ggml` backend *would* live. |

## How `ggml` registers a backend (so you know where a bridge plugs in)

Backends are wired in **`ggml/src/CMakeLists.txt`** with one line each:

```cmake
ggml_add_backend(CPU)
ggml_add_backend(WebGPU)
ggml_add_backend(OpenCL)
# ...
```

`ggml_add_backend(NAME)` (defined in that same CMakeLists) does, in effect:

- reads the option **`GGML_<NAME>`** (e.g. `GGML_WEBGPU`, `GGML_XCFE`),
- if set, adds the subdirectory **`ggml-<name>`** (e.g. `ggml-webgpu`, `ggml-xcfe`),
- that subdirectory's `CMakeLists.txt` calls **`ggml_add_backend_library(ggml-<name> …)`**,
  which links against `ggml-base` and registers the target with the `ggml` umbrella.

At runtime the backend exposes a `ggml_backend_<name>_reg()` returning a `ggml_backend_reg_t`,
and the umbrella `ggml_backend_registry` enumerates it. That registration function + the
compute-graph vtable (`supports_op`, `graph_compute`, buffer types) is what a real backend
implements.

**Install a bridge folder:** drop `ggml-<name>/` into `llama.cpp/ggml/src/`, add
`ggml_add_backend(<Name>)` to `ggml/src/CMakeLists.txt`, and configure with `-DGGML_<NAME>=ON`.

## ⚠️ Honest status of `ggml-xcfe`

`ggml-xcfe/` in this tree is **a byte-identical copy of `ggml-webgpu/`** (`diff -r` reports no
differences). Concretely:

- Its `CMakeLists.txt` calls `ggml_add_backend_library(ggml-webgpu …)` — it declares the
  **webgpu** target, not an `xcfe` one. So it is not "an xcfe backend that needs customizing";
  it is a **duplicate of webgpu** that would collide with the real `ggml-webgpu`.
- Nothing outside the folder references it: there is **no `ggml_add_backend(XCFE)`** in
  `ggml/src/CMakeLists.txt` and **no `GGML_XCFE` option**, so it is an orphan directory — not
  wired into the build and never compiled.

In short: **there is no working KHΛNARY-native `ggml` backend today.** The folder is a
placeholder name, not an implementation. Don't ship it as one.

## Making `ggml-xcfe` a real KHΛNARY backend (the actual work)

A genuine XCFE backend is a distinct target that lowers `ggml` compute-graph ops to KHΛNARY
glyph kernels (the same KLSL→WGSL(+HLSL) path the rest of KHΛNARY uses). It requires:

1. **Rename the target throughout** the folder: `ggml_add_backend_library(ggml-xcfe …)`,
   a real `../../include/ggml-xcfe.h`, and `ggml-xcfe.cpp` (not the webgpu sources).
2. **Wire the build:** add `ggml_add_backend(XCFE)` to `ggml/src/CMakeLists.txt` + a
   `GGML_XCFE` option.
3. **Implement the backend interface:** `ggml_backend_xcfe_reg()` + device/buffer types +
   `graph_compute` that maps `GGML_OP_MUL_MAT`/`SOFT_MAX`/`NORM`/etc. onto KHΛNARY glyphs
   (`G_MATMUL`, `G_ATTENTION`, `G_LAYERNORM`, `G_GELU`, `G_EMBED` — the kernels already verified
   on the HD 4600), emitted via the glyph-lowering backends.
4. **Pick the device path this rig actually supports:** D3D11 `cs_5_0` (hardware-verified here)
   or WGSL through the browser tier — not full D3D12 (this iGPU caps at FL 11_1).

**This backend now exists** in KHΛNARY at [`bridges/ggml-xcfe/`](../bridges/ggml-xcfe/): a real,
distinct, BLAS-class backend (host buffers, claims only `MUL_MAT`, registers as `XCFE` with its
own GUID). Its `graph_compute` currently runs matmuls with a portable F32 reference GEMM — the
seam (`ggml_backend_xcfe_gemm_f32`) is where the D3D11 `cs_5_0` `G_MATMUL` / DirectML dispatch
plugs in. See [`bridges/ggml-xcfe/INSTALL.md`](../bridges/ggml-xcfe/INSTALL.md) for the exact
three-step wiring. Replace the orphan `ggml-webgpu`-copy `ggml-xcfe` with it.

Until the GPU dispatch is wired, `ggml-cpu` (floor) / `ggml-opencl` (reach) / `ggml-webgpu`
(browser) remain the runtime bridges for actual GPU acceleration.
