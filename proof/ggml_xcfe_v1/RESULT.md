# ggml-xcfe de-orphaned — registers + builds (frozen)

**Part 2, milestone 1.** The KHANARY-native `ggml-xcfe` backend is no longer an orphan byte-copy of
`ggml-webgpu` — it is a **real, distinct, registering ggml backend that compiles** into the branded
fork's `ggml`, verified on this rig (MSVC 19.44, VS2022, Windows SDK 10.0.26100).

## What was built

`tools/build_khanary_llama.ps1` copied the vendored ggml source into a workspace (never editing the
read-only tree), overlaid `native/ggml-xcfe/` (replacing the webgpu copy), wired
`ggml_add_backend(XCFE)` + the static include/register in `ggml-backend-reg.cpp`, and built
`ggml` + a registry probe with `-DGGML_XCFE=ON -DGGML_BACKEND_DL=OFF`.

## Result

```
-- Including CPU backend
-- Including XCFE backend            <- CMake wired the backend
-- Configuring done / Generating done
   ggml-xcfe.cpp  ->  ggml-xcfe.lib / ggml-xcfe.dll     <- compiles clean
   xcfe_probe.exe                                        <- links against ggml
build exit=0

=== probe (walks the ggml backend registry) ===
backend[0] = XCFE
backend[1] = CPU
XCFE registered: YES
exit=0
```

The probe (`xcfe_probe.c`) enumerates `ggml_backend_reg_count()` / `_get()` / `_name()` and finds an
entry named exactly **`XCFE`** — proof the backend registers via the static path.

## Milestone 2 — XCFE claims + computes MUL_MAT (CPU baseline)

`supports_op` now claims plain 2D F32 contiguous `GGML_OP_MUL_MAT` (+ metadata view ops), and
`graph_compute` computes it with a reference GEMM matching ggml's `mul_mat` semantics
(`dst[n,m] = Σ_k a[k,n]·b[k,m]`). `xcfe_matmul_test.c` runs the **same** `ggml_mul_mat` graph on ggml's
CPU backend (ground truth) and on XCFE, and compares:

```
MUL_MAT [K=4,N=3,M=2]
  C[0]  cpu=-0.037972  xcfe=-0.037972
  ...   (all 6 elements match)
max abs err = 3.725e-09
XCFE computes MUL_MAT, matches ggml CPU: YES   exit=0
```

So XCFE is now a backend the ggml scheduler can **route MUL_MAT to** and it **computes it correctly**.
The CPU GEMM here is the **fallback baseline**; the KHANARY glyph / DirectML GEMM
(`proof/kuhul_matmul_tick_v1`) swaps into this same `graph_compute` for the GPU path — same claimed op,
two implementations.

## Honest scope (what this is / isn't)

- **IS:** `ggml-xcfe` compiles, **registers**, **claims** `MUL_MAT`, and **computes** it (matches
  ggml's CPU MUL_MAT to 3.7e-09). The branded fork's `ggml` builds with it wired in.
- **IS NOT (yet):** the *GPU* compute — `graph_compute`'s MUL_MAT is a CPU reference GEMM; the KHANARY
  glyph/DirectML kernel replaces it next. The test drives XCFE's `graph_compute` **directly**
  (`ggml_backend_graph_compute`); scheduler-level routing via `supports_op` is the standard mechanism
  (implemented) but not separately exercised here.
- **IS NOT:** a full `khanary-server` build — this builds the `ggml` library + probe + test, not
  `llama-server` (the larger full-llama build + branding is a later step).

## Reproduce
```
powershell -File tools/build_khanary_llama.ps1
# -> "-- Including XCFE backend", ggml-xcfe.dll built, probe prints "XCFE registered: YES"
```
Source: `native/ggml-xcfe/{ggml-xcfe.h,ggml-xcfe.cpp,CMakeLists.txt,xcfe_probe.c}` (snapshot here).
