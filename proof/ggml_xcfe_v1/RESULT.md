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

## Honest scope (what this is / isn't)

- **IS:** `ggml-xcfe` is a genuine backend target (reg → device → backend vtables, buffers delegate
  to CPU), it **compiles** and **registers** in the ggml registry. The branded fork's `ggml` builds
  with it wired in.
- **IS NOT:** KHANARY compute does not yet run through ggml — `supports_op` returns **false**, so the
  scheduler routes every op to CPU (`graph_compute` is a no-op success). The K'UHUL glyph-lowering
  compute (`MUL_MAT` → KHANARY glyph kernels) is the next milestone.
- **IS NOT:** a full `khanary-server` build — this builds the `ggml` library + a probe, not
  `llama-server` (that larger full-llama build + branding is a later step).

## Reproduce
```
powershell -File tools/build_khanary_llama.ps1
# -> "-- Including XCFE backend", ggml-xcfe.dll built, probe prints "XCFE registered: YES"
```
Source: `native/ggml-xcfe/{ggml-xcfe.h,ggml-xcfe.cpp,CMakeLists.txt,xcfe_probe.c}` (snapshot here).
