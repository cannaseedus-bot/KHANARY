# khanary-server — full llama-server with the KHANARY ggml-xcfe backend (frozen)

**Part 2 complete.** The full `llama-server` was built from source with the KHANARY-native
`ggml-xcfe` backend compiled in, branded to **`khanary-server.exe`**, and it **registers XCFE as a
runtime device**. Built on this rig (MSVC 19.44, VS2022) by `tools/build_khanary_server.ps1`.

## Build

Copied the full `llama.cpp` source to a workspace (never editing the read-only vendored tree),
overlaid `native/ggml-xcfe/`, wired `ggml_add_backend(XCFE)` + the static register, and built the
`llama-server` target with `-DGGML_XCFE=ON -DGGML_BACKEND_DL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF`.

```
-- Including XCFE backend
build exit=0
[ok] khanary-server bundled at dist/khanary-server
```

## Runtime — XCFE is a live device in the server

```
$ khanary-server.exe --list-devices
Available devices:
  XCFE: KHANARY XCFE (K'UHUL glyph backend, ...)  (0 MiB, 0 MiB free)
```

The KHANARY backend is not a side experiment — it is **inside the shipped server binary** and the
ggml backend registry enumerates it at startup. (The device description string was updated post-build
to "MUL_MAT via DirectML, CPU fallback"; it embeds on the next rebuild.)

## Bundle (`dist/khanary-server/`)

```
khanary-server.exe          the branded server
ggml-xcfe.dll               the KHANARY backend
ggml.dll ggml-base.dll ggml-cpu.dll   ggml runtime
llama.dll llama-common.dll llama-server-impl.dll mtmd.dll   llama runtime
dml_gemm.dll DirectML.dll   the DirectML GEMM (GPU path for XCFE's MUL_MAT)
khanary.svg                 brand asset
```

## Honest scope

- **IS:** a full, runnable `llama-server` binary with the KHANARY `ggml-xcfe` backend compiled in and
  registering at runtime; bundled with the DirectML GEMM. XCFE claims + computes MUL_MAT (GPU via
  DirectML, CPU fallback — `proof/ggml_xcfe_v1`).
- **IS NOT (yet):** the embedded **Web UI** — this binary was built `-DLLAMA_BUILD_UI=OFF` (the
  SvelteKit `vite build` needs a larger Node heap; that + KHANARY branding is Part 3, which reseals
  the binary with the branded UI). So the server runs the API + XCFE; the browser UI comes with the
  reseal.

## Reproduce
```
powershell -File tools/build_khanary_server.ps1
dist/khanary-server/khanary-server.exe --list-devices   # -> lists XCFE
```
