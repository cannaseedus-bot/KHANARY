# GPU.md — Khanary Compute Stack: Single Source of Truth

> Generated from: `kuhul_engine.exe --providers`, source audit of `json-runtime/src/`,
> `src/xvm/cpu-cluster.js`, `native/gpu_trainer/xvm_core.h`, `native/shaders/`, `scratch/dml/`.

---

## Provider inventory (`kuhul_engine.exe --providers`)

| Provider | DLL | Status |
|----------|-----|--------|
| `d3dcompiler` | D3DCompiler_47.dll | **available** |
| `shader_cache` | D3DSCache.dll | **available** |
| `directml` | DirectML.dll | **available** |
| `xcfe_directml` | ggml-xcfe.dll | **available** |
| `directml_debug` | DirectML.Debug.dll | **available** |
| `opencl` | IntelOpenCL64.dll | **available** (OpenCL 1.2 — HD 4600 is Haswell, not 2.0) |
| `intel_opencl_icd` | Intel_OpenCL_ICD64.dll | **available** |
| `clang` | common_clang64.dll | **available** (Intel OpenCL JIT) |
| `d3d12` | D3D12.dll | **available** |
| `d3d11` | d3d11.dll | **available** |
| `intel_d3d12_user` | igd12umd64.dll | **available** (HD 4600 UMD) |
| `intel_graphics_jit` | igc64.dll | **available** (GPU ISA JIT) |
| `d3d10warp` | d3d10warp.dll | **available** (WARP software fallback) |
| `opengl` | ig75icd64.dll | **available** — Intel OpenGL ICD (11 MB, System32) |
| `opengl_wrapper` | opengl32.dll | **available** — Windows OpenGL wrapper (1 MB, System32) |
| `opencl_cpu_*` | (7 DLLs) | **not found** — CPU OpenCL runtime absent |

**GPU ceiling**: Intel HD 4600 (Haswell GT2), 1792 MB VRAM, D3D12 feature level 11_0, cs_5_0.

---

## OpenGL provider — Intel HD 4600

**ICD**: `C:\Windows\System32\ig75icd64.dll` (11,473 KB)
**32-bit ICD**: `C:\Windows\SysWOW64\ig75icd32.dll` (8,530 KB)
**Driver**: 20.19.15.4835 (Intel, 2017-10-16) — last driver for HD 4600 on Windows 10
**OpenGLInstalled**: 1 (registry confirmed)
**OpenGL version**: Intel HD 4600 (Haswell GT2) supports **OpenGL 4.0** minimum; this driver revision supports **OpenGL 4.3** (compute shaders available via `GL_ARB_compute_shader`)

Intel HD 4600 OpenGL 4.3 capabilities relevant to inference:
- `GL_ARB_compute_shader` — GLSL compute shaders (`layout(local_size_x=N)`)
- `GL_ARB_shader_storage_buffer_object` (SSBO) — GPU buffer read/write from compute shaders
- `GL_ARB_program_interface_query` — shader introspection
- `GL_ARB_texture_buffer_object` — texture buffers as 1D arrays (weight storage)
- Max compute work group invocations: 1024 per dispatch
- Max SSBO size: determined by available VRAM (up to 1792 MB ceiling)

**Current project use**: OpenGL is already present in the native runtime via `opengl_frame_adapter.cpp` and `atomic.opengl.frame.manifest.json` / `atomic.opengl.asset.manifest.json` — currently wired for rendering/UI frames, not yet as a compute path for inference.

---

## GLSL — the universal backend (planned)

OpenGL 4.3 compute shaders (GLSL) are the single GPU path that covers every device without a vendor purchase:

```
Intel iGPU (2012+)    →  ig75icd64.dll   →  OpenGL 4.3  →  GLSL compute
AMD iGPU / APU        →  amdvlk / mesa   →  OpenGL 4.3  →  GLSL compute
NVIDIA (any)          →  nvoglv64.dll     →  OpenGL 4.3  →  GLSL compute
AMD discrete          →  atioglxx.dll     →  OpenGL 4.3  →  GLSL compute
Any laptop, any phone →  driver           →  OpenGL ES 3.1+  →  GLSL compute
```

CUDA requires buying specific hardware. GLSL runs on everything made since 2012 — including this HD 4600 via `ig75icd64.dll` already present in System32.

### KLSL → GLSL: addon emit target, not a rewrite

KLSL already has two emit paths from the same IR: HLSL (`khlnary_dx11.py`) and WGSL (`emit_wgsl.py`). GLSL is a third emit target. The shader logic is identical — only the binding syntax changes:

| HLSL (D3D11) | GLSL (OpenGL 4.3) |
|---|---|
| `StructuredBuffer<float> A : register(t0)` | `layout(std430,binding=0) readonly buffer A { float a[]; };` |
| `RWStructuredBuffer<float> C : register(u0)` | `layout(std430,binding=0) buffer C { float c[]; };` |
| `cbuffer Params : register(b0) { uint M; }` | `layout(std140,binding=0) uniform Params { uint M; };` |
| `[numthreads(16,16,1)]` | `layout(local_size_x=16, local_size_y=16, local_size_z=1) in;` |
| `SV_DispatchThreadID` | `gl_GlobalInvocationID` |
| `SV_GroupThreadID` | `gl_LocalInvocationID` |
| `SV_GroupID` | `gl_WorkGroupID` |
| `GroupMemoryBarrierWithGroupSync()` | `barrier()` |
| `groupshared float s[256]` | `shared float s[256]` |
| `void main(uint3 tid : SV_DispatchThreadID)` | `void main()` (reads gl_* builtins) |

All 7 shaders in `d3d11_infer.dll` translate directly. Add `#version 430` header, swap the table above, done.

### GLSL DLL: `gl_infer.dll`

Same C API as `d3d11_infer.dll` but using OpenGL:
- **Windows**: WGL headless context (`wglCreateContext` on a PIXELFORMATDESCRIPTOR with `PFD_SUPPORT_OPENGL`) → `ig75icd64.dll` takes it
- **Cross-platform**: EGL offscreen context (no window needed)
- Buffers: `glGenBuffers` / `glBindBufferBase(GL_SHADER_STORAGE_BUFFER, N, buf)`
- Dispatch: `glDispatchCompute(X, Y, Z)` + `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)`
- Shader compile: `glShaderSource` / `glCompileShader` / `glLinkProgram`

### Why GLSL matters more than DirectML or OpenCL

| Path | Requires | Works on |
|------|----------|----------|
| CUDA | NVIDIA GPU purchase | NVIDIA only |
| DirectML | Windows + D3D12 | Windows iGPU/discrete |
| OpenCL | OpenCL ICD installed | Most GPUs, platform-specific |
| **GLSL compute** | **OpenGL 4.3 driver** | **Every GPU since 2012** |

NVIDIA pushed CUDA because it sells hardware. DirectML is Microsoft-only. OpenCL fragmented. OpenGL 4.3 compute shaders have been universally available for 13 years and the ML ecosystem barely touched them — not because they can't do the math, but because the tooling ecosystem was captured early.

The d3d11_infer.dll already proved the full GPT-2 forward pass runs on this iGPU without any purchased GPU hardware. The GLSL version of those same 7 shaders would run on any machine on the planet.

### Planned additions
- [ ] `tools/emit_glsl.py` — KLSL GLSL emit target (adds to HLSL + WGSL)
- [ ] `scratch/gl/gl_infer_dll.cpp` — OpenGL 4.3 compute dispatch harness
- [ ] `scratch/gl/shaders/` — GLSL versions of the 7 inference shaders
- [ ] `gl_infer.dll` — deployed alongside `d3d11_infer.dll` in build output

---

## Architecture: three compute paths

```
                    ┌─────────────────────────────┐
                    │   SCXQ2 bytecode program     │
                    │   Mode bit per instruction:  │
                    │   CPU=0b00  GPU=0b01         │
                    └──────┬──────────┬────────────┘
                           │          │
              ─────────────┘          └──────────────
             CPU path                  GPU path
              │                         │
    ┌─────────▼──────────┐   ┌──────────▼───────────┐
    │  XVM 32-fiber      │   │  DirectML GEMM        │
    │  thread cluster    │   │  dml_gemm.dll         │
    │  xvm_run_cpu_      │   │  (KLSL forward pass)  │
    │  ticks_mt()        │   ├──────────────────────┤
    │  (hardware_        │   │  OpenCL GPU           │
    │  concurrency())    │   │  IntelOpenCL64.dll    │
    │                    │   │  + igc64.dll (JIT)    │
    │  Manifold opcodes: │   ├──────────────────────┤
    │  GEODESIC          │   │  D3D11 cs_5_0 shaders │
    │  ENTROPY_GRADIENT  │   │  xvm_compute.cso      │
    │  RIEMANN_CURVATURE │   │  xvm_fused_qkv_       │
    │  FOLD_ENTER/EXIT   │   │  attention.cso        │
    │  PRESSURE_PROPAGATE│   └──────────────────────┘
    └────────────────────┘
             │                         │
             └──────────┬──────────────┘
                        │
              ┌─────────▼──────────┐
              │  CONTROL.PARALLEL  │  (SCXQ2 fan-out)
              │  CONTROL.SYNC      │  (barrier join)
              │  MESH.MESH_EXEC    │  (cross-sidecar)
              └────────────────────┘
```

---

## CPU path — XVM 32-fiber thread cluster

**Source**: `src/xvm/cpu-cluster.js`, `native/gpu_trainer/xvm_core.h`

```
CPUCluster32 {
  fibers: XVMFiber[32 × clusterCount]
  shared: Uint32Array[1024]   ← shared memory across all fibers
  BARRIER opcode              ← all 32 fibers must reach same phase before release
}
```

Each fiber carries: `pc`, `sp`, `phase ∈ [0,6)`, `flags`, `r0-r3`, `entropy` (u8), `pressure` (u8).

**C++ multithreaded entry point** (`xvm_core.h`):
```cpp
void xvm_run_cpu_ticks_mt(XVMState& vm, uint64_t ticks,
    uint32_t threadCount = std::thread::hardware_concurrency());
```

**Manifold opcodes** (CPU gradient geometry):

| Opcode | Hex | Behaviour |
|--------|-----|-----------|
| `GEODESIC` | 0x40 | dst ← scaled distance to target fiber in (phase, entropy, pressure) space |
| `ENTROPY_GRADIENT` | 0x41 | r0 ← entropy diff vs ring neighbor; bump local entropy +1 |
| `RIEMANN_CURVATURE` | 0x42 | dst ← entropy XOR pressure (curvature estimate) |
| `FOLD_ENTER` | 0x43 | Advance phase if entropy ≥ threshold; reset entropy |
| `FOLD_EXIT` | 0x44 | Step phase back (floor 0) |
| `PHASE_TRANSITION` | 0x46 | Validated ±1 phase jump; halts fiber on illegal jump |
| `PRESSURE_PROPAGATE` | 0x45 | Push pressure delta to ring neighbors; self loses 2× |

**Training optimum** (validated):
```
batch_size: 1000, epochs: 4   ← preferred
batch_size:  500, epochs: 8   ← alternative
```
Rationale: phase-based barrier sync + manifold ops produce better gradient signal at small batch; high epoch count compensates for CPU throughput ceiling.

---

## GPU path — Native D3D11 cs_5_0 (most native path)

**DLL**: `scratch/dml/d3d11_infer.dll` → deployed to `build/bin/Release/`
**Source**: `scratch/dml/d3d11_infer_dll.cpp`
**Driver route**: `D3D11CreateDevice` → `igd10iumd64.dll` (15.7 MB) → `igdumdim64.dll` (38 MB) → `igc64.dll` → GEN7.5 EU ISA

This is the **most native GPU path on this machine** — the same route WoW (D3D11 mode) and other DX11 games take. No abstraction layer above D3D11. Shaders compiled at init time via D3DCompile → D3DCompiler_47.dll → HLSL cs_5_0 bytecode.

**C API** (C-ABI, LoadLibraryA compatible):
```c
bool d3d11_infer_init()                               // create device, compile 7 shaders
void d3d11_infer_shutdown()
int  d3d11_alloc(uint32_t n_floats)                  // allocate GPU StructuredBuffer, returns id
void d3d11_upload(int id, const void* data, uint32_t n)  // CPU→GPU (float or int32)
void d3d11_download(int id, float* out, uint32_t n)  // GPU→CPU
void d3d11_free(int id)
void d3d11_gemm(int A, int B, int C, uint32_t M, uint32_t N, uint32_t K)  // C=A*B
void d3d11_embed(int tokens, int wte, int wpe, int out, uint32_t S, uint32_t E)
void d3d11_layernorm(int x, int gamma, int beta, int y, uint32_t S, uint32_t E)
void d3d11_attention(int qkv, int out, uint32_t S, uint32_t E, uint32_t H)
void d3d11_gelu(int x, int y, uint32_t numel)
void d3d11_add_bias(int y, int b, uint32_t rows, uint32_t N)
void d3d11_add(int a, int b, uint32_t numel)          // residual add: b += a
```

**Shaders** (compiled from embedded HLSL source, cs_5_0):
| Shader | Dispatch | Role |
|--------|----------|------|
| `k_matmul` | `(N/16, M/16, 1)` `[numthreads(16,16,1)]` | 16×16 tiled GEMM with groupshared memory |
| `k_embed` | `(S, 1, 1)` `[numthreads(256,1,1)]` | Token + position embedding lookup |
| `k_layernorm` | `(S, 1, 1)` `[numthreads(256,1,1)]` | LayerNorm (mean+var reduction, gamma+beta) |
| `k_attention` | `(H, 1, 1)` `[numthreads(128,1,1)]` | Causal multi-head attention with softmax |
| `k_gelu` | `(numel/256, 1, 1)` `[numthreads(256,1,1)]` | GELU tanh approximation |
| `k_add_bias` | `(rows*N/256, 1, 1)` `[numthreads(256,1,1)]` | Broadcast bias add |
| `k_add` | `(numel/256, 1, 1)` `[numthreads(256,1,1)]` | Residual add (b += a) |

Covers the full GPT-2 forward pass: embed → (layernorm → qkv_proj → attention → proj → add) × L → layernorm → lm_head.

---

## GPU path — DirectML / KLSL forward pass

**Runtime chain**:
```
llama-server.exe  (or json_runtime.exe)
  └── ggml-xcfe.dll               ← LoadLibraryA("dml_gemm.dll") on first MUL_MAT
        └── dml_gemm.dll          ← KLSL forward pass DLL
              ├── DirectML.dll    ← Microsoft DirectML operator runtime
              └── igd12umd64.dll  ← Intel HD 4600 D3D12 user-mode driver
```

**Source**: `scratch/dml/dml_gemm_dll.cpp`
**Exports**: `dml_gemm_bt_f32(A, B, C, M, N, K)` — C[M,N] = A[M,K] @ B^T (B row-major [N,K])
**Features**: amortised D3D12+DML device, per-shape resource cache, GPU-resident weight store, one GPU sync per call.
**Fallback**: silently falls back to CPU if DLL unavailable. Log: `[ggml-xcfe] MUL_MAT path: DirectML (GPU)` or `CPU baseline`.
**Override**: `KHANARY_DML_GEMM` env var changes DLL path.

**DLL locations**:

| Consumer | DLL path |
|----------|----------|
| `llama-server.exe` | `build/bin/Release/dml_gemm.dll` (deployed by `llama-build.bat` step 5) |
| `json_runtime.exe` | `bin/ggml/dml_gemm.dll` (relative: `..\\ggml\\dml_gemm.dll` from `bin/json-runtime/`) |

### scratch/dml — DirectML development workspace

`C:\Users\canna\_khanary_inspect\scratch\dml\` is the DML build and test workspace. Everything needed to build and validate `dml_gemm.dll` lives here.

**Headers** (`scratch/dml/include/`):
- `DirectML.h` — main DirectML API
- `DirectMLX.h` — helper wrappers (graph builder, tensor desc helpers)
- `DirectMLConfig.h` — version config

**Libraries / DLLs**:
- `scratch/dml/lib/` — import libs (DirectML.lib)
- `scratch/dml/DirectML.dll` — the runtime DLL (authoritative copy; deployed to `build/bin/Release/` and `bin/ggml/` by llama-build.bat)

**Source / build**:
- `dml_gemm_dll.cpp` — source for `dml_gemm.dll`
- `build.bat` / `build_mem.bat` — MSVC build scripts (vcvars + cl.exe)

**Test executables** (all built from `.cpp` sources in same dir):
- `dml_cap_probe.exe` — capability probe (feature level, operator support)
- `dml_ops_test.exe` — operator correctness tests
- `dml_gemm_bench.exe` — GEMM throughput benchmark
- `dml_attn_run.exe` — attention layer runner
- `dml_layer_run.exe` — single transformer layer runner
- `dml_mlp_run.exe` — MLP layer runner
- `dml_model_run.exe` — full model forward pass runner
- `dml_mha_test.exe` / `dml_mha_cap_test.exe` / `dml_mha_kv_test.exe` — MHA tests
- `mem_ceiling_probe.exe` — 1792 MB VRAM ceiling measurement
- `q8_hotswap_probe.exe` — Q8 weight hot-swap validation
- `resident_generate.exe` — GPU-resident generation test
- `gen_bench.exe` / `gen_004a.exe` / `kuhul_matmul_tick.exe` — generation benchmarks

**Weight binary files** (`gen_l*.bin`, `mdl_l*.bin`, `ly_*.bin`): per-layer weight tensors exported from `from_zero` model for use by `dml_layer_run.exe` and `dml_model_run.exe`. One set per layer (0–11), covering: wq/wk/wv/wap (attention), wfc/wmp (MLP), bq/bk/bv/bap/bfc/bmp (biases), ln1g/ln1b/ln2g/ln2b (layernorm). Also `gen_wte.bin`, `gen_wpe.bin`, `gen_lmhead.bin`, `gen_lnfg.bin`/`gen_lnfb.bin`.

**Python helpers**: `attn_prep.py`, `layer_prep.py`, `model_prep.py`, `gen_prep.py` — export weights from SafeTensors → flat binary for the test executables. `compare_driver_dml.py`, `time_driver_dml.py`, `test_dml_gemm.py` — correctness and timing analysis.

---

## GPU path — XVM compute shaders (D3D11 cs_5_0)

**Source**: `native/shaders/xvm_compute.hlsl` → `native/bin/cso/xvm_compute.cso`
**Source**: `native/shaders/xvm_fused_qkv_attention.hlsl` → `native/bin/cso/xvm_attention_kv_int4.cso`

`xvm_compute.hlsl` — GPU-side XVM fiber execution: `[numthreads(64,1,1)]`, one GPU thread per fiber. Runs the same opcode table as the CPU cluster (LOAD_CONST, MOV, ADD, BARRIER, manifold ops) using StructuredBuffers for Code/Fibers/Shared/Stack/Trace.

`xvm_fused_qkv_attention.hlsl` — cs_5_0 fused QKV + attention, designed for shared-memory fallback on legacy iGPU:
- Inputs: `X [seq_len × model_dim]`, `Wqkv [model_dim × 3×head_dim]`
- groupshared `Ks[TILE][HEAD]`, `Vs[TILE][HEAD]` (TILE=64, HEAD=64)
- `[numthreads(64,1,1)]` — one group per query token

---

## GPU path — OpenCL (second independent GPU path)

Available via `IntelOpenCL64.dll` + `igc64.dll` (Intel Graphics Compiler, JIT to HD 4600 GPU ISA).
`common_clang64.dll` provides the OpenCL C front-end.

**CPU OpenCL**: NOT available — all `opencl_cpu_*` DLLs are absent. OpenCL = GPU-only on this machine.

Use cases: ops where DirectML's operator abstraction is too coarse, or where you need direct kernel control (layernorm, custom reductions). Write OpenCL C kernel → IGC compiles to GPU ISA at runtime.

---

## KLSL — WebGPU opcodes → HLSL transpiler

**llama.cpp runs its GPU backend on WebGPU opcodes (WGSL compute shaders).** KLSL converts those opcodes to HLSL for DirectX/DirectML dispatch on Windows.

```
KLSL source (.klsl)
  ├── klslc.exe  →  HLSL  →  D3D11/DirectML bytecode
  │    trainer/shaders/*.hlsl = compiled output
  │    (attn QK dot, softmax, bone argsort, fold route matmul, etc.)
  └── emit_wgsl.py  (SCXQ2 IR JSON → WGSL)  →  WebGPU dispatch table
```

**Inference path**: `dml_gemm.dll` uses DirectML high-level operator API directly — no KLSL compilation at inference time. The KLSL→HLSL work is done ahead of time for trainer shaders.

**Training path**: `trainer/shaders/*.hlsl` are KLSL-compiled HLSL compute shaders. `klslc.exe` at `C:\Users\canna\.ASX.cpp\klsl\bin\klslc.exe`.

---

## Native DX shader cache — D3DSCache.dll + SCO

`D3DSCache.dll` is Microsoft's native Direct3D shader bytecode cache — persists compiled HLSL bytecode to disk keyed by source hash.

SCO (Symbolic Cache Object, `sco.hpp`) provides the SHA-256 registry for addressing compiled GPU programs by alias. The intended wiring (not yet connected):

```
HLSL source string
  → sha256_str(source)          ← SCO hash_json()
  → SCO registry lookup
      hit  → load cached bytecode blob (D3DSCache.dll)
      miss → D3DCompile() → store in SCO + D3DSCache
```

Currently: `compile_gpu_kernel()` in `gpu_dispatch.cpp` calls `D3DCompile` on every call and discards the bytecode after counting bytes. Wiring SCO + D3DSCache eliminates recompile overhead on every `@fn:"dispatch"` call.

---

## json_runtime.exe — XCFE GPU verbs (port 8787)

| XCFE verb / `@fn` | Handler | Backend |
|---|---|---|
| `@fn:"dispatch"` | `compile_gpu_kernel()` | D3DCompiler_47.dll → cs_5_0 bytecode. Compile-only; device dispatch not yet wired. |
| `@fn:"matmul"` / `tensor.gemm` | `tensor_runtime()` | `dml_gemm.dll` (DirectML GEMM) → CPU fallback. Returns XJSON tensor + `"backend"` field. |
| `@fn:"softmax"` / `@fn:"relu"` | `tensor_runtime()` | CPU (XJSON tensor ops) |
| `@fn:"alloc"` | `alloc_tensor()` | Allocate zero-filled XJSON tensor |
| `tensor_register` / `tensor_get` / `tensor_list` | `registry_operation()` | In-process named tensor store |

XCFE stdlib `gpu` capability declares: `@gpu.dispatch`, `@gpu.buffer.write`, `@gpu.buffer.read`.
`@gpu.buffer.write` / `@gpu.buffer.read` — in manifest, not yet in C++.

### GLSL GPU sidecar (2026-08-08)

**OpenGL (not OpenCL) is the universal GPU target** — `GL_ARB_compute_shader` + SSBO runs
on every GPU since 2012 via the installed ICD. OpenCL is present but secondary.

json_runtime now admits OpenGL 4.3 compute through the **`glsl_gpu` sidecar**
(`sco/sidecars/glsl.json`, registered in `sidecars.manifest.json` + the main
manifest's `@sidecars`). This is the admission point for the Hive's Shader Expert
System (MoE shader routing: phase from shader signature, closest expert by
π-geodesic distance, top-1 dispatch).

See `docs/KUHUL_RUNTIME.md` for the phase-engine-as-versioned-runtime architecture:
K'UHUL phase engine (law) → KHL driver layer (semantic ABI) → sidecar layer
(implementation) → C++/GPU machinery. The GPU driver contract is `opengl.khl`
(`Sek -> dispatch`, `Ch'en -> collect status`, `Xul -> commit tensor state`).

| Op | What it does |
|----|--------------|
| `glsl_probe` | Probe the OpenGL 4.3 provider (ICD, GL_ARB_compute_shader + SSBO, 1024 max work-group invocations) |
| `glsl_info` | Backend contract: `gl43_compute`, HLSL→GLSL mapping, dispatch paths |
| `glsl_compile` | Compile-only GLSL validation via `@fn:dispatch @profile:glsl` |
| `glsl_dispatch` | Route compute to `gl_infer_driver.dll` (8 shaders) / `xcfe_gl_ops.dll` (17 kernels) / `GLSL_Server` (port 9060) |

`@fn:dispatch` now supports `@profile:glsl` (or `430`/`gl43`): validates `#version`,
balanced braces, and `layout(local_size_...)`, and probes the GL ICD in System32
(`ig75icd64.dll` / `igvk64.dll` / `atio6axx.dll` / `nvoglv64.dll`). Verified live:
`compiled: true, icd: ig75icd64.dll`. Full concept doc: `bin/json-runtime/SIDECARS.md`
(semantic graphic processor + REST API sandbox, phase/fold routing, micronaut hive,
JSON-without-JS).

---

## Tensor layers — storage vs runtime vs GPU compute

There are four distinct tensor layers in this stack. They do not share a format. Each has a specific job.

```
STORAGE (disk)           RUNTIME (in-flight)       GPU COMPUTE (on-device)
──────────────           ───────────────────       ───────────────────────
SafeTensors (.safetensors) → XJSON tensor       → DirectML upload heap
GGUF (.gguf)               → XJSON tensor       → dml_gemm_bt_f32 buffers
SCXQDDS (.scxqdds)         → INT8 weight buffer → kuhul_fold_compute.hlsl SRVs
```

### Layer 1 — SafeTensors (HuggingFace format, on disk)

All model weights in this project are stored as HuggingFace-compatible `.safetensors` files:

| File | Contents |
|------|----------|
| `models/from_zero/from_zero_v0.6_merged.safetensors` | Student model (GPT-2 small, 6L/6H/768E) |
| `models/from_zero/from_zero_v0.6_lora.safetensors` | LoRA adapter output from distillation |
| `models/from_zero/from_zero_v0.*.safetensors` | Phase checkpoints |

`repair_safetensors.py` (`tools/`) fixes empty-shape tensors: reads `shape_map` from a valid reference checkpoint, rebuilds target files. Uses `safetensors.torch` + PyTorch. Supported dtypes: F32, F16, BF16, I32, I64, U8.

### Layer 2 — LoRA (PyTorch, pure — no PEFT, no HuggingFace transformers)

`oss_distillation.py` (`tools/`) trains LoRA adapters via response distillation:

```
Teacher:  kuhul_engine HTTP API → port 17480 (GPT-OSS)
Student:  from_zero_v0.6 (loaded from SafeTensors, run in PyTorch)

LoRA:  W_delta = B @ A
  A: [r, in_dim]   init N(0, 0.02/r)
  B: [out_dim, r]  init zeros
  W_eff = W_frozen + B @ A   (alpha=r → unit scaling)
  Only A, B in optimizer; W frozen

Targets: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj (per layer)
Output:  from_zero_v0.6_lora.safetensors
Fallback: if engine unreachable → self-distillation (shape validation only)
```

500 steps, lr=1e-4, rank=8 default. PyTorch exists **only** in the Python tooling. It never touches the C++ runtime.

### Layer 3 — XJSON tensor (own format, in-flight at json_runtime level)

SCXQ2 `TENSOR` opcodes (MATMUL, SOFTMAX, etc.) are **compute primitives**, not storage. At runtime they operate on XJSON tensors:

```json
{ "shape": [M, K], "data": [1.0, 2.0, ...], "dtype": "f32", "backend": "khanary-directml" }
```

This is the stack's own in-flight format. JSON floats. The `"backend"` field records whether execution went to DirectML or CPU fallback. The gap for large tensors: JSON float serialization overhead — gap #5 in the hybrid trainer list (replace with binary transfer).

Named tensors persist across calls via the `tensor_register` / `tensor_get` registry in `tensor_runtime.cpp`.

### Layer 4 — DirectML / D3D12 buffers (GPU-resident, on-device)

When `dml_gemm_bt_f32(A, B, C, M, N, K)` runs, data leaves XJSON format and goes into D3D12 upload heaps → GPU-resident default heaps. `dml_gemm.dll` maintains a per-shape resource cache (amortised device + descriptor heap). The XJSON tensor is the handoff envelope — the actual compute is in D3D12 memory, dispatched via DirectML operator graph.

`kuhul_fold_compute.hlsl` consumes its weights via `StructuredBuffer<int>` (INT8 packed) with per-row dequant scales — these are SCXQDDS shards loaded directly to SRV slots, never going through XJSON.

### Mapping summary

| What | Format | Who touches it |
|------|--------|----------------|
| Weight files on disk | SafeTensors (HuggingFace) | Python tools, repair, distillation |
| LoRA adapters | SafeTensors (HuggingFace) | `oss_distillation.py` → output |
| XCFE compute values | XJSON tensor (own) | json_runtime / SCXQ2 MATMUL op |
| GPU-side weight buffers | D3D12 upload+default heap | `dml_gemm_bt_f32`, DirectML |
| Quantized expert shards | SCXQDDS INT8+CRC (own) | `kuhul_fold_compute.hlsl` SRVs |
| LoRA merge at inference | SLERP merge tool (Python) | merges SafeTensors → new SafeTensors |

---

## SCXQ2 bytecode — mode-bit dispatch

Every SCXQ2 instruction carries a 2-bit `Mode` field:

```
Mode::CPU  = 0b00   → XVM fiber cluster (xvm_run_cpu_ticks_mt)
Mode::GPU  = 0b01   → DirectML / OpenCL / D3D11 cs_5_0
Mode::HASH = 0b10   → SCO SHA-256 path
Mode::META = 0b11   → compile-time metadata
```

**Tensor opcode group** (0x14–0x17):

| Subop | Name | Implemented in C++? |
|-------|------|---------------------|
| 0 | MATMUL | yes (`tensor_runtime.cpp`) |
| 1 | DOT | no |
| 2 | TRANSPOSE | no |
| 3 | SOFTMAX | yes |
| 4 | NORMALIZE | no |
| 5 | ATTENTION | no |
| 6 | FLASH_ATTN | no |
| 7 | KV_STORE | no |
| 8 | KV_LOAD | no |
| 9 | DISPATCH | yes (compile-only) |

**Control opcodes relevant to hybrid dispatch**:
- `PARALLEL` — fan out N ops concurrently (spec'd; sequential in current xcfe.cpp executor)
- `SYNC` — barrier join
- `MESH.MESH_EXEC` — cross-sidecar dispatch (spec'd; no handler yet)

---

## Hybrid trainer: CPU+GPU split (target architecture)

```
Forward pass (per layer):

  Layer 0–5 GEMM  ──────────► GPU: dml_gemm_bt_f32()
                                     Mode::GPU, SCXQ2 MATMUL
  Layer 0–5 Attn  ──────────► GPU: xvm_fused_qkv_attention.cso
                                     (fused QKV, groupshared KV)
  LayerNorm       ──────────► GPU: OpenCL kernel via igc64.dll
                                     (or CPU via XVM fiber)

  Gradient accum  ──────────► CPU: XVM 32-fiber cluster
                                     ENTROPY_GRADIENT + RIEMANN_CURVATURE
                                     PRESSURE_PROPAGATE to neighbors
                                     KuhulPhysics antigravity scale

  CONTROL.PARALLEL wraps GPU GEMM + CPU gradient accum simultaneously
  CONTROL.SYNC joins before weight update step
```

**Gap list before this works**:
1. SCXQ2 Mode bit not yet read by `tensor_runtime.cpp` — matmul tries DirectML always, then CPU
2. `TENSOR.ATTENTION` / `FLASH_ATTN` not in C++ — need `xvm_fused_qkv_attention.cso` dispatch path
3. `CONTROL.PARALLEL` runs sequentially in xcfe.cpp — needs thread pool
4. `@gpu.buffer.write/read` not implemented — every matmul does full upload+compute+readback
5. JSON float array transfer format — replace with binary for large tensors
6. SCO + D3DSCache not connected — shader recompile on every call

---

## Skeleton & Bone System — root of the fold pipeline

The skeleton is the single most important structure in the system. It decides where data lives, how kernels are dispatched, and sculpts the physics gravity wells that shape attention.

### What the skeleton produces

```
Skeleton (bone matrices in 3D space)
         │
         ▼  cs_vertex_skin.hlsl    [Dispatch(ceil(V/64), 1, 1)  numthreads(64,1,1)]
         │  4 bones × blend weights per token
         │  → skinned position (px,py,pz) + normal (nx,ny,nz) per token
         │  → float4x4 blended transform = weighted sum of 4 bone matrices
         │
         ▼  bone_ids[i*4+0]  ← primary bone = cluster assignment
         │
         ▼  cs_bone_argsort_.hlsl  [Dispatch(1,1,1)  numthreads(128,1,1)]
         │  counting sort by primary bone ID (groupshared, 4 phases)
         │
         ├── sorted_idx[S]        ← WHERE DATA LIVES: tokens in cluster-contiguous order
         ├── cluster_start[C]     ← start offset per cluster in sorted_idx
         └── cluster_count[C]     ← token count per cluster
```

`bone_ids` is a flat `int[S*4]` buffer — 4 joint indices per token. Primary bone = `bone_ids[i*4+0]`. The argsort uses `InterlockedAdd` into groupshared counters (no global atomics), prefix-sums into `gs_start[]`, then scatters `sorted_idx` in a single pass. Constraints: S ≤ 128, N_CLUSTERS ≤ 64.

### Where data lives

`sorted_idx` re-orders tokens so cluster 0 tokens are contiguous in memory, then cluster 1, etc. This is physical layout — every downstream kernel reads `sorted_idx[cluster_start[c] + local]` to get the original token index. The primary bone assignment is the sole decider of physical placement.

```
Before: [tok4(bone1), tok0(bone0), tok2(bone1), tok1(bone0), tok3(bone2)]
After:  [tok0, tok1, tok2, tok4, tok3]
         ──bone0──  ──bone1──  bone2─
cluster_start: [0, 2, 4]
cluster_count: [2, 2, 1]
```

### How it's called

Every fold kernel receives `cluster_start[c]` and `cluster_count[c]` as its dispatch parameters. The kernel never sees the full sequence — only `[cluster_start[c] .. cluster_start[c] + cluster_count[c])`. Routing is entirely mechanical:

```
bone ID → cluster index → range in sorted_idx → kernel slice
```

In `cs_fold_route_matmul.hlsl`:
```hlsl
token_i = sorted_idx[cluster_start + t_local];
dst[token_i * n_rows + row] = dot(src[token_i], weight[row]);
```

Disjoint cluster ranges → no write conflicts, no atomics needed anywhere downstream.

### Gravity wells (physics engine)

The 4-bone blend weights don't just place tokens — they create the attention potential field. In `cs_fold_kernel_compute_.hlsl`:

```hlsl
P_buf[head * S*S + query_i * S + key_j] += gravity_scale * 2.0f * overlap;
```

`overlap` measures shared bone influence between two tokens. Tokens with heavy shared-bone weight get a +2× gravity bias added to their pre-softmax attention logit. **The bone matrix IS the gravity well** — heavier bone overlap → stronger mutual attention pull → cluster cohesion in attention space, not just in memory.

The `gravity_scale` is the KuhulPhysics antigravity scale. Intra-cluster pairs receive double the global value. Causal masking still applies using original (pre-sort) sequence positions.

### LBS shader (`cs_vertex_skin.hlsl`)

```
Input:  positions[V × positionStride],  normals[V × normalStride]
        boneWts[V × 4],  boneJoints[V × 4],  skinMats[nBones × 4 rows]
Output: outVerts[V × 6] = [px,py,pz,nx,ny,nz]

blendMatrix(i) = Σ fetchMatrix(boneJoints[i*4+k]) * boneWts[i*4+k]  for k in 0..3
sp  = mul(float4(p, 1.0), blendMatrix)       // position in bone space
sn  = mul(normal, upper-left 3×3 of blend)  // gradient direction
```

The normal channel carries gradient orientation. Bones not only place data but orient the update direction for the physics propagation step.

### Fibonacci layer on top of bones

`fibonacci_fold.h` (`KXML::Compression`) applies Fibonacci-window sampling on top of bone cluster layout:

| Class | What it does |
|---|---|
| `FibonacciFold<T>` | Compress 1D/2D tensor via Fibonacci windows (output ≈ N/φ) |
| `FibonacciAttentionFold` | O(N log_φ N) attention: fold Q+K before dot-product |
| `FibonacciTensorCompression` | Zeckendorf non-consecutive encoding for SCXQ2 (~12.5% overhead, reversible) |
| `FibonacciGeodesicFold` | Sample geodesic path at Fibonacci-indexed positions |
| `FibonacciSIMDFold` | DirectXMath SIMD: 4×4 tile fold + golden-ratio scale + Zeckendorf encode |

The XVM `GEODESIC` opcode (0x40) is the CPU-side partner: it computes manifold distance from a fiber to a target in `(phase, entropy, pressure)` space — the same geodesic sampling principle applied to the XVM fiber topology. Bone cluster assignment sets the fiber's `phase`, so manifold distances are measured within the same bone's fiber group.

---

## Fold Tensor System

The fold system is the bridge between skeleton-based cluster assignment (CPU, XVM) and GPU attention bias + expert routing. Four layers, each consuming the skeleton's output buffers.

### Layer 1 — XVM phase opcodes (CPU)

`FOLD_ENTER` (0x43): advance fiber phase if `entropy ≥ threshold`, reset entropy. Fiber "folds" into next phase.
`FOLD_EXIT` (0x44): step phase back (floored at 0).
`PHASE_TRANSITION` (0x46): validated ±1 jump; halts fiber on illegal jump.

The 32-fiber cluster uses phase as the fold-region partitioning signal. Fibers in the same phase form one fold region, matching their bone cluster assignment.

### Layer 2 — Intra-cluster attention bias (GPU)

**`cs_fold_kernel_compute_.hlsl`** — `Dispatch(n_head, cluster_count, 1)` `[numthreads(128,1,1)]`

Reads `sorted_idx`, `cluster_start`, `cluster_count`. For every (query_i, key_j) pair within the same cluster that passes causal masking (using original, pre-sort positions):

```hlsl
P_buf[head * S*S + query_i * S + key_j] += gravity_scale * 2.0f * overlap;
```

Intra-cluster pairs get 2× gravity. Cross-cluster pairs: unmodified. This amplifies attention cohesion within each bone group before softmax.

### Layer 3 — Expert-routed matmul (GPU)

**`cs_fold_route_matmul.hlsl`** — `Dispatch(ceil(n_rows/128), cluster_count, 1)` `[numthreads(128,1,1)]`

Derived from `ggml-webgpu/wgsl-shaders/mul_mat_id.wgsl`. One expert weight matrix per cluster:

```hlsl
token_i = sorted_idx[cluster_start + t_local];
dst[token_i * n_rows + row] = dot(src[token_i], weight[row]);
```

Scatters output back to original token positions. Disjoint clusters → no write conflicts.

### Layer 4 — MM-1 full fold (GPU)

**`kuhul_fold_compute.hlsl`** — 300-node 10×10×10 grid, 19,200 threads total. CM-1 gate (`ControlFlags[0] == 0x0002`) must be set before dispatch.

| Node range | Role |
|---|---|
| 0–99 | Trunk nodes — INT8 12-layer GEMM, distributed across rows |
| 100–199 | Expert nodes — 9 experts, INT8 × SwiGLU activation |
| 200–299 | Router nodes — 1024→9 logits, top-1 argmax at node 200 |

Pipeline per token: trunk matmul → router logits → top-1 argmax → selected expert matmul → token emission.

Arc outputs:
- `RouterLogitsI8` → **META_FOLD** via `arc_CF_MF` (`kuhul_fold_meta.cso`)
- `TrunkOutput` → **STORAGE_FOLD** via `arc_CF_SF` (`kuhul_fold_storage.cso`)

V6 verifier: same weights + same input → same output bytes (replay determinism).

### Layer 5 — FoldRegistry (C++)

**`native/runtime/fold_registry.h`**:
```cpp
struct FoldDescriptor { string name, id; Phase phase; string micronaut; path contract, artifact; bool observer; };
FoldRegistry::admitted(Phase, vector<string> names) → active folds for this phase
```

Loads `folds.manifest.json`. At runtime, `admitted(Phase::Pop, ...)` selects which fold kernels are dispatched. Each `FoldDescriptor.micronaut` names the MM-1 model instance it runs.

### Layer 6 — SCXQ2 IR

`RegionKind::FOLD` tags bytecode regions. `Mode::GPU=0b01` dispatches fold kernel. `Mode::CPU=0b00` runs `FOLD_ENTER`/`FOLD_EXIT` in XVM fibers. The three compiled fold shaders: `kuhul_fold_compute.cso`, `kuhul_fold_meta.cso`, `kuhul_fold_storage.cso`.

### Full data flow

```
sorted_idx + cluster_start + cluster_count   ← skeleton output (cs_bone_argsort_)
       │
       ▼
cs_fold_kernel_compute_.hlsl   ← attention bias ×2 for intra-cluster pairs
       │
       ▼
cs_fold_route_matmul.hlsl      ← scatter matmul by cluster (MoE top-1)
       │
       ▼
kuhul_fold_compute.hlsl        ← INT8 trunk → router → expert → token logits
       │
       ├── RouterLogitsI8  → META_FOLD    (kuhul_fold_meta.cso)
       └── TrunkOutput     → STORAGE_FOLD (kuhul_fold_storage.cso)
```

---

## Build artifacts and DLL deployment

### `llama-build.bat` — three build modes

Usage:
```
llama-build           — incremental (fast): UI rebuild + dml_gemm.dll rebuild + cmake --build llama-server
llama-build full      — full GPU reconfigure via build_gpu.ps1 + rebuild
llama-build clean     — wipe CMakeCache + full rebuild (same as full but clears configure cache first)
```

All modes run steps 1–2b before branching:

**Step 1 — Stale UI purge**
Deletes `tools/ui/dist/`, `build/tools/ui/.ui-stamp`, `build/tools/ui/ui.cpp`, `build/tools/ui/ui.h`.
Required because cmake's UI embed checks for `dist/index.html` and `.ui-stamp` before running npm — any leftover artifact from a prior build prevents a fresh UI bake.

**Step 2 — npm run build**
Runs SvelteKit build in `khanary-llama-build/llama.cpp/tools/ui/`. Output: `tools/ui/dist/index.html`. cmake priority-1 bakes this directly into `llama-server.exe`.

**Step 2b — Rebuild `dml_gemm.dll` (KLSL DirectML forward pass)**
```
vcvars64.bat (VS 2022 BuildTools, fallback: Community)
cl /std:c++17 /O2 /I include /LD dml_gemm_dll.cpp /link /LIBPATH:lib DirectML.lib /OUT:dml_gemm.dll
```
Built in `scratch/dml/`. If vcvars64 is missing, warns and uses existing `dml_gemm.dll` if present.

**Step 3a — Fast rebuild (default)**
```
cmake --build khanary-llama-build/llama.cpp/build --config Release --target llama-server -j 4
```

**Step 3b — Full rebuild (`llama-build full`)**
Calls `khanary-llama-build/build_gpu.ps1`. That script:
1. Fetches Khronos OpenCL headers (`CL/cl.h` etc.) from GitHub into `khanary-llama-build/opencl-headers/`
2. Generates `OpenCL.lib` import stub from `C:\Windows\System32\OpenCL.dll` via dumpbin + lib.exe
3. cmake reconfigure: `-DGGML_OPENCL=ON -DGGML_OPENCL_TARGET_VERSION=200 -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF -DOpenCL_INCLUDE_DIR=... -DOpenCL_LIBRARY=...`
4. `cmake --build . --config Release --target llama-cli llama-server -j 4`

**Step 4 — Deploy GPU runtime DLLs**
Copies from `scratch/dml/` to TWO locations:
| Destination | Consumer |
|------------|---------|
| `khanary-llama-build/llama.cpp/build/bin/Release/` | `llama-server.exe` (via ggml-xcfe.dll → LoadLibraryA) |
| `khanary-llama-build/ggml/build/bin/Release/` | `json_runtime.exe` (relative path `..\\ggml\\dml_gemm.dll`) |

DLLs copied: `dml_gemm.dll`, `DirectML.dll`. Missing either → silent CPU fallback (no GPU matmul).

### ggml subproject output (`khanary-llama-build/ggml/build/bin/Release/`)

Built by cmake as dependencies of llama-server:
- `ggml-xcfe.dll` — ggml XCFE backend (calls LoadLibraryA("dml_gemm.dll"))
- `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll` — ggml compute graph layers

### XVM compiled shaders (`native/bin/cso/`)

- `xvm_compute.cso` — GPU fiber execution (`xvm_compute.hlsl`, cs_5_0, numthreads(64,1,1))
- `xvm_attention_kv_int4.cso` — fused QKV attention (`xvm_fused_qkv_attention.hlsl`, INT4 KV path, groupshared 64×64 tile)
