# KUHUL_RUNTIME.md — K'UHUL Phase Engine as a Versioned Runtime

> Status: architecture vision + current-state audit (2026-08-08)
> See also: SIDECARS.md (json_runtime sidecars), SCXQ2.md (instruction set), GPU.md (compute paths)

---

## 0. The goal: a stable, versioned runtime (like CPython, not a one-off script)

The **phase engine should become the stable, versioned runtime** — the way Python has
CPython and Node has its runtime. Everything else plugs into that runtime rather than
redefining execution every time:

```
             APPLICATIONS
                  │
                  ▼
        K'UHUL PHASE ENGINE       ← owns execution semantics (law)
       semantic execution law
                  │
                  ▼
          KHL DRIVER LAYER        ← capability + contract ABI (semantic device drivers)
                  │
                  ▼
            SIDECAR LAYER         ← native executors / services (isolation boundary)
                  │
                  ▼
       C++ / GPU / OS / MODEL     ← inevitable machinery underneath
```

**Canonical relationship:**

```
Phase Engine = law
KHL Driver   = semantic adapter
Sidecar      = implementation
C++          = inevitable machinery underneath
```

---

## 1. What exists today (audit 2026-08-08)

| Layer | Exists? | Where | Notes |
|-------|---------|-------|-------|
| **json_runtime phase/fold engine** | ✅ YES — built-in | `bin/json-runtime/src/xcfe.cpp` `native.PHASE` | `legal_phase_transition`, `phase_manifold_object`, `@fn: phase/transition/fold/manifold`. Fully detailed in SCXQ2.md. Pop→Wo→Yax→Sek→Ch'en→Xul, illegal jumps throw |
| **kuhul_engine phase runtime** | ✅ YES — own copy | `native/runtime/phase_runtime.h` | `Kuhul::Runtime::Phase` enum (Pop…Xul), `SemanticNode` (pressure, NodeKind, Residency), fold handlers. **Separate from the sidecar phase/fold — not shared** |
| **`.kuhul` source format** | ✅ YES | `native/runtime/*.kuhul`, `examples/kuhul/*.kuhul` | K'UHUL source programs (runtime.kuhul, Chen.kuhul, folds/*.kuhul) |
| **`.khl` kernel language** | ✅ YES (source only) | `.NNC-K/bin/micronaut/programs/*.khl`, hive CP-1 SDK | K'UHUL Language glyph programs (`glyph fold::collect_all → …`). Executed by a **Python executor** (`cp1_khl_executor.py`) |
| **Compiled KHL driver library** | ❌ NO | — | No `khlc.exe`, no compiled `.khl`, no KHL ABI, no driver registry. **This is the gap to build** |
| **Sidecar protocol** | 🟡 partial | json_runtime `SidecarLoader` + `SidecarStore` | Two kinds (xcfe_manifest in-process, external_exe stdin/stdout JSON). No versioned ABI contract yet |
| **OpenGL 4.3 compute** | ✅ YES (working) | `glsl_gpu` sidecar + `gl_infer_driver.dll` + `xcfe_gl_ops.dll` | Universal GPU path. OpenCL is present but **OpenGL is the target** — every GPU since 2012 |

### Key answers to the open questions

1. **Does json_runtime include its own phase/fold engine?** — **Yes, already.** `native.PHASE`
   in xcfe.cpp is a complete phase machine (legal transitions, fold execution, manifold query).
   It is detailed in SCXQ2.md.
2. **Does kuhul_engine use the sidecar phase/fold?** — **No.** kuhul_engine has its own
   `phase_runtime.h` with a *different* implementation (pressure-scored SemanticNodes).
   The two phase engines are not wired together — that is part of the unification work below.
3. **Is there a `.kuhul` version?** — **Yes** — `.kuhul` is the K'UHUL source program format
   (`runtime.kuhul`, `examples/kuhul/*.kuhul`).
4. **Does a compiled kernel/driver library exist?** — **No.** `.khl` exists as a *language*
   (glyph source + Python executor) but there is **no compiled kernel/driver library, no KHL
   ABI, and no driver registry**. You are right — that is the missing piece.

---

## 2. The K'UHUL Runtime — versioned independently

```text
K'UHUL Runtime 1.0
K'UHUL Runtime 1.1
KHL Driver ABI 1
SCXQ2 Contract 2
Sidecar Protocol 1
```

A module declares what it needs:

```json
{
  "requires": {
    "kuhul":   ">= 1.2",
    "khl_abi": 1,
    "scxq2":   ">= 2.0"
  },
  "capabilities": ["tensor.gemm", "tensor.map", "memory.resident"]
}
```

This is what Python modules (requires-python), Node native addons (N-API version), JVM
libraries, and OS drivers already do. K'UHUL gets the same discipline.

---

## 3. KHL driver layer — the semantic device driver ABI (KAST/KSON based)

**Decision (2026-08-08):** build the compiled-driver story on **KAST** and **KSON** —
which already exist in this stack — rather than inventing a new opaque binary format.

| Layer | Exists? | Job |
|-------|---------|-----|
| **KHL** | ✅ source language | human/source-level semantic driver language (`glyph ns::op(ARGS) → …`) |
| **KAST** | ✅ grammar defined | parsed executable structure / semantic syntax tree (`protocol: kast/1`, nodes with `fold/lane/glyph/opcode/symbol/type/operands`, edges) |
| **KSON** | 🟡 new (JSON) | serialized runtime form — a KAST document as JSON (`.kson`) |
| **K'UHUL phase engine** | ✅ | executes the lifecycle law |
| **sidecar/provider** | ✅ | performs the native work |

**The pipeline:**

```text
.khl source
   ↓  khlc (tools/khlc.py — IMPLEMENTED 2026-08-08)
KAST (protocol kast/1: KastDocument / nodes / edges)
   ↓
KSON (JSON serialization, e.g. opengl.kson)
   ↓
K'UHUL phase engine
   ↓
provider / sidecar
```

`khlc` does **not** mean "compile to machine code" — it means **compile KHL into
canonical KAST/KSON**. The `.kson` carries the canonical driver contract:

```json
"@driver": {
  "@abi": 1,
  "@requires": {"kuhul": ">= 1.0", "khl_abi": 1, "scxq2": ">= 2.0"},
  "@capabilities": ["tensor.map"],
  "@phase_hooks": {"Sek": "dispatch", "Ch'en": "collect_status", "Xul": "commit_tensor_state"},
  "@provider": "opengl",
  "@resources": [],
  "@hash": "<semantic_hash>"
}
```

**Runtime startup becomes:**

```text
load KSON
→ validate schema (kast-grammar.json)
→ reconstruct/verify KAST
→ check KHL ABI (@driver.@abi)
→ resolve provider (@driver.@provider)
→ mount capabilities (@driver.@capabilities)
→ enter Pop
```

**KAST is the semantic structure, not just config data.** Each node carries `fold`
(Pop/Wo/Yax/Sek/Ch'en/Xul), `glyph`, `opcode`, `symbol`, `operands` — exactly what the
phase engine consumes. The driver's glyphs map to phases (`opengl::probe`→Pop,
`opengl::dispatch`→Sek, `opengl::collect_status`→Ch'en, `opengl::commit`→Xul).

**Compiled drivers on disk (2026-08-08, `drivers/khl/`):**

| Source | KSON | Nodes/Edges | Glyphs |
|--------|------|-------------|--------|
| `opengl.khl` | `opengl.kson` | 19/13 | 4 (probe/dispatch/collect_status/commit) |
| `fold.khl` | `fold.kson` | 107/97 | 5 |
| `attention.fold.khl` | `attention.fold.kson` | 31/28 | 2 |
| `gpt2.runtime.khl` | `gpt2.runtime.kson` | 5/4 | 1 |
| `inference.khl` | `inference.kson` | 3/2 | 1 |
| `sw.khl` | `sw.kson` | 4/3 | 1 |

Later, if startup speed matters, add a packed form **without changing the semantic model**:

```text
KSON
 ↓
packed KSON / SCXQ2 / binary cache
```

---

## 4. Sidecar layer — the isolation boundary

The **sidecar system is the isolation boundary** for things that are too large, unstable,
privileged, or independently versioned to live inside the runtime:

```text
K'UHUL
   ↓
gemm.khl
   ↓
sidecar RPC/IPC
   ↓
asx_gemm.exe
   ↓
D3D11/OpenGL/CPU
```

K'UHUL never needs to know the implementation details of `asx_gemm.exe` — only its contract:

```json
{
  "provider": "asx_gemm",
  "version": "2.1",
  "input":  ["tensor<A>", "tensor<B>"],
  "output": ["tensor<C>"],
  "operation": "gemm",
  "determinism": "strict"
}
```

**Sidecars and KHL drivers are not competing ideas — they solve different problems:**

| | KHL driver | Sidecar |
|---|-----------|---------|
| Layer | Close-to-runtime | Isolation boundary |
| Lives in | Runtime process (compiled) | Separate process / external exe |
| For | Fast, trusted, well-defined ops | Large, unstable, privileged, independently-versioned services |
| Contract | KHL ABI (in-process) | Sidecar Protocol (RPC/IPC, JSON) |
| Example | `d3d11.compute`, `filesystem` | `asx_gemm.exe`, `optical_processor.exe`, model servers |

**Sidecar Protocol v1** (what json_runtime already implements, needs versioning):
- `external_exe` — stdin JSON → stdout JSON (`invoke: "stdin_json"`), resolved + spawned by `sw.cpp`
- `xcfe_manifest` — in-process XCFE program, loaded by `SidecarLoader`, ops composed from primitives
- Route: `POST /api/sidecars/<name>/call/<op>`
- Authority: `candidate_only` — compute-only, never mutates the registry

---

## 5. OpenGL (not OpenCL) is the universal GPU target

OpenCL is present (`IntelOpenCL64.dll`, GPU-only, no CPU runtime) but the target is
**OpenGL 4.3 compute** — `GL_ARB_compute_shader` + SSBO runs on **every GPU since 2012**
(Intel, AMD, NVIDIA, mobile) via the installed ICD:

| Vendor | ICD | Status |
|--------|-----|--------|
| Intel (HD 4600 + Haswell) | `ig75icd64.dll` | ✅ present, OpenGL 4.3 |
| Intel (Arc/newer) | `igvk64.dll` | probed by gl_infer_driver |
| AMD | `atio6axx.dll` | probed |
| NVIDIA | `nvoglv64.dll` | probed |

The GLSL path is live: `glsl_gpu` sidecar on json_runtime (verified `compiled:true, icd:ig75icd64.dll`),
`gl_infer_driver.dll` (8 shaders), `xcfe_gl_ops.dll` (17 kernels on the wgpu_native GL backend).

**The KHL driver for the GPU layer is `opengl.khl`** — `Sek -> dispatch`, `Ch'en -> collect status`,
`Xul -> commit tensor state`, with the HLSL→GLSL mapping (`std430`, `barrier()`, `gl_GlobalInvocationID`)
as the contract body.

---

## 6. The unification plan

| # | Work item | Status |
|---|-----------|--------|
| 1 | json_runtime phase engine (`native.PHASE`) — **already built** | ✅ done |
| 2 | `glsl_gpu` sidecar + GLSL dispatch — **already built & live** | ✅ done |
| 3 | **`khlc` compiler** — `tools/khlc.py`: KHL → KAST (`protocol: kast/1`) → KSON with `@driver` contract (abi/requires/capabilities/phase_hooks/provider/resources/hash) — **built, 6 drivers compiled** | ✅ done |
| 4 | **Driver set in repo** — `drivers/khl/*.khl` + `*.kson` (opengl, fold, attention.fold, gpt2.runtime, inference, sw) | ✅ done |
| 5 | **Unify the two phase engines** (json_runtime `native.PHASE` + kuhul_engine `phase_runtime.h`) — one canonical `Phase` law, the other delegates | 🟡 next |
| 6 | **KSON runtime loader** — json_runtime loads `.kson`, validates against kast-grammar, reconstructs KAST, checks `@driver.@abi`, mounts capabilities, enters Pop | 🟡 next |
| 7 | **Sidecar Protocol v1 versioning** — negotiate version in the sidecar contract | 🟡 version |
| 8 | **K'UHUL Runtime version metadata** — `runtime.manifest.json` with `kuhul`, `khl_abi`, `scxq2`, `sidecar` versions | 🟡 build |
| 9 | **`opengl.khl` driver → GLSL sidecar wiring** — the compiled driver contract resolves the `glsl_gpu` sidecar as its provider | 🟡 wire |
| 10 | **Packed KSON / SCXQ2 binary cache** (startup-speed optimization, semantic model unchanged) | ❌ later |

The piece to focus on next is **the KSON runtime loader + sidecar/driver ABI** — the phase
engine already gives the execution skeleton, and `khlc` now produces the canonical
KAST/KSON driver objects. Once json_runtime can load a `.kson`, validate it against
kast-grammar, and resolve its provider to a sidecar, the K'UHUL ecosystem has its
foundation.
