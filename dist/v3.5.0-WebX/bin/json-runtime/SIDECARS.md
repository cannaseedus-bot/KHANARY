# SIDECARS.md — json_runtime Sidecar System & the Object Server

> Location: `bin/json-runtime/` (json_runtime.exe, port 8787)
> See also: `gpu.manifest.json` (`@glsl`), `sidecars.manifest.json`, `sco/sidecars/glsl.json`, `micronaut.manifest.json`, `folds.manifest.json`
> Rule: **JSON declares; XCFE/KUHUL executes.**

---

## 0. What json_runtime IS — a semantic graphic processor + REST API sandbox

`json_runtime.exe` is not just a file server. It is a **semantic graphic processor**
exposed as a **REST API sandbox** — and it is the **primary backend of the Hive**
(`E:\models\.hive`, the local AI runtime environment and swarm root).

- **Semantic graphic processor** — it ingests, stores, and executes *semantic graphs*:
  SCXQ2 IR graphs (`tensors`/`nodes`/`edges`/`regions`/`schedule`), K'UHUL ASTs
  (`kuhul.ast.v3.schema.json`), fold graphs (`folds.manifest.json`), and XJSON tensors.
  The "graphics" are the graph structures — nodes, edges, folds, lanes — not pixels.
  Per the Hive's SH-Wave-Lattice principle: **not rendering. Computing.**
- **REST API sandbox** — every capability is exposed as a sandboxed HTTP route
  (`/api/*`). Programs run inside the sandbox with authority checks
  (`RuntimeAuthority`), phase gating, and admission rules. Nothing executes outside
  XCFE/KUHUL admission. The Hive's routing contract names this exactly:
  `data_execution: control flow with data execution`, `protocol: REST API with JSON
  payloads`, `sandbox: object notation runtime`, `xml_database: extended IDB database server`.
- **Phases and folds routed** — the K'UHUL phase machine
  (`Pop→Wo→Yax→Sek→Ch'en→Xul`) gates what operations are legal at any moment;
  folds route compute across the phase manifold. `native.PHASE` enforces legal
  transitions; `folds.manifest.json` defines the fold algebra.
- **HIVE of micronauts** — the micronaut registry (`micronauts/`, `E:\models\.hive`)
  is the worker pool. Micronauts are declared as sidecars
  (`micronaut.manifest.json`), forged by the factory, packed to SCXQ2/BSON, and
  dispatched per phase/lane. The Object Server is the hive's backbone.

### The Hive is Home (hive-centric architecture)

From `E:\models\.hive\HIVE_CENTRIC_ARCHITECTURE.md`: **all skills, agents, and models
live in the hive. No external folder scanning.** json_runtime is the hive's native
C++ REST API — per `HIVE_KUHUL_INTEGRATION_COMPLETE.md`, all micronauts (CP-1, SW-1,
and the fleet) route through it as the **primary backend**, replacing hardcoded
LM Studio dependencies:

```
User Request
    ↓
Hive.kuhul Routes to Appropriate Micronaut
    ↓
    ├─→ Planning Task → PM-1 → json_runtime (REST API)
    ├─→ Brain Routing → BR-1 → json_runtime
    ├─→ K'UHUL Exec → KX-1 → Local K'UHUL
    ├─→ Swarm Dispatch → SW-1 → json_runtime (dispatcher + registry)
    └─→ …90+ micronaut agents, all coordinated through the Object Server
```

### XCFE runtime law (from `docs/specs/xcfe-runtime-spec-v1.md`)

> **JSON is the program. XCFE is the execution law. The EXE is a minimal dispatcher.**

```
XCFE = executor of definitions
Definitions = JSON
json.exe = thin interpreter, nothing more

1. Load JSON / SCXQ2
2. Validate XCFE structure (@op present)
3. Dispatch @op → resolve definition
4. Route to capability (CPU / GPU / IO)
5. Emit result
```

Everything else lives in JSON: `@ops` (instruction set), `@state` (memory/registers),
`@control` (execution flow), `@runtime` (capability declarations), `@buffers` (data).

### Shader Expert System (GPU side of the semantic processor)

From `E:\models\.hive\SHADER_EXPERT_SYSTEM.md`: shader types are **tensor experts**
with phase arrays, geodesic weights, parallel lanes, and tensor code. The router
computes a phase from the shader signature, finds the closest expert by π-geodesic
distance, and dispatches top-1 (extensible to top-K). The GLSL sidecar
(`sco/sidecars/glsl.json`) is the json_runtime admission point for this MoE shader
routing — GLSL compute is the universal expert lane.

```
REST API sandbox (/api/*)
      │
      ▼
json_runtime.exe  ── semantic graphic processor ──
      │  ├── SCXQ2 IR graphs (tensors/nodes/edges/regions)
      │  ├── K'UHUL AST (kuhul.ast.v3.schema.json)
      │  ├── Fold graphs (folds.manifest.json: Pop→Wo→Yax→Sek→Ch'en→Xul)
      │  ├── XJSON tensors (shape/dtype/data/backend)
      │  └── XCFE programs (@op node graphs)
      │
      ├── Phase machine (native.PHASE — legal transitions only)
      ├── Fold routing (fold algebra, gravity wells, lanes)
      ├── Sidecar store (glsl_gpu, math_ext, brain, …)
      ├── Shader Expert System (MoE shader routing → GLSL sidecar)
      └── Micronaut hive (factory → validate → SCXQ2 pack → BSON bind)
```

---

## 1. What a sidecar is

A **sidecar** is a JSON manifest that declares a capability bundle for `json_runtime.exe`.
It is the plugin/extension mechanism of the Object Server. A sidecar JSON is a
*bootstrap executable contract* for apps, games, programs, websites, schemas, and tools —
it says *what exists*, *what it can do*, and *how to reach it*, all in one declarative file.

```
sidecar JSON (declares)
      │
      ▼
SidecarLoader (loads, sandbox-checks the path)
      │
      ▼
XCFE / KUHUL (admits by capability + phase, then executes)
```

Two kinds of sidecar:

| Kind | What it is | How it runs |
|------|-----------|-------------|
| `xcfe_manifest` | An in-process XCFE program (JSON `@op` nodes) | Loaded by `SidecarLoader`, executed by the XCFE executor inside json_runtime — no subprocess |
| `external_exe` | A compiled binary | Spawned per call by `sw.cpp` — JSON request on stdin → JSON reply on stdout (`invoke: "stdin_json"`) |

Both speak JSON. Neither requires JavaScript.

### Lifecycle

1. **Declare** — a `*.manifest.json` (or `sco/sidecars/*.json`) declares `@sidecar` (id, capabilities, version) + `@ops` (composed operations) + optional `@routes`/`@ports`/`@schema`.
2. **Load** — `SidecarLoader::load(path, name)` sandbox-checks the path (rejects `/` and `..`), parses the JSON, registers the capability block.
3. **Admit** — `RuntimeAuthority` checks the capability against the phase contract (`Pop→Wo→Yax→Sek→Ch'en→Xul`) and the authority policy (`candidate_only` = compute-only, never mutates the registry).
4. **Dispatch** — `POST /api/sidecars/<name>/call/<op>` → `SidecarStore::call()` → in-process XCFE execution (xcfe_manifest) or subprocess (external_exe).

### What a sidecar can do

- **Declare ops** — `@ops` maps op names to composed XCFE programs (built from primitives: `ADD`, `MUL`, `SQRT`, `REDUCE`, `LOG`, `READ`, `WRITE`, `EVAL`, `CALL`, `PHASE`, …). Example: `math_ext.json` defines `HYPOTENUSE` and `MEAN` as composed ops.
- **Declare capabilities** — `@sidecar.capabilities` lists the capability tags (`MATH`, `EXTENDED`, `GLSL`, `GPU`, `OPENGL`, `COMPUTE`, …). The authority layer admits by capability.
- **Declare routes/ports/schemas** — `@routes`, `@ports`, `@schema` let a sidecar expose HTTP routes, reserve ports, and validate payloads.
- **Wrap external binaries** — `external_exe` sidecars wrap compiled workers (e.g. `quantum_hybrid.exe`) with a stdin/stdout JSON contract.
- **Run at init** — `@init` runs a boot sequence (`LOG` lines, setup ops) when the sidecar loads.

---

## 2. The GLSL sidecar (`sco/sidecars/glsl.json`)

`glsl_gpu` is the **GPU sidecar** that admits OpenGL 4.3 compute into json_runtime.
It is registered in `sidecars.manifest.json` and documented in `gpu.manifest.json` under `@glsl`.

### What it can do

| Op | What it does |
|----|--------------|
| `glsl_probe` | Probes the OpenGL 4.3 provider: ICD (`ig75icd64.dll` Intel / `igvk64.dll` Arc / `atio6axx.dll` AMD / `nvoglv64.dll` NVIDIA), `GL_ARB_compute_shader` + SSBO availability, max work-group invocations (1024) |
| `glsl_info` | Returns the backend contract: `gl43_compute`, the HLSL→GLSL mapping table, and the dispatch paths |
| `glsl_compile` | Compile-only validation of GLSL source via `@fn:dispatch @profile:glsl` (checks `#version`, balanced braces, `layout(local_size_...)`); device compile+dispatch routes to the GLSL backends |
| `glsl_dispatch` | Routes a compute op to the GLSL backends: `gl_infer_driver.dll` (8 shaders), `xcfe_gl_ops.dll` (17 kernels on the wgpu_native GL backend), or `GLSL_Server.exe` (HTTP, port 9060) |

### Dispatch paths (the actual GLSL compute engines)

1. **`gl_infer_driver.dll`** (`drivers/`) — OpenGL 4.3 compute inference. 8 GLSL shaders: `embed`, `layernorm`, `matmul`, `attention`, `gelu`, `add_bias`, `residual`, `lm_head`. HLSL→GLSL mapping: `StructuredBuffer<T> : register(tN)` → `layout(std430, binding=N) buffer`, `[numthreads(X,Y,Z)]` → `layout(local_size_x=X,...) in;`, `SV_DispatchThreadID` → `gl_GlobalInvocationID`, `GroupMemoryBarrierWithGroupSync()` → `barrier()`.
2. **`xcfe_gl_ops.dll`** (`bridges/ggml-xcfe/`) — GGML tensor bridge on the wgpu_native GL backend. 17 WGSL kernels: `mul_mat`, `get_rows`, `norm`, `rms_norm`, `gelu`, `gelu_quick`, `silu`, `relu`, `tanh`, `sigmoid`, `add`, `sub`, `mul`, `soft_max`, `rope`, `concat`, `cpy`. Seam contract v2: `xcfe_gl_run(op, inputs, n_inputs, out, ne_out, n_dims, params, n_params)`. MUL_MAT verified (max err 4.768e-07).
3. **`GLSL_Server.exe`** (`.Powernaut-v1.0.0/dist/`) — HTTP server wrapping `kuhul/glsl_server.py` (port 9060). Works from source; the PyInstaller package has a missing-`http`-module issue.

### Why GLSL is the universal GPU path

OpenGL 4.3 compute shaders run on **every GPU since 2012** — Intel iGPU, AMD, NVIDIA, mobile — with no hardware purchase. CUDA requires NVIDIA; DirectML is Windows-only; OpenCL fragmented. GLSL is the one path that covers the whole fleet, including the Intel HD 4600 (Haswell GT2) via `ig75icd64.dll` (OpenGL 4.3, `GL_ARB_compute_shader`, SSBO, 1024 max work-group invocations, up to the 1792 MB VRAM ceiling).

---

## 3. The Object Server — what json_runtime actually is

`json_runtime.exe` is an **ASX JSON/XCFE object server**. It stores, serves, and executes
JSON objects — not just files. The objects it manages include:

| Object type | Format | Example |
|-------------|--------|---------|
| Programs | XJSON `@op` programs | `{ "@op": "native.EVAL", "@fn": "add", "@args": ["$x", "$y"], "@out": "result" }` |
| Tensors | XJSON tensor | `{ "@type": "xjson/tensor", "shape": [4,8], "dtype": "f32", "data": [...] }` |
| Manifests | `*.manifest.json` | `server.manifest.json`, `gpu.manifest.json`, `sidecars.manifest.json` |
| Sidecars | `sco/sidecars/*.json` | `math_ext.json`, `brain.json`, `glsl.json` |
| SCXQ2 IR | JSON graph IR | `{ "tensors": [...], "nodes": [...], "schedule": {...} }` |
| Registry entries | Named objects | `tensor_register` / `tensor_get` / `tensor_list` |

The Object Server is the **backbone and configuration layer** of the whole stack:

- **Every service is declared as a manifest** — `server.manifest.json` (WebX root, port 7430), `rpc.manifest.json`, `mcp.manifest.json`, `gpu.manifest.json`, `agents.manifest.json`, `folds.manifest.json`, `database.manifest.json`, `projects.manifest.json`, `actions.manifest.json`, `experts.manifest.json`, `evolution.manifest.json`, `programs.manifest.json`, `scxq2.manifest.json`, `sco.manifest.json`, `game.manifest.json`, `website.manifest.json`, `xjson.manifest.json`, `xcfe.manifest.json`, `grammar.manifest.json`, `ir.manifest.json`, and ~60 more (see `RuntimeAuthority::sidecar_examples()`).
- **Every program is a JSON document** — XCFE executes `@op` node graphs directly. No compilation step required at runtime (SCXQ2 binary is an optional transport/cache format).
- **Every tensor in flight is a JSON object** — XJSON tensors carry `shape`, `dtype`, `data`, `backend`. The `"backend"` field records whether compute went to DirectML, GLSL, or CPU fallback.
- **Every model identity is a JSON document** — `atomic.manifest.json` per model (chat template, persona, sampling, tool registry, provider endpoint).

### Object Server ↔ database relationship

The Object Server is a **document store with an execution engine**:

- **Storage** — named objects persist across calls via the registry (`tensor_register`/`tensor_get`/`tensor_list`), the SCO store (SHA-256 content-addressed cache), and the sidecar store. The `database.manifest.json` and `projects.manifest.json` contracts declare the schema layer.
- **Query** — `GET /api/sidecars`, `GET /api/sidecars/<name>`, `POST /api/sidecars/<name>/call/<op>`, `/api/file-manager/*`, `/api/sco/<alias>`, `/api/discovery`.
- **Execution** — unlike a passive database, the Object Server *runs* the objects it stores: XCFE executes programs, tensor_runtime computes matmul/softmax, gpu_dispatch compiles kernels, sidecars dispatch ops.
- **Persistence** — SCXcache (a persistent DAG of execution state) is written at the Xul phase and resumes from the last committed graph state. The `@runstate`/`@boot_sequence` sidecar keys declare boot-time state.

So: **a database stores rows; the Object Server stores executable JSON documents and runs them.** It is the configuration backbone of a JSON operating system — the manifests *are* the OS configuration, the sidecars *are* the device drivers, the XCFE programs *are* the applications, and the XJSON tensors *are* the data.

### The micronaut hive (worker pool)

The hive (`E:\models\.hive`) is the swarm root: a self-contained toolchain hosting the
micronaut swarm, inference servers, model weights, 3D pipelines, and shell workspaces.
json_runtime is its **primary backend** — every micronaut routes through the native C++
REST API (`HIVE_KUHUL_INTEGRATION_COMPLETE.md`).

| Micronaut | Role | Routes through |
|-----------|------|----------------|
| PM-1 | Planning Model | json_runtime REST API |
| BR-1 | Brain Routing | json_runtime |
| VM-1 | Visual Gen | Local SVG Gen |
| OV-1 | Clustering | Local Python |
| KX-1 | K'UHUL Exec | Local K'UHUL |
| FG-1 | Frame Routing | Local Routing |
| DX-1 | DX routing | json_runtime |
| CP-1 | Copilot (SDK: expert loader, GPU orchestrator, KHL executor, brain graph) | json_runtime fallback |
| SW-1 | **Swarm dispatcher** — launcher + registry + features | json_runtime |

Micronaut lifecycle (from `micronaut.manifest.json`):

```
factory create → micronaut.json → validate (Yax) → SCXQ2 pack (Xul) → BSON bind (Xul)
semantic_key = glyph + lane + binary_runtime_array
```

Factory output is a **candidate** until validation, SCXQ2 packing, BSON checksum, and
Crown policy admission complete. Cross-micronaut communication uses HTTP, not direct
cross-stack paths — the Object Server is the hub.

---

## 4. JSON or AST — everything converges on one notation

Every layer of the stack speaks JSON or an AST that serializes to JSON:

```
Manifests  →  JSON objects (@sidecar, @gpu, @ops, @routes, @schema)
Programs   →  XJSON @op node graphs (JSON)
Tensors    →  XJSON tensors (JSON)
SCXQ2 IR   →  JSON graph IR (tensors/nodes/edges/regions/schedule)
K'UHUL AST →  JSON Schema (kuhul.ast.v3.schema.json) + EBNF
KXML       →  JSON tool registry (kuhul.tools.jsonl) + node ops (kxml_nodes.json)
Atomic DOM →  JSON manifests (atomic.manifest.json)
Micronauts →  JSON sampling profiles (micronauts/*.json)
```

The AST (K'UHUL-3D, SCXQ2 IR, XCFE programs) is the *structure*; JSON is the *encoding*.
`SCXQ2 => XCFE/XJSON <= SCX` — the binary SCXQ2 format compiles from and decompiles back
to JSON losslessly (`--roundtrip` is the correctness gate). The AST preserves the prompt/context;
K'UHUL traverses it; XCFE routes legal graph moves; opcodes perform work; compute nodes lower
to CPU, llama.cpp, WebGPU/WGSL, D3D11/HLSL, or OpenGL/GLSL.

---

## 5. JSON without JavaScript

**JSON is a data format, not a JavaScript feature.** The entire stack executes JSON with
**zero JavaScript in the runtime path**:

- `json_runtime.exe` is **C++** (MSVC). It parses JSON with `nlohmann/json` and executes
  XCFE programs with its own recursive-descent evaluator (`xcfe.cpp`, ~1237 lines) — no JS engine.
- `compile_gpu_kernel()` calls `D3DCompiler_47.dll` (C API) for HLSL and the GLSL path for
  OpenGL — no JS.
- `tensor_runtime()` calls `dml_gemm.dll` (DirectML, C API) — no JS.
- The GLSL sidecar dispatches to `gl_infer_driver.dll` / `xcfe_gl_ops.dll` / `GLSL_Server.exe`
  (Python) — no JS.
- The XCFE expression evaluator implements infix math, logic, bitwise, flow, and tensor ops
  natively in C++ — `$varname` scope resolution, `@op` dispatch, `@fn` primitives — all C++.

JavaScript (Node.js) appears only in the *orchestration* layer (`kuhul-server.cjs`), which
talks to json_runtime over HTTP. The Object Server itself — the backbone — is pure C++.

> **The point:** JSON Object Notation is a language-independent contract. Any language
> (C++, Python, PowerShell, Node, Rust, …) can declare, store, and execute the same JSON
> documents. The Object Server makes JSON the *operating system* — not a JavaScript library.

---

## 6. Quick reference

| Route | Purpose |
|-------|---------|
| `GET /api/sidecars` | List declared + loaded sidecars, store, contract, phase contract |
| `GET /api/sidecars/<name>` | Describe one sidecar (declared, loaded, kind, ops) |
| `POST /api/sidecars/<name>/call/<op>` | Dispatch an op on a sidecar |
| `GET /api/sco/<alias>` | SCO content-addressed lookup |
| `GET /api/discovery` | Discovery surface |
| `POST /api/run` | Run a JSON program |

| File | Role |
|------|------|
| `sidecars.manifest.json` | Sidecar registry (external_exe + xcfe_manifest entries) |
| `sco/sidecars/*.json` | Individual sidecar manifests (math_ext, brain, glsl, …) |
| `gpu.manifest.json` | GPU provider contracts (`@d3d11_1`, `@webgpu`, `@opencl`, `@glsl`) |
| `src/sidecar.cpp` | SidecarLoader — load, sandbox-check, capability admit |
| `src/sw.cpp` | SidecarStore — external_exe resolution + dispatch |
| `src/xcfe.cpp` | XCFE executor — runs `@op` programs |
| `src/gpu_dispatch.cpp` | `compile_gpu_kernel()` — HLSL (cs_5_0) + GLSL (gl43) compile paths |
| `src/tensor_runtime.cpp` | Tensor ops — matmul (DirectML), softmax, relu, registry |

**Launch:** `json_runtime.exe --manifest <bundle>.manifest.json --sidecar <sidecar>.manifest.json`
