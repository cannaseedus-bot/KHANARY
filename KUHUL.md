# KUHUL.md — K'UHUL Semantic Runtime & Language

> Master document for the K'UHUL execution substrate: language, compiler, IR,
> standard library, phase engine, sidecars.
> Detailed architecture: [`docs/KUHUL_RUNTIME.md`](docs/KUHUL_RUNTIME.md)

---

## What K'UHUL is

**K'UHUL is a semantic execution substrate.** JSON declares; XCFE/KUHUL executes.
It is not a JavaScript feature and not a GPU vendor SDK — it is a language-neutral
contract for *what a program wants* (semantic modules, `.kuhul`) and *how that
capability attaches to a provider* (drivers, `.khl`), executed through a canonical
phase engine (`Pop → Wo → Yax → Sek → Ch'en → Xul`).

```
Phase Engine = law
KHL Driver   = semantic adapter
Sidecar      = implementation
C++          = inevitable machinery underneath
```

---

## The toolchain — one source-to-execution path

```text
.kuhul / .khl source
   ↓  khlc            (tools/khlc.py — structural + semantic compilation, static checks)
KAST                 (canonical semantic IR, protocol kast/1)
   ↓  JSON serialization
KSON                 (runtime-loadable driver/module object, .kson)
   ↓  admission      (tools/kson_validate.py — trust/admission gate)
canonical phase engine (Pop → Wo → Yax → Sek → Ch'en → Xul)
   ↓  provider resolution
sidecar / native impl (glsl_gpu, json_runtime native, kuhul_engine, …)
```

| Layer | Exists? | Job |
|-------|---------|-----|
| **KHL** (`.khl`/`.kuhul`) | ✅ | human/source-level semantic language |
| **KAST** (`protocol: kast/1`) | ✅ | parsed executable structure — nodes carry `fold/lane/glyph/opcode/symbol/type/operands`, edges carry control flow |
| **KSON** (`.kson`) | ✅ | JSON serialization of KAST + `@driver` contract |
| **phase engine** | ✅ | execution law — legal transitions only |
| **sidecar/provider** | ✅ | native capability implementation |

### The `@driver` contract (embedded in every KSON)

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

---

## Language — two source layers

| Layer | Extension | Says | Example |
|-------|-----------|------|---------|
| Semantic module | `.kuhul` | **what** capabilities the program wants | `stdlib/pi.kuhul`, `stdlib/gravity.kuhul` |
| Driver contract | `.khl` | **how** the capability attaches to a provider/sidecar | `drivers/khl/opengl.khl`, `drivers/khl/phase.khl` |

### Phase-block syntax (KLSL idiom)

```kuhul
include "constants.kuhul"

⟁ Pop ⟁
  bind π = 3.141592653589793
  probe geometry

⟁ Wo ⟁
  bind radius = 8
  bind area = π * radius * radius

⟁ Yax ⟁
  resolve provider = geometry.compute

⟁ Sek ⟁
  dispatch provider(area)

⟁ Ch'en ⟁
  collect_status result

⟁ Xul ⟁
  commit result
```

Each `⟁ Phase ⟁` block is a fold-annotated node cluster. Verbs map to opcodes:

| Verb | Opcode | Phase |
|------|--------|-------|
| `bind` | BIND | Pop / Wo |
| `probe` | PROBE | Pop |
| `resolve` | RESOLVE | Yax |
| `dispatch` | DISPATCH | Sek |
| `collect_status` | COLLECT | Ch'en |
| `commit` | COMMIT | Xul |
| `yield` | YIELD | — |
| `op::call(ARGS) → RES` | CALL | — |

### Glyph form (driver bodies)

```khl
glyph opengl::dispatch(TENSOR_IN) →
  gl::upload(TENSOR_IN)        → GPU_BUF
  gl::bind_shader("compute")   → PROG
  gl::dispatch(GPU_BUF, PROG)  → OUT_BUF
  yield OUT_BUF
```

`include "module.kuhul"` is supported (recursive, cycle-guarded, relative paths;
provenance recorded in the KSON `includes` field).

---

## KAST — the canonical IR

`protocol: kast/1`. Each node: `id, kind, fold, lane, glyph, opcode, symbol, type,
operands, attributes`. Each edge: `id, from, to, kind, label, ordinal`.

```json
{"id": "n6", "kind": "call", "fold": "Sek", "lane": "compute",
 "glyph": "dispatch", "opcode": "DISPATCH", "symbol": "provider(area)",
 "type": "operator_call", "operands": ["area"], "attributes": {"result": ""}}
```

The document carries `protocol`, `registry_hash`, `source_kind`, `source_id`,
`entry_node_id`, `nodes`, `edges`, `semantic_hash`, `@driver`, and `includes`.

---

## Standard Library (`stdlib/`)

Semantic modules — not function collections. One-directional dependency tree:

```text
core.kuhul (folds, nodes, lanes, glyphs, opcodes, contracts, providers, types)
   ↓
constants.kuhul (e, tau, phi, c, h, N_A)
   ↓
pi.kuhul (canonical π / reference conformance program)
   ↓
functions.kuhul (length, normalize, clamp, lerp, dot, cross, sin, cos, sqrt, abs, min, max)
   ↓
┌────────────┴────────────┐
geometry.kuhul          fibonacci.kuhul (φ, F(n), golden_ratio/spiral/rectangle, lattice)
   │                         │
   └────────────┬────────────┘
                ↓
           gravity.kuhul (G, mass, field, potential, acceleration)
                ↓
           glsl.kuhul / hlsl.kuhul (provider bindings)
                ↓
           opengl.khl → glsl_gpu sidecar
```

Also: `vectors`, `matrices`, `tensors`, `statistics`, `random`, `colors`, `time`,
`audio`, `image`. Node accumulation: core 12 → constants 20 → pi 28 → functions 47 →
geometry 61 / fibonacci 59 → gravity 74 → glsl 88.

**`pi.kuhul` is the reference conformance program** — compile, validate the KAST,
load the KSON, execute every legal fold transition, invoke a provider, verify the
committed result. Its KSON node trace is the canonical sequence:

```text
n1  fold=Pop    opcode=BIND     symbol=π = 3.141592653589793
n2  fold=Pop    opcode=PROBE    symbol=geometry
n3  fold=Wo     opcode=BIND     symbol=radius = 8
n4  fold=Wo     opcode=BIND     symbol=area = π * radius * radius
n5  fold=Yax    opcode=RESOLVE  symbol=provider = geometry.compute
n6  fold=Sek    opcode=DISPATCH symbol=provider(area)
n7  fold=Ch'en  opcode=COLLECT  symbol=result
n8  fold=Xul    opcode=COMMIT   symbol=result
```

---

## Phase engine — one authority

The canonical phase law is a driver: `drivers/khl/phase.khl` → `phase.kson`
(`current→Pop, legal→Wo, transition→Yax, fold→Sek, manifold→Ch'en, commit→Xul`).

json_runtime's `native.PHASE` (in `xcfe.cpp`) is the **execution authority**; any other
runtime (e.g. kuhul_engine's `phase_runtime.h`) is a **consumer** of that authority —
not a parallel implementation. Sync drifts; authority does not.

```
             canonical PHASE authority (phase.kson + native.PHASE)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
json_runtime                 kuhul_engine
consumer                     consumer
```

---

## Admission — the trust gate

`tools/kson_validate.py` is strict enough to be a driver admission gate:

```text
load .kson
→ verify protocol == kast/1
→ validate KastDocument schema
→ verify semantic_hash
→ verify @driver.@hash
→ validate @abi
→ validate requested capabilities/resources
→ resolve provider
→ validate phase hooks
→ mount driver
→ enter Pop

invalid KSON → REJECT (never phase execution)
```

Provider registry: `opengl → glsl_gpu` sidecar, `phase → json_runtime native.PHASE`,
`gpt2.runtime → kuhul_engine`, stdlib modules → json_runtime native by default.

---

## GPU — OpenGL (not OpenCL) is the universal target

`GL_ARB_compute_shader` + SSBO runs on every GPU since 2012 via the installed ICD.
OpenCL is present but secondary. The GLSL path is live: `glsl_gpu` sidecar on
json_runtime (verified `compiled:true, icd:ig75icd64.dll`), `gl_infer_driver.dll`
(8 shaders), `xcfe_gl_ops.dll` (17 kernels on the wgpu_native GL backend).
The GPU driver contract: `opengl.khl` (`Sek→dispatch`, `Ch'en→collect status`,
`Xul→commit tensor state`).

---

## Object Server / json_runtime

`json_runtime.exe` (port 8787) is an **ASX JSON/XCFE object server** — a semantic
graphic processor + REST API sandbox. It stores, serves, and EXECUTES JSON documents
(programs, tensors, manifests, sidecars, SCXQ2 IR) behind sandboxed `/api/*` routes
with phase gating and authority admission. **JSON is a data format, not a JavaScript
feature** — the runtime is pure C++ (`nlohmann/json` + its own XCFE evaluator).
Full concept: `bin/json-runtime/SIDECARS.md` (in the install) / this doc's sibling docs.

---

## Files

| Path | Role |
|------|------|
| `tools/khlc.py` | KHL → KAST → KSON compiler + static checks |
| `tools/kson_validate.py` | KSON admission gate |
| `stdlib/*.kuhul` + `*.kson` | Semantic modules (18) + compiled KAST |
| `drivers/khl/*.khl` + `*.kson` | Driver contracts (opengl, phase, fold, attention.fold, gpt2.runtime, inference, sw) |
| `docs/KUHUL_RUNTIME.md` | Phase-engine-as-versioned-runtime architecture |
| `bin/json-runtime/SIDECARS.md` | Sidecar system + Object Server concept (install mirror) |
| `bin/json-runtime/gpu.manifest.json` | GPU provider contracts incl. `@glsl` |
| `bin/json-runtime/sco/sidecars/glsl.json` | `glsl_gpu` sidecar |

## Commands

```powershell
# compile (sources → KAST → KSON) with static checks
python tools/khlc.py stdlib/
python tools/khlc.py drivers/khl/

# admission gate (ADMITTED / REJECTED)
python tools/kson_validate.py stdlib/
python tools/kson_validate.py drivers/khl/

# tamper self-test (must REJECT)
python tools/kson_validate.py --tamper drivers/khl/opengl.kson

# live GLSL dispatch through json_runtime (rebuilt with @profile:glsl)
# POST /api/run {program: {..., "@fn":"dispatch", "@profile":"glsl", ...}}
# POST /api/sidecars/glsl_gpu/call/glsl_probe
```
