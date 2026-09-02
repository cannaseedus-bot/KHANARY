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

## Language — two surfaces (same semantics)

K'UHUL has **two source-language surfaces** sharing the same glyph/phase vocabulary
(`Pop/Wo/Yax/Sek/Ch'en/Xul`, bindings, deterministic execution, provider calls):

| | KUHUL-ES (pre-existing) | KHL / KAST (khlc) |
|---|---|---|
| Syntax | ECMAScript-flavored | KLSL-style `⟁ Phase ⟁` blocks + `glyph` bodies |
| Parse | TypeScript AST (`KUHULParser`, `compiler/src/parser.ts`) | line-oriented `tools/khlc.py` |
| Bindings | `pi x = 10;` (immutable) / `tau t = 0;` (temporal + history) | `bind var = expr` in phase blocks |
| Glyph calls | `yield* Sek('log', msg)` / `yield* Pop(...)` | `dispatch`/`probe`/`resolve`/… verbs → opcodes |
| Control | `@if` / `@for` / `@while` directives + `function*` generators | `if cond :: … done`, `for each X in C :: … done` |
| Output | transformed JS + `KUHULRuntime` (deterministic **hash chain**) | KAST (`kast/1`) → KSON → canonical phase engine |
| Host | Node (`.kuhules`/`.ts`) | any (JSON is language-neutral) |
| CLI | `kuhul-es run/compile/new/server/doctor` | `python tools/khlc.py …` / `python tools/kson_validate.py …` |

**kuhul-es 1.1.0** (2026-08-08): full audit + hardening. `run` actually executes
(KUHULRuntimeNode + physics/thought traces); `src/index.js` fixed (was a PowerShell
script); parser.js + driver-kast.js converted to CommonJS (no mixed ESM/CJS);
npm scripts added (test/build/start/doctor); **controlled thinking engine**
(`runtime/src/think.js` — `Noj` glyph + `Sek('think', …)` alias): bounded
(maxDepth/maxBreadth/maxRules), deterministic (same beliefs+rules+query → same
hash trace), auditable (every thought a KAST-like node with fold/opcode INFERE/
SHA-256), contained (read-only advice, never mutates π or the program counter),
physics-integrated (`KuhulPhysics.reflect()` updates attention/pressure/entropy).
**19/19 tests pass** (`npm test`). Live gap notes: `GAPS_AND_UPDATES.md`.
Known limitation: the runtime parser is regex-based — pass literals to `Noj(...)`,
not JS expressions.

**Driver-only KAST = secure admission surface** (1.0.23 prepared):
`compiler/src/driver-kast.js` — `toDriverOnly(fullKast)` strips ALL application
nodes/edges and emits `kind: 'driver-only'` with the hashed `@driver` contract +
`@admission` rules: allowed glyphs/opcodes/folds **derived from actual usage**
(least privilege), max nodes/edges, and resource limits (memory/workgroup/
dispatch). `verifyDriverOnly()` (JS) + `verify_driver_only()` (Python gate):
ABI, provider whitelist, capability allowlist, resource limits, contract-hash
tamper detection. Verified: ADMIT with matching caps; REJECT on missing
capability / ABI mismatch / tamper. The `.khl` driver contracts (opengl.khl,
phase.khl, …) are the khlc/Python-side implementation (`drivers/khl/`).

**Driver-only KAST (secure sandbox surface, prepared for 1.0.22):** `--driver-only`
strips ALL application layers (pi/tau value binds, glyph calls, generators,
directives) and emits only admitted capability nodes + the `@driver` contract.
For untrusted model execution: the sandbox mounts the capabilities and executes
ONLY through the declared phase hooks — the program body never ships. Declare the
surface with pi bindings (`pi provider = 'glsl_gpu'; pi capabilities = [...]`).
The `.khl` driver contracts (opengl.khl, phase.khl, …) are the khlc/Python-side
equivalent (`drivers/khl/`, not shipped in the npm package). Verified: driver-only
KSON ADMITS through the full gate (hash/abi/capabilities/provider/phase_hooks/mount).

**1.0.20 live** (2026-08-08): `compile` emits canonical KAST/KSON — `.kuhules` is a
front end into the same `kast/1 → KSON → phase engine` pipeline as `.kuhul`/`.khl`.
Semantic rules: **phase glyph ≠ opcode** (`yield* Sek('log',…)` → `fold=Sek,
glyph=Sek, opcode=DISPATCH, symbol=log`) and **application KAST ≠ driver KAST**
(plain programs emit no `@driver`; only `--driver` provider bindings carry the
contract). `pi`/`tau` (ASCII) extraction fixed (bare `pi` doesn't parse as a TS
variable statement — regex fallback added, matching the runtime). Added
`kuhul-es train <config.json>` — the GLSL trainer (semantic skeleton
EMBED→LAYERNORM→FFN→LM_HEAD→LOSS→FIELD_OPTIMIZER, physics-driven optimizer,
GLSL kernels compiled through the glsl_gpu sidecar). Verified end-to-end from the
published package: compile→admit (application + driver), trainer loss 0.156,
skeleton exports all 6 nodes.

**KUHUL-ES** (`compiler/src/parser.ts` + `.js`/`.d.ts`, `kuhul-es-1.0.18/`, CLI
`dist/khanary-server/kuhul-es.cjs`) parses with the TypeScript compiler
(`ts.createSourceFile`), detects π/τ bindings, `yield*` glyph calls, @-directives, and
generator functions, and emits a runtime with a hash-chained execution trace. It is
the JS-hosted surface; KHL/KAST is the IR/phase-engine surface. Both speak the same
K'UHUL semantics.

> **Dependency note (fixed 2026-08-08):** `kuhul-es` is published on npm as the
> user's own package — **`kuhul-es@1.0.19` live** (maintainer `xjson
> <canna.seed.us@gmail.com>`), now with the **K'UHUL physics engine**
> (`runtime/src/physics.js`). Its only runtime dependency is **commander** — an
> MIT-licensed, zero-dependency, open-source Node.js CLI argument parser (the
> standard one, by TJ Holowaychuk; the Python equivalent is argparse). Not
> vendor-specific, not closed-source.
> All three CLI entry points work: `npm i` in `kuhul-es-1.0.18/` (commander ^11),
> `npm i kuhul-es commander` + fixed pkg path in `dist/khanary-server/`, and
> package.json + `npm i` in `.Powernaut/kuhul/`.
> The **parser source** (`compiler/src/parser.ts`) remains the durable artifact.

### K'UHUL physics engine (runtime/src/physics.js, published in 1.0.19)

**Runtime physics = semantic execution metrics**, not a Newtonian simulation.
The equations are the runtime state model that influences execution — gravity gates
the learning rate, entropy/attention/pressure route attention, affinity tracks fold
replay. They are K'UHUL scheduling/execution heuristics ("not rendering. Computing.").

The semantic physics of the phase machine — matches FieldExecutionEngine
(SEMANTIC_ENGINE.md):

```
gravity_gate = clamp(1.0 + 0.35·pressure - 0.25·entropy + 0.15·attention + 0.10·affinity, 0.1, 4.0)
gravity      = 9.80665 · gravity_gate
arc_bias[i]  = 1.0 + 0.10·attention - 0.08·entropy + 0.06·pressure + 0.04·affinity
arc_weight[i]= clamp((1/√1024) · arc_bias[i], 0.01, 2.0)
velocity[i]  = 0.001 · (attention - entropy) · (1 + i%7)
```

Phase hooks: `Pop` perceive (affinity ↑, entropy decays) → `Wo` represent (pressure
builds) → `Yax` plan (attention focuses) → `Sek` execute (attention spikes, pressure
drains) → `Ch'en` project (entropy rises) → `Xul` consolidate (gravity scales up,
antigravity → 1.0). Every tick snapshots into the deterministic hash chain.
Verified trace: gravity 9.80665 → 11.92 across the cycle, affinity 0 → 0.17.

```js
// main.kuhules — KUHUL-ES surface
pi config = { name: "app", version: "1.0.0" };   // immutable
 tau frame = 0;                                    // temporal + history
function* main() {
  yield* Pop("init");
  yield* Sek('log', config.name);
  yield* Xul();
}
main();
```

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

## kuhul-folds — .NET 8 phase cycle runtime

`dist/kuhul_folds/` is the **compiled C# implementation** of the Pop→Wo→Yax→Sek→Chen→Xul
cycle. Each phase is a `IFoldStage` DLL; `FoldOrchestrator` reflects them at runtime from
`kuhul.fold-runtime.json` — no concrete type imports, composition lives entirely in JSON.

```text
kuhul.fold-runtime.json     — composition: phase→DLL path, admission thresholds
FoldOrchestrator.cs         — traversal: reflect stages, enforce closed address space, CycleIdentity
Kuhul.Folds.Contract/       — ABI: FoldContext (ref), FoldResult, IFoldStage, MicronautScore
Pop / Wo / Yax / Sek / Chen / Xul  — phase DLLs (meaning lives here)
```

### Yax — three-outcome admission gate

Yax scores the `MuField` by `S = W×C×R`; the dominant micronaut routes to **Sek** in all
three outcomes (Yax never skips Chen or Xul):

| Outcome | Condition | Sek action |
|---------|-----------|------------|
| `STRONG` | `S ≥ strong_threshold` (0.50) | execute V directly → output leases |
| `WEAK` | `weak_threshold ≤ S < strong_threshold` | route → `sidecar://micronaut-evolution/dispatch` |
| `NONE` | `S < weak_threshold` (0.15) or empty field | route → `sidecar://micronaut-factory/dispatch` |

W = competence (V magnitude, slow-evolving via MX-2 IDB).
C = confidence (K signal, per-execution, updated by Chen reward).
R = relevance (Q·K dot, set by Pop, never mutated by reward).

### Chen — reward measurement + C update

Chen computes an honest in-fold reward from two observable signals:

```
reward = projectionScore + arcScore
  projectionScore = 0.6   if Outcome==Strong AND Result non-empty
  arcScore        = min(0.4, ArcState × 0.4)   — Yax attention weight as proxy
```

For `Strong` outcomes: `C ← C + lr × (reward − C)` (per-execution EMA, immediate).
W is NOT updated within a fold — Xul logs `(node, W, C, R, reward, outcome)` to
`ProofTrace` for **MX-2 IDB** to evolve W across folds.

### Admission block (`kuhul.fold-runtime.json`)

```json
"admission": {
  "strong_threshold": 0.50,
  "weak_threshold":   0.15,
  "reward_lr":        0.10,
  "_note": "Yax: score>=strong→STRONG, >=weak→WEAK (evolve), <weak→NONE (factory). Chen: C←C+lr*(reward-C). W deferred to MX-2."
}
```

`FoldOrchestrator` seeds `FoldContext.StrongThreshold / WeakThreshold / LearningRate` from
this block before the first fold. `CycleIdentity = SHA256(SHA256(Pop.dll)‖…‖SHA256(Xul.dll))` —
binary identity, input-independent; same binaries → same identity across runs.

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
| `compiler/src/parser.ts` / `.js` / `.d.ts` | KUHUL-ES parser (TS-AST, π/τ bindings, `yield*` glyph calls) |
| `dist/khanary-server/kuhul-es.cjs` | KUHUL-ES CLI (run/compile/new/server/doctor) |

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

# KUHUL-ES (JS-hosted surface)
node dist/khanary-server/kuhul-es.cjs run main.kuhules
node dist/khanary-server/kuhul-es.cjs doctor

# live GLSL dispatch through json_runtime (rebuilt with @profile:glsl)
# POST /api/run {program: {..., "@fn":"dispatch", "@profile":"glsl", ...}}
# POST /api/sidecars/glsl_gpu/call/glsl_probe
```
