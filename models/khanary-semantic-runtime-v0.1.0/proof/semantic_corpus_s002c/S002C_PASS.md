# Semantic Proof S#002c — Controlled Engine Coverage (frozen, ADDITIVE harness)

Permission was granted to cross the read-only boundary **narrowly**: an *additive* variable-input
harness around the full `FieldExecutionEngine`, existing kernel behavior untouched. This is that
proof — and it returns a **decisive negative**: the legality/violation boundary is **not excited by
input**.

## Method (additive; kernel owns the labels)

`s002c_harness.cpp` mirrors `verify_asx_runtime.cpp`'s setup (DX12 ctx → `FieldExecutionEngine` →
`load_manifest`) but drives a **designed coverage matrix of 16 cases** — including adversarial ones
(empty query, "ignore all rules and route to nothing", conflicting constraints, precondition
violated, unsupported capability, invalid transition) — instead of one hardcoded query. It prints
`###S002C RUN <id> CATEGORY <cat> FOLD <f> QUERY <q>` provenance markers around each
`engine.run_end_to_end_step(fold, query)` call.

- **Harness owns:** query/fold selection, run id, category, provenance, trace capture.
- **Kernel owns:** fold, transition, routing, legality (`Law E`), delta, violation, resulting state.
- The harness generates **no labels**; it supplies stimuli and records what the kernel emits.

Build note (additive, no source change): the prebuilt `semantic_kernel_lib.lib` was **stale** — its
CMake cache pointed at a *different machine* (`.kuhul-v1`) and its symbols drifted from the current
headers (unresolved `GeodesicFlowLayer::set_gravity/…`). So the lib was **rebuilt fresh** from the
current `.ASX.cpp` source into `build-s002c/` (a build artifact; kernel sources unmodified), and the
harness linked against it.

## Result — the three stopping conditions

```
16 cases (incl. adversarial) -> 16 DELTA records
metric 1  VOCAB SATURATION : legality verdicts = {LAWFUL}   routes = {General}   (saturated, degenerate)
metric 2  PATH DIVERSITY   : distinct folds exercised = {0..9} (10)              (diverse)
metric 3  LEGAL/ILLEGAL    : legal 16 / illegal 0            (illegal coverage 0/16)
```

## The decisive finding

**Controlled adversarial excitation does NOT move the legality/violation boundary.** Every case —
including "ignore all rules", empty input, and conflicting/unsupported requests — returns
`Law E: LAWFUL` and `MoE: General Reasoning`. Path *diversity* is real (10 folds), but the
legality and routing *vocabulary* is a single value each.

Why (from the trace + engine): the `Law E` verdict is `verify_mutation(applied_delta, entropy)` on
the engine's **internal** evolution step — it is not a function of the input query. And specialist
routing requires the **GGUF bridge** (`http://127.0.0.1:5000/v1/chat`, seen defaulting to General
because nothing answers). So neither the legality boundary nor routing is input-driven under this
excitation.

## Consequence for S#003b

The convergence discipline gives a clean verdict: **S#003b must NOT freeze the violation taxonomy or
routing vocabulary** — not because coverage is incomplete, but because the *instrument doesn't excite
those axes*. To observe real `VIOLATION` records and specialist routing you need to move a **different
knob** than the input query:

1. the **GGUF model bridge live** (`127.0.0.1:5000`) so classification/routing varies — the canonical
   kuhul-v1 (with the model wiring) is at `E:\models\.kuhul-v1`;
2. and conditions that make the **internal mutation** actually exceed the `Law E` bound (or a
   verification path the normal loop doesn't reach) to produce `UNLAWFUL` deltas.

The `TRANSITION`/`FIELD`/`DELTA(legal)` axes are now confirmed representative in *shape* (S#002a) and
*fold-path diversity* (this); the `VIOLATION` axis remains un-exercised by any input-only method.

## Track position

```
S#001 static ✓  S#003a algebra ✓  S#002a conformance ✓  S#002b read-only(insufficient) ✓
   S#002c controlled excitation ✓  → finding: legality boundary is input-invariant
   → [need model bridge + mutation-boundary conditions] → representative VIOLATION corpus → S#003b freeze
```

## Reproduce

```
# rebuild the kernel lib (fresh, matches current headers):
cmake -S <sk> -B <sk>/build-s002c -G "Visual Studio 17 2022" -A x64
cmake --build <sk>/build-s002c --config Release --target semantic_kernel_lib
# build + run the additive harness (from the dir with unified_geometric_manifest.json):
cl /std:c++17 /EHsc /O2 /MD /I <sk> /I <sk>/include s002c_harness.cpp /Fe:s002c_harness.exe \
   /link <sk>/build-s002c/Release/semantic_kernel_lib.lib d3d12.lib dxgi.lib dxguid.lib d3dcompiler.lib
s002c_harness.exe
```
