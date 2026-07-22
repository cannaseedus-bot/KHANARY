# K'UHUL-3D vNext — Recursive Semantic + Compute Grammar

## The rule

> **The AST preserves the prompt/context; K'UHUL traverses it; XCFE chooses legal graph
> moves; opcodes perform work; compute nodes may lower to CPU, llama.cpp, WebGPU/WGSL, or
> D3D11/HLSL.**

Everything below is one contract expressing that rule. It exists so the **llama.cpp fork can
integrate incrementally**: a backend only has to understand this AST + the capability/opcode
contract — **not KXML or SCXQ2 (those lower *into* this AST later).**

## Artifacts

| File | What it is |
|---|---|
| `docs/kuhul-3d-vnext.ebnf` | the grammar (bracket surface form) |
| `docs/kuhul.ast.v3.schema.json` | the canonical AST (JSON Schema 2020-12, `$id: khanary.dev/schema/kuhul-ast-v3.json`) |
| `docs/examples/kuhul.ast.v3.example.json` | the worked example (attention + verify) |
| `docs/examples/kuhul.ast.v3.recursion.example.json` | a node whose tick nests its own nodes + tick |
| `tools/check_kuhul_ast_v3.py` | the two self-checks (both **pass**) |

## Two load-bearing design points

**1. Phases are lanes; opcodes live inside them.** `Pop / Wo / Yax / Sek / Ch'en / Xul` are the
semantic/tick traversal lanes — they are **not** in any opcode enum. Work is done by opcodes
(`OBSERVE/REPRESENT/REASON/DECIDE/REFLECT/EMIT`, `ROUTE/BRANCH/…/VERIFY`, `MATMUL/SOFTMAX/GEMM/…`,
`LOAD/STORE/…`, `CALL/DISPATCH`) that execute *inside* a lane. The grammar and schema keep this
separation visible: `phase` and `opcode` are disjoint productions; `Node.opcode` is a value, the
lane is `PhaseGroup.phase`.

**2. Recursion is real — thinking at every level.** Every `Node` may carry a `tick`: a map from
phase-lane to a `PhaseStep`, and a step may nest its own `nodes` (each again a `Node` with its own
`tick`). In the schema this is a genuine recursive `$ref` (`Node → PhaseTick → PhaseStep → Node`),
not a flattened enum. So a single `MATMUL` node can itself be a full Pop→Xul tick
(`LOAD → BIND → TILE → MATMUL → VERIFY → COMMIT`), and a nested `GEMM` inside it another.

```
K'UHUL tick
  ├─ semantic node      └─ K'UHUL tick
  ├─ micronaut          └─ K'UHUL tick
  ├─ opcode             └─ K'UHUL tick
  └─ compute dispatch   └─ K'UHUL tick
```

## XCFE routes; it does not redefine the phases

XCFE picks the **legal, best-cost backend** among registered candidates — it never re-owns the
Pop→Xul lanes. Routing is explicit instance data:

```json
"xcfe": { "route": {
  "candidates": [
    { "backend": "cache",  "requires": ["memory"], "cost": 0.01 },
    { "backend": "llama",  "requires": ["llama"],  "cost": 0.40 },
    { "backend": "webgpu", "requires": ["webgpu"], "cost": 0.25 },
    { "backend": "d3d11",  "requires": ["d3d11"],  "cost": 0.20 }
  ],
  "policy": "semantic_best_legal_path"
}}
```

## Capabilities are portable; rig-truth is instance data

The grammar's `capability` set is **general and open** (`identifier` is a valid capability) so any
backend — the llama.cpp fork, another machine's GPU — can **register** itself. A model doesn't fit
because the grammar says so; it fits because the `capabilities` object says the backend is present.

The measured truth for *this* rig is **instance data**, not baked into the contract:

```json
"capabilities": {
  "cpu": true, "numpy": true, "llama": true,   /* llama.cpp is CPU-only here */
  "d3d11": true, "hlsl": true,                 /* the working GPU path (FL 11_1) */
  "wgsl": true,                                /* via WebGL2 -> ANGLE -> D3D11 */
  "webgpu": false,                             /* Dawn/WebGPU blocklisted on HD 4600 */
  "d3d12": false                               /* only 11_x bridge; 12_0+ unsupported */
}
```

A bigger GPU flips `webgpu`/`d3d12` to `true` and the *same* AST routes differently — the contract
doesn't change, the registry does.

## The 5-gate validation contract

An AST is **ADMITTED** only after five gates pass, in order:

```
KXML / AST
  1. STRUCTURE     valid phases / nodes / edges
  2. SEMANTICS     references resolve; context preserved (source.preserve)
  3. XCFE          the transition / route is legal
  4. CAPABILITY    every requested backend is actually registered (capabilities[x] == true)
  5. COMPUTE       buffer shapes / dtypes / kernel contract valid
  -> ADMIT
```

recorded as:

```json
"validation": { "structure":"pass","semantics":"pass","xcfe":"pass",
  "capabilities":"pass","compute":"pass","admitted": true,
  "proof": { "tick": 42, "route": ["Pop:N0","Wo:N1","Yax:N2","Sek:N3","Sek:N4","Ch'en:N5","Xul:N6"] } }
```

## Incremental integration path (why this shape)

- The fork registers **existing** pieces as capabilities/backends: Micronauts, llama inference,
  WebGPU/WGSL, NumPy/wgpu-py, and the working **D3D11/HLSL** path.
- It executes this AST + opcode contract. It does **not** need KXML or SCXQ2 yet.
- KXML and SCXQ2 **lower into** this AST later — so the coder's current GPU work isn't disrupted.

## Reconciliation note (what was completed vs. the authored draft)

The authored vNext EBNF was completed — not silently rewritten — so it passes the
grammar-validator (defined-refs + reachability):

- **added `declaration = phase | xcfe | compute`** so the `xcfe` and `compute` subtrees are
  reachable from `document` (they were defined-but-unreachable);
- **`node` body now admits `xcfe | compute | tick`** — matching the AST's `kind:"xcfe"`/`"compute"`
  nodes and per-node recursion (previously nodes couldn't hold what the JSON showed);
- **defined the missing terminals/rules** referenced but absent in the draft: `body`, `operation`,
  `string`, `character`, `letter`, `digit`, `xml_attribute`, `xml_content`.

Verified by `tools/check_kuhul_ast_v3.py`:
```
[JSON] pass kuhul.ast.v3.example.json
[JSON] pass kuhul.ast.v3.recursion.example.json
[EBNF] 63 rules defined; all referenced defined; all reachable from `document`
```

## Scope

This is the **contract**, not a runtime — no XCFE router, no parser (deliberately). The reference
checker validates the contract; execution is the fork's incremental job.
