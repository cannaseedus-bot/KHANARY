# XCFE — the legal-move layer (reference)

XCFE is the **control layer over the K'UHUL phase graph**. It does not own the phases and it does not
do compute — it decides which **moves are legal** and which **backend** a compute node routes to. Think
of it as *chess moves over phase topology*: the board is the Pop→Xul lane graph, XCFE picks the legal
next move and the best-cost legal backend.

> XCFE **routes**; it never redefines the phases (see LAW P1). Same AST, capabilities vary → the route
> varies — not the meaning.

## Legal phase transitions

XCFE admits a move only if it is a legal edge of the phase state machine:

| from | to | guard |
|---|---|---|
| Pop | Wo | all inputs bound |
| Wo | Sek | intent declared |
| Wo | Yax | route selected |
| Yax | Sek | fit confirmed |
| Sek | Ch'en | execution complete |
| Ch'en | Xul | output emitted |
| Xul | Pop | reset (cycle) |

Any edge not in this table is an illegal move — XCFE rejects it before dispatch.

## Routing (the `xcfe` productions)

From `docs/kuhul-3d-vnext.ebnf`:

```ebnf
xcfe            = "[XCFE]", { xcfe_rule }, "[/XCFE]" ;
xcfe_rule       = condition_rule | route_rule | reward_rule | mutation_rule | capability_rule ;
condition_rule  = "@if", expression, { node | opcode_call | tool_call },
                  [ "@else", { node | opcode_call | tool_call } ] ;
route_rule      = "@route", expression, "->", identifier ;
reward_rule     = "@reward", expression ;
mutation_rule   = "@mutate", identifier, [expression] ;
capability_rule = "@requires", capability ;
```

A route weighs **capability-satisfying candidates** by cost and picks the best legal one:

```json
"xcfe": { "route": {
  "candidates": [
    { "backend": "cache",  "requires": ["memory"], "cost": 0.01 },
    { "backend": "d3d11",  "requires": ["d3d11"],  "cost": 0.20 },
    { "backend": "webgpu", "requires": ["webgpu"], "cost": 0.25 },
    { "backend": "llama",  "requires": ["llama"],  "cost": 0.40 }
  ],
  "policy": "semantic_best_legal_path"
}}
```

## The 5 admission gates

XCFE is gate 3 of the admission contract — an AST is **ADMITTED** only if all five pass, in order:

```
1. STRUCTURE     valid phases / nodes / edges
2. SEMANTICS     references resolve; context preserved
3. XCFE          the transition / route is legal      <-- this layer
4. CAPABILITY    every requested backend is registered
5. COMPUTE       buffer shapes / dtypes / kernel contract valid
-> ADMIT
```

## Where it sits

```
K'UHUL traversal (phases)  →  XCFE (legal moves + routing)  →  opcodes (work)  →  backend
```

XCFE is the routing/legality authority; the phases schedule, the opcodes compute, the backend
executes. Machine-checked laws P1/R1/G1 (`tools/check_kuhul_ast_v3.py`) are the admission gates.
