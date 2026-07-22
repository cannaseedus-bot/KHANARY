# What the semantic tensor should learn (design)

Grounded in the tensor-field model (`docs/tensor-fields-and-residency.md`) and Proofs #001–#003.
Direction set in design discussion; the sharpening here is that Proof #002/#003 give the semantic
tensor an **exact algebra to learn**, not a vague "phase-aware representation."

## Thesis

The semantic tensor should learn **neither English nor the fixed Pop→Wo→Yax sequence**. It should
learn the **geometry of meaning and state transition** — the invariant relationships, constraints,
transformations, and state trajectories that stay true *underneath* text / MathML / KXML / XJSON /
code / events.

> LLM:            `words → words`
> semantic tensor: `STATE + RELATIONS + CONTEXT → VALID NEXT TRANSFORMATION`

That makes it **complementary** to GPT/Qwen, not another inferior language model.

## The precise target — it's the algebra we already proved

Proof #002/#003 established, for attention KV, a general append-state law:

```
F(t+1) = Preserve(F(t)) ⊕ Δ(t)      InputState(t+1) ≡ OutputState(t)      (CONTINUITY)
```

with residency classes by **mutation geometry**: STATIC (immutable) / GROWING (append+preserve
prefix) / TRANSIENT (replaceable). **The semantic tensor learns this same algebra, lifted from
attention-K/V tensors to semantic FIELDS.** A training record is a labeled instance of it:

```
(STATE_t, EVENT_t, CONTEXT_t, RELATIONS_t, CONSTRAINTS_t)  →  STATE_{t+1}   [valid | violation]
```

The supervision is not `Sek → Ch'en` (folds are deterministic structural positions — trivial).
It is: **given this semantic field, THIS transformation is the valid one** (and these are not).

## Dataset priorities (endorsed) and what each teaches

| Pri | Dataset | Proven-law tie-in |
|---|---|---|
| 1 | State-transition trajectories | `F(t+1)=…⊕Δ`; the core |
| 1 | Positive **and negative** transitions | the semantic *manifold* boundary (valid vs `violation`) |
| 1 | Semantic graphs (real triples) | SCXQ2 `FIELD —EDGE— FIELD` |
| 1 | Equivalence / invariance sets | representation invariance (text/MathML/KXML → one canonical) |
| 1 | MathML / KXML transformation laws | exact ground truth (`a+b=b+a` ✓, `a−b=b−a` ✗) |
| 2 | Mutation trajectories | STATIC/GROWING/TRANSIENT labels; *same law as #1, different granularity* |
| 2 | Contradiction / constraint sets | what can/can't coexist ("two things can be true" — same object+time = contradiction; different time = valid transition) |
| 2 | Event / causal sequences | cause vs correlation vs dependency vs sequence |
| 2 | Code AST / semantic graphs | program meaning |
| 3 | Conversations | **repurposed**: extract intent/entities/relations/required-capability, NOT `user→assistant` text |
| 3 | Raw documents | background semantic knowledge |

Datasets #1 (state-transition) and #2 (mutation) are the **same law** — carry the mutation labels
on every transition record.

## Coordinates are structured FIELD features — NOT hashes

Remove the prototype's `token.GetHashCode() % 100` tensorization. Hashing destroys the very
geometry the tensor-field work is built on (a coordinate must *mean* something). Semantic
coordinates derive from explicit FIELD features, then learned dims refine that structure:

```
FIELD: concept · relation · role · fold · operator · state · confidence ·
       temporal_position · mutation_class · dependency · constraints
```

These are **rectilinear** semantic tensors (non-spatial geometry) — the same tensor domain as the
gpt2 grids and the birdsong mesh, one algebra with different coordinate semantics.

## Authoring format (JSONL) — reuse the KGRC field-contract labels

```json
{"id":"s001","fold":"Sek","input":{"concept":"multiply","args":[7,8]},"relations":[["7","operand_of","multiply"],["8","operand_of","multiply"]],"constraints":[],"transition":"execute","next":{"fold":"Chen","result":56},"valid":true}
{"id":"s002","fold":"Sek","input":{"concept":"multiply","args":[7,8]},"transition":"web_search","valid":false,"violation":"capability_mismatch"}
{"id":"s003","equivalence_class":"multiply_7_8","representations":[{"type":"text","value":"seven times eight"},{"type":"mathml","value":"…"},{"type":"kxml","value":"…"}],"canonical":{"operator":"multiply","args":[7,8]}}
{"id":"s004","field":"conversation_memory","class":"GROWING","mutation":"append","preserve":"prefix","before":["A","B"],"delta":"C","after":["A","B","C"],"valid":true}
```

`class` / `mutation` / `preserve` are the exact fields from
`proof/dml_mha_kv_cache_v1/field_contract.json`. Pipeline:

```
JSONL corpus → semantic normalization → KXML / canonical semantic graph → SCXQ2 → rectilinear semantic tensors → training
```

## ⚠️ KXML must be extended (dependency)

Today KXML encodes only **chat turns + tool calls** as glyph tokens:
`{role, content}` / `{role:"assistant", tool_call:{name,args}}` (see `tools/kxml_chat_template.py`).
If the semantic tensor trains on semantic-field/transition records, **KXML — as a primary canonical
representation (dataset class above) — must gain record kinds to express them**:

- `FIELD` records (concept/relation/role/state/mutation_class/constraints) as glyph tokens,
- `EDGE` records (producer → consumer / state transition),
- `TRANSITION` records (`STATE_t + Δ → STATE_{t+1}`, with `valid`/`violation`),

extending `glyph_tokenizer.encode_dialogue` beyond `encode_turn`/`encode_tool_call`. This is a
**tracked schema change**, not done here (design first) — but any change to the semantic-tensor
target implies a matching KXML schema update, kept in lock-step.

## Grounding: the target already exists as running infrastructure

This is **not greenfield**. The `.ASX.cpp` K'UHUL/KXML kernel already implements the machinery the
semantic tensor should learn to approximate:

- **`kuhul-infrastructure/registry/phases.json`** — the transition graph as data:
  `Pop → Wo → Yax → XCFE → {USE_LOCAL | INVOKE_TOOL | INVOKE_MODEL | FALLBACK} → Sek → Xul`, each
  phase declaring `inputs → outputs → valid_next`. **This is the positive state-transition schema.**
- **`kuhul-infrastructure/folds/fold_definitions.json`** — folds with `inputs/outputs/skills`
  (`intent_classification`, `entity_extraction`, `semantic_embedding`, `confidence_scoring`…).
- **`kxml-semantic-kernel/semantic_kernel_cpp/include/legality_verifier.h`** —
  `verify_mutation(AdapterDelta δ, entropy) → LegalityVerdict` and
  `verify_traversal(from_fold, to_fold, geodesic_cost)`: **the valid/violation labeler, for both
  deltas and transitions, already coded.**
- **`field_execution_engine.h`** — runs `F(t+1) = Preserve(F(t)) ⊕ Δ` live: `applied_delta` = Δ(t),
  gated by `verify_mutation`. (Alongside `kv_cache.h`, `memory_compressor.h`, `replay_lane_manager.h`
  — the STATIC/GROWING/TRANSIENT + replay machinery.)

**Practical consequence — bootstrap, don't hand-author.** The priority-1 datasets are *generated*
from this infrastructure, not written by hand:

- **Positive transitions** ← `phases.json.valid_next` + `fold_definitions` I/O + real
  `FieldExecutionEngine` traces (`applied_delta` sequences).
- **Negative transitions** ← the `LegalityVerifier`'s rejections (`verify_mutation` = ILLEGAL,
  `verify_traversal` = false) — automatically-labeled `violation` records.

So the semantic tensor is trained to **predict in one forward pass what the verify → plan → execute
loop currently computes** — collapsing the fold cycle it was built to run.

## "Update KXML" — precise location

The KXML extension is in **`kxml-semantic-kernel`** (`kxml_compiler.rs` + `semantic_kernel_cpp/
include/semantic_kernel.h`), not only the KHANARY Python template. The record types already exist
as C++ (`AdapterDelta` = Δ, `LegalityReport` = valid/violation, fold traversal = transition);
KXML must gain glyph-record kinds to **serialize FIELD / EDGE / TRANSITION / DELTA** the way it
already serializes chat turns + tool calls (`glyph_tokenizer.encode_dialogue`). Kept in lock-step
with any change to the training target.

## One sentence

The semantic tensor learns the invariant relationships, constraints, transformations, and state
trajectories that remain true underneath every surface representation — i.e. the **proven
`F(t+1)=Preserve(F(t))⊕Δ(t)` algebra applied to semantic fields** — which is what makes it a
*semantic* tensor rather than an embedding model with K'UHUL labels pasted on.
