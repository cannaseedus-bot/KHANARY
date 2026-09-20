# KXML Semantic Record Algebra — v0.x (S#003a: ALGEBRA freeze, design-only)

Derived from **observed** S#001 records (`proof/semantic_corpus_s001/`), not invented ahead of the
data. **Design-only: no `.ASX.cpp` and no runtime mutation.** This freezes the record *kinds*, their
semantic roles, the transition algebra, and provenance — and deliberately leaves vocabularies and
cardinalities open until S#002's live corpus (that is the later **S#003b schema freeze**).

## Core rule — legality is a property, not a world

> `legal` and `illegal` are **not** different structural universes. Legality is a property of an
> **attempted** transition/delta. A `VIOLATION` is the structured explanation of *why* an attempt
> failed — attached to the attempt, not a parallel record space.

```
FIELD ──EDGE── FIELD
  │
  ├── DELTA        legal → resulting FIELD        illegal → carries VIOLATION
  └── TRANSITION   legal → next FIELD/state       illegal → carries VIOLATION
```

So `TRANSITION` and `DELTA` have one shape each; `legality ∈ {legal, illegal}` is a field, and an
illegal attempt embeds/links a `VIOLATION`. The corpus generator may emit them as separate rows for
convenience, but the *algebra* treats a violation as an annotation of a failed attempt.

## The law, explicit (not implicit in KXML)

```
F(t+1) = Preserve(F(t)) ⊕ Δ(t)
         └ INVARIANT ┘  └ DELTA ┘
```

`DELTA` carries `Δ(t)`; `INVARIANT` carries `Preserve(F(t))`; together they define a `TRANSITION`.
KXML represents all three so a transition is reconstructable and checkable, not just asserted.

## The six record kinds (FROZEN fields ← S#001; EXTENSIBLE ← S#002)

Each kind lists **required** fields (frozen now, every one observed in S#001) and **extensible**
fields (open until S#002). Every record MUST carry `class`, `id`, and `source` (provenance).

### FIELD — a semantic state node (rectilinear semantic tensor)
```
required:    identity, have[]              (frozen)
extensible:  needs[], mutation_class{STATIC|GROWING|TRANSIENT}, residency, semantics, confidence
```

### EDGE — a typed relation between fields
```
required:    from, to, relation            (frozen)
extensible:  dependency, weight, temporal
```

### TRANSITION — an attempted state movement (F_t → F_{t+1})
```
required:    state, transition, next_state, legality   (frozen)
extensible:  action, violation(when illegal), cost, confidence
```

### DELTA — an attempted mutation (before ⊕ Δ → after)
```
required:    field, before, delta{applies, adds[]}, after, legality   (frozen)
extensible:  delta{removes[], changes[]}, violation(when illegal), operator
```

### INVARIANT — what must NOT change across a transition
```
required:    field, preserve[]             (frozen)
extensible:  scope, law (e.g. "prefix"), mutation{append}, growth
```

### VIOLATION — the structured reason an attempt is illegal
```
required:    attempted_operation, type, legality:illegal   (frozen)
extensible:  constraint, expected, observed/missing
```
(`type` seen in S#001: `invalid_traversal`, `dependency_missing` — a **taxonomy left open**.)

## Provenance is required (frozen)

Every record carries `source` tracing it to its origin (`phases.json:valid_next`,
`fold_definitions:io`, `fold_definitions:dependencies`, `phases.json:XCFE.decisions`, or — later —
`FieldExecutionEngine:trace` / `LegalityVerifier:verdict`). A record without provenance is invalid.

## Direct correspondence to S#001

| S#001 class | Algebra kind | Law role |
|---|---|---|
| TRANSITION | TRANSITION | the state edge |
| DELTA | DELTA | `Δ(t)` |
| INVARIANT | INVARIANT | `Preserve(F(t))` |
| VIOLATION | VIOLATION (annotation) | illegal-attempt explanation |
| EQUIVALENCE | (FIELD set + EDGE `same_as`) | notation invariance: many representations → one FIELD |

`EQUIVALENCE` is not a seventh kind — it is a set of FIELDs joined by `EDGE.relation = "equivalent"`
to one canonical FIELD (so it reuses FIELD + EDGE). This keeps the algebra at six kinds.

## KXML serialization (design intent — not implemented here)

KXML today serializes `{role, content}` / `{tool_call}` as glyph tokens via
`glyph_tokenizer.encode_dialogue` (KHANARY) — and the real kernel compiles KXML in
`kxml-semantic-kernel/kxml_compiler.rs` + `semantic_kernel_cpp/include/semantic_kernel.h`. The
extension adds glyph-record kinds for the six above, alongside chat turns and tool calls:

```
encode_dialogue → { encode_turn, encode_tool_call,
                    encode_field, encode_edge, encode_transition,
                    encode_delta, encode_invariant, encode_violation }
```

The C++ types already exist to map onto (`AdapterDelta` → DELTA, `LegalityReport` → VIOLATION, fold
traversal → TRANSITION), so KXML gains *serialization* of structures the kernel already produces —
no new semantics. **This mapping is specified, not applied.**

## Freeze boundary

**Frozen now (S#003a — this doc):**
record *kinds* (6) · their semantic roles · the `legal/illegal = property-not-world` rule · the law
`F=Preserve⊕Δ` and which kind carries which term · required fields per kind · provenance requirement.

**Deferred to S#003b (after S#002's live 10k+ corpus):**
optional-attribute set · constraint vocabulary · mutation-operator set · violation taxonomy ·
relation taxonomy · exact field cardinalities / conformance schema.

This algebra-vs-schema split stops the 66-record static bootstrap from overfitting the language.

## Track position

```
S#001 corpus extraction ✓  →  S#003a KXML ALGEBRA freeze (this)  →  S#002 live trajectories (10k+)
   →  S#003b KXML schema/conformance freeze  →  S#004 SCXQ2 tensorization  →  S#005 training  →  S#006 held-out
```

Design captured. **GPU track (#001–#004) is untouched; next action returns to #004** on
`resident-generation-amortization-v1`.
