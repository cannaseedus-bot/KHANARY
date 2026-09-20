# Semantic Proof S#002b — Representative-corpus attempt via coverage matrix (frozen)

**Goal:** move from *conformance* (S#002a) to a *representative* corpus — drive the runtime with an
intentional coverage matrix and use vocabulary **convergence** (not a record count) as the
S#003b-readiness criterion.

**Result:** the coverage *mechanism* + convergence *measurement* work; the honest conclusion is that
**the read-only-reachable data is too shallow to justify a schema freeze.** The representative corpus
requires driving the *full* FieldExecutionEngine with varied inputs — which is blocked read-only.

## What ran (READ-ONLY, no kernel mutation)

`s002b_coverage.py` drove the live `semantic_kernel_cli compile_ir` across **14 intentional queries
in 10 categories** (simple / math / code / search / tool / missing-dep / unsupported / multistep /
ambiguous / greeting), extracting **FIELD** (predicate/tense/polarity) + **EDGE** (argument role
relations) per query, and tracked vocabulary growth for convergence.

```
14 queries / 10 categories -> 40 records (FIELD 14, EDGE 26)
vocab: predicates 11, entities 23, roles 2, tenses 1, polarities 1
edge relations: agent 14, patient 12
convergence: role vocab STABLE at 2 (agent/patient) from query 1
```

## The finding: convergence is real but DEGENERATE

The structural vocabulary saturates **immediately** — 2 roles (agent/patient), 1 tense (present),
1 polarity (affirmative) — because the CLI front-end is a shallow subject-verb-object role labeler,
not the rich semantic classifier. Predicate/entity vary, but the *schema-defining* vocabulary
(intents, folds, transitions, mutation classes, constraints, **violation types**) is **not produced
at all** on this path.

So convergence-based S#003b readiness **fails for the right reason**: the read-only signal is not
representative of the semantic state space. Freezing a schema from it would overfit a shallow parser.

## Where the representative signal actually lives (and why it's blocked)

The rich distributions live in the **full FieldExecutionEngine** (folds, `Law E` legality verdicts,
`applied_delta` mutations, MoE specialist routing) — proven observable in S#002a. But its driving
query is **hardcoded** (`verify_asx_runtime.cpp:53 run_end_to_end_step(fold_id, "What is geometric
intelligence?")`). Varying it requires **rebuilding the kernel** (a source mutation), which the
standing read-only rule on `.ASX.cpp` forbids without explicit approval. The CLI's full-cycle
commands need XML IR plans, not free text.

## Decision required to advance S#002b → S#003b

To get a representative corpus, the full engine must be driven with the coverage matrix. Options:

1. **Approve a controlled harness** — a small, additive change (a query-injection loop over the
   matrix) built against the kernel, run to emit varied full-cycle trajectories. Crosses the
   read-only boundary → needs explicit go-ahead.
2. **Stand up the GGUF bridge** (`127.0.0.1:5000`) + find/expose a free-text → full-cycle path so
   varied queries exercise specialists and produce real `VIOLATION` records — read-only if such a
   path exists.
3. **Park S#003b** with S#002a conformance banked; the algebra is validated in shape, and schema
   freeze waits until the engine is drivable.

Until one of those, S#003b (vocabulary/taxonomy/cardinality freeze) **should not proceed** — exactly
the convergence discipline: freeze only what stabilizes across *representative* runs, and this run is
not representative.

## Track position

```
S#001 static ✓  S#003a algebra ✓  S#002a live conformance ✓  S#002b coverage mechanism ✓ (this)
   → [decision] drive full engine (approval) → S#002b representative corpus → S#003b schema freeze
```

## Reproduce

```
python proof/semantic_corpus_s002b/s002b_coverage.py
```
