# Semantic-track Proof S#001 — Corpus Extraction (frozen, READ-ONLY)

**Claim:** the existing deterministic K'UHUL kernel can act as the **teacher/oracle** — it emits
supervised semantic-trajectory training records (positive **and** negative) without inventing data
and **without mutating any kernel source**. This must be proven *before* extending KXML, so the real
traces — not guesswork — define the record algebra.

## What it does

`corpus_gen.py` reads (never writes) the K'UHUL registries and derives records. **The registry is
the legality oracle:**

- `kuhul-infrastructure/registry/phases.json` → `valid_next` = **legal traversals**; its complement
  = `VIOLATION / invalid_traversal` (exactly what `LegalityVerifier::verify_traversal` encodes).
- `kuhul-infrastructure/folds/fold_definitions.json` → `inputs/outputs` = **DELTA** (append) +
  **INVARIANT** (preserved prefix); `dependencies` unmet = `VIOLATION / dependency_missing`
  (what `verify_mutation` gates).
- `XCFE.decisions` (`USE_LOCAL / INVOKE_TOOL / INVOKE_MODEL / FALLBACK`) → `EQUIVALENCE` (distinct
  deltas achieving the same transition).

## Result (from 6 phases + 12 folds)

```
TRANSITION   8
DELTA       12
VIOLATION   28
INVARIANT   12
EQUIVALENCE  6
legal(pos)  20   illegal(neg) 28   TOTAL 66
=== PASS: all 5 record classes emitted, positives AND negatives, kernel sources unmutated ===
```

Output: `semantic-trajectories.jsonl` (66 records). One per class:

```json
{"class":"TRANSITION","state":{"fold":"Pop","have":["user_query","context","freshness_requirement"]},"transition":"Pop->Wo","next_state":{"fold":"Wo","needs":["classified_intent","fold_index"]},"legality":"legal","source":"phases.json:valid_next"}
{"class":"DELTA","field":"semantic.kernel","before":{"have":["raw_text","context"]},"delta":{"applies":"semantic.kernel","adds":["classified_intent","semantic_vector"]},"after":{"have":["raw_text","context","classified_intent","semantic_vector"]},"legality":"legal"}
{"class":"VIOLATION","state":{"fold":"Pop"},"transition":"Pop->Yax","legality":"illegal","violation":"invalid_traversal","expected_next":["Wo"]}
{"class":"INVARIANT","field":"semantic.kernel","preserve":["raw_text","context"],"mutation":"append","note":"prior state preserved; fold appends outputs (GROWING field)"}
{"class":"EQUIVALENCE","equivalence_class":"xcfe_resolution","representations":["USE_LOCAL","INVOKE_TOOL","INVOKE_MODEL","FALLBACK"],"canonical":{"transition":"Yax_validated -> Sek_ready"}}
```

Every record has a `source` field tracing it to the exact registry origin.

## The five classes map to the proof ladder

```
SEMANTIC #001  can the operation produce the correct field?      -> DELTA
SEMANTIC #002  can the field mutate correctly?                   -> DELTA / INVARIANT
SEMANTIC #003  do valid mutations compose into a trajectory?     -> TRANSITION
SEMANTIC #004  can it reject an invalid trajectory?              -> VIOLATION
(+ notation invariance)                                          -> EQUIVALENCE
```

Objective ≠ "predict the next fold" (the folds are deterministic positions, trivial). The fold is
*input/context*; the target is: is Δ legal, what does it modify, what is preserved, what is F(t+1),
which transition is legal next, and if illegal which constraint was violated.

## Honest scope

- **Read-only.** Reads `.ASX.cpp` registries; writes nothing to any kernel source. (Per standing
  rule: `.ASX.cpp` stays read-only.)
- Records here are derived from the **static registry contracts** (phases/folds). Scaling to the
  10k+ needed to freeze the schema comes from **live `FieldExecutionEngine` traces** (accepted Δ
  sequences) + the C++ `LegalityVerifier` on real mutations — the next step (S#002), which needs
  the running kernel.
- No KXML change yet. Only after this data shows the *real* record shapes does (b) freeze the
  KXML `FIELD/EDGE/TRANSITION/DELTA/INVARIANT/VIOLATION` extension.

## Semantic track (parallel to, not mixed with, the GPU track)

```
S#001 CORPUS EXTRACTION      (this)  ->  S#002 LEGALITY / NEGATIVE CORPUS (live traces)
   -> S#003 KXML RECORD ALGEBRA -> S#004 SCXQ2 TENSORIZATION -> S#005 TRAINING -> S#006 HELD-OUT TRAJECTORY TEST
```

## Reproduce

```
python proof/semantic_corpus_s001/corpus_gen.py
```
