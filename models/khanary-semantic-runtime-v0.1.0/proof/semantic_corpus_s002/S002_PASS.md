# Semantic Proof S#002 — Live Trajectory Extraction (frozen, READ-ONLY)

**Claim:** the frozen S#003a record algebra (designed from 66 *static* S#001 records) actually
matches **real runtime behavior** — the live `FieldExecutionEngine` emits trajectories that
conform to it. Prove this by *observing* an existing runtime, mutating no kernel source.

## Method (observe, don't invent, don't mutate)

The prebuilt `verify_asx.exe` (the FieldExecutionEngine driver) + its `unified_geometric_manifest.json`
were **copied out** of `.ASX.cpp` into a non-invasive working dir (`scratch/s002/`) and run there;
`s002_extract.py` parses its emitted per-tick stdout trace into the S#003a algebra. No `.ASX.cpp`
source is modified; the binary's own existing logs are the data.

Each live tick emits: `PHASE: PERCEPTION → ROUTING → COMPUTE → META → PROJECTION`, a
`[PASS] Law E: Mutation verified as LAWFUL` verdict, and a projected `:root` field state
(entropy/attention/pressure/replay-affinity/active-fold/expert-id/coherence).

Mapping to the algebra:
- phase edges → **TRANSITION** (legal — the runtime executed them)
- `Law E` verdict → **DELTA** + `legality` (illegal ⇒ `VIOLATION`)
- projected `:root` → **FIELD** (a GROWING/session field)

## Result

```
10 live ticks -> 60 records
  TRANSITION  40
  DELTA       10
  FIELD       10
  legal 50   illegal 0
[dist] folds visited: {0:12, 1:12, 4:8, 5:8}
[dist] MoE routes: {General Reasoning: 10}
[dist] coherence: 0.6192 (constant)
=== PASS: live trajectories conform to the frozen S#003a algebra (kernel sources unmutated) ===
```

**The algebra is confirmed representative** — records extracted from live execution slot cleanly
into the same `class`/`legality`/`source` shapes designed from the static corpus. Every live record
carries `source` provenance (`verify_asx:tick` / `:LawE` / `:projection`).

## Honest scope — conformance proven, distribution not yet representative

This run is **fixed-query and degenerate**: all 10 ticks run the same query, all verdicts are
`LAWFUL` (0 violations), MoE always defaults to General Reasoning (the trace shows a `Bridge: GGUF
call to 127.0.0.1:5000` to a local LLM server that isn't running), and coherence is constant. So:

- ✅ **Conformance** (live records fit the algebra) — proven.
- ⛔ **Distribution** (violation rates, mutation-class frequencies, relation entropy, coherence
  spread) — **not yet representative**. The statistics the S#003b schema-freeze needs require
  driving the kernel with **varied inputs** (different queries/manifests, the LLM bridge live) to
  exercise specialists and produce `VIOLATION` records. That is the S#002 *scaling* step toward the
  10k+ corpus — the runnable path is now proven; only the input variety is missing.

## Track position

```
S#001 static corpus ✓   S#003a record algebra ✓   S#002 live conformance ✓ (this)
   → S#002-scale (varied inputs → 10k+, real distributions)  → S#003b schema/vocab freeze  → S#004 SCXQ2 → training
```

The GPU ladder (#001–#004) is frozen; this is the parallel semantic track.

## Reproduce

```
# stage (copied out of .ASX.cpp, read-only): scratch/s002/{verify_asx.exe, unified_geometric_manifest.json}
python proof/semantic_corpus_s002/s002_extract.py
```
