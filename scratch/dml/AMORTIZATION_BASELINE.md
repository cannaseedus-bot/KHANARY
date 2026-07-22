# Proof #004 — Resident Generation Amortization: BASELINE (measure before changing)

Branch `resident-generation-amortization-v1` from `2d10421` (#003 tip). #003 is the oracle;
correctness must not move. This is step 1 — **measure the #003 cycle before optimizing.**

`gen_bench.cpp` = the frozen #003 runner + instrumentation (per-category timing, operation
counts, resident-byte accounting). Correctness re-verified: all G1–G5 + sequence **PASS**,
trajectory identical to #003.

## Baseline (Intel HD 4600, 8-token prompt + 6 generated = 14 ticks)

```
[latency] compile(up-front)=50.4 ms
          per-tick mean:  embed 0.74 | record/bind 31.03 | exec+sync 87.66 | readback+argmax 1.05  (ms)
          first-token=175.1 ms   mean/token=120.5 ms   total(gen)=1686.8 ms
[ops]     compiles=21 (7 fixed + 14 MHA, ALL up-front — 0/token in the loop)
          binds/token=146   syncs/token=3.0   up=3.0 KB/token   dn=196.3 KB/token (logits)
[resident] STATIC 495 MB (weights, immutable) · GROWING 1.03 MB (KV, append) · TRANSIENT 0.26 MB (scratch)
```

## What the numbers say

- **Weight uploads are already 0/token** — Proof #001's residency holds through generation.
  `up = 3 KB/token` is just the single fed-token embedding; `dn = 196 KB/token` is the logits
  readback for argmax. The model STATE never moves.
- **Dominant cost: `exec+sync` 87.7 ms/token** — this is the GPU executing **146 tiny
  single-token dispatches per tick** (12 layers × ~12 ops + ln_f + lm_head), each with dispatch
  overhead, behind 3 syncs/token.
- **Second: `record/bind` 31.0 ms/token** — creating **146 descriptor-heap + binding-table
  pairs per tick**.
- Compiles are **already amortized up-front** (21 total, 0/token in the loop) — so target (A) as
  originally framed ("recompile MHA every tick") is *already* not per-tick here; the remaining
  MHA cost is that its **binding** (past/present buffers) changes each tick because the cache is
  length-shaped.

## The two repeated per-tick setup costs to remove (one at a time)

- **(B) 146 binding-table creations/token** → *fixed-op binding reuse*: LN/GEMM/GELU/ADD bind the
  same resident weights + fixed scratch every tick, so their binding tables can be built **once**
  and reused. Only MHA's binding changes (KV grows). Low risk, no new DML op. Directly attacks
  the 31 ms/token record/bind term.
- **(A) length-shaped KV → capacity-shaped KV**: allocate `K/V` physically at `[1,Hn,MAX,Hd]`
  with a logical extent `P ≤ capacity`, so the decode MHA is a **fixed shape** and its binding is
  reusable too. Needs a de-risk: DML `MHA1` + `PastSequenceLengthsTensor` (variable valid length
  over a fixed buffer) — verify on FL 11_1 first, exactly like #002/#003. Refines the GROWING
  field with `capacity` vs `extent` (`extent(F_{t+1}) = extent(F_t)+1 ≤ capacity(F)`).

## #004 PASS criterion (structural — no arbitrary speedup required)

```
CORRECTNESS (non-negotiable):
  trajectory(#004) == trajectory(#003)   AND   G1..G5 + sequence PASS

AMORTIZATION (at least one true):
  binds/token ↓   and/or   syncs/token ↓   and/or   compiles ↓
  with latency reported honestly (before/after).
```

The #004 runner will print a **CORRECTNESS** section (G1–G5 + trajectory match vs the #003
oracle) and, separately, an **AMORTIZATION** section (per-token op counts + latency, baseline vs
optimized) — so "faster" can never quietly redefine "correct."

## Honest note

`exec+sync` (the true dominant term) is GPU dispatch overhead of many tiny ops; binding reuse
attacks `record/bind`, not `exec`. Cutting `exec` needs *fewer/larger* dispatches (op fusion /
batching) — a later, harder boundary. #004 v1 removes **(B)** first (cleanly measurable), then
de-risks **(A)**; op fusion is deferred so each step introduces one change.
