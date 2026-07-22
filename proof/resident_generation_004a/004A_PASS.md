# GPU Proof #004-A — Fixed-op Binding Reuse (frozen)

Branch `resident-generation-amortization-v1` from #003. **Oracle = frozen #003** (trajectory must
not move). Narrow claim, proven:

> Fixed-shape, fixed-resource operator bindings can persist across generation ticks **without
> altering the #003 trajectory** — binding state is *resident execution state*, not reconstructed
> tick metadata.

Only the **FIXED** op class is touched (LN, the fixed GEMMs, GELU, projections, ln_f, lm_head).
The **DYNAMIC** MHA (KV/P changes each tick) is left exactly as #003 — that's #004-B.

## Result (baseline vs reuse, same 14-tick workload, one hardware pass)

```
=== #004-A CORRECTNESS (oracle = frozen #003) ===
[traj]  trajectory(reuse) == trajectory(baseline) : EXACT  (14/14 ticks argmax-exact)
[G3]    argmax(reuse_t) == argmax(cpu_t) every tick : PASS
[G1/G5] P->P+1 all layers every tick + L0..11+ln_f+lm_head : PASS (structural)
[G2/G4] final KV vs CPU ref  baseline 5.36e-06  reuse 5.36e-06  (<=1e-3) : PASS
[seq]   reuse generated == 42447 x6  == cpu : MATCH
=== PASS: #004-A trajectory preserved under fixed-op binding reuse ===

=== #004-A AMORTIZATION (baseline -> reuse) ===
[binds]   baseline: 146/token (all rebuilt)   reuse: 134 fixed/session + 12 dynamic(MHA)/token
[record]  record/bind ms/token:  32.38  ->  9.81   (3.30x)
[exec]    exec+sync   ms/token:  90.12  ->  82.47
[total]   mean ms/token:         124.2  ->  94.0   (1.32x)
```

## PASS criterion (structural — met)

- **Correctness (non-negotiable):** `trajectory(#004-A) == trajectory(#003)` under the #003 laws.
  Exact where exact is meaningful (tokens, per-tick argmax, P-trajectory, sequence); KV within the
  established `≤1e-3` tolerance (identical `5.36e-06` both paths). No new floating-point determinism
  was required or claimed.
- **Amortization (≥1 setup class reduced):** in-loop **binding-table creations/token 146 → 12**
  (a 12× structural reduction); record/bind **3.3×**; total **1.32×**.

## Reuse ledger

| Resource | Lifetime before | Lifetime after | Reused? |
|---|---|---|---|
| Weight | model | model | yes |
| Fixed operator | model | model | yes |
| **Fixed binding** | **tick** | **session** | **NEW REUSE (134 built once)** |
| KV | growing | growing | yes |
| MHA binding | tick | tick | not yet (#004-B) |
| Scratch | tick/op | tick/op | unchanged |

## KGRC result earned

`STATIC` now splits into two things that share "construct once, preserve across ticks":

```
STATIC DATA       = weights (immutable tensor contents)         [#001]
STATIC EXEC STATE = compiled operators + persistent descriptors + invariant bindings   [#004-A]
```

## What the numbers say for #004-B

`exec+sync` (90 → 82 ms/token) barely moved — as predicted, binding reuse attacks host-side
`record/bind`, not GPU execution. The dominant term is now firmly `exec+sync` (the many tiny
single-token dispatches) plus the remaining **12 dynamic MHA bindings/token**. #004-B is the
dynamic-MHA boundary: **capacity-shaped KV** (`physical [1,Hn,MAX,Hd]`, logical extent `P ≤ C`) so
the decode MHA becomes fixed-shape and its binding reusable too — which changes how a **GROWING**
field is physically represented, so it gets its own proof (needs `MHA1` + `PastSequenceLengths`
de-risk on FL 11_1).

## Reproduce

```
python  ../../scratch/dml/gen_prep.py
cl /nologo /std:c++17 /EHsc /O2 /I include gen_004a.cpp /link /LIBPATH:lib /OUT:gen_004a.exe
gen_004a.exe    # runs baseline then reuse, compares trajectory, reports amortization
```

Frozen snapshot; evolving source `scratch/dml/gen_004a.cpp`. **Holding here** — the numbers now
determine #004-B; do not roll into MHA specialization yet.
