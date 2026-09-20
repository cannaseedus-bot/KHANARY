# K'UHUL GPU Proof Ladder — #001–#004 FROZEN (HD 4600 / DirectML)

Each rung introduces exactly one architectural claim, verified on an Intel HD 4600 (D3D11 FL 11_1
/ DirectML on D3D12). Frozen artifacts under `proof/`.

| # | Claim | Result | Proof |
|---|---|---|---|
| **#001** | Resident computation is correct | whole gpt2 forward, weights resident; logits scale-norm 1.92e-06, argmax match | `proof/gpt2_hd4600_resident_v1/` |
| **#002** | Resident state transition is correct | KV decode step: growth/preserve/append exact, output 8.08e-08 | `proof/dml_mha_kv_cache_v1/` |
| **#003** | Resident trajectory composes | 14-tick autoregressive generation; G1–G5 + sequence; KV 5.36e-06 | `proof/gpt2_resident_generation_v1/` |
| **#004-A** | Fixed execution state is reusable | binding creations 146→12/token; record/bind 3.30×; trajectory == #003 | `proof/resident_generation_004a/` |
| **#004-B1** | Capacity ≠ extent (semantics + backend diagnosis) | extent<capacity → same output 8.08e-08; **native reusable dynamic MHA unavailable on this DirectML** | `proof/resident_generation_004b1/` |

**#004 is frozen.** The amortization question is answered on this backend: the achievable host-side
win (fixed-op binding reuse) is banked in #004-A; the dynamic-MHA boundary is *diagnosed* in #004-B1
as a DirectML limitation on this hardware, not an un-optimized redundancy.

## The KGRC concepts earned along the ladder

- **STATIC** splits into **STATIC DATA** (weights) + **STATIC EXECUTION STATE** (compiled ops,
  persistent descriptors, invariant bindings) — both *construct once, preserve across ticks* (#004-A).
- **GROWING** splits into **PHYSICAL capacity C** (fixed allocation) + **SEMANTIC extent P ≤ C**
  (valid region; only the filled portion computes) (#004-B1).
- **CONTINUITY**: `InputState(t+1) ≡ OutputState(t)` across the closed `Xul→Pop` cycle (#003).
- Append law `F(t+1) = Preserve(F(t)) ⊕ Δ(t)` (#002).

## Backend-conformance principle (from #004-B1)

> **Semantic extent ≠ physical realization.** The KGRC model says *capacity fixed, extent grows*;
> each backend realizes that differently, and a backend may not realize it at all.

```
KGRC (capacity fixed, extent grows)
  ├── DirectML (HD 4600)  → present appends at physical end; native fixed-capacity op (MHA1) absent
  ├── CUDA                → compaction
  ├── Vulkan              → descriptor indexing
  └── custom SCXQ2        → native capacity buffer
```

#004-B1 is therefore a **conformance result**: DirectML's realization of a GROWING field on this
rig differs from the abstract model. That is knowledge for the runtime, not a defect to fix.

## Next chapter (not a stretch of #004)

- **#005 — Resident Capacity KV Runtime**: build a fixed `[1,Hn,MAX,Hd]` cache + manual append
  kernel + validity mask + logical-extent tracking around DirectML's MHA (the B1-diagnosed path).
  A new runtime *subsystem*, so it starts a new chapter rather than proving another property of the
  existing one. De-risked by #004-B1; optional.

## Parallel semantic track

```
S#001 corpus extraction ✓   S#003a KXML record algebra ✓
   → S#002 live FieldExecutionEngine trajectories (next)  → S#003b schema freeze → S#004 SCXQ2 → training
```
