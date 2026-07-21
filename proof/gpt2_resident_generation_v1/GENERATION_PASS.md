# KGRC Proof #003 — Resident Generation v1 (frozen)

**Frozen reference.** Composes Proof #001 (resident computation) + Proof #002 (native KV state
transition) into a multi-token autoregressive decode cycle. Its **new** architectural claim is
neither computation nor a single state edge — both are already proven — but **repeated, resident
state continuity through the closed `Xul → Pop` decode cycle**.

## Claim

> A multi-token autoregressive GPT-2 decode cycle executes on the HD 4600 using the frozen
> resident model and the proven native KV state transition, preserving correct token generation
> across successive ticks. Deterministic argmax; no sampling; no ggml.

Every token — the 8 prompt tokens (fed one at a time) and the 6 generated tokens — flows through
the **same single-token decode cycle**, so the KV cache grows exactly one row per tick. Prefill
and generation are the same cycle.

## Result — verified at EVERY tick (14 ticks)

```
tick= 0 fed= 42749 P->present=0->1  GPU=  8812 CPU=  8812 MATCH
tick= 1 fed= 32011 P->present=1->2  GPU= 42447 CPU= 42447 MATCH
tick= 2 fed= 25688 P->present=2->3  GPU=  4864 CPU=  4864 MATCH
tick= 3 fed= 13558 P->present=3->4  GPU= 49205 CPU= 49205 MATCH
tick= 4 fed= 15470 P->present=4->5  GPU= 33770 CPU= 33770 MATCH
...
tick=13 fed= 42447 P->present=13->14 GPU= 42447 CPU= 42447 MATCH

[G2/G4] final per-layer KV cache vs CPU reference (all 12 layers): maxabs 5.36e-06 : OK
gpu generated = 42447 42447 42447 42447 42447 42447
cpu generated = 42447 42447 42447 42447 42447 42447
=== PASS: Resident Generation v1 on HD4600 (14 ticks, KV-cache decode cycle) ===
```

(Full per-tick log: `token_trace.txt`; per-layer final-cache check: `state_trace.txt`.)

## Generation laws (PASS condition — all required)

- **G1 GROWTH** — every layer: `present_seq = past_seq + 1` per tick.  ✔ (`token_trace`, `state_trace`)
- **G2 PRESERVATION under repeated composition** — the final per-layer KV cache equals the CPU
  reference full-sequence K/V for **all 12 layers** (K/V maxabs ≤ 5.4e-06). ✔ (`state_trace`)
- **G3 TOKEN AGREEMENT** — `argmax(GPU_logits_t) == argmax(CPU_logits_t)` at **every** tick,
  including the varied prefill predictions (8812, 4864, 49205, 33770…). ✔
- **G4 STATE CONTINUITY** — `PastKV(t+1) ≡ PresentKV(t)`: the present buffer produced by tick t
  is bound as the past at tick t+1 (structural), and the end-of-run per-layer cache matches the
  reference — no reconstruction/reset boundary appeared between ticks. ✔
- **G5 LAYER COMPLETENESS** — every tick traverses `L0..L11 → ln_f → lm_head` exactly once. ✔
- **SEQUENCE** — GPU generated token sequence == CPU reference sequence. ✔

Numerical logits differ within the understood fp32 / erf-gelu tolerance; **token sequence, cache
trajectory, and state continuity agree exactly.**

## The closed cycle, experimentally

```
Pop   receive token t + state F(t)
Wo    bind STATIC (weights) + GROWING (KV) + TRANSIENT (scratch) fields
Yax   validate residency / cache shape / dependencies
Sek   execute L0..L11 → ln_f → lm_head
Ch'en commit PresentKV = F(t+1), logits   (one GPU sync per tick)
Xul   argmax / termination → Pop(next token)
```

#001 proved the graph. #002 proved a state edge. **#003 proves the cycle** — the closed
`Xul → Pop` edge, and thus the state trajectory `F0 → F1 → … → F13` with
`InputState(t+1) ≡ OutputState(t)` (the new **CONTINUITY** property, orthogonal to
STATIC/GROWING/TRANSIENT). See `generation_contract.json`.

## Honest scope

- Correctness only — **not performance** (per-tick op binding + MHA recompiled per cache length;
  argmax + the single fed-token embedding cross the CPU boundary, but the model STATE — weights +
  KV — stays GPU-resident throughout).
- Random-init gpt2 weights (the model repeats a token); the point is GPU↔CPU trajectory agreement,
  not text quality.
- erf-gelu (matches DirectML); deterministic argmax (sampling is a later proof).
- Next: **#004 performance/amortization**, then **#005 driver integration**, **#006 ggml-xcfe**.

## Reproduce

```
# from scratch/dml/ (needs the DirectML redist — see scratch/dml/README.md):
python  ../../scratch/dml/gen_prep.py
cl /nologo /std:c++17 /EHsc /O2 /I include resident_generate.cpp /link /LIBPATH:lib /OUT:resident_generate.exe
resident_generate.exe
```

Frozen snapshot here; evolving source at `scratch/dml/resident_generate.cpp`.
