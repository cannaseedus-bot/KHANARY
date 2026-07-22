# K'UHUL-3D LAW R1 — Executed MATMUL tick (frozen)

**LAW R1 as executed silicon, not syntax.** A single MATMUL compute node whose recursive PhaseTick
(`matmul_node.ast.json`) *drives real work* on the real Qwen weight
`transformer.h.0.attn.c_proj.weight [2048,2048]` — one shared D3D12 + DirectML device across the whole
tick. Proves `Node → PhaseTick → PhaseStep → Node` controls **residency, quantization, dispatch,
compute, and eviction**, not just recursive structure.

## The tick (each lane did real work — LAW P1: phases schedule, opcodes work)

```
[Pop]  LOAD    Q4 bytes -> D3D12 committed + MakeResident (2.1 MB); usage 5 MB
       ESCALATE per-tensor Q4 -> Q8 MakeResident (+4.0 MB); usage 11 MB
[Wo]   BIND    dequant resident Wq4 / Wq8 + F16 truth -> f32 [2048,2048]; input A[8,2048]
[Yax]  TILE    select DirectML GEMM, transB=1 (K=2048 N=2048)
[Sek]  GEMM    NESTED node/tick -> real DirectML dispatch: Cq4, Cq8 (+ Ctruth via F16 CPU ref)
[Ch'en] VERIFY normRMSE(Q4 vs truth)=0.1100  normRMSE(Q8 vs truth)=0.0115  thr=0.020
               -> admit Q8 (escalation JUSTIFIED: 9.6x lower error)
[Xul]  COMMIT  retain Q4, real D3D12 Evict of Q8 (freed 4.0 MB)
```

## Why the numbers are trustworthy

The GEMM output error tracks the *weight* quant fidelity measured independently in
`proof/qwen_quant_v1` (Q8 ≈ 42 dB, Q4 ≈ 20 dB → ~10× ratio). Here the DirectML GEMM propagates that:
Q8 output `normRMSE 0.0115` vs Q4 `0.1100` = **9.6×** — consistent, so the escalation decision is a
real fidelity gate, not a staged number. Device: Intel HD 4600, D3D12 FL 11_1, DirectML.

## What this closes

- **LAW R1 executes.** The recursion is the controller: the LOAD lane made real bytes resident, the
  Sek lane dispatched real compute, the VERIFY lane gated on real fidelity, the COMMIT lane evicted.
- **The dual-quant design runs on real data.** Per-tensor Q4→Q8 escalation + retain-Q4/evict-Q8 is
  executed against the actual Qwen artifacts (`models/khanary-qwen1_8b-v0.1.0`).

## Honest scope

One tensor, one GEMM (not a full forward). Dequant is CPU (the compact quant bytes are what stays
*resident*; a GPU dequant kernel is a later optimization). F16 "truth" is the original weight; error
is dequant/GEMM fidelity, not end-to-end perplexity.

## Reproduce
```
# build (DirectML redist in scratch/dml/{include,lib}):
cl /nologo /std:c++17 /EHsc /O2 /I include scratch/dml/kuhul_matmul_tick.cpp /link /LIBPATH:lib /OUT:kuhul_matmul_tick.exe
python tools/run_kuhul_matmul_tick.py        # resolves offsets from the manifests + safetensors, runs the tick
```
