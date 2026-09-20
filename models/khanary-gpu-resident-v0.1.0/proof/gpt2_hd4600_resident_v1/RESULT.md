# KGRC Proof #001 — Resident whole-model GPT-2 forward (frozen)

**Frozen reference.** Do not optimize this artifact in place — branch from it.

A complete GPT-2 (124M, 12 layers) forward pass runs **entirely on an Intel HD 4600 iGPU** with
the model's weights **resident on the GPU** (uploaded once, keyed by pointer) and activations
kept on-device across the whole forward. This is the architectural boundary: the model is no
longer a collection of kernels that happen to run — it is **persistent GPU state**.

## Result

```
[dev] Intel(R) HD Graphics 4600  WHOLE MODEL S=8 L=12 E=768 Hn=12 V=50257 (weights resident, on-device forward)
[verify] logits[8,50257]  max abs 4.888e-06  scale 2.54  scale-norm 1.92e-06
[verify] next-token argmax  gpu=42447  cpu(erf)=42447  MATCH
=== PASS: WHOLE gpt2 model on HD4600 (weights resident, on-device) vs CPU driver (erf-gelu) ===
```

- **Logits match** the CPU driver's erf-gelu forward at scale-norm **1.92e-06**.
- **Next-token argmax MATCHES** the driver (both `42447`).
- The DirectML erf-gelu vs the driver's tanh-gelu differs by **2.51e-04** in logits over 12
  layers but **predicts the same token** — the gelu-approximation choice does not change output.

## What is resident (the GPU working set)

- **MODEL**: embeddings (precomputed on CPU, uploaded), 12× layer weights, `ln_f`, `lm_head`.
- **EXEC**: 8 distinct DML operators compiled once (LN, GEMM E→E / E→4E / 4E→E / E→vocab, GELU,
  ADD1, MHA), reused across layers; per-layer binding tables point at that layer's resident
  weights.
- **ACTIVE**: hidden states per layer boundary + shared Q/K/V/attn/scratch buffers, on-device.

## Reproduce

```
# from scratch/dml/ (needs the DirectML redist — see scratch/dml/README.md):
python  ../../scratch/dml/model_prep.py     # dumps real gpt2 weights + embedding + refs
cl /nologo /std:c++17 /EHsc /O2 /I include dml_model_run.cpp /link /LIBPATH:lib /OUT:dml_model_run.exe
dml_model_run.exe
```

The canonical, evolving source lives at `scratch/dml/dml_model_run.cpp`; this directory is the
frozen snapshot (`SHA256SUMS`). See `model_contract.json` for the KGRC resource semantics this
proof satisfies.

## Roadmap (this is Stage A)

- **A. DONE** — resident whole-model forward (this proof).
- **B. NEXT** — resident autoregressive generation loop (Proof #002).
- **C.** — resident KV cache + incremental attention (Proof #003).
- **D.** — formalize KGRC (affinity / residence / compute+copy lease / budget / critical section).
- **E.** — XCFE resource arbitration across workloads (Proof #004).
- **F.** — driver integration, then ggml-xcfe integration.
