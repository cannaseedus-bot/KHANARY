# DISTILLATION.md — GPT-OSS Teacher → from_zero LoRA Distillation

> Covers: tensor layers, LoRA implementation, training loop, repair tools, SLERP merge.
> Tool: `tools/oss_distillation.py`
> See also: `GPU.md § Tensor layers`, `SCXQ2.md`, `SCX.md`

---

## Tensor layers — what touches what

**Critical**: these four layers are separate. Never conflate them.

```
STORAGE (disk)                  RUNTIME (Python train loop)      GPU COMPUTE (C++ runtime)
──────────────                  ───────────────────────────      ─────────────────────────
.safetensors (HuggingFace)  →   PyTorch float tensors        →   DirectML D3D12 heap
.safetensors (LoRA output)  ←   PyTorch A, B adapter params      (dml_gemm.dll — inference)
                                (no PEFT, no HF transformers)
                                        │
                                XJSON tensor                 ←   json_runtime / SCXQ2 ops
                                (in-flight, json_runtime)        ({"shape":[],"data":[]})
                                        │
                                SCXQDDS INT8+CRC             ←   kuhul_fold_compute.hlsl
                                (expert shard format)             (SRV slots, no XJSON)
```

**PyTorch = Python tooling only.** It never enters the C++ runtime stack.
**SCXQ2 Tensor ops = compute primitives** that operate on XJSON tensors, not PyTorch tensors.
**SafeTensors = the disk format** for both weight checkpoints and LoRA adapters.

---

## Distillation strategy

**Response distillation** — student learns to mimic teacher completions:

```
1. Send prompt to GPT-OSS teacher  → get completion text
2. Tokenize (prompt + completion)  → token sequence
3. Run from_zero student forward   → logits
4. Cross-entropy loss on completion tokens only (prompt tokens masked)
5. Backprop through LoRA A, B only  (base weights W frozen)
6. AdamW step on A, B params
```

No KL divergence, no intermediate layer matching — pure response distillation on token cross-entropy.

---

## Teacher

```
Model:    gpt-oss-20b-MXFP4.gguf  (served by kuhul_engine)
Endpoint: http://127.0.0.1:17474/v1/chat/completions
Protocol: OpenAI-compatible JSON, stream=False
Timeout:  30s per call
Max tokens per call: 200 (--teacher-tokens)
Fallback: if engine unreachable → self-distillation (student teaches itself)
           useful for adapter shape validation without kuhul_engine running
```

Engine check at startup: `teacher_complete(engine, 'ping', max_tokens=4)`. If `None` → switch to self-distillation mode for the full run.

---

## Student model

```
File:        models/from_zero/from_zero_v0.6_merged.safetensors
Format:      SafeTensors, F32 weights
Auto-detect: n_layer, n_embd, n_head, vocab_size from wte shape + key scan
Config:      GPT-2 small  6L/6H/768E/50270V  or  medium 12L/12H
Forward:     Pure PyTorch — LayerNorm, self-attention, MLP (GELU approx: x·σ(1.702x))
             No KV-cache. Causal mask. Tied lm_head = wte.weight.T
LoRA:        W_eff(key) = W_frozen(key) + B @ A  applied inside forward via `get(key)`
```

---

## LoRA implementation

No PEFT. No HuggingFace transformers. Implemented from scratch in ~15 lines:

```python
# Adapter factory
A = nn.Parameter(torch.randn(rank, in_dim) * (0.02 / rank))   # init: small normal
B = nn.Parameter(torch.zeros(out_dim, rank))                    # init: zeros → W_delta=0 at step 0

# Effective weight at runtime
W_eff = W_frozen + B @ A    # alpha=r gives unit scaling (no separate alpha term)

# Gradient flow
# Only A, B are in the optimizer. W_frozen is never touched.
```

**LoRA targets** (per layer, 4 projections × n_layer):
- `transformer.h.{i}.attn.c_attn.weight`  — QKV projection [E, 3E]
- `transformer.h.{i}.attn.c_proj.weight`  — output projection [E, E]
- `transformer.h.{i}.mlp.c_fc.weight`     — MLP up [E, 4E]
- `transformer.h.{i}.mlp.c_proj.weight`   — MLP down [4E, E]

Embeddings and layernorm weights are NOT adapted.

**Optimizer**: AdamW, lr=1e-4, weight_decay=0.01, grad clip=1.0

---

## Training loop

```
for step in range(args.steps):
    prompt     = prompts[step % len(prompts)]
    completion = teacher_complete(engine, prompt)   # or self if fallback
    full_text  = prompt + '\n' + completion
    token_ids  = tokenize(full_text)[:block_size]

    input_ids  = token_ids[:-1]
    target_ids = token_ids[1:]
    mask_start = len(tokenize(prompt)) - 1   # skip prompt tokens in loss

    logits = gpt2_forward(state, adapters, input_ids, cfg)
    loss   = cross_entropy(logits[mask_start:], target_ids[mask_start:])

    loss.backward()
    clip_grad_norm_(params, 1.0)
    optim.step()

print every 10 steps; track best_loss
```

**Tokenizer**: tiktoken `gpt2` encoding. Fallback: char-level (`ord(c) % 50270`) — shape validation only, not real GPT-2 tokens.

---

## Default prompts (built-in, kuhul domain)

15 prompts cycling through kuhul concepts:

```
"Explain the K'UHUL fold system and how it organizes semantic execution phases."
"Describe the role of the BOSS orchestrator in the WebX compute engine."
"How does the micronaut factory create sampling-profile micronauts?"
"What is the difference between a fold micronaut and a chat micronaut?"
"Explain the π-nary arc weighting in the KuhulPhysics gradient controller."
"How does the DirectML GEMM bridge accelerate matrix operations on Intel HD 4600?"
"What is SCXQ2 IR and how does it represent the K'UHUL execution graph?"
"Describe the KUHUL APPS app generation studio and its kuhul_engine backend."
"How do semantic micronauts influence llama-server sampling parameters?"
"Explain the SLERP merge strategy for combining two training phase checkpoints."
"What are atomic blocks in the K'UHUL semantic field system?"
"How does the grammar validator use kuhul.ebnf for constrained generation?"
"Describe the LoRA distillation pipeline from GPT-OSS to from_zero."
"What is the role of the MicrosoftSDK.ps1 in the NNC-K stack?"
"How does the Phase Pop-Wo-Sek-Chen-Xul loop work in K'UHUL execution?"
```

Override with `--prompts tools/distill_prompts.txt` (one prompt per line).

---

## CLI usage

```powershell
python tools/oss_distillation.py `
  --student  models/from_zero/from_zero_v0.6_merged.safetensors `
  --out      models/from_zero/from_zero_v0.6_lora.safetensors `
  --rank     8 `
  --steps    500 `
  --lr       1e-4 `
  --engine   http://127.0.0.1:17474 `
  --prompts  tools/distill_prompts.txt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--student` | `models/from_zero/from_zero_v0.6_merged.safetensors` | Student checkpoint |
| `--out` | `models/from_zero/from_zero_v0.6_lora.safetensors` | LoRA output file |
| `--rank` | 8 | LoRA rank r |
| `--steps` | 500 | Training steps |
| `--lr` | 1e-4 | Learning rate |
| `--engine` | `http://127.0.0.1:17474` | kuhul_engine base URL |
| `--prompts` | (built-in) | Path to prompts file |
| `--teacher-tokens` | 200 | Max completion tokens per teacher call |

---

## LoRA output format

Keys in `from_zero_v0.6_lora.safetensors`:

```
lora_A__{layer_key_with_dots_replaced_by_double_underscore}
lora_B__{layer_key_with_dots_replaced_by_double_underscore}
```

Example:
```
lora_A__transformer__h__0__attn__c_attn__weight
lora_B__transformer__h__0__attn__c_attn__weight
```

All tensors saved as F32. Total count: 4 projections × n_layer × 2 (A+B).
For 6-layer model: 48 tensors.

---

## repair_safetensors.py

`tools/repair_safetensors.py` — fixes checkpoints where the trainer wrote empty shapes `"shape":[]`.

```powershell
python tools/repair_safetensors.py `
  --ref     models/from_zero/from_zero_v0.1_folded.safetensors `
  --targets models/from_zero/from_zero_v0.4_phase1.safetensors `
             models/from_zero/from_zero_v0.5_phase2.safetensors
```

Strategy: reads correct shape map from `--ref` (a valid checkpoint of the same architecture), then for each target:
1. Parse raw SafeTensors header (offset 0: 8-byte LE length, then JSON header)
2. For each tensor key: look up shape from ref, read data bytes from target, reshape
3. If shape numel ≠ data numel → fall back to flat `[n_elements]`
4. Save as `{original}.repaired.safetensors`
5. Validate: `safe_open` the output, print key count + first 3 keys

Supported dtypes: F32, F16, BF16 (via uint16 view), I32, I64, U8.

Files repaired: `v0.4_phase1.repaired.safetensors`, `v0.5_phase2.repaired.safetensors`.

**Root cause**: gpt2_trainer's save path writes non-embedding tensors with `shape=[]`. Fix the trainer save code to write proper shapes to prevent recurrence.

---

## SLERP merge (Phase 3)

`tools/merge_models.py` — SLERP or linear merge of two same-arch SafeTensors checkpoints.

```powershell
python tools/merge_models.py `
  models/from_zero/from_zero_v0.4_phase1.safetensors `
  models/from_zero/from_zero_v0.5_phase2.safetensors `
  models/from_zero/from_zero_v0.6_merged.safetensors `
  --alpha 0.6 --method slerp
```

- `--alpha 0.0` = pure A (v0.4 general), `--alpha 1.0` = pure B (v0.5 KUHUL)
- `--alpha 0.6` recommended: keeps general language fluency while biasing toward KUHUL fold patterns
- Vocab mismatch: shared rows interpolated, extra KUHUL rows from B appended verbatim
- SLERP respects the vacuum-shaped manifold geometry; `--method linear` also available
- Prints weight-norm sanity table after saving

**Do NOT chain in earlier checkpoints (v0.1, v0.2, v0.3).** Those are intermediate stages that Phase 1 already subsumed.

Result: `from_zero_v0.6_merged.safetensors` — 148 tensors, DONE 2026-08-04.

---

## Training curriculum status

| Phase | Data | Steps | LR | Output | Status |
|-------|------|-------|----|--------|--------|
| 0a — vacuum | `vacuum_seed.bin` (50K×64) | 150 | 1e-3 | `v0.2_vacuum` | DONE — loss 0.00322 |
| 0b — vacuum+LBS | same | 200 | 5e-4 | `v0.3_vacuum_bias` | DONE — loss 0.00066 |
| 1 — header corpus | `tokens_hdr_big.bin` (200K×64) | 2000 | 3e-4 | `v0.4_phase1` | DONE 2026-08-04 |
| 2 — KUHUL corpus | `kuhul_tokens_kuhul.bin` (462 MB) | 3000 | 1e-4 | `v0.5_phase2` | DONE 2026-08-04 |
| 3 — SLERP merge | v0.4 + v0.5 | — | — | `v0.6_merged` | DONE 2026-08-04, α=0.6 |
| 4 — distillation | GPT-OSS teacher → LoRA | 500 | 1e-4 | `v0.6_lora.safetensors` | **pending** |

---

## File locations

| File | Role |
|------|------|
| `tools/oss_distillation.py` | LoRA distillation script |
| `tools/repair_safetensors.py` | Fix empty-shape SafeTensors |
| `tools/merge_models.py` | SLERP / linear merge |
| `tools/distill_prompts.txt` | Custom prompt file (optional) |
| `models/from_zero/from_zero_v0.6_merged.safetensors` | Student base weights |
| `models/from_zero/from_zero_v0.6_lora.safetensors` | LoRA adapter output (pending) |

All paths relative to `C:\Users\canna\_khanary_inspect\`.
