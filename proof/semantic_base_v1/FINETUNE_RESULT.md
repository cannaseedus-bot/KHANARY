# Finetune proof: transition data trains a correct GPT-2 (loss drops)

## The decisive test
Finetuned the SAME mini base dds_trainer used (gpt2_small_lite_tool, 124M) on 500 transition
examples via tools/finetune_hf.py (HuggingFace GPT2LMHeadModel, full finetune, CPU, AdamW lr=5e-5).

## Result -- IT LEARNS
```
step 10: loss 8.22
step 20: loss 7.34
step 30: loss 5.82
step 40: loss 5.63
step 50: loss 4.93     (8.22 -> 4.93 in 50 steps, ~11s/step CPU 4 threads)
```

## Why this matters
- HF starts at loss ~8 (a REAL GPT-2 forward on this conversational data) and drops steadily.
- dds_trainer (v0.1.1 snapshot) was stuck at ~11 = ln(50257) = the RANDOM baseline, because its
  "GPT-2 domain model" built only 1,807,872 total params -- the 124M frozen base was NOT in the
  forward path. So dds_trainer runs but can never learn; it is not a viable trainer as-is.
- The transition-shaped (Preserve+Delta) data IS learnable. Correct trainer = it learns.

## Feasibility
- Full finetune on CPU in system RAM sidesteps the ~2GB iGPU budget entirely (mini ~2GB,
  medium 355M ~5.7GB of 15.9GB). "CPU overnight" is the right frame; torch CPU is far faster
  than the naive C++ trainer's triple-loop matmul.
- Output is HF-format safetensors -> tools/gpt2_safetensors_to_gguf.py -> khanary-server.
