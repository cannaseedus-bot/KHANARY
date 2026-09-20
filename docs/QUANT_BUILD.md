# Building KHANARY dual-quant model artifacts (portable)

Turn any HuggingFace / safetensors model into the two resident-tier quant artifacts
(**Q4 base + Q8 escalation tier**) the HD 4600 residency design uses — reproducibly, on any machine.

## Why cloud and "just give instructions" are the same thing

`tools/quantize_safetensors.py` is **deterministic**: the same source produces **byte-identical**
`.kqz` payloads, so the **sha256 in the manifest is the single verification gate** for both paths.

- **Build in the cloud** (e.g. Google Cloud for a custom model) → run the quantizer → check the sha.
- **Ship build instructions** → someone runs the same command → same sha.

Either way you land the exact same bytes. Only the small manifests + this recipe need to travel; the
multi-GB `.kqz` never has to.

## Recipe

```bash
# 1. get weights (any single-file or sharded safetensors model)
#    e.g. from HuggingFace: Qwen/Qwen1.5-1.8B-Chat, google/gemma-2-2b, etc.
#    -> a model.safetensors file, or a dir with *.safetensors + model.safetensors.index.json

# 2. quantize (Q8 + Q4) — points anywhere via --src/--out; defaults reproduce this rig's Qwen-1.8B
python tools/quantize_safetensors.py --src <model.safetensors | HF model dir> \
       --out <archive dir> --name <basename>
#   options: --schemes q8,q4   --group 64 (Q4 group size)

# 3. verify container integrity + dequant fidelity (reloads from the written bytes, not the math)
python tools/verify_quant.py --src <same src> --out <same out> --name <basename>

# 4. install into a runtime version folder (fast/hot drive), re-checking sha256
#    (build_qwen_model.py is the reference installer; adapt --name/paths per model)
```

Supported source formats: **single-file** `model.safetensors` and **sharded**
(`*.safetensors` + `model.safetensors.index.json`); tensor dtypes **F16 / BF16 / F32**.

## The schemes

| scheme | layout | ~bytes/param | fidelity (measured on Qwen-1.8B) |
|---|---|---|---|
| **Q8** | per-output-channel symmetric INT8 + F16 scales | 1.0 | ~0.85% normRMSE / ~41 dB (near-lossless) |
| **Q4** | group-64 symmetric 4-bit (2 nibbles/byte) + F16 scales | 0.53 | ~10% normRMSE / ~20 dB (usable, lossy) |

1D tensors (bias / norm gains) are kept F16. Each `.kqz` has a JSON offset-manifest so a loader can
**mmap + MakeResident per tensor**; the Q8 and Q4 manifests are **tensor-aligned** (same names/order)
so a runtime can keep Q4 resident and escalate individual tensors to Q8 on demand.

## This scales *up* with hardware — the DDS angle

The residency ceiling is **per-GPU** (measured here: ~1.75 GiB on the HD 4600 —
`proof/gpu_resident_ceiling_v1`). The same quant + offset-manifest + residency approach scales:

- **Bigger card** → higher ceiling → hold a **bigger resident base** (larger model, or Q8 as the
  base, or more of it resident) — fewer swaps.
- **Smaller card / bigger model** → the base won't fit → **DDS-tile streaming**: keep a working set
  under the ceiling and stream tiles (the `smgm-16 scxq2_dds_folds` mechanism).

Same artifacts, same manifests — only the resident-vs-streamed boundary moves. That portability is
what makes the tile/DDS design general rather than a one-rig hack.

## Honest scope

- These artifacts are **weights**, not a runnable model — a forward runtime is a separate component
  (this stack's proven GPU driver is GPT-2-only; the vendored llama is CPU).
- Quant **fidelity is verified at dequant-error + container round-trip**, not end-to-end perplexity.
- **"Q8 fits resident"** on the HD 4600 is a property of this **lean per-channel scheme** (Qwen-1.8B
  Q8 = 1.712 GiB < the 1.75 GiB ceiling). Standard GGUF Q8_0 (~1.82 GiB) would not; other models
  scale by their own param count.

## Reproduce this rig's reference artifact
```bash
python tools/quantize_safetensors.py      # defaults -> E:\models\Qwen1.8B-quant\
python tools/verify_quant.py
# gate: sha256 must equal
#   q8  c81153cd798bcf9b4a4a364fdba7dd9e163256459a3b0686689d44156ae1b6c8
#   q4  bcf1523dd47cdcf85b89296e516b3231506bcb8b7ab83a4e2f831033165f2d3b
```
