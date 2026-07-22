# Qwen-1.8B Q4 + Q8 quant artifacts — provenance (frozen)

Real quantized weight bytes for the dual-quant hot-swap design, produced + verified this session.
The payloads live off-repo at `E:\models\Qwen1.8B-quant\` (2.6 GiB total); this records how they
were made, their fidelity, and the honest boundary.

## Source
```
C:\Users\canna\.lmstudio\models\Qwen-1_8B-Chat-f16\model.safetensors
FP16, 3,673,678,184 bytes, 1.837 B params, 195 tensors (24L, hidden 2048, MHA, vocab 151936)
```

## Artifacts (generator: tools/quantize_qwen.py)
| file | scheme | size | sha256 |
|---|---|---|---|
| qwen1_8b.q8.kqz | per-output-channel symmetric INT8 | 1.712 GiB | c81153cd798bcf9b4a4a364fdba7dd9e163256459a3b0686689d44156ae1b6c8 |
| qwen1_8b.q4.kqz | group-64 symmetric 4-bit | 0.909 GiB | bcf1523dd47cdcf85b89296e516b3231506bcb8b7ab83a4e2f831033165f2d3b |

1D tensors (bias / RMSNorm) kept F16. Each `.kqz` + a JSON offset-manifest (smgm-16 DDS-fold style),
mmap + MakeResident per tensor.

## Verification (tools/verify_quant.py — container + fidelity)
```
[q8] 1.712 GiB  filesize==manifest OK  195 tensors present+ordered OK
[q4] 0.909 GiB  filesize==manifest OK  195 tensors present+ordered OK
[align] Q8 and Q4 tensor-aligned (identical 195 names/order) -> per-tensor Q4->Q8 swap addressable
```
Fidelity (normalized RMSE / cosine / SNR, representative tensors):
```
        attn.c_attn        mlp.w1           lm_head          wte(embed)
  Q8   0.86% .99996 41dB  0.82% .99995 42  0.95% .99998 41  0.83% .99998 42   <- near-lossless
  Q4   10.0% .99506 20dB  10.0% .99506 20  10.2% .99458 20  10.0% .99498 20   <- usable, lossy
```
The Q8/Q4 fidelity gap (41 dB vs 20 dB) is the rationale for the design: Q4 for fast bulk work,
escalate specific tensors to Q8 when precision matters.

## How they swap (design of record)
Q8 (1.712 GiB) fits under the measured 1.75 GiB stable resident ceiling; Q4 (0.909 GiB) leaves
~0.9 GiB headroom. Since both are the SAME tensors in the SAME order, keep **Q4 resident** and
hot-swap **individual tensors up to Q8** into the headroom (per-tensor precision escalation) — never
hold both full models (2.6 GiB > the ~2.0 GiB budget wall). See proof/gpu_resident_ceiling_v1 +
proof/gpu_q8_hotswap_v1.

## Honest scope
- **Weights only, not a runnable model** — no Qwen forward path on this stack (#001 DirectML driver
  is GPT-2-only; vendored llama is CPU). The next rung would be a Qwen forward graph or pointing the
  swap probe at these real files.
- **"Q8 fits resident" is a property of THIS lean per-channel scheme** (1.712 GiB). Standard GGUF
  Q8_0 (~1.82 GiB) would exceed the ceiling.
- Fidelity verified at dequant-error + container round-trip, **not** end-to-end perplexity.

## Reproduce
```
python tools/quantize_qwen.py     # -> E:\models\Qwen1.8B-quant\*.kqz + *.manifest.json
python tools/verify_quant.py      # container integrity + dequant fidelity
```
