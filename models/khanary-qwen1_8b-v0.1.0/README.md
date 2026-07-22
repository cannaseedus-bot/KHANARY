# KHΛNARY Qwen-1.8B dual-quant — v0.1.0 (runtime copy)

One Qwen-1.8B-Chat, quantized two ways from the **same tensors** for the HD 4600 resident-tier
design. This is the **runtime copy on C:**; the build + archive of record is
`E:\models\Qwen1.8B-quant\` (backup drive — no runtime access / speed-limited, so weights are
built + archived there and copied here to run).

| tier | file | size | fidelity | role |
|---|---|---|---|---|
| **Q4 base** | `qwen1_8b.q4.kqz` | 0.909 GiB | ~10% normRMSE / 20 dB | resident base (fast, long ctx) — ~0.9 GiB headroom |
| **Q8 tier** | `qwen1_8b.q8.kqz` | 1.712 GiB | ~0.85% / 41 dB (near-lossless) | per-tensor escalation into headroom (deep-think/validate) |

Each `.kqz` has a JSON offset-manifest (`*.manifest.json`, **tracked in git**; the `.kqz` payloads
are gitignored). The two manifests are **tensor-aligned** (identical 195 names/order), so the
runtime keeps Q4 resident and hot-swaps *individual* tensors up to Q8 — never both full models
(2.6 GiB > the 2.0 GiB budget wall).

## Storage model
- **`E:\models\Qwen1.8B-quant\`** — build + archive of record (cold). Keep building/archiving here.
- **this folder (C:)** — runtime copy (hot). Re-install from the archive anytime:
  ```
  python tools/build_qwen_model.py          # copies payloads + manifests from E:, verifies sha256, writes MODEL.json
  python tools/build_qwen_model.py --no-payloads   # metadata only (payloads already present)
  ```

## Honest scope
- **Weights only, not a runnable model** — no Qwen forward path on this stack (the #001 DirectML
  driver is GPT-2-only; vendored llama is CPU). See `MODEL.json` → `honest_scope`.
- **"Q8 fits resident" is a property of this lean per-channel scheme** (1.712 GiB < 1.75 GiB stable
  ceiling); GGUF Q8_0 (~1.82 GiB) would exceed it.
- Fidelity = dequant-error + container round-trip (`tools/verify_quant.py`), not perplexity.

Provenance + fidelity frozen at `proof/qwen_quant_v1/`. Ceiling + hot-swap evidence at
`proof/gpu_resident_ceiling_v1/` and `proof/gpu_q8_hotswap_v1/`.
