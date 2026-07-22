# KHΛNARY GPU-resident proof ladder — v0.1.0

**The K'UHUL GPU Resource Contract (KGRC), earned one property at a time.** Where the gpt2 and
geometry packages certify individual glyph *kernels* bit-close on the HD 4600, this package
certifies that a whole model *lives* on the GPU as persistent state and that its residency,
state transitions, trajectory, and execution-state reuse are all **correct** — each rung a single
architectural claim, hardware-verified against a CPU reference on real gpt2 weights.

Target: **Intel HD Graphics 4600**, D3D12 feature level **11_1**, DirectML on D3D12 as an 11_x
bridge (full D3D12 12_0+ is unsupported on this 2015 iGPU).

## The ladder (frozen)

| # | Claim | Result |
|---|---|---|
| **#001** | Resident computation is correct | whole gpt2 (124M/12L) forward, weights resident; logits scale-norm **1.92e-06**; argmax gpu==cpu |
| **#002** | Resident state transition is correct | native KV decode: growth / preserve(0) / append(0) / compute **8.08e-08** — all four PASS |
| **#003** | Resident trajectory composes | 14-tick autoregressive decode, every tick MATCH; final KV **5.36e-06**; deterministic argmax |
| **#004-A** | Fixed execution state is reusable | binding creations **146→12/token**; record/bind **3.30×**; trajectory == frozen #003 |
| **#004-B1** | Capacity ≠ extent (+ backend conformance) | extent<capacity → same output **8.08e-08**; native fixed-capacity MHA1 **unavailable** on this DirectML |

## What this is (and isn't)

- **Is:** a proof-ladder capability release — verified residency/state/trajectory/amortization
  *properties*, with the frozen evidence (PASS/RESULT docs, contracts, per-rung `SHA256SUMS`)
  vendored under `proof/`.
- **Isn't:** a packaged runtime or a single dispatchable glyph. `#004-B1` is a **conformance**
  result — DirectML on this rig appends at the physical end and lacks the native fixed-capacity op,
  so a resident fixed-capacity KV runtime (chapter **#005**) is *parked, not built*.

All numbers are on this specific HD 4600 (frozen 2015 driver, WDDM 2.0); `#004-A` timings are
host-side. See `MODEL.json` → `honest_scope` and `docs/GPU_PROOF_LADDER.md`.

## Reproduce
```
python tools/build_gpu_resident_model.py     # re-vendors the frozen evidence + emits MODEL.json
# the proofs themselves are frozen references under proof/ (each with its own SHA256SUMS + harness).
```
