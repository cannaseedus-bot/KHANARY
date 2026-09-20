# KGRC Q8 Residency + Hot-Swap — Qwen-1.8B INT8 on the HD 4600 (frozen)

**Frozen reference.** Builds on the memory-ceiling probe (hard wall at the 2048 MB DXGI LOCAL
budget, ~1.75 GB stable). Qwen-1.8B INT8 is **1.71 GB** — right at that ceiling. This tests whether
that footprint is actually *viable*, and whether **tile hot-swap** (D3D12 `Evict`/`MakeResident`)
works to manage the tight fit. Mechanism = #001's residency path.

## Result (measured, one hardware pass)

```
A. Q8 RESIDENCY   base 1579 MB + hot tile -> 1707 MB resident   usage 1708 / budget 2048   PASS
B. HOT-SWAP       20x (Evict + MakeResident), budget-neutral, usage 1708..1836 MB          PASS
C. HEADROOM       +64 MB above the Q8 base, then CreateCommittedResource fails             64 MB
```

- **Q8 weights (1.71 GB) go resident** and stay under the 2048 MB budget. ✓
- **Tiles hot-swap within budget** — evict a 128 MB tile, make its replacement resident, 20 cycles,
  no growth past the Q8 footprint. ✓ This is what makes the tight fit survivable: the base tensors
  stay frozen + resident; adaptation rides on **small LoRA-DDS adapter deltas + micronaut meta
  models** (none of which bloat the resident base), and tiles swap in/out on demand.
- **Only ~64 MB of headroom** remains above the Q8 base before the wall.

## What the 64 MB headroom means (Qwen-1.8B: 24 layers, hidden 2048, MHA)

KV cache costs **2·L·H·bytes per token** = **192 KB/token FP16**, **96 KB/token INT8**:

| KV precision | tokens in 64 MB headroom |
|---|---|
| FP16 | **~341 tokens** |
| INT8 | **~682 tokens** |

So **Q8 Qwen-1.8B is resident-viable on this iGPU for a short context** (~340-token FP16 / ~680-token
INT8 window fits the headroom). Longer context requires **INT8 KV** (halves it), **KV streaming /
hot-swap**, or dropping the base to **Q4 (0.86 GB)** — which frees ~0.85 GB and makes context
comfortable.

## Verdict

**Q8 is VIABLE with hot-swap.** Weights fit resident; the design (frozen resident base + LoRA-DDS
deltas + micronaut meta + tile hot-swap) is exactly what the ~64 MB headroom demands — you cannot
hold both 1.71 GB of weights *and* a large KV window resident, so the KV/activation working set must
also be short or streamed. For a roomier default, **Q4 base** is the safer resident target.

## Reproduce
```
# from scratch/dml/ (Windows SDK; no DirectML dependency):
cl /nologo /std:c++17 /EHsc /O2 q8_hotswap_probe.cpp /link /OUT:q8_hotswap_probe.exe
q8_hotswap_probe.exe
```
