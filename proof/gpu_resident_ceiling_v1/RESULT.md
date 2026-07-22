# KGRC Memory-Ceiling Probe — Resident allocation ceiling (HD 4600, frozen)

**Frozen reference.** Companion to Proof #001 (which put ~500 MB of gpt2 weights resident). This
measures the *wall*: how many bytes actually stay **resident** on this UMA iGPU, via #001's exact
residency path — D3D12 DEFAULT-heap **committed** buffers + explicit `ID3D12Device::MakeResident`,
grown 256 MB at a time until failure, with a free-RAM safety guard.

## Device (measured)

```
Intel(R) HD Graphics 4600   (UMA — one physical DDR3 pool shared with the CPU)
DedicatedVideoMemory 112 MB   DedicatedSystemMemory 0 MB   SharedSystemMemory 2048 MB
DXGI LOCAL budget 2048 MB   (start usage 0 MB)   system RAM 16 GB
```

## Result

```
step  cumulative   LOCAL.usage   status
   1     256 MB       257 MB      resident
   ...
   7    1792 MB      1793 MB      resident            <- last stable
   8    (2048 MB)     —           CreateCommittedResource hr=0x887A0005  (DXGI_ERROR_DEVICE_REMOVED)
```

- **Hard resident ceiling ≈ 1.75 GB** (last stable = 1792 MB), **hard wall at the 2048 MB LOCAL
  budget**.
- Crossing the budget is **not graceful**: the 8th 256 MB commit — which would put usage just past
  2048 MB — returns **`DXGI_ERROR_DEVICE_REMOVED` (0x887A0005)**, i.e. the 2015 WDDM 2.0 driver
  **resets the device** rather than returning `E_OUTOFMEMORY`.
- This is **not** a system-RAM limit: **2454 MB of physical RAM was still free** at the wall. The
  ceiling is the GPU **budget**, not the 16 GB box.

## The load-bearing conclusion

The "112 MB dedicated" is a UMA cosmetic; the real resident budget is the **2048 MB shared pool**,
and the usable, stay-resident-without-a-device-reset ceiling is **~1.75 GB**. There is **no graceful
paging** past it on this driver — plan to stay under ~1.75 GB or stream, never to overcommit.

## Consequence for models (FP16 = 2 B/param)

| model | FP16 | INT8 | Q4 | verdict |
|---|---|---|---|---|
| gpt2 124M | 0.25 GB | — | — | resident (proven, #001 used ~0.5 GB FP32) |
| **Qwen-1.8B** (1.837 B, 195×F16) | **3.42 GB** ✗ | **1.71 GB** ✓(edge) | **0.86 GB** ✓ | FP16 too big; **INT8 fits at the edge, Q4 fits comfortably**; else stream/tile |

Qwen-1.8B FP16 (3.42 GB) is **1.9× over the wall** — it cannot be fully resident. INT8 (1.71 GB)
lands just under the ~1.75 GB stable ceiling (little headroom for activations/KV — risky); **Q4
(0.86 GB) is the safe resident target**. Anything that must stay FP16, or larger models, needs the
**DDS-tile / SCXQ2-fold streaming** path (working set ≤ ~1.75 GB), not full residence.

## Reproduce
```
# from scratch/dml/ (needs the Windows SDK; no DirectML dependency):
cl /nologo /std:c++17 /EHsc /O2 mem_ceiling_probe.cpp /link /OUT:mem_ceiling_probe.exe
mem_ceiling_probe.exe
```
