# GPU Proof #004-B1 — Capacity Semantics (frozen, de-risk)

Branch `resident-generation-amortization-v1`. Split from #004-B per plan: **prove semantics first,
optimize later.** Falsifiable hypothesis:

> A GROWING field whose logical **extent** changes but whose physical **capacity** is fixed can be
> attended over and produce the **same output** as an exact-extent past.

Result: **hypothesis holds for the OUTPUT** — but the *native* fixed-capacity mechanism
(`MHA1` + `PastSequenceLengths`) is **unavailable on this HD 4600**, so the fixed-capacity
*feedback* needs a manual cache (that's #004-B2).

## Three findings (measured, not assumed)

**1. Capacity-output semantics — PASS.** Physical past `[1,Hn,C,Hd]` with `C=6`, logical extent
`P=3` (+ `C−P` garbage slots), masked via `RelativePositionBias = −1e9` on the unused slots →
output vs exact-extent-`P` reference **scale-norm 8.08e-08**; real-past prefix preserved `0.0`.
`extent < capacity` gives the same attention output.

**2. Native `MHA1` / `PastSequenceLengths` — UNAVAILABLE.** Device DML feature level reports
`0x6400` (needs `0x6300`), yet `DML_OPERATOR_MULTIHEAD_ATTENTION1` fails `CreateOperator`
`E_INVALIDARG` across every desc variation tried (present on/off, `PastSequenceLengths` shape
`[1]` / `[1,1,1,1]`, Q/K/V 3-D / 4-D). Nominal feature level ≠ op availability on this 2015 iGPU.

**3. Base MHA does not compact.** With a capacity past, `present = past + 1` and the new token is
appended at the **physical end** (slot `C`), not at the logical extent `P` (confirmed `0.0`). So
base-MHA's own present cannot feed back as a fixed-capacity past — it grows every tick.

## Consequence for #004-B2 (the reusable-binding step)

The native path (MHA1 fixed present) is out. But B2 is still reachable: manage the KV cache in a
**fixed `[1,Hn,MAX,Hd]` buffer per layer**, write the new token's K/V into slot `t` **manually**
(a strided copy), and call **base MHA with `Past = the fixed cache`, a per-tick validity `mask`,
and the growing present ignored/scratch**. All MHA tensor *shapes* are then fixed (Q/K/V `[1,1,E]`,
past `[1,Hn,MAX,Hd]`, mask `[1,Hn,1,MAX+1]`); only the mask *contents* and the cache *contents*
change per tick. That makes the MHA **binding** fixed-shape → reusable (the #004-B2 claim), at the
cost of a manual append op.

This is exactly the diagnosis the split was meant to produce: *DirectML forces present growth
despite fixed capacity — so the reuse must come from a manually-managed capacity cache, not the
native present.*

## KGRC result

`GROWING` now has two independent geometries, experimentally distinguished:

```
PHYSICAL capacity = C   (fixed allocation)
SEMANTIC extent   = P   (valid region, P ≤ C)   — only the filled portion participates in compute
```

A general runtime concept (conversation memory, replay buffers, event logs, semantic graphs):
**allocate capacity once; grow the extent; never reallocate the representation.**

## Oracle unchanged

B1 is a semantics de-risk (no trajectory run). #004-B2 will use the **same frozen #003 oracle** as
#004-A: `trajectory(#004-B2) == trajectory(#003)`, G1–G5 + sequence, KV within tolerance; the only
thing that should change is `dynamic MHA binding creations/token ↓`.

## Reproduce

```
cl /nologo /std:c++17 /EHsc /O2 /I include dml_cap_probe.cpp    /link /LIBPATH:lib /OUT:dml_cap_probe.exe      # availability probe
cl /nologo /std:c++17 /EHsc /O2 /I include dml_mha_cap_test.cpp /link /LIBPATH:lib /OUT:dml_mha_cap_test.exe   # capacity-output verify
dml_cap_probe.exe ; dml_mha_cap_test.exe
```
