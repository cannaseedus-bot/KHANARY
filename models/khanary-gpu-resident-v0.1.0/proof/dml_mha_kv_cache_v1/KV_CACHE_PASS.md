# KGRC Proof #002 — Native KV-cache transition (frozen)

**Frozen reference.** The first experimentally proven **append-growing rectilinear tensor field**
in KHΛNARY. Proves resident *mutable session state* evolves correctly — distinct from Proof #001
(resident *computation* correct).

```
Target:   Intel HD 4600
Backend:  DirectML native MHA (DML_OPERATOR_MULTIHEAD_ATTENTION, KV-cache path)

Past:     K,V = [1, Hn, P,   Hd]
Present:  K,V = [1, Hn, P+1, Hd]
Axes:     [batch, heads, sequence, head_dim]   (sequence is the growth axis)
```

## Verified laws (all four, independently)

Matching only the final token is insufficient — a corrupted cache can hide for a token or two.
So the cache **state transition** is verified, not just numerical similarity at the output.

```
LAW 1 — GROWTH        present.seq = past.seq + 1                         PASS
LAW 2 — PRESERVATION  present[:,:,0:P,:] = past   (K maxabs 0, V maxabs 0)  PASS  (exact)
LAW 3 — APPEND        present[:,:,P,:]  = projected(new_token) (K 0, V 0)   PASS  (exact)
LAW 4 — COMPUTATION   Attention_GPU(past+new) ≈ Attention_CPU(past+new)     PASS  (8.08e-08)
```

Prefix error = 0 and append error = 0 ⇒ **state-transition correctness**, not just output
similarity.

## The general law this establishes

```
F_{t+1} = Preserve(F_t) ⊕ Δ_t
    F_t = existing field        Δ_t = newly admitted state
    ⊕   = contract-defined append (preserve prefix, append along the growth axis)
```

Not transformer-specific — the same law can describe semantic memory, event histories, replay
fields, and graph growth, all under one distinction between **tensor geometry**, **field
semantics**, and **physical residency**. See `docs/tensor-fields-and-residency.md`.

## Reproduce

```
# from scratch/dml/ (needs the DirectML redist — see scratch/dml/README.md):
cl /nologo /std:c++17 /EHsc /O2 /I include dml_mha_kv_test.cpp /link /LIBPATH:lib /OUT:dml_mha_kv_test.exe
dml_mha_kv_test.exe
```

`kv_cache_test.cpp` here is the frozen snapshot; the evolving source is
`scratch/dml/dml_mha_kv_test.cpp`. Field mutation policy: `field_contract.json`.

## Two independent proofs now stand

- **#001** (`proof/gpt2_hd4600_resident_v1/`) — resident computational graph is correct.
- **#002** (this) — resident mutable session state evolves correctly.

Next (do **not** combine yet): **Proof #003 — Resident Generation**, which composes #001 + #002 +
token selection + the closed decode cycle (Pop→Wo→Yax→Sek→Ch'en→Xul).
