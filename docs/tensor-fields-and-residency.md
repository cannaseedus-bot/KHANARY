# KHΛNARY tensor-field model + GPU residency classes (design notes)

Terminology adopted for SCXQ2 / KGRC. The point: KNU/K'UHUL needs **one tensor algebra with
different coordinate semantics**, not two unrelated tensor systems.

## Three separate concepts (don't conflate)

```
storage geometry   → linear contiguous memory (F32 F32 F32 …)
tensor geometry    → rank + dimensions + strides   (offset(row,col) = row·stride + col)
semantic geometry  → what the axes MEAN
```

A GPT-2 weight `[768, 2304]` is stored as a linear buffer, has tensor geometry rank-2, and its
semantic geometry is `axis0 = input feature`, `axis1 = QKV output feature`. SCXQ2 must carry all
three: **FIELD** = what it means, **TENSOR** = shape/storage, **EDGE** = relationship,
**LANE** = how it executes.

## One tensor domain, two coordinate families

```
                 KHΛNARY TENSOR DOMAIN
              ┌──────────┴──────────┐
        RECTILINEAR              SPATIAL
        (non-spatial geometry)   (spatial/topological)
        matrices [M,N]           vertices [N,3]
        vectors  [N]             topology (triangle indices)
        volumes  [S,H,D]         joints / 4×4 skin matrices
   coords: row/col/channel/      coords: x/y/z/vertex/
           head/token/feature            triangle/joint
```

Both are legitimate geometric-tensor substrates. A GPT weight is **not** a polygon mesh, but it
**does** have geometry — rectilinear, non-spatial. The birdsong/`brain2` mesh is spatial. Same
algebra, different coordinate semantics.

## A transformer layer is a tensor-field GRAPH, not a grid

A layer is the container/graph; the matrices are its fields:

```
LAYER[n]
├── LN     (weight[768], bias[768])
├── ATTENTION (QKV_W[768,2304], QKV_B[2304], proj_W[768,768], proj_B[768])
├── LN     (weight[768], bias[768])
└── MLP    (fc_W[768,3072], fc_B[3072], proj_W[3072,768], proj_B[768])
```

This matches KHΛNARY's existing folds → nodes → tensors structure.

## GPU residency classes (refines the KGRC contract)

The DirectML resident-model work distinguishes three residency lifetimes — a stronger
classification than "weights vs activations":

```
GPU RESIDENT STATE
├── STATIC     (invariant fields — upload once, never change)
│   ├── wte / wpe
│   ├── per-layer attention + MLP weights
│   ├── LN parameters
│   └── lm_head
├── SESSION    (append-growing fields — persist across a generation)
│   ├── K[0..11]   [1, heads, seq, head_dim]   (grows +1 per token)
│   └── V[0..11]
└── TRANSIENT  (per-step scratch — reused each step)
    ├── hidden, Q, logits, scratch
```

- **STATIC** = weights (Proof #001, already resident).
- **SESSION** = KV cache — a **growing rectilinear tensor** `[batch, heads, seq, head_dim]`,
  seq += 1 per decode step. Proven as a DirectML native-cache primitive (Proof #002 de-risk:
  `dml_mha_kv_test.cpp`, all 4 invariants).
- **TRANSIENT** = activations.

## KV-cache decode = the closed K'UHUL cycle

The generation loop is not an ordinary `for` loop; it's the fold:

```
Pop   = receive current token / session state
Wo    = bind resident model (STATIC) + KV (SESSION) fields
Yax   = resolve valid cache shape / dependencies
Sek   = execute transformer decode
Ch'en = commit PresentK/PresentV (SESSION) + logits
Xul   = choose continuation / termination → Pop(next token)
```

## Proof discipline for the KV-cache primitive (all four, independently)

A decode step `PastK/V [1,Hn,P,Hd] → PresentK/V [1,Hn,P+1,Hd]` is only "proven" when **all four**
hold (matching just the final token is insufficient — a corrupt cache can hide for a token or two):

1. **shape**: `present.seq == past.seq + 1`
2. **prefix**: `present[:,:,0:P,:] == past` (K and V)
3. **append**: `present[:,:,P,:] == K/V(new token)`
4. **output**: GPU decode output ≈ CPU reference

Verified on the HD 4600 (`dml_mha_kv_test.cpp`): shape OK, prefix `0.00e+00`, append `0.00e+00`,
output `8.08e-08`. DirectML's native KV-cache path is proven; the resident generation loop
(Proof #002) can build on it.
