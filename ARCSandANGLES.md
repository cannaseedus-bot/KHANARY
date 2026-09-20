# ARCSandANGLES — Fold Arc Weights and Phase Geometry

The K-CUBE has two separate but interacting numeric systems.  
This document keeps them straight.

---

## The two systems

| System | Symbol | What it is | Who touches it |
|---|---|---|---|
| **Phase angle** | `angles[i]` = i × π/3 | Fixed structural identity of each fold | Nobody — geometry |
| **Arc weight** | `fold_arcs[6,6]` | Learned flow strength from fold i → fold j | Adam optimizer |

The angles are invariant.  The arc weights are trainable.

---

## Phase angles (fixed)

Six folds arranged at π/3 intervals on the unit circle:

```
      Pop  (0)         0 rad  — observe / Q-read
      Wo   (1)        π/3     — weight / mask
      Yax  (2)       2π/3     — enumerate / K-read
      Sek  (3)         π      — compute / QKᵀ
    Ch'en  (4)       4π/3     — collect / V-gather
      Xul  (5)       5π/3     — entropy / output-project
```

Visual layout (flat):

```
               Pop (0°)
              /        \
    Xul (300°)          Wo (60°)
         |     K-CUBE    |
    Ch'en(240°)         Yax (120°)
              \        /
               Sek (180°)
```

These angles define the **structural identity** of each fold — the same identity that maps onto the attention roles: Pop=Q, Yax=K, Sek=QKᵀ, Wo=mask+weight, Ch'en=V, Xul=output.  Changing them would change what the folds *are*, not how strongly they talk to each other.

---

## Arc weights (learned)

`fold_arcs[6,6]` is a 6×6 matrix of scalar flow strengths.  
Entry `[i,j]` is how strongly the output of fold `i` flows into fold `j`.

```
                     dst
          Pop  Wo   Yax  Sek  Ch'en Xul
src Pop  [0.17 0.17 0.17 0.17 0.17  0.17]   ← initial uniform 1/6
    Wo   [0.17 0.17 ...]
    Yax  [...]
    Sek  [...]
    Ch'en[...]
    Xul  [...]
```

These weights are what the Adam optimizer updates.  They live in `AdamFoldArcs.arc_w[6][6]`.

---

## Effective arc weight

A raw arc weight is modulated by the **cosine of the phase gap** between source and destination fold:

```
effective[i→j] = arc_w[i][j] × cos(angles[j] − angles[i])
```

| Arc | Gap | cos(gap) | Meaning |
|---|---|---|---|
| Pop→Wo   | π/3  | +0.50 | adjacent — strong positive flow |
| Pop→Yax  | 2π/3 | −0.50 | two steps — opposing contribution |
| Pop→Sek  | π    | −1.00 | directly opposing — maximal cancellation |
| Pop→Ch'en| 4π/3 | −0.50 | three-quarter — partial cancellation |
| Pop→Xul  | 5π/3 | +0.50 | near-adjacent (wrap) — positive |
| Pop→Pop  | 0    | +1.00 | self-loop — full weight |

This means adjacent folds are geometrically biased to cooperate, and directly opposing folds (π gap) are biased to cancel — even before training begins.  The arc weights learned by Adam modulate *how much* of that geometric bias is used.

---

## What Adam optimizes

Adam updates `arc_w[i][j]` using gradients that flow from the loss back through the fold routing:

```
m[i][j] = β₁·m[i][j] + (1−β₁)·g[i][j]
v[i][j] = β₂·v[i][j] + (1−β₂)·g[i][j]²
arc_w[i][j] -= lr_hat · m̂[i][j] / (√v̂[i][j] + ε)
```

All 36 arcs are clipped globally (L2 norm across the full arc-gradient matrix) before each step.

---

## Fold-arc naming convention

When persisting arc weights to a checkpoint or manifest:

```json
{
  "fold_arcs": {
    "pop.wo":    0.183,
    "pop.yax":   0.141,
    "pop.sek":   0.167,
    "pop.chen":  0.152,
    "pop.xul":   0.174,
    ...
  }
}
```

Key format: `{src_fold}.{dst_fold}` — both lowercase.  `Ch'en` serialises as `chen`.

Do **not** store phase angles in checkpoints.  They are computed constants (`i * π/3`), not learned state.

---

## Files

| File | Role |
|---|---|
| `native/adam/adam.h` | C ABI header — `AdamFoldArcs` struct |
| `native/adam/adam.cpp` | Implementation — `adam_fold_arcs_step`, `adam_fold_arc_effective` |
| `tools/adam_ctypes.py` | Python wrapper — `FoldArcOptimizer` class |
| `drivers/klsl/adam.kuhul` | KXC capability manifest — `fold_adam` kernel |
| `drivers/klsl/khlc/adam.klsl` | KHLC compute shader — GPU-side Adam update |

The CPU path (Adam.dll) and the GPU path (adam.klsl → HLSL) implement the same algorithm.  Use the DLL path for small fold-arc weight tensors (36 scalars), use the HLSL path for full weight-shard updates.

---

## See also

- `K-CUBE.md` — full 6-face tensor geometry, SH projection, face↔phase mapping
- `SEMANTIC_ENGINE.md` — fold loop as micronaut memory cycle
- `KUHUL.md` — kuhul-es fold runtime, phase system Pop→Xul at π/3
