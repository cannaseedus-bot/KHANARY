# KHΛNARY semantic runtime — v0.1.0 (instrumented proof-ladder)

**The runtime counterpart to `khanary-kxml`.** khanary-kxml (v0.5.0) *registers* the tool/node
contracts but "does NOT itself execute them" — that runtime is the **FieldExecutionEngine**. This
package certifies that the live engine actually **executes trajectories conforming to the KXML
record algebra**, and honestly maps *which knob excites which record axis*.

> ⚠️ **Not a hardware-verified capability release** like gpt2 / geometry / gpu-resident. This is an
> **instrumented proof-ladder**: it certifies runtime *conformance* and characterizes excitation —
> it does **not** certify a numerical kernel against a CPU reference. Read `MODEL.json` →
> `honest_scope` before citing any number here.

## The ladder (frozen)

| stage | claim | result |
|---|---|---|
| **S#001** | kernel is a read-only teacher/oracle | 66 records, all 5 classes, 20 legal / 28 illegal, sources **unmutated** |
| **S#002** | live engine conforms to S#003a | 10 live ticks → 60 records (TRANSITION/DELTA/FIELD), all conform |
| **S#002b** | read-only coverage is insufficient | degenerate (routing roles=2) → use vocabulary **convergence**, not count |
| **S#002c** | controlled excitation — **decisive negative** | 16-case adversarial matrix, 10 folds diverse, but legality/routing **input-invariant** |
| **S#002c-A** | implement the model bridge → axes excite | routing keyword-gated (General+CODING+FACTORY); fitness varies; **1 VIOLATION** via entropy stress |

## The three things you must not overclaim

1. The S#002c-A **fitness is a hash** of the model response bytes
   (`0.30 + 0.69·(hash%1000)/1000`) — a stimulus to move the evolver off a constant, **not** a
   learned or semantic score.
2. The headline S#002c result is a **negative**: input does not excite the legality boundary. The
   one `VIOLATION` observed (`BOUNDARY_BREACH`) needed an **artificial 30-tick entropy-accumulation
   stress run**; 2 of the 3 verdict types remain **unobserved**.
3. The implemented bridge lives in a gitignored **copied header** (snapshot
   `proof/semantic_corpus_s002c/field_execution_engine.bridge.h`); the kernel `.cpp` in `.ASX.cpp`
   is **unchanged**.

## Consequence
`S#003b` (record-schema freeze) is **unblocked but not done** — it needs all three violation verdict
types covered and the vocabulary saturated across diversified batches first. See the sibling
`models/khanary-kxml-v0.5.0` for the contract layer this runtime executes.

## Reproduce
```
python tools/build_semantic_runtime_model.py   # re-vendors frozen evidence + emits MODEL.json
# to re-excite the axes, see proof/semantic_corpus_s002c/S002C_A_PASS.md (live llama-server + harness).
```
