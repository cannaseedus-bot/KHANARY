# KXC — K'UHUL Kernel Compiler

> `kxc.exe` compiles `.kuhul` kernel descriptors into GPU/CPU artifacts.
> It is the KXC stage of the AS-XCFE toolchain: source → HLSL/WGSL/SMCA/IR.

---

## Canonical location

| Artifact | Path |
|----------|------|
| Binary | `versions/kxc-v1.0.0/bin/kxc.exe` |
| Source | `versions/kxc-v1.0.0/src/` (CMake + Ninja, VS 2022 BuildTools x64) |
| Build script | `versions/kxc-v1.0.0/build_kxc.bat` |
| Grammar | `versions/kxc-v1.0.0/KXC.ebnf` (ISO/IEC 14977 — normative) |
| Registry | `versions/kxc-v1.0.0/registry/` |
| JS IR builder | `versions/kxc-v1.0.0/js/ir-format.js` |
| Examples | `versions/kxc-v1.0.0/examples/` |

---

## Stack position

```
XCFE              — semantic control flow / orchestration
K'UHUL / KLSL     — kernel descriptor language
KXC               — kernel compiler       ← this document
SMCA              — legality + registry/classification contract
SCXQ2             — backend-neutral lowered IR
HLSL / WGSL / CPU — executable backend artifacts
```

## What KXC emits

```
.kuhul descriptor
   ↓  kxc.exe --registry versions/kxc-v1.0.0/registry
.hlsl          — HLSL compute shader (D3D12 / DML target)
.wgsl          — WGSL compute shader (WebGPU target)
.cpu.cpp       — CPU fallback implementation
.smca.json     — compiled kernel contract (kernelClass, collapseClass, caps, layers, registryMatched)
.ir.json       — SCXQ2-lowered IR
```

---

## K'UHUL kernel language

A `.kuhul` kernel descriptor is a plain-text block: one `[Pop KernelName]` open,
optional `[Muwan]` and `[Sek]` properties, closed with `[Xul]`.

```kuhul
[Pop FusedAttention]
  [Muwan dispatch 64 1 1]
  [Sek needsSoftmax true]
  [Sek needsMatMul  true]
[Xul]
```

Full grammar: `versions/kxc-v1.0.0/KXC.ebnf`.
**`key = value` bare syntax is NOT supported** — use `[Sek key value]`.

---

## Classifier dispatch table (`lower.cpp::classify()`)

Priority order — first match wins:

| `[Sek]` flags | `kernelClass` | `collapseClass` |
|---------------|---------------|-----------------|
| `needsMoERoute \|\| needsMoEExpertFFN \|\| needsMoECombine` | `moe_route_top2` | `routing.top2` |
| `needsSoftmax && needsMatMul` | `tensor_attention_fused` | `attention.fused` |
| `needsMeshlet` | `mesh_meshlet_cull` | `mesh.cull` |
| `needsNormalCompute \|\| needsTangentFrame` | `mesh_normal_compute` | `mesh.normals` |
| `needsVertexProcess` | `mesh_vertex_process` | `mesh.vertex` |
| `needsAdam` | `adam_optimizer` | `training.optimizer` |
| `needsGradAccum` | `gradient_accum` | `training.accum` |
| `needsSiluGrad \|\| (needsGradClip && needsShmRead)` | `backward_pass` | `training.backward` |
| `needsValueClamp \|\| needsGradClip` | `fold_clamp` | `training.clamp` |
| _(fallthrough)_ | `generic-compute` | `compute.generic` |

---

## SMCA JSON

The `.smca.json` is the canonical runtime identity document for a compiled kernel.

```json
{
  "kernel": "FusedAttention",
  "target": "all",
  "threads": [64, 1, 1],
  "caps": { "waveOps": false, "heapTier": 1, "bindingTier": 1, "uma": true },
  "smca": {
    "kernelClass":    "tensor_attention_fused",
    "collapseClass":  "attention.fused",
    "lawful":         true,
    "registryMatched": true,
    "layers": ["MATRIX", "SCXQ2", "SCXQ7", "SCO/1", "IDB"],
    "requires": ["deterministic_join", "bounded_reduction"],
    "forbids":  ["side_effects", "order_dependence"]
  }
}
```

### Pipeline layers

| Layer | Meaning |
|-------|---------|
| `MATRIX` | source K'UHUL parsed into AST |
| `SCXQ2` | semantic ops lowered into backend-neutral IR |
| `SCXQ7` | legality and caps-aware optimization |
| `SCO/1` | backend emitters produce executable artifacts |
| `IDB` | sidecar metadata emitted for external verification |

---

## Registry (`versions/kxc-v1.0.0/registry/`)

| File | Role |
|------|------|
| `kernel-classes.json` | canonical class definitions (requires, forbids, backend, layers, lawful) |
| `kernel-aliases.json` | intermediate key → canonical name + collapseClass |
| `kernel-extras.json` | caps hints, threadsHint, fallback backend per class |

---

## Verified kernels (7/7 PASS)

| Source | `kernelClass` | `registryMatched` |
|--------|---------------|-------------------|
| `examples/fused_attention_full.kuhul` | `tensor_attention_fused` | true |
| `examples/fused_attention_simple.kuhul` | `tensor_attention_fused` | true |
| `examples/binary_split_test.kuhul` | `generic-compute` | true |
| `examples/adam_optimizer_test.kuhul` | `adam_optimizer` | true |
| `examples/gradient_accum_test.kuhul` | `gradient_accum` | true |
| `examples/backward_pass_test.kuhul` | `backward_pass` | true |
| `examples/fold_clamp_test.kuhul` | `fold_clamp` | true |

End-to-end test: `python tools/test_adam_flow.py` — 11/11 PASS.

---

## JS IR builder (`versions/kxc-v1.0.0/js/ir-format.js`)

`createKernelIR(opts)` — builds a validated IR object matching the SMCA contract.
`validateKernelIR(ir)` — checks ir_version, smca block, capability flags, forbid list.
`computeStackCid(artifacts)` — FNV1A-64 over sorted artifact list → stable stack identity.

Stack ID: `asx-xcfe-stack/v1`
