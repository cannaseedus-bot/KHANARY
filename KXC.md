# KXC.md — K'UHUL Kernel Compiler

> `kxc.exe` compiles `.kuhul` kernel descriptors into executable GPU/CPU artifacts.
> It is the KXC stage of the AS-XCFE toolchain: source → HLSL/WGSL/SMCA/IR.

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

## What KXC is

**KXC is the kernel compiler for the AS-XCFE stack.** It takes a `.kuhul` kernel
descriptor, classifies the kernel against the registry, and emits five artifacts per
kernel: HLSL, WGSL, CPU C++, SMCA JSON, and IR JSON.

```
.kuhul descriptor
   ↓  kxc.exe
   classify → kernel-aliases.json → kernel-classes.json
.hlsl       — HLSL compute shader (D3D12 / DML target)
.wgsl       — WGSL compute shader (WebGPU target)
.cpu.cpp    — CPU fallback implementation
.smca.json  — compiled kernel contract (kernelClass, collapseClass, caps, layers, registry result)
.ir.json    — SCXQ2-lowered IR
```

Binary: `C:\Users\canna\.ASX.cpp\kxc.exe` — 32-bit x86 MSVC debug PE.  
Output dir: `C:\Users\canna\.ASX.cpp\` (artifacts written alongside the binary).  
Registry dir: `C:\Users\canna\_khanary_inspect\versions\kxc-v1.0.0\registry\`

---

## K'UHUL kernel language

A `.kuhul` kernel descriptor is a plain-text block: one `[Pop KernelName]` open,
optional `[Muwan]` and `[Sek]` properties, closed with `[Xul]`.

```kuhul
[Pop FusedAttention]
  [Muwan dispatch 64 1 1]
  [Sek needsDecompress true]
  [Sek needsSoftmax true]
  [Sek needsMatMul true]
  [Sek kvInt4 true]
[Xul]
```

### Token reference

| Token | Syntax | Role |
|-------|--------|------|
| `[Pop KernelName]` | required, first | opens kernel block; sets kernel name |
| `[Muwan dispatch X Y Z]` | optional | thread group dimensions (exactly 3 integers) |
| `[Sek propertyName value]` | optional, repeatable | kernel property flag (bool / int / string) |
| `[Yax ...]` | optional | supported; syntax not yet documented |
| `[Xul]` | required, last | closes kernel block and triggers compilation |

**`key = value` bare syntax is NOT supported** — kxc exits 1 with "unsupported syntax".
Always use `[Sek key value]`.

### Confirmed `[Sek]` properties

`needsDecompress`, `needsSoftmax`, `needsMatMul`, `kvInt4`,
`needsMoERoute`, `needsMoEExpertFFN`, `needsMoECombine`, `needsPhaseMatch`

---

## Usage

```powershell
# from the kxc output dir (artifacts land alongside kxc.exe)
cd C:\Users\canna\.ASX.cpp
.\kxc.exe C:\path\to\kernel.kuhul
```

Output on success: `emitted: KernelName.cpp KernelName.hlsl KernelName.wgsl KernelName.cpu.cpp KernelName.smca.json`

---

## SMCA JSON

The `.smca.json` is the canonical runtime identity document for a compiled kernel.

```json
{
  "kernel": "FusedAttention",
  "target": "all",
  "threads": [64, 1, 1],
  "caps": {
    "waveOps": false,
    "heapTier": 1,
    "bindingTier": 1,
    "uma": true
  },
  "smca": {
    "kernelClass": "tensor_attention_fused",
    "collapseClass": "attention.fused",
    "lawful": true,
    "registryMatched": true,
    "layers": ["MATRIX", "SCXQ2", "SCXQ7", "SCO/1", "IDB"],
    "requires": ["deterministic_join", "bounded_reduction"],
    "forbids": ["side_effects", "order_dependence"],
    "notes": [...]
  }
}
```

| Field | Meaning |
|-------|---------|
| `kernelClass` | canonical kernel class name from registry |
| `collapseClass` | collapse/merge class for XVM scheduling |
| `lawful` | kernel passes legality check |
| `registryMatched` | kernel found in SMCA/kxc registry |
| `layers` | compilation pipeline stages traversed |
| `requires` / `forbids` | semantic constraints from kernel-classes.json |

### Pipeline layers

| Layer | Meaning |
|-------|---------|
| `MATRIX` | source K'uhul parsed into AST |
| `SCXQ2` | semantic ops lowered into backend-neutral IR |
| `SCXQ7` | legality and caps-aware optimization |
| `SCO/1` | backend emitters produce executable artifacts |
| `IDB` | sidecar metadata emitted for external verification |

---

## Registry

Three JSON files under `C:\Users\canna\_khanary_inspect\versions\kxc-v1.0.0\registry\`:

### `kernel-aliases.json` — intermediate key → canonical name

Maps the intermediate class names assigned by the classifier to their canonical
registry names and collapse classes.

```json
{
  "generic-compute": {
    "local": "binary_split", "@kernel": "binary_split",
    "canonical": "binary_split", "collapseClass": "compute.binary", "family": "compute"
  },
  "fused-attention": {
    "local": "tensor_attention_fused", "@kernel": "tensor_attention_fused",
    "canonical": "tensor_attention_fused", "collapseClass": "attention.fused", "family": "attention"
  },
  "moe-routing": {
    "local": "moe_route_top2", "@kernel": "moe_route_top2",
    "canonical": "moe_route_top2", "collapseClass": "routing.top2", "family": "routing"
  }
}
```

### `kernel-classes.json` — canonical class definitions

Per canonical class: `requires`, `forbids`, `backend`, `layers`, `lawful`, `collapseClass`, `family`.

### `kernel-extras.json` — overlay hints

Per canonical class: `smcaAnnotation`, `threadsHint`, `fallback` backend.

---

## Kernel classification

The classifier at `0x66090` inspects the kernel name and assigns `kernelClass` and
`collapseClass` to the kernel object before registry lookup:

| Kernel name first char | Assigned kernelClass | collapseClass |
|------------------------|----------------------|---------------|
| `F` (FusedAttention) | `tensor_attention_fused` | `attention.fused` |
| anything else | `generic-compute` | `compute.generic` |

**Current limitation — registry capability > classifier capability.**
The registry already defines entries for MoE routing and can accommodate further
classes, but the binary classifier currently resolves to exactly two effective
`kernelClass` values. All non-FusedAttention kernels collapse to `generic-compute`
regardless of semantic properties.

Classes prepared in the registry but not yet reachable by the classifier:

| Canonical class | family | collapseClass |
|-----------------|--------|---------------|
| `moe_route_top2` | routing | `routing.top2` |
| *(moe_expert_ffn, moe_combine, phase_match)* | — | — |

The next compiler milestone is extending the classifier to dispatch these classes from
kernel name or `[Sek]` property signals, at which point `kernelClass` and
`collapseClass` become a real dispatch taxonomy instead of a binary split.

---

## Test kernels

All four verified — exit 0, `registryMatched: true`:

| Source | kernelClass | threads |
|--------|-------------|---------|
| `drivers/klsl/examples/fused_attention_full.kuhul` | `tensor_attention_fused` | [64,1,1] |
| `drivers/klsl/examples/fused_attention_simple.kuhul` | `tensor_attention_fused` | [64,1,1] |
| `drivers/klsl/examples/binary_split_test.kuhul` | `generic-compute` | [32,1,1] |
| `drivers/klsl/examples/neural_layer_kuhul_test.kuhul` | `generic-compute` | [32,16,1] |

All paths relative to `C:\Users\canna\_khanary_inspect\`.

---

## Files

| Path | Role |
|------|------|
| `C:\Users\canna\.ASX.cpp\kxc.exe` | kernel compiler binary |
| `C:\Users\canna\_khanary_inspect\versions\kxc-v1.0.0\registry\kernel-aliases.json` | intermediate → canonical name map |
| `C:\Users\canna\_khanary_inspect\versions\kxc-v1.0.0\registry\kernel-classes.json` | canonical class definitions |
| `C:\Users\canna\_khanary_inspect\versions\kxc-v1.0.0\registry\kernel-extras.json` | caps hints and fallback backends |
| `C:\public_html\MX2LM\codex\AS-XCFE\native\xvm-d3d12\SMCA\registry\kernel-classes\v1.json` | SMCA registry (runtime side) |
| `drivers/klsl/examples/*.kuhul` | test kernel descriptors |
