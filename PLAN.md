# KHANARY Trainer Build Plan

## Goal
Standalone GPT-2 D3D11 trainer inside `_khanary_inspect/trainer/`, buildable with CMake + MSVC without the full `.ASX.cpp` monorepo. Target: compile `gpt2_trainer.exe`, run v0.2 training on `E:/data/kuhul_synthetic.jsonl`.

---

## Trainer architecture (Phase 3)

| Component | Implementation |
|-----------|----------------|
| Forward pass | **CPU** — exact layernorm / matmul / attention / GELU |
| Backward pass | **CPU** — exact backprop through all layers |
| Adam update | **GPU** via `cs_adam_` compute shader (D3D11 cs_5_0) |
| Phase 3 fwd/bwd | **GPU** — full pipeline via 19 compute shaders when `use_gpu_fwd=true` |
| Shader compilation | Runtime, `D3DCompileFromFile`, HLSL → cs_5_0 |

HD 4600 constraint: per-layer `ComPtr<ID3D11Buffer>` vectors everywhere (flat `[NL, ...]` layouts break because the driver ignores `SRV.FirstElement` for structured buffers).

---

## Trainer folder

```
trainer/
  CMakeLists.txt          standalone CMake build (FetchContent nlohmann_json)
  d3d11_engine.h          trainer-specific D3D11Engine (rawDevice()/rawCtx())
  d3d11_engine.cpp        D3D11CreateDevice + adapter name via DXGI
  gpt2_config.h           GPT-2 small: V=50260 S=1024 E=768 H=12 L=6
  gpt2_trainer.h          GPT2Trainer class (Phase 3 full GPU fwd+bwd+Adam)
  gpt2_trainer.cpp        ~1800 LOC: CPU math helpers + Phase 3 dispatch
  gpt2_train_main.cpp     CLI: --model --data --out --steps --batch --block --lr
  pi_kuhul/
    KuhulPhysics.h        adaptive gradient-gravity controller (header-only)
    SphericalGeometryAVX2.h
    DirectXMathAVX2.h
    Fold2DCompiler.h
  shaders/
    gpt2_adam.hlsl
    gpt2_attn_fwd.hlsl  gpt2_attn_bwd.hlsl
    gpt2_bias_bwd.hlsl
    gpt2_embed_fwd.hlsl gpt2_embed_bwd.hlsl
    gpt2_gelu_fwd.hlsl  gpt2_gelu_bwd.hlsl
    gpt2_layernorm_fwd.hlsl gpt2_layernorm_bwd.hlsl
    gpt2_loss.hlsl
    gpt2_matmul_fwd.hlsl gpt2_matmul_bwd.hlsl
    gpt2_residual_add.hlsl
```

---

## Build steps

### Prerequisites
- VS 2022 or VS 2026 BuildTools (MSVC, x64)
- Windows SDK 10.0.26100.0 (for d3d11.h, dxgi.h, d3dcompiler.h, DirectXMath.h)
- CMake ≥ 3.20
- Internet access for FetchContent (nlohmann_json v3.11.3 from GitHub)

### Configure + build

```powershell
# Open a VS x64 Native Tools command prompt, then:
cd C:\Users\canna\_khanary_inspect\trainer
cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Or with VS 2026 Insiders:
```powershell
# via vs-insiders agent: Enter-VsDevShell c67d4bb6
cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Run (from build/Release/ so ../shaders/ resolves)

```powershell
cd C:\Users\canna\_khanary_inspect\trainer\build\Release
.\gpt2_trainer.exe `
  --model "C:\Users\canna\.ASX.cpp\trainer\from_zero_v0.1.safetensors" `
  --data  "E:\data\kuhul_tokens.bin" `
  --out   "C:\Users\canna\_khanary_inspect\models\from_zero\from_zero_v0.2.safetensors" `
  --steps 5000 --batch 4 --block 128 --lr 3e-5
```

Env flags:
```
$env:GPT2_FULLSEQ       = "1"   # full-sequence loss over all S positions
$env:GPT2_ADAPTIVE_CLIP = "1"   # KuhulPhysics gradient-gravity controller
```

---

## Training data pipeline

### Corpus / dataset links — `E:\data\`

**Root files:**

| File | Size | Notes |
|------|------|-------|
| `kuhul_synthetic.jsonl` | ~340 MB | **Primary** — 350,388 π-KUHUL examples |
| `khanary_transitions.jsonl` | 261.8 MB | Seed transitions (101,562 records) |
| `khanary_transitions_kxml.jsonl` | 139.8 MB | KXML-annotated transitions |
| `khanary_clean_train.jsonl` | 105.8 MB | Cleaned Khanary train split |
| `kuhul_tokens.bin` | 487 MB | Packed token bins — 951,264 seqs × 128 tokens |
| `kuhul_flat.bin` | 487 MB | Flat (un-headed) version of kuhul_tokens.bin |
| `merged_for_gpt2.jsonl` | — | Merged GPT-2 corpus |
| `glyph-corpus-v3.jsonl` | — | Glyph/K'UHUL corpus v3 |
| `evidence_15k.jsonl` | — | 15k evidence samples (C2/C2b PMI experiments) |
| `heldout_400.jsonl` | — | 400-sample held-out eval set |
| `trinity_field.json` | — | Trinity field data |
| `field_c0*.json` | — | C2/C2b PMI experiment field captures |

**Subdirectories:**

| Dir | Size | Contents |
|-----|------|----------|
| `smgm16_gpu_bridge/` | 11 GB | GPU bridge data |
| `claude_history/` | 4.7 GB | Claude conversation history |
| `KxmlGpt2Corpus/` | 4.5 GB | KXML-formatted GPT-2 training corpus |
| `chunks/` | 1.9 GB | Chunked corpus splits |
| `semantic-json/` | 1.3 GB | Semantic JSON training data |
| `coder_outputs/` | 1 GB | Coder model outputs |
| `chat-data/` | 736 MB | Chat format training data |
| `xshard_jsonl/` | 579 MB | XShard JSONL shards |
| `coding-corpus/` | 488 MB | Coding corpus |
| `ultrachat_jsonl/` | 299 MB | UltraChat JSONL |
| `math_data/` | 154 MB | Math training data |
| `deepseek_data/` | 127 MB | DeepSeek-style data |
| `KodCode/` | 70 MB | KodCode dataset |

**Previous v0.1 bins (`.ASX.cpp`):**

| File | Size |
|------|------|
| `C:\Users\canna\.ASX.cpp\trainer\tokens_hdr.bin` | 12 MB |
| `C:\Users\canna\.ASX.cpp\trainer\tokens_hdr_big.bin` | 49 MB |

### Phase=0 / gravity diagnosis

`[split] text=X phase=0 total=X` — **`phase=0` is expected and correct** for the current training run. It means no tokens with ID ≥ 50257 appear in `kuhul_tokens.bin`. The data was tokenized before the KUHUL vocab extension existed, so `<THINK>`, `<AGENT>` etc. got split into regular GPT-2 subword tokens. The KuhulPhysics gravity IS working (adaptive clip fires, EMA runs) but there are no glyph tokens yet to count as "phase."

**Fix:** Re-tokenize `kuhul_synthetic.jsonl` after:
1. `extend_vocab.py` patches the checkpoint (`wte`: [50260,768] → [50270,768])
2. Tokenizer is updated to map `<THINK>` → 50262, `<AGENT>` → 50260, etc. (see `tokenizer_config.json`)

### Tokenize → pack → train

```powershell
# 1. Generate synthetic corpus (already done: 350,388 examples)
python tools/gen_kuhul_training.py E:\data\khanary_transitions.jsonl -o E:\data\kuhul_synthetic.jsonl

# 2. Tokenize
python C:\Users\canna\.ASX.cpp\trainer\tokenize_transitions.py E:\data\kuhul_synthetic.jsonl flat.bin

# 3. Pack into headed format
python tools\pack_tokens.py flat.bin E:\data\kuhul_tokens.bin --seq-len 128

# 4. Train
$env:GPT2_FULLSEQ=1; $env:GPT2_ADAPTIVE_CLIP=1
cd C:\Users\canna\_khanary_inspect\trainer\build\Release
.\gpt2_trainer.exe --model "C:\Users\canna\.ASX.cpp\trainer\from_zero_v0.1.safetensors" `
  --data E:\data\kuhul_tokens.bin `
  --out  "C:\Users\canna\_khanary_inspect\models\from_zero\from_zero_v0.2.safetensors" `
  --steps 5000 --batch 4 --block 128 --lr 3e-5
```

---

## Model registry — `models/`

| Model | Key Files | Notes |
|-------|-----------|-------|
| `khanary-gpu-resident-v0.1.0/` | `proof/`, `MODEL.json`, `README.md` | Proof of HD 4600 1792 MB ceiling + Q8 hot-swap |
| `khanary-kxml-v0.5.0/` | `kxml_chat_template.jinja/json`, `kxml_nodes.json`, `kuhul.tools.jsonl`, `kxml_alignment.json`, `MODEL.json`, `source/kuhul_functions.h`, `source/kuhul_tool_runtime.h` | KXML tool/op registry: 12 tools, 7 node ops, 5 phases (Pop/Wo/Sek/Chen/Xul). Trainable chat-template + tool-call layer. |
| `khanary-qwen1_8b-v0.1.0/` | `qwen1_8b.q4.kqz` (931 MB), `qwen1_8b.q8.kqz` (1753 MB), manifests | Q4 resident (0.909 GiB, ~20 dB) + Q8 hot-swap (1.712 GiB, ~41 dB). Archive at `E:\models\Qwen1.8B-quant\`. Weights only — no Qwen forward path yet. |
| `khanary-semantic-runtime-v0.1.0/` | `proof/`, algebra doc, `MODEL.json` | Semantic runtime proofs |
| `from_zero/` | `from_zero_v0.1.f32.gguf` (623.6 MB) | v0.1 checkpoint as GGUF. Safetensors source at `C:\Users\canna\.ASX.cpp\trainer\from_zero_v0.1.safetensors`. v0.2 training writing to `from_zero_v0.2.safetensors`. |
| `khanary-geometry-v0.3.0/` | `kernels/vertex_skin.hlsl`, `kernels/vertex_skin.wgsl`, `kernels/vertex_transform.hlsl/.wgsl`, KNU JSON, `data/birdsong_mesh.stb` (2 MB) | Vertex skinning + transform kernels. Relevant to K'UHUL attention bias "skinning" analogy. |
| `khanary-gpt2-v0.4.0/` | `trainer/gpt2_trainer.cpp` (106.9 KB), `trainer/gpu_fwdbwd_new.cpp` (32.6 KB), `trainer/gpt2_train_main.cpp`, `trainer/include/KuhulPhysics.h`, `trainer/shaders/` (full set), `trainer/engine/d3d11_engine.h` (XVM), `kernels/`, `knu/`, `data/gpt2_c_proj.stb` | Authoritative source for 7 additional shaders (copied to `trainer/shaders/`). NOTE: `gpu_fwdbwd_new.cpp` uses flat `[NL,...]` SRV offsets — broken on HD 4600; standalone `gpt2_trainer.cpp` uses per-layer buffers instead. Engine is XVM version (not compatible with trainer build). |

### v0.4.0 shaders added to `trainer/shaders/`
Copied from `khanary-gpt2-v0.4.0/trainer/shaders/` — these were missing from the standalone build:
- `gpt2_lm_head.hlsl` — unembedding (last position → logits)
- `gpt2_lm_head_bwd.hlsl` — LM head backward (scatter dL/dwte)
- `gpt2_qkv_split.hlsl` — split QKV into per-head Q, K, V
- `gpt2_adam_wte.hlsl` — WTE-specific Adam (numthreads(768,1,1) = one group per vocab token)
- `gpt2_embed.hlsl` — combined embed (alt to embed_fwd; loop-per-dim style)
- `gpt2_matmul.hlsl` — add-into matmul (8×8 tiled)
- `gpt2_residual.hlsl` — non-in-place residual (x + y → result)

---

## GPU inference path: ggml-xcfe vs. custom shader pipeline

### The MUL_MAT-only limitation

When loading a GGUF model through `llama-server` + the `ggml-xcfe` DirectML backend, the startup log shows:

```
warning: no usable GPU found, --gpu-layers option will be ignored
[ggml-xcfe] MUL_MAT path: DirectML (GPU)
```

The first line is llama.cpp's standard GPU detection (it found no Vulkan/CUDA/Metal). The second is the custom xcfe backend intercepting **only** MUL_MAT ops and routing them to DirectML. All other ops run on CPU:

| Op | ggml-xcfe path | Custom trainer path |
|----|---------------|---------------------|
| Token + position embedding | CPU | `cs_embed_fwd_` (GPU) |
| LayerNorm (mean/var/norm) | CPU | `cs_lnorm_fwd_` (GPU) |
| QKV projection (matmul) | **DirectML GEMM** | `cs_matmul_fwd_` (GPU) |
| QK^T attention + softmax | CPU | `cs_attn_fwd_` (GPU) |
| 4-bone LBS bias | — | `cs_kuhul_think_bias_` (GPU) |
| GELU activation | CPU | `cs_gelu_fwd_` (GPU) |
| Residual add | CPU | `cs_resadd_*` (GPU) |
| LM head unembedding | CPU | `cs_matmul_fwd_transb_` (GPU) |
| Loss + backward | CPU | `cs_loss_` + bwd shaders (GPU) |

With batch=1 on a small model, attention and layernorm dominate wall time. Routing only MUL_MAT to DirectML gives roughly **30-40% FLOP reduction** — the CPU-bound ops (softmax, norm, residual) remain the bottleneck.

### Full GPU coverage: ggml-webgpu already has all ops

The `ggml-xcfe` DirectML backend is a custom intercept that only wired up MUL_MAT. The llama.cpp build in this repo already ships a complete **ggml-webgpu** backend at `ggml/src/ggml-webgpu/wgsl-shaders/` with WGSL kernels for every op the CPU was handling:

```
soft_max.wgsl         → GGML_OP_SOFT_MAX (attention softmax)
row_norm.wgsl         → GGML_OP_NORM
rms_norm_mul.wgsl     → GGML_OP_RMS_NORM
rope.wgsl             → GGML_OP_ROPE (rotary position embedding)
flash_attn.wgsl       → fused QKV attention (tiled + vec variants)
unary.wgsl            → GELU, SiLU, and all element-wise activations
binary.wgsl           → ADD, MUL, residual connections
get_rows.wgsl         → token embedding lookup
mul_mat_*.wgsl        → all MUL_MAT variants (reg-tile, vec, subgroup)
quantize_q8.wgsl      → Q8 quantization on GPU
```

The MUL_MAT-only limitation is specific to the ggml-xcfe custom DirectML backend, not to llama.cpp. Switching the build to target the **ggml-webgpu** backend gives full GPU coverage with no CPU round-trips between ops.

### KLSL → WGSL for K'UHUL-specific extensions

KLSL (K'UHUL Level Shading Language) provides the WebGPU op generation layer for ops that don't exist in upstream ggml-webgpu — K'UHUL-specific kernels like the 4-bone LBS attention bias, fold-aware routing, and pi-nary arc weighting. These compile to WGSL and slot into the ggml-webgpu dispatch table alongside the upstream shaders. New K'UHUL ops are authored in KLSL rather than raw WGSL.

### Build target: GGML_WEBGPU=ON

Switch the CMake build from ggml-xcfe to ggml-webgpu:

```powershell
# In the VS x64 dev shell:
cd C:\Users\canna\_khanary_inspect\khanary-llama-build\llama.cpp
cmake -B build-webgpu -S . -G "Visual Studio 17 2022" -A x64 `
  -DGGML_WEBGPU=ON `
  -DGGML_XCFE=OFF `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-webgpu --config Release --target llama-server
```

ggml-xcfe can be left in the tree (`GGML_XCFE=OFF` just doesn't register the backend); the two
are independent plugins and don't conflict.

**Where KUHUL WGSL shaders slot in:** after emitting from KLSL → WGSL (via `emit_wgsl.py`),
copy the output file into `ggml/src/ggml-webgpu/wgsl-shaders/` and register the op dispatch
entry in `ggml-webgpu.cpp` — the same pattern used by all upstream shaders in that folder.

### HD 4600 compatibility — WebGPU via D3D12

Intel HD 4600 (GT2, Haswell) supports D3D12 feature level 11_0. Dawn (the WebGPU C++ library
that llama.cpp's ggml-webgpu backend uses) adapts to D3D12 on Windows — it does **not** require
Vulkan. Dawn's `D3D12Backend` has been validated on Haswell since WebGPU shipped in Chrome 113.

Adapter selection is automatic when only one GPU is present. To verify at runtime:

```
[ggml-webgpu] adapter: Intel(R) HD Graphics 4600 (D3D12, feature level 11_0)
```

If Dawn falls back to WARP (software rasterizer), force the discrete adapter:
```powershell
$env:DAWN_BACKEND = "d3d12"
$env:DAWN_ADAPTER_NAME = "Intel"   # substring match against adapter description
.\llama-server.exe --model model.gguf --n-gpu-layers 99
```

The ggml-webgpu shader set (`wgsl-shaders/`) already compiles cleanly against
WGSL 2024 / D3D12 11_0 — no SM 6.x features required.

### Native Glyph Engine → WebGPU mapping

`C:\Users\canna\.NNC-K\bin\Kuhul-c++\native_glyph_engine.cpp` and `glyph.h` map directly to WebGPU via KLSL:

| Native C++ | WebGPU equivalent |
|---|---|
| `GlyphEntry` packed struct (32 bytes) | `struct GlyphEntry` in WGSL `storage` buffer — exact same layout |
| `GlyphOpcode` function pointer | KLSL-compiled WGSL compute entry point |
| `register_opcode(code, fn)` table | KLSL opcode registry → dispatched by opcode ID in shader |
| Windows named file mapping (`OpenFileMappingA`) | WebGPU `GPUBuffer` (storage, mappable) |
| `status: 0=empty / 1=ready / 2=processed` polling loop | WebGPU command queue ordering — no status word needed |
| `for (i=0; i<n; i++)` sequential op loop | Single compute dispatch, all `glyphCount` entries in parallel |

The `features[16]` field (128 bits per glyph) is the semantic payload slot — on the WebGPU path this carries K'UHUL fold context: expert bone IDs, blend weights, arc depth, and node classification packed into the same 16-byte region.

The performance argument: the current native engine processes glyphs O(n) sequentially. The WGSL dispatch processes all N glyphs in one call across GPU threads — critical for paragraph-scale glyph graphs where N can be thousands.

---

## KLSL compiler architecture

### Overview

**llama.cpp runs its GPU backend on WebGPU opcodes (WGSL compute shaders).** KLSL is the transpiler layer that takes those WebGPU-style opcodes and converts them into HLSL for DirectX/DirectML dispatch on Windows — this is what allows llama's WebGPU compute graph to run on Intel HD 4600 via DirectML without Vulkan or CUDA.

KLSL (K'UHUL Level Shading Language) is the extension shading language that writes K'UHUL-specific GPU kernels (4-bone LBS bias, fold routing, pi-nary arc) without hand-coding HLSL or WGSL. It has two independent compilation paths:

```
KLSL source
  ├── klslc.exe  (two-pass, line-oriented)  →  HLSL → DirectML / D3D11 bytecode
  │    trainer/shaders/*.hlsl = compiled HLSL output (attn, softmax, bone argsort, etc.)
  └── emit_wgsl.py  (SCXQ2 IR JSON → WGSL)  →  WebGPU dispatch table
```

**Inference path** (`scratch/dml/dml_gemm_dll.cpp` → `dml_gemm.dll`): uses DirectML's high-level operator API directly. `ggml-xcfe.dll` calls `LoadLibraryA("dml_gemm.dll")` at the first MUL_MAT dispatch. This DLL carries the full KLSL forward pass kernel (GEMM, amortised D3D12+DML device, per-shape resource cache, GPU-resident weight store).

**Training path** (`trainer/shaders/*.hlsl`): HLSL shaders are the compiled output of KLSL kernels via `klslc.exe`. These are DirectML compute shaders for GPT-2 training ops (attention QK dot, softmax, bone argsort, fold route matmul, etc.).

Both paths share **SCXQ2 IR** as the canonical graph representation. KLSL source is the human-facing authoring format; SCXQ2 IR is what backends consume.

---

### KLSL syntax (glyph keywords)

KLSL uses Unicode glyph prefixes (`⟁` = U+27C1) as phase markers. The compiler is line-oriented — each line starts with one of these:

| KLSL token | Role | HLSL output |
|---|---|---|
| `⟁ shader <name>` | Shader block open | — (metadata) |
| `⟁Xul⟁` | Shader block close | — |
| `⟁Wo⟁ stage "compute"` | Set shader stage | → `[numthreads(...)]` |
| `⟁Wo⟁ threads [X, Y, Z]` | Set thread group | → `[numthreads(X,Y,Z)]` |
| `⟁Wo⟁ StructuredBuffer<T> name : register(tN)` | Buffer decl | → verbatim buffer decl |
| `[Pop <name>]` | Function open | → `void <name>(...) {` |
| `[Xul]` | Function close | → `}` |
| `⟁Wo⟁ <type> <name> = <expr>` | Local var decl | → `<type> <name> = <expr>;` |
| `⟁Sek⟁ if (cond)` | Branch | → `if (cond) {` |
| `⟁Sek⟁ return <expr>` | Return | → `return <expr>;` |
| `⟁Sek⟁ <expr>` | Assignment/call | → `<expr>;` |
| `⟁K'ayab'⟁ <type> k in 0 .. N` | For-loop | → `for (<type> k = 0; k < N; ++k) {` |
| `⟁Kumk'u⟁` | Loop close | → `}` |
| `⟁Ch'en⟁ <expr>` | Assignment target | → `// [→ <expr>]` (comment) |
| `⟁Yax⟁ <expr>` | Load target | → `// [← <expr>]` (comment) |

Plain HLSL lines (no glyph prefix) inside `[Pop]`/`[Xul]` are passed through verbatim.

**SV_ parameter injection**: the compiler scans the entry function body for `SV_DispatchThreadID`, `SV_GroupThreadID`, `SV_GroupID` and auto-generates the parameter list — no manual declaration needed.

**Example** (`examples/neural_layer.klsl` → dense forward pass):
```klsl
⟁ shader dense_layer
  ⟁Wo⟁ stage "compute"
  ⟁Wo⟁ threads [16, 16, 1]
  ⟁Wo⟁ StructuredBuffer<float>    input_buf  : register(t0)
  ⟁Wo⟁ StructuredBuffer<float>    weight_buf : register(t1)
  ⟁Wo⟁ ConstantBuffer<DenseConst> cb         : register(b0)
  ⟁Wo⟁ RWStructuredBuffer<float>  out_buf    : register(u0)

  [Pop dense_forward]
    ⟁Wo⟁ uint row = SV_DispatchThreadID.y
    ⟁Wo⟁ uint col = SV_DispatchThreadID.x
    ⟁Sek⟁ if (row >= cb.out_rows || col >= cb.out_cols) return
    ⟁Wo⟁ float acc = 0.0f
    ⟁K'ayab'⟁ uint k in 0 .. cb.in_dim
      ⟁Wo⟁ float a = input_buf[row * cb.in_dim + k]
      ⟁Sek⟁ acc += a * weight_buf[col * cb.in_dim + k]
    ⟁Kumk'u⟁
    ⟁Sek⟁ out_buf[row * cb.out_cols + col] = acc
  [Xul]
⟁Xul⟁
```

---

### SCXQ2 IR (C++ struct layout)

`scxq2_ir.h` defines the canonical graph IR used by all backends:

```cpp
struct SCXQ2IR {
    std::vector<Tensor>     tensors;   // shape + dtype + layout + storage
    std::vector<Node>       nodes;     // opcode + inputs + outputs + attrs
    std::vector<Edge>       edges;     // from_node → tensor → to_node
    std::vector<Region>     regions;   // control flow: SEQUENCE/BRANCH/LOOP/PARALLEL/FOLD
    Schedule                schedule;  // wave-based execution order
    SymbolTable             symbols;
    std::vector<Constant>   constants;
    std::vector<MemBuffer>  memory;
};
```

**RegionKind::FOLD** exists natively in the IR — K'UHUL fold scopes compile directly to a `FOLD` region.

**K'UHUL phase opcodes** (semantic annotations, not compute):
```
PHASE_POP  = 0x80   PHASE_WO   = 0x81   PHASE_SEK  = 0x82
PHASE_CHEN = 0x88   PHASE_XUL  = 0x83
```

**Backend targets listed in header:**
- WGSL, HLSL, CUDA, Metal, Vulkan, AVX2, AVX512, LLVM
- **Frontends:** KXML, XCFE, MathML, JSON, SVG-3D

---

### WGSL emitter: Python path (`emit_wgsl.py`)

The Python emitter consumes SCXQ2 IR in JSON form and emits WGSL source. IR JSON schema:

```json
{
  "version": "SCXQ2-IR/1",
  "tensors": [{"id": 0, "shape": [4,8], "dtype": "f32", "storage": "gpu", "read_only": true}],
  "nodes":   [{"id": 0, "opcode": "MATMUL", "inputs": [0,1], "outputs": [2],
               "attrs": {"M":4,"K":8,"N":3}, "workgroup_x": 256}],
  "schedule":{"passes": [{"wave": 0, "nodes": [0]}]}
}
```

Supported opcodes: `ADD`, `SUB`, `MUL`, `DIV`, `MATMUL`, `SILU`, `GELU`, `RELU`, `SOFTMAX`, `RMS_NORM`, `CROSS_ENTROPY`.

Output: `generated_shaders/kernel.wgsl` — flat storage buffers, one `@compute fn main(@builtin(global_invocation_id) gid)`.

HLSL→WGSL type mapping: `float→f32`, `float2→vec2<f32>`, `int→i32`, `uint→u32`, `float4x4→mat4x4<f32>`.
Intrinsic mapping: `frac→fract`, `lerp→mix`, `saturate→clamp`, `mad→fma`, `rsqrt→inverseSqrt`.

---

### Authoring the 4-bone LBS bias as a KLSL kernel

The existing `trainer/shaders/gpt2_kuhul_think_bias.hlsl` is hand-written HLSL. When porting to ggml-webgpu, the equivalent KLSL source would look like this (sketch):

```klsl
⟁ shader kuhul_lbs_bias
  ⟁Wo⟁ stage "compute"
  ⟁Wo⟁ threads [256, 1, 1]

  ⟁Wo⟁ StructuredBuffer<float>   think_depth_buf  : register(t0)  // [S]
  ⟁Wo⟁ StructuredBuffer<int>     bone_ids_buf     : register(t1)  // [S*4]
  ⟁Wo⟁ StructuredBuffer<float>   bone_weights_buf : register(t2)  // [S*4]
  ⟁Wo⟁ RWStructuredBuffer<float> P_buf            : register(u0)  // [S*S]
  ⟁Wo⟁ ConstantBuffer<ThinkBiasCB> cb             : register(b0)

  [Pop think_bias_main]
    ⟁Wo⟁ uint idx = SV_DispatchThreadID.x
    ⟁Sek⟁ if (idx >= cb.S * cb.S) return
    ⟁Wo⟁ uint i = idx / cb.S
    ⟁Wo⟁ uint j = idx % cb.S
    ⟁Wo⟁ float lbs = 0.0f
    // 4×4 bone overlap accumulation (expand via plain HLSL inside [Pop])
    [unroll]
    for (int ki = 0; ki < 4; ki++) {
        int bi = bone_ids_buf[i * 4 + ki];
        if (bi < 0) continue;
        float wi = bone_weights_buf[i * 4 + ki];
        [unroll]
        for (int kj = 0; kj < 4; kj++) {
            if (bone_ids_buf[j * 4 + kj] == bi)
                lbs += wi * bone_weights_buf[j * 4 + kj];
        }
    }
    ⟁Sek⟁ P_buf[idx] += cb.brain_scale * lbs
  [Xul]
⟁Xul⟁
```

The 4×4 unrolled inner loop mixes KLSL `⟁Sek⟁` and plain HLSL for the `[unroll]` attribute (compiler passes non-glyph lines through verbatim). This hybrid is valid — KLSL is a thin transpiler, not a full language.

---

### HLSLTarget defaults (`hlsl_target.h`)

| Field | Default |
|---|---|
| `shader_version` | `cs_5_0` (D3D11 compute) |
| `entry_point` | `main` |
| `thread_group_x/y/z` | 16 / 16 / 1 (override via `⟁Wo⟁ threads [...]`) |
| `use_structured_buffers` | true |
| `use_half_precision` | false |
| `enable_debug_info` | false |

Register allocators in `HLSLContext`: `next_register_t` (SRV t#), `next_register_u` (UAV u#), `next_register_b` (cbuffer b#), `next_register_s` (sampler s#). The compiler fills these during Pass 1 buffer parsing; buffers declared with explicit `register(tN)` in KLSL source skip the allocator.

---

## Atomic Block DOM — per-model manifests

Each model has an `atomic.manifest.json` that drives `kuhul_engine.exe --Atomic.DOM <manifest>`.
This is the khanary equivalent of llama's GGUF Jinja chat template — it binds the model's
NPC persona, KXML chat template, sampling params, micronaut routing, and provider endpoint
into a single declarative file.

### Manifest locations

| Model | Manifest | Size | GPU? | Purpose |
|-------|----------|------|------|---------|
| `from_zero_v0.6` | `models/from_zero/atomic.manifest.json` | 475 MB st | yes (Q8 gguf) | KUHUL domain chat, KXML tool calls, LoRA adapter |
| `khanary-kxml-v0.5.0` | `models/khanary-kxml-v0.5.0/atomic.manifest.json` | — | yes | Trained-in T_<NAME> tool-call agent |
| `gpt2-xl-tools-mcp` | `models/gpt2-xl/atomic.manifest.json` | 1668 MB | **yes** (fits 1792 MB) | GPT-2 XL Q8 MCP-baked, resident GPU tool agent |
| `lfm2.5-1.2b-instruct` | `models/lfm2-1b/atomic.manifest.json` | 1188 MB | **yes** | LFM2 SSM, 128K context, native tool calls |
| `gemma-3-1b-it-qat` | `models/gemma-3-1b/atomic.manifest.json` | 687 MB | **yes** | Fastest GPU-resident model, QAT quality |
| `gemma-3-4b-it` | `models/gemma-3-4b/atomic.manifest.json` | ~2.7 GB | no (CPU) | Downloading. Better quality, CPU inference |
| `gemma-4-e2b-it` | `models/gemma-4-e2b/atomic.manifest.json` | 3.2 GB + 941 MB mmproj | no (CPU) | Multimodal vision, 4.2 GB total |
| `gpt-oss-20b` | `models/gpt-oss/atomic.manifest.json` | 11.28 GB | no (CPU) | Phase 4 distillation teacher |

#### Batch 2 — additional LM Studio + ASX models

| Alias | Manifest | Size MB | GPU? | Notes |
|-------|----------|---------|------|-------|
| `phi3-mini` | `models/phi3-mini-4k/` | 2282 | no (CPU) | Phi-3 mini Q4 micronaut-tagged, tool-call agent |
| `dolphin` | `models/dolphin-phi2/` | 1844 | no (CPU) | Dolphin 2.6 Phi-2 Q5_K_S, uncensored creative |
| `gemma-1b-q8` | `models/gemma-3-1b-q8/` | 1020 | **yes** | Gemma 3 1B Q8_0 unsloth, higher fidelity than QAT |
| `qwen-1b8` | `models/qwen-1b8-chat/` | 3504 F16 st | no | Safetensors — must convert: `convert_hf_to_gguf.py → q8` |
| `qwen-story` | `models/qwen25-05b-story/` | 644 | **yes** | Qwen 2.5 0.5B — **story/creative mode only**, small system prompt, small repeat_penalty; hallucinates on factual tasks |
| `mgguf-gpt2` | `models/mgguf-gpt2-2expert/` | 1408 | **yes** | GPT-2 2-expert MoE ASX mgguf — moe_gguf_runtime.exe |
| `mgguf-qwen` | `models/mgguf-qwen-1expert/` | 1862 | no (CPU) | Qwen 1-expert MoE ASX mgguf — moe_gguf_runtime.exe |

### Launch

```bat
AtomicDOM                   :: from_zero default
AtomicDOM kxml              :: KXML tool-call agent
AtomicDOM gpt-oss           :: GPT-OSS teacher
AtomicDOM path\to\manifest  :: explicit path
```

Reference implementation: `C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\AtomicChat.cmd` and
`C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\AtomicDOM\` (frame/body/feed/header/menu/footer manifests).

### Schema fields (key)

| Field | Purpose |
|-------|---------|
| `model.gguf` / `model.safetensors` / `model.lora` | Weight paths |
| `chat_template` | Role tokens + Jinja path + tool_call / reasoning open/close |
| `sampling` | temperature, repeat_penalty, stop tokens |
| `app.npc` | System prompt + persona rules |
| `app.provider.endpoint` | kuhul_engine at port 17474 |
| `app.micronauts` | Per-intent micronaut map |
| `app.distillation` | (gpt-oss only) student pointer + oss_distillation.py params |

---

## json_runtime.exe — GPU operations

`json_runtime.exe` (port 8787) is not just a hosting/file-manager API — it also exposes GPU compute through its XCFE stdlib.

### GPU verbs (XCFE stdlib `gpu` capability)

| Verb / `@fn` | C++ handler | What it does |
|---|---|---|
| `@fn: "dispatch"` | `compile_gpu_kernel()` | Compile HLSL shader source at runtime via `D3DCompiler_47.dll`. Accepts `@source` (HLSL string), `@entry` (default `"main"`), `@profile` (default `"cs_5_0"`). Returns `{compiled, bytecode_bytes, profile, entry}`. Currently compile-only; device dispatch is the next step. |
| `@fn: "matmul"` / `tensor.matmul` / `tensor.gemm` | `tensor_runtime()` → `matmul()` | Matrix multiply via DirectML GEMM. Loads `dml_gemm.dll` from `..\\ggml\\dml_gemm.dll` (KLSL forward pass DLL). Falls back to CPU triple-loop if DLL unavailable. Returns XJSON tensor with `"backend": "khanary-directml"` or `"cpu-fallback"`. |
| `@fn: "relu"` / `@fn: "softmax"` | `tensor_runtime()` → `unary()` | Element-wise unary ops on XJSON tensors (CPU-side). |
| `@fn: "alloc"` | `alloc_tensor()` | Allocate a zero-filled XJSON tensor of given `@shape`. |
| `tensor_register` / `tensor_get` / `tensor_list` | `registry_operation()` | Named tensor registry — store, retrieve, and enumerate tensors across operations within a session. |

XCFE stdlib declares these under the `gpu` capability block:
```json
"gpu": ["@gpu.dispatch", "@gpu.buffer.write", "@gpu.buffer.read"]
```
`@gpu.buffer.write` and `@gpu.buffer.read` are declared in the manifest but not yet implemented in C++.

### XJSON tensor format

```json
{
  "@type": "xjson/tensor",
  "shape": [4, 8],
  "dtype": "f32",
  "device": "cpu",
  "layout": "row_major",
  "phase": "Pop",
  "data": [...]
}
```

`tensor.matmul` example (via XCFE `function_call`):
```json
{
  "@fn": "function_call",
  "@name": "tensor.matmul",
  "@args": { "A": { "@type": "xjson/tensor", "shape":[2,3], "data":[...] },
              "B": { "@type": "xjson/tensor", "shape":[3,4], "data":[...] } }
}
```

### DLL paths

`json_runtime.exe` lives at `C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\json_runtime.exe`, run from `bin/json-runtime/`. It loads:

| DLL | Path (relative to runtime working dir) |
|-----|----------------------------------------|
| `dml_gemm.dll` | `..\\ggml\\dml_gemm.dll` → `bin/ggml/dml_gemm.dll` |
| `DirectML.dll` | `..\\ggml\\DirectML.dll` → `bin/ggml/DirectML.dll` |

Override via `KHANARY_DML_GEMM` env var (same as `ggml-xcfe.dll`). The `bin/ggml/` directory is already populated from the ggml subproject build output.

### gpu.manifest.json policy

```json
{
  "@gpu": {"policy": "D3D11_1/WebGL2/WebGPU/OpenCL providers are declared, measured, then admitted by XCFE/KUHUL.", "fallback": "cpu"},
  "@d3d11_1": {"primary": true, "shader_model": "cs_5_0"},
  "@webgpu": {"optional": true},
  "@opencl": {"optional": true}
}
```

D3D11_1 (cs_5_0) is primary — matches the HD 4600's feature level 11_1. WebGPU/OpenCL are optional admittable providers.

---

## Scratch — standalone GPU verification harnesses

`scratch/` contains proven standalone test programs that validate the full inference pipeline
independently from the main trainer. Key assets:

| File | What it proves |
|------|---------------|
| `scratch/infer/gpt2_infer_run.cpp` | **Full-model GPT-2** on HD 4600: embed→[block×N]→ln_f→lm_head, KV cache, greedy decode. Matches CPU oracle. |
| `scratch/infer/gpt2_infer_run.exe` | Pre-built binary — runs immediately |
| `scratch/block/gpt2_block_run.cpp` | **Single transformer block** GPU chain: ln1→qkv→attn→proj→res→ln2→fc→gelu→proj→res |
| `scratch/block/gpt2_block_run.exe` | Pre-built binary |
| `scratch/dml/` | DirectML attention experiments + amortization baseline |
| `scratch/lora_smoke.gguf` | LoRA smoke test GGUF artifact |
| `scratch/fz_test.err` | `from_zero_v0.1.f32.gguf` serving at 8181: 9 tok/s on DirectML, load in 3.8s |
| `scratch/knu_*.hlsl` | KNU glyph kernel shaders (attn/embed/gelu/layernorm/matmul/skin/xform) |

The `scratch/infer/` KV-cache implementation is the reference for the full inference path:
- Prefill: embed prompt → N blocks (populates K/V cache) → lm_head → argmax
- Decode: 1-row block per token, online softmax over cached K/V

---

## Open decisions (NOT yet implemented — need confirmation)

### Decision A — "mesh shader" interpretation
**Blocker**: HD 4600 is D3D11 feature level 11_1. D3D12 mesh shaders (SM 6.5) are physically impossible on this GPU. The `MeshletMS.hlsl` files in the tree are sample rendering code.

**Proposed interpretation**: Add a `cs_5_0` compute pass that applies K'UHUL geometric weight tensor bias to attention logits (P_buf) between QK dot and softmax. This requires splitting `gpt2_attn_fwd.hlsl` into two kernels:
1. `gpt2_attn_qk.hlsl` — QK dot → scale → causal mask → write P_buf (no softmax)
2. `gpt2_attn_kuhul_bias.hlsl` — apply geometric bias `g[0..4]` per head to P_buf
3. `gpt2_attn_softmax_v.hlsl` — softmax → V aggregation

New per-layer learned parameter: `kuhul_bias[L, H, 5]` (5 geometric coefficients × L layers × H heads).

**Confirm before implementing.**

### Decision B — DML GEMM bridge
**Status**: `dml_gemm_bt_f32` (from `dml_gemm.dll` / `ggml-xcfe.dll`) takes **CPU float pointers**. The trainer keeps all data in D3D11 GPU buffers.

**Cost of wiring in**: D3D11→CPU readback → DML GEMM (D3D12 device, upload/compute/readback) → CPU→D3D11 upload, per matmul call. This is **slower** than the existing on-GPU `cs_matmul_fwd_` shader. It is a correctness/parity proof only, not a perf improvement.

**Options**:
- Skip DML GEMM in the trainer (it already has GPU matmul); wire it only if switching to D3D12.
- Add `GPT2_DML_GEMM=1` env flag that routes matmul through the CPU bridge (parity test mode).

**Confirm before implementing.**

---

## Training curriculum status

| Phase | Data | Steps | LR | Output | Status |
|---|---|---|---|---|---|
| 0a — vacuum | `vacuum_seed.bin` (50K × 64) | 150 | 1e-3 | `v0.2_vacuum` | DONE — loss floor 0.00322 |
| 0b — vacuum+LBS | same | 200 | 5e-4 | `v0.3_vacuum_bias` | DONE — loss floor 0.00066 |
| 1 — header corpus | `tokens_hdr_big.bin` (200K × 64) | 2000 | 3e-4 | `v0.4_phase1` | DONE 2026-08-04 — antigravity→1.0 at step ~1200 |
| 2 — KUHUL corpus | `kuhul_tokens_kuhul.bin` (462 MB) | 3000 | 1e-4 | `v0.5_phase2` | **DONE** 2026-08-04 |
| 3 — merge | v0.4 + v0.5 | — | — | `v0.6_merged` | **DONE** 2026-08-04 — SLERP α=0.6, 148 tensors |
| 4 — distillation | GPT-OSS teacher → LoRA | 500 | 1e-4 | `v0.6_lora.safetensors` | **pending** — see DISTILLATION.md |

### Safetensors repair note

The trainer wrote all non-embedding tensors with empty shapes `"shape":[]`. v0.4 and v0.5 were
repaired using `tools/repair_safetensors.py` (borrows shapes from v0.1_folded, validates output).
Repaired files: `v0.4_phase1.repaired.safetensors`, `v0.5_phase2.repaired.safetensors`.
Fix the trainer's save path to write proper shapes to prevent this in future phases.

### Phase 2 command

```powershell
$env:GPT2_ADAPTIVE_CLIP = "1"
$env:GPT2_THINK_BIAS    = "1"
$env:GPT2_BRAIN_EXPERTS = "C:\Users\canna\_khanary_inspect\brain2\experts_kuhul.bin"
cd C:\Users\canna\_khanary_inspect\trainer\build\Release
.\gpt2_trainer.exe `
  --model    "C:\Users\canna\_khanary_inspect\models\from_zero\from_zero_v0.4_phase1.safetensors" `
  --data     "E:\data\kuhul_tokens_kuhul.bin" `
  --out      "C:\Users\canna\_khanary_inspect\models\from_zero\from_zero_v0.5_phase2.safetensors" `
  --steps 3000 --batch 4 --block 64 --lr 1e-4 --save-every 200
```

### Phase 3 — model merge

`tools/merge_models.py` — SLERP / linear merge of two same-arch safetensors checkpoints.

```powershell
python tools/merge_models.py `
  models/from_zero/from_zero_v0.4_phase1.safetensors `
  models/from_zero/from_zero_v0.5_phase2.safetensors `
  models/from_zero/from_zero_v0.6_merged.safetensors `
  --alpha 0.6 --method slerp
```

- `--alpha 0.0` = pure A (v0.4 general), `--alpha 1.0` = pure B (v0.5 KUHUL)
- `--alpha 0.6` recommended: keeps general language fluency while biasing toward KUHUL fold patterns
- Vocab mismatch handling built-in: if models differ in wte/lm_head vocab dim, shared rows are interpolated, extra KUHUL rows from B are appended verbatim
- SLERP respects the vacuum-shaped manifold geometry; linear interpolation is also supported via `--method linear`
- Prints weight-norm sanity table after saving

**Do NOT chain in earlier checkpoints (v0.1, v0.2, v0.3).** Those are intermediate stages
that Phase 1 already subsumed. SLERP between a mature model and its own early draft pulls
the result backward. Use them only as fallback recovery points if v0.5 proves overfit.

---

## Micronaut sampling contracts

Each micronaut carries its own sampling parameters — callers pick a micronaut by name and
the dispatch layer injects the right values into the llama-server request body.

```json
// tool_call.micronaut.json
{
  "name": "tool_call",
  "sampling": {
    "repeat_penalty": 1.0,
    "temperature": 0.1,
    "stop": ["</tool_call>"]
  }
}

// chat.micronaut.json
{
  "name": "chat",
  "sampling": {
    "repeat_penalty": 1.3,
    "temperature": 0.8,
    "repeat_last_n": 64
  }
}
```

Verified: llama-server `/completion` accepts `repeat_penalty`, `repeat_last_n`, `temperature`,
and `stop` per-request, overriding server-level defaults. A model that doesn't repeat tool-call
tokens (JSON brackets, `"name"`, `"arguments"`) doesn't need penalty — penalty=1.0 is the
neutral pass-through. A model in free-text chat mode benefits from penalty=1.3 to break loops.
The micronaut definition is the right place to encode this, not the caller.

---

## Status

- [x] `trainer/` folder created with source + shaders
- [x] `trainer/d3d11_engine.h/.cpp` — trainer-specific D3D11 (no XVM dependency)
- [x] `trainer/CMakeLists.txt` — standalone CMake with FetchContent nlohmann_json
- [x] Dataset links captured (above) — full `E:\data` inventory
- [x] Model registry captured — 7 models documented
- [x] `tools/gen_kuhul_training.py` — synthetic corpus generator (350,388 examples)
- [x] `tools/kuhul_dataset_validator.py` — validate + compile π-KUHUL structured records
- [x] `tools/extend_vocab.py` — patch checkpoint wte [50260,768] → [50270,768]
- [x] `tools/merge_models.py` — SLERP/linear merge of two same-arch checkpoints (vocab-mismatch-aware)
- [x] `tokenizer_config.json` — KUHUL token IDs 50260–50269 at repo root
- [x] `pi_kuhul/` — KuhulPhysics.h, SphericalGeometryAVX2.h, DirectXMathAVX2.h, Fold2DCompiler.h
- [x] `trainer/shaders/gpt2_kuhul_think_bias.hlsl` — π-nary geodesic arc + 4-bone LBS attention bias
- [x] 7 missing shaders copied from v0.4.0 into `trainer/shaders/`
- [x] Build: `gpt2_trainer.exe` 279 KB, compiled MSVC 19.44, running on HD 4600
- [x] **Re-tokenized** `kuhul_synthetic.jsonl` → `E:\data\kuhul_tokens_kuhul.bin` (946,503 seqs × 128; KUHUL tokens present)
- [x] **4-bone LBS upgrade**: replaced `seq_expert_buf_` with `seq_bone_ids_buf_` + `seq_bone_weights_buf_`; rewritten `buildThinkDepth()`; updated shader to compute `lbs_overlap(i,j)`
- [x] Phase 0a vacuum — DONE (loss floor 0.00322, `v0.2_vacuum`)
- [x] Phase 0b vacuum+LBS — DONE (loss floor 0.00066, `v0.3_vacuum_bias`)
- [x] Phase 1 header corpus — DONE 2026-08-04 (`v0.4_phase1`, antigravity→1.0 at step ~1200)
- [x] Phase 2 KUHUL corpus — DONE 2026-08-04 (`v0.5_phase2`, 3000 steps)
- [x] Phase 3 merge — DONE 2026-08-04 (`v0.6_merged`, SLERP α=0.6, 148 tensors)
- [ ] Phase 4 distillation — GPT-OSS → LoRA adapter (`tools/oss_distillation.py`, 500 steps)
- [ ] Decision A: split `gpt2_attn_fwd.hlsl` → QK / think-bias / softmax+V
- [ ] Decision B: DML GEMM bridge mode
- [x] KUHUL APPS studio — stack service + gateway MCP client + canvas route (`/chat/[id]/canvas`)
- [x] KUHUL APPS studio — project manifest lifecycle (`projects.service.ts`: file-manager init/read/write + port allocation; live-verified 2026-08-05; deploy/mount in item 5)
- [ ] KUHUL APPS studio — Build / Deploy / Open actions + sidebar entry
- [x] KUHUL APPS studio — task routing (studio-task-router + task-engine schema + AtomicDOM model registry)
- [x] `llama-build.bat` — full rebuild sequence: (1) stale UI purge (dist/ + .ui-stamp + ui.cpp/h), (2) npm run build → tools/ui/dist/, (3) KLSL forward pass recompile via vcvars64 + cl.exe → dml_gemm.dll from scratch/dml/dml_gemm_dll.cpp, (4) cmake --build llama-server, (5) GPU DLL deploy: dml_gemm.dll + DirectML.dll → build/bin/Release/. Usage: `llama-build` / `llama-build full` / `llama-build clean`

### Brain expert routing (GPT2_THINK_BIAS=1)

**How it works (3 modifiers stacked, post-softmax on P_buf):**
1. π-nary arc: `sin(depth/2)²` — tokens inside `<THINK>…</THINK>` get geodesic-distance-weighted attention boost
2. KuhulPhysics scale: `antigravity_scale` (0.1→1.0) — strengthens as training stabilises
3. Brain expert cluster: `brain_experts_[tok % 30628]` ∈ [0,60] — same Delaunay cluster → co-attention boost

**Path:** `brain2/experts.bin` relative to build dir, or `$env:GPT2_BRAIN_EXPERTS=<full_path>`.

**Convergence note:** loss ~6.0 at step 116-117 on `kuhul_tokens_kuhul.bin` (487 MB, 121M tokens) is **expected** — the model has seen 60K tokens (0.05% of data). CPU hit 0.00 overnight on the small `tokens_hdr.bin` (12 MB, repeating). To verify GPU backward works: run a quick overfit with `--data E:\data\kuhul_tokens.bin --steps 200 --lr 1e-3` on a tiny slice, or use `tokens_hdr.bin`.

---

## GPT-OSS Distillation — Phase 4

Goal: use `gpt-oss-20b-MXFP4.gguf` (teacher, served at port 17474) to distil knowledge into
a LoRA adapter for `from_zero_v0.6_merged`. The adapter captures KUHUL domain knowledge from
the large model without full fine-tuning of the base weights.

### GPT-OSS model paths

| Format | Path | Size |
|--------|------|------|
| GGUF (MXFP4) — teacher server | `C:\Users\canna\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf` | 11.28 GB |
| HF sharded (24 layers, xshard) | `E:\models\GPT-OSS\hf\layer_00` … `layer_23` | ~12 GB total |
| HF model config | `E:\models\GPT-OSS\hf\model_config.json` | arch: hidden=2880, heads=64, kv_heads=8, experts=32, top_k=8, vocab=200064 |

> **GPU note**: MXFP4 GGUF is 11.28 GB — exceeds HD 4600's 1792 MB ceiling. Run with `-ngl 0`
> (CPU inference only) when serving as the distillation teacher. Throughput is low but sufficient
> for generating 500 distillation completions. The xshard format at `E:\models\GPT-OSS\hf\` is
> the kuhul engine's hot-swap layer format for streaming individual layers on-demand.

### Strategy: response distillation

1. Send KUHUL-domain prompts to GPT-OSS teacher via `/v1/chat/completions`
2. Tokenize `(prompt + completion)` as a full sequence
3. Run student forward pass (from_zero_v0.6)
4. Cross-entropy loss on completion tokens (prompt tokens masked)
5. Backprop only through LoRA adapter weights (A and B matrices); base weights frozen

### LoRA adapter design

```
W_effective = W_base + B @ A
  A: [rank, in_dim]  — initialized N(0, 0.02/rank)
  B: [out_dim, rank] — initialized zeros
```

Applied to: `c_attn.weight`, `c_proj.weight`, `mlp.c_fc.weight`, `mlp.c_proj.weight`
for all 6 (or 12) transformer layers. Default rank=8.

### Run command

```powershell
cd C:\Users\canna\_khanary_inspect
# Start kuhul_engine first (teacher):
# node dist/khanary-server/kuhul-server.cjs  (auto-starts engine)

python tools/oss_distillation.py \
  --student  models/from_zero/from_zero_v0.6_merged.safetensors \
  --out      models/from_zero/from_zero_v0.6_lora.safetensors \
  --rank     8 \
  --steps    500 \
  --lr       1e-4 \
  --engine   http://127.0.0.1:17474
```

If engine is unreachable: falls back to self-distillation (student teaches itself — useful for
adapter shape validation).

### OSS-distillation DLL (future)

An `oss-distillation.dll` built against the `pi_kuhul/` C++ headers would accelerate the
distillation forward passes on DirectML. The Python script above validates the pipeline first;
once the LoRA architecture is confirmed correct, wrap in a DLL for GPU-native speed.

Does NOT need to be a new LoRA format — standard rank decomposition, GGUF-compatible LoRA or
plain safetensors adapter both work.

---

## KUHUL APPS — App Generation Studio

`khanary-llama-build/llama.cpp/tools/ui/` becomes **KUHUL APPS**: an AI app generation studio
powered by `kuhul_engine.exe`. Users chat to describe and build apps; each conversation is
a project that can be opened, renamed, and edited.

### Architecture

```
Left sidebar            Center canvas              Right chat
─────────────────       ─────────────────────      ────────────────────
Projects list           Live preview iframe        kuhul_engine chat
(conversations)         (generated HTML/code       AI App Assistant
Theme switcher          from latest msg)
Settings                Export / Publish btns
```

### Database mapping (no schema changes needed)

| IndexedDB table | KUHUL APPS concept |
|---|---|
| `conversations` | Projects |
| `messages`      | Project generation history |
| `conversation.name` | Project name (auto-generated from first prompt) |
| `message.content`   | App code / chat / generated HTML |

Every conversation IS a project. Chat messages are the generation timeline.
The canvas area renders the latest assistant message that contains a `<!DOCTYPE html>` or
triple-backtick HTML block in an isolated `<iframe srcdoc="...">`.

### Theme system

| Theme | Default | CSS variables |
|---|---|---|
| Dark (KUHUL default) | YES | `--bg: #0d0d0d`, `--sidebar: #1e293b`, `--accent: #6366f1` |
| Light | no | `--bg: #f8fafc`, `--sidebar: #f1f5f9`, `--accent: #4f46e5` |
| Kuhul indigo | no | `--bg: #1e1b4b`, `--sidebar: #312e81`, `--accent: #a5b4fc` |

Switched via `VITE_PUBLIC_APP_NAME='KUHUL APPS'` + CSS class on `<html>`.

### Key files changed

| File | Change |
|---|---|
| `src/lib/assets/logo.svg` | Replaced llama logo with KUHUL K glyph (indigo gradient) |
| `src/lib/constants/app.ts` | APP_NAME = 'KUHUL APPS' (via `VITE_PUBLIC_APP_NAME`) |
| `src/lib/constants/ui.ts` | 'New chat' → 'New project' |
| `src/lib/constants/title-generation.ts` | FALLBACK: 'New Project' |
| `SidebarNavigation.svelte` | 'Rename conversation' → 'Rename project', delete dialog updated |
| `SidebarNavigationActions.svelte` | Search placeholder: 'Search projects...' |
| `ChatScreenGreeting.svelte` | Greeting changed to KUHUL APPS |
| `.env` | `VITE_PUBLIC_APP_NAME='KUHUL APPS'`, server origin = port 17474 |

### Studio implementation status

The three-panel studio (left sidebar / center canvas / right chat) is implemented as an
overlay route on the existing chat `[id]` route.

Route: `(chat)/chat/[id]/canvas`

| Piece | File | Status |
|---|---|---|
| Stack status + gateway discovery | `src/lib/services/kuhul-stack.service.ts` | DONE — probes gateway (8764), engine (17474), json_runtime (8787); never throws |
| Gateway MCP client | `src/lib/services/gateway-mcp.service.ts` | DONE — `POST /mcp` JSON-RPC `tools/call` for kuhul_task_boss / kuhul_json_runtime / kuhul_wwa_host / kuhul_forge |
| HTML extraction | `src/lib/utils/extract-html-doc.ts` | DONE — newest assistant message, ```html fence or raw `<!DOCTYPE html>` |
| Canvas component | `src/lib/components/app/chat/canvas/CanvasPreview.svelte` | DONE — sandboxed iframe, Preview/Live tabs, refresh / open / copy / export |
| Studio route | `src/routes/(chat)/chat/[id]/canvas/+page.svelte` | DONE — left project + stack panel, center canvas, right studio chat (send → TaskEngine with chat fallback) + plan checklist |
| Live chat input + model selector | `canvas/+page.svelte` right panel + `src/lib/constants/atomicdom-models.ts` | DONE (2026-08-05) — textarea (Enter sends, Shift+Enter newline), GPU/CPU grouped `<select>`, 15 AtomicDOM aliases |
| TaskEngine schema | `src/lib/types/task-engine.ts` | DONE — verb/targetKind re-export, TaskPlanItem / TaskPlan / StudioTaskRequest / StudioTaskResult |
| Task router | `src/lib/services/studio-task-router.ts` | DONE — classifyPrompt → taskBoss → parsePlanMarkdown (`- [ ]`/`[x]`/`[>]`/`[!]`/`[~]`, indented subtasks); verb badge via classify() |
| Project manifest lifecycle | `src/lib/services/projects.service.ts` | DONE — file-manager init/list/read/write/stat, manifest read/write, port allocation (8800-8899), `/api/sidecars` hosted probe; live-verified vs json_runtime (8787). Deploy/mount → item 5 |
| Build / Deploy / Open actions + sidebar entry | — | pending — item 5 |

Verified: `npm run check` → svelte-check 0 errors / 0 warnings.

Note: kuhul-server's `kuhul_task_boss` handler currently reads only `verb` /
`target_kind` / `tasks` — the studio's `prompt` and `modelAlias` fields are accepted
by the router but not yet forwarded into TaskEngine admission. Extend
`kuhul-server.cjs` when the engine's admission contract needs them.

### Stack backend

- `.env` sets `VITE_PUBLIC_SERVER_ORIGIN='http://localhost:17474'` — chat streams directly to
  kuhul_engine (OpenAI-compatible `/v1/chat/completions`).
- kuhul-server (port 8764) is the gateway: micronauts, MCP tools (`kuhul_task_boss` with the
  verbs `task.plan` / `app.create` / `app.inspect` / `build.game` / `build.website` /
  `build.program` / `build.micronaut`), engine auto-start + health, forge.
- json_runtime (port 8787) is the hosting API: per-project `server.manifest.json` mounted as
  a sidecar declares folder + port; ports are managed by the runtime.
- WWAHost.exe launches generated WWA apps; kuhul_engine creates the app files (WWA.dll-backed)
  that make each project runnable.

Live state (2026-08-05): json_runtime running on 8787; kuhul_engine (17474) and kuhul-server
(8764) come up together via `node dist/khanary-server/kuhul-server.cjs` (auto-starts the engine).

---

## PRIMEOS — Stack Management Layer

`C:\Users\canna\_khanary_inspect\desktop\PRIMEOS\bin\Release\net8.0-windows\PRIMEOS.exe`

PRIMEOS is a .NET 8 Windows desktop app that acts as the management UI for the full NNC-K
/ KUHUL APPS stack. Long-term goal: all services (kuhul_engine, JSON runtime, MCP tools,
micronaut factory, kuhul-server) can be started, stopped, monitored, and updated from PRIMEOS.

### Integration surface

| Component | How PRIMEOS manages it |
|---|---|
| `kuhul_engine.exe` | Start/stop via process API; port 17474 health check |
| `kuhul-server.cjs` | Start/stop; reads `.kuhul-server.port`; shows bound port |
| `json_runtime.exe` | Run programs; view output; update manifests |
| `MicrosoftSDK.ps1` | Invoke commands; show tasklist; run persona/manifest |
| `WWAHost.exe` | Launch apps; manage WWA manifests |
| MCP tools (all 10) | Invoke any MCP tool from a native Windows UI |
| micronaut factory | Create/view/forge micronauts; show auto-created list |
| LoRA distillation | Trigger `oss_distillation.py`; show training progress |

### khanary ↔ JSON runtime

khanary (via kuhul-server MCP tool `kuhul_json_runtime`) can already run and update JSON runtime
programs. PRIMEOS adds a visual program editor and manifest browser on top of the same runtime.
Both use `json_runtime.exe` at `bin/json-runtime/` as the execution engine — no duplication.

### Pending

- Define the PRIMEOS ↔ kuhul-server API contract (REST or named pipe)
- Wire kuhul-server `GET /kuhul/engine/status` into PRIMEOS health dashboard
- Add PRIMEOS as a startup item so it auto-starts kuhul-server and kuhul_engine on login