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

KLSL (K'UHUL Level Shading Language) is the extension shading language that writes K'UHUL-specific GPU kernels (4-bone LBS bias, fold routing, pi-nary arc) without hand-coding HLSL or WGSL. It has two independent compilation paths:

```
KLSL source
  ├── klsl_compiler.cpp  (two-pass, line-oriented)  →  HLSL → D3D11 bytecode
  └── emit_wgsl.py       (SCXQ2 IR JSON → WGSL)    →  WebGPU dispatch table
```

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
| 2 — KUHUL corpus | `kuhul_tokens_kuhul.bin` (462 MB) | 3000 | 1e-4 | `v0.5_phase2` | **RUNNING** 2026-08-04 |
| 3 — merge | v0.4 + v0.5 | — | — | `v0.6_merged` | pending Phase 2 completion |

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
- [ ] Phase 2 KUHUL corpus — RUNNING (3000 steps, lr=1e-4, `v0.5_phase2`)
- [ ] Phase 3 merge — pending Phase 2 (`merge_models.py --alpha 0.6`, `v0.6_merged`)
- [ ] Decision A: split `gpt2_attn_fwd.hlsl` → QK / think-bias / softmax+V
- [ ] Decision B: DML GEMM bridge mode

### Brain expert routing (GPT2_THINK_BIAS=1)

**How it works (3 modifiers stacked, post-softmax on P_buf):**
1. π-nary arc: `sin(depth/2)²` — tokens inside `<THINK>…</THINK>` get geodesic-distance-weighted attention boost
2. KuhulPhysics scale: `antigravity_scale` (0.1→1.0) — strengthens as training stabilises
3. Brain expert cluster: `brain_experts_[tok % 30628]` ∈ [0,60] — same Delaunay cluster → co-attention boost

**Path:** `brain2/experts.bin` relative to build dir, or `$env:GPT2_BRAIN_EXPERTS=<full_path>`.

**Convergence note:** loss ~6.0 at step 116-117 on `kuhul_tokens_kuhul.bin` (487 MB, 121M tokens) is **expected** — the model has seen 60K tokens (0.05% of data). CPU hit 0.00 overnight on the small `tokens_hdr.bin` (12 MB, repeating). To verify GPU backward works: run a quick overfit with `--data E:\data\kuhul_tokens.bin --steps 200 --lr 1e-3` on a tiny slice, or use `tokens_hdr.bin`.
