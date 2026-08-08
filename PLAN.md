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
| `from_zero/` | `from_zero_v0.6_merged.gguf` (654 MB) + `from_zero_v0.6_kuhul.gguf` (654 MB) | v0.6 merged checkpoint as GGUF (149 tensors, gpt2 arch). `v0.6_kuhul.gguf` (2026-08-07) is the KUHUL-vocab variant from `tools/gpt2_kuhul_to_gguf.py` — preserves all 50270 embedding rows + KUHUL special tokens from `tokenizer_config.json`; this is the GGUF that matches the 50270-vocab tokenizer and is referenced by `active-model.json` + `atomic.manifest.json`. Safetensors source at `models/from_zero/from_zero_v0.6_merged.safetensors`.
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

### The XCFE GL seam (`xcfe_gl_ops.dll`) — full F32 tensor set on OpenGL

2026-08-06: the MUL_MAT-only framing above is superseded. The bridge now has a second GPU
seam that covers the **whole F32 tensor set**, on the OpenGL backend — the route the HD 4600
driver actually supports (wgpu_native GL backend, not D3D12):

- `bridges/ggml-xcfe/xcfe_gl_ops.cpp` → `xcfe_gl_ops.dll` (no link dependencies; lazily loads
  `wgpu_native-release.dll`, GL backend preferred, then D3D11, then D3D12)
- Seam contract v2: `xcfe_gl_run(op, inputs, n_inputs, out, ne_out, n_dims, params, n_params)`
  — params carry op scalars: K for mul_mat; eps (norm/rms_norm); scale (soft_max);
  n_dims/freq_base/freq_scale (rope); dim/ne0a/ne1a (concat); table rows (get_rows)
- 17 WGSL kernels staged: mul_mat, get_rows, norm, rms_norm, gelu, gelu_quick, silu, relu,
  tanh, sigmoid, add, sub, mul, soft_max, rope, concat, cpy — one fixed binding layout
  (uniform params + 2 storage inputs + 1 storage output), readback via a separate map buffer
- MUL_MAT verified through the seam on this rig (max err 4.768e-07):

```
[ggml-xcfe] tensor dispatch: KHANARY GL ops (GPU)
[xcfe_gl_ops] device up: OpenGL (wgpu_native GL backend)
MUL_MAT [K=64,N=32,M=8]  (256 elems)
  sample: C[0] cpu=-0.579257 xcfe=-0.579257   C[255] cpu=-0.071912 xcfe=-0.071912
  max abs err = 4.768e-07   (scale-normalized 2.188e-07)
XCFE computes MUL_MAT, matches ggml CPU: YES
```

Remaining: per-op verification of the other 16 kernels (op-level test harness), then an
end-to-end F32 model run (`models/from_zero` f32) with `-ngl 999` on the GL backend. Q8 /
GPT-OSS models won't route through XCFE (the claim is F32-src0 only).

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
| `app.provider.endpoint` | kuhul_engine at port 17480 |
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
| `scratch/fz_test.err` | `from_zero_v0.1.f32.gguf` serving at 8181: 9 tok/s on DirectML, load in 3.8s (historical log — the v0.1 GGUF it loaded no longer exists in `models/from_zero/`; superseded by the v0.6 GGUFs) |
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
| 4 — distillation | GPT-OSS teacher → LoRA | 400 (cloud, 128-tok) | 1e-4 | `v0.6_lora.safetensors` | **DONE 2026-08-07** — real GPT-OSS 120B cloud teacher via Ollama (`gpt-oss:120b-cloud`). 200 fresh steps + 200 resumed steps, rank=8, teacher-tokens=128, best_loss=7.0734. Micronaut/runtime context supplies stack-specific details at inference, so the base model does not need to memorize the entire KUHUL stack. JSONL logs in `logs/distillation_*.jsonl` include per-prompt atomic blocks. |

### Phase 4 distillation notes

**Intended teacher:** GPT-OSS 120B (`gpt-oss:120b-cloud`) served through local Ollama at `http://127.0.0.1:11434`, which forwards to the cloud model.  
**Run completed:** 2026-08-07, 400 effective teacher steps (200 fresh + 200 `--resume`), rank=8, lr=1e-4, 128 teacher tokens per prompt.  
**Result:** `models/from_zero/from_zero_v0.6_lora.safetensors` — 4.5 MB, 96 tensors, best_loss=7.0734.  
**Scope:** The LoRA distills GPT-OSS's style/tool-calling/KUHUL-domain response patterns from the 15 kuhul prompts in `distill_prompts.txt`. It does **not** need to memorize every stack detail — the Atomic DOM manifest, micronauts, and runtime context inject stack-specific facts at inference time.

**Improvements made during this session:**
- `tools/oss_distillation.py` now uses Ollama `/api/chat` with `reasoning: false` and 128-token targets, plus retries and an in-memory completion cache.
- Empty cloud-teacher responses are skipped instead of training the student on silence.
- Every prompt is tagged with its `micronaut` and `atomic_blocks` (HEADER/MENU/BODY/FEED/FOOTER) and logged to JSONL.
- `--resume` loads an existing LoRA checkpoint and continues training.
- `--log-dir` controls where JSONL logs are written.

The earlier 50-step run and the 500-step mixed run were superseded by this 400-step teacher-only run. A backup of the mixed checkpoint is kept at `models/from_zero/from_zero_v0.6_lora_backup_2026-08-07.safetensors`.

The earlier 500-step self-distillation run was replaced by this real-teacher run. `tools/oss_distillation.py` now supports `--ollama-url` / `--ollama-model` for cloud-teacher distillation.

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

### Trainer & build
- [x] `trainer/` folder created with source + shaders
- [x] `trainer/d3d11_engine.h/.cpp` — trainer-specific D3D11 (no XVM dependency)
- [x] `trainer/CMakeLists.txt` — standalone CMake with FetchContent nlohmann_json
- [x] Build: `gpt2_trainer.exe` — confirmed at `trainer/build/Release/gpt2_trainer.exe`
- [x] `llama-build.bat` — full rebuild sequence (UI + KLSL + cmake + GPU DLL deploy)
- [x] `pi_kuhul/` — KuhulPhysics.h, SphericalGeometryAVX2.h, DirectXMathAVX2.h, Fold2DCompiler.h (repo root + `trainer/pi_kuhul/`)

### Data & tools
- [x] Dataset links captured — full `E:\data` inventory
- [x] `tools/gen_kuhul_training.py` — synthetic corpus generator (350,388 examples)
- [x] `tools/kuhul_dataset_validator.py` — validate + compile π-KUHUL structured records
- [x] `tools/extend_vocab.py` — patch checkpoint wte [50260,768] → [50270,768]
- [x] `tools/merge_models.py` — SLERP/linear merge of two same-arch checkpoints (vocab-mismatch-aware)
- [x] `tokenizer_config.json` — KUHUL token IDs 50260–50269 at repo root
- [x] **Re-tokenized** `kuhul_synthetic.jsonl` → `E:\data\kuhul_tokens_kuhul.bin` (946,503 seqs × 128; KUHUL tokens present)

### Shaders
- [x] `trainer/shaders/gpt2_kuhul_think_bias.hlsl` — π-nary geodesic arc + 4-bone LBS attention bias
- [x] 7 missing shaders copied from v0.4.0 into `trainer/shaders/` (lm_head, lm_head_bwd, qkv_split, adam_wte, embed, matmul, residual)
- [x] 5 extra shaders present beyond PLAN.md listing: `cs_bone_argsort_`, `cs_fold_kernel_compute_`, `cs_fold_route_matmul`, `cs_gravity_field_layer_`, `cs_vertex_skin` (not yet wired into trainer)
- [x] Decision A: split `gpt2_attn_fwd.hlsl` → QK / think-bias / softmax+V — `gpt2_attn_qk_dot_` and `gpt2_attn_softmax_` shaders on disk **and wired into trainer dispatch** (`gpt2_trainer.cpp` compiles both at lines 234–235, dispatches at 1588/1688)
- [ ] Decision B: DML GEMM bridge mode

### Training phases
- [x] Phase 0a vacuum — DONE (loss floor 0.00322, `v0.2_vacuum`)
- [x] Phase 0b vacuum+LBS — DONE (loss floor 0.00066, `v0.3_vacuum_bias`)
- [x] Phase 1 header corpus — DONE 2026-08-04 (`v0.4_phase1`, antigravity→1.0 at step ~1200)
- [x] Phase 2 KUHUL corpus — DONE 2026-08-04 (`v0.5_phase2`, 3000 steps)
- [x] Phase 3 merge — DONE 2026-08-04 (`v0.6_merged`, SLERP α=0.6, 148 tensors)
- [ ] **GPT-2 Medium merge** — two 1449 MB safetensors at `E:\models\GPT2\med-GPT\` (`gpt2_medium_merged_a035` + `model`). Same architecture (355M params). Merge: `python tools/merge_models.py E:\models\GPT2\med-GPT\gpt2_medium_merged_a035.safetensors E:\models\GPT2\med-GPT\model.safetensors E:\models\GPT2\med-GPT\gpt2_medium_merged.safetensors --alpha 0.5 --method slerp`
- [ ] **GPT-2 XL split-train-merge** — XL GGUF → safetensors (`tools/gguf_to_safetensors.py`), split into A=tool-train + B=chat, merge A+B via SLERP.
- [ ] **Qwen 1.8B split-train-merge** — base safetensors at `.lmstudio/models/Qwen-1_8B-Chat-f16/model.safetensors`. Quantized variants at `E:\models\Qwen1.8B-quant/` (q4.kqz 931 MB + q8.kqz 1753 MB). Same split-train-merge pattern: A = tool-train (adds tool calling to Qwen's chat ability), B = keep original, merge A+B. The `qwen_infer_driver` already handles the Qwen architecture (RoPE, RMSNorm, SiLU, 24 layers, 2048 emb, 16 heads).
- [x] **Phase 4 distillation** — `from_zero_v0.6_lora.safetensors` produced 2026-08-07 via `tools/oss_distillation.py` (400 steps: 200 fresh + 200 resumed, rank=8, lr=1e-4, 128 teacher tokens) using real GPT-OSS 120B cloud teacher through Ollama. Micronauts + Atomic DOM supply stack-specific context at runtime, so the model does not need to memorize the full stack. JSONL logs include atomic block tags.
- [x] **Fix trainer safetensors save path** — `AdamParam::shape` field added, populated from safetensors metadata during load, written correctly during save (2026-08-06)

### Model conversion
- [x] **GGUF conversion** — `from_zero_v0.6_merged.gguf` (654 MB, 149 tensors, gpt2 arch) — **DONE 2026-08-06**. Verified: 12 layers, 768 emb, 12 heads, 3072 FF, 1024 ctx, 50270 vocab. **Tested inference**: answers factual Qs (capital of France), pushes tool calls when prompted. Mini-GPT fluent in chat + tool calling.
- [x] **KUHUL-vocab GGUF** — `from_zero_v0.6_kuhul.gguf` (654 MB) — **DONE 2026-08-07** via `tools/gpt2_kuhul_to_gguf.py` from `from_zero_v0.6_merged.safetensors`. Unlike `gpt2_safetensors_to_gguf.py`, this converter reads vocab size from the `wte` tensor (50270), extends the GPT-2 tokenizer metadata with the KUHUL special tokens from `tokenizer_config.json`, and preserves all 50270 embedding rows so llama.cpp's shape check passes. This is the GGUF now referenced by `active-model.json` and `models/from_zero/atomic.manifest.json`.
- [x] **Atomic DOM launch** — `AtomicDOM.cmd` / `AtomicChat.cmd` at `.NNC-K/bin/v3.5.0-WebX/` launch `kuhul_engine.exe --Atomic.DOM <manifest>`. Each model's `atomic.manifest.json` defines chat template, system prompt, micronaut sampling, tool registry, and execution gating. The DOM layer is what enables tool calling — not the raw model.
- [x] **Qwen forward path** — `qwen_infer_driver.h/.cpp` created (2026-08-07). 9 D3D11 compute shader ops: embed, rms_norm, matmul, rope, attention, silu, add_bias, residual, lm_head. Architecture: 24 layers, 2048 emb, 16 heads, 128 head_dim, RoPE theta 1e6. Qwen 1.8B config default. **DLL compiled 2026-08-07**; full shader dispatch still stubbed.

### Model registry
- [x] 14 `atomic.manifest.json` files confirmed on disk (from_zero, khanary-kxml-v0.5.0, gpt2-xl, lfm2-1b, gemma-3-1b, gemma-3-4b, gemma-4-e2b, gpt-oss, phi3-mini-4k, dolphin-phi2, gemma-3-1b-q8, qwen-1b8-chat, qwen25-05b-story, mgguf-gpt2-2expert)
- [ ] `mgguf-qwen-1expert/` — no `atomic.manifest.json` on disk (MoE ASX mgguf, manifest TBD)

### KUHUL APPS — HYBRID MCP STUDIO (revised architecture)
- [x] **Copy C++ sources** — 8 files from `.NNC-K/native/runtime/` + driver source in `drivers/` (alongside `json_runtime_lib.dll`) (2026-08-06)
- [x] **Driver DLL adaptation layer** — `khanary_driver.h/.cpp` (7 C ABI exports), `webx_stubs.h`, `studio_tasklist_example.json`
- [x] **Compile khanary_driver.dll** — **DONE 2026-08-07**. MSVC BuildTools 2022 (cl.exe 14.44.35207) is present at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe`. The "no C++ toolchain" diagnosis in prior PLAN.md audits was wrong. Added missing `drivers/task_engine.h` stub so `khanary_driver.cpp` compiles. All 5 source-only driver DLLs now build via `drivers/build_drivers.bat`.
- [x] **Deploy DLL / ffi loading** — kuhul-server.cjs loads the driver DLLs, but `ffi-napi` does not compile on Node 24.15.0. Added `dist/khanary-server/ffi-shim.js` backed by `koffi` (installed and working) to provide the same `ffi.Library` + `ref.readCString` surface. 2 new MCP tools (`kuhul_driver_plan`, `kuhul_driver_dispatch`) are wired and functional.
- [x] **Wire llama.cpp native MCP server** — tools served from model backend (port 17480) via MCP_TOOLS registry (11 tools total)
- [x] **Micronaut hive dispatch** — 5 providers registered (micronaut-coder, micronaut-factory, micronaut-base, json-runtime, kuhul-engine), bot.py dispatch path documented
- [x] **Canvas route** — `(chat)/chat/[id]/canvas/+page.svelte` (373 lines): 3-panel grid, MCP task_boss dispatch, HTML extraction, model selector
- [x] **Canvas iframe component** — `CanvasPreview.svelte`: Preview/Source tabs, sandboxed srcdoc iframe, refresh button
- [x] **Plan checklist component** — `PlanChecklist.svelte`: TaskPlanItem[] render with ○/◉/●/⊗ status icons, toggle handler
- [x] **Build / Deploy / Open actions + sidebar entry** — `ROUTES.CANVAS(id)` constant added, Export/Copy/Open actions in canvas sidebar

### XCFE GL backend (`bridges/ggml-xcfe/`)
- [x] `xcfe_gl_ops.dll` **compiled** at `bridges/ggml-xcfe/xcfe_gl_ops.dll` — source at `xcfe_gl_ops.cpp`. Loaded by `ggml-xcfe.dll` at runtime via `LoadLibraryA`. MUL_MAT verified (max err 4.768e-07). Covers the full F32 tensor set via wgpu_native GL backend (17 WGSL kernels staged, 42 shaders on disk). Uses seam contract v2: `xcfe_gl_run(op, inputs, n_inputs, out, ne_out, n_dims, params, n_params)`.
- [x] `ggml-xcfe.dll` **compiled** at `dist/khanary-server/ggml-xcfe.dll` — the bridge DLL that routes GGML tensor ops through `xcfe_gl_ops.dll` for GPU dispatch or falls back to CPU.
- [x] **42 WGSL shaders on disk** at `khanary-llama-build/ggml/src/ggml-webgpu/wgsl-shaders/` — flash_attn (5 variants), mul_mat (5 variants), set_rows (quantized), rms_norm_mul, row_norm, rope, soft_max, concat, binary, unary, argsort, cumsum, ssm_conv, ssm_scan, gated_delta_net, glu, scale, pad, repeat, cpy, get_rows, quantize_q8, im2col, conv2d, upscale, solv_tri, argmax, add_id, set, memset
- [ ] **Verify WGSL kernels** — op-level test harness needed for per-op CPU reference comparison (only MUL_MAT verified so far)
- [ ] End-to-end F32 model run (`models/from_zero` f32) with `-ngl 999` on GL backend

### SCXQ2 / XVM — known issues (`.NNC-K` runtime)

Audit of `dist/xvm-d3d12/src/` revealed 6 bugs:

| Issue | Files | Impact |
|---|---|---|
| **SCXQ2_FAMILY_COLLISION** | `scxq2_format.cpp`, `scxq2_format_v1_2.cpp` | Three incompatible parsers share one family name → loader mis-selection |
| **XVM_OPCODE_SEMANTICS_DRIFT** | `xvm_runtime.cpp` vs `xvm_core.cpp` | `0x04` = EMIT in runtime but ADD in core; `0x05` vs `0x3F` return semantics |
| **XVM_CORE_IMPL_SPLIT** | local vs `desp/engine/xvm/` | Two xvm_core.cpp differ in phase/barrier/entropy — deterministic claims diverge |
| **SCXQ2_V12_UNWIRED** | `test_scxq2_format_v1_2.cpp`, `CMakeLists.txt` | v1.2 parser + test exist but not in CMake target graph — dead code |
| **SCXQ2_V12_UNDERFLOW** | `scxq2_format_v1_2.cpp` | `lane_data_len = lane_len - 4` without guard for `lane_len < 4` → OOB read |
| **SCXQ2_STREAM_ALIAS** | `d3d12_stream_adapter.cpp` | `streamSCXQ2ToCodeBuffer` routes through DDSStream with SCXQDDS semantics |

These are in the `.NNC-K` XVM/D3D12 runtime — not the `_khanary_inspect` workspace. Fix by:
1. Rename family tags to avoid collision (`SCXQ2-v1`, `SCXQ2-v1.2`, `SCX2-runtime`)
2. Normalize opcode tables (pick one canonical xvm_core.cpp)
3. Wire v1.2 into CMake, add bounds check, unify stream adapter contracts

### Build system + runtime issues (audit 2026-08-07)

| Issue | Location | Impact |
|---|---|---|
| **BUILD_CACHE_ROOT_DRIFT** | `dist/xvm-d3d12/build-ninja3/CMakeCache.txt` | Cache points to `C:/public_html/...` not workspace → local rebuilds fail |
| **BUILD_DEP_PATH_DRIFT** | `dist/xvm-d3d12/CMakeLists.txt` | References `asx_scx/dependencies/json/include` not in this checkout |
| **SCXQDDS_ZSTD_GATING** | `CMakeLists.txt`, `scxqdds_chunks_loader.cpp` | zstd compression compile-time optional → silently unavailable |
| **MCP_SCHEMA_NOT_ENFORCED** | `kuhul-server.cjs` | Tool `inputSchema` declared but not validated before dispatch → **FIXED 2026-08-07**: `validateToolArgs()` checks required + type before every dispatch; `schema_error` on violation |
| **TRACE_NONDETERMINISM** | `kuhul-server.cjs`, `kuhul-es.cjs` | `Date.now`/`toISOString` in trace → provenance hashes non-replayable → **FIXED 2026-08-07**: monotonic `seq` in hash payload, wall_ts display-only, both trace writers |
| **NO_EVAL_GATE_MISSING** | `.NNC-K` runtime | `new Function` dynamic execution → authority bypass → **FIXED 2026-08-07**: `src/xcfe/eval-gate.js` (fail-closed; origins codegen/micronaut-verified/system; `KHANARY_ALLOW_EVAL=1` to open; `KHANARY_EVAL_TRACE=1` audit). Wired into Sandbox.run() + @function handler. Verified deny/fail-closed/open |
| **SUPERNAUT_ACTION_DRIFT** | `src/supernaut/routes.js` | Routes map to action names with no local implementation → non-executable surface |
| **PROJECTION_SOURCE_SPLIT** | `dist/`, `.micronauts`, `DESP` | Three active projections partly aligned, partly divergent → ambiguous canonical source |

### Audit fixes (SCXQ2/XVM + build/runtime, 2026-08-07)
- [x] **SCXQ2 family collision** — parser tags renamed: `SCX2-runtime` (scxq2_format.cpp errors), `SCXQ2-v1.2` (scxq2_format_v1_2.cpp header documents magic 0x53435151 "SCQQ" + v0x02, distinct from SCX2-runtime)
- [x] **XVM opcode semantics** — normalized `xvm_runtime.cpp`: `kOpEmit = 0x47` (host event dispatch; 0x04 = OP_ADD in xvm_core), `kOpReturn = 0x3F` (matches xvm_core)
- [x] **XVM core split** — local `xvm_core.cpp` (4434ec3a, superset) canonical; DESP copy (89ddf17e, stale) documented at `E:\models\.desp-v1\DESP-V1.2\engine\xvm`
- [x] **SCXQ2 v1.2 wire-up** — `scxq2_format_v1_2.cpp` added to xvm_d12 target + `scxq2_v1_2_test` executable (test_scxq2_format_v1_2.cpp)
- [x] **SCXQ2 v1.2 underflow** — `lane_len < 4` guard added before `lane_len - 4` (fix_underflow.py)
- [x] **SCXQ2 stream alias** — renamed `streamSCXQ2ToCodeBuffer` → `streamSCXQDDSChunkToCodeBuffer` (3 files: scxqdds_chunks_loader.cpp, stream adapter, header)
- [x] **Build cache root** — `build-ninja3/CMakeCache.txt` `CMAKE_HOME_DIRECTORY` → `C:/Users/canna/_khanary_inspect/dist/xvm-d3d12`
- [x] **Build dep path** — CMakeLists.json fallback include dir → `../../trainer/build/_deps/nlohmann_json-src/single_include`
- [x] **ZSTD gating** — `DecompressZstdToMemory` raw-copy fallback when `XVM_USE_ZSTD` not defined
- [x] **MCP schema enforcement** — `validateToolArgs()` checks required + type against inputSchema before dispatch; `schema_error` returned on violation
- [x] **Deterministic trace** — monotonic `seq` in hash payload, wall_ts display-only; both `kuhul-server.cjs` + `kuhul-es.cjs` trace writers
- [x] **Eval gate** — new `src/xcfe/eval-gate.js` (fail-closed); wired into `Sandbox.run()` (micronaut-parallelism.js) + `@function` handler (imperative-layer.js); verified deny untrusted / fail-closed / open with env
- [ ] **Supernaut bindings** — verify route → action implementations exist in `src/supernaut`
- [x] **Projection canonical** — IDB pattern adopted from `E:\models\.micronaut\kuhul\` into `brain2/idb/` (2026-08-07): `idb_memory.py` (KV heap + causal DAG + schema registry), `mx2db.py` (KUHUL tables + optional Object Server with Python fallbacks), `idb.schema.xsd` (verbatim), fresh `idb.instance.xml` seeded with SCXQ2/SCXQ2-v1.2/CM-1/SCXQ7/SCX-BSON. `KHANARY_IDB_PATH` configurable; `KHANARY_OBJECT_SERVER=0` default (server optional). SCXQ2 golden vector generator copied to `dist/xvm-d3d12/tests/generate_scxq2_vectors.py` (13 glyphs, 16 opcodes). Verified: KV set/get/flush, causal steps, KUHUL tables, cm1_verify + kuhul_tsg fallbacks, XML parse, module import.
- [x] **Studio domain in mx2db** — 7 app-layer tables added (users, projects, conversations, messages, apps, sessions, micronauts = 16 total with KUHUL engine tables) + `studio.*` RAM keys. Per-user profiles at `micronauts/user-<ID>.micronaut.json` with template `micronauts/user-template.micronaut.json` (sampling, active, projects, owned_micronauts, tools allow/deny). `save_user_micronaut`/`load_user_micronaut` helpers: deep-merge over existing file or template (partial updates preserve tools/projects/sampling), mirror row into users table. `KHANARY_MICRONAUTS_DIR` configurable. Verified: deep-merge two-pass update, template fallback, 16 tables, clean instance.
- [x] **User profile forge + ROM build** — `studio-dist/user-profile.html` (custom agent forge: user ID, agent name, temp/penalty/last_n sliders, system message persona, MCP tool allow/deny). MCP tools `kuhul_user_profile_get`/`kuhul_user_profile_save` in kuhul-server.cjs (deep-merge, IDB mirror via mx2db). Compiled into ROM: `⚙ profile` topbar button → iframe modal; `send()` injects profile system message + sampling into `/v1/chat/completions`. Native WWA packaging: `ke_package_wwa`/`ke_launch_wwa`/`ke_wwa_runtime_available` exports added to `kuhul_engine_driver` (deterministic STORE zipcontainer, CRC32, sorted entries — mirrors KuhulAppCreator layout). MCP tools `kuhul_wwa_package`/`kuhul_wwa_launch` (native via DLL, PowerShell fallback). `ROM-build.bat` (validate → package studio.wwa via `tools/package_wwa.cjs` → deploy/launch modes). Verified: ROM validates 6 core files, studio.wwa builds (7 entries incl. user-profile.html, no self-inclusion), valid zip, koffi-backed ffi-shim path + PowerShell fallback both work.
- [x] **Minimal llama mode (inference-only)** — llama.cpp made UI-optional so the studio replaces llama's embedded web UI:
  - `LLAMA_BUILD_UI` option gated at three levels: root CMakeLists option (deduped with upstream), `tools/CMakeLists.txt` only adds `ui/` subdir when ON, `tools/server/CMakeLists.txt` links `llama-ui` + defines `LLAMA_BUILD_UI=1` only when ON (else inference-only link)
  - `server-http.cpp` guards `#include "ui.h"` and the `frontend_paths` lambda with `#if defined(LLAMA_BUILD_UI)` (inference-only server serves just `/v1/*` + `/health`; the asset-serving block was already behind `LLAMA_UI_HAS_ASSETS`)
  - `llama-build minimal` — skips stale-UI clearing + npm entirely, configures `-DLLAMA_BUILD_UI=OFF`, builds llama-server only
  - `build_gpu.ps1 -UiOff` — full GPU rebuild in inference-only mode; script ASCII-ized + UTF-8 BOM added (PS 5.1 ANSI parse fix: 8 em-dashes, 13→0 parse errors)
  - With all drivers deployed, llama is a pure inference core: llama.dll + ggml.dll + ggml-base/cpu/xcfe + dml_gemm + DirectML + llama-server-impl; studio-dist/ (ROM) is the UI

### PRIMEOS (desktop management layer)
- [x] PRIMEOS .NET 8 WPF app scaffolded (`desktop/PRIMEOS/` — App.xaml, Shell, .csproj)
- [x] **PRIMEOS ↔ kuhul-server API contract** — 2 endpoints added to kuhul-server.cjs (2026-08-07):

| Endpoint | What PRIMEOS gets |
|---|---|
| `GET /kuhul/engine/status` | Engine PID/port/up, gateway PID/port/tools/drivers, json_runtime status, WWA host, SHM name, uptime |
| `GET /kuhul/stack/status` | All 4 services (kuhul-server, engine, json_runtime, llama-server), MCP tool count, micronaut count |

PRIMEOS already hits `/health` on llama-server and SHM reads `Local\KuhulGeometricState`
every 500ms. The new endpoints give it stack-level visibility for the health dashboard.
- [ ] Wire kuhul-server endpoints into PRIMEOS health dashboard UI
- [ ] Add PRIMEOS as startup item (auto-start kuhul-server + kuhul_engine on login)

### Inference backends
- [ ] `gl_infer.dll` — OpenGL 4.3 compute-shader inference backend (README: "in progress"; no DLL in repo)
- [x] `d3d11_infer.dll` — D3D11 compute-shader inference (referenced in README; trainer uses D3D11 directly via d3d11_engine)
- [x] Server stack deployed: `dist/khanary-server/` — khanary-server.exe, kuhul-server.cjs, all GGML/llama DLLs, dml_gemm.dll, DirectML.dll

### Driver DLL inventory (`drivers/`)

| Driver | Source | Status | Exports |
|---|---|---|---|
| `json_runtime_lib.dll` | (prebuilt) | **Compiled** | Hosting API (sidecars, file-manager) |
| `native_glyph_engine.dll` | (prebuilt) | **Compiled** | K'UHUL glyph IPC rendering |
| `native_glyph_engine_abi.dll` | (prebuilt) | **Compiled** | Glyph engine ABI variant |
| `xcfe_gl_ops.dll` | `bridges/ggml-xcfe/xcfe_gl_ops.cpp` | **Compiled** | GGML → OpenGL 4.3 compute shader bridge (seam contract v2) |
| `ggml-xcfe.dll` | `bridges/ggml-xcfe/` | **Compiled** | Routes GGML tensor ops to GPU (xcfe_gl_ops) or CPU fallback |
| `khanary_driver.dll` | `khanary_driver.h/.cpp` + `DAG.cpp` + `task_engine.h` (added) | **Compiled** | 7 exports: kd_create/load_tasks/plan/run/dispatch |
| `khanary_glyph_driver.dll` | `khanary_glyph_driver.h/.cpp` | **Compiled** | 12 phases + 13 lanes = 25 entries, dump/dispatch |
| `kuhul_engine_driver.dll` | `kuhul_engine_driver.h/.cpp` | **Compiled** | 10 exports: ke_create/load_model/load_dom/chat/status |
| `gl_infer_driver.dll` | `gl_infer_driver.h/.cpp` | **Compiled** | 8 exports: gli_create/load_model/forward/sample/probe + 8 GLSL compute shader ops |
| `qwen_infer_driver.dll` | `qwen_infer_driver.h/.cpp` | **Compiled** | 6 exports: qw_create/load_model/forward/sample/probe + 9 D3D11 compute shader ops (RoPE, RMSNorm, SiLU) |

### MCP tool registry (20 tools)

| Tool | Handler | Role |
|---|---|---|
| `kuhul_chat` | Proxy to kuhul_engine | Model inference |
| `kuhul_tasklist` | MicrosoftSDK.ps1 | Generate TaskList JSON |
| `kuhul_task_boss` | driver-first → execFile fallback | Execute TaskList |
| `kuhul_driver_plan` | khanary_driver.dll | Plan tasks in-process |
| `kuhul_driver_dispatch` | khanary_driver.dll | Single-task dispatch |
| `kuhul_json_runtime` | execFile | Run JSON runtime programs |
| `kuhul_manifest` | MicrosoftSDK.ps1 | Stack manifest |
| `kuhul_engine_status` | Health probe | Engine/JSON runtime/WWA status |
| `kuhul_wwa_host` | spawn | Launch WWA apps |
| `kuhul_wwa_package` | kuhul_engine_driver.dll (ke_package_wwa) / PowerShell fallback | Package an app folder into a `.wwa` zip container |
| `kuhul_wwa_launch` | kuhul_engine_driver.dll (ke_launch_wwa) / PowerShell fallback | Launch a packaged `.wwa` app |
| `kuhul_grammar` | File read | K'UHUL EBNF grammar |
| `kuhul_grammar_validate` | kuhul_engine | Validate K'UHUL source → AST JSON or validation errors |
| `kuhul_forge` | execFile | Forge memory micronaut |
| `kuhul_glyph_phase` | khanary_glyph_driver.dll | Dispatch phase/fold glyph |
| `kuhul_glyph_registry` | khanary_glyph_driver.dll | Dump 25-entry glyph+lane registry |
| `kuhul_gpu_probe` | gl_infer_driver.dll | OpenGL 4.3 GPU probe (vendor, renderer, shader support) |
| `kuhul_qwen_probe` | qwen_infer_driver.dll | Qwen 1.8B D3D11 architecture probe |
| `kuhul_engine_dom` | kuhul_engine_driver.dll | Load Atomic DOM manifest, extract tools + persona + gating |
| `kuhul_build_micronaut` | fs.writeFile | **Self-extending**: GPT-OSS creates new micronauts |

### WWA system integration

Windows Web Application host is a native Windows subsystem. The studio ROM can be hosted
as a first-class Windows app via the WWA stack in `System32/`:

| DLL | Role | Adopt for |
|---|---|---|
| `WWAHost.exe` | Web app host process | Launch studio ROM as native window |
| `WwaApi.dll` / `WwaExt.dll` | WWA API + extensions | System integration (notifications, tiles) |
| `xmllite.dll` | Lightweight XML parser | Parse manifests without pulling in full XML stack |
| `zipcontainer.dll` | ZIP container | Pack/unpack studio ROM as `.wwa` bundle |
| `wwapi.dll` | Windows Web API | Native HTTP server for local studio hosting |
| `XamlTileRender.dll` | XAML tile renderer | Live tile updates from micronaut task status |

kuhul-server already launches WWA apps via `kuhul_wwa_host` MCP tool. The studio ROM
can be bundled as a `.wwa` package and launched via `WWAHost.exe studio-dist\` without
a separate HTTP server.

### Studio ROM (`studio-dist/`)

Self-contained 8-file PWA served as a static ROM via json_runtime sidecar. Single HTML file (754 lines) with JetBrains Mono + Chakra Petch fonts, obsidian/jade/gold theme. Service worker for offline FLASH RAM persistence.

| File | Role |
|---|---|
| `index.html` | Full studio shell — chat, canvas, theme |
| `kuhul-icon.svg` | PWA icon (SVG, maskable) |
| `kuhl-studio.webmanifest` | PWA manifest (standalone display) |
| `sw.js` | Service worker (cache, persistence) |
| `studio.manifest.json` | ROM manifest — COOP/COEP/CSP headers, port 8820-8829, MCP tool bindings |
| `user-profile.html` | Custom agent forge (user ID, agent name, temp/penalty/last_n sliders, MCP tool allow/deny) |
| `studio.wwa` | Packaged WWA bundle (built by `ROM-build.bat` via `tools/package_wwa.cjs`) |
| `launch.bat` | Launch helper |

**Security (SvelteKit-equivalent):** COOP `same-origin` + COEP `require-corp` for SharedArrayBuffer. CSP: inline scripts, Google Fonts, localhost MCP connect. Frame-src `data:` for sandboxed canvas.

**Mount:** `json_runtime.exe --manifest studio.manifest.json` → `http://127.0.0.1:8820`

**Build all drivers (MSVC):**
```powershell
cd drivers
.\build_drivers.bat
```
This builds all 5 source-only driver DLLs with the correct `/std:c++17` flag and `vcvars64.bat` setup. Individual builds (not recommended):
```powershell
cl /std:c++17 /EHsc /O2 /LD /I. khanary_driver.cpp DAG.cpp /Fe:khanary_driver.dll
cl /std:c++17 /EHsc /O2 /LD /I. khanary_glyph_driver.cpp /Fe:khanary_glyph_driver.dll
cl /std:c++17 /EHsc /O2 /LD /I. kuhul_engine_driver.cpp /Fe:kuhul_engine_driver.dll
cl /std:c++17 /EHsc /O2 /LD /I. gl_infer_driver.cpp /Fe:gl_infer_driver.dll
cl /std:c++17 /EHsc /O2 /LD /I. qwen_infer_driver.cpp /Fe:qwen_infer_driver.dll
```

### Build pipeline (standardized 2026-08-06)

Four batch files consolidated into one build → deploy → launch pipeline:

| Script | Role | Status |
|---|---|---|
| `llama-build.bat` | **Build** — clears stale UI cache, npm build, cmake, GPU DLLs | Authoritative |
| `llama-build.bat deploy` | **Deploy** — copies fresh binary + DLLs to `dist/khanary-server/` | New |
| `START-SERVERS.bat` | **Launch** — starts 4 services, writes `active-model.json`, prints MCP hookup | Authoritative |
| `build-khanary.bat` | **Alias** — delegates to `llama-build.bat` (no more stale builds) | Fixed |
| `SERVER-LAUNCHER.bat` | **Deprecated** — forwards to `START-SERVERS.bat` | Deprecated |

**Standard workflow:**
```powershell
llama-build deploy    # build + npm + cmake + deploy to dist/
START-SERVERS         # launch all 4 services
```

**Stale cache fix:** `llama-build.bat` deletes 4 artifacts before each build:
- `tools/ui/dist/` — priority-1 source dist (cmake bakes this directly)
- `build/tools/ui/.ui-stamp` — npm-skip stamp
- `build/tools/ui/ui.cpp` — embedded file (stale = old UI in binary)
- `build/tools/ui/ui.h` — embedded header

These survive `cmake --build` and cause every subsequent build to reuse old UI.
`build-khanary.bat` no longer runs cmake directly — it delegates to `llama-build.bat`.

### Brain expert routing (GPT2_THINK_BIAS=1)

**How it works (3 modifiers stacked, post-softmax on P_buf):**
1. π-nary arc: `sin(depth/2)²` — tokens inside `<THINK>…</THINK>` get geodesic-distance-weighted attention boost
2. KuhulPhysics scale: `antigravity_scale` (0.1→1.0) — strengthens as training stabilises
3. Brain expert cluster: `brain_experts_[tok % 30628]` ∈ [0,60] — same Delaunay cluster → co-attention boost

**Path:** `brain2/experts.bin` relative to build dir, or `$env:GPT2_BRAIN_EXPERTS=<full_path>`.

**Convergence note:** loss ~6.0 at step 116-117 on `kuhul_tokens_kuhul.bin` (487 MB, 121M tokens) is **expected** — the model has seen 60K tokens (0.05% of data). CPU hit 0.00 overnight on the small `tokens_hdr.bin` (12 MB, repeating). To verify GPU backward works: run a quick overfit with `--data E:\data\kuhul_tokens.bin --steps 200 --lr 1e-3` on a tiny slice, or use `tokens_hdr.bin`.

---

## Gaps discovered 2026-08-06

Full workspace audit against PLAN.md claims. Cross-referenced every `[x]` task against files on disk.

### Critical — studio files claimed done but missing

PLAN.md claimed three KUHUL APPS studio items as `[x]` done. Two of the three referenced files still do not exist in `khanary-ui-build/src/` (scanned all `.ts` / `.svelte` source files):

| Claimed item | Claimed file | Status |
|---|---|---|
| Stack service + gateway MCP + canvas route | `projects.service.ts`, `/chat/[id]/canvas` | **Canvas route EXISTS** — `(chat)/chat/[id]/canvas/+page.svelte` + `CanvasPreview.svelte` + `PlanChecklist.svelte` (TaskPlanItem types present). `projects.service.ts` still absent — intentionally replaced by the native driver DLL pivot (see below). |
| Project manifest lifecycle | `projects.service.ts` | **File not found** — replaced by `json_runtime` `/api/file-manager/*` |
| Task routing | `studio-task-router.ts` | **File not found** — replaced by `khanary_driver.dll` (kd_plan / kd_dispatch) |

`grep_files` for `TaskPlanItem`, `StudioTaskRequest`, `StudioTaskResult` (the types PLAN.md describes at lines 968-973) returned zero matches. These may have existed in a prior workspace, been deleted, or were planned but never committed.

**Update 2026-08-08**: `TaskPlanItem` now matches in `+page.svelte` and `PlanChecklist.svelte`. The two missing service files are not being rebuilt — the architecture pivoted to backend-driven MCP tools with a thin frontend shell (see the HYBRID MCP STUDIO section).

**Resolution options:**
1. Recreate `projects.service.ts` + `studio-task-router.ts` from the PLAN.md spec (implementation plan exists at lines 965-982)
2. If they exist in another workspace/checkout, copy them into `_khanary_inspect/khanary-ui-build/src/lib/services/`
3. If they never existed, reset those items to `[ ]` and implement from spec

### Missing artifacts

| Item | PLAN.md claim | Ground truth |
|---|---|---|
| `from_zero_v0.1.f32.gguf` (623.6 MB) | Model registry says in `models/from_zero/` | **Not present** (historical — the log `scratch/fz_test.err` shows it was served from `models/from_zero/` on 2026-08-05, then removed). Superseded by `from_zero_v0.6_merged.gguf` + `from_zero_v0.6_kuhul.gguf` (both present, 654 MB each). |
| `v0.6_lora.safetensors` | Phase 4 distillation output | **Exists** (2026-08-07, 400 cloud-teacher steps, 128 tokens/target, best_loss=7.0734) |
| Trainer safetensors shape fix | Root cause | `tools/repair_safetensors.py` exists (fixes files post-hoc). **Root cause FIXED 2026-08-06**: `AdamParam::shape` is populated from safetensors metadata during load, preserved, and written correctly on save (`gpt2_trainer.cpp` lines 325–342, 441, 2787). New checkpoints no longer write empty `"shape":[]`. |
| `gl_infer.dll` | README: "in progress" | **No DLL in repo.** The OpenGL 4.3 inference backend exists as a concept in README.md but no compiled artifact. |
| `mgguf-qwen-1expert/` atomic manifest | Model registry row says "Qwen 1-expert MoE" | Directory contains only `atomic.manifest.json` — no weights. Manifest exists but models are not present. |

### Shader inventory — extra files on disk not in PLAN.md

5 HLSL shaders present in `trainer/shaders/` that aren't listed in the PLAN.md shader tree (line 24-47):

| File | Likely purpose |
|---|---|
| `cs_bone_argsort_.hlsl` | 4-bone LBS bone-ID sorting for attention routing |
| `cs_fold_kernel_compute_.hlsl` | K'UHUL fold kernel for region-based computation |
| `cs_fold_route_matmul.hlsl` | Fold-routed matrix multiply |
| `cs_gravity_field_layer_.hlsl` | Gravity field physics computation (KuhulPhysics GPU path) |
| `cs_vertex_skin.hlsl` | Vertex skinning kernel (geometry model) |

These are not wired into `gpt2_trainer.cpp` dispatch. They represent GPU physics / geometry paths that exist as shaders but have no trainer integration yet. Added as a shader item with clarification that they're on disk but not wired.

**Update 2026-08-08**: 3 of the 5 are now wired — `cs_gravity_field_layer_` (compiled, gravity sync), `cs_bone_argsort_` (compiled + dispatched, CPU-mirror match), `cs_fold_kernel_compute_` (compiled + fold dispatch table). `cs_fold_route_matmul` and `cs_vertex_skin` remain on disk but unwired.

### Decision A clarification

The `gpt2_attn_fwd.hlsl` split (Decision A) already has two partial shaders on disk:
- `gpt2_attn_qk_dot_.hlsl` — Q·K^T dot product
- `gpt2_attn_softmax_.hlsl` — softmax normalization

These aren't listed in the PLAN.md shader tree either. They exist alongside the monolithic `gpt2_attn_fwd.hlsl` but the trainer dispatch code still uses the monolithic path. Decision A is about wiring these into the trainer's forward pass — not writing new shaders from scratch.

### XCFE GL — verification gap

`xcfe_gl_ops.dll` is built and MUL_MAT passes verification (line 244-246). But PLAN.md says "17 WGSL kernels staged" and only 1 (MUL_MAT) has been verified against CPU reference. The remaining 16 kernels (get_rows, norm, rms_norm, gelu, gelu_quick, silu, relu, tanh, sigmoid, add, sub, mul, soft_max, rope, concat, cpy) need an op-level test harness. Without this, `xcfe_gl_ops.dll` cannot be trusted for training backward-pass compute.

### PRIMEOS — scaffold only

PRIMEOS has a .NET 8 WPF project skeleton (`App.xaml`, `Shell`, `.csproj`) but:
- No API contract with kuhul-server
- No health dashboard integration
- No startup item registration
- No process management, no MCP tool invocation, no micronaut factory UI
- The PRIMEOS.md spec describes a full management layer; the code is a scaffold

### Corrected status summary

| Category | Before audit | After audit |
|---|---|---|
| Trainer build artifacts | 1 claimed (279 KB exe) | 1 confirmed present |
| Training phases done | 5 | 5 confirmed (v0.2 through v0.6) |
| Model manifests | 7 claimed | 14 confirmed on disk + 1 missing (mgguf-qwen) |
| Studio tasks done | 3 claimed done | 1 confirmed (canvas route + CanvasPreview + PlanChecklist); 2 service files absent — pivoted to native driver DLLs |
| Native driver DLLs | 5 claimed compiled | **5 confirmed compiled 2026-08-07** (khanary_driver, khanary_glyph, kuhul_engine, gl_infer, qwen_infer) |
| Shaders listed in PLAN.md | 14 | 14 listed + 7 extra on disk |
| GGUF models in repo | 1 claimed | 0 confirmed |
| XCFE GL ops verified | 1 of 17 | 1 of 17 |
| PRIMEOS functionality | "management UI" | Scaffold + kuhul-server endpoints wired; dashboard UI not yet integrated |

### Architectural pivot: HYBRID MCP STUDIO (2026-08-06)

The 8 missing studio TypeScript files are not being rebuilt. The architecture shifts to:

1. **khanary_driver.dll** — native driver DLL compiled from `.NNC-K/native/runtime/` sources via MSVC. Flat C ABI (7 exports). Loaded by kuhul-server via ffi-napi. Handles load/validate/plan/run/dispatch with DAG + provider resolution.
2. **llama.cpp native MCP server hosting** — tools served from the model backend directly (port 17480), eliminating the separate kuhul-server MCP gateway hop.
3. **Micronaut hive** (`E:\models\.hive`, `.micronaut*`) — `bot.py` per micronaut domain. khanary_driver.dll dispatches to the right micronaut via MCP.
4. **Individual vendor drivers** — same 7-function ABI, swappable at load time (OpenCL, DML, CUDA variants if needed).
5. **Svelte frontend = thin shell** — chat input, canvas iframe, MCP client. No TypeScript orchestration services.

See the revised KUHUL APPS — HYBRID MCP STUDIO section below for the full architecture.

---

## GPT-OSS Distillation — Phase 4

### GPT-OSS — task engine model (dual role)

GPT-OSS 20B serves two roles in the stack:

| Role | Function | Runtime |
|---|---|---|
| **Task orchestrator** | Plans, builds, verifies, researches — has access to all MCP tools | kuhul-engine :17480 |
| **Distillation teacher** | Generates KUHUL-domain completions for LoRA training | `oss_distillation.py` |

GPT-OSS can also dispatch to **cloud worker agents** for offloaded compute:

| Worker | Type | Models | Role |
|---|---|---|---|
| DeepSeek Flash | `worker` | `deepseek-chat` | Fast, cheap code generation + tool calling for micronaut tasks |
| Ollama Cloud | `worker` | `llama3.1, mistral, qwen2.5` | Secondary worker for non-critical / parallel tasks |
| OpenAI | `advisor` | `gpt-4o-mini` | Optional high-quality advisor for complex plans |
| Anthropic | `advisor` | `claude-sonnet` | Optional advisor for reasoning-heavy tasks |

**Cloud dispatch config** (`ModelCloud.toml` + `.env`):

```toml
[providers.deepseek-flash]
type = "worker"
endpoint = "https://api.deepseek.com/v1/chat/completions"
api_token_env = "DEEPSEEK_API_KEY"
capabilities = ["agent", "worker", "fast", "tools", "code"]
```

AgentServer.ps1 reads `ModelCloud.toml` at startup, resolves providers by capability,
and routes GPT-OSS task dispatches to the right worker. `.env.example` provides the
API token template — copy to `.env` and fill in keys.

**bots.py — micronaut orchestration engine** (`.Powernaut-v1.0.0/kuhul/MX-2/bots.py`, 173 lines):

4-tier fallback chain per micronaut:

| Tier | Backend | Endpoint | When |
|---|---|---|---|
| 1 | `micronaut.exe` CLI | Native K-shell subprocess | Always preferred if available |
| 2 | Ollama local | `localhost:11434` | CLI missing, Ollama running |
| 3 | LM Studio local | `localhost:1234` | Ollama not running |
| 4 | DeepSeek / Ollama Cloud | API endpoints | No local runtime, API token set in `.env` |

Each micronaut can copy this pattern — change the manifest (`MX-2.xjson`), change the
weights path, and `run_prompt()` handles the rest. The orchestrator (`launch-orchestrator.ps1`,
`orchestrator-simple.mjs`) calls `bots.py run_prompt()` per micronaut domain.

**MX-2 is an ngram learning engine.** It uses `MX-2.xjson` as its manifest, learns from
interactions via `evolve()`/`save()`/`load()` calls on the native CLI, and persists
ngram patterns in `brains/` and `model.forge.bin`. The `evolution/` directory drives
the learning loop via `evolution_bot.h` from the semantic engine.

**Worker dispatch flow:**
```
GPT-OSS plans task
  → kuhul_task_boss resolves provider
    → if provider = deepseek-flash:
        AgentServer.ps1 → POST https://api.deepseek.com/v1/chat/completions
    → if provider = ollama-cloud:
        AgentServer.ps1 → POST https://api.ollama.com/v1/chat/completions
    → result flows back to GPT-OSS for next planning step
```

**Tool training:** GPT-OSS was tool-trained with a Jinja chat template. The GGUF metadata
header (`tokenizer.chat_template` key) contains the full tool-calling schema — every tool
the model knows how to invoke. kuhul-server reads this at engine startup and registers the
tools in the MCP_TOOLS registry.

**Updating tool definitions:** To add/remove tools the model can call:
1. Extract current GGUF metadata:
   ```powershell
   cd khanary-llama-build\llama.cpp\gguf-py
   python -c "from gguf import GGUFReader; r=GGUFReader(r'C:\Users\canna\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf'); [print(f'{k}: {v}') for k,v in r.metadata.items() if 'tool' in k.lower() or 'template' in k.lower() or 'chat' in k.lower()]"
   ```
2. Patch `tokenizer.chat_template` with updated Jinja tool definitions
3. Re-pack GGUF using `gguf-py/gguf/gguf_writer.py`:
   ```powershell
   python examples/writer.py --set-metadata tokenizer.chat_template=@updated_template.jinja
   ```
4. Restart kuhul-engine — kuhul-server re-reads GGUF metadata on next health probe

**Current tool set** (from model training): task.plan, app.create, app.inspect, build.game,
build.website, build.program, build.micronaut — same verbs as MCP `kuhul_task_boss`.

**No GGUF patching needed.** The model was trained with a generic Jinja template that handles
any tool schema. Tool definitions are split between two layers:

| Layer | Location | What it holds | Updated how |
|---|---|---|---|
| **Tool format** | GGUF `tokenizer.chat_template` | How to structure a tool call (TypeScript types, enums, objects) | Retrain only if format changes |
| **Tool list** | kuhul-server `MCP_TOOLS` registry | Which 13 tools are available right now | Add to `MCP_TOOLS` array, restart |

When new tools are added (e.g. `kuhul_glyph_phase`, `kuhul_driver_dispatch`), only the
MCP_TOOLS registry needs updating. kuhul-server injects the tool list into the system prompt
at inference time. GPT-OSS outputs tool calls in its trained format; kuhul-server routes them.

### Self-extending system: GPT-OSS creates micronauts

The `build.micronaut` verb lets GPT-OSS spawn new micronauts at runtime. Each micronaut is a
~10-line JSON file with a name + sampling profile (temperature, repeat_penalty, stop tokens):

| Micronaut | Temp | Penalty | Purpose |
|---|---|---|---|
| `tool_call` | 0.1 | 1.0 | Deterministic tool calling, stops on `</tool_call>` |
| `coder` | 0.15 | 1.0 | Code generation, low temp for precision |
| `memory` | 0.4 | 1.15 | Conservative memory recall |
| `factory` | 0.7 | 1.1 | App scaffolding, moderate creativity |
| `evolution` | 1.0 | 1.35 | Creative exploration, highest entropy |
| `chat` | 0.8 | 1.3 | Free-text conversation |
| *(new)* | *GPT-OSS decides* | *GPT-OSS decides* | *Created on-the-fly for capability gaps* |

**Self-extending loop:**
```
GPT-OSS identifies capability gap
  → calls build.micronaut with name + sampling config
    → kuhul-server writes micronauts/<name>.json
      → new micronaut immediately dispatchable
        → improves data confidence for future runs
```

24 micronaut JSONs in `micronauts/` — the model can grow this directory on its own.

**Where tool calls live in the stack:**

| Layer | Role | Example |
|---|---|---|
| GGUF `chat_template` | **Formats** how the model outputs tool calls | Jinja template with TypeScript types |
| Atomic DOM manifest | **Defines** what tools are available | System prompt injection: "You have access to: task.plan, app.create..." |
| MCP_TOOLS registry | **Routes** tool calls to handlers | kuhul-server receives JSON-RPC, dispatches to driver/execFile |
| Micronaut worker | **Executes** the tool | coder generates code, factory scaffolds project |
| Studio plan checklist | **Displays** results to user | Plan items with ○/◉/●/⊗ status icons in canvas sidebar |

The Atomic DOM tells the model WHAT exists. The MCP handles the HOW. The studio handles the SHOW.

**Why Atomic DOM + micronauts is the complete identity layer:** The model knows HOW to
tool-call (Jinja format). The Atomic DOM manifest knows WHO the model is (NPC persona,
system prompt, sampling config, phase gating). The micronauts know WHAT the model can DO
(coder writes code, factory scaffolds, base handles general tasks, toolcall routes MCP,
research investigates, design styles). The model never sees the raw infrastructure — the
DOM layer injects identity and the micronaut layer executes actions. Swap the manifest or
swap the micronaut, you swap the model's behavior — same weights, different personality,
different capabilities.

Goal: use `gpt-oss-20b-MXFP4.gguf` (teacher, served at port 17480) to distil knowledge into
a LoRA adapter for `from_zero_v0.6_merged`. The adapter captures KUHUL domain knowledge from
the large model without full fine-tuning of the base weights.

### Micronaut worker architecture

GPT-OSS doesn't build alone — it plans tasks and dispatches to **micronaut workers**.
Each micronaut is a specialized agent that handles one task domain:

| Micronaut | Location | Task domain |
|---|---|---|
| `micronaut-coder` | `.NNC-K/bin/micronaut-coder/` | Code generation, refactoring, app builds |
| `micronaut-factory` | `.NNC-K/bin/micronaut-factory/` | App scaffolding, manifests, project structure |
| `micronauts/` (10 variants) | `.NNC-K/bin/micronauts/` | Agent, coder, design, kuhul, math, numatic, research, skill, supernaut, toolcall |
| `supernaut-cpp` | `.NNC-K/bin/supernaut-cpp/` | Orchestrator: assembly-line-executor, specialist registry, query router, result aggregator |
| `CodeWASM` | `.NNC-K/bin/CodeWASM/` | 18 tree-sitter WASM parsers (kuhul, python, cpp, js, ts, json, html, css...) |

**GPT-OSS → micronaut flow:**
```
User prompt
  → GPT-OSS plan (task.plan verb)
    → kuhul_task_boss dispatches to micronaut by task domain
      → micronaut-coder generates code
      → micronaut-factory scaffolds project
      → json_runtime allocates port + mounts sidecar
      → supernaut-cpp orchestrates multi-step builds
```

**tree-sitter WASM modules** — copied to `drivers/wasm/` (9 parsers):
`tree-sitter.wasm`, `tree-sitter-kuhul.wasm`, `tree-sitter-python.wasm`,
`tree-sitter-cpp.wasm`, `tree-sitter-javascript.wasm`, `tree-sitter-json.wasm`,
`tree-sitter-typescript.wasm`, `tree-sitter-css.wasm`, `tree-sitter-html.wasm`

These provide K'UHUL/KXML semantic parsing and code generation in the browser
via WASM — the XCFE-tree-sitter integration from the architecture plan.

### Micronaut C++ runtime (attention engine)

The micronaut worker architecture is backed by a native C++ runtime at
`.NNC-K/native/micronaut_cpp_runtime/` (50+ files):

| Subsystem | Files | What it does |
|---|---|---|
| **SCXQ2 IR** | `scxq2_runtime.cpp/.h` | SCXQ2 graph IR execution — runs the same IR the KLSL compiler emits |
| **Attention registry** | `attention_registry.cpp/.h`, `attention.registry.json` | Maps attention lanes (flash, linear, ring, sink, sliding, sparse, paged, cross, local) to implementations |
| **Attention ops** | `softmax.cpp`, `rotary.cpp`, `qkv_split.cpp`, `matmul.cpp`, `scale.cpp`, `mask.cpp` | Core attention math ops |
| **KV cache** | `kv_cache.cpp` | GPU-resident key/value cache management |
| **Variants** | `flash_attn.cpp`, `linear_attn.cpp`, `ring_attn.cpp`, `sink_attn.cpp`, `sliding.cpp`, `sparse_attn.cpp`, `paged_attn.cpp`, `cross_attn.cpp` | 8 attention algorithm variants |
| **Head routing** | `head_routing.cpp`, `mqa.cpp`, `gqa.cpp`, `mla.cpp` | Multi-Query / Grouped-Query / Multi-Latent Attention routing |
| **XCFE router** | `xcfe_router.cpp/.h` | Routes task list verbs to attention lanes via XCFE |
| **Scheduler** | `scheduler.cpp/.h`, `process.cpp/.h`, `batches.cpp/.h` | Task scheduling + batch processing |
| **Atomic DOM sandbox** | `frames.cpp/.h`, `canvas.cpp/.h`, `chat.cpp/.h`, `input.cpp/.h`, `output.cpp/.h` + `.Powernaut-v1.0.0/` (workspace) | **Model execution sandbox** — Atomic DOM blocks (FRAME, HEADER, MENU, BODY, GRID, FEED, FOOTER) rendered as SVG-3D via glyph engine with spherical geometry → `.apng` output. Powernaut wraps this in a full shell: MCP server (`mcp_server.mjs`), agent bridge scripts, GLSL/WGSL shader pipelines, 3D demo launchers, and model cloud routing (`ModelCloud.toml`). Models run INSIDE the Atomic DOM frame system — it's their sandbox, not just a renderer. |

### Powernaut surface (`.Powernaut-v1.0.0/` in workspace)

| Component | Files | Role |
|---|---|---|
| MCP server | `mcp_server.mjs`, `MCPSERVER.ps1` | MCP JSON-RPC endpoint for tool dispatch |
| Micronaut server | `micronaut-server.ps1` | Micronaut worker launcher |
| Agent server | `AgentServer.ps1` | Agent orchestration (spawn/control/list) |
| Model cloud | `ModelCloud.toml` | Model routing config (which model serves which task) |
| GLSL pipeline | `GLSL.ps1/.psd1/.psm1`, `neural_layer.glsl` | OpenGL 4.3 shader compilation + dispatch. **System-aware**: probes GPU ICD at startup (ig75icd64/atio6axx/nvoglv64), detects GL version, compiles shaders for target GPU. |
| WGSL pipeline | `wgsl.bat`, `neural_layer.wgsl`, `server.wgsl.json` | WebGPU shader pipeline |
| GLSL Server | `dist/GLSL_Server.exe`, `build/GLSL_Server_entry.py` | PyInstaller-packaged HTTP server wrapping `kuhul/glsl_server.py`. Probes GPU + serves GLSL compute results. Currently has PyInstaller packaging issue (missing `http` stdlib module) — works from source via `python kuhul/glsl_server.py`. |
| Kuhul3D | `build/Kuhul3D.cs`, `Kuhul3D.psm1` | C# + PowerShell SVG-3D renderer. Renders frames to PNG, animates to `.apng`. Test scripts in `build/` for terminal, glyph DLL, flipbook, export. |
| 3D demos | `demo_kuhul3d.ps1`, `demo_kuhul3d_anim.ps1`, `demo_kuhul3d_player.ps1` | SVG-3D → .apng demo launchers |
| Shell | `KuhulShell.ps1/.cmd`, `KUHUL.CMD`, `kuhul-shell-config.json` | PowerShell-based model shell |
| Bridges | `sk_bridge.py`, `kuhul_agent_bridge.py`, `kuhul_native_bridge.py`, `kuhul_vertex_bridge.py`, `kuhul_natives_bridge.py` | Python bridges to native engines |
| Server | `server.ps1`, `dist/GLSL_Server.exe` | GLSL compute server + compiled binary |
| UI | `kuhul-studio.html`, `thinking-splash.html`, `public/`, `ui/` | Web-based studio shell |
| Programs | `programs/` | JSON runtime programs for task execution |
| Routing | `routing_tools.json`, `model_tiers.json` | Tool + model tier routing config |
| **Threading** | `threads.cpp/.h` | Thread pool execution |

**The 4-bone LBS attention bias** from the GPT-2 trainer maps directly to this runtime:
`gpt2_kuhul_think_bias.hlsl` (trainer shader) → `attention_registry.cpp` (C++ registry) →
`sparse_attn.cpp` (sparse attention with bone-ID routing) → `head_routing.cpp` (expert head dispatch).

The subset `micronaut_attention_cpp` is the same runtime stripped to just the attention registry + ops — used for lightweight micronaut attention dispatch without the full SCXQ2 runtime.

### Micronaut sampling profiles

Each micronaut carries its own sampling parameters — callers pick a micronaut by name and
the dispatch layer injects the right values into the llama-server request body.
24 JSON profiles in `micronauts/`:

| Micronaut | Temp | Penalty | Purpose |
|---|---|---|---|
| `tool_call` | 0.1 | 1.0 | Deterministic tool calling, stops on `</tool_call>` |
| `coder` | 0.15 | 1.0 | Code generation, low temp for precision |
| `memory` | 0.4 | 1.15 | Conservative memory recall |
| `factory` | 0.7 | 1.1 | App scaffolding, moderate creativity |
| `evolution` | 1.0 | 1.35 | Creative exploration, highest entropy |
| `chat` | 0.8 | 1.3 | Free-text conversation |
| `default`/`khanary` | 0.8 | 1.1 | General-purpose fallback |
| `eliza` | 0.9 | 1.2 | Conversational agent |
| `librarian` | 0.3 | 1.05 | Knowledge retrieval, low temp |
| `chen`/`sek`/`wo`/`pop`/`xul`/`yax` | 0.5 | 1.0 | Phase-gated profiles (per K'UHUL phase) |

### MX-2 ngram learning engine

MX-2 (`bots.py`, 173 lines) is the micronaut orchestration engine with a 4-tier fallback chain:

| Tier | Backend | Endpoint | When |
|---|---|---|---|
| 1 | `micronaut.exe` CLI | Native K-shell subprocess | Always preferred if available |
| 2 | Ollama local | `localhost:11434` | CLI missing, Ollama running |
| 3 | LM Studio local | `localhost:1234` | Ollama not running |
| 4 | DeepSeek / Ollama Cloud | API endpoints | No local runtime, API token set in `.env` |

Each micronaut copies this pattern — change `MX-2.xjson` manifest, change weights path,
and `run_prompt()` handles dispatch. Learns from interactions via `evolve()`/`save()`/`load()`,
persists ngram patterns in `brains/` and `model.forge.bin`. The `evolution/` loop drives
continuous improvement via `evolution_bot.h` from the semantic engine.

### Self-extending: GPT-OSS builds new micronauts

The `kuhul_build_micronaut` MCP tool lets GPT-OSS spawn new micronauts at runtime.
Each new micronaut is a ~10-line JSON file written to `micronauts/<name>.json` with
sampling config, registered in MCP_TOOLS, and immediately dispatchable:

```
GPT-OSS identifies capability gap
  → calls kuhul_build_micronaut with name + sampling config
    → kuhul-server writes micronauts/<name>.json
      → registers in MCP_TOOLS dynamically
        → immediately available for dispatch
          → improves data confidence for future runs
```

Compiled micronaut DLLs in `dist/khanary-server/`:
- `micronaut_evolution.dll` — evolution engine (creative exploration)
- `micronaut_factory_core.dll` — factory engine (app scaffolding)

### Micronaut model format (`.micronaut`)

A working micronaut model format exists at `E:\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\`:

| File | Size | Format | Role |
|---|---|---|---|
| `model.micronaut` | 48 MB | JSON (`xcfe-model-1` schema) | Topology + routing metadata, SVG coordinate frame |
| `micronaut_brain.bson` | 2 GB | BSON | Trained brain weights (compressed binary) |
| `micronaut-weights-v2.s7` | 401 MB | S7 | Weight format v2 (quantized) |
| `micronaut.registry.xjson` | XJSON | Registry | Micronaut registry with skill bindings |

**Runtime layer** (17 TypeScript files): entropy ledger, GPU determinism harness,
portable composition operators, WebGPU backend binding, XCFE WGSL enforcement,
EBPD validation gates, per-platform determinism harness.

This format is production-ready — GPT-2 Medium DX11 was trained and deployed with it.
The MBX2LM brain (`MX2LLM/brain/`) directory structure maps directly to the MX-2
ngram learning engine pattern.

These provide K'UHUL/KXML semantic parsing and code generation in the browser
via WASM — the XCFE-tree-sitter integration from the architecture plan.

### Micronaut C++ runtime (attention engine)

The micronaut worker architecture is backed by a native C++ runtime at
`.NNC-K/native/micronaut_cpp_runtime/` (50+ files):

| Subsystem | Files | What it does |
|---|---|---|
| **SCXQ2 IR** | `scxq2_runtime.cpp/.h` | SCXQ2 graph IR execution — runs the same IR the KLSL compiler emits |
| **Attention registry** | `attention_registry.cpp/.h`, `attention.registry.json` | Maps attention lanes (flash, linear, ring, sink, sliding, sparse, paged, cross, local) to implementations |
| **Attention ops** | `softmax.cpp`, `rotary.cpp`, `qkv_split.cpp`, `matmul.cpp`, `scale.cpp`, `mask.cpp` | Core attention math ops |
| **KV cache** | `kv_cache.cpp` | GPU-resident key/value cache management |
| **Variants** | `flash_attn.cpp`, `linear_attn.cpp`, `ring_attn.cpp`, `sink_attn.cpp`, `sliding.cpp`, `sparse_attn.cpp`, `paged_attn.cpp`, `cross_attn.cpp` | 8 attention algorithm variants |
| **Head routing** | `head_routing.cpp`, `mqa.cpp`, `gqa.cpp`, `mla.cpp` | Multi-Query / Grouped-Query / Multi-Latent Attention routing |
| **XCFE router** | `xcfe_router.cpp/.h` | Routes task list verbs to attention lanes via XCFE |
| **Scheduler** | `scheduler.cpp/.h`, `process.cpp/.h`, `batches.cpp/.h` | Task scheduling + batch processing |
| **Atomic DOM sandbox** | `frames.cpp/.h`, `canvas.cpp/.h`, `chat.cpp/.h`, `input.cpp/.h`, `output.cpp/.h` + `.Powernaut-v1.0.0/` (workspace) | **Model execution sandbox** — Atomic DOM blocks (FRAME, HEADER, MENU, BODY, GRID, FEED, FOOTER) rendered as SVG-3D via glyph engine with spherical geometry → `.apng` output. Powernaut wraps this in a full shell: MCP server (`mcp_server.mjs`), agent bridge scripts, GLSL/WGSL shader pipelines, 3D demo launchers, and model cloud routing (`ModelCloud.toml`). Models run INSIDE the Atomic DOM frame system — it's their sandbox, not just a renderer. |

### Powernaut surface (`.Powernaut-v1.0.0/` in workspace)

| Component | Files | Role |
|---|---|---|
| MCP server | `mcp_server.mjs`, `MCPSERVER.ps1` | MCP JSON-RPC endpoint for tool dispatch |
| Micronaut server | `micronaut-server.ps1` | Micronaut worker launcher |
| Agent server | `AgentServer.ps1` | Agent orchestration (spawn/control/list) |
| Model cloud | `ModelCloud.toml` | Model routing config (which model serves which task) |
| GLSL pipeline | `GLSL.ps1/.psd1/.psm1`, `neural_layer.glsl` | OpenGL 4.3 shader compilation + dispatch. **System-aware**: probes GPU ICD at startup (ig75icd64/atio6axx/nvoglv64), detects GL version, compiles shaders for target GPU. |
| WGSL pipeline | `wgsl.bat`, `neural_layer.wgsl`, `server.wgsl.json` | WebGPU shader pipeline |
| GLSL Server | `dist/GLSL_Server.exe`, `build/GLSL_Server_entry.py` | PyInstaller-packaged HTTP server wrapping `kuhul/glsl_server.py`. Probes GPU + serves GLSL compute results. Currently has PyInstaller packaging issue (missing `http` stdlib module) — works from source via `python kuhul/glsl_server.py`. |
| Kuhul3D | `build/Kuhul3D.cs`, `Kuhul3D.psm1` | C# + PowerShell SVG-3D renderer. Renders frames to PNG, animates to `.apng`. Test scripts in `build/` for terminal, glyph DLL, flipbook, export. |
| 3D demos | `demo_kuhul3d.ps1`, `demo_kuhul3d_anim.ps1`, `demo_kuhul3d_player.ps1` | SVG-3D → .apng demo launchers |
| Shell | `KuhulShell.ps1/.cmd`, `KUHUL.CMD`, `kuhul-shell-config.json` | PowerShell-based model shell |
| Bridges | `sk_bridge.py`, `kuhul_agent_bridge.py`, `kuhul_native_bridge.py`, `kuhul_vertex_bridge.py`, `kuhul_natives_bridge.py` | Python bridges to native engines |
| Server | `server.ps1`, `dist/GLSL_Server.exe` | GLSL compute server + compiled binary |
| UI | `kuhul-studio.html`, `thinking-splash.html`, `public/`, `ui/` | Web-based studio shell |
| Programs | `programs/` | JSON runtime programs for task execution |
| Routing | `routing_tools.json`, `model_tiers.json` | Tool + model tier routing config |
| **Threading** | `threads.cpp/.h` | Thread pool execution |

**The 4-bone LBS attention bias** from the GPT-2 trainer maps directly to this runtime:
`gpt2_kuhul_think_bias.hlsl` (trainer shader) → `attention_registry.cpp` (C++ registry) →
`sparse_attn.cpp` (sparse attention with bone-ID routing) → `head_routing.cpp` (expert head dispatch).

The subset `micronaut_attention_cpp` is the same runtime stripped to just the attention registry + ops — used for lightweight micronaut attention dispatch without the full SCXQ2 runtime.

### GPT-OSS model paths

| Format | Path | Size |
|--------|------|------|
| GGUF (MXFP4) — teacher server | `C:\Users\canna\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf` | 11.28 GB |
| HF sharded (24 layers, xshard) | `E:\models\GPT-OSS\hf\layer_00` … `layer_23` | ~12 GB total |
| HF model config | `E:\models\GPT-OSS\hf\model_config.json` | arch: hidden=2880, heads=64, kv_heads=8, experts=32, top_k=8, vocab=200064 |
| Conversion report | `E:\models\GPT-OSS\hf\conversion_report.json` | **DONE** — 24/24 layers converted, 4640s total (experts 4527s, attention 94s) |

### Semantic engine — dual role

The semantic engine (`desktop/semantic_engine/include/`, 28 headers) serves two functions in the stack:

| Role | What it does | Key headers |
|---|---|---|
| **Inference executor** | Runs model forward pass through FieldExecutionEngine | `field_execution_engine.h`, `dx12_executor.h`, `executor.h` |
| **Model converter** | GGUF → xshard hot-swap layer format | `dds_shard_loader.h`, `tokenizer.h`, `quant.h`, `kv_cache.h` |

**Conversion pipeline (GPT-OSS 20B, proven):**
```
gpt-oss-20b-MXFP4.gguf (11.28 GB)
  → dds_shard_loader → layer_00/layer_01/.../layer_23/
    ├── q.bin / k.bin / v.bin / o.bin     (attention weights)
    └── gate.bin / up.bin / down.bin       (expert FFN weights)
  → hot_swap_classes: [0]  (class 0 = experts, streamed on-demand)
  → cold_load_once_classes: [2]  (class 2 = embeddings, loaded once)
```

**Why xshard:** HD 4600 has 1792 MB VRAM — can't fit 11.28 GB. xshard streams individual layers on-demand via `dds_shard_loader.h`. The kuhul_engine hot-swap layer format loads only the layers needed for the current forward pass, unloads them after.

**Architecture: 32 experts, top-8 routing** — the 4-bone LBS attention bias from the trainer maps to the expert routing layer. Each expert is a separate FFN shard; the glyph engine's lanes map to specific expert classes.

### Conversion pipeline scripts (`.NNC-K/scripts/`)

The full GGUF → xshard conversion + forward pass pipeline lives at `C:\Users\canna\.NNC-K\scripts\`:

| Script | Role |
|---|---|
| `gguf_to_xshard.py` | Main conversion: GGUF → per-layer shards |
| `gguf_experts_to_xshard.py` | Expert FFN shard extraction (32 experts → gate/up/down per layer) |
| `gguf_layer_to_safetensors.py` | Single layer extraction to safetensors |
| `safetensors_to_xshard.py` | Safetensors → xshard (for from_zero model) |
| `convert_all_layers_to_xshard.py` | Batch convert all 24 layers |
| `gptoss_harmony_forward.py` | Full GPT-OSS forward pass (harmony = MoE routing) |
| `gptoss_multi_layer_forward.py` | Multi-layer forward pass |
| `gptoss_layer_forward.py` | Single-layer forward pass |
| `adaptive_layer_streaming.py` | Adaptive layer streaming (hot-swap scheduler) |
| `xcfe_router.py` | XCFE routing from task list to model layers |
| `research_bot.py` | Research bot — automated analysis across xshard layers |
| `NNCK-Runtime.psm1` | PowerShell runtime module (SDK bridge) |
| `folds.toml` | Fold configuration (TENSOR, ARC, KERNEL, PHASE) |
| `CM1_FOLD_TENSOR.s7` | CM-1 brain fold tensor program |
| `BOSS-Guide.md` | TaskEngine BOSS layer documentation |

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
  --engine   http://127.0.0.1:17480
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

## KUHUL APPS — HYBRID MCP STUDIO (Revised Architecture)

`khanary-llama-build/llama.cpp/tools/ui/` becomes **KUHUL APPS**: an AI app generation studio.
The architecture pivots from "TypeScript services in the Svelte frontend" to **backend-driven MCP
tools with a thin frontend shell**. The orchestration layer lives in C++/WASM, not TypeScript.

### Architecture principle: backend-heavy, UI-thin

```
┌─ Svelte Frontend (thin shell) ──────────────────────────────────────┐
│  Chat input → MCP dispatch → canvas iframe + plan checklist          │
│  No orchestration logic. No task routing. No custom service layer.   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ MCP JSON-RPC
                               ▼
┌─ Backend Stack ─────────────────────────────────────────────────────┐
│                                                                      │
│  llama.cpp (port 17480)          kuhul-server (port 8764)            │
│  ├─ /v1/chat/completions         ├─ MCP gateway                      │
│  ├─ Native MCP server host       │  ├─ kuhul_task_boss               │
│  │  └─ tools served from         │  ├─ kuhul_json_runtime            │
│  │     model backend directly    │  ├─ kuhul_forge                   │
│  └─ AtomicDOM model registry     │  └─ kuhul_wwa_host                │
│                                  │                                   │
│  json_runtime (port 8787)        │  khanary_driver.dll                │
│  ├─ /api/file-manager/*          │  ├─ TaskEngine + DAG               │
│  ├─ /api/sidecars (8800-8899)    │  ├─ kd_load_tasks / kd_plan        │
│  └─ 180+ route surface           │  │  / kd_run / kd_dispatch        │
│                                  │  ├─ Loaded by kuhul-server         │
│  Micronaut Hive                  │  │  via ffi-napi/koffi            │
│  E:\models\.hive                 │  └─ Dispatches to bot.py per       │
│  ├─ bot.py per micronaut         │     micronaut domain               │
│  └─ khanary_driver.dll           │                                    │
│     dispatches tasks to           │                                    │
│     micronaut by task domain      │                                    │
└──────────────────────────────────┴──────────────────────────────────┘
```

### Why this replaces the 8 missing TypeScript files

The previous PLAN.md listed 8 service/component files as "DONE" — none exist on disk
(audited 2026-08-06, 523 source files scanned). Rather than rebuild them in TypeScript,
each function maps to existing C++ infrastructure in one native driver DLL:

| Old TS file (missing) | Replaced by | Location |
|---|---|---|
| `kuhul-stack.service.ts` | kuhul-server health probes | Already live on 8764 |
| `gateway-mcp.service.ts` | llama.cpp native MCP server host | Model backend, no separate client |
| `studio-task-router.ts` | `khanary_driver.dll` — kd_plan / kd_dispatch | `drivers/khanary_driver.cpp` |
| `projects.service.ts` | `json_runtime` `/api/file-manager/*` | Already live on 8787 |
| `task-engine.ts` (types) | `TaskList.json` / `TaskList.kuhul` schemas | `drivers/` |
| `CanvasPreview.svelte` | Simple iframe + srcdoc binding | ~30 lines, no service needed |
| `extract-html-doc.ts` | Regex in canvas component | ~10 lines |
| `atomicdom-models.ts` | `models/*/atomic.manifest.json` scan | Read manifests at build time |

### Driver DLL architecture

One native DLL (`khanary_driver.dll`) replaces 3 WASM compilation targets. Flat C ABI,
compiles with MSVC, loaded by kuhul-server via `ffi-napi` or `koffi`. Same sources as the
WASM plan but no Emscripten, no separate artifacts per module. Individual vendor drivers
(OpenCL, DML, CUDA) can be separate DLLs exporting the same 7-function ABI — swapped at
load time without JS changes.

**Load path in kuhul-server.cjs:**
```js
const ffi = require('ffi-napi');
const driver = ffi.Library('khanary_driver', {
  kd_create:      ['pointer', ['string']],
  kd_load_tasks:  ['int',    ['pointer', 'string', 'pointer', 'int']],
  kd_plan:        ['string', ['pointer']],
  kd_run:         ['string', ['pointer']],
  kd_dispatch:    ['string', ['pointer', 'string']],
  kd_task_count:  ['int',    ['pointer']],
  kd_destroy:     ['void',   ['pointer']],
  kd_free_string: ['void',   ['string']],
});
```

**C ABI surface (7 exports):**
| Function | Role |
|---|---|
| `kd_create(providers_json)` → handle | Init driver with provider registry |
| `kd_load_tasks(handle, json, err_buf, err_len)` → 1/0 | Parse + validate TaskList JSON |
| `kd_plan(handle)` → JSON string | Topological plan with provider resolution |
| `kd_run(handle)` → JSON string | Plan + mark ready for MCP dispatch |
| `kd_dispatch(handle, task_json)` → JSON string | Single-task dispatch to micronaut |
| `kd_task_count(handle)` → int | Number of loaded tasks |
| `kd_destroy(handle)` | Free driver |

**Build (MSVC, no Emscripten):**
```powershell
cd drivers
cl /LD /EHsc /O2 /Fe:khanary_driver.dll ^
  khanary_driver.cpp DAG.cpp ^
  /I. /link /OUT:khanary_driver.dll
```

**Files written / fixed (2026-08-06 / 2026-08-07):**

| File | Purpose |
|---|---|
| `khanary_driver.h` | Public C ABI header (`__declspec(dllexport)`) |
| `khanary_driver.cpp` | Implementation — TaskEngine + DAG + JSON parser + 7 exports |
| `webx_stubs.h` | Minimal WebX::Provider / ProviderManager stubs |
| `task_engine.h` | **Added** — minimal TaskSpec/TaskResult/TaskEngine stub so `khanary_driver.cpp` compiles standalone |
| `studio_tasklist_example.json` | Reference TaskList for MCP micronaut dispatch. **Fixed 2026-08-07**: tasks now live at top-level `"tasks"` array so `kd_load_tasks` can parse it directly. |
| `TaskList.json` / `TaskList.kuhul` | Canonical schemas (copied from `.NNC-K`) |

### Micronaut hive dispatch

The `E:\models\.hive` system uses `bot.py` per micronaut. Each micronaut handles one task domain:

| Micronaut | Task domain | bot.py role |
|---|---|---|
| `.micronaut-coder` | Code generation, refactoring | Generates app source from TaskList |
| `.micronaut-factory` | App scaffolding, manifests | Creates project structure, manifest.json |
| `.micronaut` (base) | General task execution | Fallback for unclassified tasks |
| `MICRONAUT_V0`–`V2` | Versioned micronaut engines | Progressive capability tiers |

khanary_driver.dll resolves the provider (micronaut) from the TaskList, then kuhul-server
dispatches to the right bot.py. Each bot.py is a standalone Python script that:
1. Receives a TaskSpec JSON via stdin or HTTP
2. Executes the task (generate code, create files, compile)
3. Returns TaskResult JSON

Duplication is by design — `bot.py` is ~200 lines, trivially copied per micronaut domain.

### DAG scheduling (inside khanary_driver.dll)

**DAG.cpp** — included in the driver DLL build, no separate module needed:
- Topological sort of task dependencies
- Cycle detection
- Dependency validation (missing deps → error)

SCXcache persistence and XCFE-tree-sitter parsing are deferred to Phase 2 —
the driver DLL ships the DAG scheduler and TaskEngine core first. The full
FieldGraph orchestration (node scheduling by pressure, retry, phase gating)
can be added to the driver as additional exports when needed.

### llama.cpp MCP server hosting

llama.cpp already supports MCP server integration natively. Instead of a separate
`kuhul-server.cjs` acting as the MCP gateway, the model backend hosts tools directly:

```
Client (Svelte)  ──MCP JSON-RPC──▶  llama.cpp (port 17480)
                                       ├─ /v1/chat/completions  (model inference)
                                       └─ /mcp                   (tool host)
                                            ├─ tools/list
                                            ├─ tools/call
                                            │   ├─ task.plan      → khanary_driver.dll
                                            │   ├─ app.create     → KuhulAppCreator
                                            │   ├─ app.inspect    → json_runtime
                                            │   ├─ build.game     → micronaut-factory
                                            │   ├─ build.website  → micronaut-coder
                                            │   └─ build.program  → micronaut-coder
                                            └─ resources/*
```

This eliminates the separate kuhul-server gateway hop. llama.cpp's MCP support
means tools are defined in the model server config, not in a Node.js process.
If llama.cpp's MCP implementation doesn't cover all verbs yet, kuhul-server (8764)
remains as the fallback gateway — but the target is single-process.

### Three-panel studio layout (unchanged)

```
Left sidebar            Center canvas              Right chat
─────────────────       ─────────────────────      ────────────────────
Projects list           Live preview iframe        Chat input
(conversations)         (generated HTML/code       Model selector
Theme switcher          from TaskEngine output)    Plan checklist
Settings                Export / Publish btns      (khanary_driver.dll)
```

### Theme system (unchanged)

| Theme | Default | CSS variables |
|---|---|---|
| Dark (KUHUL default) | YES | `--bg: #0d0d0d`, `--sidebar: #1e293b`, `--accent: #6366f1` |
| Light | no | `--bg: #f8fafc`, `--sidebar: #f1f5f9`, `--accent: #4f46e5` |
| Kuhul indigo | no | `--bg: #1e1b4b`, `--sidebar: #312e81`, `--accent: #a5b4fc` |

### Key files changed (branding — already done, retained)

| File | Change |
|---|---|
| `src/lib/assets/logo.svg` | Replaced llama logo with KUHUL K glyph (indigo gradient) |
| `src/lib/constants/app.ts` | APP_NAME = 'KUHUL APPS' (via `VITE_PUBLIC_APP_NAME`) |
| `src/lib/constants/ui.ts` | 'New chat' → 'New project' |
| `SidebarNavigation.svelte` | 'Rename conversation' → 'Rename project' |
| `ChatScreenGreeting.svelte` | Greeting changed to KUHUL APPS |
| `.env` | `VITE_PUBLIC_APP_NAME='KUHUL APPS'`, server origin = port 17480 |

### Stack backend (unchanged — live state)

- kuhul_engine (port 17480): OpenAI-compatible `/v1/chat/completions` + MCP tool host
- kuhul-server (port 8764): MCP gateway fallback, micronaut routing, health probes
- json_runtime (port 8787): hosting API, file-manager, sidecars (ports 8800-8899)
- `dist/khanary-server/`: `node kuhul-server.cjs` auto-starts engine + server

---

## PRIMEOS — Stack Management Layer

`C:\Users\canna\_khanary_inspect\desktop\PRIMEOS\bin\Release\net8.0-windows\PRIMEOS.exe`

PRIMEOS is a .NET 8 Windows desktop app that acts as the management UI for the full NNC-K
/ KUHUL APPS stack. Long-term goal: all services (kuhul_engine, JSON runtime, MCP tools,
micronaut factory, kuhul-server) can be started, stopped, monitored, and updated from PRIMEOS.

### Integration surface

| Component | How PRIMEOS manages it |
|---|---|
| `kuhul_engine.exe` | Start/stop via process API; port 17480 health check |
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