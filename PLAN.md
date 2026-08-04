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

## Status

- [x] `trainer/` folder created with source + shaders
- [x] `trainer/d3d11_engine.h/.cpp` — trainer-specific D3D11 (no XVM dependency)
- [x] `trainer/CMakeLists.txt` — standalone CMake with FetchContent nlohmann_json
- [x] Dataset links captured (above) — full `E:\data` inventory
- [x] Model registry captured — 7 models documented
- [x] `tools/gen_kuhul_training.py` — synthetic corpus generator (350,388 examples)
- [x] `tools/kuhul_dataset_validator.py` — validate + compile π-KUHUL structured records
- [x] `tools/extend_vocab.py` — patch checkpoint wte [50260,768] → [50270,768]
- [x] `tokenizer_config.json` — KUHUL token IDs 50260–50269 at repo root
- [x] `pi_kuhul/` — KuhulPhysics.h, SphericalGeometryAVX2.h, DirectXMathAVX2.h, Fold2DCompiler.h
- [x] `trainer/shaders/gpt2_kuhul_think_bias.hlsl` — π-nary geodesic arc + brain expert routing shader (3 stacked modifiers)
- [x] 7 missing shaders copied from v0.4.0 into `trainer/shaders/`
- [x] Build: `gpt2_trainer.exe` 279 KB, compiled MSVC 19.44, running on HD 4600
- [x] Shader compile flag: `D3DCOMPILE_SKIP_OPTIMIZATION` → `D3DCOMPILE_OPTIMIZATION_LEVEL3`
- [x] v0.2 training run started — 5000 steps, `kuhul_tokens.bin`, lr=3e-5
- [x] **Re-tokenized** `kuhul_synthetic.jsonl` with KUHUL tag injection → `E:\data\kuhul_tokens_kuhul.bin` (946,503 seqs × 128; 203,124 KUHUL tokens = 0.17% of stream)
- [x] **Brain expert routing wired** (`loadBrainExperts`, `buildThinkDepth`, dispatch, all 3 files): activate with `$env:GPT2_THINK_BIAS=1` (+ optional `$env:GPT2_BRAIN_EXPERTS=<path>`)
- [ ] Rebuild `gpt2_trainer.exe` to pick up new code
- [ ] Run `extend_vocab.py` on `from_zero_v0.1.safetensors` → `from_zero_v0.1_kuhul.safetensors`
- [ ] v0.3 training run with extended vocab + KUHUL tokens in data
- [ ] Decision A: split `gpt2_attn_fwd.hlsl` → QK / think-bias / softmax+V (currently post-softmax, which works but is non-standard)
- [ ] Decision B: DML GEMM bridge mode

### Brain expert routing (GPT2_THINK_BIAS=1)

**How it works (3 modifiers stacked, post-softmax on P_buf):**
1. π-nary arc: `sin(depth/2)²` — tokens inside `<THINK>…</THINK>` get geodesic-distance-weighted attention boost
2. KuhulPhysics scale: `antigravity_scale` (0.1→1.0) — strengthens as training stabilises
3. Brain expert cluster: `brain_experts_[tok % 30628]` ∈ [0,60] — same Delaunay cluster → co-attention boost

**Path:** `brain2/experts.bin` relative to build dir, or `$env:GPT2_BRAIN_EXPERTS=<full_path>`.

**Convergence note:** loss ~6.0 at step 116-117 on `kuhul_tokens_kuhul.bin` (487 MB, 121M tokens) is **expected** — the model has seen 60K tokens (0.05% of data). CPU hit 0.00 overnight on the small `tokens_hdr.bin` (12 MB, repeating). To verify GPU backward works: run a quick overfit with `--data E:\data\kuhul_tokens.bin --steps 200 --lr 1e-3` on a tiny slice, or use `tokens_hdr.bin`.
