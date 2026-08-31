<p align="center"><img src=https://github.com/cannaseedus-bot/KHANARY/blob/main/khanary.png style="width:350px;"></p>

## Multi-alphabet Semantic Encoding and Execution Substrate for Deterministic Neural Compute Pipelines

```
KHΛNARY encodes tensor operations and control flow into 32-bit **Knowledge Numeric Unit** (KNU) words using the `KHΛ-2-DENSE-32`
profile, enabling deterministic replay of neural compute workloads on CPU with optional iGPU acceleration via WebGPU.
```

---

## Build

```powershell
# One command — clears stale UI cache, builds Svelte frontend, compiles
# llama.cpp with GPU backends, deploys to dist/khanary-server/
llama-build deploy

# Launch the full KUHUL APPS stack (4 services)
START-SERVERS
```

| Script | Role |
|---|---|
| `llama-build.bat` | Full build: clears stale UI cache → npm build → cmake → GPU DLLs |
| `llama-build.bat deploy` | Copies fresh binary + DLLs to `dist/khanary-server/` |
| `START-SERVERS.bat` | Launches json_runtime, kuhul-server, kuhul_engine, llama-server |
| `build-khanary.bat` | Alias for `llama-build` (no more stale UI builds) |

**Prerequisites:** VS 2022/2026 BuildTools (MSVC, C++ workload), CMake ≥ 3.20, Node.js ≥ 18, npm.

**Drivers** live in `drivers/` — `json_runtime_lib.dll` (hosting API), `native_glyph_engine.dll` + `native_glyph_engine_abi.dll` (K'UHUL glyph rendering), `khanary_driver.dll` (TaskEngine + DAG + provider dispatch), and `khanary_glyph_driver.dll` (unified phase/fold glyph + compute lane registry — 12 phase opcodes + 13 GGML lanes). kuhul-server loads them via ffi-napi. CLI entry at `bin/native_glyph_engine_cli.exe`.

> **Raw `.dll` files are not committed** (GitHub flags them on push). They ship in
> [`khanary-driver-dlls.zip`](khanary-driver-dlls.zip) at the repo root — **unzip at
> the repo root**; the paths inside mirror the repo layout. Full inventory in
> `driver.manifest.json`.

**K'UHUL toolchain** (semantic modules + compiler + runtime):

```powershell
npm i kuhul-es            # the K'UHUL-ES runtime (pi/tau/glyph + physics engine) — published on npm
python tools/khlc.py stdlib/            # compile .kuhul/.khl semantic modules -> KAST -> .kson
python tools/kson_validate.py stdlib/   # admission gate (ADMITTED / REJECTED)
kuhul-es compile examples/hello.kuhules # .kuhules -> canonical KAST (.kson)
kuhul-es train examples/train_sin.json  # GLSL trainer (semantic skeleton + physics)
```

**Compiler repos:**

| Repo | Compiler | Input | Output |
|------|----------|-------|--------|
| https://github.com/cannaseedus-bot/KXC | KXC (`kxc.exe`) | `.kuhul` kernel descriptor | HLSL / WGSL / SMCA JSON / CPU C++ |
| https://github.com/cannaseedus-bot/KHLC-PY | KHLC (`khlc.py`) | `.kuhul` / `.khl` semantic source | KAST / KSON |
| https://github.com/cannaseedus-bot/SMCA | SMCA registry | architecture spec | kernel-classes / collapse-geometry |

**Ports after launch:**

| Port | Service |
|---|---|
| 9000 | KUHUL APPS Studio (llama-server) |
| 8764 | MCP Gateway (kuhul-server) |
| 17480 | Inference Engine (kuhul_engine) |
| 8787 | Hosting API (json_runtime) |

**json_runtime engine role:** it is a manifest-driven **lane/phase/opcode
engine**, split by layer:

| Layer | Responsibility | Repo authority |
|---|---|---|
| Lane engine | Routes work to JSON runtime, K'UHUL folds, SCXQ2 sidecars, native glyph engine, coder, GPU, or trainer lanes | `programs/async.manifest.json`, `programs/actions.manifest.json`, `programs/functions.manifest.json` |
| Phase engine | Enforces Pop → Wo → Yax → Sek → Ch'en → Xul law and fold transitions | `stdlib/*.kuhul`, `programs/kuhul_dispatch.json`, `dist/Kuhul-c++/build/Release/native_glyph_engine*.dll` |
| Opcode engine | Executes primitive ops such as GET, SET, CALL, GPU_DISPATCH, FOLD_ENTER, and FOLD_EXIT | `programs/stdlib.json`, `dist/json-runtime/build/Release/json_runtime.exe`, `json_runtime_lib.dll` |

Short version: **XJSON is the lane/control manifest, K'UHUL is the phase law,
and json_runtime is the opcode + side-effect executor.** Full inventory and
contracts: [`JSON-RUNTIME.md`](JSON-RUNTIME.md).

Product framing: [`MAKE-CAKE.md`](MAKE-CAKE.md) defines this stack as the
**K'UHUL Animated Comic Book Engine**: a semantic object server with NPC/game
engine execution mechanics.

**Studio canvas:** open a chat, then append `/canvas` to the URL.

---

## The gap KHANARY fills — universal GPU inference without a hardware purchase

llama.cpp ships backends for: CUDA (NVIDIA only), HIP/ROCm (AMD only), Metal (Apple only), Vulkan, DirectML, OpenCL (experimental), CPU. **It has no OpenGL backend.**

OpenGL 4.3 compute shaders have existed since 2012. They run on every GPU — Intel integrated, AMD, NVIDIA, mobile, decade-old discrete cards. `GL_ARB_compute_shader` + SSBOs give you the same compute primitives as CUDA for inference: matrix multiply, reductions, elementwise ops. The hardware has been sitting there for 13 years on every PC that shipped with a graphics card.

The ML ecosystem defaulted to CUDA because NVIDIA funds developer relations, writes the tutorials, and sponsors the research. OpenGL got associated with "old graphics" exactly when ML inference took off — even though OpenGL 4.3 compute has nothing to do with rasterization or the display pipeline. The result: someone with a $300 laptop from 2014 has a fully working compute-capable GPU that inference frameworks refuse to use, and they're told to buy a $1,500 card.

**KHANARY closes this gap.**

KLSL (the project's shader IR) already emits to HLSL (D3D11) and WGSL (WebGPU). GLSL is a third emit target — not a rewrite, purely syntactic translation:

| HLSL | GLSL |
|------|------|
| `StructuredBuffer<float> A : register(t0)` | `layout(std430, binding=0) readonly buffer A { float a[]; }` |
| `[numthreads(16,16,1)]` | `layout(local_size_x=16, local_size_y=16, local_size_z=1) in;` |
| `SV_DispatchThreadID` | `gl_GlobalInvocationID` |
| `groupshared float s[256]` | `shared float s[256]` |
| `GroupMemoryBarrierWithGroupSync()` | `barrier()` |

The same 7 inference shaders that power `d3d11_infer.dll` (full GPT-2 forward pass: GEMM, embed, layernorm, attention, GELU, add_bias, residual add) translate line-for-line to GLSL. `gl_infer.dll` (in progress) exposes the same C API and dispatches through `ig75icd64.dll` — Intel's OpenGL 4.3 ICD, confirmed present on every Intel iGPU since ~2013 — and through the equivalent ICD on every AMD and NVIDIA GPU that has shipped a driver in the last decade.

| Backend | Hardware required | Purchase required |
|---------|------------------|-------------------|
| CUDA | NVIDIA GPU | Yes — $300–$2000+ |
| HIP/ROCm | AMD discrete GPU | Yes — $200–$800+ |
| Metal | Apple Silicon or AMD Mac GPU | Yes — Apple hardware |
| **OpenGL 4.3 (GLSL)** | **Any GPU since 2012** | **No** |

**Every GPU since 2012. No hardware purchase. No NVIDIA. No Apple. No AMD discrete.**

That is what KHANARY contributes to the ML ecosystem: the GPU path that was always there, that every machine already has, that llama.cpp and every major inference framework chose not to implement because NVIDIA captured the tooling first.

---

## What the K'UHUL Semantic Engine and XJSON Micronauts add to llama.cpp

llama.cpp is `POST /v1/chat/completions` → tokens. It is a very fast inference engine with no opinion about routing, state, model identity, or what happens between calls. Stateless. Single model at a time. Sampling params come in per request or from a hardcoded system prompt.

That is the entire surface. Everything above the inference call is your problem.

### K'UHUL Semantic Engine — physics-based routing above inference

llama.cpp has no routing layer. The `FieldExecutionEngine` puts one above the inference call:

- **Gravity wells** — `clamp(1.0 + 0.35·pressure - 0.25·entropy + 0.15·attention + 0.10·affinity, 0.1, 4.0)`. The engine maintains runtime physics state (entropy, attention, pressure) that evolves per tick and shapes *how* a request is routed, not just where it goes.
- **Semantic MoE routing** — "code"/"refactor" → AgentCoder, "create"/"new" → AgentFactory, else → GGUF inference. Content determines the execution path.
- **Phase gating** (Pop→Wo→Yax→Sek→Ch'en→Xul) — a state machine layered over the inference call. What operations are available at Sek are not available at Pop. llama.cpp is a function; this is a gated execution substrate.
- **SharedMemoryBridge** (`Local\KuhulGeometricState`) — cross-process telemetry so PRIMEOS sees real engine state every tick. llama.cpp has no equivalent visibility surface.
- **SCXcache DAG** — persistent semantic working-set written at Xul phase, readable at Pop of the next call. llama.cpp is stateless between requests; the cache is not.

### XJSON Micronauts / Atomic DOM — model identity separate from model weights

llama.cpp's model "identity" is a GGUF header and a chat template string. Nothing else.

The Atomic DOM (`models/{alias}/atomic.manifest.json`) gives each model a complete identity document:

- **Chat template** — `kxml/v1` for KHANARY-trained models (glyph tokens baked into weights), `chatml` for stock GGUF. The same DOM drives both. The raw JSON is never sent to the model — the DOM layer constructs exactly what the model sees.
- **`npc.system_prompt`** — the only text injected as a system message. Kept short and plain. Small models (Qwen 0.5B) get one sentence. The system prompt does not describe the runtime; it describes the model's role.
- **Micronauts** — context-aware sampling overlays. `tool_call`, `memory`, `coder`, and `chat` contexts each map to a different behavior profile (temperature, repeat_penalty, stop tokens). No retraining. Swap on context switch.
- **Tool registry** (`kuhul.tools.jsonl`) — per-model tool surface. Tools with `glyph_token: null` are internal to the runtime. The model only sees what the DOM exposes.
- **`execution_gated`** flag — `true` means K'UHUL phase gating applies; `false` means direct passthrough (story/creative tasks bypass the physics loop entirely).

### Planner/executor split — the model cannot execute, only propose

llama.cpp has no authority model. Whatever the model outputs can be treated as authoritative by naive callers.

KHANARY enforces a structural separation:

- **PM-1** (micronaut brain) = **PLANNER**. Model output. Non-authoritative. Produces a `TaskList` JSON proposing what should happen.
- **TaskEngine.cpp** (C++ authority) = **EXECUTOR**. Validates, plans, and runs. The model can never directly execute — it can only propose.

Model hallucinations stay proposals. The C++ layer decides what actually runs.

### The complete picture

| Layer | llama.cpp | KHANARY |
|-------|-----------|---------|
| Inference | Fast, multi-backend | Same (llama.cpp under the hood) |
| Routing | None — one endpoint | Semantic MoE, gravity-weighted dispatch |
| State | Stateless between calls | Physics state evolves per tick |
| Model identity | GGUF header + chat template | Atomic DOM — identity, tools, behavior profiles |
| Behavior switching | Re-prompt or retrain | Micronaut overlay, zero retraining |
| Authority | Model output is final | Planner proposes, TaskEngine.cpp decides |
| Observability | Request logs | SharedMemoryBridge cross-process telemetry |
| Persistence | None | SCXcache DAG survives across calls |

llama.cpp is inference. KHANARY is inference + orchestration + identity + semantic routing + authority separation. The Semantic Engine and Micronauts are the layer llama.cpp explicitly doesn't provide.

---

## Use KXML with any GGUF model (no retraining)

**Do users just paste the KXML template into their GGUF header?** No — a stock GGUF (llama/qwen/phi/gemma/LFM…) doesn't have KHΛNARY's glyph tokens in its vocab, so its header can't render KXML directly. Instead, the **stock-model adapter** (`tools/kxml_stock_adapter.py`) translates a KXML dialogue onto the target model's *own* `chat_template` + tool-call convention — read straight from the GGUF header — so **one KXML front-end drives any GGUF today**:

```python
from tools.kxml_stock_adapter import render_for_gguf
kxml = [
  {"role": "user", "content": "read config.txt and tell me the port"},
  {"role": "assistant", "tool_call": {"name": "Read", "args": "config.txt"}},
  {"role": "tool", "content": "port=8080"},
  {"role": "assistant", "content": "The port is 8080."},
]
prompt = render_for_gguf(kxml, gguf_path="model.gguf",
                         tool_style="openai")   # "inline" for models without native tools
```

The adapter reads the GGUF's `chat_template`/`bos`/`eos` from its header and renders the same KXML dialogue in the model's native format. Verified end-to-end against real models:

- **Phi-3** (no native tool support) → `tool_style="inline"`: tool call/result fold into `<|user|>`/`<|assistant|>` text turns.
- **LFM2.5-1.2B-Instruct** (tool-calling fine-tuned in) → `tool_style="openai"`: emits its native `<|tool_call_start|>[Read(input="config.txt")]<|tool_call_end|>` + `tool` role.

So: a **KHΛNARY-trained** model gets KXML *trained in* (roles/tools are glyph tokens — no header needed); a **stock GGUF** is driven through this adapter. Either way the front-end speaks one KXML dialogue format. See the [KXML section](#kxml--the-trainable-chat-template--tool-call-layer).

## Architecture

```
Python source / weights
        │
        ▼
┌──────────────────┐
│   KUHUL Layer    │  Semantic glyph definitions (tensor, attention, control flow)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  KHΛNARY Layer   │  32-bit KNU encoding (KHΛ-2-DENSE-32 profile)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  KLSL Emit Layer  │  Same IR → three shader targets  │
│  emit_hlsl.py     │  D3D11 cs_5_0  (Windows iGPU)   │
│  emit_wgsl.py     │  WebGPU/WGSL   (browser/cross)  │
│  emit_glsl.py     │  OpenGL 4.3    (universal)       │
└────────┬─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  Backend Runtime                                     │
│  d3d11_infer.dll  → igd10iumd64.dll  (HD 4600 native│
│  gl_infer.dll     → ig75icd64.dll   (OpenGL 4.3)    │
│  ggml-opencl.dll  → IntelOpenCL64   (OpenCL 2.0)    │
│  ggml-cpu         → CPU fallback                     │
└──────────────────────────────────────────────────────┘
```

**KNU word layout** (32 bits):

| Bits | Field | Purpose |
|------|-------|---------|
| 31–28 | `VER` | Version tag |
| 27–20 | `GLYPH_ID` | Semantic operation |
| 19–16 | `ARITY` | Operand count |
| 15–12 | `FLAGS` | Immediate / bin_ref / shape_desc |
| 11–4 | `PAYLOAD` | Immediate data or descriptor |
| 3–1 | `AUTH_CLASS` | Authority level |
| 0 | `PARITY` | Even parity validation |

**Design philosophy**: CPU-first with lightweight iGPU offload. SVG clusters handle the structural graph layer, keeping the runtime lean — no heavyweight GPU toolkit dependencies.

---

## K'UHUL Semantic Runtime — compiler, IR, stdlib, GLSL trainer (2026-08-08)

KHANARY now has a real source-to-execution toolchain around the K'UHUL phase engine,
with one canonical IR across all front ends:

```text
.kuhul / .khl / .kuhules
   ↓ khlc (tools/khlc.py)  /  kuhul-es compile (compiler/src/parser.js)
KAST (protocol kast/1 — nodes carry fold/lane/glyph/opcode/symbol/type/operands)
   ↓ JSON serialization
KSON (.kson — with @driver contract for provider bindings)
   ↓ admission (tools/kson_validate.py — REJECT on invalid, never phase execution)
canonical phase engine (Pop → Wo → Yax → Sek → Ch'en → Xul)
   ↓ provider resolution
sidecar / native impl (glsl_gpu, json_runtime native, kuhul_engine, …)
```

`json_runtime` is the native XJSON object-server/runtime substrate in that final
provider layer. It does not replace K'UHUL phase law; it executes XJSON-declared
lanes, primitive opcodes, sidecar calls, API routes, health checks, and native
glyph/runtime calls. See [`JSON-RUNTIME.md`](JSON-RUNTIME.md) for the
`dist/json-runtime`, `dist/Kuhul-c++`, `programs/`, and `stdlib/` contract map.

**Two source layers:** `.kuhul` semantic modules say *what* capabilities the program
wants; `.khl` drivers say *how* the capability attaches to a provider/sidecar.

**Standard Library (`stdlib/`)** — 18 semantic modules on a one-directional dependency
tree: `core → constants → pi → functions → {geometry, fibonacci} → gravity → glsl/hlsl`
(+ vectors/matrices/tensors/statistics/random/colors/time/audio/image).
[`pi.kuhul`](stdlib/pi.kuhul) is the reference conformance program — one file that
proves the whole stack (source → KAST → KSON → admission → phase engine → provider).

**kuhul-es** (`npm i kuhul-es`) — the ECMAScript front end, published on npm:
`pi`/`tau` bindings, `yield* Pop/Wo/Yax/Sek/Ch'en/Xul(...)` glyph calls, a
hash-chained deterministic runtime, and a **runtime physics engine** (semantic
execution metrics: gravity gate, entropy/attention/pressure, affinity — not a
Newtonian simulation). Its `compile` emits the same canonical KAST/KSON.

**GLSL trainer** (`kuhul-es train`) — the network is built as a semantic skeleton:
EMBED(Pop) → LAYERNORM(Wo) → FFN(Sek) → LM_HEAD(Xul) → LOSS(Ch'en) →
FIELD_OPTIMIZER(Ch'en), each tensor a node with fold/lane/glyph/opcode/gravity,
trained by a physics-driven optimizer (momentum + gravity-class boost + π-nary arc
weights + adaptive clip). GLSL matmul/layernorm/gelu kernels compile through the
live `glsl_gpu` sidecar; the trained skeleton exports as a KAST-like document.

**OpenGL (not OpenCL) is the universal GPU target** — `GL_ARB_compute_shader` + SSBO
runs on every GPU since 2012 via the installed ICD (`ig75icd64.dll` on the HD 4600).
json_runtime admits it through the `glsl_gpu` sidecar (`sco/sidecars/glsl.json`);
`gl_infer_driver.dll` (8 shaders) + `xcfe_gl_ops.dll` (17 kernels on the wgpu_native
GL backend) provide the compute. Verified live: `@fn:dispatch @profile:glsl` →
`compiled:true, icd:ig75icd64.dll`.

Master doc: [`KUHUL.md`](KUHUL.md) · Runtime architecture:
[`docs/KUHUL_RUNTIME.md`](docs/KUHUL_RUNTIME.md)

---

## Geometry backend (D3D11 `cs_5_0` + WGSL)

Geometry is a first-class KHΛNARY opcode: a geometry glyph in the KNU stream drives which kernel the lowering emits. Two additive glyphs — `G_VERTEX_TRANSFORM` (`0x40`) and `G_VERTEX_SKIN` (`0x41`) — lower to **two co-equal backends** from the same stream:

- **D3D11 `cs_5_0`** (`tools/khlnary_dx11.py`) — a byte-addressable vertex transform and weighted-joint skinning (position + normal). **Hardware-verified** on an Intel HD 4600 (feature level 11_1) bit-exact vs CPU (`max abs err 0.00e+00`): transform (256 verts), skinning (128 verts), and the real **30,628-vertex `brain2` birdsong mesh**. Goes directly to `igd10iumd64.dll` — the same D3D11 native driver that games like WoW use on this hardware.
- **WebGPU / WGSL** (`tools/khlnary_webgpu.py`) — structural mirror emitted by the same glyph-driven lowering, for rigs where WebGPU is available. (Not executed on the HD 4600, where WebGPU is blocklisted — the reason the D3D11 backend exists.)
- **OpenGL 4.3 / GLSL** (`tools/emit_glsl.py`, planned) — third emit target from the same KLSL IR. GLSL compute shaders are syntactically near-identical to HLSL (`groupshared`→`shared`, `SV_DispatchThreadID`→`gl_GlobalInvocationID`, `StructuredBuffer`→SSBO). Runs on `ig75icd64.dll` (confirmed present, 11 MB, System32) on this rig — and on every Intel/AMD/NVIDIA GPU since 2012 without a hardware purchase. CUDA requires buying NVIDIA; GLSL requires nothing beyond a standard GPU driver.

The `brain2` → `.stb` bridge (`tools/brain_to_stb.py`) turns a birdsong spectrogram graph into an SVG-Tensor `.stb`, closing the chain **audio → ridges → graph → `.stb` → KNU glyph → GPU geometry**. The capability is packaged as a versioned model under `models/khanary-geometry-v0.3.0/` (manifest, both backends' kernels, KNU streams, and the mesh data), reproducible via `python tools/build_geometry_model.py`. See its `MODEL.json` for honest scope (HLSL hardware-verified vs WGSL structural-parity; diagonal-matrix test coverage).

---

## Compute glyphs — the five GPT-2 forward ops

Beyond geometry, the same glyph-driven lowering emits **compute** kernels. **All five forward ops of a GPT-2 block are now glyphs**, each promoted from the trainer's HLSL and lowered to both `cs_5_0` HLSL and WGSL:

| Glyph | Op | HD 4600 verification (vs NumPy f64) |
|---|---|---|
| `G_MATMUL` (`0x50`) | dense GEMM `A@B` (16×16 tiled) | real `768×2304` weight → `1.01e-06`, **~3.5× faster than naive** |
| `G_ATTENTION` (`0x51`) | causal MHA (softmax + mask) | real QKV, `S=16,E=768,H=12` → `6.39e-08` |
| `G_LAYERNORM` (`0x52`) | LayerNorm (groupshared reduction) | real gpt2 `ln_1` γ/β → `1.27e-07` |
| `G_GELU` (`0x53`) | GELU (tanh approx) | `3.31e-08` |
| `G_EMBED` (`0x54`) | token + positional embedding | `0.00e+00` (exact) |

All dispatched on the iGPU against real GPT-2 tensors from a safetensors checkpoint.

**Inference driver** (`tools/kxml_inference_driver.py`): walks the full-model `.stb` + manifest `forward_graph` and runs the whole model — `embed → [ln, attn, ln, ffn+gelu]×N → ln → lm_head` over all 148 tensors. Its logits **match HuggingFace GPT-2** (`scale-norm ~1e-6`) on the real 124M weights. The driver's op bodies are numpy mirrors of the glyphs. **A GPU version is proven for a full transformer block** (`scratch/block/gpt2_block_run.cpp`): `ln→qkv→attn→proj→+res→ln→fc→gelu→proj→+res` chained on the HD 4600 entirely from the glyph kernels (+ `G_ADD`/`G_ADD_BIAS` glue), matching the CPU driver at scale-norm `3.8e-07`. The **full-model GPU driver is complete** (`scratch/infer/gpt2_infer_run.cpp`): the whole 12-layer gpt2 runs on the HD 4600 with a **KV cache** (prefill fills per-layer K/V; decode uses a flash-style online-softmax attention over the cache, O(t) per token). Its greedy generation matches the CPU driver, and the KV-cache decode logits match a CPU full-recompute at scale-norm `1.3e-06`.

`tools/safetensors_to_stb.py` bridges safetensors weights into KHΛNARY `.stb` tensors (the weight-side sibling of `brain_to_stb.py`), so trained weights flow **safetensors → `.stb` → KNU `G_MATMUL` → GPU GEMM**. This is packaged as the compute model `models/khanary-gpt2-v0.4.0/` — all five forward kernels, the glyph tokenizer, a real weight `.stb`, and the vendored native D3D11 GPT-2 trainer (reference-only). *Honest scope:* correctness is complete end-to-end (CPU driver matches HuggingFace; the full 12-layer model runs on the iGPU with a KV cache). Performance work has begun: `G_MATMUL` is now a **16×16 groupshared tiled GEMM — measured ~3.5× faster than the naive kernel on the HD 4600, correctness preserved** (the full block + KV-cache pipeline still match the CPU driver). What remains is more **performance** (fused kernels, batching) and, for GGUF weights, a dequant→`.stb` step. See its `MODEL.json`.

**Runtime bridges (llama.cpp `ggml`).** KHΛNARY's tiered backends map onto llama.cpp's `ggml`
backend registry — `ggml-cpu` (floor), `ggml-opencl` (reach, runs where WebGPU is blocklisted),
`ggml-webgpu` (browser/WGSL), and `ggml-xcfe` (the KHΛNARY-native slot). These install into
`llama.cpp/ggml/src/`. See [`docs/llama-ggml-bridges.md`](docs/llama-ggml-bridges.md) for how
`ggml` registers a backend and the honest status of `ggml-xcfe` (today a verbatim copy of
`ggml-webgpu`, not yet a real backend).

---

## KXML — the trainable chat-template + tool-call layer

Where llama.cpp uses a prompt-time **chat template** to structure tool calls, KXML structures them as a declarative node stream that lowers to the **glyph tokens the model is trained on**, and whose tool nodes the **runtime dispatches** (the semantic kernel). `models/khanary-kxml-v0.5.0/` registers *all* of it:

- **12 tool calls** (`read_file`, `write_file`, `exec`, `shell`, `tool`, `agent`, `micronaut`, `skill`, `action`, `verb`, `bot`, `http`) emitted as `kuhul.tools.jsonl` — the runtime-loadable registry `kuhul_tool_runtime.h` expects (previously missing).
- **7 compute node ops** (Attention / FFN / LayerNorm / Embed / LmHead / Loss / FieldOptimizer) with the `Pop→Wo→Sek→Chen→Xul` phase machine.
- **Alignment**: each tool → the glyph tokenizer's tool tier (8/12 have a trained token today), each node → a KNU compute glyph (`ATTENTION_NODE→G_ATTENTION`, `FFN_NODE→G_MATMUL`, `LAYERNORM_NODE→G_LAYERNORM`, `EMBED_NODE→G_EMBED`; only the training-only Loss/FieldOptimizer nodes are not glyphs).

This is the alignment point between the compute glyphs, the glyph tokenizer, and the K'UHUL semantic kernel — the tool-augmented runtime chat, trained in rather than templated at inference.

**Chat template** (`kxml_chat_template.{json,jinja}`): where llama.cpp uses a prompt-time chat template, KXML's is **trained in** — `render_tokens(messages)` emits glyph tokens (roles → `I_EXPLAIN`/`I_QUESTION`/`I_ANSWER`, tool calls → `TOOL_CALL·T_xxx·TOOL_RESULT`, turns wrapped `BOS…SEP…EOS`), with a **llama-compatible `.jinja` surface** for interop.

**Full-model `.stb`**: `tools/safetensors_to_model_stb.py` packs a whole checkpoint into one `.stb` (all weights) + a `.stb.json` manifest = config + name→id index + the forward graph over the five glyphs — the same layering as GGUF/safetensors. Verified on gpt2-small (148 tensors, 124M, 497.8 MB); fits the `.stb` u8 tensor-id (larger models need per-layer sharding or a u16 v2).

---

## K'UHUL-3D — the execution contract (AST v3)

Above the KNU/backend layers sits a **frozen, machine-checked semantic contract** (`docs/KUHUL_3D_VNEXT.md`, `docs/kuhul.ast.v3.schema.json`). The rule:

> The **AST preserves** the prompt/context; **K'UHUL traverses** it (phase lanes); **XCFE routes** legal graph moves; **opcodes perform work**; **compute nodes lower** to CPU, llama.cpp, WebGPU/WGSL, or D3D11/HLSL.

It exists so the **llama.cpp fork integrates incrementally** — a backend need only understand this AST + the capability/opcode contract, **not** KXML or SCXQ2 (those lower *into* it later). Three **normative laws** are enforced by `tools/check_kuhul_ast_v3.py`, not by convention:

| Law | Statement | Check |
|---|---|---|
| **P1** | `Phase ∩ Opcode = ∅` — phases schedule, opcodes work | 6 phases disjoint from 30 opcodes |
| **R1** | `Node → PhaseTick → PhaseStep → Node` — a node *is* a tick (thinking at every level) | recursive `$ref` cycle present |
| **G1** | Glyph nativity — the glyph **is** the token (source codepoints == declared Unicode, byte-stable; rendering is a separate projection) | `⟁Sek⟁` → `U+27C1 U+0053 U+0065 U+006B U+27C1` |

Capability is **runtime-resolved** (`{requires, preferred, backend: "resolved_at_runtime"}`) so the *same* AST routes differently per rig — `ASX ≅ XCFE ≅ XJSON ≅ K'UHUL ≅ AST` is **projection equivalence**, not literal equality. `docs/lowering-rules.md` §7 documents the `SVG → KUHUL → KNU → AST v3` lowering row.

**Executed, not aspirational.** `proof/kuhul_matmul_tick_v1` runs a real `G_MATMUL` node's full PhaseTick on the HD 4600 (one shared D3D12 + DirectML device) against the real Qwen weight `transformer.h.0.attn.c_proj.weight [2048,2048]`: `[Pop]` real `MakeResident` of Q4 bytes → per-tensor **Q4→Q8 escalation**; `[Sek]` real DirectML GEMM (`Cq4`, `Cq8`) vs an F16 CPU truth; `[Ch'en]` fidelity gate (`normRMSE` Q4 `0.1100` vs Q8 `0.0115` → admit Q8, escalation 9.6× lower); `[Xul]` real `Evict`. LAW R1 controls **real residency, quantization, dispatch, compute, and eviction** — not just recursive syntax.

---

## Resident LLM inference + dual-quant (Qwen on the HD 4600)

The K'UHUL GPU Resource Contract (KGRC) proof ladder (`docs/GPU_PROOF_LADDER.md`, `models/khanary-gpu-resident-v0.1.0/`) certifies a whole model living as **persistent GPU state** on the Intel HD 4600, each rung one hardware-verified property:

| # | claim | result |
|---|---|---|
| #001 | resident computation | whole gpt2 forward, weights resident → logits scale-norm `1.92e-06` |
| #002 | resident state transition | native DirectML KV decode: growth/preserve/append exact, `8.08e-08` |
| #003 | resident trajectory | 14-tick autoregressive, every tick matches; KV `5.36e-06` |
| #004-A | fixed execution state reusable | binding creations **146 → 12/token**; trajectory == #003 |
| #004-B1 | capacity ≠ extent (+ backend conformance) | extent<capacity → same output; native fixed-capacity MHA **absent** on this DirectML |

The **measured resident ceiling** is `~1.75 GiB` (hard wall at the 2048 MiB DXGI budget; the 2015 driver returns `DEVICE_REMOVED` on overcommit — `proof/gpu_resident_ceiling_v1`). That budget drives a **dual-quant** design (`tools/quantize_safetensors.py`, `docs/QUANT_BUILD.md`, `models/khanary-qwen1_8b-v0.1.0/`): one Qwen-1.8B, two quantizations of the **same tensors** —

| tier | scheme | size | fidelity | role |
|---|---|---|---|---|
| **Q4** | group-64 4-bit | 0.909 GiB | ~10% normRMSE / 20 dB | resident base (long ctx, ~0.9 GiB headroom) |
| **Q8** | per-channel INT8 | 1.712 GiB | ~0.85% / 41 dB (near-lossless) | per-tensor escalation into headroom (validate/deep-think) |

Because both are the same tensors in the same order (tensor-aligned manifests), the runtime keeps **Q4 resident and hot-swaps individual tensors up to Q8** — never both full models (2.6 GiB > the 2.0 GiB wall). The quantizer is deterministic (byte-identical sha256 anywhere → *cloud build == build instructions*) and handles any single-file or sharded HF safetensors (F16/BF16/F32). This is the design the executed MATMUL tick (above) runs on real data.

---

## Birdsong geometry — now a formal grammar

The birdsong pipeline is not just data — it has a **formal grammar** (`docs/birdsong-geometry.ebnf`, `docs/birdsong-brain.schema.json`, `docs/BIRDSONG_GEOMETRY.md`) mapping `audio → spectrogram → ridges → mesh → graph → experts` onto the `Pop→Xul` fold cycle, in three equivalent forms (EBNF = syntax, JSON = dataset, KXML = execution tree). It is **grounded in the real `.stb`** (`tools/check_birdsong.py` confirms the example's totals against `birdsong_mesh.stb`: **30,628 nodes / 91,863 Delaunay edges / 183,726 CSR neighbours**), so bird-song becomes parseable like code and trainable directly on geometric structure instead of text tokens.

---

## Novel innovation — birdsong as *executable geometry*

Today's birdsong AI maps audio to **labels, embeddings, or generated sequences** through learned neural inference:

| System | Role | Method |
|---|---|---|
| **BirdNET** (Cornell) | global species identification | CNN inference, audio → label |
| **TweetyBERT** | self-supervised syntax segmentation | transformer, syllables → latent clusters |
| **FinchGPT** | song-syntax language model | attention over textualized sequences |
| **eBirdNet-Nano** | edge species monitoring | quantized inference on microcontrollers/NPU |

KHΛNARY sits on a **different axis**. It does not classify, embed, or generate — it **compiles the song's spectrogram structure into a deterministic, replayable geometric-compute artifact**. The spectrogram's ridge graph (`brain2`) becomes a Delaunay mesh; the mesh becomes an SVG-Tensor `.stb`; the `.stb` becomes a **KNU glyph stream**; the glyphs drive **real GPU geometry kernels** (vertex transform / skinning). The song is manipulated with GPU *geometry* ops — not convolution or attention.

What is genuinely new here:

- **Deterministic, not inferential.** Same input → bit-identical output (`max abs err 0.00e+00`), fully replayable from the KNU words — the opposite of weight-dependent, stochastic NN inference.
- **Geometry-native representation.** A song is carried as *executable vertex geometry*, so downstream work is transform / skinning / mesh algebra rather than a black-box embedding.
- **Runs where modern GPU stacks don't.** Verified on a frozen **2015 Intel HD 4600 iGPU** (D3D11 feature level 11_1) — no CUDA, no WebGPU (blocklisted on this rig), no heavyweight toolkit. A reach/edge angle that complements the CPU-bound analyzers above.
- **One stream, co-equal backends.** A single KNU glyph stream lowers to both HLSL `cs_5_0` and WGSL.

**Complementary, not competitive.** KHΛNARY makes no species-ID or accuracy claim and is not a benchmark rival to BirdNET et al. — it could *consume* their detections or embeddings. Its contribution is the **deterministic geometric encoding + commodity-iGPU execution substrate** that turns a birdsong graph into hardware-verified, replayable geometry.

---

## Roadmap

### Phase 1 — Specification & Core Encoding ✅

Foundational spec work and the reference encoder/decoder.

- [x] KHΛNARY v0.1 mapping law and `KHΛ-2-DENSE-32` draft layout
- [x] KHΛNARY v0.2 concrete 32-bit profile with tensor/attention glyphs
- [x] KNU encoder/decoder with parity validation
- [x] Lane-bundle format (128-bit, 4 × KNU)
- [x] Python AST → KNU lowering for a compact Python subset

### Phase 1.1 — SVG-Tensor Binary Format ✅

Memory-mapped tensor interchange via `.stb` files.

- [x] STB format v0.1 spec with glyph-to-header wiring
- [x] Deterministic `.stb` writer/reader (Python)
- [x] KHΛNARY payload wiring helpers

### Phase 2 — End-to-End Neural Compute Pipeline ✅

Full compiler toolchain: Python → KHΛNARY KNUs → backend artifacts.

- [x] KUHUL v0.2 glyph catalog (tensor, activation, attention, control flow)
- [x] Formal EBNF grammar for KHΛNARY v0.2
- [x] AST schemas (JSON Schema + Protobuf)
- [x] Backend-lowering rules (CPU / WebGPU contract)
- [x] WebGPU/WGSL skeleton emitter with binding generation
- [x] End-to-end demo pipeline (weights → `.stb` → KNUs → WGSL/JS artifacts)
- [x] Vertical-stack integration tests

### Phase 3 — Validation & Hardening (current)

Strengthen the toolchain for real workloads.

- [ ] Expand test coverage (edge cases, malformed input, round-trip fidelity)
- [ ] CPU native execution backend
- [ ] WebGPU in-browser execution validation (iGPU offload)
- [ ] Parity and authority-class enforcement across all backends
- [ ] Performance profiling of encode/decode and lowering passes

### Phase 3.1 — Geometry ops & native iGPU execution ✅

Real geometry kernels, glyph-driven, executed on hardware.

- [x] Geometry glyphs `G_VERTEX_TRANSFORM` / `G_VERTEX_SKIN` (additive within `KHΛ-2-DENSE-32`)
- [x] Glyph-driven kernel selection in both HLSL and WGSL lowering (co-equal backends)
- [x] D3D11 `cs_5_0` backend: vertex transform + weighted-joint skinning (position + normal)
- [x] Hardware-verified bit-exact vs CPU on Intel HD 4600 (incl. a 30,628-vertex real mesh)
- [x] `brain2` birdsong graph → `.stb` bridge (SVG-Tensor link made real)
- [x] Versioned model folder `models/khanary-geometry-v0.3.0/` with reproducible generator

### Phase 3.2 — Compute glyphs (GEMM) ✅

The first compute op beyond the copy skeleton.

- [x] `G_MATMUL` glyph → dense `cs_5_0` GEMM (HLSL) + WGSL mirror, glyph-driven
- [x] `G_ATTENTION` glyph → causal multi-head attention (softmax + mask), verified on HD 4600
- [x] Verified on Intel HD 4600 with a real GPT-2 weight (scale-normalized err `1.0e-06` vs NumPy)
- [x] `safetensors → .stb` bridge (`tools/safetensors_to_stb.py`)
- [x] Compute model folder `models/khanary-gpt2-v0.4.0/` with glyph tokenizer + vendored trainer
- [x] Layernorm / gelu / embedding compute glyphs (verified on HD 4600) — all 5 fwd ops done
- [x] End-to-end inference driver — walks the .stb+manifest graph; matches HuggingFace GPT2 (~1e-6)
- [x] GPU driver — a full gpt2 block chained on the HD 4600 from the glyph kernels (scale-norm 3.8e-07 vs CPU)
- [x] Full-model GPU inference with KV cache — 12-layer gpt2 on the HD 4600, matches the CPU driver (decode logits 1.3e-06)
- [x] Tiled GEMM — 16×16 groupshared `G_MATMUL`, ~3.5× faster than naive on the HD 4600, correctness preserved
- [ ] GGUF → `.stb` dequant path (safetensors is already dense float32)

### Phase 3.25 — Native D3D11 + GLSL universal backend

The GPU insight: Intel HD 4600 has **three independent GPU paths**, all going to the same execution units through `igc64.dll`. The project was routing through DirectML (D3D12 compat shim, thin 4 MB wrapper) when the native path was always D3D11:

```
igd10iumd64.dll (15.7 MB)  ← D3D11 native — what WoW actually uses
igdumdim64.dll  (38.0 MB)  ← immediate context — the real renderer
igd12umd64.dll  ( 4.1 MB)  ← D3D12 compat shim → internally calls D3D11
ig75icd64.dll   (11.0 MB)  ← OpenGL 4.3 ICD — equally native
```

- [x] `scratch/dml/d3d11_infer_dll.cpp` — native D3D11 cs_5_0 inference DLL: `d3d11_gemm` / `d3d11_embed` / `d3d11_layernorm` / `d3d11_attention` / `d3d11_gelu` / `d3d11_add_bias` / `d3d11_add`. Full GPT-2 forward pass. Goes directly to `igd10iumd64.dll`, no abstraction layer. Built: `d3d11_infer.dll` (162 KB), deployed to `build/bin/Release/`.
- [x] `llama-build.bat` step 2c — builds `d3d11_infer.dll` alongside `dml_gemm.dll` on every build
- [ ] `tools/emit_glsl.py` — GLSL emit target for KLSL (adds to existing HLSL + WGSL targets). Same IR, purely syntactic translation: `StructuredBuffer`→SSBO, `[numthreads]`→`layout(local_size_x=...)`, `SV_DispatchThreadID`→`gl_GlobalInvocationID`, `groupshared`→`shared`, `GroupMemoryBarrierWithGroupSync()`→`barrier()`.
- [ ] `scratch/gl/gl_infer_dll.cpp` — OpenGL 4.3 compute dispatch harness (`gl_infer.dll`). WGL headless context on Windows → `ig75icd64.dll`. Cross-platform via EGL. Same C API as `d3d11_infer.dll`.
- [ ] `scratch/gl/shaders/` — GLSL versions of the 7 inference shaders (direct translation of `scratch/infer/*.hlsl`)

**Why GLSL matters**: GLSL compute shaders run on every GPU since 2012 — Intel, AMD, NVIDIA, mobile — without a hardware purchase. The ML ecosystem ignored them because NVIDIA captured the tooling early with CUDA. The same 7 shaders that power `d3d11_infer.dll` translate line-for-line to GLSL and would run universally. See `GPU.md` for the full comparison.

### Phase 3.3 — KXML tool/op registry (semantic-kernel layer) ✅

The trainable chat-template + tool-call layer, fully enumerated.

- [x] All 12 KXML tool calls registered as `kuhul.tools.jsonl` (runtime-loadable)
- [x] All 7 compute node ops + phase machine (`kxml_nodes.json`)
- [x] Tool→glyph-token and node→KNU-glyph alignment (`kxml_alignment.json`)
- [x] `models/khanary-kxml-v0.5.0/` registry with vendored semantic-kernel source
- [ ] Trained tokens for the 4 unmapped tools (micronaut / action / verb / bot)

### Phase 3.4 — K'UHUL-3D execution contract + resident LLM / dual-quant ✅

The semantic execution layer, machine-enforced, running on real weights.

- [x] K'UHUL-3D AST v3 contract (`docs/kuhul.ast.v3.schema.json`, `docs/kuhul-3d-vnext.ebnf`) + design doc
- [x] Normative laws **P1** (phase∩opcode=∅) / **R1** (recursive tick) / **G1** (glyph nativity), machine-checked by `tools/check_kuhul_ast_v3.py`
- [x] Runtime capability resolution (`{requires, preferred, backend:"resolved_at_runtime"}`)
- [x] KGRC GPU proof ladder #001–#004 + resident ceiling (`~1.75 GiB`) — `models/khanary-gpu-resident-v0.1.0/`, `proof/`
- [x] Dual-quant Qwen-1.8B (Q4 group-64 + Q8 per-channel, tensor-aligned) — `tools/quantize_safetensors.py`, `models/khanary-qwen1_8b-v0.1.0/`, `docs/QUANT_BUILD.md`
- [x] **Executed** MATMUL PhaseTick — real Q4→Q8 residency + DirectML GEMM + fidelity gate (`proof/kuhul_matmul_tick_v1/`)
- [x] Birdsong geometry formal grammar grounded in the real `.stb` (`docs/birdsong-geometry.ebnf`, `tools/check_birdsong.py`)
- [x] `SVG → KUHUL → KNU → AST v3` lowering row (`docs/lowering-rules.md` §7)
- [x] Semantic-runtime proof track (`models/khanary-semantic-runtime-v0.1.0/`)
- [ ] Second opcode (`SOFTMAX`/`ATTENTION`) through the same native PhaseTick path
- [ ] GPU dequant kernel (keep the compact quant bytes resident; dequant on-device)

### Phase 4 — Extended Glyph Support

Broaden the semantic surface beyond basic tensor ops.

- [ ] Conv2D / pooling operation compilation
- [ ] Full attention-mechanism compilation (QKV projection → scaled dot-product)
- [ ] Normalization glyphs (LayerNorm, BatchNorm)
- [ ] Backward-pass / autograd glyph lowering
- [ ] Custom glyph registration API

### Phase 4B — Ecosystem & Enterprise Features (blueprint)

- [x] Package registry architecture blueprint (templates, search, cache/store, publish/install)
- [x] Community glyph marketplace blueprint (verification, reviews, analytics)
- [x] Multi-tenant authority management blueprint (isolation, quotas, billing hooks)
- [x] Audit/compliance blueprint (structured audit, reports, DSAR workflows, dashboard)
- [x] Integration and rollout plan documented for Python-first KHANARY repository

See `docs/phase4-ecosystem-enterprise.md`.

### Phase 5 — Packaging & Ecosystem

Make KHΛNARY usable as a standalone tool.

- [ ] `pyproject.toml` / installable package
- [ ] CLI entry point (`khanary compile`, `khanary run`)
- [ ] CI pipeline (lint, test, coverage)
- [ ] Documentation site or expanded docs
- [ ] Example model zoo (small networks compiled through the full stack)
- [ ] **PRIMEOS desktop app** — two-part architecture, shippable in order (KHANARY plugs in at the
      `ggml` backend layer, not the UI):
  - [x] **1. WebView2 shell that launches llama-server.** `desktop/PRIMEOS/` now **starts
        `llama-server.exe` itself** (bundled `.\llama\` runtime, or the `.ASX.cpp` dev build) on a free
        loopback port and hosts its **built-in web UI** (baked into the binary — no npm/SvelteKit build)
        in a **WebView2** control. Kills the server on close; model configurable. Builds clean on net8
        (`dotnet build -c Release`, 0 warnings). Runs on stock `ggml-cpu`/`ggml-opencl`.
  - [ ] **2. A branded KHANARY build with a real `ggml-xcfe` backend.** The full compilable
        `llama.cpp/ggml` source is present (vendored under `.ASX.cpp/llama-b9968-.../llama.cpp/`), and
        `ggml/src/ggml-xcfe/` already exists — but today it's an **orphan byte-copy of `ggml-webgpu`**
        (its CMake declares the `ggml-webgpu` target and there is no `ggml_add_backend(XCFE)` — see
        `docs/llama-ggml-bridges.md`). Steps:
    - [ ] copy the source out of the read-only vendored tree into a KHANARY build workspace;
    - [x] make `ggml-xcfe` a genuine, **registering, compiling** target that **claims + computes
          MUL_MAT** — `native/ggml-xcfe/` (real reg→device→backend vtables, CPU-delegated buffers;
          `supports_op` claims 2D F32 contiguous `GGML_OP_MUL_MAT`; `graph_compute` = CPU reference GEMM
          baseline). Wired + built via `tools/build_khanary_llama.ps1`. **Verified** (`proof/ggml_xcfe_v1/`):
          `XCFE registered: YES`, XCFE's MUL_MAT matches ggml's CPU MUL_MAT, **and `graph_compute`
          runs it on the HD 4600 via DirectML** (`dml_gemm.dll`, loaded at runtime; CPU fallback) —
          matching CPU to `2.7e-07`. The executed MATMUL tick and the ggml backend are now **joined**;
    - [x] build a custom llama with `-DGGML_XCFE=ON` → **`dist/khanary-server/khanary-server.exe`**
          (full llama-server, XCFE compiled in, bundled with the ggml/llama runtime + `dml_gemm.dll`).
          **Verified**: `khanary-server --list-devices` lists **XCFE** as a live device
          (`proof/khanary_server_v1/`). Built `-DLLAMA_BUILD_UI=OFF` (Web UI + branding = Part 3 reseal).
          *Next:* bundle KHANARY's MCP server(s) alongside.
  - [ ] **3. KHANARY UI pages (incl. model training).** Add KHANARY-native pages on top of the shell —
        a menu app with **train-a-model pages** (dataset → LoRA/train run → export), model management,
        and the KHANARY tooling. Note: custom pages mean a real UI build (fork llama's SvelteKit webui
        + `npm build`, **or** ship a separate KHANARY web app the WebView2 shell also hosts) — the
        "no npm build" property only holds for llama's *stock* baked-in UI.

  Full recipe + the **page → backend** map (each page surfaces a capability that already exists):
  [`docs/KHANARY_BUILD.md`](docs/KHANARY_BUILD.md).
  - [x] **Grammar / developer-sandbox pages** — W3Schools-style static reference (syntax + schema +
        worked examples + "Try it") for K'UHUL-3D, Birdsong, and XCFE, generated from the grammar docs
        by `tools/build_grammar_pages.py` → `sandbox/` (self-contained HTML, **no npm build**, hostable
        in the WebView2 shell). Add a grammar by extending the generator.
  - [x] **Branded UI built + sealed into khanary-server** — `tools/build_khanary_ui.ps1` builds the
        SvelteKit UI with KHANARY branding (name + ASX-teal accent, `--max-old-space-size` fixes the
        vite OOM); `tools/reseal_khanary_server.ps1` embeds it into the binary (priority-1 pre-built
        assets, no npm) and rebuilds. **Verified** (`proof/khanary_ui_v1/`): the branded UI is in
        `llama-server-impl.dll`, and `khanary-server` serves the SvelteKit UI at `/` at runtime.
  - [x] KHANARY **feature routes** inside the SvelteKit app — **Models** (lists `/v1/models` + the
        KHANARY packages) and **Train** (the quantize/train pipeline reference), with sidebar nav items,
        overlaid onto `tools/ui` by `tools/build_khanary_ui.ps1` (mirrors the `mcp-servers` pattern).
        **Verified served** by `khanary-server` at runtime (`proof/khanary_ui_v1/`). *Next:* wire Train
        to a real host training endpoint (browser-driven runs); the grammar sandbox is the static app.

---

## Project Structure

```
KHANARY/
├── KUHUL.md                       Master doc — K'UHUL semantic runtime & language
├── KXC.md                         KXC kernel compiler reference (grammar, SMCA output, registry, classifier)
├── driver.manifest.json           Driver DLL inventory (exports, loaders, zip distribution)
├── khanary-driver-dlls.zip        Driver DLLs (unzip at repo root — GitHub flags raw .dll)
├── stdlib/                        K'UHUL Standard Library — 18 semantic modules (core → pi → gravity → glsl)
│   ├── pi.kuhul                  Reference conformance program (source → KAST → KSON → phase engine)
│   ├── gravity.kuhul / glsl.kuhul / hlsl.kuhul
│   └── *.kson                    Compiled KAST
├── drivers/khl/                   Driver contracts (.khl): opengl, phase, fold, attention.fold, …
├── examples/                      hello.kuhules (KUHUL-ES), hello.kson (compiled KAST), train_sin.json
├── docs/                          Specifications
│   ├── KUHUL_RUNTIME.md          Phase engine as versioned runtime + KHL/KAST/KSON
│   ├── khlnary-v0.1.md           v0.1 foundational mapping law
│   ├── khlnary-v2.md             v0.2 concrete 32-bit profile
│   ├── stb-format.md             SVG-Tensor Binary format spec
│   ├── lowering-rules.md         Backend-lowering contract (CPU / WebGPU)
│   ├── grammar.ebnf              Formal KHΛNARY v0.2 grammar
│   ├── khlnary-ast.schema.json   JSON Schema for AST nodes
│   ├── khlnary-ast.proto         Protobuf AST interchange schema
│   ├── KUHUL_3D_VNEXT.md         K'UHUL-3D execution contract + laws P1/R1/G1
│   ├── kuhul-3d-vnext.ebnf       K'UHUL-3D grammar (bracket surface)
│   ├── kuhul.ast.v3.schema.json  K'UHUL-3D AST v3 (recursive PhaseTick)
│   ├── BIRDSONG_GEOMETRY.md      Birdsong grammar (EBNF + JSON + KXML)
│   ├── birdsong-geometry.ebnf    Birdsong formal grammar
│   ├── birdsong-brain.schema.json  Birdsong dataset schema
│   ├── GPU_PROOF_LADDER.md       KGRC resident proof ladder #001–#004
│   └── QUANT_BUILD.md            Portable dual-quant build recipe
├── tools/                         Reference implementations
│   ├── khlnary_encoder.py        KNU encoder/decoder + Python AST lowering
│   ├── khlnary_compiler.py       Compiler (KUHUL encoding + .stb registration)
│   ├── kuhul_glyphs.py           KUHUL v0.2 glyph catalog
│   ├── stb.py                    .stb writer/reader
│   ├── khlnary_webgpu.py         KHΛNARY → WGSL emitter (+ geometry kernels)
│   ├── khlnary_dx11.py           KHΛNARY → D3D11 cs_5_0 HLSL backend (+ geometry kernels)
│   ├── emit_glsl.py              KHΛNARY → OpenGL 4.3 GLSL compute emitter (universal backend, planned)
│   ├── brain_to_stb.py           brain2 birdsong graph → SVG-Tensor .stb bridge
│   ├── build_geometry_model.py   generator for the geometry model version folder
│   ├── safetensors_to_stb.py     safetensors weights → SVG-Tensor .stb bridge
│   ├── build_gpt2_model.py       generator for the gpt2 compute model version folder
│   ├── huggingface_tensor_trainer.py  chat JSONL -> HF safetensors finetune pipeline
│   ├── finetune_hf_tokenbin.py   token-bin datasets -> HF safetensors finetune pipeline
│   ├── kxml_ops.py               registry of all KXML tool calls + compute node ops
│   ├── build_kxml_registry.py    generator for the KXML tool/op registry folder
│   ├── kxml_chat_template.py     KXML chat template (trained-in tokens + llama .jinja)
│   ├── kxml_stock_adapter.py     run a KXML dialogue on any stock GGUF (reads its chat_template)
│   ├── safetensors_to_model_stb.py  pack a whole checkpoint into one full-model .stb
│   ├── kxml_inference_driver.py  walk the .stb+manifest graph and run the model (matches HF GPT-2)
│   ├── quantize_safetensors.py   deterministic dual-quant (Q4+Q8) of any safetensors model
│   ├── verify_quant.py           quant container + dequant-fidelity verifier
│   ├── check_kuhul_ast_v3.py     K'UHUL-3D self-check (schema + EBNF + laws P1/R1/G1)
│   ├── check_birdsong.py         Birdsong self-check (schema + EBNF + real .stb totals)
│   ├── run_kuhul_matmul_tick.py  run the executed MATMUL PhaseTick proof
│   └── demo_end_to_end.py        Full pipeline demo
├── models/                        Versioned KHΛNARY models
│   ├── khanary-geometry-v0.3.0/  Geometry ops: manifest, HLSL+WGSL kernels, KNU, mesh
│   ├── khanary-gpt2-v0.4.0/      Compute: G_MATMUL + G_ATTENTION, glyph tokenizer, vendored trainer
│   ├── khanary-kxml-v0.5.0/      KXML tool/op registry: kuhul.tools.jsonl, node ops, alignment
│   ├── khanary-gpu-resident-v0.1.0/    KGRC resident proof ladder (hardware-verified capability)
│   ├── khanary-semantic-runtime-v0.1.0/  FieldExecutionEngine proof track (instrumented)
│   └── khanary-qwen1_8b-v0.1.0/  Qwen-1.8B dual-quant (Q4 base + Q8 escalation tier)
├── proof/                         Frozen, SHA256-verified proof artifacts
│   ├── gpu_resident_ceiling_v1/  measured ~1.75 GiB resident ceiling
│   ├── qwen_quant_v1/            dual-quant provenance + fidelity
│   └── kuhul_matmul_tick_v1/     executed LAW R1: MATMUL tick drives real residency + GEMM
├── desktop/                       Desktop UI tier
│   └── PRIMEOS/                   WPF + WebView2 shell (net8) over the llama-server web UI (Part 1 done)
└── tests/                         Test suite
    ├── test_khlnary_encoder.py   KNU codec + parity tests
    ├── test_stb_minimal.py       .stb format tests
    ├── test_lowering_skeletons.py Backend lowering + WGSL geometry-glyph parity tests
    ├── test_dx11_backend.py      D3D11 HLSL backend + geometry-glyph selection tests
    └── test_vertical_stack.py    Full-stack integration tests
```

## Model paths — Phase 4 distillation

### Student: from_zero v0.6

| File | Path |
|------|------|
| v0.6 merged (SLERP α=0.6) | `models/from_zero/from_zero_v0.6_merged.safetensors` |
| v0.6 LoRA adapter (output) | `models/from_zero/from_zero_v0.6_lora.safetensors` |
| v0.6 GGUF (KUHUL vocab) | `models/from_zero/from_zero_v0.6_kuhul.gguf` |

Architecture: 12 layers, n_embd=768, n_head=12, vocab=50270

### Teacher: GPT-OSS 20B

| Format | Path | Size |
|--------|------|------|
| GGUF MXFP4 | `C:\Users\canna\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf` | 11.28 GB |
| HF shards (layer_00–23) | `E:\models\GPT-OSS\hf\` | ~12 GB |
| Config | `E:\models\GPT-OSS\hf\model_config.json` | 24L · hidden=2880 · 32 experts · top_k=8 · vocab=200064 |

Served by `kuhul_engine.exe` on port **17480**. Exceeds HD 4600 VRAM (1792 MB) — runs CPU inference.

### Run distillation (Phase 4)

```powershell
# kuhul_engine.exe already serves at port 17480 — start it first if not running
cd C:\Users\canna\_khanary_inspect
python tools/oss_distillation.py `
  --student  models/from_zero/from_zero_v0.6_merged.safetensors `
  --out      models/from_zero/from_zero_v0.6_lora.safetensors `
  --rank 8 --steps 500 --lr 1e-4 --engine http://127.0.0.1:17480
```

Falls back to self-distillation if engine is unreachable (useful for shape validation).

---

## Quick Checks

```bash
# Compile-check all modules
python -m compileall tools/kuhul_glyphs.py tools/khlnary_compiler.py tools/khlnary_encoder.py tools/stb.py tools/khlnary_webgpu.py tools/khlnary_dx11.py tools/brain_to_stb.py tools/build_geometry_model.py tools/safetensors_to_stb.py tools/build_gpt2_model.py tools/demo_end_to_end.py

# Run test suite
python -m pytest tests/ -q

# K'UHUL semantic runtime — compile + admission (all modules + drivers ADMITTED)
python tools/khlc.py stdlib/
python tools/khlc.py drivers/khl/
python tools/kson_validate.py stdlib/
python tools/kson_validate.py drivers/khl/
python tools/kson_validate.py --tamper drivers/khl/opengl.kson   # must REJECT

# KUHUL-ES front end (npm i kuhul-es) — compile to canonical KAST + GLSL train
node dist/khanary-server/node_modules/kuhul-es/bin/kuhul-es.js compile examples/hello.kuhules
node dist/khanary-server/node_modules/kuhul-es/bin/kuhul-es.js train examples/train_sin.json

# Machine-check the grammar contracts (laws P1/R1/G1 + Birdsong, grounded in the real .stb)
python tools/check_kuhul_ast_v3.py
python tools/check_birdsong.py

# Build the model version folders (kernels emitted from the real lowering)
python tools/build_geometry_model.py
python tools/build_gpt2_model.py

# Reproduce the executed MATMUL PhaseTick (needs the Qwen dual-quant artifacts + DirectML)
python tools/run_kuhul_matmul_tick.py
```

## License

See repository for license details.
<img style="text-align:center;" src="https://github.com/cannaseedus-bot/KHANARY/blob/main/khanary.svg" alt="KHΛNARY">
