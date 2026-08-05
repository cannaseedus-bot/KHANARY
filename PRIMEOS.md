# PRIMEOS.md — Layer 6 Orchestration Shell

> Location: `C:\Users\canna\_khanary_inspect\desktop\PRIMEOS\`
> See also: SEMANTIC_ENGINE.md, GPU.md, SCX.md, DISTILLATION.md

---

## What it is

PRIMEOS is a **native WPF desktop app** (C#, .NET 8, WebView2) that bottles the khanary-llama stack into a self-contained orchestration shell. The key insight: llama-server doesn't need a standalone browser — PRIMEOS launches it internally, wraps its built-in web UI inside a WebView2 canvas, and adds the full K'UHUL execution layer on top (delta commands, registry, MCP routing, phase management, live physics telemetry from FieldExecutionEngine).

```
PRIMEOS.exe  ──  launches  ──  llama-server.exe
     │                              │
     │  WebView2 canvas             │  /v1/chat/completions
     │  (embeds llama web UI)       │  /health  /status
     │                              │
     ├──────────────────────────────┘   port: dynamic (FreePort())
     │
     ├── kuhul_engine  :17474   → inference (RouteToLlamaInference)
     │
     ├── kuhul-server  :8764    → MCP gateway (CallBossAsync → kuhul_task_boss)
     │
     └── Local\KuhulGeometricState  → SHM read @ 500ms (FieldExecutionEngine telemetry)
```

**Self-contained**: llama-server.exe bundles its own web UI (SvelteKit, compiled in). No npm, no external browser. WebView2 (Edge Chromium runtime) is the only UI dependency.

---

## Architecture

### Layout (3-column × 3-row WPF Grid, 1400×900, dark green-on-black)

```
┌──────────────── HEADER: Mode + Status + CONNECT LLAMA ───────────────────────┐
├── LEFT (280px) ──────┬── CENTER (*) ─────────────────┬── RIGHT (350px) ──────┤
│  REGISTRY             │  COMMAND INPUT (TextBox)       │  PHASE STATE          │
│  ▸ AGENTS             │  ─────────────────────────     │  Current Phase        │
│  ▸ SKILLS             │  CANVAS OUTPUT (WebView2)      │  Execution Time       │
│  ▸ PLUGINS            │    ← llama web UI OR           │  Mode                 │
│  ▸ TOOLS              │       HTML/SVG inference output │  LLAMA Status         │
│  ▸ MICRONAUTS         │                                │  GEODESIC STATE       │
│  ▸ OPCODES            │  ─── footer status bar ────    │  (SHM reader)         │
├───────────────────────┤                                ├───────────────────────┤
│  Δ DELTA COMMANDS     │                                │  CHAT / INFERENCE     │
│  [CREATE] New Agent   │                                │  Chat history list    │
│  [UPDATE] Skill Config│                                │  [FAST/AUTO/DEEP]     │
│  [DELETE] Micronaut   │                                │  ChatInput + SEND     │
│  [ROUTE] Expert Query │                                │  MODEL selector       │
│  [VALIDATE] Phase     │                                └───────────────────────┘
│  [EXECUTE] Opcode     │
└───────────────────────┘
```

### Key C# classes / fields

| Field | Value | Notes |
|-------|-------|-------|
| `_llamaServerUri` | dynamic port | resolved at launch via FreePort() |
| `_bossUri` | `http://127.0.0.1:8764` | kuhul-server MCP gateway |
| `KuhulEngineExe` | `...v3.5.0-WebX\build\bin\Release\kuhul_engine.exe` | path to kuhul_engine binary |
| `SHM_NAME` | `Local\KuhulGeometricState` | FieldExecutionEngine shared memory |
| `_shmTimer` | 500ms tick | reads KuhulSharedState struct → updates Geodesic State panel |

---

## llama-server launch flow

```
InitCanvasAsync()
  └── LaunchLlamaServerAsync()
        ├── ResolveLlamaServer()
        │     1. .\llama\llama-server.exe  (bundled next to PRIMEOS.exe)
        │     2. .ASX.cpp dev build fallback
        ├── FreePort()  — bind random free loopback port
        ├── ResolveModel()
        │     1. %USERPROFILE%\.lmstudio\models\gpt2.Q8_0.gguf
        │     2. Qwen-1_8B-Chat-f16
        │     (null → UI starts without inference)
        ├── Process.Start(llama-server.exe --host 127.0.0.1 --port {port} -m {model})
        └── Poll /health (30s, 500ms steps) → then load WebView2 → canvas shows llama UI
```

WebView2 canvas source: `CanvasDisplay.Source = new Uri(_llamaServerUri)` — the llama web UI itself.

When a kuhul response contains HTML/SVG (`<`…`>`), `DisplayCanvasOutput()` overrides the canvas with inline HTML (dark theme, Consolas, green border).

---

## Delta command system

Commands typed in the center CommandInput panel:

| Command | Routes to | Notes |
|---------|-----------|-------|
| `[CREATE] agent MyAgent` | BOSS `:8764` `app.create` / `build.micronaut` | also updates local registry |
| `[UPDATE] skill ...` | local only | config update, no BOSS call |
| `[DELETE] micronaut ...` | local only | removes from in-memory registry |
| `[ROUTE] Expert Query: "..."` | BOSS `:8764` `task.plan` | expert routing via MCP |
| `[VALIDATE] Phase WO` | local only | triggers TransitionPhase() |
| `[EXECUTE] Opcode PUSH_PHASE` | BOSS `:8764` `task.plan` | opcode execution |
| `query: <text>` | kuhul_engine `:17474` | direct inference |

BOSS call format (JSON-RPC):
```json
{"jsonrpc":"2.0","method":"tools/call","id":1,
 "params":{"name":"kuhul_task_boss","arguments":{"verb":"app.create","prompt":"..."}}}
```
Endpoint: `POST http://127.0.0.1:8764/mcp`

---

## Geodesic State panel (live SHM reader)

Reads `Local\KuhulGeometricState` every 500ms via `MemoryMappedFile.OpenExisting()`:

```csharp
[StructLayout(LayoutKind.Sequential, Pack = 4)]
struct KuhulSharedState {
    uint  Version;
    uint  ActiveFold;
    uint  TickCount;
    float Entropy;
    float Attention;
    float Pressure;
    // float reserve[10] — not mapped
}
```

Displays: entropy, attention, pressure, fold/tick counter. Shows `⊘ SHM offline` when FieldExecutionEngine is not running.

This is how PRIMEOS sees live physics state without polling kuhul_engine over HTTP.

---

## Phase system

PRIMEOS uses all 6 K'UHUL phases (unlike the 5-phase executor.h):

```
POP (0) → WO (1) → YAX (2) → SEK (3) → CH'EN (4) → XUL (5) → wrap to POP
```

`[VALIDATE] Phase WO` triggers `TransitionPhase()` → increments to next phase, shown in right panel.

---

## Model selector

Bottom-right panel. Maps alias tags → `atomic.manifest.json` paths relative to repo root:

| Tag | Model | GPU/CPU | VRAM |
|-----|-------|---------|------|
| `qwen-story` | Qwen 2.5 0.5B story | GPU | 644 MB |
| `gemma-1b` | Gemma 3 1B QAT | GPU | 687 MB |
| `lfm2` | LFM2 1.2B (128K ctx) | GPU | 1188 MB |
| `gemma-1b-q8` | Gemma 1B Q8 | GPU | 1020 MB |
| `mgguf-gpt2` | GPT-2 MoE 2-expert | GPU | 1408 MB |
| `gpt2-xl` | GPT-2 XL (MCP baked) | GPU | 1668 MB |
| `from_zero` | from_zero K'UHUL Chat | GPU | — |
| `kxml` | KXML Tool Agent | GPU | — |
| `mgguf-qwen` | Qwen MoE 1-expert | CPU | 1862 MB |
| `dolphin` | Dolphin Phi-2 creative | CPU | 1844 MB |
| `phi3-mini` | Phi-3 Mini 4K | CPU | 2282 MB |
| `gemma-4b` | Gemma 3 4B | CPU | downloading |
| `gemma-4-e2b` | Gemma 4 Vision | CPU | 4.2 GB |
| `qwen-1b8` | Qwen 1.8B multilingual | CPU | convert req'd |
| `gpt-oss` | GPT-OSS 20B distillation teacher | CPU | — |

All GPU models fit within the HD 4600 ≤1792 MB ceiling. Models above 1668 MB → CPU only.

Selecting a model triggers a type-style load sequence in the footer status bar (`LOADING [...] MANIFEST OK → GPU/CPU · READY ▌`).

---

## Registry

In-memory registry (no persistence yet):

```csharp
{ "agents",     ["semantic-router", "expert-classifier", "plan-generator"] }
{ "skills",     ["geometry-validator", "phase-manager", "cache-optimizer"] }
{ "plugins",    ["llama-gguf-bridge", "mcp-server-link", "canvas-sync"] }
{ "tools",      ["execute-query", "measure-latency", "profile-memory"] }
{ "micronauts", ["qwen-2.5-micronaut", "validator-micronaut", "planner-micronaut"] }
{ "opcodes",    ["PUSH_PHASE", "EXECUTE", "VALIDATE", "ROUTE", "CACHE_HIT", "RETURN"] }
```

Shown in the left Registry panel. `[CREATE]` / `[DELETE]` commands mutate the in-memory list + send a BOSS task.

---

## Inference routing

Chat input → `RouteToLlamaInference(query)`:
```
POST http://127.0.0.1:{dynamic_port}/v1/chat/completions
{ model:"kuhul", messages:[{role:"user",content:query}], stream:false }
```

Response containing HTML/SVG → rendered in the canvas. All other responses → appended to the chat panel.

---

## Build

.NET 8 WPF, self-contained `WinExe`. Only external dependency: `Microsoft.Web.WebView2` NuGet package (Edge Chromium runtime must be installed on the machine — ships with Edge on Win10/11).

```
dotnet build PRIMEOS.csproj -c Release
```

Output: `bin\Release\net8.0-windows\PRIMEOS.exe`

**Note**: llama-server.exe's sibling ggml\*.dll files must be in the same directory as llama-server.exe. `WorkingDirectory` for the launched process is set to `Path.GetDirectoryName(exe)` so they load correctly.

---

## Connection to the rest of the stack

```
PRIMEOS (this shell)
  │
  ├── WebView2 canvas    ←── llama-server (bundled, dynamic port)
  │                              └── ggml backend / KHANARY DirectML
  │
  ├── HTTP :17474        ←── kuhul_engine  (main inference)
  │
  ├── HTTP :8764         ←── kuhul-server  (MCP gateway: kuhul_task_boss)
  │
  └── SHM Local\KuhulGeometricState ←── FieldExecutionEngine (ASX Runtime)
```

**The key point**: PRIMEOS is the **native app shell** that wraps everything. The khanary build of llama (in `khanary-llama-build/`) doesn't have to be accessed through a browser — PRIMEOS launches it, embeds its web UI, and adds the full orchestration layer. Any model in the stack can be selected from the model picker and routed through the same inference path.

---

## File locations

| File | Role |
|------|------|
| `desktop/PRIMEOS/PRIMEOS-Shell.xaml` | WPF layout — all panels, styles, model selector |
| `desktop/PRIMEOS/PRIMEOS-Shell.xaml.cs` | C# code-behind — all logic |
| `desktop/PRIMEOS/App.xaml` | WPF application entry point |
| `desktop/PRIMEOS/App.xaml.cs` | Empty application class |
| `desktop/PRIMEOS/PRIMEOS.csproj` | .NET 8 WPF project, WebView2 package ref |
| `desktop/PRIMEOS/README.md` | PRIMEOS readme |
| `desktop/semantic_engine/include/` | FieldExecutionEngine headers (see SEMANTIC_ENGINE.md) |
