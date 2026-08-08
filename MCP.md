# MCP.md — K'UHUL WebX MCP / Service Mesh

> Source: `C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\`
> See also: PRIMEOS.md, SEMANTIC_ENGINE.md, SCXQ2.md, ATOMIC_DOM.md

---

## What it is

The K'UHUL MCP system is the full service mesh for the WebX stack. It is separate from both kuhul_engine (the model server) and PRIMEOS (the desktop shell). The MCP layer handles: task planning, task execution, agent orchestration, code generation, evolution, manifest routing, and model proxying — all behind a REST API with explicit policy admission.

**This was completely undocumented before this file was created.**

---

## Port map (complete)

| Port | Service | Role |
|------|---------|------|
| 7430 | KUHUL WebX-3D server | Static file server (COOP/COEP for SharedArrayBuffer) |
| 7431 | Trainer SSE bridge | `native/trainer-server.cjs` — iGPU/CPU trainer stream |
| 8090 | Agent orchestration | JSON-RPC agent app |
| 8088 | gRPC compat bridge | Replaces any old 8080-style contracts |
| 8787 | json_runtime | ASX JSON/XCFE object server |
| 9080 | Model server | llama.cpp or LM Studio compatible local model |
| 1234 | LM Studio | OpenAI-compatible headless LLM |
| 6002 | SCO runtime | SCO alias registry server |
| 17480 | kuhul_engine | Main inference (OpenAI-compatible `/v1/chat/completions`) |
| 8764 | kuhul-server | MCP gateway — `kuhul_task_boss` JSON-RPC tool |

**Port 8080 is NOT a valid port** — all 8080-style contracts should reference port 8088 (gRPC compat). See `grpc_compat` in server.manifest.json.

---

## Planner / TaskEngine separation

This is the critical architectural point:

```
User or agent prompt
        │
        ▼
  MicrosoftSDK.ps1  (semantic-kernel bridge)
  [SK-Agent / PM-1]  ← PLANNER — model-non-authoritative
        │
        │  produces
        ▼
  TaskList JSON
  { verb, target_kind, tasks[{id, action, description, provider, depends_on}] }
        │
        │  load() → validate() → plan() → run()
        ▼
  TaskEngine.cpp     ← EXECUTOR — C++ authority
  [Kuhul::Runtime::TaskEngine]
        │
        └── WebX::ProviderManager (opencl, opencl_cpu_executor, etc.)
              └── task_executor_abi.cpp — Intel OpenCL executor ABI
```

The model (SK-Agent, PM-1) is a **planner only** — it cannot execute, invoke tools, or bypass TaskEngine admission. It produces a declarative TaskList and hands off. `TaskEngine.cpp` validates required fields, resolves providers, checks dependencies (DAG), and executes.

Allowed planner verbs: `task.plan`, `app.create`, `app.inspect`, `build.game`, `build.website`, `build.program`, `build.micronaut`.

---

## MicrosoftSDK.ps1 — Semantic Kernel bridge

Location: `native/semantic-kernel/MicrosoftSDK.ps1`

This script bridges Microsoft Semantic Kernel's .NET SDK to the kuhul stack. It is part of the MCP system, not the model server.

```
MicrosoftSDK.ps1 commands:
  discover       List SK .csproj files
  manifest       Full SK capability JSON (capabilities, persona, agent, supernaut)
  persona        SK planner persona definition
  stack-manifest Read all stack manifests (manifest.json, server.manifest.json, etc.)
  actions        MicronautActions.json
  runtime        Launch json_runtime.exe with args
  toolchain      Dotnet toolchain probe (Roslyn csc/vbc, formatter)
  build          dotnet build SK-dotnet.slnx
  test           dotnet test
  format         dotnet format
  invoke         POST to kuhul_engine:17480 with SK persona system prompt
  tasklist       Full planner flow → TaskList JSON → optional BOSS dispatch
```

SK persona (`microsoft-sdk-planner`):
- `authority: user` — model output is never authoritative
- `executionOwner: bridge-host-or-taskengine` — only TaskEngine.cpp or the bridge host executes
- System prompt: *"You are a planner and bridge, not an executor or inference authority. Do not invent tool results or capabilities. Keep model output declarative; TaskEngine.cpp validates, admits, schedules, and executes any resulting TaskList."*
- Capabilities: `semantic-kernel-discovery`, `intent-to-tasklist`, `dependency-declaration`, `provider-recommendation`, `plugin-function-routing`, `structured-output-contracts`
- Cannot: execute shell commands, invent tool results, bypass TaskEngine admission, select unavailable providers

SK reads from `registry/sdk-system-micronauts.registry.json` for system-level behavior profiles — these are injected as "System reminders" in the `invoke` and `tasklist` commands, never as model-facing context terms.

---

## TaskEngine.cpp — C++ task executor

Location: `native/runtime/TaskEngine.cpp` — `namespace Kuhul::Runtime`

```cpp
struct TaskSpec { id, action, description, provider, dependsOn };
struct TaskResult { id, provider, status, detail };
class TaskEngine {
    load(path)         // reads TaskList.json or TaskList.kuhul
    validate()         // checks required fields, no duplicate IDs
    plan()             // returns TaskResult[] with provider resolution
    run(executor)      // executes via TaskExecutor callback
};
```

Supported formats:
- `.json` — standard TaskList JSON (`{tasks:[{id,action,description,provider,depends_on}]}`)
- `.kuhul` — K'UHUL DSL task format (parsed line by line)

Provider resolution: `WebX::ProviderManager` — maps provider IDs to executors. Intel OpenCL executor probed via `?GetTaskExecutor@TaskExecutor@OpenCL@Intel@@...` ABI export. Falls back to `provider_available` status without execution.

---

## Micronaut brain IDs (PM-1 and siblings)

The `agents.manifest.json` defines micronaut brain IDs. These are the "TASK planners":

```
CM-1   Compute Model brain
PM-1   Planning Model brain  ← this is what MicrosoftSDK.ps1 and hive.ps1 route to
TM-1   Task Model brain
HM-1   Hive Model brain
SM-1   Semantic Model brain
MM-1   Memory Model brain
XM-1   eXecution Model brain
VM-1   Validation Model brain 1
VM-2   Validation Model brain 2
```

`hive.ps1` defaults to `PM-1` — it calls PM-1's endpoint for planning requests. PM-1 is a behavior profile (sampling config + endpoint), not the TaskEngine. The separation holds: PM-1 proposes, TaskEngine executes.

---

## Micronaut registry + selection (kuhul-server)

`kuhul-server` (port 8764 or an OS-assigned port written to `dist/khanary-server/.kuhul-server.port`) loads `micronauts/registry.json` from the project root and exposes:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/micronauts` | List all micronaut behavior profiles |
| GET | `/micronauts/<name>` | Get one micronaut profile |
| POST | `/micronauts/factory` | Auto-create a micronaut from personality parameters |
| GET/POST | `/micronauts/select?prompt=...` | Pick the best micronaut for a prompt + return Atomic DOM block mapping |

`/micronauts/select` returns:
```json
{
  "prompt": "how does kuhul route tasks",
  "selected": { "name": "stack_doc", "confidence": 0.9, "reason": "keyword match" },
  "atomic_blocks": ["MENU", "BODY"],
  "available": [ ... ]
}
```

Keyword routing covers: `coder`, `memory`, `ui`, `stack_doc`, `primeos_guide`, `scx_guide`, `asx_guide`, `distillation_guide`, `tool_call`, `chat`, and explicit fold names (`pop`, `wo`, `yax`, `sek`, `chen`, `xul`). The selected micronaut only changes sampling parameters; the Atomic DOM block list tells PRIMEOS which UI regions to render. Test with `node tests/micronaut_select_test.js`.

---

## JSON-runtime route surface (abridged)

The `bin/json-runtime/server.manifest.json` defines 180+ routes. Key route groups:

| Group | Routes | Status |
|-------|--------|--------|
| **Core** | `/api/health`, `/api/run`, `/api/file`, `/api/graph`, `/api/phases` | implemented |
| **File manager** | `/api/file-manager/*` (init/list/read/write/patch/stat/search) | implemented |
| **Sidecar** | `/api/sidecars/<name>/call/<op>` | implemented |
| **SCO** | `/api/sco/<alias>`, `/.well-known/sco` | implemented |
| **Discovery** | `/api/discovery` | implemented |
| **Model proxy** | `/v1/chat/completions`, `/api/llama/*`, `/api/lmstudio/*`, `/api/sdk/*`, `/api/cloud/*` | proxy_contract |
| **Code** | `/api/code-generation`, `/api/code/*` (review/diff/refactor/optimize/test/patch) | proxy_contract |
| **Actions** | `/api/actions/<class>/<method>` | requires_policy |
| **Experts** | `/api/experts/*` | sidecar_contract |
| **Evolution** | `/api/evolution/*` (mutate/reward/select) | requires_policy |
| **Micronaut** | `/api/micronaut/factory`, `/api/micronaut/worker/*` | proxy_contract |
| **MX2LM** | `/api/mx2lm/*` (parse/emit/run/ngram/auto) | proxy_contract |
| **Supernaut** | `/api/supernaut/*` | proxy_contract |
| **Chat Hive** | `/api/chat-hive/*` | proxy_contract |
| **PrimeOS Agents** | `/api/agents/*` (spawn/control/list/train) | requires_policy |
| **NNC** | `/api/nnc/*` (compile/execute/examples) | requires_policy |
| **Gravity Wells** | `/api/gravity-wells`, `/api/gravity-wells/query` | sidecar_contract |
| **Crown/Actor/Personality** | `/api/crown-matrix/*`, `/api/actor-matrix/*`, `/api/personality-matrix/*` | requires_policy |
| **BSON** | `/api/bson-micronaut/*` | requires_policy |
| **Bot research** | `/api/bots/*` | requires_policy |
| **Aliases** | `/api/read`, `/api/write`, `/api/edit`, `/api/patch`, `/api/get` | alias_contract |
| **RPC** | `/api/rpc` | methods: sidecar.describe, sidecar.call, program.run |
| **Angles/Lanes** | `/api/angles/*`, `/api/lanes/*` | sidecar_contract |
| **Para-graph** | `/api/para-graph/*` | sidecar_contract + requires_policy |

Route status levels:
- `implemented` — live in server.cpp
- `sidecar_contract` — sidecar must be mounted first
- `proxy_contract` — proxied to sub-service
- `requires_policy` — needs admission policy object in body
- `alias_contract` — shorthand for another route

---

## SCXcache — symbolic cache execution

`native/runtime/SCXcache.manifest.json` — schema: `scx-cache-manifest-v1`

The SCXcache is a persistent DAG of K'UHUL execution state:
- **Working set selection**: by pressure (highest pressure nodes execute next)
- **Ordering**: topological (respects DAG dependencies)
- **Repair**: max 3 attempts per failed node
- **Node state** required fields: inputs, outputs, pressure, confidence, residency, provider, history, completion
- **Shader cache**: `D3DSCache.dll` — keyed by node_id + provider + shader_hash + compiler_version + resource_contract + device_signature. Eviction: recompile or switch provider.
- **Persistence**: written at Xul phase, resumes from last committed graph state, preserves provenance

---

## Atomic DOM — system-level manifests

The `native/runtime/` directory and `AtomicDOM/` both contain system-level Atomic DOM manifests distinct from per-model `models/*/atomic.manifest.json`:

```
AtomicDOM/
  frame.manifest.json    — FRAME block: [header, menu, body, grid, feed, footer], vertical, HOT residency
  header.manifest.json   — HEADER block
  menu.manifest.json     — MENU block
  body.manifest.json     — BODY block
  grid.manifest.json     — GRID block
  feed.manifest.json     — FEED block
  footer.manifest.json   — FOOTER block

native/runtime/
  atomic.manifest.json           — base Atomic DOM entry
  atomic.frame.manifest.json     — FRAME with GRID block added
  atomic.chat.manifest.json      — chat app type
  atomic.game.manifest.json      — game app type
  atomic.page.manifest.json      — page/web type
  atomic.scene.manifest.json     — 3D scene type
  atomic.grid.manifest.json      — grid layout
  atomic.strategy.manifest.json  — strategy game
  atomic.widgets.manifest.json   — widget system
  atomic.opengl.asset.manifest.json
  atomic.opengl.frame.manifest.json
```

These are the **type definitions** for the Atomic DOM. Per-model manifests reference these types via their `block: "FRAME"` field.

---

## native/runtime/ — full inventory

The native runtime is the most complete C++ subsystem in the stack:

```
Core runtime:
  runtime.cpp / runtime.kuhul         main runtime
  domain_runtime.cpp / .h             domain runtime layer
  phase_runtime.h                     phase state machine
  Pop/Wo/Yax/Sek/Chen/Xul.cpp/.kuhul  per-phase execution
  manifest.kuhul / Personality.kuhul  runtime persona/manifest

Task engine:
  TaskEngine.cpp / task_engine.h      planner→executor gap
  task_executor_abi.cpp / .h          Intel OpenCL executor ABI
  task_helper.cpp                     task utilities
  TaskList.json / TaskList.kuhul      task schema files
  DAG.cpp / dag.h                     dependency graph

AI/inference:
  Inference.cpp                       inference layer
  CodeGenGraph.cpp / codegen_graph.h  code generation DAG
  GrammarGraph.cpp / grammar_graph.h  grammar AST
  TokenRAG.cpp / token_rag.h          token-level RAG
  RAG.cpp / rag.h                     retrieval-augmented generation
  ChatHistory.cpp / chat_history.h    conversation context
  Learning.cpp                        online learning
  Confidence.cpp / confidence.h       confidence scoring

Registry / fold:
  fold_registry.cpp / .h              FoldRegistry (same as .NNC-K kxml includes)
  node_registry.cpp / .h              node type registry
  nodes/                              node type implementations
  folds/                              fold implementation files
  SCX.cpp                             SCX integration point
  SCXcache.manifest.json              persistent DAG cache spec

App creation:
  KuhulAppCreator.cpp / .h            creates apps/games/websites from TaskList
  atomic_shell_manifest.cpp / .h      Atomic DOM shell manifest
  APIWriter.cpp / api_writer.h        API surface writer
  CodeGenGraph.cpp                    code generation planning

DirectX / OpenCL:
  DirectML.cpp / directml.h           DirectML integration
  OpenCLTaskAdapter.cpp / .h          OpenCL task dispatch
  opencl_helper.cpp                   OpenCL probing
  Shader.cpp                          shader compilation

Rendering:
  opengl_frame_adapter.cpp / .h       OpenGL frame → Atomic DOM
  particle_effect.cpp / .h            particle system
  world_tile.cpp / .h                 world tile system

Agents:
  instant_agent.cpp / .h              spawn instant agents
  Provider.cpp                        provider resolution
  Forge.cpp                           Forge creation

Misc:
  users.manifest.json                 user session manifest
  adam12.schema.json                  ADAM-12 schema
  Compile.cpp                         compilation utilities
  Chen.cpp                            Chen (verify) phase impl
```

---

## MCP agent registry (abridged)

From `agents.manifest.json` — agents are organized by subsystem:

| Group | Agent IDs |
|-------|-----------|
| DotNet/MCP | DOTNET-CLI, MSBUILD-GRAPH, ROSLYN-CS, ROSLYN-VB, MCP-SDK-SAMPLES, MCP-STDIO-SERVER, MCP-HTTP-SERVER, MCP-CLIENT, MCP-AUTH, MCP-TASKS, DOTNET-MCP-MICRONAUT |
| K'UHUL | REASON-mu, CREATE-mu, DOTNET-CLI, KXML-REPLAY-BUILD |
| MX2LM | MX2LM-NGRAM-RUNTIME, MX2LM-AUTO-ADMIT, ASX-BRAINJS-POLICY |
| Micronaut brains | CM-1, PM-1, TM-1, HM-1, SM-1, MM-1, XM-1, VM-1, VM-2 |
| Chat Hive | ASX-PRIME-CHAT-HIVE, CHAT-HIVE-MODEL-MANAGER, CHAT-HIVE-RLHF-PLANNER |
| PrimeOS | PRIMEOS-AGENT-SPAWNER, PRIMEOS-STATIC-TUNNEL, PRIMEOS-TRAINING-BOT |
| Crown/Actor/Personality | CROWN-MATRIX, ACTOR-MATRIX, PERSONALITY-MATRIX, SHEOGORATH-ACTOR |
| AI Specialists | FRONTEND-AI-SPECIALIST, BACKEND-AI-SPECIALIST, DESIGN-AI-SPECIALIST |

---

## WebX static server

`server.manifest.json` (root level, port 7430):
- Static file server with COOP/COEP headers (required for SharedArrayBuffer)
- Routes: `/`, `/splash`, `/app`, `/docs/*`, `/examples/*`, `/shaders/*`, `/src/*`, `/kuhul/*`, `/registry/*`
- API: `/api/manifest` → server.manifest.json, `/api/cache` → cache.manifest.json
- Services: trainer at port 7431 (`native/trainer-server.cjs`) — auto_start: false
- Env: `KUHUL_PORT=7430`, `KUHUL_MODELS=E:/models/GPT2`, `KUHUL_TRAINER_EXE=native/bin/gpt2_trainer.exe`

---

## File locations

```
.NNC-K/bin/v3.5.0-WebX/
  server.manifest.json                   — WebX root server (port 7430)
  AtomicDOM/                             — Atomic DOM block type definitions
  native/runtime/                        — C++ runtime (TaskEngine, phases, fold registry, etc.)
  native/semantic-kernel/MicrosoftSDK.ps1  — SK planner bridge
  native/semantic-kernel/dotnet/         — SK .NET source (SK-dotnet.slnx)
  bin/json-runtime/server.manifest.json  — 180+ route surface (ports 8787, 8090, etc.)
  bin/json-runtime/agents.manifest.json  — agent registry
  bin/json-runtime/rpc.manifest.json     — RPC contracts
  bin/json-runtime/actions.manifest.json — action manifests
  bin/json-runtime/folds.manifest.json   — fold registry
  registry/sdk-system-micronauts.registry.json  — SK system behavior profiles
  registry/micronauts.registry.json      — main behavior profile registry
  registry/agents-net.registry.json      — .NET agent registry
```
