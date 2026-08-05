# ATOMIC_DOM.md — Atomic Document Object Model

> Location: `models/*/atomic.manifest.json`
> See also: PRIMEOS.md, SCXQ2.md, SCX.md, GPU.md

---

## What it is

The **Atomic DOM** is the document layer that mediates between a model type and the runtime. Each model in the fleet has exactly one `atomic.manifest.json`. This document defines everything specific to that model: its weights, chat format, persona, sampling, tool access, and behavior profile selection.

**Raw JSON files are never sent to models.** The Atomic DOM is what the runtime reads. It constructs the actual model input from the chat template + `npc.system_prompt` + conversation history. Behavior profiles (`micronauts/*.json`) are referenced as sampling overlays — they adjust temperature and penalty, never add text to model context.

```
atomic.manifest.json
      │
      ├── chat_template  ─────────────────────────────────┐
      │    (jinja / chatml / kxml/v1)                     │
      │                                                    ▼
      ├── npc.system_prompt ──────────────────── what the model reads
      │
      ├── app.micronauts ─── {context → profile path}
      │    "default":   "micronauts/khanary.json"  ──────  sampling parameters only
      │    "tool_call": "micronauts/tool_call.json"        (never injected into context)
      │    "memory":    "micronauts/memory.json"
      │
      └── provider.endpoint ─────────────────────────────  where it's sent
```

---

## Schema

```json
{
  "$schema": "atomics://local.dns.route",
  "id":      "unique-model-id",
  "block":   "FRAME",
  "version": "1.0.0",
  "backend": "kuhul_engine | mgguf_runtime",
  "execution_gated": true,
  "feed_parser": "xcfe://tree-sitter-wasm",
  "blocks": ["HEADER", "MENU", "BODY", "FEED", "FOOTER"]
}
```

| Field | Values | Meaning |
|-------|--------|---------|
| `$schema` | `atomics://local.dns.route` | Atomic DOM schema URI |
| `block` | `FRAME` | Top-level document block type |
| `blocks` | HEADER/MENU/BODY/FEED/FOOTER | PRIMEOS layout blocks for rendering |
| `backend` | `kuhul_engine` / `mgguf_runtime` | Which runner handles inference |
| `execution_gated` | `true` / `false` | `true` = K'UHUL phase gating; `false` = direct passthrough (story/creative modes) |
| `feed_parser` | `xcfe://tree-sitter-wasm` | Output stream is parsed by XCFE/tree-sitter for structured extraction |

---

## `display`

Optional section for PRIMEOS model picker:

```json
"display": {
  "alias": "Qwen Story",
  "description": "Qwen 2.5 0.5B — story/creative generation only (tiny system prompt required)",
  "group": "Creative"
}
```

---

## `model`

Model identity and weight file locations:

```json
"model": {
  "id":         "from_zero_v0.6",
  "arch":       "gpt2",
  "n_layer":    12,
  "n_embd":     768,
  "n_head":     12,
  "vocab_size": 50270,
  "params":     "1.5B",
  "quant":      "Q8_0",
  "size_mb":    1668,
  "gpu_fit":    true,
  "gguf":       "models/from_zero/from_zero_v0.1.f32.gguf",
  "safetensors":"models/from_zero/from_zero_v0.6_merged.safetensors",
  "lora":       "models/from_zero/from_zero_v0.6_lora.safetensors",
  "constraint": "STORY MODE ONLY. ...",
  "note":       "644 MB — very comfortable on HD 4600.",
  "tool_registry": "models/.../kuhul.tools.jsonl",
  "node_ops":      "models/.../kxml_nodes.json"
}
```

`constraint` is a machine-readable restriction. The runtime enforces it — e.g. Qwen story mode blocks tool calls, sets max system prompt to ~10 words.

---

## `runtime`

Only present for non-standard runtimes (e.g. mgguf_runtime):

```json
"runtime": {
  "exe":           "path/moe_gguf_runtime.exe",
  "streaming_exe": "path/moe_gguf_streaming_test.exe",
  "test_exe":      "path/moe_gguf_test.exe",
  "benchmark_exe": "path/moe_gguf_benchmark.exe",
  "xjson_exe":     "path/micronaut_xjson.exe",
  "launch_args":   ["--model", "{mgguf}", "--chat"]
}
```

---

## `chat_template`

The template controls how conversation turns are formatted before reaching the model.

### KXML format (`kxml/v1`)

Used by GPT-2 based models trained with KXML glyph tokens. The glyph tokens are **trained-in vocabulary entries**, not strings:

```json
"chat_template": {
  "format": "kxml/v1",
  "trained_in": true,
  "jinja": "models/khanary-kxml-v0.5.0/kxml_chat_template.jinja",
  "roles": {
    "system":    "I_EXPLAIN",
    "user":      "I_QUESTION",
    "assistant": "I_ANSWER",
    "tool":      "TOOL_RESULT"
  },
  "specials": {
    "bos": "BOS",
    "eos": "EOS",
    "turn_sep": "SEP",
    "pad": "PAD"
  },
  "tool_call": {
    "open":       "TOOL_CALL",
    "name_token": "T_<NAME>",
    "close":      "TOOL_RESULT"
  },
  "reasoning": {
    "open":  "THINK_START",
    "close": "THINK_END"
  },
  "generation_prompt": "I_ANSWER"
}
```

Jinja template (the actual rendered form):
```
{{ '<BOS>' }}
{% for m in messages %}
  role token (I_EXPLAIN / I_QUESTION / I_ANSWER / TOOL_RESULT)
  {% if tool_call %}<TOOL_CALL><T_ToolName>args<TOOL_RESULT>{% else %}content{% endif %}
  {{ '<SEP>' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<I_ANSWER>' }}{% endif %}
```

### ChatML format (`chatml`)

Used by commercial/fine-tuned models (Qwen, Gemma, Phi, etc.):

```json
"chat_template": {
  "format": "chatml",
  "roles": {
    "system":    "<|im_start|>system",
    "user":      "<|im_start|>user",
    "assistant": "<|im_start|>assistant"
  },
  "specials": {
    "bos":      "<|im_start|>",
    "eos":      "<|im_end|>",
    "turn_end": "<|im_end|>",
    "turn_sep": "<|im_end|>\n"
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "stock_adapter": "tools/kxml_stock_adapter.py",
  "stock_adapter_style": "inline"
}
```

`stock_adapter` is a Python shim that converts kuhul conversation state to chatml format inline.

---

## `sampling`

Default sampling parameters for this model type. Overridden per-context by the selected behavior profile:

```json
"sampling": {
  "temperature": 0.65,
  "repeat_penalty": 1.2,
  "repeat_last_n": 64,
  "top_p": 0.95,
  "top_k": 50,
  "stop": ["EOS", "SEP"],
  "note": "Small repeat_penalty (1.02) is intentional — higher values degrade story coherence."
}
```

---

## `app` — full application definition

### `kind`
- `"chat"` — interactive conversation
- `"tool_call"` — structured tool invocation only

### `input` / `io_plane`
- `input: "terminal"` — user types in the chat panel
- `input: "api"` — programmatic input only
- `io_plane: "terminal"` — output is plain text
- `io_plane: "json"` — output is structured JSON

### `npc` — the persona

This is what shapes model behavior. The `system_prompt` here is the ONLY text that gets injected as a system message:

```json
"npc": {
  "id":    "kuhul-guide",
  "name":  "KUHUL",
  "role":  "K'UHUL semantic execution assistant",
  "tone":  "precise, minimal",
  "system_prompt": "You are KUHUL...",
  "rules": [
    "Reason in Pop→Wo→Sek→Chen→Xul phases.",
    "Use <THINK>…</THINK> blocks for non-trivial steps."
  ],
  "context": {
    "source":           "state://chat.history",
    "max_turns":        12,
    "include_manifest": false
  }
}
```

**Critical**: keep `system_prompt` short. Small models (0.5B–1B) confuse badly with long system prompts. The Qwen story model uses `"You write stories."` — one sentence. Use `include_manifest: false` for small models (prevents manifest content leaking into context).

### `conversation` — turn format contract

```json
"conversation": {
  "kind": "kxml | tool_call",
  "required": true,
  "reply_format": {
    "turn":    "integer",
    "role":    "user|assistant|tool|system",
    "content": "string",
    "route":   "string",
    "status":  "admitted|deferred|rejected"
  }
}
```

### `provider` — inference endpoint

```json
"provider": {
  "kind":             "kuhul_engine",
  "endpoint":         "http://127.0.0.1:17474/v1/chat/completions",
  "model":            "from_zero_v0.6",
  "gpu_layers":       999,
  "request_route":    "task://chat.submit",
  "response_route":   "state://chat.reply_log"
}
```

| route | meaning |
|-------|---------|
| `task://chat.submit` | general chat routing |
| `task://tool.call` | structured tool call routing |
| `state://chat.reply_log` | response stored to chat log |
| `state://tool.result` | response stored as tool result |

### `micronauts` — behavior profile selection

```json
"micronauts": {
  "default":   "micronauts/khanary.json",
  "tool_call": "micronauts/tool_call.json",
  "memory":    "micronauts/memory.json",
  "coder":     "micronauts/coder.json",
  "fallback":  "micronauts/khanary.json"
}
```

Keys are context labels. The runtime selects the profile based on what kind of turn is happening. The profile file contains only sampling parameters — it is never injected into model context. This is the complete list of what a profile contains:

```json
{ "name": "tool_call", "sampling": { "repeat_penalty": 1.0, "temperature": 0.1, "repeat_last_n": 64, "stop": ["</tool_call>"] } }
```

---

## KXML node ops (`kxml_nodes.json`)

Defines the compute graph nodes for KXML-based models. Each node has a phase assignment and gravity type:

| Node | Phase | Gravity | Glyph |
|------|-------|---------|-------|
| EMBED_NODE | Pop | Embed | G_EMBED |
| LAYERNORM_NODE | Wo | Heavy | G_LAYERNORM |
| ATTENTION_NODE | Sek | Normal | G_ATTENTION |
| FFN_NODE | Sek | Normal | G_MATMUL |
| LM_HEAD_NODE | Xul | Heavy | G_MATMUL |
| LOSS_NODE | Ch'en | Heavy | — |
| FIELD_OPTIMIZER_NODE | Ch'en | Normal | — |

`FIELD_OPTIMIZER_NODE` uses K'UHUL physics fields: `ATTRACTION_WELL`, `SCROLL_INERTIA`, `WIND_FIELD`, `NAVIGATION_FORCE`.

Gravity types: Float (lightest) → Embed → Normal → Heavy (most anchored)

---

## Tool registry (`kuhul.tools.jsonl`)

Each line is a tool definition. Only tools with a `glyph_token` are visible to the model at inference time. Tools with `"glyph_token": null` are internal routing primitives:

| Tool | Glyph | Effect | Sandbox |
|------|-------|--------|---------|
| read_file | T_READ | io | restricted |
| write_file | T_WRITE | io | restricted |
| exec | T_BASH | process | process_isolation |
| shell | T_BASH | shell | process_isolation |
| tool (MCP) | T_MCP_CALL | tool | api_gateway |
| agent | T_AGENT_SPAWN | agent | isolated |
| skill | T_SKILL | skill | isolated |
| http | T_WEB_FETCH | network | network_sandbox |
| micronaut | null | capsule | thread_pool |
| action | null | action | isolated |
| verb | null | verb | readonly |
| bot | null | bot | isolated |

Tool dispatch format in model output: `TOOL_CALL T_<ToolName> <args> TOOL_RESULT`

---

## Fleet manifest

All current models with their key constraints:

| ID | Arch | Format | Backend | GPU | MB | Gated |
|----|------|--------|---------|-----|-----|-------|
| `qwen25-0_5b-story` | qwen2 | GGUF Q8 | kuhul_engine | ✓ | 644 | ✗ (story only) |
| `gemma-3-1b` | gemma | GGUF QAT | kuhul_engine | ✓ | 687 | ✓ |
| `gemma-3-1b-q8` | gemma | GGUF Q8 | kuhul_engine | ✓ | 1020 | ✓ |
| `lfm2-1b` | lfm2 | GGUF | kuhul_engine | ✓ | 1188 | ✓ |
| `mgguf-gpt2-2expert` | gpt2_moe | mgguf | mgguf_runtime | ✓ | 1408 | ✓ |
| `gpt2-xl` | gpt2 | GGUF Q8 | kuhul_engine | ✓ | 1668 | ✓ |
| `from_zero` | gpt2 | GGUF + ST | kuhul_engine | ✓ | — | ✓ |
| `khanary-kxml` | gpt2 | — | kuhul_engine | ✓ | — | ✓ |
| `mgguf-qwen-1expert` | qwen_moe | mgguf | mgguf_runtime | ✗ | 1862 | ✓ |
| `dolphin-phi2` | phi2 | — | kuhul_engine | ✗ | 1844 | ✓ |
| `phi3-mini-4k` | phi3 | GGUF | kuhul_engine | ✗ | 2282 | ✓ |
| `gemma-3-4b` | gemma | GGUF | kuhul_engine | ✗ | — | ✓ |
| `gemma-4-e2b` | gemma4 | — | kuhul_engine | ✗ | 4200 | ✓ |
| `qwen-1b8-chat` | qwen | — | kuhul_engine | ✗ | — | ✓ |
| `gpt-oss` | — | GGUF MXFP4 | kuhul_engine | ✗ | — | ✓ |

---

## File locations

```
models/{alias}/atomic.manifest.json     — one per model
models/khanary-kxml-v0.5.0/
  kxml_chat_template.jinja              — KXML jinja template
  kxml_chat_template.json              — KXML template (JSON form)
  kxml_nodes.json                      — compute graph node definitions
  kuhul.tools.jsonl                    — tool registry (one JSON per line)
  kxml_alignment.json                  — alignment config
  MODEL.json                           — model card
  source/                              — source checkpoints

models/khanary-qwen1_8b-v0.1.0/
  qwen1_8b.q8.manifest.json            — older per-quant manifest format
  qwen1_8b.q4.manifest.json

tools/kxml_stock_adapter.py            — chatml ↔ kuhul conversation converter
micronauts/registry.json               — behavior profile index
micronauts/*.json                      — individual behavior profiles
```

Full docs: this file. See also PRIMEOS.md (how PRIMEOS reads atomic.manifest.json), GPU.md (compute paths), DISTILLATION.md (from_zero / LoRA layers).
