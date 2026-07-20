# KHΛNARY KXML registry — v0.5.0

**KXML is the trainable chat-template + tool-call layer.** Where llama.cpp uses a prompt-time
chat template to structure tool calls, KXML structures them as a declarative node stream that
lowers to the **glyph tokens the model is trained on**, and whose tool nodes the **runtime
dispatches** (the semantic kernel). This version registers *all* of it.

## Contents
- **`kuhul.tools.jsonl`** — every KXML tool call as a runtime record (the file
  `kuhul_tool_runtime.h` loads; it was previously missing). 12 tools:
  read_file, write_file, exec, shell, tool, agent, micronaut, skill, action, verb, bot, http.
- **`kxml_nodes.json`** — the 7 compute node ops (Attention/FFN/LayerNorm/Embed/
  LmHead/Loss/FieldOptimizer) with phase/domain/gravity + the phase machine.
- **`kxml_alignment.json`** — how each tool maps to the glyph tokenizer's tool tier and each
  node to a KHΛNARY KNU compute glyph.
- **`source/`** — the vendored `kxml-semantic-kernel` headers/settings this is derived from.

## Alignment (honest)
- Tools with a trained glyph token: 8/12. Without yet:
  micronaut, action, verb, bot (candidates to add to the tokenizer tool tier).
- Node ops already a KNU glyph: `ATTENTION_NODE→G_ATTENTION`, `FFN_NODE`/`LM_HEAD_NODE→G_MATMUL`.
  Not yet glyphs (trainer shaders exist): LAYERNORM_NODE, EMBED_NODE, LOSS_NODE, FIELD_OPTIMIZER_NODE.

See `MODEL.json` → `honest_scope`: this registers + makes loadable the contracts; it does not
re-implement the runtime (vendored under `source/`), and some builtins there are stubs.

## Reproduce
```
python tools/build_kxml_registry.py
python -m pytest tests/ -q
```
