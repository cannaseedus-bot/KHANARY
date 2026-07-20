# build_kxml_registry.py — emit the KHANARY KXML registry (v0.5.0).
#
# Materializes ALL KXML tool calls + compute node ops (tools/kxml_ops.py) into a versioned
# KHANARY artifact: the runtime-loadable kuhul.tools.jsonl (the file kuhul_tool_runtime.h loads,
# previously missing), the node-op graph, and the alignment maps to the glyph tokenizer's tool
# tier (semantic-kernel surface) and the KHANARY compute glyphs. Vendors the source
# kxml-semantic-kernel headers/settings as reference.
import os, sys, json, shutil, hashlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "tools"))
from tools.kxml_ops import all_tools, all_nodes, PHASES, DOMAINS, DEVICES, GRAVITY, KXML_TOOLS, KXML_NODES
from tools.khlnary_encoder import GLYPH_IDS

VERSION = "0.5.0"
MODEL_DIR = os.path.join(_ROOT, "models", f"khanary-kxml-v{VERSION}")
SK = r"C:\Users\canna\.ASX.cpp\kxml-semantic-kernel"
TOKENIZER = os.path.join(_ROOT, "models", "khanary-gpt2-v0.4.0", "tokenizer", "glyph_tokenizer.py")
VENDOR = ["kxml_settings.xml", "kuhul_functions.h", "kuhul_tool_runtime.h"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def load_token_ids():
    """The glyph tokenizer's tool-tier name->id map, for validating tool alignment."""
    ids = {}
    if os.path.exists(TOKENIZER):
        import ast
        tree = ast.parse(open(TOKENIZER, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
               and isinstance(node.value.value, int) and len(node.targets) == 1 \
               and isinstance(node.targets[0], ast.Name):
                ids[node.targets[0].id] = node.value.value
    return ids


def main():
    os.makedirs(os.path.join(MODEL_DIR, "source"), exist_ok=True)
    tok_ids = load_token_ids()

    tools = all_tools()
    nodes = all_nodes()

    # 1) kuhul.tools.jsonl — the runtime-loadable tool registry (one JSON object per line)
    with open(os.path.join(MODEL_DIR, "kuhul.tools.jsonl"), "w", encoding="utf-8") as f:
        for t in tools:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 2) kxml_nodes.json — the compute-graph node ops
    json.dump({"phases": PHASES, "domains": DOMAINS, "devices": DEVICES, "gravity": GRAVITY,
               "nodes": nodes},
              open(os.path.join(MODEL_DIR, "kxml_nodes.json"), "w", encoding="utf-8"), indent=2)

    # 3) alignment: tool -> glyph token, node -> KNU compute glyph; flag unmapped
    tool_align, tool_unmapped = {}, []
    for t in tools:
        tk = t["glyph_token"]
        if tk and tk in tok_ids:
            tool_align[t["name"]] = {"token": tk, "id": tok_ids[tk]}
        else:
            tool_unmapped.append(t["name"])
            tool_align[t["name"]] = {"token": tk, "id": None}
    node_align, node_unmapped = {}, []
    for n in nodes:
        g = n["kuhul_glyph"]
        if g and g in GLYPH_IDS:
            node_align[n["name"]] = {"glyph": g, "id": GLYPH_IDS[g]}
        else:
            node_unmapped.append(n["name"])
            node_align[n["name"]] = {"glyph": g, "id": None}
    alignment = {
        "tool_to_glyph_token": tool_align,
        "node_to_kuhul_glyph": node_align,
        "tools_without_trained_token": tool_unmapped,
        "nodes_without_glyph": node_unmapped,
        "note": ("tools_without_trained_token are KXML calls with no dedicated glyph-tokenizer token "
                 "yet (candidates to add to the tool tier). nodes_without_glyph have a trainer HLSL "
                 "shader but are not yet a KNU compute glyph (next promotions after G_MATMUL/G_ATTENTION)."),
    }
    json.dump(alignment, open(os.path.join(MODEL_DIR, "kxml_alignment.json"), "w", encoding="utf-8"), indent=2)

    # 4) vendor the source semantic-kernel files (reference)
    vendored = []
    for fn in VENDOR:
        src = os.path.join(SK, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(MODEL_DIR, "source", fn)); vendored.append(f"source/{fn}")
        else:
            print(f"  WARN missing (skipped): {src}")

    manifest = {
        "name": "khanary-kxml", "version": VERSION, "kind": "KXML tool/op registry (semantic-kernel layer)",
        "summary": ("KXML is the trainable chat-template + tool-call layer: a declarative node stream "
                    "that lowers to glyph tokens the model is trained on, and whose tool nodes the "
                    "runtime dispatches (the semantic kernel). This registry enumerates ALL of it."),
        "counts": {"tool_calls": len(tools), "node_ops": len(nodes)},
        "tool_calls": [t["name"] for t in tools],
        "node_ops": [n["name"] for n in nodes],
        "phase_machine": PHASES, "domains": DOMAINS, "gravity": GRAVITY,
        "artifacts": {
            "kuhul.tools.jsonl": "runtime-loadable tool registry (kuhul_tool_runtime.h format)",
            "kxml_nodes.json": "compute-graph node ops + enums",
            "kxml_alignment.json": "tool->glyph-token and node->KNU-glyph maps",
            "source/": vendored,
        },
        "alignment_summary": {
            "tools_with_trained_token": len(tools) - len(tool_unmapped),
            "tools_without_trained_token": tool_unmapped,
            "nodes_with_glyph": [n["name"] for n in nodes if n["name"] not in node_unmapped],
            "nodes_without_glyph": node_unmapped,
        },
        "provenance": {"tool_calls": "kuhul_functions.h", "node_ops": "kxml_settings.xml",
                       "runtime": "kuhul_tool_runtime.h", "source_tree": ".ASX.cpp/kxml-semantic-kernel"},
        "honest_scope": [
            "This registry + kuhul.tools.jsonl make the KXML tool calls enumerable and runtime-loadable; "
            "it does NOT itself execute them (the runtime is kuhul_tool_runtime.h, vendored under source/).",
            "Several tool builtins in kuhul_functions.h are stubs (http/agent/bot/micronaut return "
            "placeholder maps); the registry records their contract, not a real implementation.",
        ],
        "generator": "tools/build_kxml_registry.py",
    }
    json.dump(manifest, open(os.path.join(MODEL_DIR, "MODEL.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    readme = f"""# KHΛNARY KXML registry — v{VERSION}

**KXML is the trainable chat-template + tool-call layer.** Where llama.cpp uses a prompt-time
chat template to structure tool calls, KXML structures them as a declarative node stream that
lowers to the **glyph tokens the model is trained on**, and whose tool nodes the **runtime
dispatches** (the semantic kernel). This version registers *all* of it.

## Contents
- **`kuhul.tools.jsonl`** — every KXML tool call as a runtime record (the file
  `kuhul_tool_runtime.h` loads; it was previously missing). {len(tools)} tools:
  {", ".join(t["name"] for t in tools)}.
- **`kxml_nodes.json`** — the {len(nodes)} compute node ops (Attention/FFN/LayerNorm/Embed/
  LmHead/Loss/FieldOptimizer) with phase/domain/gravity + the phase machine.
- **`kxml_alignment.json`** — how each tool maps to the glyph tokenizer's tool tier and each
  node to a KHΛNARY KNU compute glyph.
- **`source/`** — the vendored `kxml-semantic-kernel` headers/settings this is derived from.

## Alignment (honest)
- Tools with a trained glyph token: {len(tools) - len(tool_unmapped)}/{len(tools)}. Without yet:
  {", ".join(tool_unmapped) or "none"} (candidates to add to the tokenizer tool tier).
- Node ops already a KNU glyph: `ATTENTION_NODE→G_ATTENTION`, `FFN_NODE`/`LM_HEAD_NODE→G_MATMUL`.
  Not yet glyphs (trainer shaders exist): {", ".join(node_unmapped)}.

See `MODEL.json` → `honest_scope`: this registers + makes loadable the contracts; it does not
re-implement the runtime (vendored under `source/`), and some builtins there are stubs.

## Reproduce
```
python tools/build_kxml_registry.py
python -m pytest tests/ -q
```
"""
    open(os.path.join(MODEL_DIR, "README.md"), "w", encoding="utf-8").write(readme)

    print(f"wrote {os.path.relpath(MODEL_DIR, _ROOT)}")
    for dp, _, fs in os.walk(MODEL_DIR):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            print(f"  {os.path.relpath(p, MODEL_DIR):34} {os.path.getsize(p):>7} B")
    print(f"tools={len(tools)} nodes={len(nodes)} | tools w/ token={len(tools)-len(tool_unmapped)} | unmapped tools={tool_unmapped}")


if __name__ == "__main__":
    main()
