# kxml_ops.py — canonical registry of ALL KXML tool calls + compute node ops.
#
# KXML is the declarative chat/tool-call + compute-graph layer (the trainable analog of an LLM
# chat template): a KXML node stream lowers to glyph tokens the model is trained on, and the
# runtime dispatches the tool nodes to real functions (the "semantic kernel"). This module is the
# single source of truth for that alignment, derived verbatim from the kxml-semantic-kernel:
#   - tool calls  <- kuhul_functions.h   (Kuhul::Functions::FunctionRegistry builtins)
#   - node ops    <- kxml_settings.xml   (KXML.Nodes structs)
#   - enums       <- kxml_settings.xml   (PHASE / DOMAIN / DEVICE / GRAVITY)
#
# "token" aligns each tool to the glyph tokenizer's tool tier (models/.../tokenizer/
# glyph_tokenizer.py); None = no trained token yet (candidate to add). "glyph" aligns each
# compute node to a KHANARY KNU compute glyph; None = shader exists, not yet a glyph.

# ── enums (kxml_settings.xml) ────────────────────────────────────────────────
PHASES   = ["Pop", "Wo", "Sek", "Chen", "Xul"]           # phase machine
DOMAINS  = ["Compute", "Storage", "Routing", "Meta", "Ui"]
DEVICES  = ["Cpu", "Gpu", "Auto"]
GRAVITY  = ["Float", "Embed", "Normal", "Heavy"]

# ── ALL KXML tool calls (kuhul_functions.h RegisterBuiltins) ─────────────────
# (name, effect, args, returns, permissions, sandbox, cmd, glyph_token, help)
KXML_TOOLS = [
    ("read_file",  "io",        ["path:string"],                        "string", ["fs.read"],  "restricted",        "kuhul.fs.read",   "T_READ",        "Read entire file contents"),
    ("write_file", "io",        ["path:string", "content:string"],      "bool",   ["fs.write"], "restricted",        "kuhul.fs.write",  "T_WRITE",       "Write content to a file"),
    ("exec",       "process",   ["cmd:string", "args:list"],            "list",   ["os.exec"],  "process_isolation", "kuhul.os.exec",   "T_BASH",        "Execute a process, capture stdout lines"),
    ("shell",      "shell",     ["cmd:string"],                         "bool",   ["os.exec"],  "process_isolation", "kuhul.os.shell",  "T_BASH",        "Run a shell command, return success"),
    ("tool",       "tool",      ["name:string", "input:map"],           "map",    ["tool.call"],"api_gateway",       "kuhul.tool",      "T_MCP_CALL",    "Invoke a registered tool by name"),
    ("agent",      "agent",     ["name:string", "prompt:string"],       "string", ["ai.infer"], "isolated",          "kuhul.agent",     "T_AGENT_SPAWN", "Route a prompt to a named agent"),
    ("micronaut",  "micronaut", ["name:string", "args:map"],            "map",    ["compute"],  "thread_pool",       "kuhul.micronaut", None,            "Spawn a micronaut capsule"),
    ("skill",      "skill",     ["name:string", "input:any"],           "any",    ["skill.call"],"isolated",         "kuhul.skill",     "T_SKILL",       "Invoke a skill by name"),
    ("action",     "action",    ["name:string", "params:map"],          "map",    ["action"],   "isolated",          "kuhul.action",    None,            "Perform a named action"),
    ("verb",       "verb",      ["name:string", "subject:string", "object:string"], "string", [], "readonly",        "kuhul.verb",      None,            "Apply a verb to subject/object"),
    ("bot",        "bot",       ["name:string", "message:string"],      "string", ["ai.infer"], "isolated",          "kuhul.bot",       None,            "Send a message to a named bot"),
    ("http",       "network",   ["method:string", "url:string", "body:string"], "map", ["network"], "network_sandbox", "kuhul.http",   "T_WEB_FETCH",   "Perform an HTTP request"),
]

# ── ALL KXML compute node ops (kxml_settings.xml KXML.Nodes) ─────────────────
# (name, phase, domain, gravity, params{}, glyph)  glyph=None -> trainer shader, not yet a KNU glyph
KXML_NODES = [
    ("ATTENTION_NODE", "Sek",  "Compute", "Normal",
        {"N_HEADS": 12, "HEAD_DIM": 64, "SEQ_LEN": 1024, "CAUSAL_MASK": True}, "G_ATTENTION"),
    ("FFN_NODE", "Sek", "Compute", "Normal",
        {"HIDDEN_DIM": 768, "INTER_DIM": 3072, "ACTIVATION": "gelu"}, "G_MATMUL"),
    ("LAYERNORM_NODE", "Wo", "Compute", "Heavy",
        {"DIM": 768, "EPSILON": 1e-5}, "G_LAYERNORM"),
    ("EMBED_NODE", "Pop", "Compute", "Embed",
        {"VOCAB_SIZE": 50257, "N_CTX": 1024, "N_EMBD": 768}, "G_EMBED"),
    ("LM_HEAD_NODE", "Xul", "Compute", "Heavy",
        {"VOCAB_SIZE": 50257}, "G_MATMUL"),
    ("LOSS_NODE", "Chen", "Compute", "Heavy",
        {"LOSS_TYPE": "cross_entropy", "TARGET_LOSS": 0.5}, None),
    ("FIELD_OPTIMIZER_NODE", "Chen", "Compute", "Normal",
        {"LR": 1e-5, "BETA1": 0.9, "BETA2": 0.999, "WEIGHT_DECAY": 0.1, "GRAD_CLIP": 1.0,
         "ATTRACTION_WELL": True, "SCROLL_INERTIA": True, "WIND_FIELD": True,
         "NAVIGATION_FORCE": True}, None),
]


def tool_dict(t):
    """A KXML tool as a kuhul.tools.jsonl record (the format kuhul_tool_runtime.h loads)."""
    name, effect, args, returns, perms, sandbox, cmd, token, help_ = t
    return {
        "id": f"kxml_{name}", "type": "function", "name": name, "version": "1.0.0",
        "effect": effect, "permissions": perms, "sandbox": sandbox,
        "batchable": effect in ("io", "pure", "network"), "thread_safe": effect != "shell",
        "timeout_ms": 5000, "cmd": cmd, "args": args, "returns": returns,
        "glyph_token": token, "help": help_,
    }


def node_dict(n):
    name, phase, domain, gravity, params, glyph = n
    return {"name": name, "phase": phase, "domain": domain, "gravity": gravity,
            "params": params, "kuhul_glyph": glyph}


def all_tools():
    return [tool_dict(t) for t in KXML_TOOLS]


def all_nodes():
    return [node_dict(n) for n in KXML_NODES]


__all__ = ["PHASES", "DOMAINS", "DEVICES", "GRAVITY", "KXML_TOOLS", "KXML_NODES",
           "tool_dict", "node_dict", "all_tools", "all_nodes"]
