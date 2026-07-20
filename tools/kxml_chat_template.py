# kxml_chat_template.py — the KXML chat template (two halves).
#
# llama.cpp uses a prompt-time Jinja chat template to structure roles + tool calls. KXML's chat
# template is the same contract, but it is TRAINED IN: it renders to the glyph tokens the model
# learns (glyph_tokenizer.encode_dialogue / encode_tool_call), not a formatted string. We keep a
# llama-compatible .jinja surface too, so KXML chat interops with GGUF-style runtimes.
#
#   render_tokens(messages)  -> glyph token ids  (the trained-in form)
#   to_jinja()               -> a llama-style chat_template string (the interop surface)
#
# Run: python tools/kxml_chat_template.py   # emits the artifacts into the kxml v0.5.0 model folder
import os, sys, json

# role -> the glyph tokenizer's intent/special token that prefixes the turn
ROLES = {"system": "I_EXPLAIN", "user": "I_QUESTION", "human": "I_QUESTION",
         "assistant": "I_ANSWER", "tool": "TOOL_RESULT"}
SPECIALS = {"bos": "BOS", "eos": "EOS", "turn_sep": "SEP", "pad": "PAD"}
TOOL_CALL = {"open": "TOOL_CALL", "tool_token": "T_<NAME>", "close": "TOOL_RESULT"}
REASONING = {"open": "THINK_START", "close": "THINK_END"}
GENERATION_PROMPT = "I_ANSWER"

TEMPLATE_SPEC = {
    "name": "kxml-chat/v1",
    "trained_in": True,
    "roles": ROLES, "specials": SPECIALS,
    "tool_call": TOOL_CALL, "reasoning": REASONING,
    "generation_prompt": GENERATION_PROMPT,
    "renders_via": "glyph_tokenizer.encode_dialogue / encode_turn / encode_tool_call",
    "note": ("KXML chat is trained into the token stream (roles/tools are glyph tokens, not string "
             "tags). The .jinja is the llama-compatible interop surface describing the same structure."),
}


def _tokenizer():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok_dir = os.path.join(here, "models", "khanary-gpt2-v0.4.0", "tokenizer")
    sys.path.insert(0, tok_dir)
    import glyph_tokenizer  # noqa
    return glyph_tokenizer.GlyphTokenizer(), glyph_tokenizer


def render_tokens(messages, max_len=512):
    """The TRAINED-IN chat template: messages -> glyph token ids. A message may carry
    {"role","content"} or {"role":"assistant","tool_call":{"name","args"}}. Built from the
    non-padding encode_raw path so multi-turn dialogues aren't truncated by per-turn padding."""
    tok, G = _tokenizer()
    intent = {"system": G.I_EXPLAIN, "user": G.I_QUESTION, "human": G.I_QUESTION,
              "assistant": G.I_ANSWER, "tool": G.TOOL_RESULT}
    ids = [G.BOS]
    for m in messages:
        tc = m.get("tool_call")
        if tc:
            ids += tok.encode_tool_call(tc["name"], tc.get("args", ""))
            continue
        role = m.get("role", "assistant")
        if role in intent:
            ids.append(intent[role])
        ids += tok.encode_raw(m.get("content", m.get("text", "")), add_special=False)
        ids.append(G.SEP)
    ids.append(G.EOS)
    return ids[:max_len]


def to_jinja():
    """A llama.cpp-compatible chat_template string. Uses the KXML special-token *names* as markers
    so a GGUF-style runtime can render the same role structure KXML trains in."""
    return (
        "{{ '<BOS>' }}"
        "{% for m in messages %}"
        "{% if m['role'] == 'system' %}{{ '<I_EXPLAIN>' }}"
        "{% elif m['role'] in ['user', 'human'] %}{{ '<I_QUESTION>' }}"
        "{% elif m['role'] == 'assistant' %}{{ '<I_ANSWER>' }}"
        "{% elif m['role'] == 'tool' %}{{ '<TOOL_RESULT>' }}"
        "{% endif %}"
        "{% if m.get('tool_call') %}{{ '<TOOL_CALL>' }}{{ '<T_' ~ m['tool_call']['name'] ~ '>' }}"
        "{{ m['tool_call'].get('args', '') }}{{ '<TOOL_RESULT>' }}"
        "{% else %}{{ m['content'] }}{% endif %}"
        "{{ '<SEP>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<I_ANSWER>' }}{% endif %}"
    )


def emit(model_dir):
    json.dump({**TEMPLATE_SPEC,
               "example": {
                   "messages": [
                       {"role": "system", "content": "be concise"},
                       {"role": "user", "content": "read config.txt"},
                       {"role": "assistant", "tool_call": {"name": "Read", "args": "config.txt"}},
                       {"role": "tool", "content": "port=8080"},
                       {"role": "assistant", "content": "the port is 8080"},
                   ]}},
              open(os.path.join(model_dir, "kxml_chat_template.json"), "w", encoding="utf-8"), indent=2)
    open(os.path.join(model_dir, "kxml_chat_template.jinja"), "w", encoding="utf-8").write(to_jinja() + "\n")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md = os.path.join(here, "models", "khanary-kxml-v0.5.0")
    emit(md)
    ids = render_tokens([{"role": "system", "content": "be concise"},
                         {"role": "user", "content": "hi"},
                         {"role": "assistant", "tool_call": {"name": "Read", "args": "f.txt"}}])
    print("emitted kxml_chat_template.{json,jinja}")
    print("sample render_tokens ->", ids[:12], "...")
