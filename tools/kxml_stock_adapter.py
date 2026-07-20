# kxml_stock_adapter.py — run a KXML dialogue on a STOCK GGUF (no retraining).
#
# A KHANARY-trained model gets the KXML chat template TRAINED IN (glyph tokens). A stock GGUF
# (llama/qwen/phi/gemma/LFM...) does NOT have those tokens — you cannot paste the KXML .jinja into
# its header. This adapter is the bridge: it translates a KXML message list onto the target
# model's OWN chat_template + tool-call convention, so one KXML front-end drives any GGUF today.
#
#   read_gguf_metadata(path)                     -> {chat_template, bos_token, eos_token, arch, ...}
#   kxml_to_messages(kxml_messages, tool_style)  -> OpenAI-style messages (roles + tool calls)
#   render_for_gguf(kxml_messages, gguf_path)    -> the final prompt string for that model
#
# Run: python tools/kxml_stock_adapter.py <model.gguf> [inline|openai]
import os, sys, json, struct

# ── minimal GGUF metadata reader (header only; does not touch tensors) ───────────
_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf_metadata(path, resolve_special=True):
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    struct.unpack("<I", f.read(4))            # version
    struct.unpack("<Q", f.read(8))            # tensor_count
    (nkv,) = struct.unpack("<Q", f.read(8))

    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def rval(t):
        if t == 8:  # string
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")
        if t in _SCALAR:
            b = f.read(_SCALAR[t])
            if t == 7: return b[0] != 0                       # bool
            if t in (4, 5): return struct.unpack("<i", b)[0]  # (u)int32
            if t in (10, 11): return struct.unpack("<q", b)[0]
            return None
        if t == 9:  # array
            (et,) = struct.unpack("<I", f.read(4))
            (cnt,) = struct.unpack("<Q", f.read(8))
            arr = [rval(et) for _ in range(cnt)]
            return arr
        raise ValueError(f"unknown gguf value type {t}")

    md = {}
    want = {"tokenizer.chat_template", "general.architecture",
            "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"}
    tokens = None
    for _ in range(nkv):
        key = rstr()
        (t,) = struct.unpack("<I", f.read(4))
        v = rval(t)
        if key == "tokenizer.ggml.tokens":
            tokens = v            # keep for bos/eos string resolution
        elif key in want:
            md[key] = v
    f.close()

    out = {"arch": md.get("general.architecture"),
           "chat_template": md.get("tokenizer.chat_template", ""),
           "bos_token": "", "eos_token": ""}
    if resolve_special and tokens:
        bi, ei = md.get("tokenizer.ggml.bos_token_id"), md.get("tokenizer.ggml.eos_token_id")
        if isinstance(bi, int) and 0 <= bi < len(tokens): out["bos_token"] = tokens[bi]
        if isinstance(ei, int) and 0 <= ei < len(tokens): out["eos_token"] = tokens[ei]
    return out


# ── KXML -> stock messages ───────────────────────────────────────────────────────
_ROLE = {"system": "system", "user": "user", "human": "user",
         "assistant": "assistant", "tool": "tool"}


def _args_obj(args):
    """Arguments as an object — templates like LFM2/qwen iterate arguments.items()."""
    return args if isinstance(args, dict) else {"input": args}


def _args_json(args):
    return json.dumps(_args_obj(args))


def kxml_to_messages(kxml_messages, tool_style="inline"):
    """Translate a KXML message list into the shape a stock chat_template consumes.

    tool_style="inline"  — tool calls/results become text inside user/assistant turns; works with
                           ANY chat_template (even those with no tool support). Most portable.
    tool_style="openai"  — assistant.tool_calls[] + role:"tool" results; for tool-aware templates
                           (qwen/hermes/LFM2/functionary)."""
    out = []
    for m in kxml_messages:
        role = _ROLE.get(m.get("role", "assistant"), "user")
        tc = m.get("tool_call")
        if tc:
            name, args = tc["name"], tc.get("args", "")
            if tool_style == "openai":
                out.append({"role": "assistant", "content": "",
                            "tool_calls": [{"id": f"call_{len(out)}", "type": "function",
                                            "function": {"name": name, "arguments": _args_obj(args)}}]})
            else:
                out.append({"role": "assistant",
                            "content": f'<tool_call>\n{{"name": "{name}", "arguments": {_args_json(args)}}}\n</tool_call>'})
            continue
        if role == "tool" and tool_style != "openai":
            out.append({"role": "user",
                        "content": f'<tool_response>\n{m.get("content", "")}\n</tool_response>'})
        else:
            out.append({"role": role, "content": m.get("content", m.get("text", ""))})
    return out


def render_for_gguf(kxml_messages, gguf_path=None, chat_template=None, bos_token="", eos_token="",
                    tool_style="inline", add_generation_prompt=True):
    """Translate + render a KXML dialogue into the final prompt string for a stock GGUF, using that
    model's own chat_template (read from the GGUF header if a path is given)."""
    if gguf_path:
        md = read_gguf_metadata(gguf_path)
        chat_template = chat_template or md["chat_template"]
        bos_token = bos_token or md["bos_token"]
        eos_token = eos_token or md["eos_token"]
    if not chat_template:
        raise ValueError("no chat_template (pass gguf_path or chat_template)")
    messages = kxml_to_messages(kxml_messages, tool_style)
    import jinja2
    env = jinja2.Environment()
    env.filters.setdefault("tojson", lambda o, **k: json.dumps(o))
    tmpl = env.from_string(chat_template)
    return tmpl.render(messages=messages, bos_token=bos_token, eos_token=eos_token,
                       add_generation_prompt=add_generation_prompt)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/kxml_stock_adapter.py <model.gguf> [inline|openai]"); sys.exit(1)
    gguf, style = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "inline")
    md = read_gguf_metadata(gguf)
    print(f"[gguf] arch={md['arch']} bos={md['bos_token']!r} eos={md['eos_token']!r} "
          f"chat_template={len(md['chat_template'])} chars")
    kxml = [{"role": "system", "content": "You are concise."},
            {"role": "user", "content": "read config.txt and tell me the port"},
            {"role": "assistant", "tool_call": {"name": "Read", "args": "config.txt"}},
            {"role": "tool", "content": "port=8080"},
            {"role": "assistant", "content": "The port is 8080."}]
    prompt = render_for_gguf(kxml, gguf_path=gguf, tool_style=style)
    print(f"\n===== KXML dialogue rendered for {os.path.basename(gguf)} (style={style}) =====\n")
    print(prompt)
