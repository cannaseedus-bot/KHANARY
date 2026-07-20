"""Tests for the KXML -> stock-GGUF adapter (tools/kxml_stock_adapter.py).

Translation logic is pure (no GGUF file needed); render is exercised with an inline ChatML
template so it needs no model on disk.
"""

from tools.kxml_stock_adapter import kxml_to_messages, render_for_gguf

DIALOGUE = [
    {"role": "system", "content": "be concise"},
    {"role": "user", "content": "read config"},
    {"role": "assistant", "tool_call": {"name": "Read", "args": "config.txt"}},
    {"role": "tool", "content": "port=8080"},
    {"role": "assistant", "content": "the port is 8080"},
]

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def test_inline_style_folds_tools_into_text():
    msgs = kxml_to_messages(DIALOGUE, tool_style="inline")
    roles = [m["role"] for m in msgs]
    # only system/user/assistant — no 'tool' role (works with any template)
    assert set(roles) <= {"system", "user", "assistant"}
    assert any("<tool_call>" in m["content"] and "Read" in m["content"] for m in msgs)
    assert any("<tool_response>" in m["content"] and "port=8080" in m["content"] for m in msgs)


def test_openai_style_emits_structured_tool_calls():
    msgs = kxml_to_messages(DIALOGUE, tool_style="openai")
    call = next(m for m in msgs if m.get("tool_calls"))
    fn = call["tool_calls"][0]["function"]
    assert fn["name"] == "Read"
    assert fn["arguments"] == {"input": "config.txt"}          # object, not a JSON string
    assert any(m["role"] == "tool" and "port=8080" in m["content"] for m in msgs)


def test_dict_args_pass_through():
    msgs = kxml_to_messages([{"role": "assistant",
                              "tool_call": {"name": "Bash", "args": {"command": "ls"}}}],
                            tool_style="openai")
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


def test_render_with_supplied_template():
    prompt = render_for_gguf(DIALOGUE, chat_template=CHATML, tool_style="inline")
    assert "<|im_start|>system" in prompt and "be concise" in prompt
    assert "<tool_call>" in prompt and "Read" in prompt          # tool folded into a turn
    assert prompt.rstrip().endswith("<|im_start|>assistant")     # generation prompt
