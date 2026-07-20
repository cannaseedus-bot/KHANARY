"""Tests for the KXML chat template (tools/kxml_chat_template.py).

The trained-in renderer produces a structured glyph-token stream; the .jinja is the
llama-compatible surface. Both must express the same role + tool-call structure.
"""

import sys, os

from tools.kxml_chat_template import render_tokens, to_jinja, TEMPLATE_SPEC

# tokenizer special-token ids (mirror glyph_tokenizer.py)
BOS, EOS, SEP, TOOL_CALL, TOOL_RESULT = 1, 2, 4, 11, 12
T_READ, I_QUESTION, I_ANSWER, I_EXPLAIN = 48, 120, 121, 123


def test_render_tokens_has_role_and_tool_structure():
    ids = render_tokens([
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "read config"},
        {"role": "assistant", "tool_call": {"name": "Read", "args": "config.txt"}},
        {"role": "tool", "content": "port 8080"},
        {"role": "assistant", "content": "the port is 8080"},
    ])
    assert ids[0] == BOS and ids[-1] == EOS
    assert I_EXPLAIN in ids and I_QUESTION in ids and I_ANSWER in ids   # roles
    # tool call appears as TOOL_CALL, T_READ, ..., TOOL_RESULT (trained-in tool tokens)
    assert TOOL_CALL in ids and T_READ in ids and TOOL_RESULT in ids
    assert ids[ids.index(TOOL_CALL) + 1] == T_READ


def test_render_tokens_multi_turn_not_truncated():
    # regression: per-turn padding once collapsed multi-turn dialogues to a single turn
    ids = render_tokens([{"role": "user", "content": "a"},
                         {"role": "assistant", "content": "b"},
                         {"role": "user", "content": "c"}])
    assert ids.count(SEP) == 3           # three turns each terminated by SEP
    assert ids.count(I_QUESTION) == 2 and ids.count(I_ANSWER) == 1


def test_jinja_surface_covers_roles_and_generation_prompt():
    j = to_jinja()
    for marker in ("<BOS>", "<I_EXPLAIN>", "<I_QUESTION>", "<I_ANSWER>", "<TOOL_CALL>", "<SEP>"):
        assert marker in j
    assert "add_generation_prompt" in j


def test_spec_role_map():
    assert TEMPLATE_SPEC["roles"]["system"] == "I_EXPLAIN"
    assert TEMPLATE_SPEC["roles"]["assistant"] == "I_ANSWER"
    assert TEMPLATE_SPEC["trained_in"] is True
