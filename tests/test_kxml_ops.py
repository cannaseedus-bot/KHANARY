"""Tests for the KXML tool/op registry (tools/kxml_ops.py).

Verifies ALL KXML tool calls + compute node ops from the kxml-semantic-kernel are present,
correctly typed, and aligned to the KHANARY compute glyphs.
"""

import json

from tools.kxml_ops import KXML_TOOLS, KXML_NODES, all_tools, all_nodes
from tools.khlnary_encoder import GLYPH_IDS


def test_all_twelve_tool_calls_present():
    names = {t[0] for t in KXML_TOOLS}
    expected = {"read_file", "write_file", "exec", "shell", "tool", "agent",
                "micronaut", "skill", "action", "verb", "bot", "http"}
    assert names == expected
    assert len(KXML_TOOLS) == 12


def test_all_seven_node_ops_present():
    names = {n[0] for n in KXML_NODES}
    expected = {"ATTENTION_NODE", "FFN_NODE", "LAYERNORM_NODE", "EMBED_NODE",
                "LM_HEAD_NODE", "LOSS_NODE", "FIELD_OPTIMIZER_NODE"}
    assert names == expected
    assert len(KXML_NODES) == 7


def test_tool_effects_match_source():
    eff = {t[0]: t[1] for t in KXML_TOOLS}
    assert eff["read_file"] == "io" and eff["write_file"] == "io"
    assert eff["exec"] == "process" and eff["shell"] == "shell"
    assert eff["http"] == "network" and eff["agent"] == "agent"


def test_tools_serialize_as_runtime_jsonl_records():
    for t in all_tools():
        # each is a valid kuhul.tools.jsonl record
        s = json.dumps(t)
        back = json.loads(s)
        assert back["type"] == "function"
        assert back["cmd"].startswith("kuhul.")
        assert isinstance(back["args"], list) and back["returns"]


def test_node_glyph_alignment_resolves():
    glyph = {n["name"]: n["kuhul_glyph"] for n in all_nodes()}
    # the two compute ops we actually promoted to glyphs
    assert glyph["ATTENTION_NODE"] == "G_ATTENTION"
    assert GLYPH_IDS["G_ATTENTION"] == 0x51
    assert glyph["FFN_NODE"] == "G_MATMUL"
    assert GLYPH_IDS["G_MATMUL"] == 0x50
    # layernorm/embed/loss/optimizer are not yet glyphs (trainer shaders)
    assert glyph["LAYERNORM_NODE"] is None
    assert glyph["EMBED_NODE"] is None
