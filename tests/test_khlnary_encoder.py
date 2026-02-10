import unittest

from tools.khlnary_encoder import (
    GLYPH_IDS,
    KhlNaryParityError,
    compile_python_to_khlnary_words,
    decode_knu,
    encode_knu,
    pack_lane_bundle_u128,
    pack_lane_bundles,
)


class TestKhlNaryEncoder(unittest.TestCase):
    def test_nop_example_word(self):
        self.assertEqual(encode_knu("G_NOP", arity=0, profile_flags=0, payload=0), 0x10000002)

    def test_const_i8_example_word(self):
        self.assertEqual(encode_knu("G_CONST_I8", arity=0, profile_flags=0x1, payload=0x05), 0x10101052)

    def test_decode_rejects_bad_parity(self):
        good = encode_knu("G_NOP")
        bad = good ^ 0x1
        with self.assertRaises(KhlNaryParityError):
            decode_knu(bad)

    def test_compile_python_expr_subset(self):
        words = compile_python_to_khlnary_words("1 + 2")
        decoded = [decode_knu(word)["glyph_name"] for word in words]
        self.assertEqual(decoded, ["G_CONST_I8", "G_CONST_I8", "G_ADD_I32", "G_RET"])

    def test_if_else_lowering_has_conditional_and_unconditional_jump(self):
        src = "if 1 == 0:\n    1\nelse:\n    2\n"
        words = compile_python_to_khlnary_words(src)
        decoded = [decode_knu(word) for word in words]
        glyphs = [d["glyph_name"] for d in decoded]
        self.assertIn("G_IFZ_JUMP8", glyphs)
        self.assertIn("G_JUMP8", glyphs)

        ifz_idx = glyphs.index("G_IFZ_JUMP8")
        jump_payload = decoded[ifz_idx]["payload"]
        self.assertNotEqual(jump_payload, 0)

    def test_while_lowering_has_markers_and_back_jump(self):
        src = "x = 0\nwhile x < 2:\n    x = x + 1\n"
        words = compile_python_to_khlnary_words(src)
        decoded = [decode_knu(word) for word in words]
        glyphs = [d["glyph_name"] for d in decoded]
        self.assertIn("G_WHILE_HEAD", glyphs)
        self.assertIn("G_WHILE_TAIL", glyphs)

        jump_indices = [i for i, g in enumerate(glyphs) if g == "G_JUMP8"]
        self.assertTrue(jump_indices)
        back_jump_payload = decoded[jump_indices[-1]]["payload"]
        self.assertGreaterEqual(back_jump_payload, 0x80)

    def test_function_def_and_call(self):
        src = "def add(a, b):\n    return a + b\n\nadd(1, 2)\n"
        words = compile_python_to_khlnary_words(src)
        decoded = [decode_knu(word) for word in words]
        glyphs = [d["glyph_name"] for d in decoded]
        self.assertIn("G_FUNC_DEF", glyphs)
        self.assertIn("G_FUNC_END", glyphs)
        self.assertIn("G_CALL", glyphs)

        func_def = decoded[glyphs.index("G_FUNC_DEF")]
        func_call = decoded[glyphs.index("G_CALL")]
        self.assertEqual(func_def["payload"], func_call["payload"])

    def test_lane_bundle_packing(self):
        words = [
            encode_knu("G_CONST_I8", profile_flags=1, payload=1),
            encode_knu("G_CONST_I8", profile_flags=1, payload=2),
            encode_knu("G_ADD_I32", arity=2),
            encode_knu("G_RET", arity=1),
            encode_knu("G_NOP"),
        ]
        bundles = pack_lane_bundles(words)
        self.assertEqual(len(bundles), 2)
        self.assertEqual(len(bundles[0]), 4)
        self.assertEqual(len(bundles[1]), 4)

        lane = pack_lane_bundle_u128(bundles[0])
        expected = (
            (bundles[0][0] << 96)
            | (bundles[0][1] << 64)
            | (bundles[0][2] << 32)
            | bundles[0][3]
        )
        self.assertEqual(lane, expected)

    def test_new_control_flow_ids_are_stable(self):
        self.assertEqual(GLYPH_IDS["G_IFZ_JUMP8"], 0x10)
        self.assertEqual(GLYPH_IDS["G_JUMP8"], 0x11)
        self.assertEqual(GLYPH_IDS["G_FUNC_DEF"], 0x20)


if __name__ == "__main__":
    unittest.main()
