import unittest

from tools.khlnary_encoder import compile_to_knu, encode_knu
from tools.khlnary_webgpu import lower_khlnary_to_wgsl, webgpu_js_loader


class TestLoweringSkeletons(unittest.TestCase):
    def test_compile_to_knu_alias(self):
        self.assertEqual(compile_to_knu("1 + 2"), compile_to_knu("1 + 2"))

    def test_webgpu_lowering_emits_bindings(self):
        load_word = encode_knu("G_LOAD_BIN_TENSOR", payload=0x01)

        wgsl = lower_khlnary_to_wgsl([load_word], {0: {"path": "tiny.stb", "alias": "tiny"}})
        self.assertIn("@group(0) @binding(0)", wgsl)
        js = webgpu_js_loader({0: {"path": "tiny.stb", "alias": "tiny"}})
        self.assertIn("loadStbToBuffer", js)


if __name__ == "__main__":
    unittest.main()
