import unittest

from tools.khlnary_cuda import lower_khlnary_to_cuda
from tools.khlnary_encoder import compile_to_knu, encode_knu
from tools.khlnary_webgpu import lower_khlnary_to_wgsl, webgpu_js_loader


class TestLoweringSkeletons(unittest.TestCase):
    def test_compile_to_knu_alias(self):
        self.assertEqual(compile_to_knu("1 + 2"), compile_to_knu("1 + 2"))

    def test_cuda_lowering_emits_init_and_kernel(self):
        knus = [encode_knu("G_NOP"), encode_knu("G_NOP")]
        # inject one G_LOAD_BIN_TENSOR word with payload=0x01
        load_word = encode_knu("G_NOP")
        load_word &= ~(0xFF << 20)
        load_word |= (0x30 << 20)
        load_word &= ~(0xFF << 4)
        load_word |= (0x01 << 4)
        knus.append(load_word)

        src = lower_khlnary_to_cuda(knus, {0: {"path": "tiny.stb", "alias": "tiny"}})
        self.assertIn("init_t_0_1", src)
        self.assertIn("__global__ void khlnary_kernel", src)

    def test_webgpu_lowering_emits_bindings(self):
        load_word = encode_knu("G_NOP")
        load_word &= ~(0xFF << 20)
        load_word |= (0x30 << 20)
        load_word &= ~(0xFF << 4)
        load_word |= (0x01 << 4)

        wgsl = lower_khlnary_to_wgsl([load_word], {0: {"path": "tiny.stb", "alias": "tiny"}})
        self.assertIn("@group(0) @binding(0)", wgsl)
        js = webgpu_js_loader({0: {"path": "tiny.stb", "alias": "tiny"}})
        self.assertIn("loadStbToBuffer", js)


if __name__ == "__main__":
    unittest.main()
