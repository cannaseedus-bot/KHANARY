import unittest

from tools.khlnary_compiler import KhlnaryCompiler
from tools.khlnary_webgpu import WebGpuBackend


class TestVerticalStack(unittest.TestCase):
    def test_compiler_emits_expected_sequence_for_linear(self):
        compiler = KhlnaryCompiler()
        compiler.compile_linear_layer(
            weight_file="weights/l1.stb",
            weight_id=0,
            bias_file="weights/l1.stb",
            bias_id=1,
            weight_shape=(8, 16),
        )
        module = compiler.build_module()
        glyph_ids = [((w >> 20) & 0xFF) for w in module.knus]
        self.assertEqual(glyph_ids, [0x30, 0x30, 0x40, 0x41])

    def test_webgpu_backend_binds_storage_buffers(self):
        compiler = KhlnaryCompiler()
        compiler.compile_attention_layer(hidden_size=8, num_heads=2, file_path="weights/attn.stb")
        module = compiler.build_module()
        wgsl = WebGpuBackend().generate_wgsl_shader(module)
        self.assertIn("@group(0) @binding(0)", wgsl)
        self.assertIn("Constants", wgsl)


if __name__ == "__main__":
    unittest.main()
