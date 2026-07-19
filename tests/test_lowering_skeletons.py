import unittest

from tools.khlnary_encoder import compile_to_knu, encode_knu
from tools.khlnary_webgpu import WebGpuBackend, lower_khlnary_to_wgsl, webgpu_js_loader


class TestLoweringSkeletons(unittest.TestCase):
    def test_compile_to_knu_alias(self):
        self.assertEqual(compile_to_knu("1 + 2"), compile_to_knu("1 + 2"))

    def test_webgpu_lowering_emits_bindings(self):
        load_word = encode_knu("G_LOAD_BIN_TENSOR", payload=0x01)

        wgsl = lower_khlnary_to_wgsl([load_word], {0: {"path": "tiny.stb", "alias": "tiny"}})
        self.assertIn("@group(0) @binding(0)", wgsl)
        js = webgpu_js_loader({0: {"path": "tiny.stb", "alias": "tiny"}})
        self.assertIn("loadStbToBuffer", js)

    def test_wgsl_geometry_glyph_dispatch_parity(self):
        # WGSL parity with the HLSL backend: a geometry glyph in the KNU stream selects the
        # real geometry kernel, not the tensor-copy skeleton.
        xform = lower_khlnary_to_wgsl([encode_knu("G_VERTEX_TRANSFORM", payload=0)], {})
        self.assertEqual(xform, WebGpuBackend().generate_vertex_transform_wgsl())
        self.assertIn("xf.M * vec4<f32>(p, 1.0)", xform)   # actual transform
        self.assertNotIn("output_buf[idx] = input_buf[idx]", xform)  # not the copy skeleton

        # G_VERTEX_SKIN wins over G_VERTEX_TRANSFORM, mirroring lower_khlnary_to_hlsl.
        skin = lower_khlnary_to_wgsl(
            [encode_knu("G_VERTEX_TRANSFORM", payload=0), encode_knu("G_VERTEX_SKIN", payload=0)], {})
        self.assertEqual(skin, WebGpuBackend().generate_skinning_wgsl())
        self.assertIn("array<mat4x4<f32>>", skin)
        self.assertIn("let sn = m3 * nrm;", skin)          # normal skinning

    def test_wgsl_matmul_glyph_dispatch_parity(self):
        # G_MATMUL selects a GEMM kernel in the WGSL backend too (co-equal with HLSL).
        mm = lower_khlnary_to_wgsl([encode_knu("G_MATMUL", payload=0)], {})
        self.assertEqual(mm, WebGpuBackend().generate_matmul_wgsl())
        self.assertIn("acc = acc + A[row * g.K + k] * B[k * g.N + col];", mm)
        self.assertIn("@workgroup_size(16, 16)", mm)
        self.assertNotIn("output_buf[idx] = input_buf[idx]", mm)  # not the copy skeleton

    def test_wgsl_attention_glyph_dispatch_parity(self):
        # G_ATTENTION selects the causal-MHA kernel in the WGSL backend too (co-equal with HLSL).
        att = lower_khlnary_to_wgsl([encode_knu("G_ATTENTION", payload=0)], {})
        self.assertEqual(att, WebGpuBackend().generate_attention_wgsl())
        self.assertIn("struct AttnFwdParams", att)
        self.assertIn("let e = exp(P_buf[p_row + j] - mx);", att)  # max-stable softmax
        self.assertIn("@builtin(workgroup_id) gid", att)           # one workgroup per head


if __name__ == "__main__":
    unittest.main()
