"""Tests for the D3D11 cs_5_0 (HLSL) backend — sibling of the WebGPU backend.

Structural checks (portable; no fxc/GPU needed). The HLSL these produce has been
verified to compile as cs_5_0 with fxc on an Intel HD 4600 (where WebGPU is blocklisted).
"""

from types import SimpleNamespace

from tools.khlnary_dx11 import Dx11Backend, lower_khlnary_to_hlsl
from tools.khlnary_encoder import encode_knu


def test_generate_hlsl_shader_bindings_and_entry():
    mod = SimpleNamespace(tensors=[SimpleNamespace(ptr_name="w0"),
                                   SimpleNamespace(ptr_name="b0")])
    hlsl = Dx11Backend().generate_hlsl_shader(mod)
    # one StructuredBuffer per module tensor, at t0..t(N-1)
    assert "StructuredBuffer<float> w0 : register(t0);" in hlsl
    assert "StructuredBuffer<float> b0 : register(t1);" in hlsl
    # input at t(N), output UAV at u0, constants cbuffer, cs_5_0 shape
    assert "input_buffer  : register(t2);" in hlsl
    assert "RWStructuredBuffer<float> output_buffer : register(u0);" in hlsl
    assert "cbuffer Constants : register(b0)" in hlsl
    assert "[numthreads(256, 1, 1)]" in hlsl
    assert "void main(uint3 gid : SV_DispatchThreadID)" in hlsl


def test_lower_khlnary_to_hlsl_binds_load_bin_tensor_words():
    knus = [
        encode_knu("G_LOAD_BIN_TENSOR", payload=(0 << 4) | 0),
        encode_knu("G_LOAD_BIN_TENSOR", payload=(0 << 4) | 1),
    ]
    hlsl = lower_khlnary_to_hlsl(knus, {0: {"path": "x.stb"}})
    assert "StructuredBuffer<float> t_0_0 : register(t0);" in hlsl
    assert "StructuredBuffer<float> t_0_1 : register(t1);" in hlsl
    assert "RWStructuredBuffer<float> output_buf : register(u0);" in hlsl
    assert "[numthreads(64, 1, 1)]" in hlsl


def test_generate_vertex_transform_is_real_geometry_kernel():
    # Not the copy-skeleton: a real ByteAddressBuffer Load3/Store3 vertex transform.
    # This HLSL was dispatched on an Intel HD 4600 and verified bit-exact vs CPU.
    hlsl = Dx11Backend().generate_vertex_transform_hlsl()
    assert "ByteAddressBuffer   inPos  : register(t0);" in hlsl
    assert "RWByteAddressBuffer outPos : register(u0);" in hlsl
    assert "row_major float4x4 M;" in hlsl
    assert "inPos.Load3(i * 12)" in hlsl          # tight 12-byte stride read
    assert "outPos.Store3(i * 12," in hlsl        # tight 12-byte stride write
    assert "mul(M, float4(p, 1.0f))" in hlsl      # actual transform (not a copy)
    assert "[numthreads(64, 1, 1)]" in hlsl


def test_knu_vertex_transform_glyph_selects_geometry_kernel():
    # The KNU stream DRIVES kernel selection: a G_VERTEX_TRANSFORM word makes the lowering
    # emit the real geometry kernel instead of the tensor-copy skeleton.
    knus = [encode_knu("G_VERTEX_TRANSFORM", payload=0)]
    hlsl = lower_khlnary_to_hlsl(knus, {})
    assert hlsl == Dx11Backend().generate_vertex_transform_hlsl()
    assert "inPos.Load3(i * 12)" in hlsl
    assert "RWStructuredBuffer<float> output_buf" not in hlsl  # not the copy skeleton


def test_knu_vertex_skin_glyph_selects_skinning_kernel_and_wins():
    # G_VERTEX_SKIN selects the richer skinning kernel; it wins even if a transform glyph
    # is also present.
    knus = [encode_knu("G_VERTEX_TRANSFORM", payload=0),
            encode_knu("G_VERTEX_SKIN", payload=0)]
    hlsl = lower_khlnary_to_hlsl(knus, {})
    assert hlsl == Dx11Backend().generate_skinning_hlsl()
    assert "StructuredBuffer<float4x4>  skinMatrices : register(t4);" in hlsl
    assert "float3 sn = mul((float3x3)m, nrm);" in hlsl  # normal skinning
    assert "outVerts.Store3(b + 12, asuint(sn));" in hlsl


def test_knu_matmul_glyph_selects_gemm_kernel():
    # The G_MATMUL glyph selects a real GEMM compute kernel (C = A @ B), not the copy skeleton.
    # This HLSL was dispatched on an Intel HD 4600 with a real gpt2 weight (scale-normalized
    # err 1.0e-06 vs numpy).
    hlsl = lower_khlnary_to_hlsl([encode_knu("G_MATMUL", payload=0)], {})
    assert hlsl == Dx11Backend().generate_matmul_hlsl()
    assert "StructuredBuffer<float>   A : register(t0);" in hlsl
    assert "RWStructuredBuffer<float> C : register(u0);" in hlsl
    assert "cbuffer GemmCB : register(b0) { uint M; uint N; uint K; uint _pad; };" in hlsl
    assert "acc += A[row * K + k] * B[k * N + col];" in hlsl   # the dot product
    assert "[numthreads(16, 16, 1)]" in hlsl
    assert "RWStructuredBuffer<float> output_buf" not in hlsl  # not the copy skeleton


def test_lower_missing_bin_file_id_raises():
    knus = [encode_knu("G_LOAD_BIN_TENSOR", payload=(1 << 4) | 0)]
    try:
        lower_khlnary_to_hlsl(knus, {})  # bin_file_id 1 not in table
        assert False, "expected KeyError"
    except KeyError:
        pass
