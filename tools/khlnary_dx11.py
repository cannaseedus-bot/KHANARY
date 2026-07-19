"""KHΛNARY -> D3D11 cs_5_0 (HLSL) backend — sibling of khlnary_webgpu.

For rigs where WebGPU is unavailable (e.g. Intel HD 4600: Dawn/WebGPU blocklisted, but
D3D11 cs_5_0 works). Mirrors WebGpuBackend / lower_khlnary_to_wgsl one-for-one, emitting
StructuredBuffer/RWStructuredBuffer + numthreads instead of WGSL storage bindings. Uses
float32 (STB's real dtype; cs_5_0 has no first-class f16 structured buffers).
"""

from __future__ import annotations

from typing import List, Mapping

from tools.khlnary_compiler import KhlnaryModule
from tools.khlnary_encoder import GLYPH_IDS, decode_knu


class Dx11Backend:
    def generate_hlsl_shader(self, module: KhlnaryModule) -> str:
        bindings = [
            f"StructuredBuffer<float> {tensor.ptr_name} : register(t{idx});"
            for idx, tensor in enumerate(module.tensors)
        ]
        n = len(module.tensors)
        prefix = "\n".join(bindings)
        return (
            (prefix + "\n" if prefix else "")
            + f"StructuredBuffer<float>   input_buffer  : register(t{n});\n"
            + "RWStructuredBuffer<float> output_buffer : register(u0);\n"
            + "cbuffer Constants : register(b0) "
            + "{ uint batch_size; uint seq_len; uint hidden_size; uint _pad; };\n"
            + "[numthreads(256, 1, 1)]\n"
            + "void main(uint3 gid : SV_DispatchThreadID) {\n"
            + "  uint idx = gid.x;\n"
            + "  uint n = batch_size * seq_len * hidden_size;\n"
            + "  if (idx >= n) { return; }\n"
            + "  output_buffer[idx] = input_buffer[idx];\n"
            + "}\n"
        )

    def generate_vertex_transform_hlsl(self) -> str:
        """A REAL geometry kernel (not the copy skeleton): apply a 4x4 transform to each
        vertex position via byte-addressable Load3/Store3 with a tight 12-byte stride.
        This is the cs_5_0 form of the toji 'compute vertex data' / skinning pattern.
        row_major so the uploaded matrix maps to dot(row_i, [p,1]) unambiguously.
        """
        return (
            "cbuffer XformCB : register(b0) {\n"
            "    row_major float4x4 M;\n"
            "    uint vertexCount;\n"
            "    uint3 _pad;\n"
            "};\n"
            "ByteAddressBuffer   inPos  : register(t0);   // tight float3 stream (12 B/vertex)\n"
            "RWByteAddressBuffer outPos : register(u0);\n"
            "[numthreads(64, 1, 1)]\n"
            "void main(uint3 tid : SV_DispatchThreadID) {\n"
            "    uint i = tid.x;\n"
            "    if (i >= vertexCount) { return; }\n"
            "    float3 p = asfloat(inPos.Load3(i * 12));\n"
            "    float3 q = mul(M, float4(p, 1.0f)).xyz;   // manifold transform\n"
            "    outPos.Store3(i * 12, asuint(q));\n"
            "}\n"
        )

    def generate_skinning_hlsl(self) -> str:
        """Full weighted joint skinning: per-vertex position+normal transformed by a blend of
        4x4 skin matrices (position 12 B + normal 12 B -> tight 24 B output stride).
        Byte-addressable Load3/Store3, StructuredBuffer<float4x4> joints. Verified cs_5_0."""
        return (
            "cbuffer VertexUniforms : register(b0) {\n"
            "    uint positionStride; uint positionOffset;\n"
            "    uint normalStride;   uint normalOffset;\n"
            "    uint vertexCount;    float3 packingPadding;  // cbuffer 16-byte alignment\n"
            "};\n"
            "ByteAddressBuffer          positions    : register(t0);\n"
            "ByteAddressBuffer          normals      : register(t1);\n"
            "ByteAddressBuffer          weights      : register(t2);\n"
            "Buffer<uint4>              joints       : register(t3);\n"
            "StructuredBuffer<float4x4>  skinMatrices : register(t4);\n"
            "RWByteAddressBuffer        outVerts     : register(u0);\n"
            "float4x4 SkinMatrix(uint i) {\n"
            "    uint4 j = joints[i];\n"
            "    float4 w = asfloat(weights.Load4(i * 16));\n"
            "    float4x4 m = skinMatrices[j.x] * w.x;\n"
            "    m += skinMatrices[j.y] * w.y;\n"
            "    m += skinMatrices[j.z] * w.z;\n"
            "    m += skinMatrices[j.w] * w.w;\n"
            "    return m;\n"
            "}\n"
            "[numthreads(64, 1, 1)]\n"
            "void main(uint3 tid : SV_DispatchThreadID) {\n"
            "    uint i = tid.x; if (i >= vertexCount) { return; }\n"
            "    float3 p   = asfloat(positions.Load3((i * positionStride + positionOffset) * 4));\n"
            "    float3 nrm = asfloat(normals.Load3((i * normalStride + normalOffset) * 4));\n"
            "    float4x4 m = SkinMatrix(i);\n"
            "    float3 sp = mul(m, float4(p, 1.0f)).xyz;\n"
            "    float3 sn = mul((float3x3)m, nrm);\n"
            "    uint b = i * 24;\n"
            "    outVerts.Store3(b, asuint(sp));\n"
            "    outVerts.Store3(b + 12, asuint(sn));\n"
            "}\n"
        )

    def generate_matmul_hlsl(self) -> str:
        """A real compute kernel: dense GEMM C[M,N] = A[M,K] @ B[K,N], all row-major float32.
        Naive one-thread-per-output-element (2D 16x16 tiles) — correctness-first, the first
        compute glyph beyond the copy skeleton. cs_5_0. This is the path a GGUF/safetensors
        weight would run through once dequantized into an .stb tensor."""
        return (
            "StructuredBuffer<float>   A : register(t0);   // [M,K] row-major\n"
            "StructuredBuffer<float>   B : register(t1);   // [K,N] row-major\n"
            "RWStructuredBuffer<float> C : register(u0);   // [M,N] row-major\n"
            "cbuffer GemmCB : register(b0) { uint M; uint N; uint K; uint _pad; };\n"
            "[numthreads(16, 16, 1)]\n"
            "void main(uint3 tid : SV_DispatchThreadID) {\n"
            "    uint row = tid.y;\n"
            "    uint col = tid.x;\n"
            "    if (row >= M || col >= N) { return; }\n"
            "    float acc = 0.0f;\n"
            "    for (uint k = 0; k < K; ++k) {\n"
            "        acc += A[row * K + k] * B[k * N + col];\n"
            "    }\n"
            "    C[row * N + col] = acc;\n"
            "}\n"
        )

    @staticmethod
    def generate_dispatch_stub() -> str:
        """D3D11 dispatch skeleton (the cs_5_0 counterpart of the WebGPU JS loader)."""
        return "\n".join(
            [
                "// D3D11 cs_5_0 dispatch (mirror of the WebGPU JS loader):",
                "// 1. D3DCompileFromFile(hlsl, entry=\"main\", target=\"cs_5_0\") -> CreateComputeShader",
                "// 2. .stb tensors -> StructuredBuffers: SRVs t0..tN (+ input at tN), output RWStructuredBuffer u0",
                "// 3. Map Constants cbuffer { batch_size, seq_len, hidden_size }",
                "// 4. ctx->Dispatch(ceil(n/256),1,1); CopyResource(staging, output); Map -> read",
            ]
        )


def lower_khlnary_to_hlsl(
    knus: List[int], bin_file_table: Mapping[int, Mapping[str, str]]
) -> str:
    # Geometry glyphs in the KNU stream select a real geometry kernel (not the tensor-copy
    # skeleton). G_VERTEX_SKIN takes precedence over G_VERTEX_TRANSFORM when both appear.
    glyphs = {decode_knu(w)["glyph_id"] for w in knus}
    if GLYPH_IDS["G_VERTEX_SKIN"] in glyphs:
        return Dx11Backend().generate_skinning_hlsl()
    if GLYPH_IDS["G_VERTEX_TRANSFORM"] in glyphs:
        return Dx11Backend().generate_vertex_transform_hlsl()
    if GLYPH_IDS["G_MATMUL"] in glyphs:
        return Dx11Backend().generate_matmul_hlsl()

    bindings = []
    for w in knus:
        k = decode_knu(w)
        if k["glyph_id"] == GLYPH_IDS["G_LOAD_BIN_TENSOR"]:
            bin_file_id = (k["payload"] >> 4) & 0xF
            tensor_id = k["payload"] & 0xF
            if bin_file_id not in bin_file_table:
                raise KeyError(f"Missing bin_file_id in table: {bin_file_id}")
            binding_idx = len(bindings)
            bindings.append((binding_idx, bin_file_id, tensor_id))

    hlsl_buffers = [
        f"StructuredBuffer<float> t_{bin_id}_{tid} : register(t{idx});"
        for idx, bin_id, tid in bindings
    ]
    n = len(bindings)
    return "\n".join(hlsl_buffers) + f"""

StructuredBuffer<float>   input_buf  : register(t{n});
RWStructuredBuffer<float> output_buf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {{
  uint idx = gid.x;
  output_buf[idx] = input_buf[idx];
}}
"""


__all__ = ["lower_khlnary_to_hlsl", "Dx11Backend"]
