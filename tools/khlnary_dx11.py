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
        """Dense GEMM C[M,N] = A[M,K] @ B[K,N], row-major float32, with 16x16 GROUPSHARED tiling
        (shared-memory blocking): each tile of A and B is loaded once into groupshared and reused
        by all 256 threads, instead of re-fetching B from global memory per k. Bounds-guarded for
        non-multiple-of-16 dims. cs_5_0. Drop-in for the naive kernel (same buffers/cbuffer/dispatch)."""
        return (
            "#define TS 16\n"
            "StructuredBuffer<float>   A : register(t0);   // [M,K] row-major\n"
            "StructuredBuffer<float>   B : register(t1);   // [K,N] row-major\n"
            "RWStructuredBuffer<float> C : register(u0);   // [M,N] row-major\n"
            "cbuffer GemmCB : register(b0) { uint M; uint N; uint K; uint _pad; };\n"
            "groupshared float As[TS][TS];\n"
            "groupshared float Bs[TS][TS];\n"
            "[numthreads(TS, TS, 1)]\n"
            "void main(uint3 dtid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {\n"
            "    uint row = dtid.y, col = dtid.x;\n"
            "    float acc = 0.0f;\n"
            "    uint nT = (K + TS - 1) / TS;\n"
            "    for (uint t = 0; t < nT; ++t) {\n"
            "        uint aC = t * TS + lid.x, bR = t * TS + lid.y;\n"
            "        As[lid.y][lid.x] = (row < M && aC < K) ? A[row * K + aC] : 0.0f;\n"
            "        Bs[lid.y][lid.x] = (bR < K && col < N) ? B[bR * N + col] : 0.0f;\n"
            "        GroupMemoryBarrierWithGroupSync();\n"
            "        [unroll] for (uint k = 0; k < TS; ++k) acc += As[lid.y][k] * Bs[k][lid.x];\n"
            "        GroupMemoryBarrierWithGroupSync();\n"
            "    }\n"
            "    if (row < M && col < N) C[row * N + col] = acc;\n"
            "}\n"
        )

    def generate_attention_hlsl(self) -> str:
        """Causal multi-head attention forward, promoted near-verbatim from the gpt2 trainer's
        gpt2_attn_fwd.hlsl. Dispatch(n_head,1,1), numthreads(128,1,1): group = head h, thread =
        query position i. Reads qkv[S,3E] (Q|K|V interleaved), writes attn_out[S,E] and the
        softmax weights P_buf[H,S,S]. Scaled dot-product + causal mask + max-stable softmax."""
        return (
            "cbuffer AttnFwdParams : register(b0) {\n"
            "    uint  seq_len;\n"
            "    uint  n_embd;   // E\n"
            "    uint  head_dim; // D = E/H\n"
            "    float scale;    // 1/sqrt(D)\n"
            "};\n"
            "StructuredBuffer<float>   qkv      : register(t0);  // [S, 3E]\n"
            "RWStructuredBuffer<float> attn_out : register(u0);  // [S, E]\n"
            "RWStructuredBuffer<float> P_buf    : register(u1);  // [H, S, S]\n"
            "[numthreads(128, 1, 1)]\n"
            "void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {\n"
            "    const uint h = gid.x;\n"
            "    const uint i = lid.x;\n"
            "    const uint S = seq_len;\n"
            "    const uint E = n_embd;\n"
            "    const uint D = head_dim;\n"
            "    if (i >= S) return;\n"
            "    const uint p_row = h * S*S + i * S;\n"
            "    // scores = Q[i] . K[j] * scale  (causal: j <= i)\n"
            "    float mx = -1e30f;\n"
            "    for (uint j = 0; j <= i; ++j) {\n"
            "        float dot = 0.f;\n"
            "        for (uint d = 0; d < D; ++d)\n"
            "            dot += qkv[i*3*E + h*D + d] * qkv[j*3*E + E + h*D + d];\n"
            "        dot *= scale;\n"
            "        P_buf[p_row + j] = dot;\n"
            "        if (dot > mx) mx = dot;\n"
            "    }\n"
            "    for (uint j = i+1; j < S; ++j) P_buf[p_row + j] = -1e30f;\n"
            "    // softmax\n"
            "    float sum_e = 0.f;\n"
            "    for (uint j = 0; j <= i; ++j) {\n"
            "        float e = exp(P_buf[p_row + j] - mx);\n"
            "        P_buf[p_row + j] = e;\n"
            "        sum_e += e;\n"
            "    }\n"
            "    for (uint j = 0; j <= i; ++j) P_buf[p_row + j] /= sum_e;\n"
            "    for (uint j = i+1; j < S; ++j) P_buf[p_row + j] = 0.f;\n"
            "    // attn_out[i, h*D+d] = sum_j P[i,j] * V[j,d]\n"
            "    for (uint d = 0; d < D; ++d) {\n"
            "        float acc = 0.f;\n"
            "        for (uint j = 0; j <= i; ++j)\n"
            "            acc += P_buf[p_row + j] * qkv[j*3*E + 2*E + h*D + d];\n"
            "        attn_out[i*E + h*D + d] = acc;\n"
            "    }\n"
            "}\n"
        )

    def generate_layernorm_hlsl(self) -> str:
        """LayerNorm forward, promoted from gpt2_layernorm_fwd.hlsl. Dispatch(seq_len,1,1),
        numthreads(256): groupshared parallel reduction for mean/var per row, then
        y = gamma * (x-mean)*inv_std + beta. Also saves xhat + inv_std (backward)."""
        return (
            "cbuffer LNFwdParams : register(b0) { uint n_embd; uint seq_len; float eps; uint pad; };\n"
            "StructuredBuffer<float>   x_in   : register(t0);  // [S, E]\n"
            "StructuredBuffer<float>   gamma  : register(t1);  // [E]\n"
            "StructuredBuffer<float>   beta   : register(t2);  // [E]\n"
            "RWStructuredBuffer<float> y_out  : register(u0);  // [S, E]\n"
            "RWStructuredBuffer<float> xhat   : register(u1);  // [S, E]\n"
            "RWStructuredBuffer<float> inv_std: register(u2);  // [S]\n"
            "groupshared float gs_s[256];\n"
            "groupshared float gs_s2[256];\n"
            "[numthreads(256, 1, 1)]\n"
            "void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {\n"
            "    const uint s = gid.x; const uint tid = lid.x; const uint base = s * n_embd;\n"
            "    float lsum = 0.f, lsum2 = 0.f;\n"
            "    for (uint i = tid; i < n_embd; i += 256) { float v = x_in[base + i]; lsum += v; lsum2 += v*v; }\n"
            "    gs_s[tid] = lsum; gs_s2[tid] = lsum2;\n"
            "    GroupMemoryBarrierWithGroupSync();\n"
            "    [unroll] for (uint stride = 128; stride >= 1; stride >>= 1) {\n"
            "        if (tid < stride) { gs_s[tid] += gs_s[tid+stride]; gs_s2[tid] += gs_s2[tid+stride]; }\n"
            "        GroupMemoryBarrierWithGroupSync();\n"
            "    }\n"
            "    const float mean = gs_s[0] / (float)n_embd;\n"
            "    const float var  = gs_s2[0] / (float)n_embd - mean * mean;\n"
            "    const float istd = 1.0f / sqrt(var + eps);\n"
            "    if (tid == 0) inv_std[s] = istd;\n"
            "    for (uint i = tid; i < n_embd; i += 256) {\n"
            "        float xh = (x_in[base + i] - mean) * istd;\n"
            "        xhat[base + i] = xh; y_out[base + i] = gamma[i] * xh + beta[i];\n"
            "    }\n"
            "}\n"
        )

    def generate_gelu_hlsl(self) -> str:
        """GELU forward (tanh approx), promoted from gpt2_gelu_fwd.hlsl. Includes the HD 4600
        tanh clamp (tanh overflows for |k|>~10 on this driver; saturates at +/-1 anyway)."""
        return (
            "static const float SQRT_2_OVER_PI = 0.7978845608f;\n"
            "static const float COEFF = 0.044715f;\n"
            "cbuffer GeluParams : register(b0) { uint numel; uint x_in_offset; uint2 pad; };\n"
            "StructuredBuffer<float>   x_in : register(t0);\n"
            "RWStructuredBuffer<float> y    : register(u0);\n"
            "[numthreads(256, 1, 1)]\n"
            "void main(uint3 tid : SV_DispatchThreadID) {\n"
            "    const uint i = tid.x; if (i >= numel) return;\n"
            "    const float x = x_in[i + x_in_offset];\n"
            "    const float k = SQRT_2_OVER_PI * (x + COEFF * x * x * x);\n"
            "    const float kc = clamp(k, -10.0f, 10.0f);\n"
            "    y[i] = 0.5f * x * (1.0f + tanh(kc));\n"
            "}\n"
        )

    def generate_embed_hlsl(self) -> str:
        """Token + positional embedding lookup, promoted from gpt2_embed_fwd.hlsl:
        hidden[i,d] = wte[tokens[i], d] + wpe[i, d]. Dispatch(seq_len,1,1), numthreads(256)."""
        return (
            "cbuffer EmbedParams : register(b0) { uint seq_len; uint n_embd; uint2 pad; };\n"
            "StructuredBuffer<int>     tokens : register(t0);  // [S]\n"
            "StructuredBuffer<float>   wte    : register(t1);  // [V, E]\n"
            "StructuredBuffer<float>   wpe    : register(t2);  // [ctx, E]\n"
            "RWStructuredBuffer<float> h_out  : register(u0);  // [S, E]\n"
            "[numthreads(256, 1, 1)]\n"
            "void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {\n"
            "    const uint i = gid.x; if (i >= seq_len) return;\n"
            "    const uint tok = (uint)tokens[i];\n"
            "    for (uint d = lid.x; d < n_embd; d += 256)\n"
            "        h_out[i * n_embd + d] = wte[tok * n_embd + d] + wpe[i * n_embd + d];\n"
            "}\n"
        )

    def generate_add_hlsl(self) -> str:
        """Elementwise add (residual): y[i] += r[i]. The glue between glyphs in a block."""
        return (
            "cbuffer AddParams : register(b0) { uint len; uint3 pad; };\n"
            "RWStructuredBuffer<float> y : register(u0);\n"
            "StructuredBuffer<float>   r : register(t0);\n"
            "[numthreads(256, 1, 1)]\n"
            "void main(uint3 t : SV_DispatchThreadID) {\n"
            "    uint i = t.x; if (i >= len) return;\n"
            "    y[i] = y[i] + r[i];\n"
            "}\n"
        )

    def generate_add_bias_hlsl(self) -> str:
        """Broadcast add (bias): y[i] += b[i % N] over a [rows, N] row-major buffer."""
        return (
            "cbuffer BiasParams : register(b0) { uint rows; uint N; uint2 pad; };\n"
            "RWStructuredBuffer<float> y : register(u0);\n"
            "StructuredBuffer<float>   b : register(t0);\n"
            "[numthreads(256, 1, 1)]\n"
            "void main(uint3 t : SV_DispatchThreadID) {\n"
            "    uint i = t.x; if (i >= rows * N) return;\n"
            "    y[i] = y[i] + b[i % N];\n"
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
    if GLYPH_IDS["G_ATTENTION"] in glyphs:
        return Dx11Backend().generate_attention_hlsl()
    if GLYPH_IDS["G_LAYERNORM"] in glyphs:
        return Dx11Backend().generate_layernorm_hlsl()
    if GLYPH_IDS["G_GELU"] in glyphs:
        return Dx11Backend().generate_gelu_hlsl()
    if GLYPH_IDS["G_EMBED"] in glyphs:
        return Dx11Backend().generate_embed_hlsl()
    if GLYPH_IDS["G_ADD_BIAS"] in glyphs:
        return Dx11Backend().generate_add_bias_hlsl()
    if GLYPH_IDS["G_ADD"] in glyphs:
        return Dx11Backend().generate_add_hlsl()

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
