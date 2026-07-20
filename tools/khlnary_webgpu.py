"""KHΛNARY -> WebGPU lowering helpers and module-oriented backend."""

from __future__ import annotations

from typing import List, Mapping

from tools.khlnary_compiler import KhlnaryModule
from tools.khlnary_encoder import GLYPH_IDS, decode_knu


class WebGpuBackend:
    def generate_wgsl_shader(self, module: KhlnaryModule) -> str:
        bindings = [
            f"@group(0) @binding({idx}) var<storage, read> {tensor.ptr_name} : array<f16>;"
            for idx, tensor in enumerate(module.tensors)
        ]
        prefix = "\n".join(bindings)
        return (
            prefix
            + "\n\n@group(0) @binding(100) var<storage, read> input_buffer : array<f16>;\n"
            + "@group(0) @binding(101) var<storage, read_write> output_buffer : array<f16>;\n"
            + "struct Constants { batch_size: u32, seq_len: u32, hidden_size: u32 };\n"
            + "@group(0) @binding(102) var<uniform> constants : Constants;\n"
            + "@compute @workgroup_size(256)\n"
            + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            + "  let idx = gid.x;\n"
            + "  let n = constants.batch_size * constants.seq_len * constants.hidden_size;\n"
            + "  if (idx >= n) { return; }\n"
            + "  output_buffer[idx] = input_buffer[idx];\n"
            + "}\n"
        )

    def generate_vertex_transform_wgsl(self) -> str:
        """Real geometry kernel (WGSL mirror of Dx11Backend.generate_vertex_transform_hlsl):
        apply a 4x4 to each vertex of a tight f32x3 position stream. WGSL mat4x4<f32> is
        column-major, so the host uploads M transposed vs the HLSL row_major upload to get
        identical numbers — same math, backend-native layout."""
        return (
            "struct Xform { M : mat4x4<f32>, vertexCount : u32 };\n"
            "@group(0) @binding(0) var<uniform> xf : Xform;\n"
            "@group(0) @binding(1) var<storage, read>       inPos  : array<f32>;  // tight f32x3\n"
            "@group(0) @binding(2) var<storage, read_write> outPos : array<f32>;\n"
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let i = gid.x;\n"
            "  if (i >= xf.vertexCount) { return; }\n"
            "  let b = i * 3u;\n"
            "  let p = vec3<f32>(inPos[b], inPos[b + 1u], inPos[b + 2u]);\n"
            "  let q = (xf.M * vec4<f32>(p, 1.0)).xyz;   // manifold transform\n"
            "  outPos[b] = q.x; outPos[b + 1u] = q.y; outPos[b + 2u] = q.z;\n"
            "}\n"
        )

    def generate_skinning_wgsl(self) -> str:
        """Weighted joint skinning (WGSL mirror of Dx11Backend.generate_skinning_hlsl):
        position + normal transformed by a blend of 4x4 skin matrices, written to a tight
        f32x6 (pos+normal) output stream."""
        return (
            "struct SkinU { positionStride : u32, positionOffset : u32,\n"
            "               normalStride : u32, normalOffset : u32, vertexCount : u32 };\n"
            "@group(0) @binding(0) var<uniform> u : SkinU;\n"
            "@group(0) @binding(1) var<storage, read>       positions    : array<f32>;\n"
            "@group(0) @binding(2) var<storage, read>       normals      : array<f32>;\n"
            "@group(0) @binding(3) var<storage, read>       weights      : array<f32>;  // 4/vertex\n"
            "@group(0) @binding(4) var<storage, read>       joints       : array<u32>;  // 4/vertex\n"
            "@group(0) @binding(5) var<storage, read>       skinMatrices : array<mat4x4<f32>>;\n"
            "@group(0) @binding(6) var<storage, read_write> outVerts     : array<f32>;  // 6/vertex\n"
            "fn skinMatrix(i : u32) -> mat4x4<f32> {\n"
            "  let jb = i * 4u;\n"
            "  var m = skinMatrices[joints[jb]]      * weights[jb];\n"
            "  m = m + skinMatrices[joints[jb + 1u]] * weights[jb + 1u];\n"
            "  m = m + skinMatrices[joints[jb + 2u]] * weights[jb + 2u];\n"
            "  m = m + skinMatrices[joints[jb + 3u]] * weights[jb + 3u];\n"
            "  return m;\n"
            "}\n"
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let i = gid.x;\n"
            "  if (i >= u.vertexCount) { return; }\n"
            "  let pb = i * u.positionStride + u.positionOffset;\n"
            "  let p = vec3<f32>(positions[pb], positions[pb + 1u], positions[pb + 2u]);\n"
            "  let nb = i * u.normalStride + u.normalOffset;\n"
            "  let nrm = vec3<f32>(normals[nb], normals[nb + 1u], normals[nb + 2u]);\n"
            "  let m = skinMatrix(i);\n"
            "  let sp = (m * vec4<f32>(p, 1.0)).xyz;\n"
            "  let m3 = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);\n"
            "  let sn = m3 * nrm;\n"
            "  let ob = i * 6u;\n"
            "  outVerts[ob] = sp.x; outVerts[ob + 1u] = sp.y; outVerts[ob + 2u] = sp.z;\n"
            "  outVerts[ob + 3u] = sn.x; outVerts[ob + 4u] = sn.y; outVerts[ob + 5u] = sn.z;\n"
            "}\n"
        )

    def generate_matmul_wgsl(self) -> str:
        """WGSL mirror of Dx11Backend.generate_matmul_hlsl: dense GEMM
        C[M,N] = A[M,K] @ B[K,N], row-major f32, one thread per output element."""
        return (
            "@group(0) @binding(0) var<storage, read>       A : array<f32>;  // [M,K]\n"
            "@group(0) @binding(1) var<storage, read>       B : array<f32>;  // [K,N]\n"
            "@group(0) @binding(2) var<storage, read_write> C : array<f32>;  // [M,N]\n"
            "struct GemmU { M : u32, N : u32, K : u32 };\n"
            "@group(0) @binding(3) var<uniform> g : GemmU;\n"
            "@compute @workgroup_size(16, 16)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let row = gid.y;\n"
            "  let col = gid.x;\n"
            "  if (row >= g.M || col >= g.N) { return; }\n"
            "  var acc = 0.0;\n"
            "  for (var k = 0u; k < g.K; k = k + 1u) {\n"
            "    acc = acc + A[row * g.K + k] * B[k * g.N + col];\n"
            "  }\n"
            "  C[row * g.N + col] = acc;\n"
            "}\n"
        )

    def generate_attention_wgsl(self) -> str:
        """WGSL mirror of Dx11Backend.generate_attention_hlsl: causal multi-head attention fwd,
        one workgroup per head, one thread per query position."""
        return (
            "struct AttnFwdParams { seq_len : u32, n_embd : u32, head_dim : u32, scale : f32 };\n"
            "@group(0) @binding(0) var<uniform> P : AttnFwdParams;\n"
            "@group(0) @binding(1) var<storage, read>       qkv      : array<f32>;  // [S,3E]\n"
            "@group(0) @binding(2) var<storage, read_write> attn_out : array<f32>;  // [S,E]\n"
            "@group(0) @binding(3) var<storage, read_write> P_buf    : array<f32>;  // [H,S,S]\n"
            "@compute @workgroup_size(128)\n"
            "fn main(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {\n"
            "  let h = gid.x; let i = lid.x;\n"
            "  let S = P.seq_len; let E = P.n_embd; let D = P.head_dim;\n"
            "  if (i >= S) { return; }\n"
            "  let p_row = h * S * S + i * S;\n"
            "  var mx = -1e30;\n"
            "  for (var j = 0u; j <= i; j = j + 1u) {\n"
            "    var dot = 0.0;\n"
            "    for (var d = 0u; d < D; d = d + 1u) {\n"
            "      dot = dot + qkv[i*3u*E + h*D + d] * qkv[j*3u*E + E + h*D + d];\n"
            "    }\n"
            "    dot = dot * P.scale;\n"
            "    P_buf[p_row + j] = dot;\n"
            "    if (dot > mx) { mx = dot; }\n"
            "  }\n"
            "  for (var j = i + 1u; j < S; j = j + 1u) { P_buf[p_row + j] = -1e30; }\n"
            "  var sum_e = 0.0;\n"
            "  for (var j = 0u; j <= i; j = j + 1u) {\n"
            "    let e = exp(P_buf[p_row + j] - mx);\n"
            "    P_buf[p_row + j] = e;\n"
            "    sum_e = sum_e + e;\n"
            "  }\n"
            "  for (var j = 0u; j <= i; j = j + 1u) { P_buf[p_row + j] = P_buf[p_row + j] / sum_e; }\n"
            "  for (var j = i + 1u; j < S; j = j + 1u) { P_buf[p_row + j] = 0.0; }\n"
            "  for (var d = 0u; d < D; d = d + 1u) {\n"
            "    var acc = 0.0;\n"
            "    for (var j = 0u; j <= i; j = j + 1u) {\n"
            "      acc = acc + P_buf[p_row + j] * qkv[j*3u*E + 2u*E + h*D + d];\n"
            "    }\n"
            "    attn_out[i*E + h*D + d] = acc;\n"
            "  }\n"
            "}\n"
        )

    def generate_layernorm_wgsl(self) -> str:
        """WGSL mirror of Dx11Backend.generate_layernorm_hlsl (workgroup reduction for mean/var)."""
        return (
            "struct LNFwdParams { n_embd : u32, seq_len : u32, eps : f32 };\n"
            "@group(0) @binding(0) var<uniform> P : LNFwdParams;\n"
            "@group(0) @binding(1) var<storage, read>       x_in    : array<f32>;\n"
            "@group(0) @binding(2) var<storage, read>       gamma   : array<f32>;\n"
            "@group(0) @binding(3) var<storage, read>       beta    : array<f32>;\n"
            "@group(0) @binding(4) var<storage, read_write> y_out   : array<f32>;\n"
            "@group(0) @binding(5) var<storage, read_write> xhat    : array<f32>;\n"
            "@group(0) @binding(6) var<storage, read_write> inv_std : array<f32>;\n"
            "var<workgroup> gs_s  : array<f32, 256>;\n"
            "var<workgroup> gs_s2 : array<f32, 256>;\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {\n"
            "  let s = gid.x; let tid = lid.x; let base = s * P.n_embd;\n"
            "  var lsum = 0.0; var lsum2 = 0.0;\n"
            "  for (var i = tid; i < P.n_embd; i = i + 256u) { let v = x_in[base + i]; lsum = lsum + v; lsum2 = lsum2 + v*v; }\n"
            "  gs_s[tid] = lsum; gs_s2[tid] = lsum2;\n"
            "  workgroupBarrier();\n"
            "  for (var stride = 128u; stride >= 1u; stride = stride >> 1u) {\n"
            "    if (tid < stride) { gs_s[tid] = gs_s[tid] + gs_s[tid+stride]; gs_s2[tid] = gs_s2[tid] + gs_s2[tid+stride]; }\n"
            "    workgroupBarrier();\n"
            "  }\n"
            "  let mean = gs_s[0] / f32(P.n_embd);\n"
            "  let varr = gs_s2[0] / f32(P.n_embd) - mean * mean;\n"
            "  let istd = 1.0 / sqrt(varr + P.eps);\n"
            "  if (tid == 0u) { inv_std[s] = istd; }\n"
            "  for (var i = tid; i < P.n_embd; i = i + 256u) {\n"
            "    let xh = (x_in[base + i] - mean) * istd;\n"
            "    xhat[base + i] = xh; y_out[base + i] = gamma[i] * xh + beta[i];\n"
            "  }\n"
            "}\n"
        )

    def generate_gelu_wgsl(self) -> str:
        """WGSL mirror of Dx11Backend.generate_gelu_hlsl (tanh-approx GELU)."""
        return (
            "struct GeluParams { numel : u32, x_in_offset : u32 };\n"
            "@group(0) @binding(0) var<uniform> P : GeluParams;\n"
            "@group(0) @binding(1) var<storage, read>       x_in : array<f32>;\n"
            "@group(0) @binding(2) var<storage, read_write> y    : array<f32>;\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) tid : vec3<u32>) {\n"
            "  let i = tid.x; if (i >= P.numel) { return; }\n"
            "  let x = x_in[i + P.x_in_offset];\n"
            "  let k = 0.7978845608 * (x + 0.044715 * x * x * x);\n"
            "  let kc = clamp(k, -10.0, 10.0);\n"
            "  y[i] = 0.5 * x * (1.0 + tanh(kc));\n"
            "}\n"
        )

    def generate_embed_wgsl(self) -> str:
        """WGSL mirror of Dx11Backend.generate_embed_hlsl (token + positional embedding)."""
        return (
            "struct EmbedParams { seq_len : u32, n_embd : u32 };\n"
            "@group(0) @binding(0) var<uniform> P : EmbedParams;\n"
            "@group(0) @binding(1) var<storage, read>       tokens : array<i32>;\n"
            "@group(0) @binding(2) var<storage, read>       wte    : array<f32>;\n"
            "@group(0) @binding(3) var<storage, read>       wpe    : array<f32>;\n"
            "@group(0) @binding(4) var<storage, read_write> h_out  : array<f32>;\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {\n"
            "  let i = gid.x; if (i >= P.seq_len) { return; }\n"
            "  let tok = u32(tokens[i]);\n"
            "  for (var d = lid.x; d < P.n_embd; d = d + 256u) {\n"
            "    h_out[i * P.n_embd + d] = wte[tok * P.n_embd + d] + wpe[i * P.n_embd + d];\n"
            "  }\n"
            "}\n"
        )

    def generate_add_wgsl(self) -> str:
        """WGSL mirror of Dx11Backend.generate_add_hlsl (elementwise residual add)."""
        return (
            "struct AddParams { len : u32 };\n"
            "@group(0) @binding(0) var<uniform> P : AddParams;\n"
            "@group(0) @binding(1) var<storage, read_write> y : array<f32>;\n"
            "@group(0) @binding(2) var<storage, read>       r : array<f32>;\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) t : vec3<u32>) {\n"
            "  let i = t.x; if (i >= P.len) { return; }\n"
            "  y[i] = y[i] + r[i];\n"
            "}\n"
        )

    def generate_add_bias_wgsl(self) -> str:
        """WGSL mirror of Dx11Backend.generate_add_bias_hlsl (broadcast bias add)."""
        return (
            "struct BiasParams { rows : u32, N : u32 };\n"
            "@group(0) @binding(0) var<uniform> P : BiasParams;\n"
            "@group(0) @binding(1) var<storage, read_write> y : array<f32>;\n"
            "@group(0) @binding(2) var<storage, read>       b : array<f32>;\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) t : vec3<u32>) {\n"
            "  let i = t.x; if (i >= P.rows * P.N) { return; }\n"
            "  y[i] = y[i] + b[i % P.N];\n"
            "}\n"
        )

    @staticmethod
    def generate_javascript_loader() -> str:
        return """
async function loadSTBBuffer(device, url) {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const gpuBuffer = device.createBuffer({
    size: arrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gpuBuffer, 0, arrayBuffer);
  return gpuBuffer;
}

async function createKHlnaryPipeline(device, shaderCode) {
  const shaderModule = device.createShaderModule({ code: shaderCode });
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
  });
  return pipeline;
}
""".strip()


def lower_khlnary_to_wgsl(knus: List[int], bin_file_table: Mapping[int, Mapping[str, str]]) -> str:
    # Geometry glyphs in the KNU stream select a real geometry kernel (co-equal with the HLSL
    # backend in lower_khlnary_to_hlsl). G_VERTEX_SKIN takes precedence over G_VERTEX_TRANSFORM.
    glyphs = {decode_knu(w)["glyph_id"] for w in knus}
    if GLYPH_IDS["G_VERTEX_SKIN"] in glyphs:
        return WebGpuBackend().generate_skinning_wgsl()
    if GLYPH_IDS["G_VERTEX_TRANSFORM"] in glyphs:
        return WebGpuBackend().generate_vertex_transform_wgsl()
    if GLYPH_IDS["G_MATMUL"] in glyphs:
        return WebGpuBackend().generate_matmul_wgsl()
    if GLYPH_IDS["G_ATTENTION"] in glyphs:
        return WebGpuBackend().generate_attention_wgsl()
    if GLYPH_IDS["G_LAYERNORM"] in glyphs:
        return WebGpuBackend().generate_layernorm_wgsl()
    if GLYPH_IDS["G_GELU"] in glyphs:
        return WebGpuBackend().generate_gelu_wgsl()
    if GLYPH_IDS["G_EMBED"] in glyphs:
        return WebGpuBackend().generate_embed_wgsl()
    if GLYPH_IDS["G_ADD_BIAS"] in glyphs:
        return WebGpuBackend().generate_add_bias_wgsl()
    if GLYPH_IDS["G_ADD"] in glyphs:
        return WebGpuBackend().generate_add_wgsl()

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

    wgsl_buffers = [
        f"@group(0) @binding({idx}) var<storage, read> t_{bin_id}_{tid} : array<f16>;"
        for idx, bin_id, tid in bindings
    ]
    return "\n".join(wgsl_buffers) + """

@group(0) @binding(10) var<storage, read> input_buf : array<f16>;
@group(0) @binding(11) var<storage, read_write> output_buf : array<f16>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  output_buf[idx] = input_buf[idx];
}
"""


def webgpu_js_loader(bin_file_table: Mapping[int, Mapping[str, str]]) -> str:
    _ = bin_file_table
    return "\n".join(
        [
            "async function loadStbToBuffer(device, url) {",
            "  const resp = await fetch(url);",
            "  const buf = await resp.arrayBuffer();",
            "  const gpuBuf = device.createBuffer({",
            "    size: buf.byteLength,",
            "    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,",
            "  });",
            "  device.queue.writeBuffer(gpuBuf, 0, buf);",
            "  return gpuBuf;",
            "}",
        ]
    )


__all__ = ["lower_khlnary_to_wgsl", "webgpu_js_loader", "WebGpuBackend"]
