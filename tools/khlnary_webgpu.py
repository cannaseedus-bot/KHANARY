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
