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
