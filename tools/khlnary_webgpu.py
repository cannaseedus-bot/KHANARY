"""KHΛNARY -> WebGPU lowering skeleton.

Emits WGSL storage buffer bindings by scanning G_LOAD_BIN_TENSOR glyphs and
provides minimal JS buffer-loader glue.
"""

from __future__ import annotations

from typing import List, Mapping

from tools.khlnary_cuda import GLYPH_IDS, decode_knu


def lower_khlnary_to_wgsl(knus: List[int], bin_file_table: Mapping[int, Mapping[str, str]]) -> str:
    bindings = []
    for w in knus:
        k = decode_knu(w)
        if k["glyph_id"] == GLYPH_IDS["G_LOAD_BIN_TENSOR"]:
            bin_file_id = (k["payload"] >> 4) & 0xF
            tensor_id = (k["payload"] >> 0) & 0xF
            if bin_file_id not in bin_file_table:
                raise KeyError(f"Missing bin_file_id in table: {bin_file_id}")
            binding_idx = len(bindings)
            bindings.append((binding_idx, bin_file_id, tensor_id))

    wgsl_buffers = []
    for idx, bin_id, tid in bindings:
        wgsl_buffers.append(f"@group(0) @binding({idx}) var<storage, read> t_{bin_id}_{tid} : array<f16>;")

    wgsl = "\n".join(wgsl_buffers) + """

@group(0) @binding(10) var<storage, read> input_buf : array<f16>;
@group(0) @binding(11) var<storage, read_write> output_buf : array<f16>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  // TODO: use t_bin_tensor buffers + input_buf to compute output_buf
}
"""
    return wgsl


def webgpu_js_loader(bin_file_table: Mapping[int, Mapping[str, str]]) -> str:
    _ = bin_file_table
    lines = [
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
        "",
        "async function createPipeline(device) {",
        "  // TODO: load WGSL, create pipeline, bind groups using loadStbToBuffer",
        "}",
    ]
    return "\n".join(lines)


__all__ = ["lower_khlnary_to_wgsl", "webgpu_js_loader"]
