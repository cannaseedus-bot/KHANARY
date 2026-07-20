struct EmbedParams { seq_len : u32, n_embd : u32 };
@group(0) @binding(0) var<uniform> P : EmbedParams;
@group(0) @binding(1) var<storage, read>       tokens : array<i32>;
@group(0) @binding(2) var<storage, read>       wte    : array<f32>;
@group(0) @binding(3) var<storage, read>       wpe    : array<f32>;
@group(0) @binding(4) var<storage, read_write> h_out  : array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let i = gid.x; if (i >= P.seq_len) { return; }
  let tok = u32(tokens[i]);
  for (var d = lid.x; d < P.n_embd; d = d + 256u) {
    h_out[i * P.n_embd + d] = wte[tok * P.n_embd + d] + wpe[i * P.n_embd + d];
  }
}
