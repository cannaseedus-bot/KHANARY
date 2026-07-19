struct AttnFwdParams { seq_len : u32, n_embd : u32, head_dim : u32, scale : f32 };
@group(0) @binding(0) var<uniform> P : AttnFwdParams;
@group(0) @binding(1) var<storage, read>       qkv      : array<f32>;  // [S,3E]
@group(0) @binding(2) var<storage, read_write> attn_out : array<f32>;  // [S,E]
@group(0) @binding(3) var<storage, read_write> P_buf    : array<f32>;  // [H,S,S]
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let h = gid.x; let i = lid.x;
  let S = P.seq_len; let E = P.n_embd; let D = P.head_dim;
  if (i >= S) { return; }
  let p_row = h * S * S + i * S;
  var mx = -1e30;
  for (var j = 0u; j <= i; j = j + 1u) {
    var dot = 0.0;
    for (var d = 0u; d < D; d = d + 1u) {
      dot = dot + qkv[i*3u*E + h*D + d] * qkv[j*3u*E + E + h*D + d];
    }
    dot = dot * P.scale;
    P_buf[p_row + j] = dot;
    if (dot > mx) { mx = dot; }
  }
  for (var j = i + 1u; j < S; j = j + 1u) { P_buf[p_row + j] = -1e30; }
  var sum_e = 0.0;
  for (var j = 0u; j <= i; j = j + 1u) {
    let e = exp(P_buf[p_row + j] - mx);
    P_buf[p_row + j] = e;
    sum_e = sum_e + e;
  }
  for (var j = 0u; j <= i; j = j + 1u) { P_buf[p_row + j] = P_buf[p_row + j] / sum_e; }
  for (var j = i + 1u; j < S; j = j + 1u) { P_buf[p_row + j] = 0.0; }
  for (var d = 0u; d < D; d = d + 1u) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
      acc = acc + P_buf[p_row + j] * qkv[j*3u*E + 2u*E + h*D + d];
    }
    attn_out[i*E + h*D + d] = acc;
  }
}
