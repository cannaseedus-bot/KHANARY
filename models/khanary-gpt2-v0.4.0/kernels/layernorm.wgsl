struct LNFwdParams { n_embd : u32, seq_len : u32, eps : f32 };
@group(0) @binding(0) var<uniform> P : LNFwdParams;
@group(0) @binding(1) var<storage, read>       x_in    : array<f32>;
@group(0) @binding(2) var<storage, read>       gamma   : array<f32>;
@group(0) @binding(3) var<storage, read>       beta    : array<f32>;
@group(0) @binding(4) var<storage, read_write> y_out   : array<f32>;
@group(0) @binding(5) var<storage, read_write> xhat    : array<f32>;
@group(0) @binding(6) var<storage, read_write> inv_std : array<f32>;
var<workgroup> gs_s  : array<f32, 256>;
var<workgroup> gs_s2 : array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let s = gid.x; let tid = lid.x; let base = s * P.n_embd;
  var lsum = 0.0; var lsum2 = 0.0;
  for (var i = tid; i < P.n_embd; i = i + 256u) { let v = x_in[base + i]; lsum = lsum + v; lsum2 = lsum2 + v*v; }
  gs_s[tid] = lsum; gs_s2[tid] = lsum2;
  workgroupBarrier();
  for (var stride = 128u; stride >= 1u; stride = stride >> 1u) {
    if (tid < stride) { gs_s[tid] = gs_s[tid] + gs_s[tid+stride]; gs_s2[tid] = gs_s2[tid] + gs_s2[tid+stride]; }
    workgroupBarrier();
  }
  let mean = gs_s[0] / f32(P.n_embd);
  let varr = gs_s2[0] / f32(P.n_embd) - mean * mean;
  let istd = 1.0 / sqrt(varr + P.eps);
  if (tid == 0u) { inv_std[s] = istd; }
  for (var i = tid; i < P.n_embd; i = i + 256u) {
    let xh = (x_in[base + i] - mean) * istd;
    xhat[base + i] = xh; y_out[base + i] = gamma[i] * xh + beta[i];
  }
}
