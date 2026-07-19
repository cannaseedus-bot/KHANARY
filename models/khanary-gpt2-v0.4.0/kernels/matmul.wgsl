@group(0) @binding(0) var<storage, read>       A : array<f32>;  // [M,K]
@group(0) @binding(1) var<storage, read>       B : array<f32>;  // [K,N]
@group(0) @binding(2) var<storage, read_write> C : array<f32>;  // [M,N]
struct GemmU { M : u32, N : u32, K : u32 };
@group(0) @binding(3) var<uniform> g : GemmU;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= g.M || col >= g.N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < g.K; k = k + 1u) {
    acc = acc + A[row * g.K + k] * B[k * g.N + col];
  }
  C[row * g.N + col] = acc;
}
