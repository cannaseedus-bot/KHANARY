@group(0) @binding(0) var<storage, read>       A : array<f32>;  // [M,K]
@group(0) @binding(1) var<storage, read>       B : array<f32>;  // [K,N]
@group(0) @binding(2) var<storage, read_write> C : array<f32>;  // [M,N]
struct GemmU { M : u32, N : u32, K : u32 };
@group(0) @binding(3) var<uniform> g : GemmU;
var<workgroup> As : array<array<f32, 16>, 16>;
var<workgroup> Bs : array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) dtid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = dtid.y; let col = dtid.x;
  var acc = 0.0;
  let nT = (g.K + 15u) / 16u;
  for (var t = 0u; t < nT; t = t + 1u) {
    let aC = t * 16u + lid.x; let bR = t * 16u + lid.y;
    As[lid.y][lid.x] = select(0.0, A[row * g.K + aC], row < g.M && aC < g.K);
    Bs[lid.y][lid.x] = select(0.0, B[bR * g.N + col], bR < g.K && col < g.N);
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) { acc = acc + As[lid.y][k] * Bs[k][lid.x]; }
    workgroupBarrier();
  }
  if (row < g.M && col < g.N) { C[row * g.N + col] = acc; }
}
