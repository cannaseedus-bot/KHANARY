struct Xform { M : mat4x4<f32>, vertexCount : u32 };
@group(0) @binding(0) var<uniform> xf : Xform;
@group(0) @binding(1) var<storage, read>       inPos  : array<f32>;  // tight f32x3
@group(0) @binding(2) var<storage, read_write> outPos : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= xf.vertexCount) { return; }
  let b = i * 3u;
  let p = vec3<f32>(inPos[b], inPos[b + 1u], inPos[b + 2u]);
  let q = (xf.M * vec4<f32>(p, 1.0)).xyz;   // manifold transform
  outPos[b] = q.x; outPos[b + 1u] = q.y; outPos[b + 2u] = q.z;
}
