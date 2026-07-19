struct SkinU { positionStride : u32, positionOffset : u32,
               normalStride : u32, normalOffset : u32, vertexCount : u32 };
@group(0) @binding(0) var<uniform> u : SkinU;
@group(0) @binding(1) var<storage, read>       positions    : array<f32>;
@group(0) @binding(2) var<storage, read>       normals      : array<f32>;
@group(0) @binding(3) var<storage, read>       weights      : array<f32>;  // 4/vertex
@group(0) @binding(4) var<storage, read>       joints       : array<u32>;  // 4/vertex
@group(0) @binding(5) var<storage, read>       skinMatrices : array<mat4x4<f32>>;
@group(0) @binding(6) var<storage, read_write> outVerts     : array<f32>;  // 6/vertex
fn skinMatrix(i : u32) -> mat4x4<f32> {
  let jb = i * 4u;
  var m = skinMatrices[joints[jb]]      * weights[jb];
  m = m + skinMatrices[joints[jb + 1u]] * weights[jb + 1u];
  m = m + skinMatrices[joints[jb + 2u]] * weights[jb + 2u];
  m = m + skinMatrices[joints[jb + 3u]] * weights[jb + 3u];
  return m;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u.vertexCount) { return; }
  let pb = i * u.positionStride + u.positionOffset;
  let p = vec3<f32>(positions[pb], positions[pb + 1u], positions[pb + 2u]);
  let nb = i * u.normalStride + u.normalOffset;
  let nrm = vec3<f32>(normals[nb], normals[nb + 1u], normals[nb + 2u]);
  let m = skinMatrix(i);
  let sp = (m * vec4<f32>(p, 1.0)).xyz;
  let m3 = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
  let sn = m3 * nrm;
  let ob = i * 6u;
  outVerts[ob] = sp.x; outVerts[ob + 1u] = sp.y; outVerts[ob + 2u] = sp.z;
  outVerts[ob + 3u] = sn.x; outVerts[ob + 4u] = sn.y; outVerts[ob + 5u] = sn.z;
}
