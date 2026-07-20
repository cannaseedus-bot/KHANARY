struct GeluParams { numel : u32, x_in_offset : u32 };
@group(0) @binding(0) var<uniform> P : GeluParams;
@group(0) @binding(1) var<storage, read>       x_in : array<f32>;
@group(0) @binding(2) var<storage, read_write> y    : array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) tid : vec3<u32>) {
  let i = tid.x; if (i >= P.numel) { return; }
  let x = x_in[i + P.x_in_offset];
  let k = 0.7978845608 * (x + 0.044715 * x * x * x);
  let kc = clamp(k, -10.0, 10.0);
  y[i] = 0.5 * x * (1.0 + tanh(kc));
}
