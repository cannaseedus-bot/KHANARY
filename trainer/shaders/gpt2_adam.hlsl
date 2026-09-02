// gpt2_adam.hlsl — GPU Adam optimizer update
// Dispatch(X, Y, 1) where X*Y*256 >= numel and X,Y <= 65535
// Use 2D dispatch for large params (e.g. wte with 51M elements).
// cs_5_0 compatible (D3D11 compute)
//
// HD 4600 (Ivy Bridge FL 11.0) fixes:
//   - Explicit packoffset on all cbuffer members: Intel driver cbuffer validation
//     fails silently in CreateComputeShader when mixed scalar/vector types appear
//     without explicit layout.
//   - Bitwise NaN test via asuint() instead of isnan(): Intel Ivy Bridge drivers
//     miscompile the isnan DXBC instruction at O3, returning incorrect results or
//     causing CreateComputeShader to reject the bytecode.
//   - max(0,v_hat) guard on sqrt(): Intel's sqrt can NaN on tiny negatives caused
//     by FP accumulation rounding; clamp avoids silent moment corruption.

cbuffer AdamParams : register(b0) {
    float lr           : packoffset(c0.x);
    float beta1        : packoffset(c0.y);
    float beta2        : packoffset(c0.z);
    float eps          : packoffset(c0.w);
    float weight_decay : packoffset(c1.x);
    float bias_corr1   : packoffset(c1.y);  // 1 / (1 - beta1^t)
    float bias_corr2   : packoffset(c1.z);  // 1 / (1 - beta2^t)
    uint  numel        : packoffset(c1.w);
    uint  stride_x     : packoffset(c2.x);  // dispatch X dim for 2D->1D index
    float grad_clip    : packoffset(c2.y);  // clip bound (fixed=1.0 or KuhulPhysics)
    uint  pad0         : packoffset(c2.z);
    uint  pad1         : packoffset(c2.w);
};

RWStructuredBuffer<float> weights : register(u0);
RWStructuredBuffer<float> grads   : register(u1);
RWStructuredBuffer<float> m       : register(u2);   // first moment
RWStructuredBuffer<float> v       : register(u3);   // second moment

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    // 2D group -> flat element index: i = (gid.y * stride_x + gid.x) * 256 + lid.x
    const uint i = (gid.y * stride_x + gid.x) * 256 + lid.x;
    if (i >= numel) return;

    float g = grads[i];

    // Gradient sanitization + clipping (stability guardrail).
    // The last-position backward can explode through 12 layers on this iGPU;
    // NaN -> 0 (direction lost), Inf/huge -> clamp to +-grad_clip (sign preserved).
    // Bitwise NaN check: exponent bits all 1 and mantissa non-zero.
    // Avoids isnan() DXBC instruction which misfires on Intel Ivy Bridge at O3.
    const uint g_bits = asuint(g);
    const bool g_nan  = (g_bits & 0x7FFFFFFFu) > 0x7F800000u;
    g = g_nan ? 0.0f : clamp(g, -grad_clip, grad_clip);

    // L2 weight decay folded into gradient
    g += weight_decay * weights[i];

    // Moment updates (EMA)
    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    // Bias-corrected update.
    // max(0,v_hat) guards sqrt() against tiny FP negatives from moment accumulation.
    const float m_hat = mi * bias_corr1;
    const float v_hat = vi * bias_corr2;

    weights[i] -= lr * m_hat / (sqrt(max(0.0f, v_hat)) + eps);

    // Zero gradient for next step
    grads[i] = 0.0f;
}
