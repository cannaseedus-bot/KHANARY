// gpt2_adam_wte.hlsl — GPU Adam update for wte (embedding table).
// Dispatch(vocab_size, 1, 1), numthreads(n_embd, 1, 1)
// V=50257 < D3D11 group limit (65535). E=768 threads per group.
// One group per vocab token; each thread handles one embedding dimension.
// Gradient accumulator is zeroed here — no separate clear needed.

cbuffer AdamParams : register(b0) {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_corr1;   // 1 / (1 - beta1^t)
    float bias_corr2;   // 1 / (1 - beta2^t)
    uint  numel;        // V * E
};

RWStructuredBuffer<float> weights : register(u0);  // [V, E] wte weights
RWStructuredBuffer<float> grads   : register(u1);  // [V, E] gradient accumulator
RWStructuredBuffer<float> m       : register(u2);  // [V, E] first moment
RWStructuredBuffer<float> v       : register(u3);  // [V, E] second moment

[numthreads(768, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint i = gid.x * 768u + tid.x;
    if (i >= numel) return;

    float g = grads[i];
    g += weight_decay * weights[i];

    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    const float m_hat = mi * bias_corr1;
    const float v_hat = vi * bias_corr2;

    weights[i] -= lr * m_hat / (sqrt(v_hat) + eps);

    grads[i] = 0.0f;
}
