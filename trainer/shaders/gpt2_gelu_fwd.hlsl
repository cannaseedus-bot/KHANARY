// gpt2_gelu_fwd.hlsl — GELU forward
// GELU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// Dispatch(ceil(numel/256), 1, 1)
// Intel HD 4600: SRV FirstElement is ignored by the driver.
// Offsets are passed as CB constants and applied in the shader instead.

static const float SQRT_2_OVER_PI = 0.7978845608f;
static const float COEFF = 0.044715f;

cbuffer GeluParams : register(b0) {
    uint numel;
    uint x_in_offset;  // element offset into x_in buffer (= mpre for layer l)
    uint2 pad;
};

StructuredBuffer<float>   x_in : register(t0);  // full mlp_pre_buf_ (first=0)
RWStructuredBuffer<float> y    : register(u0);  // GELU output → mlp_gelu_buf_[mgel..]

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= numel) return;

    const float x = x_in[i + x_in_offset];
    const float k = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    // Intel HD 4600: tanh() overflows for |k|>~10 (exp(2k)=INF→NaN or wrong result).
    // Clamp before tanh: tanh saturates at ±1 for |k|>=5, so result is exact.
    const float kc = clamp(k, -10.0f, 10.0f);
    y[i] = 0.5f * x * (1.0f + tanh(kc));
}
