// gpt2_bias_bwd.hlsl — accumulate bias gradient (column-sum of upstream gradient)
// d_bias[j] += sum_{i=0..M-1} dC[i*N + j]
//
// Dispatch(ceil(N/256), 1, 1)  numthreads(256, 1, 1)

cbuffer BiasBwdParams : register(b0) {
    uint M;    // rows (e.g. seq_len)
    uint N;    // cols (bias dimension)
    uint2 pad;
};

StructuredBuffer<float>   dC    : register(t0);  // [M, N] upstream gradient
RWStructuredBuffer<float> dbias : register(u0);  // [N] bias gradient (accumulate +=)

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= N) return;
    float acc = 0.f;
    for (uint i = 0; i < M; ++i)
        acc += dC[i * N + j];
    dbias[j] += acc;
}
