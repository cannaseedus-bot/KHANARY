// gpt2_residual_add.hlsl — elementwise residual addition, cs_5_0
//
// CSMain_add3:  C[i] = A[i] + B[i]   A(t0), B(t1) → C(u0)
// CSMain_addto: C[i] += A[i]          A(t0)         → C(u0)
//
// Dispatch(ceil(numel/256), 1, 1)  numthreads(256, 1, 1)

cbuffer ResAddParams : register(b0) {
    uint numel;
    uint3 pad;
};

StructuredBuffer<float>   A : register(t0);
StructuredBuffer<float>   B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(256, 1, 1)]
void CSMain_add3(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= numel) return;
    C[i] = A[i] + B[i];
}

[numthreads(256, 1, 1)]
void CSMain_addto(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= numel) return;
    C[i] += A[i];
}
