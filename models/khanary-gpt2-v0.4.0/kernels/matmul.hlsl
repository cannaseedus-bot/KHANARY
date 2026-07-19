StructuredBuffer<float>   A : register(t0);   // [M,K] row-major
StructuredBuffer<float>   B : register(t1);   // [K,N] row-major
RWStructuredBuffer<float> C : register(u0);   // [M,N] row-major
cbuffer GemmCB : register(b0) { uint M; uint N; uint K; uint _pad; };
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.y;
    uint col = tid.x;
    if (row >= M || col >= N) { return; }
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
