// gpt2_matmul_bwd.hlsl  C[M,N] = A[M,K] @ B[K,N]  backward, cs_5_0
//
// CSMain_dA: dA[M,K] += dC[M,N] @ B^T    Dispatch(ceil(K/16), ceil(M/16), 1)
// CSMain_dB: dB[K,N] += A^T   @ dC[M,N]  Dispatch(ceil(N/16), ceil(K/16), 1)

cbuffer MMBwdParams : register(b0) {
    uint M; uint K; uint N; uint pad;
};

StructuredBuffer<float>   A  : register(t0);
StructuredBuffer<float>   B  : register(t1);
StructuredBuffer<float>   dC : register(t2);
RWStructuredBuffer<float> dA : register(u0);
RWStructuredBuffer<float> dB : register(u1);

#define TILE 16
groupshared float Ts[TILE][TILE];
groupshared float Us[TILE][TILE];

// dA[m,k] = sum_n dC[m,n] * B[k,n]  → dA = dC @ B^T
[numthreads(TILE, TILE, 1)]
void CSMain_dA(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint row = gid.y*TILE + lid.y;  // m
    const uint col = gid.x*TILE + lid.x;  // k
    float acc = 0.f;
    for (uint t = 0; t < (N + TILE-1)/TILE; ++t) {
        // Ts = tile of dC[row, t*TILE+lid.x]
        uint nc = t*TILE + lid.x;
        Ts[lid.y][lid.x] = (row < M && nc < N) ? dC[row*N + nc] : 0.f;
        // Us = tile of B^T, i.e. B[col, t*TILE+lid.y]
        uint nb = t*TILE + lid.y;
        Us[lid.y][lid.x] = (col < K && nb < N) ? B[col*N + nb] : 0.f;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < TILE; ++k) acc += Ts[lid.y][k] * Us[k][lid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < K) dA[row*K + col] += acc;
}

// dB[k,n] = sum_m A[m,k] * dC[m,n]  → dB = A^T @ dC
[numthreads(TILE, TILE, 1)]
void CSMain_dB(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint row = gid.y*TILE + lid.y;  // k
    const uint col = gid.x*TILE + lid.x;  // n
    float acc = 0.f;
    for (uint t = 0; t < (M + TILE-1)/TILE; ++t) {
        // Ts = tile of A^T[row, t*TILE+lid.x] = A[(t*TILE+lid.x)*K + row]
        uint ma = t*TILE + lid.x;
        Ts[lid.y][lid.x] = (ma < M && row < K) ? A[ma*K + row] : 0.f;
        // Us = tile of dC[t*TILE+lid.y, col]
        uint md = t*TILE + lid.y;
        Us[lid.y][lid.x] = (md < M && col < N) ? dC[md*N + col] : 0.f;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < TILE; ++k) acc += Ts[lid.y][k] * Us[k][lid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < K && col < N) dB[row*N + col] += acc;
}
