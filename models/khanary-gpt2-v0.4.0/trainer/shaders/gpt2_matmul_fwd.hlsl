// gpt2_matmul_fwd.hlsl — Tiled 16×16 matmul for cs_5_0
// CSMain:       C[M,N]  = A[M,K] @ B[K,N]  + bias[N]   (standard)
// CSMain_transB: C[M,N] = A[M,K] @ B^T[N,K]             (LM head: B=wte[V,E])
// Dispatch(ceil(N/16), ceil(M/16), 1)  numthreads(16,16,1)

cbuffer MMParams : register(b0) {
    uint M;
    uint K;
    uint N;
    uint use_bias;       // 0 or 1
    uint A_row_offset;   // add to row index when reading A (0 for most uses)
    uint B_row_offset;   // add to br index when reading B (0 for most uses)
    uint2 pad;
};

StructuredBuffer<float>   A    : register(t0);
StructuredBuffer<float>   B    : register(t1);
StructuredBuffer<float>   bias : register(t2);
RWStructuredBuffer<float> C    : register(u0);  // accumulate +=

#define TILE 16
groupshared float As[TILE][TILE];
groupshared float Bs[TILE][TILE];

[numthreads(TILE, TILE, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint row = gid.y * TILE + lid.y;
    const uint col = gid.x * TILE + lid.x;
    float acc = 0.f;
    const uint ntiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < ntiles; ++t) {
        uint ac = t * TILE + lid.x; As[lid.y][lid.x] = (row < M && ac < K) ? A[(A_row_offset+row)*K + ac] : 0.f;
        uint br = t * TILE + lid.y; Bs[lid.y][lid.x] = (br  < K && col < N) ? B[(B_row_offset+br)*N + col] : 0.f;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < TILE; ++k) acc += As[lid.y][k] * Bs[k][lid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N)
        C[row*N + col] += acc + (use_bias ? bias[col] : 0.f);
}

// CSMain_transB: C[M,N] = A[M,K] @ B^T  where B is stored as [N,K]
// Used for LM head: A=hidden[1,E], B=wte[V,E], C=logits[1,V]
[numthreads(TILE, TILE, 1)]
void CSMain_transB(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint row = gid.y * TILE + lid.y;  // M
    const uint col = gid.x * TILE + lid.x;  // N
    float acc = 0.f;
    const uint ntiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < ntiles; ++t) {
        uint ac = t * TILE + lid.x; As[lid.y][lid.x] = (row < M && ac < K) ? A[(A_row_offset+row)*K + ac] : 0.f;
        uint bk = t * TILE + lid.y; Bs[lid.y][lid.x] = (bk  < K && col < N) ? B[col*K + bk]              : 0.f;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < TILE; ++k) acc += As[lid.y][k] * Bs[k][lid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N)
        C[row*N + col] = acc;  // overwrite (no bias for LM head)
}
