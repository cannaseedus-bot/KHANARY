// gpt2_embed_bwd.hlsl — gradient scatter for embeddings
// One group per embedding dimension d (gid.x = d), one thread.
// Thread d loops over all seq positions — no race (each d is unique per group).
//
// Dispatch(n_embd, 1, 1)  numthreads(1, 1, 1)

cbuffer EmbedBwdParams : register(b0) {
    uint seq_len;
    uint n_embd;
    uint2 pad;
};

StructuredBuffer<int>     tokens  : register(t0);  // [S]
StructuredBuffer<float>   dh      : register(t1);  // [S, E]
RWStructuredBuffer<float> d_wte   : register(u0);  // [V, E]  accumulate +=
RWStructuredBuffer<float> d_wpe   : register(u1);  // [ctx, E]

[numthreads(1, 1, 1)]
void CSMain(uint3 gid : SV_GroupID) {
    const uint d = gid.x;
    if (d >= n_embd) return;
    for (uint i = 0; i < seq_len; ++i) {
        float g = dh[i * n_embd + d];
        d_wpe[i * n_embd + d] += g;
        d_wte[(uint)tokens[i] * n_embd + d] += g;  // safe: only thread d writes column d
    }
}
