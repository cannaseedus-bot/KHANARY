// gpt2_lm_head_bwd.hlsl — LM head backward: scatter dL/dwte onto the wte gradient buffer.
// Dispatch(vocab_size, 1, 1), numthreads(n_embd, 1, 1)
// V=50257 < D3D11 group limit (65535). E=768 threads per group.
// One group per vocab token, each thread handles one embedding dimension.

cbuffer LMHeadBwdCB : register(b0) {
    uint V;         // vocab_size (50257)
    uint E;         // n_embd     (768)
    uint last_pos;  // flat index of last sequence position = B*S - 1
    uint pad;
};

StructuredBuffer<float>   dlogits : register(t0);  // [V]      dL/dlogits from loss shader
StructuredBuffer<float>   ln_out  : register(t1);  // [B*S, E] final layer-norm output
RWStructuredBuffer<float> g_wte   : register(u0);  // [V, E]   wte gradient accumulator

[numthreads(768, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint v = gid.x;
    const uint d = tid.x;
    if (v >= V || d >= E) return;

    // dL/dwte[v, d] = dlogits[v] * ln_out[last_pos, d]
    g_wte[v * E + d] += dlogits[v] * ln_out[last_pos * E + d];
}
