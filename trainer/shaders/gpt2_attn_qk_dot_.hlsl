// gpt2_attn_qk_dot_.hlsl — Causal QK score computation (split attention pass 1/2)
// Writes raw QK logits into P_buf. Gravity bias (or any pre-softmax injection)
// should run after this and before gpt2_attn_softmax_.hlsl.
// Dispatch(n_head, 1, 1)  numthreads(128, 1, 1)
// gid.x = head h,  lid.x = query position i

cbuffer AttnFwdParams : register(b0) {
    uint  seq_len;
    uint  n_embd;   // E
    uint  head_dim; // D = E/H
    float scale;    // 1/sqrt(D)
};

StructuredBuffer<float>   qkv   : register(t0);  // [S, 3E]
RWStructuredBuffer<float> P_buf : register(u0);  // [H, S, S]  — raw logits out

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint h = gid.x;
    const uint i = lid.x;
    const uint S = seq_len;
    const uint E = n_embd;
    const uint D = head_dim;

    if (i >= S) return;

    const uint p_row = h * S*S + i * S;

    // Q[i] · K[j] * scale for causal window
    for (uint j = 0; j <= i; ++j) {
        float dot = 0.f;
        for (uint d = 0; d < D; ++d)
            dot += qkv[i*3*E + h*D + d] * qkv[j*3*E + E + h*D + d];
        P_buf[p_row + j] = dot * scale;
    }
    // Fill future positions with large negative so softmax zeroes them
    for (uint j = i+1; j < S; ++j) P_buf[p_row + j] = -1e30f;
}
