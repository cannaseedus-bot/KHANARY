// gpt2_attn_fwd.hlsl — Causal multi-head attention forward
// Dispatch(n_head, 1, 1)  numthreads(128, 1, 1)
// gid.x = head h,  lid.x = query position i  (i < seq_len)
// Input:  qkv_buf[S, 3E]  (result of QKV linear projection)
// Output: attn_out[S, E]  (= 0 before dispatch; each head writes its D columns)
//         P_buf[H, S, S]  (softmax weights; saved for backward)

cbuffer AttnFwdParams : register(b0) {
    uint  seq_len;
    uint  n_embd;   // E
    uint  head_dim; // D = E/H
    float scale;    // 1/sqrt(D)
};

StructuredBuffer<float>   qkv      : register(t0);  // [S, 3E]
RWStructuredBuffer<float> attn_out : register(u0);  // [S, E]
RWStructuredBuffer<float> P_buf    : register(u1);  // [H, S, S]

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint h   = gid.x;
    const uint i   = lid.x;
    const uint S   = seq_len;
    const uint E   = n_embd;
    const uint D   = head_dim;

    if (i >= S) return;

    const uint qkv_q_off = i * 3*E + h*D;
    const uint p_row      = h * S*S + i * S;

    // --- scores = Q[i] · K[j] * scale  (causal: j <= i) ---
    float mx = -1e30f;
    for (uint j = 0; j <= i; ++j) {
        float dot = 0.f;
        for (uint d = 0; d < D; ++d)
            dot += qkv[i*3*E + h*D + d] * qkv[j*3*E + E + h*D + d];
        dot *= scale;
        P_buf[p_row + j] = dot;
        if (dot > mx) mx = dot;
    }
    for (uint j = i+1; j < S; ++j) P_buf[p_row + j] = -1e30f;

    // --- softmax ---
    float sum_e = 0.f;
    for (uint j = 0; j <= i; ++j) {
        float e = exp(P_buf[p_row + j] - mx);
        P_buf[p_row + j] = e;
        sum_e += e;
    }
    for (uint j = 0; j <= i; ++j) P_buf[p_row + j] /= sum_e;
    for (uint j = i+1; j < S; ++j) P_buf[p_row + j] = 0.f;

    // --- attn output: sum_j P[i,j] * V[j,d] → attn_out[i, h*D+d] ---
    for (uint d = 0; d < D; ++d) {
        float acc = 0.f;
        for (uint j = 0; j <= i; ++j)
            acc += P_buf[p_row + j] * qkv[j*3*E + 2*E + h*D + d];
        attn_out[i*E + h*D + d] = acc;  // each head writes distinct columns — no race
    }
}
