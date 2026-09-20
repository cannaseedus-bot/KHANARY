// gpt2_attn_softmax_.hlsl — Softmax + V accumulation (split attention pass 2/2)
// Reads P_buf (raw logits, optionally pre-softmax-biased by gravity field),
// applies numerically-stable row-wise softmax in-place, then accumulates V.
// Dispatch(n_head, 1, 1)  numthreads(128, 1, 1)
// gid.x = head h,  lid.x = query position i

cbuffer AttnFwdParams : register(b0) {
    uint  seq_len;
    uint  n_embd;   // E
    uint  head_dim; // D = E/H
    float scale;    // unused here; kept to match AttnFwdParams layout
};

StructuredBuffer<float>   qkv      : register(t0);  // [S, 3E] — for V columns
RWStructuredBuffer<float> P_buf    : register(u0);  // [H, S, S] — logits in, softmax out
RWStructuredBuffer<float> attn_out : register(u1);  // [S, E]   — output (pre-zeroed)

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint h = gid.x;
    const uint i = lid.x;
    const uint S = seq_len;
    const uint E = n_embd;
    const uint D = head_dim;

    if (i >= S) return;

    const uint p_row = h * S*S + i * S;

    // Numerically-stable softmax over causal window [0..i]
    float mx = -1e30f;
    for (uint j = 0; j <= i; ++j)
        if (P_buf[p_row + j] > mx) mx = P_buf[p_row + j];

    float sum_e = 0.f;
    for (uint j = 0; j <= i; ++j) {
        float e = exp(P_buf[p_row + j] - mx);
        P_buf[p_row + j] = e;
        sum_e += e;
    }
    for (uint j = 0; j <= i; ++j) P_buf[p_row + j] /= sum_e;
    for (uint j = i+1; j < S; ++j) P_buf[p_row + j] = 0.f;

    // V accumulation: attn_out[i, h*D+d] = sum_j P[i,j] * V[j,d]
    for (uint d = 0; d < D; ++d) {
        float acc = 0.f;
        for (uint j = 0; j <= i; ++j)
            acc += P_buf[p_row + j] * qkv[j*3*E + 2*E + h*D + d];
        attn_out[i*E + h*D + d] = acc;  // heads write distinct columns — no race
    }
}
