// gpt2_loss.hlsl — Cross-entropy loss + dlogits for LM objective
// Dispatch(1, 1, 1)  numthreads(1024, 1, 1)
// Handles vocab_size up to 1024*50 = 51200 (covers V=50260)

cbuffer LossParams : register(b0) {
    uint  vocab_size;   // V
    uint  target;       // target token id (seq[S-1])
    float inv_batch;    // 1/batch_size for gradient scaling
    uint  pad;
};

StructuredBuffer<float>   logits  : register(t0);  // [V]
RWStructuredBuffer<float> dlogits : register(u0);  // [V]
RWStructuredBuffer<float> loss    : register(u1);  // [1]

groupshared float gs_max[1024];
groupshared float gs_sum[1024];

[numthreads(1024, 1, 1)]
void CSMain(uint3 lid : SV_GroupThreadID) {
    const uint tid = lid.x;
    const uint V   = vocab_size;

    // Parallel max
    float lmax = -1e30f;
    for (uint v = tid; v < V; v += 1024) lmax = max(lmax, logits[v]);
    gs_max[tid] = lmax;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 512; s >= 1; s >>= 1) {
        if (tid < s) gs_max[tid] = max(gs_max[tid], gs_max[tid+s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float gmax = gs_max[0];

    // Parallel sum(exp(logit - max))
    float lsum = 0.f;
    for (uint v = tid; v < V; v += 1024) lsum += exp(logits[v] - gmax);
    gs_sum[tid] = lsum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 512; s >= 1; s >>= 1) {
        if (tid < s) gs_sum[tid] += gs_sum[tid+s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float gsum = gs_sum[0];

    if (tid == 0)
        loss[0] = log(gsum) + gmax - logits[target];

    const float inv_sum = 1.0f / gsum;
    for (uint v = tid; v < V; v += 1024) {
        float sm = exp(logits[v] - gmax) * inv_sum;
        dlogits[v] = (sm - (v == target ? 1.0f : 0.0f)) * inv_batch;
    }
}
