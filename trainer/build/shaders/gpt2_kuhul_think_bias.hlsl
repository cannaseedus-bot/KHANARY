// gpt2_kuhul_think_bias.hlsl -- pi-nary K'UHUL attention bias (geodesic arc + 4-bone LBS routing)
//
// THREE STACKED ATTENTION MODIFIERS (applied post-softmax to P_buf):
//
// 1. PI-NARY ARC (geodesic depth)
//    think_depth[t] = GeodesicDistance(wte[think_start], wte[tokens[t]], R)  in [0, pi]
//    arc weight = sin(depth / 2):  0 at boundaries, 1.0 at midpoint (pi/2)
//    Tokens outside <THINK>: depth = 0 -> no arc bias
//
// 2. KUHUL PHYSICS (antigravity scale)
//    antigravity_scale = KuhulPhysicsSolver::GetAntigravityFloat()  [0.1 -> 1.0]
//    grows as training enters stable orbit -- bias automatically strengthens
//
// 3. 4-BONE LINEAR BLEND SKINNING (fold mesh overlap)
//    Each token carries 4 bone IDs + 4 blend weights (sum=1.0).
//    Bone priority: innermost fold -> parent fold -> base hash bone
//    lbs_overlap(i,j) = sum_k sum_m  w_i[k] * w_j[m] * (bone_ids_i[k] == bone_ids_j[m])
//    This is a soft dot product of weight vectors masked by shared bone membership.
//    Same fold -> high overlap; different folds -> low/zero overlap.
//
// Buffer layout:
//   bone_ids_buf[t*4 + k]     : int32, k-th bone ID for token t (-1 = unused)
//   bone_weights_buf[t*4 + k] : float, k-th blend weight for token t
//
// Dispatch(n_head, seq_len, 1)  numthreads(128,1,1)
//   GroupID.x = head h,  GroupID.y = query i,  ThreadID.x = key j
// P_buf layout: P_buf[h*S*S + i*S + j]
// Activated by GPT2_THINK_BIAS=1 env flag.

static const float PI = 3.14159265f;

cbuffer ThinkBiasCB : register(b0) {
    uint  seq_len;            // S: current sequence length
    uint  n_head;             // H: number of attention heads
    float g0_preserve;        // KuhulPhysics weight g0: preserve_ratio [0,1]
    float g1_delta;           // KuhulPhysics weight g1: delta_entropy  [0,1]  -- vec4 boundary
    float g2_confidence;      // KuhulPhysics weight g2: op confidence  [0,1]
    float g3_op_index;        // KuhulPhysics weight g3: op family      [0,1]
    float g4_temporal;        // KuhulPhysics weight g4: pi-time        [0,1]
    float antigravity_scale;  // KuhulPhysicsSolver::GetAntigravityFloat() [0.1->1.0]  -- vec4 boundary
    float brain_scale;        // LBS routing weight = antigravity_scale * 0.5
    float _pad0;
    float _pad1;
    float _pad2;              // total = 12 x 4 = 48 bytes, 3 x vec4 -- D3D11 aligned
};

// In-place post-softmax attention weights [H, S, S]
RWStructuredBuffer<float> P_buf            : register(u0);

// Modifier 1: geodesic arc depth [S], float in [0, pi] inside <THINK>, 0.0 outside
StructuredBuffer<float>   think_depth      : register(t0);

// Modifier 3: 4-bone LBS -- interleaved [S*4], bone ID -1 = unused bone slot
StructuredBuffer<int>     bone_ids_buf     : register(t1);
StructuredBuffer<float>   bone_weights_buf : register(t2);

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint h = gid.x;
    uint i = gid.y;
    uint j = tid.x;

    if (h >= n_head || i >= seq_len || j >= seq_len)
        return;

    float di = think_depth[i];
    float dj = think_depth[j];
    bool  i_in = (di > 0.001f);
    bool  j_in = (dj > 0.001f);

    // ── Modifier 3: 4-bone LBS overlap ───────────────────────────────────────
    // Compute dot product of bone weight vectors masked by shared bone membership.
    // lbs_overlap in [0, 1]: 1.0 = identical bone assignment, 0.0 = no shared bones.
    float lbs_overlap = 0.0f;
    [unroll]
    for (int ki = 0; ki < 4; ki++) {
        int bi = bone_ids_buf[i * 4 + ki];
        if (bi < 0) continue;
        float wi = bone_weights_buf[i * 4 + ki];
        [unroll]
        for (int kj = 0; kj < 4; kj++) {
            if (bone_ids_buf[j * 4 + kj] == bi)
                lbs_overlap += wi * bone_weights_buf[j * 4 + kj];
        }
    }

    // Early exit: no think arc and no structural overlap
    if (!i_in && !j_in && lbs_overlap < 0.01f)
        return;

    // ── Modifier 1: pi-nary arc weights ──────────────────────────────────────
    float arc_i = sin(di * 0.5f);
    float arc_j = sin(dj * 0.5f);

    // ── Modifier 2: physics-derived bias scalars ──────────────────────────────
    float bias_add   = clamp(g2_confidence * g0_preserve * antigravity_scale * 4.0f, 0.0f, 8.0f);
    float shield_sub = clamp((1.0f - g2_confidence) * g1_delta * antigravity_scale * 2.0f, 0.0f, 8.0f);
    float temporal   = 1.0f + g4_temporal * 0.25f;

    uint  idx = h * seq_len * seq_len + i * seq_len + j;

    if (i_in && j_in) {
        // Both inside <THINK>: pi-nary arc gravity + LBS fold overlap boost
        float arc_term = bias_add * arc_i * arc_j * temporal;
        P_buf[idx] += arc_term * (1.0f + brain_scale * lbs_overlap);
    }
    else if (i_in && !j_in) {
        // Thought attending outward: attentional shield, softened by fold overlap
        float shield = shield_sub * arc_i * (1.0f - brain_scale * 0.5f * lbs_overlap);
        P_buf[idx] -= max(0.0f, shield);
    }
    else if (!i_in && !j_in && lbs_overlap > 0.01f) {
        // Both outside <THINK> but share fold structure: soft co-attention, proportional to overlap
        P_buf[idx] += brain_scale * lbs_overlap;
    }
    // j_in && !i_in: causal mask handles it
}
