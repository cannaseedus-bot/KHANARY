#define ADAM_DLL_EXPORTS
#include "adam.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cassert>

/* π/3 phase angles — fixed structural geometry, never optimized */
static constexpr float PI_OVER_3 = 1.04719755119659775f;

static const float FOLD_ANGLES[ADAM_FOLD_COUNT] = {
    0.0f,            /* Pop   — observe / Q-read             */
    PI_OVER_3,       /* Wo    — weight / mask                */
    PI_OVER_3 * 2,   /* Yax   — enumerate / K-read           */
    PI_OVER_3 * 3,   /* Sek   — compute / QKᵀ               */
    PI_OVER_3 * 4,   /* Ch'en — collect / V-gather           */
    PI_OVER_3 * 5,   /* Xul   — entropy / output-project    */
};

/* -----------------------------------------------------------------------
 * AdamState
 * ----------------------------------------------------------------------- */

int adam_init(AdamState* s, uint32_t n,
              float lr, float beta1, float beta2,
              float eps, float grad_clip)
{
    if (!s || n == 0) return -1;
    s->n         = n;
    s->step      = 0;
    s->lr        = lr;
    s->beta1     = beta1;
    s->beta2     = beta2;
    s->eps       = eps;
    s->grad_clip = grad_clip;
    /* w and g are caller-owned — never touch them here */
    s->m         = (float*)calloc(n, sizeof(float));
    s->v         = (float*)calloc(n, sizeof(float));
    if (!s->m || !s->v) {
        free(s->m); free(s->v);
        s->m = s->v = nullptr;
        return -2;
    }
    return 0;
}

int adam_clip_grads(AdamState* s)
{
    if (!s || !s->g || s->grad_clip <= 0.0f || s->n == 0) return 0;
    double norm2 = 0.0;
    for (uint32_t i = 0; i < s->n; ++i) {
        double gi = s->g[i];
        norm2 += gi * gi;
    }
    float norm = (float)std::sqrt(norm2);
    if (norm > s->grad_clip) {
        float ratio = s->grad_clip / norm;
        for (uint32_t i = 0; i < s->n; ++i)
            s->g[i] *= ratio;
    }
    return 0;
}

int adam_step(AdamState* s)
{
    if (!s || !s->w || !s->m || !s->v || !s->g || s->n == 0) return -1;

    adam_clip_grads(s);

    s->step++;
    const float b1      = s->beta1;
    const float b2      = s->beta2;
    const float lr      = s->lr;
    const float eps     = s->eps;

    /* Bias correction: fused into lr_hat to avoid redundant pow per element */
    const float bc1    = 1.0f - std::pow(b1, (float)s->step);
    const float bc2    = 1.0f - std::pow(b2, (float)s->step);
    const float lr_hat = lr * std::sqrt(bc2) / bc1;

    float* __restrict w = s->w;
    float* __restrict m = s->m;
    float* __restrict v = s->v;
    const float* __restrict g = s->g;

    for (uint32_t i = 0; i < s->n; ++i) {
        float gi  = g[i];
        float mi  = b1 * m[i] + (1.0f - b1) * gi;
        float vi  = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i]      = mi;
        v[i]      = vi;
        w[i]     -= lr_hat * mi / (std::sqrt(vi) + eps);
    }
    return 0;
}

void adam_free(AdamState* s)
{
    if (!s) return;
    free(s->m); s->m = nullptr;
    free(s->v); s->v = nullptr;
}

/* -----------------------------------------------------------------------
 * AdamFoldArcs
 * ----------------------------------------------------------------------- */

int adam_fold_arcs_init(AdamFoldArcs* fa, float lr)
{
    if (!fa) return -1;
    std::memset(fa, 0, sizeof(AdamFoldArcs));

    /* Structural angles — fixed geometry */
    for (int i = 0; i < ADAM_FOLD_COUNT; ++i)
        fa->angles[i] = FOLD_ANGLES[i];

    /* Uniform initial arc weights: 1/6 so rows sum to ~1 */
    constexpr float INIT_W = 1.0f / ADAM_FOLD_COUNT;
    for (int i = 0; i < ADAM_FOLD_COUNT; ++i)
        for (int j = 0; j < ADAM_FOLD_COUNT; ++j)
            fa->arc_w[i][j] = INIT_W;

    fa->lr        = lr;
    fa->beta1     = 0.9f;
    fa->beta2     = 0.999f;
    fa->eps       = 1e-8f;
    fa->grad_clip = 1.0f;
    fa->step      = 0;
    return 0;
}

int adam_fold_arcs_step(AdamFoldArcs* fa, float g[ADAM_FOLD_COUNT][ADAM_FOLD_COUNT])
{
    if (!fa || !g) return -1;

    /* Clip gradients: compute global L2 norm across all 36 arc grads */
    if (fa->grad_clip > 0.0f) {
        double norm2 = 0.0;
        for (int i = 0; i < ADAM_FOLD_COUNT; ++i)
            for (int j = 0; j < ADAM_FOLD_COUNT; ++j) {
                double gi = g[i][j];
                norm2 += gi * gi;
            }
        float norm = (float)std::sqrt(norm2);
        if (norm > fa->grad_clip) {
            float ratio = fa->grad_clip / norm;
            for (int i = 0; i < ADAM_FOLD_COUNT; ++i)
                for (int j = 0; j < ADAM_FOLD_COUNT; ++j)
                    g[i][j] *= ratio;
        }
    }

    fa->step++;
    const float b1     = fa->beta1;
    const float b2     = fa->beta2;
    const float bc1    = 1.0f - std::pow(b1, (float)fa->step);
    const float bc2    = 1.0f - std::pow(b2, (float)fa->step);
    const float lr_hat = fa->lr * std::sqrt(bc2) / bc1;

    for (int i = 0; i < ADAM_FOLD_COUNT; ++i) {
        for (int j = 0; j < ADAM_FOLD_COUNT; ++j) {
            float gi             = g[i][j];
            float mi             = b1 * fa->arc_m[i][j] + (1.0f - b1) * gi;
            float vi             = b2 * fa->arc_v[i][j] + (1.0f - b2) * gi * gi;
            fa->arc_m[i][j]      = mi;
            fa->arc_v[i][j]      = vi;
            fa->arc_w[i][j]     -= lr_hat * mi / (std::sqrt(vi) + fa->eps);
        }
    }
    return 0;
}

float adam_fold_arc_effective(const AdamFoldArcs* fa, int i, int j)
{
    if (!fa || i < 0 || i >= ADAM_FOLD_COUNT || j < 0 || j >= ADAM_FOLD_COUNT)
        return 0.0f;
    /* Geometric modulation: arcs spanning small phase deltas carry more weight.
     * Pop→Wo (π/3 gap) > Pop→Sek (π gap) > Pop→Yax (2π/3 gap), etc.    */
    float delta = fa->angles[j] - fa->angles[i];
    float mod   = std::cos(delta);   /* range [-1, 1]; negative = opposing phase */
    return fa->arc_w[i][j] * mod;
}

void adam_fold_arcs_reset_moments(AdamFoldArcs* fa)
{
    if (!fa) return;
    std::memset(fa->arc_m, 0, sizeof(fa->arc_m));
    std::memset(fa->arc_v, 0, sizeof(fa->arc_v));
    fa->step = 0;
}

/* -----------------------------------------------------------------------
 * Version
 * ----------------------------------------------------------------------- */
const char* adam_version(void) { return "1.0.0"; }
