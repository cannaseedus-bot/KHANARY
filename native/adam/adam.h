#pragma once
/* adam.h  --  Khanary Adam optimizer DLL  --  flat C ABI
 * Callable from: Python ctypes / C++ / .NET P/Invoke / node-ffi
 * Pattern follows: versions/khlc-v1.0.0/bin/dag.dll
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifdef _WIN32
#  ifdef ADAM_DLL_EXPORTS
#    define ADAM_API __declspec(dllexport)
#  else
#    define ADAM_API __declspec(dllimport)
#  endif
#else
#  define ADAM_API
#endif

/* -----------------------------------------------------------------------
 * AdamState  --  one tensor's optimizer state
 * Caller owns w[] and g[].  DLL owns m[] and v[] after adam_init().
 * ----------------------------------------------------------------------- */
typedef struct AdamState {
    float*   w;         /* weights — updated in-place by adam_step()       */
    float*   m;         /* 1st moment — DLL-allocated by adam_init()       */
    float*   v;         /* 2nd moment — DLL-allocated by adam_init()       */
    float*   g;         /* gradients — read-only; caller sets each step    */
    uint32_t n;         /* element count                                   */
    uint32_t step;      /* current step index (1-based; set by adam_step)  */
    float    lr;        /* learning rate              (default 1e-5)       */
    float    beta1;     /* 1st moment decay           (default 0.9)        */
    float    beta2;     /* 2nd moment decay           (default 0.999)      */
    float    eps;       /* denominator stability term (default 1e-8)       */
    float    grad_clip; /* L2 gradient clip threshold (0 = disabled)       */
} AdamState;

/* Allocate m/v buffers for n elements, set hyperparams, zero state.
 * Returns 0 on success, -1 bad args, -2 alloc failure.               */
ADAM_API int adam_init(AdamState* s, uint32_t n,
                       float lr, float beta1, float beta2,
                       float eps, float grad_clip);

/* Apply one Adam step; s->g must be set by caller before each call.
 * Increments s->step, updates m/v/w in-place.
 * Returns 0 on success, -1 bad state.                                */
ADAM_API int adam_step(AdamState* s);

/* Clip s->g by L2 norm in-place (without updating weights/moments).
 * No-op if s->grad_clip <= 0.  Returns 0.                            */
ADAM_API int adam_clip_grads(AdamState* s);

/* Free DLL-owned m/v buffers; sets pointers to NULL.                 */
ADAM_API void adam_free(AdamState* s);

/* -----------------------------------------------------------------------
 * AdamFoldArcs  --  6×6 arc weight matrix for K-CUBE fold geometry
 *
 * Each arc[i][j] holds the learned flow strength from fold i to fold j.
 * Fold index order: 0=Pop  1=Wo  2=Yax  3=Sek  4=Ch'en  5=Xul
 * angles[i] = i * π/3  (fixed structural identity — NOT optimized)
 *
 * arc_w[i][j] are scalar weights (n_arc elements each); Adam optimizes
 * these.  For simple arc strength n_arc=1.  For richer mixing n_arc=d.
 * ----------------------------------------------------------------------- */

#define ADAM_FOLD_COUNT 6

typedef struct AdamFoldArcs {
    /* Inline weight/moment storage — no separate heap allocation needed
     * for n_arc == 1 (the common case).  For larger n_arc use the heap
     * variant via adam_fold_arcs_init_heap().                           */
    float     arc_w[ADAM_FOLD_COUNT][ADAM_FOLD_COUNT];  /* learned weights   */
    float     arc_m[ADAM_FOLD_COUNT][ADAM_FOLD_COUNT];  /* 1st moment        */
    float     arc_v[ADAM_FOLD_COUNT][ADAM_FOLD_COUNT];  /* 2nd moment        */
    float     arc_g[ADAM_FOLD_COUNT][ADAM_FOLD_COUNT];  /* current gradients */
    float     angles[ADAM_FOLD_COUNT];                  /* i * π/3, read-only*/
    uint32_t  step;
    float     lr, beta1, beta2, eps, grad_clip;
} AdamFoldArcs;

/* Initialize scalar (n_arc=1) arc weight matrix with sane defaults.
 * arc_w initialized to 1/6 (uniform flow), angles set to i*π/3.     */
ADAM_API int  adam_fold_arcs_init(AdamFoldArcs* fa, float lr);

/* Apply one Adam step over all 36 arc weights.
 * g[6][6] = gradient matrix for this step.
 * Returns 0 on success.                                               */
ADAM_API int  adam_fold_arcs_step(AdamFoldArcs* fa, float g[ADAM_FOLD_COUNT][ADAM_FOLD_COUNT]);

/* Return the effective weight for arc i→j, modulated by phase delta:
 *   effective = arc_w[i][j] * cos(angles[j] - angles[i])
 * Nearby folds (small angle gap) carry more weight.                   */
ADAM_API float adam_fold_arc_effective(const AdamFoldArcs* fa, int i, int j);

/* Reset moments + step counter, keep weights.                         */
ADAM_API void adam_fold_arcs_reset_moments(AdamFoldArcs* fa);

/* -----------------------------------------------------------------------
 * Version
 * ----------------------------------------------------------------------- */
ADAM_API const char* adam_version(void);  /* returns "1.0.0" */

#ifdef __cplusplus
}
#endif
