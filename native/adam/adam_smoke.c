/* adam_smoke.c  --  minimal smoke test for Adam.dll
 * Compile (after building Adam.dll):
 *   cl adam_smoke.c /I. Adam.lib /link /out:adam_smoke.exe
 * or from Python:
 *   python ../../../tools/adam_ctypes.py --smoke
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "adam.h"

static int fail(const char* msg) { fprintf(stderr, "FAIL: %s\n", msg); return 1; }

int main(void) {
    printf("Adam.dll version: %s\n", adam_version());

    /* --- Single-tensor smoke --- */
    const uint32_t N = 16;
    float w[16], g[16];
    for (uint32_t i = 0; i < N; ++i) { w[i] = 0.5f; g[i] = 0.01f; }

    AdamState s = {0};
    s.w = w;
    s.g = g;
    if (adam_init(&s, N, 1e-3f, 0.9f, 0.999f, 1e-8f, 1.0f) != 0)
        return fail("adam_init");

    for (int step = 0; step < 10; ++step) {
        for (uint32_t i = 0; i < N; ++i) g[i] = 0.01f * (step + 1);
        if (adam_step(&s) != 0) return fail("adam_step");
    }
    printf("w[0] after 10 steps: %.6f  (expect ~0.49)\n", w[0]);
    if (w[0] >= 0.5f || w[0] <= 0.48f) return fail("weight not decreasing as expected");

    adam_free(&s);

    /* --- Fold arcs smoke --- */
    AdamFoldArcs fa;
    if (adam_fold_arcs_init(&fa, 1e-3f) != 0) return fail("fold_arcs_init");

    float g6[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            g6[i][j] = 0.005f;

    for (int step = 0; step < 5; ++step)
        if (adam_fold_arcs_step(&fa, g6) != 0) return fail("fold_arcs_step");

    /* Pop→Wo (π/3 gap) effective weight should be positive */
    float eff_pop_wo  = adam_fold_arc_effective(&fa, 0, 1);
    /* Pop→Sek (π gap) effective weight should be negative (opposing phase) */
    float eff_pop_sek = adam_fold_arc_effective(&fa, 0, 3);
    printf("effective Pop→Wo:  %.4f  (expect > 0)\n", eff_pop_wo);
    printf("effective Pop→Sek: %.4f  (expect < 0)\n", eff_pop_sek);
    if (eff_pop_wo  <= 0.0f) return fail("Pop→Wo should be positive");
    if (eff_pop_sek >= 0.0f) return fail("Pop→Sek should be negative (π phase)");

    printf("PASS\n");
    return 0;
}
