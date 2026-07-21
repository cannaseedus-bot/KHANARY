// ggml-xcfe.h — KHΛNARY XCFE backend for llama.cpp's ggml.
//
// A distinct ggml backend (registers as "XCFE", GUID of its own) in the same class as the BLAS
// backend: it reuses the CPU host-memory buffer type and claims only GGML_OP_MUL_MAT, so ggml's
// scheduler routes matmuls here and everything else to the CPU backend. MUL_MAT is where the
// KHΛNARY glyph GEMM (D3D11 cs_5_0 `G_MATMUL`, or DirectML) is meant to execute.
//
// Install: copy this file to llama.cpp/ggml/include/ and the ggml-xcfe/ folder to
// llama.cpp/ggml/src/ ; see INSTALL.md.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize an XCFE backend (stream).
GGML_BACKEND_API ggml_backend_t ggml_backend_xcfe_init(void);

// Is this backend an XCFE backend?
GGML_BACKEND_API bool ggml_backend_is_xcfe(ggml_backend_t backend);

// Number of worker threads for the reference GEMM (until the GPU dispatch is wired).
GGML_BACKEND_API void ggml_backend_xcfe_set_n_threads(ggml_backend_t backend_xcfe, int n_threads);

// Backend registry entry (referenced by ggml_add_backend(XCFE) via GGML_XCFE).
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xcfe_reg(void);

#ifdef __cplusplus
}
#endif
