# ggml-xcfe backend — KHΛNARY XCFE glyph runtime.
# Enabled by -DGGML_XCFE=ON, wired via ggml_add_backend(XCFE) in ggml/src/CMakeLists.txt.
ggml_add_backend_library(ggml-xcfe
                         ggml-xcfe.cpp
                        )

# The reference GEMM reuses the CPU backend's host buffers (ggml_backend_cpu_buffer_type /
# ggml_backend_cpu_buffer_from_ptr) — same as the BLAS backend. No extra link deps yet.
# When the KHΛNARY GPU dispatch is wired, add its libraries here (D3D11: d3d11 d3dcompiler;
# DirectML: DirectML d3d12 dxgi).
