# KHANARY ggml-xcfe backend — a real, distinct target (NOT the orphan webgpu copy).
ggml_add_backend_library(ggml-xcfe
    ggml-xcfe.cpp
    ../../include/ggml-xcfe.h
)
