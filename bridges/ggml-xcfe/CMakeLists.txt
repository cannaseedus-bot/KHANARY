# ggml-xcfe backend — KHΛNARY XCFE glyph runtime.
# Enabled by -DGGML_XCFE=ON, wired via ggml_add_backend(XCFE) in ggml/src/CMakeLists.txt.
ggml_add_backend_library(ggml-xcfe
                         ggml-xcfe.cpp
                        )

# ggml-xcfe reuses CPU host buffers — same as the BLAS backend. No extra link deps for the
# ggml static library target (xcfe_gl_ops.dll is loaded lazily via LoadLibraryA at runtime).

# xcfe_gl_ops.dll — OpenGL 4.3 compute ops, loaded lazily at runtime from next to the binary.
# B001 fix: this target was previously missing from CMake; only buildable via manual cl.exe.
if(WIN32)
    add_library(xcfe_gl_ops SHARED
        "${CMAKE_CURRENT_LIST_DIR}/xcfe_gl_ops.cpp"
    )
    target_compile_features(xcfe_gl_ops PRIVATE cxx_std_17)
    target_link_libraries(xcfe_gl_ops PRIVATE opengl32 gdi32 user32)
    set_target_properties(xcfe_gl_ops PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        PREFIX ""
    )
endif()
