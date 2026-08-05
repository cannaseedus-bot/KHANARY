cmake_minimum_required(VERSION 3.20)
project(gpt2_trainer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── third-party: nlohmann/json (header-only, fetched from GitHub) ─────────────
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ── executable ────────────────────────────────────────────────────────────────
add_executable(gpt2_trainer
    gpt2_train_main.cpp
    gpt2_trainer.cpp
    d3d11_engine.cpp
)

target_include_directories(gpt2_trainer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}   # d3d11_engine.h, gpt2_config.h, gpt2_trainer.h
    # Note: gpt2_trainer.cpp uses #include "../pi_kuhul/KuhulPhysics.h" (relative to trainer/).
    # That resolves to <repo_root>/pi_kuhul/ — the headers must live there, not under trainer/.
)

target_compile_definitions(gpt2_trainer PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    UNICODE _UNICODE
)

# MSVC-specific: warning level, multi-processor compile
if(MSVC)
    target_compile_options(gpt2_trainer PRIVATE /W3 /MP /arch:AVX2)
endif()

target_link_libraries(gpt2_trainer PRIVATE
    d3d11
    dxgi
    d3dcompiler
    nlohmann_json::nlohmann_json
)

# ── shaders: copy to <build>/shaders/ so "../shaders/" resolves at runtime ────
# The trainer uses D3DCompileFromFile with base "../shaders/" relative to CWD.
# Convention: run the exe from <build>/Release/ → ../shaders/ = <build>/shaders/.
set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_command(TARGET gpt2_trainer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHADER_SRC_DIR}" "${SHADER_OUT_DIR}"
    COMMENT "Copying HLSL shaders to ${SHADER_OUT_DIR}"
)

message(STATUS "gpt2_trainer: shaders will be staged to ${SHADER_OUT_DIR}")
message(STATUS "gpt2_trainer: run exe from <build>/Release/ — shaders resolve via ../shaders/")
