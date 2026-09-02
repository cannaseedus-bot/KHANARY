cmake_minimum_required(VERSION 3.20)
project(gpt2_trainer LANGUAGES CXX)

set(DIRECTXTK_ROOT "C:/Users/canna/.ASX.cpp/DirectXTK" CACHE PATH
    "DirectX Tool Kit source tree used by the DDS residency provider")

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

find_program(FXC_EXECUTABLE fxc.exe
    PATHS
        "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/10.0.26100.0/x64"
        "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/10.0.22621.0/x64"
        "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/10.0.19041.0/x64"
)

if(FXC_EXECUTABLE)
    add_custom_command(
        OUTPUT "${SHADER_OUT_DIR}/xshard_adapt_fold.cso"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT_DIR}"
        COMMAND "${FXC_EXECUTABLE}" /T cs_5_0 /E CSMain
                "${SHADER_SRC_DIR}/xshard_adapt_fold.hlsl"
                /Fo "${SHADER_OUT_DIR}/xshard_adapt_fold.cso"
                /O3 /Zi
        DEPENDS "${SHADER_SRC_DIR}/xshard_adapt_fold.hlsl"
        COMMENT "Compiling xshard_adapt_fold.hlsl to cs_5_0 DXBC"
        VERBATIM
    )
    add_custom_target(xshard_adapt_fold_cso ALL
        DEPENDS "${SHADER_OUT_DIR}/xshard_adapt_fold.cso"
    )
    add_dependencies(xshard_adapt_fold_cso gpt2_trainer)

    # ── LBS + gravity + fold-route shaders (CSS-Micronaut-PIGCCP GPU layer) ─────
    foreach(_SHADER IN ITEMS
            cs_gravity_field_layer_
            cs_fold_route_matmul
            gpt2_kuhul_think_bias
            cs_bone_argsort_
            cs_vertex_skin
            cs_fold_kernel_compute_)
        add_custom_command(
            OUTPUT  "${SHADER_OUT_DIR}/${_SHADER}.cso"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT_DIR}"
            COMMAND "${FXC_EXECUTABLE}" /T cs_5_0 /E CSMain
                    "${SHADER_SRC_DIR}/${_SHADER}.hlsl"
                    /Fo "${SHADER_OUT_DIR}/${_SHADER}.cso"
                    /O3 /Zi
            DEPENDS "${SHADER_SRC_DIR}/${_SHADER}.hlsl"
            COMMENT "Compiling ${_SHADER}.hlsl → cs_5_0 DXBC"
            VERBATIM
        )
        add_custom_target(${_SHADER}_cso ALL
            DEPENDS "${SHADER_OUT_DIR}/${_SHADER}.cso"
        )
        add_dependencies(${_SHADER}_cso gpt2_trainer)
    endforeach()
else()
    message(WARNING "fxc.exe not found; xshard_adapt_fold.cso will not be precompiled by CMake")
endif()

message(STATUS "gpt2_trainer: shaders will be staged to ${SHADER_OUT_DIR}")
message(STATUS "gpt2_trainer: run exe from <build>/Release/ — shaders resolve via ../shaders/")

# ── scx_dds_loader — DirectXTK-backed DDS residency provider ────────────────
# This DLL owns physical DDS inspection/loading only. It does not mutate model
# weights or participate in gradient calculation; the field graph remains the
# authority for semantic identity and revision publication.
add_library(scx_dds_loader SHARED
    scx_dds_loader.cpp
    "${DIRECTXTK_ROOT}/Src/DDSTextureLoader.cpp"
)

target_include_directories(scx_dds_loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${DIRECTXTK_ROOT}/Inc"
    "${DIRECTXTK_ROOT}/Src"
)

target_compile_definitions(scx_dds_loader PRIVATE
    SCX_DDS_LOADER_BUILD
    DIRECTX_TOOLKIT_EXPORT
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    UNICODE _UNICODE
    _WIN32_WINNT=0x0A00
)

if(MSVC)
    target_compile_options(scx_dds_loader PRIVATE /W3 /MP)
endif()

target_link_libraries(scx_dds_loader PRIVATE
    d3d11
    dxgi
    windowscodecs
)

add_executable(scx_dds_loader_smoke
    scx_dds_loader_smoke.cpp
)
target_include_directories(scx_dds_loader_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${DIRECTXTK_ROOT}/Inc"
    "${DIRECTXTK_ROOT}/Src"
)
target_compile_definitions(scx_dds_loader_smoke PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE
)
if(MSVC)
    target_compile_options(scx_dds_loader_smoke PRIVATE /W3 /MP)
endif()
target_link_libraries(scx_dds_loader_smoke PRIVATE d3d11 dxgi scx_dds_loader)

set(SCX_DDS_STAGE_DIR "${CMAKE_SOURCE_DIR}/../dist/kuhul-runtime-v1/drivers")
add_custom_command(TARGET scx_dds_loader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SCX_DDS_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:scx_dds_loader>" "${SCX_DDS_STAGE_DIR}/scx_dds_loader.dll"
    COMMENT "Staging scx_dds_loader.dll to runtime-v1/drivers"
)
add_custom_command(TARGET scx_dds_loader_smoke POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SCX_DDS_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:scx_dds_loader_smoke>" "${SCX_DDS_STAGE_DIR}/scx_dds_loader_smoke.exe"
    COMMENT "Staging scx_dds_loader_smoke.exe to runtime-v1/drivers"
)

message(STATUS "scx_dds_loader: DirectXTK=${DIRECTXTK_ROOT}")

# ── xshard_info (header-only, stdlib only — no D3D, no nlohmann) ─────────────
add_executable(xshard_info
    xshard_info.cpp
)

target_include_directories(xshard_info PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}   # xshard.h
)

target_compile_definitions(xshard_info PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    UNICODE _UNICODE
)

if(MSVC)
    target_compile_options(xshard_info PRIVATE /W3 /MP)
endif()

message(STATUS "xshard_info: header-only XSHARD/1 inspector CLI")

# ── xshard_adapt — D3D11 host wrapper for xshard_adapt_fold.cso ──────────────
add_executable(xshard_adapt
    xshard_adapt.cpp
    d3d11_engine.cpp
)

target_include_directories(xshard_adapt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(xshard_adapt PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    UNICODE _UNICODE
)

if(MSVC)
    target_compile_options(xshard_adapt PRIVATE /W3 /MP /arch:AVX2)
endif()

target_link_libraries(xshard_adapt PRIVATE
    d3d11
    dxgi
)

if(TARGET xshard_adapt_fold_cso)
    add_dependencies(xshard_adapt xshard_adapt_fold_cso)
endif()

message(STATUS "xshard_adapt: D3D11 XSHARD F32 shard update CLI")

# ── xshard_backward — streamed token/data bin -> gradient XSHARD producer ────
add_executable(xshard_backward
    xshard_backward.cpp
)

target_include_directories(xshard_backward PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(xshard_backward PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    UNICODE _UNICODE
)

if(MSVC)
    target_compile_options(xshard_backward PRIVATE /W3 /MP /arch:AVX2)
endif()

target_link_libraries(xshard_backward PRIVATE
    bcrypt
)

message(STATUS "xshard_backward: streamed token/data bin to gradient XSHARD CLI")
