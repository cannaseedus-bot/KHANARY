cmake_minimum_required(VERSION 3.16)
project(micronaut_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(micronaut_xjson
    main.cpp
    common.cpp
    input.cpp
    output.cpp
    process.cpp
    chat.cpp
    batches.cpp
    data_index.cpp
    kuhule.cpp
    threads.cpp
    scheduler.cpp
    canvas.cpp
    frames.cpp
    xcfe_router.cpp
    scxq2_runtime.cpp
    toml_loader.cpp
    xjson_loader.cpp
    bots_worker.cpp
    phase_router.cpp
)

if (WIN32)
    target_compile_definitions(micronaut_xjson PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# C++17 filesystem
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(micronaut_xjson PRIVATE stdc++fs)
endif()
