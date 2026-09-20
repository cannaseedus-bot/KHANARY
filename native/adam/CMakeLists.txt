cmake_minimum_required(VERSION 3.20)
project(Adam VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(Adam SHARED adam.cpp)
target_include_directories(Adam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(Adam PRIVATE
        /W4         # warning level 4
        /O2         # speed optimisation
        /GL         # whole-program optimisation
        /fp:fast    # fast float — safe for Adam arithmetic
        /arch:AVX   # vectorise the Adam inner loop on HD 4600 CPU fallback
    )
    target_link_options(Adam PRIVATE /LTCG)
    # Export the flat C ABI via a .def for stable ordinals (optional)
    # set_target_properties(Adam PROPERTIES LINK_FLAGS "/DEF:adam.def")
endif()

set_target_properties(Adam PROPERTIES
    OUTPUT_NAME "Adam"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

install(TARGETS Adam
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
install(FILES adam.h DESTINATION include)
