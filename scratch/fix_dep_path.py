import pathlib

p = pathlib.Path(r"C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\CMakeLists.txt")
t = p.read_text(encoding="utf-8")

old = 'set(NLOHMANN_JSON_INCLUDE_DIR "${XVM_REPO_ROOT}/asx_scx/dependencies/json/include")'
new = (
    'set(NLOHMANN_JSON_INCLUDE_DIR "${XVM_REPO_ROOT}/asx_scx/dependencies/json/include")\n'
    'if (NOT EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")\n'
    '  # Fallback: nlohmann_json from trainer FetchContent (present in this checkout)\n'
    '  set(NLOHMANN_JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../trainer/build/_deps/nlohmann_json-src/single_include")\n'
    'endif()'
)

assert old in t, "dep path line not found"
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("dep path fallback added")
