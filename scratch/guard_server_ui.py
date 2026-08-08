import pathlib

p = pathlib.Path(r"C:\Users\canna\_khanary_inspect\khanary-llama-build\llama.cpp\tools\server\server-http.cpp")
t = p.read_text(encoding="utf-8")

# 1. Guard the ui.h include
old_inc = '#include "server-http.h"\n#include "server-common.h"\n#include "ui.h"'
new_inc = '#include "server-http.h"\n#include "server-common.h"\n#if defined(LLAMA_BUILD_UI)\n#include "ui.h"\n#endif'
assert old_inc in t, "include block not found"
t = t.replace(old_inc, new_inc, 1)

# 2. Guard the frontend_paths lambda (unconditional llama_ui_get_assets call)
old_fp = """    // Frontend paths - all embedded UI assets
    static const std::unordered_set<std::string> frontend_paths = []() {
        std::unordered_set<std::string> paths { "/" };
        for (const llama_ui_asset & a : llama_ui_get_assets()) {
            paths.insert("/" + a.name);
        }
        return paths;
    }();"""
new_fp = """    // Frontend paths - all embedded UI assets
#if defined(LLAMA_BUILD_UI)
    static const std::unordered_set<std::string> frontend_paths = []() {
        std::unordered_set<std::string> paths { "/" };
        for (const llama_ui_asset & a : llama_ui_get_assets()) {
            paths.insert("/" + a.name);
        }
        return paths;
    }();
#else
    static const std::unordered_set<std::string> frontend_paths = { "/" };
#endif"""
assert old_fp in t, "frontend_paths block not found"
t = t.replace(old_fp, new_fp, 1)

p.write_text(t, encoding="utf-8")
print("server-http.cpp UI guards added")
