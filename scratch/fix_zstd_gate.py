import pathlib

p = pathlib.Path(r"C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\src\scxqdds_chunks_loader.cpp")
t = p.read_text(encoding="utf-8")

old = """bool DecompressZstdToMemory(const std::filesystem::path& chunk, uint8_t* dst, uint64_t rawBytes) {
#ifndef XVM_USE_ZSTD
    return false;
#else"""

new = """bool DecompressZstdToMemory(const std::filesystem::path& chunk, uint8_t* dst, uint64_t rawBytes) {
#ifndef XVM_USE_ZSTD
    // Graceful fallback: zstd not compiled in. Treat chunk as raw bytes.
    FILE* f = nullptr;
    _wfopen_s(&f, chunk.c_str(), L"rb");
    if (!f) return false;
    const size_t read = fread(dst, 1, static_cast<size_t>(rawBytes), f);
    fclose(f);
    return read == static_cast<size_t>(rawBytes);
#else"""

assert old in t, "zstd gate pattern not found"
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("zstd graceful fallback added")
