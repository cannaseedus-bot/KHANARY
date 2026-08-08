import pathlib

p = pathlib.Path(r"C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\src\scxq2_format_v1_2.cpp")
t = p.read_text(encoding="utf-8")

anchor = "const std::uint8_t* lane_data = data + pos + 10;"
assert anchor in t, "anchor not found"
idx = t.find(anchor)

guard = (
    "    // Guard against underflow: lane_len must include the 4-byte CRC32\n"
    "    if (lane_len < 4) {\n"
    '      error = "Lane length too small (< 4)";\n'
    "      return false;\n"
    "    }\n\n"
)

p.write_text(t[:idx] + guard + t[idx:], encoding="utf-8")
print("underflow guard added")
