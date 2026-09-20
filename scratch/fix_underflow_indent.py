import pathlib

p = pathlib.Path(r"C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\src\scxq2_format_v1_2.cpp")
t = p.read_text(encoding="utf-8")

old = (
    "    // Parse lane data\n"
    "        // Guard against underflow: lane_len must include the 4-byte CRC32\n"
    "    if (lane_len < 4) {\n"
    '      error = "Lane length too small (< 4)";\n'
    "      return false;\n"
    "    }\n"
    "\n"
    "const std::uint8_t* lane_data = data + pos + 10;\n"
    "    std::size_t lane_data_len = lane_len - 4;  // -4 for CRC32"
)
new = (
    "    // Guard against underflow: lane_len must include the 4-byte CRC32\n"
    "    if (lane_len < 4) {\n"
    '      error = "Lane length too small (< 4)";\n'
    "      return false;\n"
    "    }\n"
    "\n"
    "    // Parse lane data\n"
    "    const std::uint8_t* lane_data = data + pos + 10;\n"
    "    std::size_t lane_data_len = lane_len - 4;  // -4 for CRC32"
)

assert old in t, "pattern not found"
p.write_text(t.replace(old, new), encoding="utf-8")
print("indentation fixed")
