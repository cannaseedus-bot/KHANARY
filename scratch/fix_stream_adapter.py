import pathlib

files = [
    r"C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\src\d3d11_engine.h",
    r"C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\src\d3d12_engine.h",
    r"C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\src\d3d12_stream_adapter.cpp",
]

old = "streamSCXQ2ToCodeBuffer"
new = "streamSCXQDDSChunkToCodeBuffer"

for f in files:
    p = pathlib.Path(f)
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n:
        t = t.replace(old, new)
        p.write_text(t, encoding="utf-8")
        print(f"{p.name}: renamed {n} occurrence(s)")
    else:
        print(f"{p.name}: no match")
