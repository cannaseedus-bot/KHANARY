import pathlib

p = pathlib.Path(r"C:\Users\canna\_khanary_inspect\khanary-llama-build\build_gpu.ps1")
t = p.read_text(encoding="utf-8")

# Replace non-ASCII em-dashes with ASCII hyphens so PowerShell 5.1 (ANSI default,
# no BOM) can parse the file regardless of system locale.
before = t.count("\u2014")
t = t.replace("\u2014", "-")
# Also normalize any stray non-ASCII bullets/quotes that could break ANSI parse
t = t.replace("\u2013", "-").replace("\u2022", "*").replace("\u2018", "'").replace("\u2019", "'")
t = t.replace("\u201c", '"').replace("\u201d", '"')
t = t.replace("\u00d7", "x").replace("\u2192", "->")

# Ensure UTF-8 BOM so PowerShell always reads it as UTF-8
p.write_bytes(b"\xef\xbb\xbf" + t.encode("utf-8"))
print(f"ASCII-ized {before} em-dashes, wrote UTF-8 BOM")
