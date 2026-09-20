# check_birdsong.py — self-checks for the Birdsong Geometry grammar.
#   1. JSON  : the example validates against birdsong-brain.schema.json.
#   2. EBNF  : every referenced non-terminal defined; all reachable from `BirdSong`.
#   3. GROUND: the example's whole-graph totals match the REAL birdsong_mesh.stb.
import os, re, sys, json, struct
import jsonschema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "docs", "birdsong-brain.schema.json")
EBNF = os.path.join(ROOT, "docs", "birdsong-geometry.ebnf")
EXAMPLE = os.path.join(ROOT, "docs", "examples", "birdsong.example.json")
STB = os.path.join(ROOT, "models", "khanary-geometry-v0.3.0", "data", "birdsong_mesh.stb")

def check_json():
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    v = jsonschema.Draft202012Validator(schema)
    errs = sorted(v.iter_errors(json.load(open(EXAMPLE, encoding="utf-8"))), key=lambda e: list(e.path))
    if errs:
        print(f"[JSON] FAIL {os.path.basename(EXAMPLE)}")
        for e in errs[:8]:
            print(f"       - {list(e.path)}: {e.message}")
        return False
    print(f"[JSON] pass {os.path.basename(EXAMPLE)}")
    return True

def check_ebnf():
    text = open(EBNF, encoding="utf-8").read()
    text = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.S)
    text = re.sub(r"\?[^?]*\?", " ", text)
    # strip quoted terminals in one L->R pass: consume whichever quote opens first, to its close.
    # (order-independent; handles both '"' and "double" tokens with embedded quotes.)
    text = re.sub(r"\"[^\"]*\"|'[^']*'", " ", text)
    rules = {m.group(1): m.group(2) for m in re.finditer(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)\s*=(.*?);", text, flags=re.S)}
    defined = set(rules)
    ref = {k: set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", v)) for k, v in rules.items()}
    undefined = sorted(set().union(*ref.values()) - defined) if ref else []
    seen, stack = set(), ["BirdSong"]
    while stack:
        r = stack.pop()
        if r in seen or r not in rules: continue
        seen.add(r); stack.extend(ref.get(r, ()))
    unreachable = sorted(defined - seen - {"BirdSong"})
    ok = True
    print(f"[EBNF] {len(defined)} rules defined; start=BirdSong")
    if undefined: ok = False; print(f"[EBNF] FAIL referenced-but-undefined: {undefined}")
    else: print("[EBNF] pass  all referenced non-terminals defined")
    if unreachable: ok = False; print(f"[EBNF] FAIL defined-but-unreachable: {unreachable}")
    else: print("[EBNF] pass  all defined rules reachable from `BirdSong`")
    return ok

def check_ground():
    b = open(STB, "rb").read()
    assert b[:4] == b"STB0"
    T = []
    tc = struct.unpack("<H", b[6:8])[0]
    for i in range(tc):
        e = b[32+i*32:32+(i+1)*32]
        T.append(struct.unpack("<III", e[20:32]))
    N, E, NB = T[0][0], T[3][0], T[5][0]
    g = json.load(open(EXAMPLE, encoding="utf-8"))["graph"]
    ok = (g["nodes"], g["edges"], g["neighbors"]) == (N, E, NB)
    print(f"[GROUND] example graph totals {(g['nodes'],g['edges'],g['neighbors'])} vs .stb {(N,E,NB)}: {'pass' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    r = all([check_json(), check_ebnf(), check_ground()])
    print("[done]", "ALL PASS" if r else "FAILURES ABOVE")
    sys.exit(0 if r else 1)
