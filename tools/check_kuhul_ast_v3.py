# check_kuhul_ast_v3.py — the two decisive self-checks for the K'UHUL-3D vNext contract.
#
#   1. JSON side  : validate the provided example ASTs against kuhul.ast.v3.schema.json.
#                   (If their own examples don't pass, the schema is wrong.)
#   2. EBNF side  : every referenced non-terminal defined, every defined rule reachable
#                   from `document` (the grammar-validator discipline).
import os, re, sys, json
import jsonschema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "docs", "kuhul.ast.v3.schema.json")
EBNF = os.path.join(ROOT, "docs", "kuhul-3d-vnext.ebnf")
EXAMPLES = [
    os.path.join(ROOT, "docs", "examples", "kuhul.ast.v3.example.json"),
    os.path.join(ROOT, "docs", "examples", "kuhul.ast.v3.recursion.example.json"),
    os.path.join(ROOT, "docs", "examples", "kuhul.ast.v3.glyph_atom.example.json"),
]

def check_json():
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    v = jsonschema.Draft202012Validator(schema)
    ok = True
    for ex in EXAMPLES:
        inst = json.load(open(ex, encoding="utf-8"))
        errs = sorted(v.iter_errors(inst), key=lambda e: e.path)
        if errs:
            ok = False
            print(f"[JSON] FAIL {os.path.basename(ex)}")
            for e in errs[:8]:
                print(f"       - {list(e.path)}: {e.message}")
        else:
            print(f"[JSON] pass {os.path.basename(ex)}")
    return ok

def check_ebnf():
    text = open(EBNF, encoding="utf-8").read()
    text = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.S)          # strip comments
    text = re.sub(r"\?[^?]*\?", " ", text)                         # strip ?special? terminals
    # strip quoted terminals in one L->R pass: consume whichever quote opens first, to its close.
    # (handles both '"' and "[Ch'en" — order-independent, unlike two separate substitutions.)
    text = re.sub(r"\"[^\"]*\"|'[^']*'", " ", text)
    rules = {}
    for m in re.finditer(r"(?m)^([a-z_][a-z0-9_]*)\s*=(.*?);", text, flags=re.S):
        rules[m.group(1)] = m.group(2)
    defined = set(rules)
    referenced = {}
    for name, rhs in rules.items():
        referenced[name] = set(re.findall(r"[a-z_][a-z0-9_]*", rhs))
    all_refs = set().union(*referenced.values()) if referenced else set()

    undefined = sorted(all_refs - defined)
    # reachability from `document`
    seen, stack = set(), ["document"]
    while stack:
        r = stack.pop()
        if r in seen or r not in rules:
            continue
        seen.add(r)
        stack.extend(referenced.get(r, ()))
    unreachable = sorted(defined - seen - {"document"})

    ok = True
    print(f"[EBNF] {len(defined)} rules defined; start=document")
    if undefined:
        ok = False; print(f"[EBNF] FAIL referenced-but-undefined: {undefined}")
    else:
        print("[EBNF] pass  all referenced non-terminals defined")
    if unreachable:
        ok = False; print(f"[EBNF] FAIL defined-but-unreachable: {unreachable}")
    else:
        print("[EBNF] pass  all defined rules reachable from `document`")
    return ok

def _refs(node):
    out = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str):
                out.add(v.rsplit("/", 1)[-1])
            else:
                out |= _refs(v)
    elif isinstance(node, list):
        for v in node:
            out |= _refs(v)
    return out

def check_laws():
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    defs = schema["$defs"]
    phases = set(defs["PhaseName"]["enum"])
    # opcode terminals from the EBNF *_opcode rules
    text = open(EBNF, encoding="utf-8").read()
    text = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.S)
    opcodes = set()
    for m in re.finditer(r"(?m)^[a-z_]*opcode\s*=(.*?);", text, flags=re.S):
        opcodes |= set(re.findall(r'"([A-Z_]+)"', m.group(1)))
    ok = True
    # LAW P1 — Phase ∩ Opcode = ∅
    inter = phases & opcodes
    if inter:
        ok = False; print(f"[LAW P1] FAIL Phase intersect Opcode = {inter}")
    else:
        print(f"[LAW P1] pass  Phase({len(phases)}) disjoint Opcode({len(opcodes)}) -> empty")
    # LAW R1 — recursive cycle Node -> PhaseTick -> PhaseStep -> Node
    cyc = ("PhaseTick" in _refs(defs["Node"])
           and "PhaseStep" in _refs(defs["PhaseTick"])
           and "Node" in _refs(defs["PhaseStep"]))
    if cyc:
        print("[LAW R1] pass  Node -> PhaseTick -> PhaseStep -> Node cycle present")
    else:
        ok = False; print("[LAW R1] FAIL recursive tick cycle missing in schema")
    return ok

def check_glyph_nativity():
    # LAW G1 — the glyph IS the token: source codepoints == declared unicode.codepoints
    # (source <-> unicode <-> lexical token identity; rendering is a separate projection).
    ok = True; n = 0
    for ex in EXAMPLES:
        doc = json.load(open(ex, encoding="utf-8"))
        for atom in doc.get("context", {}).get("atoms", []):
            n += 1
            got = [f"U+{ord(c):04X}" for c in atom["source"]]
            want = [c.upper() for c in atom["unicode"]["codepoints"]]
            if got != want:
                ok = False
                print(f"[LAW G1] FAIL {os.path.basename(ex)} '{atom['source']}': {got} != {want}")
    if n == 0:
        print("[LAW G1] (no glyph_atom instances to check)")
    elif ok:
        print(f"[LAW G1] pass  {n} glyph_atom(s): source codepoints == declared unicode (byte-stable)")
    return ok

if __name__ == "__main__":
    j = check_json()
    e = check_ebnf()
    L = check_laws()
    G = check_glyph_nativity()
    print("[done]", "ALL PASS" if (j and e and L and G) else "FAILURES ABOVE")
    sys.exit(0 if (j and e and L and G) else 1)
