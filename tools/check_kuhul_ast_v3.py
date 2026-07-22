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
    text = re.sub(r'"[^"]*"', " ", text)                           # strip "double" terminals
    text = re.sub(r"'[^']*'", " ", text)                           # strip 'single' terminals
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

if __name__ == "__main__":
    j = check_json()
    e = check_ebnf()
    print("[done]", "ALL PASS" if (j and e) else "FAILURES ABOVE")
    sys.exit(0 if (j and e) else 1)
