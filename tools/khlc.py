#!/usr/bin/env python3
"""khlc.py — KHL → KAST → KSON compiler.

Compiles K'UHUL Language (.khl) driver/source semantics into a canonical
KAST document (protocol kast/1) serialized as KSON (JSON).

Pipeline:
    opengl.khl → khlc → opengl.kson (KastDocument + @driver contract)
    runtime: load KSON → validate schema → reconstruct KAST → check KHL ABI
             → resolve provider → mount capabilities → enter Pop

KHL grammar (line-oriented, glyph source):
    /* comments */
    @bind manifest.path → VAR          # bind manifest data
    glyph ns::name(ARGS) →             # glyph (driver) definition
      op::call(ARGS) → RESULT          # function call / assignment
      if cond :: ... done              # conditional
      for each X in COLL :: ... done   # loop
      yield EXPR                       # return

Usage:
    python tools/khlc.py fold.khl                  # → fold.kson
    python tools/khlc.py fold.khl -o out.kson
    python tools/khlc.py --compile-dir dir/        # all .khl in dir
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# ── KHL tokenizer ─────────────────────────────────────────────────────────────

def strip_comments(src: str) -> str:
    """Remove /* ... */ (including multiline) and // line comments."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def parse_binds(lines):
    """@bind manifest.path → VAR  →  {VAR: "manifest.path"}"""
    binds = {}
    for ln in lines:
        m = re.match(r"@bind\s+([^\s→]+)\s*→\s*([A-Za-z_][A-Za-z0-9_]*)", ln)
        if m:
            binds[m.group(2)] = m.group(1)
    return binds


def parse_calls(body_lines):
    """Return list of statement dicts for a glyph body.

    Handles:  op::call(ARGS) → RESULT   (also RESULT = op::call)
              if cond :: ... done
              for each X in COLL :: ... done
              yield EXPR
              VAR = EXPR / VAR.set(...) / map:new etc.
    """
    stmts = []
    i = 0
    n = len(body_lines)
    while i < n:
        ln = body_lines[i].strip()
        i += 1
        if not ln:
            continue

        # yield EXPR
        if ln.startswith("yield"):
            stmts.append({"kind": "yield", "expr": ln[5:].strip()})
            continue

        # for each X in COLL ::
        m = re.match(r"for\s+each\s+([A-Za-z_][\w, ]*?)\s+in\s+(.+)\s*::\s*$", ln)
        if m:
            inner = []
            depth = 1
            while i < n and depth > 0:
                sub = body_lines[i].strip()
                i += 1
                if sub.startswith("for each") or sub.startswith("for "):
                    depth += 1
                if sub == "done":
                    depth -= 1
                    if depth == 0:
                        break
                inner.append(sub)
            stmts.append({
                "kind": "loop",
                "var": m.group(1),
                "collection": m.group(2),
                "body": parse_calls(inner),
            })
            continue

        # if cond ::
        if ln.startswith("if ") and ln.rstrip().endswith("::"):
            cond = ln[3:-2].strip()
            inner = []
            depth = 1
            while i < n and depth > 0:
                sub = body_lines[i].strip()
                i += 1
                if sub.startswith("if "):
                    depth += 1
                if sub == "done":
                    depth -= 1
                    if depth == 0:
                        break
                inner.append(sub)
            stmts.append({
                "kind": "if",
                "cond": cond,
                "body": parse_calls(inner),
            })
            continue

        # op::call(ARGS) → RESULT     (call arrow)
        m = re.match(r"([A-Za-z_][\w:]*)\(([^)]*)\)\s*→\s*([A-Za-z_]\w*)", ln)
        if m:
            stmts.append({
                "kind": "call",
                "op": m.group(1),
                "args": [a.strip() for a in m.group(2).split(",") if a.strip()],
                "result": m.group(3),
            })
            continue

        # RESULT = op::call(ARGS)     (assignment)
        m = re.match(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z_][\w:]*)\(([^)]*)\)", ln)
        if m:
            stmts.append({
                "kind": "call",
                "op": m.group(2),
                "args": [a.strip() for a in m.group(3).split(",") if a.strip()],
                "result": m.group(1),
            })
            continue

        # bare statement (VAR.set(...), stack:new, etc.)
        stmts.append({"kind": "stmt", "text": ln})

    return stmts


def parse_glyphs(lines):
    """Parse glyph definitions into (name, args, body_stmts) tuples."""
    glyphs = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        i += 1
        m = re.match(r"glyph\s+([A-Za-z_][\w:]*)(?:\(([^)]*)\))?\s*→\s*$", ln)
        if not m:
            continue
        name = m.group(1)
        args = [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]
        body = []
        while i < n and not lines[i].strip().startswith("glyph "):
            body.append(lines[i])
            i += 1
        glyphs.append((name, args, parse_calls(body)))
    return glyphs


# ── KAST document builder ────────────────────────────────────────────────────

def fold_for_glyph(name: str) -> str:
    """Heuristic phase assignment for a glyph (driver contract may override)."""
    lname = name.lower()
    if any(k in lname for k in ("load", "perceive", "bind", "probe", "check")):
        return "Pop"
    if any(k in lname for k in ("represent", "build", "create", "init")):
        return "Wo"
    if any(k in lname for k in ("plan", "predict", "route")):
        return "Yax"
    if any(k in lname for k in ("compute", "dispatch", "execute", "process", "attend")):
        return "Sek"
    if any(k in lname for k in ("project", "output", "collect", "status")):
        return "Ch'en"
    if any(k in lname for k in ("commit", "consolidate", "store", "replay", "watchdog")):
        return "Xul"
    return "Sek"  # default: execute lane


def build_kast(source: str, source_id: str, driver: dict) -> dict:
    """Compile KHL source text into a KAST document (protocol kast/1)."""
    clean = strip_comments(source)
    lines = [l for l in clean.splitlines() if l.strip()]
    binds = parse_binds(lines)
    glyphs = parse_glyphs(lines)

    nodes = []
    edges = []
    node_id = 0
    edge_id = 0
    entry = None

    # @bind statements → attribute nodes
    for var, path in binds.items():
        nodes.append({
            "id": f"n{node_id}",
            "kind": "bind",
            "fold": "Pop",
            "lane": "config",
            "glyph": "bind",
            "opcode": "BIND",
            "symbol": path,
            "type": "manifest_ref",
            "operands": [var],
            "attributes": {"manifest_path": path},
        })
        node_id += 1

    for name, args, stmts in glyphs:
        gid = f"n{node_id}"
        node_id += 1
        if entry is None:
            entry = gid
        nodes.append({
            "id": gid,
            "kind": "glyph",
            "fold": fold_for_glyph(name),
            "lane": "driver",
            "glyph": name,
            "opcode": "GLYPH",
            "symbol": name,
            "type": "driver_glyph",
            "operands": args,
            "attributes": {"driver": driver.get("provider", "")},
        })
        prev = gid

        def walk(stmts, parent):
            nonlocal node_id, edge_id
            for s in stmts:
                if s["kind"] == "call":
                    cid = f"n{node_id}"
                    node_id += 1
                    nodes.append({
                        "id": cid,
                        "kind": "call",
                        "fold": fold_for_glyph(s["op"]),
                        "lane": "compute",
                        "glyph": s["op"],
                        "opcode": "CALL",
                        "symbol": s["op"],
                        "type": "operator_call",
                        "operands": s["args"],
                        "attributes": {"result": s.get("result", "")},
                    })
                    edges.append({
                        "id": f"e{edge_id}", "from": parent, "to": cid,
                        "kind": "control", "label": "call", "ordinal": edge_id,
                    })
                    edge_id += 1
                    parent = cid
                elif s["kind"] in ("if", "loop"):
                    nid = f"n{node_id}"
                    node_id += 1
                    nodes.append({
                        "id": nid,
                        "kind": s["kind"],
                        "fold": "Sek",
                        "lane": "control",
                        "glyph": s["kind"],
                        "opcode": s["kind"].upper(),
                        "symbol": s.get("cond", s.get("collection", "")),
                        "type": "control_flow",
                        "operands": [],
                        "attributes": {},
                    })
                    edges.append({
                        "id": f"e{edge_id}", "from": parent, "to": nid,
                        "kind": "control", "label": s["kind"], "ordinal": edge_id,
                    })
                    edge_id += 1
                    parent = nid
                    walk(s.get("body", []), nid)
                elif s["kind"] == "yield":
                    yid = f"n{node_id}"
                    node_id += 1
                    nodes.append({
                        "id": yid,
                        "kind": "yield",
                        "fold": "Xul",
                        "lane": "output",
                        "glyph": "yield",
                        "opcode": "YIELD",
                        "symbol": s["expr"],
                        "type": "return_value",
                        "operands": [],
                        "attributes": {},
                    })
                    edges.append({
                        "id": f"e{edge_id}", "from": parent, "to": yid,
                        "kind": "control", "label": "yield", "ordinal": edge_id,
                    })
                    edge_id += 1
                elif s["kind"] == "stmt":
                    sid = f"n{node_id}"
                    node_id += 1
                    nodes.append({
                        "id": sid,
                        "kind": "stmt",
                        "fold": "Sek",
                        "lane": "compute",
                        "glyph": "expr",
                        "opcode": "STMT",
                        "symbol": s["text"],
                        "type": "expression",
                        "operands": [],
                        "attributes": {},
                    })
                    edges.append({
                        "id": f"e{edge_id}", "from": parent, "to": sid,
                        "kind": "control", "label": "next", "ordinal": edge_id,
                    })
                    edge_id += 1
                    parent = sid

        walk(stmts, gid)

    # semantic hash over the node/edge structure
    sem = hashlib.sha256(
        json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True).encode()
    ).hexdigest()

    doc = {
        "protocol": "kast/1",
        "registry_hash": hashlib.sha256(source.encode()).hexdigest(),
        "source_kind": "khl",
        "source_id": source_id,
        "entry_node_id": entry,
        "nodes": nodes,
        "edges": edges,
        "semantic_hash": sem,
        "@driver": driver,
    }
    return doc


# ── Driver contract from CLI ─────────────────────────────────────────────────

def default_driver(source_id: str) -> dict:
    stem = Path(source_id).stem if source_id else "driver"
    return {
        "@abi": 1,
        "@requires": {"kuhul": ">= 1.0", "khl_abi": 1, "scxq2": ">= 2.0"},
        "@capabilities": ["tensor.map"],
        "@phase_hooks": {"Sek": "dispatch", "Ch'en": "collect_status", "Xul": "commit_tensor_state"},
        "@provider": stem,
        "@resources": [],
        "@hash": "",
    }


# ── main ─────────────────────────────────────────────────────────────────────

def compile_file(path: Path, out: Path = None, driver: dict = None):
    source = path.read_text(encoding="utf-8")
    d = driver or default_driver(path.name)
    kast = build_kast(source, path.name, d)
    kast["@driver"]["@hash"] = kast["semantic_hash"]
    out_path = out or path.with_suffix(".kson")
    out_path.write_text(json.dumps(kast, indent=2), encoding="utf-8")
    n_glyphs = sum(1 for n in kast["nodes"] if n["kind"] == "glyph")
    print(f"[khlc] {path.name} -> {out_path.name}  "
          f"({len(kast['nodes'])} nodes, {len(kast['edges'])} edges, "
          f"{n_glyphs} glyphs, fold={kast['nodes'][0]['fold'] if kast['nodes'] else '?'})")


def main():
    ap = argparse.ArgumentParser(description="KHL → KAST → KSON compiler")
    ap.add_argument("input", nargs="+", help=".khl file(s) or a directory")
    ap.add_argument("-o", "--out", help="output .kson path (single input only)")
    ap.add_argument("--abi", type=int, default=1, help="KHL ABI version")
    ap.add_argument("--provider", default=None, help="driver provider id")
    ap.add_argument("--compile-dir", action="store_true",
                    help="compile every .khl in the input directory")
    args = ap.parse_args()

    files = []
    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.khl")))
        else:
            files.append(p)

    if not files:
        print("no .khl files found", file=sys.stderr)
        sys.exit(1)

    for f in files:
        driver = default_driver(f.name)
        if args.provider:
            driver["@provider"] = args.provider
        driver["@abi"] = args.abi
        out = Path(args.out) if (args.out and len(files) == 1) else None
        compile_file(f, out, driver)


if __name__ == "__main__":
    main()
