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


INCLUDE_RE = re.compile(r'include\s+"([^"]+)"')


def expand_includes(source: str, source_path: Path, seen=None) -> list:
    """Inline `include "file.kuhul"` statements (recursive, cycle-guarded).

    Returns a flat list of source lines. Includes resolve relative to the
    including file's directory; each file is included at most once.
    """
    if seen is None:
        seen = set()
    out = []
    for ln in source.splitlines():
        m = INCLUDE_RE.match(ln.strip())
        if m:
            inc = Path(source_path).parent / m.group(1)
            key = str(inc.resolve())
            if key in seen or not inc.exists():
                continue
            seen.add(key)
            inc_src = inc.read_text(encoding="utf-8")
            out.extend(expand_includes(inc_src, inc, seen))
        else:
            out.append(ln)
    return out


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

PHASE_GLYPH = {
    "bind": "BIND", "probe": "PROBE", "resolve": "RESOLVE",
    "dispatch": "DISPATCH", "collect_status": "COLLECT",
    "commit": "COMMIT", "yield": "YIELD",
}

PHASE_BLOCK_RE = re.compile(r"⟁\s*([A-Za-z'’]+)\s*⟁?$")


def parse_phase_blocks(lines):
    """Parse KLSL-style phase blocks:  ⟁ Pop ⟁  ...  ⟁ Wo ⟁  ...

    Returns [(phase, [stmt_dicts])]. Statements:
        bind VAR = EXPR                     -> BIND
        probe geometry                      -> PROBE
        resolve provider = geometry.compute -> RESOLVE
        dispatch provider(AREA)             -> DISPATCH
        collect_status RESULT               -> COLLECT
        commit RESULT                       -> COMMIT
        op::call(ARGS) -> RESULT            -> CALL
        yield EXPR                          -> YIELD
    """
    blocks = []
    current_phase = None
    body = []
    for raw in lines:
        s = raw.strip()
        m = PHASE_BLOCK_RE.match(s)
        if m:
            if current_phase is not None:
                blocks.append((current_phase, parse_phase_stmts(body)))
            current_phase = m.group(1)
            body = []
            continue
        if s and not s.startswith("@bind"):
            body.append(s)
    if current_phase is not None:
        blocks.append((current_phase, parse_phase_stmts(body)))
    return blocks


def parse_phase_stmts(body):
    stmts = []
    for ln in body:
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"bind\s+([^\s=]+)(?:\s*=\s*(.+))?", s)
        if m:
            stmts.append({"kind": "bind", "var": m.group(1),
                          "value": (m.group(2) or "").strip()})
            continue
        m = re.match(r"([a-z_]+)\(([^)]*)\)", s)
        if m and m.group(1) in PHASE_GLYPH:
            stmts.append({"kind": "verb", "verb": m.group(1),
                          "target": m.group(1),
                          "args": [a.strip() for a in m.group(2).split(",") if a.strip()]})
            continue
        m = re.match(r"([a-z_]+)\s+(.+)", s)
        if m and m.group(1) in PHASE_GLYPH:
            stmts.append({"kind": "verb", "verb": m.group(1),
                          "target": m.group(2).strip(), "args": []})
            continue
        m = re.match(r"([A-Za-z_][\w:]*)\(([^)]*)\)\s*→\s*([A-Za-z_]\w*)", s)
        if m:
            stmts.append({"kind": "call", "op": m.group(1),
                          "args": [a.strip() for a in m.group(2).split(",") if a.strip()],
                          "result": m.group(3)})
            continue
        if s.startswith("yield"):
            stmts.append({"kind": "yield", "expr": s[5:].strip()})
            continue
        stmts.append({"kind": "stmt", "text": s})
    return stmts


def fold_for_glyph(name: str) -> str:
    """Heuristic phase assignment for a glyph (driver contract may override)."""
    lname = name.lower()
    if "manifold" in lname:
        return "Ch'en"  # (before 'fold' match — 'manifold' contains 'fold')
    if any(k in lname for k in ("load", "perceive", "bind", "probe", "check",
                                "current", "read", "state")):
        return "Pop"
    if any(k in lname for k in ("represent", "build", "create", "init",
                                "legal", "validate", "define")):
        return "Wo"
    if any(k in lname for k in ("plan", "predict", "route", "schedule",
                                "transition", "decide")):
        return "Yax"
    if any(k in lname for k in ("compute", "dispatch", "execute", "process",
                                "attend", "fold", "run")):
        return "Sek"
    if any(k in lname for k in ("project", "output", "collect", "status",
                                "manifold", "emit", "report")):
        return "Ch'en"
    if any(k in lname for k in ("commit", "consolidate", "store", "replay",
                                "watchdog", "persist")):
        return "Xul"
    return "Sek"  # default: execute lane


def build_kast(source: str, source_id: str, driver: dict, source_path: Path = None) -> dict:
    """Compile KHL source text into a KAST document (protocol kast/1)."""
    lines = source.splitlines()
    if source_path is not None:
        lines = expand_includes(source, source_path)
    clean = strip_comments("\n".join(lines))
    lines = [l for l in clean.splitlines() if l.strip()]
    binds = parse_binds(lines)

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

    # ⟁ Phase-block form (pi.kuhul reference program): each block is a
    # fold-annotated node cluster — the phase engine sees the exact
    # Pop→Wo→Yax→Sek→Ch'en→Xul legality inline.
    if "⟁" in source:
        blocks = parse_phase_blocks(lines)
        for phase, stmts in blocks:
            for s in stmts:
                nid = f"n{node_id}"
                node_id += 1
                if entry is None:
                    entry = nid
                if s["kind"] == "bind":
                    nodes.append({
                        "id": nid, "kind": "bind", "fold": phase,
                        "lane": "config", "glyph": "bind", "opcode": "BIND",
                        "symbol": s["var"], "type": "constant",
                        "operands": [s["var"]],
                        "attributes": {"value": s["value"]},
                    })
                elif s["kind"] == "verb":
                    opcode = PHASE_GLYPH.get(s["verb"], "CALL")
                    nodes.append({
                        "id": nid, "kind": "call", "fold": phase,
                        "lane": "compute", "glyph": s["verb"], "opcode": opcode,
                        "symbol": s["target"], "type": "operator_call",
                        "operands": s["args"],
                        "attributes": {"result": ""},
                    })
                elif s["kind"] == "call":
                    nodes.append({
                        "id": nid, "kind": "call", "fold": phase,
                        "lane": "compute", "glyph": s["op"], "opcode": "CALL",
                        "symbol": s["op"], "type": "operator_call",
                        "operands": s["args"],
                        "attributes": {"result": s.get("result", "")},
                    })
                elif s["kind"] == "yield":
                    nodes.append({
                        "id": nid, "kind": "yield", "fold": phase,
                        "lane": "output", "glyph": "yield", "opcode": "YIELD",
                        "symbol": s["expr"], "type": "return_value",
                        "operands": [], "attributes": {},
                    })
                else:
                    nodes.append({
                        "id": nid, "kind": "stmt", "fold": phase,
                        "lane": "compute", "glyph": "expr", "opcode": "STMT",
                        "symbol": s["text"], "type": "expression",
                        "operands": [], "attributes": {},
                    })
                # control edge from previous node (phase legality is visible in folds)
                if len(nodes) > 1:
                    prev = nodes[-2]["id"]
                    edges.append({
                        "id": f"e{edge_id}", "from": prev, "to": nid,
                        "kind": "control", "label": "next", "ordinal": edge_id,
                    })
                    edge_id += 1
    glyphs = parse_glyphs(lines)
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
    stem = stem.lower()
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

# Phase law: Pop -> Wo -> Yax -> Sek -> Ch'en -> Xul -> Pop (wrap)
PHASE_CYCLE = ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"]
PHASE_IDX = {p: i for i, p in enumerate(PHASE_CYCLE)}

# Built-in op namespaces (declared capabilities are added on top)
BUILTIN_OPS = {"gl", "tensor", "map", "list", "stack", "attention", "system",
               "gpt2", "fold", "manifest", "memory", "os", "math",
               "state", "manifold", "phase", "fold"}

SUPPORTED_ABI = {1}


def check_kast(kast: dict) -> tuple:
    """Static driver checks. Returns (errors, warnings).

    - structural: duplicate/missing node ids, edges to missing nodes
    - unreachable: statements after a yield in the same control chain
    - phase: @phase_hooks must follow the legal cycle; call folds outside
      the glyph's phase are suspicious (provider call from wrong fold)
    - capabilities: call op namespace must be built-in or declared
    """
    errors, warnings = [], []
    nodes = kast.get("nodes", [])
    edges = kast.get("edges", [])
    driver = kast.get("@driver", {})

    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        errors.append("duplicate node ids")
    id_set = set(ids)
    for e in edges:
        if e.get("from") not in id_set:
            errors.append(f"edge {e.get('id')}: missing 'from' node {e.get('from')}")
        if e.get("to") not in id_set:
            errors.append(f"edge {e.get('id')}: missing 'to' node {e.get('to')}")

    # unreachable: node following a yield in the same body (no incoming edge)
    yield_ids = {n["id"] for n in nodes if n["kind"] == "yield"}
    from_ids = {e.get("from") for e in edges}
    to_ids = {e.get("to") for e in edges}
    for n in nodes:
        if n["id"] not in to_ids and n["kind"] not in ("bind", "glyph"):
            warnings.append(f"node {n['id']} ({n['symbol'][:30]}) has no incoming edge")

    # phase hooks must follow the legal cycle
    hooks = driver.get("@phase_hooks", {})
    hook_order = [p for p in PHASE_CYCLE if p in hooks]
    for i in range(len(hook_order) - 1):
        a, b = hook_order[i], hook_order[i + 1]
        if PHASE_IDX[b] != (PHASE_IDX[a] + 1) % 6:
            errors.append(f"illegal phase jump in @phase_hooks: {a} -> {b}")

    # per-glyph: call folds should stay within the glyph's phase family
    decl_caps = set(driver.get("@capabilities", []))
    for n in nodes:
        if n["kind"] != "glyph":
            continue
        glyph_fold = n["fold"]
        for e in edges:
            if e.get("from") != n["id"]:
                continue
            child = next((x for x in nodes if x["id"] == e["to"]), None)
            if child and child["kind"] in ("call", "stmt") and child["fold"] != glyph_fold:
                warnings.append(
                    f"{n['symbol']} (fold {glyph_fold}) calls {child['symbol'][:30]} "
                    f"from fold {child['fold']} — provider call from another fold")

    # capabilities: call op namespace must be built-in or declared
    for n in nodes:
        if n["kind"] != "call":
            continue
        sym = n["symbol"]
        ns = sym.split("::")[0] if "::" in sym else sym.split(":")[0]
        if ns not in BUILTIN_OPS and ns not in decl_caps and ns != driver.get("@provider", ""):
            warnings.append(f"undeclared capability: {sym} (namespace '{ns}' not built-in or @capabilities)")

    # ABI gate
    if driver.get("@abi") not in SUPPORTED_ABI:
        errors.append(f"unsupported KHL ABI: {driver.get('@abi')} (supported: {sorted(SUPPORTED_ABI)})")

    # semantic hash recompute
    sem = hashlib.sha256(
        json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True).encode()
    ).hexdigest()
    if sem != kast.get("semantic_hash"):
        errors.append("semantic_hash mismatch (nodes/edges changed after compile)")
    if driver.get("@hash") != kast.get("semantic_hash"):
        errors.append("@driver.@hash != semantic_hash")

    return errors, warnings


def compile_file(path: Path, out: Path = None, driver: dict = None):
    source = path.read_text(encoding="utf-8")
    d = driver or default_driver(path.name)
    kast = build_kast(source, path.name, d, source_path=path)
    kast["@driver"]["@hash"] = kast["semantic_hash"]
    # record includes for provenance
    includes = [ln.strip() for ln in source.splitlines()
                if INCLUDE_RE.match(ln.strip())]
    if includes:
        kast["includes"] = [INCLUDE_RE.match(i).group(1) for i in includes]
    errors, warnings = check_kast(kast)
    out_path = out or path.with_suffix(".kson")
    out_path.write_text(json.dumps(kast, indent=2), encoding="utf-8")
    n_glyphs = sum(1 for n in kast["nodes"] if n["kind"] == "glyph")
    status = "OK" if not errors else "FAIL"
    print(f"[khlc] {path.name} -> {out_path.name}  "
          f"({len(kast['nodes'])} nodes, {len(kast['edges'])} edges, "
          f"{n_glyphs} glyphs) [{status}]")
    for w in warnings:
        print(f"  ! warn: {w}")
    for e in errors:
        print(f"  x err : {e}")
    return errors


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
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
            files.extend(sorted(p.glob("*.kuhul")))
        else:
            files.append(p)

    if not files:
        print("no .khl files found", file=sys.stderr)
        sys.exit(1)

    failed = False
    for f in files:
        driver = default_driver(f.name)
        if args.provider:
            driver["@provider"] = args.provider
        driver["@abi"] = args.abi
        out = Path(args.out) if (args.out and len(files) == 1) else None
        if compile_file(f, out, driver):
            failed = True
    if failed:
        print("[khlc] static checks FAILED — driver(s) rejected", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
