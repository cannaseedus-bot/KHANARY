#!/usr/bin/env python3
"""kson_validate.py — KSON driver admission gate.

Strict admission for KHL/KAST/KSON driver objects (.kson). A driver is either
ADMITTED (mounted, phase engine enters Pop) or REJECTED — never phase execution
on an invalid driver.

Admission sequence (per docs/KUHUL_RUNTIME.md):
    load .kson
    -> verify protocol == kast/1
    -> validate KastDocument schema
    -> verify semantic_hash
    -> verify @driver.@hash
    -> validate @abi
    -> validate requested capabilities/resources
    -> resolve provider
    -> validate phase hooks
    -> mount driver
    -> enter Pop

Usage:
    python tools/kson_validate.py drivers/khl/opengl.kson
    python tools/kson_validate.py drivers/khl/            # all .kson in dir
    python tools/kson_validate.py --registry providers.json
    python tools/kson_validate.py --tamper drivers/khl/opengl.kson  # self-test
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROTOCOL = "kast/1"
SUPPORTED_ABI = {1}
PHASE_CYCLE = ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"]
PHASE_IDX = {p: i for i, p in enumerate(PHASE_CYCLE)}

# Canonical provider registry: driver @provider -> sidecar / service
DEFAULT_PROVIDERS = {
    "opengl":          {"sidecar": "glsl_gpu",      "kind": "xcfe_manifest", "ops": ["glsl_probe", "glsl_compile", "glsl_dispatch", "glsl_info"]},
    "gpt2.runtime":    {"sidecar": "kuhul_engine",  "kind": "engine",        "ops": ["chat", "forward"]},
    "fold":            {"sidecar": "json_runtime",  "kind": "native",        "ops": ["phase", "fold", "transition"]},
    "phase":           {"sidecar": "json_runtime",  "kind": "native",        "ops": ["phase", "transition", "fold", "manifold"]},
    "sw":              {"sidecar": "json_runtime",  "kind": "native",        "ops": ["watchdog"]},
    "inference":       {"sidecar": "kuhul_engine",  "kind": "engine",        "ops": ["chat", "forward"]},
    "attention.fold":  {"sidecar": "json_runtime",  "kind": "native",        "ops": ["attention", "fold"]},
    "pi":               {"sidecar": "json_runtime",  "kind": "native",        "ops": ["bind", "probe", "resolve", "dispatch", "collect_status", "commit"]},
    "gravity":          {"sidecar": "json_runtime",  "kind": "native",        "ops": ["gravity.field", "gravity.acceleration", "gravity.solve", "gravity.state"]},
    "glsl":             {"sidecar": "glsl_gpu",      "kind": "xcfe_manifest", "ops": ["glsl_probe", "glsl_compile", "glsl_dispatch", "glsl_info"]},
    "glsl_gpu":         {"sidecar": "glsl_gpu",      "kind": "xcfe_manifest", "ops": ["glsl_probe", "glsl_compile", "glsl_dispatch", "glsl_info"]},
}

REQUIRED_DOC = {"protocol", "registry_hash", "source_kind", "source_id",
                "entry_node_id", "nodes", "edges", "semantic_hash"}
REQUIRED_NODE = {"id", "kind", "fold", "lane", "glyph", "opcode", "symbol"}
REQUIRED_EDGE = {"id", "from", "to", "kind"}
REQUIRED_DRIVER = {"@abi", "@requires", "@capabilities", "@phase_hooks",
                   "@provider", "@resources", "@hash"}


class AdmissionResult:
    def __init__(self, ok, steps, errors=None, driver=None):
        self.ok = ok
        self.steps = steps          # list of (step, status)
        self.errors = errors or []
        self.driver = driver

    def __repr__(self):
        s = "ADMITTED" if self.ok else "REJECTED"
        return f"<AdmissionResult {s} steps={len(self.steps)} errors={len(self.errors)}>"


def admit(path: Path, providers=None) -> AdmissionResult:
    providers = providers or DEFAULT_PROVIDERS
    is_stdlib = "stdlib" in str(path).replace("\\", "/")
    steps = []
    errors = []

    def step(name, ok, detail=""):
        steps.append((name, "ok" if ok else "fail", detail))
        if not ok:
            errors.append(f"{name}: {detail}")

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return AdmissionResult(False, [("load", "fail", str(e))], [f"load: {e}"])

    # 1. protocol
    step("protocol", doc.get("protocol") == PROTOCOL,
         f"expected {PROTOCOL}, got {doc.get('protocol')!r}")

    # 2. KastDocument schema
    missing_doc = REQUIRED_DOC - set(doc)
    step("schema.doc", not missing_doc, f"missing keys: {sorted(missing_doc)}")
    for i, n in enumerate(doc.get("nodes", [])):
        missing_n = REQUIRED_NODE - set(n)
        if missing_n:
            step("schema.node", False, f"node[{i}] missing: {sorted(missing_n)}")
    for i, e in enumerate(doc.get("edges", [])):
        missing_e = REQUIRED_EDGE - set(e)
        if missing_e:
            step("schema.edge", False, f"edge[{i}] missing: {sorted(missing_e)}")
    if not steps or all(s[1] == "ok" for s in steps[:2]):
        pass

    # 3. semantic_hash
    sem = hashlib.sha256(
        json.dumps({"nodes": doc.get("nodes", []), "edges": doc.get("edges", [])},
                   sort_keys=True).encode()).hexdigest()
    step("semantic_hash", sem == doc.get("semantic_hash"),
         f"recomputed {sem[:12]}… != {str(doc.get('semantic_hash'))[:12]}…")

    # 4-8. @driver contract — only for driver KAST (provider bindings).
    # Applications (kuhul-es programs, stdlib modules) have no @driver and
    # mount directly to the phase engine as programs.
    driver = doc.get("@driver")
    is_driver = isinstance(driver, dict) and bool(driver)
    prov = None
    if is_driver:
        step("driver.hash", driver.get("@hash") == doc.get("semantic_hash"),
             "@driver.@hash must equal semantic_hash")
        abi = driver.get("@abi")
        step("abi", abi in SUPPORTED_ABI, f"abi={abi}, supported={sorted(SUPPORTED_ABI)}")
        caps = driver.get("@capabilities", [])
        res = driver.get("@resources", [])
        step("capabilities", isinstance(caps, list) and len(caps) > 0,
             f"capabilities={caps}")
        step("resources", isinstance(res, list), f"resources={res}")
        provider = driver.get("@provider")
        prov = providers.get(provider)
        if prov is None and is_stdlib:
            prov = {"sidecar": "json_runtime", "kind": "native",
                    "ops": ["phase", "fold", "dispatch"]}
        step("provider", prov is not None,
             f"provider={provider!r} not in registry "
             f"({', '.join(sorted(providers))})" + (" [stdlib -> native]" if prov else ""))
        hooks = driver.get("@phase_hooks", {})
        hook_order = [p for p in PHASE_CYCLE if p in hooks]
        legal = True
        for i in range(len(hook_order) - 1):
            a, b = hook_order[i], hook_order[i + 1]
            if PHASE_IDX[b] != (PHASE_IDX[a] + 1) % 6:
                legal = False
                break
        step("phase_hooks", legal, f"hooks={hook_order}")
    else:
        steps.append(("driver", "skip", "application KAST — no @driver contract; mounts as program"))

    # 9. entry node exists
    entry = doc.get("entry_node_id")
    node_ids = {n["id"] for n in doc.get("nodes", [])}
    step("entry", entry in node_ids, f"entry_node_id={entry!r}")

    ok = not errors
    if ok:
        if prov:
            steps.append(("mount", "ok", f"driver '{provider}' -> sidecar "
                         f"{prov['sidecar']} ({prov['kind']})"))
        else:
            steps.append(("mount", "ok", "application -> canonical phase engine (json_runtime native)"))
        steps.append(("enter_pop", "ok", "phase engine entered Pop"))
    return AdmissionResult(ok, steps, errors, doc)


def main():
    ap = argparse.ArgumentParser(description="KSON driver admission gate")
    ap.add_argument("input", nargs="+", help=".kson file(s) or a directory")
    ap.add_argument("--registry", default=None, help="provider registry JSON")
    ap.add_argument("--tamper", action="store_true",
                    help="self-test: flip a byte in each file to prove REJECT")
    ap.add_argument("-v", "--verbose", action="store_true", help="show every step")
    args = ap.parse_args()

    providers = DEFAULT_PROVIDERS
    if args.registry:
        providers = json.loads(Path(args.registry).read_text(encoding="utf-8"))

    files = []
    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.kson")))
        else:
            files.append(p)

    all_ok = True
    for f in files:
        if args.tamper:
            # prove the gate rejects tampered KSON
            txt = f.read_text(encoding="utf-8")
            flipped = txt.replace("\"protocol\": \"kast/1\"",
                                  "\"protocol\": \"kast/2\"", 1)
            f.write_text(flipped, encoding="utf-8")
            r = admit(f, providers)
            f.write_text(txt, encoding="utf-8")  # restore
            print(f"[tamper-test] {f.name}: protocol tamper -> "
                  f"{'REJECTED (correct)' if not r.ok else 'ADMITTED (WRONG!)'}")
            if r.ok:
                all_ok = False
            continue

        r = admit(f, providers)
        tag = "ADMITTED" if r.ok else "REJECTED"
        drv = (r.driver or {}).get("@driver") or {}
        prov = drv.get("@provider", "application" if not drv else "?")
        print(f"[kson] {f.name}: {tag}  (provider={prov})")
        if args.verbose:
            for name, status, detail in r.steps:
                print(f"    {status:4s} {name}: {detail}")
        if not r.ok:
            all_ok = False
            for e in r.errors:
                print(f"    x {e}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()


# ── driver-only verification (cross-language port of driver-kast.js verifyDriverOnly) ──
def verify_driver_only(doc, runtime_caps):
    """Verify a driver-only KAST against runtime capabilities.
    Returns (admitted, reason)."""
    driver = doc.get("@driver", {})
    admission = doc.get("@admission", {})
    if driver.get("@abi") != runtime_caps.get("abi"):
        return False, f"ABI mismatch: {driver.get('@abi')} != {runtime_caps.get('abi')}"
    wl = runtime_caps.get("provider_whitelist")
    if wl and driver.get("@provider") not in wl:
        return False, f"Provider {driver.get('@provider')} not in whitelist"
    for cap in driver.get("@capabilities", []):
        if cap not in runtime_caps.get("capabilities", []):
            return False, f"Missing capability: {cap}"
    mn = runtime_caps.get("max_nodes")
    if mn and admission.get("max_nodes", 0) > mn:
        return False, f"Node count {admission.get('max_nodes')} exceeds limit {mn}"
    mm = runtime_caps.get("max_memory")
    rl = admission.get("resource_limits", {})
    if mm and rl.get("max_memory_mb", 0) > mm:
        return False, f"Memory limit {rl.get('max_memory_mb')}MB exceeds {mm}MB"
    # tamper check: recompute hash over contract minus hash/signature
    contract = {k: v for k, v in driver.items() if k not in ("@hash", "@signature")}
    canon = json.dumps(contract, sort_keys=True)
    computed = hashlib.sha256(canon.encode()).hexdigest()
    if computed != driver.get("@hash"):
        return False, "Driver contract hash mismatch - possible tampering"
    return True, "admitted"
