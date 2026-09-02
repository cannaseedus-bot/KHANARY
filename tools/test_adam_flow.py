"""test_adam_flow.py — end-to-end test for the adam.kuhul training flow.

Stages
------
1. kxc compile  : drivers/klsl/adam.kuhul → kernelClass=adam_optimizer, registryMatched=true
2. xshard_backward : test.xshard + synthetic token bin → grad.xshard
3. xshard_adapt (grad-xshard) : dry-run Adam update via real grad xshard (no --apply)
4. xshard_adapt (grad-scale)  : dry-run Adam update via synthetic probe (no --apply)
5. adam_ctypes smoke          : Python AdamOptimizer + FoldArcOptimizer

Usage
-----
    python tools/test_adam_flow.py
    python tools/test_adam_flow.py --stage 2          # run one stage
    python tools/test_adam_flow.py --stop-on-fail      # abort on first failure
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root and binary paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
KXC      = ROOT / "versions/kxc-v1.0.0/bin/kxc.exe"
REGISTRY = ROOT / "versions/kxc-v1.0.0/registry"
ADAM_KUHUL   = ROOT / "drivers/klsl/adam.kuhul"
XSHARD_BACK  = ROOT / "trainer/build/Release/xshard_backward.exe"
XSHARD_ADAPT  = ROOT / "trainer/build/Release/xshard_adapt.exe"
ADAPT_SHADER  = ROOT / "trainer/build/shaders/xshard_adapt_fold.cso"
TEST_XSHARD  = ROOT / "trainer/test.xshard"
ADAM_DLL     = ROOT / "versions/khlc-v1.0.0/bin/Adam.dll"

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
_results: list[tuple[int, str, bool, str]] = []

def _record(stage: int, name: str, ok: bool, detail: str = "") -> None:
    _results.append((stage, name, ok, detail))
    sym = "PASS" if ok else "FAIL"
    tag = f"[{sym}]  stage {stage}: {name}"
    print(tag + (f"  — {detail}" if detail else ""))

def _run(cmd: list[str | Path], **kw) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def stage1_kxc_compile(outdir: Path) -> bool:
    if not KXC.exists():
        _record(1, "kxc binary", False, f"not found: {KXC}")
        return False
    if not ADAM_KUHUL.exists():
        _record(1, "adam.kuhul exists", False, str(ADAM_KUHUL))
        return False

    r = _run([KXC, ADAM_KUHUL, "--outdir", outdir, "--registry", REGISTRY])
    if r.returncode != 0:
        _record(1, "kxc compile", False, r.stderr.strip()[:200])
        return False

    smca_files = list(outdir.glob("*.smca.json"))
    if not smca_files:
        _record(1, "smca.json produced", False, "no .smca.json in output dir")
        return False

    data = json.loads(smca_files[0].read_text())
    kc = data.get("smca", {}).get("kernelClass", "")
    rm = data.get("smca", {}).get("registryMatched", False)

    ok_class = kc == "adam_optimizer"
    ok_reg   = rm is True
    _record(1, "kernelClass=adam_optimizer", ok_class, f"got: {kc!r}")
    _record(1, "registryMatched=true",       ok_reg,   f"got: {rm!r}")
    return ok_class and ok_reg


def stage2_xshard_backward(tmpdir: Path) -> tuple[bool, Path]:
    grad_xshard = tmpdir / "adam_test_grad.xshard"

    if not XSHARD_BACK.exists():
        _record(2, "xshard_backward binary", False, str(XSHARD_BACK))
        return False, grad_xshard

    if not TEST_XSHARD.exists():
        _record(2, "test.xshard exists", False, str(TEST_XSHARD))
        return False, grad_xshard

    # Write a minimal synthetic token bin (256 random-ish bytes so stream_token_signal
    # produces a deterministic non-zero signal)
    token_bin = tmpdir / "synthetic_tokens.bin"
    payload = bytes([(i * 7 + 13) & 0xFF for i in range(256)])
    token_bin.write_bytes(payload)

    r = _run([
        XSHARD_BACK, TEST_XSHARD,
        "--token-bin", token_bin,
        "--output", grad_xshard,
        "--max-shards", "3",
        "--grad-scale", "1e-3",
    ])

    if r.returncode != 0:
        _record(2, "xshard_backward exit 0", False, r.stderr.strip()[:200])
        return False, grad_xshard

    if not grad_xshard.exists() or grad_xshard.stat().st_size < 64:
        _record(2, "grad xshard produced", False, "file missing or too small")
        return False, grad_xshard

    # Verify XSHD magic in gradient xshard
    magic = grad_xshard.read_bytes()[:4]
    ok_magic = magic == b"XSHD"
    _record(2, "grad xshard XSHD magic", ok_magic, f"got: {magic!r}")

    # Parse manifest to confirm @kind
    manifest_len = struct.unpack_from("<I", grad_xshard.read_bytes(), 8)[0]
    try:
        manifest_raw = grad_xshard.read_bytes()[16:16 + manifest_len]
        mj = json.loads(manifest_raw)
        kind = mj.get("@kind", "")
        ok_kind = "gradient" in kind
        _record(2, "grad manifest @kind contains 'gradient'", ok_kind, f"got: {kind!r}")
        n_shards = mj.get("n_shards", 0)
        ok_shards = n_shards > 0
        _record(2, f"grad n_shards > 0", ok_shards, f"got: {n_shards}")
    except Exception as e:
        _record(2, "grad manifest parseable", False, str(e))
        return False, grad_xshard

    return ok_magic and ok_kind and ok_shards, grad_xshard


def stage3_adapt_grad_xshard(grad_xshard: Path, tmpdir: Path) -> bool:
    if not XSHARD_ADAPT.exists():
        _record(3, "xshard_adapt binary", False, str(XSHARD_ADAPT))
        return False
    if not grad_xshard.exists():
        _record(3, "grad xshard available", False, "stage 2 produced no grad xshard")
        return False

    ledger = tmpdir / "adapt_grad.jsonl"
    r = _run([
        XSHARD_ADAPT, TEST_XSHARD,
        "--shader", ADAPT_SHADER,
        "--grad-xshard", grad_xshard,
        "--lr", "1e-5",
        "--max-shards", "3",
        "--ledger", ledger,
        # no --apply -> dry-run
    ])

    ok_exit = r.returncode == 0
    _record(3, "xshard_adapt (grad-xshard) exit 0", ok_exit,
            (r.stderr or r.stdout).strip()[:200] if not ok_exit else "")

    if ok_exit and ledger.exists():
        lines = [l for l in ledger.read_text().splitlines() if l.strip()]
        ok_ledger = len(lines) > 0
        _record(3, "adapt ledger written", ok_ledger, f"{len(lines)} entries")
    else:
        ok_ledger = not ok_exit  # if exit failed, don't penalise ledger
        if ok_exit:
            _record(3, "adapt ledger written", False, "ledger file not created")

    return ok_exit and ok_ledger


def stage4_adapt_grad_scale(tmpdir: Path) -> bool:
    if not XSHARD_ADAPT.exists():
        _record(4, "xshard_adapt binary", False, str(XSHARD_ADAPT))
        return False

    ledger = tmpdir / "adapt_synth.jsonl"
    r = _run([
        XSHARD_ADAPT, TEST_XSHARD,
        "--shader", ADAPT_SHADER,
        "--grad-scale", "1e-4",
        "--lr", "1e-5",
        "--max-shards", "3",
        "--ledger", ledger,
        # no --apply -> dry-run
    ])

    ok_exit = r.returncode == 0
    _record(4, "xshard_adapt (grad-scale) exit 0", ok_exit,
            (r.stderr or r.stdout).strip()[:200] if not ok_exit else "")

    if ok_exit and ledger.exists():
        lines = [l for l in ledger.read_text().splitlines() if l.strip()]
        ok_ledger = len(lines) > 0
        _record(4, "synth adapt ledger written", ok_ledger, f"{len(lines)} entries")
        # check first entry parses as JSON with an "ok" or "applied" field
        try:
            first = json.loads(lines[0])
            has_fields = any(k in first for k in ("ok", "applied", "shard_id", "seq", "kernelClass", "fold"))
            _record(4, "ledger entry has recognisable fields", has_fields, str(list(first.keys()))[:120])
        except Exception as e:
            _record(4, "ledger entry is valid JSON", False, str(e))
    else:
        ok_ledger = not ok_exit
        if ok_exit:
            _record(4, "synth adapt ledger written", False, "ledger file not created")

    return ok_exit and ok_ledger


def stage5_adam_ctypes() -> bool:
    if not ADAM_DLL.exists():
        _record(5, "Adam.dll exists", False, str(ADAM_DLL))
        return False

    script = ROOT / "tools/adam_ctypes.py"
    if not script.exists():
        _record(5, "adam_ctypes.py exists", False, str(script))
        return False

    r = _run([sys.executable, script, "--smoke"])
    ok = r.returncode == 0 and "PASS" in r.stdout
    _record(5, "adam_ctypes --smoke PASS", ok,
            (r.stderr or r.stdout).strip()[-200:] if not ok else
            r.stdout.strip().splitlines()[-1])
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0,
                    help="run only this stage (0 = all)")
    ap.add_argument("--stop-on-fail", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("adam.kuhul training flow — end-to-end test")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="adam_flow_") as td:
        tmpdir = Path(td)
        kxc_out = tmpdir / "kxc_out"
        kxc_out.mkdir()

        ok1 = ok2 = ok3 = ok4 = ok5 = True
        grad_xshard = tmpdir / "adam_test_grad.xshard"

        run_all = args.stage == 0

        if run_all or args.stage == 1:
            print("\n--- Stage 1: kxc compile adam.kuhul ---")
            ok1 = stage1_kxc_compile(kxc_out)
            if not ok1 and args.stop_on_fail:
                return _summary()

        if run_all or args.stage == 2:
            print("\n--- Stage 2: xshard_backward -> grad xshard ---")
            ok2, grad_xshard = stage2_xshard_backward(tmpdir)
            if not ok2 and args.stop_on_fail:
                return _summary()

        if run_all or args.stage == 3:
            print("\n--- Stage 3: xshard_adapt --grad-xshard (dry-run) ---")
            ok3 = stage3_adapt_grad_xshard(grad_xshard, tmpdir)
            if not ok3 and args.stop_on_fail:
                return _summary()

        if run_all or args.stage == 4:
            print("\n--- Stage 4: xshard_adapt --grad-scale (synthetic probe, dry-run) ---")
            ok4 = stage4_adapt_grad_scale(tmpdir)
            if not ok4 and args.stop_on_fail:
                return _summary()

        if run_all or args.stage == 5:
            print("\n--- Stage 5: adam_ctypes.py --smoke ---")
            ok5 = stage5_adam_ctypes()

    return _summary()


def _summary() -> int:
    print("\n" + "=" * 60)
    total = len(_results)
    passed = sum(1 for _, _, ok, _ in _results if ok)
    failed = total - passed
    print(f"Results: {passed}/{total} PASS  ({failed} FAIL)")
    print("=" * 60)
    if failed:
        print("\nFailed checks:")
        for stage, name, ok, detail in _results:
            if not ok:
                print(f"  stage {stage}: {name}" + (f"  [{detail}]" if detail else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
