"""
MCDR-1 bots.py — AST code reviewer JSON-line worker
Yax fold. Tasks: review, review_file, parse_ast, diff, todos, document, test, explain, github_review, health.

Wraps micronaut_code_reviewer.exe (CLI) + node parse.js (tree-sitter).
Falls back to pure-Python pattern review if native exe is unavailable.

Reads  {"task": "...", "payload": {...}} from stdin.
Writes one JSON result line to stdout per request.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

_REVIEWER_CANDIDATES = [
    os.path.join(_HERE, "Release", "micronaut_code_reviewer.exe"),
    os.path.join(_HERE, "build-role2", "bin", "micronaut_code_reviewer.exe"),
    os.path.join(_HERE, "build", "bin", "Release", "micronaut_coder.exe"),
]
_REVIEWER_EXE: Optional[str] = next((p for p in _REVIEWER_CANDIDATES if os.path.isfile(p)), None)

_PARSE_JS = os.path.join(_HERE, "CodeWASM", "parse.js")
_NODE_EXE = "node"

# Tree-sitter language → grammar file mapping
_TS_LANGS = {
    "bash": "bash", "c": "c", "csharp": "c_sharp", "cpp": "cpp",
    "css": "css", "go": "go", "html": "html", "java": "java",
    "javascript": "javascript", "json": "json", "kuhul": "kuhul",
    "php": "php", "python": "python", "ruby": "ruby", "rust": "rust",
    "scala": "scala", "tsx": "tsx", "typescript": "typescript",
}

# Language detection patterns (same as CODE-1 but extended)
_LANG_PATTERNS: list[tuple[str, list[str]]] = [
    ("python",     [r"\bdef\b", r"\bimport\b", r"\bprint\s*\("]),
    ("javascript", [r"\bconst\b", r"\bconsole\.", r"=>"]),
    ("typescript", [r":\s*\w+\s*[=;{]", r"interface\s+\w+", r"<\w+>"]),
    ("cpp",        [r"#include\s*<", r"\bstd::", r"int\s+main\b"]),
    ("csharp",     [r"\bnamespace\b", r"\busing\b", r"\.cs\b"]),
    ("java",       [r"\bpublic\s+class\b", r"\bSystem\.out\."]),
    ("rust",       [r"\bfn\s+\w+", r"\blet\s+mut\b", r"\bimpl\b"]),
    ("go",         [r"\bfunc\s+\w+", r"\bpackage\s+\w+", r"\bgo\s+\w+"]),
]

def _detect_language(code: str) -> str:
    scores: dict[str, int] = {}
    for lang, patterns in _LANG_PATTERNS:
        scores[lang] = sum(1 for p in patterns if re.search(p, code))
    best = max(scores, key=lambda k: scores[k], default="unknown")
    return best if scores.get(best, 0) > 0 else "unknown"

# ---------------------------------------------------------------------------
# Native reviewer subprocess helper
# ---------------------------------------------------------------------------

def _run_reviewer(cmd_args: list[str], timeout: int = 15) -> tuple[bool, str]:
    if not _REVIEWER_EXE:
        return False, "micronaut_code_reviewer.exe not found"
    try:
        result = subprocess.run(
            [_REVIEWER_EXE] + cmd_args,
            capture_output=True, text=True, timeout=timeout,
            cwd=_HERE,
        )
        out = result.stdout + result.stderr
        return result.returncode == 0, out.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return False, str(e)


def _run_reviewer_on_code(code: str, command: str, language: str, extra_args: list[str] = []) -> tuple[bool, str]:
    suffix = {"python": ".py", "javascript": ".js", "typescript": ".ts",
               "cpp": ".cpp", "csharp": ".cs", "java": ".java",
               "rust": ".rs", "go": ".go"}.get(language, ".txt")
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        return _run_reviewer([command, tmp] + extra_args)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Tree-sitter AST via parse.js
# ---------------------------------------------------------------------------

def _parse_ast_node(code: str, language: str) -> dict:
    ts_lang = _TS_LANGS.get(language)
    if not ts_lang:
        return {"error": f"no tree-sitter grammar for language '{language}'", "available": list(_TS_LANGS)}
    if not os.path.isfile(_PARSE_JS):
        return {"error": "CodeWASM/parse.js not found"}

    suffix = ".txt"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [_NODE_EXE, _PARSE_JS, ts_lang, tmp],
            capture_output=True, text=True, timeout=20, cwd=_HERE,
        )
        output = result.stdout + result.stderr
        return {
            "language":   language,
            "ts_grammar": ts_lang,
            "exit_code":  result.returncode,
            "output":     output.strip()[:4000],
            "success":    result.returncode == 0,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"error": str(e), "language": language}
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Fallback pattern-based review (when native exe unavailable)
# ---------------------------------------------------------------------------

_REVIEW_PATTERNS = [
    ("bug",      "high",   r"\beval\s*\(",                  "eval() is a security risk"),
    ("bug",      "medium", r"except\s*:\s*$",               "bare except catches all exceptions"),
    ("bug",      "high",   r"==\s*None\b",                  "use 'is None' not '== None'"),
    ("security", "high",   r"\bpassword\s*=\s*['\"][^'\"]+", "hardcoded credential"),
    ("security", "high",   r"\bsecret\s*=\s*['\"][^'\"]+",  "hardcoded secret"),
    ("perf",     "medium", r"\bfor\b.{0,40}\bfor\b",        "nested loop — verify O(n^2) is acceptable"),
    ("perf",     "medium", r'"\s*\+\s*"',                   "string concat in loop — use join()"),
    ("style",    "low",    r"\bprint\s*\(",                  "debug print left in code"),
    ("style",    "low",    r"\t",                            "tab character — prefer spaces"),
    ("memory",   "high",   r"\bnew\b\s+\w+(?!\s*\[)",       "raw heap allocation — check delete/RAII"),
]

def _pattern_review(code: str, language: str) -> list[dict]:
    issues = []
    for lineno, line in enumerate(code.splitlines(), 1):
        for kind, sev, pat, msg in _REVIEW_PATTERNS:
            if re.search(pat, line):
                issues.append({"type": kind, "severity": sev, "line": lineno,
                                "message": msg, "snippet": line.strip()[:100]})
    return issues


# ---------------------------------------------------------------------------
# Unified diff helper
# ---------------------------------------------------------------------------

def _unified_diff(old: str, new: str, label_old: str = "old", label_new: str = "new") -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=label_old, tofile=label_new, lineterm="",
    ))


# ---------------------------------------------------------------------------
# Doc / test / github templates
# ---------------------------------------------------------------------------

def _doc_template(code: str, language: str, style: str) -> str:
    lines = code.splitlines()
    fn_re = {
        "python":     r"^\s*def\s+(\w+)\s*\(([^)]*)\)",
        "javascript": r"^\s*function\s+(\w+)\s*\(([^)]*)\)",
        "typescript": r"^\s*(?:export\s+)?function\s+(\w+)\s*\(([^)]*)\)",
        "cpp":        r"^\s*\w[\w\s\*&]+\s+(\w+)\s*\(([^)]*)\)",
        "java":       r"^\s*(?:public|private|protected|static).*\s+(\w+)\s*\(([^)]*)\)",
    }.get(language, r"^\s*def\s+(\w+)\s*\(([^)]*)\)")

    docs = []
    for lineno, line in enumerate(lines, 1):
        m = re.match(fn_re, line)
        if not m:
            continue
        name, params = m.group(1), m.group(2).strip()
        param_list = [p.strip().split()[-1] for p in params.split(",") if p.strip()]

        if style == "jsdoc" or language in ("javascript", "typescript"):
            block = ["/**", f" * {name} — TODO: describe", " *"]
            block += [f" * @param {p} - TODO" for p in param_list if p]
            block += [" * @returns TODO", " */"]
        elif style == "doxygen" or language in ("cpp", "c"):
            block = [f"/** @brief {name} — TODO: describe"]
            block += [f" * @param {p} TODO" for p in param_list if p]
            block += [" * @return TODO", " */"]
        else:
            block = [f'"""{name} — TODO: describe', ""]
            block += [f"Args:", *[f"    {p}: TODO" for p in param_list if p], "", "Returns:", "    TODO", '"""']

        docs.append({"function": name, "line": lineno, "doc": "\n".join(block)})

    return json.dumps({"language": language, "style": style, "docs": docs, "count": len(docs)}, indent=2)


_TEST_FRAMEWORKS = {
    "python": "pytest", "javascript": "jest", "typescript": "jest",
    "cpp": "gtest", "java": "junit", "rust": "builtin", "go": "testing",
}

def _test_template(code: str, language: str, framework: str) -> str:
    fw = framework or _TEST_FRAMEWORKS.get(language, "unknown")
    fn_re = {
        "python":     r"^\s*def\s+(\w+)\s*\(",
        "javascript": r"^\s*(?:function\s+|(?:const|let|var)\s+)(\w+)\s*",
        "typescript": r"^\s*(?:export\s+)?function\s+(\w+)\s*\(",
        "cpp":        r"^\s*\w[\w\s\*&]+\s+(\w+)\s*\(",
        "java":       r"^\s*(?:public|private).*\s+(\w+)\s*\(",
    }.get(language, r"^\s*def\s+(\w+)\s*\(")

    fns = []
    for line in code.splitlines():
        m = re.match(fn_re, line)
        if m and not m.group(1).startswith("_"):
            fns.append(m.group(1))

    tests = []
    for fn in fns[:10]:
        if fw == "pytest":
            tests.append(f"def test_{fn}():\n    # TODO: implement\n    assert {fn}() is not None")
        elif fw == "jest":
            tests.append(f"test('{fn}', () => {{\n  // TODO: implement\n  expect({fn}()).toBeDefined();\n}});")
        elif fw == "gtest":
            tests.append(f"TEST({fn}Test, Basic) {{\n  // TODO: implement\n  EXPECT_TRUE(true);\n}}")
        elif fw == "junit":
            tests.append(f"@Test\npublic void test{fn.capitalize()}() {{\n  // TODO: implement\n  assertTrue(true);\n}}")
        else:
            tests.append(f"// test for {fn} — framework: {fw}")

    return json.dumps({"language": language, "framework": fw, "functions": fns, "tests": tests}, indent=2)


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------

def health() -> dict:
    return {
        "status":       "ready",
        "agent":        "MCDR-1",
        "fold":         "Yax",
        "domain":       "ast_code_review",
        "port":         3216,
        "reviewer_exe": _REVIEWER_EXE or "not found",
        "parse_js":     _PARSE_JS if os.path.isfile(_PARSE_JS) else "not found",
        "ts_grammars":  list(_TS_LANGS),
        "capabilities": ["review", "review_file", "parse_ast", "diff", "todos", "document", "test", "explain", "github_review"],
    }


def review(payload: dict) -> dict:
    code     = str(payload.get("code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or _detect_language(code)
    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    ok, native_out = _run_reviewer_on_code(code, "review", language)
    latency = (time.monotonic() - t0) * 1000

    if ok and native_out:
        return {"task": "review", "language": language, "source": "native",
                "output": native_out, "latency_ms": round(latency, 2)}

    # fallback
    issues = _pattern_review(code, language)
    return {
        "task": "review", "language": language, "source": "fallback_pattern",
        "issues": issues, "issue_count": len(issues),
        "verdict": "needs_attention" if issues else "clean",
        "latency_ms": round(latency, 2),
    }


def review_file(payload: dict) -> dict:
    path = str(payload.get("path", "")).strip()
    if not path:
        return {"error": "path is required"}
    if not os.path.isfile(path):
        return {"error": f"file not found: {path}"}

    t0 = time.monotonic()
    ok, out = _run_reviewer(["review", path])
    latency = (time.monotonic() - t0) * 1000

    return {"task": "review_file", "path": path, "success": ok,
            "output": out[:8000], "latency_ms": round(latency, 2)}


def parse_ast(payload: dict) -> dict:
    code     = str(payload.get("code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or _detect_language(code)
    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    result = _parse_ast_node(code, language)
    result["task"] = "parse_ast"
    result["latency_ms"] = round((time.monotonic() - t0) * 1000, 2)
    return result


def diff(payload: dict) -> dict:
    old      = str(payload.get("old_code", "")).strip()
    new      = str(payload.get("new_code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or _detect_language(new or old)
    if not old or not new:
        return {"error": "old_code and new_code are required"}

    t0 = time.monotonic()
    patch = _unified_diff(old, new)
    old_issues = _pattern_review(old, language)
    new_issues = _pattern_review(new, language)
    resolved = len(old_issues) - len(new_issues)
    latency = (time.monotonic() - t0) * 1000

    return {
        "task": "diff", "language": language,
        "patch": patch[:6000],
        "lines_added":   sum(1 for l in patch.splitlines() if l.startswith("+")),
        "lines_removed": sum(1 for l in patch.splitlines() if l.startswith("-")),
        "issues_before": len(old_issues),
        "issues_after":  len(new_issues),
        "issues_resolved": max(0, resolved),
        "new_issues":    [i for i in new_issues if i not in old_issues],
        "latency_ms":    round(latency, 2),
    }


def todos(payload: dict) -> dict:
    code    = str(payload.get("code", "")).strip()
    service = str(payload.get("service", "MCDR-1")).strip()
    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    # Try native reviewer first
    ok, native_out = _run_reviewer_on_code(code, "todos", _detect_language(code))
    if ok and native_out:
        return {"task": "todos", "service": service, "source": "native",
                "output": native_out, "latency_ms": round((time.monotonic() - t0) * 1000, 2)}

    # Fallback: regex extraction
    _TODO_RE = re.compile(r"(?:TODO|FIXME|HACK|XXX)[:\s]+(.*)", re.IGNORECASE)
    items = []
    for lineno, line in enumerate(code.splitlines(), 1):
        m = _TODO_RE.search(line)
        if m:
            marker = line.upper()
            priority = 5 if "CRITICAL" in marker else 4 if "FIXME" in marker else 2 if "HACK" in marker else 3
            items.append({"line": lineno, "text": m.group(1).strip()[:80],
                          "priority": priority, "status": "pending"})
    return {"task": "todos", "service": service, "source": "fallback",
            "total": len(items), "todos": items,
            "latency_ms": round((time.monotonic() - t0) * 1000, 2)}


def document(payload: dict) -> dict:
    code     = str(payload.get("code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or _detect_language(code)
    style    = str(payload.get("style", "")).strip().lower() or "docstring"
    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    ok, native_out = _run_reviewer_on_code(code, "document", language)
    if ok and native_out:
        return {"task": "document", "language": language, "source": "native",
                "output": native_out, "latency_ms": round((time.monotonic() - t0) * 1000, 2)}

    doc_json = _doc_template(code, language, style)
    return {"task": "document", "language": language, "style": style, "source": "template",
            "result": json.loads(doc_json), "latency_ms": round((time.monotonic() - t0) * 1000, 2)}


def test(payload: dict) -> dict:
    code      = str(payload.get("code", "")).strip()
    language  = str(payload.get("language", "")).strip().lower() or _detect_language(code)
    framework = str(payload.get("framework", "")).strip().lower()
    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    ok, native_out = _run_reviewer_on_code(code, "test", language)
    if ok and native_out:
        return {"task": "test", "language": language, "source": "native",
                "output": native_out, "latency_ms": round((time.monotonic() - t0) * 1000, 2)}

    test_json = _test_template(code, language, framework)
    return {"task": "test", "language": language, "source": "template",
            "result": json.loads(test_json), "latency_ms": round((time.monotonic() - t0) * 1000, 2)}


def explain(payload: dict) -> dict:
    code     = str(payload.get("code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or _detect_language(code)
    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    ok, native_out = _run_reviewer_on_code(code, "explain", language)
    if ok and native_out:
        return {"task": "explain", "language": language, "source": "native",
                "output": native_out, "latency_ms": round((time.monotonic() - t0) * 1000, 2)}

    # fallback: structural summary
    lines = code.splitlines()
    fn_pat = r"^\s*(?:def|function|func|fn)\s+(\w+)"
    fns = [re.match(fn_pat, l).group(1) for l in lines if re.match(fn_pat, l)]
    issues = _pattern_review(code, language)
    return {
        "task": "explain", "language": language, "source": "fallback",
        "line_count": len(lines), "functions": fns,
        "notable_issues": [i["message"] for i in issues if i["severity"] == "high"][:5],
        "latency_ms": round((time.monotonic() - t0) * 1000, 2),
    }


def github_review(payload: dict) -> dict:
    code       = str(payload.get("code", "")).strip()
    language   = str(payload.get("language", "")).strip().lower() or _detect_language(code)
    pr_context = str(payload.get("pr_context", "")).strip()
    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    ok, native_out = _run_reviewer_on_code(code, "github-review", language)
    if ok and native_out:
        return {"task": "github_review", "language": language, "source": "native",
                "output": native_out, "latency_ms": round((time.monotonic() - t0) * 1000, 2)}

    # fallback: format pattern issues as GitHub markdown
    issues = _pattern_review(code, language)
    lines = ["## MCDR-1 Code Review", ""]
    if pr_context:
        lines += [f"> Context: {pr_context}", ""]
    if not issues:
        lines += ["**No issues found.** Code looks structurally sound.", ""]
    else:
        sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for i in issues:
            icon = sev_icon.get(i["severity"], "⚪")
            lines.append(f"- {icon} **[{i['type'].upper()}]** Line {i['line']}: {i['message']}")
            lines.append(f"  ```\n  {i['snippet']}\n  ```")
    lines += ["", f"*Reviewed by MCDR-1 (Yax fold, tree-sitter AST reviewer)*"]
    return {
        "task": "github_review", "language": language, "source": "fallback",
        "markdown": "\n".join(lines),
        "issue_count": len(issues),
        "latency_ms": round((time.monotonic() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_HANDLERS = {
    "health":        lambda p: health(),
    "review":        review,
    "review_file":   review_file,
    "parse_ast":     parse_ast,
    "diff":          diff,
    "todos":         todos,
    "document":      document,
    "test":          test,
    "explain":       explain,
    "github_review": github_review,
}


def dispatch(task: str, payload: dict) -> dict:
    handler = _HANDLERS.get(task)
    if handler is None:
        return {"error": f"unknown task: {task}", "available": list(_HANDLERS)}
    try:
        return handler(payload)
    except Exception as exc:
        return {"error": str(exc), "task": task}


# ---------------------------------------------------------------------------
# JSON-line stdio loop
# ---------------------------------------------------------------------------

def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg     = json.loads(line)
            task    = str(msg.get("task", "")).strip()
            payload = msg.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            result = dispatch(task, payload)
        except json.JSONDecodeError as exc:
            result = {"error": f"invalid JSON: {exc}"}
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
