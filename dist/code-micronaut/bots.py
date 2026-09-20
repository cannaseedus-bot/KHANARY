"""
CODE-1 bots.py — code expert JSON-line worker
Pop fold. Tasks: generate, review, refactor, optimize, explain, todos, health.

Reads {"task": "...", "payload": {...}} from stdin.
Writes one JSON result line to stdout per request.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import importlib.util
from typing import Optional

# Pull in the shared bridge (TodoCreator, DeterministicV6, TraceLogger)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "include"))
from micronaut_native import TodoCreator, DeterministicV6, TodoJsonSerializer, TraceLogger

_TRACE = TraceLogger(os.path.join(_HERE, "code1_trace.jsonl"))

# ---------------------------------------------------------------------------
# ELIZA-1 — metacognitive brain (Chen fold)
# ---------------------------------------------------------------------------

_ELIZA = os.path.normpath(os.path.join(_HERE, "..", "eliza-micronaut", "bots.py"))
_eliza = None
try:
    _espec = importlib.util.spec_from_file_location("eliza_bots", _ELIZA)
    _emod  = importlib.util.module_from_spec(_espec)
    _espec.loader.exec_module(_emod)
    _eliza = _emod
except Exception:
    pass

def _eliza_intent(text: str) -> dict:
    return _eliza.intent(text) if _eliza else {}

def _eliza_question(context: str) -> dict:
    return _eliza.question(context) if _eliza else {}

def _eliza_plan(context: str, user_intent: str = None) -> dict:
    return _eliza.plan(context, user_intent) if _eliza else {}

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_LANG_PATTERNS: list[tuple[str, list[str]]] = [
    ("python",     [r"\bdef\b", r"\bimport\b", r"\bprint\s*\(", r"#\s"]),
    ("javascript", [r"\bconst\b", r"\blet\b", r"\bvar\b", r"\bconsole\.", r"=>"]),
    ("csharp",     [r"\bnamespace\b", r"\busing\b", r"\bpublic\b.*\bclass\b", r"\.cs\b"]),
    ("cpp",        [r"#include\s*<", r"\bstd::", r"\bvoid\b.*\(", r"\bint\s+main\b"]),
    ("java",       [r"\bpublic\s+class\b", r"\bSystem\.out\.", r"\bimport\s+java\."]),
]

def detect_language(code: str) -> str:
    scores: dict[str, int] = {}
    for lang, patterns in _LANG_PATTERNS:
        scores[lang] = sum(1 for p in patterns if re.search(p, code))
    best = max(scores, key=lambda k: scores[k], default="unknown")
    return best if scores.get(best, 0) > 0 else "unknown"


# ---------------------------------------------------------------------------
# Issue patterns for review / optimize
# ---------------------------------------------------------------------------

_REVIEW_PATTERNS = [
    # (type, severity, pattern, message)
    ("bug",      "high",   r"\beval\s*\(",                 "eval() is a security risk — never pass untrusted input"),
    ("bug",      "medium", r"except\s*:\s*$",              "bare except catches all exceptions including KeyboardInterrupt"),
    ("bug",      "medium", r"except\s+Exception\s*:\s*$",  "catching Exception without binding swallows error context"),
    ("bug",      "high",   r"==\s*None\b",                 "use 'is None' not '== None'"),
    ("security", "high",   r"\bpassword\s*=\s*['\"][^'\"]+['\"]", "hardcoded credential detected"),
    ("security", "high",   r"\bsecret\s*=\s*['\"][^'\"]+['\"]",  "hardcoded secret detected"),
    ("style",    "low",    r"\bprint\s*\(",                "debug print left in production code"),
    ("style",    "low",    r"\t",                          "tab character found — prefer spaces"),
    ("perf",     "medium", r"\.append\(.*\)\s*$",          "list.append in loop — consider list comprehension"),
    ("perf",     "medium", r'"\s*\+\s*"',                  "string concatenation in loop — use join()"),
    ("perf",     "high",   r"\bfor\b.*\bfor\b",            "nested loop detected — verify O(n²) is acceptable"),
]

_OPTIMIZE_PATTERNS = [
    ("O(n²)",    "high",   r"\bfor\b.{0,40}\bfor\b",       "nested loop — consider hash map or sort-based approach"),
    ("memory",   "medium", r"\blist\s*\(",                  "list() on iterator materializes all elements — use generator if streaming"),
    ("io",       "high",   r"open\(.+\)\s*$",              "file open outside context manager — resource may not close on exception"),
    ("cpu",      "medium", r"\btime\.sleep\b",              "blocking sleep in hot path — consider async or threading"),
    ("alloc",    "low",    r"\bnew\b.*\*\b",               "raw pointer allocation — prefer smart pointer or RAII wrapper"),
]

def _scan_patterns(code: str, patterns) -> list[dict]:
    issues = []
    lines = code.splitlines()
    for lineno, line in enumerate(lines, start=1):
        for kind, severity, pat, msg in patterns:
            if re.search(pat, line):
                issues.append({
                    "type": kind,
                    "severity": severity,
                    "line": lineno,
                    "message": msg,
                    "snippet": line.strip()[:120],
                })
    return issues


# ---------------------------------------------------------------------------
# Refactor suggestions
# ---------------------------------------------------------------------------

_REFACTOR_PATTERNS = [
    (r"if\s+\w+\s*==\s*True\b",       "simplify: `if x == True` → `if x`"),
    (r"if\s+\w+\s*==\s*False\b",      "simplify: `if x == False` → `if not x`"),
    (r"return\s+True\b.*\nreturn\s+False\b", "simplify: `return cond` instead of return True/False"),
    (r"len\(\w+\)\s*==\s*0\b",        "simplify: `len(x) == 0` → `not x`"),
    (r"len\(\w+\)\s*>\s*0\b",         "simplify: `len(x) > 0` → `if x`"),
    (r"lambda\s+\w+\s*:\s*\w+\.\w+", "extract lambda to named function for readability"),
    (r"(\w+)\s*=\s*\1\s*\+\s*1\b",   "use `+=` instead of `x = x + 1`"),
    (r"(\w+)\s*=\s*\1\s*\+\s*",      "consider `+=` compound assignment"),
    (r"#\s*TODO",                     "TODO comment present — extract to issue tracker"),
    (r"#\s*FIXME",                    "FIXME comment present — schedule for next sprint"),
    (r"#\s*HACK",                     "HACK comment present — document the constraint it works around"),
]

def _refactor_suggestions(code: str) -> list[dict]:
    suggestions = []
    lines = code.splitlines()
    seen = set()
    for lineno, line in enumerate(lines, start=1):
        for pat, msg in _REFACTOR_PATTERNS:
            if re.search(pat, line) and msg not in seen:
                suggestions.append({"line": lineno, "suggestion": msg, "snippet": line.strip()[:80]})
                seen.add(msg)
    return suggestions


# ---------------------------------------------------------------------------
# Explanation: decompose code into structural components
# ---------------------------------------------------------------------------

def _explain_components(code: str, language: str) -> list[dict]:
    components = []
    lines = code.splitlines()

    # Functions / methods
    fn_pat = {
        "python":     r"^\s*def\s+(\w+)\s*\(",
        "javascript": r"^\s*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
        "csharp":     r"^\s*(?:public|private|protected|static|async).*\s+(\w+)\s*\(",
        "cpp":        r"^\s*(?:\w+\s+)+(\w+)\s*\(",
        "java":       r"^\s*(?:public|private|protected|static|final).*\s+(\w+)\s*\(",
    }.get(language, r"^\s*def\s+(\w+)\s*\(")

    for lineno, line in enumerate(lines, start=1):
        m = re.match(fn_pat, line)
        if m:
            name = next((g for g in m.groups() if g), "anonymous")
            components.append({
                "kind": "function",
                "name": name,
                "line": lineno,
                "desc": f"{name}() defined at line {lineno}",
            })

    # Classes
    cls_pat = {
        "python":     r"^\s*class\s+(\w+)",
        "javascript": r"^\s*class\s+(\w+)",
        "csharp":     r"^\s*(?:public|private|internal|sealed|abstract)?\s*class\s+(\w+)",
        "cpp":        r"^\s*(?:class|struct)\s+(\w+)",
        "java":       r"^\s*(?:public|private|abstract|final)?\s*class\s+(\w+)",
    }.get(language, r"^\s*class\s+(\w+)")

    for lineno, line in enumerate(lines, start=1):
        m = re.match(cls_pat, line)
        if m:
            components.append({
                "kind": "class",
                "name": m.group(1),
                "line": lineno,
                "desc": f"class {m.group(1)} defined at line {lineno}",
            })

    # Imports
    import_pat = {
        "python":     r"^\s*(?:import|from)\s+(\S+)",
        "javascript": r"^\s*import\s+.*\s+from\s+['\"]([^'\"]+)['\"]",
        "csharp":     r"^\s*using\s+(\S+);",
        "cpp":        r"^\s*#include\s+[<\"]([^>\"]+)[>\"]",
        "java":       r"^\s*import\s+(\S+);",
    }.get(language, r"^\s*import\s+(\S+)")

    imports = []
    for line in lines:
        m = re.match(import_pat, line)
        if m:
            imports.append(m.group(1))
    if imports:
        components.append({
            "kind": "imports",
            "names": imports[:20],
            "desc": f"{len(imports)} import(s) detected",
        })

    if not components:
        components.append({
            "kind": "raw",
            "lines": len(lines),
            "desc": f"{len(lines)} line(s) of {language} code — no top-level structure detected",
        })

    return components


# ---------------------------------------------------------------------------
# Code generation templates
# ---------------------------------------------------------------------------

_GEN_TEMPLATES = {
    "python": """\
def solution({params}):
    \"\"\"
    {description}
    \"\"\"
    # TODO: implement
    raise NotImplementedError
""",
    "javascript": """\
/**
 * {description}
 * @param {{{params}}} input
 */
function solution({params}) {{
    // TODO: implement
    throw new Error('Not implemented');
}}
""",
    "csharp": """\
/// <summary>
/// {description}
/// </summary>
public static object Solution({params}) {{
    // TODO: implement
    throw new NotImplementedException();
}}
""",
    "cpp": """\
// {description}
auto solution({params}) {{
    // TODO: implement
    throw std::runtime_error("not implemented");
}}
""",
    "java": """\
/**
 * {description}
 */
public static Object solution({params}) {{
    // TODO: implement
    throw new UnsupportedOperationException();
}}
""",
}

def _generate_template(prompt: str, language: str) -> str:
    words = [w.lower() for w in re.findall(r"\w+", prompt)]
    # simple param guess from keywords
    params = "input"
    if any(w in words for w in ("array", "list", "nums", "numbers")):
        params = {"python": "nums: list", "javascript": "nums", "java": "int[] nums",
                  "csharp": "int[] nums", "cpp": "vector<int>& nums"}.get(language, "nums")
    elif any(w in words for w in ("string", "str", "text")):
        params = {"python": "s: str", "javascript": "s", "java": "String s",
                  "csharp": "string s", "cpp": "const std::string& s"}.get(language, "s")
    elif any(w in words for w in ("tree", "node", "root")):
        params = {"python": "root", "javascript": "root", "java": "TreeNode root",
                  "csharp": "TreeNode root", "cpp": "TreeNode* root"}.get(language, "root")

    template = _GEN_TEMPLATES.get(language, _GEN_TEMPLATES["python"])
    description = prompt[:120].replace('"', "'")
    return template.format(params=params, description=description)


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------

def health() -> dict:
    return {
        "status": "ready",
        "agent": "CODE-1",
        "fold": "Pop",
        "domain": "code",
        "port": 3215,
        "capabilities": ["generate", "review", "refactor", "optimize", "explain", "todos"],
        "expert_model": "code-expert-v1",
        "training_pairs": 109118,
        "confidence": 0.88,
        "eliza": "wired" if _eliza is not None else "absent",
    }


def generate(payload: dict) -> dict:
    prompt   = str(payload.get("prompt", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or detect_language(prompt)
    if language == "unknown":
        language = "python"

    if not prompt:
        return {"error": "prompt is required"}

    # ELIZA intent — classify what the user is asking for
    eliza_i = _eliza_intent(prompt)

    t0 = time.monotonic()
    code = _generate_template(prompt, language)
    latency = (time.monotonic() - t0) * 1000

    input_hash  = DeterministicV6.sha256_hex(prompt)
    output_hash = DeterministicV6.sha256_hex(code)
    _TRACE.log_tool_execution("generate", prompt, code, latency)

    return {
        "task": "generate",
        "language": language,
        "code": code,
        "confidence": 0.75,
        "source": "template",
        "model_version": "code-expert-v1",
        "training_pairs": 109118,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "latency_ms": round(latency, 2),
        "note": "Pop-fold: template scaffold. Implement body per the prompt description.",
        "eliza_intent": eliza_i.get("intent_class", ""),
        "eliza_domain": eliza_i.get("alice_domain", "unknown"),
    }


def review(payload: dict) -> dict:
    code     = str(payload.get("code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or detect_language(code)

    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    issues = _scan_patterns(code, _REVIEW_PATTERNS)
    latency = (time.monotonic() - t0) * 1000

    suggestions = list({i["message"] for i in issues})
    _TRACE.log_tool_execution("review", code[:256], json.dumps(issues), latency)

    result = {
        "task": "review",
        "language": language,
        "issues": issues,
        "issue_count": len(issues),
        "suggestions": suggestions,
        "confidence": 0.88,
        "verdict": "needs_attention" if issues else "clean",
        "latency_ms": round(latency, 2),
    }
    # ELIZA: when issues found, produce a plan for what to fix first
    if issues:
        issue_summary = f"code review in {language}: {len(issues)} issues — " + \
                        "; ".join(i["message"] for i in issues[:3])
        ep = _eliza_plan(issue_summary)
        result["eliza_next"] = ep.get("next", [])[:3]
        result["eliza_domain"] = ep.get("semantic", {}).get("alice_domain", "unknown")
    return result


def refactor(payload: dict) -> dict:
    code     = str(payload.get("code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or detect_language(code)

    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    suggestions = _refactor_suggestions(code)
    latency = (time.monotonic() - t0) * 1000

    _TRACE.log_tool_execution("refactor", code[:256], json.dumps(suggestions), latency)

    return {
        "task": "refactor",
        "language": language,
        "suggestions": suggestions,
        "suggestion_count": len(suggestions),
        "confidence": 0.82,
        "note": "Apply suggestions top-to-bottom; re-run review after each change.",
        "latency_ms": round(latency, 2),
    }


def optimize(payload: dict) -> dict:
    code     = str(payload.get("code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or detect_language(code)

    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    issues = _scan_patterns(code, _OPTIMIZE_PATTERNS)
    latency = (time.monotonic() - t0) * 1000

    _TRACE.log_tool_execution("optimize", code[:256], json.dumps(issues), latency)

    return {
        "task": "optimize",
        "language": language,
        "performance_issues": issues,
        "issue_count": len(issues),
        "confidence": 0.80,
        "verdict": "optimize_needed" if issues else "no_bottlenecks_found",
        "latency_ms": round(latency, 2),
    }


def explain(payload: dict) -> dict:
    code     = str(payload.get("code", "")).strip()
    language = str(payload.get("language", "")).strip().lower() or detect_language(code)

    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    components = _explain_components(code, language)
    lines      = code.splitlines()
    issues     = _scan_patterns(code, _REVIEW_PATTERNS)
    latency    = (time.monotonic() - t0) * 1000

    _TRACE.log_tool_execution("explain", code[:256], json.dumps(components), latency)

    return {
        "task": "explain",
        "language": language,
        "line_count": len(lines),
        "components": components,
        "component_count": len(components),
        "notable_issues": [i["message"] for i in issues if i["severity"] in ("high",)][:5],
        "confidence": 0.85,
        "latency_ms": round(latency, 2),
    }


def todos(payload: dict) -> dict:
    code    = str(payload.get("code", "")).strip()
    service = str(payload.get("service", "CODE-1")).strip()

    if not code:
        return {"error": "code is required"}

    t0 = time.monotonic()
    creator = TodoCreator(service)
    items   = creator.extract_todos(code)
    latency = (time.monotonic() - t0) * 1000

    _TRACE.log_tool_execution("todos", code[:256], str(len(items)), latency)

    return {
        "task": "todos",
        "service": service,
        "total": len(items),
        "todos": [
            {
                "id":          t.id,
                "title":       t.title,
                "category":    t.category,
                "priority":    t.priority,
                "line":        t.line_number,
                "confidence":  t.confidence,
                "status":      t.status,
            }
            for t in items
        ],
        "latency_ms": round(latency, 2),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_HANDLERS = {
    "health":   lambda p: health(),
    "generate": generate,
    "review":   review,
    "refactor": refactor,
    "optimize": optimize,
    "explain":  explain,
    "todos":    todos,
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
