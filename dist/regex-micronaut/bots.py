"""
REGEX-1 RegexMicronaut — bots.py
Gram-fold pure pattern matching µ. Zero external deps.
Backend: Python re module.
Fold: ⟁GRAM_FOLD⟁
"""

import sys
import re
import json
import time
from pathlib import Path

_MICRONAUT_ID   = "REGEX-1"
_MICRONAUT_NAME = "RegexMicronaut"
_MICRONAUT_FOLD = "⟁GRAM_FOLD⟁"

_SCHEMAS: dict[str, str] = {
    "kxml_token":    r"^[⊗⊕⊖⊘⊛⊜⊝⊞]",
    "aiml_pattern":  r"^[A-Z0-9 _*#]+$",
    "port":          r"^([1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$",
    "semver":        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$",
    "sha256":        r"^[0-9a-f]{64}$",
}

_FLAG_MAP = {
    "i": re.IGNORECASE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
    "x": re.VERBOSE,
}

def _parse_flags(flag_str: str | None) -> int:
    flags = 0
    for c in (flag_str or ""):
        flags |= _FLAG_MAP.get(c.lower(), 0)
    return flags

# ---------------------------------------------------------------------------

def health_check() -> dict:
    return {"status": "ok", "micronaut": _MICRONAUT_ID, "engine": "Python re", "schemas": list(_SCHEMAS.keys())}


def match(pattern: str, input_text: str, flags: str = "") -> dict:
    t0 = time.perf_counter()
    try:
        m = re.fullmatch(pattern, input_text, _parse_flags(flags))
        latency = (time.perf_counter() - t0) * 1000
        return {
            "micronaut": _MICRONAUT_ID,
            "fold": _MICRONAUT_FOLD,
            "pattern": pattern,
            "input": input_text,
            "matched": m is not None,
            "groups": m.groupdict() if m else {},
            "span": list(m.span()) if m else None,
            "latency_ms": round(latency, 2),
        }
    except re.error as exc:
        return {"error": str(exc), "pattern": pattern, "micronaut": _MICRONAUT_ID}


def search(pattern: str, input_text: str, flags: str = "") -> dict:
    t0 = time.perf_counter()
    try:
        m = re.search(pattern, input_text, _parse_flags(flags))
        latency = (time.perf_counter() - t0) * 1000
        return {
            "micronaut": _MICRONAUT_ID,
            "fold": _MICRONAUT_FOLD,
            "pattern": pattern,
            "matched": m is not None,
            "groups": m.groupdict() if m else {},
            "span": list(m.span()) if m else None,
            "latency_ms": round(latency, 2),
        }
    except re.error as exc:
        return {"error": str(exc), "micronaut": _MICRONAUT_ID}


def findall(pattern: str, input_text: str, flags: str = "") -> dict:
    t0 = time.perf_counter()
    try:
        results = re.findall(pattern, input_text, _parse_flags(flags))
        latency = (time.perf_counter() - t0) * 1000
        return {
            "micronaut": _MICRONAUT_ID,
            "fold": _MICRONAUT_FOLD,
            "pattern": pattern,
            "count": len(results),
            "results": results[:100],
            "latency_ms": round(latency, 2),
        }
    except re.error as exc:
        return {"error": str(exc), "micronaut": _MICRONAUT_ID}


def validate(schema_name: str, input_text: str) -> dict:
    pat = _SCHEMAS.get(schema_name)
    if pat is None:
        return {"error": f"Unknown schema '{schema_name}'", "valid_schemas": list(_SCHEMAS.keys())}
    result = match(pat, input_text)
    result["schema"] = schema_name
    return result


def dispatch(task: str, payload: dict | None = None) -> str:
    payload = payload or {}
    t = task.lower()
    if t == "match":
        return json.dumps(match(payload.get("pattern",""), payload.get("input",""), payload.get("flags","")))
    if t == "search":
        return json.dumps(search(payload.get("pattern",""), payload.get("input",""), payload.get("flags","")))
    if t == "findall":
        return json.dumps(findall(payload.get("pattern",""), payload.get("input",""), payload.get("flags","")))
    if t in ("validate", "schema"):
        return json.dumps(validate(payload.get("schema",""), payload.get("input","")))
    if t == "health":
        return json.dumps(health_check())
    return json.dumps({"error": f"Unknown task '{task}'",
                       "supported": ["match","search","findall","validate","health"],
                       "micronaut": _MICRONAUT_ID})


if __name__ == "__main__":
    print(f"[{_MICRONAUT_ID}] {_MICRONAUT_NAME} smoke test")
    print(f"[1] health: {health_check()}")
    r = json.loads(dispatch("match", {"pattern": r"hello (\w+)", "input": "hello world", "flags": "i"}))
    print(f"[2] match: matched={r['matched']} groups={r['groups']}")
    r = json.loads(dispatch("validate", {"schema": "sha256", "input": "a" * 64}))
    print(f"[3] validate sha256: matched={r['matched']}")
    r = json.loads(dispatch("validate", {"schema": "semver", "input": "1.0.0"}))
    print(f"[4] validate semver 1.0.0: matched={r['matched']}")
    r = json.loads(dispatch("validate", {"schema": "port", "input": "17474"}))
    print(f"[5] validate port 17474: matched={r['matched']}")
    print(f"[{_MICRONAUT_ID}] done")
