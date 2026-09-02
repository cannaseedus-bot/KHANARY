"""
ELIZA-1 ElizaMicronaut — bots.py
Ch'en-fold reasoning verifier: hypothesis checking, correction capture, semantic-cube integration.
Backend: kuhul-engine HTTP API on http://localhost:17474  (µ=eliza)
Fold: ⟁CHEN_FOLD⟁
"""

import sys
import time
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

_INCLUDE_DIR = Path(__file__).parent / "include"
if _INCLUDE_DIR.exists() and str(_INCLUDE_DIR) not in sys.path:
    sys.path.insert(0, str(_INCLUDE_DIR))

_MICRONAUT_ID   = "ELIZA-1"
_MICRONAUT_NAME = "ElizaMicronaut"
_MICRONAUT_FOLD = "⟁CHEN_FOLD⟁"

_KUHUL_BASE     = "http://localhost:17474"
_MU_NAME        = "eliza"

# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": str(exc)}


def _get(url: str, timeout: int = 5) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": str(exc)}

# ---------------------------------------------------------------------------

def health_check() -> dict:
    return _get(f"{_KUHUL_BASE}/health")


def verify(hypothesis: str, context: str = "", cube_state: dict | None = None) -> str:
    """
    POST hypothesis + context to kuhul-engine (µ=eliza).
    Returns verification result: {verified, confidence, correction}.
    """
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": (
            "You are ELIZA, a hypothesis verifier. Evaluate the candidate answer or "
            "tool call against known constraints. Return JSON: "
            "{verified: bool, confidence: float, correction: string|null, cube_face_updates: {}}"
        )},
        {"role": "user", "content": hypothesis if not context else f"Context: {context}\n\nHypothesis: {hypothesis}"}
    ]
    result = _post(f"{_KUHUL_BASE}/v1/chat/completions", {
        "model": _MU_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
    })
    latency = (time.perf_counter() - t0) * 1000

    text = ""
    if "choices" in result:
        text = result["choices"][0].get("message", {}).get("content", "")
    elif "error" in result:
        return json.dumps({"status": "error", "micronaut": _MICRONAUT_ID, "error": result["error"]})

    return json.dumps({
        "micronaut": _MICRONAUT_ID,
        "fold": _MICRONAUT_FOLD,
        "latency_ms": round(latency, 2),
        "result": text,
    })


def correct(original: str, correction: str, confidence: float = 0.9) -> str:
    """Append a correction event to the episodic log."""
    event = {
        "op": "correction",
        "micronaut": _MICRONAUT_ID,
        "original": original,
        "correction": correction,
        "confidence": confidence,
        "hash": hashlib.sha256(f"{original}|{correction}".encode()).hexdigest(),
    }
    log_path = Path(__file__).parents[2] / "micronauts" / "tool_dispatch_proof.json"
    try:
        existing = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(event)
        log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:
        event["log_error"] = str(exc)
    return json.dumps(event)


def dispatch(task: str, payload: dict | None = None) -> str:
    payload = payload or {}
    t = task.lower()
    if t in ("verify", "check", "validate"):
        return verify(payload.get("hypothesis", payload.get("input", "")),
                      payload.get("context", ""),
                      payload.get("cube_state"))
    if t in ("correct", "correction"):
        return correct(payload.get("original", ""), payload.get("correction", ""),
                       payload.get("confidence", 0.9))
    if t == "health":
        return json.dumps(health_check())
    return json.dumps({"error": f"Unknown task '{task}'",
                       "supported": ["verify", "correct", "health"],
                       "micronaut": _MICRONAUT_ID})


if __name__ == "__main__":
    print(f"[{_MICRONAUT_ID}] {_MICRONAUT_NAME} smoke test")
    h = health_check()
    print(f"[1] health: {h}")
    if "error" not in h:
        r = verify("The semantic cube has 6 faces.", "Testing ELIZA verification.")
        print(f"[2] verify: {r[:200]}")
    print(f"[{_MICRONAUT_ID}] done")
