"""
AM-1 AdamMicronaut — bots.py
Autonomous Deterministic Adaptive Model inference bot.
Calls ADAM HTTP API on http://localhost:3167.

Backend: api_custom (adam.exe --port 3167)
Fold:    ⟁COMPUTE_FOLD⟁
"""

import sys
import time
import json
import hashlib
import importlib.util
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# ELIZA-1 — metacognitive brain (Chen fold)
# ---------------------------------------------------------------------------

_ELIZA = Path(__file__).parent.parent / "eliza-micronaut" / "bots.py"
_eliza = None
try:
    _espec = importlib.util.spec_from_file_location("eliza_bots", str(_ELIZA))
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
# Native bridge (micronaut_native.py co-located in include/)
# ---------------------------------------------------------------------------
_INCLUDE_DIR = Path(__file__).parent / "include"
if _INCLUDE_DIR.exists() and str(_INCLUDE_DIR) not in sys.path:
    sys.path.insert(0, str(_INCLUDE_DIR))

_NATIVE_BRIDGE = False
try:
    from micronaut_native import DeterministicV6, TraceLogger, NATIVE_AVAILABLE
    _NATIVE_BRIDGE = True
except ImportError:
    NATIVE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Micronaut identity
# ---------------------------------------------------------------------------
_MICRONAUT_ID   = "AM-1"
_MICRONAUT_NAME = "AdamMicronaut"
_MICRONAUT_FOLD = "⟁COMPUTE_FOLD⟁"

_ADAM_BASE      = "http://localhost:3167"
_ADAM_EXE       = Path(__file__).parent / "adam.exe"

# Trace log
try:
    _TRACE_LOG_PATH = Path(__file__).parent.parents[6] / "logs" / "am1_trace.jsonl"
except IndexError:
    _TRACE_LOG_PATH = Path(__file__).parent / "am1_trace.jsonl"
_trace_logger = TraceLogger(str(_TRACE_LOG_PATH)) if _NATIVE_BRIDGE else None

# ---------------------------------------------------------------------------
# HTTP helpers (zero external deps)
# ---------------------------------------------------------------------------

def _http_get(path: str, timeout: int = 5) -> dict:
    url = f"{_ADAM_BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": str(exc), "url": url}


def _http_post(path: str, payload: dict, timeout: int = 10) -> dict:
    url = f"{_ADAM_BASE}{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": str(exc), "url": url}


# ---------------------------------------------------------------------------
# V6 wrapper
# ---------------------------------------------------------------------------

def _v6_wrap(tool_name: str, input_text: str, output_text: str,
             latency_ms: float) -> str:
    if not _NATIVE_BRIDGE:
        return output_text
    envelope = DeterministicV6.create_tool_result(
        tool_name=tool_name,
        input_text=input_text,
        output_text=output_text,
        latency_ms=latency_ms,
    )
    if _trace_logger:
        _trace_logger.log_tool_execution(
            tool_name=tool_name,
            input_hash=hashlib.sha256(input_text.encode()).hexdigest()[:16],
            output_hash=hashlib.sha256(output_text.encode()).hexdigest()[:16],
            latency_ms=latency_ms,
            micronaut_id=_MICRONAUT_ID,
        )
    return json.dumps(envelope, indent=2)


# ---------------------------------------------------------------------------
# Core tool methods
# ---------------------------------------------------------------------------

def health_check() -> dict:
    """GET /health — verify ADAM is running."""
    result = _http_get("/health")
    result["eliza"] = "wired" if _eliza is not None else "absent"
    return result


def generate(prompt: str, mode: str = "fast") -> str:
    """
    POST /generate — run ADAM inference.
    Routes prompt to reasoning/code/math expert via bigram scoring.
    Returns V6-wrapped result envelope.
    ELIZA classifies intent to auto-select mode when not explicitly set.
    """
    # ELIZA intent — auto-promote to deep mode for plan/clarification requests
    eliza_i = _eliza_intent(prompt)
    intent_class = eliza_i.get("intent_class", "")
    alice_domain = eliza_i.get("alice_domain", "unknown")
    if mode == "fast" and intent_class in ("plan", "clarification"):
        mode = "deep"

    t0 = time.perf_counter()
    result = _http_post("/generate", {"prompt": prompt, "mode": mode})
    latency = (time.perf_counter() - t0) * 1000

    if "error" in result:
        return json.dumps({"status": "error", "micronaut": _MICRONAUT_ID,
                           "error": result["error"]})

    result["eliza_intent"] = intent_class
    result["eliza_domain"] = alice_domain
    output_text = json.dumps(result)
    return _v6_wrap("generate", prompt, output_text, latency)


def evolve(feedback: dict | None = None) -> str:
    """POST /evolve — send feedback to ADAM evolution engine."""
    t0 = time.perf_counter()
    result = _http_post("/evolve", {"feedback": feedback or {}})
    latency = (time.perf_counter() - t0) * 1000
    output_text = json.dumps(result)
    return _v6_wrap("evolve", json.dumps(feedback or {}), output_text, latency)


def get_metrics() -> dict:
    """GET /metrics — engine telemetry."""
    return _http_get("/metrics")


def verify(input_hash: str) -> str:
    """POST /verify — determinism proof for a given input hash."""
    t0 = time.perf_counter()
    result = _http_post("/verify", {"input_hash": input_hash})
    latency = (time.perf_counter() - t0) * 1000
    output_text = json.dumps(result)
    return _v6_wrap("verify", input_hash, output_text, latency)


# ---------------------------------------------------------------------------
# Dispatch — BotOrchestrator entry point
# ---------------------------------------------------------------------------

_MICRONAUT_TO_FOLD = {
    "AM-1": "⟁COMPUTE_FOLD⟁",
}


def dispatch(task: str, payload: dict | None = None) -> str:
    """
    Route orchestrator task to the correct ADAM endpoint.

    Tasks:
      generate      → POST /generate  (payload: {prompt, mode?})
      evolve        → POST /evolve    (payload: {feedback?})
      metrics       → GET  /metrics
      verify        → POST /verify    (payload: {input_hash})
      health        → GET  /health
    """
    payload = payload or {}
    task_lower = task.lower()

    if task_lower in ("generate", "infer", "run"):
        prompt = payload.get("prompt", payload.get("input", ""))
        mode   = payload.get("mode", "fast")
        return generate(prompt, mode)

    if task_lower in ("evolve", "adapt", "feedback"):
        return evolve(payload.get("feedback"))

    if task_lower in ("metrics", "status", "telemetry"):
        return json.dumps(get_metrics(), indent=2)

    if task_lower in ("verify", "proof"):
        raw  = payload.get("input_hash") or payload.get("prompt", "")
        ihash = raw if len(raw) == 64 else hashlib.sha256(raw.encode()).hexdigest()
        return verify(ihash)

    if task_lower == "health":
        return json.dumps(health_check(), indent=2)

    return json.dumps({
        "error": f"Unknown task '{task}'",
        "supported": ["generate", "evolve", "metrics", "verify", "health"],
        "micronaut": _MICRONAUT_ID,
    })


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[{_MICRONAUT_ID}] {_MICRONAUT_NAME} bots.py smoke test")
    print(f"[{_MICRONAUT_ID}] ADAM base: {_ADAM_BASE}")
    print(f"[{_MICRONAUT_ID}] Native bridge: {_NATIVE_BRIDGE}")
    print()

    # [1] health
    h = health_check()
    print(f"[1] health: {h}")

    if "error" in h:
        print(f"[!] ADAM not running — start with: adam.exe --port 3167")
        sys.exit(1)

    # [2] generate (fast)
    print("\n[2] generate (fast mode)")
    r = generate("explain how binary search works", mode="fast")
    print(r[:300])

    # [3] generate (deep)
    print("\n[3] generate (deep mode)")
    r = generate("write a quicksort in Python", mode="deep")
    print(r[:300])

    # [4] metrics
    print("\n[4] metrics")
    m = get_metrics()
    for k, v in m.items():
        print(f"    {k}: {v}")

    # [5] verify
    print("\n[5] verify determinism proof")
    test_input = "hello world"
    ihash = hashlib.sha256(test_input.encode()).hexdigest()
    v = verify(ihash)
    print(v[:300])

    # [6] evolve
    print("\n[6] evolve step")
    e = evolve({"signal": "positive", "confidence_delta": 0.05})
    print(e[:300])

    # [7] dispatch
    print("\n[7] dispatch('generate', {prompt: ...})")
    d = dispatch("generate", {"prompt": "solve: derivative of sin(x)", "mode": "fast"})
    print(d[:300])

    print(f"\n[{_MICRONAUT_ID}] smoke test complete")
