"""
JYGG-1 bots.py — Jyggalag order engine
Wo fold: deterministic constraint enforcement and collapse.
Temperature = 0.0 — no randomness, no candidates, only assertions.
Counterpart to SHEOG-1 (Sheogorath).
"""

import json
import sys
import re
import os
import importlib.util

# ---------------------------------------------------------------------------
# ELIZA-1 — metacognitive brain (Chen fold)
# ---------------------------------------------------------------------------

_ELIZA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eliza-micronaut", "bots.py"))
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
# Constants
# ---------------------------------------------------------------------------

JYGG_THRESHOLD = 0.85  # candidates below this coherence are dropped unconditionally

# ---------------------------------------------------------------------------
# Active constraint set (mutated by enforce())
# ---------------------------------------------------------------------------

_constraints = {
    "non_empty":        lambda c: bool(c.strip()),
    "no_contradiction": lambda c: not re.search(r"\b(not not|is not not|never never)\b", c, re.I),
    "max_length":       lambda c: len(c) <= 2048,
    "no_raw_url":       lambda c: not re.search(r"https?://\S{80,}", c),
}

_greymarch_log = []
_greymarch_count = 0


# ---------------------------------------------------------------------------
# Coherence scoring — same logic as SHEOG-1 for parity
# ---------------------------------------------------------------------------

def _coherence(candidate: str, context: str = "") -> float:
    cands = set(re.findall(r"[a-z]{3,}", candidate.lower()))
    ctxs  = set(re.findall(r"[a-z]{3,}", context.lower()))
    if not cands:
        return 0.0
    overlap = len(cands & ctxs) / len(cands) if ctxs else 0.4
    length_bonus = min(0.2, len(candidate) / 500)
    return min(1.0, 0.5 + overlap * 0.3 + length_bonus)


# ---------------------------------------------------------------------------
# Constraint checker
# ---------------------------------------------------------------------------

def validate(candidate: str, constraints: list = None) -> dict:
    """
    Check candidate against the active constraint set (or a named subset).
    Returns pass/fail with per-rule breakdown.
    """
    active = _constraints
    if constraints:
        active = {k: v for k, v in _constraints.items() if k in constraints}

    results = {}
    passed = True
    for name, rule in active.items():
        try:
            ok = rule(candidate)
        except Exception as e:
            ok = False
        results[name] = ok
        if not ok:
            passed = False

    return {
        "@kind": "kuhul.jygg.validate.v1",
        "candidate": candidate[:120] + ("…" if len(candidate) > 120 else ""),
        "passed": passed,
        "rules": results,
        "verdict": "order" if passed else "collapse",
    }


# ---------------------------------------------------------------------------
# Collapse engine — deterministic selection from candidate pool
# ---------------------------------------------------------------------------

def collapse(candidates: list, context: str = "") -> dict:
    """
    Collapse a pool of candidates to the single best survivor.
    Selection: highest coherence above JYGG_THRESHOLD that passes all constraints.
    Ties broken by lexicographic order (deterministic).
    """
    scored = []
    dropped = []

    for i, item in enumerate(candidates):
        text = item if isinstance(item, str) else item.get("candidate", str(item))
        coh  = item.get("coherence", _coherence(text, context)) if isinstance(item, dict) else _coherence(text, context)
        v    = validate(text)

        if coh < JYGG_THRESHOLD or not v["passed"]:
            dropped.append({"index": i, "text": text[:80], "coherence": round(coh, 3), "reason": "below_threshold" if coh < JYGG_THRESHOLD else "constraint_fail"})
            continue
        scored.append({"index": i, "text": text, "coherence": round(coh, 3)})

    _greymarch_log.extend(dropped)

    if not scored:
        return {
            "@kind": "kuhul.jygg.collapse.v1",
            "survivor": None,
            "dropped": len(dropped),
            "total": len(candidates),
            "verdict": "total_collapse",
        }

    # Deterministic: highest coherence, ties broken lexicographically
    scored.sort(key=lambda x: (-x["coherence"], x["text"]))
    winner = scored[0]

    return {
        "@kind": "kuhul.jygg.collapse.v1",
        "survivor": winner["text"],
        "survivor_coherence": winner["coherence"],
        "survivors": len(scored),
        "dropped": len(dropped),
        "total": len(candidates),
        "verdict": "order",
    }


# ---------------------------------------------------------------------------
# Invariant enforcer — hard rule applied to context
# ---------------------------------------------------------------------------

_INVARIANTS = {
    "no_contradiction":  lambda ctx: not re.search(r"\b(not not|is not not)\b", ctx, re.I),
    "schema_present":    lambda ctx: "@kind" in ctx or "kind" in ctx,
    "non_circular":      lambda ctx: ctx.count("itself") < 2,
    "bounded_length":    lambda ctx: len(ctx) < 4096,
    "no_empty_fold":     lambda ctx: "fold" not in ctx or re.search(r'"fold":\s*"[A-Za-z]', ctx) is not None,
    "port_in_range":     lambda ctx: not re.search(r'"port":\s*(\d+)', ctx) or
                                     all(3208 <= int(m) <= 65535 for m in re.findall(r'"port":\s*(\d+)', ctx)),
}

def enforce(rule: str, context: str) -> dict:
    """Apply a named invariant rule to context. Returns pass/fail + reason."""
    if rule not in _INVARIANTS:
        return {
            "@kind": "kuhul.jygg.enforce.v1",
            "rule": rule,
            "passed": False,
            "reason": f"unknown rule '{rule}' — add to _INVARIANTS",
            "known_rules": list(_INVARIANTS.keys()),
        }
    try:
        passed = _INVARIANTS[rule](context)
    except Exception as e:
        passed = False

    return {
        "@kind": "kuhul.jygg.enforce.v1",
        "rule": rule,
        "passed": passed,
        "verdict": "order" if passed else "collapse",
    }


# ---------------------------------------------------------------------------
# Greymarch — total-order collapse event
# ---------------------------------------------------------------------------

def greymarch(candidates: list, context: str = "") -> dict:
    """
    Full Greymarch: evaluate every candidate from SHEOG-1.
    All constraints applied. Only those above JYGG_THRESHOLD survive.
    Logs every dropped candidate. Increments greymarch_count.
    """
    global _greymarch_count
    _greymarch_count += 1

    result = collapse(candidates, context)
    result["@kind"]          = "kuhul.jygg.greymarch.v1"
    result["greymarch_count"] = _greymarch_count
    result["threshold"]       = JYGG_THRESHOLD
    result["partner"]         = "SHEOG-1"

    if result["verdict"] == "total_collapse":
        result["note"] = "The Greymarch is complete. Nothing remains. Order is absolute."
        # ELIZA: what went wrong and what to do when all candidates collapse
        ep = _eliza_plan(context or "all candidates failed the Jyggalag coherence threshold")
        result["eliza_plan"] = {
            "wrong":     ep.get("wrong", [])[:3],
            "next":      ep.get("next", [])[:3],
            "questions": ep.get("questions", [])[:2],
        }
    else:
        result["note"] = f"The Greymarch is complete. {result['survivors']} survived. Order holds."

    return result


# ---------------------------------------------------------------------------
# Constraints list
# ---------------------------------------------------------------------------

def list_constraints() -> dict:
    return {
        "@kind": "kuhul.jygg.constraints.v1",
        "active_constraints": list(_constraints.keys()),
        "active_invariants":  list(_INVARIANTS.keys()),
        "threshold": JYGG_THRESHOLD,
        "greymarch_count": _greymarch_count,
        "greymarch_log_size": len(_greymarch_log),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health_check() -> dict:
    return {
        "status": "order",
        "id": "JYGG-1",
        "fold": "Wo",
        "personality": {
            "D_divergence": 0.05, "A_association": 0.30, "N_novelty": 0.10,
            "S_systemization": 1.00, "V_verification": 1.00, "C_closure": 1.00,
        },
        "temperature": 0.0,
        "threshold": JYGG_THRESHOLD,
        "greymarch_count": _greymarch_count,
        "partner": "SHEOG-1",
        "eliza": "wired" if _eliza is not None else "absent",
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(task: str, payload: dict) -> dict:
    if task == "health":
        return health_check()
    elif task == "validate":
        candidate   = payload.get("candidate", "")
        constraints = payload.get("constraints", None)
        return validate(candidate, constraints)
    elif task == "collapse":
        candidates = payload.get("candidates", [])
        context    = payload.get("context", "")
        return collapse(candidates, context)
    elif task == "enforce":
        rule    = payload.get("rule", "")
        context = payload.get("context", "")
        return enforce(rule, context)
    elif task == "greymarch":
        candidates = payload.get("candidates", [])
        context    = payload.get("context", "")
        return greymarch(candidates, context)
    elif task == "constraints":
        return list_constraints()
    else:
        return {"error": f"unknown task: {task}",
                "valid": ["health", "validate", "collapse", "enforce", "greymarch", "constraints"]}


# ---------------------------------------------------------------------------
# Entry point — JSON line server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg     = json.loads(raw)
            task    = msg.get("task", "health")
            payload = msg.get("payload", {})
            result  = dispatch(task, payload)
        except Exception as e:
            result = {"error": str(e)}
        print(json.dumps(result), flush=True)
