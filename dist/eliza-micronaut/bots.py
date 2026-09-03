"""
ELIZA-1 bots.py — metacognitive reflection engine
Chen fold: question → compare → plan. Temperature 0.35.
Never executes. Never calls external services.
Structured outputs only: intent, next, works, wrong, right, questions.
"""

import json
import sys
import re

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_INTENT_CLASSES = {
    "action":        r"\b(do|run|build|create|add|delete|fix|change|update|deploy|write|make)\b",
    "question":      r"\b(what|how|why|when|where|which|who|explain|describe|show me)\b",
    "complaint":     r"\b(broken|wrong|bug|error|fail|not working|doesn't work|issue|problem)\b",
    "request":       r"\b(please|can you|could you|would you|i need|i want|i'd like)\b",
    "clarification": r"\b(mean|meant|actually|no not|wait|hold on|i said)\b",
    "plan":          r"\b(plan|roadmap|next|steps|strategy|approach|design|architecture)\b",
}

def intent(message: str) -> dict:
    """Classify user intent from a message."""
    msg_lower = message.lower()
    scores = {}
    for cls, pattern in _INTENT_CLASSES.items():
        scores[cls] = len(re.findall(pattern, msg_lower))

    best = max(scores, key=lambda k: (scores[k], list(_INTENT_CLASSES).index(k)))
    confidence = min(1.0, scores[best] * 0.25 + 0.3)

    sub = None
    if best == "action" and re.search(r"\b(fix|bug|error|broken)\b", msg_lower):
        sub = "debug"
    elif best == "action" and re.search(r"\b(create|build|add|make)\b", msg_lower):
        sub = "implement"
    elif best == "action" and re.search(r"\b(update|change|refactor)\b", msg_lower):
        sub = "modify"
    elif best == "question" and re.search(r"\bhow\b", msg_lower):
        sub = "how_to"
    elif best == "question" and re.search(r"\bwhy\b", msg_lower):
        sub = "why_does"

    open_questions = []
    if confidence < 0.5:
        open_questions.append("What specifically are you trying to accomplish?")
    if scores.get("action", 0) > 0 and scores.get("question", 0) > 0:
        open_questions.append("Are you asking how to do this, or asking me to do it?")

    return {
        "@kind": "kuhul.eliza.intent.v1",
        "intent_class": best,
        "confidence": round(confidence, 2),
        "sub_intent": sub,
        "scores": scores,
        "questions": open_questions,
    }


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

_QUESTION_TEMPLATES = {
    "ambiguity":   "What exactly do you mean by {term}?",
    "scope":       "Should this affect {scope_hint}, or only the part you mentioned?",
    "priority":    "Is {item} the most important part, or is there something more urgent?",
    "constraint":  "Are there constraints I should know about (time, compatibility, style)?",
    "success":     "What does success look like here?",
    "failure":     "What broke, and when did it start?",
    "intent":      "What are you actually trying to accomplish — what's the end goal?",
    "next":        "What's the very next thing that needs to happen?",
    "blockers":    "Is anything blocked right now? What's in the way?",
    "wrong":       "What's currently wrong or not working as expected?",
    "right":       "What is working correctly that we should preserve?",
}

def question(context: str) -> dict:
    """Generate probing questions ordered by diagnostic value."""
    ctx_lower = context.lower()
    selected = []

    if len(context.split()) < 15:
        selected.append(_QUESTION_TEMPLATES["intent"])

    vague = re.findall(r"\b(it|this|that|the thing|stuff|something)\b", ctx_lower)
    if vague:
        selected.append(_QUESTION_TEMPLATES["ambiguity"].format(term=vague[0]))

    if re.search(r"\b(broken|error|fail|bug|wrong|issue)\b", ctx_lower):
        selected.append(_QUESTION_TEMPLATES["failure"])
        selected.append(_QUESTION_TEMPLATES["wrong"])

    if re.search(r"\b(build|create|add|implement|make)\b", ctx_lower):
        selected.append(_QUESTION_TEMPLATES["success"])
        selected.append(_QUESTION_TEMPLATES["scope"].format(scope_hint="the whole system"))

    if len(selected) >= 2:
        selected.append(_QUESTION_TEMPLATES["right"])

    if not selected:
        selected = [_QUESTION_TEMPLATES["intent"], _QUESTION_TEMPLATES["next"]]

    return {
        "@kind": "kuhul.eliza.question.v1",
        "questions": selected[:5],
        "context_length": len(context.split()),
        "signal": "ambiguous" if len(selected) > 3 else "clear",
    }


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

def compare(a: str, b: str, criteria: list = None) -> dict:
    """Compare two options or states. Returns works/does_not_work per side."""
    a_words = set(re.findall(r"[a-z]{3,}", a.lower()))
    b_words = set(re.findall(r"[a-z]{3,}", b.lower()))
    shared  = a_words & b_words

    a_unique = a_words - b_words
    b_unique = b_words - a_words

    a_has_negation = bool(re.search(r"\b(not|no|never|without|missing|fail|broken)\b", a.lower()))
    b_has_negation = bool(re.search(r"\b(not|no|never|without|missing|fail|broken)\b", b.lower()))
    a_has_positive = bool(re.search(r"\b(works|correct|valid|good|right|pass|success)\b", a.lower()))
    b_has_positive = bool(re.search(r"\b(works|correct|valid|good|right|pass|success)\b", b.lower()))

    a_verdict = "works" if a_has_positive and not a_has_negation else \
                "unclear" if not a_has_negation else "does_not_work"
    b_verdict = "works" if b_has_positive and not b_has_negation else \
                "unclear" if not b_has_negation else "does_not_work"

    if a_verdict == "works" and b_verdict != "works":
        overall = "a_preferred"
    elif b_verdict == "works" and a_verdict != "works":
        overall = "b_preferred"
    elif a_verdict == b_verdict:
        overall = "equivalent"
    else:
        overall = "inconclusive"

    applies = criteria if criteria else ["content", "clarity", "completeness"]

    return {
        "@kind": "kuhul.eliza.compare.v1",
        "a": {"text": a[:120], "verdict": a_verdict, "unique_terms": sorted(a_unique)[:8]},
        "b": {"text": b[:120], "verdict": b_verdict, "unique_terms": sorted(b_unique)[:8]},
        "shared_terms": sorted(shared)[:8],
        "criteria": applies,
        "overall": overall,
        "wrong": (["A: " + a[:80]] if a_verdict == "does_not_work" else []) +
                 (["B: " + b[:80]] if b_verdict == "does_not_work" else []),
        "right": (["A: " + a[:80]] if a_verdict == "works" else []) +
                 (["B: " + b[:80]] if b_verdict == "works" else []),
    }


# ---------------------------------------------------------------------------
# Think — deep comparison with verdict
# ---------------------------------------------------------------------------

def think(a: str, b: str) -> dict:
    """Deep comparison of two candidates with an explicit better/worse verdict."""
    result = compare(a, b)
    better = "a" if result["overall"] == "a_preferred" else \
             "b" if result["overall"] == "b_preferred" else "neither"
    worse  = "b" if better == "a" else "a" if better == "b" else "neither"

    why_parts = []
    if result["a"]["unique_terms"]:
        why_parts.append(f"A introduces: {', '.join(result['a']['unique_terms'][:4])}")
    if result["b"]["unique_terms"]:
        why_parts.append(f"B introduces: {', '.join(result['b']['unique_terms'][:4])}")
    if result["shared_terms"]:
        why_parts.append(f"Both share: {', '.join(result['shared_terms'][:4])}")

    verdict = "prefer_a" if better == "a" else "prefer_b" if better == "b" else "no_clear_winner"

    return {
        "@kind": "kuhul.eliza.think.v1",
        "better": better,
        "worse": worse,
        "why": "; ".join(why_parts) if why_parts else "insufficient signal to distinguish",
        "verdict": verdict,
        "detail": result,
    }


# ---------------------------------------------------------------------------
# Plan — structured output: intent + next + works + wrong + right
# ---------------------------------------------------------------------------

def plan(context: str, user_intent: str = None) -> dict:
    """
    Produce a structured plan from context.
    wrong is listed before right (fix before building).
    """
    intent_result = intent(context) if not user_intent else None
    resolved_intent = user_intent or (intent_result["intent_class"] + (
        f"/{intent_result['sub_intent']}" if intent_result and intent_result["sub_intent"] else ""
    ))

    ctx_lower = context.lower()

    wrong = []
    if re.search(r"\b(broken|error|fail|bug|missing|wrong|issue)\b", ctx_lower):
        wrong.append("Something in the context is broken or missing — identify it before proceeding")
    if re.search(r"\bnot\s+\w+ing\b", ctx_lower):
        wrong.append("A process or feature is failing — diagnose root cause first")

    works = []
    if re.search(r"\b(works|correct|valid|pass|good|success)\b", ctx_lower):
        works.append("Parts of the current state are working — preserve them")
    if not wrong:
        works.append("No explicit failures detected in context")

    right = []
    if re.search(r"\b(correct|right|proper|valid|good|best)\b", ctx_lower):
        right.append("The approach direction appears sound — continue on this path")
    if works:
        right.append("Working components should not be modified")

    next_steps = []
    if wrong:
        next_steps.append("Diagnose and fix what is broken before adding anything new")
    if intent_result and intent_result["intent_class"] == "action":
        next_steps.append("Define the exact scope of the action before starting")
        next_steps.append("Verify preconditions are met")
    if intent_result and intent_result["intent_class"] == "plan":
        next_steps.append("List all components involved")
        next_steps.append("Order steps by dependency, not preference")
    next_steps.append("Confirm intent is understood before executing")
    if not wrong:
        next_steps.append("Proceed with the intended action")

    open_qs = []
    if not user_intent:
        open_qs.append("Is the stated intent the real goal, or is there a higher-level objective?")
    if wrong:
        open_qs.append("What caused the failure — code, config, or design?")
    if not works:
        open_qs.append("What parts of the current state are we preserving?")

    return {
        "@kind": "kuhul.eliza.plan.v1",
        "intent":    resolved_intent,
        "wrong":     wrong,
        "right":     right,
        "works":     works,
        "next":      next_steps,
        "questions": open_qs,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health_check() -> dict:
    return {
        "status": "questioning",
        "id": "ELIZA-1",
        "fold": "Chen",
        "role": "metacognitive: question → compare → plan",
        "temperature": 0.35,
        "invariants": [
            "never execute",
            "wrong before right",
            "questions before plans when intent is ambiguous",
        ],
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(task: str, payload: dict) -> dict:
    if task == "health":
        return health_check()
    elif task == "question":
        return question(payload.get("context", ""))
    elif task == "compare":
        return compare(
            payload.get("a", ""),
            payload.get("b", ""),
            payload.get("criteria", None),
        )
    elif task == "plan":
        return plan(
            payload.get("context", ""),
            payload.get("intent", None),
        )
    elif task == "intent":
        return intent(payload.get("message", ""))
    elif task == "think":
        return think(payload.get("a", ""), payload.get("b", ""))
    else:
        return {
            "error": f"unknown task: {task}",
            "valid": ["health", "question", "compare", "plan", "intent", "think"],
        }


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
