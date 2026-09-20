"""
ELIZA-1 bots.py — metacognitive reflection engine
Chen fold: question → compare → plan. Temperature 0.35.
Never executes. Never calls external services.
Structured outputs only: intent, next, works, wrong, right, questions.

ALICE-1 (Yax fold, port 3210) is ELIZA's semantic sub-µ. ELIZA calls
alice.profile() / alice.classify() to resolve stack-domain meaning
before doing intent classification, question generation, and planning.
Falls back to internal-only logic if alice-micronaut/bots.py is absent.
"""

import json
import sys
import re
import os
import importlib.util

# ---------------------------------------------------------------------------
# Bootstrap ALICE-1 as semantic sub-µ
# ---------------------------------------------------------------------------

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ALICE  = os.path.join(_HERE, "..", "alice-micronaut", "bots.py")
_alice  = None

try:
    _spec  = importlib.util.spec_from_file_location("alice_bots", os.path.normpath(_ALICE))
    _mod   = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _alice = _mod
except Exception:
    pass  # ALICE absent — ELIZA falls back to regex-only mode

def _alice_profile(text: str) -> dict:
    if _alice:
        try:
            return _alice.profile(text)
        except Exception:
            pass
    return {}

def _alice_classify(text: str) -> dict:
    if _alice:
        try:
            return _alice.classify(text)
        except Exception:
            pass
    return {}

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
    """Classify user intent from a message. ALICE resolves semantic domain first."""
    msg_lower = message.lower()
    scores = {}
    for cls, pattern in _INTENT_CLASSES.items():
        scores[cls] = len(re.findall(pattern, msg_lower))

    best = max(scores, key=lambda k: (scores[k], list(_INTENT_CLASSES).index(k)))
    confidence = min(1.0, scores[best] * 0.25 + 0.3)

    # ALICE semantic domain — boosts confidence when domain is known
    alice_cls = _alice_classify(message)
    alice_domain   = alice_cls.get("domain", "unknown")
    alice_conf     = alice_cls.get("confidence", 0.0)
    alice_concepts = alice_cls.get("abstract_concepts", [])

    # Domain-aware confidence boost: if ALICE is confident about domain, lift ELIZA's confidence
    if alice_domain != "unknown" and alice_conf >= 0.3:
        confidence = min(1.0, confidence + alice_conf * 0.2)

    # Domain-specific intent refinements
    if alice_domain == "training" and best == "action":
        scores["plan"] = scores.get("plan", 0) + 2  # training requests usually need a plan
    if alice_domain in ("phase", "inference", "micronaut") and best == "question":
        confidence = min(1.0, confidence + 0.1)

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
        "alice_domain":    alice_domain,
        "alice_concepts":  alice_concepts,
        "alice_scores":    alice_cls.get("scores", {}),
    }


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

_DOMAIN_QUESTIONS = {
    "phase":       "Which K-CUBE phase are you working in (Pop/Wo/Yax/Sek/Chen/Xul)?",
    "face":        "Which K-CUBE face does this concern — Phi, Gram, Geodesic, Projection, or Entropy?",
    "inference":   "Which inference backend is involved — DirectML, llama.cpp, or kuhul-engine?",
    "training":    "Which training stage — forward pass, backward pass, or shard export?",
    "micronaut":   "Which micronaut or agent is responsible for this?",
    "personality": "Is this a CHEESE signal, a personality vector drift, or a greymarch event?",
    "semantic":    "Are you working at the bigram/trigram level, or at the coarse-concept level?",
    "port":        "Is the port question about routing, registration, or process startup?",
}

_CONCEPT_QUESTIONS = {
    "PHASE_LOC":      "Which phase position (fold angle) are you targeting?",
    "INFERENCE_OP":   "Which inference operation is failing or being optimized?",
    "TRAINING_OP":    "Which training operation — forward, backward, gradient accum, or fold clamp?",
    "SEMANTIC_FIELD": "Is this about the gravity well, arc weights, or field projection?",
    "MICRONAUT_ROLE": "Which micronaut role — coordinator, executor, or semantic resolver?",
    "MEMORY_ACCESS":  "Is this a Pop-phase read or Xul-phase write to shared memory?",
    "STACK_ARCH":     "Which stack layer — kuhul-engine, XCFE executor, or SCXQ2 IR?",
    "PERSONALITY":    "Is this a divergence score issue, a CHEESE reward signal, or a personality vector adjustment?",
    "PLANNING_CYCLE": "Are you in the wrong/right analysis step, or the next-steps determination?",
    "COLLAPSE_EVENT": "Are you seeing a greymarch collapse, or anticipating one?",
    "ENTROPY_EVENT":  "Is this a coherent_novelty signal or useless_randomness being rejected?",
    "GPU_RESOURCE":   "Is this a VRAM budget constraint, a DirectML dispatch issue, or a hot-swap decision?",
}

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
    """
    Generate probing questions ordered by diagnostic value.
    ALICE provides semantic profile — domain and abstract concepts drive
    domain-specific questions before generic template fallbacks.
    """
    ctx_lower = context.lower()
    selected  = []

    # ALICE semantic profile — resolves domain + concepts from context
    ap = _alice_profile(context)
    alice_domain    = ap.get("domain", "unknown")
    alice_concepts  = ap.get("abstract_concepts", [])
    alice_resolved  = ap.get("resolved_terms", [])
    alice_unknown   = ap.get("unknown_tokens", [])

    # Domain-specific question (highest priority when domain is known)
    if alice_domain in _DOMAIN_QUESTIONS:
        selected.append(_DOMAIN_QUESTIONS[alice_domain])

    # Concept-specific questions (one per detected concept, capped at 2)
    for concept in alice_concepts[:2]:
        if concept in _CONCEPT_QUESTIONS:
            q = _CONCEPT_QUESTIONS[concept]
            if q not in selected:
                selected.append(q)

    # Unknown tokens — ask user to clarify terms ALICE couldn't resolve
    if alice_unknown and len(alice_unknown) <= 4:
        selected.append(
            f"What do you mean by '{alice_unknown[0]}'? — it's not a recognized stack term."
        )

    # Generic template-based questions
    if len(context.split()) < 15:
        if _QUESTION_TEMPLATES["intent"] not in selected:
            selected.append(_QUESTION_TEMPLATES["intent"])

    vague = re.findall(r"\b(it|this|that|the thing|stuff|something)\b", ctx_lower)
    if vague:
        q = _QUESTION_TEMPLATES["ambiguity"].format(term=vague[0])
        if q not in selected:
            selected.append(q)

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
        "alice_domain":   alice_domain,
        "alice_concepts": alice_concepts,
        "alice_resolved": alice_resolved,
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

_DOMAIN_WRONG = {
    "training":   "Training state may be inconsistent — verify shard CRC and gradient accum before resuming",
    "inference":  "Check VRAM headroom and DirectML session validity before re-running inference",
    "phase":      "A phase transition may have stalled — check K-CUBE face state and SHM flags",
    "micronaut":  "Micronaut registration or port binding may be wrong — verify config.@.toml and lane engine",
    "personality":"Personality vector may have drifted past JYGG-1 threshold — check divergence score",
    "semantic":   "Semantic resolution may be incomplete — unknown tokens in context need classification",
}

_DOMAIN_NEXT = {
    "training":   "Verify xshard tile CRC, then re-run xshard_adapt before any backward pass",
    "inference":  "Confirm model is loaded into resident VRAM budget before sending tokens",
    "phase":      "Confirm Pop-phase SHM read succeeds before scheduling the Wo-phase action",
    "micronaut":  "Check that the micronaut's lane engine and fold processor are both running",
    "personality":"Re-evaluate CHEESE signal before allowing further divergence steps",
    "semantic":   "Run ALICE resolve on the unknown tokens before classifying intent",
}


def plan(context: str, user_intent: str = None) -> dict:
    """
    Produce a structured plan from context.
    wrong is listed before right (fix before building).
    ALICE enriches wrong/next with domain-specific risks.
    """
    intent_result = intent(context) if not user_intent else None
    resolved_intent = user_intent or (intent_result["intent_class"] + (
        f"/{intent_result['sub_intent']}" if intent_result and intent_result["sub_intent"] else ""
    ))

    ctx_lower = context.lower()

    # ALICE semantic profile — domain + concepts for enrichment
    ap = _alice_profile(context)
    alice_domain    = ap.get("domain", "unknown")
    alice_concepts  = ap.get("abstract_concepts", [])
    alice_resolved  = ap.get("resolved_terms", [])
    alice_unknown   = ap.get("unknown_tokens", [])

    wrong = []
    if re.search(r"\b(broken|error|fail|bug|missing|wrong|issue)\b", ctx_lower):
        wrong.append("Something in the context is broken or missing — identify it before proceeding")
    if re.search(r"\bnot\s+\w+ing\b", ctx_lower):
        wrong.append("A process or feature is failing — diagnose root cause first")
    # ALICE domain-specific wrong item
    if alice_domain in _DOMAIN_WRONG:
        dom_wrong = _DOMAIN_WRONG[alice_domain]
        if dom_wrong not in wrong:
            wrong.append(dom_wrong)
    # Unknown tokens are unresolved semantic risk
    if alice_unknown:
        wrong.append(
            f"Unresolved terms in context: {', '.join(alice_unknown[:4])} — classify before acting"
        )

    works = []
    if re.search(r"\b(works|correct|valid|pass|good|success)\b", ctx_lower):
        works.append("Parts of the current state are working — preserve them")
    if not wrong:
        works.append("No explicit failures detected in context")
    if alice_resolved:
        works.append(f"ALICE resolved: {', '.join(alice_resolved[:6])}")

    right = []
    if re.search(r"\b(correct|right|proper|valid|good|best)\b", ctx_lower):
        right.append("The approach direction appears sound — continue on this path")
    if works:
        right.append("Working components should not be modified")

    next_steps = []
    if wrong:
        next_steps.append("Diagnose and fix what is broken before adding anything new")
    # ALICE domain-specific next step (prepend — it's more specific)
    if alice_domain in _DOMAIN_NEXT:
        next_steps.insert(0, _DOMAIN_NEXT[alice_domain])
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
    if alice_unknown:
        open_qs.append(
            f"What are these terms: {', '.join(alice_unknown[:3])}? ALICE could not resolve them."
        )

    return {
        "@kind": "kuhul.eliza.plan.v1",
        "intent":    resolved_intent,
        "wrong":     wrong,
        "right":     right,
        "works":     works,
        "next":      next_steps,
        "questions": open_qs,
        "semantic": {
            "alice_domain":    alice_domain,
            "alice_concepts":  alice_concepts,
            "alice_resolved":  alice_resolved,
            "alice_unknown":   alice_unknown,
        },
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health_check() -> dict:
    alice_status = "wired" if _alice is not None else "absent (regex-only fallback)"
    return {
        "status": "questioning",
        "id": "ELIZA-1",
        "fold": "Chen",
        "role": "metacognitive: question → compare → plan",
        "temperature": 0.35,
        "sub_mu": {
            "id": "ALICE-1",
            "fold": "Yax",
            "port": 3210,
            "status": alice_status,
            "provides": ["profile", "classify", "resolve"],
            "used_in": ["intent", "question", "plan"],
        },
        "invariants": [
            "never execute",
            "wrong before right",
            "questions before plans when intent is ambiguous",
            "alice resolves semantics before eliza classifies intent",
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
