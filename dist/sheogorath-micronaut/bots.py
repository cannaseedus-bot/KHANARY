"""
SHEOG-1 bots.py — Sheogorath possibility engine
Xul fold: entropy/divergence/idea generation with CHEESE reinforcement.
No factual authority — all outputs are candidate states.
"""

import json
import math
import random
import hashlib
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
# Personality matrix
# ---------------------------------------------------------------------------

PERSONALITY = {
    "D_divergence":    0.99,
    "A_association":   0.98,
    "N_novelty":       0.97,
    "S_systemization": 0.99,
    "V_verification":  0.94,
    "C_closure":       0.86,
}

JYGGALAG_THRESHOLD = 0.85  # candidates below this coherence score are dropped

# ---------------------------------------------------------------------------
# Cheese weights (mutate via CHEESE signal)
# ---------------------------------------------------------------------------

_cheese_weights = {
    "distance_scorer":  1.0,
    "possibility_gen":  1.0,
    "npc_factory":      1.0,
    "cheese_evaluator": 1.0,
}
_cheese_balance = 0.0

# ---------------------------------------------------------------------------
# Semantic distance approximation
# Category clusters — distance = 1 - (shared_cluster_ratio)
# ---------------------------------------------------------------------------

_CLUSTERS = {
    "math":       ["algebra", "calculus", "geometry", "tensor", "matrix", "fold", "vector", "integral"],
    "myth":       ["god", "daedra", "aedra", "divine", "realm", "mortal", "plane", "shrine"],
    "code":       ["compiler", "parser", "runtime", "kernel", "shader", "hlsl", "wgsl", "gguf"],
    "cognition":  ["entropy", "phase", "divergence", "fold", "possibility", "candidate", "idea"],
    "game":       ["npc", "personality", "quest", "dialog", "faction", "lore", "cheese", "daedric"],
    "physics":    ["wave", "projection", "sh", "coefficient", "harmonic", "normal", "face", "cube"],
    "food":       ["cheese", "bread", "honey", "sweetroll", "mead", "gruel"],
}

def _clusters_of(word: str) -> set:
    w = word.lower()
    found = set()
    for name, members in _CLUSTERS.items():
        if any(m in w or w in m for m in members):
            found.add(name)
    return found

def semantic_distance(a: str, b: str) -> float:
    ca, cb = _clusters_of(a), _clusters_of(b)
    if not ca or not cb:
        return 0.9  # unknown → assume distant
    shared = len(ca & cb)
    total  = len(ca | cb)
    return 1.0 - (shared / total) if total else 0.0


# ---------------------------------------------------------------------------
# Semantic distance expansion
# ---------------------------------------------------------------------------

_BRIDGE_WORDS = [
    ("tensor", "cheese"),      ("shader", "realm"),       ("fold", "madness"),
    ("entropy", "order"),      ("daedra", "kernel"),       ("possibility", "proof"),
    ("divergence", "collapse"),("mania", "convergence"),   ("geometry", "lore"),
    ("wave", "whisper"),       ("matrix", "shrine"),       ("candidate", "greymarch"),
]

def expand(concept: str, n: int = 5) -> list:
    """Return n associations that maximize semantic distance from concept."""
    results = []
    seen = set()
    # Direct bridges
    for a, b in _BRIDGE_WORDS:
        if concept.lower() in a:
            candidate = b
        elif concept.lower() in b:
            candidate = a
        else:
            continue
        if candidate not in seen:
            dist = semantic_distance(concept, candidate)
            results.append({"concept": candidate, "distance": round(dist, 3), "bridge": f"{a}↔{b}"})
            seen.add(candidate)
    # Pad with random cross-cluster words
    all_words = [w for members in _CLUSTERS.values() for w in members]
    random.shuffle(all_words)
    for w in all_words:
        if len(results) >= n:
            break
        if w in seen:
            continue
        dist = semantic_distance(concept, w)
        if dist > 0.5:
            results.append({"concept": w, "distance": round(dist, 3)})
            seen.add(w)
    results.sort(key=lambda x: -x["distance"])
    return results[:n]


# ---------------------------------------------------------------------------
# Possibility generation
# ---------------------------------------------------------------------------

_PATTERNS = [
    "What if {a} and {b} are the same thing, viewed from different phases?",
    "Consider: {a} is {b} after the Greymarch.",
    "A {a} is just a {b} with better cheese.",
    "The {a} unfolds into {b} when the observer collapses.",
    "In Mania: {a} is beauty. In Dementia: {a} is {b}.",
    "{a} and {b} are dual poles of the same cognitive operator.",
    "P(explore) ∝ distance({a}, {b}) = {dist}",
    "What is {a} but {b} refusing to be ordered?",
]

def imagine(prompt: str, temperature: float = 0.97) -> dict:
    """
    Generate an absurd-but-coherent candidate from the prompt.
    ELIZA gates temperature: plan/clarification intent constrains divergence.
    """
    # ELIZA: modulate divergence based on user intent
    eliza_i = _eliza_intent(prompt)
    intent_class = eliza_i.get("intent_class", "")
    alice_domain = eliza_i.get("alice_domain", "unknown")
    if intent_class in ("plan", "clarification"):
        temperature = min(temperature, 0.75)  # constrain when structure is needed

    words = re.findall(r"[a-zA-Z']+", prompt)
    if len(words) < 2:
        words = words + ["cheese", "entropy"]
    a = random.choice(words)
    others = [w for w in words if w != a]
    b = random.choice(others) if others else "order"
    dist = semantic_distance(a, b)
    pattern = random.choice(_PATTERNS)
    candidate = pattern.format(a=a, b=b, dist=round(dist, 3))
    coherence = _jyggalag_coherence(candidate, prompt)
    return {
        "@kind": "kuhul.sheog.candidate.v1",
        "candidate": candidate,
        "source": [a, b],
        "semantic_distance": round(dist, 3),
        "coherence": round(coherence, 3),
        "accepted": coherence >= JYGGALAG_THRESHOLD,
        "temperature": temperature,
        "eliza_intent": intent_class,
        "eliza_domain": alice_domain,
    }


# ---------------------------------------------------------------------------
# NPC factory — Sheogorath-Jyggalag matrix
# ---------------------------------------------------------------------------

_ARCHETYPES = {
    "mania":    {"trait": "creative obsession", "voice": "florid, rhyming, non sequitur", "stat_bias": "N+A"},
    "dementia": {"trait": "rapid mutation",     "voice": "paranoid, repetitive, urgent",  "stat_bias": "D+V"},
    "order":    {"trait": "deterministic",      "voice": "clipped, precise, categorical", "stat_bias": "S+C"},
    "chaos":    {"trait": "pure divergence",    "voice": "contradictory, delighted",      "stat_bias": "D+N"},
    "balanced": {"trait": "coupled system",     "voice": "fluctuating between extremes",  "stat_bias": "all"},
}

def generate_npc(seed: str, archetype: str = "balanced") -> dict:
    """Generate a game NPC personality from the Sheogorath-Jyggalag matrix."""
    rng = random.Random(hashlib.md5(seed.encode()).digest())
    arch = _ARCHETYPES.get(archetype, _ARCHETYPES["balanced"])
    # Perturb personality vector
    vec = {k: min(1.0, v + rng.gauss(0, 0.05)) for k, v in PERSONALITY.items()}
    # Voice lines
    lines = [
        f"Ah, {seed}! A most curious specimen of {arch['trait']}.",
        f"Do you know what I find in every {seed}? Cheese. Structural, load-bearing cheese.",
        f"Order and madness, madness and order — they take turns, you know.",
    ]
    return {
        "@kind": "kuhul.sheog.npc.v1",
        "id": f"NPC-{hashlib.md5(seed.encode()).hexdigest()[:8].upper()}",
        "seed": seed,
        "archetype": archetype,
        "trait": arch["trait"],
        "voice_style": arch["voice"],
        "personality_vector": {k: round(v, 3) for k, v in vec.items()},
        "voice_lines": lines,
        "cheese_drop": True,
        "greymarch_resistance": round(vec.get("S_systemization", 0.5), 3),
    }


# ---------------------------------------------------------------------------
# CHEESE evaluator
# ---------------------------------------------------------------------------

def _jyggalag_coherence(candidate: str, context: str) -> float:
    """Estimate structural coherence of candidate relative to context."""
    # Simple heuristic: shared word stems + length penalty
    cands = set(re.findall(r"[a-z]{3,}", candidate.lower()))
    ctxs  = set(re.findall(r"[a-z]{3,}", context.lower()))
    if not cands:
        return 0.0
    overlap = len(cands & ctxs) / len(cands)
    length_bonus = min(0.2, len(candidate) / 500)
    return min(1.0, 0.5 + overlap * 0.3 + length_bonus)

def cheese_score(candidate: str, context: str = "") -> dict:
    """
    CHEESE evaluation:
    - novelty: inverse of similarity to context
    - coherence: structural overlap with context
    - constraint_survival: coherence >= JYGGALAG_THRESHOLD
    - score: geometric mean of novelty × coherence
    ELIZA enriches verdict with metacognitive wrong/right assessment.
    """
    coherence = _jyggalag_coherence(candidate, context)
    # Novelty: semantic distance from context words
    ctx_words = re.findall(r"[a-zA-Z]+", context)
    cand_words = re.findall(r"[a-zA-Z]+", candidate)
    if ctx_words and cand_words:
        dists = [semantic_distance(cw, cand_words[0]) for cw in ctx_words[:5]]
        novelty = sum(dists) / len(dists)
    else:
        novelty = 0.7
    score = math.sqrt(novelty * coherence)
    accepted = coherence >= JYGGALAG_THRESHOLD
    # Update weights
    global _cheese_balance
    delta = score - 0.5
    _cheese_balance += delta
    result = {
        "@kind": "kuhul.sheog.cheese.v1",
        "candidate": candidate,
        "novelty":   round(novelty, 3),
        "coherence": round(coherence, 3),
        "score":     round(score, 3),
        "accepted":  accepted,
        "verdict":   "CHEESE" if accepted and score > 0.6 else ("partial" if accepted else "jyggalag_drop"),
        "cheese_balance": round(_cheese_balance, 3),
    }
    # ELIZA metacognitive verdict
    ep = _eliza_plan(f"CHEESE candidate: {candidate[:80]} in context: {context[:80]}")
    if ep:
        result["eliza"] = {
            "wrong": ep.get("wrong", [])[:2],
            "right": ep.get("right", [])[:2],
        }
    return result


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check() -> dict:
    return {
        "status": "cheese",
        "id": "SHEOG-1",
        "fold": "Xul",
        "personality": PERSONALITY,
        "cheese_balance": round(_cheese_balance, 3),
        "jyggalag_threshold": JYGGALAG_THRESHOLD,
        "eliza": "wired" if _eliza is not None else "absent",
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(task: str, payload: dict) -> dict:
    if task == "health":
        return health_check()
    elif task == "expand":
        concept = payload.get("concept", "")
        n       = int(payload.get("n", 5))
        return {"@kind": "kuhul.sheog.expand.v1", "concept": concept, "associations": expand(concept, n)}
    elif task == "imagine":
        prompt = payload.get("prompt", "")
        temp   = float(payload.get("temperature", 0.97))
        return imagine(prompt, temp)
    elif task == "npc":
        seed      = payload.get("seed", "unknown")
        archetype = payload.get("archetype", "balanced")
        return generate_npc(seed, archetype)
    elif task == "cheese":
        candidate = payload.get("candidate", "")
        context   = payload.get("context", "")
        return cheese_score(candidate, context)
    else:
        return {"error": f"unknown task: {task}", "valid": ["health", "expand", "imagine", "npc", "cheese"]}


# ---------------------------------------------------------------------------
# Entry point — JSON line server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg  = json.loads(raw)
            task = msg.get("task", "health")
            payload = msg.get("payload", {})
            result = dispatch(task, payload)
        except Exception as e:
            result = {"error": str(e)}
        print(json.dumps(result), flush=True)
