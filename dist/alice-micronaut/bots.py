"""
ALICE-1 bots.py — semantic resolver + gram engine
Yax fold: classify, enumerate, resolve meaning. Sub-µ of ELIZA-1.
Three gram layers: bigrams (local pairs) + trigrams (local triples)
+ coarse grams (concept-window chunks, n=4 default).
Pure Python. No external calls. No generation.
"""

import json
import sys
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    """Lowercase alphanumeric+underscore tokens. Strips punctuation."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def _clean(words: list) -> list:
    """Remove very short noise tokens."""
    stop = {"a", "an", "the", "is", "in", "of", "to", "and", "or", "it",
            "be", "as", "at", "by", "do", "if", "on", "we"}
    return [w for w in words if len(w) > 1 and w not in stop]


# ---------------------------------------------------------------------------
# Gram extraction
# ---------------------------------------------------------------------------

def bigrams(text: str) -> dict:
    """
    Sliding window of 2. Returns sorted list of {gram, count}.
    Captures local word pairs — immediate syntactic neighbours.
    """
    words = _tokenize(text)
    pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    counts = Counter(pairs)
    return {
        "@kind": "kuhul.alice.bigrams.v1",
        "length": len(words),
        "unique_bigrams": len(counts),
        "grams": [
            {"gram": list(k), "count": v}
            for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ],
    }


def trigrams(text: str) -> dict:
    """
    Sliding window of 3. Captures local phrase patterns and co-occurrences.
    More specific than bigrams — good for detecting stack-specific phrases.
    """
    words = _tokenize(text)
    triples = [(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)]
    counts = Counter(triples)
    return {
        "@kind": "kuhul.alice.trigrams.v1",
        "length": len(words),
        "unique_trigrams": len(counts),
        "grams": [
            {"gram": list(k), "count": v}
            for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ],
    }


def coarse_grams(text: str, n: int = 4) -> dict:
    """
    Sliding window of n (default 4). Captures concept-level chunks —
    broader than trigrams, coarser than full sentences.
    Good for detecting argument structures and multi-word concepts.
    Also emits abstract concept signatures by mapping known synonyms
    to canonical coarse labels (e.g. 'fold position' and 'phase angle'
    both emit PHASE_LOC).
    """
    words = _tokenize(text)
    windows = [tuple(words[i: i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(windows)

    # Concept abstraction — map known phrase patterns to coarse labels
    abstract = []
    joined = " ".join(words)
    for label, patterns in _COARSE_CONCEPTS.items():
        for pat in patterns:
            if pat in joined:
                abstract.append(label)
                break

    return {
        "@kind": "kuhul.alice.coarse_grams.v1",
        "window_size": n,
        "length": len(words),
        "unique_coarse_grams": len(counts),
        "grams": [
            {"gram": list(k), "count": v}
            for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:20]
        ],
        "abstract_concepts": sorted(set(abstract)),
    }


# ---------------------------------------------------------------------------
# Coarse concept map — known multi-word concepts → canonical labels
# ---------------------------------------------------------------------------

_COARSE_CONCEPTS = {
    "PHASE_LOC":       ["fold position", "phase angle", "k cube face", "cube face"],
    "INFERENCE_OP":    ["forward pass", "matmul op", "softmax", "attention head", "token embed"],
    "TRAINING_OP":     ["backward pass", "gradient accum", "xshard adapt", "fold clamp", "adam step"],
    "SEMANTIC_FIELD":  ["semantic cube", "gravity well", "arc weight", "field exec", "projection face"],
    "MICRONAUT_ROLE":  ["swarm role", "agent id", "micronaut id", "fold engine", "expert pool"],
    "MEMORY_ACCESS":   ["shm read", "shm write", "pop phase", "xul phase", "memory cycle"],
    "STACK_ARCH":      ["kuhul engine", "khanary server", "xcfe executor", "scxq2 mode"],
    "PERSONALITY":     ["personality vector", "divergence score", "cheese eval", "coherent novelty"],
    "PLANNING_CYCLE":  ["next steps", "what works", "what is wrong", "user intent", "what is right"],
    "COLLAPSE_EVENT":  ["greymarch", "jygg threshold", "below threshold", "total collapse"],
    "ENTROPY_EVENT":   ["expand concept", "sheogorath", "coherent novelty", "semantic distance"],
    "GPU_RESOURCE":    ["vram budget", "directml", "hd 4600", "resident ceiling", "hot swap"],
}


# ---------------------------------------------------------------------------
# Semantic domain vocabulary — stack-aware term → meaning
# ---------------------------------------------------------------------------

_DOMAIN_VOCAB = {
    # Folds / phases
    "pop":      ("phase", "Pop phase — memory read / state observation. Angle 0. Face: Phi."),
    "wo":       ("phase", "Wo phase — scheduling / planning. Angle π/3. Face: Fold."),
    "yax":      ("phase", "Yax phase — classification / enumeration. Angle 2π/3. Face: Gram."),
    "sek":      ("phase", "Sek phase — dispatch / execution. Angle π. Face: Geodesic."),
    "chen":     ("phase", "Chen/Ch'en phase — metacognition / verification. Angle 4π/3. Face: Projection."),
    "xul":      ("phase", "Xul phase — emission / entropy output. Angle 5π/3. Face: Entropy."),
    "fold":     ("phase", "A K-CUBE phase position. Each fold maps to a face angle and micronaut role."),

    # K-CUBE faces
    "phi":        ("face", "Phi face — intent confidence. Pop phase."),
    "gram":       ("face", "Gram face — structural correctness of plan. Yax phase."),
    "geodesic":   ("face", "Geodesic face — distance from user goal. Sek phase."),
    "projection": ("face", "Projection face — current best 'what next' answer. Chen phase."),
    "entropy":    ("face", "Entropy face — ambiguity / chaos level. Xul phase."),

    # Micronauts
    "eliza":      ("micronaut", "ELIZA-1 — metacognitive questioner/thinker/planner. Chen fold. Port 3209."),
    "alice":      ("micronaut", "ALICE-1 — semantic resolver + gram engine. Yax fold. Port 3210. Sub-µ of ELIZA."),
    "sheogorath": ("micronaut", "SHEOG-1 — entropy engine / possibility generator. Xul fold. Port 3213."),
    "jyggalag":   ("micronaut", "JYGG-1 — order engine / collapse evaluator. Wo fold. Port 3214. Greymarch threshold 0.85."),
    "cube":       ("micronaut", "CUBE-1 — K-CUBE geometry engine. Port 3211."),
    "adam":       ("micronaut", "AM-1 — adaptive router / n-gram bigram routing. Sek fold. Port 3208."),
    "regex":      ("micronaut", "REGEX-1 — pattern matcher. Port 3212."),

    # Training
    "xshard":     ("training", "xshard — DDS/SCXQDDS tile-based training format. Used by xshard_adapt+xshard_backward."),
    "scxqdds":    ("training", "SCXQDDS — quantized expert shard format. INT8 tensor records with CRC validation."),
    "xshard_adapt":   ("training", "GPU forward-pass kernel for xshard tiles. DML/HLSL."),
    "xshard_backward":("training", "GPU backward-pass gradient kernel for xshard tiles."),
    "adam":       ("training", "Adam optimizer — KXC registered kernel, adam_optimizer class."),
    "gradient":   ("training", "Gradient accumulation — KXC registered, gradient_accum class."),
    "fold_clamp": ("training", "Fold clamp — clips gradient to fold-valid range. KXC kernel."),

    # Inference
    "gguf":       ("inference", "GGUF — quantized model format (Q4/Q8). Used by llama.cpp."),
    "directml":   ("inference", "DirectML — Microsoft DML inference backend. Used on HD 4600."),
    "mgguf":      ("inference", "MGGUF — khanary mixture-of-experts GGUF format."),
    "scxq2":      ("inference", "SCXQ2 — khanary IR bytecode. Binary format with mode bits and phase system."),
    "xcfe":       ("inference", "XCFE — khanary executor backend. Runs SCXQ2 mode bits."),
    "klsl":       ("inference", "KLSL — khanary shader language. Compiles to HLSL/WGSL for GPU dispatch."),
    "kuhul":      ("inference", "kuhul-engine — main inference + routing daemon. Port 17474."),

    # Semantic
    "semantic":   ("semantic", "Semantic — relating to meaning. In this stack: K-CUBE face projection, domain mapping."),
    "bigram":     ("semantic", "Bigram — two-word local pair. Captures syntactic neighbourhood."),
    "trigram":    ("semantic", "Trigram — three-word local triple. Captures phrase-level patterns."),
    "coarse_gram":("semantic", "Coarse gram — n-word concept window (n≥4). Captures argument structure."),
    "gram":       ("semantic", "Gram — general n-gram. Alice resolves all three layers (bi/tri/coarse)."),
    "intent":     ("semantic", "Intent — what the user is trying to accomplish. ELIZA classifies; Alice resolves term meaning within it."),
    "resolve":    ("semantic", "Resolve — Alice's core operation: given a term or concept, return its meaning in stack context."),

    # CHEESE / personality
    "cheese":     ("personality", "CHEESE — reinforcement signal. Positive: coherent_novelty. Negative: useless_randomness."),
    "coherent_novelty":   ("personality", "Positive CHEESE signal — a divergent idea that is still meaningful."),
    "useless_randomness": ("personality", "Negative CHEESE signal — divergent but incoherent; rejected by JYGG-1."),
    "greymarch":  ("personality", "Greymarch — JYGG-1 periodic total-order collapse. Candidates below 0.85 coherence are dropped."),
    "personality_vector": ("personality", "6-axis vector: D=Divergence / A=Association / N=Novelty / S=Systemization / V=Verification / C=Closure."),

    # Ports
    "17474":  ("port", "kuhul-engine — main inference daemon."),
    "8764":   ("port", "PRIMEOS WebX gateway."),
    "3208":   ("port", "AM-1 — adaptive router."),
    "3209":   ("port", "ELIZA-1 — metacognitive questioner/planner."),
    "3210":   ("port", "ALICE-1 — semantic resolver (this service)."),
    "3211":   ("port", "CUBE-1 — K-CUBE geometry."),
    "3212":   ("port", "REGEX-1 — pattern matcher."),
    "3213":   ("port", "SHEOG-1 — entropy engine."),
    "3214":   ("port", "JYGG-1 — order engine."),
}

# Domain priority for classify() tie-breaking
_DOMAIN_ORDER = ["phase", "face", "micronaut", "inference", "training",
                 "semantic", "personality", "port"]


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def resolve(text: str, domain: str = None) -> dict:
    """
    Resolve the meaning of a term or phrase in khanary stack context.
    Looks up each token in _DOMAIN_VOCAB. Returns best matches.
    """
    words = _tokenize(text)
    hits = []

    # Exact token lookup
    for w in words:
        if w in _DOMAIN_VOCAB:
            dom, meaning = _DOMAIN_VOCAB[w]
            if domain and dom != domain:
                continue
            hits.append({"term": w, "domain": dom, "meaning": meaning, "score": 1.0})

    # Multi-word lookup (bigram keys)
    bg = bigrams(text)
    for entry in bg["grams"]:
        key = "_".join(entry["gram"])
        if key in _DOMAIN_VOCAB:
            dom, meaning = _DOMAIN_VOCAB[key]
            if domain and dom != domain:
                continue
            hits.append({"term": key, "domain": dom, "meaning": meaning,
                         "score": 0.9, "gram": entry["gram"]})

    # Deduplicate by term, keep highest score
    seen = {}
    for h in hits:
        t = h["term"]
        if t not in seen or h["score"] > seen[t]["score"]:
            seen[t] = h
    unique = sorted(seen.values(), key=lambda x: (-x["score"], x["term"]))

    # Build abstract concepts
    cg = coarse_grams(text)
    abstract = cg.get("abstract_concepts", [])

    return {
        "@kind": "kuhul.alice.resolve.v1",
        "input": text,
        "domain_filter": domain,
        "resolved": unique,
        "abstract_concepts": abstract,
        "total_hits": len(unique),
        "unknown_tokens": [w for w in _clean(words)
                           if w not in _DOMAIN_VOCAB and
                           "_".join([w]) not in _DOMAIN_VOCAB],
    }


def classify(text: str, domains: list = None) -> dict:
    """
    Classify text into a semantic domain.
    Uses vocabulary hit counts per domain to score, weighted by bigram overlap.
    """
    words = _tokenize(text)
    domain_scores = Counter()

    for w in words:
        if w in _DOMAIN_VOCAB:
            dom, _ = _DOMAIN_VOCAB[w]
            domain_scores[dom] += 1

    # Coarse concept boost
    cg = coarse_grams(text)
    concept_domain_map = {
        "PHASE_LOC":    "phase",
        "INFERENCE_OP": "inference",
        "TRAINING_OP":  "training",
        "SEMANTIC_FIELD": "semantic",
        "MICRONAUT_ROLE": "micronaut",
        "MEMORY_ACCESS": "phase",
        "STACK_ARCH":   "inference",
        "PERSONALITY":  "personality",
        "PLANNING_CYCLE": "semantic",
        "COLLAPSE_EVENT": "personality",
        "ENTROPY_EVENT": "personality",
        "GPU_RESOURCE": "inference",
    }
    for concept in cg.get("abstract_concepts", []):
        mapped = concept_domain_map.get(concept)
        if mapped:
            domain_scores[mapped] += 2  # concept match is worth more

    # Filter to requested domains if given
    if domains:
        domain_scores = Counter({k: v for k, v in domain_scores.items() if k in domains})

    if not domain_scores:
        top_domain = "unknown"
        confidence = 0.0
    else:
        total = sum(domain_scores.values())
        top_domain, top_count = domain_scores.most_common(1)[0]
        confidence = round(min(1.0, top_count / max(1, len(words)) * 2), 2)

    return {
        "@kind": "kuhul.alice.classify.v1",
        "input": text,
        "domain": top_domain,
        "confidence": confidence,
        "scores": dict(domain_scores.most_common()),
        "abstract_concepts": cg.get("abstract_concepts", []),
    }


def similarity(a: str, b: str) -> dict:
    """
    Semantic similarity between two texts using gram overlap at all three levels.
    Weighted: bigram (0.3) + trigram (0.4) + coarse (0.3).
    Returns 0.0–1.0 score.
    """
    def jaccard(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        return len(set_a & set_b) / len(union) if union else 0.0

    a_words = _tokenize(a)
    b_words = _tokenize(b)

    a_bi = set(zip(a_words, a_words[1:]))
    b_bi = set(zip(b_words, b_words[1:]))
    bi_score = jaccard(a_bi, b_bi)

    a_tri = set(zip(a_words, a_words[1:], a_words[2:]))
    b_tri = set(zip(b_words, b_words[1:], b_words[2:]))
    tri_score = jaccard(a_tri, b_tri)

    n = 4
    a_cg = set(tuple(a_words[i:i+n]) for i in range(len(a_words)-n+1))
    b_cg = set(tuple(b_words[i:i+n]) for i in range(len(b_words)-n+1))
    cg_score = jaccard(a_cg, b_cg)

    # Unigram (vocab domain overlap)
    a_dom = set(w for w in a_words if w in _DOMAIN_VOCAB)
    b_dom = set(w for w in b_words if w in _DOMAIN_VOCAB)
    dom_score = jaccard(a_dom, b_dom)

    combined = round(bi_score * 0.25 + tri_score * 0.30 + cg_score * 0.20 + dom_score * 0.25, 4)

    return {
        "@kind": "kuhul.alice.similarity.v1",
        "a": a[:80],
        "b": b[:80],
        "bigram_score":   round(bi_score, 4),
        "trigram_score":  round(tri_score, 4),
        "coarse_score":   round(cg_score, 4),
        "domain_score":   round(dom_score, 4),
        "combined":       combined,
        "verdict": "similar" if combined > 0.4 else "related" if combined > 0.15 else "distinct",
    }


def profile(text: str) -> dict:
    """
    Full gram profile of a text: bigrams + trigrams + coarse grams + semantic classification.
    Alice's primary report for ELIZA to consume when resolving meaning of an input.
    """
    bg  = bigrams(text)
    tg  = trigrams(text)
    cg  = coarse_grams(text)
    cls = classify(text)
    res = resolve(text)

    return {
        "@kind": "kuhul.alice.profile.v1",
        "input": text,
        "token_count": bg["length"],
        "domain": cls["domain"],
        "domain_confidence": cls["confidence"],
        "abstract_concepts": cg["abstract_concepts"],
        "resolved_terms": [h["term"] for h in res["resolved"]],
        "unknown_tokens": res["unknown_tokens"],
        "bigrams":   {"unique": bg["unique_bigrams"],  "top": bg["grams"][:8]},
        "trigrams":  {"unique": tg["unique_trigrams"], "top": tg["grams"][:8]},
        "coarse_grams": {"window": cg["window_size"], "unique": cg["unique_coarse_grams"],
                         "top": cg["grams"][:8]},
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health_check() -> dict:
    return {
        "status": "semantic",
        "id": "ALICE-1",
        "fold": "Yax",
        "role": "semantic resolver — bigrams / trigrams / coarse grams / domain classification",
        "parent": "ELIZA-1",
        "vocab_size": len(_DOMAIN_VOCAB),
        "concept_classes": len(_COARSE_CONCEPTS),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(task: str, payload: dict) -> dict:
    if task == "health":
        return health_check()
    elif task == "resolve":
        return resolve(payload.get("text", ""), payload.get("domain", None))
    elif task == "bigrams":
        return bigrams(payload.get("text", ""))
    elif task == "trigrams":
        return trigrams(payload.get("text", ""))
    elif task in ("coarse_grams", "coarse"):
        return coarse_grams(payload.get("text", ""), payload.get("n", 4))
    elif task == "profile":
        return profile(payload.get("text", ""))
    elif task == "classify":
        return classify(payload.get("text", ""), payload.get("domains", None))
    elif task == "similarity":
        return similarity(payload.get("a", ""), payload.get("b", ""))
    else:
        return {
            "error": f"unknown task: {task}",
            "valid": ["health", "resolve", "bigrams", "trigrams",
                      "coarse_grams", "profile", "classify", "similarity"],
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
