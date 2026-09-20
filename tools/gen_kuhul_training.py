#!/usr/bin/env python3
"""
gen_kuhul_training.py -- Generate synthetic pi-KUHUL structured training examples.

Seeds from khanary_transitions.jsonl (state-transition records) and emits
{"text": "..."} JSONL directly compatible with the gpt2_trainer pipeline:
  tokenize_transitions.py -> pack_tokens.py -> gpt2_trainer.exe

Three example types per seed transition:
  A  agent_think   -- [USER]/[THOUGHT]/[AGENT] neuro-symbolic dialog with <THINK>/<RESPONSE>
  B  state_trans   -- [TRANSITION] algebraic record (F(t+1)=Preserve(F(t))+Delta(t))
  C  phase_trace   -- Pop->Wo->Yax->Sek->Ch'en->Xul execution trace

Plus ~15% violation records (valid=false) drawn from cross-wired state pairs
to teach the model the manifold boundary (valid vs violation).

Weight tensor (5 floats, embedded as text in [THOUGHT]):
  g0: preserve_ratio      -- fraction of context carried forward
  g1: delta_entropy       -- magnitude of new information
  g2: confidence          -- op-based certainty
  g3: op_index            -- op family encoding [0,1]
  g4: temporal_position   -- deterministic pi-time (seeded from state hash)

Usage:
  python tools/gen_kuhul_training.py E:/data/khanary_transitions.jsonl \\
    -o E:/data/kuhul_synthetic.jsonl
  python tools/gen_kuhul_training.py E:/data/khanary_transitions.jsonl \\
    -o E:/data/kuhul_synthetic.jsonl --max 5000 --seed 42
"""

import sys, json, argparse, random, math, hashlib
from pathlib import Path

# ─── Op metadata (LAW P1: Phase INTERSECT Opcode = empty set) ─────────────────

# K'UHUL phase assigned to each op
_OP_PHASE = {
    "DEFINE":    "Wo",       # planning / structuring
    "INSTRUCT":  "Sek",      # direct execution
    "COMPARE":   "Yax",      # branching / evaluating
    "ENUMERATE": "Yax",      # branching / listing
    "TRANSFORM": "Sek",      # transforming / modifying
    "GENERATE":  "Sek",      # producing / synthesizing
    "COMPUTE":   "Sek",      # calculating
    "LOCATE":    "Pop",      # observing / finding
    "EXPLAIN":   "Ch'en",    # verifying / clarifying
}

_OP_CONFIDENCE = {
    "DEFINE":    0.85,
    "INSTRUCT":  0.90,
    "COMPARE":   0.75,
    "ENUMERATE": 0.80,
    "TRANSFORM": 0.88,
    "GENERATE":  0.72,
    "COMPUTE":   0.95,
    "LOCATE":    0.82,
    "EXPLAIN":   0.78,
}

# Residency mutation class from the F(t+1)=Preserve+Delta algebra
_OP_MUTATION = {
    "DEFINE":    "STATIC",
    "INSTRUCT":  "GROWING",
    "COMPARE":   "TRANSIENT",
    "ENUMERATE": "GROWING",
    "TRANSFORM": "TRANSIENT",
    "GENERATE":  "GROWING",
    "COMPUTE":   "TRANSIENT",
    "LOCATE":    "STATIC",
    "EXPLAIN":   "GROWING",
}

# Normalized op index for weight tensor dimension g3
_OP_NAMES_SORTED = sorted(_OP_PHASE.keys())
_OP_INDEX = {op: i / max(len(_OP_NAMES_SORTED) - 1, 1)
             for i, op in enumerate(_OP_NAMES_SORTED)}

_PHASES_ORDERED = ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"]


# ─── Weight tensor derivation ──────────────────────────────────────────────────

def derive_weight_tensor(op: str, preserve, delta, seed_str: str) -> list:
    """
    Returns 5 floats in [0,1] as geometric bias coefficients g_ij.
    Deterministic: same inputs always produce the same tensor.
    """
    p_len = len(preserve) if isinstance(preserve, list) else len(str(preserve).split())
    d_len = len(delta)    if isinstance(delta,    list) else len(str(delta).split())
    total = p_len + d_len
    g0 = p_len / total if total > 0 else 0.5                    # preserve_ratio
    g1 = min(1.0, d_len / 20.0)                                 # delta_entropy
    g2 = _OP_CONFIDENCE.get(op, 0.80)                           # confidence
    g3 = _OP_INDEX.get(op, 0.5)                                 # op_index
    h  = int(hashlib.sha1(seed_str.encode("utf-8", errors="replace")).hexdigest()[:8], 16)
    g4 = (h % 1000) / 1000.0                                    # temporal_position
    return [round(g0, 3), round(g1, 3), round(g2, 3), round(g3, 3), round(g4, 3)]


def _wt_str(wt: list) -> str:
    return "[" + ", ".join(f"{x:.3f}" for x in wt) + "]"


# ─── Text helpers ──────────────────────────────────────────────────────────────

def _trunc(s: str, n: int = 120) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + "..."


def _word_list(words, limit: int = 12) -> str:
    if isinstance(words, list):
        return ", ".join(str(w) for w in words[:limit])
    return str(words)[:100]


def _first_line(text: str, n: int = 80) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:n]
    return text[:n]


def _get(rec: dict, *keys):
    for k in keys:
        v = rec.get(k)
        if v is not None:
            return v
    return ""


# ─── Example type A: agent_think ──────────────────────────────────────────────

_INSTRUCT_PREAMBLE = (
    "You are a pi-KUHUL runtime agent operating under the KHANARY semantic substrate. "
    "Apply the state-transition law F(t+1) = Preserve(F(t)) + Delta(t). "
    "Respond using K'UHUL phases: Pop -> Wo -> Yax -> Sek -> Ch'en -> Xul."
)

def gen_agent_think(rec: dict, wt: list) -> str:
    op     = rec.get("op", "EXPLAIN")
    lane   = rec.get("lane", "general")
    phase  = _OP_PHASE.get(op, "Sek")
    conf   = _OP_CONFIDENCE.get(op, 0.80)
    mut    = _OP_MUTATION.get(op, "GROWING")
    pres   = rec.get("preserve") or []
    delt   = rec.get("delta") or []
    state  = (_get(rec, "state") or "").strip()
    s_next = (_get(rec, "state_next") or "").strip()
    pi_t   = round(wt[4], 4)  # fraction of pi: 0.0-1.0

    pwords = _word_list(pres)
    dwords = _word_list(delt)
    st_sum = _first_line(state, 50)
    dn_sum = _word_list(delt, 6) or "new-context"

    return "\n".join([
        "[INSTRUCT]",
        _INSTRUCT_PREAMBLE,
        "[/INSTRUCT]",
        "",
        "[USER]",
        state or "[empty state]",
        "[/USER]",
        "",
        "[THOUGHT]",
        f"FOCUS: {op} on '{st_sum}';",
        f"AGENDA: [{op}, verify-continuity, commit];",
        f"INFER: '{dn_sum}' extends state with confidence {conf:.2f};",
        f"MUTATION: {mut};",
        f"PHASE: {phase};",
        f"WEIGHT: {_wt_str(wt)};",
        f"pi-time: {pi_t:.4f}pi;",
        "[/THOUGHT]",
        "",
        "[AGENT]",
        "<THINK>",
        f'  @apply "{op}" to "{st_sum}";',
        f"  PRESERVE: [{pwords}];",
        f"  DELTA: [{dwords}];",
        "</THINK>",
        "<RESPONSE>",
        s_next or "[empty next state]",
        "</RESPONSE>",
        "[/AGENT]",
    ])


# ─── Example type B: state_transition algebraic record ────────────────────────

def gen_state_transition(rec: dict, wt: list) -> str:
    op     = rec.get("op", "EXPLAIN")
    lane   = rec.get("lane", "general")
    phase  = _OP_PHASE.get(op, "Sek")
    conf   = _OP_CONFIDENCE.get(op, 0.80)
    mut    = _OP_MUTATION.get(op, "GROWING")
    pres   = rec.get("preserve") or []
    delt   = rec.get("delta") or []
    state  = (_get(rec, "state") or "").strip()
    s_next = (_get(rec, "state_next") or "").strip()

    return "\n".join([
        "[TRANSITION]",
        f"FOLD: {phase}",
        f"LANE: {lane}",
        f"OP: {op}",
        f"MUTATION: {mut}",
        f"CONFIDENCE: {conf:.2f}",
        f"WEIGHT: {_wt_str(wt)}",
        f"STATE_T: {_trunc(state)}",
        f"PRESERVE: [{_word_list(pres)}]",
        f"DELTA: [{_word_list(delt)}]",
        f"STATE_T1: {_trunc(s_next)}",
        "VALID: true",
        "LAW: F(t+1) = Preserve(F(t)) + Delta(t)",
    ])


# ─── Example type C: phase_trace (Pop->Wo->Yax->Sek->Ch'en->Xul) ─────────────

def gen_phase_trace(rec: dict, wt: list) -> str:
    op     = rec.get("op", "EXPLAIN")
    lane   = rec.get("lane", "general")
    phase  = _OP_PHASE.get(op, "Sek")
    conf   = _OP_CONFIDENCE.get(op, 0.80)
    mut    = _OP_MUTATION.get(op, "GROWING")
    pres   = rec.get("preserve") or []
    delt   = rec.get("delta") or []
    state  = (_get(rec, "state") or "").strip()
    s_next = (_get(rec, "state_next") or "").strip()

    pwords = _word_list(pres)
    dwords = _word_list(delt)
    st_s   = _first_line(state, 60)
    sn_s   = _first_line(s_next, 60)

    # One line per phase; mark the active phase
    def pline(p: str, body: str) -> str:
        marker = " [ACTIVE]" if p == phase else ""
        return f"{p}:{marker}  {body}"

    return "\n".join([
        "[PHASE_TRACE]",
        f"OP: {op}",
        f"LANE: {lane}",
        f"MUTATION: {mut}",
        "",
        pline("Pop",    f"observe state_t='{st_s}'; lane='{lane}';"),
        pline("Wo",     f"schedule {op}; preserve=[{pwords}]; delta=[{dwords}];"),
        pline("Yax",    f"branch -- apply? confidence={conf:.2f} >= 0.50 -> YES;"),
        pline("Sek",    f"execute {op}; emit state_t1='{sn_s}';"),
        pline("Ch'en",  f"verify F(t+1)=Preserve(F(t))+Delta(t); MUTATION={mut}; VALID=true;"),
        pline("Xul",    f"commit state_t1; WEIGHT={_wt_str(wt)};"),
        "",
        f"RESULT: F(t+1) = Preserve(F(t)) + Delta(t);  VALID=true;  CONFIDENCE={conf:.2f};",
    ])


# ─── Example type D: violation (valid=false) ──────────────────────────────────

def gen_violation(rec: dict, decoy_next: str, rng: random.Random) -> str:
    op    = rec.get("op", "EXPLAIN")
    lane  = rec.get("lane", "general")
    state = (_get(rec, "state") or "").strip()

    wrong_ops   = [o for o in _OP_PHASE if o != op]
    wrong_op    = rng.choice(wrong_ops) if wrong_ops else "GENERATE"
    wrong_phase = _OP_PHASE.get(wrong_op, "Sek")

    return "\n".join([
        "[TRANSITION]",
        f"FOLD: {wrong_phase}",
        f"LANE: {lane}",
        f"OP: {wrong_op}",
        "MUTATION: TRANSIENT",
        "CONFIDENCE: 0.00",
        f"STATE_T: {_trunc(state)}",
        "PRESERVE: []",
        "DELTA: [corrupted]",
        f"STATE_T1: {_trunc(decoy_next)}",
        "VALID: false",
        "VIOLATION: continuity_break -- state jumped without valid Preserve(F(t))+Delta(t) path",
        "LAW: F(t+1) = Preserve(F(t)) + Delta(t)",
    ])


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic pi-KUHUL training examples from khanary_transitions.jsonl"
    )
    ap.add_argument("input",  help="Path to khanary_transitions.jsonl")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL path")
    ap.add_argument("--max",  type=int, default=0, metavar="N",
                    help="Max seed records to read (0=all)")
    ap.add_argument("--seed", type=int, default=1337,
                    help="Random seed for shuffling and violation selection (default 1337)")
    ap.add_argument("--violation-rate", type=float, default=0.15,
                    help="Fraction of total examples to emit as violations (default 0.15)")
    a = ap.parse_args()

    rng = random.Random(a.seed)

    # ── Pass 1: read seed records ──────────────────────────────────────────────
    records  = []
    parse_err = skip = 0
    with open(a.input, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                parse_err += 1
                continue
            if not isinstance(rec, dict):
                parse_err += 1
                continue
            if not rec.get("state") or not rec.get("state_next"):
                skip += 1
                continue
            records.append(rec)
            if a.max and len(records) >= a.max:
                break

    print(f"Seed records: {len(records):,}  (parse_err={parse_err}, no_state_skip={skip})")
    if not records:
        print("ERROR: no usable seed records")
        sys.exit(1)

    all_next = [r.get("state_next", "") for r in records]

    # ── Pass 2: generate examples A/B/C per record ────────────────────────────
    examples = []
    for rec in records:
        op       = rec.get("op", "EXPLAIN")
        preserve = rec.get("preserve") or []
        delta    = rec.get("delta")    or []
        seed_str = (rec.get("state") or "") + (rec.get("state_next") or "")
        wt       = derive_weight_tensor(op, preserve, delta, seed_str)

        examples.append(gen_agent_think(rec, wt))
        examples.append(gen_state_transition(rec, wt))
        examples.append(gen_phase_trace(rec, wt))

    n_base = len(examples)

    # ── Pass 3: add violation examples ────────────────────────────────────────
    n_violations = int(n_base * a.violation_rate)
    for _ in range(n_violations):
        rec   = rng.choice(records)
        decoy = rng.choice(all_next)
        examples.append(gen_violation(rec, decoy, rng))

    # ── Shuffle + write ────────────────────────────────────────────────────────
    rng.shuffle(examples)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for text in examples:
            if not text.strip():
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1

    out_mb = out.stat().st_size / 1e6
    print(f"Written: {written:,} examples  ->  {out}  ({out_mb:.1f} MB)")
    print(f"  A agent_think:  {len(records):,}")
    print(f"  B state_trans:  {len(records):,}")
    print(f"  C phase_trace:  {len(records):,}")
    print(f"  D violation:    {n_violations:,}")
    print()
    print("Next steps:")
    print(f"  python C:/Users/canna/.ASX.cpp/trainer/tokenize_transitions.py \\")
    print(f"    {out.resolve()} flat.bin")
    print(f"  python tools/pack_tokens.py flat.bin packed.bin --seq-len 128")


if __name__ == "__main__":
    main()
