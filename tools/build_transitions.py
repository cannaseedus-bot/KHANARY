#!/usr/bin/env python3
# build_transitions.py -- slice #1 of the semantic base: turn flat {query,response} pairs into
# SEMANTIC-PATTERN (transition-shaped) training examples that teach the transition FUNCTION,
# not just recall.  Each example is a  F(t+1) = Preserve (+) Delta  step (the stack's semantic
# tensor target) -- the discourse-tier analogue of the glyph engine's DET+NOUN->NP composition.
#
#   STATE  = the query's semantic field
#   OP     = the semantic operation (ELIZA/RegEx decompose of intent) -- a WORD label, never a
#            hex opcode, so it can't shadow native_glyph_engine's opcode space
#   KEEP   = content carried Q->R (Preserve)
#   NEW    = content introduced in R (Delta)
#   STATE' = the response
#
# Usage: python tools/build_transitions.py <in_clean.jsonl> <out_transitions.jsonl>

import sys, os, re, json, collections

STOP = set("the a an and or but if then of to in on at for with as is are was were be been being "
           "this that these those it its they them their you your i me my we our he she his her "
           "do does did done have has had will would can could should may might must not no yes "
           "so what which who whom whose why how when where can could please give me some few "
           "about into from by via out up down over under can't cannot don't".split())

WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]{1,}")
# these mirror the corpus cleaner -- the transition file must stay opcode/image/base64-free
HEX_RUN = re.compile(r'(?:(?:0[xX])?[0-9A-Fa-f]{2}[\s,]+){7,}(?:0[xX])?[0-9A-Fa-f]{2}')
IMG     = re.compile(r'data:image/[a-z0-9.+-]+[;,][^\s"\')]{40,}')

# ELIZA-style decompose, SCORED over the whole query+response (not just the query prefix), so the
# op distribution isn't swamped by a generic fallback. Ops are chosen to map onto the code/instruct/
# math experts (the router signal for slice #2).
OP_PAT = {
    "DEFINE":    re.compile(r"\bwhat(?:'?s| is| are| was| were)\b|\bdefine\b|\bmeaning of\b|\bwho (is|are|was)\b", re.I),
    "EXPLAIN":   re.compile(r"\bwhy\b|\bexplain\b|\bhow (?:does|do|did) .{0,40}\bwork\b|\breasons?\b|\bpurpose of\b", re.I),
    "COMPARE":   re.compile(r"\b(vs\.?|versus|difference(s)? between|compared? to|pros and cons|better than|whereas|on the other hand|trade[- ]?offs?)\b", re.I),
    "ENUMERATE": re.compile(r"\b(list|enumerate|name (?:some|a few|the)|what are the .{0,30}(types|kinds|ways|examples|benefits|steps))\b", re.I),
    "INSTRUCT":  re.compile(r"\bhow (?:to|do i|can i|would i|should i|do you)\b|\bstep[- ]by[- ]step\b|\bsteps? (?:to|for)\b|\bguide\b|\btutorial\b|\binstructions?\b", re.I),
    "TRANSFORM": re.compile(r"\b(convert|translate|rewrite|refactor|summar(?:ize|ise)|fix|debug|improve|optimi(?:ze|se)|correct|revise|edit)\b", re.I),
    "GENERATE":  re.compile(r"\b(write|create|generate|make|build|design|implement|draft|compose|develop|produce|code(?: me| a)?)\b", re.I),
    "COMPUTE":   re.compile(r"\b(calculate|compute|solve|evaluate|how (?:much|many))\b|\bwhat is\s+[-\d(]", re.I),
    "LOCATE":    re.compile(r"^\s*(when|where)\b", re.I),
}
DECLINE_PAT = re.compile(r"^\s*(i\s+(?:can'?t|cannot|won'?t|am unable)|i'?m sorry|as an ai|i (?:do not|don'?t) (?:have|provide|support))", re.I)
LIST_STRUCT = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*+])\s+\S")
STEP_STRUCT = re.compile(r"(?mi)^\s*(?:step\s*\d|first[,:]|then[,:]|next[,:]|finally[,:])")
CODE_SIG = re.compile(r"```|\bdef \b|\bfunction\b|\bimport \b|\bclass \b|#include|=>|\bconsole\.log\b|\bprintf\b|\b(?:SELECT|INSERT|UPDATE)\b|</?[a-z][a-z0-9]*>|\bnpm \b|\bpip install\b", re.I)
CODE_KW  = re.compile(r"\b(code|function|bug|compile|debug|python|javascript|java|c\+\+|rust|golang|sql|html|css|api|regex|algorithm|script|syntax|program|variable|array|loop)\b", re.I)
MATH_SIG = re.compile(r"\$[^$\n]{2,}\$|\\(?:frac|sum|int|sqrt|prod)|\b\d+\s*[-+*/^=]\s*\d|\b\d+\s*(?:x|y|n)\b", re.I)
MATH_KW  = re.compile(r"\b(calculate|solve|equation|theorem|prove|proof|integral|derivative|matrix|probability|algebra|geometry|arithmetic|factorial|polynomial|multiply|divide|sum of)\b", re.I)

def content(t):
    return [w.lower() for w in WORD.findall(t) if w.lower() not in STOP and len(w) > 2]

def lane_of(q, r):
    if CODE_SIG.search(q + "\n" + r) or CODE_KW.search(q): return "code"
    if MATH_SIG.search(q + "\n" + r) or MATH_KW.search(q): return "math"
    return "general"

def classify(q, r):
    if DECLINE_PAT.search(r):
        return "DECLINE"
    rr = r[:600]
    scores = {op: len(p.findall(q)) + 0.5 * len(p.findall(rr)) for op, p in OP_PAT.items()}
    if LIST_STRUCT.search(rr): scores["ENUMERATE"] += 1.0    # response structure as evidence
    if STEP_STRUCT.search(rr): scores["INSTRUCT"]  += 1.0
    if "```" in rr:            scores["GENERATE"]  += 1.0
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "RESPOND"

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    ops = collections.Counter()
    lanes = collections.Counter()
    n = kp = dl = 0
    bad = 0
    with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as o:
        for line in f:
            try: rec = json.loads(line)
            except Exception: continue
            q, r = rec.get("query", ""), rec.get("response", "")
            if not q or not r: continue
            qc, rc = content(q), content(r)
            if not qc or not rc: continue
            qset = set(qc)
            preserve = [w for w in dict.fromkeys(rc) if w in qset]          # carried Q->R (ordered, unique)
            delta    = [w for w in dict.fromkeys(rc) if w not in qset][:24] # newly introduced (cap noise)
            op = classify(q, r)
            lane = lane_of(q, r)      # code / math / general -> which expert handles this transition
            # the transition-shaped training text (bracket markers, no hex -> opcode-safe)
            text = (f"[STATE] {q.strip()}\n[LANE] {lane}\n[OP] {op}\n"
                    f"[KEEP] {' '.join(preserve[:24])}\n[NEW] {' '.join(delta)}\n"
                    f"[STATE'] {r.strip()}")
            if HEX_RUN.search(text) or IMG.search(text):   # guard: never emit opcode/image contamination
                bad += 1; continue
            o.write(json.dumps({
                "state": q.strip(), "lane": lane, "op": op,
                "preserve": preserve[:24], "delta": delta,
                "state_next": r.strip(), "text": text,
            }, ensure_ascii=False) + "\n")
            ops[op] += 1; lanes[lane] += 1; n += 1; kp += len(preserve); dl += len(delta)
    print(f"[ok] {outp}: {n} transitions, {os.path.getsize(outp)/1e6:.1f} MB  (skipped contaminated: {bad})")
    print(f"     avg Preserve={kp/max(n,1):.1f}  avg Delta={dl/max(n,1):.1f} content words/example")
    print("     op coverage:")
    for op, c in ops.most_common():
        print(f"        {op:10s} {c:7d}  ({100*c/max(n,1):4.1f}%)")
    print("     lane coverage (expert routing):")
    for lane, c in lanes.most_common():
        print(f"        {lane:10s} {c:7d}  ({100*c/max(n,1):4.1f}%)")

if __name__ == "__main__":
    main()
