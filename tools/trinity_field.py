#!/usr/bin/env python3
# trinity_field.py -- slice A of the "better brain": fit Trinity's weighted SEMANTIC FIELD to the
# observed structure of the transition dataset, and expose the @flux INLINE-UPDATE rule.
#
# This is NOT more ELIZA rules. It's the learnable field the hybrid was missing: a prior over
# meaning-transitions, bootstrapped from data, that experience (execution/@flux evidence) tunes
# inline (A->B: .70 -> .73 -> .77). ELIZA/RegEx/Roslyn stay on explicit recognition; THIS carries
# the weighted geometry. Later: port these weights into quantum_trinity_hybrid.cpp; promote stable
# ones to SCXQ2 persistent state.
#
# Fits, from each {lane, op, preserve[], delta[]} transition (F(t+1)=Preserve(+)Delta):
#   op_given_lane[lane][op]      P(operation | lane)            -- routing structure
#   delta_given_op[op][tok]      P(new-concept | operation)     -- what each op introduces
#   transition[p_tok][d_tok]     Preserve-concept -> Delta-concept weight  -- the semantic "bigram"
#   pd_stats[op]                 mean Preserve / Delta sizes     -- the transition geometry
#
# Usage: python tools/trinity_field.py fit  <transitions.jsonl> <field.json> [--limit N] [--vocab 2000] [--topk 20]
#        python tools/trinity_field.py demo <field.json>

import sys, json, argparse, collections, math

class TrinityField:
    def __init__(self):
        self.op_given_lane = {}     # lane -> {op: prob}
        self.delta_given_op = {}    # op -> {token: prob}   (top-k)
        self.transition = {}        # preserve_tok -> {delta_tok: weight}  (top-k, the semantic bigram)
        self.pd_stats = {}          # op -> {"preserve": mean, "delta": mean}
        self.meta = {}

    # ---- fit the PRIOR from observed structure ------------------------------
    def fit(self, jsonl, limit=0, vocab_n=2000, topk=20):
        freq = collections.Counter()
        rows = []
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                try: r = json.loads(line)
                except Exception: continue
                p, d = r.get("preserve", []), r.get("delta", [])
                freq.update(p); freq.update(d)
                rows.append((r.get("lane", "general"), r.get("op", "RESPOND"), p, d))
                if limit and len(rows) >= limit: break
        vocab = set(t for t, _ in freq.most_common(vocab_n))

        lane_op = collections.defaultdict(collections.Counter)
        op_delta = collections.defaultdict(collections.Counter)
        cooc     = collections.defaultdict(collections.Counter)
        pd_sum   = collections.defaultdict(lambda: [0, 0, 0])   # [preserve_sum, delta_sum, n]
        for lane, op, p, d in rows:
            lane_op[lane][op] += 1
            pv = [t for t in p if t in vocab]
            dv = [t for t in d if t in vocab]
            for t in dv: op_delta[op][t] += 1
            for pt in pv:
                for dt in dv:
                    cooc[pt][dt] += 1
            s = pd_sum[op]; s[0] += len(p); s[1] += len(d); s[2] += 1

        norm = lambda c: {k: v / sum(c.values()) for k, v in c.items()} if c else {}
        top  = lambda c, k: dict(collections.Counter(c).most_common(k))
        self.op_given_lane = {l: norm(c) for l, c in lane_op.items()}
        self.delta_given_op = {o: norm(top(c, topk)) for o, c in op_delta.items()}
        self.transition = {p: {d: w / sum(cd.values()) for d, w in top(cd, topk).items()}
                           for p, cd in cooc.items()}
        self.pd_stats = {o: {"preserve": s[0]/max(s[2],1), "delta": s[1]/max(s[2],1)}
                         for o, s in pd_sum.items()}
        self.meta = {"rows": len(rows), "vocab": len(vocab), "topk": topk,
                     "transition_nodes": len(self.transition)}
        return self

    # ---- @flux INLINE update: evidence tunes the field ----------------------
    def flux(self, p_tok, d_tok, reward, lr=0.05):
        """Adjust a Preserve->Delta transition from execution evidence.
        reward>0 (it happened / succeeded) pushes the weight up toward 1; reward<0 pushes toward 0."""
        row = self.transition.setdefault(p_tok, {})
        w = row.get(d_tok, 0.0)
        w = w + lr * reward * (1.0 - w) if reward >= 0 else w + lr * reward * w
        row[d_tok] = max(0.0, min(1.0, w))
        return row[d_tok]

    def guidance(self, state_tokens, k=10):
        """Given current Preserve-state concepts, rank the likely Delta concepts (the field's steer)."""
        agg = collections.Counter()
        for t in state_tokens:
            for d, w in self.transition.get(t, {}).items(): agg[d] += w
        return agg.most_common(k)

    def save(self, path):
        json.dump({"meta": self.meta, "op_given_lane": self.op_given_lane,
                   "delta_given_op": self.delta_given_op, "transition": self.transition,
                   "pd_stats": self.pd_stats}, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    def load(self, path):
        d = json.load(open(path, encoding="utf-8"))
        for k in ("meta", "op_given_lane", "delta_given_op", "transition", "pd_stats"):
            setattr(self, k, d.get(k, {}))
        return self

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fit"); pf.add_argument("jsonl"); pf.add_argument("out")
    pf.add_argument("--limit", type=int, default=0); pf.add_argument("--vocab", type=int, default=2000)
    pf.add_argument("--topk", type=int, default=20)
    pd = sub.add_parser("demo"); pd.add_argument("field")
    pa = sub.add_parser("adapt"); pa.add_argument("field_in"); pa.add_argument("evidence"); pa.add_argument("field_out")
    pa.add_argument("--lr", type=float, default=0.05); pa.add_argument("--limit", type=int, default=0)
    pa.add_argument("--pmi", action="store_true",
                    help="C2 value-proxy: reward=sign(PMI(A,B)) instead of +1 (same iteration/budget)")
    pa.add_argument("--graded", action="store_true",
                    help="C2b: reward=sign(PMI)*tanh(|PMI|/scale) -- direction + bounded strength")
    pa.add_argument("--pmi-scale", type=float, default=2.0, help="C2b bound (declared before run, fixed)")
    a = ap.parse_args()

    if a.cmd == "adapt":
        # Model stays FROZEN; only the field learns via @flux over the SAME (preserve x delta) pairs.
        # C1: reward=+1 for every observation (frequency). C2: reward=sign(PMI) (value-discriminative).
        fld = TrinityField().load(a.field_in)
        rows = []
        with open(a.evidence, encoding="utf-8") as f:
            for line in f:
                try: r = json.loads(line)
                except Exception: continue
                rows.append((r.get("preserve", [])[:12], r.get("delta", [])[:12]))
                if a.limit and len(rows) >= a.limit: break

        pmi = None
        if a.pmi:
            # pass 1: marginals + co-occurrence over the SAME pair population C1 visits
            mA, mB, cooc, T = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter), 0
            for p, d in rows:
                for pt in p:
                    for dt in d:
                        mA[pt] += 1; mB[dt] += 1; cooc[pt][dt] += 1; T += 1
            pmi = {}
            for pt, row in cooc.items():
                pmi[pt] = {dt: math.log((c * T) / (mA[pt] * mB[dt])) for dt, c in row.items()}  # log P(A,B)/P(A)P(B)
            json.dump(pmi, open(a.field_out + ".pmi.json", "w", encoding="utf-8"))  # raw magnitude for C2b

        n = upd = pos = neg = 0
        for p, d in rows:
            for pt in p:
                for dt in d:
                    if pmi is not None:
                        m = pmi[pt][dt]
                        if a.graded:                        # C2b: direction + bounded strength
                            reward = math.copysign(math.tanh(abs(m) / a.pmi_scale), m) if m != 0 else 0.0
                        else:                               # C2: direction only
                            reward = 1.0 if m > 0 else (-1.0 if m < 0 else 0.0)
                        pos += reward > 0; neg += reward < 0
                    else:
                        reward = 1.0; pos += 1
                    fld.flux(pt, dt, reward=reward, lr=a.lr); upd += 1
            n += 1
        fld.save(a.field_out)
        tag = (f"sign(PMI)*tanh(|PMI|/{a.pmi_scale})" if a.graded else "sign(PMI)") if a.pmi else "+1"
        print(f"[ok] @flux-adapted field ({tag}): {n} rows, {upd} updates (reinforce={pos} weaken={neg}) -> {a.field_out}"
              + (f"  [raw PMI -> {a.field_out}.pmi.json]" if a.pmi else ""))
        return

    if a.cmd == "fit":
        fld = TrinityField().fit(a.jsonl, a.limit, a.vocab, a.topk)
        fld.save(a.out)
        print(f"[ok] fit field -> {a.out}  {fld.meta}")
        print("  P(op|lane):")
        for lane, ops in fld.op_given_lane.items():
            top = sorted(ops.items(), key=lambda x: -x[1])[:4]
            print(f"    {lane:8s}: " + ", ".join(f"{o}={p:.2f}" for o, p in top))
        print("  Preserve/Delta geometry (mean tokens):")
        for op, s in sorted(fld.pd_stats.items(), key=lambda x: -x[1]['delta'])[:6]:
            print(f"    {op:10s} preserve={s['preserve']:.1f} delta={s['delta']:.1f}")
    else:
        fld = TrinityField().load(a.field)
        print(f"[field] {fld.meta}")
        # pick a real Preserve->Delta transition and show @flux inline learning in action
        p0 = next(iter(fld.transition)); d0 = max(fld.transition[p0], key=fld.transition[p0].get)
        print(f"\n@flux inline-learning demo on transition  '{p0}' -> '{d0}':")
        print(f"  Trinity prior: {fld.transition[p0][d0]:.3f}")
        for i in range(4):
            w = fld.flux(p0, d0, reward=+1.0)
            print(f"  +evidence (succeeded x{i+1}): {w:.3f}")
        for i in range(2):
            w = fld.flux(p0, d0, reward=-1.0)
            print(f"  -evidence (failed x{i+1}):    {w:.3f}")
        print(f"\n  guidance('{p0}') -> likely next concepts: " +
              ", ".join(f"{t}({w:.2f})" for t, w in fld.guidance([p0], 6)))

if __name__ == "__main__":
    main()
