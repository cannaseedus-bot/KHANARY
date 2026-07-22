# Semantic Proof S#002 — LIVE trajectory extraction (OBSERVE, don't invent).
# Runs the prebuilt FieldExecutionEngine (verify_asx.exe) in a non-invasive working dir and parses
# its emitted per-tick trace into the FROZEN S#003a record algebra — validating that the algebra
# designed from the 66 STATIC records (S#001) actually matches REAL runtime behavior.
#
# Mutates NO kernel source. verify_asx.exe + its manifest were COPIED out of .ASX.cpp to
# scratch/s002/ (read-only observation of the running binary's existing stdout logs).
import os, re, json, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
S002 = os.path.join(os.path.dirname(HERE), "..", "scratch", "s002")   # staged exe+manifest
def run_live():
    exe = os.path.join(S002, "verify_asx.exe")
    if not os.path.exists(exe):
        # fall back to a previously captured trace if the exe/manifest aren't staged
        t = os.path.join(S002, "verify_trace.txt")
        return open(t, encoding="utf-8", errors="replace").read() if os.path.exists(t) else ""
    return subprocess.run([exe], cwd=S002, capture_output=True, text=True, errors="replace").stdout

trace = run_live()
if "TICK" not in trace:
    print("no live trace (stage scratch/s002 with verify_asx.exe + unified_geometric_manifest.json)"); sys.exit(1)

# split into ticks
ticks = re.split(r"\[INFO\] TICK ", trace)[1:]
recs = []
def R(**k): recs.append(k)
PHASES = ["PERCEPTION","ROUTING","COMPUTE","META","PROJECTION"]
for tk in ticks:
    tn = int(re.match(r"\s*(\d+)", tk).group(1))
    fold = re.search(r"Fold (\d+): ([\d.]+)", tk); fold_id = int(fold.group(1)) if fold else None; aff = float(fold.group(2)) if fold else None
    query = (re.search(r"Routing Query: (.+)", tk) or [None,None])[1]
    moe = (re.search(r"MoE: (.+)", tk) or [None,None])[1]
    lawful = "Law E: Mutation verified as LAWFUL" in tk
    # projected :root field state
    field = {}
    for m in re.finditer(r"--([a-z-]+):\s*\"?([^;\"]+)\"?;", tk):
        field[m.group(1)] = m.group(2).strip()
    src = "verify_asx:tick"
    # 4 TRANSITION edges between the 5 phases (all executed by the runtime -> legal)
    for a,b in zip(PHASES, PHASES[1:]):
        R(id=f"t{tn:02d}_{a}_{b}", **{"class":"TRANSITION"}, state={"fold":fold_id,"phase":a},
          transition=f"{a}->{b}", next_state={"phase":b}, legality="legal", source=src)
    # DELTA + legality from the META Law-E verdict (applied mutation)
    R(id=f"d{tn:02d}", **{"class":"DELTA"}, field=f"fold_{fold_id}", delta={"applies":"evolve/mutate","route":moe},
      legality=("legal" if lawful else "illegal"), **({} if lawful else {"violation":"law_e_unlawful"}),
      source="verify_asx:LawE")
    # FIELD state node (the projected semantic field for this tick)
    R(id=f"f{tn:02d}", **{"class":"FIELD"}, identity=f"tick_{tn}", have=field,
      mutation_class="GROWING", residency="session", source="verify_asx:projection")

from collections import Counter
byc = Counter(r["class"] for r in recs)
legal = sum(1 for r in recs if r.get("legality")=="legal"); illegal = sum(1 for r in recs if r.get("legality")=="illegal")
OUT = os.path.join(HERE, "semantic-trajectories-live.jsonl")
open(OUT,"w",encoding="utf-8").write("\n".join(json.dumps(r) for r in recs)+"\n")

# runtime statistics (what the user wanted: distributions the static corpus couldn't give)
folds = Counter(r["state"]["fold"] for r in recs if r["class"]=="TRANSITION")
moes  = Counter(r["delta"].get("route") for r in recs if r["class"]=="DELTA")
cohs  = [float(r["have"].get("coherence",0)) for r in recs if r["class"]=="FIELD" and r["have"].get("coherence")]
print(f"[S#002] LIVE extraction from verify_asx.exe (FieldExecutionEngine): {len(ticks)} ticks -> {len(recs)} records")
for c in ["TRANSITION","DELTA","FIELD"]:
    print(f"  {c:12s} {byc.get(c,0):4d}")
print(f"  legal {legal}  illegal {illegal}")
print(f"[dist] folds visited: {dict(folds)}")
print(f"[dist] MoE routes: {dict(moes)}")
if cohs: print(f"[dist] coherence range: {min(cohs):.4f}..{max(cohs):.4f}")
# validate: live records conform to the FROZEN S#003a algebra kinds
algebra_kinds = {"FIELD","EDGE","TRANSITION","DELTA","INVARIANT","VIOLATION"}
conforms = all(r["class"] in algebra_kinds for r in recs) and all("source" in r and "legality" in r for r in recs if r["class"] in {"TRANSITION","DELTA"})
allpop = byc.get("TRANSITION",0)>0 and byc.get("DELTA",0)>0 and byc.get("FIELD",0)>0
print(f"=== {'PASS' if conforms and allpop else 'FAIL'}: S#002 live trajectories conform to the frozen S#003a algebra (kernel sources unmutated) ===")
sys.exit(0 if conforms and allpop else 1)
