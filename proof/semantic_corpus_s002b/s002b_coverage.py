# Semantic Proof S#002b — Representative corpus via a designed COVERAGE MATRIX (READ-ONLY).
# Drives the live semantic_kernel_cli (compile_ir) across intentionally varied queries and measures
# whether the semantic vocabulary SATURATES (convergence), rather than chasing a record count.
# Extracts FIELD (predicate/tense/polarity) + EDGE (argument role relations) per query.
#
# Scope honesty: this covers the CLASSIFICATION axis (front-end parse), which is runnable read-only.
# The legality/VIOLATION-taxonomy axis needs the FULL FieldExecutionEngine driven with varied inputs
# -- blocked read-only because verify_asx's query is hardcoded (rebuild = kernel source mutation).
import os, json, subprocess, sys
from collections import Counter, defaultdict

CLI = r"C:\Users\canna\.ASX.cpp\kxml-semantic-kernel\semantic_kernel_cpp\build\Release\semantic_kernel_cli.exe"
S002 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scratch", "s002")
os.makedirs(S002, exist_ok=True)

# intentional coverage matrix (cover the semantic state space, not maximize count)
MATRIX = [
    ("simple",     "what is the capital of france"),
    ("math",       "multiply seven by eight"),
    ("math",       "what is the derivative of x squared"),
    ("code",       "write a python function to sort a list"),
    ("code",       "refactor this loop into a comprehension"),
    ("search",     "search for flights to phoenix on friday"),
    ("search",     "find the latest news about mars"),
    ("tool",       "read the file config.txt and tell me the port"),
    ("missing_dep","summarize the document i did not upload"),
    ("unsupported","generate an image of a sunset"),
    ("multistep",  "book a flight then add it to my calendar"),
    ("ambiguous",  "run it"),
    ("ambiguous",  "the thing from before"),
    ("greeting",   "hello there"),
]

recs=[]; vocab={"predicate":set(),"tense":set(),"polarity":set(),"role":set(),"entity":set()}
conv=[]   # vocabulary-size convergence curve as queries are added
for i,(cat,q) in enumerate(MATRIX):
    out=os.path.join(S002,f"ir_{i:02d}.json")
    subprocess.run([CLI,"compile_ir",q,out], cwd=os.path.dirname(CLI), capture_output=True, text=True)
    if not os.path.exists(out): continue
    d=json.load(open(out))
    pred=d.get("predicate"); tense=d.get("tense"); pol=d.get("polarity"); args=d.get("arguments",[])
    vocab["predicate"].add(pred); vocab["tense"].add(tense); vocab["polarity"].add(pol)
    # FIELD: the classified semantic field for this query
    recs.append({"id":f"f{i:02d}","class":"FIELD","identity":f"q{i:02d}","category":cat,
                 "have":{"predicate":pred,"tense":tense,"polarity":pol},"mutation_class":"TRANSIENT",
                 "source":"cli:compile_ir","query":q})
    # EDGE: each argument is a role relation from the predicate to an entity
    for a in args:
        role=a.get("role"); ent=a.get("entity"); vocab["role"].add(role); vocab["entity"].add(ent)
        recs.append({"id":f"e{i:02d}_{role}","class":"EDGE","from":pred,"to":ent,"relation":role,
                     "category":cat,"source":"cli:compile_ir"})
    conv.append({"queries":i+1, **{k:len(v) for k,v in vocab.items()}})

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"semantic-classification-corpus.jsonl")
open(OUT,"w",encoding="utf-8").write("\n".join(json.dumps(r) for r in recs)+"\n")

byc=Counter(r["class"] for r in recs)
bycat=Counter(r["category"] for r in recs if r["class"]=="FIELD")
roles=Counter(r["relation"] for r in recs if r["class"]=="EDGE")
print(f"[S#002b] coverage matrix: {len(MATRIX)} intentional queries across {len(set(c for c,_ in MATRIX))} categories -> {len(recs)} records")
print(f"  FIELD {byc.get('FIELD',0)}  EDGE {byc.get('EDGE',0)}")
print(f"[vocab] predicates={len(vocab['predicate'])} roles={len(vocab['role'])} entities={len(vocab['entity'])} tenses={len(vocab['tense'])} polarities={len(vocab['polarity'])}")
print(f"[dist]  categories: {dict(bycat)}")
print(f"[dist]  edge relations: {dict(roles)}")
print("[convergence] role-vocab size as queries added:", " ".join(str(c["role"]) for c in conv))
# convergence: did role vocab stop growing over the last third of the matrix?
tail=[c["role"] for c in conv[-max(3,len(conv)//3):]]
role_stable = len(set(tail))==1
print(f"[convergence] role vocabulary over last {len(tail)} queries: {tail} -> {'STABLE' if role_stable else 'still growing'}")
allpop = byc.get("FIELD",0)>0 and byc.get("EDGE",0)>0
print(f"=== {'PASS' if allpop else 'FAIL'}: S#002b classification-axis coverage (FIELD+EDGE from a designed matrix; convergence measured) ===")
print("[SCOPE] legality/VIOLATION-taxonomy coverage NOT included -- requires driving the full engine")
print("        with varied inputs (verify_asx query is hardcoded); that is blocked read-only.")
sys.exit(0 if allpop else 1)
