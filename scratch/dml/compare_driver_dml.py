# Prove the DirectML matmul path inside the KHANARY inference driver: run the SAME forward pass
# with the numpy G_MATMUL and with the DirectML DLL path, compare logits + the greedy argmax.
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import kxml_inference_driver as K

stb = os.path.join(ROOT, "scratch", "gpt2_model.stb")
man = os.path.join(ROOT, "scratch", "gpt2_model.stb.json")
m = K.KxmlModel(stb, man)
print(f"[model] {m.cfg['n_layer']}L n_embd={m.cfg['n_embd']} vocab={m.cfg['vocab']} | {len(m.graph)} nodes")

rng = np.random.default_rng(0)
toks = rng.integers(0, m.cfg["vocab"], size=8).tolist()

K._USE_DML = False
ref = np.asarray(m.forward(toks), dtype=np.float64)
K._USE_DML = True
dml = np.asarray(m.forward(toks), dtype=np.float64)

scale = np.abs(ref).max()
nrm = np.abs(dml - ref).max() / scale
arg_ref = int(np.argmax(ref[-1]))
arg_dml = int(np.argmax(dml[-1]))
print(f"[compare] logits{list(ref.shape)}  max abs {np.abs(dml-ref).max():.3e}  scale-norm {nrm:.2e}")
print(f"[compare] next-token argmax  numpy={arg_ref}  dml={arg_dml}  {'MATCH' if arg_ref==arg_dml else 'DIFFER'}")
ok = nrm < 1e-3 and arg_ref == arg_dml
print(f"=== {'PASS' if ok else 'FAIL'}: DirectML matmul path threads the full 12-layer driver, matches numpy ===")
sys.exit(0 if ok else 1)
