import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import kxml_inference_driver as K
m = K.KxmlModel(os.path.join(ROOT,"scratch","gpt2_model.stb"), os.path.join(ROOT,"scratch","gpt2_model.stb.json"))
toks = np.random.default_rng(0).integers(0, m.cfg["vocab"], size=8).tolist()
K._USE_DML = True
m.forward(toks)                       # warm (compile ops per shape)
t0=time.perf_counter()
for _ in range(3): m.forward(toks)
dt=(time.perf_counter()-t0)/3
print(f"[time] DML forward (8 tok, 12L): {dt*1000:.1f} ms/pass")
