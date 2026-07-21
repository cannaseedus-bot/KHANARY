# Dump a full gpt2 model (real weights) for the on-device DirectML runner, + a CPU reference.
# Embedding is precomputed on CPU; the runner does 12 fused layers + ln_f + lm_head on the GPU.
# Reference logits use ERF gelu (to match DirectML's DML_OPERATOR_ACTIVATION_GELU); we also
# report the driver's TANH-gelu logits so the modeling-choice gap is explicit.
import os, sys, math
import numpy as np
sys.path.insert(0, "tools")
import kxml_inference_driver as K

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = K.KxmlModel(os.path.join(ROOT, "scratch", "gpt2_model.stb"), os.path.join(ROOT, "scratch", "gpt2_model.stb.json"))
cfg = m.cfg
S, L, E, Hn, vocab = 8, cfg["n_layer"], cfg["n_embd"], cfg["n_head"], cfg["vocab"]
H = 4 * E
toks = np.random.default_rng(0).integers(0, vocab, size=S).tolist()
W = lambda n: np.asarray(m.W[m.name2id[n]], dtype=np.float32)

D = os.path.join(ROOT, "scratch", "dml")
def dump(n, a): np.ascontiguousarray(a, dtype=np.float32).tofile(os.path.join(D, n))
def tile(b): return np.broadcast_to(b, (S, b.shape[0]))

wte = W("transformer.wte.weight"); wpe = W("transformer.wpe.weight")
dump("mdl_embed.bin", wte[toks] + wpe[:S])                       # [S,E] embedded input

for i in range(L):
    p = f"transformer.h.{i}."
    dump(f"mdl_l{i}_ln1g.bin", W(p+"ln_1.weight")); dump(f"mdl_l{i}_ln1b.bin", W(p+"ln_1.bias"))
    ca = W(p+"attn.c_attn.weight"); cab = W(p+"attn.c_attn.bias")   # [E,3E], [3E]
    dump(f"mdl_l{i}_wq.bin", ca[:, :E]);      dump(f"mdl_l{i}_wk.bin", ca[:, E:2*E]);    dump(f"mdl_l{i}_wv.bin", ca[:, 2*E:])
    dump(f"mdl_l{i}_bq.bin", tile(cab[:E]));  dump(f"mdl_l{i}_bk.bin", tile(cab[E:2*E])); dump(f"mdl_l{i}_bv.bin", tile(cab[2*E:]))
    dump(f"mdl_l{i}_wap.bin", W(p+"attn.c_proj.weight")); dump(f"mdl_l{i}_bap.bin", tile(W(p+"attn.c_proj.bias")))
    dump(f"mdl_l{i}_ln2g.bin", W(p+"ln_2.weight")); dump(f"mdl_l{i}_ln2b.bin", W(p+"ln_2.bias"))
    dump(f"mdl_l{i}_wfc.bin", W(p+"mlp.c_fc.weight")); dump(f"mdl_l{i}_bfc.bin", tile(W(p+"mlp.c_fc.bias")))
    dump(f"mdl_l{i}_wmp.bin", W(p+"mlp.c_proj.weight")); dump(f"mdl_l{i}_bmp.bin", tile(W(p+"mlp.c_proj.bias")))

dump("mdl_lnfg.bin", W("transformer.ln_f.weight")); dump("mdl_lnfb.bin", W("transformer.ln_f.bias"))
dump("mdl_lmhead.bin", wte.T.copy())                            # [E, vocab] (weight-tied)

# reference logits: ERF-gelu forward (matches DirectML). Then TANH (driver default) for the gap.
tanh = K.op_gelu
K.op_gelu = lambda x: (0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))).astype(np.float32)
erf_logits = np.asarray(m.forward(toks), dtype=np.float64)
K.op_gelu = tanh
tanh_logits = np.asarray(m.forward(toks), dtype=np.float64)
dump("mdl_ref.bin", erf_logits)                                # C++ verifies against this

gap = np.abs(erf_logits - tanh_logits).max() / np.abs(tanh_logits).max()
same = int(np.argmax(erf_logits[-1])) == int(np.argmax(tanh_logits[-1]))
open(os.path.join(D, "mdl_dims.txt"), "w").write(f"{S} {E} {Hn} {H} {L} {vocab}\n")
print(f"[prep] S={S} L={L} E={E} Hn={Hn} vocab={vocab}")
print(f"[prep] erf-ref logits dumped. erf-vs-tanh(driver) logits scale-norm {gap:.2e}, next-token argmax {'MATCH' if same else 'DIFFER'} "
      f"(erf={int(np.argmax(erf_logits[-1]))} tanh={int(np.argmax(tanh_logits[-1]))})")
