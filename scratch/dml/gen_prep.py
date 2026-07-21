# Reference for Proof #003 (Resident Generation): a numpy erf-gelu GPT-2 that (a) autoregressively
# generates N tokens from an S-token prompt (deterministic argmax), and (b) captures the per-layer
# KV trajectory for the full sequence, so the on-device generation can be verified at every tick
# AND its final KV cache checked per layer. Weights dumped with a gen_ prefix (biases as raw [N]
# vectors, since every decode step is a single token).
import os, sys, math
import numpy as np
sys.path.insert(0, "tools")
import kxml_inference_driver as K

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = K.KxmlModel(os.path.join(ROOT,"scratch","gpt2_model.stb"), os.path.join(ROOT,"scratch","gpt2_model.stb.json"))
cfg = m.cfg
S, N, E, Hn, V = 8, 6, cfg["n_embd"], cfg["n_head"], cfg["vocab"]
Hd, H, Ln = E//Hn, 4*cfg["n_embd"], cfg["n_layer"]
W = lambda n: np.asarray(m.W[m.name2id[n]], dtype=np.float32)
wte = W("transformer.wte.weight"); wpe = W("transformer.wpe.weight")

# per-layer weights (split QKV from fused c_attn)
lw = []
for i in range(Ln):
    p=f"transformer.h.{i}."; ca=W(p+"attn.c_attn.weight"); cab=W(p+"attn.c_attn.bias")
    lw.append(dict(ln1g=W(p+"ln_1.weight"),ln1b=W(p+"ln_1.bias"),
        wq=ca[:,:E],wk=ca[:,E:2*E],wv=ca[:,2*E:], bq=cab[:E],bk=cab[E:2*E],bv=cab[2*E:],
        wap=W(p+"attn.c_proj.weight"),bap=W(p+"attn.c_proj.bias"),
        ln2g=W(p+"ln_2.weight"),ln2b=W(p+"ln_2.bias"),
        wfc=W(p+"mlp.c_fc.weight"),bfc=W(p+"mlp.c_fc.bias"),
        wmp=W(p+"mlp.c_proj.weight"),bmp=W(p+"mlp.c_proj.bias")))
lnfg=W("transformer.ln_f.weight"); lnfb=W("transformer.ln_f.bias")

def LN(x,g,b,e=1e-5): mu=x.mean(-1,keepdims=True); return (x-mu)/np.sqrt(x.var(-1,keepdims=True)+e)*g+b
def gelu(x): return 0.5*x*(1.0+np.vectorize(math.erf)(x/math.sqrt(2.0)))
def mha(q,k,v):
    n=q.shape[0]; o=np.zeros_like(q); sc=1/np.sqrt(Hd); mask=np.triu(np.ones((n,n)),1).astype(bool)
    for h in range(Hn):
        sl=slice(h*Hd,(h+1)*Hd); s=(q[:,sl]@k[:,sl].T)*sc; s[mask]=-1e9; s=s-s.max(-1,keepdims=True)
        e=np.exp(s); o[:,sl]=(e/e.sum(-1,keepdims=True))@v[:,sl]
    return o

def forward(seq, cap=False):
    h = wte[seq] + wpe[:len(seq)]; KV=[]
    for L in lw:
        ln1=LN(h,L["ln1g"],L["ln1b"]); q=ln1@L["wq"]+L["bq"]; k=ln1@L["wk"]+L["bk"]; v=ln1@L["wv"]+L["bv"]
        if cap: KV.append((k.copy(),v.copy()))
        h = h + (mha(q,k,v)@L["wap"]+L["bap"])
        ln2=LN(h,L["ln2g"],L["ln2b"]); h = h + (gelu(ln2@L["wfc"]+L["bfc"])@L["wmp"]+L["bmp"])
    return (LN(h,lnfg,lnfb)@wte.T), KV

# autoregressive generation (deterministic argmax) + per-tick prediction trace
prompt = np.random.default_rng(0).integers(0,V,size=S).tolist()
seq = list(prompt); cpu_pred=[]
for t in range(S+N):
    logits,_ = forward(seq[:t+1])
    p = int(np.argmax(logits[-1])); cpu_pred.append(p)
    if t >= S-1 and t < S+N-1: seq.append(p)
T = len(seq)   # == S+N

# reference KV for the FULL final sequence, per layer, laid out [Hn, T, Hd] (DML present layout)
_, KV = forward(seq, cap=True)

D = os.path.join(ROOT,"scratch","dml")
def dump(n,a): np.ascontiguousarray(a,dtype=np.float32).tofile(os.path.join(D,n))
dump("gen_wte.bin", wte); dump("gen_wpe.bin", wpe[:T]); dump("gen_lnfg.bin",lnfg); dump("gen_lnfb.bin",lnfb); dump("gen_lmhead.bin", wte.T.copy())
for i,L in enumerate(lw):
    for k_,v_ in L.items(): dump(f"gen_l{i}_{k_}.bin", v_)
    k,v = KV[i]                                   # [T,E] -> [Hn,T,Hd]
    kr = k.reshape(T,Hn,Hd).transpose(1,0,2); vr = v.reshape(T,Hn,Hd).transpose(1,0,2)
    dump(f"gen_l{i}_kref.bin", kr); dump(f"gen_l{i}_vref.bin", vr)
open(os.path.join(D,"gen_seq.txt"),"w").write(
    f"{T} {S} {N} {E} {Hn} {V}\n" + " ".join(map(str,seq)) + "\n" + " ".join(map(str,cpu_pred)) + "\n")
print(f"[prep] S={S} N={N} T={T} E={E} Hn={Hn}")
print(f"[prep] prompt={prompt}")
print(f"[prep] generated (cpu, erf-gelu) = {seq[S:]}")
print(f"[prep] per-tick cpu_pred = {cpu_pred}")
