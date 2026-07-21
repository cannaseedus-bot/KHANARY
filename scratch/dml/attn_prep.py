# Inputs + numpy reference for the fused attention block:
#   ln = LayerNorm(x); Q=ln@Wq+bq; K=ln@Wk+bk; V=ln@Wv+bv;
#   attn = causal_MHA(Q,K,V);  proj = attn@Wproj+bproj;  out = proj + x
import os
import numpy as np

S, E, Hn = 8, 768, 12
Hd = E // Hn
rng = np.random.default_rng(11)
x   = (rng.standard_normal((S, E)) * 0.5).astype(np.float32)
lng = (1 + 0.02 * rng.standard_normal(E)).astype(np.float32)
lnb = (0.01 * rng.standard_normal(E)).astype(np.float32)
def W(): return (rng.standard_normal((E, E)) / np.sqrt(E)).astype(np.float32)
def b(): return (0.01 * rng.standard_normal(E)).astype(np.float32)
Wq,Wk,Wv,Wp = W(),W(),W(),W()
bq,bk,bv,bp = b(),b(),b(),b()

def layernorm(v,g,bb,eps=1e-5):
    m=v.mean(-1,keepdims=True); return (v-m)/np.sqrt(v.var(-1,keepdims=True)+eps)*g+bb

def causal_mha(Q,K,V):
    out=np.zeros_like(Q); scale=1.0/np.sqrt(Hd); mask=np.triu(np.ones((S,S)),1).astype(bool)
    for h in range(Hn):
        sl=slice(h*Hd,(h+1)*Hd)
        sc=(Q[:,sl]@K[:,sl].T)*scale
        sc[mask]=-1e9; sc=sc-sc.max(-1,keepdims=True)
        e=np.exp(sc); p=e/e.sum(-1,keepdims=True)
        out[:,sl]=p@V[:,sl]
    return out

ln=layernorm(x,lng,lnb)
Q=ln@Wq+bq; K=ln@Wk+bk; V=ln@Wv+bv
attn=causal_mha(Q,K,V)
proj=attn@Wp+bp
out=proj+x

D=os.path.dirname(os.path.abspath(__file__))
def dump(n,a): np.ascontiguousarray(a,dtype=np.float32).tofile(os.path.join(D,n))
dump("attn_x.bin",x); dump("attn_lng.bin",lng); dump("attn_lnb.bin",lnb)
dump("attn_wq.bin",Wq); dump("attn_wk.bin",Wk); dump("attn_wv.bin",Wv); dump("attn_wp.bin",Wp)
dump("attn_bq.bin",np.broadcast_to(bq,(S,E))); dump("attn_bk.bin",np.broadcast_to(bk,(S,E)))
dump("attn_bv.bin",np.broadcast_to(bv,(S,E))); dump("attn_bp.bin",np.broadcast_to(bp,(S,E)))
dump("attn_ref.bin",out)
open(os.path.join(D,"attn_dims.txt"),"w").write(f"{S} {E} {Hn}\n")
print(f"[prep] S={S} E={E} Hn={Hn} Hd={Hd}  dumped. out[0,:3]={out[0,:3]}")
