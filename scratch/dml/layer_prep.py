# Inputs + numpy reference for a full gpt2 transformer LAYER (attention block + MLP block):
#   ln1=LN(x); Q,K,V=ln1@Wq/k/v+b; attn=causalMHA(Q,K,V); ap=attn@Wap+bap; x1=ap+x
#   ln2=LN(x1); fc=ln2@Wfc+bfc; g=gelu_erf(fc); mp=g@Wmp+bmp; out=mp+x1
import os, math
import numpy as np

S, E, Hn = 8, 768, 12
Hd, H = E // Hn, 4 * E
rng = np.random.default_rng(23)
def R(a, b): return (rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32)
def v(n): return (0.01 * rng.standard_normal(n)).astype(np.float32)
x   = (rng.standard_normal((S, E)) * 0.5).astype(np.float32)
ln1g=(1+0.02*rng.standard_normal(E)).astype(np.float32); ln1b=v(E)
ln2g=(1+0.02*rng.standard_normal(E)).astype(np.float32); ln2b=v(E)
Wq,Wk,Wv,Wap = R(E,E),R(E,E),R(E,E),R(E,E); bq,bk,bv,bap = v(E),v(E),v(E),v(E)
Wfc,Wmp = R(E,H),R(H,E); bfc,bmp = v(H),v(E)

def LN(t,g,b,e=1e-5): m=t.mean(-1,keepdims=True); return (t-m)/np.sqrt(t.var(-1,keepdims=True)+e)*g+b
def gelu_erf(t): return 0.5*t*(1.0+np.vectorize(math.erf)(t/math.sqrt(2.0)))
def cmha(Q,K,V):
    o=np.zeros_like(Q); sc0=1/np.sqrt(Hd); mask=np.triu(np.ones((S,S)),1).astype(bool)
    for h in range(Hn):
        sl=slice(h*Hd,(h+1)*Hd); s=(Q[:,sl]@K[:,sl].T)*sc0; s[mask]=-1e9; s=s-s.max(-1,keepdims=True)
        e=np.exp(s); o[:,sl]=(e/e.sum(-1,keepdims=True))@V[:,sl]
    return o

ln1=LN(x,ln1g,ln1b)
Q=ln1@Wq+bq; K=ln1@Wk+bk; Vv=ln1@Wv+bv
x1=(cmha(Q,K,Vv)@Wap+bap)+x
ln2=LN(x1,ln2g,ln2b)
out=(gelu_erf(ln2@Wfc+bfc)@Wmp+bmp)+x1

D=os.path.dirname(os.path.abspath(__file__))
def d(n,a): np.ascontiguousarray(a,dtype=np.float32).tofile(os.path.join(D,n))
d("ly_x.bin",x); d("ly_ln1g.bin",ln1g); d("ly_ln1b.bin",ln1b); d("ly_ln2g.bin",ln2g); d("ly_ln2b.bin",ln2b)
d("ly_wq.bin",Wq); d("ly_wk.bin",Wk); d("ly_wv.bin",Wv); d("ly_wap.bin",Wap); d("ly_wfc.bin",Wfc); d("ly_wmp.bin",Wmp)
for n,bb,sz in [("bq",bq,E),("bk",bk,E),("bv",bv,E),("bap",bap,E),("bfc",bfc,H),("bmp",bmp,E)]:
    d(f"ly_{n}.bin", np.broadcast_to(bb,(S,sz)))
d("ly_ref.bin",out)
open(os.path.join(D,"ly_dims.txt"),"w").write(f"{S} {E} {Hn} {H}\n")
print(f"[prep] S={S} E={E} Hn={Hn} H={H}  full-layer ref dumped. out[0,:3]={out[0,:3]}")
