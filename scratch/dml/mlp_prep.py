# Generate inputs + a numpy reference for the fused-on-device MLP block test.
# MLP: ln = LayerNorm(x); fc = ln@Wfc + bfc; g = gelu_erf(fc); proj = g@Wproj + bproj; out = proj + x
# gelu is the EXACT/erf form to match DirectML's DML_OPERATOR_ACTIVATION_GELU (not the driver's tanh).
import os, math
import numpy as np

S, E = 8, 768
H = 4 * E
rng = np.random.default_rng(7)
x     = rng.standard_normal((S, E)).astype(np.float32) * 0.5
ln_g  = (1.0 + 0.02 * rng.standard_normal(E)).astype(np.float32)
ln_b  = (0.01 * rng.standard_normal(E)).astype(np.float32)
Wfc   = (rng.standard_normal((E, H)) / np.sqrt(E)).astype(np.float32)
bfc   = (0.01 * rng.standard_normal(H)).astype(np.float32)
Wproj = (rng.standard_normal((H, E)) / np.sqrt(H)).astype(np.float32)
bproj = (0.01 * rng.standard_normal(E)).astype(np.float32)

def layernorm(v, g, b, eps=1e-5):
    mu = v.mean(-1, keepdims=True); var = v.var(-1, keepdims=True)
    return (v - mu) / np.sqrt(var + eps) * g + b

def gelu_erf(v):
    return 0.5 * v * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))

ln   = layernorm(x, ln_g, ln_b)
fc   = ln @ Wfc + bfc
g    = gelu_erf(fc)
proj = g @ Wproj + bproj
out  = proj + x

D = os.path.dirname(os.path.abspath(__file__))
def dump(name, arr): np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(D, name))
dump("mlp_x.bin", x)
dump("mlp_lng.bin", ln_g);   dump("mlp_lnb.bin", ln_b)
dump("mlp_wfc.bin", Wfc)
dump("mlp_bfc.bin", np.broadcast_to(bfc, (S, H)))       # pre-tiled to [S,H] (avoid GEMM C broadcast strides)
dump("mlp_wproj.bin", Wproj)
dump("mlp_bproj.bin", np.broadcast_to(bproj, (S, E)))   # pre-tiled to [S,E]
dump("mlp_ref.bin", out)
open(os.path.join(D, "mlp_dims.txt"), "w").write(f"{S} {E} {H}\n")
print(f"[prep] S={S} E={E} H={H}  dumped x/weights/biases/ref (erf-gelu). out[0,:3]={out[0,:3]}")
