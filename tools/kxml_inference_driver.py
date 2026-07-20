# kxml_inference_driver.py — walk a full-model .stb + manifest forward graph and run the model.
#
# The capstone: the 5 forward glyphs (embed/layernorm/matmul/attention/gelu) were each verified on
# the iGPU; this driver WALKS the manifest's forward_graph and threads them into a real GPT-2
# forward pass over the weights in the .stb. The op bodies here are numpy references that mirror the
# verified HLSL glyphs exactly — so this is both a runnable model and the reference a GPU driver
# (same graph walk, swapping numpy for the verified glyph dispatches) must match.
#
# Run: python tools/kxml_inference_driver.py <model.stb> <model.stb.json> [--verify-hf <safetensors>]
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stb import read_stb


# ── op kernels (numpy mirrors of the verified KNU glyphs) ─────────────────────────
def op_embed(tokens, wte, wpe):
    return wte[tokens] + wpe[:len(tokens)]                       # G_EMBED


def op_layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta          # G_LAYERNORM


def op_matmul(x, W, bias=None, transpose_B=False):
    y = x @ (W.T if transpose_B else W)                          # G_MATMUL (gpt2 Conv1D: x @ W)
    return y + bias if bias is not None else y


def op_gelu(x):
    k = np.clip(0.7978845608 * (x + 0.044715 * x**3), -10, 10)
    return 0.5 * x * (1.0 + np.tanh(k))                          # G_GELU (tanh approx + clamp)


def op_attention(qkv, n_head, head_dim):                        # G_ATTENTION (causal MHA)
    S = qkv.shape[0]; E = n_head * head_dim
    out = np.zeros((S, E), dtype=qkv.dtype)
    scale = 1.0 / np.sqrt(head_dim)
    causal = np.triu(np.ones((S, S), bool), 1)
    for h in range(n_head):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        Q, K, V = qkv[:, sl], qkv[:, E + h * head_dim:E + (h + 1) * head_dim], qkv[:, 2 * E + h * head_dim:2 * E + (h + 1) * head_dim]
        s = (Q @ K.T) * scale
        s[causal] = -1e30
        s -= s.max(-1, keepdims=True)
        w = np.exp(s); w /= w.sum(-1, keepdims=True)
        out[:, sl] = w @ V
    return out


class KxmlModel:
    def __init__(self, stb_path, manifest_path):
        self.man = json.load(open(manifest_path, encoding="utf-8"))
        self.cfg = self.man["config"]
        self.graph = self.man["forward_graph"]
        self.name2id = {k: v["id"] for k, v in self.man["tensors"].items()}
        raw = read_stb(stb_path)
        self.W = {tid: np.asarray(raw[tid]["array"]) for tid in raw}

    def _t(self, ref):
        if isinstance(ref, dict):        # resolved {name,id}
            return self.W[ref["id"]]
        return self.W[self.name2id[ref]]  # name string (e.g. a bias)

    def forward(self, tokens):
        cfg = self.cfg
        x = None; residual = None
        for step in self.graph:
            g = step["glyph"]; reads = step.get("reads", {})
            if g == "G_EMBED":
                x = op_embed(np.asarray(tokens), self._t(reads["wte"]), self._t(reads["wpe"]))
            elif g == "G_LAYERNORM":
                if not step["step"].startswith("ln_f"):
                    residual = x                                 # residual saved before the LN
                x = op_layernorm(x, self._t(reads["gamma"]), self._t(reads["beta"]), cfg["ln_eps"])
            elif g == "G_MATMUL":
                bias = self._t(step["bias"]) if "bias" in step else None
                x = op_matmul(x, self._t(reads["B"]), bias, step.get("transpose_B", False))
                if step.get("residual"):
                    x = x + residual
            elif g == "G_ATTENTION":
                x = op_attention(x, cfg["n_head"], cfg["n_embd"] // cfg["n_head"])
            elif g == "G_GELU":
                x = op_gelu(x)
            else:
                raise ValueError(f"unknown glyph {g}")
        return x                                                 # logits [S, vocab]

    def generate(self, tokens, n=8, greedy=True):
        toks = list(tokens)
        for _ in range(n):
            logits = self.forward(toks[-self.cfg["n_ctx"]:])
            nxt = int(np.argmax(logits[-1])) if greedy else int(np.argmax(logits[-1]))
            toks.append(nxt)
        return toks


def _verify_hf(model, safetensors_path, S=12):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config
    cfg = model.cfg
    hf = GPT2LMHeadModel(GPT2Config(vocab_size=cfg["vocab"], n_positions=cfg["n_ctx"],
                                    n_embd=cfg["n_embd"], n_layer=cfg["n_layer"], n_head=cfg["n_head"]))
    from safetensors.torch import load_file
    sd = load_file(safetensors_path)
    hf.transformer.load_state_dict({k[len("transformer."):]: v for k, v in sd.items()
                                    if k.startswith("transformer.")}, strict=False)
    hf.lm_head.weight = hf.transformer.wte.weight
    hf.eval()
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, cfg["vocab"], size=S).tolist()
    ours = model.forward(tokens)
    with torch.no_grad():
        ref = hf(torch.tensor([tokens])).logits[0].numpy()
    scale = np.abs(ref).max()
    nrm = np.abs(ours - ref).max() / scale
    print(f"[verify] driver vs HuggingFace GPT2  logits[{S},{cfg['vocab']}]  "
          f"max abs {np.abs(ours-ref).max():.3e}  scale-norm {nrm:.2e}")
    print(f"=== {'PASS' if nrm < 1e-4 else 'FAIL'}: graph-walking driver matches HF GPT2 ===")
    return nrm < 1e-4


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python tools/kxml_inference_driver.py <model.stb> <model.stb.json> [--verify-hf <st>]")
        sys.exit(1)
    m = KxmlModel(sys.argv[1], sys.argv[2])
    print(f"[model] {m.cfg['n_layer']}L n_embd={m.cfg['n_embd']} vocab={m.cfg['vocab']} "
          f"| {len(m.graph)} graph nodes | {len(m.W)} tensors")
    if "--verify-hf" in sys.argv:
        ok = _verify_hf(m, sys.argv[sys.argv.index("--verify-hf") + 1])
        sys.exit(0 if ok else 1)
    out = m.generate([50256], n=6)
    print("[generate] greedy 6 tokens from BOS:", out)
