#!/usr/bin/env python3
# gpt2_safetensors_to_gguf.py -- convert ANY GPT-2 (.safetensors) into a gpt2 GGUF that
# khanary-server (llama.cpp `gpt2` arch) can load + run on the XCFE backend.
#
# Handles, by auto-detection:
#   - naming:  `transformer.h.N`/`transformer.wte` (HF) OR `h.N`/`wte` (this stack's trainer)
#   - size:    gpt2 small/medium/large/xl (n_layer from tensors, n_embd from wte)
#   - shapes:  reconstructed from byte size (works even when the header shape is [] -- the
#              D3D11 trainer writes empty shapes)
#   - skips HF causal-mask buffers `attn.bias` / `attn.masked_bias`
# Maps HF names -> llama gpt2 GGUF names, TRANSPOSES the four Conv1D weight matrices, copies the
# vetted GPT-2 vocab.  llama's load-time shape check is the correctness gate.
#
# Usage: python gpt2_safetensors_to_gguf.py <in.safetensors> <out.gguf> [--vocab <ggml-vocab-gpt-2.gguf>]

import sys, os, json, struct, argparse
import numpy as np
import gguf

GPT2_VOCAB = 50257
DEFAULT_VOCAB = r"C:\Users\canna\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama.cpp\models\ggml-vocab-gpt-2.gguf"
# HF Conv1D weights are [in,out]; llama's Linear-style gpt2 loader wants [out,in] -> transpose.
CONV1D_W = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
SKIP = {"attn.bias", "attn.masked_bias"}  # HF causal-mask buffers, not weights

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); base = 8 + n
    return hdr, base

def norm(name):
    """strip a leading 'transformer.' so we work in one namespace: h.N.* / wte / wpe / ln_f.*"""
    return name[len("transformer."):] if name.startswith("transformer.") else name

def numel(hdr, name):
    a, b = hdr[name]["data_offsets"]; return (b - a) // 4  # F32

def detect(hdr):
    names = [k for k in hdr if k != "__metadata__"]
    norms = {norm(k): k for k in names}
    wte = norms.get("wte.weight") or norms.get("wte")
    n_embd = round(numel(hdr, wte) / GPT2_VOCAB)
    vocab  = numel(hdr, wte) // n_embd
    layers = max(int(k.split(".")[1]) for k in norms if k.startswith("h.")) + 1
    wpe = norms.get("wpe.weight") or norms.get("wpe")
    n_ctx = numel(hdr, wpe) // n_embd
    cfc = next((norms[k] for k in norms if k.endswith("mlp.c_fc.weight")), None)
    n_ff = numel(hdr, cfc) // n_embd if cfc else 4 * n_embd
    return dict(n_embd=n_embd, vocab=vocab, n_layer=layers, n_ctx=n_ctx, n_ff=n_ff,
                n_head=n_embd // 64, names=names)

def role_shape(nm, d):
    E, F, V, C = d["n_embd"], d["n_ff"], d["vocab"], d["n_ctx"]
    if nm == "wte.weight": return (V, E)
    if nm == "wpe.weight": return (C, E)
    if nm in ("ln_f.weight", "ln_f.bias"): return (E,)
    tail = nm.split(".", 2)[2]  # after h.N.
    return {"ln_1.weight": (E,), "ln_1.bias": (E,), "ln_2.weight": (E,), "ln_2.bias": (E,),
            "attn.c_attn.weight": (E, 3*E), "attn.c_attn.bias": (3*E,),
            "attn.c_proj.weight": (E, E),   "attn.c_proj.bias": (E,),
            "mlp.c_fc.weight": (E, F),      "mlp.c_fc.bias": (F,),
            "mlp.c_proj.weight": (F, E),    "mlp.c_proj.bias": (E,)}[tail]

def gguf_name(nm):
    if nm == "wte.weight": return "token_embd.weight"
    if nm == "wpe.weight": return "position_embd.weight"
    if nm == "ln_f.weight": return "output_norm.weight"
    if nm == "ln_f.bias":   return "output_norm.bias"
    i = nm.split(".")[1]; tail = ".".join(nm.split(".")[2:])
    return {"ln_1.weight": f"blk.{i}.attn_norm.weight", "ln_1.bias": f"blk.{i}.attn_norm.bias",
            "attn.c_attn.weight": f"blk.{i}.attn_qkv.weight", "attn.c_attn.bias": f"blk.{i}.attn_qkv.bias",
            "attn.c_proj.weight": f"blk.{i}.attn_output.weight", "attn.c_proj.bias": f"blk.{i}.attn_output.bias",
            "ln_2.weight": f"blk.{i}.ffn_norm.weight", "ln_2.bias": f"blk.{i}.ffn_norm.bias",
            "mlp.c_fc.weight": f"blk.{i}.ffn_up.weight", "mlp.c_fc.bias": f"blk.{i}.ffn_up.bias",
            "mlp.c_proj.weight": f"blk.{i}.ffn_down.weight", "mlp.c_proj.bias": f"blk.{i}.ffn_down.bias"}[tail]

def field_value(reader, key):
    f = reader.get_field(key)
    if f is None: return None
    try: return f.contents()
    except Exception: return f.parts[f.data[0]] if f.data else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile"); ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    a = ap.parse_args()

    hdr, base = read_header(a.infile)
    d = detect(hdr)
    print(f"[detect] n_layer={d['n_layer']} n_embd={d['n_embd']} n_head={d['n_head']} "
          f"n_ff={d['n_ff']} n_ctx={d['n_ctx']} vocab={d['vocab']}")
    with open(a.infile, "rb") as f: blob = f.read()

    w = gguf.GGUFWriter(a.outfile, "gpt2")
    w.add_name(os.path.splitext(os.path.basename(a.infile))[0])
    w.add_context_length(d["n_ctx"]); w.add_embedding_length(d["n_embd"])
    w.add_block_count(d["n_layer"]); w.add_feed_forward_length(d["n_ff"])
    w.add_head_count(d["n_head"]); w.add_layer_norm_eps(1e-5)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    r = gguf.GGUFReader(a.vocab)
    w.add_tokenizer_model(field_value(r, "tokenizer.ggml.model"))
    pre = field_value(r, "tokenizer.ggml.pre")
    if pre: w.add_tokenizer_pre(pre)
    dec = lambda xs: [x.decode("utf-8") if isinstance(x, bytes) else x for x in xs]
    w.add_token_list(dec(field_value(r, "tokenizer.ggml.tokens")))
    w.add_token_types([int(x) for x in field_value(r, "tokenizer.ggml.token_type")])
    w.add_token_merges(dec(field_value(r, "tokenizer.ggml.merges")))
    for k, add in (("tokenizer.ggml.bos_token_id", w.add_bos_token_id),
                   ("tokenizer.ggml.eos_token_id", w.add_eos_token_id)):
        v = field_value(r, k)
        if v is not None: add(int(v))

    wte_arr = None; nt = 0
    for name in d["names"]:
        nm = norm(name)
        tail = nm.split(".", 2)[2] if nm.startswith("h.") else nm
        if tail in SKIP: continue
        a0, b0 = hdr[name]["data_offsets"]
        arr = np.frombuffer(blob[base+a0:base+b0], dtype=np.float32).reshape(role_shape(nm, d))
        if nm == "wte.weight": wte_arr = arr
        if tail in CONV1D_W: arr = arr.T
        w.add_tensor(gguf_name(nm), np.ascontiguousarray(arr, dtype=np.float32)); nt += 1
    w.add_tensor("output.weight", np.ascontiguousarray(wte_arr, dtype=np.float32)); nt += 1  # tied

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"[ok] {a.outfile} ({os.path.getsize(a.outfile)/1e6:.1f} MB, {nt} tensors)")

if __name__ == "__main__":
    main()
