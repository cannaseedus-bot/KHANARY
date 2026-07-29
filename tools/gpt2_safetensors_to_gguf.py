#!/usr/bin/env python3
# gpt2_safetensors_to_gguf.py -- convert a KHANARY-trained GPT-2 (.safetensors) into a GGUF
# that khanary-server (llama.cpp `gpt2` arch) can load and run on the XCFE backend.
#
# Why this exists: this llama.cpp (b9968) DROPPED gpt2 from convert_hf_to_gguf.py (the runtime
# still supports `gpt2`), and the trainer wrote the safetensors WITHOUT shapes ('shape': []).
# So we reconstruct GPT-2 shapes, map HF names -> llama gpt2 GGUF names (TRANSPOSING the four
# Conv1D weight matrices, the classic gpt2 gotcha), and copy the GPT-2 vocab from the vetted
# ggml-vocab-gpt-2.gguf. Load-time shape validation in llama.cpp is the correctness check.
#
# Usage: python gpt2_safetensors_to_gguf.py <in.safetensors> <out.gguf> [--vocab <ggml-vocab-gpt-2.gguf>]

import sys, os, json, struct, argparse
import numpy as np
import gguf

# GPT-2 small (124M) hyper-params -- fixed by the 497.8MB/148-tensor fingerprint we measured.
N_LAYER, N_EMBD, N_HEAD, N_CTX, N_FF, N_VOCAB, LN_EPS = 12, 768, 12, 1024, 3072, 50257, 1e-5
DEFAULT_VOCAB = r"C:\Users\canna\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama.cpp\models\ggml-vocab-gpt-2.gguf"

def hf_shape(name):
    """The HF GPT-2 shape for a tensor (used to reshape the flat F32 blob)."""
    if name == "transformer.wte.weight":  return (N_VOCAB, N_EMBD)
    if name == "transformer.wpe.weight":  return (N_CTX, N_EMBD)
    if name in ("transformer.ln_f.weight", "transformer.ln_f.bias"): return (N_EMBD,)
    tail = name.split(".", 3)[3]  # after "transformer.h.N."
    return {
        "ln_1.weight": (N_EMBD,), "ln_1.bias": (N_EMBD,),
        "ln_2.weight": (N_EMBD,), "ln_2.bias": (N_EMBD,),
        "attn.c_attn.weight": (N_EMBD, 3*N_EMBD), "attn.c_attn.bias": (3*N_EMBD,),
        "attn.c_proj.weight": (N_EMBD, N_EMBD),   "attn.c_proj.bias": (N_EMBD,),
        "mlp.c_fc.weight": (N_EMBD, N_FF),        "mlp.c_fc.bias": (N_FF,),
        "mlp.c_proj.weight": (N_FF, N_EMBD),      "mlp.c_proj.bias": (N_EMBD,),
    }[tail]

# HF Conv1D weights are stored [in, out]; llama's Linear-style gpt2 loader wants [out, in] -> transpose.
CONV1D_W = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}

def gguf_name(name):
    """Map an HF GPT-2 tensor name to the llama.cpp gpt2 GGUF tensor name."""
    if name == "transformer.wte.weight": return "token_embd.weight"
    if name == "transformer.wpe.weight": return "position_embd.weight"
    if name == "transformer.ln_f.weight": return "output_norm.weight"
    if name == "transformer.ln_f.bias":   return "output_norm.bias"
    parts = name.split(".")           # transformer h N <tail...>
    i = parts[2]; tail = ".".join(parts[3:])
    m = {
        "ln_1.weight": f"blk.{i}.attn_norm.weight", "ln_1.bias": f"blk.{i}.attn_norm.bias",
        "attn.c_attn.weight": f"blk.{i}.attn_qkv.weight", "attn.c_attn.bias": f"blk.{i}.attn_qkv.bias",
        "attn.c_proj.weight": f"blk.{i}.attn_output.weight", "attn.c_proj.bias": f"blk.{i}.attn_output.bias",
        "ln_2.weight": f"blk.{i}.ffn_norm.weight", "ln_2.bias": f"blk.{i}.ffn_norm.bias",
        "mlp.c_fc.weight": f"blk.{i}.ffn_up.weight", "mlp.c_fc.bias": f"blk.{i}.ffn_up.bias",
        "mlp.c_proj.weight": f"blk.{i}.ffn_down.weight", "mlp.c_proj.bias": f"blk.{i}.ffn_down.bias",
    }
    return m[tail]

def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        f.seek(0); blob = f.read()
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__": continue
        a, b = meta["data_offsets"]
        shp = hf_shape(name)
        arr = np.frombuffer(blob[base+a:base+b], dtype=np.float32).reshape(shp)
        out[name] = arr
    return out

def field_value(reader, key):
    f = reader.get_field(key)
    if f is None: return None
    try: return f.contents()          # modern gguf-py
    except Exception: pass
    # fallback: single-value scalar/string
    return f.parts[f.data[0]] if f.data else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    a = ap.parse_args()

    print(f"[1/4] read {a.infile}")
    tensors = read_safetensors(a.infile)
    print(f"      {len(tensors)} tensors")

    print(f"[2/4] GGUFWriter (arch=gpt2)")
    w = gguf.GGUFWriter(a.outfile, "gpt2")
    w.add_name("from_zero_v0.1")
    w.add_context_length(N_CTX); w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER); w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD); w.add_layer_norm_eps(LN_EPS)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    print(f"[3/4] copy GPT-2 vocab from {os.path.basename(a.vocab)}")
    r = gguf.GGUFReader(a.vocab)
    w.add_tokenizer_model(field_value(r, "tokenizer.ggml.model"))
    pre = field_value(r, "tokenizer.ggml.pre")
    if pre: w.add_tokenizer_pre(pre)
    toks = [t.decode("utf-8") if isinstance(t, bytes) else t for t in field_value(r, "tokenizer.ggml.tokens")]
    w.add_token_list(toks)
    w.add_token_types([int(x) for x in field_value(r, "tokenizer.ggml.token_type")])
    merges = [m.decode("utf-8") if isinstance(m, bytes) else m for m in field_value(r, "tokenizer.ggml.merges")]
    w.add_token_merges(merges)
    for k, add in (("tokenizer.ggml.bos_token_id", w.add_bos_token_id),
                   ("tokenizer.ggml.eos_token_id", w.add_eos_token_id)):
        v = field_value(r, k)
        if v is not None: add(int(v))
    print(f"      vocab={len(toks)} merges={len(merges)}")

    print(f"[4/4] add tensors (transpose Conv1D weights)")
    nt = 0
    for name, arr in tensors.items():
        tail = ".".join(name.split(".")[3:]) if name.startswith("transformer.h.") else name
        data = arr.T.copy() if tail in CONV1D_W else arr
        w.add_tensor(gguf_name(name), np.ascontiguousarray(data, dtype=np.float32)); nt += 1
    # gpt2 ties the output projection to the token embedding
    w.add_tensor("output.weight", np.ascontiguousarray(tensors["transformer.wte.weight"], dtype=np.float32)); nt += 1
    print(f"      wrote {nt} tensors")

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"[ok] {a.outfile} ({os.path.getsize(a.outfile)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
