#!/usr/bin/env python3
# gpt2_kuhul_to_gguf.py -- convert a KUHUL-extended GPT-2 safetensors into a gpt2 GGUF
# that llama.cpp / kuhul_engine can load.
#
# Unlike gpt2_safetensors_to_gguf.py, this converter:
#   - reads the vocab size from the wte tensor (supports 50270-token models)
#   - extends the GPT-2 tokenizer metadata with KUHUL special tokens from tokenizer_config.json
#   - preserves all 50270 embedding rows so llama.cpp's shape check passes
#
# Usage:
#   python tools/gpt2_kuhul_to_gguf.py models/from_zero/from_zero_v0.6_merged.safetensors models/from_zero/from_zero_v0.6_kuhul.gguf

import sys, os, json, struct, argparse
from pathlib import Path
import numpy as np
import gguf

GGUF_PY = Path(__file__).resolve().parent.parent / "khanary-llama-build" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))
from gguf import GGUFReader

DEFAULT_VOCAB = r"C:\Users\canna\_khanary_inspect\bin\ggml-vocab-gpt-2.gguf"
KUHUL_TOKENS_PATH = Path(__file__).resolve().parent.parent / "tokenizer_config.json"

# HF Conv1D weights are [in,out]; llama's Linear-style gpt2 loader wants [out,in] -> transpose.
CONV1D_W = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
SKIP = {"attn.bias", "attn.masked_bias"}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    return hdr, base


def norm(name):
    return name[len("transformer."): ] if name.startswith("transformer.") else name


def numel(hdr, name):
    a, b = hdr[name]["data_offsets"]
    return (b - a) // 4  # F32


def detect(hdr):
    names = [k for k in hdr if k != "__metadata__"]
    norms = {norm(k): k for k in names}
    wte = norms.get("wte.weight") or norms.get("wte")
    raw_vocab = numel(hdr, wte)
    # Infer n_embd from wpe or a layer bias
    wpe = norms.get("wpe.weight") or norms.get("wpe")
    n_embd_candidates = []
    if wpe:
        # wpe shape will be [n_ctx, n_embd]; try common n_ctx values
        for n_ctx in [1024, 2048, 512]:
            if raw_vocab % n_ctx == 0:
                n_embd_candidates.append(raw_vocab // n_ctx)
    # Use ln_f.weight to pin n_embd
    ln_f = norms.get("ln_f.weight") or norms.get("ln_f")
    if ln_f:
        n_embd_candidates.append(numel(hdr, ln_f))
    n_embd = max(set(n_embd_candidates), key=n_embd_candidates.count) if n_embd_candidates else 768
    vocab = numel(hdr, wte) // n_embd
    layers = max(int(k.split(".")[1]) for k in norms if k.startswith("h.")) + 1
    n_ctx = numel(hdr, wpe) // n_embd if wpe else 1024
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


def load_kuhul_tokens(path=KUHUL_TOKENS_PATH):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    extras = {}
    # Map explicit KUHUL tokens (50260-50269)
    for t in cfg.get("additional_special_tokens", []):
        extras[t["id"]] = t["content"]
    return extras


def extend_gpt2_vocab(base_tokens, base_types, base_merges, extras, target_vocab):
    """Extend a 50257-token GPT-2 vocab to target_vocab by appending specials.

    extras: dict {token_id: token_string} for IDs >= len(base_tokens)
    Missing IDs between len(base_tokens) and target_vocab are filled with placeholder specials.
    """
    tokens = list(base_tokens)
    types = list(base_types)
    merges = list(base_merges)
    n_base = len(tokens)
    if target_vocab < n_base:
        raise ValueError(f"target_vocab {target_vocab} < base vocab {n_base}")
    for i in range(n_base, target_vocab):
        tok = extras.get(i, f"<KUHUL_{i}>")
        tokens.append(tok)
        # llama.cpp token types: 1=normal, 2=unknown, 3=control, 4=user-defined, 5=unused, 6=byte
        types.append(3)  # control
    return tokens, types, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="Input safetensors (e.g. from_zero_v0.6_merged.safetensors)")
    ap.add_argument("outfile", help="Output GGUF")
    ap.add_argument("--vocab", default=DEFAULT_VOCAB, help="Reference GPT-2 vocab GGUF")
    ap.add_argument("--kuhul-tokens", default=str(KUHUL_TOKENS_PATH), help="tokenizer_config.json with KUHUL token IDs")
    a = ap.parse_args()

    hdr, base = read_header(a.infile)
    d = detect(hdr)
    vocab = d["vocab"]
    print(f"[detect] n_layer={d['n_layer']} n_embd={d['n_embd']} n_head={d['n_head']} "
          f"n_ff={d['n_ff']} n_ctx={d['n_ctx']} vocab={vocab}")

    with open(a.infile, "rb") as f: blob = f.read()

    print(f"[vocab] loading reference GPT-2 vocab from {a.vocab}")
    r = GGUFReader(a.vocab)
    dec = lambda xs: [x.decode("utf-8") if isinstance(x, bytes) else x for x in xs]
    base_tokens = dec(field_value(r, "tokenizer.ggml.tokens"))
    base_types = [int(x) for x in field_value(r, "tokenizer.ggml.token_type")]
    base_merges = dec(field_value(r, "tokenizer.ggml.merges"))
    print(f"[vocab] base tokens={len(base_tokens)} types={len(base_types)} merges={len(base_merges)}")

    extras = load_kuhul_tokens(Path(a.kuhul_tokens))
    print(f"[vocab] KUHUL extras: {extras}")
    tokens, types, merges = extend_gpt2_vocab(base_tokens, base_types, base_merges, extras, vocab)
    print(f"[vocab] extended to {len(tokens)} tokens")

    w = gguf.GGUFWriter(a.outfile, "gpt2")
    w.add_name(os.path.splitext(os.path.basename(a.infile))[0])
    w.add_context_length(d["n_ctx"])
    w.add_embedding_length(d["n_embd"])
    w.add_block_count(d["n_layer"])
    w.add_feed_forward_length(d["n_ff"])
    w.add_head_count(d["n_head"])
    w.add_layer_norm_eps(1e-5)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    w.add_tokenizer_model(field_value(r, "tokenizer.ggml.model"))
    pre = field_value(r, "tokenizer.ggml.pre")
    if pre: w.add_tokenizer_pre(pre)
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    for k, add in (("tokenizer.ggml.bos_token_id", w.add_bos_token_id),
                   ("tokenizer.ggml.eos_token_id", w.add_eos_token_id)):
        v = field_value(r, k)
        if v is not None: add(int(v))

    # Add KUHUL metadata so consumers know this is an extended vocab
    w.add_bool("kuhul.extended_vocab", True)
    w.add_string("kuhul.special_tokens", json.dumps(extras))

    wte_arr = None; nt = 0
    for name in d["names"]:
        nm = norm(name)
        tail = nm.split(".", 2)[2] if nm.startswith("h.") else nm
        if tail in SKIP: continue
        a0, b0 = hdr[name]["data_offsets"]
        arr = np.frombuffer(blob[base+a0:base+b0], dtype=np.float32).reshape(role_shape(nm, d))
        if nm == "wte.weight":
            wte_arr = arr
            print(f"[wte] shape {arr.shape}")
        if tail in CONV1D_W: arr = arr.T
        w.add_tensor(gguf_name(nm), np.ascontiguousarray(arr, dtype=np.float32)); nt += 1
    w.add_tensor("output.weight", np.ascontiguousarray(wte_arr, dtype=np.float32)); nt += 1  # tied

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"[ok] {a.outfile} ({os.path.getsize(a.outfile)/1e6:.1f} MB, {nt} tensors)")


if __name__ == "__main__":
    main()
