# export_gpt2_flat.py — export a gpt2 safetensors as the flat blob the native GPU inference
# driver (scratch/infer/gpt2_infer_run.cpp) loads: weights in canonical order + dims + a prompt
# and the CPU driver's greedy generation (the correctness oracle the GPU run is checked against).
#
# Run: python tools/export_gpt2_flat.py <gpt2.safetensors> <out_dir> [prompt tokens...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from safetensors import safe_open


def canonical_order(n_layer):
    order = ["transformer.wte.weight", "transformer.wpe.weight"]
    for l in range(n_layer):
        p = f"transformer.h.{l}."
        order += [p + s for s in ("ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias",
                                  "attn.c_proj.weight", "attn.c_proj.bias", "ln_2.weight", "ln_2.bias",
                                  "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")]
    order += ["transformer.ln_f.weight", "transformer.ln_f.bias"]
    return order


def main():
    src, out = sys.argv[1], sys.argv[2]
    prompt = [int(t) for t in sys.argv[3:]] or [15496, 11, 314, 716]
    os.makedirs(out, exist_ok=True)
    with safe_open(src, framework="numpy") as f:
        E = f.get_tensor("transformer.wte.weight").shape[1]
        V = f.get_tensor("transformer.wte.weight").shape[0]
        CTX = f.get_tensor("transformer.wpe.weight").shape[0]
        n_layer = 1 + max(int(k.split(".")[2]) for k in f.keys() if k.startswith("transformer.h."))
        NH = E // 64
        with open(os.path.join(out, "gpt2.weights"), "wb") as w:
            for k in canonical_order(n_layer):
                w.write(np.ascontiguousarray(f.get_tensor(k).astype(np.float32)).tobytes())
    open(os.path.join(out, "gpt2.dims"), "w").write(f"{n_layer} {E} {NH} {CTX} {V}")
    open(os.path.join(out, "gpt2.plen"), "w").write(str(len(prompt)))
    np.array(prompt, np.int32).tofile(os.path.join(out, "gpt2.prompt"))

    # CPU-driver greedy oracle (needs the full-model .stb + manifest built by safetensors_to_model_stb)
    try:
        from tools.kxml_inference_driver import KxmlModel
        stb = os.path.join(os.path.dirname(out), "gpt2_model.stb")
        man = os.path.join(os.path.dirname(os.path.dirname(out)),
                           "models", "khanary-gpt2-v0.4.0", "model", "gpt2_model.stb.json")
        if os.path.exists(stb) and os.path.exists(man):
            gen = KxmlModel(stb, man).generate(prompt, n=8)
            np.array(gen, np.int32).tofile(os.path.join(out, "gpt2.expected"))
            print(f"CPU greedy oracle: {gen}")
    except Exception as e:
        print(f"(oracle skipped: {e})")
    print(f"exported {n_layer}L E={E} V={V} -> {out}/gpt2.weights + dims/plen/prompt")


if __name__ == "__main__":
    main()
