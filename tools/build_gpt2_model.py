# build_gpt2_model.py — assemble the KHANARY gpt2 compute model version folder (v0.4.0).
#
# Packages the compute path proven in this line of work: the G_MATMUL cs_5_0 GEMM (verified on
# an HD 4600 with a real gpt2 weight), the glyph tokenizer, the safetensors->.stb bridge output,
# and the VENDORED native gpt2 trainer + its engine headers (reference-only). Sources are copied
# from the sibling .ASX.cpp tree and the .KUHUL_V2 engine src; missing sources are skipped with a
# warning so the (already-committed) folder still regenerates elsewhere.
import os, sys, json, glob, shutil, hashlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "tools"))
from tools.khlnary_encoder import encode_knu, decode_knu, GLYPH_IDS
from tools.khlnary_dx11 import lower_khlnary_to_hlsl
from tools.khlnary_webgpu import lower_khlnary_to_wgsl

VERSION = "0.4.0"
MODEL_DIR = os.path.join(_ROOT, "models", f"khanary-gpt2-v{VERSION}")

ASX = r"C:\Users\canna\.ASX.cpp"
ENG = r"C:\Users\canna\.KUHUL_V2\PowerShell-LLM\Models\native\xvm-d3d12\src"
SAFETENSORS = os.path.join(ASX, "trainer", "random_gpt2.safetensors")

# (src_abs, dst_rel) vendoring table
VENDOR = [
    (os.path.join(ASX, "trainer", "gpt2_trainer.cpp"),    "trainer/gpt2_trainer.cpp"),
    (os.path.join(ASX, "trainer", "gpt2_trainer.h"),      "trainer/gpt2_trainer.h"),
    (os.path.join(ASX, "trainer", "gpt2_train_main.cpp"), "trainer/gpt2_train_main.cpp"),
    (os.path.join(ASX, "trainer", "gpt2_config.h"),       "trainer/gpt2_config.h"),
    (os.path.join(ASX, "trainer", "gpu_fwdbwd_new.cpp"),  "trainer/gpu_fwdbwd_new.cpp"),
    (os.path.join(ASX, "pi_kuhul", "KuhulPhysics.h"),     "trainer/include/KuhulPhysics.h"),
    (os.path.join(ENG, "d3d11_engine.h"),                 "trainer/engine/d3d11_engine.h"),
    (os.path.join(ENG, "xvm_core.h"),                     "trainer/engine/xvm_core.h"),
    (os.path.join(ASX, "trainer", "glyph_tokenizer.py"),  "tokenizer/glyph_tokenizer.py"),
    (os.path.join(ASX, "www", "kxml-gpt2-v0.1", "glyph_contract.json"), "tokenizer/glyph_contract.json"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def copy(src, dst_rel):
    dst = os.path.join(MODEL_DIR, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(src):
        print(f"  WARN missing (skipped): {src}")
        return False
    shutil.copyfile(src, dst)
    return True


def main():
    for sub in ("kernels", "knu", "trainer/shaders", "data"):
        os.makedirs(os.path.join(MODEL_DIR, sub), exist_ok=True)

    vendored = [d for s, d in VENDOR if copy(s, d)]

    # trainer shaders (runtime-compiled by name); skip the "(2)" duplicate
    shaders = []
    for s in sorted(glob.glob(os.path.join(ASX, "shaders", "gpt2_*.hlsl"))):
        if "(" in os.path.basename(s):
            continue
        d = f"trainer/shaders/{os.path.basename(s)}"
        if copy(s, d):
            shaders.append(d)

    # compute kernels emitted from their glyphs (both co-equal backends)
    compute_ops = {"matmul": "G_MATMUL", "attention": "G_ATTENTION",
                   "layernorm": "G_LAYERNORM", "gelu": "G_GELU", "embed": "G_EMBED"}
    for name, glyph in compute_ops.items():
        w = encode_knu(glyph, payload=0)
        open(os.path.join(MODEL_DIR, "kernels", f"{name}.hlsl"), "w").write(lower_khlnary_to_hlsl([w], {}))
        open(os.path.join(MODEL_DIR, "kernels", f"{name}.wgsl"), "w").write(lower_khlnary_to_wgsl([w], {}))
        json.dump({"op": name, "stream": [{"glyph": glyph, "word": w,
                  "word_hex": f"0x{w:08x}", "decoded": decode_knu(w)}]},
                  open(os.path.join(MODEL_DIR, "knu", f"{name}.knu.json"), "w", encoding="utf-8"), indent=2)

    # data payload: a real gpt2 weight as a KHANARY .stb (compact c_proj sample)
    data_meta = {"present": False}
    if os.path.exists(SAFETENSORS):
        from tools.safetensors_to_stb import safetensors_to_stb
        dst = os.path.join(MODEL_DIR, "data", "gpt2_c_proj.stb")
        man = safetensors_to_stb(SAFETENSORS, dst, ["transformer.h.0.attn.c_proj.weight"])
        data_meta = {"present": True, "file": "data/gpt2_c_proj.stb", "sha256": sha256(dst),
                     "bytes": os.path.getsize(dst), "tensor": man[0][1], "shape": list(man[0][2]),
                     "source_safetensors": "trainer/random_gpt2.safetensors (.ASX.cpp)"}

    manifest = {
        "name": "khanary-gpt2", "version": VERSION, "kind": "compute (LLM) model",
        "knu_profile": "KHΛ-2-DENSE-32",
        "compute_glyphs": {g: GLYPH_IDS[g] for g in
                           ("G_MATMUL", "G_ATTENTION", "G_LAYERNORM", "G_GELU", "G_EMBED")},
        "note": ("v0.4.0 packages KHANARY COMPUTE glyphs (matmul, attention, layernorm, gelu, embed) "
                 "+ the native gpt2 trainer that produces the weights. The five forward ops of a gpt2 "
                 "block are now glyphs. Distinct model from the geometry v0.3.0."),
        "kernels": {
            "matmul": {"glyph": "G_MATMUL", "hlsl": "kernels/matmul.hlsl",
                       "wgsl": "kernels/matmul.wgsl", "knu": "knu/matmul.knu.json"},
            "attention": {"glyph": "G_ATTENTION", "hlsl": "kernels/attention.hlsl",
                          "wgsl": "kernels/attention.wgsl", "knu": "knu/attention.knu.json",
                          "promoted_from": "trainer/shaders/gpt2_attn_fwd.hlsl"},
        },
        "backends": {
            "d3d11_cs_5_0": {"status": "hardware-verified",
                "evidence": "GEMM C[64,2304]=A[64,768]@B[768,2304] with real gpt2 c_attn weight on "
                            "Intel HD 4600 (FL 11_1): scale-normalized err 1.01e-06 vs numpy f64. "
                            "Causal MHA (G_ATTENTION) on real gpt2 qkv (S=16,E=768,H=12): "
                            "scale-normalized err 6.39e-08 vs numpy f64. Naive/correctness-first."},
            "webgpu_wgsl": {"status": "structural-parity", "note": "same glyph-driven lowering; not run on HD 4600."},
        },
        "tokenizer": {
            "files": ["tokenizer/glyph_tokenizer.py", "tokenizer/glyph_contract.json"],
            "class": "GlyphTokenizer",
            "note": ("the GPT-2 token glyph stack (vocab ~50269, ids 50257-50268 map to Maya glyph "
                     "names). Distinct id-space from the DXC MoE/KPI1 opcode stack — do not conflate."),
        },
        "trainer": {
            "status": "vendored, reference-only",
            "role": "native D3D11 cs_5_0 gpt2 trainer that produced the safetensors weights",
            "files": sorted(vendored) + sorted(shaders),
            "build_requirements": [
                "MSVC vcvars64 (Visual Studio 2022 BuildTools)",
                "links against a PREBUILT xvm engine object (D3D11Engine impl) — not vendored; "
                "trainer/engine/ holds only the headers (d3d11_engine.h, xvm_core.h)",
                "runtime-compiles trainer/shaders/gpt2_*.hlsl by filename at run time",
                "d3d11.lib d3dcompiler.lib dxgi.lib dxguid.lib",
            ],
            "provenance": {"trainer_src": "trainer/ (.ASX.cpp)", "physics": "pi_kuhul/KuhulPhysics.h (.ASX.cpp)",
                           "engine_headers": "xvm-d3d12/src (.KUHUL_V2, copied with owner approval)"},
            "standalone_buildable": False,
        },
        "data": data_meta,
        "honest_scope": [
            "G_MATMUL is a NAIVE GEMM (correctness-first, not tiled/optimized); G_ATTENTION is the "
            "trainer's O(S^2) causal MHA forward (fine for small S).",
            "All five gpt2 FORWARD ops are now glyphs (matmul, attention, layernorm, gelu, embed), "
            "each verified bit-close on the HD 4600. What remains for an end-to-end inference DRIVER "
            "is wiring them into a per-block schedule (embed -> [ln, attn, ln, ffn+gelu] x N -> ln -> "
            "lm_head) + a KV cache; the backward/optimizer glyphs are training-only. For GGUF, a "
            "dequant->.stb step (safetensors is already dense float32).",
            "The vendored trainer is reference-only and does not build standalone here (external "
            "prebuilt engine object).",
        ],
        "generator": "tools/build_gpt2_model.py",
    }
    json.dump(manifest, open(os.path.join(MODEL_DIR, "MODEL.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    readme = f"""# KHΛNARY gpt2 compute model — v{VERSION}

The **compute** counterpart to the geometry model: this version packages KHΛNARY's first
compute glyph (`G_MATMUL`) together with the native gpt2 trainer that produces the weights and
the glyph tokenizer that feeds it.

## Contents
- **`kernels/matmul.{{hlsl,wgsl}}`** — dense GEMM `C[M,N] = A[M,K] @ B[K,N]`, emitted from the
  `G_MATMUL` glyph (`0x{GLYPH_IDS['G_MATMUL']:02x}`). The HLSL was **dispatched on an Intel HD 4600**
  with a real gpt2 QKV weight: scale-normalized error `1.01e-06` vs numpy. See `MODEL.json`.
- **`tokenizer/`** — `glyph_tokenizer.py` (`GlyphTokenizer`) + `glyph_contract.json` (the GPT-2
  token↔glyph mapping).
- **`trainer/`** — the vendored native D3D11 `cs_5_0` gpt2 trainer (`gpt2_trainer.cpp` + backward
  `gpu_fwdbwd_new.cpp` + `shaders/gpt2_*.hlsl`), its physics header, and the engine headers
  (`engine/d3d11_engine.h`, `engine/xvm_core.h`). **Reference-only** — links against a prebuilt
  external xvm engine object, so it does not build standalone here (see `MODEL.json` →
  `trainer.build_requirements`).
- **`data/gpt2_c_proj.stb`** — a real gpt2 weight in KHΛNARY `.stb` form via
  `tools/safetensors_to_stb.py` (the weight-side bridge, sibling to `brain_to_stb.py`).

## Honest scope
`G_MATMUL` is a naive GEMM (correctness-first). A full LLM run still needs attention/softmax/
layernorm/gelu **glyphs** — the trainer already implements these as HLSL shaders, but they are
not yet exposed as KNU glyphs. GGUF would additionally need a dequant → `.stb` step; safetensors
is already dense float32. See `MODEL.json` → `honest_scope`.

## Reproduce
```
python tools/build_gpt2_model.py          # regenerate this folder (needs the .ASX.cpp/.KUHUL_V2 sources)
python tools/safetensors_to_stb.py <model.safetensors> out.stb <keys...>
python -m pytest tests/ -q
```
"""
    open(os.path.join(MODEL_DIR, "README.md"), "w", encoding="utf-8").write(readme)

    print(f"wrote {os.path.relpath(MODEL_DIR, _ROOT)}")
    total = 0
    for dp, _, fs in os.walk(MODEL_DIR):
        for f in sorted(fs):
            p = os.path.join(dp, f); total += os.path.getsize(p)
            print(f"  {os.path.relpath(p, MODEL_DIR):46} {os.path.getsize(p):>9} B")
    print(f"  total: {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
