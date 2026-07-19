# build_geometry_model.py — emit the KHANARY geometry model version folder.
#
# Packages the geometry-op capability as a self-contained, versioned KHANARY model:
# the KNU glyph streams, both co-equal backends' kernels (emitted from the REAL lowering,
# not hand-copied), the birdsong mesh data payload, and the manifest/verification record.
#
# Run: python tools/build_geometry_model.py
import os, sys, json, struct, hashlib, shutil
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                                   # so `tools.*` package imports resolve
sys.path.insert(0, os.path.join(_ROOT, "tools"))            # so `from stb import ...` resolves

from tools.khlnary_encoder import encode_knu, decode_knu, GLYPH_IDS
from tools.khlnary_dx11 import lower_khlnary_to_hlsl
from tools.khlnary_webgpu import lower_khlnary_to_wgsl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "0.3.0"
MODEL_DIR = os.path.join(ROOT, "models", f"khanary-geometry-v{VERSION}")
STB_SRC = r"C:\ffmpeg\bin\brain2\khanary_brain.stb"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def knu_record(glyph):
    word = encode_knu(glyph, payload=0)
    d = decode_knu(word)
    return {"glyph": glyph, "word": word, "word_hex": f"0x{word:08x}", "decoded": d}


def main():
    os.makedirs(os.path.join(MODEL_DIR, "kernels"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "knu"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "data"), exist_ok=True)

    ops = {
        "vertex_transform": [encode_knu("G_VERTEX_TRANSFORM", payload=0)],
        "vertex_skin":      [encode_knu("G_VERTEX_SKIN", payload=0)],
    }

    # --- kernels: emit from the REAL lowering so the folder mirrors runtime output ---
    for name, knus in ops.items():
        open(os.path.join(MODEL_DIR, "kernels", f"{name}.hlsl"), "w").write(lower_khlnary_to_hlsl(knus, {}))
        open(os.path.join(MODEL_DIR, "kernels", f"{name}.wgsl"), "w").write(lower_khlnary_to_wgsl(knus, {}))

    # --- KNU streams (the glyph words that select each kernel) ---
    for name in ("vertex_transform", "vertex_skin"):
        glyph = "G_VERTEX_TRANSFORM" if name == "vertex_transform" else "G_VERTEX_SKIN"
        json.dump({"op": name, "stream": [knu_record(glyph)]},
                  open(os.path.join(MODEL_DIR, "knu", f"{name}.knu.json"), "w", encoding="utf-8"), indent=2)

    # --- data payload: birdsong mesh (brain2 -> .stb), copied self-contained + hashed ---
    data_meta = {"present": False}
    if os.path.exists(STB_SRC):
        dst = os.path.join(MODEL_DIR, "data", "birdsong_mesh.stb")
        shutil.copyfile(STB_SRC, dst)
        magic, ver, nodes, edges, csr, K = (None,) * 6
        try:
            from stb import read_stb
            t = read_stb(dst)
            nodes = int(t[0]["dims"][0])
        except Exception:
            pass
        data_meta = {"present": True, "file": "data/birdsong_mesh.stb",
                     "sha256": sha256(dst), "bytes": os.path.getsize(dst),
                     "nodes": nodes, "source": "brain2 canary-song spectrogram graph",
                     "tensors": {"0": "node time (SVG x)", "1": "node freq (SVG y)",
                                 "2": "node energy", "3": "Delaunay edges", "4": "CSR index",
                                 "5": "CSR neighbours", "6": "geometric MoE expert id"}}

    knu_ver = decode_knu(ops["vertex_transform"][0])["ver"]  # actual VER emitted by encode_knu
    manifest = {
        "name": "khanary-geometry",
        "version": VERSION,
        "knu_profile": "KHΛ-2-DENSE-32 (v0.1 draft, tools/khlnary_encoder.py)",
        "knu_ver_field": knu_ver,
        "version_note": (f"geometry ops are additive GLYPH_ID entries; the emitted KNU words carry "
                         f"VER={knu_ver} (encoder profile). Model version {VERSION} is a CAPABILITY "
                         f"release, NOT a KNU wire-format bump. Do not conflate model version with "
                         f"KNU VER. (The separate khlnary_compiler.py module tags its module words 0x2.)"),
        "glyphs": {g: GLYPH_IDS[g] for g in ("G_VERTEX_TRANSFORM", "G_VERTEX_SKIN")},
        "ops": {
            "vertex_transform": {"glyph": "G_VERTEX_TRANSFORM",
                                 "desc": "apply 4x4 to each vertex (tight 12B stride)",
                                 "kernels": ["kernels/vertex_transform.hlsl", "kernels/vertex_transform.wgsl"],
                                 "knu": "knu/vertex_transform.knu.json"},
            "vertex_skin": {"glyph": "G_VERTEX_SKIN",
                            "desc": "weighted joint skinning, position + normal (tight 24B out stride)",
                            "kernels": ["kernels/vertex_skin.hlsl", "kernels/vertex_skin.wgsl"],
                            "knu": "knu/vertex_skin.knu.json"},
        },
        "backends": {
            "d3d11_cs_5_0": {
                "status": "hardware-verified",
                "device": "Intel HD Graphics 4600, D3D11 feature level 11_1 (0xb100)",
                "evidence": ["vertex_transform: 256 verts, max abs err 0.00e+00 vs CPU",
                             "vertex_skin: 128 verts pos+normal, max abs err 0.00e+00 vs CPU",
                             "birdsong_mesh: 30628 real verts, max abs err 0.00e+00 vs CPU"],
                "harnesses": ["scratch/xform_run.cpp", "scratch/skin_run.cpp", "scratch/brain_xform_run.cpp"],
            },
            "webgpu_wgsl": {
                "status": "structural-parity",
                "note": ("emitted by the same glyph-driven lowering as HLSL; NOT executed on this "
                         "rig (Dawn/WebGPU blocklisted on HD 4600 — the reason the D3D11 backend "
                         "exists). Runnable where WebGPU is available."),
            },
        },
        "coverage_caveats": [
            "skinning/mesh verification uses diagonal (uniform-scale) matrices + axis-aligned "
            "normals: proves dispatch + buffer plumbing + affine blend bit-exact and discriminates "
            "column-major (translation applied), but does NOT exercise rotation / non-uniform scale, "
            "so the normal path mul((float3x3)m, nrm) (no inverse-transpose) is untested there.",
            "StructuredBuffer<float4x4> is column-major (no row_major qualifier); host uploads and "
            "CPU reference match that convention.",
        ],
        "data": data_meta,
        "provenance": {
            "chain": "audio -> spectrogram ridges -> Delaunay graph -> .stb (SVG-Tensor) -> "
                     "KNU glyph stream -> backend lowering -> GPU dispatch",
            "encoder": "tools/khlnary_encoder.py", "backends": ["tools/khlnary_dx11.py", "tools/khlnary_webgpu.py"],
            "generator": "tools/build_geometry_model.py",
        },
    }
    json.dump(manifest, open(os.path.join(MODEL_DIR, "MODEL.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    readme = f"""# KHΛNARY geometry model — v{VERSION}

A versioned KHΛNARY **model**: geometry operations expressed as KNU glyph streams and lowered
to two co-equal backends. Geometry is a first-class KHΛNARY opcode — the KNU word selects the
kernel (see `MODEL.json` -> `ops`).

## What this version adds
- Glyphs `G_VERTEX_TRANSFORM` (0x{GLYPH_IDS['G_VERTEX_TRANSFORM']:02x}) and
  `G_VERTEX_SKIN` (0x{GLYPH_IDS['G_VERTEX_SKIN']:02x}) — additive within the `KHΛ-2-DENSE-32`
  v0.2 wire format (KNU VER stays 0x2; this is a capability release).
- Glyph-driven kernel selection in **both** `lower_khlnary_to_hlsl` and `lower_khlnary_to_wgsl`.

## Backends
- **D3D11 `cs_5_0`** — hardware-verified on Intel HD 4600 (FL 11_1). Transform, skinning, and the
  real 30628-vertex birdsong mesh all bit-exact vs CPU (`max abs err 0.00e+00`).
- **WebGPU WGSL** — structural parity (same glyph-driven lowering); not executed on this rig
  (WebGPU blocklisted here).

## Data
`data/birdsong_mesh.stb` — the brain2 canary-song graph as an SVG-Tensor `.stb` (nodes = spectrogram
ridge points). This is the mesh transformed bit-exact on the iGPU in the verification harnesses.

## Reproduce
```
python tools/build_geometry_model.py     # regenerate this folder
python -m pytest tests/ -q               # glyph-selection + parity tests
```
See `MODEL.json` -> `coverage_caveats` for honest scope.
"""
    open(os.path.join(MODEL_DIR, "README.md"), "w", encoding="utf-8").write(readme)
    print(f"wrote {os.path.relpath(MODEL_DIR, ROOT)}")
    for dp, _, fs in os.walk(MODEL_DIR):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            print(f"  {os.path.relpath(p, MODEL_DIR):42} {os.path.getsize(p):>10} B")


if __name__ == "__main__":
    main()
