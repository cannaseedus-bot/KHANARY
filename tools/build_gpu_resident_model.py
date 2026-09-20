# build_gpu_resident_model.py — emit the KHANARY GPU-resident proof-ladder model (v0.1.0).
#
# Packages the KGRC GPU proof ladder (#001-#004-B1) as a versioned KHANARY artifact. Unlike the
# other three model packages this one certifies a RESIDENCY / EXECUTION capability rather than a
# single glyph kernel: whole-model residence, native KV state transition, closed-trajectory
# generation, and fixed-op binding amortization — each HARDWARE-VERIFIED on the Intel HD 4600
# (D3D11 FL 11_1 / DirectML on D3D12). It vendors the frozen evidence (PASS/RESULT docs, contracts,
# SHA256SUMS) out of proof/ — it does not re-run the proofs (those artifacts are frozen references).
import os, json, shutil, hashlib

VERSION = "0.1.0"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models", f"khanary-gpu-resident-v{VERSION}")
PROOF = os.path.join(_ROOT, "proof")

# (rung, source proof dir, evidence files to vendor)
RUNGS = [
    ("#001",    "gpt2_hd4600_resident_v1",   ["RESULT.md", "model_contract.json", "SHA256SUMS"]),
    ("#002",    "dml_mha_kv_cache_v1",        ["KV_CACHE_PASS.md", "field_contract.json", "SHA256SUMS"]),
    ("#003",    "gpt2_resident_generation_v1",["GENERATION_PASS.md", "generation_contract.json", "SHA256SUMS"]),
    ("#004-A",  "resident_generation_004a",   ["004A_PASS.md", "SHA256SUMS"]),
    ("#004-B1", "resident_generation_004b1",  ["B1_PASS.md", "SHA256SUMS"]),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def main():
    evidence_dir = os.path.join(MODEL_DIR, "proof")
    os.makedirs(evidence_dir, exist_ok=True)
    vendored = {}
    for rung, src, files in RUNGS:
        dst = os.path.join(evidence_dir, src)
        os.makedirs(dst, exist_ok=True)
        for fn in files:
            sp = os.path.join(PROOF, src, fn)
            if not os.path.exists(sp):
                raise SystemExit(f"[gen-error] missing frozen evidence: {sp}")
            shutil.copy2(sp, os.path.join(dst, fn))
            vendored[f"proof/{src}/{fn}"] = sha256(sp)

    model = {
        "name": "khanary-gpu-resident",
        "version": VERSION,
        "kind": "GPU residency + execution proof ladder (KGRC)",
        "knu_profile": "KHΛ-2-DENSE-32",
        "summary": "The K'UHUL GPU Resource Contract (KGRC) proof ladder: each rung certifies exactly one "
                   "residency/execution property of a whole model living as persistent GPU state on the "
                   "Intel HD 4600, hardware-verified against a CPU reference. This packages the frozen "
                   "evidence, not a re-runnable kernel.",
        "target": {
            "device": "Intel(R) HD Graphics 4600",
            "d3d12_feature_level": "11_1 (0xb100)",
            "dml_feature_level": "0x6200 (#001) / device reports 0x6400 (#004-B1)",
            "api": "DirectML on D3D12 (11_x bridge); D3D11 cs_5_0 is the primary compute tier"
        },
        "ladder": [
            {"rung": "#001", "claim": "Resident computation is correct",
             "result": "whole gpt2 (124M, 12L) forward, weights resident on-device; logits scale-norm "
                       "1.92e-06 vs CPU erf-gelu; next-token argmax gpu==cpu==42447",
             "proof": "proof/gpt2_hd4600_resident_v1/",
             "budget": {"weights_mb": 500, "dispatches_per_forward": 146, "seq_len": 8}},
            {"rung": "#002", "claim": "Resident state transition is correct",
             "result": "native DirectML MHA KV-cache decode step; LAW1 growth / LAW2 preserve (maxabs 0) / "
                       "LAW3 append (maxabs 0) / LAW4 compute 8.08e-08 — all four independently PASS",
             "proof": "proof/dml_mha_kv_cache_v1/"},
            {"rung": "#003", "claim": "Resident trajectory composes",
             "result": "14-tick autoregressive decode (8 prompt + 6 generated) through the closed "
                       "Xul->Pop cycle; every tick GPU==CPU argmax MATCH; final per-layer KV vs CPU "
                       "maxabs 5.36e-06; deterministic argmax, no sampling, no ggml",
             "proof": "proof/gpt2_resident_generation_v1/"},
            {"rung": "#004-A", "claim": "Fixed execution state is reusable (amortization)",
             "result": "binding-table creations 146->12 per token (134 fixed/session + 12 dynamic MHA/tok); "
                       "record/bind 32.38->9.81 ms/tok (3.30x); total 124.2->94.0 ms/tok (1.32x); "
                       "trajectory(reuse) == frozen #003, 14/14 ticks argmax-exact (oracle preserved)",
             "proof": "proof/resident_generation_004a/"},
            {"rung": "#004-B1", "claim": "Capacity != extent (semantics + backend conformance)",
             "result": "physical capacity C=6 past with logical extent P=3, unused slots masked "
                       "(RelativePositionBias -1e9) -> output scale-norm 8.08e-08 vs exact-extent-P "
                       "reference; prefix preserved 0.0. DIAGNOSIS: native fixed-capacity op "
                       "DML_OPERATOR_MULTIHEAD_ATTENTION1 + PastSequenceLengths is UNAVAILABLE on this "
                       "iGPU (E_INVALIDARG at CreateOperator across all desc variants; device DML FL "
                       "0x6400 nominally >= required 0x6300, but the op is absent). Base MHA appends at "
                       "the physical end, not the logical extent.",
             "proof": "proof/resident_generation_004b1/"}
        ],
        "kgrc_concepts": {
            "STATIC": "STATIC DATA (weights) + STATIC EXECUTION STATE (compiled ops, persistent "
                      "descriptors, invariant bindings) — construct once, preserve across ticks (#004-A)",
            "GROWING": "PHYSICAL capacity C (fixed alloc) + SEMANTIC extent P<=C (valid region computes) (#004-B1)",
            "CONTINUITY": "InputState(t+1) == OutputState(t) across the closed Xul->Pop cycle (#003)",
            "append_law": "F(t+1) = Preserve(F(t)) (+) Delta(t) (#002)",
            "backend_conformance": "Semantic extent != physical realization: the KGRC model (capacity "
                                   "fixed, extent grows) is realized differently per backend and may not "
                                   "be realized at all — DirectML on this rig appends at the physical end "
                                   "and lacks the native fixed-capacity op. A conformance fact for the "
                                   "runtime, not a defect."
        },
        "backends": {
            "d3d11_cs_5_0": {"status": "primary compute tier (measured; FL 11_1 full)"},
            "directml_d3d12_11x": {"status": "hardware-verified for this ladder",
                                   "note": "DirectML runs at D3D12 FL 11_1 as an 11_x bridge; full D3D12 "
                                           "(12_0+) is UNSUPPORTED on the HD 4600 (DXGI_ERROR_UNSUPPORTED)."}
        },
        "evidence_sha256": vendored,
        "honest_scope": [
            "This is a PROOF-LADDER capability release: it certifies residency/state/trajectory/amortization "
            "PROPERTIES verified vs a CPU reference on real gpt2 weights — it is not a packaged runtime or a "
            "single dispatchable glyph kernel. The frozen harness sources live under proof/ in the repo; only "
            "the evidence docs + contracts + SHA256SUMS are vendored here.",
            "#004-B1 is a NEGATIVE/conformance result on the native fixed-capacity path: MHA1 is unavailable "
            "on this DirectML/HD 4600. The capacity-vs-extent SEMANTICS pass (8.08e-08), but a resident "
            "fixed-capacity KV runtime needs a manual cache + mask (parked as chapter #005, not built).",
            "Numbers are on this specific 2015 iGPU (frozen 2015 driver, WDDM 2.0). Discrete-GPU migration "
            "does not map 1:1 (integrated = system memory). Timings (#004-A) are host-side, this build only."
        ],
        "provenance": {
            "proofs": "proof/ (.ASX.cpp GPU proof ladder, frozen with per-rung SHA256SUMS)",
            "ladder_doc": "docs/GPU_PROOF_LADDER.md",
            "engine_headers": "xvm-d3d12/src (.KUHUL_V2) + DirectML redist (scratch/dml/)",
            "weights": "real gpt2 124M (scratch/gpt2_model.stb)"
        },
        "generator": "tools/build_gpu_resident_model.py"
    }

    with open(os.path.join(MODEL_DIR, "MODEL.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"[gen] wrote {MODEL_DIR}/MODEL.json  ({len(vendored)} evidence files vendored)")


if __name__ == "__main__":
    main()
