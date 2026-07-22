# build_semantic_runtime_model.py — emit the KHANARY semantic-runtime proof-ladder model (v0.1.0).
#
# Packages the FieldExecutionEngine semantic-runtime proof track (S#001 -> S#002c-A). Where
# khanary-kxml (v0.5.0) registers the tool/node CONTRACTS but "does NOT itself execute them", this
# package is about the RUNTIME that executes: it certifies that the live engine emits trajectories
# conforming to the KXML record algebra, and it honestly characterizes what excites each record axis.
#
# THIS IS NOT A HARDWARE-VERIFIED CAPABILITY RELEASE like gpt2/geometry/gpu-resident. It is an
# INSTRUMENTED PROOF-LADDER: the headline S#002c finding is a decisive NEGATIVE (input does not
# excite legality); the S#002c-A "fitness" is a hash of the response bytes, not a learned signal;
# the single VIOLATION came from an artificial entropy-accumulation stress run; and the model bridge
# lives in a gitignored COPIED header, not the kernel .cpp. honest_scope states all of this plainly.
import os, json, shutil, hashlib

VERSION = "0.1.0"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models", f"khanary-semantic-runtime-v{VERSION}")
PROOF = os.path.join(_ROOT, "proof")

# (source proof dir, evidence files to vendor)
STAGES = [
    ("semantic_corpus_s001",  ["S001_PASS.md", "semantic-trajectories.jsonl", "SHA256SUMS"]),
    ("semantic_corpus_s002",  ["S002_PASS.md", "semantic-trajectories-live.jsonl", "SHA256SUMS"]),
    ("semantic_corpus_s002b", ["SHA256SUMS"]),
    ("semantic_corpus_s002c", ["S002C_PASS.md", "S002C_A_PASS.md", "field_execution_engine.bridge.h",
                               "semantic-controlled-corpus.jsonl", "SHA256SUMS"]),
]
RECORD_ALGEBRA_DOC = os.path.join("docs", "kxml-semantic-record-algebra.md")


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
    for src, files in STAGES:
        dst = os.path.join(evidence_dir, src)
        os.makedirs(dst, exist_ok=True)
        for fn in files:
            sp = os.path.join(PROOF, src, fn)
            if not os.path.exists(sp):
                raise SystemExit(f"[gen-error] missing frozen evidence: {sp}")
            shutil.copy2(sp, os.path.join(dst, fn))
            vendored[f"proof/{src}/{fn}"] = sha256(sp)
    # record-algebra reference doc (S#003a), if present
    rad = os.path.join(_ROOT, RECORD_ALGEBRA_DOC)
    if os.path.exists(rad):
        shutil.copy2(rad, os.path.join(MODEL_DIR, "kxml-semantic-record-algebra.md"))
        vendored[RECORD_ALGEBRA_DOC] = sha256(rad)

    model = {
        "name": "khanary-semantic-runtime",
        "version": VERSION,
        "kind": "semantic-runtime proof ladder (FieldExecutionEngine) — INSTRUMENTED, not HW-verified",
        "summary": "The runtime counterpart to khanary-kxml: khanary-kxml registers the tool/node "
                   "contracts, this certifies that the live FieldExecutionEngine EXECUTES trajectories "
                   "conforming to the KXML record algebra, and honestly maps which knob excites each "
                   "record axis (routing / fitness / field / legality-VIOLATION).",
        "record_algebra": {
            "classes": ["TRANSITION", "DELTA", "FIELD", "INVARIANT", "EQUIVALENCE", "VIOLATION"],
            "doc": "kxml-semantic-record-algebra.md (S#003a)",
            "legality_note": "legality is a PROPERTY of a record (legal/illegal), not a separate structural "
                             "world; provenance is required on every record."
        },
        "ladder": [
            {"stage": "S#001", "claim": "kernel is a teacher/oracle (static corpus), read-only",
             "result": "66 records from 6 phases + 12 folds; all 5 classes; 20 legal / 28 illegal "
                       "(positives AND negatives); kernel sources UNMUTATED",
             "proof": "proof/semantic_corpus_s001/"},
            {"stage": "S#002", "claim": "live engine conforms to the S#003a algebra",
             "result": "prebuilt verify_asx.exe observed (copied out, not mutated); 10 live ticks -> 60 "
                       "records (TRANSITION 40 / DELTA 10 / FIELD 10), all conform; all legal",
             "proof": "proof/semantic_corpus_s002/"},
            {"stage": "S#002b", "claim": "read-only coverage is insufficient",
             "result": "compile_ir over 14 queries is DEGENERATE (routing roles=2, saturates at query 1) "
                       "-> record COUNT is the wrong stopping metric; use vocabulary CONVERGENCE",
             "proof": "proof/semantic_corpus_s002b/"},
            {"stage": "S#002c", "claim": "controlled additive excitation — DECISIVE NEGATIVE",
             "result": "additive harness drives the full FieldExecutionEngine with a 16-case coverage "
                       "matrix (incl. adversarial). Path diversity real (10 folds), but every case returns "
                       "Law E: LAWFUL and MoE: General -> legality/routing vocabulary is INPUT-INVARIANT. "
                       "The verdict is on the engine's INTERNAL mutation, not the query.",
             "proof": "proof/semantic_corpus_s002c/S002C_PASS.md"},
            {"stage": "S#002c-A", "claim": "implement the model bridge -> all axes excite",
             "result": "bridge implemented in a COPIED header (kernel .cpp untouched); it POSTs to a live "
                       "llama-server (gpt2.Q8_0.gguf). Then: (1) routing is KEYWORD-gated not dead "
                       "(code/refactor->CODING, create/new->FACTORY, else General) — the S#002c matrix "
                       "had a keyword coverage gap; (2) fitness now varies (was constant 0.941); (3) one "
                       "VIOLATION (Law B BOUNDARY_BREACH) emerged only from a 30-tick entropy-accumulation "
                       "stress run — violations are a STATE-TRAJECTORY property, not per-tick input.",
             "proof": "proof/semantic_corpus_s002c/S002C_A_PASS.md"}
        ],
        "axis_excitation": {
            "routing": "keyword-gated (General + CODING + FACTORY) — excited by matrix keywords",
            "fitness": "response-derived (~0.33..0.65), was constant stub 0.941 — see honest_scope #2",
            "field_coherence": "5 distinct values with the live bridge (was 1)",
            "legality_VIOLATION": "1 BOUNDARY_BREACH via accumulated-entropy stress; TENSOR_INSTABILITY + "
                                  "CAUSALITY_VIOLATION reachable via targeted conditions, NOT yet observed"
        },
        "legality_verifier": {
            "checks": ["Law B entropy threshold (m_max_entropy_threshold)",
                       "tensor NaN/Inf/oversize (m_max_mutation_delta)",
                       "gen-0 causality"],
            "note": "verify_mutation is a real 3-check verifier operating on the internal evolution step."
        },
        "honest_scope": [
            "INSTRUMENTED PROOF-LADDER, not a hardware-verified capability release. It does not certify a "
            "numerical kernel against a CPU reference the way gpt2/geometry/gpu-resident do; it certifies "
            "runtime CONFORMANCE + characterizes axis excitation.",
            "The S#002c-A 'fitness' is fitness = 0.30 + 0.69 * (hash(response_bytes) % 1000)/1000 — a HASH of "
            "the model response, a stimulus to move the evolver off a constant, NOT a learned or semantic "
            "score. Do not read it as model quality.",
            "The headline S#002c result is a decisive NEGATIVE: input does not excite the legality boundary. "
            "The single VIOLATION observed required an ARTIFICIAL 30-tick entropy-accumulation stress run; no "
            "organic input produced one, and 2 of the 3 verdict types remain UNOBSERVED.",
            "The implemented bridge lives in a gitignored COPIED header (desktop/semantic_engine/include/, "
            "frozen snapshot field_execution_engine.bridge.h); the kernel .cpp in .ASX.cpp is UNCHANGED. "
            "Several kxml tool builtins remain stubs (see khanary-kxml honest_scope).",
            "S#003b (schema freeze) is UNBLOCKED but NOT DONE: it needs all 3 violation verdict types covered "
            "and vocabulary saturation across diversified batches first."
        ],
        "provenance": {
            "proofs": "proof/semantic_corpus_s001..s002c (.ASX.cpp, frozen with per-stage SHA256SUMS)",
            "engine": "kxml-semantic-kernel/semantic_kernel_cpp FieldExecutionEngine + LegalityVerifier (.ASX.cpp, read-only)",
            "registry_sibling": "models/khanary-kxml-v0.5.0 (the contract layer this runtime executes)",
            "bridge_target": "llama-server on C:/Users/canna/.lmstudio/models/gpt2.Q8_0.gguf (/v1/chat/completions)"
        },
        "evidence_sha256": vendored,
        "generator": "tools/build_semantic_runtime_model.py"
    }

    with open(os.path.join(MODEL_DIR, "MODEL.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"[gen] wrote {MODEL_DIR}/MODEL.json  ({len(vendored)} evidence files vendored)")


if __name__ == "__main__":
    main()
