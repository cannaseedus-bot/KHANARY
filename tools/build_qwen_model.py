# build_qwen_model.py — install the Qwen-1.8B dual-quant artifacts from the E: ARCHIVE into the
# runtime KHANARY version folder, and emit MODEL.json.
#
# Workflow: E:\models\Qwen1.8B-quant\ is the build+archive of record (backup drive: no runtime
# access / speed-limited). The runtime copy lives on C: inside models/khanary-qwen1_8b-vX/. This
# script copies the payloads + manifests across, verifies sha256 against the archive, and writes
# MODEL.json. The large .kqz payloads are gitignored (manifests ARE tracked, like the gpt2 pkg .stb).
import os, json, shutil, hashlib

VERSION = "0.1.0"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = r"E:\models\Qwen1.8B-quant"
PKG = os.path.join(_ROOT, "models", f"khanary-qwen1_8b-v{VERSION}")
SRC_SAFETENSORS = r"C:\Users\canna\.lmstudio\models\Qwen-1_8B-Chat-f16\model.safetensors"

ARTIFACTS = [
    ("q8", "qwen1_8b.q8.kqz", "qwen1_8b.q8.manifest.json", "per-output-channel symmetric INT8",
     "c81153cd798bcf9b4a4a364fdba7dd9e163256459a3b0686689d44156ae1b6c8", "~0.85% normRMSE / ~41 dB (near-lossless)"),
    ("q4", "qwen1_8b.q4.kqz", "qwen1_8b.q4.manifest.json", "group-64 symmetric 4-bit (2 nibbles/byte)",
     "bcf1523dd47cdcf85b89296e516b3231506bcb8b7ab83a4e2f831033165f2d3b", "~10% normRMSE / ~20 dB (usable, lossy)"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(install_payloads=True):
    os.makedirs(PKG, exist_ok=True)
    entries = []
    for tag, kqz, man, scheme, want_sha, fidelity in ARTIFACTS:
        # manifests are always installed (small, tracked)
        shutil.copy2(os.path.join(ARCHIVE, man), os.path.join(PKG, man))
        dst_kqz = os.path.join(PKG, kqz)
        if install_payloads:
            if not os.path.exists(dst_kqz) or os.path.getsize(dst_kqz) != os.path.getsize(os.path.join(ARCHIVE, kqz)):
                shutil.copy2(os.path.join(ARCHIVE, kqz), dst_kqz)
            got = sha256(dst_kqz)
            if got != want_sha:
                raise SystemExit(f"[gen-error] {kqz} sha256 {got} != archived {want_sha}")
            size = os.path.getsize(dst_kqz)
        else:
            size = os.path.getsize(os.path.join(ARCHIVE, kqz))
        man_j = json.load(open(os.path.join(PKG, man)))
        entries.append({"scheme": tag, "payload": kqz, "manifest": man, "quant": scheme,
                        "bytes": size, "gib": round(size/1024**3, 3), "sha256": want_sha,
                        "tensors": len(man_j["tensors"]), "fidelity": fidelity,
                        "payload_gitignored": True})

    model = {
        "name": "khanary-qwen1_8b",
        "version": VERSION,
        "kind": "quantized base-model weights (dual-quant, resident-tiered) — weights only, no forward path",
        "summary": "One Qwen-1.8B, two quantizations of the SAME tensors: a Q4 resident base + a Q8 tier "
                   "escalated per-tensor into headroom. Real weight bytes for the HD 4600 residency / "
                   "hot-swap design; not a runnable model on this stack.",
        "source": {"safetensors": SRC_SAFETENSORS, "params": "1.837B", "tensors": 195,
                   "arch": "Qwen (24L, hidden 2048, MHA, RMSNorm/SwiGLU/RoPE, vocab 151936)", "dtype": "F16"},
        "artifacts": entries,
        "residency": {
            "measured_ceiling": "~1.75 GiB stable / 2.0 GiB hard wall on the HD 4600 (proof/gpu_resident_ceiling_v1)",
            "q8_fits": "1.712 GiB < 1.75 GiB stable ceiling — a property of THIS lean per-channel scheme; "
                       "standard GGUF Q8_0 (~1.82 GiB) would exceed it",
            "q4_headroom": "0.909 GiB base leaves ~0.9 GiB headroom",
            "swap_design": "Q8/Q4 manifests are tensor-aligned (identical names/order); keep Q4 resident, "
                           "hot-swap INDIVIDUAL tensors up to Q8 into headroom (per-tensor precision "
                           "escalation) — never hold both full models (2.6 GiB > the 2.0 GiB wall)"
        },
        "storage": {
            "archive_of_record": ARCHIVE + " (E: backup drive — build + archive; no runtime access / speed-limited)",
            "runtime_copy": f"models/khanary-qwen1_8b-v{VERSION}/ (C: — hot/runtime)",
            "git": "manifests tracked; *.kqz payloads gitignored (re-install from the E: archive via this generator)"
        },
        "honest_scope": [
            "WEIGHTS ONLY, not a runnable model — this stack has no Qwen forward path (the proven #001 "
            "DirectML driver is GPT-2-only: LayerNorm/GELU/50257, not Qwen RMSNorm/SwiGLU/RoPE/151936; "
            "the vendored llama is CPU-only, off the GPU budget).",
            "\"Q8 fits resident\" holds for this lean per-channel INT8 scheme (1.712 GiB), not universally.",
            "Fidelity verified at dequant-error + container round-trip (tools/verify_quant.py), NOT "
            "end-to-end perplexity.",
            "Q4 is genuinely lossy (~10% normRMSE / 20 dB) — it is the fast bulk-work tier; escalate "
            "tensors to Q8 (~41 dB) when precision matters."
        ],
        "provenance": {
            "quantizer": "tools/quantize_qwen.py",
            "verifier": "tools/verify_quant.py",
            "proof": "proof/qwen_quant_v1/ (frozen provenance + fidelity)",
            "container_style": "smgm-16 DDS-fold offset-manifest (E:\\models\\GPT2\\smgm-16)"
        },
        "generator": "tools/build_qwen_model.py"
    }
    with open(os.path.join(PKG, "MODEL.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"[gen] wrote {PKG}\\MODEL.json  ({len(entries)} artifacts, payloads {'installed' if install_payloads else 'referenced'})")


if __name__ == "__main__":
    import sys
    main(install_payloads="--no-payloads" not in sys.argv)
