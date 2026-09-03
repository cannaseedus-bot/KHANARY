"""
CUBE-1 SemanticCubeMicronaut — bots.py
K-CUBE geometry engine: 6-face SHM tensor reads/writes + SH projection → dominant µ.
Backend: native_glyph_engine_cli.exe + direct SHM via subprocess.
Fold: ⟁CUBE_FOLD⟁
"""

import sys
import json
import time
import math
import subprocess
import importlib.util
from pathlib import Path

_MICRONAUT_ID   = "CUBE-1"
_MICRONAUT_NAME = "SemanticCubeMicronaut"
_MICRONAUT_FOLD = "⟁CUBE_FOLD⟁"

# ---------------------------------------------------------------------------
# ELIZA-1 — metacognitive brain (Chen fold)
# ---------------------------------------------------------------------------

_ELIZA = Path(__file__).parent.parent / "eliza-micronaut" / "bots.py"
_eliza = None
try:
    _espec = importlib.util.spec_from_file_location("eliza_bots", str(_ELIZA))
    _emod  = importlib.util.module_from_spec(_espec)
    _espec.loader.exec_module(_emod)
    _eliza = _emod
except Exception:
    pass

def _eliza_intent(text: str) -> dict:
    return _eliza.intent(text) if _eliza else {}

def _eliza_question(context: str) -> dict:
    return _eliza.question(context) if _eliza else {}

def _eliza_plan(context: str, user_intent: str = None) -> dict:
    return _eliza.plan(context, user_intent) if _eliza else {}

_GLYPH_CLI = Path(__file__).parent / "native_glyph_engine_cli.exe"
_XJSON_EXE = Path(__file__).parent / "micronaut_xjson.exe"

# Face normals on a unit sphere (one per K-CUBE face, matching Phase angles)
_FACE_NORMALS = {
    "Phi":        (1.0,  0.0,  0.0),   # Pop   φ=0
    "Fold":       (0.5,  0.866, 0.0),  # Wo    φ=π/3
    "Gram":       (-0.5, 0.866, 0.0),  # Yax   φ=2π/3
    "Geodesic":   (-1.0, 0.0,  0.0),   # Sek   φ=π
    "Projection": (-0.5,-0.866, 0.0),  # Chen  φ=4π/3
    "Entropy":    (0.5, -0.866, 0.0),  # Xul   φ=5π/3
}

# µ name → primary face (for dominant µ lookup)
_MU_FACE = {
    "pop":  "Phi", "wo": "Fold", "yax": "Gram",
    "sek":  "Geodesic", "chen": "Projection", "xul": "Entropy",
}

# In-process face state (fallback when SHM not available)
_FACES: dict[str, float] = {k: 0.0 for k in _FACE_NORMALS}

# ---------------------------------------------------------------------------

def _run_cli(args: list[str]) -> dict:
    if not _GLYPH_CLI.exists():
        return {"error": "native_glyph_engine_cli.exe not found"}
    try:
        result = subprocess.run(
            [str(_GLYPH_CLI)] + args, capture_output=True, text=True, timeout=5
        )
        out = result.stdout.strip()
        return json.loads(out) if out.startswith("{") else {"raw": out, "rc": result.returncode}
    except Exception as exc:
        return {"error": str(exc)}

# ---------------------------------------------------------------------------

def health_check() -> dict:
    cli_ok = _GLYPH_CLI.exists()
    return {
        "status": "ok" if cli_ok else "degraded",
        "micronaut": _MICRONAUT_ID,
        "glyph_cli": cli_ok,
        "shm": "Local\\KuhulGeometricState",
        "faces": list(_FACE_NORMALS.keys()),
        "eliza": "wired" if _eliza is not None else "absent",
    }


def get_face(face: str) -> dict:
    if face not in _FACE_NORMALS:
        return {"error": f"Unknown face '{face}'", "valid": list(_FACE_NORMALS.keys())}
    r = _run_cli(["--get-face", face])
    if "value" in r:
        _FACES[face] = float(r["value"])
    return {"face": face, "value": _FACES.get(face, 0.0)}


def set_face(face: str, value: float) -> dict:
    if face not in _FACE_NORMALS:
        return {"error": f"Unknown face '{face}'"}
    value = max(0.0, min(1.0, float(value)))
    _FACES[face] = value
    r = _run_cli(["--set-face", face, str(value)])
    return {"face": face, "value": value, "shm": r}


def project() -> dict:
    """
    Run SH softmax projection over current face activations → dominant µ.
    ELIZA interprets the dominant face as a cognitive state and suggests next steps.
    """
    t0 = time.perf_counter()
    activations = {k: _FACES.get(k, 0.0) for k in _FACE_NORMALS}
    temperature = 1.0
    total = sum(math.exp(v / temperature) for v in activations.values())
    weights = {k: math.exp(v / temperature) / total for k, v in activations.items()}
    dominant_face = max(weights, key=weights.__getitem__)
    # face → fold µ name
    face_to_mu = {v: k for k, v in _MU_FACE.items()}
    dominant_mu = face_to_mu.get(dominant_face, dominant_face.lower())
    latency = (time.perf_counter() - t0) * 1000
    result = {
        "micronaut": _MICRONAUT_ID,
        "fold": _MICRONAUT_FOLD,
        "activations": activations,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "dominant_face": dominant_face,
        "dominant_mu": dominant_mu,
        "confidence": round(weights[dominant_face], 4),
        "latency_ms": round(latency, 2),
    }
    # ELIZA: interpret dominant cognitive state → next steps
    ep = _eliza_plan(
        f"K-CUBE dominant face is {dominant_face} ({dominant_mu} phase) "
        f"with confidence {round(weights[dominant_face], 4)}"
    )
    if ep:
        result["eliza_interpretation"] = {
            "phase":       dominant_mu,
            "next":        ep.get("next", [])[:2],
            "alice_domain": ep.get("semantic", {}).get("alice_domain", "unknown"),
        }
    return result


def snapshot() -> dict:
    return {"micronaut": _MICRONAUT_ID, "faces": dict(_FACES)}


def dispatch(task: str, payload: dict | None = None) -> str:
    payload = payload or {}
    t = task.lower()
    if t in ("project", "dominant", "dominant_mu"):
        return json.dumps(project())
    if t in ("get", "get_face"):
        return json.dumps(get_face(payload.get("face", "")))
    if t in ("set", "set_face"):
        return json.dumps(set_face(payload.get("face", ""), payload.get("value", 0.0)))
    if t in ("snapshot", "state"):
        return json.dumps(snapshot())
    if t == "health":
        return json.dumps(health_check())
    return json.dumps({"error": f"Unknown task '{task}'",
                       "supported": ["project", "get_face", "set_face", "snapshot", "health"],
                       "micronaut": _MICRONAUT_ID})


if __name__ == "__main__":
    print(f"[{_MICRONAUT_ID}] {_MICRONAUT_NAME} smoke test")
    print(f"[1] health: {health_check()}")
    set_face("Phi", 0.9)
    set_face("Gram", 0.3)
    set_face("Entropy", 0.1)
    r = project()
    print(f"[2] project: dominant={r['dominant_mu']} confidence={r['confidence']}")
    print(f"[3] snapshot: {snapshot()}")
    print(f"[{_MICRONAUT_ID}] done")
