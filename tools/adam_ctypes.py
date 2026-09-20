"""adam_ctypes.py — Python ctypes wrapper for Adam.dll

Usage:
    from tools.adam_ctypes import AdamOptimizer, AdamFoldArcs
    opt = AdamOptimizer(weights, lr=1e-4)
    opt.step(grads)

Smoke test:
    python tools/adam_ctypes.py --smoke
"""
from __future__ import annotations
import ctypes
import os
import sys
import math
from pathlib import Path
import numpy as np

# --- DLL location -----------------------------------------------------------
_DEFAULT_DLL = Path(__file__).parent.parent / "versions" / "khlc-v1.0.0" / "bin" / "Adam.dll"

def _load(path: Path | str | None = None) -> ctypes.CDLL:
    p = Path(path) if path else _DEFAULT_DLL
    if not p.exists():
        raise FileNotFoundError(f"Adam.dll not found at {p}\nRun native/adam/build_adam.bat first.")
    return ctypes.CDLL(str(p))

_lib: ctypes.CDLL | None = None

def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib

# --- C struct mirrors -------------------------------------------------------
FOLD_COUNT = 6

class _AdamState(ctypes.Structure):
    _fields_ = [
        ("w",         ctypes.POINTER(ctypes.c_float)),
        ("m",         ctypes.POINTER(ctypes.c_float)),
        ("v",         ctypes.POINTER(ctypes.c_float)),
        ("g",         ctypes.POINTER(ctypes.c_float)),
        ("n",         ctypes.c_uint32),
        ("step",      ctypes.c_uint32),
        ("lr",        ctypes.c_float),
        ("beta1",     ctypes.c_float),
        ("beta2",     ctypes.c_float),
        ("eps",       ctypes.c_float),
        ("grad_clip", ctypes.c_float),
    ]

_Float6x6 = (ctypes.c_float * FOLD_COUNT) * FOLD_COUNT

class _AdamFoldArcs(ctypes.Structure):
    _fields_ = [
        ("arc_w",     _Float6x6),
        ("arc_m",     _Float6x6),
        ("arc_v",     _Float6x6),
        ("arc_g",     _Float6x6),
        ("angles",    ctypes.c_float * FOLD_COUNT),
        ("step",      ctypes.c_uint32),
        ("lr",        ctypes.c_float),
        ("beta1",     ctypes.c_float),
        ("beta2",     ctypes.c_float),
        ("eps",       ctypes.c_float),
        ("grad_clip", ctypes.c_float),
    ]

# --- Prototype binding helper -----------------------------------------------
def _bind(lib: ctypes.CDLL) -> None:
    lib.adam_init.restype      = ctypes.c_int
    lib.adam_init.argtypes     = [
        ctypes.POINTER(_AdamState), ctypes.c_uint32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,
    ]
    lib.adam_step.restype      = ctypes.c_int
    lib.adam_step.argtypes     = [ctypes.POINTER(_AdamState)]
    lib.adam_clip_grads.restype  = ctypes.c_int
    lib.adam_clip_grads.argtypes = [ctypes.POINTER(_AdamState)]
    lib.adam_free.restype      = None
    lib.adam_free.argtypes     = [ctypes.POINTER(_AdamState)]

    lib.adam_fold_arcs_init.restype    = ctypes.c_int
    lib.adam_fold_arcs_init.argtypes   = [ctypes.POINTER(_AdamFoldArcs), ctypes.c_float]
    lib.adam_fold_arcs_step.restype    = ctypes.c_int
    lib.adam_fold_arcs_step.argtypes   = [ctypes.POINTER(_AdamFoldArcs), ctypes.POINTER(_Float6x6)]
    lib.adam_fold_arc_effective.restype  = ctypes.c_float
    lib.adam_fold_arc_effective.argtypes = [ctypes.POINTER(_AdamFoldArcs), ctypes.c_int, ctypes.c_int]
    lib.adam_fold_arcs_reset_moments.restype  = None
    lib.adam_fold_arcs_reset_moments.argtypes = [ctypes.POINTER(_AdamFoldArcs)]
    lib.adam_version.restype   = ctypes.c_char_p
    lib.adam_version.argtypes  = []

_bound = False
def _ensure_bound() -> ctypes.CDLL:
    global _bound
    lib = _get_lib()
    if not _bound:
        _bind(lib)
        _bound = True
    return lib

# --- High-level Python API --------------------------------------------------

class AdamOptimizer:
    """Wraps a single AdamState for one numpy float32 weight tensor."""

    def __init__(self, weights: np.ndarray, *,
                 lr: float = 1e-5,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 grad_clip: float = 1.0):
        self._lib  = _ensure_bound()
        self._w    = np.ascontiguousarray(weights, dtype=np.float32)
        self._s    = _AdamState()
        n          = self._w.size
        rc = self._lib.adam_init(ctypes.byref(self._s), n, lr, beta1, beta2, eps, grad_clip)
        if rc != 0:
            raise RuntimeError(f"adam_init failed: {rc}")
        # Set caller-owned pointers after init (init only allocates m/v)
        self._s.w  = self._w.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def step(self, grads: np.ndarray) -> None:
        g = np.ascontiguousarray(grads, dtype=np.float32)
        assert g.size == self._s.n, "grad/weight size mismatch"
        self._s.g = g.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self._lib.adam_step(ctypes.byref(self._s))
        if rc != 0:
            raise RuntimeError(f"adam_step failed: {rc}")

    @property
    def weights(self) -> np.ndarray:
        return self._w

    @property
    def step_count(self) -> int:
        return int(self._s.step)

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_s"):
            self._lib.adam_free(ctypes.byref(self._s))


class FoldArcOptimizer:
    """Wraps AdamFoldArcs — 6×6 arc weight matrix with Adam updates."""

    FOLD_NAMES = ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"]

    def __init__(self, lr: float = 1e-3):
        self._lib = _ensure_bound()
        self._fa  = _AdamFoldArcs()
        rc = self._lib.adam_fold_arcs_init(ctypes.byref(self._fa), lr)
        if rc != 0:
            raise RuntimeError(f"adam_fold_arcs_init failed: {rc}")

    def step(self, grad_matrix: np.ndarray) -> None:
        g = np.ascontiguousarray(grad_matrix, dtype=np.float32).reshape(FOLD_COUNT, FOLD_COUNT)
        g_c = (_Float6x6)(*[
            (ctypes.c_float * FOLD_COUNT)(*g[i].tolist()) for i in range(FOLD_COUNT)
        ])
        rc = self._lib.adam_fold_arcs_step(ctypes.byref(self._fa), ctypes.byref(g_c))
        if rc != 0:
            raise RuntimeError(f"fold_arcs_step failed: {rc}")

    def weights(self) -> np.ndarray:
        return np.array([[self._fa.arc_w[i][j]
                          for j in range(FOLD_COUNT)]
                         for i in range(FOLD_COUNT)], dtype=np.float32)

    def effective(self, src: int, dst: int) -> float:
        return float(self._lib.adam_fold_arc_effective(ctypes.byref(self._fa), src, dst))

    def effective_matrix(self) -> np.ndarray:
        return np.array([[self.effective(i, j)
                          for j in range(FOLD_COUNT)]
                         for i in range(FOLD_COUNT)], dtype=np.float32)

    def reset_moments(self) -> None:
        self._lib.adam_fold_arcs_reset_moments(ctypes.byref(self._fa))

    @property
    def step_count(self) -> int:
        return int(self._fa.step)

# --- Smoke test -------------------------------------------------------------

def _smoke():
    lib = _ensure_bound()
    print(f"Adam.dll version: {lib.adam_version().decode()}")

    # Single tensor
    w = np.full(16, 0.5, dtype=np.float32)
    opt = AdamOptimizer(w, lr=1e-3)
    for step in range(10):
        g = np.full(16, 0.01 * (step + 1), dtype=np.float32)
        opt.step(g)
    print(f"w[0] after 10 steps: {w[0]:.6f}  (expect ~0.49)")
    assert 0.48 < w[0] < 0.5, f"unexpected w[0]={w[0]}"

    # Fold arcs
    fa = FoldArcOptimizer(lr=1e-3)
    g6 = np.full((6, 6), 0.005, dtype=np.float32)
    for _ in range(5):
        fa.step(g6)
    eff_po_wo  = fa.effective(0, 1)   # Pop→Wo: π/3 gap → positive
    eff_po_sek = fa.effective(0, 3)   # Pop→Sek: π gap → negative
    print(f"effective Pop->Wo:  {eff_po_wo:.4f}  (expect > 0)")
    print(f"effective Pop->Sek: {eff_po_sek:.4f}  (expect < 0)")
    assert eff_po_wo  > 0, "Pop→Wo should be positive"
    assert eff_po_sek < 0, "Pop→Sek should be negative (π phase)"

    print("PASS")

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        print(__doc__)
