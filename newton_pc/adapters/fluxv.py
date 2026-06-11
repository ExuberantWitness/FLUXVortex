"""FLUXV (ANCF shell + UVLM) adapters for the window predictor-corrector.

Wraps the MATLAB-exact validated machinery in ``tools/matlab_exact/``
(march + solve_chain) behind the ``StructuralEntry`` / ``ForceProvider``
protocols WITHOUT modifying the validated files. This pairing carries the
1e-6 MATLAB ground truth, so it is the regression arena proving the generic
coupler preserves the validated scheme.

Force-set semantics (mirrors the reference chain exactly):
  - the four force families (Qf_p vector, added-mass matrix, mat0, lift2)
    interpolate affinely across the window;
  - the slip-term auxiliaries (Gamma, dt_Amat1, dt_Amat2_Gamma, V_wake_plate)
    are NOT interpolated — they follow the newer anchor of each pass
    (predictor: last accepted solve; corrector: the fresh window solve),
    which `affine`/`lincomb` implement by inheriting them from the newer
    operand.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_ME = os.path.join(_ROOT, "tools", "matlab_exact")


def _load_pilot(mscale: float = 1.0):
    """Import the validated pilot module with neutral arguments."""
    if _ME not in sys.path:
        sys.path.insert(0, _ME)
        sys.path.insert(0, os.path.join(_ROOT, "src"))
        sys.path.insert(0, os.path.join(_ROOT, "tests"))
    argv_save = sys.argv
    sys.argv = ["newton_pc", "linear", "1", "34", "--mscale", str(mscale)]
    try:
        import pilot_grad_pc as P  # noqa: PLC0415
    finally:
        sys.argv = argv_save
    return P


class FluxvForceSet:
    """Interpolable wrapper around the pilot's ``Aero`` force families."""

    def __init__(self, aero, payload: dict | None = None):
        self.aero = aero
        self.payload = payload  # raw solve_chain output (wake etc.) for commit

    def affine(self, other: "FluxvForceSet", beta: float) -> "FluxvForceSet":
        P = other._P if other._P is not None else self._P
        a = P.Aero()
        for key in P.Aero.KEYS:
            sv = getattr(self.aero, key)
            ov = getattr(other.aero, key)
            setattr(a, key, sv + (ov - sv) * beta)
        # slip-term auxiliaries follow the NEWER anchor (reference semantics)
        for key in ("Gamma", "dA1", "dA2G", "Vwp"):
            setattr(a, key, getattr(other.aero, key))
        out = FluxvForceSet(a)
        out._P = P
        return out

    def lincomb(self, pairs) -> "FluxvForceSet":
        P = self._P
        a = P.Aero()
        for key in P.Aero.KEYS:
            acc = None
            for fs, w in pairs:
                term = getattr(fs.aero, key) * w
                acc = term if acc is None else acc + term
            setattr(a, key, acc)
        newest = pairs[-1][0]
        for key in ("Gamma", "dA1", "dA2G", "Vwp"):
            setattr(a, key, getattr(newest.aero, key))
        out = FluxvForceSet(a)
        out._P = P
        return out

    _P = None  # module handle, set by factory


class FluxvEntry:
    """StructuralEntry over the validated single-substep march."""

    def __init__(self, P, X0: np.ndarray, d_t: float):
        self.P = P
        self.X = X0.copy()
        self.d_t = d_t

    def snapshot(self) -> np.ndarray:
        return self.X.copy()

    def restore(self, snap: np.ndarray) -> None:
        self.X = snap.copy()

    def substep(self, t: float, dt: float, forces: FluxvForceSet) -> None:
        P = self.P
        it = int(round(t / self.d_t))
        # constant-force ctx: force_at(beta) returns the pre-interpolated set
        ctx = dict(mode="linear", Fk=forces.aero, Fk1=forces.aero,
                   Fkm1=None, dFk=None, dFk1=None)
        self.X = P.march(self.X, [it], 0.0, forces.aero, ctx)

    def state(self) -> np.ndarray:
        return self.X.copy()


class FluxvProvider:
    """ForceProvider over the validated full fluid solve (solve_chain)."""

    def __init__(self, P):
        self.P = P
        self.wake = None
        self.Gp = np.zeros(P.Ne)
        self.Gp2 = np.zeros(P.Ne)
        self.iw = 1
        self.n_solves = 0

    def solve(self, state: np.ndarray) -> FluxvForceSet:
        P = self.P
        out = P.ms.solve_chain(state, self.wake, self.Gp, self.Gp2,
                               first_wake=(self.iw == 1))
        self.n_solves += 1
        fs = FluxvForceSet(P.Aero.from_out(out), payload=out)
        fs._P = P
        return fs

    def commit(self, forces: FluxvForceSet) -> None:
        out = forces.payload
        if out is None:
            return  # zero boot set — nothing to accept
        self.wake = out["wake"]
        self.Gp2 = self.Gp
        self.Gp = out["Gamma"]
        self.iw += 1


def make_fluxv_pair(mscale: float = 1.0):
    """Build (entry, provider, zero_forces, P) at the validated initial state."""
    P = _load_pilot(mscale)
    FluxvForceSet._P = P
    hX = np.asarray(P.f3s["h_X_vec"])
    X0 = hX[:, 0].copy()
    entry = FluxvEntry(P, X0, P.d_t)
    provider = FluxvProvider(P)
    zero = FluxvForceSet(P.Aero())
    zero._P = P
    return entry, provider, zero, P
