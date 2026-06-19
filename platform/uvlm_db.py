"""UVLM-derived aero database for the fast control (PPO) environment.

The full multi-surface UVLM is ~163 ms/solve — too slow for PPO's thousands of
rollouts. Following the established NL-VLM paradigm (Proietti 2022; the 3D VLM supplies
the force, a fast model carries it through the rollout), this precomputes the VALIDATED
UVLM total force coefficients vs angle of attack (feathered to the attached domain) ONCE,
then interpolates them in microseconds inside the flight env. The aero stays UVLM-grounded
(not strip theory): every coefficient is the actual multi-surface UVLM solve.

Builds CL(alpha), CD(alpha) for the feathered aircraft, plus the lift-curve slope and the
control-effectiveness used by the fast 6-DOF env. Cached to docs/uvlm_db.npz.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.join(_FLUXV, "src"), os.path.join(_FLUXV, "tests"),
          os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                  # noqa: E402
import uvlm_aircraft as U                                          # noqa: E402
from aircraft_geom import Aircraft                                 # noqa: E402

RHO = 1.225
ATTACHED_DEG = 14.0          # shallow-stall attached limit (literature)


def build_db(V=10.0, alphas_deg=None, cache=True):
    """Precompute UVLM CL/CD vs alpha for the aircraft. Returns dict + caches."""
    cache_path = os.path.join(_FLUXV, "docs", "uvlm_db.npz")
    if cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}
    wp.init()
    ac = Aircraft()
    surfs = U.build_aircraft_surfaces(ac)
    msu = U.MultiSurfaceUVLM(surfs)
    ps = {s["body"]: (np.zeros(3), np.array([0, 0, 0, 1.0])) for s in surfs}
    ts = {s["body"]: (np.zeros(3), np.zeros(3)) for s in surfs}
    if alphas_deg is None:
        alphas_deg = np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=float)
    S = sum(s["nc"] * s["ns"] for s in surfs) * 0.0 + 2 * ac.wing.area  # ref area (~2 wings)
    S = float(ac.wing.area)
    q = 0.5 * RHO * V * V
    CL, CD = [], []
    for a in alphas_deg:
        ar = np.deg2rad(a)
        Vinf = np.array([V * np.cos(ar), 0.0, -V * np.sin(ar)])
        out = msu.solve(ps, ts, Vinf)
        CL.append(out["lift"] / (q * S))
        CD.append(abs(out["drag"]) / (q * S))
    CL, CD = np.array(CL), np.array(CD)
    # lift-curve slope (1/rad) from the linear part
    CLa = float(np.polyfit(np.deg2rad(alphas_deg[:6]), CL[:6], 1)[0])
    db = dict(alpha_deg=alphas_deg, CL=CL, CD=CD, CLa=np.array(CLa), S=np.array(S),
              V_ref=np.array(V), attached_deg=np.array(ATTACHED_DEG))
    if cache:
        np.savez(cache_path, **db)
    return db


class AeroDB:
    """Fast interpolated UVLM aero: CL/CD(alpha) feathered to the attached domain."""

    def __init__(self, db=None):
        self.db = db or build_db()
        self.S = float(self.db["S"]); self.CLa = float(self.db["CLa"])
        self.att = np.deg2rad(float(self.db["attached_deg"]))
        self._a = np.deg2rad(self.db["alpha_deg"])
        self._cl = self.db["CL"]; self._cd = self.db["CD"]

    def coeffs(self, alpha):
        """CL, CD at effective AoA alpha (rad), feathered to the attached limit."""
        a = float(np.clip(abs(alpha), 0.0, self.att))      # feather to attached domain
        cl = np.interp(a, self._a, self._cl) * np.sign(alpha)
        cd = np.interp(a, self._a, self._cd)
        return cl, cd


if __name__ == "__main__":
    db = build_db(cache=False)
    print("UVLM aero database (feathered, attached domain):")
    print("  alpha(deg): " + "  ".join(f"{a:5.0f}" for a in db["alpha_deg"]))
    print("  CL        : " + "  ".join(f"{c:5.2f}" for c in db["CL"]))
    print("  CD        : " + "  ".join(f"{c:5.3f}" for c in db["CD"]))
    print(f"  lift-curve slope CLa = {float(db['CLa']):.2f} /rad  ref area S={float(db['S']):.3f} m^2")
    np.savez(os.path.join(_FLUXV, "docs", "uvlm_db.npz"), **db)
    print("  cached -> docs/uvlm_db.npz")
