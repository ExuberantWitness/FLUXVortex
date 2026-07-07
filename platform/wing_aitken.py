"""Force-space Aitken Delta^2 dynamic under-relaxation (Kuettler-Wall 2008)
for the window-level Picard loop in WindowPredictorCorrector.

Recipe (docs/p2_s5_coupling_research.md #2, citations verified):
  - omega_k = -omega_{k-1} * (r_{k-1} . (r_k - r_{k-1})) / ||r_k - r_{k-1}||^2
  - window start: omega INHERITED from the previous window's end value,
    sign kept, clamped to min(|omega|, start_max=0.5); first window omega0.
    (The scaffold _wf_full.py reset omega per window — measured KW deviation.)
  - in-iteration clip [omin, omax] kept as a safety rail (documented deviation:
    KW leaves the iteration value unclamped; negative/large omegas are where
    the acceleration comes from — hence the wide default rail).
  - struggling window: after `shrink_after` iterations the LOCAL omax halves
    progressively (recorded in stats) — the in-window analogue of the plan's
    "halve omega_max and retry" without re-running committed wake state.
  - madd passthrough: relaxation applies to the force VECTOR only; the madd
    operator rides along from the newest solve (it acts on the LHS; relaxing
    it would double-relax the added-mass channel).
"""
from __future__ import annotations

import numpy as np

from newton_pc.adapters.flap import NodalForceSet


def add_aitken(provider, omega0=0.5, omin=0.05, omax=1.0, start_max=0.5,
               shrink_after=12):
    """Wrap provider.solve/commit with KW Aitken relaxation. Returns a stats
    dict that live-updates: iters (per-window list), omega_end, n_shrinks."""
    orig_solve, orig_commit = provider.solve, provider.commit
    st = {"x": None, "r_prev": None, "omega": omega0, "omega_carry": omega0,
          "k": 0, "omax_local": omax, "stats": dict(iters=[], n_shrinks=0)}

    def solve(state):
        out = orig_solve(state)
        g = out.f.copy()
        st["k"] += 1
        if st["x"] is None:                          # first solve of the window
            st["x"] = g
            st["r_prev"] = None
            w = st["omega_carry"]
            st["omega"] = float(np.sign(w) or 1.0) * min(abs(w), start_max)
            st["omax_local"] = omax
            return out
        r = g - st["x"]
        if st["r_prev"] is not None:
            dr = r - st["r_prev"]
            den = float(dr @ dr) + 1e-30
            st["omega"] = float(np.clip(
                -st["omega"] * float(st["r_prev"] @ dr) / den,
                omin, st["omax_local"]))
        if st["k"] > shrink_after:                   # struggling: shrink rail
            st["omax_local"] = max(st["omax_local"] * 0.5, omin)
            st["stats"]["n_shrinks"] += 1
        x_new = st["x"] + st["omega"] * r
        st["x"] = x_new
        st["r_prev"] = r
        return NodalForceSet(x_new, payload=out.payload, madd=out.madd,
                             a_lag=getattr(out, "a_lag", None))

    def commit(F_new):
        st["stats"]["iters"].append(st["k"])
        st["omega_carry"] = st["omega"]              # cross-window inheritance
        st["x"] = None; st["r_prev"] = None; st["k"] = 0
        return orig_commit(F_new)

    provider.solve = solve
    provider.commit = commit
    return st["stats"]
