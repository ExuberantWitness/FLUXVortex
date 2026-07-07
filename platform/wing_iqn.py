"""IQN-ILS interface quasi-Newton (Degroote et al. 2009, C&S 87:793) on the
force interface, for the window-level Picard in WindowPredictorCorrector.

Trigger (docs/p2_s5_coupling_research.md #2): Aitken plateaus/limit-cycles on
the strong added-mass windows (measured: residual flat at 1.5e-5 for 50+
iterations, 2-cycles at 5.6e-3) — the fixed-point map has several dominant
modes, a scalar relaxation cannot damp them; IQN-ILS builds a low-rank secant
model of the interface Jacobian from the iteration history.

Update (standard IQN-ILS):
    x~^k = g(x^k)   (raw provider output)      r^k = x~^k - x^k
    Delta-R columns:  r^i - r^{i-1}            Delta-X~ columns: x~^i - x~^{i-1}
    lambda = argmin || Delta-R lambda + r^k ||     (economy least squares,
             rank-filtered — Degroote filter, drop near-dependent columns)
    x^{k+1} = x^k + r^k + Delta-X~ lambda
First iteration of each window: relaxed step with omega0 (no history yet).
History is per-window (no reuse across windows — simplest honest variant;
reuse 2-10 is the documented next lever if iteration counts stay high).

madd / a_lag passthrough: same convention as wing_aitken — the quasi-Newton
combination applies to the force VECTOR; the operator pair rides the newest
solve (LHS operator; at convergence identical).
"""
from __future__ import annotations

import numpy as np

from newton_pc.adapters.flap import NodalForceSet


def add_iqnils(provider, omega0=0.5, filt=1e-8, reuse=4, max_cols=40,
               mann_after=12, mann_omega=0.25, final_at=29):
    """Wrap provider.solve/commit with IQN-ILS. Returns live stats dict.

    reuse: number of PREVIOUS windows whose secant pairs are kept in the
    least-squares system (Degroote reuse 2-10: the interface Jacobian varies
    slowly window-to-window; a fresh window would otherwise re-learn it from
    scratch exactly where the fixed point is most repelling). max_cols caps
    the total secant history (oldest dropped).

    mann_after: on windows where IQN has not converged after this many
    iterations (measured: limit cycles / noisy-branch chatter at the violent
    stroke phases), switch to Krasnoselskii-Mann averaging — small relaxed
    steps x + mann_omega*r with a RUNNING MEAN of the iterates; the mean of a
    bounded orbit approximates the orbit center, and that averaged force is
    what gets committed (instead of a random sample of the cycle, which was
    measured to pump a 2-window sawtooth through the delayed-Kutta shed)."""
    orig_solve, orig_commit = provider.solve, provider.commit
    st = {"x": None, "gs": [], "rs": [], "hist": [], "xbar": None, "nbar": 0,
          "stats": dict(iters=[], n_filtered=0, n_mann=0)}

    def _pairs_this_window():
        k = len(st["rs"]) - 1
        cols = []
        for i in range(2, k + 1):
            cols.append((st["rs"][i] - st["rs"][i - 1],
                         st["gs"][i] - st["gs"][i - 1]))
        return cols

    def solve(state):
        out = orig_solve(state)
        g = out.f.copy()
        if st["x"] is None:                        # first solve of the window
            st["x"] = g                            # take the raw output
            st["gs"] = [g]
            st["rs"] = [np.zeros_like(g)]
            return out
        r = g - st["x"]
        st["gs"].append(g)
        st["rs"].append(r)
        k = len(st["rs"]) - 1
        if k > mann_after:                         # limit-cycle regime: Mann
            st["stats"]["n_mann"] += 1
            x_new = st["x"] + mann_omega * r
            st["nbar"] += 1
            if st["xbar"] is None:
                st["xbar"] = x_new.copy()
            else:
                st["xbar"] += (x_new - st["xbar"]) / st["nbar"]
            # iterate on x_new; hand out the ORBIT MEAN only for the final
            # (committing) march of a capped window
            x_out = st["xbar"] if k >= final_at else x_new
        else:
            cols = _pairs_this_window() + [p for w_ in st["hist"] for p in w_]
            if not cols:                           # no secant data yet: relax
                x_new = st["x"] + omega0 * r
            else:
                dR = np.stack([c[0] for c in cols], axis=1)
                dG = np.stack([c[1] for c in cols], axis=1)
                lam, res_, rank, sv = np.linalg.lstsq(dR, -r, rcond=filt)
                if rank < dR.shape[1]:
                    st["stats"]["n_filtered"] += dR.shape[1] - rank
                x_new = st["x"] + r + dG @ lam
            x_out = x_new
        st["x"] = x_new
        return NodalForceSet(x_out, payload=out.payload, madd=out.madd,
                             a_lag=getattr(out, "a_lag", None))

    def commit(F_new):
        st["stats"]["iters"].append(max(len(st["rs"]) - 1, 1))
        st["xbar"] = None; st["nbar"] = 0
        pw = _pairs_this_window()
        if pw:
            st["hist"].insert(0, pw)
        st["hist"] = st["hist"][:reuse]
        total = sum(len(w_) for w_ in st["hist"])
        while total > max_cols and st["hist"]:
            total -= len(st["hist"].pop())
        st["x"] = None; st["gs"] = []; st["rs"] = []
        return orig_commit(F_new)

    provider.solve = solve
    provider.commit = commit
    return st["stats"]
