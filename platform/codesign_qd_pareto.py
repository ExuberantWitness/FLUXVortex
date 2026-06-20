"""Multi-objective (gust-rejection × control-effort) Pareto from the co-design archive (Plan §7
抗风×效率→ here gust-load-alleviation: deflection vs actuation energy).

The archive (codesign_qd_unsteady.py) optimises gust deflection over the (stiffness taper × control
gain) behaviour space. Here we recover, for every illuminated elite, its **control effort**
E_ctrl = Σ_t ‖u_t‖²·dt with u_t = −k·q̇_t·(position DOFs) — the actuation energy spent to achieve
that gust rejection — by re-running the (cheap) coupled unsteady forward and reading the velocity
trajectory. The (deflection, effort) scatter over all elites traces the achievable gust-load-
alleviation Pareto front, coloured by morphology (stiffness taper): the co-design question is which
structures reach low deflection at low control effort. No re-search; pure post-evaluation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import warp as wp                                                # noqa: E402
import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import codesign_qd_unsteady as qd                                # noqa: E402
import diff_coupled_unsteady_gpu as cg                           # noqa: E402

HERE = os.path.dirname(__file__)


def control_effort(env, theta):
    """E_ctrl = Σ_t ‖u_t‖²·dt, u_t = −k·q̇_t·pos, from the coupled-unsteady velocity trajectory."""
    E, R, k = qd._fields(env, theta)
    pos = np.zeros(env.ndof)
    for n in range(env.ndof // 9):
        pos[9 * n:9 * n + 3] = 1.0
    pos *= env.fmask
    qN, qs = cg.coupled_unsteady_forward_gpu(env.sh, env.C, env.P, env.dist, env.q0, env.dq0,
                                             qd.NSTEP, qd.DT, E, R, env.nx, env.ny, use_wake=True,
                                             fb_gain=k, cg_tol=qd.CG_TOL)
    qs = np.array(qs)                                            # (N+1, ndof)
    dq = np.diff(qs, axis=0) / qd.DT                             # q̇ per step
    eff = float((k * k) * qd.DT * np.sum((dq * pos) ** 2))
    defl = float(np.sum(((qN - env.qref) * env.fmask) ** 2))
    return defl, eff


def main(npz=os.path.join(HERE, "qd_unsteady_archive.npz"), out=os.path.join(HERE, "qd_pareto.png")):
    wp.init()
    d = np.load(npz); thetas = d["thetas"]; b1 = d["b1"]; b2 = d["b2"]
    env = qd.Env()
    defl = np.zeros(len(thetas)); eff = np.zeros(len(thetas))
    for i, th in enumerate(thetas):
        try:
            defl[i], eff[i] = control_effort(env, th)
        except Exception:
            defl[i] = np.nan; eff[i] = np.nan
    ok = np.isfinite(defl) & np.isfinite(eff) & (defl < 1.0)
    defl, eff, tb1, tb2 = defl[ok], eff[ok], b1[ok], b2[ok]
    # Pareto front (minimise both deflection and effort)
    order = np.argsort(defl)
    front, best_e = [], np.inf
    for idx in order:
        if eff[idx] <= best_e + 1e-30:
            front.append(idx); best_e = eff[idx]
    front = np.array(front)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    sc = ax.scatter(defl, eff, c=tb1, cmap="coolwarm", s=34, norm=plt.matplotlib.colors.LogNorm(),
                    edgecolor="k", linewidth=0.3, alpha=0.85)
    fo = front[np.argsort(defl[front])]
    ax.plot(defl[fo], eff[fo], "-", color="k", lw=1.8, label=f"Pareto front ({len(front)} designs)")
    ax.scatter(defl[fo], eff[fo], facecolor="none", edgecolor="k", s=80, linewidth=1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gust deflection energy  $\\|q_N-q_{ref}\\|^2$  (← better gust rejection)")
    ax.set_ylabel("control effort  $\\sum_t\\|u_t\\|^2 dt$  (↓ less actuation energy)")
    ax.set_title("Aeroservoelastic gust-load-alleviation co-design — achievable Pareto front\n"
                 f"{len(defl)} co-designed elites on the differentiable unsteady FSI (single RTX 4090); "
                 "colour = stiffness taper $E_{tip}/E_{root}$", fontsize=10)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("stiffness taper $E_{tip}/E_{root}$")
    ax.legend(loc="upper right"); ax.grid(alpha=0.25, which="both")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved Pareto -> {out}")
    # honest finding: correlate the front's stiffness with position on the trade-off
    fr_b1 = tb1[fo]
    print(f"  {len(defl)} elites; Pareto front {len(front)} designs spanning "
          f"deflection {defl[fo].min():.2e}..{defl[fo].max():.2e}, effort {eff[fo].min():.2e}..{eff[fo].max():.2e}")
    print(f"  front stiffness taper {fr_b1.min():.2f}..{fr_b1.max():.2f} "
          f"(low-effort end taper={fr_b1[np.argmin(eff[fo])]:.2f}, low-deflection end taper={fr_b1[np.argmin(defl[fo])]:.2f})")
    return out


if __name__ == "__main__":
    main()
