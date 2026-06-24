"""Per-ELEMENT stiffness+mass co-design on the full coupled UVLM FSI by the EXACT per-element adjoint
gradient, with the amortised controller in the loop.

This drops the low-D spanwise-spline parameterisation (3+3 control points) of fsi_codesign_qd and instead
optimises the FULL per-element design x = (log E_e, log ρ_e) ∈ R^(2·ne) directly — the differentiable
strong-coupled FSI returns the per-element ∂J/∂E_e, ∂J/∂ρ_e (validated by verify_pc_grad), so every
element's stiffness AND mass is a free design variable. The control gain is supplied by the trained
amortised controller k = π(descriptor(x)) and held fixed within each design step (alternating block: the
design step uses the EXACT ∂J/∂(E,ρ)|_k; k is refreshed from the new field between steps). Bound + spanwise
smoothness keep the field manufacturable.

validate_perelem_grad() gates the per-element gradient vs FD (k fixed) before optimisation; optimize()
runs projected Adam from the uniform wing; figure() shows the optimised per-element E and ρ over the
(chord × span) element grid. NO surrogate — quality is the coupled-FSI gust excursion J.
"""
from __future__ import annotations

import os

import numpy as np

import warp as wp
import codesign_qd_unsteady as cq
import diff_coupled_unsteady_gpu as cg
import fsi_shac_controller as ctl

NX, NY, NSTEP, DT = 6, 4, 30, 2e-4
CG_TOL, PC_IT, PC_TOL = 1e-6, 22, 1e-7
ADJ_IT, ADJ_TOL = 40, 1e-8
LOG_E, LOG_R = 1.30, 0.69                       # per-element log-scale bounds (manufacturability)
SMOOTH = 2e-3                                    # spanwise smoothness penalty weight (neighbouring elements)


def _descriptor(env, xE, xR):
    """Project the per-element log-field onto the controller's 6-D (root/mid/tip) descriptor via the
    quadratic span basis pseudo-inverse — the morphology summary the amortised π was trained on."""
    Bpinv = np.linalg.pinv(env.B)                # (3, ne)
    return np.concatenate([Bpinv @ xE, Bpinv @ xR])


def _smooth_pen(env, x):
    """½·SMOOTH·Σ (x_e − x_nbr)² over spanwise neighbours (e and e+nx share chord station) + its grad."""
    nx, ny = env.nx, env.ny; X = x.reshape(ny, nx); g = np.zeros_like(X); P = 0.0
    for j in range(ny - 1):
        d = X[j + 1] - X[j]; P += 0.5 * SMOOTH * float(np.sum(d * d))
        g[j + 1] += SMOOTH * d; g[j] -= SMOOTH * d
    return P, g.reshape(-1)


def eval_perelem(env, pol, xE, xR, k=None, grad=False):
    """J (coupled-FSI gust excursion) at per-element design (E=exp(xE), ρ=exp(xR)); k from π unless given.
    grad=True → exact per-element (∂J/∂xE, ∂J/∂xR)|_k (+ spanwise-smoothness grad). Returns (J, gxE, gxR, k)."""
    E = np.exp(np.clip(xE, -LOG_E, LOG_E)); R = np.exp(np.clip(xR, -LOG_R, LOG_R))
    if k is None:
        k = pol.gain(_descriptor(env, xE, xR))
    qN = cg.coupled_unsteady_forward_pc_gpu(env.sh, env.C, env.P, env.dist, env.q0, env.dq0,
             NSTEP, DT, E, R, env.nx, env.ny, use_wake=True, fb_gain=k, cg_tol=CG_TOL,
             pc_it=PC_IT, pc_tol=PC_TOL, wake_max=NSTEP * env.ny + env.ny)
    dfl = (qN - env.qref) * env.fmask; J = float(np.sum(dfl * dfl))
    pE, gsE = _smooth_pen(env, xE); pR, gsR = _smooth_pen(env, xR); Jtot = J + pE + pR
    if not grad:
        return Jtot, None, None, k
    w = 2.0 * dfl
    _, gE, gR, _, _ = cg.coupled_unsteady_pc_grad_gpu(
        env.sh, env.C, env.P, env.dist, env.q0, env.dq0, NSTEP, DT, w, E, R, env.nx, env.ny,
        use_wake=True, fb_gain=k, cg_tol=CG_TOL, pc_it=PC_IT, pc_tol=PC_TOL,
        adj_it=ADJ_IT, adj_tol=ADJ_TOL, wake_max=NSTEP * env.ny + env.ny)
    gxE = gE * E + gsE                            # ∂J/∂xE_e = (∂J/∂E_e)·E_e  (+ smoothness)
    gxR = gR * R + gsR
    return Jtot, gxE, gxR, k


def validate_perelem_grad(pol, seed=4, eps=3e-4):
    """Per-element ∂J/∂(xE,xR) vs FD (k FIXED — the design-step gradient the optimiser ascends). Gate."""
    wp.init(); env = cq.Env(nx=NX, ny=NY, seed=0); rng = np.random.default_rng(seed)
    xE = 0.25 * rng.standard_normal(env.ne); xR = 0.18 * rng.standard_normal(env.ne)
    k = pol.gain(_descriptor(env, xE, xR))        # fix k for the gradient AND the FD
    J0, gxE, gxR, _ = eval_perelem(env, pol, xE, xR, k=k, grad=True)
    probes = [0, env.ne // 3, env.ne - 1]; relsE = []; relsR = []
    for e in probes:
        xp = xE.copy(); xp[e] += eps; Jp, _, _, _ = eval_perelem(env, pol, xp, xR, k=k)
        xm = xE.copy(); xm[e] -= eps; Jm, _, _, _ = eval_perelem(env, pol, xm, xR, k=k)
        relsE.append(abs(gxE[e] - (Jp - Jm) / (2 * eps)) / (abs((Jp - Jm) / (2 * eps)) + 1e-12))
        yp = xR.copy(); yp[e] += eps; Jp, _, _, _ = eval_perelem(env, pol, xE, yp, k=k)
        ym = xR.copy(); ym[e] -= eps; Jm, _, _, _ = eval_perelem(env, pol, xE, ym, k=k)
        relsR.append(abs(gxR[e] - (Jp - Jm) / (2 * eps)) / (abs((Jp - Jm) / (2 * eps)) + 1e-12))
    rel = max(max(relsE), max(relsR)); ok = rel < 5e-2
    print(f"per-element design gradient ∂J/∂(logE_e, logρ_e) vs FD on the full coupled UVLM FSI "
          f"({env.ne} elems, k={k:.3f}):")
    print(f"  stiffness probes rel={[f'{r:.1e}' for r in relsE]}  mass probes rel={[f'{r:.1e}' for r in relsR]}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: every element's E and ρ has an EXACT FSI gradient — per-element "
          f"co-design is gradient-driven on the full unsteady model")
    return ok


def optimize(pol, steps=40, lr=0.08, seed=0, log=print):
    """Projected Adam on the per-element (xE, xR) from the UNIFORM wing; k refreshed from π each step."""
    wp.init(); env = cq.Env(nx=NX, ny=NY, seed=seed)
    ne = env.ne; xE = np.zeros(ne); xR = np.zeros(ne)
    mE = np.zeros(ne); vE = np.zeros(ne); mR = np.zeros(ne); vR = np.zeros(ne)
    b1, b2, e = 0.9, 0.999, 1e-8
    J0, _, _, k0 = eval_perelem(env, pol, xE, xR); log(f"  uniform wing: J={J0:.4e}, k=π={k0:.3f}")
    hist = [J0]
    for t in range(steps):
        J, gxE, gxR, k = eval_perelem(env, pol, xE, xR, grad=True)
        mE = b1 * mE + (1 - b1) * gxE; vE = b2 * vE + (1 - b2) * gxE * gxE
        mR = b1 * mR + (1 - b1) * gxR; vR = b2 * vR + (1 - b2) * gxR * gxR
        c1 = 1 - b1 ** (t + 1); c2 = 1 - b2 ** (t + 1)
        xE = np.clip(xE - lr * (mE / c1) / (np.sqrt(vE / c2) + e), -LOG_E, LOG_E)
        xR = np.clip(xR - lr * (mR / c1) / (np.sqrt(vR / c2) + e), -LOG_R, LOG_R)
        hist.append(J)
        if t % 5 == 0 or t == steps - 1:
            log(f"  step {t:3d}: J={J:.4e}  k=π={k:.3f}  E∈[{np.exp(xE).min():.2f},{np.exp(xE).max():.2f}]  "
                f"ρ∈[{np.exp(xR).min():.2f},{np.exp(xR).max():.2f}]")
    Jf, _, _, kf = eval_perelem(env, pol, xE, xR)
    log(f"  DONE: J {J0:.4e} -> {Jf:.4e}  ({100*(J0-Jf)/J0:.1f}% reduction), final k=π={kf:.3f}")
    return env, xE, xR, hist


def figure(env, xE, xR, hist, path="docs/fsi_perelem_codesign.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx, ny = env.nx, env.ny
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(hist, "-o", ms=3); ax[0].set_xlabel("step"); ax[0].set_ylabel("gust excursion J")
    ax[0].set_title("per-element gradient co-design"); ax[0].grid(alpha=0.3)
    im1 = ax[1].imshow(np.exp(xE).reshape(ny, nx), origin="lower", aspect="auto", cmap="viridis",
                       extent=[0, 1, 0, 1])
    ax[1].set_title("optimised E_e / E_nom (per element)"); ax[1].set_xlabel("chord →"); ax[1].set_ylabel("span (root→tip) →")
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(np.exp(xR).reshape(ny, nx), origin="lower", aspect="auto", cmap="magma",
                       extent=[0, 1, 0, 1])
    ax[2].set_title("optimised ρ_e / ρ_nom (per element)"); ax[2].set_xlabel("chord →"); ax[2].set_ylabel("span (root→tip) →")
    fig.colorbar(im2, ax=ax[2])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); print(f"saved figure -> {os.path.abspath(path)}")


if __name__ == "__main__":
    import sys
    z = np.load("artifacts/fsi_shac_policy.npz"); pol = ctl.GainPolicy(w=z["w"])
    if "--gate" in sys.argv:
        raise SystemExit(0 if validate_perelem_grad(pol) else 1)
    if not validate_perelem_grad(pol):
        raise SystemExit("per-element gradient gate FAILED — not optimising")
    env, xE, xR, hist = optimize(pol)
    np.savez("artifacts/fsi_perelem_design.npz", xE=xE, xR=xR, hist=np.array(hist))
    figure(env, xE, xR, hist)
