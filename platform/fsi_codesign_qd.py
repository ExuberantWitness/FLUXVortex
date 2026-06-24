"""Full-UVLM meta-RL co-design — Step 3 (FLAGSHIP): MAP-Elites over the two STRUCTURAL distributions
(spanwise stiffness × spanwise mass) on the FULL coupled unsteady free-wake UVLM ⊗ ANCF FSI, with the
control layer AMORTIZED by the SHAC-trained design-conditioned controller k=π(θ_E,θ_R) (fsi_shac_controller).

This is the co-design the directive asks for — "基于元强化学习 + 刚柔分布 + 质量分布 + 设计参数的基于 MAP-Elites
的 co-design" — on the full nonlinear unsteady model, NO reduced surrogate:

  · design layer  = MAP-Elites over θ = (θ_E[3], θ_R[3]) — root/mid/tip spline control points of the
                    spanwise stiffness and mass fields (E=exp(B·θ_E), ρ=exp(B·θ_R)).
  · control layer = the amortized controller π(θ) supplies the per-design gust-rejection gain k (no per-
                    niche control search; the strong-coupled PC forward keeps high-k designs STABLE).
  · descriptors   = (E_tip/E_root  stiffness taper [翼面 axis],  ρ_tip/ρ_root  mass taper [质量 axis]).
  · quality       = −J, J = gust excursion ‖(q_N−q_ref)·free‖² on the STRONG-coupled UVLM FSI.
  · DQD emitter   = the EXACT design gradient ∂J/∂θ through the differentiable strong-coupled FSI,
                    INCLUDING the controller response dk/dθ:  ∂J/∂θ = ∂J/∂θ|_k + (∂J/∂k)(∂π/∂θ).

The archive illuminates the diverse (stiffness × mass) morphologies with their amortized controllers; each
cell stores the best (design, π-gain, J, efficiency). Reduced aggregates (cruise L/D, tip-inertia gust
resistance) are recorded per cell for the 抗风×效率 reading, but the QUALITY is the real coupled-FSI J.
"""
from __future__ import annotations

import os
import time

import numpy as np

import warp as wp
import codesign_qd_unsteady as cq
import diff_coupled_unsteady_gpu as cg
import fsi_shac_controller as ctl
import design_field as dfld

NX, NY, NSTEP, DT = 6, 4, 30, 2e-4
CG_TOL, PC_IT, PC_TOL = 1e-6, 22, 1e-7        # quality-ranking tolerances (validation used 1e-9)
ADJ_IT, ADJ_TOL = 40, 1e-8                     # DQD descent direction needs far fewer adjoint iters than the 1e-11 gate
B1_LO, B1_HI = 0.30, 3.30                      # E_tip/E_root  stiffness taper  (翼面 axis)
B2_LO, B2_HI = 0.50, 2.00                      # ρ_tip/ρ_root  mass taper       (质量 axis)
NB1, NB2 = 12, 12
LOG_E, LOG_R = 1.30, 0.69                      # θ box (exp(±) gives the descriptor ranges)
PENALTY = -1e9


def _descriptors(theta):
    b1 = float(np.exp(theta[2] - theta[0]))    # E_tip/E_root
    b2 = float(np.exp(theta[5] - theta[3]))    # ρ_tip/ρ_root
    return b1, b2


def _aggregates(theta):
    """Reduced physical reads for the 抗风×效率 interpretation (NOT the quality)."""
    sE = dfld.StiffnessField(np.exp(np.asarray(theta[0:3])))
    mR = dfld.MassField(np.exp(np.asarray(theta[3:6])))
    return dict(LD=dfld.cruise_efficiency(sE), m_gust=mR.m_gust(), m_total=mR.m_total())


def eval_design(env, pol, theta, grad=False):
    """k=π(θ); strong-coupled PC forward → J. grad=True also returns the EXACT ∂(−J)/∂θ (incl. dk/dθ)."""
    d6 = np.asarray(theta[0:6]); k = pol.gain(d6)
    E = np.exp(env.B @ d6[0:3]); R = np.exp(env.B @ d6[3:6])
    try:
        qN = cg.coupled_unsteady_forward_pc_gpu(env.sh, env.C, env.P, env.dist, env.q0, env.dq0,
                 NSTEP, DT, E, R, env.nx, env.ny, use_wake=True, fb_gain=k, cg_tol=CG_TOL,
                 pc_it=PC_IT, pc_tol=PC_TOL, wake_max=NSTEP * env.ny + env.ny)
    except Exception:
        return (PENALTY, np.inf, None)
    if (not np.all(np.isfinite(qN))) or np.max(np.abs(qN)) > 1e3:
        return (PENALTY, np.inf, None)
    dfl = (qN - env.qref) * env.fmask; J = float(np.sum(dfl * dfl))
    if not grad:
        return (-J, J, None)
    try:
        w = 2.0 * dfl                                          # ∂J/∂q_N
        _, gE, gR, _, dL_dk = cg.coupled_unsteady_pc_grad_gpu(
            env.sh, env.C, env.P, env.dist, env.q0, env.dq0, NSTEP, DT, w, E, R, env.nx, env.ny,
            use_wake=True, fb_gain=k, cg_tol=CG_TOL, pc_it=PC_IT, pc_tol=PC_TOL,
            adj_it=ADJ_IT, adj_tol=ADJ_TOL,
            wake_max=NSTEP * env.ny + env.ny)                  # match the J-forward's wake (no truncation)
        g = np.zeros(6)
        g[0:3] = env.B.T @ (E * gE)                            # ∂J/∂θ_E|_k
        g[3:6] = env.B.T @ (R * gR)                            # ∂J/∂θ_ρ|_k
        g += dL_dk * pol.dgain_dtheta(d6)                     # + (∂J/∂k)(∂k/∂θ): the controller response (∂k/∂θ, NOT ∂k/∂w)
        if not np.all(np.isfinite(g)):
            return (-J, J, None)
        return (-J, J, -g)                                     # quality=−J ⇒ ascent dir = −∂J/∂θ
    except Exception:
        return (-J, J, None)


class Archive:
    def __init__(self):
        self.cells = {}
        self.e1 = np.linspace(B1_LO, B1_HI, NB1 + 1); self.e2 = np.linspace(B2_LO, B2_HI, NB2 + 1)

    def _cell(self, b1, b2):
        i = int(np.clip(np.searchsorted(self.e1, b1) - 1, 0, NB1 - 1))
        j = int(np.clip(np.searchsorted(self.e2, b2) - 1, 0, NB2 - 1))
        return i, j

    def add(self, theta, qual, b1, b2, extra):
        c = self._cell(b1, b2)
        if c not in self.cells or qual > self.cells[c]["qual"]:
            self.cells[c] = dict(theta=theta.copy(), qual=qual, b1=b1, b2=b2, **extra); return True
        return False

    def coverage(self):
        return len(self.cells) / (NB1 * NB2)

    def best(self):
        return max(self.cells.values(), key=lambda d: d["qual"]) if self.cells else None


def _clamp(th):
    th = th.copy()
    th[0:3] = np.clip(th[0:3], -LOG_E, LOG_E); th[3:6] = np.clip(th[3:6], -LOG_R, LOG_R)
    return th


def _rand(rng):
    return _clamp(np.array([*(0.8 * rng.standard_normal(3)), *(0.45 * rng.standard_normal(3))]))


def run(pol, n_init=20, n_iter=160, n_dqd=45, dqd_lr=0.12, seed=0, log=print):
    wp.init(); env = cq.Env(nx=NX, ny=NY, seed=seed); rng = np.random.default_rng(seed + 1)
    arch = Archive(); nev = 0; nstab = 0; t0 = time.time()
    while len(arch.cells) < n_init:
        th = _rand(rng); q, J, _ = eval_design(env, pol, th); nev += 1
        if q > PENALTY:
            b1, b2 = _descriptors(th); arch.add(th, q, b1, b2, dict(k=pol.gain(th[:6]), J=J, **_aggregates(th))); nstab += 1
        if nev > 20 * n_init:
            break
    log(f"  init: {len(arch.cells)} niches from {nev} evals in {time.time()-t0:.0f}s")
    for it in range(n_iter):
        parent = list(arch.cells.values())[rng.integers(len(arch.cells))]
        th = parent["theta"].copy()
        if it % 3 == 0 and it < n_dqd * 3:
            _, _, g = eval_design(env, pol, th, grad=True)
            th = _clamp(th + dqd_lr * g / (np.linalg.norm(g) + 1e-9)) if g is not None \
                else _clamp(th + 0.25 * rng.standard_normal(6))
        else:
            th = _clamp(th + 0.25 * rng.standard_normal(6))
        q, J, _ = eval_design(env, pol, th); nev += 1
        if q > PENALTY:
            b1, b2 = _descriptors(th); arch.add(th, q, b1, b2, dict(k=pol.gain(th[:6]), J=J, **_aggregates(th))); nstab += 1
        if (it + 1) % 50 == 0:
            log(f"  iter {it+1}: {len(arch.cells)} niches, cov {arch.coverage()*100:.0f}%, "
                f"best −J {arch.best()['qual']:.3e}, {nev} evals, {time.time()-t0:.0f}s")
    log(f"  DONE: {len(arch.cells)}/{NB1*NB2} cells (cov {arch.coverage()*100:.0f}%), "
        f"{nstab}/{nev} stable evals in {time.time()-t0:.0f}s")
    return env, arch


def validate_design_grad(pol, seed=3, eps=2e-4):
    """Gate the DQD emitter: the assembled design gradient ∂J/∂θ (incl. the controller response dk/dθ)
    vs FD of J(θ) with k=π(θ) re-evaluated — the FULL closed-loop design gradient the archive ascends."""
    wp.init(); env = cq.Env(nx=NX, ny=NY, seed=0); rng = np.random.default_rng(seed)
    th = _clamp(np.array([*(0.5 * rng.standard_normal(3)), *(0.3 * rng.standard_normal(3))]))
    negJ, J0, neg_g = eval_design(env, pol, th, grad=True)
    if neg_g is None:
        print("  design-grad gate: design infeasible, retry seed"); return False
    g = -neg_g                                                # ∂J/∂θ
    g_fd = np.zeros(6)
    for i in range(6):
        tp = th.copy(); tp[i] += eps; _, Jp, _ = eval_design(env, pol, tp)
        tm = th.copy(); tm[i] -= eps; _, Jm, _ = eval_design(env, pol, tm)
        g_fd[i] = (Jp - Jm) / (2 * eps)
    rel = np.max(np.abs(g - g_fd)) / (np.max(np.abs(g_fd)) + 1e-30); ok = rel < 5e-2
    print(f"DQD design gradient ∂J/∂θ (incl. controller dk/dθ) vs FD on the full coupled UVLM FSI:")
    print(f"  analytic ‖g‖={np.linalg.norm(g):.3e}  vs FD ‖g‖={np.linalg.norm(g_fd):.3e}  max-rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the archive ascends an EXACT design gradient through the FSI")
    return ok


def figure(arch, path="docs/fsi_codesign_qd_archive.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.full((NB2, NB1), np.nan); kgrid = np.full((NB2, NB1), np.nan)
    for (i, j), c in arch.cells.items():
        grid[j, i] = c["J"]; kgrid[j, i] = c["k"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = ax[0].imshow(grid, origin="lower", aspect="auto", cmap="viridis_r",
                       extent=[B1_LO, B1_HI, B2_LO, B2_HI])
    ax[0].set_xlabel("E_tip/E_root (stiffness taper)"); ax[0].set_ylabel("ρ_tip/ρ_root (mass taper)")
    ax[0].set_title("gust excursion J (full UVLM FSI)"); fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(kgrid, origin="lower", aspect="auto", cmap="plasma",
                       extent=[B1_LO, B1_HI, B2_LO, B2_HI])
    ax[1].set_xlabel("E_tip/E_root (stiffness taper)"); ax[1].set_ylabel("ρ_tip/ρ_root (mass taper)")
    ax[1].set_title("amortized controller gain k=π(θ)"); fig.colorbar(im1, ax=ax[1])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); print(f"saved figure -> {os.path.abspath(path)}")


if __name__ == "__main__":
    import sys
    z = np.load("artifacts/fsi_shac_policy.npz"); pol = ctl.GainPolicy(w=z["w"])
    if "--gate" in sys.argv:                                   # quick design-gradient gate only
        raise SystemExit(0 if validate_design_grad(pol) else 1)
    if not validate_design_grad(pol):                         # gate the long archive run
        raise SystemExit("design-gradient gate FAILED — not running the archive")
    env, arch = run(pol)
    cells = list(arch.cells.values())
    np.savez("artifacts/fsi_codesign_qd_archive.npz",
             b1=np.array([c["b1"] for c in cells]), b2=np.array([c["b2"] for c in cells]),
             qual=np.array([c["qual"] for c in cells]), k=np.array([c["k"] for c in cells]),
             J=np.array([c["J"] for c in cells]), LD=np.array([c["LD"] for c in cells]),
             m_gust=np.array([c["m_gust"] for c in cells]), thetas=np.array([c["theta"] for c in cells]))
    figure(arch)
    b = arch.best()
    print(f"FLAGSHIP archive: {len(cells)} niches; best J={b['J']:.3e} at taper(E)={b['b1']:.2f} "
          f"taper(ρ)={b['b2']:.2f}, k=π={b['k']:.2f}; J spans {min(c['J'] for c in cells):.2e}.."
          f"{max(c['J'] for c in cells):.2e}, L/D {min(c['LD'] for c in cells):.1f}..{max(c['LD'] for c in cells):.1f}")
