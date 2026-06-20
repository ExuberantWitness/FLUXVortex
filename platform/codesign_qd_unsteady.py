"""MAP-Elites co-design on the differentiable UNSTEADY coupled FSI (Plan Phase F, P5 iteration-1).

The scientific experiment: illuminate the (翼面 morphology × 动力系统 control) behavior space with
co-designed (刚柔/质量 distribution + closed-loop control) solutions, evaluated on the validated
differentiable unsteady free-wake coupled FSI (diff_coupled_unsteady_gpu). Each archive cell holds
the best gust-rejecting design AT that (stiffness-taper, control-gain) — so distinct cells carry
distinct 本体 AND control, the MAP-Elites phenomenon.

Genotype θ (7-D, low-dim spline per plan): log-stiffness control points (root/mid/tip) + log-mass
control points + closed-loop feedback gain k. E_scale=exp(B·θ_E), ρ_scale=exp(B·θ_ρ) over the span
(B = quadratic Lagrange basis); control u_t = -k·dq_t (the validated closed-loop policy).
Quality = -J, J = residual deflection energy ‖q_N(free)‖² after a gust IC (lower = better gust
rejection). Behaviour descriptors: b1 = E_tip/E_root (spanwise stiffness taper, 翼面 axis),
b2 = k (control gain, 动力系统 axis). Emitters: random mutation (forward eval) + a DQD
gradient-arborescence emitter using the EXACT coupled-unsteady design+control gradient.

Runs on a single RTX 4090 (fp64); scale chosen so a full archive fills in minutes. Honest: this is
iteration-1 (gust-rejection objective, spanwise spline design, global-damping control); the
multi-objective (抗风×效率) and per-element/policy-net extensions are the documented next steps.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

for p in (os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants      # noqa: E402
import diff_coupled_unsteady as dcu                             # noqa: E402
import diff_coupled_unsteady_gpu as cg                          # noqa: E402
from diff_struct_design import _build_shell                     # noqa: E402

NX, NY, NSTEP, DT = 6, 4, 40, 1e-5
CG_TOL = 1e-4                       # QD eval tolerance (validation used 1e-12; quality ranking needs far less)
K_LO, K_HI = 0.0, 9.0              # closed-loop control gain (position-DOF velocity feedback; stable ≤~10)
B1_LO, B1_HI = 0.30, 3.30          # E_tip/E_root  stiffness taper (翼面 axis)
B2_LO, B2_HI = K_LO, K_HI          # control gain k                (动力系统 axis)
NB1, NB2 = 14, 14                  # archive grid


def _basis(nx, ny):
    """Quadratic Lagrange basis B (ne×3) mapping (root,mid,tip) control points -> per-element field
    along the span. Element e=j*nx+i sits at span fraction s=j/(ny-1)."""
    L = lambda s: np.array([2 * (s - 0.5) * (s - 1.0), -4 * s * (s - 1.0), 2 * s * (s - 0.5)])
    ne = nx * ny; B = np.zeros((ne, 3))
    for j in range(ny):
        s = j / max(ny - 1, 1)
        for i in range(nx):
            B[j * nx + i] = L(s)
    return B


class Env:
    def __init__(self, nx=NX, ny=NY, seed=0):
        self.nx, self.ny = nx, ny
        self.sh = _build_shell(nx=nx, ny=ny)
        self.C = ANCFConstants(self.sh, device=cfg.DEVICE)
        self.ne = self.sh.ne; self.ndof = self.sh.ndof
        self.B = _basis(nx, ny)
        free = np.array(sorted(set(range(self.ndof)) - set(self.sh._bc_dofs)))
        self.free = free; self.fmask = np.zeros(self.ndof); self.fmask[free] = 1.0
        self.qref = self.sh.q.copy()                   # undeformed reference (deflection is measured from here)
        rng = np.random.default_rng(seed)
        self.q0 = self.sh.q.copy(); self.q0[free] += 1e-3 * rng.standard_normal(len(free))
        self.dq0 = np.zeros(self.ndof); self.dq0[free] = 3e-2 * rng.standard_normal(len(free))  # gust IC
        self.P, self.dist = dcu._index_maps(self.sh, nx, ny)
        self.Mff = self.sh.M[np.ix_(free, free)].toarray()


def _fields(env, theta):
    E = np.exp(env.B @ theta[0:3]); R = np.exp(env.B @ theta[3:6]); k = float(theta[6])
    return E, R, k


def descriptors(env, theta):
    b1 = float(np.exp(theta[2] - theta[0]))            # E_tip/E_root  (stiffness taper, 翼面 axis)
    b2 = float(theta[6])                               # control gain k (动力系统 axis)
    return b1, b2


PENALTY = -1e9                                         # unstable/infeasible design (standard QD)


def eval_forward(env, theta):
    E, R, k = _fields(env, theta)
    try:
        qN, _ = cg.coupled_unsteady_forward_gpu(env.sh, env.C, env.P, env.dist, env.q0, env.dq0,
                                                NSTEP, DT, E, R, env.nx, env.ny, use_wake=True, fb_gain=k, cg_tol=CG_TOL)
    except Exception:                                  # any solver/Warp failure ⇒ infeasible design
        return PENALTY, np.inf, None
    if (not np.all(np.isfinite(qN))) or np.max(np.abs(qN)) > 1e3:
        return PENALTY, np.inf, None                   # divergent rollout = infeasible
    d = (qN - env.qref) * env.fmask                    # gust-induced deflection from the undeformed reference
    J = float(np.sum(d * d))
    return -J, J, qN


def eval_grad(env, theta):
    """∂(quality)/∂θ via the EXACT coupled-unsteady DESIGN+CONTROL gradient (gE,gR,dL/dk) + chain rule.
    Returns None for unstable designs (the DQD step is skipped)."""
    E, R, k = _fields(env, theta)
    try:
        qN, _ = cg.coupled_unsteady_forward_gpu(env.sh, env.C, env.P, env.dist, env.q0, env.dq0,
                                                NSTEP, DT, E, R, env.nx, env.ny, use_wake=True, fb_gain=k, cg_tol=CG_TOL)
        if (not np.all(np.isfinite(qN))) or np.max(np.abs(qN)) > 1e3:
            return None
        w = 2.0 * (qN - env.qref) * env.fmask           # ∂J/∂q_N for J=‖(q_N−q_ref)(free)‖²
        _, gE, gR, _, dL_dk = cg.coupled_unsteady_grad_gpu(
            env.sh, env.C, env.P, env.dist, env.q0, env.dq0, NSTEP, DT, w, E, R, env.nx, env.ny,
            use_wake=True, fb_gain=k, cg_tol=CG_TOL)
    except Exception:
        return None
    g = np.zeros(7)
    g[0:3] = env.B.T @ (E * gE)                         # ∂J/∂θ_E (chain through exp + spline)
    g[3:6] = env.B.T @ (R * gR)                         # ∂J/∂θ_ρ
    g[6] = dL_dk                                        # ∂J/∂k   (closed-loop control gain)
    if not np.all(np.isfinite(g)):
        return None
    return -g                                           # quality = -J


class Archive:
    def __init__(self):
        self.cells = {}                                # (i,j) -> dict(theta, qual, b1, b2)
        self.b1edges = np.linspace(B1_LO, B1_HI, NB1 + 1)
        self.b2edges = np.linspace(B2_LO, B2_HI, NB2 + 1)

    def _cell(self, b1, b2):
        i = int(np.clip(np.searchsorted(self.b1edges, b1) - 1, 0, NB1 - 1))
        j = int(np.clip(np.searchsorted(self.b2edges, b2) - 1, 0, NB2 - 1))
        return i, j

    def add(self, theta, qual, b1, b2):
        c = self._cell(b1, b2)
        if c not in self.cells or qual > self.cells[c]["qual"]:
            self.cells[c] = dict(theta=theta.copy(), qual=qual, b1=b1, b2=b2)
            return True
        return False

    def coverage(self):
        return len(self.cells) / (NB1 * NB2)

    def best(self):
        return max(self.cells.values(), key=lambda d: d["qual"]) if self.cells else None


LOG_E, LOG_R = 1.3, 0.7                                # stiffness can taper wide; mass kept stable (light mass → CFL blowup)


def _clamp(th):
    th = th.copy()
    th[0:3] = np.clip(th[0:3], -LOG_E, LOG_E)
    th[3:6] = np.clip(th[3:6], -LOG_R, LOG_R)
    th[6] = np.clip(th[6], K_LO, K_HI)
    return th


def rand_theta(rng):
    return _clamp(np.array([*(0.8 * rng.standard_normal(3)), *(0.5 * rng.standard_normal(3)),
                            rng.uniform(K_LO, K_HI)]))


def run(n_init=40, n_iter=450, n_dqd=100, dqd_lr=0.15, seed=0, log=print):
    wp.init()
    env = Env(seed=seed); rng = np.random.default_rng(seed + 1)
    arch = Archive(); nev = 0; nstable = 0; t0 = time.time()
    while len(arch.cells) < n_init:                    # random init until we seed real niches
        th = rand_theta(rng); q, _, _ = eval_forward(env, th); nev += 1
        if q > PENALTY:
            b1, b2 = descriptors(env, th); arch.add(th, q, b1, b2); nstable += 1
        if nev > 20 * n_init:
            break
    log(f"  init: {len(arch.cells)} niches from {nev} evals ({nstable} stable) in {time.time()-t0:.0f}s")
    for it in range(n_iter):                           # MAP-Elites: mutation (cheap) + DQD (sharpen)
        parent = list(arch.cells.values())[rng.integers(len(arch.cells))]
        th = parent["theta"].copy()
        if it % 3 == 0 and it < n_dqd * 3:             # DQD gradient-arborescence emitter
            g = eval_grad(env, th)
            if g is not None:
                th = _clamp(th + dqd_lr * g / (np.linalg.norm(g) + 1e-9))
            else:
                th = _clamp(th + 0.30 * rng.standard_normal(7) * np.array([1, 1, 1, 1, 1, 1, 5.0]))
        else:                                          # iso+line random mutation
            th = _clamp(th + 0.30 * rng.standard_normal(7) * np.array([1, 1, 1, 1, 1, 1, 5.0]))
        q, _, _ = eval_forward(env, th); nev += 1
        if q > PENALTY:
            b1, b2 = descriptors(env, th); arch.add(th, q, b1, b2); nstable += 1
        if (it + 1) % 100 == 0:
            log(f"  iter {it+1}: {len(arch.cells)} niches, coverage {arch.coverage()*100:.0f}%, "
                f"best quality {arch.best()['qual']:.3e}, {nev} evals, {time.time()-t0:.0f}s")
    log(f"  DONE: {len(arch.cells)} niches / {NB1*NB2} cells (coverage {arch.coverage()*100:.0f}%), "
        f"{nstable}/{nev} stable evals in {time.time()-t0:.0f}s on the 4090")
    return env, arch


if __name__ == "__main__":
    env, arch = run()
    # honest phenomenon check: do filled niches genuinely span distinct morphology AND control?
    cells = list(arch.cells.values())
    b1s = np.array([c["b1"] for c in cells]); b2s = np.array([c["b2"] for c in cells])
    quals = np.array([c["qual"] for c in cells])
    np.savez(os.path.join(os.path.dirname(__file__), "qd_unsteady_archive.npz"),
             b1=b1s, b2=b2s, qual=quals,
             thetas=np.array([c["theta"] for c in cells]),
             b1edges=arch.b1edges, b2edges=arch.b2edges)
    print("  saved archive -> qd_unsteady_archive.npz")
    top = sorted(cells, key=lambda d: d["qual"], reverse=True)[:max(1, len(cells) // 5)]
    tb1 = np.array([c["b1"] for c in top]); tb2 = np.array([c["b2"] for c in top])
    print(f"\n  PHENOMENON CHECK ({len(cells)} niches):")
    print(f"    stiffness taper spans {b1s.min():.2f}..{b1s.max():.2f}; control gain k spans {b2s.min():.2f}..{b2s.max():.2f}")
    print(f"    top-20% niches: stiffness taper {np.ptp(tb1):.2f} wide, control gain {np.ptp(tb2):.2f} wide "
          f"-> {'DIVERSE 本体 AND control across high-quality niches' if (np.ptp(tb1) > 0.4 and np.ptp(tb2) > 1.0) else 'concentrated (report honestly)'}")
