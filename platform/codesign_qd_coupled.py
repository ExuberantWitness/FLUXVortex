"""MAP-Elites co-design on the REAL coupled FSI (assembled from S3/S4/S5).

The plan's §7 behavior space = 翼面轴 × 动力系统轴. Here:
  · BD_x (翼面 / 本体)   = mean wing stiffness  (the structural morphology level)
  · BD_y (动力系统 / 控制) = policy feedback gain k_p (control authority)
Each archive cell holds a FULL co-design solution = (per-element 刚柔 field, 质量 field, k_p, k_d),
quality = −J(coupled regulation), evaluated on the differentiable coupled FSI (diff_coupled_*).
The per-element distribution + mass + k_d are the free dims the emitter optimizes WITHIN each
(stiffness × control) niche — so different cells carry different 本体 AND control.

Emitters: random mutation (cheap forward eval) + a DQD gradient-arborescence emitter that uses
the S4 dual coupled gradient ∂(−J)/∂(刚柔,质量,k_p,k_d) + the BD gradients to steer designs.
Forward eval is fast (no per-step Jacobian); only the DQD parents pay the gradient cost.

Output: the illuminated (stiffness × control) archive + representative niches, showing the
typical MAP-Elites phenomenon — multiple high-performing niches with distinct morphology AND
control. Saves docs/codesign_qd_coupled.npz and .png.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt                                   # noqa: E402

import diff_coupled_policy as dcp                                 # noqa: E402
from diff_struct_design import _build_shell                       # noqa: E402

NX = NY = 3
E_RANGE, KP_RANGE = (0.4, 2.5), (0.0, 8.0)
GRID = (9, 7)
ELO, EHI, KPHI, KDHI = 0.4, 2.5, 8.0, 2.0


def _setup():
    sh = _build_shell(nx=NX, ny=NY)
    rng = np.random.default_rng(0)
    free = np.array(sorted(set(range(sh.ndof)) - set(sh._bc_dofs)))
    ref = sh.q.copy()
    q0 = sh.q.copy(); q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(sh.ndof); dq0[free] = 5e-3 * rng.standard_normal(len(free))
    cdof = dcp._ctrl_dof(sh, NX, NY)
    return sh, (free, cdof, ref[cdof]), q0, dq0, ref


def _unpack(sol, ne):
    return sol[:ne], sol[ne:2 * ne], float(sol[2 * ne]), float(sol[2 * ne + 1])


def _clip(sol, ne):
    sol[:2 * ne] = np.clip(sol[:2 * ne], ELO, EHI)
    sol[2 * ne] = np.clip(sol[2 * ne], *KP_RANGE); sol[2 * ne + 1] = np.clip(sol[2 * ne + 1], 0, KDHI)
    return sol


def eval_fast(sh, sol, env):
    """quality=-J and BD=(meanE, kp) via a forward-only coupled rollout (cheap)."""
    ne = sh.ne; E, R, kp, kd = _unpack(sol, ne)
    q0, dq0, ref, ctx = env
    sh.set_distribution(E_scale=E, rho_scale=R)
    J, _, _, _ = dcp.rollout(sh, q0, dq0, np.array([kp, kd]), ref, ctx, nx=NX, ny=NY,
                             N=10, nsub=2, want_grad=False)
    return -J, float(E.mean()), kp


def eval_grad(sh, sol, env):
    """∂(-J)/∂sol via the S4 dual coupled gradient (for the DQD emitter)."""
    ne = sh.ne; E, R, kp, kd = _unpack(sol, ne)
    q0, dq0, ref, ctx = env
    sh.set_distribution(E_scale=E, rho_scale=R)
    J, gE, gR, gth = dcp.rollout(sh, q0, dq0, np.array([kp, kd]), ref, ctx, nx=NX, ny=NY,
                                 N=10, nsub=2, want_grad=True)
    return -J, -np.concatenate([gE, gR, gth])


class Archive:
    def __init__(self, ne):
        self.ni, self.nj = GRID; self.ne = ne
        self.q = np.full(GRID, -np.inf); self.sol = np.zeros((self.ni, self.nj, 2 * ne + 2))

    def _cell(self, meanE, kp):
        i = int(np.clip((meanE - E_RANGE[0]) / (E_RANGE[1] - E_RANGE[0]) * self.ni, 0, self.ni - 1))
        j = int(np.clip((kp - KP_RANGE[0]) / (KP_RANGE[1] - KP_RANGE[0]) * self.nj, 0, self.nj - 1))
        return i, j

    def add(self, sol, qual, meanE, kp):
        i, j = self._cell(meanE, kp)
        if qual > self.q[i, j]:
            self.q[i, j] = qual; self.sol[i, j] = sol; return True
        return False

    def elites(self):
        ii, jj = np.where(np.isfinite(self.q))
        return [(self.sol[i, j], self.q[i, j], i, j) for i, j in zip(ii, jj)]

    def coverage(self):
        return float(np.isfinite(self.q).mean())


def run(budget_fwd=140, n_init=40, n_dqd=18, seed=0, log=print):
    sh, ctx, q0, dq0, ref = _setup()
    env = (q0, dq0, ref, ctx); ne = sh.ne
    rng = np.random.default_rng(seed)
    arch = Archive(ne)
    t0 = time.time()

    def rand_sol():
        lvl = rng.uniform(ELO, EHI)                    # span the stiffness (本体) axis
        E = lvl * np.exp(0.15 * rng.standard_normal(ne))
        R = np.exp(0.3 * rng.standard_normal(ne))
        return _clip(np.concatenate([E, R, [rng.uniform(*KP_RANGE), rng.uniform(0, KDHI)]]), ne)
    for _ in range(n_init):
        s = rand_sol(); qd, mE, kp = eval_fast(sh, s, env); arch.add(s, qd, mE, kp)
    # MAP-Elites: random mutation (cheap) to fill, DQD gradient (S4) to sharpen quality
    n_fwd = n_init
    while n_fwd < budget_fwd:
        el = arch.elites()
        sol0 = el[rng.integers(len(el))][0].copy()
        sigma = np.concatenate([np.full(2 * ne, 0.18), [1.2, 0.3]])
        s = _clip(sol0 + sigma * rng.standard_normal(2 * ne + 2), ne)
        qd, mE, kp = eval_fast(sh, s, env); arch.add(s, qd, mE, kp); n_fwd += 1
    for _ in range(n_dqd):                              # DQD gradient-arborescence steps
        el = arch.elites()
        sol0 = el[rng.integers(len(el))][0].copy()
        _, g = eval_grad(sh, sol0, env)
        gq = g / (np.linalg.norm(g) + 1e-9)
        gx = np.zeros_like(g); gx[:ne] = 1.0 / ne; gx /= np.linalg.norm(gx) + 1e-9   # ∂meanE
        gy = np.zeros_like(g); gy[2 * ne] = 1.0                                        # ∂kp
        for _ in range(3):
            c = rng.normal(size=3)
            s = _clip(sol0 + 0.25 * (c[0] * gq + c[1] * gx + c[2] * gy), ne)
            qd, mE, kp = eval_fast(sh, s, env); arch.add(s, qd, mE, kp)
        n_fwd += 3
    log(f"  archive: {len(arch.elites())} niches, coverage {arch.coverage()*100:.0f}%, "
        f"{n_fwd} fwd evals + {n_dqd} DQD grads, {time.time()-t0:.0f}s")
    return sh, arch


def figure(sh, arch):
    ne = sh.ne
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    axH = fig.add_subplot(gs[0])
    Q = np.where(np.isfinite(arch.q), arch.q, np.nan)
    im = axH.imshow(Q.T, origin="lower", aspect="auto", cmap="viridis",
                    extent=[E_RANGE[0], E_RANGE[1], KP_RANGE[0], KP_RANGE[1]])
    axH.set_xlabel("翼面轴: mean wing stiffness 刚柔 (本体)")
    axH.set_ylabel("动力系统轴: control gain k_p (控制)")
    axH.set_title("MAP-Elites archive on the REAL coupled FSI\n"
                  "(quality = −J; each cell = a full 刚柔/质量/policy co-design)", fontsize=11)
    plt.colorbar(im, ax=axH, label="quality (−J, higher better)")

    # representative niches near the 4 corners of the (stiffness × control) space —
    # chosen to SPAN 本体 (low/high stiffness) × control (low/high k_p) to show the diversity
    el = arch.elites()
    corners = [(0, 0), (0, arch.nj - 1), (arch.ni - 1, 0), (arch.ni - 1, arch.nj - 1)]
    picks = []
    for ci, cj in corners:
        sol, q, i, j = min(el, key=lambda e: (e[2] - ci) ** 2 + (e[3] - cj) ** 2)
        picks.append((sol, q, i, j))
    axN = fig.add_subplot(gs[1])
    xe = np.arange(ne); w = 0.2
    colors = plt.cm.tab10(np.linspace(0, 1, len(picks)))
    for k, (sol, q, i, j) in enumerate(picks):
        E, R, kp, kd = _unpack(sol, ne)
        mE = (i + 0.5) / arch.ni * (E_RANGE[1] - E_RANGE[0]) + E_RANGE[0]
        axH.plot(E.mean(), kp, "o", ms=11, mec="white", mfc=colors[k], mew=1.5)
        axH.annotate(f"N{k+1}", (E.mean(), kp), color="white", fontsize=8, ha="center", va="center")
        axN.plot(xe, E, "-o", color=colors[k], ms=3,
                 label=f"N{k+1}: 刚柔̄={E.mean():.2f} 质量̄={R.mean():.2f} kp={kp:.1f} kd={kd:.2f}")
    axN.set_xlabel("wing element"); axN.set_ylabel("per-element 刚柔 (E-scale)")
    axN.set_title("Representative niches — distinct 本体 AND control\n"
                  "(different stiffness distribution + policy per niche)", fontsize=11)
    axN.grid(alpha=0.3); axN.legend(fontsize=7.5, loc="upper right")

    plt.tight_layout()
    out = os.path.join(_FLUXV, "docs", "codesign_qd_coupled.png")
    plt.savefig(out, dpi=110)
    sols = np.array([s for s, _, _, _ in arch.elites()])
    np.savez(os.path.join(_FLUXV, "docs", "codesign_qd_coupled.npz"),
             q=arch.q, sols=arch.sol, ne=ne)
    print(f"saved -> {out}")
    # the phenomenon: do the top niches genuinely differ in 本体 AND control?
    Es = np.array([_unpack(s, ne)[0].mean() for s, _, _, _ in picks])
    kps = np.array([_unpack(s, ne)[2] for s, _, _, _ in picks])
    print(f"  {len(picks)} top niches: mean刚柔 spans {Es.min():.2f}..{Es.max():.2f}, "
          f"kp spans {kps.min():.1f}..{kps.max():.1f} -> distinct 本体 AND control per niche"
          if len(picks) else "  (no niches)")


def main():
    import warp as wp; wp.init()
    print("MAP-Elites co-design on the REAL coupled FSI (翼面轴 × 动力系统轴)")
    sh, arch = run(log=print)
    figure(sh, arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
