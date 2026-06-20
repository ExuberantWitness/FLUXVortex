"""Numerical stabilization of the coupled FSI (fix 1a) — over-flex / stiff designs must not
blow up (the real-FSI pipeline saw a design diverge to ~5e9 N). The explicit coupled step is
CFL-limited (stiff modes blow up when ω·dt>2); the stabilization is sub-stepping (smaller
effective dt) + mass-proportional structural damping, both kept differentiable so the S3
coupled design gradient still validates under stabilization.

verify():
  (1) a stiff+light design DIVERGES with the plain step (nsub=1, no damping);
  (2) the SAME design stays BOUNDED with sub-stepping + damping;
  (3) the coupled design gradient still matches FD under stabilization.
"""
from __future__ import annotations

import numpy as np

import diff_coupled_fsi as dc
from diff_struct_design import _build_shell


def _setup(nx=3, ny=3, E=3.5, R=0.4, seed=0):
    sh = _build_shell(nx=nx, ny=ny)
    ne = sh.ne
    sh.set_distribution(E_scale=np.full(ne, E), rho_scale=np.full(ne, R))   # stiff + light -> CFL-prone
    rng = np.random.default_rng(seed)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    return sh, free, q0, dq0


def _peak(sh, q0, dq0, free, N, dt, nx, ny, alpha, nsub):
    """Peak DEFLECTION growth from the initial state (excludes the ~1 reference slope DOFs).
    A diverged rollout NaNs/raises (singular assembly) -> reported as (inf, False)."""
    P, dist = dc._index_maps(sh, nx, ny)
    Mff = sh.M[np.ix_(free, free)].toarray()
    try:
        with np.errstate(all="ignore"):
            qs, _ = dc._forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny, alpha, nsub)
        dev = np.abs(qs[-1] - q0)[free]
        if not np.all(np.isfinite(qs[-1])):
            return float("inf"), False
        return float(np.max(dev)), True
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return float("inf"), False


def verify(nx=3, ny=3, N=14, dt=4.5e-4):
    sh, free, q0, dq0 = _setup(nx, ny)
    A2, NS2 = 10.0, 16
    # (1) plain explicit step -> diverges (CFL violated for the stiff+light design)
    pk0, fin0 = _peak(sh, q0, dq0, free, N, dt, nx, ny, alpha=0.0, nsub=1)
    # (2) sub-stepping + damping -> bounded
    pk1, fin1 = _peak(sh, q0, dq0, free, N, dt, nx, ny, alpha=A2, nsub=NS2)
    diverged = (not fin0) or pk0 > 1e2
    bounded = fin1 and pk1 < 1.0
    print("Numerical stabilization of the coupled FSI (fix 1a):")
    print(f"  (1) plain step (nsub=1, no damping):   peak deflection={pk0:.2e} finite={fin0} "
          f"-> {'DIVERGES' if diverged else 'ok'}")
    print(f"  (2) sub-step + damping (nsub={NS2}, α={A2:.0f}): peak deflection={pk1:.2e} "
          f"finite={fin1} -> {'BOUNDED ✓' if bounded else 'still unstable'}")

    # (3) coupled design gradient still valid under stabilization
    ne = sh.ne; rng = np.random.default_rng(3)
    Es = np.exp(0.15 * rng.standard_normal(ne)); Rs = np.exp(0.15 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    alpha, nsub, Ng, dtg, eps = 6.0, 4, 8, 4e-5, 1e-5
    _, gE, gR = dc.loss_and_grad(sh, q0, dq0, Ng, dtg, free, w, nx, ny, alpha, nsub)

    def Jonly(E_, R_):
        sh.set_distribution(E_scale=E_, rho_scale=R_)
        P, dist = dc._index_maps(sh, nx, ny)
        Mff = sh.M[np.ix_(free, free)].toarray()
        qs, _ = dc._forward(sh, q0, dq0, Ng, dtg, free, Mff, P, dist, nx, ny, alpha, nsub)
        return float(w @ qs[-1])
    gE_fd = np.zeros(ne); gR_fd = np.zeros(ne)
    for e in range(ne):
        ep = Es.copy(); ep[e] += eps; em = Es.copy(); em[e] -= eps
        gE_fd[e] = (Jonly(ep, Rs) - Jonly(em, Rs)) / (2 * eps)
        rp = Rs.copy(); rp[e] += eps; rm = Rs.copy(); rm[e] -= eps
        gR_fd[e] = (Jonly(Es, rp) - Jonly(Es, rm)) / (2 * eps)
    relE = np.max(np.abs(gE - gE_fd)) / (np.max(np.abs(gE_fd)) + 1e-30)
    relR = np.max(np.abs(gR - gR_fd)) / (np.max(np.abs(gR_fd)) + 1e-30)
    okg = relE < 1e-3 and relR < 1e-3
    print(f"  (3) coupled gradient UNDER stabilization vs FD: ∂刚柔 rel={relE:.2e} "
          f"∂质量 rel={relR:.2e} -> {'PASS' if okg else 'FAIL'}")
    ok = diverged and bounded and okg
    print(f"  -> {'PASS' if ok else 'FAIL'}: stabilization tames the over-flex blow-up and "
          f"keeps the design gradient differentiable")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
