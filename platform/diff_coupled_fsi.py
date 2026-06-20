"""Differentiable COUPLED FSI design gradient (S3) — the design gradient ∂L/∂(刚柔, 质量)
flowing through the FULL aeroelastic coupling: ANCF structure (mass + stiffness adjoints,
diff_struct_design) ⊗ VLM aero load on the DEFORMED geometry (diff_vlm).

Coupled rollout (ANCF nodes ARE the VLM lattice corners — direct geometry/load transfer):
    corners_t = node positions(q_t)
    F_aero_t  = distribute( VLM(corners_t) ) → nodal forces
    a_t       = M(ρ)⁻¹ ( F_aero_t − Qint(q_t; E) )            (symplectic structural step)

Adjoint chains BOTH physics per step:
    adj_rhs   = M⁻¹ adj_a
    aero  :   adj_q += Pᵀ · J_vlm(q_t)ᵀ · (distributeᵀ adj_rhs)     ← the coupled aero term
    struct:   adj_q += K_t(q_t) · (−adj_rhs)        ;  ∂L/∂E_e, ∂L/∂ρ_e accumulate as before
with J_vlm the EXACT complex-step VLM Jacobian (diff_vlm.panel_jacobian). Mass (ρ) is
differentiable because M(ρ) is linear (diff_struct_design); the aero adds the geometry
coupling. This is the gradient ‘穿完整耦合 FSI’ — quasi-steady aero load (unsteady wake is a
later refinement); the linear-solve VJP inside the aero is exact (np.linalg.solve / complex
step), the structural K_t adjoint is the validated exact tangent.

verify(): ∂L/∂E and ∂L/∂ρ from the coupled adjoint vs central finite differences of the same
coupled forward (re-running structure+aero with set_distribution) — the S3 red line.
"""
from __future__ import annotations

import numpy as np

from diff_struct_design import (_build_shell, _assemble, _elem_mass_unit)
import diff_vlm

VINF = np.array([12.0, 0.0, 1.2])     # freestream with small AoA so the aero load is nonzero


def _index_maps(sh, nx, ny):
    """P (ncorner·3 × ndof): corners = node positions(q). dist (ndof × npan·3): panel force
    -> nodal force (1/4 to each of a panel's 4 nodes, on the position DOFs)."""
    nc = (nx + 1) * (ny + 1)
    P = np.zeros((nc * 3, sh.ndof))
    for i in range(nx + 1):
        for j in range(ny + 1):
            node = j * (nx + 1) + i
            cflat = (i * (ny + 1) + j) * 3
            for d in range(3):
                P[cflat + d, 9 * node + d] = 1.0
    dist = np.zeros((sh.ndof, nx * ny * 3))
    for pi in range(nx):
        for pj in range(ny):
            pflat = (pi * ny + pj) * 3
            nodes = [j2 * (nx + 1) + i2 for (i2, j2) in
                     ((pi, pj), (pi + 1, pj), (pi, pj + 1), (pi + 1, pj + 1))]
            for node in nodes:
                for d in range(3):
                    dist[9 * node + d, pflat + d] += 0.25
    return P, dist


def _corners(q, P, nx, ny):
    return (P @ q).reshape(nx + 1, ny + 1, 3)


def _aero_nodal(q, P, dist, nx, ny):
    Fp = diff_vlm.panel_forces_flat(_corners(q, P, nx, ny), nx, ny, VINF)
    return dist @ Fp


def _forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny, alpha=0.0, nsub=1):
    """Symplectic coupled rollout. STABILIZATION (1a): `nsub` sub-steps of dt/nsub fix the
    explicit-step CFL blow-up for stiff/over-flex designs; mass-proportional damping `alpha`
    (a -= alpha·dq) bleeds energy. alpha=0,nsub=1 reproduces the validated S3 rollout."""
    sdt = dt / nsub
    q, dq = q0.copy(), dq0.copy(); qs = [q.copy()]; as_ = []
    for _ in range(N * nsub):
        Qint, _, _ = _assemble(sh, q)
        rhs = _aero_nodal(q, P, dist, nx, ny) - Qint
        a = np.zeros(sh.ndof); a[free] = np.linalg.solve(Mff, rhs[free])
        a = a - alpha * dq                                       # mass-proportional damping
        dq = dq + sdt * a; q = q + sdt * dq
        qs.append(q.copy()); as_.append(a.copy())
    return qs, as_


def loss_and_grad(sh, q0, dq0, N, dt, free, w, nx, ny, alpha=0.0, nsub=1):
    P, dist = _index_maps(sh, nx, ny)
    Mff = sh.M[np.ix_(free, free)].toarray()
    sdt = dt / nsub
    qs, as_ = _forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny, alpha, nsub)
    L = float(w @ qs[-1])
    ndof = sh.ndof; rho, h = sh.rho, sh.h
    gE = np.zeros(sh.ne); gR = np.zeros(sh.ne)
    Mu = [_elem_mass_unit(sh, e) for e in range(sh.ne)]
    adj_q = w.copy(); adj_dq = np.zeros(ndof)
    for t in reversed(range(N * nsub)):
        aq1 = adj_q
        ad1 = adj_dq + sdt * aq1
        adj_a = sdt * ad1
        adj_dq_t = ad1 - alpha * adj_a                          # damping: a depends on dq_t
        adj_rhs = np.zeros(ndof)
        adj_rhs[free] = np.linalg.solve(Mff, adj_a[free])
        # --- aero coupling adjoint: adj_q += Pᵀ J_vlmᵀ distᵀ adj_rhs ---
        adj_Fp = dist.T @ adj_rhs
        Jv = diff_vlm.panel_jacobian(_corners(qs[t], P, nx, ny), nx, ny, VINF)
        adj_corners = Jv.T @ adj_Fp
        adj_q_aero = P.T @ adj_corners
        # --- structural adjoint + design grads ---
        adj_Qint = -adj_rhs
        _, Kt_t, per_e = _assemble(sh, qs[t])
        for e, (dofs, Qe) in enumerate(per_e):
            gE[e] += float(adj_Qint[dofs] @ (Qe / sh.E_scale_e[e]))
            gR[e] += float(-(adj_rhs[dofs] @ (Mu[e] * rho * h) @ as_[t][dofs]))
        adj_q = aq1 + Kt_t @ adj_Qint + adj_q_aero
        adj_dq = adj_dq_t
    return L, gE, gR


def verify(nx=3, ny=3, N=6, dt=1e-5, eps=1e-6, seed=0):
    sh = _build_shell(nx=nx, ny=ny)
    rng = np.random.default_rng(seed)
    ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))

    L, gE, gR = loss_and_grad(sh, q0, dq0, N, dt, free, w, nx, ny)

    def loss_only(Es_, Rs_):
        sh.set_distribution(E_scale=Es_, rho_scale=Rs_)
        P, dist = _index_maps(sh, nx, ny)
        Mff = sh.M[np.ix_(free, free)].toarray()
        qs, _ = _forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny)
        return float(w @ qs[-1])

    gE_fd = np.zeros(ne); gR_fd = np.zeros(ne)
    for e in range(ne):
        ep = Es.copy(); ep[e] += eps; em = Es.copy(); em[e] -= eps
        gE_fd[e] = (loss_only(ep, Rs) - loss_only(em, Rs)) / (2 * eps)
        rp = Rs.copy(); rp[e] += eps; rm = Rs.copy(); rm[e] -= eps
        gR_fd[e] = (loss_only(Es, rp) - loss_only(Es, rm)) / (2 * eps)

    relE = np.max(np.abs(gE - gE_fd)) / (np.max(np.abs(gE_fd)) + 1e-30)
    relR = np.max(np.abs(gR - gR_fd)) / (np.max(np.abs(gR_fd)) + 1e-30)
    okE, okR = relE < 1e-3, relR < 1e-3
    print(f"Differentiable COUPLED FSI design gradient (ANCF structure ⊗ VLM aero), "
          f"{ne} elements, {N}-step coupled rollout:")
    print(f"  ∂L/∂E_scale  (刚柔)  coupled adjoint vs FD: rel={relE:.2e}  -> {'PASS' if okE else 'FAIL'}")
    print(f"  ∂L/∂rho_scale(质量)  coupled adjoint vs FD: rel={relR:.2e}  -> {'PASS' if okR else 'FAIL'}")
    print(f"  -> design gradient flows through the FULL aeroelastic coupling (aero load on the "
          f"deformed geometry + structure + mass); the SHAC coupled design gradient (S3)")
    return okE and okR


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
