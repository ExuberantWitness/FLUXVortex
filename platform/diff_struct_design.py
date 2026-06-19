"""Differentiable per-element (刚柔 + 质量) STRUCTURAL design gradient — Stage 2 of the
real-FSI co-design (the piece that makes BOTH the stiffness field AND the mass field
differentiable design variables on the validated ANCF shell).

The earlier diff machinery (diff_step) differentiates the structural rollout w.r.t. state
via the exact K_t adjoint, but treats the mass matrix as CONSTANT — so mass (ρ) was not a
differentiable design variable. Here we add BOTH per-element design adjoints, exploiting
that the ANCF internal force is LINEAR in the per-element stiffness scale and the mass block
is LINEAR in the per-element density scale:

  a_t = M(ρ)⁻¹ (F − Qint(q_t; E))                       (symplectic structural step)
  ∂L/∂E_e  += adj_Qint_t[e] · (Qe(q_t)/E_e)             (Qint ∝ E_e  per element)
  ∂L/∂ρ_e  += −adj_rhs_t[e]ᵀ (M̃_e·ρ·h) a_t[e]           (M_e ∝ ρ_e ; M̃_e unit-density block)

with adj_rhs_t = M⁻¹ adj_a_t and adj_Qint_t = −adj_rhs_t (the adjoint of the mass solve and
the residual), and the state adjoint adj_q += K_t(q_t)·adj_Qint chaining over the rollout.

verify(): ∂L/∂E and ∂L/∂ρ from this adjoint vs central finite differences (re-running the
real shell forward with set_distribution) — the red line for differentiable 刚柔+质量 design.
This is the STRUCTURAL design gradient; coupling the UVLM aero adjoint + a policy is the
remaining SHAC work (Stage 3/4, A100).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
for p in (_SRC, _TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from fluxvortex.ancf_shell import (ANCFShell, _gauss_legendre, NDOF_ELEM,  # noqa: E402
                                   NDOF_NODE)


def _build_shell(nx=4, ny=3, L=0.4, W=0.3, h=1.5e-3, rho=1200.0, E=1.0e6, nu=0.3):
    xs = np.linspace(0, L, nx + 1); ys = np.linspace(0, W, ny + 1)
    nodes = np.array([[x, y, 0.0] for y in ys for x in xs])
    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            quads.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    quads = np.array(quads)
    sh = ANCFShell(nodes, quads, h, rho, E, E, nu)
    return sh


def _assemble(shell, q):
    """Global Qint, K_t, and per-element (dofs, Qe) at configuration q."""
    ndof = shell.ndof
    Qint = np.zeros(ndof); Kt = np.zeros((ndof, ndof)); per_e = []
    for e in range(shell.ne):
        Qe, Kte = shell._elem_forces_and_tangent(e, q)
        dofs = shell._elem_dofs(e)
        Qint[dofs] += Qe
        Kt[np.ix_(dofs, dofs)] += Kte
        per_e.append((np.asarray(dofs), Qe))
    return Qint, Kt, per_e


def _elem_mass_unit(shell, e):
    """Unit-density element mass block M̃_e (= ∫ Sᵀ S detJ), so M_e = M̃_e · ρ_e · h."""
    ed = shell._elems[e]
    pts, wts = _gauss_legendre(shell.n_gauss)
    Me = np.zeros((NDOF_ELEM, NDOF_ELEM))
    for i in range(shell.n_gauss):
        for j in range(shell.n_gauss):
            S = ed.S[i, j]
            Me += wts[i] * wts[j] * ed.detJ * (S.T @ S)
    return Me


def _forward(shell, q0, dq0, F, N, dt, free, Mff):
    q, dq = q0.copy(), dq0.copy()
    qs = [q.copy()]; as_ = []
    for _ in range(N):
        Qint, _, _ = _assemble(shell, q)
        rhs = F - Qint
        a = np.zeros(shell.ndof)
        a[free] = np.linalg.solve(Mff, rhs[free])
        dq = dq + dt * a
        q = q + dt * dq
        qs.append(q.copy()); as_.append(a.copy())
    return qs, as_


def loss_and_grad(shell, q0, dq0, F, N, dt, free, w):
    """L = w·q_N and the per-element design gradients ∂L/∂E_scale, ∂L/∂rho_scale (adjoint)."""
    Mff = shell.M[np.ix_(free, free)].toarray()
    qs, as_ = _forward(shell, q0, dq0, F, N, dt, free, Mff)
    L = float(w @ qs[-1])
    ndof = shell.ndof
    rho, h = shell.rho, shell.h
    gE = np.zeros(shell.ne); gR = np.zeros(shell.ne)
    Mu = [_elem_mass_unit(shell, e) for e in range(shell.ne)]
    adj_q = w.copy(); adj_dq = np.zeros(ndof)
    for t in reversed(range(N)):
        aq1 = adj_q                                   # cotangent of q_{t+1}
        ad1 = adj_dq + dt * aq1                        # cotangent of dq_{t+1} (+= dt·adj_q)
        adj_dq_t = ad1.copy()
        adj_a = dt * ad1
        adj_rhs = np.zeros(ndof)
        adj_rhs[free] = np.linalg.solve(Mff, adj_a[free])     # M symmetric -> same solve
        adj_Qint = -adj_rhs
        _, Kt_t, per_e = _assemble(shell, qs[t])
        for e, (dofs, Qe) in enumerate(per_e):
            gE[e] += float(adj_Qint[dofs] @ (Qe / shell.E_scale_e[e]))   # Qint ∝ E_e
            md = dofs
            gR[e] += float(-(adj_rhs[md] @ (Mu[e] * rho * h) @ as_[t][md]))  # M_e ∝ ρ_e
        adj_q = aq1 + Kt_t @ adj_Qint                 # state chain: adj_q += K_t·adj_Qint
        adj_dq = adj_dq_t
    return L, gE, gR


def verify(N=20, dt=2e-5, eps=1e-6, seed=0):
    sh = _build_shell()
    rng = np.random.default_rng(seed)
    ne = sh.ne
    Es = np.exp(0.25 * rng.standard_normal(ne))       # per-element 刚柔 field
    Rs = np.exp(0.25 * rng.standard_normal(ne))       # per-element 质量 field
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy()
    q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-2 * rng.standard_normal(len(free))
    F = np.zeros(ndof); F[free] = 0.5 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))

    L, gE, gR = loss_and_grad(sh, q0, dq0, F, N, dt, free, w)

    def loss_only(Es_, Rs_):
        sh.set_distribution(E_scale=Es_, rho_scale=Rs_)
        Mff = sh.M[np.ix_(free, free)].toarray()
        qs, _ = _forward(sh, q0, dq0, F, N, dt, free, Mff)
        return float(w @ qs[-1])

    gE_fd = np.zeros(ne); gR_fd = np.zeros(ne)
    for e in range(ne):
        ep = Es.copy(); ep[e] += eps; em = Es.copy(); em[e] -= eps
        gE_fd[e] = (loss_only(ep, Rs) - loss_only(em, Rs)) / (2 * eps)
        rp = Rs.copy(); rp[e] += eps; rm = Rs.copy(); rm[e] -= eps
        gR_fd[e] = (loss_only(Es, rp) - loss_only(Es, rm)) / (2 * eps)

    relE = np.max(np.abs(gE - gE_fd)) / (np.max(np.abs(gE_fd)) + 1e-30)
    relR = np.max(np.abs(gR - gR_fd)) / (np.max(np.abs(gR_fd)) + 1e-30)
    # 1e-3 is the repo's established tolerance for the K_t-adjoint path (diff_step.verify):
    # the validated tangent matches FD-dQint to ~7e-6 and the FD loss is solve-noise-limited.
    okE, okR = relE < 1e-3, relR < 1e-3
    print(f"Differentiable structural design gradient (per-element 刚柔 + 质量), "
          f"{ne} elements, {N}-step rollout:")
    print(f"  ∂L/∂E_scale  (刚柔)  adjoint vs FD: rel={relE:.2e}  -> {'PASS' if okE else 'FAIL'}")
    print(f"  ∂L/∂rho_scale(质量)  adjoint vs FD: rel={relR:.2e}  -> {'PASS' if okR else 'FAIL'}")
    print(f"  -> 刚柔 AND 质量 distribution are both differentiable design variables on the "
          f"validated ANCF (mass adjoint added; M_e ∝ ρ_e, Qint ∝ E_e)")
    return okE and okR


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
