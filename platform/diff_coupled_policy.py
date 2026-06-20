"""Joint policy + structure SHAC backward (S4) — the control policy AND the (刚柔, 质量)
design are optimized through the SAME differentiable coupled-FSI rollout, one backward pass
giving BOTH ∂J/∂design and ∂J/∂θ_policy. This is the plan's §6 SHAC joint co-design on the
coupled physics (the cheap surrogate stays the DEV stand-in; here it is the real coupled
aeroelastic rollout from S3 + the stabilization from 1a).

A differentiable PD feedback policy applies a control force at a tip node during the coupled
rollout:
    u_t = −k_p·(q_t[c] − z*) − k_d·dq_t[c]        (θ = [k_p, k_d])
    a_t = M(ρ)⁻¹( F_aero(q_t) + F_ctrl_t − Qint(q_t; E) ) − α·dq_t
Objective:  J = ½‖q_N − ref‖² + λ_u·Σ u_t²  + μ·Σρ   (regulate + control effort + mass).

The adjoint chains: design (E,ρ) via the validated coupled adjoint, AND the policy θ via the
control-force adjoint (u enters the residual at the tip DOF and the effort term). Dual
backward = SHAC.

verify(): ∂J/∂刚柔, ∂J/∂质量, ∂J/∂k_p, ∂J/∂k_d all vs FD; then a joint Adam step on (design,
policy) reduces J — the joint co-design.
"""
from __future__ import annotations

import numpy as np

import diff_coupled_fsi as dc
import diff_vlm
from diff_struct_design import _build_shell

LO, HI = 0.4, 2.5


def _ctrl_dof(sh, nx, ny):
    node = ny * (nx + 1) + (nx // 2)         # a tip-row node (max span), mid-chord
    return 9 * node + 2                       # its z-position DOF


def rollout(sh, q0, dq0, theta, ref, ctx, N=12, dt=4e-5, nx=3, ny=3,
            alpha=6.0, nsub=4, lam_u=2e-2, mu=2e-3, want_grad=True):
    """Coupled+policy rollout. want_grad=True returns J and the dual gradient (gE,gR,gtheta);
    want_grad=False is forward-only (fast — no per-step complex-step Jacobian), returns J and
    Nones (the cheap evaluation MAP-Elites uses for the bulk of candidates)."""
    free, cdof, zt = ctx
    kp, kd = float(theta[0]), float(theta[1])
    P, dist = dc._index_maps(sh, nx, ny)
    Mff = sh.M[np.ix_(free, free)].toarray()
    sdt = dt / nsub
    ndof = sh.ndof; rho, h = sh.rho, sh.h
    # forward (store pre-update q_t, dq_t, control u_t, and the raw M⁻¹·rhs = a_raw)
    q, dq = q0.copy(), dq0.copy()
    qs, dqs, us, araw = [], [], [], []
    u2 = 0.0
    for _ in range(N * nsub):
        if want_grad:
            qs.append(q.copy()); dqs.append(dq.copy())
        Qint, _, _ = dc._assemble(sh, q)
        u = -kp * (q[cdof] - zt) - kd * dq[cdof]
        u2 += u * u
        if want_grad:
            us.append(u)
        rhs = dc._aero_nodal(q, P, dist, nx, ny) - Qint
        rhs[cdof] += u
        a = np.zeros(ndof); a[free] = np.linalg.solve(Mff, rhs[free])
        if want_grad:
            araw.append(a.copy())                   # undamped M⁻¹·rhs (for the mass adjoint)
        a = a - alpha * dq
        dq = dq + sdt * a; q = q + sdt * dq
    J = 0.5 * float((q - ref) @ (q - ref)) + lam_u * u2 + mu * float(np.sum(sh.rho_scale_e))
    if not want_grad:
        return J, None, None, None
    # backward (dual: design + policy)
    gE = np.zeros(sh.ne); gR = np.zeros(sh.ne); gkp = 0.0; gkd = 0.0
    Mu = [dc._elem_mass_unit(sh, e) for e in range(sh.ne)]
    adj_q = (q - ref).copy(); adj_dq = np.zeros(ndof)
    for t in reversed(range(N * nsub)):
        qt, dqt, ut = qs[t], dqs[t], us[t]
        aq1 = adj_q
        ad1 = adj_dq + sdt * aq1
        adj_a = sdt * ad1
        adj_dq_t = ad1 - alpha * adj_a
        adj_rhs = np.zeros(ndof)
        adj_rhs[free] = np.linalg.solve(Mff, adj_a[free])
        # control force u: cotangent from the residual (adj_rhs[cdof]) + effort (2 λu u)
        adj_u = adj_rhs[cdof] + 2.0 * lam_u * ut
        gkp += adj_u * (-(qt[cdof] - zt))
        gkd += adj_u * (-dqt[cdof])
        adj_q_extra = np.zeros(ndof); adj_q_extra[cdof] = adj_u * (-kp)   # u's q-dependence
        adj_dq_t[cdof] += adj_u * (-kd)                                    # u's dq-dependence
        # aero + structure + design grads
        adj_Fp = dist.T @ adj_rhs
        Jv = diff_vlm.panel_jacobian(dc._corners(qt, P, nx, ny), nx, ny, dc.VINF)
        adj_q_aero = P.T @ (Jv.T @ adj_Fp)
        adj_Qint = -adj_rhs
        _, Kt_t, per_e = dc._assemble(sh, qt)
        for e, (dofs, Qe) in enumerate(per_e):
            gE[e] += float(adj_Qint[dofs] @ (Qe / sh.E_scale_e[e]))
            gR[e] += float(-(adj_rhs[dofs] @ (Mu[e] * rho * h) @ araw[t][dofs]))
        adj_q = aq1 + Kt_t @ adj_Qint + adj_q_aero + adj_q_extra
        adj_dq = adj_dq_t
    gR = gR + mu
    return J, gE, gR, np.array([gkp, gkd])


def verify(nx=3, ny=3, seed=0):
    sh = _build_shell(nx=nx, ny=ny)
    ne = sh.ne
    rng = np.random.default_rng(seed)
    free = np.array(sorted(set(range(sh.ndof)) - set(sh._bc_dofs)))
    ref = sh.q.copy()
    q0 = sh.q.copy(); q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(sh.ndof); dq0[free] = 5e-3 * rng.standard_normal(len(free))
    cdof = _ctrl_dof(sh, nx, ny); zt = ref[cdof]
    ctx = (free, cdof, zt)
    Es = np.exp(0.15 * rng.standard_normal(ne)); Rs = np.exp(0.15 * rng.standard_normal(ne))
    theta = np.array([5.0, 0.5])
    sh.set_distribution(E_scale=Es, rho_scale=Rs)

    J0, gE, gR, gth = rollout(sh, q0, dq0, theta, ref, ctx, nx=nx, ny=ny)

    def Jonly(E_, R_, th):
        sh.set_distribution(E_scale=E_, rho_scale=R_)
        return rollout(sh, q0, dq0, th, ref, ctx, nx=nx, ny=ny)[0]
    eps = 1e-5
    # design grads (spot-check a few elements) + policy grads
    idx = [0, ne // 2, ne - 1]
    relE = relR = 0.0
    for e in idx:
        ep = Es.copy(); ep[e] += eps; em = Es.copy(); em[e] -= eps
        fd = (Jonly(ep, Rs, theta) - Jonly(em, Rs, theta)) / (2 * eps)
        relE = max(relE, abs(gE[e] - fd) / (abs(fd) + 1e-8))
        rp = Rs.copy(); rp[e] += eps; rm = Rs.copy(); rm[e] -= eps
        fd = (Jonly(Es, rp, theta) - Jonly(Es, rm, theta)) / (2 * eps)
        relR = max(relR, abs(gR[e] - fd) / (abs(fd) + 1e-8))
    gth_fd = np.zeros(2)
    for k in range(2):
        tp = theta.copy(); tp[k] += eps; tm = theta.copy(); tm[k] -= eps
        gth_fd[k] = (Jonly(Es, Rs, tp) - Jonly(Es, Rs, tm)) / (2 * eps)
    relT = np.max(np.abs(gth - gth_fd)) / (np.max(np.abs(gth_fd)) + 1e-8)
    okg = relE < 2e-3 and relR < 2e-3 and relT < 2e-3
    print("Joint policy + structure SHAC backward (S4):")
    print(f"  dual-backward gradients vs FD: ∂刚柔 rel={relE:.2e}  ∂质量 rel={relR:.2e}  "
          f"∂policy rel={relT:.2e}  -> {'PASS' if okg else 'FAIL'}")

    # joint Adam: design + policy together
    Es = np.full(ne, 1.0); Rs = np.full(ne, 1.0); th = np.array([2.0, 0.2])
    mE = np.zeros(ne); vE = np.zeros(ne); mR = np.zeros(ne); vR = np.zeros(ne)
    mT = np.zeros(2); vT = np.zeros(2)
    Js = []
    for it in range(25):
        sh.set_distribution(E_scale=Es, rho_scale=Rs)
        J, gE, gR, gth = rollout(sh, q0, dq0, th, ref, ctx, nx=nx, ny=ny)
        Js.append(J)
        for (x, g, m, v, lr) in [(Es, gE, mE, vE, 0.05), (Rs, gR, mR, vR, 0.03)]:
            m[:] = 0.9 * m + 0.1 * g; v[:] = 0.999 * v + 0.001 * g * g
            x -= lr * (m / (1 - 0.9 ** (it + 1))) / (np.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
        mT[:] = 0.9 * mT + 0.1 * gth; vT[:] = 0.999 * vT + 0.001 * gth * gth
        th = th - 0.1 * (mT / (1 - 0.9 ** (it + 1))) / (np.sqrt(vT / (1 - 0.999 ** (it + 1))) + 1e-8)
        Es = np.clip(Es, LO, HI); Rs = np.clip(Rs, LO, HI); th = np.clip(th, 0.0, 50.0)
    ok = okg and Js[-1] < Js[0]
    print(f"  joint Adam (design + policy): J {Js[0]:.3e} -> {Js[-1]:.3e}  "
          f"(θ=[{th[0]:.2f},{th[1]:.2f}], mean刚柔={Es.mean():.2f} mean质量={Rs.mean():.2f})")
    print(f"  -> {'PASS' if ok else 'FAIL'}: policy AND (刚柔,质量) co-designed through one "
          f"coupled-FSI backward (SHAC)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
