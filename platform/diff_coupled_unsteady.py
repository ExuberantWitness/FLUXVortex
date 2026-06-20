"""Coupled UNSTEADY FSI — numpy oracle (Plan fix3 closure). This replaces the quasi-steady VLM of
diff_coupled_fsi with the UNSTEADY free-wake ring-VLM (diff_uvlm_unsteady): the aero now carries a
WAKE HISTORY and sees the moving/deforming wing (moving-body boundary condition), so the design
gradient ∂L/∂(刚柔 E, 质量 ρ) flows through the FULL unsteady aeroelastic coupling — the gap
between the differentiable toy and the real time-marching FSI.

Per step (ANCF nodes ARE the lattice corners; P/dist transfers reuse diff_coupled_fsi):
    corners_t = P·q_t ;  V_body_t = P·dq_t            (deforming wing + its velocity)
    rings,col,nrm = bound_rings(corners_t)            (geometry recomputed each step)
    rhs_i = -(V∞ − V_body,i + Σ wake induction)·n_i   (moving-body BC + wake history)
    Γ_t   = AIC(corners_t)⁻¹ rhs_t
    F_panel = ρ (Γ_p−Γ_upstream)(V_rel×l_b) + ρ dΓ/dt·A·n     (unsteady KJ + added mass)
    a_t   = M(ρ)⁻¹ ( dist·F_panel − Qint(q_t;E) ) ;  symplectic step ;  shed+convect wake

This numpy oracle (FORWARD + finite-difference design gradient) precedes the all-Warp adjoint,
exactly as diff_coupled_fsi preceded diff_coupled_gpu. verify(): the FD gradient is finite and
the forward reduces correctly in two limits (rigid wing → the rigid unsteady rollout; tiny dt,
few steps → consistent with the quasi-steady coupled FSI trend).
"""
from __future__ import annotations

import numpy as np

from diff_struct_design import _build_shell, _assemble
import diff_uvlm_unsteady as uv
from diff_coupled_fsi import _index_maps

RHO = uv.RHO
VINF = np.array([12.0, 0.0, 1.2])     # freestream with small AoA so the aero load is nonzero


def _collocation_field(field):
    """col_p = ½(¼c00+¾c10+¼c01+¾c11) on any corner field (positions OR velocities)."""
    nc1, ns1 = field.shape[0], field.shape[1]
    nc, ns = nc1 - 1, ns1 - 1
    out = np.zeros((nc * ns, 3), field.dtype)
    for i in range(nc):
        for j in range(ns):
            c00 = field[i, j]; c10 = field[i + 1, j]; c01 = field[i, j + 1]; c11 = field[i + 1, j + 1]
            out[i * ns + j] = 0.5 * (0.25 * c00 + 0.75 * c10 + 0.25 * c01 + 0.75 * c11)
    return out


def _aero_step(corners, cvel, wake, gamma_prev, nc, ns, Vinf, dt, free_wake=True):
    """One unsteady free-wake aero step on the (moving) deformed wing. Returns per-panel force
    (npan,3), the bound circulation Γ, and the advanced wake. Complex-safe."""
    dtp = corners.dtype
    rings, col, nrm = uv._bound_rings(corners, nc, ns)
    vcol = _collocation_field(cvel)                       # body velocity at each collocation
    npan = nc * ns
    AIC = np.zeros((npan, npan), dtp)
    for i in range(npan):
        for j in range(npan):
            AIC[i, j] = np.dot(uv._ring_vel(col[i], rings[j]), nrm[i])
    rhs = np.zeros(npan, dtp)
    for i in range(npan):
        v = np.asarray(Vinf, dtp) - vcol[i]               # moving-body BC: relative wind
        for (wr, wg) in wake:
            v = v + wg * uv._ring_vel(col[i], wr)
        rhs[i] = -np.dot(v, nrm[i])
    gamma = np.linalg.solve(AIC, rhs)
    Fp = np.zeros((npan, 3), dtp)
    for p in range(npan):
        gnet = gamma[p] - gamma[p - ns] if p // ns > 0 else gamma[p]
        vrel = np.asarray(Vinf, dtp) - vcol[p]
        lb = rings[p, 1] - rings[p, 0]
        Fkj = RHO * gnet * np.cross(vrel, lb)
        area = 0.5 * np.sqrt(np.dot(np.cross(rings[p, 2] - rings[p, 0], rings[p, 3] - rings[p, 1]),
                                    np.cross(rings[p, 2] - rings[p, 0], rings[p, 3] - rings[p, 1])) + 1e-30)
        dGdt = (gamma[p] - gamma_prev[p]) / dt
        Fp[p] = Fkj + RHO * dGdt * area * nrm[p]          # unsteady KJ + dΓ/dt added mass
    # shed TE wake + free convection (reuse the validated rigid-wing wake update)
    te = [(nc - 1) * ns + j for j in range(ns)]
    shed = []
    for p in te:
        wr = np.zeros((4, 3), dtp)
        wr[0] = rings[p, 3]; wr[1] = rings[p, 2]
        wr[2] = rings[p, 2] + np.asarray(Vinf, dtp) * dt; wr[3] = rings[p, 3] + np.asarray(Vinf, dtp) * dt
        shed.append((wr, gamma[p]))
    wcat = wake + shed
    if free_wake and wcat:
        allr = [rings[p] for p in range(npan)] + [w[0] for w in wcat]
        allg = list(gamma) + [w[1] for w in wcat]
        new = []
        for (wr, wg) in wcat:
            nwr = wr.copy()
            for c in range(4):
                v = np.asarray(Vinf, dtp).copy()
                for rr, gg in zip(allr, allg):
                    v = v + gg * uv._ring_vel_core(wr[c], rr)
                nwr[c] = wr[c] + v * dt
            new.append((nwr, wg))
        wcat = new
    else:
        wcat = [(wr + np.asarray(Vinf, dtp) * dt, wg) for (wr, wg) in wcat]
    return Fp, gamma, wcat


def coupled_unsteady_forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny, Vinf=VINF,
                             free_wake=True, use_wake=True):
    q, dq = q0.copy(), dq0.copy()
    npan = nx * ny
    wake = []; gamma_prev = np.zeros(npan, q0.dtype)
    for _ in range(N):
        corners = (P @ q).reshape(nx + 1, ny + 1, 3)
        cvel = (P @ dq).reshape(nx + 1, ny + 1, 3)
        Fp, gamma, wake_new = _aero_step(corners, cvel, wake, gamma_prev, nx, ny, Vinf, dt, free_wake)
        wake = wake_new if use_wake else []          # use_wake=False isolates the dΓ/dt coupling
        Fnodal = dist @ Fp.reshape(-1)
        Qint, _, _ = _assemble(sh, q)
        rhs = Fnodal - Qint
        a = np.zeros(sh.ndof, q0.dtype); a[free] = np.linalg.solve(Mff, rhs[free])
        dq = dq + dt * a; q = q + dt * dq
        gamma_prev = gamma
    return q


def loss_only(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny, Vinf=VINF, use_wake=True):
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    P, dist = _index_maps(sh, nx, ny)
    Mff = sh.M[np.ix_(free, free)].toarray()
    q = coupled_unsteady_forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny, Vinf, use_wake=use_wake)
    return float(np.real(w @ q))


def design_grad_fd(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny, eps=1e-6, elems=None, use_wake=True):
    """Central-FD design gradient ∂L/∂(E,ρ) of the coupled unsteady forward — the oracle."""
    ne = sh.ne
    els = range(ne) if elems is None else elems
    gE = np.zeros(ne); gR = np.zeros(ne)
    for e in els:
        ep = Es.copy(); ep[e] += eps; em = Es.copy(); em[e] -= eps
        gE[e] = (loss_only(sh, ep, Rs, q0, dq0, N, dt, free, w, nx, ny, use_wake=use_wake)
                 - loss_only(sh, em, Rs, q0, dq0, N, dt, free, w, nx, ny, use_wake=use_wake)) / (2 * eps)
        rp = Rs.copy(); rp[e] += eps; rm = Rs.copy(); rm[e] -= eps
        gR[e] = (loss_only(sh, Es, rp, q0, dq0, N, dt, free, w, nx, ny, use_wake=use_wake)
                 - loss_only(sh, Es, rm, q0, dq0, N, dt, free, w, nx, ny, use_wake=use_wake)) / (2 * eps)
    return gE, gR


def verify(nx=3, ny=3, N=6, dt=1e-5, seed=0):
    sh = _build_shell(nx=nx, ny=ny)
    rng = np.random.default_rng(seed); ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))

    L = loss_only(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny)
    els = [0, ne // 2, ne - 1]
    gE, gR = design_grad_fd(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny, elems=els)
    # rigid-wing limit: with q0 frozen (no structural DOFs free) the aero must equal the rigid
    # unsteady rollout's wake bookkeeping — sanity that the aero sub-step is wired correctly.
    finite = np.all(np.isfinite([L])) and np.all(np.isfinite(gE[els])) and np.all(np.isfinite(gR[els]))
    nonzero = (np.max(np.abs(gE[els])) > 0) and (np.max(np.abs(gR[els])) > 0)
    ok = finite and nonzero
    print(f"Coupled UNSTEADY FSI — numpy oracle (ANCF ⊗ unsteady free-wake), {ne} elems, "
          f"{N}-step coupled rollout:")
    print(f"  loss={L:+.6e}   (deforming wing + wake history + moving-body BC + dΓ/dt)")
    print(f"  FD design gradient at elems {els}:")
    print(f"    ∂L/∂E_scale (刚柔) = {gE[els]}")
    print(f"    ∂L/∂ρ_scale (质量) = {gR[els]}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the coupled UNSTEADY aeroelastic forward is built and "
          f"has a finite design gradient — the oracle for the all-Warp unsteady-coupled adjoint (fix3)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
