"""P2-S1 geometrically-exact 3D beam (Warp) — verification ladder gates 1-5.

Gate 1a: internal force == dE/dq (central FD of the GPU energy), rel < 1e-6.
         Non-circular: force comes from the analytic variation, energy is a
         separate scalar kernel — a wrong Jacobian convention fails this gate.
Gate 1b: tangent at q=0 == independently hand-derived linear Timoshenko(1GP) Ke.
Gate 2 : rigid translation+rotation (30/90/150 deg) -> zero strain/force.
Gate 3a: cantilever tip-force deflection vs Timoshenko analytic (<0.5%).
Gate 3b: pure-moment roll-up to a quarter circle (geometrically exact benchmark,
         within the |psi|<pi chart) — tip position error < 1% of L.
Gates 4/5 (modal, Newmark) live in platform/beam3d_solver.py (need the solver).

Run: cd FLUXV && python tests/test_beam3d_warp.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import warp as wp

import fluxvortex.warp_fsi.config as cfg
from fluxvortex.warp_fsi import kernels_beam3d as kb

DEV = cfg.DEVICE
NP = cfg.NP_DTYPE
SEC = kb.MAIN_SPAR


def _wp_q(q_np):
    return wp.array(np.ascontiguousarray(q_np, dtype=NP), dtype=cfg.DTYPE, device=DEV)


def build_polyline(points, section=SEC):
    nodes = np.asarray(points, float)
    elems = np.array([[i, i + 1] for i in range(len(nodes) - 1)])
    return kb.Beam3DConstants(nodes, elems, section, device=DEV)


def build_cantilever(nel, L=0.8, section=SEC):
    nodes = np.zeros((nel + 1, 3))
    nodes[:, 0] = np.linspace(0.0, L, nel + 1)
    C = build_polyline(nodes, section)
    C.set_bc([0], fix_rot=True)
    return C


def rot(axis, ang):
    axis = np.array(axis, dtype=float)          # copy! (never mutate the caller's q)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def log_so3(R):
    ct = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(ct)
    if th < 1e-10:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def jr(psi):
    th = np.linalg.norm(psi)
    K = np.array([[0, -psi[2], psi[1]], [psi[2], 0, -psi[0]], [-psi[1], psi[0], 0]])
    if th < 1e-6:
        return np.eye(3) - K / 2 + K @ K / 6
    return (np.eye(3) - (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K)


# ─── Newton static solver (test-only, dense) ─────────────────────────────────

def newton_static(C, fext_fn, q0=None, tol=1e-8, max_it=30):
    """Solve Q_int(q) = F_ext(q) on free DOFs. Full-step Newton (convergence is
    non-monotone in |r| for this element — do NOT line-search on the residual),
    with an explosion guard. Dense, B=1, test-only."""
    q = np.zeros(C.ndof) if q0 is None else q0.copy()
    free = np.array(sorted(set(range(C.ndof)) - C.bc_dofs))
    load = max(1.0, np.linalg.norm(fext_fn(q)))
    hj = 1e-7

    def load_jac(qv):
        """FD Jacobian of the (possibly configuration-dependent) external load."""
        J = np.zeros((C.ndof, C.ndof))
        base = fext_fn(qv)
        cols = np.nonzero(np.abs(base) > 0)[0]
        # only rotation DOFs of loaded nodes can carry configuration dependence here
        probe = set()
        for d in cols:
            n = d // 6
            probe.update(range(6 * n + 3, 6 * n + 6))
        for d in sorted(probe):
            qp = qv.copy(); qp[d] += hj
            qm = qv.copy(); qm[d] -= hj
            J[:, d] = (fext_fn(qp) - fext_fn(qm)) / (2 * hj)
        return J

    for it in range(max_it):
        Q = kb.beam_internal_force(_wp_q(q[None, :]), C).numpy()[0]
        r = fext_fn(q) - Q
        rn = np.linalg.norm(r[free])
        if rn < tol * load:
            return q, it
        if not np.isfinite(rn) or rn > 1e10 * load:
            raise RuntimeError(f"Newton diverged: |r|={rn:.3e}")
        Kblk = kb.assemble_beam_kblocks(_wp_q(q[None, :]), C, symmetrize=False).numpy()[0]
        K = kb.scatter_beam_global(Kblk, C.edofs_np, C.ndof) - load_jac(q)
        dq = np.zeros(C.ndof)
        dq[free] = np.linalg.solve(K[np.ix_(free, free)], r[free])
        q += dq
    raise RuntimeError(f"Newton stalled: |r|={rn:.3e}")


# ─── Gate 1a: force == dE/dq ─────────────────────────────────────────────────

def test_force_energy_fd():
    rng = np.random.default_rng(7)
    pts = np.array([[0, 0, 0], [0.2, 0.02, -0.01], [0.4, -0.03, 0.02],
                    [0.55, 0.1, 0.08], [0.7, 0.12, 0.2]])   # general polyline
    C = build_polyline(pts)
    B = 3
    q = np.zeros((B, C.ndof))
    for b in range(B):
        u = rng.normal(0, 0.05, (C.nn, 3))
        ps = rng.normal(0, 0.4, (C.nn, 3))
        q[b] = np.concatenate([np.concatenate([u[n], ps[n]]) for n in range(C.nn)])
    qw = _wp_q(q)
    Qf = kb.beam_internal_force(qw, C).numpy()
    h = 1e-6
    worst = 0.0
    scale = np.abs(Qf).max()
    for b in range(B):
        for d in rng.choice(C.ndof, size=24, replace=False):
            qp = q.copy(); qp[b, d] += h
            qm = q.copy(); qm[b, d] -= h
            Ep = kb.beam_energy_total(_wp_q(qp), C).numpy()[b]
            Em = kb.beam_energy_total(_wp_q(qm), C).numpy()[b]
            fd = (Ep - Em) / (2 * h)
            rel = abs(fd - Qf[b, d]) / max(abs(fd), 1e-8 * scale)
            worst = max(worst, rel)
    assert worst < 1e-6, f"force!=dE/dq: worst rel {worst:.3e}"
    print(f"Gate 1a PASS: force==dE/dq, worst rel {worst:.3e} (24 dofs x {B} envs, |psi| up to ~1 rad)")


# ─── Gate 1b: K(0) vs hand-derived linear 1GP-Timoshenko Ke ──────────────────

def test_k0_linear():
    C = build_cantilever(1, L=0.8)
    L = 0.8
    qw = _wp_q(np.zeros((1, C.ndof)))
    Kfd = kb.assemble_beam_kblocks(qw, C, symmetrize=False).numpy()[0, 0]
    # independent linearized derivation: Gam = (u2-u1)/L + e1 x (ps1+ps2)/2 ;
    # kap = (ps2-ps1)/L ; Ke = L B^T diag(CF,CM) B
    e1x = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0.]])   # skew(e1)
    Bm = np.zeros((6, 12))
    Bm[0:3, 0:3] = -np.eye(3) / L; Bm[0:3, 6:9] = np.eye(3) / L
    Bm[0:3, 3:6] = 0.5 * e1x;      Bm[0:3, 9:12] = 0.5 * e1x
    Bm[3:6, 3:6] = -np.eye(3) / L; Bm[3:6, 9:12] = np.eye(3) / L
    D = np.diag(np.concatenate([C.CF_np[0], C.CM_np[0]]))
    Ke = L * Bm.T @ D @ Bm
    rel = np.abs(Kfd - Ke).max() / np.abs(Ke).max()
    assert rel < 1e-7, f"K(0) != linear Ke: rel {rel:.3e}"
    print(f"Gate 1b PASS: K(0) matches hand-derived linear 1GP-Timoshenko Ke, rel {rel:.3e}")


# ─── Gate 2: rigid motion -> zero force ──────────────────────────────────────

def test_rigid_motion():
    pts = np.array([[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0.4, 0.15, 0], [0.4, 0.3, 0.1]])
    C = build_polyline(pts)  # L-shaped frame (exercises Lam0 != I)
    t = np.array([0.03, -0.02, 0.05])
    worst = 0.0
    for deg in (30.0, 90.0, 150.0):
        R = rot([0.3, 0.9, 0.5], np.deg2rad(deg))
        psi = log_so3(R)
        q = np.zeros(C.ndof)
        for n in range(C.nn):
            q[6*n:6*n+3] = (R - np.eye(3)) @ C.nodes_np[n] + t
            q[6*n+3:6*n+6] = psi
        Q = kb.beam_internal_force(_wp_q(q[None, :]), C).numpy()[0]
        rel = np.abs(Q).max() / SEC.EA
        worst = max(worst, rel)
        assert rel < 1e-12, f"rigid {deg}deg: |Q|/EA = {rel:.3e}"
    print(f"Gate 2 PASS: rigid translation+rotation 30/90/150deg, max|Q|/EA {worst:.3e}")


# ─── Gate 3a: cantilever tip force vs Timoshenko analytic ────────────────────

def test_cantilever_static():
    L, nel, F = 0.8, 16, 1.0
    C = build_cantilever(nel, L=L)
    tip_dof = C.dof_map_np[C.nn - 1][2]        # z-translation at tip

    def fext(q):
        f = np.zeros(C.ndof); f[tip_dof] = F
        return f

    q, it = newton_static(C, fext)
    delta = q[tip_dof]
    ana = F * L**3 / (3 * SEC.EI) + F * L / SEC.GAks
    rel = abs(delta - ana) / ana
    assert rel < 0.005, f"cantilever: {delta:.6e} vs {ana:.6e} rel {rel:.3e}"
    print(f"Gate 3a PASS: tip deflection {delta*1e3:.4f} mm vs analytic {ana*1e3:.4f} mm "
          f"(rel {rel:.2e}, {it} Newton its)")


# ─── Gate 3b: pure-moment roll-up (quarter circle) ───────────────────────────

def test_rollup_quarter():
    L, nel = 0.8, 32
    C = build_cantilever(nel, L=L)
    alpha = np.pi / 2                            # tip rotation (within |psi|<pi chart)
    M_full = alpha * SEC.EI / L                  # dead moment about +z
    tipn = C.nn - 1

    q = np.zeros(C.ndof)
    for lam in np.linspace(1.0 / 32, 1.0, 32):   # 32 load increments
        m = np.array([0.0, 0.0, lam * M_full])

        def fext(qv):
            f = np.zeros(C.ndof)
            psi = qv[6*tipn+3:6*tipn+6]
            R = rot(psi, np.linalg.norm(psi)) if np.linalg.norm(psi) > 0 else np.eye(3)
            f[6*tipn+3:6*tipn+6] = jr(psi).T @ R.T @ m   # moment conjugate to psi
            return f

        q, _ = newton_static(C, fext, q0=q, tol=1e-8)

    R_arc = L / alpha
    tip_ana = np.array([R_arc * np.sin(alpha), R_arc * (1 - np.cos(alpha)), 0.0])
    tip_num = C.nodes_np[tipn] + q[6*tipn:6*tipn+3]
    err = np.linalg.norm(tip_num - tip_ana) / L
    psi_tip = q[6*tipn+3:6*tipn+6]
    rot_err = abs(np.linalg.norm(psi_tip) - alpha) / alpha
    assert err < 0.01, f"roll-up tip pos err {err:.3e}"
    assert rot_err < 0.01, f"roll-up tip rot err {rot_err:.3e}"
    print(f"Gate 3b PASS: quarter-circle roll-up, tip pos err {err:.2e} of L, "
          f"tip rotation {np.linalg.norm(psi_tip):.6f} vs {alpha:.6f}")


# ─── Gate 4: analytic cantilever modes ───────────────────────────────────────

def test_modal():
    from scipy.linalg import eigh
    L, nel = 0.8, 32
    C = build_cantilever(nel, L=L)
    Kblk = kb.assemble_beam_kblocks(_wp_q(np.zeros((1, C.ndof))), C,
                                    symmetrize=False).numpy()[0]
    K = kb.scatter_beam_global(Kblk, C.edofs_np, C.ndof)
    M = kb.scatter_beam_global(C.Me_np, C.edofs_np, C.ndof)
    free = np.array(sorted(set(range(C.ndof)) - C.bc_dofs))
    w = np.sqrt(np.abs(eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
                            eigvals_only=True)))
    ana = {
        "bend1": 1.8751**2 * np.sqrt(SEC.EI / (SEC.m_lin * L**4)),
        "torsion1": (np.pi / 2) * np.sqrt(SEC.GJ / (SEC.rhoJ * L**2)),
        "axial1": (np.pi / 2) * np.sqrt(SEC.EA / (SEC.m_lin * L**2)),
    }
    msg = []
    for name, wa in ana.items():
        rel = np.abs(w - wa).min() / wa
        assert rel < 0.02, f"{name}: nearest mode off by {rel:.3e}"
        msg.append(f"{name} {wa/2/np.pi:.1f}Hz rel {rel:.1e}")
    # first bending is a degenerate pair (tube symmetry)
    pair = abs(w[0] - w[1]) / w[0]
    assert pair < 1e-6, f"bending pair split {pair:.3e}"
    print(f"Gate 4 PASS: modes vs analytic ({'; '.join(msg)}; y/z pair split {pair:.1e})")


# ─── Gate 5: gpu_newmark_step ring-down ──────────────────────────────────────

def test_newmark():
    from fluxvortex.warp_fsi.batched_solver import gpu_newmark_step
    L, nel = 0.8, 16
    C = build_cantilever(nel, L=L)
    tip_dof = C.dof_map_np[C.nn - 1][2]
    F = 2.0

    def fext(qv):
        f = np.zeros(C.ndof); f[tip_dof] = F
        return f

    q0, _ = newton_static(C, fext)              # static sag as IC, then release
    a0 = q0[tip_dof]
    B, dt, nsteps = 1, 1e-4, 2000
    q = _wp_q(q0[None, :])
    dq = wp.zeros((B, C.ndof), dtype=cfg.DTYPE, device=DEV)
    zeros = wp.zeros((B, C.ndof), dtype=cfg.DTYPE, device=DEV)

    def recompute_bend(q_p1):
        return kb.beam_internal_force(q_p1, C)

    # cross-check: one dense-LU step vs the verified PCG reference step
    # (nonzero dq: exactly-zero rhs is a degenerate case where the PCG reference
    #  itself NaNs via 0/0 alpha — dense handles it cleanly)
    rngv = np.random.default_rng(11)
    dqx = rngv.normal(0, 1e-3, (B, C.ndof)); dqx[:, sorted(C.bc_dofs)] = 0.0
    dqw = _wp_q(dqx)
    Kblk = kb.assemble_beam_kblocks(q, C)
    Qb = kb.beam_internal_force(q, C)
    qA, dqA = kb.beam_newmark_step(q, dqw, Kblk, C, zeros, Qb, recompute_bend,
                                   0.5, 2.0, dt)
    qB, dqB = gpu_newmark_step(q, dqw, Kblk, C.Me, C.edofs, C.free, C.ndof,
                               zeros, zeros, Qb, zeros, recompute_bend, None,
                               0.5, 2.0, dt)
    xref = max(np.abs(qB.numpy()).max(), 1e-30)
    dstep = np.abs(qA.numpy() - qB.numpy()).max() / xref
    assert dstep < 1e-6, f"dense vs PCG Newmark step rel {dstep:.3e}"

    tip = np.zeros(nsteps)
    for k in range(nsteps):
        Kblk = kb.assemble_beam_kblocks(q, C)   # symmetrized tangent
        Qb = kb.beam_internal_force(q, C)
        q, dq = kb.beam_newmark_step(q, dq, Kblk, C, zeros, Qb, recompute_bend,
                                     0.5, 2.0, dt)
        tip[k] = q.numpy()[0, tip_dof]
    assert np.all(np.isfinite(tip)), "ring-down not finite"
    growth = np.abs(tip).max() / abs(a0)
    assert growth < 1.05, f"amplitude grew {growth:.3f}x"
    # frequency from zero crossings (linear interp on first/last crossing)
    s = tip
    cross = np.where(np.diff(np.sign(s)) != 0)[0]
    tc = cross + s[cross] / (s[cross] - s[cross + 1])
    f_num = (len(tc) - 1) / (2 * (tc[-1] - tc[0]) * dt)
    f_ana = 1.8751**2 * np.sqrt(SEC.EI / (SEC.m_lin * L**4)) / (2 * np.pi)
    rel = abs(f_num - f_ana) / f_ana
    assert rel < 0.02, f"ring-down freq {f_num:.2f}Hz vs {f_ana:.2f}Hz rel {rel:.3e}"
    print(f"Gate 5 PASS: ring-down finite, peak/initial {growth:.3f}, "
          f"freq {f_num:.2f}Hz vs analytic {f_ana:.2f}Hz (rel {rel:.2e})")


if __name__ == "__main__":
    print("=" * 70)
    print(f"P2-S1 geometrically-exact beam gates ({cfg.summary()})")
    print("=" * 70)
    test_force_energy_fd()
    test_k0_linear()
    test_rigid_motion()
    test_cantilever_static()
    test_rollup_quarter()
    test_modal()
    test_newmark()
    print("\nAll S1 gates 1-5 passed.")
