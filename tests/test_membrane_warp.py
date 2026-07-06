"""P2-S2 tension-field CST membrane (Warp) — verification ladder.

M1: force == dE/dq (central FD of GPU energy) across taut/wrinkled/slack states.
M2: K(0) == independent analytic CST plane-stress stiffness + N0 geometric
    stiffness (the K_G(N0) that carries ALL out-of-plane stiffness).
M3: rigid motion — energy invariant, nodal forces rotate with the frame.
M4: pretensioned square membrane modes vs analytic f_mn = 1/2 sqrt(N0/rho h)
    sqrt((m/a)^2+(n/a)^2), including the f ~ sqrt(N0) scaling signature
    (the discriminating test the old shell path failed).
M5: static pressure bulge vs analytic series w_c (Poisson N0 lap(w) = -p).
M6: element constitutive kink — n2(e2) bends at the taut/wrinkled boundary
    (slope -> eta * h*beta) and n1 follows the uniaxial branch; slack: eta only.
M7: Newmark ring-down (dense beam_newmark_step, duck-typed constants) —
    finite, bounded, frequency matches the mesh's own eigenfrequency.

Run: cd FLUXV && python tests/test_membrane_warp.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import warp as wp

import fluxvortex.warp_fsi.config as cfg
from fluxvortex.warp_fsi import kernels_membrane as km
from fluxvortex.warp_fsi.kernels_beam3d import beam_newmark_step, scatter_beam_global

DEV = cfg.DEVICE
NP = cfg.NP_DTYPE

# Mylar skin + research-anchored pretension baseline (docs/p2_s2_membrane_research.md)
H, E_Y, NU, RHO = 5e-5, 4e9, 0.3, 1390.0
N0, ETA = 30.0, 1e-4
RHOH = RHO * H                                   # 0.0695 kg/m^2


def _wp_q(q_np):
    return wp.array(np.ascontiguousarray(q_np, dtype=NP), dtype=cfg.DTYPE, device=DEV)


def build_square(n, a=0.1, N0=N0):
    """Structured (n+1)^2 grid on [0,a]^2, z=0, 2 tris per cell."""
    xs = np.linspace(0, a, n + 1)
    nodes = np.array([[x, y, 0.0] for y in xs for x in xs])
    tris = []
    for j in range(n):
        for i in range(n):
            p = j * (n + 1) + i
            tris.append([p, p + 1, p + n + 2])
            tris.append([p, p + n + 2, p + n + 1])
    C = km.MembraneConstants(nodes, np.array(tris), H, E_Y, NU, RHO, N0, ETA, device=DEV)
    bnd = [k for k, nd in enumerate(nodes)
           if min(nd[0], nd[1]) < 1e-12 or max(nd[0], nd[1]) > a - 1e-12]
    return C, bnd


# ─── M1: force == dE/dq across branches ──────────────────────────────────────

def test_force_energy_fd():
    rng = np.random.default_rng(2)
    C, _ = build_square(4)
    kappa = N0 / C.hb                            # pretension strain scale
    e1sl = -N0 * (1 - NU) / C.hE
    B = 3
    q = np.zeros((B, C.ndof))
    # Smooth controlled fields, one branch per env, with EXPLICIT margins from
    # both branch boundaries and the e1==e2 degeneracy (wrinkle direction
    # undefined at equibiaxial states — genuine measure-zero kink of isotropic
    # tension-field theory; boundary behavior itself is covered by gate M6,
    # degeneracy is the P4 sqrt(D+eps^2) regularization target).
    a = 0.1
    X, Y = C.nodes_np[:, 0], C.nodes_np[:, 1]
    ripple = 0.05 * np.sin(2 * np.pi * X / a) * np.sin(2 * np.pi * Y / a)
    strains = [(0.3, 1.0), (0.5, -4.0), (-3.0, -6.0)]        # (exx, eyy)/kappa
    for b, (s1, s2) in enumerate(strains):
        u = np.zeros((C.nn, 3))
        u[:, 0] = s1 * kappa * X * (1 + ripple)
        u[:, 1] = s2 * kappa * Y * (1 + ripple)
        u[:, 2] = 0.02 * kappa * a * ripple
        q[b] = u.reshape(-1)
    st = km.membrane_state(_wp_q(q), C).numpy()
    branches = []
    for b in range(B):
        e1, e2, _, n2 = st[b, :, 0], st[b, :, 1], st[b, :, 2], st[b, :, 3]
        assert (e1 - e2).min() > 0.3 * kappa, "degeneracy margin violated"
        if b == 0:
            assert n2.min() > 0.2 * N0; branches.append("taut")
        elif b == 1:
            nf2 = n2 / ETA                        # eta-blended residual -> full
            assert nf2.max() < -0.5 * N0 and (e1 - e1sl).min() > 0.3 * kappa
            branches.append("wrinkled")
        else:
            assert (e1sl - e1).min() > 0.3 * kappa; branches.append("slack")
    Qf = km.membrane_internal_force(_wp_q(q), C).numpy()
    h = 1e-8            # energy is quartic in q (Green^2): balance truncation/roundoff
    worst = 0.0
    for b in range(B):
        scale = np.abs(Qf[b]).max()              # per-env force scale: near-zero
        for d in rng.choice(C.ndof, size=20, replace=False):   # components carry
            qp = q.copy(); qp[b, d] += h          # FD truncation noise, so
            qm = q.copy(); qm[b, d] -= h          # normalize by the field scale
            Ep = km.membrane_energy_total(_wp_q(qp), C).numpy()[b]
            Em = km.membrane_energy_total(_wp_q(qm), C).numpy()[b]
            fd = (Ep - Em) / (2 * h)
            worst = max(worst, abs(fd - Qf[b, d]) / scale)
    assert worst < 1e-6, f"force!=dE/dq: worst rel-of-scale {worst:.3e}"
    print(f"M1 PASS: force==dE/dq worst rel-of-scale {worst:.3e} (envs: {branches})")


# ─── M2: K(0) vs independent analytic (material + K_G(N0)) ──────────────────

def test_k0_analytic():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    C = km.MembraneConstants(nodes, np.array([[0, 1, 2]]), H, E_Y, NU, RHO, N0, ETA,
                             device=DEV)
    Kfd = km.assemble_membrane_kblocks(_wp_q(np.zeros((1, C.ndof))), C,
                                       symmetrize=False).numpy()[0, 0]
    # independent: CST b/c coefficients on this triangle (A=1/2)
    A = 0.5
    b = np.array([-1.0, 1.0, 0.0])               # dN/dx
    c = np.array([-1.0, 0.0, 1.0])               # dN/dy
    Bmat = np.zeros((3, 6))
    for i in range(3):
        Bmat[0, 2*i] = b[i]; Bmat[1, 2*i+1] = c[i]
        Bmat[2, 2*i] = c[i]; Bmat[2, 2*i+1] = b[i]
    beta = E_Y / (1 - NU**2)
    Cps = H * beta * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    Kin = A * Bmat.T @ Cps @ Bmat                # (6,6) on x,y dofs
    Ke = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            Ke[3*i:3*i+2, 3*j:3*j+2] += Kin[2*i:2*i+2, 2*j:2*j+2]
            Ke[3*i:3*i+3, 3*j:3*j+3] += A * N0 * (b[i]*b[j] + c[i]*c[j]) * np.eye(3)
    rel = np.abs(Kfd - Ke).max() / np.abs(Ke).max()
    assert rel < 1e-7, f"K(0) mismatch rel {rel:.3e}"
    print(f"M2 PASS: K(0) == CST material + K_G(N0) analytic, rel {rel:.3e}")


# ─── M3: rigid motion ────────────────────────────────────────────────────────

def test_rigid_motion():
    C, _ = build_square(3)
    ax = np.array([0.3, 0.9, 0.5]); ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    W0 = km.membrane_energy_total(_wp_q(np.zeros((1, C.ndof))), C).numpy()[0]
    Q0 = km.membrane_internal_force(_wp_q(np.zeros((1, C.ndof))), C).numpy()[0]
    worst_W = worst_Q = 0.0
    for deg in (30.0, 120.0):
        th = np.deg2rad(deg)
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        q = ((R - np.eye(3)) @ C.nodes_np.T).T + np.array([0.02, -0.01, 0.05])
        qv = q.reshape(1, -1)
        W = km.membrane_energy_total(_wp_q(qv), C).numpy()[0]
        Q = km.membrane_internal_force(_wp_q(qv), C).numpy()[0].reshape(-1, 3)
        Wscale = N0 * C.A0_np.sum()               # physical energy scale (W0 == 0)
        worst_W = max(worst_W, abs(W - W0) / Wscale)
        worst_Q = max(worst_Q, np.abs(Q - Q0.reshape(-1, 3) @ R.T).max() / np.abs(Q0).max())
    assert worst_W < 1e-12 and worst_Q < 1e-11, f"rigid: dW {worst_W:.2e} dQ {worst_Q:.2e}"
    print(f"M3 PASS: rigid motion — energy invariant {worst_W:.2e}, force rotates {worst_Q:.2e}")


# ─── M4: modes vs analytic + sqrt(N0) scaling ────────────────────────────────

def _modes(C, bnd, k=6, vectors=False):
    from scipy.linalg import eigh
    C.set_bc(bnd)
    q0 = _wp_q(np.zeros((1, C.ndof)))
    Kblk = km.assemble_membrane_kblocks(q0, C, symmetrize=False).numpy()[0]
    K = scatter_beam_global(Kblk, C.edofs_np, C.ndof)
    M = scatter_beam_global(C.Me_np, C.edofs_np, C.ndof)
    free = np.array(sorted(set(range(C.ndof)) - C.bc_dofs))
    if not vectors:
        w2 = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True,
                  subset_by_index=[0, k - 1])
        return np.sqrt(np.abs(w2)) / (2 * np.pi)
    w2, V = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
                 subset_by_index=[0, k - 1])
    phi = np.zeros((C.ndof, k))
    phi[free] = V
    return np.sqrt(np.abs(w2)) / (2 * np.pi), phi


def test_modal_sqrtN0():
    a, n = 0.1, 16
    f_ana = lambda N0v, m, nn_: 0.5 * np.sqrt(N0v / RHOH) * np.sqrt((m/a)**2 + (nn_/a)**2)
    C1, bnd = build_square(n, a, N0=30.0)
    f1 = _modes(C1, bnd)
    rel11 = abs(f1[0] - f_ana(30, 1, 1)) / f_ana(30, 1, 1)
    # degenerate pair (1,2)/(2,1) then (2,2)
    rel12 = abs(f1[1] - f_ana(30, 1, 2)) / f_ana(30, 1, 2)
    C4, bnd4 = build_square(n, a, N0=120.0)
    f4 = _modes(C4, bnd4)
    ratio = f4[0] / f1[0]
    assert rel11 < 0.02 and rel12 < 0.02, f"modal: rel11 {rel11:.3e} rel12 {rel12:.3e}"
    assert abs(ratio - 2.0) < 1e-3, f"sqrt(N0) scaling: f(4N0)/f(N0) = {ratio:.5f}"
    print(f"M4 PASS: f11 {f1[0]:.1f}Hz vs {f_ana(30,1,1):.1f}Hz (rel {rel11:.1e}), "
          f"f12 rel {rel12:.1e}, f(4N0)/f(N0) = {ratio:.5f}")


# ─── M5: static pressure bulge vs analytic series ────────────────────────────

def test_bulge():
    # p=1 Pa keeps the gate in the LINEAR regime: at p=4 the bulge slope already
    # self-tensions the membrane by ~3.6% of N0 (geometric stiffening — physics,
    # not error) and the linear series overpredicts by the same amount.
    a, n, p = 0.1, 16, 1.0
    C, bnd = build_square(n, a)
    C.set_bc(bnd)
    free = np.array(sorted(set(range(C.ndof)) - C.bc_dofs))
    # dead pressure load: p*A0/3 to each node of each tri, +z
    f = np.zeros(C.ndof)
    for e, tri in enumerate(C.tris_np):
        for nd in tri:
            f[C.dof_map_np[nd][2]] += p * C.A0_np[e] / 3.0
    q = np.zeros(C.ndof)
    for it in range(30):
        Q = km.membrane_internal_force(_wp_q(q[None, :]), C).numpy()[0]
        r = f - Q
        rn = np.linalg.norm(r[free])
        if rn < 1e-10 * max(1.0, np.linalg.norm(f)):
            break
        Kblk = km.assemble_membrane_kblocks(_wp_q(q[None, :]), C, symmetrize=False).numpy()[0]
        K = scatter_beam_global(Kblk, C.edofs_np, C.ndof)
        dq = np.zeros(C.ndof)
        dq[free] = np.linalg.solve(K[np.ix_(free, free)], r[free])
        q += dq
    # center node z
    ic = (n // 2) * (n + 1) + n // 2
    assert abs(C.nodes_np[ic, 0] - a/2) < 1e-9 and abs(C.nodes_np[ic, 1] - a/2) < 1e-9
    wc = q[C.dof_map_np[ic][2]]
    # analytic series for N0 lap(w) = -p, w=0 on boundary
    s = 0.0
    for m in range(1, 40, 2):
        for nn_ in range(1, 40, 2):
            s += (np.sin(m*np.pi/2) * np.sin(nn_*np.pi/2)) / (m * nn_ * (m**2 + nn_**2))
    wc_ana = 16 * p * a**2 / (np.pi**4 * N0) * s
    rel = abs(wc - wc_ana) / wc_ana
    assert rel < 0.02, f"bulge: {wc:.4e} vs {wc_ana:.4e} rel {rel:.3e}"
    print(f"M5 PASS: bulge w_c {wc*1e3:.4f}mm vs analytic {wc_ana*1e3:.4f}mm "
          f"(rel {rel:.1e}, {it} Newton its)")


# ─── M6: constitutive kink at the wrinkling boundary ─────────────────────────

def test_wrinkle_kink():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    C = km.MembraneConstants(nodes, np.array([[0, 1, 2]]), H, E_Y, NU, RHO, N0, ETA,
                             device=DEV)
    kappa = N0 / C.hb
    ex = 0.5 * kappa                                  # fixed small stretch in x
    e2_bnd = -NU * ex - kappa                         # analytic taut/wrinkled boundary
    eys, n1s, n2s = [], [], []
    for ey in np.linspace(0.5 * e2_bnd, 2.5 * e2_bnd, 41):
        q = np.zeros((1, C.ndof))
        q[0, C.dof_map_np[1][0]] = ex                 # u_x at node (1,0): exx = ex+ex^2/2~ex
        q[0, C.dof_map_np[2][1]] = ey
        st = km.membrane_state(_wp_q(q), C).numpy()[0, 0]
        eys.append(st[1]); n1s.append(st[2]); n2s.append(st[3])
    eys, n1s, n2s = map(np.array, (eys, n1s, n2s))
    # taut side: dn2/de2 = hb; wrinkled side: eta*hb
    taut = eys > e2_bnd * 0.9
    wr = eys < e2_bnd * 1.1
    s_taut = np.polyfit(eys[taut], n2s[taut], 1)[0]
    s_wr = np.polyfit(eys[wr], n2s[wr], 1)[0]
    r1 = abs(s_taut - C.hb) / C.hb
    r2 = abs(s_wr - ETA * C.hb) / (ETA * C.hb)
    assert r1 < 0.02 and r2 < 0.05, f"kink slopes: taut rel {r1:.2e}, wrinkled rel {r2:.2e}"
    # wrinkled n1 follows the uniaxial branch (within eta)
    iw = np.argmin(np.abs(eys - 2.0 * e2_bnd))
    e1w = ex * (1 + 0.5 * ex)                         # Green exx for stretch ex
    n1_uni = N0 * (1 - NU) + C.hE * e1w
    r3 = abs(n1s[iw] - n1_uni) / n1_uni
    assert r3 < 5e-3, f"uniaxial n1 rel {r3:.2e}"
    print(f"M6 PASS: n2 kink at wrinkling boundary (slopes hb/{ETA}*hb rel {r1:.1e}/{r2:.1e}), "
          f"wrinkled n1 uniaxial rel {r3:.1e}")


# ─── M7: Newmark ring-down (dense driver, duck-typed constants) ──────────────

def test_newmark_ringdown():
    a, n = 0.1, 6              # n=6: single-thread dense LU cost per env ~ ndof^3
    C, bnd = build_square(n, a)
    # IC = first eigenmode (pure out-of-plane; a static-bulge IC carries fast
    # in-plane content, under-resolved at this dt -> bounded ~7% first-swing
    # transient that clutters the growth gate)
    f_ref, phi = _modes(C, bnd, k=1, vectors=True)
    f_ref = f_ref[0]
    ic = (n // 2) * (n + 1) + n // 2
    zdof = C.dof_map_np[ic][2]
    q = phi[:, 0] * (2e-5 / phi[zdof, 0])        # 0.02 mm modal amplitude
    a0 = q[zdof]
    B, dt, nsteps = 1, 5e-5, 1000
    qw = _wp_q(q[None, :])
    dqw = wp.zeros((B, C.ndof), dtype=cfg.DTYPE, device=DEV)
    zeros = wp.zeros((B, C.ndof), dtype=cfg.DTYPE, device=DEV)
    rec = lambda qp1: km.membrane_internal_force(qp1, C)
    tip = np.zeros(nsteps)
    for k in range(nsteps):
        Kblk = km.assemble_membrane_kblocks(qw, C)
        Qb = km.membrane_internal_force(qw, C)
        qw, dqw = beam_newmark_step(qw, dqw, Kblk, C, zeros, Qb, rec, 0.5, 2.0, dt)
        tip[k] = qw.numpy()[0, zdof]
    assert np.all(np.isfinite(tip)), "ring-down not finite"
    growth = np.abs(tip).max() / abs(a0)
    assert growth < 1.05, f"amplitude grew {growth:.3f}"
    s = tip - tip.mean()
    cross = np.where(np.diff(np.sign(s)) != 0)[0]
    tc = cross + s[cross] / (s[cross] - s[cross + 1])
    f_num = (len(tc) - 1) / (2 * (tc[-1] - tc[0]) * dt)
    rel = abs(f_num - f_ref) / f_ref
    assert rel < 0.03, f"ring-down {f_num:.1f}Hz vs discrete {f_ref:.1f}Hz rel {rel:.3e}"
    print(f"M7 PASS: ring-down finite, peak/initial {growth:.3f}, "
          f"freq {f_num:.1f}Hz vs mesh eigenfreq {f_ref:.1f}Hz (rel {rel:.1e})")


if __name__ == "__main__":
    print("=" * 70)
    print(f"P2-S2 tension-field membrane gates ({cfg.summary()})")
    print(f"Mylar h={H} E={E_Y/1e9}GPa nu={NU} rho={RHO}; N0={N0} N/m eta={ETA}")
    print("=" * 70)
    test_force_energy_fd()
    test_k0_analytic()
    test_rigid_motion()
    test_modal_sqrtN0()
    test_bulge()
    test_wrinkle_kink()
    test_newmark_ringdown()
    print("\nAll S2 membrane gates (M1-M7) passed.")
