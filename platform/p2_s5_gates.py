"""P2-S5 gates (strong-coupling migration, docs/p2_s5_coupling_research.md).

G1: provider madd (full n(x)n blocks, symmetrized + sign-projected) on the
    RIGID rest wing vs the analytic flat-plate strip distribution
    m_a = rho*pi*c^2/4 per unit span:
      - -Ms must be PSD (sign projection worked; no destabilizing directions)
      - uniform a_z: total reaction magnitude within +-30% of the analytic
        total, sign OPPOSING the acceleration
      - spanwise distribution tracks c^2 (mid-span stations within +-35%)
      - clip fraction (wrong-sign spectral mass removed) reported honestly
G2: rho_f sweep {0.5, 1.225, 3.0}: coupled-iteration count must be near-
    independent of rho_f (Lefrancois criterion) — run AFTER T2/T3 wiring.

Run: cd FLUXV && python platform/p2_s5_gates.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_ROOT, _HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                             # noqa: E402
from fluxvortex.warp_fsi import config as cfg                 # noqa: E402
from wing_system import WingModel, WingEntry                  # noqa: E402
from newton_pc.adapters.flap import (FlapKinematics,          # noqa: E402
                                     FlapUVLMProvider)
import _v2_robogeom as rg                                     # noqa: E402

D_STAR = 3.77e-3
CHORD, U, AOA_DEG = 0.287, 8.0, 5.0
RHO = 1.225


def gate_G1(rho=RHO, verbose=True):
    alpha = np.deg2rad(AOA_DEG)
    V_vec = U * np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    m = WingModel(rib_depth=D_STAR)
    entry = WingEntry(m, FlapKinematics(0.0, 1.0 / 2.3))      # rest clamp
    st = entry.state()
    dtw = (CHORD / m.nc) / U
    prov = FlapUVLMProvider(V_vec, rho, dtw, K=6, nu=15.06e-6, chord=CHORD,
                            particles=False, max_particles=1,
                            added_mass_operator=True)
    F = prov.solve(st)
    nn = (m.nc + 1) * (m.ns + 1)
    tidx = (9 * np.arange(nn)[:, None] + np.arange(3)[None, :]).ravel()
    Ms = F.madd[np.ix_(tidx, tidx)]

    # (a) -Ms PSD (largest eigenvalue of Ms must be <= 0 up to roundoff)
    ev = np.linalg.eigvalsh(Ms)
    assert ev.max() < 1e-10 * max(1.0, -ev.min()), f"sign projection failed: {ev.max():.2e}"

    # (b) uniform a_z reaction vs analytic plate total
    a = np.tile([0.0, 0.0, 1.0], nn)
    Fn = (Ms @ a).reshape(nn, 3)
    Fz = Fn[:, 2]
    tot = Fz.sum()
    ys = np.linspace(0.0, 0.8, 400)
    m_ana = rho * np.pi / 4.0 * np.trapezoid(rg.chord_at(ys) ** 2, ys)
    assert tot < 0.0, f"reaction does not oppose acceleration: {tot:+.3e}"
    ratio = -tot / m_ana
    assert 0.7 < ratio < 1.3, f"|madd| vs analytic plate: ratio {ratio:.3f}"

    # (c) spanwise distribution tracks c^2 (mid stations, edge lumping excluded)
    row = np.add.reduceat(Fz, np.arange(0, nn, m.nc + 1))     # per j station
    yj = np.linspace(0.0, 0.8, m.ns + 1)
    dy = yj[1] - yj[0]
    ana_row = rho * np.pi / 4.0 * rg.chord_at(yj) ** 2 * dy
    mid = slice(2, 14)
    rr = -row[mid] / ana_row[mid]
    assert np.all((rr > 0.65) & (rr < 1.35)), f"spanwise ratio out of band: {np.round(rr, 2)}"

    if verbose:
        print(f"G1 PASS: madd full-n(x)n | -Ms PSD (max eig {ev.max():.1e}) | "
              f"uniform a_z reaction {-tot*1e3:.1f} g vs analytic "
              f"{m_ana*1e3:.1f} g (ratio {ratio:.3f}) | spanwise c^2 band "
              f"[{rr.min():.2f},{rr.max():.2f}] | clip fraction "
              f"{prov.madd_clip_frac:.3f}")
    return dict(ratio=ratio, clip=prov.madd_clip_frac, Ms=Ms, model=m)


def coupled_run(rho=RHO, n_windows=10, iterations=30, verbose=False):
    """Short coupled run (stroke-top start, IQN-ILS, madd), mirrors p2_s4_fsi.
    Returns (iters list, lift list, ok)."""
    from wing_iqn import add_iqnils
    from newton_pc import WindowPredictorCorrector
    alpha = np.deg2rad(AOA_DEG)
    period = 1.0 / 2.3
    dtw = (CHORD / 8) / U
    substeps = 20
    V_vec = U * np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    model = WingModel(rib_depth=D_STAR)
    kin0 = FlapKinematics(np.deg2rad(45.0), period)

    class TopStartKin:
        def angles(self, t):
            return kin0.angles(t + period / 4.0)
    kin = TopStartKin()
    entry = WingEntry(model, kin, ramp_T=0.0, load_ramp_T=4 * dtw)
    p0 = FlapUVLMProvider(V_vec, rho, dtw, K=6, nu=15.06e-6, chord=CHORD,
                          particles=False, max_particles=1)
    F0 = p0.solve(entry.state())
    prov = FlapUVLMProvider(V_vec, rho, dtw, K=6, nu=15.06e-6, chord=CHORD,
                            particles=False, max_particles=1,
                            added_mass_operator=True)
    add_iqnils(prov)
    th0, thd0, thdd0 = kin.angles(0.0)
    c0_, s0_ = np.cos(th0), np.sin(th0)
    R0 = np.array([[1, 0, 0], [0, c0_, -s0_], [0, s0_, c0_]])
    pos_rot = (model.nodes + entry.q[model.trans_map]) @ R0.T
    entry.q[model.trans_map.ravel()] = (pos_rot - model.nodes).ravel()
    al = np.array([thdd0, 0.0, 0.0])
    entry.dq[:] = 0.0
    entry.a[model.trans_map.ravel()] = np.cross(al, pos_rot).ravel()
    for n_ in model.beam_nodes:
        entry.q[model.dof6[n_, 3:]] = [th0, 0.0, 0.0]
        entry.a[model.dof6[n_, 3:]] = al
    pc = WindowPredictorCorrector(
        entry=entry, provider=prov, substeps=substeps, dt=dtw / substeps,
        mode="two-pass", iterations=iterations, min_iterations=3,
        adaptive_tol=1e-8, adaptive_tol_rel=1e-3,
        residual_norm=lambda a, b: float(np.linalg.norm(
            np.asarray(b["verts"]) - np.asarray(a["verts"]))))
    pc.initialize(F0)
    pc.advance(n_substeps=1)
    iters, lift = [], []
    for w in range(n_windows):
        try:
            s = pc.advance()
        except RuntimeError as e:
            if verbose:
                print(f"    rho={rho}: FAIL at w{w}: {e}")
            return iters, lift, False
        iters.append(s.iterations)
        lift.append(float(pc._F_cur.f.reshape(-1, 9)[:, 2].sum()))
        if verbose:
            print(f"    rho={rho} w{w}: it={s.iterations} L={lift[-1]:+.2f}")
    return iters, lift, True


def gate_G2(rhos=(0.5, 1.225, 3.0), n_windows=10):
    """Lefrancois criterion: with the UVLM-consistent added mass on the LHS,
    coupled-iteration count must be near-INDEPENDENT of rho_f (a plain
    staggered/relaxed scheme degrades or diverges with growing rho_f)."""
    means = {}
    for rho in rhos:
        iters, lift, ok = coupled_run(rho=rho, n_windows=n_windows)
        assert ok, f"G2: coupled run failed at rho_f={rho}"
        means[rho] = float(np.mean(iters))
        print(f"  rho_f={rho:5.3f}: iters/window mean {means[rho]:.1f} "
              f"max {max(iters)}  L_last={lift[-1]:+.2f} N")
    spread = max(means.values()) / max(min(means.values()), 1e-9)
    assert spread < 2.0, f"iteration count strongly rho_f-dependent: {means}"
    print(f"G2 PASS: iters/window near rho_f-independent "
          f"(spread x{spread:.2f} over rho_f x{max(rhos)/min(rhos):.0f})")
    return means


if __name__ == "__main__":
    import sys as _sys
    print(cfg.summary())
    which = _sys.argv[1] if len(_sys.argv) > 1 else "all"
    if which in ("all", "G1"):
        gate_G1()
    if which in ("all", "G2"):
        gate_G2()
    print(f"S5 gates passed ({which}).")
