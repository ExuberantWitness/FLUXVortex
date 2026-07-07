"""P2-S4: coupled flexible-wing FSI smoke — Stein cold start + three anchors.

Case: 8 m/s, AoA 5 deg, flapping +-45 deg @ 2.3 Hz, 2 cycles, half-wing.
Cold start (Stein-Benney-Tezduyar 2001 paradigm, docs/p2_s2_membrane_research.md §3):
  0. pretension pre-equilibrium (inside WingEntry; free-edge relaxation)
  1. freeze the first-window aero load -> latched static Newton preload
  2. flow pre-development on the frozen structure (wake spin-up)
  3. couple from the preloaded state, zero velocities, flap amplitude ramped
     0 -> 45 deg over T/2 (C1 cosine); bounded settling expected & accepted.

Anchors (comparison targets, honest scope):
  A1 old ANCF-shell path : diverged in <10 windows (docs/p2_step0_3mat_probe.md)
  A2 rigid UVLM lift     : ~4.2 N (static geometry, same provider family)
  A3 measured flapping   : ~7.79 N cycle-mean lift (paper, 2 wings)
  NOTE: lightweight FlapUVLMProvider (no LEV/stall closures) — LIFT SCALE ONLY;
  thrust is NOT comparable until the S5 closure port.

Run: cd FLUXV && python platform/p2_s4_fsi.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_ROOT, _HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                            # noqa: E402
from fluxvortex.warp_fsi import config as cfg                # noqa: E402
from wing_system import WingModel, WingEntry                 # noqa: E402
from wing_iqn import add_iqnils                              # noqa: E402
from newton_pc import WindowPredictorCorrector               # noqa: E402
from newton_pc.adapters.flap import (FlapKinematics,         # noqa: E402
                                     FlapUVLMProvider)

D_STAR = 3.78e-3          # rib depth from the K_MEAS calibration (p2_s3 gate W2,
                          # assembly v3: flat + three straight rods to the edge arc)
CHORD = 0.287
U, AOA_DEG, FREQ, AMP_DEG = 8.0, 5.0, 2.3, 45.0
N_CYCLES = 2


def force_to_wing(model, fset):
    """Provider 9-dof/node layout (aero fraction grid, order j*(nc+1)+i) ->
    wing global dof vector via the work-consistent transfer f_s = W^T f_a."""
    f = np.zeros(model.ndof)
    f9 = np.asarray(fset.f).reshape(-1, 9)
    f[model.trans_map.ravel()] = (model.W_a2s.T @ f9[:, 0:3]).ravel()
    return f


def main():
    alpha = np.deg2rad(AOA_DEG)
    period = 1.0 / FREQ
    nc, ns = 8, 16
    dtw = (CHORD / nc) / U                        # panel-transit window (as run_fsi)
    wpc = int(round(period / dtw))
    n_windows = int(round(N_CYCLES * period / dtw))
    substeps = 40                                  # dt_sub ~ 1.1e-4: fine enough that
    # adaptive halving never triggers at the violent stroke phases — the halving
    # BRANCH FLIP made the window map g(x) discontinuous (measured: Picard limit
    # cycles -> committed 2-window sawtooth pumped through the delayed-Kutta shed)

    print(f"S4 FSI: U={U} AoA={AOA_DEG} flap ±{AMP_DEG}deg@{FREQ}Hz | "
          f"windows/cycle={wpc}, total={n_windows}, substeps={substeps} "
          f"(dt_sub={dtw/substeps:.2e})")

    model = WingModel(nc=nc, ns=ns, rib_depth=D_STAR)
    # S5 cold start: START AT THE STROKE TOP (theta=+A, theta_dot=0) — the
    # test-rig-realistic release point. Full-amplitude kinematics from t=0
    # (the wake is measured unstable on near-frozen geometry — S4 blocker #2)
    # but with ZERO interface velocity at t=0: aero loads grow continuously
    # from static values instead of the 9 m/s impulsive start (measured: the
    # mid-stroke start left w>=3 windows on a strongly repelling fixed point,
    # L -15 -> -125 -> -413 N even under IQN-ILS). Short load ramp (~4
    # windows, SHARPy style) absorbs the Wagner transient.
    kin0 = FlapKinematics(np.deg2rad(AMP_DEG), period)

    class TopStartKin:
        """theta(t) = A cos(2 pi t / T): stroke-top release, theta_dot(0)=0."""
        def angles(self, t):
            return kin0.angles(t + period / 4.0)
    kin = TopStartKin()
    entry = WingEntry(model, kin, ramp_T=0.0, load_ramp_T=4 * dtw)
    print(f"  pre-eq: |r|_soft={entry.preeq_info['resid']:.2e} N, "
          f"wrinkled {entry.preeq_info['n_wrinkled']}/{entry.preeq_info['ne']}")

    V_vec = U * np.array([np.cos(alpha), 0.0, np.sin(alpha)])

    # ── Stein step 1: frozen first-window load -> static preload ────────────
    provider0 = FlapUVLMProvider(V_vec, 1.225, dtw, K=6, nu=15.06e-6, chord=CHORD,
                                 particles=False, max_particles=1)
    F0 = provider0.solve(entry.state())
    Fz0 = F0.f.reshape(-1, 9)[:, 2].sum()
    # NOTE: no static preload — with the aero-load ramp-in the wing must START
    # UNLOADED (a full-load preload + zero ramped load = an inconsistent 6-mm
    # release transient, measured unconvergeable). Pre-equilibrium (pretension
    # relaxation) + C1 load ramp IS the consistent cold start.
    print(f"  step1: impulsive-start aero Fz={Fz0:+.2f} N (ramped in over T/2, "
          f"no static preload)")

    # ── Stein step 2 (adapted): wake development happens DURING the ramped
    # first coupled cycle — committing this lightweight UVLM on a FROZEN
    # geometry is measured unstable (stacked rings: Fz -> 5.8e4 N in 8
    # commits), so the standalone spin-up is folded into step 3; the
    # impulsive-start (Wagner) transient is absorbed by the amplitude ramp.
    #
    # S5 strong coupling (docs/p2_s5_coupling_research.md, Lefrancois route):
    #   - provider madd = UVLM-consistent added-mass matrix (full n(x)n,
    #     symmetrized + sign-projected; G1-gated vs the analytic plate) goes
    #     onto the structural LHS inside WingEntry.substep (M_eff = M - madd);
    #     explicit dGamma/dt is zeroed by the provider (no double count, no
    #     a_lag channel);
    #   - window-level Picard to convergence + Kuettler-Wall Aitken (omega
    #     inherited across windows, sign-kept, clamped 0.5), min 3 iterations
    #     (SHARPy), rel tol 1e-3 on the interface displacement residual.
    provider = FlapUVLMProvider(V_vec, 1.225, dtw, K=6, nu=15.06e-6,
                                chord=CHORD, particles=False, max_particles=1,
                                added_mass_operator=True)
    ait_stats = add_iqnils(provider)               # IQN-ILS w/ history reuse

    # ── Stein step 3: kinematically consistent stroke-top IC ────────────────
    # The WHOLE wing (pre-equilibrium shape) rigidly rotated to theta(0)=+A;
    # dq = 0 exactly (stroke reversal); a = theta_dd x r (angular acceleration
    # of the release, theta_dot=0 so no centripetal term). Root row and rod
    # psi match the prescribed _cb(0) values by construction.
    th0, thd0, thdd0 = kin.angles(0.0)
    c0_, s0_ = np.cos(th0), np.sin(th0)
    R0 = np.array([[1, 0, 0], [0, c0_, -s0_], [0, s0_, c0_]])
    u_pre = entry.q[model.trans_map]               # pre-eq displacements
    pos_rot = (model.nodes + u_pre) @ R0.T
    entry.q[model.trans_map.ravel()] = (pos_rot - model.nodes).ravel()
    al = np.array([thdd0, 0.0, 0.0])               # flap axis = +x (see _cb)
    entry.dq[:] = 0.0
    entry.a[model.trans_map.ravel()] = np.cross(al, pos_rot).ravel()
    for n_ in model.beam_nodes:
        entry.q[model.dof6[n_, 3:]] = [th0, 0.0, 0.0]
        entry.a[model.dof6[n_, 3:]] = al
    pc = WindowPredictorCorrector(
        entry=entry, provider=provider, substeps=substeps, dt=dtw / substeps,
        mode="two-pass", iterations=30, min_iterations=3,
        # abs tol at the PHYSICAL scale (30 um interface displacement): the
        # iteration has a ~1.5e-5 m noise floor (branchy substep algorithms);
        # demanding rel 1e-3 below it burned the full budget on already-
        # converged windows and committed limit-cycle samples on the rest
        adaptive_tol=3e-5, adaptive_tol_rel=1e-3,
        residual_norm=lambda a, b: float(np.linalg.norm(
            np.asarray(b["verts"]) - np.asarray(a["verts"]))))
    pc.initialize(F0)
    pc.advance(n_substeps=1)

    cap = int(os.environ.get("S4_NWIN", "0") or 0)     # short-run hook (G2)
    if cap:
        n_windows = min(n_windows, cap)
    lift, thrust, bend, iters = [], [], [], []
    t0 = time.time()
    for w in range(n_windows):
        stat = pc.advance()
        iters.append(stat.iterations)
        st = entry.state()
        if not np.all(np.isfinite(st["verts"])):
            print(f"  NON-FINITE at window {w}"); return False
        F = (pc._F_cur.payload["f_panel"].sum(axis=(0, 1))
             if pc._F_cur.payload else np.zeros(3))
        lift.append(float(-F[0] * np.sin(alpha) + F[2] * np.cos(alpha)))
        thrust.append(float(-(F[0] * np.cos(alpha) + F[2] * np.sin(alpha))))
        th = entry._angles(pc._t)[0]         # ramped kinematics
        c, s_ = np.cos(th), np.sin(th)
        R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
        g0 = np.zeros((ns + 1, nc + 1, 3))   # rest AERO surface (flat + camber)
        g0[..., 0:2] = model.aero_rest2d
        g0[..., 2] = model.aero_off
        rigid = (g0.reshape(-1, 3) @ R.T).reshape(ns + 1, nc + 1, 3).transpose(1, 0, 2)
        bend.append(float(np.abs(st["verts"][..., 2] - rigid[..., 2]).max()))
        if w % 10 == 0:
            print(f"  w={w:3d} t={pc._t:.3f}s th={np.rad2deg(th):+6.1f}deg "
                  f"L={lift[-1]:+7.2f}N bend={bend[-1]*1e3:6.1f}mm "
                  f"it={iters[-1]:2d} [{time.time()-t0:.0f}s]", flush=True)

    lift = np.array(lift); thrust = np.array(thrust); iters = np.array(iters)
    Lcyc = 2.0 * float(lift[-wpc:].mean())        # x2: half-wing channel
    Tcyc = 2.0 * float(thrust[-wpc:].mean())
    print("\n=== S5 coupled anchors ===")
    print(f"  A1 old ANCF-shell path : diverged <10 windows (this run: "
          f"{n_windows} windows finite, bend_max {max(bend)*1e3:.1f} mm)")
    print(f"  A2 rigid UVLM lift     : ~4.2 N   | flexible cycle-mean 2L = {Lcyc:+.2f} N")
    print(f"  A3 measured flapping   : ~7.79 N  | (lift scale only; thrust "
          f"2T = {Tcyc:+.2f} N NOT comparable pre-S5b closure port)")
    print(f"  Picard iters/window    : mean {iters.mean():.1f} max {iters.max()} "
          f"(IQN-ILS w/ reuse; Degroote strong-coupling reference ~6-10); "
          f"rank-filtered cols {ait_stats['n_filtered']}")
    print(f"  wall {time.time()-t0:.0f}s")
    return True


if __name__ == "__main__":
    print(cfg.summary())
    ok = main()
    print("S4 COUPLED SMOKE", "PASS" if ok else "FAIL")
