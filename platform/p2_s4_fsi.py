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
from newton_pc import WindowPredictorCorrector               # noqa: E402
from newton_pc.adapters.flap import (FlapKinematics,         # noqa: E402
                                     FlapUVLMProvider, NodalForceSet)

D_STAR = 4.43e-3          # rib depth from the K_MEAS calibration (p2_s3 gate W2)
CHORD = 0.287
U, AOA_DEG, FREQ, AMP_DEG = 8.0, 5.0, 2.3, 45.0
N_CYCLES = 2


def force_to_wing(model, fset):
    """Provider 9-dof/node layout -> wing global dof vector (translations)."""
    f = np.zeros(model.ndof)
    f9 = np.asarray(fset.f).reshape(-1, 9)
    f[model.trans_map.ravel()] = f9[:, 0:3].ravel()
    return f


def main():
    alpha = np.deg2rad(AOA_DEG)
    period = 1.0 / FREQ
    nc, ns = 8, 16
    dtw = (CHORD / nc) / 8.0 / U * 8.0            # panel-transit window (as run_fsi)
    dtw = (CHORD / nc) / 8.0
    wpc = int(round(period / dtw))
    n_windows = int(round(N_CYCLES * period / dtw))
    substeps = 20                                  # dt_sub ~ 2.2e-4 (ring-down regime)

    print(f"S4 FSI: U={U} AoA={AOA_DEG} flap ±{AMP_DEG}deg@{FREQ}Hz | "
          f"windows/cycle={wpc}, total={n_windows}, substeps={substeps} "
          f"(dt_sub={dtw/substeps:.2e})")

    model = WingModel(nc=nc, ns=ns, rib_depth=D_STAR)
    kin = FlapKinematics(np.deg2rad(AMP_DEG), period)
    entry = WingEntry(model, kin, ramp_T=period / 2, load_ramp_T=period / 2)
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
    provider_raw = FlapUVLMProvider(V_vec, 1.225, dtw, K=6, nu=15.06e-6,
                                    chord=CHORD, particles=False, max_particles=1)

    class UnderRelaxedProvider:
        """Force under-relaxation between provider solves (omega=0.35): the
        membrane added-mass ratio ~5 gives the loose two-pass PC a window loop
        gain ~1.6 (measured 2-window sawtooth, L alternating to +-1e2 N by w=8)
        — the research-recommended remedy (决策点3: 欠松弛+子迭代 before damping).
        The provider's madd operator is NOT used: measured INDEFINITE
        (eigs +-5e-4 ~ structural node mass) and historically never enabled."""

        def __init__(self, inner, omega=0.35):
            self.inner = inner
            self.omega = omega
            self._rel = None

        def solve(self, state):
            F = self.inner.solve(state)
            fmix = F.f if self._rel is None else self._rel.f + self.omega * (F.f - self._rel.f)
            pay = dict(F.payload or {})
            pay["a_lag"] = entry.a.copy()      # freeze the accel at solve time
            self._rel = NodalForceSet(fmix, payload=pay)
            return self._rel

        def commit(self, forces):
            self.inner.commit(forces)

    provider = UnderRelaxedProvider(provider_raw)

    # ── Stein step 3: coupled, zero velocities, amplitude ramp ──────────────
    entry.dq[:] = 0.0
    entry.a[:] = 0.0
    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=substeps, dt=dtw / substeps,
                                  mode="two-pass")
    pc.initialize(F0)
    pc.advance(n_substeps=1)

    lift, thrust, bend = [], [], []
    t0 = time.time()
    for w in range(n_windows):
        pc.advance()
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
        rigid = (model.nodes @ R.T).reshape(ns + 1, nc + 1, 3).transpose(1, 0, 2)
        bend.append(float(np.abs(st["verts"][..., 2] - rigid[..., 2]).max()))
        if w % 10 == 0:
            print(f"  w={w:3d} t={pc._t:.3f}s th={np.rad2deg(th):+6.1f}deg "
                  f"L={lift[-1]:+7.2f}N bend={bend[-1]*1e3:6.1f}mm "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    lift = np.array(lift); thrust = np.array(thrust)
    Lcyc = 2.0 * float(lift[-wpc:].mean())        # x2: half-wing channel
    Tcyc = 2.0 * float(thrust[-wpc:].mean())
    print("\n=== S4 anchors ===")
    print(f"  A1 old ANCF-shell path : diverged <10 windows (this run: "
          f"{n_windows} windows finite, bend_max {max(bend)*1e3:.1f} mm)")
    print(f"  A2 rigid UVLM lift     : ~4.2 N   | flexible cycle-mean 2L = {Lcyc:+.2f} N")
    print(f"  A3 measured flapping   : ~7.79 N  | (lift scale only; thrust "
          f"2T = {Tcyc:+.2f} N NOT comparable pre-S5)")
    print(f"  wall {time.time()-t0:.0f}s")
    return True


if __name__ == "__main__":
    print(cfg.summary())
    ok = main()
    print("S4 COUPLED SMOKE", "PASS" if ok else "FAIL")
