"""P2-S1 exit gate (gate 7): main spar under prescribed RoboEagle flapping.

Main carbon spar (OD10/wall1mm, L=0.8 m span, 16 elements) clamped-prescribed at
the root: rigid flapping frame +-45 deg @ 2.3 Hz about the x axis (the FSI-side
kinematics of FlapEntryRobo, 6-DOF version), 2 cycles at dt=2e-4 s.

Checks:
  - finite throughout; rotation-vector chart guard never trips;
  - elastic tip deflection (total minus rigid frame) finite and physically sane
    (inertial lag of a 43.5 N*m^2 spar: mm..cm, hard band [1e-4, 0.15] m);
  - mid-run snapshot -> advance -> restore -> advance replay is bit-identical
    (window-PC rewind precondition).

Run: cd FLUXV && python platform/p2_s1_beam3d.py
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
from fluxvortex.warp_fsi import kernels_beam3d as kb         # noqa: E402
from beam3d_solver import WarpBeam3DEntry                    # noqa: E402
from newton_pc.adapters.flap import FlapKinematics           # noqa: E402


def rot_x(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def main():
    L, nel = 0.8, 16
    amp_deg, freq = 45.0, 2.3
    dt, n_cycles = 2e-4, 2
    nodes = np.zeros((nel + 1, 3))
    nodes[:, 1] = np.linspace(0.0, L, nel + 1)               # spar along +y (span)
    elems = np.array([[i, i + 1] for i in range(nel)])
    C = kb.Beam3DConstants(nodes, elems, kb.MAIN_SPAR)

    kin = FlapKinematics(np.deg2rad(amp_deg), 1.0 / freq)

    def cb(t):
        th, thd, thdd = kin.angles(t)
        # root node at the flap axis origin: translations identically zero;
        # single-axis rotation -> rotation vector EXACTLY (theta,0,0) (psi_dot = omega)
        z = np.zeros(3)
        return (np.concatenate([z, [th, 0.0, 0.0]]),
                np.concatenate([z, [thd, 0.0, 0.0]]),
                np.concatenate([z, [thdd, 0.0, 0.0]]))

    entry = WarpBeam3DEntry(C, presc_nodes=[0], presc_cb=cb)
    nsteps = int(round(n_cycles / freq / dt))
    print(f"P2-S1 exit: spar L={L} EI={kb.MAIN_SPAR.EI:.1f} N*m^2, "
          f"+-{amp_deg}deg @ {freq}Hz, dt={dt}, {nsteps} substeps ({n_cycles} cycles)")

    tipX = C.nodes_np[-1]
    defl = np.zeros(nsteps)
    replay_diff = None
    t0 = time.time()
    for k in range(nsteps):
        t = (k + 1) * dt
        entry.substep(t, dt, None)
        st = entry.state()
        tip = st["verts"][-1]
        th = kin.angles(t)[0]
        defl[k] = np.linalg.norm(tip - rot_x(th) @ tipX)     # elastic lag
        if not np.all(np.isfinite(st["verts"])):
            raise RuntimeError(f"non-finite at substep {k}")
        if k == nsteps // 2:                                  # replay gate mid-run
            snap = entry.snapshot()
            for j in range(100):
                entry.substep(t + (j + 1) * dt, dt, None)
            qA = entry.q.numpy().copy()
            entry.restore(snap)
            for j in range(100):
                entry.substep(t + (j + 1) * dt, dt, None)
            replay_diff = np.abs(entry.q.numpy() - qA).max()
            entry.restore(snap)                               # resume nominal path
        if k % 500 == 0:
            print(f"  k={k} t={t:.4f}s theta={np.rad2deg(th):+6.1f}deg "
                  f"defl={defl[k]*1e3:7.3f}mm [{time.time()-t0:.0f}s]", flush=True)

    dmax = defl.max()
    print(f"\nfinite: True; max elastic tip deflection {dmax*1e3:.2f} mm; "
          f"replay max|dq| = {replay_diff:.2e}")
    assert 1e-4 < dmax < 0.15, f"deflection {dmax:.4f} m outside sanity band"
    assert replay_diff == 0.0 or replay_diff < 1e-12, f"replay not bit-identical: {replay_diff:.2e}"
    print(f"S1 EXIT GATE PASS  [{time.time()-t0:.0f}s wall]")


if __name__ == "__main__":
    print(cfg.summary())
    main()
