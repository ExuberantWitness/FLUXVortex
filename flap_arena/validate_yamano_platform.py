"""Platform-config validation vs MATLAB (Yamano cantilever-plate FSI).

The UNIVERSAL platform configuration (physical-convention UVLM + particle
far-field wake + two-pass PC) runs the Yamano case in physical units:
1m x 1m plate, h=1mm, E=2.352 GPa, rho_s=1225, V=10 m/s, alpha=0,
clamped at the LEADING edge (x=0), pulse load 0.5*rho_f*V^2*sin(pi t*/0.2)
for t*<0.2. Truth: MATLAB fixtures_traj h_X_vec tip-z at block boundaries.

NOTE (documented sensitivity): the MATLAB-exact adapter config reproduces
the trajectory to 1e-6; the universal config differs in regularization and
omits the added-mass operator transfer, so deviation GROWS with horizon —
this case measures that fidelity gap honestly.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scipy.io import loadmat
from newton_pc import WindowPredictorCorrector
from newton_pc.adapters.flap import (FlapEntry, FlapKinematics,
                                     FlapUVLMProvider, NodalForceSet)

L = W = 1.0; H = 1e-3; E = 2.352e9; RHO_S = 1225.0
V = 10.0; RHO_F = 1.225; NC, NS = 15, 10
DT = 2e-4                  # physical substep (MATLAB d_t=2e-3 nondim x 0.1)
SUBSTEPS = 34              # MATLAB block
DTW = SUBSTEPS * DT

kin = FlapKinematics(0.0, 1.0)
entry = FlapEntry(L, W, NC, NS, kin, mode="elastic", thickness=H,
                  rho_s=RHO_S, E0=E, clamp_edge="x0")
fd = 0.5 * RHO_F * V * V / H            # body-force density of the pulse
pulse_nodal = entry.shell.distributed_load(np.array([0.0, 0.0, fd]))
T_P = 0.2 * L / V                        # pulse duration (t*<0.2)
entry.extra_force_fn = lambda t: (pulse_nodal * np.sin(np.pi * t / T_P)
                                  if t < T_P else 0.0 * pulse_nodal)

provider = FlapUVLMProvider(V * np.array([1.0, 0.0, 0.0]), RHO_F, DTW,
                            K=8, particles=True, chord=L,
                            added_mass_operator=True)
pc = WindowPredictorCorrector(entry=entry, provider=provider,
                              substeps=SUBSTEPS, dt=DT, mode="two-pass")
pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))

f4 = loadmat("FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/"
             "fixture_step5_t0.5000.mat", squeeze_me=True,
             struct_as_record=False)
hX = np.asarray(f4["h_X_vec"])
ZDOF = 9 * 175 + 2

pc.advance(n_substeps=1)
b = 1
import time; t0 = time.time()
for w in range(21):                       # to t* = 1+21*34 blocks ~ 1.43
    pc.advance()
    b += SUBSTEPS
    tip = entry.shell.q[ZDOF]
    if b < hX.shape[1]:
        ml = hX[ZDOF, b]
        r = tip / ml if abs(ml) > 1e-12 else float("nan")
        print(f"b={b:4d} t*={b*2e-3:.3f} tip={tip:+.6e} ml={ml:+.6e} "
              f"ratio={r:.4f} ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"b={b:4d} t*={b*2e-3:.3f} tip={tip:+.6e} (beyond truth)",
              flush=True)
