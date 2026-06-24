import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from newton_pc import WindowPredictorCorrector
from newton_pc.adapters.flap import (FlapEntry, FlapKinematics, FlapUVLMProvider,
                                      NodalForceSet)

# Bare newton_pc wing aero, RIGID prescribed flapping (mode=kinematic) -> NO structural
# feedback, NO flight dynamics. If lift is physical (~few N) the aero is fine and the
# instability is in the FSI coupling/flight; if it is +-100s of N the aero itself is broken.
chord, span, nc, ns = 0.29, 0.85, 4, 6
flap_hz, amp_deg, V0 = 3.0, 22.0, 6.0
dtw = (chord / nc) / V0
substeps = 16
area = chord * span
q_dyn = 0.5 * 1.225 * V0 ** 2
print(f"wing {chord}x{span}m area={area:.3f}m^2  V={V0}m/s  q_dyn={q_dyn:.1f}Pa  "
      f"=> physical lift scale ~ q*S*CL ~ {q_dyn*area:.1f}N (CL=1)", flush=True)

kin = FlapKinematics(np.deg2rad(amp_deg), 1.0 / flap_hz)
entry = FlapEntry(chord, span, nc, ns, kin, mode="kinematic", thickness=1.2e-3,
                  rho_s=135.0, E0=50e9)
# freestream at 6 deg AoA (the feathered cruise condition)
aoa = np.deg2rad(6.0)
Vinf = V0 * np.array([1.0, 0.0, np.tan(aoa)])
provider = FlapUVLMProvider(Vinf, 1.225, dtw, K=6, chord=chord, particles=False,
                            max_particles=1)
pc = WindowPredictorCorrector(entry=entry, provider=provider, substeps=substeps,
                              dt=dtw / substeps, mode="two-pass")
pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))
pc.advance(n_substeps=1)

Fz = []; Fx = []
for i in range(24):
    pc.advance()
    payload = pc._F_cur.payload
    F = payload["f_panel"].sum(axis=(0, 1)) if payload else np.zeros(3)
    Fz.append(float(F[2])); Fx.append(float(F[0]))
    if i % 3 == 0:
        print(f"  win {i:2d}: F=({F[0]:+8.2f}, {F[1]:+8.2f}, {F[2]:+8.2f}) N", flush=True)
Fz = np.array(Fz); Fx = np.array(Fx)
print(f"cycle-mean lift Fz={np.mean(Fz[4:]):+.2f}N (std {np.std(Fz[4:]):.2f})  "
      f"thrust Fx={np.mean(Fx[4:]):+.2f}N", flush=True)
print(f"VERDICT: {'PHYSICAL (~few N) -> aero OK, problem is FSI/flight' if abs(np.mean(Fz[4:]))<50 and np.std(Fz[4:])<100 else 'GARBAGE (100s of N) -> bare aero is BROKEN'}", flush=True)
print("DONE", flush=True)
