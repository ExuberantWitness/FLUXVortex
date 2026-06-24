"""Proof-of-concept: does a sectional LDVM (with LEV shedding) sustain the flapping-wing lift where
the attached UVLM collapses? Drive ONE mid-span strip's LDVM with the flapping kinematics
(plunge from the +-45deg flap folded into an effective AoA + twist), get the cycle-mean sectional CL.
If CL ~ the steady value (LEV sustains it) the LEV approach is validated; if it collapses, it isn't."""
import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from lev_dvm import LDVM2D

U = 8.0; chord = 0.287; half_span = 0.80
flap_amp = np.radians(45.0); freq = 2.0; Om = 2 * np.pi * freq
body_aoa = np.radians(5.0)
y = 0.40                                  # mid-span strip
n_cycle = 6; steps_per_cycle = 80; dt = (1.0 / freq) / steps_per_cycle

for twist_deg in (0.0, 22.5):
    A_t = np.radians(twist_deg)
    ldvm = LDVM2D(U=U, c=chord, n=30, lesp_crit=0.11, dt=dt)
    a_prev = None; CLs = []; nlev_max = 0
    for it in range(n_cycle * steps_per_cycle):
        t = it * dt
        th = flap_amp * np.sin(Om * t); thd = flap_amp * Om * np.cos(Om * t)
        hdot = y * thd                                   # plunge velocity of the strip (flap about root)
        psi = A_t * (y / half_span) * np.sin(Om * t - np.pi / 2)   # twist, 90deg phase
        psidot = A_t * (y / half_span) * Om * np.cos(Om * t - np.pi / 2)   # geometric pitch RATE (twist)
        a_eff = body_aoa + np.arctan2(-hdot, U) + psi    # plunge folded into alpha (uniform downwash)
        r = ldvm.step(a_eff, psidot)                     # dalpha = twist rate ONLY (not the plunge-AoA rate)
        nlev_max = max(nlev_max, r['n_lev'])
        if it >= (n_cycle - 1) * steps_per_cycle:         # last cycle
            CLs.append(r['CL'])
    clm = np.mean(CLs)
    a_amp = np.degrees(np.arctan2(y * flap_amp * Om, U))
    print(f"  twist {twist_deg:4.1f}deg: eff-AoA swing +-{a_amp:.0f}deg  cycle-mean CL={clm:+.3f}  "
          f"(steady CL@5deg~0.4; attached flapping collapses to ~0.1)  LEVs shed={nlev_max}", flush=True)
print("DONE", flush=True)
