"""Verify the speed claim: does the wing reach the weight anchor at the real cruise speed (10 m/s)?
Sweep AoA at U=10 (and 11) m/s, RoboEagle geometry, ±45° flap. Report absolute lift (N) vs the
weight anchors (E-Flap 6.9N, HIT-Hawk 11.3N)."""
import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flap_flight_validate as ffv

G = 9.81
print("Speed check — does the wing trim to weight at the real cruise (10-11 m/s)?", flush=True)
print(f"  anchors: E-Flap 6.9N, HIT-Hawk 11.3N.  RoboEagle geom, flap ±45°, 2 Hz, both wings", flush=True)
for U in (10.0, 11.0):
    print(f"\n  U={U} m/s:  AoA -> lift(N)  thrust(N)  power(W)", flush=True)
    lifts = []; aoas = (8.0, 10.0, 12.0, 15.0)
    for aoa in aoas:
        r = ffv.gpu_run(nc=4, ns=10, chord=0.287, half_span=0.80, mass=0.70, U=U,
                        aoa_deg=aoa, flap_amp_deg=45.0, freq=2.0, n_cycle=5, steps_per_cycle=40,
                        verbose=False)
        lifts.append(r["L"])
        print(f"   {aoa:4.1f}: {r['L']:+7.2f}  {r['T']:+7.2f}   {r['P']:6.1f}", flush=True)
    lifts = np.array(lifts); aoas = np.array(aoas)
    for W, name in ((0.70 * G, "E-Flap 700g"), (1.15 * G, "HIT-Hawk 1.15kg")):
        if lifts.max() >= W >= lifts.min():
            print(f"   -> trims to {name} ({W:.1f}N) at AoA {np.interp(W, lifts, aoas):.1f}°", flush=True)
        else:
            print(f"   -> max lift {lifts.max():.2f}N {'>=' if lifts.max()>=W else '<'} {name} {W:.1f}N", flush=True)
print("DONE", flush=True)
