"""Absolute lift trim: sweep AoA at the REAL RoboEagle geometry + ±45° flapping, report the
absolute cycle-mean LIFT (N) at each AoA, and find the trim AoA where lift = weight. The RoboEagle
paper gives NO absolute lift, so the only absolute anchor is weight (lift=weight for level flight):
E-Flap 700g -> 6.9N, HIT-Hawk 1.15kg -> 11.3N. Report power at trim vs the published 40-82 W band."""
import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flap_flight_validate as ffv

G = 9.81
print("Absolute LIFT trim — RoboEagle geometry (chord 0.287, half-span 0.80), flap ±45°, 8 m/s, 2 Hz", flush=True)
print("(no twist = lift baseline; twist trades lift for thrust). Both wings.", flush=True)
print(f"  weight anchors: E-Flap 700g -> {0.700*G:.1f}N ;  HIT-Hawk 1.15kg -> {1.15*G:.1f}N", flush=True)
print("  AoA :  lift(N)  thrust(N)  power(W)", flush=True)
res = []
for aoa in (0.0, 5.0, 8.0, 10.0, 12.0, 15.0):
    r = ffv.gpu_run(nc=4, ns=10, chord=0.287, half_span=0.80, mass=0.70, U=8.0,
                    aoa_deg=aoa, flap_amp_deg=45.0, freq=2.0, n_cycle=5, steps_per_cycle=40,
                    verbose=False)
    res.append((aoa, r["L"], r["T"], r["P"]))
    print(f"  {aoa:4.1f}: {r['L']:+7.2f}  {r['T']:+7.2f}   {r['P']:6.1f}", flush=True)
# interpolate the trim AoA for lift = E-Flap weight (6.9N)
aoas = np.array([x[0] for x in res]); lifts = np.array([x[1] for x in res])
W = 0.70 * G
if lifts.max() >= W >= lifts.min():
    aoa_trim = float(np.interp(W, lifts, aoas))
    print(f"\n  -> trims to E-Flap weight {W:.1f}N at AoA = {aoa_trim:.1f}°", flush=True)
else:
    print(f"\n  -> does NOT reach {W:.1f}N within 0-15° (max lift {lifts.max():.2f}N at 15°)", flush=True)
print("DONE", flush=True)
