"""Diagnose the lift gap: sweep flap amplitude + mesh at 8 m/s, 5deg AoA, 2 Hz, NO twist, vs the
RoboEagle measured 678 gf = 6.65 N (Fig 18d, 8 m/s 2 Hz twist 0, both wings). Hypothesis: my attached
UVLM loses lift as flap amplitude grows (effective AoA swings to +-44deg at +-45 flap, where the real
wing sustains lift via LEV/dynamic stall). If lift drops sharply with amplitude -> the gap is
LEV/dynamic-stall, not mesh."""
import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flap_flight_validate as ffv

G = 9.81
U = 8.0; aoa = 5.0; freq = 2.0; chord = 0.287; hs = 0.80
PAPER = 0.678 * G   # 678 gf at twist 0, 8 m/s, 2 Hz, 5deg (Fig 18d) -> N, both wings
# effective-AoA swing estimate at the tip:
for amp in (45.0,):
    vft = np.radians(amp) * 2 * np.pi * freq * hs
    print(f"  flap +-{amp}deg @ {freq}Hz: tip flap speed {vft:.1f} m/s vs U {U} -> eff-AoA swing +-{np.degrees(np.arctan2(vft,U)):.0f}deg", flush=True)
print(f"\nPAPER (twist 0, 8 m/s, 2 Hz, 5deg, both wings): lift = {PAPER:.2f} N", flush=True)
print("  flap-amplitude sweep (attached UVLM, no twist):", flush=True)
for amp in (0.0, 5.0, 10.0, 20.0, 30.0, 45.0):
    r = ffv.gpu_run(nc=4, ns=10, chord=chord, half_span=hs, mass=0.7, U=U, aoa_deg=aoa,
                    flap_amp_deg=amp, freq=freq, n_cycle=5, steps_per_cycle=40, verbose=False)
    print(f"   flap +-{amp:4.1f}deg: lift={r['L']:+6.2f}N  thrust={r['T']:+6.2f}N  (paper {PAPER:.2f}N, ratio {r['L']/PAPER:.2f})", flush=True)
print("  mesh sweep at flap +-45deg (is it under-resolution?):", flush=True)
for nc, ns in ((4, 10), (8, 16), (12, 24)):
    r = ffv.gpu_run(nc=nc, ns=ns, chord=chord, half_span=hs, mass=0.7, U=U, aoa_deg=aoa,
                    flap_amp_deg=45.0, freq=freq, n_cycle=5, steps_per_cycle=40, verbose=False)
    print(f"   nc={nc} ns={ns}: lift={r['L']:+6.2f}N  (paper {PAPER:.2f}N, ratio {r['L']/PAPER:.2f})", flush=True)
print("DONE", flush=True)
