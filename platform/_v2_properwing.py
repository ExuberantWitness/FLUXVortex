"""Build the wing CORRECTLY in the user's order: (1) rounded tip geometry, (2) NACA-2406 camber
surface, (3) cosine chordwise distribution. Test the steady lift at 0deg/5deg and the chordwise
nc-convergence on the PROPER wing (does building it right fix the flat-plate nc-divergence?)."""
import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flap_flight_validate as ffv
import robowing as rw

ORIG = ffv.flat_wing          # capture original to avoid recursion
U = 8.0; chord = 0.287; hs = 0.80; q = 0.5 * 1.225 * U ** 2


def run(geomfn, aoa, nc=8, ns=16, area=None):
    ffv.flat_wing = geomfn
    r = ffv.gpu_run(nc=nc, ns=ns, chord=chord, half_span=hs, mass=0.7, U=U, aoa_deg=aoa,
                    flap_amp_deg=0.0, freq=1.0, n_cycle=6, steps_per_cycle=40, verbose=False)
    ffv.flat_wing = ORIG
    return r['L']


G = {
 '1 flat plate uniform':        lambda nc, ns, c, h: ORIG(nc, ns, c, h),
 '2 +rounded tip':              lambda nc, ns, c, h: rw.robowing(nc, ns, c, h, round_tip=True, cosine_chord=False, camber_m=0.0),
 '3 +NACA2406 camber':          lambda nc, ns, c, h: rw.robowing(nc, ns, c, h, round_tip=True, cosine_chord=False, camber_m=0.02),
 '4 +cosine chord (PROPER)':    lambda nc, ns, c, h: rw.robowing(nc, ns, c, h, round_tip=True, cosine_chord=True, camber_m=0.02),
}

print("STEADY lift (N, both wings) at 0deg and 5deg, building the wing up in order:", flush=True)
print("  (data anchor: 0deg flapping ~2.9N, 5deg steady ~6-7N; camber should give lift at 0deg)", flush=True)
for name, gf in G.items():
    L0 = run(gf, 0.0); L5 = run(gf, 5.0)
    print(f"  {name:28s}: L(0deg)={L0:+5.2f}N   L(5deg)={L5:+5.2f}N", flush=True)

print("\nchordwise nc-convergence on the PROPER wing (rounded+NACA2406+cosine), 5deg:", flush=True)
print("  (flat-plate uniform DIVERGED: nc 2->8 gave 0.41->0.21x; does proper wing converge?)", flush=True)
gf = G['4 +cosine chord (PROPER)']
for nc in (2, 4, 8, 12):
    L = run(gf, 5.0, nc=nc)
    print(f"  nc={nc:2d}: L(5deg)={L:.2f}N", flush=True)
print("DONE", flush=True)
