import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM 5deg AoA, twist0, 2.3Hz: L(full panel_force) vs Lkj(Vinf-only, no plunge tilt):")
print("  paper flapping ~7.8N, steady ~6.8N\n")
for fa in (0.0, 25.0, 45.0):
    r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=fa,
                           twist_amp_deg=0.0, freq=2.3, n_cycle=5, steps_per_cycle=40)
    print(f"  flap={fa:4.1f}deg: L_panelforce={r['L']:+6.2f}N   L_kj(Vinf-only)={r['Lkj']:+6.2f}N", flush=True)
print("DONE")
