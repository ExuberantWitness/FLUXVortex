import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM flap=45deg 5deg twist0 2.3Hz: effect of wake length (steady 6.8N, paper ~7.8N):")
for wr in (1, 3, 10, 50):
    r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                           freq=2.3, n_cycle=5, steps_per_cycle=40, wake_rows=wr)
    print(f"  wake_rows={wr:3d}: L={r['L']:+6.2f}N  T={r['T']:+6.2f}N", flush=True)
print("DONE")
