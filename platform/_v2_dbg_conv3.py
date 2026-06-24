import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM flap=45 5deg twist0 2.3Hz: finer time steps (target strip=7.41N/paper7.8N):", flush=True)
for spc in (480, 640, 900):
    r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                           freq=2.3, n_cycle=3, steps_per_cycle=spc, wake_rows=80)
    print(f"  steps/cycle={spc:4d} (dt={(1/2.3)/spc*1000:4.2f}ms): L={r['L']:+6.2f}N  T={r['T']:+6.2f}N", flush=True)
print("DONE", flush=True)
