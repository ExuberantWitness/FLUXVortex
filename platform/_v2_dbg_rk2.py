import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM flap=45 5deg twist0 2.3Hz: Euler vs RK2 convection (expected rigid value 0.736*6.81=5.0N):", flush=True)
print(f"{'spc':>5} {'Euler':>8} {'RK2':>8}", flush=True)
for spc in (60, 120, 240):
    rE = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                            freq=2.3, n_cycle=4, steps_per_cycle=spc, wake_rows=spc, rk2=False)
    rR = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                            freq=2.3, n_cycle=4, steps_per_cycle=spc, wake_rows=spc, rk2=True)
    print(f"{spc:5d} {rE['L']:7.2f}N {rR['L']:7.2f}N", flush=True)
print("DONE")
