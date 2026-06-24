import numpy as np, warp as wp, time
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM flap=45 5deg twist0 2.3Hz: CLEAN convergence (wake=1 cycle for ALL dt):", flush=True)
for spc in (80, 160, 240):
    t0=time.time()
    r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                           freq=2.3, n_cycle=4, steps_per_cycle=spc, wake_rows=spc)  # wake=1 full cycle
    print(f"  spc={spc:4d} wake=1cyc (dt={(1/2.3)/spc*1000:4.2f}ms): L={r['L']:+6.2f}N T={r['T']:+6.2f}N  [{time.time()-t0:.0f}s]", flush=True)
print("DONE", flush=True)
