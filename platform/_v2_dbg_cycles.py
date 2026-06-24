import numpy as np, warp as wp, time
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
spc=120; ncyc=10
print(f"3D UVLM flap=45 5deg twist0 2.3Hz, spc={spc}, wake=3cyc: per-cycle lift (is it still climbing?)", flush=True)
print("  (data ~7.79N; tests whether wake needs many cycles to develop)\n", flush=True)
t0=time.time()
r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                       freq=2.3, n_cycle=ncyc, steps_per_cycle=spc, wake_rows=spc*3)
Lh=r['Lh']*2
for c in range(ncyc):
    seg=Lh[c*spc:(c+1)*spc]
    print(f"  cycle {c+1:2d}: mean L={seg.mean():+6.2f}N", flush=True)
print(f"  [last-cycle reported value = {r['L']:+.2f}N]  [{time.time()-t0:.0f}s]", flush=True)
print("DONE")
