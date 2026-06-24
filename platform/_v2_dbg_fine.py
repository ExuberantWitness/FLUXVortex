import numpy as np, warp as wp, time
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM flap=45 5deg twist0 2.3Hz: FINE dt (te_traj, wake=1cyc) — rigid 5N or data 7N?", flush=True)
for spc in (360, 540, 760):
    t0=time.time()
    r = robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=5.0,flap_amp_deg=45.0,twist_amp_deg=0.0,freq=2.3,
                           n_cycle=3,steps_per_cycle=spc,wake_rows=spc,te_traj=True)
    print(f"  spc={spc} (dt={(1/2.3)/spc*1000:.2f}ms): L={r['L']:.2f}N  [{time.time()-t0:.0f}s]", flush=True)
print("DONE")
