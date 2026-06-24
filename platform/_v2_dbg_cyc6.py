import numpy as np, warp as wp, time
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
spc=120; ncyc=12
print(f"3D UVLM flap=45 5deg twist0 2.3Hz spc={spc}: per-cycle lift with 6-CYCLE wake retained", flush=True)
print("  (vs 3-cyc wake settled at 2.76N; does longer wake keep developing?)\n", flush=True)
t0=time.time()
r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                       freq=2.3, n_cycle=ncyc, steps_per_cycle=spc, wake_rows=spc*6)
Lh=r['Lh']*2
for c in range(ncyc):
    seg=Lh[c*spc:(c+1)*spc]; print(f"  cycle {c+1:2d}: mean L={seg.mean():+6.2f}N", flush=True)
print(f"  [{time.time()-t0:.0f}s]"); print("DONE")
