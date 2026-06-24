import numpy as np, warp as wp
wp.init()
import flap_flight_validate as ffv, _v2_robogeom as rg
ffv.flat_wing = lambda nc,ns,c,h: rg.robowing_real(nc, ns, h)
import _v2_robo as robo
GF=9.81/1000
data = {1.4:626.1, 2.0:759.4, 2.3:794.2, 2.6:788.4}
print("LEV tune: converged dt (spc=240) + K_v sweep @2.3Hz, then freq sweep at best K_v:", flush=True)
# tune K_v at 2.3Hz (data 7.79N)
for Kv in (4.0, 5.5, 7.0):
    r = robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=5.0,flap_amp_deg=45.0,twist_amp_deg=0.0,freq=2.3,
                           n_cycle=4,steps_per_cycle=240,wake_rows=240,swept_axis=True,lev=True,K_v=Kv)
    print(f"  2.3Hz K_v={Kv}: L={r['L']:.2f}N (data 7.79, ratio {r['L']/7.79:.2f})", flush=True)
print("DONE", flush=True)
