import numpy as np, warp as wp
wp.init()
import flap_flight_validate as ffv, _v2_robogeom as rg
ffv.flat_wing = lambda nc,ns,c,h: rg.robowing_real(nc, ns, h)
import _v2_robo as robo
GF=9.81/1000
data = {1.4:626.1, 2.0:759.4, 2.3:794.2, 2.6:788.4}   # Fig18b @8m/s lift (g)
print("3D UVLM + dynamic-stall LEV (Polhamus) vs RoboEagle Fig18b @8m/s 5deg twist0, real geom, spc=160:", flush=True)
print(f"{'freq':>5} {'data(N)':>8} {'attached':>9} {'+LEV':>8} {'ratio':>6}", flush=True)
for fq in (1.4, 2.0, 2.3, 2.6):
    r0 = robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=5.0,flap_amp_deg=45.0,twist_amp_deg=0.0,freq=fq,
                            n_cycle=4,steps_per_cycle=160,wake_rows=160,swept_axis=True,lev=False)
    r1 = robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=5.0,flap_amp_deg=45.0,twist_amp_deg=0.0,freq=fq,
                            n_cycle=4,steps_per_cycle=160,wake_rows=160,swept_axis=True,lev=True,K_v=2.5)
    dN = data[fq]*GF
    print(f"{fq:5.1f} {dN:8.2f} {r0['L']:8.2f}N {r1['L']:7.2f}N {r1['L']/dN:6.2f}", flush=True)
print("DONE", flush=True)
