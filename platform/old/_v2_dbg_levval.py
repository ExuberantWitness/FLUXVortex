import numpy as np, warp as wp
wp.init()
import flap_flight_validate as ffv, _v2_robogeom as rg
ffv.flat_wing = lambda nc,ns,c,h: rg.robowing_real(nc, ns, h)
import _v2_robo as robo
GF=9.81/1000; Kv=5.5
def run(U,aoa,fq): return robo.gpu_run_twist(nc=6,ns=12,U=U,aoa_deg=aoa,flap_amp_deg=45.0,twist_amp_deg=0.0,
        freq=fq,n_cycle=4,steps_per_cycle=240,wake_rows=240,swept_axis=True,lev=True,K_v=Kv)['L']
print(f"VALIDATE single K_v={Kv} (calibrated @2.3Hz/5deg/8m/s) across conditions, spc=240:", flush=True)
print("\n--- freq sweep @8m/s 5deg (Fig18b) ---", flush=True)
for fq,d in [(1.4,6.14),(2.0,7.45),(2.3,7.79),(2.6,7.73)]:
    L=run(8.0,5.0,fq); print(f"  {fq}Hz: L={L:.2f}N data={d:.2f} ratio={L/d:.2f}", flush=True)
print("--- AoA sweep @8m/s 2.3Hz (Fig19b) ---", flush=True)
for aoa,d in [(0.0,2.90),(10.0,12.11),(15.0,14.27)]:
    L=run(8.0,aoa,2.3); print(f"  {aoa}deg: L={L:.2f}N data={d:.2f} ratio={L/d:.2f}", flush=True)
print("DONE", flush=True)
