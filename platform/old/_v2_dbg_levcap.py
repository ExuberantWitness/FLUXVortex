import numpy as np, warp as wp
wp.init()
import flap_flight_validate as ffv, _v2_robogeom as rg
ffv.flat_wing = lambda nc,ns,c,h: rg.robowing_real(nc, ns, h)
import _v2_robo as robo
def run(aoa,Kv,ds): return robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=aoa,flap_amp_deg=45.0,twist_amp_deg=0.0,
        freq=2.3,n_cycle=4,steps_per_cycle=240,wake_rows=240,swept_axis=True,lev=True,K_v=Kv,lev_ds_deg=ds)['L']
data={0.0:2.90,5.0:7.80,10.0:12.11,15.0:14.27}
print("AoA sweep with deep-stall cap (ds=33deg), tune K_v so 5deg~7.8; check slope flattens at 15deg:", flush=True)
for Kv in (6.5, 8.0):
    print(f"  --- K_v={Kv}, ds=33 ---", flush=True)
    for aoa in (0.0,5.0,10.0,15.0):
        L=run(aoa,Kv,33.0); print(f"    {aoa}deg: L={L:.2f}N data={data[aoa]:.2f} ratio={L/data[aoa]:.2f}", flush=True)
print("DONE", flush=True)
