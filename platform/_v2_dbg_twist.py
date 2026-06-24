import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM flap=45deg, 5deg AoA, 2.3Hz — effect of feathering twist (paper flapping ~7.8N):")
for ta in (0.0, 22.5, 45.0):
    for ph in (-90.0, 90.0):
        if ta==0.0 and ph==90.0: continue
        r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0,
                               twist_amp_deg=ta, twist_phase_deg=ph, freq=2.3, n_cycle=5, steps_per_cycle=40)
        tag = "no twist" if ta==0 else f"twist{ta:.0f} ph{ph:+.0f}"
        print(f"  {tag:16s}: L={r['L']:+6.2f}N  T={r['T']:+6.2f}N", flush=True)
print("DONE")
