import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)   # proper RoboEagle wing
import _v2_robo as robo
print("3D UVLM gpu_run_twist at 5deg AoA, twist0, 8 m/s — flap OFF vs ON:")
for fa in (0.0, 10.0, 25.0, 45.0):
    r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=fa,
                           twist_amp_deg=0.0, freq=2.3, n_cycle=5, steps_per_cycle=40)
    print(f"  flap={fa:4.1f}deg: L={r['L']:+6.2f}N  T={r['T']:+6.2f}N  (paper flapping ~7.8N)", flush=True)
print("DONE")
