import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
# static check (flap=0): te_traj must not change it (~6.81N)
rs0 = robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=5.0,flap_amp_deg=0.0,twist_amp_deg=0.0,freq=2.3,
                         n_cycle=5,steps_per_cycle=40,wake_rows=80,te_traj=False)
rs1 = robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=5.0,flap_amp_deg=0.0,twist_amp_deg=0.0,freq=2.3,
                         n_cycle=5,steps_per_cycle=40,wake_rows=80,te_traj=True)
print(f"STATIC flap=0 (sanity): standard={rs0['L']:.2f}N  te_traj={rs1['L']:.2f}N  (must match ~6.8N)\n", flush=True)
print("FLAPPING flap=45 5deg 2.3Hz: standard vs te_traj (expected rigid ~5.0N):", flush=True)
print(f"{'spc':>5} {'standard':>9} {'te_traj':>9}", flush=True)
for spc in (60, 120, 240):
    r0 = robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=5.0,flap_amp_deg=45.0,twist_amp_deg=0.0,freq=2.3,
                            n_cycle=4,steps_per_cycle=spc,wake_rows=spc,te_traj=False)
    r1 = robo.gpu_run_twist(nc=6,ns=12,U=8.0,aoa_deg=5.0,flap_amp_deg=45.0,twist_amp_deg=0.0,freq=2.3,
                            n_cycle=4,steps_per_cycle=spc,wake_rows=spc,te_traj=True)
    print(f"{spc:5d} {r0['L']:8.2f}N {r1['L']:8.2f}N", flush=True)
print("DONE")
