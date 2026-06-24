import numpy as np, warp as wp, time
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM flap=45 5deg 2.3Hz at PROPER resolution (spc=240,wake=1cyc): does feathering recover lift?", flush=True)
print("  (earlier 'twist barely matters' was at under-resolved spc=40). paper~7.8N\n", flush=True)
cases = [("twist0",0.0,-90.0),("twist22 ph-90",22.5,-90.0),("twist22 ph+90",22.5,90.0),
         ("twist45 ph-90",45.0,-90.0),("twist45 ph+90",45.0,90.0)]
for tag,ta,ph in cases:
    t0=time.time()
    r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=ta,
                           twist_phase_deg=ph, freq=2.3, n_cycle=4, steps_per_cycle=240, wake_rows=240)
    print(f"  {tag:15s}: L={r['L']:+6.2f}N T={r['T']:+6.2f}N  [{time.time()-t0:.0f}s]", flush=True)
print("DONE", flush=True)
