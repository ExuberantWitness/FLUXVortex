import numpy as np, warp as wp
wp.init()
import flap_flight_validate as ffv, _v2_robogeom as rg
ffv.flat_wing = lambda nc,ns,c,h: rg.robowing_real(nc, ns, h)   # REAL planform (raked TE)
import _v2_robo as robo
print("REAL geom (raked TE) + REAL swept flap/twist axis, flap=45 5deg 2.3Hz spc=240:", flush=True)
print("  paper: twist 22.5deg gives +7.8% lift, +47% thrust vs untwisted; flapping lift ~7.79N\n", flush=True)
# untwisted baseline + active twist 22.5 both phases, with swept axis
for tag, ta, ph, sw in [("twist0           ",0.0,-90.0,True),
                        ("twist22 ph-90 SWEPT",22.5,-90.0,True),
                        ("twist22 ph+90 SWEPT",22.5,90.0,True),
                        ("twist22 ph+90 OLDaxis",22.5,90.0,False)]:
    r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=ta,
                           twist_phase_deg=ph, freq=2.3, n_cycle=4, steps_per_cycle=240,
                           wake_rows=240, swept_axis=sw)
    print(f"  {tag}: L={r['L']:+6.2f}N  T={r['T']:+6.2f}N", flush=True)
print("DONE", flush=True)
