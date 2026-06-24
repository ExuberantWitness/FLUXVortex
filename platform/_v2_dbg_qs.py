import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
print("3D UVLM flap=45deg 5deg twist0: lift vs FREQUENCY (isolate geometric dihedral vs unsteady motion)")
print("  quasi-steady (low freq, plunge->0) should give <cos^2(theta)>*6.8 ~ 5.1N if geometry is OK\n")
for fq in (0.1, 0.3, 1.0, 2.3):
    # keep dt sane: scale steps so dt ~ const-ish
    spc = max(40, int(40*fq/2.3*6))
    r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                           freq=fq, n_cycle=4, steps_per_cycle=spc, wake_rows=60)
    vplunge = 0.4*np.radians(45)*2*np.pi*fq
    print(f"  freq={fq:4.1f}Hz (mid-span plunge {vplunge:4.1f} m/s): L={r['L']:+6.2f}N  T={r['T']:+6.2f}N", flush=True)
print("DONE")
