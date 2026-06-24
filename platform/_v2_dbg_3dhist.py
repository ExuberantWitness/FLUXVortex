import numpy as np, warp as wp
wp.init()
import robowing as rw, flap_flight_validate as ffv
ffv.flat_wing = lambda nc,ns,c,h: rw.robowing(nc,ns,c,h)
import _v2_robo as robo
spc=40
r = robo.gpu_run_twist(nc=6, ns=12, U=8.0, aoa_deg=5.0, flap_amp_deg=45.0,
                       twist_amp_deg=0.0, freq=2.3, n_cycle=5, steps_per_cycle=spc)
Lh = r['Lh']*2; Xh = r['Xh']*2   # both wings
last = Lh[(5-1)*spc:]; lastX = Xh[(5-1)*spc:]
print(f"flap=45deg 5deg: cycle-mean L={last.mean():+.2f}N (steady ~6.8N)  T={-lastX.mean():+.2f}N")
print(f"  per-step lift over last cycle (steady would oscillate around +6.8):")
th = np.degrees(np.radians(45.0)*np.sin(2*np.pi*np.arange(spc)/spc))
for i in range(0,spc,4):
    print(f"    dihedral={th[i]:+5.0f}deg  L={last[i]:+7.2f}N  T={-lastX[i]:+7.2f}N")
print(f"  L range [{last.min():+.1f},{last.max():+.1f}]  mean {last.mean():+.2f}")
