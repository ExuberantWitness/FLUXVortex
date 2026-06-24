import numpy as np
from flap_ldvm import FlapLDVM
# Drive a mid-span strip with the flapping kinematics at several body AoA; LEV on vs off; cycle-mean lift.
# Want: LEV-on gives POSITIVE extra lift (not negative=over-cap), and a CONCAVE AoA dependence (stall).
U=8.0; chord=0.287; half_span=0.80; freq=2.3; Om=2*np.pi*freq
flap_amp=np.radians(45.0); y=0.45; spc=120; dt=(1/freq)/spc
print("flap_ldvm sectional dynamic stall (mid-span strip, flapping), LEV on vs off, AoA sweep:", flush=True)
print(f"{'AoA':>4} {'attached':>9} {'+LEV':>8} {'increment':>10} {'maxLEVs':>8}", flush=True)
for aoa_deg in (0.0, 5.0, 10.0, 15.0):
    a_b=np.radians(aoa_deg); res={}
    for lev,crit in [("off",99.0),("on",0.20)]:
        m=FlapLDVM(U=U,c=chord,n=30,dt=dt,rho=1.225,lesp_crit=crit,max_wake=700)
        L=[]; nlev=0
        for it in range(5*spc):
            t=it*dt; thd=flap_amp*Om*np.cos(Om*t); hdot=y*thd
            psidot=0.0
            r=m.step(a_b, psidot, hdot); nlev=max(nlev,r['n_lev'])
            if it>=4*spc: L.append(r['lift'])
        res[lev]=(np.mean(L),nlev)
    inc=res['on'][0]-res['off'][0]
    print(f"{aoa_deg:4.0f} {res['off'][0]:8.2f} {res['on'][0]:7.2f} {inc:+10.2f} {res['on'][1]:8d}", flush=True)
print("DONE", flush=True)
