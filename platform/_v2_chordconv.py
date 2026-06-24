import numpy as np, sys
sys.path.insert(0,'.'); sys.path.insert(0,'..')
import flap_flight_validate as ffv
# High-AR wing -> 2D limit: CL slope should -> 2*pi*AR/(AR+2). Check chordwise (nc) convergence.
U=10.0; chord=0.10; hs=2.0; aoa=5.0
S=2*chord*hs; AR=(2*hs)**2/S; CLa=2*np.pi*AR/(AR+2); q=0.5*1.225*U**2
print(f"high-AR wing: chord {chord} span {2*hs} AR {AR:.0f}  -> analytic CL@5={CLa*np.radians(aoa):.3f} (2D 2pi*a={2*np.pi*np.radians(aoa):.3f})",flush=True)
print("chordwise(nc) convergence, STEADY (flap 0), ns=20:",flush=True)
for nc in (1,2,4,8):
    r=ffv.gpu_run(nc=nc,ns=20,chord=chord,half_span=hs,mass=1.0,U=U,aoa_deg=aoa,flap_amp_deg=0.0,freq=1.0,n_cycle=8,steps_per_cycle=40,verbose=False)
    cl=r['L']/(q*S)
    print(f"  nc={nc}: CL={cl:.3f}  (analytic {CLa*np.radians(aoa):.3f}, ratio {cl/(CLa*np.radians(aoa)):.2f})",flush=True)
print("spanwise(ns) convergence at nc=4:",flush=True)
for ns in (10,20,40):
    r=ffv.gpu_run(nc=4,ns=ns,chord=chord,half_span=hs,mass=1.0,U=U,aoa_deg=aoa,flap_amp_deg=0.0,freq=1.0,n_cycle=8,steps_per_cycle=40,verbose=False)
    cl=r['L']/(q*S)
    print(f"  ns={ns}: CL={cl:.3f}  (ratio {cl/(CLa*np.radians(aoa)):.2f})",flush=True)
print("DONE",flush=True)
