import numpy as np, sys
sys.path.insert(0,'.'); sys.path.insert(0,'..')
import flap_flight_validate as ffv
# PteraSoftware benchmark geometry: chord 1, half_span 2 (symmetric->full span 4), AR 4, AoA 5, V 10
U=10.0; chord=1.0; hs=2.0; aoa=5.0
S=2*chord*hs; AR=(2*hs)**2/S; CLa=2*np.pi*AR/(AR+2)
q=0.5*1.225*U**2
print(f"PteraSoftware geom: AR={AR:.1f}  analytic CL@5={CLa*np.radians(aoa):.3f}  L_analytic={q*S*CLa*np.radians(aoa):.1f}N",flush=True)
for nc,ns in ((5,10),(8,16)):
    r=ffv.gpu_run(nc=nc,ns=ns,chord=chord,half_span=hs,mass=1.0,U=U,aoa_deg=aoa,flap_amp_deg=0.0,freq=1.0,n_cycle=6,steps_per_cycle=40,verbose=False)
    cl=r['L']/(q*S)
    print(f"  my standalone UVLM nc={nc} ns={ns}: L={r['L']:.1f}N  CL={cl:.3f}  (analytic {CLa*np.radians(aoa):.3f}, ratio {cl/(CLa*np.radians(aoa)):.2f})",flush=True)
print("DONE",flush=True)
