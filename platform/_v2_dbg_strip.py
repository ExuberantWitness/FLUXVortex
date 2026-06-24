import numpy as np
from flap_ldvm import FlapLDVM
U=8.0; c=0.287; freq=2.3; Om=2*np.pi*freq; dt=(1/freq)/60
a_b=np.radians(5.0)
h0=0.31   # plunge amplitude -> effective AoA swing ~+-29deg (matches mid-span flap)
m=FlapLDVM(U=U,c=c,n=40,dt=dt,rho=1.225,lesp_crit=0.20,max_wake=120)
L=[]; T=[]; AE=[]
for it in range(4*60):
    t=it*dt
    hdot=h0*Om*np.cos(Om*t)
    aeff = a_b + np.arctan2(-hdot,U)   # effective AoA (for reference)
    r=m.step(a_b,0.0,hdot)
    if it>=3*60:
        L.append(r['lift']); T.append(r['thrust']); AE.append(np.degrees(aeff))
L=np.array(L); T=np.array(T); AE=np.array(AE)
q=0.5*1.225*U*U
print(f"steady 5deg section lift ~ {0.518* q*c*(5/5):.2f} N/m (CL~0.52)")
print(f"plunge+5deg: mean lift={L.mean():+.2f} N/m  mean thrust={T.mean():+.2f} N/m")
print(f"  eff-AoA range [{AE.min():.0f},{AE.max():.0f}]deg")
print(f"  lift over last cycle (every 6th step):")
for i in range(0,60,6):
    print(f"    aeff={AE[i]:+5.0f}deg  lift={L[i]:+7.2f}  thrust={T[i]:+6.2f}")
