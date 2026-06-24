import numpy as np
from flap_ldvm import FlapLDVM
U=8.0; c=0.287; freq=2.3; Om=2*np.pi*freq
a_b=np.radians(5.0); h0=0.31
q=0.5*1.225*U*U
print(f"steady 5deg section lift ~ {0.52*q*c:.2f} N/m")
for lesp,spc,tag in [(99.0,120,"LEV off, 120/cyc"),(99.0,240,"LEV off, 240/cyc"),
                     (0.20,120,"LEV on,  120/cyc"),(0.20,240,"LEV on,  240/cyc")]:
    dt=(1/freq)/spc
    m=FlapLDVM(U=U,c=c,n=40,dt=dt,rho=1.225,lesp_crit=lesp,max_wake=200)
    L=[]; T=[]
    for it in range(5*spc):
        t=it*dt; hdot=h0*Om*np.cos(Om*t)
        r=m.step(a_b,0.0,hdot)
        if it>=4*spc: L.append(r['lift']); T.append(r['thrust'])
    print(f"  {tag}: mean lift={np.mean(L):+.2f} N/m  mean thrust={np.mean(T):+.2f}  (std lift={np.std(L):.1f})")
