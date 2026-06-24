import numpy as np
# eiDATA force-displacement (displacement mm, force N). Both labeled "11b" in file but one is 11a.
d1 = np.array([[0,0.01488],[4.994,0.20833],[9.988,0.40179],[15.054,0.55060],[20.048,0.69940],
[25.042,0.90774],[30.036,1.10119],[35.030,1.27976],[40.024,1.48810],[45.018,1.83036],
[50.012,2.02381],[55.006,2.33631]])
d2 = np.array([[0,0],[4.923,1.07143],[9.988,1.50298],[14.982,2.21726],[19.976,3.08036],
[25.042,4.10714],[29.964,4.83631],[34.958,6.01190],[40.024,6.65179],[44.946,7.50000],
[50.012,8.11012],[55.006,9.25595]])
L = 0.2735   # local chord at the measurement point (spanwise 588.6mm, chordwise 273.5mm)
for tag, d in [("dataset1 (flexible)", d1), ("dataset2 (stiff)", d2)]:
    k = np.polyfit(d[:,0]/1000.0, d[:,1], 1)[0]   # N/m (force vs displacement in m)
    EI = k * L**3 / 3.0
    print(f"  {tag}: k={k:.1f} N/m  ->  EI = k*L^3/3 = {EI:.4f} N*m^2")
k1=np.polyfit(d1[:,0]/1000,d1[:,1],1)[0]; k2=np.polyfit(d2[:,0]/1000,d2[:,1],1)[0]
print(f"  ratio stiff/flexible = {k2/k1:.2f}  (paper says EI ratio 3.98)")
print(f"  => 11a (original RoboEagle, flexible) = dataset1 EI~{k1*L**3/3:.3f} ; 11b (optimized, stiff) = dataset2 EI~{k2*L**3/3:.3f}")
print(f"  paper: aero experiments use STIFFENED 11b (minimize passive twist, isolate active twist)")
