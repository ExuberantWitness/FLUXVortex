import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TE = np.array([
[800.0,-15.94],[800.0,-25.76],[800.0,-42.93],[797.77,-58.87],[794.44,-77.27],[789.99,-98.12],
[778.86,-122.65],[765.51,-148.41],[757.72,-159.44],[741.03,-181.52],[716.55,-207.28],[688.73,-228.13],
[668.71,-241.62],[634.21,-257.56],[613.07,-267.38],[565.23,-279.64],[526.29,-285.77],[499.58,-288.23],
[435.05,-287.0],[408.34,-287.0],[356.05,-287.0],[305.98,-287.0],[240.33,-287.0],[193.60,-285.77],
[119.05,-285.77],[13.35,-288.23]])
y_te = TE[:,0]; x_te = TE[:,1]
o = np.argsort(y_te); y_te=y_te[o]; x_te=x_te[o]

fig, ax = plt.subplots(figsize=(11,5))
# real planform: LE straight at x=0 (y 0..800), TE = points
ax.plot([0,800],[0,0],'b-',lw=2,label='LE (spar, x=0)')
ax.plot(y_te, x_te,'b.-',lw=1.5,ms=8,label='real TE (paper points)')
ax.fill_between(y_te, 0, x_te, alpha=0.12, color='blue')
# robowing outline
import robowing as rw
C0 = rw.robowing(2,40,0.287,0.80)
yr = C0[0,:,1]*1000; ler = C0[0,:,0]*1000; ter = C0[-1,:,0]*1000
ax.plot(yr, ter-ler,'r--',lw=1.5,label='my robowing TE (approx)')
# flap/twist axis AB: (0,-96.89)->(800,0)
ax.plot([0,800],[-96.8932,0],'g-',lw=2.5,label='flap+twist axis (real: 33.8%c root -> LE tip)')
# my old twist axis (quarter chord, 25%): -0.25*chord, chord~287 inboard tapering
ax.plot([0,526,800],[-0.25*287,-0.25*287,0],'m:',lw=2,label='my old twist axis (1/4 chord)')
ax.plot(588.6,-273.5,'k*',ms=16,label='EI measurement pt')
ax.set_xlabel('spanwise y (mm)'); ax.set_ylabel('chordwise x (mm), LE=0')
ax.set_title('RoboEagle wing planform: REAL (paper TE points) vs my robowing + flap/twist axis')
ax.legend(loc='lower left',fontsize=8); ax.grid(alpha=0.3); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig('_v2_planform.png',dpi=110)
print("saved _v2_planform.png")
# key metrics
chord=-x_te; area=np.trapezoid(chord,y_te)/1e6
print(f"real: half area {area*1e4:.0f}cm2, both {2*area:.4f}m2; taper from y~{y_te[chord>280].max():.0f}mm")
print(f"axis sweep: root x={-96.8932:.1f}mm ({96.8932/287*100:.1f}% chord) -> tip x=0 (LE)")
