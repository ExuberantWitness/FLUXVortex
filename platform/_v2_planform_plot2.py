import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _v2_robogeom as rg

# real measured TE points (data convention: TE at negative chordwise x)
TE = np.array([
[800.0,-15.94],[800.0,-25.76],[800.0,-42.93],[797.77,-58.87],[794.44,-77.27],[789.99,-98.12],
[778.86,-122.65],[765.51,-148.41],[757.72,-159.44],[741.03,-181.52],[716.55,-207.28],[688.73,-228.13],
[668.71,-241.62],[634.21,-257.56],[613.07,-267.38],[565.23,-279.64],[526.29,-285.77],[499.58,-288.23],
[435.05,-287.0],[408.34,-287.0],[356.05,-287.0],[305.98,-287.0],[240.33,-287.0],[193.60,-285.77],
[119.05,-285.77],[13.35,-288.23]])

# NEW robowing_real mesh (built from those points): x=0 LE, x=+chord TE -> negate for data convention
nc, ns = 6, 24
C = rg.robowing_real(nc, ns)          # (nc+1, ns+1, 3): [x_chord, y_span, z_camber]
ym = C[..., 1] * 1000                  # mm
xm = -C[..., 0] * 1000                 # negate -> data convention (TE negative)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(TE[:,0], TE[:,1], 'k.', ms=11, label='REAL measured TE points (target)', zorder=5)
# robowing_real LE + TE outlines
ax.plot(ym[0,:], xm[0,:], 'b-', lw=2, label='robowing_real LE (x=0)')
ax.plot(ym[-1,:], xm[-1,:], 'r-', lw=2, label='robowing_real TE (built from points)')
# mesh
for j in range(0, ns+1, 2):
    ax.plot(ym[:,j], xm[:,j], 'g-', lw=0.4, alpha=0.5)
for i in range(nc+1):
    ax.plot(ym[i,:], xm[i,:], 'g-', lw=0.3, alpha=0.4)
# swept flap/twist axis (negate to data convention)
ax.plot([0,800], [-96.8932, 0], 'm-', lw=2.5, label='flap/twist axis (33.8%c root -> LE tip)')
ax.plot(588.6, -273.5, 'c*', ms=18, label='EI measurement pt', zorder=6)
ax.set_xlabel('spanwise y (mm)'); ax.set_ylabel('chordwise x (mm), LE=0')
ax.set_title('FIXED geometry: robowing_real (mesh) overlaid on REAL measured TE points')
ax.legend(loc='lower left', fontsize=8); ax.grid(alpha=0.3); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig('_v2_planform.png', dpi=115)
# how well does robowing_real match the real TE?
te_y = TE[:,0]; te_x = -TE[:,1]
fit = np.interp(te_y, ym[-1,::-1] if ym[-1,0]>ym[-1,-1] else ym[-1,:], 
                xm[-1,::-1]*-1 if ym[-1,0]>ym[-1,-1] else xm[-1,:]*-1)
print(f"saved _v2_planform.png  ;  robowing_real both-wing area = {rg.area_both():.4f} m^2 (target 0.4174)")
print(f"  max |TE mismatch| vs real points: {np.max(np.abs(rg.chord_at(te_y/1000)*1000 - te_x)):.0f} mm")
