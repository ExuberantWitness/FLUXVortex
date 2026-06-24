import numpy as np
# Real TE points from RoboEagle paper (spanwise mm, chordwise mm; root LE=[0,0], LE straight at x=0)
TE = np.array([
[800.0,-15.94],[800.0,-25.76],[800.0,-42.93],[797.77,-58.87],[794.44,-77.27],[789.99,-98.12],
[778.86,-122.65],[765.51,-148.41],[757.72,-159.44],[741.03,-181.52],[716.55,-207.28],[688.73,-228.13],
[668.71,-241.62],[634.21,-257.56],[613.07,-267.38],[565.23,-279.64],[526.29,-285.77],[499.58,-288.23],
[435.05,-287.0],[408.34,-287.0],[356.05,-287.0],[305.98,-287.0],[240.33,-287.0],[193.60,-285.77],
[119.05,-285.77],[13.35,-288.23]])
y = TE[:,0]; chord = -TE[:,1]   # local chord = |chordwise TE| (LE at x=0)
order = np.argsort(y); y=y[order]; chord=chord[order]
print("Real RoboEagle planform (from TE points), LE straight at x=0:")
print(f"  root chord (y~13mm): {chord[0]:.0f}mm ;  half-span (max y): {y.max():.0f}mm")
print(f"  chord constant ~287mm up to y~{y[chord>280].max():.0f}mm, then tapers to {chord[-3:].mean():.0f}mm at tip")
# area of half-wing (trapz of chord over span), both wings x2
area_half = np.trapezoid(chord, y)/1e6  # m^2
print(f"  half-wing area = {area_half*1e4:.1f} cm^2 ;  BOTH wings = {2*area_half:.4f} m^2")
# chord at sample spanwise stations
print("  chord(y):", {int(yy): f"{cc:.0f}mm" for yy,cc in zip(y[::3], chord[::3])})

# compare to my robowing (constant 287 + quarter-circle round tip over last chord/2)
import robowing as rw
C0 = rw.robowing(6, 12, 0.287, 0.80)   # (nc+1, ns+1, 3)
yr = C0[0,:,1]*1000   # spanwise stations (mm)
# robowing local chord = LE-to-TE span at each spanwise station
le = C0[0,:,0]; te = C0[-1,:,0]; chord_r = (le-te)*1000
print("\nMy robowing planform:")
print(f"  BOTH wings area: 0.4415 m^2 (recorded earlier)")
print(f"  chord(y):", {int(yy): f"{cc:.0f}mm" for yy,cc in zip(yr[::2], chord_r[::2])})
# interp real chord onto robowing stations to compare
cr_real = np.interp(yr, y, chord)
print(f"\n  diff robowing vs real (mm) at stations:")
for yy, cr, crr in zip(yr, chord_r, cr_real):
    print(f"    y={yy:5.0f}mm: robowing={cr:5.0f}  real={crr:5.0f}  diff={cr-crr:+5.0f}mm")
