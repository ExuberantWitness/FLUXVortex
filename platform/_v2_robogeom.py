"""REAL RoboEagle wing geometry + flap/twist axis, from the paper's measured trailing-edge points
and the two fixed axis points (NOT my earlier robowing approximation / quarter-chord axis).

Planform: LE straight at x=0 (carbon spar), TE = measured points -> local chord(y). Constant ~287mm
to y~526mm, then raked/tapered tip to ~28mm at y=800mm. NACA-2406 camber on the surface.

Flap+twist axis (two measured fixed points, root LE = [0,0]): root (y=0) at chordwise 96.89mm
(=33.8% chord) aft of LE, swept to the LE (0) at the tip (y=800mm). x_axis(y) = 0.09689*(1 - y/0.80).
Both the +-45deg flap (dihedral) and the spanwise twist pitch about THIS swept line.
"""
import numpy as np

# measured TE points: spanwise y (mm), local chord = |chordwise| (mm). (root + tip-edge added)
_Y = np.array([0.0, 13.35, 119.05, 193.60, 240.33, 305.98, 356.05, 408.34, 435.05, 499.58,
               526.29, 565.23, 613.07, 634.21, 668.71, 688.73, 716.55, 741.03, 757.72, 765.51,
               778.86, 789.99, 794.44, 797.77, 800.0])
_C = np.array([287.0, 288.23, 285.77, 285.77, 287.0, 287.0, 287.0, 287.0, 287.0, 288.23,
               285.77, 279.64, 267.38, 257.56, 241.62, 228.13, 207.28, 181.52, 159.44, 148.41,
               122.65, 98.12, 77.27, 58.87, 28.0])           # tip edge ~28mm (mean of 16/26/43)
AXIS_ROOT_X = 0.09689     # 96.89 mm aft of LE = 33.8% root chord (measured fixed point)
HALF_SPAN = 0.80


def chord_at(y_m):
    """Local chord (m) at spanwise y (m), from the measured TE points."""
    return np.interp(np.asarray(y_m) * 1000.0, _Y, _C) / 1000.0


def axis_x(y_m, half_span=HALF_SPAN):
    """Chordwise x (m, aft of LE) of the flap/twist axis at spanwise y: swept 33.8%c(root)->LE(tip)."""
    return AXIS_ROOT_X * (1.0 - np.asarray(y_m) / half_span)


def naca_camber(xc, m=0.02, p=0.40):
    xc = np.asarray(xc, float)
    return np.where(xc < p, m / p ** 2 * (2 * p * xc - xc ** 2),
                    m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xc - xc ** 2))


def robowing_real(nc, ns, half_span=HALF_SPAN, camber_m=0.02, camber_p=0.40, cosine_chord=True):
    """Lattice corners (nc+1, ns+1, 3): real planform (LE straight x=0, raked TE chord(y)) + NACA2406
    camber. Drop-in for ffv.flat_wing(nc, ns, chord, half_span) — chord arg ignored (uses measured)."""
    ys = np.linspace(0.0, half_span, ns + 1)
    cy = chord_at(ys)
    xf = (0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nc + 1))) if cosine_chord
          else np.linspace(0.0, 1.0, nc + 1))
    zc = naca_camber(xf, camber_m, camber_p)
    C = np.zeros((nc + 1, ns + 1, 3))
    for j, y in enumerate(ys):
        c = max(cy[j], 1e-4)
        for i, f in enumerate(xf):
            C[i, j] = [f * c, y, zc[i] * c]      # LE at x=0, TE at x=c (chord in +x = flow dir)
    return C


def area_both(half_span=HALF_SPAN):
    ys = np.linspace(0.0, half_span, 400)
    return 2.0 * float(np.trapezoid(chord_at(ys), ys))


if __name__ == "__main__":
    print(f"real RoboEagle: both-wing area = {area_both():.4f} m^2 (target 0.4174)")
    print(f"  chord: root {chord_at(0.0)*1000:.0f}mm, mid {chord_at(0.4)*1000:.0f}mm, "
          f"y=0.6 {chord_at(0.6)*1000:.0f}mm, tip {chord_at(0.79)*1000:.0f}mm")
    print(f"  flap/twist axis x: root {axis_x(0.0)*1000:.1f}mm ({axis_x(0.0)/chord_at(0)*100:.1f}%c), "
          f"mid {axis_x(0.4)*1000:.1f}mm, tip {axis_x(0.8)*1000:.1f}mm")
