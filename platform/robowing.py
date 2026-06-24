"""RoboEagle wing geometry: rectangular inner section + ROUNDED (quarter-circle / 'quarter-sphere')
wingtip, with optional COSINE chordwise clustering (refines the leading & trailing edges, per the
user's note that LE/TE refinement forms the leading-edge suction). Symmetric about the mid-chord line.

chord(y) = chord                              for y <= y_round
         = 2*sqrt(r^2 - (y-y_round)^2)        for y in [y_round, half_span],  r = chord/2,
i.e. the tip chord collapses to 0 at y=half_span following a quarter-circle (rounded tip cap).
"""
import numpy as np


def naca_camber(xc, m=0.02, p=0.40):
    """NACA 4-digit mean camber line yc/c (NACA 2406 -> m=0.02 max camber at p=0.40 chord).
    The VLM is built on this CAMBER SURFACE — a cambered airfoil produces lift at 0deg AoA,
    which is why the real wing has lift at 0deg where a flat plate has none."""
    xc = np.asarray(xc, float)
    return np.where(xc < p,
                    m / p ** 2 * (2 * p * xc - xc ** 2),
                    m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xc - xc ** 2))


def robowing(nc, ns, chord=0.287, half_span=0.80, round_tip=True, cosine_chord=True,
             round_frac=None, camber_m=0.02, camber_p=0.40):
    """Lattice corners (nc+1, ns+1, 3): rounded ('quarter-sphere') tip planform + NACA-2406 camber
    surface (z = yc(x/c)*c) + optional cosine chordwise clustering (refines LE & TE). Root at y=0.
    camber_m=0 -> flat plate."""
    r = (round_frac * half_span) if round_frac else (chord / 2.0)
    y_round = half_span - r
    midc = chord / 2.0
    ys = np.linspace(0.0, half_span, ns + 1)
    if cosine_chord:
        xf = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nc + 1)))
    else:
        xf = np.linspace(0.0, 1.0, nc + 1)
    zc = naca_camber(xf, camber_m, camber_p)          # camber/chord at each chordwise fraction
    C = np.zeros((nc + 1, ns + 1, 3))
    for j, y in enumerate(ys):
        if round_tip and y > y_round:
            dy = min(y - y_round, r)
            c = 2.0 * np.sqrt(max(r * r - dy * dy, 0.0))
        else:
            c = chord
        le = midc - c / 2.0; te = midc + c / 2.0
        for i, f in enumerate(xf):
            C[i, j] = [le + f * (te - le), y, zc[i] * c]   # z = camber * local chord
    return C


def area_both(chord, half_span, round_tip=True, round_frac=None):
    """Planform area of BOTH wings (rounded reduces it vs rectangular 2*chord*half_span)."""
    r = (round_frac * half_span) if round_frac else (chord / 2.0)
    y_round = half_span - r
    rect = chord * y_round
    cap = 0.25 * np.pi * r * chord   # integral of 2*sqrt(r^2-dy^2) over [0,r] = (pi/4)*... = (pi r^2)/2; with c=2r -> area
    # exact: ∫_0^r 2*sqrt(r^2-dy^2) dy = 2*(pi r^2/4) = pi r^2/2
    cap = 0.5 * np.pi * r * r
    return 2.0 * (rect + cap)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = robowing(12, 24, cosine_chord=True)
    Crect = robowing(12, 24, round_tip=False, cosine_chord=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    # plot planform outline (LE and TE)
    ax.plot(C[0, :, 1], C[0, :, 0], 'b-', label='rounded LE'); ax.plot(C[-1, :, 1], C[-1, :, 0], 'b-', label='rounded TE')
    ax.plot(Crect[0, :, 1], Crect[0, :, 0], 'r--', alpha=0.5); ax.plot(Crect[-1, :, 1], Crect[-1, :, 0], 'r--', alpha=0.5, label='rectangular')
    for j in range(0, 25, 2):
        ax.plot(C[:, j, 1], C[:, j, 0], 'k-', lw=0.3)
    ax.set_xlabel('span y (m)'); ax.set_ylabel('chord x (m)'); ax.set_aspect('equal'); ax.legend(); ax.invert_yaxis()
    ax.set_title(f'RoboEagle wing: rounded tip + cosine chord. area={area_both(0.287,0.80):.3f} vs rect {2*0.287*0.80:.3f} m^2')
    import os
    os.makedirs('docs', exist_ok=True); fig.tight_layout(); fig.savefig('docs/robowing_planform.png', dpi=120)
    print(f"saved docs/robowing_planform.png; rounded area(both)={area_both(0.287,0.80):.4f} m^2, "
          f"rectangular={2*0.287*0.80:.4f} m^2 (ratio {area_both(0.287,0.80)/(2*0.287*0.80):.3f})")
