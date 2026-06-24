"""De-risk increment 1: does the LEV PARTICLE cloud roll up into a coherent core near the leading edge?
Side view (x-z plane = chordwise x vertical, where a spanwise LEV rolls up) at mid-span, animated.
Particles colored by circulation sign, sized by |alpha|. Wing section overlaid. Zoomed on the wing +
near wake (NOT the far convected wake). If the recent LE particles wind into a spiral/concentrated core
instead of trailing as a flat sheet, the rollup works."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v2_robo as R

FREQ = 2.0; U = 8.0; NS = 10; SPC = 80
frames = []
res = R.gpu_run_twist(nc=4, ns=NS, U=U, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                      freq=FREQ, n_cycle=3, steps_per_cycle=SPC, wake_rows=50, swept_axis=True,
                      real_geom=True, real_lev=True, lesp_crit_deg=11, lev_klev=1.0,
                      part_lev=True, frames_out=frames, frame_skip=2)
print(f"recorded {len(frames)} frames; L_bern={res['L_bern']:.2f}N", flush=True)

# mid-span band (y around half the half-span) to project onto the x-z plane
yband = (0.30, 0.55)
fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")

def update(fi):
    ax.cla()
    f = frames[fi]
    b = f["bound"]                                            # bound rings (npan,4,3)
    # wing section at mid-span: panels whose centroid y is in the band, draw chord line in x-z
    cz = b.reshape(-1, 4, 3).mean(1)
    msk = (cz[:, 1] > yband[0]) & (cz[:, 1] < yband[1])
    for ring in b[msk]:
        ax.plot(ring[[0, 1, 2, 3, 0], 0], ring[[0, 1, 2, 3, 0], 2], "-", color="#333", lw=1.0)
    # LEV particles in the band
    pp = f.get("pp", np.zeros((0, 3))); pa = f.get("pa", np.zeros((0, 3)))
    if len(pp):
        m = (pp[:, 1] > yband[0]) & (pp[:, 1] < yband[1]) & (np.linalg.norm(pa, axis=1) > 1e-9)
        P = pp[m]; A = pa[m]
        if len(P):
            g = A[:, 1]                                       # spanwise circulation component (sign)
            sz = 8 + 120 * np.linalg.norm(A, axis=1) / (np.abs(g).max() + 1e-12)
            ax.scatter(P[:, 0], P[:, 2], c=g, cmap="coolwarm", s=sz, alpha=0.8,
                       vmin=-np.abs(g).max(), vmax=np.abs(g).max(), edgecolors="none")
    ax.set_xlim(-0.2, 1.3); ax.set_ylim(-0.7, 0.9)
    ax.set_xlabel("x (chordwise, flow →)"); ax.set_ylabel("z (normal)")
    ax.set_aspect("equal")
    ax.set_title(f"LEV particle rollup (side view, mid-span)  t={f['t']:.3f}s\n"
                 f"do recent LE particles wind into a coherent core?  ({U:.0f} m/s, {FREQ} Hz, ±45°)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    return []

anim = FuncAnimation(fig, update, frames=len(frames), interval=90, blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v2_lev_rollup.gif")
anim.save(out, writer="pillow", fps=11, dpi=85)
print(f"saved {out}", flush=True)
