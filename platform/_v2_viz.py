"""Visualize the RoboEagle flapping UVLM working process (like figures/hybrid_k05_free.gif):
wing lattice (panels, red=LEV-separated / blue=attached) + shed TRAILING WAKE rings (colored by
circulation sign = reverse-von-Karman street) + lift/thrust history. Saves an animated GIF."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v2_robo as R

FREQ = 2.3; U = 8.0
frames = []
res = R.gpu_run_twist(nc=4, ns=12, U=U, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=22.5,
                      freq=FREQ, n_cycle=3, steps_per_cycle=100, wake_rows=60, swept_axis=True,
                      real_geom=True, real_lev=True, lesp_crit_deg=11, lev_klev=1.0,
                      frames_out=frames, frame_skip=2)
print(f"recorded {len(frames)} frames; L={res['L']:.2f}N T_inviscid={res['T']:.2f}N", flush=True)

Lh, Xh = res["Lh"], res["Xh"]
dt = (1.0 / FREQ) / 100
fig = plt.figure(figsize=(15, 6.2), facecolor="white")
ax = fig.add_subplot(121, projection="3d"); axh = fig.add_subplot(122)
# fixed limits
allw = np.concatenate([f["wr"].reshape(-1, 3) for f in frames if len(f["wr"])] + [frames[0]["bound"].reshape(-1, 3)])
xr = [allw[:, 0].min() - 0.05, allw[:, 0].max() + 0.05]
yr = [-0.05, 0.85]; zr = [allw[:, 2].min() - 0.05, allw[:, 2].max() + 0.05]
wgmax = max((np.abs(f["wg"]).max() if len(f["wg"]) else 1e-9) for f in frames) + 1e-9

def update(fi):
    ax.cla(); axh.cla()
    f = frames[fi]
    # wing panels: red = LEV-separated, blue = attached
    polys = [b for b in f["bound"]]
    cols = ["#d62728" if s else "#4a90d9" for s in f["sep"]]
    pc = Poly3DCollection(polys, alpha=0.85, facecolors=cols, edgecolor="#222", linewidth=0.3)
    ax.add_collection3d(pc)
    # mirror wing (other side, y<0) faint for context
    bm = f["bound"].copy(); bm[..., 1] *= -1
    ax.add_collection3d(Poly3DCollection([b for b in bm], alpha=0.15, facecolors="#4a90d9", linewidth=0))
    # wake rings: TEV (trailing) faint by circulation sign; LEV (leading-edge vortex) bold green/magenta
    if len(f["wr"]):
        wt = f.get("wtype", np.zeros(len(f["wr"]), int))
        segt, lct, segl, lcl = [], [], [], []
        for ring, g, ty in zip(f["wr"], f["wg"], wt):
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                if ty == 1:   # LEV — the real leading-edge vortex
                    segl.append([ring[a], ring[b]])
                    lcl.append((0.0, 0.7, 0.0, 0.85) if g > 0 else (0.8, 0.0, 0.8, 0.85))
                else:         # TEV trailing wake
                    segt.append([ring[a], ring[b]])
                    lct.append((0.85, 0.3, 0.3, 0.35) if g > 0 else (0.3, 0.3, 0.85, 0.35))
        if segt: ax.add_collection3d(Line3DCollection(segt, colors=lct, linewidths=0.4))
        if segl: ax.add_collection3d(Line3DCollection(segl, colors=lcl, linewidths=1.3))
    ax.set_xlim(xr); ax.set_ylim(yr); ax.set_zlim(zr)
    ax.set_xlabel("x (chordwise, flow →)"); ax.set_ylabel("y (span)"); ax.set_zlabel("z")
    ax.set_title(f"RoboEagle 3D LDVM (REAL leading-edge vortex)  t={f['t']:.3f}s\n"
                 f"GREEN/MAGENTA rings = shed LEV (±Γ),  faint red/blue = TEV trailing wake", fontsize=10)
    ax.view_init(elev=18, azim=-65)
    # lift/thrust history
    tt = np.arange(len(Lh)) * dt; cur = f["t"]
    axh.plot(tt, 2 * Lh, "b-", lw=0.8, alpha=0.4); axh.plot(tt, -2 * Xh, "g-", lw=0.8, alpha=0.4)
    m = tt <= cur
    axh.plot(tt[m], 2 * Lh[m], "b-", lw=1.6, label="lift (both wings)")
    axh.plot(tt[m], -2 * Xh[m], "g-", lw=1.6, label="thrust=−Fx (inviscid)")
    axh.axvline(cur, color="r", ls="--", lw=0.8); axh.axhline(0, color="gray", lw=0.5)
    axh.set_xlabel("time (s)"); axh.set_ylabel("force (N)"); axh.legend(fontsize=9, loc="upper right")
    axh.set_title(f"forces  ({U:.0f} m/s, {FREQ} Hz, ±45° flap, 22.5° twist)", fontsize=10)
    axh.grid(alpha=0.3)
    return []

anim = FuncAnimation(fig, update, frames=len(frames), interval=90, blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v2_real_lev.gif")
anim.save(out, writer="pillow", fps=11, dpi=80)
print(f"saved {out}", flush=True)
