"""Visualize the nc=12 grid-independence config (joint spc=720) at the REAL Fig17 condition
(8 m/s, ±45° flap, 22.5° twist, Ansari-LEV) so we can eyeball that the physics is normal:
wing lattice (red=LE-separated/blue=attached), Ansari LEV sheet rolling up over the suction surface
(green/magenta = +/-Gamma), TEV trailing wake (faint = reverse-vK thrust street), and lift/thrust history."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v2_robo as R
from grid_indep import MODEL, COND

FREQ = COND['freq']; U = COND['U']; SPC = 720; NC = 12
frames = []
res = R.gpu_run_twist(**{**MODEL, 'n_cycle': 2}, nc=NC, ns=16, steps_per_cycle=SPC, wake_rows=SPC,
                      **COND, frames_out=frames, frame_skip=8)
print(f"recorded {len(frames)} frames; L_wind={res['L_wind']:.2f}N T_wind={res['T_wind']:.2f}N "
      f"L_bern={res['L_bern']:.2f}N", flush=True)

def smooth(a, w=9):
    k = np.ones(w) / w; return np.convolve(np.pad(a, w // 2, mode="edge"), k, mode="valid")[:len(a)]
LhB = smooth(2 * res["Lh_bern"]); dt = (1.0 / FREQ) / SPC
fig = plt.figure(figsize=(15, 6.2), facecolor="white")
ax = fig.add_subplot(121, projection="3d"); axh = fig.add_subplot(122)
allw = [f["bound"].reshape(-1, 3) for f in frames]
allw += [f["lev_rings"].reshape(-1, 3) for f in frames if len(f["lev_rings"])]
allw = np.concatenate(allw)
xr = [allw[:, 0].min() - 0.05, allw[:, 0].max() + 0.05]; yr = [-0.05, 0.85]
zr = [allw[:, 2].min() - 0.05, allw[:, 2].max() + 0.05]


def update(fi):
    ax.cla(); axh.cla()
    f = frames[fi]
    cols = ["#d62728" if s else "#4a90d9" for s in f["sep"]]
    ax.add_collection3d(Poly3DCollection([b for b in f["bound"]], alpha=0.85, facecolors=cols, edgecolor="#222", linewidth=0.3))
    bm = f["bound"].copy(); bm[..., 1] *= -1
    ax.add_collection3d(Poly3DCollection([b for b in bm], alpha=0.12, facecolors="#4a90d9", linewidth=0))
    if len(f["wr"]):                                   # TEV trailing wake (faint)
        seg, lc = [], []
        for ring, g in zip(f["wr"], f["wg"]):
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                seg.append([ring[a], ring[b]]); lc.append((.85, .3, .3, .28) if g > 0 else (.3, .3, .85, .28))
        ax.add_collection3d(Line3DCollection(seg, colors=lc, linewidths=0.4))
    if len(f["lev_rings"]):                            # Ansari LEV sheet over the suction surface (bold)
        seg, lc = [], []
        for ring, g in zip(f["lev_rings"], f["lev_g"]):
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                seg.append([ring[a], ring[b]]); lc.append((0., .7, 0., .9) if g > 0 else (.8, 0., .8, .9))
        ax.add_collection3d(Line3DCollection(seg, colors=lc, linewidths=1.4))
    ax.set_xlim(xr); ax.set_ylim(yr); ax.set_zlim(zr)
    ax.set_xlabel("x (chord, flow→)"); ax.set_ylabel("y (span)"); ax.set_zlabel("z")
    ax.set_title(f"nc=12 (spc=720) Ansari-LEV  t={f['t']:.3f}s\nGREEN/MAGENTA=LEV sheet (±Γ over suction surf), faint=TEV wake", fontsize=10)
    ax.view_init(elev=18, azim=-65)
    tt = np.arange(len(LhB)) * dt; cur = f["t"]; m = tt <= cur
    axh.plot(tt, LhB, "b-", lw=0.7, alpha=0.3)
    axh.plot(tt[m], LhB[m], "b-", lw=1.7, label=f"lift (Bernoulli ⟨{res['L_bern']:.1f}N⟩)")
    axh.axvline(cur, color="r", ls="--", lw=0.8); axh.axhline(0, color="gray", lw=0.5)
    axh.axhline(7.67, color="b", ls=":", lw=0.9, alpha=0.6)
    axh.set_xlim(dt * SPC * 0.0, tt[-1]); axh.set_ylim(-20, 30)
    axh.set_xlabel("time (s)"); axh.set_ylabel("lift (N)")
    axh.set_title(f"lift  ({U:.0f} m/s, {FREQ}Hz, ±45° flap, 22.5° twist)\nmeasured ~7.67N (dotted) — model L_wind={res['L_wind']:.1f}N", fontsize=9.5)
    axh.legend(fontsize=8.5, loc="upper right"); axh.grid(alpha=0.3)
    return []


anim = FuncAnimation(fig, update, frames=len(frames), interval=90, blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "_v2_nc12.gif")
anim.save(out, writer="pillow", fps=12, dpi=80)
print(f"saved {out}", flush=True); print("DONE", flush=True)
