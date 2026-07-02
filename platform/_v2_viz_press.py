"""GIF: wing SURFACE PRESSURE (Cp, per-panel unsteady-Bernoulli) + LEV (green/magenta ±Γ) + TEV wake (faint
reverse-vK street) at the real RoboEagle flapping condition, with the sqrtx LE-suction + Ansari LEV. Lets you
eyeball whether the system works: pressure concentrated near the LE (suction), LEV rolling up over the suction
surface, TEV trailing behind, force signs sane.  python _v2_viz_press.py [nc] [twist]"""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v2_robo as R
from grid_indep import MODEL

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TW = float(sys.argv[2]) if len(sys.argv) > 2 else 22.5
FREQ, U, SPC = 2.0, 8.0, 160
b = {k: v for k, v in MODEL.items() if k not in ('n_cycle', 'lev_place')}
frames = []
res = R.gpu_run_twist(U=U, aoa_deg=5.0, freq=FREQ, twist_amp_deg=TW, twist_phase_deg=90.0, nc=NC, ns=16,
                      n_cycle=2, steps_per_cycle=SPC, wake_rows=SPC, a0_mode='sqrtx', lev_place='wake',
                      lev_consistent=False, frames_out=frames, frame_skip=2, **b)
print(f"recorded {len(frames)} frames; L_wind={res['L_wind']:.2f}N T_wind={res['T_wind']:.2f}N", flush=True)

# global Cp colour range (robust 92nd pct so the LE suction peak doesn't wash out the rest)
allcp = np.concatenate([f['cp'] for f in frames])
cmax = float(np.percentile(np.abs(allcp), 92)) or 1.0
norm = colors.Normalize(vmin=-cmax, vmax=cmax); cmap = cm.get_cmap('RdBu_r')   # red=high p (under), blue=suction


def smooth(a, w=7):
    k = np.ones(w) / w; return np.convolve(np.pad(a, w // 2, mode="edge"), k, mode="valid")[:len(a)]
LhB = smooth(2 * res["Lh_bern"]); Tnet = smooth(2 * (-(res["Xh_bern"] + res["Xh_les"]))); dt = (1.0 / FREQ) / SPC
fig = plt.figure(figsize=(15, 6.4), facecolor="white")
ax = fig.add_subplot(121, projection="3d"); axh = fig.add_subplot(122)
allw = np.concatenate([f["bound"].reshape(-1, 3) for f in frames] +
                      [f["lev_rings"].reshape(-1, 3) for f in frames if len(f["lev_rings"])])
xr = [allw[:, 0].min() - 0.05, allw[:, 0].max() + 0.05]; yr = [-0.05, 0.85]
zr = [allw[:, 2].min() - 0.05, allw[:, 2].max() + 0.05]
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02); cbar.set_label("Cp (blue = suction)")


def update(fi):
    ax.cla(); axh.cla()
    f = frames[fi]
    facec = cmap(norm(f['cp']))                                   # per-panel surface pressure colour
    ax.add_collection3d(Poly3DCollection([q for q in f["bound"]], facecolors=facec, edgecolor="#333",
                                         linewidth=0.2, alpha=1.0))
    bm = f["bound"].copy(); bm[..., 1] *= -1                       # mirror wing (faint, same Cp)
    ax.add_collection3d(Poly3DCollection([q for q in bm], facecolors=facec, linewidth=0, alpha=0.25))
    # wake rings split by type: wtype==1 = LEV (bold green/magenta ±Γ), wtype==0 = TEV (faint red/blue)
    if len(f["wr"]):
        wt = f.get("wtype", np.zeros(len(f["wr"]), int))
        segt, lct, segl, lcl = [], [], [], []
        for ring, g, ty in zip(f["wr"], f["wg"], wt):
            for a_, b_ in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                if ty == 1:
                    segl.append([ring[a_], ring[b_]]); lcl.append((0, .75, 0, .95) if g > 0 else (.85, 0, .85, .95))
                else:
                    segt.append([ring[a_], ring[b_]]); lct.append((.85, .3, .3, .28) if g > 0 else (.3, .3, .85, .28))
        if segt: ax.add_collection3d(Line3DCollection(segt, colors=lct, linewidths=0.4))
        if segl: ax.add_collection3d(Line3DCollection(segl, colors=lcl, linewidths=1.6))
    if len(f["lev_rings"]):                                       # (fallback) anchored-Ansari sheet if used
        seg, lc = [], []
        for ring, g in zip(f["lev_rings"], f["lev_g"]):
            for a_, b_ in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                seg.append([ring[a_], ring[b_]]); lc.append((0, .75, 0, .95) if g > 0 else (.85, 0, .85, .95))
        ax.add_collection3d(Line3DCollection(seg, colors=lc, linewidths=1.6))
    ax.set_xlim(xr); ax.set_ylim(yr); ax.set_zlim(zr)
    ax.set_xlabel("x (chord, flow→)"); ax.set_ylabel("y (span)"); ax.set_zlabel("z")
    ax.set_title(f"surface Cp + LEV(green/magenta) + TEV(faint)   nc={NC} sqrtx  t={f['t']:.3f}s\n"
                 f"8 m/s, ±45° flap, {TW:.0f}° twist, 2 Hz", fontsize=10)
    ax.view_init(elev=20, azim=-60)
    tt = np.arange(len(LhB)) * dt; cur = f["t"]; m = tt <= cur
    axh.plot(tt, LhB, "b-", lw=0.7, alpha=0.3); axh.plot(tt, Tnet, "g-", lw=0.7, alpha=0.3)
    axh.plot(tt[m], LhB[m], "b-", lw=1.7, label=f"lift ⟨{res['L_wind']:.1f}N⟩")
    axh.plot(tt[m], Tnet[m], "g-", lw=1.7, label=f"net thrust ⟨{res['T_wind']:.1f}N⟩")
    axh.axvline(cur, color="r", ls="--", lw=0.8); axh.axhline(0, color="gray", lw=0.5)
    axh.axhline(7.67, color="b", ls=":", lw=0.8, alpha=0.5)
    axh.set_xlim(0, tt[-1]); axh.set_ylim(-20, 30)
    axh.set_xlabel("time (s)"); axh.set_ylabel("force (N)"); axh.legend(fontsize=9, loc="upper right")
    axh.set_title(f"forces — model L={res['L_wind']:.1f}N T={res['T_wind']:.1f}N", fontsize=10); axh.grid(alpha=0.3)
    return []


anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", f"_v2_press_nc{NC}_tw{int(TW)}.gif")
anim.save(out, writer="pillow", fps=12, dpi=80)
print(f"saved {out}", flush=True); print("DONE", flush=True)
