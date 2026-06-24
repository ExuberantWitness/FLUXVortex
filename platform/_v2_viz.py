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

FREQ = 2.3; U = 8.0; SPC = 100
frames = []
res = R.gpu_run_twist(nc=4, ns=12, U=U, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                      freq=FREQ, n_cycle=3, steps_per_cycle=SPC, wake_rows=60, swept_axis=True,
                      real_geom=True, real_lev=True, lesp_crit_deg=11, lev_klev=1.0,
                      visc=True, les_suction=True, les_eta=1.0,
                      frames_out=frames, frame_skip=2)
print(f"recorded {len(frames)} frames; L_bern={res['L_bern']:.2f}N T_net={res['T_net']:.2f}N "
      f"T_lesp={res['T_lesp']:.2f}N D_visc={res['D_visc']:.2f}N", flush=True)

# per-step force histories (both wings, x2). Lift = unsteady-Bernoulli (captures the LEV, the 85% result).
# Thrust = LE-suction (Garrick/DeLaurier) - induced(Bernoulli x) - friction.  All x2 for two wings.
def smooth(a, w=7):   # short moving average to suppress per-step dGamma/dt shedding spikes (display only)
    k = np.ones(w) / w; return np.convolve(np.pad(a, w // 2, mode="edge"), k, mode="valid")[:len(a)]
LhB = smooth(2 * res["Lh_bern"])                          # lift (Bernoulli, both wings)
Tnet = smooth(-2 * (res["Xh_bern"] + res["Xh_vis"] + res["Xh_les"]))   # net thrust history
Tles = smooth(-2 * res["Xh_les"])                         # LE-suction thrust component
dt = (1.0 / FREQ) / SPC
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
    # lift/thrust history (correct Bernoulli lift + decomposed thrust)
    tt = np.arange(len(LhB)) * dt; cur = f["t"]
    axh.plot(tt, LhB, "b-", lw=0.8, alpha=0.35); axh.plot(tt, Tnet, "g-", lw=0.8, alpha=0.35)
    axh.plot(tt, Tles, color="#e08000", lw=0.8, alpha=0.35)
    m = tt <= cur
    axh.plot(tt[m], LhB[m], "b-", lw=1.7, label=f"lift (Bernoulli, ⟨L⟩={res['L_bern']:.1f}N)")
    axh.plot(tt[m], Tnet[m], "g-", lw=1.7, label=f"net thrust (⟨T⟩={res['T_net']:.1f}N)")
    axh.plot(tt[m], Tles[m], color="#e08000", lw=1.4, label=f"LE-suction (⟨{res['T_lesp']:.1f}N⟩)")
    axh.axvline(cur, color="r", ls="--", lw=0.8); axh.axhline(0, color="gray", lw=0.5)
    axh.axhline(7.79, color="b", ls=":", lw=0.9, alpha=0.6)   # measured lift @2.3Hz
    axh.set_xlim(dt * SPC * 0.4, tt[-1])                       # skip the startup transient
    axh.set_ylim(-12, 22)                                      # focus on the steady oscillation
    axh.set_xlabel("time (s)"); axh.set_ylabel("force (N)"); axh.legend(fontsize=8.5, loc="upper right")
    axh.set_title(f"forces  ({U:.0f} m/s, {FREQ} Hz, ±45° flap, no twist)\n"
                  f"measured lift 7.79N (blue dotted) — model {res['L_bern']:.1f}N = {100*res['L_bern']/7.79:.0f}%",
                  fontsize=9.5)
    axh.grid(alpha=0.3)
    return []

anim = FuncAnimation(fig, update, frames=len(frames), interval=90, blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v2_real_lev.gif")
anim.save(out, writer="pillow", fps=11, dpi=80)
print(f"saved {out}", flush=True)
