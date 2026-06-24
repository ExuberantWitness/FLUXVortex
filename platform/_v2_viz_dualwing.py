"""Two-wing symmetric flapping (root symmetry plane, sym=True). The solver computes the RIGHT wing
(y>=0) with a y=0 image of the OTHER wing; here we DRAW both: the right wing + its mirror (left wing)
flapping symmetrically (both tips up/down together), each with its shed wake. Shows the corrected
wing-pair loading the symmetry-plane fix produces. Left: 3D both wings + wakes. Right: lift history."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v2_robo as R

FREQ = 2.0; U = 8.0; SPC = 90
frames = []
res = R.gpu_run_twist(nc=4, ns=12, U=U, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0,
                      freq=FREQ, n_cycle=3, steps_per_cycle=SPC, wake_rows=55, swept_axis=True,
                      real_geom=True, sym=True, frames_out=frames, frame_skip=2)
print(f"recorded {len(frames)} frames; sym L_bern={res['L_bern']:.2f}N (both wings)", flush=True)


def smooth(a, w=7):
    k = np.ones(w) / w; return np.convolve(np.pad(a, w // 2, mode="edge"), k, mode="valid")[:len(a)]
LhB = smooth(2 * res["Lh_bern"]); dt = (1.0 / FREQ) / SPC

fig = plt.figure(figsize=(15, 6.4), facecolor="white")
ax = fig.add_subplot(121, projection="3d"); axh = fig.add_subplot(122)
allw = np.concatenate([f["wr"].reshape(-1, 3) for f in frames if len(f["wr"])] + [frames[0]["bound"].reshape(-1, 3)])
xr = [allw[:, 0].min() - 0.05, allw[:, 0].max() + 0.05]
zr = [min(-0.65, allw[:, 2].min() - 0.05), max(0.65, allw[:, 2].max() + 0.05)]


def mirror(a):                                   # reflect across y=0 root symmetry plane
    b = a.copy(); b[..., 1] *= -1; return b


def update(fi):
    ax.cla(); axh.cla()
    f = frames[fi]
    bR = f["bound"]; bL = mirror(bR)             # right (solved) + left (mirror) wings
    ax.add_collection3d(Poly3DCollection([b for b in bR], alpha=0.9, facecolors="#4a90d9", edgecolor="#1a3a5a", linewidth=0.3))
    ax.add_collection3d(Poly3DCollection([b for b in bL], alpha=0.9, facecolors="#d98a4a", edgecolor="#5a3a1a", linewidth=0.3))
    if len(f["wr"]):                             # both wakes, colored by circulation sign
        seg, lc = [], []
        for side, rings in ((1, f["wr"]), (-1, mirror(f["wr"]))):
            for ring, g in zip(rings, f["wg"]):
                for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    seg.append([ring[a], ring[b]])
                    lc.append((0.85, 0.3, 0.3, 0.30) if g > 0 else (0.3, 0.3, 0.85, 0.30))
        ax.add_collection3d(Line3DCollection(seg, colors=lc, linewidths=0.4))
    ax.set_xlim(xr); ax.set_ylim(-0.85, 0.85); ax.set_zlim(zr)
    ax.set_xlabel("x (chord, flow →)"); ax.set_ylabel("y (span, both wings)"); ax.set_zlabel("z (flap)")
    ax.set_title(f"RoboEagle TWO-WING symmetric flapping (root symmetry plane)  t={f['t']:.3f}s\n"
                 f"blue=solved wing / orange=mirror wing — both flap ±45° together", fontsize=10)
    ax.view_init(elev=16, azim=-72)
    tt = np.arange(len(LhB)) * dt; cur = f["t"]
    axh.plot(tt, LhB, "b-", lw=0.8, alpha=0.35)
    m = tt <= cur
    axh.plot(tt[m], LhB[m], "b-", lw=1.7, label=f"lift, both wings (⟨L⟩={res['L_bern']:.1f}N)")
    axh.axhline(7.45, color="g", ls=":", lw=1.0, alpha=0.7, label="measured 7.45N")
    axh.axvline(cur, color="r", ls="--", lw=0.8); axh.axhline(0, color="gray", lw=0.5)
    axh.set_xlim(dt * SPC * 0.4, tt[-1]); axh.set_ylim(-12, 24)
    axh.set_xlabel("time (s)"); axh.set_ylabel("lift (N)"); axh.legend(fontsize=9, loc="upper right")
    axh.set_title(f"lift ({U:.0f} m/s, {FREQ} Hz, ±45° flap, no twist, sym plane ON)", fontsize=10)
    axh.grid(alpha=0.3)
    return []


anim = FuncAnimation(fig, update, frames=len(frames), interval=90, blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v2_dualwing.gif")
anim.save(out, writer="pillow", fps=12, dpi=80)
print(f"saved {out}", flush=True)
