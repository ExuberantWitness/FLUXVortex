"""HIRATO Fig.11 VALIDATION: rectangular AR=6 wing, uniform PITCH RAMP 0->45deg at K=ȧc/2U=0.3, LESP_crit=0.27
(Hirato et al. 2019 case 1). Renders the convecting-wake LEV (green/magenta ±Γ) + TEV to check it sheds from the
LE and ROLLS UP into a coherent spanwise spiral over the suction surface (vs the J≈1 flapping where it spreads)."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v2_robo as R

U, CH, HS = 8.0, 0.287, 0.861          # AR = 2*HS/CH = 6.0 ; freestream 8 m/s
NC, NS, SPC, FREQ = 10, 20, 160, 8.0   # freq only sets dt: U*dt=U/(freq*spc)=0.006m ≈ 2%c LEV ring spacing
TMAX = 7.0                              # t* to render up to (ramp done ~2.3, rollup 2-6)
frames = []
res = R.gpu_run_twist(U=U, aoa_deg=0.0, flap_amp_deg=0.0, twist_amp_deg=0.0, freq=FREQ, nc=NC, ns=NS,
                      chord=CH, half_span=HS, n_cycle=2, steps_per_cycle=SPC, wake_rows=250,
                      real_geom=False, sym=True, les_suction=True, les_eta=1.0, a0_crit=0.27,
                      lev_shed_mode='kelvin', lev_sheet=True, lev_place='wake', lev_sign=1.0,
                      lev_consistent=False, a0_mode='sqrtx', lev_overlap=0.12, lev_roll_core=0.002,
                      pitch_ramp=True, pitch_max=-45.0, pitch_K=0.3, pitch_t0star=1.0,
                      frames_out=frames, frame_skip=2)
frames = [f for f in frames if f['t'] * U / CH <= TMAX]   # trim to the rollup window
tstar = np.array([f['t'] * U / CH for f in frames])
print(f"recorded {len(frames)} frames, t*=[{tstar.min():.1f},{tstar.max():.1f}]  L_wind={res['L_wind']:.2f}N", flush=True)

fig = plt.figure(figsize=(9, 8), facecolor="white")
ax = fig.add_subplot(111, projection="3d")
allw = np.concatenate([f["bound"].reshape(-1, 3) for f in frames] +
                      [f["wr"].reshape(-1, 3) for f in frames if len(f["wr"])])
xr = [allw[:, 0].min() - 0.05, allw[:, 0].max() + 0.05]; yr = [-0.05, HS + 0.05]
zr = [allw[:, 2].min() - 0.05, allw[:, 2].max() + 0.05]


def update(fi):
    ax.cla()
    f = frames[fi]
    ax.add_collection3d(Poly3DCollection([q for q in f["bound"]], facecolors="#c8d8e8", edgecolor="#333",
                                         linewidth=0.2, alpha=0.9))
    if len(f["wr"]):
        wt = f.get("wtype", np.zeros(len(f["wr"]), int))
        segt, lct, segl, lcl = [], [], [], []
        for ring, g, ty in zip(f["wr"], f["wg"], wt):
            for a_, b_ in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                if ty == 1:
                    segl.append([ring[a_], ring[b_]]); lcl.append((0, .7, 0, .95) if g > 0 else (.85, 0, .85, .95))
                else:
                    segt.append([ring[a_], ring[b_]]); lct.append((.4, .4, .8, .25))
        if segt: ax.add_collection3d(Line3DCollection(segt, colors=lct, linewidths=0.4))
        if segl: ax.add_collection3d(Line3DCollection(segl, colors=lcl, linewidths=1.4))
    ax.set_xlim(xr); ax.set_ylim(yr); ax.set_zlim(zr)
    ax.set_xlabel("x (chord, flow→)"); ax.set_ylabel("y (span)"); ax.set_zlabel("z")
    ax.set_title(f"HIRATO pitch-ramp (AR=6, K=0.3, LESP_crit=0.27)  t*={f['t']*U/CH:.2f}\n"
                 f"LEV(green/magenta) sheds from LE, should roll up over suction surface (cf. Fig.11)", fontsize=10)
    ax.view_init(elev=10, azim=-88)                # near side view (x-z) to see the chord-normal roll-up spiral
    return []


anim = FuncAnimation(fig, update, frames=len(frames), interval=90, blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "_v2_hirato_rollup.gif")
anim.save(out, writer="pillow", fps=12, dpi=85)
print(f"saved {out}", flush=True); print("DONE", flush=True)
