"""2D LDVM (flap_ldvm) LEV visualization — the research-grade sectional model that sheds DISCRETE
leading-edge (LEV) + trailing-edge (TEV) vortex particles (the one gold-validated for LE suction).
Shows airfoil + LE particles (red=+, blue=-) + TE wake, like figures/hybrid_k05_free.gif. Saves GIF."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flap_ldvm import FlapLDVM

# pitching+plunging section that sheds a strong dynamic-stall LEV (classic LDVM case)
U = 1.0; c = 1.0; k = 0.25; Om = 2 * U / c * k; dt = 0.025; amax = np.radians(35.0); h0 = 0.0
m = FlapLDVM(U=U, c=c, n=70, dt=dt, rho=1.0, lesp_crit=0.11, lev_shed=True, max_wake=900)
frames = []
nsteps = 360
for it in range(nsteps):
    t = it * dt
    a = amax * np.sin(Om * t); da = amax * Om * np.cos(Om * t)
    r = m.step(a, da, 0.0)
    if it % 2 == 0:
        sx, sy = m.sx, m.sy; ca, sa = np.cos(a), np.sin(a)
        le = (sx, sy); te = (sx + c * ca, sy - c * sa)
        frames.append(dict(t=t, le=le, te=te,
                           lx=np.array(m.lx), ly=np.array(m.ly), lg=np.array(m.lg),
                           tx=np.array(m.tx), ty=np.array(m.ty), tg=np.array(m.tg),
                           a=np.degrees(a), CL=r["CL"], lesp=r["lesp"], nlev=r["n_lev"]))
print(f"recorded {len(frames)} frames, final LEVs={len(m.lx)}, TEVs={len(m.tx)}", flush=True)

fig, ax = plt.subplots(figsize=(11, 6.2)); fig.set_facecolor("white")
xs = np.concatenate([f["tx"] for f in frames if len(f["tx"])] + [np.array([0.0])])
xr = [xs.min() - 0.5, 1.5]; zr = [-2.2, 2.2]

def update(fi):
    ax.cla()
    f = frames[fi]
    # TE wake particles (the trailing vortex street), colored by sign
    if len(f["tx"]):
        sz = 6 + 60 * np.abs(f["tg"]) / (np.abs(f["tg"]).max() + 1e-9)
        ax.scatter(f["tx"], f["ty"], c=np.sign(f["tg"]), cmap="bwr", vmin=-1, vmax=1,
                   s=sz, alpha=0.55, edgecolors="none")
    # LE vortex particles (the LEV!), larger + outlined
    if len(f["lx"]):
        sz = 12 + 120 * np.abs(f["lg"]) / (np.abs(f["lg"]).max() + 1e-9)
        ax.scatter(f["lx"], f["ly"], c=np.sign(f["lg"]), cmap="bwr", vmin=-1, vmax=1,
                   s=sz, alpha=0.9, edgecolors="k", linewidths=0.4, zorder=5)
    # airfoil
    ax.plot([f["le"][0], f["te"][0]], [f["le"][1], f["te"][1]], "k-", lw=3, zorder=6)
    ax.plot(f["le"][0], f["le"][1], "go", ms=8, zorder=7)   # LE marker (green)
    ax.set_xlim(xr); ax.set_ylim(zr); ax.set_aspect("equal")
    ax.set_xlabel("x (chordwise, flow →)"); ax.set_ylabel("z")
    ax.set_title(f"2D LDVM dynamic-stall LEV  (pitch ±35°, k=0.25)\n"
                 f"t={f['t']:.2f}s  α={f['a']:+.0f}°  LESP={f['lesp']:+.2f}  "
                 f"LEVs={f['nlev']}  |  green=LE,  big outlined=LEV,  small=TE wake (red/blue=±Γ)",
                 fontsize=10)
    ax.grid(alpha=0.25)
    return []

anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v2_lev_working.gif")
anim.save(out, writer="pillow", fps=12, dpi=85)
print(f"saved {out}", flush=True)
