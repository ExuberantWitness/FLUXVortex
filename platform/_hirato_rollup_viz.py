"""Visualize the LEV vortex-SHEET ROLLUP for the Hirato Case-1 pitch-ramp condition, colored by vortex-ring
strength Gamma (cf. paper Fig.10 colorbar / Fig.11 LEV-sheet surface). Produces a montage of time instants
(like Fig.11a/b/c) + an animated gif, viewed to show the LEV sheet lifting off the suction surface and rolling
up into a spanwise-coherent spiral near the leading edge.

  python _hirato_rollup_viz.py [ns] [spc] [core_ring]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp as wp; wp.init()
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
import _v2_robo as R

CH=0.10; HS=0.30; U=3.0
NS = int(sys.argv[1]) if len(sys.argv)>1 else 16
SPC = int(sys.argv[2]) if len(sys.argv)>2 else 200
CORE = float(sys.argv[3]) if len(sys.argv)>3 else 0.25   # smaller -> tighter rollup (stronger self-induction)
frames=[]
res=R.gpu_run_twist(U=U, aoa_deg=0.0, freq=10.0, n_cycle=1, steps_per_cycle=SPC, wake_rows=SPC, nc=6, ns=NS,
                    chord=CH, half_span=HS, flap_amp_deg=0.0, twist_amp_deg=0.0, real_geom=False, sym=True,
                    tc_thick=0.085, pitch_ramp=True, pitch_max=45.0, pitch_K=0.3, pitch_t0star=1.0,
                    lev_shed_mode='hirato', lev_place='wake', lev_sheet=True, les_suction=True, les_eta=1.0,
                    a0_crit=0.27, lev_sign=1.0, lev_le_off=0.10, lev_core_ring=CORE,
                    frames_out=frames, frame_skip=2)
print(f"recorded {len(frames)} frames; L_wind={res['L_wind']:.2f}N", flush=True)

# color range from LEV ring strengths (robust)
allg=np.concatenate([np.abs(f['wg'][f.get('wtype',np.zeros(len(f['wg']),int))==1]) for f in frames if len(f['wg'])] + [np.array([1e-6])])
gmax=float(np.percentile(allg[allg>1e-6], 95)) if (allg>1e-6).any() else 1.0
norm=colors.Normalize(vmin=0, vmax=gmax); cmap=cm.get_cmap('jet')   # paper Fig.10 uses a jet-like Gamma colorbar

def tstar(f): return f['t']*U/CH

def draw(ax, f, side=False):
    ax.cla()
    wr=f['wr']; wt=f.get('wtype',np.zeros(len(wr),int)); wg=np.asarray(f['wg'])
    ax.add_collection3d(Poly3DCollection([q for q in f['bound']], facecolors='0.8', edgecolor='0.5', linewidths=0.2, alpha=0.9))
    lev=wt==1
    if lev.sum()>0:
        levr=wr[lev]; levg=np.abs(wg[lev]); strong=levg>0.002
        if strong.sum()>0:
            polys=[q for q in levr[strong]]; fc=cmap(norm(levg[strong]))
            ax.add_collection3d(Poly3DCollection(polys, facecolors=fc, edgecolor='k', linewidths=0.12, alpha=0.82))
    tev=wt==0
    if tev.sum()>0:
        seg=[]
        for ring in wr[tev][::4]:
            for a,b in [(0,1),(1,2),(2,3),(3,0)]: seg.append([ring[a],ring[b]])
        ax.add_collection3d(Line3DCollection(seg, colors=(0.3,0.3,0.3,0.12), linewidths=0.3))
    ax.set_xlim(-0.04,0.24); ax.set_ylim(-0.02,0.32); ax.set_zlim(-0.05,0.14)
    ax.set_xlabel('x (chord, flow→)'); ax.set_zlabel('z')
    if side:                                   # look ALONG the span (x-z profile) -> reveals the rollup curl
        ax.view_init(elev=2, azim=-89); ax.set_ylabel('')
        try: ax.set_box_aspect((0.28,0.34,0.19))
        except Exception: pass
    else:
        ax.set_ylabel('y (span)'); ax.view_init(elev=20, azim=-70)
        try: ax.set_box_aspect((0.28,0.34,0.19))
        except Exception: pass

# montage: TOP row = perspective, BOTTOM row = side (x-z) profile revealing the rollup curl
picks=[1.9,2.2,2.5,2.8]
fig=plt.figure(figsize=(18,8))
for k,ts in enumerate(picks):
    f=min(frames, key=lambda ff: abs(tstar(ff)-ts))
    ax=fig.add_subplot(2,4,k+1,projection='3d'); draw(ax,f,side=False)
    ax.set_title(f"perspective   t*={tstar(f):.2f}", fontsize=10)
    ax2=fig.add_subplot(2,4,k+5,projection='3d'); draw(ax2,f,side=True)
    ax2.set_title(f"side (x-z)   t*={tstar(f):.2f}", fontsize=10)
sm=cm.ScalarMappable(norm=norm,cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.02, label='LEV ring strength |Γ| (m²/s)')
fig.suptitle(f'LEV vortex-sheet ROLLUP over the suction surface — Hirato Case 1 (cf. Fig.11), core_ring={CORE}', fontsize=13)
out=os.path.join('docs','hirato_rollup_montage.png'); fig.savefig(out,dpi=110,bbox_inches='tight')
print(f"saved {out}", flush=True)

# animated gif
fig2=plt.figure(figsize=(7,6)); ax2=fig2.add_subplot(111,projection='3d')
sm2=cm.ScalarMappable(norm=norm,cmap=cmap); sm2.set_array([]); fig2.colorbar(sm2,ax=ax2,shrink=0.6,label='|Γ| (m²/s)')
def upd(i):
    draw(ax2, frames[i]); ax2.set_title(f"LEV rollup  t*={tstar(frames[i]):.2f}", fontsize=11); return []
anim=FuncAnimation(fig2, upd, frames=range(0,len(frames),1), interval=90, blit=False)
gif=os.path.join('docs','hirato_rollup.gif'); anim.save(gif, writer='pillow', fps=11, dpi=70)
print(f"saved {gif}", flush=True); print("DONE", flush=True)
