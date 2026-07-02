"""Definitive rollup check: plot the LEV ring centroids of a MID-SPAN slice in the x-z plane, colored by age
(shed order), at the final developed instant — for several convection cores. If the LEV curls up over the
suction surface (spiral), rollup is achieved (cf. Hirato Fig.11/14 x-z sheet profile); if it trails flat aft,
it is not. Overlays the wing chord line.  python _hirato_rollup_profile.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp as wp; wp.init()
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm
import _v2_robo as R

CH=0.10; HS=0.30; U=3.0; NS=16; SPC=240
CORES=[0.4, 0.2, 0.1, 0.05]
fig,axes=plt.subplots(1,len(CORES),figsize=(4.6*len(CORES),4.4),sharey=True)
for ax,core in zip(axes,CORES):
    frames=[]
    r=R.gpu_run_twist(U=U, aoa_deg=0.0, freq=10.0, n_cycle=1, steps_per_cycle=SPC, wake_rows=SPC, nc=6, ns=NS,
                      chord=CH, half_span=HS, flap_amp_deg=0.0, twist_amp_deg=0.0, real_geom=False, sym=True,
                      tc_thick=0.085, pitch_ramp=True, pitch_max=45.0, pitch_K=0.3, pitch_t0star=1.0,
                      lev_shed_mode='hirato', lev_place='wake', lev_sheet=True, les_suction=True, les_eta=1.0,
                      a0_crit=0.27, lev_sign=1.0, lev_le_off=0.10, lev_core_ring=core,
                      frames_out=frames, frame_skip=3)
    f=min(frames, key=lambda ff: abs(ff['t']*U/CH-2.8))          # near end of ramp / hold
    wr=f['wr']; wt=f.get('wtype',np.zeros(len(wr),int)); wg=np.abs(np.asarray(f['wg'])); born=None
    lev=(wt==1)&(wg>0.002)
    bnd=f['bound'].reshape(-1,3); lex=bnd[:,0].min(); c=bnd[:,0].max()-lex
    # mid-span slice: rings with centroid y in [0.4,0.6]*HS
    cen=wr.mean(1); ins=lev&(cen[:,1]>0.35*HS)&(cen[:,1]<0.65*HS)
    if ins.sum()>0:
        xz=cen[ins]; age=np.arange(ins.sum())                    # order ~ shed order (approx age)
        sc=ax.scatter((xz[:,0]-lex)/c, xz[:,2]/c, c=np.linspace(0,1,ins.sum()), cmap='viridis', s=18)
    # wing chord (rotated flat plate at ~45deg at t*=2.8): draw LE->TE of a mid-span bound strip
    js=NS//2; le=bnd[js]; te=bnd[(6-1)*NS+js] if 6>1 else bnd[js]
    ax.plot([(le[0]-lex)/c,(te[0]-lex)/c],[le[2]/c,te[2]/c],'k-',lw=2,label='wing chord')
    ax.axhline(0,color='0.7',lw=0.5); ax.set_xlim(-0.3,2.0); ax.set_ylim(-0.4,1.2)
    ax.set_xlabel('(x−LE)/c'); ax.set_title(f'core_ring={core}\nt*={f["t"]*U/CH:.2f}, mid-span slice'); ax.grid(alpha=0.3)
axes[0].set_ylabel('z / c  (lift-off above suction surface)')
fig.suptitle('LEV rollup PROFILE (mid-span x-z), color=shed order — does the sheet CURL UP (rollup) or trail flat?', fontsize=12)
fig.tight_layout(rect=[0,0,1,0.93])
out=os.path.join('docs','hirato_rollup_profile.png'); fig.savefig(out,dpi=115); print(f"saved {out}",flush=True); print("DONE",flush=True)
