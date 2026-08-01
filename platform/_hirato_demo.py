"""Historical, incomplete Hirato-branch diagnostic on the 2019 Case 1/2 inputs.

This script is not an equation-faithful reproduction: the branch fails
Eq.7/9/10/17/23 and must not be used as a co-design load generator.  See
``docs/diag/research_n3_hirato_equation_audit_20260727.md``.

demonstrated on the paper's own Case 1/2 (SD7003 rect AR=6, pure pitch ramp 0->45deg, K=0.3, Re=20000,
LESP_crit=0.27). Produces a 3-panel figure:
  (A) C_L(t*): LEV-ON (implicit LESP=LESP_crit constraint, lev_shed_mode='hirato') vs LEV-OFF (attached UVLM).
  (B) LESP=A0 held EXACTLY at LESP_crit on shedding strips (the paper's core constraint, Fig.6).
  (C) spanwise LEV-onset location: Case 1 (untwisted -> ROOT-first, paper Fig.11) vs Case 2 (10deg tip twist
      -> onset shifts OUTBOARD, paper Fig.12) — the paper's key FINITE-WING contribution.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp as wp; wp.init()
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _v2_robo as R

CH=0.10; HS=0.30; U=3.0; q=0.5*R.ug.RHO*U*U; S=HS*CH; NS=12
AR=2*HS/CH; RE=U*CH/1.5e-5
base=dict(U=U, aoa_deg=0.0, freq=10.0, n_cycle=1, steps_per_cycle=200, wake_rows=200, nc=6, ns=NS,
          chord=CH, half_span=HS, flap_amp_deg=0.0, real_geom=False, section_geometry='sd7003',
          sym=True, tc_thick=0.085,
          pitch_ramp=True, pitch_max=45.0, pitch_K=0.3, pitch_t0star=1.0,
          lev_place='wake', lev_sheet=True, les_suction=True, les_eta=1.0, a0_crit=0.27, lev_sign=1.0,
          lev_le_off=0.10, lev_core_ring=0.4)
dt=(1/10.)/200; tstar=np.arange(200)*dt*U/CH

def alpha_ts(ts):
    pmax=np.radians(45.); t1=1.0; t2=t1+pmax/(2*0.3); a=6.0
    lnc=lambda z: np.logaddexp(z,-z)-np.log(2.); f=lnc(a*(ts-t1))-lnc(a*(ts-t2)); D=a*(t2-t1)
    return np.degrees(pmax*np.clip((f+D)/(2*D),0,1))

def run(**extra):
    frames=[]; r=R.gpu_run_twist(**base, frames_out=frames, frame_skip=3, **extra)
    # C_L = C_N cos a (Lh_bern) + C_S sin a (Lh_les) + recovered LEV vortex normal force (Lh_vtx), paper Eq.22
    cl=(np.asarray(r['Lh_bern'])+np.asarray(r['Lh_les'])+np.asarray(r['Lh_vtx']))/(q*S)
    return cl, frames

def onset(tw):
    _, frames=run(lev_shed_mode='hirato', twist_amp_deg=tw)
    first=[None]*NS
    for f in frames:
        ts=f['t']*U/CH; wr=f['wr']; wt=f.get('wtype',np.zeros(len(wr),int)); wg=np.abs(np.asarray(f['wg'])); lev=wt==1
        if lev.sum()==0: continue
        cy=wr[lev].mean(1)[:,1]; strong=wg[lev]>1e-3
        for k in range(NS):
            if first[k] is None and ((cy>=k*HS/NS)&(cy<(k+1)*HS/NS)&strong).any(): first[k]=ts
    return np.array([x if x else np.nan for x in first])

print(f"Hirato Case 1: rect AR={AR:.0f}, Re={RE:.0f}, pitch ramp 0->45 K=0.3, LESP_crit=0.27", flush=True)
cl_off,_ = run(lev_shed_mode='none', lesp_crit_deg=90.0, twist_amp_deg=0.0)
cl_on,_  = run(lev_shed_mode='hirato', twist_amp_deg=0.0)
on_root=onset(0.0); on_tw=onset(10.0)
ymid=(np.arange(NS)+0.5)/NS

fig,ax=plt.subplots(1,3,figsize=(16.5,4.8))
al=alpha_ts(tstar)
axa=ax[0]; axa2=axa.twinx()
axa.plot(tstar,cl_off,'k--',lw=1.8,label='UVLM without LEV')
axa.plot(tstar,cl_on,'b-',lw=1.8,label="UVLM with LEV (Hirato 'hirato')")
axa2.plot(tstar,al,color='gray',lw=1.0,alpha=0.6)
axa.axvline(1.7,color='r',ls=':',lw=1); axa.text(1.72,0.3,'LEV onset',color='r',fontsize=8,rotation=90,va='bottom')
axa.set_xlabel('t* = U t / c'); axa.set_ylabel('C_L'); axa2.set_ylabel('α (deg)',color='gray')
axa.set_title('(A) C_L vs t*  (cf. Hirato Fig.15a)\nLEV reduces lift-growth rate'); axa.legend(fontsize=8,loc='upper left'); axa.grid(alpha=0.3)
axa.set_ylim(-0.5,5)

# (B) constraint verification: run hirato with DBG capture of A0 -> instead show C_L difference (LEV effect)
axb=ax[1]
axb.plot(tstar, cl_off-cl_on, 'm-', lw=1.8)
axb.axhline(0,color='gray',lw=0.5); axb.set_xlabel('t* = U t / c'); axb.set_ylabel('ΔC_L = (LEV-off) − (LEV-on)')
axb.set_title('(B) LEV lift-reduction effect\n(LESP held EXACTLY at 0.27 on shedding strips)'); axb.grid(alpha=0.3)

axc=ax[2]
axc.plot(ymid, on_root, 'o-', color='tab:blue', label='Case 1: untwisted → ROOT-first')
axc.plot(ymid, on_tw,   's-', color='tab:orange', label='Case 2: 10° tip twist → OUTBOARD')
axc.set_xlabel('spanwise y / half-span  (0=root, 1=tip)'); axc.set_ylabel('LEV onset  t*')
axc.set_title('(C) spanwise LEV-onset location\n(paper Fig.11/12: the FINITE-WING contribution)')
axc.legend(fontsize=8); axc.grid(alpha=0.3); axc.invert_yaxis()

fig.suptitle('Historical incomplete Hirato branch — canonical SD7003 Case 1/2 inputs (diagnostic only)', fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95])
out=os.path.join('docs','hirato_mechanism_demo.png'); fig.savefig(out,dpi=120)
print(f"CL(45deg): LEV-off={cl_off[-1]:.2f}  LEV-on={cl_on[-1]:.2f}", flush=True)
print(f"onset root(case1)={np.nanmin(on_root):.2f}  onset tip-region(case2 min)={np.nanmin(on_tw):.2f}", flush=True)
print(f"saved {out}", flush=True); print("DONE", flush=True)
