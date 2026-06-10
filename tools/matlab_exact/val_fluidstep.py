"""Closed-book fluid solve at boundary 171 vs fixture step3 (every quantity)."""
import os, sys
os.chdir('/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from ml_fluid_step import MatlabFluidStep

F3 = 'FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step3_t0.3000.mat'
f3s = loadmat(F3, squeeze_me=True, struct_as_record=False)   # for var_param etc.
f3r = loadmat(F3, squeeze_me=False)                          # for assembly module
g = lambda f, k: (f[k].toarray() if sp.issparse(f[k]) else np.asarray(f[k], dtype=float))

# build solver constants (mix: MatlabFluidStep wants squeeze_me version for var_param,
# raw version for the assembly module)
class Mixed(dict):
    def __init__(self, fs, fr):
        self.fs, self.fr = fs, fr
    def __getitem__(self, k):
        return self.fr[k]
    def __contains__(self, k):
        return k in self.fr
mx = Mixed(f3s, f3r)
ms = MatlabFluidStep.__new__(MatlabFluidStep)
# manual init mixing the two load styles
sq = lambda k: np.asarray(f3s[k]).squeeze()
ms.Nx = int(sq('Nx')); ms.Ny = int(sq('Ny')); ms.Ne = ms.Nx*ms.Ny
ms.Nq = int(sq('N_q_all'))
vp = f3s['var_param']
ms.Length = float(np.asarray(vp.Length).squeeze())
ms.r_eps_fine = float(np.asarray(vp.r_eps.fine).squeeze())
ms.r_eps_rough = float(np.asarray(vp.r_eps.rough).squeeze())
ms.Ncore = int(np.asarray(vp.Ncore).squeeze())
ms.eps_v = float(np.asarray(vp.eps_v).squeeze())
ms.d_t_wake = float(sq('d_t_wake'))
ms.U_in = float(sq('U_in'))
ms.V_in = g(f3s, 'V_in')
ms.Rtrunc = 5.5*ms.Length; ms.Rnochange = ms.Rtrunc - 1.5*ms.Length
ms.Sc_col = f3s['Sc_mat_col_global']; ms.S31 = f3s['Sc_mat_31']; ms.S24 = f3s['Sc_mat_24']
ms.Sp = [f3s[f'Sc_mat_panel_global_{k}'] for k in (1, 2, 3, 4)]
from ml_fluidforce import MatlabFluidForce
ms.asm = MatlabFluidForce(f3r)
ms.idof = ms.asm.idof
ms.Sc_col_d = g(f3s, 'Sc_mat_col_global')

# ---- inputs at boundary 171 ----
hX3 = np.asarray(f3s['h_X_vec'])
X = hX3[:, 170].copy()                    # X_vec read by solve_fluid (= fixture X_vec)
print('X check:', np.abs(X - np.asarray(f3s['X_vec']).ravel()).max())
iw = int(sq('i_wake_time'))               # i_wake_time AFTER increment? dump during solve at 171
hrw = f3s['h_r_wake']; hgw = f3s['h_Gamma_wake']
print('i_wake_time =', iw, ' len h_r_wake =', len(hrw))
# previous wake = h_r_wake{iw-1}... cell indexing: stored at CURRENT step index
# fixture r_wake_* = post-advect of THIS step; previous = h_r_wake{iw-1}
prev = np.asarray(hrw[iw-2])              # 1-based {iw-1} -> 0-based [iw-2]
Nw_prev = prev.shape[0] // 4
wake_in = dict(r1=prev[:Nw_prev], r2=prev[Nw_prev:2*Nw_prev],
               r3=prev[2*Nw_prev:3*Nw_prev], r4=prev[3*Nw_prev:])
# input Gamma_wake = current fixture Gamma_wake rows Ny: (pre-prepend state w/ trail update)
Gw_fix = g(f3s, 'Gamma_wake').ravel()
wake_in['Gam'] = Gw_fix[ms.Ny:].copy()
old_Gamma = g(f3s, 'old_Gamma').ravel()   # at dump, old_Gamma = Gamma_171? check below

# old_Gamma at generate_wake time = Gamma from solve 137. After this solve's
# calc_fluid_force line 183 old_Gamma=Gamma_171. The dump is AFTER that line, so
# fixture old_Gamma = Gamma_171?? compare with fixture Gamma:
print('|old_Gamma - Gamma| =', np.abs(old_Gamma - g(f3s,'Gamma').ravel()).max(),
      ' (0 => old_Gamma already overwritten; need Gamma_137 elsewhere)')
# Gamma_137: available as h_Gamma{iw-1}? check key
if 'h_Gamma' in f3r:
    hG = f3s['h_Gamma']
    Gamma_137 = np.asarray(hG[iw-2]).ravel()
    print('using h_Gamma{iw-1} for Gamma_137')
else:
    # reconstruct: trail rows of fixture Gamma_wake[:Ny] = Gamma_trail = Gamma_137 TE.
    # Full Gamma_137 not needed except TE rows for trail + bound-source Gamma for
    # wake advection. For advection, MATLAB uses workspace Gamma = Gamma_137 (full).
    Gamma_137 = None
    print('h_Gamma missing — need Gamma_137 from fixture step2/137 chain')

# fallback: solve once at boundary-137 state to get Gamma_137 (closed-book chain)
if Gamma_137 is None:
    # boundary 137 state = hX3[:,136] (corrected start of predictor)? NO — fluid at 137
    # was solved at the PREDICTOR pass-1 state of block 103->137, which is in f2's hX.
    f2s = loadmat('FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step2_t0.2000.mat',
                  squeeze_me=True, struct_as_record=False)
    # f2 dumped at boundary 103; its h_X_vec in-block cols are pass-1 of block 69..103.
    # The 137-solve input state is NOT directly stored. Instead reconstruct Gamma_137
    # from fixture step3's own wake bookkeeping: Gamma_wake[:Ny] (trail) = Gamma_137 TE.
    # For ADVECTION the bound source is full Gamma_137 — approximate test: advection
    # bound influence uses Gamma_137; we lack it. USE trail-extended approximation NO —
    # instead: verify the rest of the chain with fixture's own post-advect wake state
    # (skip generate_wake) to isolate solve+forces first.
    Gamma_137 = None

# ---- TEST A: skip generate_wake (use fixture post-advect wake) -> isolate solve+forces ----
wk_fix = dict(r1=g(f3s,'r_wake_1'), r2=g(f3s,'r_wake_2'), r3=g(f3s,'r_wake_3'),
              r4=g(f3s,'r_wake_4'), Gam=Gw_fix.copy(),
              dt1=g(f3s,'dt_r_wake_1') if 'dt_r_wake_1' in f3r else None,
              dt2=g(f3s,'dt_r_wake_2'), dt3=g(f3s,'dt_r_wake_3'),
              dt4=g(f3s,'dt_r_wake_4'))
if wk_fix['dt1'] is None:
    wk_fix['dt1'] = g(f3s,'dt_r_wake_1')

Ny = ms.Ny; Nq = ms.Nq
q = X[:Nq]; dtq = X[Nq:]
bP = ms.panels(q); dt_bP = [np.asarray(S@dtq).reshape(-1,3) for S in ms.Sp]
rc = ms.colloc(q); dt_rc = ms.colloc(dtq)
nv, dtn = ms.normals(q, dtq)
rel = lambda a, b: np.linalg.norm(np.asarray(a)-np.asarray(b))/(np.linalg.norm(np.asarray(b))+1e-30)
print('--- geometry ---')
print('rc rel:', rel(rc, g(f3s,'rc_vec')), ' n rel:', rel(nv, g(f3s,'n_vec_i')),
      ' dt_n rel:', rel(dtn, g(f3s,'dt_n_vec_i')))
for k in range(4):
    print(f'r_panel_{k+1} rel:', rel(bP[k], g(f3s,f'r_panel_vec_{k+1}')))
from ml_uvlm import aic_from_q1234
Vq = ms.q1234(rc, bP, fine=True)
A = aic_from_q1234(Vq, nv)
print('A rel:', rel(A, g(f3s,'A_mat')))
wP=[wk_fix['r1'],wk_fix['r2'],wk_fix['r3'],wk_fix['r4']]
# RHS uses Gamma_wake with OLD trail (pre line-157)... fixture Gamma_wake has NEW trail.
# Gamma_wake at RHS time: rows[:Ny] = Gamma_trail (prepended in generate_wake) = old TE
# of Gamma_137 — which IS what line 137 prepended: Gamma_trail = old_Gamma(TE) where
# old_Gamma=Gamma_137. And line 157 sets the SAME value again!! (Gamma_trail
# unchanged between). So fixture Gamma_wake == RHS Gamma_wake. Good.
Vwp_force, q_wake = ms.vwake(rc, wP, wk_fix['Gam'], fine=True)
print('V_wake_plate (force, new trail) rel:', rel(Vwp_force, g(f3s,'V_wake_plate')))
# RHS wake circulation: trail = Gamma_103 TE (two solves ago), per delayed Kutta
hG = f3s['h_Gamma']
Gamma_103 = np.asarray(hG[iw-3]).ravel()   # h_Gamma{iw-2} 1-based
Gw_rhs = wk_fix['Gam'].copy()
Gw_rhs[:Ny] = Gamma_103[-Ny:]
Vwp_rhs = np.einsum('tsc,s->tc', q_wake, Gw_rhs)
Vn = np.einsum('tc,tc->t', dt_rc - ms.V_in - Vwp_rhs, nv)
print('V_normal (delayed-Kutta trail) rel:', rel(Vn, g(f3s,'V_normal')))
Gam = np.linalg.solve(A, Vn)
print('Gamma rel:', rel(Gam, g(f3s,'Gamma')))
# downstream force quantities with corrected Gamma
Vg = np.einsum('tsc,s->tc', Vq, Gam)
V_surf1 = Vg + Vwp_force + ms.V_in
t21=bP[1]-bP[0]; t34=bP[2]-bP[3]; t14=bP[0]-bP[3]; t23=bP[1]-bP[2]
tx=(t21+t34)/2; ty=(t14+t23)/2
dxn=np.linalg.norm(tx,axis=1,keepdims=True); dyn=np.linalg.norm(ty,axis=1,keepdims=True)
tx/=dxn; ty/=dyn
Gm=Gam.reshape(ms.Nx,Ny); dxm=dxn.reshape(ms.Nx,Ny); dym=dyn.reshape(ms.Nx,Ny)
dxG=np.vstack([Gm[:1],np.diff(Gm,axis=0)])/dxm
Gm2=np.hstack([np.zeros((ms.Nx,1)),Gm,np.zeros((ms.Nx,1))])
dyG=(Gm2[:,2:]-Gm2[:,:-2])/(2*dym); dyG[:,0]=Gm[:,0]/dym[:,0]; dyG[:,-1]=-Gm[:,-1]/dym[:,-1]
txdx=tx*dxG.reshape(-1,1); tydy=ty*dyG.reshape(-1,1)
dp1=np.einsum('tc,tc->t',V_surf1,txdx+tydy)
dp2=-(txdx+tydy)
print('dp_lift1 rel:', rel(dp1, g(f3s,'dp_lift1').ravel()))
dp2_ml=np.squeeze(g(f3s,'dp_lift2')); dp2_ml=dp2_ml.T if dp2_ml.shape[0]==3 else dp2_ml
print('dp_lift2 rel:', rel(dp2, dp2_ml))
# Mf2_vec1 with fixture wake dt
from ml_fluid_step import dt_q1234_mat
dtwP=[wk_fix['dt1'],wk_fix['dt2'],wk_fix['dt3'],wk_fix['dt4']]
dtq_w=dt_q1234_mat(rc,wP,dt_rc,dtwP)
Gw_dt_n=np.einsum('tc,tc->t',np.einsum('tsc,s->tc',dtq_w,wk_fix['Gam']),nv)
mv1=np.linalg.solve(A,-Gw_dt_n)
print('Mf2_vec1 rel:', rel(mv1, g(f3s,'Mf2_vec1').ravel()))
# dt_Amat1, dA2G
dt_bP2=[np.asarray(S@dtq).reshape(-1,3) for S in ms.Sp]
dtq_b=dt_q1234_mat(rc,bP,dt_rc,dt_bP2)
dA1=np.einsum('tsc,tc->ts',dtq_b,nv)
print('dt_Amat1 rel:', rel(dA1, g(f3s,'dt_Amat1')))
print('dt_Amat2_Gamma rel:', rel(Vg, g(f3s,'dt_Amat2_Gamma')))
# Mf1, Mf2, assembly
Ne=ms.Ne
nvec_Sc=np.zeros((Ne,ms.Nq))
for e in range(Ne):
    rows=ms.Sc_col_d[3*e:3*e+3][:,ms.idof[e]]
    nvec_Sc[e,ms.idof[e]]=nv[e]@rows
Mf1=np.linalg.solve(A,nvec_Sc); Mf2=np.linalg.inv(A)
print('Mf1_mat rel:', rel(Mf1, g(f3s,'Mf1_mat')))
print('Mf2_mat rel:', rel(Mf2, g(f3s,'Mf2_mat')))
Qv,M0,L2,Mm=ms.asm.assemble(dp1,mv1,dp2,Mf2,Mf1,nv)
print('CLOSED-BOOK Qf_p_global  rel:', rel(Qv, g(f3s,'Qf_p_global').ravel()))
print('CLOSED-BOOK mat0_global  rel:', rel(M0, g(f3s,'Qf_p_mat0_global')))
print('CLOSED-BOOK lift2_global rel:', rel(L2, g(f3s,'Qf_p_lift2_mat_global')))
print('CLOSED-BOOK mat_global   rel:', rel(Mm, g(f3s,'Qf_p_mat_global')))

# ---- TEST B: generate_wake (RK4) from previous wake state vs fixture ----
print('--- TEST B: RK4 wake advection ---')
Gamma_137 = np.asarray(hG[iw-2]).ravel()      # bound source at advection
Gamma_103_TE = np.asarray(hG[iw-3]).ravel()[-Ny:]   # prepended trail (two solves ago)
wk_new = ms.generate_wake(False, bP, dt_bP, Gamma_137, wake_in, Gamma_103_TE)
for k,fk in [('r1','r_wake_1'),('r2','r_wake_2'),('r3','r_wake_3'),('r4','r_wake_4'),
             ('dt2','dt_r_wake_2'),('dt3','dt_r_wake_3')]:
    print(f'{fk:12s} rel:', rel(wk_new[k], g(f3s,fk)))
# circulation: fixture trail post-157 = Γ_137 TE; ours pre-157 = Γ_103 TE; compare rest
print('Gam[Ny:]    rel:', rel(wk_new['Gam'][Ny:], Gw_fix[Ny:]))
print('Gam[:Ny] (pre-157, vs Γ_103 TE):', rel(wk_new['Gam'][:Ny], Gamma_103_TE))
