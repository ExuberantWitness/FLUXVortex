"""Block-replay harness: replicate MATLAB solve_structure.m corrector pass
step-by-step in MATLAB-native DOF order/units, starting from MATLAB state with
MATLAB fluid anchors, and compare each step against MATLAB h_X_vec ground truth.

Block: boundary i_time=137 (time_fluid=0.274) -> 171. Corrector steps 138..171.
  start X = f3.h_X_vec(:,138)   [stable corrected column]
  anchors: F_k = Qf_p_*_a == old_Qf_p_*, F_{k+1} = Qf_p_*   (from f3 dump)
  truth:   f4.h_X_vec(:,139..172)
"""
import os, sys
os.chdir('/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV')
sys.path.insert(0,'src'); sys.path.insert(0,'tests')
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.linalg import lu_factor, lu_solve
from run_standalone_yamano import yamano_params, build_yamano_shell

import sys as _sys
PRED = '--pred' in _sys.argv
F2='FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step2_t0.2000.mat'
F3='FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step3_t0.3000.mat'
F4='FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step4_t0.4000.mat'
f2=loadmat(F2,squeeze_me=True,struct_as_record=False)
f3=loadmat(F3,squeeze_me=True,struct_as_record=False)
f4=loadmat(F4,squeeze_me=True,struct_as_record=False)
g=lambda f,k:(f[k].toarray() if sp.issparse(f[k]) else np.asarray(f[k],dtype=float))

N=1584; Nel=150; d_t=0.002; dtw=0.068; tf=float(f3['time_fluid'])  # 0.274
alpha=0.5; C_damp=2.0
SCALE_F=122.5   # MATLAB nondim force -> physical (Python) and back

# ---- anchors (F_k from _a==old_, F_{k+1} from current) ----
Fp_a   = g(f3,'Qf_p_global_a').ravel();   Fp_n   = g(f3,'Qf_p_global').ravel()
Mat_a  = g(f3,'Qf_p_mat_global_a');       Mat_n  = g(f3,'Qf_p_mat_global')
Mat0_a = g(f3,'Qf_p_mat0_global_a');      Mat0_n = g(f3,'Qf_p_mat0_global')
L2_a   = g(f3,'Qf_p_lift2_mat_global_a'); L2_n   = g(f3,'Qf_p_lift2_mat_global')
if PRED:
    # predictor pass: F(t) = F_k + (F_k - F_{k-1})*beta  -> "_n" := 2F_k - F_{k-1} so a+( n-a)*beta works
    Fkm1   = g(f2,'Qf_p_global').ravel()
    Matkm1 = g(f2,'Qf_p_mat_global')
    Mat0km1= g(f2,'Qf_p_mat0_global')
    L2km1  = g(f2,'Qf_p_lift2_mat_global')
    Fp_n   = 2*Fp_a   - Fkm1
    Mat_n  = 2*Mat_a  - Matkm1
    Mat0_n = 2*Mat0_a - Mat0km1
    L2_n   = 2*L2_a   - L2km1
M_global = g(f3,'M_global')
Qf_time  = g(f3,'Qf_time_global').ravel()
Sc_col = f3['Sc_mat_col_global'];  S31 = f3['Sc_mat_31'];  S24 = f3['Sc_mat_24']
V_in   = g(f3,'V_in'); Vwp = g(f3,'V_wake_plate'); dA2G = g(f3,'dt_Amat2_Gamma')
dA1    = g(f3,'dt_Amat1'); Gamma = g(f3,'Gamma').ravel()

# ---- BCs (MATLAB 1-based node lists; r, dx_r, dy_r all constrained) ----
vp=f3['var_param']
nodes_c = np.asarray(vp.node_r_0,dtype=int).ravel()
i_vec=[]
for n0 in nodes_c:
    b=9*(n0-1)
    i_vec += [b+0,b+1,b+2,b+3,b+4,b+5,b+6,b+7,b+8]
i_vec=np.array(sorted(i_vec)); free=np.setdiff1d(np.arange(N),i_vec)
nf=len(free)
print(("[PREDICTOR] " if PRED else "[CORRECTOR] ")+f"constrained dofs={len(i_vec)}, free={nf}")

# ---- Python shell for elastic forces (perm + scale, validated bit-exact) ----
params=yamano_params()
shell,_,_,_=build_yamano_shell(params,nx=15,ny=10)
def perm_ml2py(Nx,Ny):
    nn=(Nx+1)*(Ny+1); p=np.empty(9*nn,dtype=int)
    for j in range(Ny+1):
        for i in range(Nx+1):
            kp=j*(Nx+1)+i; km=i*(Ny+1)+j
            for d in range(9): p[9*kp+d]=9*km+d
    return p
perm=perm_ml2py(15,10)           # v_py = v_ml[perm]
invp=np.empty_like(perm); invp[perm]=np.arange(N)   # v_ml = v_py[invp]

def elastic(q_ml):
    q_py=q_ml[perm]
    Qm,Qb=shell._internal_forces_separated(q_py)
    Qe_ml=np.asarray(Qm).ravel()[invp]/SCALE_F
    Qk_ml=np.asarray(Qb).ravel()[invp]/SCALE_F
    return Qe_ml,Qk_ml

def kmem(q_ml):
    K=shell._tangent_K_mem(q_ml[perm])
    K=K.toarray() if sp.issparse(K) else np.asarray(K)
    return K[np.ix_(invp,invp)]/SCALE_F

def dt_n_vec(q,dtq):
    r13=(S31@q).reshape(-1,3); r42=(S24@q).reshape(-1,3)
    dt13=(S31@dtq).reshape(-1,3); dt42=(S24@dtq).reshape(-1,3)
    cr=np.cross(r13,r42); nrm=np.linalg.norm(cr,axis=1,keepdims=True)
    nv=cr/nrm
    dtc=(np.cross(dt13,r42)+np.cross(r13,dt42))/nrm
    dtn=dtc-nv*np.sum(dtc*nv,axis=1,keepdims=True)
    return nv,dtn

def aero_terms(beta,dtq,q):
    """Per-step time-extrapolated aero (solve_structure lines 15-66)."""
    Fp   = Fp_a   + (Fp_n  - Fp_a )*beta     # old_==_a so (new-old)=(new-a)
    Mat  = Mat_a  + (Mat_n - Mat_a)*beta
    Mat0 = Mat0_a + (Mat0_n- Mat0_a)*beta
    L2   = L2_a   + (L2_n  - L2_a )*beta
    nv,dtn=dt_n_vec(q,dtq)
    dt_rc=(Sc_col@dtq).reshape(-1,3)
    slip=np.sum((dt_rc - V_in - Vwp - dA2G)*dtn,axis=1) - dA1@Gamma
    f_mat0 = Mat0@slip
    f_l2   = L2@dt_rc.ravel()
    return Fp,Mat,f_mat0,f_l2,(Mat0,L2)

def step(X,i_time):
    time=i_time*d_t; beta=(time-tf)/dtw
    q=X[:N].copy(); dtq=X[N:].copy()
    Fp,Mat,f_mat0_n,f_l2_n,(Mat0_tv,L2_tv)=aero_terms(beta,dtq,q)
    Qe_n,Qk_n=elastic(q)
    dqQe=kmem(q)
    # M_eff and block-reduced theta-method (new_X_func_FAST, Fmat=I, Qd=J=0)
    Meff=(M_global - Mat)[np.ix_(free,free)]
    D21=(C_damp*d_t/2.0)*dqQe[np.ix_(free,free)]
    S=Meff + alpha*d_t*D21
    lu=lu_factor(S)
    qf=q[free]; dqf=dtq[free]
    b1=qf+(1.0-alpha)*d_t*dqf
    b2=D21@qf + Meff@dqf
    def solveA1(c1,c2):
        x2=lu_solve(lu, c2 - D21@c1)
        return c1+alpha*d_t*x2, x2
    a1,a2=solveA1(b1,b2)                       # A1^{-1} A2 X (cached both stages)
    # stage 0
    Qf0=Qf_time*0.0 + Fp + f_mat0_n + f_l2_n   # q_in_norm=0 for t>0.2
    Q0=(Qf0 - (Qe_n+Qk_n))[free]
    s1,s2=solveA1(np.zeros(nf), Q0)
    Xp=X.copy(); Xp[free]=a1+d_t*s1; Xp[N+free]=a2+d_t*s2
    # predictor re-evals (solve_structure lines 108-146)
    qp=Xp[:N]; dtqp=Xp[N:]
    nv_p,dtn_p=dt_n_vec(qp,dtqp)
    dt_rc_p=(Sc_col@dtqp).reshape(-1,3)
    slip_p=np.sum((dt_rc_p - V_in - Vwp - dA2G)*dtn_p,axis=1) - dA1@Gamma
    f_mat0_p=Mat0_tv@slip_p
    f_l2_p=L2_tv@dt_rc_p.ravel()
    _,Qk_p=elastic(qp)
    # stage 1
    Qf1=Fp + (f_mat0_n+f_mat0_p)/2.0 + (f_l2_n+f_l2_p)/2.0
    Qe1=Qe_n + (Qk_n+Qk_p)/2.0
    Q1=(Qf1 - Qe1)[free]
    t1,t2=solveA1(np.zeros(nf), Q1)
    Xn=X.copy(); Xn[free]=a1+d_t*t1; Xn[N+free]=a2+d_t*t2
    return Xn

hX3=np.asarray(f3['h_X_vec']); hX4=np.asarray(f4['h_X_vec'])
if PRED:
    hX4=hX3   # truth = predictor pass stored in f3's own history
X=hX3[:,137].copy()        # MATLAB h_X_vec(:,138) = state at start of step 138
zdof=None
# tip z dof in MATLAB order: node with max x,y -> last node (16*11=176) -> dof 9*175+3
zdof=9*175+2
print(f"start tip z={X[zdof]:+.6e}  truth start={hX4[:,137][zdof]:+.6e}")
for k,i_time in enumerate(range(138,172)):
    X=step(X,i_time)
    truth=hX4[:,i_time]    # h_X_vec(:,i_time+1) 1-based = [:,i_time] 0-based
    err=np.abs(X-truth).max()
    rel=err/(np.abs(truth).max())
    if k%5==0 or k==33:
        print(f"i_time={i_time}  t={i_time*d_t:.3f}  max|X-truth|={err:.3e}  rel={rel:.3e}  tip py={X[zdof]:+.6e} ml={truth[zdof]:+.6e}")
print(f"\nFINAL block-replay error: {np.abs(X-hX4[:,171]).max():.3e}")

# ---- error DOF pattern diagnosis ----
truth=hX4[:,171]
d=np.abs(X-truth)
dq_err=d[:N]; dtq_err=d[N:]
print(f"q-part err max={dq_err.max():.3e}  dt_q-part err max={dtq_err.max():.3e}")
def label(i):
    node=i//9; comp=i%9
    im=node//11; jm=node%11   # MATLAB node = i*(Ny+1)+j
    kind=['rx','ry','rz','dxrx','dxry','dxrz','dyrx','dyry','dyrz'][comp]
    return f"node(i={im},j={jm})/{kind}"
print("top-8 q errs:")
for i in np.argsort(dq_err)[::-1][:8]:
    print(f"  {label(i):26s} err={dq_err[i]:.3e}  X={X[i]:+.5e} truth={truth[i]:+.5e}")
print("top-8 dt_q errs:")
for i in np.argsort(dtq_err)[::-1][:8]:
    print(f"  {label(i):26s} err={dtq_err[i]:.3e}  X={X[N+i]:+.5e} truth={truth[N+i]:+.5e}")
