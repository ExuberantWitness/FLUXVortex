"""Step 4 — full-trajectory chain: MATLAB exe.m two-pass scheme with the
closed-book Python fluid solve, from t=0, validated per-boundary vs fixtures.

Per exe.m: boundaries at i_time = 1, 35, 69, ... Each block [b_prev+1 .. b]:
  PREDICTOR pass: F(t) = a + (cur-old)*(t-tf)/dtw   [a=cur=F_k, old=F_{k-1}]
  fluid solve at state h_X(b) (pre-boundary-step predictor state)
  CORRECTOR pass: F(t) = a + (new-old')*(t-tf)/dtw  [a=old'=F_k, new=F_{k+1}]
  anchors update at second boundary visit: a <- F_{k+1}, tf <- t_b
Wake bookkeeping (validated): shed prepends Gamma_trail = G_{k-1}(TE); advection
bound source = G_k; post-solve trail update = G_k(TE).
Usage: python chain_full.py [N_BLOCKS]
"""
import os, sys, time as tmod
os.chdir('/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.linalg import lu_factor, lu_solve
from ml_fluid_step import MatlabFluidStep, dt_q1234_mat
from vpm_hybrid import HybridFluidStep
from ml_fluidforce import MatlabFluidForce
from run_standalone_yamano import yamano_params, build_yamano_shell

N_BLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
K_RINGS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

F3 = 'FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step3_t0.3000.mat'
f3s = loadmat(F3, squeeze_me=True, struct_as_record=False)
f3r = loadmat(F3, squeeze_me=False)
g = lambda f, k: (f[k].toarray() if sp.issparse(f[k]) else np.asarray(f[k], dtype=float))

N = 1584; Ne = 150; Nx_, Ny_ = 15, 10
d_t = 0.002; dtw = 0.068
alpha = 0.5; C_damp = 2.0
SCALE_F = 122.5

# ---- fluid step (constants from fixture; they are state-independent) ----
sq = lambda k: np.asarray(f3s[k]).squeeze()
ms = HybridFluidStep.__new__(HybridFluidStep)
vp = f3s['var_param']
ms.Nx, ms.Ny, ms.Ne, ms.Nq = Nx_, Ny_, Ne, N
ms.Length = float(np.asarray(vp.Length).squeeze())
ms.r_eps_fine = float(np.asarray(vp.r_eps.fine).squeeze())
ms.r_eps_rough = float(np.asarray(vp.r_eps.rough).squeeze())
ms.Ncore = int(np.asarray(vp.Ncore).squeeze())
ms.eps_v = float(np.asarray(vp.eps_v).squeeze())
ms.d_t_wake = dtw
ms.U_in = float(sq('U_in'))
ms.V_in = g(f3s, 'V_in')
ms.Rtrunc = 5.5 * ms.Length
ms.Rnochange = ms.Rtrunc - 1.5 * ms.Length
ms.Sc_col = f3s['Sc_mat_col_global']; ms.S31 = f3s['Sc_mat_31']; ms.S24 = f3s['Sc_mat_24']
ms.Sp = [f3s[f'Sc_mat_panel_global_{k}'] for k in (1, 2, 3, 4)]
ms.asm = MatlabFluidForce(f3r)
ms.idof = ms.asm.idof
ms.Sc_col_d = g(f3s, 'Sc_mat_col_global')
ms.init_hybrid(K_RINGS)

M_global = g(f3s, 'M_global')
Qf_time = g(f3s, 'Qf_time_global').ravel()
q_in_norm = lambda t: 0.5 * np.sin(np.pi * t / 0.2) if t < 0.2 else 0.0

# BCs
nodes_c = np.asarray(vp.node_r_0, dtype=int).ravel()
i_vec = np.array(sorted(int(9 * (n0 - 1) + d) for n0 in nodes_c for d in range(9)))
free = np.setdiff1d(np.arange(N), i_vec); nf = len(free)

# elastic via python shell (validated route)
params = yamano_params()
shell, _, _, _ = build_yamano_shell(params, nx=Nx_, ny=Ny_)
def perm_ml2py(Nx, Ny):
    nn = (Nx + 1) * (Ny + 1); p = np.empty(9 * nn, dtype=int)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            kp = j * (Nx + 1) + i; km = i * (Ny + 1) + j
            for d in range(9): p[9 * kp + d] = 9 * km + d
    return p
perm = perm_ml2py(Nx_, Ny_)
invp = np.empty_like(perm); invp[perm] = np.arange(N)

def elastic(q_ml):
    Qm, Qb = shell._internal_forces_separated(q_ml[perm])
    return (np.asarray(Qm).ravel()[invp] / SCALE_F,
            np.asarray(Qb).ravel()[invp] / SCALE_F)

def kmem(q_ml):
    K = shell._tangent_K_mem(q_ml[perm])
    K = K.toarray() if sp.issparse(K) else np.asarray(K)
    return K[np.ix_(invp, invp)] / SCALE_F

def dt_n_vec(q, dtq):
    r13 = np.asarray(ms.S31 @ q).reshape(-1, 3); r42 = np.asarray(ms.S24 @ q).reshape(-1, 3)
    d13 = np.asarray(ms.S31 @ dtq).reshape(-1, 3); d42 = np.asarray(ms.S24 @ dtq).reshape(-1, 3)
    cr = np.cross(r13, r42); nrm = np.linalg.norm(cr, axis=1, keepdims=True)
    nv = cr / nrm
    dtc = (np.cross(d13, r42) + np.cross(r13, d42)) / nrm
    return nv, dtc - nv * np.sum(dtc * nv, axis=1, keepdims=True)


class AeroState:
    """Fluid-solve outputs frozen over a structural pass."""
    def __init__(self):
        z = np.zeros
        self.Fp = z(N); self.mat = z((N, N)); self.mat0 = z((N, Ne)); self.l2 = z((N, 3 * Ne))
        self.Gamma = z(Ne); self.dA1 = z((Ne, Ne)); self.dA2G = z((Ne, 3)); self.Vwp = z((Ne, 3))

def aero_from_fluid(out):
    a = AeroState()
    a.Fp = out['Qf_p']; a.mat = out['mat']; a.mat0 = out['mat0']; a.l2 = out['lift2']
    a.Gamma = out['Gamma']; a.dA1 = out['dt_Amat1']; a.dA2G = out['dt_Amat2_Gamma']
    a.Vwp = out['Vwp']
    return a


def march(X, steps, anc_a, anc_slope, tf, wq):
    """March structural steps (list of i_time) per solve_structure.m.
    anc_a/anc_slope: AeroState anchors (F(t) = a + slope*(t-tf)/dtw).
    wq: AeroState giving frozen wake quantities (Gamma, dA1, dA2G, Vwp)."""
    X = X.copy()
    for it in steps:
        t = it * d_t
        beta = (t - tf) / dtw
        Fp = anc_a.Fp + anc_slope.Fp * beta
        Mat = anc_a.mat + anc_slope.mat * beta
        Mat0 = anc_a.mat0 + anc_slope.mat0 * beta
        L2 = anc_a.l2 + anc_slope.l2 * beta
        q = X[:N]; dtq = X[N:]
        nv, dtn = dt_n_vec(q, dtq)
        dt_rc = np.asarray(ms.Sc_col @ dtq).reshape(-1, 3)
        slip = np.einsum('ec,ec->e', dt_rc - ms.V_in - wq.Vwp - wq.dA2G, dtn) - wq.dA1 @ wq.Gamma
        f_mat0_n = Mat0 @ slip
        f_l2_n = L2 @ dt_rc.ravel()
        Qe_n, Qk_n = elastic(q)
        dqQe = kmem(q)
        Meff = (M_global - Mat)[np.ix_(free, free)]
        D21 = (C_damp * d_t / 2.0) * dqQe[np.ix_(free, free)]
        S = Meff + alpha * d_t * D21
        lu = lu_factor(S)
        qf = q[free]; dqf = dtq[free]
        b1 = qf + (1.0 - alpha) * d_t * dqf
        b2 = D21 @ qf + Meff @ dqf
        def solveA1(c1, c2):
            x2 = lu_solve(lu, c2 - D21 @ c1)
            return c1 + alpha * d_t * x2, x2
        a1, a2 = solveA1(b1, b2)
        pulse = Qf_time * q_in_norm(t)
        Qf0 = pulse + Fp + f_mat0_n + f_l2_n
        s1, s2 = solveA1(np.zeros(nf), (Qf0 - (Qe_n + Qk_n))[free])
        Xp = X.copy(); Xp[free] = a1 + d_t * s1; Xp[N + free] = a2 + d_t * s2
        qp = Xp[:N]; dtqp = Xp[N:]
        nv_p, dtn_p = dt_n_vec(qp, dtqp)
        dt_rc_p = np.asarray(ms.Sc_col @ dtqp).reshape(-1, 3)
        slip_p = np.einsum('ec,ec->e', dt_rc_p - ms.V_in - wq.Vwp - wq.dA2G, dtn_p) - wq.dA1 @ wq.Gamma
        f_mat0_p = Mat0 @ slip_p
        f_l2_p = L2 @ dt_rc_p.ravel()
        _, Qk_p = elastic(qp)
        Qf1 = pulse + Fp + (f_mat0_n + f_mat0_p) / 2.0 + (f_l2_n + f_l2_p) / 2.0
        Qe1 = Qe_n + (Qk_n + Qk_p) / 2.0
        t1, t2 = solveA1(np.zeros(nf), (Qf1 - Qe1)[free])
        X[free] = a1 + d_t * t1; X[N + free] = a2 + d_t * t2
    return X


def zero_aero():
    return AeroState()

# ---- initial state (flat plate) ----
hX3 = np.asarray(f3s['h_X_vec'])
X0 = hX3[:, 0].copy()
zdof = 9 * 175 + 2  # tip z (MATLAB order)

# fluid history state
F_old = zero_aero()        # F_{k-1}
F_cur = zero_aero()        # F_k
F_a = zero_aero()          # anchor a
tf = 0.0
wake = None
Gamma_prev = np.zeros(Ne)      # G_k (advection source, trail update)
Gamma_prev2 = np.zeros(Ne)     # G_{k-1} (shed prepend)
i_wake_time = 1

X = X0.copy()
boundaries = [1 + 34 * k for k in range(N_BLOCKS + 1)]
t0 = tmod.time()
hG_ours = []
prev_b = None
print(f"chain: {N_BLOCKS} blocks, boundaries {boundaries[:5]}...", flush=True)
for bi, b in enumerate(boundaries):
    steps = [b] if prev_b is None else list(range(prev_b + 1, b + 1))
    slope_pred = AeroState()
    for f_ in ('Fp', 'mat', 'mat0', 'l2'):
        setattr(slope_pred, f_, getattr(F_cur, f_) - getattr(F_old, f_))
    # PREDICTOR pass (anchor a = F_a, slope = F_cur - F_old, wake quantities = F_cur's)
    Xp_end = march(X, steps, F_a, slope_pred, tf, F_cur)
    # fluid solve at state BEFORE the boundary step of the predictor pass
    if len(steps) > 1:
        X_fluid = march(X, steps[:-1], F_a, slope_pred, tf, F_cur)
    else:
        X_fluid = X.copy()    # boundary 1: fluid solved at X after step... see exe: X_vec at i_time=1 = X0
    out = ms.solve_chain(X_fluid, wake, Gamma_prev, Gamma_prev2,
                         first_wake=(i_wake_time == 1))
    i_wake_time += 1
    wake = out['wake']
    F_new = aero_from_fluid(out)
    # CORRECTOR pass: a = old = F_cur(=F_a after boundary update), slope = F_new - F_cur
    slope_corr = AeroState()
    for f_ in ('Fp', 'mat', 'mat0', 'l2'):
        setattr(slope_corr, f_, getattr(F_new, f_) - getattr(F_cur, f_))
    X = march(X, steps, F_a, slope_corr, tf, F_new)
    # anchor updates (second boundary visit): a = F_new, tf = t_b
    Gamma_prev2 = Gamma_prev; Gamma_prev = out['Gamma']
    F_old = F_cur; F_cur = F_new; F_a = F_new
    tf = b * d_t
    prev_b = b
    print(f"  boundary {b:4d} t={b*d_t:.3f}  tip={X[zdof]:+.6e}  wake_rows={wake['r2'].shape[0]//Ny_ if wake else 0} npart={out.get('n_particles',0)}  ({tmod.time()-t0:.0f}s)", flush=True)
    np.save(f'/tmp/chainH{K_RINGS}_X_b{b}.npy', X)

print("done", flush=True)

# ---- validation vs fixture corrected trajectory ----
print("\n=== validation vs fixture h_X_vec (corrected cols from f4) ===")
F4='FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step8_t1.0000.mat'
f4s=loadmat(F4,squeeze_me=True,struct_as_record=False)
hX4=np.asarray(f4s['h_X_vec'])
for b in boundaries:
    if b+1 <= hX4.shape[1] and b <= 477:
        Xc=np.load(f'/tmp/chain_X_b{b}.npy')
        truth=hX4[:,b]      # h_X(:,b+1) 1-based
        err=np.abs(Xc-truth).max()
        print(f"  boundary {b:4d} t={b*d_t:.3f}  max|X-ml|={err:.3e}  tip ours={Xc[zdof]:+.6e} ml={truth[zdof]:+.6e}  ratio={Xc[zdof]/truth[zdof] if abs(truth[zdof])>1e-12 else float('nan'):.6f}")
