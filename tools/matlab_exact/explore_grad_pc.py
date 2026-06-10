"""Exploratory study: gradient-enhanced predictor-corrector for the two-pass
strong-coupling scheme.

The MATLAB scheme is ONE Picard iteration of the block fixed point
    F* = Fluid( StructMarch_end(F*) ).
Questions:
  E1  How far is one Picard step from the converged fixed point? (intrinsic
      splitting error of the MATLAB scheme; Picard-k as reference)
  E2  Does a TANGENT (JVP-along-trajectory) predictor — replacing the
      time-extrapolation slope by a directional derivative of the fluid map —
      place the fluid solve at a more consistent state, and reduce the error
      vs the converged reference?
  E3  Accuracy/cost frontier at doubled block size (dtw x2: half the fluid
      solves) for standard vs enhanced variants.

Metrics per boundary: tip, pred_state_err = max|X_fluidinput - X_corrected|,
n_fluid_solves. Usage:
  python explore_grad_pc.py MODE N_BLOCKS [BLOCK_STEPS]
  MODE in {std, picard2, picard3, tangent}
"""
import os, sys, time as tmod
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.linalg import lu_factor, lu_solve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.chdir(ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from ml_fluid_step import MatlabFluidStep
from ml_fluidforce import MatlabFluidForce
from run_standalone_yamano import yamano_params, build_yamano_shell

MODE = sys.argv[1] if len(sys.argv) > 1 else 'std'
N_BLOCKS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BLOCK_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 34

F3 = 'FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step3_t0.3000.mat'
f3s = loadmat(F3, squeeze_me=True, struct_as_record=False)
f3r = loadmat(F3, squeeze_me=False)
g = lambda f, k: (f[k].toarray() if sp.issparse(f[k]) else np.asarray(f[k], dtype=float))

N = 1584; Ne = 150; Nx_, Ny_ = 15, 10
d_t = 0.002
dtw = BLOCK_STEPS * d_t          # block size (MATLAB: 0.068)
alpha = 0.5; C_damp = 2.0
SCALE_F = 122.5

# ---- fluid step constants ----
sq = lambda k: np.asarray(f3s[k]).squeeze()
ms = MatlabFluidStep.__new__(MatlabFluidStep)
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

M_global = g(f3s, 'M_global')
Qf_time = g(f3s, 'Qf_time_global').ravel()
q_in_norm = lambda t: 0.5 * np.sin(np.pi * t / 0.2) if t < 0.2 else 0.0
nodes_c = np.asarray(vp.node_r_0, dtype=int).ravel()
i_vec = np.array(sorted(int(9 * (n0 - 1) + d) for n0 in nodes_c for d in range(9)))
free = np.setdiff1d(np.arange(N), i_vec); nf = len(free)

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


class Aero:
    def __init__(self):
        z = np.zeros
        self.Fp = z(N); self.mat = z((N, N)); self.mat0 = z((N, Ne)); self.l2 = z((N, 3 * Ne))
        self.Gamma = z(Ne); self.dA1 = z((Ne, Ne)); self.dA2G = z((Ne, 3)); self.Vwp = z((Ne, 3))

    @staticmethod
    def from_out(o):
        a = Aero()
        a.Fp = o['Qf_p']; a.mat = o['mat']; a.mat0 = o['mat0']; a.l2 = o['lift2']
        a.Gamma = o['Gamma']; a.dA1 = o['dt_Amat1']; a.dA2G = o['dt_Amat2_Gamma']; a.Vwp = o['Vwp']
        return a

    def axpy(self, other, c):
        r = Aero()
        for k in ('Fp', 'mat', 'mat0', 'l2'):
            setattr(r, k, getattr(self, k) + c * getattr(other, k))
        for k in ('Gamma', 'dA1', 'dA2G', 'Vwp'):
            setattr(r, k, getattr(self, k))
        return r

    def diff(self, other):
        r = Aero()
        for k in ('Fp', 'mat', 'mat0', 'l2'):
            setattr(r, k, getattr(self, k) - getattr(other, k))
        return r


def march(X, steps, anc_a, slope, tf, wq):
    X = X.copy()
    for it in steps:
        t = it * d_t
        beta = (t - tf) / dtw
        Fp = anc_a.Fp + slope.Fp * beta
        Mat = anc_a.mat + slope.mat * beta
        Mat0 = anc_a.mat0 + slope.mat0 * beta
        L2 = anc_a.l2 + slope.l2 * beta
        q = X[:N]; dtq = X[N:]
        nv, dtn = dt_n_vec(q, dtq)
        drc = np.asarray(ms.Sc_col @ dtq).reshape(-1, 3)
        slip = np.einsum('ec,ec->e', drc - ms.V_in - wq.Vwp - wq.dA2G, dtn) - wq.dA1 @ wq.Gamma
        fm0 = Mat0 @ slip; fl2 = L2 @ drc.ravel()
        Qe_n, Qk_n = elastic(q)
        dqQe = kmem(q)
        Meff = (M_global - Mat)[np.ix_(free, free)]
        D21 = (C_damp * d_t / 2.0) * dqQe[np.ix_(free, free)]
        S = Meff + alpha * d_t * D21
        lu = lu_factor(S)
        qf = q[free]; dqf = dtq[free]
        b1 = qf + (1.0 - alpha) * d_t * dqf
        b2 = D21 @ qf + Meff @ dqf
        def sol(c1, c2):
            x2 = lu_solve(lu, c2 - D21 @ c1)
            return c1 + alpha * d_t * x2, x2
        a1, a2 = sol(b1, b2)
        pulse = Qf_time * q_in_norm(t)
        s1, s2 = sol(np.zeros(nf), (pulse + Fp + fm0 + fl2 - (Qe_n + Qk_n))[free])
        Xp = X.copy(); Xp[free] = a1 + d_t * s1; Xp[N + free] = a2 + d_t * s2
        qp = Xp[:N]; dtqp = Xp[N:]
        nv2, dtn2 = dt_n_vec(qp, dtqp)
        drc2 = np.asarray(ms.Sc_col @ dtqp).reshape(-1, 3)
        slip2 = np.einsum('ec,ec->e', drc2 - ms.V_in - wq.Vwp - wq.dA2G, dtn2) - wq.dA1 @ wq.Gamma
        fm0b = Mat0 @ slip2; fl2b = L2 @ drc2.ravel()
        _, Qk_p = elastic(qp)
        Qf1 = pulse + Fp + (fm0 + fm0b) / 2.0 + (fl2 + fl2b) / 2.0
        Qe1 = Qe_n + (Qk_n + Qk_p) / 2.0
        t1, t2 = sol(np.zeros(nf), (Qf1 - Qe1)[free])
        X[free] = a1 + d_t * t1; X[N + free] = a2 + d_t * t2
    return X


def run_chain(mode, n_blocks):
    hX3 = np.asarray(f3s['h_X_vec'])
    X = hX3[:, 0].copy()
    zdof = 9 * 175 + 2
    F_old = Aero(); F_cur = Aero(); F_a = Aero()
    tf = 0.0; wake = None
    Gp = np.zeros(Ne); Gp2 = np.zeros(Ne)
    iw = 1; prev_b = None
    n_fluid = 0
    boundaries = [1 + BLOCK_STEPS * k for k in range(n_blocks + 1)]
    rows = []
    t0 = tmod.time()
    for b in boundaries:
        steps = [b] if prev_b is None else list(range(prev_b + 1, b + 1))
        slope_p = F_cur.diff(F_old)
        # ---- tangent predictor: slope from directional derivative of fluid map ----
        if mode == 'tangent' and prev_b is not None:
            delta = 2  # look-ahead steps
            X_d = march(X, steps[:delta], F_a, slope_p, tf, F_cur)
            out0 = ms.solve_chain(X, wake, Gp, Gp2, first_wake=(iw == 1)); n_fluid += 1
            outd = ms.solve_chain(X_d, wake, Gp, Gp2, first_wake=(iw == 1)); n_fluid += 1
            Fk0 = Aero.from_out(out0); Fkd = Aero.from_out(outd)
            scale = dtw / (delta * d_t)
            slope_p = Fkd.diff(Fk0)
            for k in ('Fp', 'mat', 'mat0', 'l2'):
                setattr(slope_p, k, getattr(slope_p, k) * scale)
            anc_pred = Fk0
        else:
            anc_pred = F_a
        # ---- predictor pass ----
        if len(steps) > 1:
            Xf = march(X, steps[:-1], anc_pred, slope_p, tf, F_cur)
        else:
            Xf = X.copy()
        out = ms.solve_chain(Xf, wake, Gp, Gp2, first_wake=(iw == 1)); n_fluid += 1
        F_new = Aero.from_out(out)
        wake_new = out['wake']
        # ---- corrector (+ optional Picard re-iterations) ----
        n_pic = {'std': 1, 'tangent': 1, 'picard2': 2, 'picard3': 3}[mode]
        for pic in range(n_pic):
            slope_c = F_new.diff(F_cur)
            Xc = march(X, steps, F_a, slope_c, tf, F_new)
            if pic < n_pic - 1:
                # re-solve fluid at the corrected pre-boundary state
                Xc_pre = march(X, steps[:-1], F_a, slope_c, tf, F_new) if len(steps) > 1 else X.copy()
                out = ms.solve_chain(Xc_pre, wake, Gp, Gp2, first_wake=(iw == 1)); n_fluid += 1
                F_new = Aero.from_out(out)
                wake_new = out['wake']
                Xf = Xc_pre
        # predictor-state inconsistency metric
        Xc_pre_final = march(X, steps[:-1], F_a, F_new.diff(F_cur), tf, F_new) if len(steps) > 1 else X.copy()
        pred_err = np.abs(Xf - Xc_pre_final).max()
        X = Xc
        iw += 1
        wake = wake_new
        Gp2 = Gp; Gp = out['Gamma']
        F_old = F_cur; F_cur = F_new; F_a = F_new
        tf = b * d_t
        prev_b = b
        rows.append((b, b * d_t, X[zdof], pred_err, n_fluid))
        print(f"[{mode}] b={b:4d} t*={b*d_t:.3f} tip={X[zdof]:+.6e} pred_err={pred_err:.3e} fluids={n_fluid} ({tmod.time()-t0:.0f}s)", flush=True)
    np.save(f'/tmp/explore_{mode}_bs{BLOCK_STEPS}.npy', np.array(rows))
    return rows


if __name__ == '__main__':
    run_chain(MODE, N_BLOCKS)
