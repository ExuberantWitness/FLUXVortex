"""Pilot experiments for the gradient-enhanced predictor-corrector study.

P1 (interpolation order): corrector force-family interpolation
    linear (baseline) | quad (3-point history, gradient-free) |
    hermite (cubic w/ JVP endpoint derivatives, frozen-wake) |
    dense<N> (re-solve forces every N substeps with frozen wake = reference)
P2 (sub-shedding): --subshed S sheds S short rows per window (ring length
    U*dtw/S) while keeping one force solve per window.
P3 (mass ratio): --mscale m scales the structural mass matrix (heavy-loading
    probe; reference is self-consistent picard within same m).

Usage:
  python pilot_grad_pc.py MODE N_BLOCKS BLOCK_STEPS [--subshed S] [--mscale m] [--picard K]
  MODE in {linear, quad, hermite, dense1, dense4, ...}
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

from ml_uvlm import _cnorm
from ml_fluid_step import MatlabFluidStep, dt_q1234_mat
from ml_fluidforce import MatlabFluidForce
from run_standalone_yamano import yamano_params, build_yamano_shell

args = sys.argv[1:]
MODE = args[0] if args else 'linear'
N_BLOCKS = int(args[1]) if len(args) > 1 else 8
BLOCK_STEPS = int(args[2]) if len(args) > 2 else 34
SUBSHED = int(args[args.index('--subshed') + 1]) if '--subshed' in args else 1
MSCALE = float(args[args.index('--mscale') + 1]) if '--mscale' in args else 1.0
PICARD = int(args[args.index('--picard') + 1]) if '--picard' in args else 1
JVP = args[args.index('--jvp') + 1] if '--jvp' in args else 'march'
ACCEL = args[args.index('--accel') + 1] if '--accel' in args else 'none'
CONVTEST = int(args[args.index('--convtest') + 1]) if '--convtest' in args else 0
AITERS = int(args[args.index('--aiters') + 1]) if '--aiters' in args else 3
DENSE_N = int(MODE[5:]) if MODE.startswith('dense') else 0

F3 = 'FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step3_t0.3000.mat'
f3s = loadmat(F3, squeeze_me=True, struct_as_record=False)
f3r = loadmat(F3, squeeze_me=False)
g = lambda f, k: (f[k].toarray() if sp.issparse(f[k]) else np.asarray(f[k], dtype=float))

N = 1584; Ne = 150; Nx_, Ny_ = 15, 10
d_t = 0.002
dtw = BLOCK_STEPS * d_t
alpha = 0.5; C_damp = 2.0
SCALE_F = 122.5

sq = lambda k: np.asarray(f3s[k]).squeeze()
ms = MatlabFluidStep.__new__(MatlabFluidStep)
vp = f3s['var_param']
ms.Nx, ms.Ny, ms.Ne, ms.Nq = Nx_, Ny_, Ne, N
ms.Length = float(np.asarray(vp.Length).squeeze())
ms.r_eps_fine = float(np.asarray(vp.r_eps.fine).squeeze())
ms.r_eps_rough = float(np.asarray(vp.r_eps.rough).squeeze())
ms.Ncore = int(np.asarray(vp.Ncore).squeeze())
ms.eps_v = float(np.asarray(vp.eps_v).squeeze())
ms.d_t_wake = dtw / SUBSHED          # shedding/advection cadence
ms.U_in = float(sq('U_in'))
ms.V_in = g(f3s, 'V_in')
ms.Rtrunc = 5.5 * ms.Length
ms.Rnochange = ms.Rtrunc - 1.5 * ms.Length
ms.Sc_col = f3s['Sc_mat_col_global']; ms.S31 = f3s['Sc_mat_31']; ms.S24 = f3s['Sc_mat_24']
ms.Sp = [f3s[f'Sc_mat_panel_global_{k}'] for k in (1, 2, 3, 4)]
ms.asm = MatlabFluidForce(f3r)
ms.idof = ms.asm.idof
ms.Sc_col_d = g(f3s, 'Sc_mat_col_global')

M_global = g(f3s, 'M_global') * MSCALE
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
    KEYS = ('Fp', 'mat', 'mat0', 'l2')
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
    def diff(self, o):
        r = Aero()
        for k in Aero.KEYS: setattr(r, k, getattr(self, k) - getattr(o, k))
        return r
    def scale(self, c):
        r = Aero()
        for k in Aero.KEYS: setattr(r, k, getattr(self, k) * c)
        for k in ('Gamma', 'dA1', 'dA2G', 'Vwp'): setattr(r, k, getattr(self, k))
        return r


def frozen_force(X, wk, Gamma_prev, trail_update=None):
    """Force families at state X with FROZEN wake (no advect/shed): the dense
    sampling primitive. Mirrors solve_chain minus generate_wake/truncate.
    trail_update: if given (Ny,), post-solve trail circulation for the force
    side (solve_chain line-157 semantics)."""
    Nq = N
    q = X[:Nq]; dtq = X[Nq:]
    bP = ms.panels(q)
    dt_bP = [np.asarray(S @ dtq).reshape(-1, 3) for S in ms.Sp]
    rc = ms.colloc(q); dt_rc = ms.colloc(dtq)
    nv, dtn = ms.normals(q, dtq)
    from ml_uvlm import aic_from_q1234
    Vq = ms.q1234(rc, bP, fine=True)
    A = aic_from_q1234(Vq, nv)
    wP = [wk['r1'], wk['r2'], wk['r3'], wk['r4']]
    Vwp_rhs, q_wake = ms.vwake(rc, wP, wk['Gam'], fine=True)
    Vn = np.einsum('tc,tc->t', dt_rc - ms.V_in - Vwp_rhs, nv)
    Gamma = np.linalg.solve(A, Vn)
    if trail_update is not None:
        Gam2 = wk['Gam'].copy(); Gam2[:Ny_] = trail_update
        Vwp = np.einsum('tsc,s->tc', q_wake, Gam2)
        wk = dict(wk); wk['Gam'] = Gam2
    else:
        Vwp = Vwp_rhs
    Vg = np.einsum('tsc,s->tc', Vq, Gamma)
    V_surf1 = Vg + Vwp + ms.V_in
    t21 = bP[1] - bP[0]; t34 = bP[2] - bP[3]
    t14 = bP[0] - bP[3]; t23 = bP[1] - bP[2]
    tx = (t21 + t34) / 2; ty = (t14 + t23) / 2
    dx = _cnorm(tx)[:, None]; dy = _cnorm(ty)[:, None]
    tx /= dx; ty /= dy
    Gm = Gamma.reshape(Nx_, Ny_)
    dxm = dx.reshape(Nx_, Ny_); dym = dy.reshape(Nx_, Ny_)
    dxG = np.vstack([Gm[:1], np.diff(Gm, axis=0)]) / dxm
    Gm2 = np.hstack([np.zeros((Nx_, 1), dtype=Gm.dtype), Gm, np.zeros((Nx_, 1), dtype=Gm.dtype)])
    dyG = (Gm2[:, 2:] - Gm2[:, :-2]) / (2 * dym)
    dyG[:, 0] = Gm[:, 0] / dym[:, 0]; dyG[:, -1] = -Gm[:, -1] / dym[:, -1]
    txdx = tx * dxG.reshape(-1, 1); tydy = ty * dyG.reshape(-1, 1)
    dp1 = np.einsum('tc,tc->t', V_surf1, txdx + tydy)
    dp2 = -(txdx + tydy)
    dtwP = [wk['dt1'], wk['dt2'], wk['dt3'], wk['dt4']]
    dtq_w = dt_q1234_mat(rc, wP, dt_rc, dtwP)
    Gw_dt_n = np.einsum('tc,tc->t', np.einsum('tsc,s->tc', dtq_w, wk['Gam']), nv)
    mv1 = np.linalg.solve(A, -Gw_dt_n)
    dtq_b = dt_q1234_mat(rc, bP, dt_rc, dt_bP)
    dA1 = np.einsum('tsc,tc->ts', dtq_b, nv)
    nvec_Sc = np.zeros((Ne, N), dtype=nv.dtype)
    for e in range(Ne):
        rows = ms.Sc_col_d[3*e:3*e+3][:, ms.idof[e]]
        nvec_Sc[e, ms.idof[e]] = nv[e] @ rows
    Mf1 = np.linalg.solve(A, nvec_Sc); Mf2 = np.linalg.inv(A)
    Qv, M0, L2m, Mm = ms.asm.assemble(dp1, mv1, dp2, Mf2, Mf1, nv)
    return dict(Qf_p=Qv, mat=Mm, mat0=M0, lift2=L2m, Gamma=Gamma, dt_Amat1=dA1,
                dt_Amat2_Gamma=Vg, Vwp=Vwp)


def force_at(beta, ctx):
    """Window force families at normalized time beta in [0,1] per MODE."""
    mode = ctx['mode']
    Fk, Fk1 = ctx['Fk'], ctx['Fk1']
    a = Aero()
    if mode == 'quad' and ctx['Fkm1'] is not None:
        # parabola through F_{k-1}(beta=-1), F_k(0), F_{k+1}(1)
        Fkm1 = ctx['Fkm1']
        for key in Aero.KEYS:
            f0 = getattr(Fk, key); f1 = getattr(Fk1, key); fm = getattr(Fkm1, key)
            c2 = (f1 + fm) / 2.0 - f0
            c1 = (f1 - fm) / 2.0
            setattr(a, key, f0 + c1 * beta + c2 * beta * beta)
    elif mode == 'hermite' and ctx['dFk'] is not None and ctx['dFk1'] is not None:
        dFk, dFk1 = ctx['dFk'], ctx['dFk1']   # dF/dbeta at endpoints
        h00 = 2*beta**3 - 3*beta**2 + 1; h10 = beta**3 - 2*beta**2 + beta
        h01 = -2*beta**3 + 3*beta**2;    h11 = beta**3 - beta**2
        for key in Aero.KEYS:
            setattr(a, key, h00*getattr(Fk, key) + h10*getattr(dFk, key)
                          + h01*getattr(Fk1, key) + h11*getattr(dFk1, key))
    else:  # linear
        for key in Aero.KEYS:
            setattr(a, key, getattr(Fk, key) + (getattr(Fk1, key) - getattr(Fk, key)) * beta)
    return a


def march(X, steps, tf, wq, ctx, dense_samples=None, pen_out=None):
    X = X.copy()
    for it in steps:
        if pen_out is not None and it == steps[-1]:
            pen_out.append(X.copy())
        t = it * d_t
        beta = (t - tf) / dtw
        if dense_samples is not None:
            # linear interpolation BETWEEN dense force samples
            x = beta * (len(dense_samples) - 1)
            i0 = min(int(x), len(dense_samples) - 2)
            w = x - i0
            A0, A1 = dense_samples[i0], dense_samples[i0 + 1]
            Fa = Aero()
            for key in Aero.KEYS:
                setattr(Fa, key, (1.0 - w) * getattr(A0, key) + w * getattr(A1, key))
            Fp_, Mat, Mat0, L2_ = Fa.Fp, Fa.mat, Fa.mat0, Fa.l2
        else:
            Fa = force_at(beta, ctx)
            Fp_, Mat, Mat0, L2_ = Fa.Fp, Fa.mat, Fa.mat0, Fa.l2
        q = X[:N]; dtq = X[N:]
        nv, dtn = dt_n_vec(q, dtq)
        drc = np.asarray(ms.Sc_col @ dtq).reshape(-1, 3)
        slip = np.einsum('ec,ec->e', drc - ms.V_in - wq.Vwp - wq.dA2G, dtn) - wq.dA1 @ wq.Gamma
        fm0 = Mat0 @ slip; fl2 = L2_ @ drc.ravel()
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
        s1, s2 = sol(np.zeros(nf), (pulse + Fp_ + fm0 + fl2 - (Qe_n + Qk_n))[free])
        Xp = X.copy(); Xp[free] = a1 + d_t * s1; Xp[N + free] = a2 + d_t * s2
        qp = Xp[:N]; dtqp = Xp[N:]
        nv2, dtn2 = dt_n_vec(qp, dtqp)
        drc2 = np.asarray(ms.Sc_col @ dtqp).reshape(-1, 3)
        slip2 = np.einsum('ec,ec->e', drc2 - ms.V_in - wq.Vwp - wq.dA2G, dtn2) - wq.dA1 @ wq.Gamma
        fm0b = Mat0 @ slip2; fl2b = L2_ @ drc2.ravel()
        _, Qk_p = elastic(qp)
        Qf1 = pulse + Fp_ + (fm0 + fm0b) / 2.0 + (fl2 + fl2b) / 2.0
        Qe1 = Qe_n + (Qk_n + Qk_p) / 2.0
        t1, t2 = sol(np.zeros(nf), (Qf1 - Qe1)[free])
        X[free] = a1 + d_t * t1; X[N + free] = a2 + d_t * t2
    return X


def jvp_force(X, wk, Gamma_prev, steps_ahead, tf, wq, ctx0):
    """dF/dbeta via frozen-wake look-ahead: march `steps_ahead` substeps with
    held F, frozen-wake force at both states, scaled difference."""
    F0 = Aero.from_out(frozen_force(X, wk, Gamma_prev))
    Xd = march(X, steps_ahead, tf, wq, ctx0)
    Fd = Aero.from_out(frozen_force(Xd, wk, Gamma_prev))
    scale = float(BLOCK_STEPS) / len(steps_ahead)   # d/dbeta
    return F0, Fd.diff(F0).scale(scale)


def jvp_traj(Xpt, V, wk, Gamma_prev, h):
    """dF/dbeta = dtw * [F(X+hV) - F(X-hV)]/(2h), frozen-wake force map,
    V = trajectory tangent dX/dt (consistent direction, no look-ahead march)."""
    Fp = Aero.from_out(frozen_force(Xpt + h * V, wk, Gamma_prev))
    Fm = Aero.from_out(frozen_force(Xpt - h * V, wk, Gamma_prev))
    return Fp.diff(Fm).scale(dtw / (2.0 * h))


def jvp_cs(Xpt, V, wk, Gamma_prev):
    """EXACT dF/dbeta via complex-step (machine-precision forward-mode AD):
    dF/dt . dtw = Im[F(X + i h V)]/h . dtw, h far below roundoff floor."""
    h = 1e-100
    out = frozen_force(Xpt.astype(complex) + (1j * h) * V, wk, Gamma_prev)
    imag = {k: (np.imag(v) / h if isinstance(v, np.ndarray) else v)
            for k, v in out.items()}
    return Aero.from_out(imag).scale(dtw)


def run():
    hX3 = np.asarray(f3s['h_X_vec'])
    X = hX3[:, 0].copy()
    zdof = 9 * 175 + 2
    F_old = Aero(); F_cur = Aero(); F_a = Aero()
    V_start = None
    dF_cache = None
    tf = 0.0; wake = None
    Gp = np.zeros(Ne); Gp2 = np.zeros(Ne)
    iw = 1; prev_b = None
    n_fluid = 0
    t0 = tmod.time()
    boundaries = [1 + BLOCK_STEPS * k for k in range(N_BLOCKS + 1)]
    rows = []
    for b in boundaries:
        steps = [b] if prev_b is None else list(range(prev_b + 1, b + 1))
        # ---- predictor (always linear extrapolation, MATLAB-style) ----
        lin_end = Aero()
        for key in Aero.KEYS:
            setattr(lin_end, key, getattr(F_a, key) + getattr(F_cur, key) - getattr(F_old, key))
        ctx_p = dict(mode='linear', Fk=F_a, Fk1=lin_end, Fkm1=None, dFk=None, dFk1=None)
        _pen_p = []
        Xf = march(X, steps[:-1], tf, F_cur, ctx_p, pen_out=_pen_p) if len(steps) > 1 else X.copy()
        V_end_pred = (Xf - _pen_p[0]) / d_t if _pen_p else None
        # ---- window fluid solve (with SUBSHED wake substeps) ----
        if SUBSHED == 1:
            out = ms.solve_chain(Xf, wake, Gp, Gp2, first_wake=(iw == 1)); n_fluid += 1
        else:
            # P2 sub-shedding: advect+shed S short rows at dtw/S cadence (held
            # bound Gamma sources, held trail circ), then ONE force solve on the
            # pre-advected wake with the standard post-solve trail update.
            wk = wake
            bP = ms.panels(Xf[:N])
            dt_bP = [np.asarray(Sm @ Xf[N:]).reshape(-1, 3) for Sm in ms.Sp]
            for s_ in range(SUBSHED):
                first = (wk is None)
                wk = ms.generate_wake(first, bP, dt_bP, Gp, wk, Gp2[-Ny_:])
            out = frozen_force(Xf, wk, Gp, trail_update=Gp[-Ny_:])
            out['wake'] = dict(wk)
            g2 = wk['Gam'].copy(); g2[:Ny_] = Gp[-Ny_:]
            out['wake']['Gam'] = g2          # carry line-157 trail update forward
            n_fluid += 1
        F_new = Aero.from_out(out)
        wake_new = out['wake']
        # ---- corrector force representation ----
        n_pic = PICARD
        dense_samples = None
        ctx_c = dict(mode=MODE if MODE in ('quad', 'hermite') else 'linear',
                     Fk=F_cur, Fk1=F_new, Fkm1=(F_old if iw > 2 else None),
                     dFk=None, dFk1=None)
        if MODE == 'hermite' and prev_b is not None:
            if JVP == 'cs' and V_start is not None and V_end_pred is not None:
                ctx_c['dFk'] = jvp_cs(X, V_start, wake_new, Gp); n_fluid += 1
                ctx_c['dFk1'] = jvp_cs(Xf, V_end_pred, wake_new, out['Gamma']); n_fluid += 1
            elif JVP == 'traj' and V_start is not None and V_end_pred is not None:
                # trajectory-consistent central-difference JVP (no look-ahead march)
                hfd = d_t
                ctx_c['dFk'] = jvp_traj(X, V_start, wake_new, Gp, hfd); n_fluid += 2
                ctx_c['dFk1'] = jvp_traj(Xf, V_end_pred, wake_new, out['Gamma'], hfd); n_fluid += 2
            else:
                la = steps[:2]
                _, dFk = jvp_force(X, wake_new, Gp, la, tf, F_new, ctx_c); n_fluid += 2
                Xe = march(X, steps, tf, F_new, dict(mode='linear', Fk=F_cur, Fk1=F_new,
                                                     Fkm1=None, dFk=None, dFk1=None))
                _, dFk1 = jvp_force(Xe, wake_new, out['Gamma'], la, tf, F_new, ctx_c); n_fluid += 2
                ctx_c['dFk'] = dFk; ctx_c['dFk1'] = dFk1
        if DENSE_N > 0 and prev_b is not None:
            # dense force sampling reference: frozen-wake solve every DENSE_N substeps
            dense_samples = []
            Xs = X.copy()
            nseg = max(1, len(steps) // DENSE_N)
            ctx_lin = dict(mode='linear', Fk=F_cur, Fk1=F_new, Fkm1=None, dFk=None, dFk1=None)
            for si in range(0, len(steps), DENSE_N):
                Fs = Aero.from_out(frozen_force(Xs, wake_new, Gp)); n_fluid += 1
                dense_samples.append(Fs)
                seg = steps[si:si + DENSE_N]
                Xs = march(Xs, seg, tf, F_new, ctx_lin)
            dense_samples.append(Aero.from_out(frozen_force(Xs, wake_new, Gp))); n_fluid += 1
        if CONVTEST and b == boundaries[CONVTEST] and len(steps) > 1:
            # ---- single-window residual-convergence test (clean solver metric) ----
            def Hmap(Xp_):
                o = ms.solve_chain(Xp_, wake, Gp, Gp2, first_wake=(iw == 1))
                Fn = Aero.from_out(o)
                cx = dict(mode='linear', Fk=F_cur, Fk1=Fn, Fkm1=None, dFk=None, dFk1=None)
                return march(X, steps[:-1], tf, Fn, cx)
            import numpy.linalg as la
            nX = la.norm(Xf)
            print(f"== CONVTEST window b={b} t*={b*d_t:.3f} M*={MSCALE} ==", flush=True)
            # Picard
            Xp = Xf.copy()
            for i in range(6):
                Xn = Hmap(Xp); r = la.norm(Xn - Xp) / nX
                print(f"  picard   it{i+1}: res={r:.3e}", flush=True)
                Xp = Xn
            # Anderson(1D Aitken) with damping beta=0.5
            Xp = Xf.copy(); hx, hg = [], []
            for i in range(6):
                Xn = Hmap(Xp); g = Xn - Xp; r = la.norm(g) / nX
                print(f"  anderson it{i+1}: res={r:.3e}", flush=True)
                hx.append(Xp.copy()); hg.append(g.copy())
                if len(hg) >= 2:
                    dG = hg[-1] - hg[-2]; den = float(dG @ dG)
                    th = float(dG @ hg[-1]) / den if den > 0 else 0.0
                    Xp = (1 - th) * (Xp + g) + th * (hx[-2] + hg[-2])
                else:
                    Xp = Xn
            # Newton-GMRES (FD-JVP), 3 outer x (1 res + 3 JVP)
            from scipy.sparse.linalg import LinearOperator, gmres
            Xp = Xf.copy()
            for i in range(3):
                Xn = Hmap(Xp); R0 = Xn - Xp; r = la.norm(R0) / nX
                print(f"  newton   it{i+1}: res={r:.3e}", flush=True)
                eps = 1e-7 * (la.norm(Xp) + 1.0)
                def mv(v, _Xp=Xp, _Xn=Xn, _eps=eps):
                    nv_ = la.norm(v)
                    if nv_ == 0: return v
                    Hv = (Hmap(_Xp + (_eps / nv_) * v) - _Xn) * (nv_ / _eps)
                    return v - Hv
                A = LinearOperator((len(Xp), len(Xp)), matvec=mv)
                dx, _ = gmres(A, R0, rtol=1e-2, maxiter=3, restart=3)
                Xp = Xp + dx
            Xn = Hmap(Xp)
            print(f"  newton   final: res={la.norm(Xn - Xp) / nX:.3e}", flush=True)
            import sys as _s; _s.exit(0)
        if ACCEL != 'none' and prev_b is not None and len(steps) > 1:
            # ---- window fixed point X_pre* = H(X_pre*), H = fluid solve + corrector re-march ----
            def Hmap(Xp):
                o = ms.solve_chain(Xp, wake, Gp, Gp2, first_wake=(iw == 1))
                Fn = Aero.from_out(o)
                cx = dict(mode='linear', Fk=F_cur, Fk1=Fn, Fkm1=None, dFk=None, dFk1=None)
                Xn = march(X, steps[:-1], tf, Fn, cx)
                return Xn, o, Fn, cx
            Xp = Xf.copy()
            if ACCEL == 'anderson':
                # Anderson(m=2) on the window map
                hist_x, hist_g = [], []
                for it_a in range(AITERS):
                    Xn, out, F_new, ctx_c = Hmap(Xp); n_fluid += 1
                    g = Xn - Xp
                    hist_x.append(Xp.copy()); hist_g.append(g.copy())
                    if len(hist_g) >= 2:
                        dG = hist_g[-1] - hist_g[-2]
                        denom = float(dG @ dG)
                        th = float(dG @ hist_g[-1]) / denom if denom > 0 else 0.0
                        Xp = (1 - th) * (Xp + g) + th * (hist_x[-2] + hist_g[-2])
                    else:
                        Xp = Xn
            elif ACCEL == 'newton':
                # Newton-Krylov (FD-JVP, GMRES k small) on R(Xp)=H(Xp)-Xp
                from scipy.sparse.linalg import LinearOperator, gmres
                for it_n in range(AITERS):
                    Xn, out, F_new, ctx_c = Hmap(Xp); n_fluid += 1
                    R = Xn - Xp
                    rn = np.linalg.norm(R) / (np.linalg.norm(Xp) + 1e-30)
                    if rn < 1e-9:
                        break
                    eps = 1e-7 * (np.linalg.norm(Xp) + 1.0)
                    def mv(v, _Xp=Xp, _R=R, _eps=eps):
                        nv_ = np.linalg.norm(v)
                        if nv_ == 0:
                            return v
                        Xv, _, _, _ = Hmap(_Xp + (_eps / nv_) * v)
                        Hv = (Xv - (_Xp + _R)) * (nv_ / _eps)   # H'(Xp)·v approx
                        return v - Hv                            # (I - H')v
                    A = LinearOperator((len(Xp), len(Xp)), matvec=lambda v: mv(v))
                    dx, _ = gmres(A, R, rtol=1e-2, maxiter=3, restart=3)
                    n_fluid += 3
                    Xp = Xp + dx
            # FINAL evaluation at the accelerated X_pre (without this the
            # accelerated iterate's force never reaches the corrector!)
            _, out, F_new, ctx_c = Hmap(Xp); n_fluid += 1
            wake_new = out['wake']
            Xc = march(X, steps, tf, F_new, ctx_c)
            _pen_c = [march(X, steps[:-1], tf, F_new, ctx_c)] if len(steps) > 1 else []
        else:
          for pic in range(n_pic):
            _pen_c = []
            Xc = march(X, steps, tf, F_new, ctx_c, dense_samples=dense_samples, pen_out=_pen_c)
            if pic < n_pic - 1:
                Xc_pre = march(X, steps[:-1], tf, F_new, ctx_c, dense_samples=dense_samples) \
                    if len(steps) > 1 else X.copy()
                out = ms.solve_chain(Xc_pre, wake, Gp, Gp2, first_wake=(iw == 1)); n_fluid += 1
                F_new = Aero.from_out(out)
                wake_new = out['wake']
                ctx_c['Fk1'] = F_new
        V_start = (Xc - _pen_c[0]) / d_t if _pen_c else None
        X = Xc
        iw += 1
        wake = wake_new
        Gp2 = Gp; Gp = out['Gamma']
        F_old = F_cur; F_cur = F_new; F_a = F_new
        tf = b * d_t
        prev_b = b
        rows.append((b, b * d_t, X[zdof], n_fluid))
        print(f"[{MODE} bs={BLOCK_STEPS} ss={SUBSHED} m={MSCALE} pic={PICARD} acc={ACCEL}] "
              f"b={b:4d} t*={b*d_t:.3f} tip={X[zdof]:+.6e} fluids={n_fluid} "
              f"({tmod.time()-t0:.0f}s)", flush=True)
    tag = f"{MODE}_bs{BLOCK_STEPS}_ss{SUBSHED}_m{MSCALE}_p{PICARD}_a{ACCEL}"
    np.save(f'/tmp/pilot_{tag}.npy', np.array(rows))


if __name__ == '__main__':
    run()
