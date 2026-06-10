"""MATLAB-exact GPU chain: two-pass block scheme with GPU fluid (ml_fluid) and
GPU structural Newmark (kernels_ancf + batched_solver), validated against the
CPU closed-book chain (/tmp/chain_full.py) and MATLAB fixtures.

Units/order: fluid side runs in MATLAB nondim units & DOF order (exact);
structure runs on the existing GPU stack in physical units & Python order.
Exact interface conversions: q_py=q_ml[perm], dq_py=dt_q_ml[perm]*10,
F_py=F_ml[perm]*122.5, M_added_py=1.225*mat_ml[perm,perm], t_py=t_ml*0.1.
"""
from __future__ import annotations
import numpy as np
import warp as wp
import scipy.sparse as sp
from . import config
from .kernels_ancf import ANCFConstants, assemble_kmem_blocks, assemble_internal_force_sep
from .batched_solver import gpu_newmark_step
from .kernels_coupling import CSR
from .ml_fluid import MlGpuFluid

DTYPE = config.DTYPE
SCALE_F = 122.5
SCALE_M = 1.225
VL = 10.0          # V/L: dq_py = dt_q_ml * VL; t_ml = t_py * VL


def perm_ml2py(Nx, Ny):
    nn = (Nx + 1) * (Ny + 1)
    p = np.empty(9 * nn, dtype=int)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            kp = j * (Nx + 1) + i
            km = i * (Ny + 1) + j
            for d in range(9):
                p[9 * kp + d] = 9 * km + d
    return p


class AeroML:
    """Fluid-solve outputs (MATLAB units) frozen over a pass."""
    def __init__(self, N, Ne):
        z = np.zeros
        self.Fp = z(N); self.mat = z((N, N)); self.mat0 = z((N, Ne)); self.l2 = z((N, 3 * Ne))
        self.Gamma = z(Ne); self.dA1 = z((Ne, Ne)); self.dA2G = z((Ne, 3)); self.Vwp = z((Ne, 3))

    @staticmethod
    def from_out(out, N, Ne):
        a = AeroML(N, Ne)
        a.Fp = out['Qf_p']; a.mat = out['mat']; a.mat0 = out['mat0']; a.l2 = out['lift2']
        a.Gamma = out['Gamma']; a.dA1 = out['dt_Amat1']; a.dA2G = out['dt_Amat2_Gamma']
        a.Vwp = out['Vwp']
        return a


class MlGpuChain:
    def __init__(self, fixture_sq, fixture_raw, shell, B=1, device=None,
                 cg_tol=1e-12):
        self.device = device or config.DEVICE
        self.B = B
        self.cg_tol = cg_tol
        self.fluid = MlGpuFluid(fixture_sq, fixture_raw, B, device)
        self.N = self.fluid.Nq; self.Ne = self.fluid.Ne
        self.Nx, self.Ny = self.fluid.Nx, self.fluid.Ny
        self.d_t_ml = 0.002; self.dtw = 0.068
        self.dt_py = self.d_t_ml / VL
        self.perm = perm_ml2py(self.Nx, self.Ny)
        self.invp = np.empty_like(self.perm); self.invp[self.perm] = np.arange(self.N)
        self.shell = shell
        self.C = ANCFConstants(shell, device=self.device)
        f = fixture_sq
        self.Qf_time = np.asarray(fixture_raw['Qf_time_global'], dtype=float).ravel()
        self.q_in_norm = lambda t_ml: 0.5 * np.sin(np.pi * t_ml / 0.2) if t_ml < 0.2 else 0.0
        # geometry helpers (ML units; tiny host matvecs)
        self.Sc_col = self.fluid.Sc_col; self.S31 = self.fluid.S31; self.S24 = self.fluid.S24
        self.V_in = self.fluid.V_in

    # ---- ML-unit velocity force (mat0 + lift2 slip terms) at given ML state ----
    def _vel_force_ml(self, dtq_ml, Mat0, L2, wq):
        drc = np.asarray(self.Sc_col @ dtq_ml).reshape(-1, 3)
        # dt_n requires q too — caller supplies via closure (set per step)
        nv, dtn = self._nv_dtn
        slip = (np.einsum('ec,ec->e', drc - self.V_in - wq.Vwp - wq.dA2G, dtn)
                - wq.dA1 @ wq.Gamma)
        return Mat0 @ slip + L2 @ drc.ravel()

    # ---- one structural pass (list of ML step indices) on the GPU stack ----
    def march(self, q_py, dq_py, steps, anc_a, anc_slope, tf_ml, wq):
        """q_py/dq_py: (B,ndof) device arrays. Returns updated device arrays."""
        B, N = self.B, self.N
        NP = config.NP_DTYPE
        dev = self.device
        for it in steps:
            t_ml = it * self.d_t_ml
            beta = (t_ml - tf_ml) / self.dtw
            Fp = anc_a.Fp + anc_slope.Fp * beta
            Mat = anc_a.mat + anc_slope.mat * beta
            Mat0 = anc_a.mat0 + anc_slope.mat0 * beta
            L2 = anc_a.l2 + anc_slope.l2 * beta
            # ML state of env0 (lockstep) for dt_n at q_n
            q0 = q_py.numpy()[0][self.invp]
            dq0 = dq_py.numpy()[0][self.invp] / VL
            r13 = np.asarray(self.S31 @ q0).reshape(-1, 3)
            r42 = np.asarray(self.S24 @ q0).reshape(-1, 3)
            d13 = np.asarray(self.S31 @ dq0).reshape(-1, 3)
            d42 = np.asarray(self.S24 @ dq0).reshape(-1, 3)
            cr = np.cross(r13, r42); nrm = np.linalg.norm(cr, axis=1, keepdims=True)
            nv = cr / nrm
            dtc = (np.cross(d13, r42) + np.cross(r13, d42)) / nrm
            self._nv_dtn = (nv, dtc - nv * np.sum(dtc * nv, axis=1, keepdims=True))
            # constant force (pulse + interpolated Bernoulli) -> physical
            pulse_ml = self.Qf_time * self.q_in_norm(t_ml)
            Fc_ml = pulse_ml + Fp
            Fc_py = (Fc_ml[self.perm] * SCALE_F).astype(NP)
            Fc = wp.array(np.broadcast_to(Fc_py, (B, N)).copy(), dtype=DTYPE, device=dev)
            # velocity force callbacks (ML compute, physical output)
            Fv_ml_n = self._vel_force_ml(dq0, Mat0, L2, wq)
            Fv_py_n = (Fv_ml_n[self.perm] * SCALE_F).astype(NP)
            Fvel_n = wp.array(np.broadcast_to(Fv_py_n, (B, N)).copy(), dtype=DTYPE, device=dev)

            def rfvel(qp1, dqp1):
                dq1 = dqp1.numpy()[0][self.invp] / VL
                q1 = qp1.numpy()[0][self.invp]
                r13b = np.asarray(self.S31 @ q1).reshape(-1, 3)
                r42b = np.asarray(self.S24 @ q1).reshape(-1, 3)
                d13b = np.asarray(self.S31 @ dq1).reshape(-1, 3)
                d42b = np.asarray(self.S24 @ dq1).reshape(-1, 3)
                crb = np.cross(r13b, r42b); nrmb = np.linalg.norm(crb, axis=1, keepdims=True)
                nvb = crb / nrmb
                dtcb = (np.cross(d13b, r42b) + np.cross(r13b, d42b)) / nrmb
                dtnb = dtcb - nvb * np.sum(dtcb * nvb, axis=1, keepdims=True)
                drcb = np.asarray(self.Sc_col @ dq1).reshape(-1, 3)
                slip = (np.einsum('ec,ec->e', drcb - self.V_in - wq.Vwp - wq.dA2G, dtnb)
                        - wq.dA1 @ wq.Gamma)
                Fml = Mat0 @ slip + L2 @ drcb.ravel()
                Fpy = (Fml[self.perm] * SCALE_F).astype(NP)
                return wp.array(np.broadcast_to(Fpy, (B, N)).copy(), dtype=DTYPE, device=dev)

            # time-interpolated added mass (ML->physical CSR)
            Mat_py = sp.csr_matrix(
                (Mat[np.ix_(self.perm, self.perm)] * SCALE_M))
            madd = CSR(Mat_py, self.device)
            madd_diag = wp.array(np.asarray(Mat_py.diagonal()).astype(NP),
                                 dtype=DTYPE, device=dev)
            Kblk = assemble_kmem_blocks(q_py, self.C, dev)
            Qmem, Qbend = assemble_internal_force_sep(q_py, self.C, dev)

            def recompute_bend(qp1):
                return assemble_internal_force_sep(qp1, self.C, dev)[1]

            q_py, dq_py = gpu_newmark_step(
                q_py, dq_py, Kblk, self.C.Me, self.C.edofs, self.C.free, N,
                Fc, Qmem, Qbend, Fvel_n, recompute_bend, rfvel,
                alpha_v=0.5, c_damp=2.0, dt=self.dt_py, cg_tol=self.cg_tol,
                device=dev, madd=madd, madd_diag=madd_diag)
        return q_py, dq_py

    # ---- full chain ----
    def run(self, X0_ml, n_blocks, on_boundary=None):
        B, N, Ne = self.B, self.N, self.Ne
        NP = config.NP_DTYPE
        dev = self.device
        q0 = X0_ml[:N][self.perm]
        dq0 = X0_ml[N:][self.perm] * VL
        q = wp.array(np.broadcast_to(q0, (B, N)).astype(NP).copy(), dtype=DTYPE, device=dev)
        dq = wp.array(np.broadcast_to(dq0, (B, N)).astype(NP).copy(), dtype=DTYPE, device=dev)
        F_old = AeroML(N, Ne); F_cur = AeroML(N, Ne); F_a = AeroML(N, Ne)
        tf = 0.0
        wake = None
        Gp = np.zeros(Ne); Gp2 = np.zeros(Ne)
        iw = 1
        prev_b = None
        boundaries = [1 + 34 * k for k in range(n_blocks + 1)]
        for b in boundaries:
            steps = [b] if prev_b is None else list(range(prev_b + 1, b + 1))
            slope_p = AeroML(N, Ne)
            for fkey in ('Fp', 'mat', 'mat0', 'l2'):
                setattr(slope_p, fkey, getattr(F_cur, fkey) - getattr(F_old, fkey))
            # snapshot for rewind
            q_snap = wp.clone(q); dq_snap = wp.clone(dq)
            # PREDICTOR
            if len(steps) > 1:
                qf, dqf = self.march(q, dq, steps[:-1], F_a, slope_p, tf, F_cur)
            else:
                qf, dqf = q, dq
            # fluid at predictor pre-boundary state (ML units)
            X_ml = np.empty(2 * N)
            X_ml[:N] = qf.numpy()[0][self.invp]
            X_ml[N:] = dqf.numpy()[0][self.invp] / VL
            out = self.fluid.solve_chain(X_ml, wake, Gp, Gp2, first_wake=(iw == 1))
            iw += 1
            wake = out['wake']
            F_new = AeroML.from_out(out, N, Ne)
            # CORRECTOR from snapshot
            slope_c = AeroML(N, Ne)
            for fkey in ('Fp', 'mat', 'mat0', 'l2'):
                setattr(slope_c, fkey, getattr(F_new, fkey) - getattr(F_cur, fkey))
            q, dq = self.march(q_snap, dq_snap, steps, F_a, slope_c, tf, F_new)
            # anchors
            Gp2 = Gp; Gp = out['Gamma']
            F_old = F_cur; F_cur = F_new; F_a = F_new
            tf = b * self.d_t_ml
            prev_b = b
            if on_boundary is not None:
                X_out = np.empty(2 * N)
                X_out[:N] = q.numpy()[0][self.invp]
                X_out[N:] = dq.numpy()[0][self.invp] / VL
                on_boundary(b, X_out)
        return q, dq
