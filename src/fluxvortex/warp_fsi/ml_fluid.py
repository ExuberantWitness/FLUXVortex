"""MATLAB-exact GPU fluid solve (solve_fluid.m + generate_wake.m RK4 +
calc_fluid_force_strong.m) — the validated /tmp/ml_fluid_step.py chain with the
O(N^2) induction on device (kernels_ml_exact) and host-side wake bookkeeping.

All quantities in MATLAB nondim units / MATLAB DOF order. Per-env batching via
leading B dim on device arrays; v1 keeps tiny wake node bookkeeping on host
(arrays <=200 nodes; the N-body stage evaluations run on GPU).

Constants come from a fixture .mat (Sc matrices, V_in, var_param, C-tensors for
the rank-1 force assembly).
"""
from __future__ import annotations
import numpy as np
import warp as wp
import scipy.sparse as sp
from . import config
from .kernels_ml_exact import (ml_rcore_kernel, ml_aic_kernel, ml_induce_kernel,
                               ml_dt_induce_kernel, ml_dt_aic_kernel)
from .batched_solver import batched_dense_solve

DTYPE = config.DTYPE
VEC3 = config.VEC3


def _dense(f, k):
    v = f[k]
    return v.toarray() if sp.issparse(v) else np.asarray(v, dtype=float)


class MlGpuFluid:
    """One MATLAB-exact fluid solve on GPU (B envs lockstep)."""

    def __init__(self, fixture_sq, fixture_raw, B, device=None):
        device = device or config.DEVICE
        self.device = device
        self.B = B
        NP = config.NP_DTYPE
        f = fixture_sq
        sq = lambda k: np.asarray(f[k]).squeeze()
        self.Nx = int(sq('Nx')); self.Ny = int(sq('Ny'))
        self.Ne = self.Nx * self.Ny
        self.Nq = int(sq('N_q_all'))
        vp = f['var_param']
        self.Length = float(np.asarray(vp.Length).squeeze())
        self.r_fine = NP(float(np.asarray(vp.r_eps.fine).squeeze()))
        self.r_rough = NP(float(np.asarray(vp.r_eps.rough).squeeze()))
        self.eps_v = NP(float(np.asarray(vp.eps_v).squeeze()))
        self.LNx = NP(self.Length / self.Nx)
        self.dtw = NP(float(sq('d_t_wake')))
        self.U_in = float(sq('U_in'))
        self.V_in = _dense(f, 'V_in')                       # (Ne,3)
        self.Rtrunc = 5.5 * self.Length
        self.Rnochange = self.Rtrunc - 1.5 * self.Length
        # Sc matrices (scipy sparse, host) — geometry matvecs are tiny; host ok v1
        self.Sc_col = f['Sc_mat_col_global'].tocsr()
        self.S31 = f['Sc_mat_31'].tocsr(); self.S24 = f['Sc_mat_24'].tocsr()
        self.Sp = [f[f'Sc_mat_panel_global_{k}'].tocsr() for k in (1, 2, 3, 4)]
        self.Sc_col_d = self.Sc_col.toarray()
        # rank-1 assembly constants (bundled; /tmp fallback for dev)
        import os
        _cpath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'data', 'asm_C_tensors_15x10.npz')
        if not os.path.exists(_cpath):
            _cpath = '/tmp/asm_C_tensors.npz'
        d = np.load(_cpath)
        self.C = d['C']                                     # (Ne,3,36,3)
        self.src = d['src']                                 # (Ne,3)
        fr = fixture_raw
        iv = fr['i_vec_v']; iv = iv[0] if iv.ndim == 2 else iv
        self.idof = np.stack([np.asarray(iv[e], dtype=int).ravel() - 1
                              for e in range(self.Ne)])     # (Ne,36)
        # device scratch
        self._alloc()

    def _alloc(self):
        B, Ne, dev = self.B, self.Ne, self.device
        z = lambda shape, dt: wp.zeros(shape, dtype=dt, device=dev)
        self.colw = z((B, Ne), VEC3); self.nw = z((B, Ne), VEC3)
        self.dtcw = z((B, Ne), VEC3)
        self.bc = [z((B, Ne), VEC3) for _ in range(4)]
        self.bdt = [z((B, Ne), VEC3) for _ in range(4)]
        self.rcore_b = z((B, Ne), DTYPE)
        self.A = z((B, Ne, Ne), DTYPE)
        self.dtA = z((B, Ne, Ne), DTYPE)
        self.Vbuf = z((B, Ne), VEC3)

    # ---------- host<->device helpers (B-broadcast upload) ----------
    def up(self, a, out=None, dt=VEC3):
        NP = config.NP_DTYPE
        arr = np.broadcast_to(a, (self.B,) + a.shape).astype(NP)
        if out is not None and tuple(out.shape) == arr.shape[:out.ndim] and dt == VEC3:
            wp.copy(out, wp.array(np.ascontiguousarray(arr), dtype=dt, device=self.device))
            return out
        return wp.array(np.ascontiguousarray(arr), dtype=dt, device=self.device)

    # ---------- induction wrappers (B lockstep, env0 host return) ----------
    def _rcore(self, corners_dev, S, r_eps):
        rc = wp.zeros((self.B, S), dtype=DTYPE, device=self.device)
        wp.launch(ml_rcore_kernel, dim=(self.B, S),
                  inputs=[corners_dev[0], corners_dev[1], corners_dev[2],
                          corners_dev[3], DTYPE(self.LNx), DTYPE(r_eps)],
                  outputs=[rc], device=self.device)
        return rc

    def induce(self, targets_np, rings_np, gamma_np, fine):
        """N-body ring induction on device. targets (T,3); rings 4x(S,3);
        gamma (S,). Returns (T,3) numpy (env 0)."""
        T = targets_np.shape[0]; S = rings_np[0].shape[0]
        tw = self.up(targets_np)
        cw = [self.up(r) for r in rings_np]
        gw = self.up(gamma_np, dt=DTYPE)
        rc = self._rcore(cw, S, self.r_fine if fine else self.r_rough)
        V = wp.zeros((self.B, T), dtype=VEC3, device=self.device)
        wp.launch(ml_induce_kernel, dim=(self.B, T, S),
                  inputs=[tw, cw[0], cw[1], cw[2], cw[3], gw, rc, DTYPE(self.eps_v)],
                  outputs=[V], device=self.device)
        wp.synchronize()
        return V.numpy()[0]

    # ---------- geometry (host matvecs, tiny) ----------
    def geometry(self, q, dtq):
        bP = [np.asarray(S @ q).reshape(-1, 3) for S in self.Sp]
        bdt = [np.asarray(S @ dtq).reshape(-1, 3) for S in self.Sp]
        rc = np.asarray(self.Sc_col @ q).reshape(-1, 3)
        drc = np.asarray(self.Sc_col @ dtq).reshape(-1, 3)
        r13 = np.asarray(self.S31 @ q).reshape(-1, 3)
        r42 = np.asarray(self.S24 @ q).reshape(-1, 3)
        d13 = np.asarray(self.S31 @ dtq).reshape(-1, 3)
        d42 = np.asarray(self.S24 @ dtq).reshape(-1, 3)
        cr = np.cross(r13, r42)
        nrm = np.linalg.norm(cr, axis=1, keepdims=True)
        nv = cr / nrm
        dtc = (np.cross(d13, r42) + np.cross(r13, d42)) / nrm
        dtn = dtc - nv * np.sum(dtc * nv, axis=1, keepdims=True)
        return bP, bdt, rc, drc, nv, dtn

    # ---------- generate_wake (RK4; stage math on GPU) ----------
    def generate_wake(self, first, bP, dt_bP, Gamma, wake, Gamma_trail):
        Ny = self.Ny; dtw = float(self.dtw)
        i_trail = np.arange(self.Ne - Ny, self.Ne)
        p2_end = bP[1][i_trail]; p3_end = bP[2][i_trail]
        p31_end = p3_end[0:1]
        dt_p2_end = dt_bP[1][i_trail]; dt_p3_end = dt_bP[2][i_trail]
        if first:
            tr = np.vstack([p2_end, p31_end])
            Vg = self.induce(tr, bP, Gamma, fine=False)
            V2 = Vg[:-1] + self.V_in[i_trail] - dt_p2_end
            V31 = Vg[-1:] + self.V_in[:1] - dt_p3_end[0:1]
            r2 = p2_end + V2 * dtw
            r31 = p31_end + V31 * dtw
            r1 = p2_end.copy(); r4 = p3_end.copy()
            r3 = np.vstack([r31, r2[:-1]])
            dt_r3 = np.vstack([V31, V2[:-1]])
            return dict(r1=r1, r2=r2, r3=r3, r4=r4, Gam=Gamma_trail.copy(),
                        dt1=dt_p2_end.copy(), dt2=V2, dt3=dt_r3,
                        dt4=dt_p3_end.copy())
        r1o, r2o, r3o, r4o = wake['r1'], wake['r2'], wake['r3'], wake['r4']
        Gw = wake['Gam']
        Nw = r2o.shape[0]
        old_r2 = np.vstack([p2_end, r2o])
        old_r31 = np.vstack([p31_end, r3o[::Ny]])
        Nwt = Nw + Ny
        dt_p2w = np.zeros((Nwt, 3)); dt_p2w[:Ny] = dt_p2_end
        dt_p31w = np.zeros((Nwt // Ny, 3)); dt_p31w[0] = dt_p3_end[0]
        Vin_w = np.zeros((Nwt, 3)); Vin_w[:, 0] = self.U_in
        cx = (r1o[:, 0] + r4o[:, 0]) / 2.0
        idx_nc = np.where(cx > self.Rnochange)[0]
        if idx_nc.size:
            i0 = (idx_nc[0] // Ny) * Ny + 1
            nc2 = np.arange(i0 - 1, Nwt)
            nc31 = np.arange((idx_nc[0] // Ny), old_r31.shape[0])
        else:
            nc2 = nc31 = None

        def stage_rings(r2s, r31s):
            r3s = np.zeros((Nwt, 3))
            r3s[::Ny] = r31s
            idx = np.arange(Nwt); idx = idx[idx % Ny != 0]
            r3s[idx] = r2s[idx - 1]
            r1s = np.vstack([p2_end, r2s[:-Ny]])
            r4s = np.vstack([p3_end, r3s[:-Ny]])
            return r1s, r3s, r4s

        def stage_vel(r2s, r31s, rings, Gw_s):
            tg = np.vstack([r2s, r31s])
            Vb = self.induce(tg, bP, Gamma, fine=False)
            Vw = self.induce(tg, rings, Gw_s, fine=False)
            V = Vb + Vw
            V2 = V[:Nwt] + Vin_w - dt_p2w
            V31 = V[Nwt:] + Vin_w[:old_r31.shape[0]] - dt_p31w
            return V2, V31

        k1_2, k1_31 = stage_vel(old_r2, old_r31, [r1o, r2o, r3o, r4o], Gw)
        r2_k2 = old_r2 + k1_2 * dtw / 2; r31_k2 = old_r31 + k1_31 * dtw / 2
        Gw2 = np.concatenate([Gamma_trail, Gw])
        r1_k2, r3_k2, r4_k2 = stage_rings(r2_k2, r31_k2)
        k2_2, k2_31 = stage_vel(r2_k2, r31_k2, [r1_k2, r2_k2, r3_k2, r4_k2], Gw2)
        r2_k3 = old_r2 + k2_2 * dtw / 2; r31_k3 = old_r31 + k2_31 * dtw / 2
        r1_k3, r3_k3, r4_k3 = stage_rings(r2_k3, r31_k3)
        k3_2, k3_31 = stage_vel(r2_k3, r31_k3, [r1_k3, r2_k3, r3_k3, r4_k3], Gw2)
        r2_k4 = old_r2 + k3_2 * dtw; r31_k4 = old_r31 + k3_31 * dtw
        r1_k4, r3_k4, r4_k4 = stage_rings(r2_k4, r31_k4)
        k4_2, k4_31 = stage_vel(r2_k4, r31_k4, [r1_k4, r2_k4, r3_k4, r4_k4], Gw2)
        V2 = (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6.0
        V31 = (k1_31 + 2 * k2_31 + 2 * k3_31 + k4_31) / 6.0
        if nc2 is not None:
            V2[nc2] = Vin_w[nc2]
            V31[nc31] = Vin_w[:old_r31.shape[0]][nc31]
        r2n = old_r2 + V2 * dtw; r31n = old_r31 + V31 * dtw
        r1n, r3n, r4n = stage_rings(r2n, r31n)
        dt3 = np.zeros((Nwt, 3)); dt3[::Ny] = V31
        idx = np.arange(Nwt); idx = idx[idx % Ny != 0]
        dt3[idx] = V2[idx - 1]
        dt1 = np.vstack([dt_p2_end, V2[:-Ny]])
        dt4 = np.vstack([dt_p3_end, dt3[:-Ny]])
        out = dict(r1=r1n, r2=r2n, r3=r3n, r4=r4n, Gam=Gw2,
                   dt1=dt1, dt2=V2, dt3=dt3, dt4=dt4)
        cx = (r1n[:, 0] + r4n[:, 0]) / 2.0
        idx_tr = np.where(cx > self.Rtrunc)[0]
        if idx_tr.size:
            i0 = (idx_tr[0] // Ny) * Ny
            for k in ('r1', 'r2', 'r3', 'r4', 'dt1', 'dt2', 'dt3', 'dt4'):
                out[k] = out[k][:i0]
            out['Gam'] = out['Gam'][:i0]
        return out

    # ---------- one fluid solve ----------
    def solve_chain(self, X, wake, Gamma_prev, Gamma_prev2, first_wake=False):
        NP = config.NP_DTYPE
        Ny, Ne, Nq, B = self.Ny, self.Ne, self.Nq, self.B
        q = X[:Nq]; dtq = X[Nq:]
        bP, bdt, rc, drc, nv, dtn = self.geometry(q, dtq)
        # AIC on device
        colw = self.up(rc); nw = self.up(nv)
        cw = [self.up(p) for p in bP]
        rcb = self._rcore(cw, Ne, self.r_fine)
        wp.launch(ml_aic_kernel, dim=(B, Ne, Ne),
                  inputs=[colw, nw, cw[0], cw[1], cw[2], cw[3], rcb,
                          DTYPE(self.eps_v)], outputs=[self.A], device=self.device)
        wp.synchronize()
        A0 = self.A.numpy()[0]
        # wake advect + shed (GPU stage math)
        trail_shed = Gamma_prev2[-Ny:]
        wk = self.generate_wake(first_wake, bP, bdt, Gamma_prev, wake, trail_shed)
        wP = [wk['r1'], wk['r2'], wk['r3'], wk['r4']]
        # RHS with shed trail; force with updated trail
        Vwp_rhs = self.induce(rc, wP, wk['Gam'], fine=True)
        Vn = np.einsum('tc,tc->t', drc - self.V_in - Vwp_rhs, nv)
        Gamma = np.linalg.solve(A0, Vn)
        wk['Gam'] = wk['Gam'].copy(); wk['Gam'][:Ny] = Gamma_prev[-Ny:]
        Vwp = self.induce(rc, wP, wk['Gam'], fine=True)
        Vg = self.induce(rc, bP, Gamma, fine=True)
        V_surf1 = Vg + Vwp + self.V_in
        # dp_lift1/2
        t21 = bP[1] - bP[0]; t34 = bP[2] - bP[3]
        t14 = bP[0] - bP[3]; t23 = bP[1] - bP[2]
        tx = (t21 + t34) / 2; ty = (t14 + t23) / 2
        dx = np.linalg.norm(tx, axis=1, keepdims=True)
        dy = np.linalg.norm(ty, axis=1, keepdims=True)
        tx /= dx; ty /= dy
        Gm = Gamma.reshape(self.Nx, Ny)
        dxm = dx.reshape(self.Nx, Ny); dym = dy.reshape(self.Nx, Ny)
        dxG = np.vstack([Gm[:1], np.diff(Gm, axis=0)]) / dxm
        Gm2 = np.hstack([np.zeros((self.Nx, 1)), Gm, np.zeros((self.Nx, 1))])
        dyG = (Gm2[:, 2:] - Gm2[:, :-2]) / (2 * dym)
        dyG[:, 0] = Gm[:, 0] / dym[:, 0]
        dyG[:, -1] = -Gm[:, -1] / dym[:, -1]
        txdx = tx * dxG.reshape(-1, 1); tydy = ty * dyG.reshape(-1, 1)
        dp_lift1 = np.einsum('tc,tc->t', V_surf1, txdx + tydy)
        dp_lift2 = -(txdx + tydy)
        # Mf2_vec1 (GPU dt-induce over wake)
        Sw = wP[0].shape[0]
        dtcw = self.up(drc)
        cwk = [self.up(r) for r in wP]
        dwk = [self.up(wk[f'dt{k}']) for k in (1, 2, 3, 4)]
        gww = self.up(wk['Gam'], dt=DTYPE)
        Vdt = wp.zeros((B, Ne), dtype=VEC3, device=self.device)
        wp.launch(ml_dt_induce_kernel, dim=(B, Ne, Sw),
                  inputs=[colw, dtcw, cwk[0], cwk[1], cwk[2], cwk[3],
                          dwk[0], dwk[1], dwk[2], dwk[3], gww],
                  outputs=[Vdt], device=self.device)
        wp.synchronize()
        Gw_dt_n = np.einsum('tc,tc->t', Vdt.numpy()[0], nv)
        Mf2_vec1 = np.linalg.solve(A0, -Gw_dt_n)
        # dt_Amat1 (GPU)
        bdtw = [self.up(d) for d in bdt]
        wp.launch(ml_dt_aic_kernel, dim=(B, Ne, Ne),
                  inputs=[colw, dtcw, nw, cw[0], cw[1], cw[2], cw[3],
                          bdtw[0], bdtw[1], bdtw[2], bdtw[3]],
                  outputs=[self.dtA], device=self.device)
        wp.synchronize()
        dt_Amat1 = self.dtA.numpy()[0]
        # Mf1 / Mf2
        nvec_Sc = np.zeros((Ne, Nq))
        for e in range(Ne):
            rows = self.Sc_col_d[3 * e:3 * e + 3][:, self.idof[e]]
            nvec_Sc[e, self.idof[e]] = nv[e] @ rows
        Mf2 = np.linalg.inv(A0)
        Mf1 = np.linalg.solve(A0, nvec_Sc)
        # rank-1 assembly (host v1; device-trivial later)
        scal = dp_lift1 + Mf2_vec1
        Qv = np.zeros(Nq); M0 = np.zeros((Nq, Ne))
        L2 = np.zeros((Nq, 3 * Ne)); Mm = np.zeros((Nq, Nq))
        for e in range(Ne):
            for s in range(3):
                j = self.src[e, s]
                if j < 0:
                    continue
                u = self.C[e, s] @ nv[j]
                idx = self.idof[e]
                Qv[idx] += u * scal[j]
                M0[idx, :] += np.outer(u, Mf2[j])
                L2[idx, 3 * e:3 * e + 3] += np.outer(u, dp_lift2[j])
                Mm[np.ix_(idx, idx)] += np.outer(u, Mf1[j][idx])
        return dict(A=A0, Gamma=Gamma, wake=wk, Vwp=Vwp, dp_lift1=dp_lift1,
                    dp_lift2=dp_lift2, Mf2_vec1=Mf2_vec1, dt_Amat1=dt_Amat1,
                    dt_Amat2_Gamma=Vg, Qf_p=Qv, mat0=M0, lift2=L2, mat=Mm)
