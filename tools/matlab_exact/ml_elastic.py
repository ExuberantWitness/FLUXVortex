"""MATLAB-native elastic forces (membrane + bending + membrane tangent) in numpy,
implemented verbatim from dq_eps_Dm_eps.m / dq_k_Dk_k_FAST.m / generate_stiff_matrices.m,
using the per-element shape-matrix structs stored in the fixture. MATLAB DOF order,
MATLAB nondim units. Eliminates the Python-shell conversion entirely.

Per gauss point (xi_i, eta_j):
  membrane: eps = [.5(q'A1q-1); .5(q'A2q-1); .5 q'A3q],  dq_eps = [q'A1; q'A2; q'A3]
            Qe_gp = dq_eps' D eps
            K_gp  = sum_s (D eps)_s A_s + dq_eps' D dq_eps
  bending:  n = dx_r x dy_r, nh = n/|n|
            kappa = [dx2_r.nh; dy2_r.nh; 2 dxy_r.nh]
            dq_n[:,k] = cross(dxS[:,k], dy_r) + cross(dx_r, dyS[:,k])
            dq_nh = (dq_n - nh (nh.dq_n))/|n|
            dq_kappa = [nh@dx2S + dx2_r@dq_nh; nh@dy2S + dy2_r@dq_nh; 2(nh@dxyS + dxy_r@dq_nh)]
            Qk_gp = dq_kappa' D kappa
  integrate dL*dW/4 * w_i w_j;  globals: Qe *= zeta_m, Qk *= eta_m, K_mem *= zeta_m.
"""
import numpy as np
import scipy.sparse as sp


class MatlabElastic:
    def __init__(self, fixture):
        f = fixture
        self.Ne = int(np.asarray(f['N_element']).ravel()[0])
        self.Nq_all = int(np.asarray(f['N_q_all']).ravel()[0])
        self.D = np.asarray(f['Dp_mat'], dtype=float)            # (3,3)
        self.w = np.asarray(f['w_vec'], dtype=float).ravel()     # (5,)
        self.zeta_m = float(np.asarray(f['zeta_m']).ravel()[0])
        self.eta_m = float(np.asarray(f['eta_m']).ravel()[0])
        self.dL = np.asarray(f['dL_vec'], dtype=float).ravel()
        self.dW = np.asarray(f['dW_vec'], dtype=float).ravel()
        iv = f['i_vec_v']
        iv = iv[0] if iv.ndim == 2 else iv
        self.idof = [np.asarray(iv[e], dtype=int).ravel() - 1 for e in range(self.Ne)]
        st = f['dx_n_Sc_struct']
        st = st[0] if st.ndim == 2 else st
        ng = len(self.w)
        self.ng = ng
        # per-element gauss-point shape matrices, reordered to (ng,ng,3,36)/(ng,ng,36,36)
        def mat4(e, name):
            a = np.ascontiguousarray(st[e][name], dtype=float)   # (3,36,ng,ng)
            return np.transpose(a, (2, 3, 0, 1))
        def mat44(e, name):
            a = np.ascontiguousarray(st[e][name], dtype=float)   # (36,36,ng,ng)
            return np.transpose(a, (2, 3, 0, 1))
        self.dxS=[];  self.dyS=[];  self.dx2S=[]; self.dy2S=[]; self.dxyS=[]
        self.A1=[]; self.A2=[]; self.A3=[]
        for e in range(self.Ne):
            self.dxS.append(mat4(e, 'dx_Sc_mat_v_o'))
            self.dyS.append(mat4(e, 'dy_Sc_mat_v_o'))
            self.dx2S.append(mat4(e, 'dx2_Sc_mat_v_o'))
            self.dy2S.append(mat4(e, 'dy2_Sc_mat_v_o'))
            self.dxyS.append(mat4(e, 'dxy_Sc_mat_v_o'))
            self.A1.append(mat44(e, 'dx_Sc_dx_Sc_mat'))
            self.A2.append(mat44(e, 'dy_Sc_dy_Sc_mat'))
            self.A3.append(mat44(e, 'dxSc_dySc_p_dySc_dxSc_mat'))
        # gauss weight grid (ng,ng)
        self.wgrid = np.outer(self.w, self.w)

    def forces(self, q, want_K=True):
        """Returns (Qe_global, Qk_global, dq_Qe_global_dense_or_None) in MATLAB units."""
        Nq = self.Nq_all
        Qe = np.zeros(Nq); Qk = np.zeros(Nq)
        K = np.zeros((Nq, Nq)) if want_K else None
        D = self.D
        for e in range(self.Ne):
            qe = q[self.idof[e]]                                  # (36,)
            scale = self.dL[e] * self.dW[e] / 4.0
            wg = self.wgrid * scale                               # (ng,ng)
            # ---- membrane ----
            g1 = self.A1[e] @ qe                                  # (ng,ng,36)
            g2 = self.A2[e] @ qe
            g3 = self.A3[e] @ qe
            e1 = 0.5 * (g1 @ qe - 1.0)                            # (ng,ng)
            e2 = 0.5 * (g2 @ qe - 1.0)
            e3 = 0.5 * (g3 @ qe)
            eps = np.stack([e1, e2, e3], axis=-1)                 # (ng,ng,3)
            Deps = eps @ D.T                                      # (ng,ng,3)
            dq_eps = np.stack([g1, g2, g3], axis=-2)              # (ng,ng,3,36)
            Qm = np.einsum('ij,ijsk,ijs->k', wg, dq_eps, Deps)
            Qe[self.idof[e]] += self.zeta_m * Qm
            if want_K:
                # material part: dq_eps' D dq_eps ; geometric: sum_s (D eps)_s A_s
                Kmat = np.einsum('ij,ijsk,st,ijtl->kl', wg, dq_eps, D, dq_eps)
                Kg = (np.einsum('ij,ij,ijkl->kl', wg, Deps[..., 0], self.A1[e])
                      + np.einsum('ij,ij,ijkl->kl', wg, Deps[..., 1], self.A2[e])
                      + np.einsum('ij,ij,ijkl->kl', wg, Deps[..., 2], self.A3[e]))
                K[np.ix_(self.idof[e], self.idof[e])] += self.zeta_m * (Kmat + Kg)
            # ---- bending ----
            dx_r = self.dxS[e] @ qe                               # (ng,ng,3)
            dy_r = self.dyS[e] @ qe
            dx2_r = self.dx2S[e] @ qe
            dy2_r = self.dy2S[e] @ qe
            dxy_r = self.dxyS[e] @ qe
            n = np.cross(dx_r, dy_r)                              # (ng,ng,3)
            nn = np.linalg.norm(n, axis=-1, keepdims=True)        # (ng,ng,1)
            nh = n / nn
            k1 = np.einsum('ijc,ijc->ij', dx2_r, nh)
            k2 = np.einsum('ijc,ijc->ij', dy2_r, nh)
            k3 = 2.0 * np.einsum('ijc,ijc->ij', dxy_r, nh)
            kap = np.stack([k1, k2, k3], axis=-1)                 # (ng,ng,3)
            Dkap = kap @ D.T
            # dq_n[:, :, c, k] = cross(dxS[...,k], dy_r) + cross(dx_r, dyS[...,k])
            dq_n = (np.cross(np.moveaxis(self.dxS[e], -1, 2), dy_r[:, :, None, :])
                    + np.cross(dx_r[:, :, None, :], np.moveaxis(self.dyS[e], -1, 2)))
            dq_n = np.moveaxis(dq_n, 2, 3)                        # (ng,ng,3,36)
            nT_dq_n = np.einsum('ijc,ijck->ijk', nh, dq_n)        # (ng,ng,36)
            dq_nh = (dq_n - nh[..., None] * nT_dq_n[:, :, None, :]) / nn[..., None]
            dk1 = np.einsum('ijc,ijck->ijk', nh, self.dx2S[e]) \
                + np.einsum('ijc,ijck->ijk', dx2_r, dq_nh)
            dk2 = np.einsum('ijc,ijck->ijk', nh, self.dy2S[e]) \
                + np.einsum('ijc,ijck->ijk', dy2_r, dq_nh)
            dk3 = 2.0 * (np.einsum('ijc,ijck->ijk', nh, self.dxyS[e])
                         + np.einsum('ijc,ijck->ijk', dxy_r, dq_nh))
            dq_kap = np.stack([dk1, dk2, dk3], axis=-2)           # (ng,ng,3,36)
            Qb = np.einsum('ij,ijsk,ijs->k', wg, dq_kap, Dkap)
            Qk[self.idof[e]] += self.eta_m * Qb
        return Qe, Qk, K
