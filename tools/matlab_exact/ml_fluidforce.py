"""MATLAB-native fluid force assembly (calc_fluid_force_strong.m verbatim).

Inputs (panel level, MATLAB element order i-outer/j-inner, Ny inner):
  dp_lift1 (Ne,), Mf2_vec1 (Ne,), dp_lift2 (Ne,3), Mf2_mat (Ne,Ne),
  Mf1_mat (Ne, Nq_all), n_vec_i (Ne,3)
Outputs:
  Qf_p_global (Nq,), Qf_p_mat0_global (Nq,Ne), Qf_p_lift2_mat_global (Nq,3Ne),
  Qf_p_mat_global (Nq,Nq dense)
Pressure chordwise interpolation: p_interp.m (piecewise linear between vortex
stations, 3/4-chord break, TE pressure -> 0), Gauss 5x5 per element.
"""
import numpy as np


def p_interp(x, ii_c, dL_f, Nx):
    """p_interp.m verbatim. x: gauss abscissa in [0,dL]; ii_c: 1-based chordwise
    station; dL_f: (Nx,) chordwise element lengths. Returns (3,) stencil weights
    (prev, self, next)."""
    H = lambda v: 1.0 if v >= 0 else 0.0
    i = ii_c
    if 1 < i < Nx:
        w1 = (3*dL_f[i-1] - 4*x)/(3*dL_f[i-1] + dL_f[i-2])*(H(x) - H(x - 0.75*dL_f[i-1]))
        w2 = ((dL_f[i-2] + 4*x)/(3*dL_f[i-1] + dL_f[i-2])*(H(x) - H(x - 0.75*dL_f[i-1]))
              + (3*dL_f[i] - 4*(x - dL_f[i]))/(3*dL_f[i] + dL_f[i-1])
              * (H(x - 0.75*dL_f[i-1]) - H(x - dL_f[i-1])))
        w3 = ((dL_f[i-1] + 4*(x - dL_f[i]))/(3*dL_f[i] + dL_f[i-1])
              * (H(x - 0.75*dL_f[i-1]) - H(x - dL_f[i-1])))
    elif i == 1:
        w1 = 0.0
        w2 = ((H(x) - H(x - 0.75*dL_f[0]))
              + (3*dL_f[1] - 4*(x - dL_f[1]))/(3*dL_f[1] + dL_f[0])
              * (H(x - 0.75*dL_f[0]) - H(x - dL_f[0])))
        w3 = ((dL_f[0] + 4*(x - dL_f[1]))/(3*dL_f[1] + dL_f[0])
              * (H(x - 0.75*dL_f[0]) - H(x - dL_f[0])))
    else:  # i == Nx, TE pressure -> 0
        w1 = (3*dL_f[i-1] - 4*x)/(3*dL_f[i-1] + dL_f[i-2])*(H(x) - H(x - 0.75*dL_f[i-1]))
        w2 = ((dL_f[i-2] + 4*x)/(3*dL_f[i-1] + dL_f[i-2])*(H(x) - H(x - 0.75*dL_f[i-1]))
              + (4*dL_f[i-1] - 4*x)/dL_f[i-1]*(H(x - 0.75*dL_f[i-1]) - H(x - dL_f[i-1])))
        w3 = 0.0
    return np.array([w1, w2, w3])


class MatlabFluidForce:
    def __init__(self, fixture):
        f = fixture
        self.Ne = int(np.asarray(f['N_element']).ravel()[0])
        self.Nq = int(np.asarray(f['N_q_all']).ravel()[0])
        self.Nx = int(np.asarray(f['Nx']).ravel()[0])
        self.Ny = int(np.asarray(f['Ny']).ravel()[0])
        self.dL = np.asarray(f['dL_vec'], dtype=float).ravel()
        self.dW = np.asarray(f['dW_vec'], dtype=float).ravel()
        self.w = np.asarray(f['w_vec'], dtype=float).ravel()
        self.p = np.asarray(f['p_vec'], dtype=float).ravel()
        # Sc_mat_v (3,36,ng,ng,Ne) -> (Ne,ng,ng,3,36)
        self.Sc = np.transpose(np.ascontiguousarray(f['Sc_mat_v'], dtype=float),
                               (4, 2, 3, 0, 1))
        iv = f['i_vec_v']; iv = iv[0] if iv.ndim == 2 else iv
        self.idof = [np.asarray(iv[e], dtype=int).ravel() - 1 for e in range(self.Ne)]
        self.dL_f = self.dL[::self.Ny]          # chordwise station lengths (Nx,)
        ng = len(self.w); self.ng = ng
        # precompute stencil weights per element & gauss-xi: (Ne, ng, 3)
        self.pw = np.zeros((self.Ne, ng, 3))
        for e in range(self.Ne):
            ii_c = e // self.Ny + 1            # 1-based chordwise station
            for a in range(ng):
                x_i = self.dL[e]*(self.p[a] + 1.0)/2.0
                self.pw[e, a] = p_interp(x_i, ii_c, self.dL_f, self.Nx)
        self.wgrid = np.outer(self.w, self.w)   # (ng, ng)

    def _stencil(self, nodal):
        """nodal: (Ne, ...) per-element nodal values -> (Ne, 3, ...) chordwise
        (prev, self, next) stencil with zero ends (MATLAB dp_nvec_i pattern)."""
        Ny = self.Ny
        s = np.zeros((self.Ne, 3) + nodal.shape[1:], dtype=nodal.dtype)
        s[:, 1] = nodal
        s[Ny:, 0] = nodal[:-Ny]
        s[:-Ny, 2] = nodal[Ny:]
        return s

    def assemble(self, dp_lift1, Mf2_vec1, dp_lift2, Mf2_mat, Mf1_mat, n_vec_i):
        Ne, Nq, ng = self.Ne, self.Nq, self.ng
        # nodal pressure stencils
        sc_vec = self._stencil(((dp_lift1 + Mf2_vec1)[:, None] * n_vec_i))      # (Ne,3,3)
        sc_l2 = self._stencil(np.einsum('ec,ed->ecd', n_vec_i, dp_lift2))       # (Ne,3,3,3) n⊗dp2
        sc_m0 = self._stencil(np.einsum('ec,ej->ecj', n_vec_i, Mf2_mat))        # (Ne,3,3,Ne)
        sc_m1 = self._stencil(np.einsum('ec,ej->ecj', n_vec_i, Mf1_mat))        # (Ne,3,3,Nq)
        _dt = np.result_type(dp_lift1, Mf2_mat, n_vec_i)
        Qv = np.zeros(Nq, dtype=_dt)
        M0 = np.zeros((Nq, Ne), dtype=_dt)
        L2 = np.zeros((Nq, 3*Ne), dtype=_dt)
        Mm = np.zeros((Nq, Nq), dtype=_dt)
        for e in range(Ne):
            scale = self.dL[e]*self.dW[e]/4.0
            # interp over chord per gauss-xi: weights pw (ng,3)
            v_i = np.einsum('as,sc->ac', self.pw[e], sc_vec[e])                 # (ng,3)
            l2_i = np.einsum('as,scd->acd', self.pw[e], sc_l2[e])               # (ng,3,3)
            m0_i = np.einsum('as,scj->acj', self.pw[e], sc_m0[e])               # (ng,3,Ne)
            m1_i = np.einsum('as,scj->acj', self.pw[e], sc_m1[e][:, :, self.idof[e]])  # (ng,3,36)
            # Gauss: sum_a sum_b w_a w_b Sc[a,b].T @ (.)
            Sc = self.Sc[e]                                                     # (ng,ng,3,36)
            Wb = self.wgrid * scale                                             # (ng,ng)
            Qv[self.idof[e]] += np.einsum('ab,abck,ac->k', Wb, Sc, v_i)
            L2e = np.einsum('ab,abck,acd->kd', Wb, Sc, l2_i)                    # (36,3)
            M0e = np.einsum('ab,abck,acj->kj', Wb, Sc, m0_i)                    # (36,Ne)
            Mme = np.einsum('ab,abck,acj->kj', Wb, Sc, m1_i)                    # (36,36)
            L2[self.idof[e], 3*e:3*e+3] += L2e
            M0[np.ix_(self.idof[e], range(Ne))] += M0e
            Mm[np.ix_(self.idof[e], self.idof[e])] += Mme
        return Qv, M0, L2, Mm
