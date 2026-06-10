"""Closed-book MATLAB fluid step: solve_fluid.m + generate_wake.m (RK4) +
calc_fluid_force.m/_strong.m, verbatim in numpy, MATLAB units/order.

One call = one fluid solve at a block boundary:
  inputs : X_vec (3168,), wake rings (r_wake_1..4, Gamma_wake) from previous
           step, old_Gamma (previous solve's bound circulation)
  outputs: Gamma, four global force matrices, updated wake state, and all
           intermediate panel-level quantities (for fixture validation).
"""
import numpy as np
from ml_uvlm import q1234_mat, aic_from_q1234
from ml_fluidforce import MatlabFluidForce

MEPS = np.finfo(float).eps


def dt_q1234_mat(rc, P, dt_rc, dtP):
    """dt_generate_q1234_mat.m verbatim: analytic d/dt of the UNregularized
    q1234 (no core, no eps_v). P/dtP: lists of 4 corner arrays (Ns,3)."""
    pairs = [(0, 3), (1, 0), (2, 1), (3, 2)]
    Nt = rc.shape[0]
    Ns = P[0].shape[0]
    out = np.zeros((Nt, Ns, 3))
    R = rc[:, None, :]; dR = dt_rc[:, None, :]
    for a, b in pairs:
        r1 = R - P[a][None]; r2 = R - P[b][None]
        d1 = dR - dtP[a][None]; d2 = dR - dtP[b][None]
        r0 = r1 - r2; d0 = d1 - d2
        cr = np.cross(r1, r2)
        ncr2 = np.einsum('tsc,tsc->ts', cr, cr)
        dcr = np.cross(d1, r2) + np.cross(r1, d2)
        term1 = (dcr / ncr2[..., None]
                 - 2.0 * cr * np.einsum('tsc,tsc->ts', dcr, cr)[..., None] / (ncr2**2)[..., None])
        n1 = np.linalg.norm(r1, axis=-1)[..., None]
        n2 = np.linalg.norm(r2, axis=-1)[..., None]
        u = r1 / n1 - r2 / n2
        cr_n2 = cr / ncr2[..., None]
        du = (d1 / n1 - r1 * np.einsum('tsc,tsc->ts', d1, r1)[..., None] / n1**3
              - d2 / n2 + r2 * np.einsum('tsc,tsc->ts', d2, r2)[..., None] / n2**3)
        out += (term1 * np.einsum('tsc,tsc->ts', r0, u)[..., None]
                + cr_n2 * np.einsum('tsc,tsc->ts', d0, u)[..., None]
                + cr_n2 * np.einsum('tsc,tsc->ts', r0, du)[..., None])
    return -out / (4.0 * np.pi)


class MatlabFluidStep:
    def __init__(self, fixture):
        f = fixture
        sq = lambda k: np.asarray(f[k]).squeeze()
        self.Nx = int(sq('Nx')); self.Ny = int(sq('Ny'))
        self.Ne = self.Nx * self.Ny
        self.Nq = int(sq('N_q_all'))
        vp = f['var_param']
        # squeeze_me=True gives mat_struct
        self.Length = float(np.asarray(vp.Length).squeeze())
        self.r_eps_fine = float(np.asarray(vp.r_eps.fine).squeeze())
        self.r_eps_rough = float(np.asarray(vp.r_eps.rough).squeeze())
        self.Ncore = int(np.asarray(vp.Ncore).squeeze())
        self.eps_v = float(np.asarray(vp.eps_v).squeeze())
        self.d_t_wake = float(sq('d_t_wake'))
        self.U_in = float(sq('U_in')) if 'U_in' in f else 1.0
        self.V_in = np.asarray(f['V_in'], dtype=float)            # (Ne,3)
        self.Rtrunc = 5.5 * self.Length
        self.Rnochange = self.Rtrunc - 1.5 * self.Length
        import scipy.sparse as sp
        g = lambda k: (f[k].toarray() if sp.issparse(f[k]) else np.asarray(f[k], dtype=float))
        self.Sc_col = f['Sc_mat_col_global']
        self.S31 = f['Sc_mat_31']; self.S24 = f['Sc_mat_24']
        self.Sp = [f[f'Sc_mat_panel_global_{k}'] for k in (1, 2, 3, 4)]
        self.asm = MatlabFluidForce(f)
        self.idof = self.asm.idof
        # nvec_Sc rows are built from Sc_col rows; keep dense Sc_col for that
        self.Sc_col_d = g('Sc_mat_col_global')

    # ---- geometry helpers ----
    def panels(self, q):
        return [np.asarray(S @ q).reshape(-1, 3) for S in self.Sp]

    def colloc(self, v):
        return np.asarray(self.Sc_col @ v).reshape(-1, 3)

    def normals(self, q, dtq):
        r13 = np.asarray(self.S31 @ q).reshape(-1, 3)
        r42 = np.asarray(self.S24 @ q).reshape(-1, 3)
        d13 = np.asarray(self.S31 @ dtq).reshape(-1, 3)
        d42 = np.asarray(self.S24 @ dtq).reshape(-1, 3)
        cr = np.cross(r13, r42)
        nrm = np.linalg.norm(cr, axis=1, keepdims=True)
        nv = cr / nrm
        dtc = (np.cross(d13, r42) + np.cross(r13, d42)) / nrm
        dtn = dtc - nv * np.sum(dtc * nv, axis=1, keepdims=True)
        return nv, dtn

    def q1234(self, rc, P, fine):
        eps = self.r_eps_fine if fine else self.r_eps_rough
        return q1234_mat(rc, P[0], P[1], P[2], P[3], self.Length, self.Nx,
                         eps, self.Ncore, self.eps_v)

    def vwake(self, rc, P, Gam, fine):
        V = self.q1234(rc, P, fine)
        return np.einsum('tsc,s->tc', V, Gam), V

    # ---- generate_wake.m (RK4, frozen-base stages) ----
    def generate_wake(self, first, bP, dt_bP, Gamma, wake, Gamma_trail):
        """bP/dt_bP: bound panel corners/velocities (4 lists). wake = dict with
        r1..r4 (Nw,3), Gam (Nw,) or None at first shed. Returns new wake dict
        incl. dt_r_wake corner velocities."""
        Ny = self.Ny; dtw = self.d_t_wake
        i_trail = np.arange(self.Ne - Ny, self.Ne)
        p2_end = bP[1][i_trail]; p3_end = bP[2][i_trail]
        p31_end = p3_end[0:1]
        dt_p2_end = dt_bP[1][i_trail]; dt_p3_end = dt_bP[2][i_trail]
        if first:
            tr = np.vstack([p2_end, p31_end])
            Vg, _ = self.vwake(tr, bP, Gamma, fine=False)
            V2 = Vg[:-1] + self.V_in[i_trail] - dt_p2_end
            V31 = Vg[-1:] + self.V_in[:1] - dt_p3_end[0:1]
            r2 = p2_end + V2 * dtw
            r31 = p31_end + V31 * dtw
            r1 = p2_end.copy(); r4 = p3_end.copy()
            r3 = np.vstack([r31, r2[:-1]])
            dt_r2 = V2; dt_r31 = V31
            dt_r3 = np.vstack([dt_r31, dt_r2[:-1]])
            dt_r1 = dt_p2_end.copy(); dt_r4 = dt_p3_end.copy()
            return dict(r1=r1, r2=r2, r3=r3, r4=r4, Gam=Gamma_trail.copy(),
                        dt1=dt_r1, dt2=dt_r2, dt3=dt_r3, dt4=dt_r4)
        # ---- RK4 ----
        r1o, r2o, r3o, r4o = wake['r1'], wake['r2'], wake['r3'], wake['r4']
        Gw = wake['Gam']
        Nw = r2o.shape[0]
        old_r2 = np.vstack([p2_end, r2o])                       # (Nw+Ny,3)
        old_r31 = np.vstack([p31_end, r3o[::Ny]])               # (Nw/Ny+1,3)
        Nwt = Nw + Ny
        dt_p2w = np.zeros((Nwt, 3)); dt_p2w[:Ny] = dt_p2_end
        dt_p31w = np.zeros((Nwt // Ny, 3)); dt_p31w[0] = dt_p3_end[0]
        Vin_w = np.zeros((Nwt, 3)); Vin_w[:, 0] = self.U_in
        # no-change mask (computed from PRE-advect ring x positions)
        cx = (r1o[:, 0] + r4o[:, 0]) / 2.0
        idx_nc = np.where(cx > self.Rnochange)[0]
        if idx_nc.size:
            i0 = (idx_nc[0] // Ny) * Ny + 1          # MATLAB 1-based row start
            nc2 = np.arange(i0 - 1, Nwt)             # rings >= i0 (0-based)
            nc31 = np.arange((idx_nc[0] // Ny) + 1 - 1, old_r31.shape[0])
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
            Vb, _ = self.vwake(tg, bP, Gamma, fine=False)
            Vw, _ = self.vwake(tg, rings, Gw_s, fine=False)
            V = Vb + Vw
            V2 = V[:Nwt] + Vin_w - dt_p2w
            V31 = V[Nwt:] + Vin_w[:old_r31.shape[0]] - dt_p31w
            return V2, V31

        # k1: OLD rings (without new trail), Gw
        k1_2, k1_31 = stage_vel(old_r2, old_r31, [r1o, r2o, r3o, r4o], Gw)
        # k2 positions
        r2_k2 = old_r2 + k1_2 * dtw / 2; r31_k2 = old_r31 + k1_31 * dtw / 2
        # prepend trail circulation BETWEEN k1 and k2 (MATLAB line 137)
        Gw2 = np.concatenate([Gamma_trail, Gw])
        r1_k2, r3_k2, r4_k2 = stage_rings(r2_k2, r31_k2)
        k2_2, k2_31 = stage_vel(r2_k2, r31_k2, [r1_k2, r2_k2, r3_k2, r4_k2], Gw2)
        r2_k3 = old_r2 + k2_2 * dtw / 2; r31_k3 = old_r31 + k2_31 * dtw / 2
        r1_k3, r3_k3, r4_k3 = stage_rings(r2_k3, r31_k3)
        k3_2, k3_31 = stage_vel(r2_k3, r31_k3, [r1_k3, r2_k3, r3_k3, r4_k3], Gw2)
        r2_k4 = old_r2 + k3_2 * dtw; r31_k4 = old_r31 + k3_31 * dtw
        r1_k4, r3_k4, r4_k4 = stage_rings(r2_k4, r31_k4)
        k4_2, k4_31 = stage_vel(r2_k4, r31_k4, [r1_k4, r2_k4, r3_k4, r4_k4], Gw2)
        V2 = (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6.0
        V31 = (k1_31 + 2*k2_31 + 2*k3_31 + k4_31) / 6.0
        if nc2 is not None:
            V2[nc2] = Vin_w[nc2]
            V31[nc31] = Vin_w[:old_r31.shape[0]][nc31]
        r2n = old_r2 + V2 * dtw; r31n = old_r31 + V31 * dtw
        r1n, r3n, r4n = stage_rings(r2n, r31n)
        dt2 = V2; dt31 = V31
        dt3 = np.zeros((Nwt, 3)); dt3[::Ny] = dt31
        idx = np.arange(Nwt); idx = idx[idx % Ny != 0]
        dt3[idx] = dt2[idx - 1]
        dt1 = np.vstack([dt_p2_end, dt2[:-Ny]])
        dt4 = np.vstack([dt_p3_end, dt3[:-Ny]])
        out = dict(r1=r1n, r2=r2n, r3=r3n, r4=r4n, Gam=Gw2,
                   dt1=dt1, dt2=dt2, dt3=dt3, dt4=dt4)
        # truncation (row-wise, by ring centroid x of r1/r4)
        cx = (r1n[:, 0] + r4n[:, 0]) / 2.0
        idx_tr = np.where(cx > self.Rtrunc)[0]
        if idx_tr.size:
            i0 = (idx_tr[0] // Ny) * Ny
            for k in ('r1', 'r2', 'r3', 'r4', 'dt1', 'dt2', 'dt3', 'dt4'):
                out[k] = out[k][:i0]
            out['Gam'] = out['Gam'][:i0]
        return out

    # ---- one full fluid solve, validated trail semantics ----
    def solve_chain(self, X, wake, Gamma_prev, Gamma_prev2, first_wake=False):
        """Gamma_prev  = G_k   (bound advection source; post-solve trail update)
           Gamma_prev2 = G_{k-1} (shed-prepend trail, delayed Kutta)"""
        Nq = self.Nq; Ny = self.Ny; Ne = self.Ne
        q = X[:Nq]; dtq = X[Nq:]
        bP = self.panels(q)
        dt_bP = [np.asarray(S @ dtq).reshape(-1, 3) for S in self.Sp]
        rc = self.colloc(q); dt_rc = self.colloc(dtq)
        nv, dtn = self.normals(q, dtq)
        Vq = self.q1234(rc, bP, fine=True)
        A = aic_from_q1234(Vq, nv)
        trail_shed = Gamma_prev2[-Ny:]
        wk = self.generate_wake(first_wake, bP, dt_bP, Gamma_prev, wake, trail_shed)
        wP = [wk['r1'], wk['r2'], wk['r3'], wk['r4']]
        # RHS: wake circulation as built (trail = G_{k-1} TE)
        Vwp_rhs, q_wake = self.vwake(rc, wP, wk['Gam'], fine=True)
        Vn = np.einsum('tc,tc->t', dt_rc - self.V_in - Vwp_rhs, nv)
        Gamma = np.linalg.solve(A, Vn)
        # post-solve trail update -> force-side wake velocity
        wk['Gam'] = wk['Gam'].copy()
        wk['Gam'][:Ny] = Gamma_prev[-Ny:]
        Vwp = np.einsum('tsc,s->tc', q_wake, wk['Gam'])
        Vg = np.einsum('tsc,s->tc', Vq, Gamma)
        V_surf1 = Vg + Vwp + self.V_in
        # dp_lift1 / dp_lift2
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
        # Mf2_vec1
        dtwP = [wk['dt1'], wk['dt2'], wk['dt3'], wk['dt4']]
        dtq_w = dt_q1234_mat(rc, wP, dt_rc, dtwP)
        Gw_dt_n = np.einsum('tc,tc->t',
                            np.einsum('tsc,s->tc', dtq_w, wk['Gam']), nv)
        Mf2_vec1 = np.linalg.solve(A, -Gw_dt_n)
        # dt_Amat1 / dt_Amat2_Gamma
        dtq_b = dt_q1234_mat(rc, bP, dt_rc, dt_bP)
        dt_Amat1 = np.einsum('tsc,tc->ts', dtq_b, nv)
        # Mf1 / Mf2
        nvec_Sc = np.zeros((Ne, Nq))
        for e in range(Ne):
            rows = self.Sc_col_d[3*e:3*e+3][:, self.idof[e]]
            nvec_Sc[e, self.idof[e]] = nv[e] @ rows
        Mf1 = np.linalg.solve(A, nvec_Sc)
        Mf2 = np.linalg.inv(A)
        Qv, M0, L2, Mm = self.asm.assemble(dp_lift1, Mf2_vec1, dp_lift2, Mf2, Mf1, nv)
        return dict(A=A, Gamma=Gamma, wake=wk, Vwp=Vwp, dp_lift1=dp_lift1,
                    dp_lift2=dp_lift2, Mf2_vec1=Mf2_vec1, dt_Amat1=dt_Amat1,
                    dt_Amat2_Gamma=Vg, Qf_p=Qv, mat0=M0, lift2=L2, mat=Mm)

    # ---- one full fluid solve ----
    def solve(self, X, wake, old_Gamma, first_wake=False):
        Nq = self.Nq; Ny = self.Ny; Ne = self.Ne
        q = X[:Nq]; dtq = X[Nq:]
        bP = self.panels(q)
        dt_bP = [np.asarray(S @ dtq).reshape(-1, 3) for S in self.Sp]
        rc = self.colloc(q); dt_rc = self.colloc(dtq)
        nv, dtn = self.normals(q, dtq)
        # AIC
        Vq = self.q1234(rc, bP, fine=True)
        A = aic_from_q1234(Vq, nv)
        # wake advect + shed (Gamma = previous solve's = old_Gamma here per
        # exe flow: solve_fluid's workspace Gamma at generate_wake time)
        Gamma_trail = old_Gamma[-Ny:]
        wk = self.generate_wake(first_wake, bP, dt_bP, old_Gamma, wake, Gamma_trail)
        wP = [wk['r1'], wk['r2'], wk['r3'], wk['r4']]
        # wake influence (fine) and RHS
        Vwp, q_wake = self.vwake(rc, wP, wk['Gam'], fine=True)
        Vn = np.einsum('tc,tc->t', dt_rc - self.V_in - Vwp, nv)
        Gamma = np.linalg.solve(A, Vn)
        # trail circulation update for forces
        wk['Gam'] = wk['Gam'].copy()
        wk['Gam'][:Ny] = Gamma_trail
        Vwp = np.einsum('tsc,s->tc', q_wake, wk['Gam'])
        Vg = np.einsum('tsc,s->tc', Vq, Gamma)
        V_surf1 = Vg + Vwp + self.V_in
        # dp_lift1 / dp_lift2
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
        dp_lift2 = -(txdx + tydy)                                  # (Ne,3)
        # Mf2_vec1 (wake dt term)
        dtwP = [wk['dt1'], wk['dt2'], wk['dt3'], wk['dt4']]
        dtq_w = dt_q1234_mat(rc, wP, dt_rc, dtwP)
        Gw_dt_n = np.einsum('tc,tc->t',
                            np.einsum('tsc,s->tc', dtq_w, wk['Gam']), nv)
        Mf2_vec1 = np.linalg.solve(A, -Gw_dt_n)
        # dt_Amat1, dt_Amat2_Gamma
        dtq_b = dt_q1234_mat(rc, bP, dt_rc, dt_bP)
        dt_Amat1 = np.einsum('tsc,tc->ts', dtq_b, nv)
        dt_Amat2_Gamma = Vg.copy()
        # Mf1 / Mf2
        nvec_Sc = np.zeros((Ne, Nq))
        for e in range(Ne):
            rows = self.Sc_col_d[3*e:3*e+3][:, self.idof[e]]
            nvec_Sc[e, self.idof[e]] = nv[e] @ rows
        Mf1 = np.linalg.solve(A, nvec_Sc)
        Mf2 = np.linalg.inv(A)
        # assembly
        Qv, M0, L2, Mm = self.asm.assemble(dp_lift1, Mf2_vec1, dp_lift2, Mf2, Mf1, nv)
        return dict(A=A, Gamma=Gamma, wake=wk, Vwp=Vwp, dp_lift1=dp_lift1,
                    dp_lift2=dp_lift2, Mf2_vec1=Mf2_vec1, dt_Amat1=dt_Amat1,
                    dt_Amat2_Gamma=dt_Amat2_Gamma, Mf1=Mf1, Mf2=Mf2,
                    Qf_p=Qv, mat0=M0, lift2=L2, mat=Mm, nv=nv, dtn=dtn)
