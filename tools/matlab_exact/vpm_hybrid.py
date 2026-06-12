"""P0-2 — FLUXVortex-style hybrid wake: nearest K rows stay UVLM ring vortices
(Kutta/Mf2 fidelity), older rows convert to regularized vortex particles
(far field, cheap kernels), per the original FLUXVortex repo pattern
(standalone_uvlm.get_wake_particle_sources / warp_vpm.py).

Conversion: each ring (corners c1..c4, circulation G) -> 4 segment particles,
alpha = G*(b-a) at segment midpoints, segment order (1->4),(2->1),(3->2),(4->3)
matching the ring kernel. Particle induction (MATLAB -V sign, matching
ml_uvlm.q1234):
    u(x) = -(1/4pi) * (alpha x r)/|r|^3 * g(|r|/sigma),  g(rho)=rho^3/(rho^2+1)^1.5
Particles convect with the local velocity (freestream + bound + ring wake +
particles) by forward Euler at the fluid-solve cadence.

Known, measured approximations (quantified by the K-sweep):
  - particle dt-contribution to Mf2_vec1 is dropped (dt kernel ~1/r^2, far field)
  - Euler (not RK4) particle convection
Red line: K large must converge back to the full-ring result.
"""
import numpy as np
from ml_fluid_step import MatlabFluidStep, dt_q1234_mat


def rings_to_particles(wk, K, Ny):
    """Strip ring rows beyond the first K (newest-first storage) into particles.
    Returns (trimmed wake dict, new particle pos (M,3), alpha (M,3), sigma (M,))."""
    Nw = wk['Gam'].shape[0]
    keep = K * Ny
    if Nw <= keep:
        return wk, None, None, None
    pos, alp, sig = [], [], []
    pairs = [(0, 3), (1, 0), (2, 1), (3, 2)]
    P = [wk['r1'], wk['r2'], wk['r3'], wk['r4']]
    for i in range(keep, Nw):
        G = wk['Gam'][i]
        seg_lens = []
        for a, b in pairs:
            xa, xb = P[a][i], P[b][i]
            pos.append(0.5 * (xa + xb))
            alp.append(G * (xb - xa))
            seg_lens.append(np.linalg.norm(xb - xa))
        smax = max(seg_lens)
        sig.extend([smax] * 4)
    out = dict(wk)
    for k in ('r1', 'r2', 'r3', 'r4', 'dt1', 'dt2', 'dt3', 'dt4'):
        out[k] = wk[k][:keep]
    out['Gam'] = wk['Gam'][:keep]
    return out, np.array(pos), np.array(alp), np.array(sig)


def particle_induce(targets, pos, alpha, sigma, r_eps):
    """Regularized particle Biot-Savart, MATLAB -V sign. sigma scaled by r_eps
    (rough core scale, consistent with the ring wake kernels)."""
    if pos is None or len(pos) == 0:
        return np.zeros_like(targets)
    r = targets[:, None, :] - pos[None, :, :]          # (T,M,3)
    d2 = np.einsum('tmc,tmc->tm', r, r)
    d = np.sqrt(d2)
    sg = np.maximum(sigma * r_eps, 1e-12)[None, :]
    rho2 = d2 / sg**2
    g = rho2 * np.sqrt(rho2) / (rho2 + 1.0)**1.5       # rho^3/(rho^2+1)^{3/2}
    cross = np.cross(alpha[None, :, :], r)             # (T,M,3)
    u = cross * (g / np.maximum(d, 1e-30)**3)[..., None] / (4.0 * np.pi)
    return -np.sum(u, axis=1)


class HybridFluidStep(MatlabFluidStep):
    """MatlabFluidStep with hybrid ring+particle wake. K = retained ring rows.
    Particle state lives on the instance (ppos/palpha/psigma)."""

    def init_hybrid(self, K, particle_core_eps=0.1, merge_eps=None):
        self.merge_eps = merge_eps   # None = off; e.g. 1e-3 enables merging
        self.K = K
        self.p_eps = particle_core_eps          # rough-style core scale
        self.ppos = None; self.palpha = None; self.psigma = None

    # particle velocity field at arbitrary points (MATLAB sign)
    def pv(self, targets):
        return particle_induce(targets, self.ppos, self.palpha, self.psigma, self.p_eps)

    def merge_pass(self, ref_pts, eps_rel=1e-3, protect=80, vref=1.0):
        """Pairwise moment-conserving merge with at-wing error threshold
        (same scheme as newton_pc flap provider; UAV-VPM precedent)."""
        if self.ppos is None or len(self.ppos) < 2 * protect:
            return 0
        P = len(self.ppos)
        n_free = P - protect
        pos, alpha, sig = self.ppos[:n_free], self.palpha[:n_free], self.psigma[:n_free]
        wc = ref_pts.mean(axis=0)
        cell = 0.1 + 0.15 * np.linalg.norm(pos - wc, axis=1)
        key = np.floor(pos / cell[:, None]).astype(np.int64)
        order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
        ks = key[order]
        same = np.all(ks[1:] == ks[:-1], axis=1)
        used = np.zeros(n_free, bool)
        ci, cj = [], []
        for a in np.nonzero(same)[0]:
            i, j = order[a], order[a + 1]
            if used[i] or used[j] or alpha[i] @ alpha[j] <= 0:
                continue
            used[i] = used[j] = True
            ci.append(i); cj.append(j)
        if not ci:
            return 0
        ci = np.array(ci); cj = np.array(cj)
        a1, a2 = alpha[ci], alpha[cj]
        x1, x2 = pos[ci], pos[cj]
        w1 = np.linalg.norm(a1, axis=1, keepdims=True)
        w2 = np.linalg.norm(a2, axis=1, keepdims=True)
        am = a1 + a2
        xm = (w1 * x1 + w2 * x2) / (w1 + w2 + 1e-30)
        sm = np.maximum(sig[ci], sig[cj])

        def vel(R, X, A, S):
            r = R[:, None, :] - X[None, :, :]
            d = np.linalg.norm(r, axis=-1) + 1e-30
            sg = np.maximum(S * self.p_eps, 1e-12)[None, :]
            rho_ = d / sg
            g = rho_ ** 3 / (rho_ ** 2 + 1.0) ** 1.5
            return -(g / (4 * np.pi * d ** 3))[..., None] * np.cross(A[None], r)

        dv = (vel(ref_pts, x1, a1, sig[ci]) + vel(ref_pts, x2, a2, sig[cj])
              - vel(ref_pts, xm, am, sm))
        err = np.abs(dv).max(axis=(0, 2))
        ok = err < eps_rel * vref
        if not ok.any():
            return 0
        ci, cj = ci[ok], cj[ok]
        keep = np.ones(P, bool); keep[cj] = False
        self.ppos[ci] = xm[ok]; self.palpha[ci] = am[ok]; self.psigma[ci] = sm[ok]
        self.ppos = self.ppos[keep]; self.palpha = self.palpha[keep]
        self.psigma = self.psigma[keep]
        return int(ok.sum())

    def _convect_particles(self, bP, Gamma, wk):
        if self.ppos is None or len(self.ppos) == 0:
            return
        tg = self.ppos
        V = np.zeros_like(tg)
        V[:, 0] += self.U_in
        V += self.induce_like(tg, bP, Gamma, fine=False)
        V += self.induce_like(tg, [wk['r1'], wk['r2'], wk['r3'], wk['r4']],
                              wk['Gam'], fine=False)
        V += self.pv(tg)
        self.ppos = self.ppos + V * self.d_t_wake
        # drop particles beyond truncation
        m = self.ppos[:, 0] <= self.Rtrunc
        if not m.all():
            self.ppos = self.ppos[m]; self.palpha = self.palpha[m]; self.psigma = self.psigma[m]
        if self.merge_eps is not None and self.ppos is not None and len(self.ppos):
            self.merge_pass(np.asarray(bP[0])[::17][:9],
                            eps_rel=self.merge_eps, vref=self.U_in)

    def induce_like(self, targets, P, Gam, fine):
        v, _ = self.vwake(targets, P, Gam, fine)
        return v

    def solve_chain(self, X, wake, Gamma_prev, Gamma_prev2, first_wake=False):
        """Same sequence as the validated base solve_chain, with: particle
        convection before ring advection; particle induction added to wake-ring
        RK4 stages, to the gamma RHS, and to the force-side V_wake_plate."""
        Nq = self.Nq; Ny = self.Ny; Ne = self.Ne
        q = X[:Nq]; dtq = X[Nq:]
        bP = self.panels(q)
        dt_bP = [np.asarray(S @ dtq).reshape(-1, 3) for S in self.Sp]
        rc = self.colloc(q); dt_rc = self.colloc(dtq)
        nv, dtn = self.normals(q, dtq)
        from ml_uvlm import aic_from_q1234
        Vq = self.q1234(rc, bP, fine=True)
        A = aic_from_q1234(Vq, nv)
        # convect particles in the pre-advect field (uses incoming ring wake)
        if wake is not None:
            self._convect_particles(bP, Gamma_prev, wake)
        trail_shed = Gamma_prev2[-Ny:]
        # ring advection with particle background flow: monkey-patch stage_vel
        # by temporarily adding particle field through V_in trick is incorrect
        # (V_in is colloc-shaped) -> wrap generate_wake with particle add-on.
        wk = self._generate_wake_with_particles(first_wake, bP, dt_bP,
                                                Gamma_prev, wake, trail_shed)
        # convert old rows to particles
        wk, npos, nalp, nsig = rings_to_particles(wk, self.K, Ny)
        if npos is not None:
            if self.ppos is None:
                self.ppos, self.palpha, self.psigma = npos, nalp, nsig
            else:
                self.ppos = np.vstack([self.ppos, npos])
                self.palpha = np.vstack([self.palpha, nalp])
                self.psigma = np.concatenate([self.psigma, nsig])
        wP = [wk['r1'], wk['r2'], wk['r3'], wk['r4']]
        Vwp_rhs, q_wake = self.vwake(rc, wP, wk['Gam'], fine=True)
        Vp_col = self.pv(rc)
        Vn = np.einsum('tc,tc->t', dt_rc - self.V_in - Vwp_rhs - Vp_col, nv)
        Gamma = np.linalg.solve(A, Vn)
        wk['Gam'] = wk['Gam'].copy(); wk['Gam'][:Ny] = Gamma_prev[-Ny:]
        Vwp = np.einsum('tsc,s->tc', q_wake, wk['Gam']) + Vp_col
        Vg = np.einsum('tsc,s->tc', Vq, Gamma)
        V_surf1 = Vg + Vwp + self.V_in
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
        dtwP = [wk['dt1'], wk['dt2'], wk['dt3'], wk['dt4']]
        dtq_w = dt_q1234_mat(rc, wP, dt_rc, dtwP)
        Gw_dt_n = np.einsum('tc,tc->t',
                            np.einsum('tsc,s->tc', dtq_w, wk['Gam']), nv)
        Mf2_vec1 = np.linalg.solve(A, -Gw_dt_n)     # particle dt-term dropped
        dtq_b = dt_q1234_mat(rc, bP, dt_rc, dt_bP)
        dt_Amat1 = np.einsum('tsc,tc->ts', dtq_b, nv)
        nvec_Sc = np.zeros((Ne, Nq))
        for e in range(Ne):
            rows = self.Sc_col_d[3*e:3*e+3][:, self.idof[e]]
            nvec_Sc[e, self.idof[e]] = nv[e] @ rows
        Mf1 = np.linalg.solve(A, nvec_Sc)
        Mf2 = np.linalg.inv(A)
        Qv, M0, L2, Mm = self.asm.assemble(dp_lift1, Mf2_vec1, dp_lift2, Mf2, Mf1, nv)
        return dict(A=A, Gamma=Gamma, wake=wk, Vwp=Vwp, dp_lift1=dp_lift1,
                    dp_lift2=dp_lift2, Mf2_vec1=Mf2_vec1, dt_Amat1=dt_Amat1,
                    dt_Amat2_Gamma=Vg, Qf_p=Qv, mat0=M0, lift2=L2, mat=Mm,
                    n_particles=0 if self.ppos is None else len(self.ppos))

    def _generate_wake_with_particles(self, first, bP, dt_bP, Gamma, wake, trail):
        """generate_wake with particle background velocity added to every RK4
        stage (wraps the validated base implementation by temporarily extending
        the stage velocity through vwake interception)."""
        if self.ppos is None or len(self.ppos) == 0 or first or wake is None:
            return self.generate_wake(first, bP, dt_bP, Gamma, wake, trail)
        # intercept: base generate_wake calls self.vwake twice per stage
        # (bound, wake). We add the particle field once per stage by wrapping
        # the BOUND call (first of the pair).
        orig_vwake = self.vwake
        state = {'count': 0}
        def wrapped(tg, P, Gam, fine):
            v, qm = orig_vwake(tg, P, Gam, fine)
            state['count'] += 1
            if state['count'] % 2 == 1:        # bound call of each stage pair
                v = v + self.pv(tg)
            return v, qm
        self.vwake = wrapped
        try:
            wk = self.generate_wake(first, bP, dt_bP, Gamma, wake, trail)
        finally:
            self.vwake = orig_vwake
        return wk
