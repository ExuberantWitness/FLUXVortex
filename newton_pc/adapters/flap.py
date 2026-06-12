"""Flapping-wing adapters: ANCF shell with driven root + hybrid-wake UVLM.

Arena (congruent with flap_arena/ptera_baseline.py):
  rectangular flat plate, chord x span = C x S, nc x ns panels,
  sinusoidal flapping theta(t) = A*sin(2*pi*t/T) about the x-axis through the
  root chord edge (y=0), freestream V*(cos(alpha), 0, sin(alpha)).

Entry modes:
  - "kinematic" (L1): every node follows the rigid rotation analytically; no
    structural solve. Validates the aero + wake + kinematics chain alone.
  - "elastic"  (L2+): ANCF shell, root edge prescribed via
    ANCFShell.set_prescribed_motion, stiffness scaled by `kscale`
    (large kscale -> quasi-rigid; kscale ~ 1 -> aeroelastic platform mode).

Provider: StandaloneUVLM on the deforming vertex grid with a free ring wake;
rows older than `K` are converted to vortex particles (4 segments/ring,
polynomial-kernel induction — FLUXVortex far-field treatment), convected with
the freestream. solve() is repeatable (wake state deep-copied per call);
commit() accepts the post-solve wake exactly once per window.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Any

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from fluxvortex.ancf_shell import ANCFShell  # noqa: E402
from fluxvortex.standalone_uvlm import StandaloneUVLM  # noqa: E402


# ── force container ──────────────────────────────────────────────────────
class NodalForceSet:
    """Per-node (ndof,) force vector + optional added-mass OPERATOR.

    The operator M_add = dF_aero/d(accel) rides the same interpolation and is
    absorbed into the structural effective mass (M_eff = M - M_add), exactly
    the MATLAB Qf_p_mat treatment - mandatory for added-mass ratios ~1.
    """

    def __init__(self, f: np.ndarray, payload: dict | None = None,
                 madd: np.ndarray | None = None):
        self.f = f
        self.payload = payload
        self.madd = madd

    def affine(self, other: "NodalForceSet", beta: float) -> "NodalForceSet":
        ma = None
        if self.madd is not None or other.madd is not None:
            a = self.madd if self.madd is not None else 0.0 * other.madd
            b = other.madd if other.madd is not None else 0.0 * self.madd
            ma = a + (b - a) * beta
        return NodalForceSet(self.f + (other.f - self.f) * beta, madd=ma)

    def lincomb(self, pairs) -> "NodalForceSet":
        acc = None
        ma = None
        for fs, w in pairs:
            term = fs.f * w
            acc = term if acc is None else acc + term
            if fs.madd is not None:
                ma = fs.madd * w if ma is None else ma + fs.madd * w
        return NodalForceSet(acc, madd=ma)


# ── kinematics ───────────────────────────────────────────────────────────
class FlapKinematics:
    """theta(t) = amp * sin(2*pi*t/period) about the x-axis."""

    def __init__(self, amp_rad: float, period: float):
        self.A = amp_rad
        self.Om = 2.0 * np.pi / period

    def angles(self, t: float):
        th = self.A * np.sin(self.Om * t)
        thd = self.A * self.Om * np.cos(self.Om * t)
        thdd = -self.A * self.Om ** 2 * np.sin(self.Om * t)
        return th, thd, thdd

    def rot(self, th: float) -> np.ndarray:
        c, s = np.cos(th), np.sin(th)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_p(self, th: float) -> np.ndarray:
        c, s = np.cos(th), np.sin(th)
        return np.array([[0, 0, 0], [0, -s, -c], [0, c, -s]])


def build_plate_shell(chord, span, nc, ns, thickness, rho_s, E, nu=0.3):
    x = np.arange(nc + 1) / nc * chord
    y = np.arange(ns + 1) / ns * span
    nn = (nc + 1) * (ns + 1)
    nodes = np.zeros((nn, 3))
    for j in range(ns + 1):
        for i in range(nc + 1):
            nodes[j * (nc + 1) + i, 0] = x[i]
            nodes[j * (nc + 1) + i, 1] = y[j]
    quads = np.zeros((nc * ns, 4), dtype=np.int32)
    for j in range(ns):
        for i in range(nc):
            n1 = j * (nc + 1) + i
            quads[j * nc + i] = (n1, n1 + 1, n1 + nc + 2, n1 + nc + 1)
    return ANCFShell(nodes, quads, h=thickness, rho=rho_s,
                     Ex=E, Ey=E, nu_xy=nu), nodes


# ── structural entry ─────────────────────────────────────────────────────
class FlapEntry:
    """StructuralEntry: rigid-kinematic (L1) or driven-root elastic (L2+)."""

    def __init__(self, chord, span, nc, ns, kin: FlapKinematics,
                 mode="kinematic", kscale=1.0, hscale=1.0,
                 thickness=2e-3, rho_s=1200.0, E0=5e9, nu_xy=0.3,
                 clamp_edge="y0", extra_force_fn=None):
        self.kin = kin
        self.mode = mode
        self.nc, self.ns = nc, ns
        self.extra_force_fn = extra_force_fn
        self.shell, self.nodes0 = build_plate_shell(
            chord, span, nc, ns, thickness * hscale, rho_s, E0 * kscale,
            nu=nu_xy)
        self.t = 0.0
        axis = 1 if clamp_edge == "y0" else 0
        if mode == "elastic":
            root = [n for n in range(self.shell.nn)
                    if abs(self.nodes0[n, axis]) < 1e-12]
            pd = np.array([9 * n + d for n in sorted(root)
                           for d in range(9)])
            q0 = self.shell.q[pd].reshape(-1, 3, 3)

            def cb(t):
                th, thd, _thdd = kin.angles(t)
                R, Rp = kin.rot(th), kin.rot_p(th)
                Rpp = -kin.rot(th) @ np.diag([0.0, 1.0, 1.0])  # d2R/dth2 (x-rot)
                dR = thd * Rp
                ddR = _thdd * Rp + thd ** 2 * Rpp
                return ((q0 @ R.T).reshape(-1), (q0 @ dR.T).reshape(-1),
                        (q0 @ ddR.T).reshape(-1))

            if abs(kin.A) < 1e-15:
                self.shell.set_bc(root, fix_slopes=True)   # static clamp (FSI)
            else:
                self.shell.set_prescribed_motion(root, cb)

    # protocol -----------------------------------------------------------
    def snapshot(self) -> Any:
        return (self.t, self.shell.q.copy(), self.shell.dq.copy())

    def restore(self, snap: Any) -> None:
        self.t, q, dq = snap[0], snap[1].copy(), snap[2].copy()
        self.shell.q[:] = q
        self.shell.dq[:] = dq

    def substep(self, t: float, dt: float, forces: NodalForceSet) -> None:
        if self.mode == "kinematic":
            th, thd, _ = self.kin.angles(t)
            R, Rp = self.kin.rot(th), self.kin.rot_p(th)
            r0 = self.nodes0
            grid9 = self.shell.q.reshape(-1, 9)
            grid9[:, 0:3] = r0 @ R.T
            dgrid9 = self.shell.dq.reshape(-1, 9)
            dgrid9[:, 0:3] = (thd * (r0 @ Rp.T))
        else:
            f = forces.f
            if self.extra_force_fn is not None:
                f = f + self.extra_force_fn(t)
            if forces.madd is not None:
                from scipy.sparse import csr_matrix
                self.shell.set_added_mass_matrix(csr_matrix(forces.madd))
            self.shell.step_newmark(f, dt, t_end=t)
        self.t = t

    def state(self) -> dict:
        nc, ns = self.nc, self.ns
        q9 = self.shell.q.reshape(-1, 9)
        dq9 = self.shell.dq.reshape(-1, 9)
        verts = q9[:, 0:3].reshape(ns + 1, nc + 1, 3).transpose(1, 0, 2)
        vels = dq9[:, 0:3].reshape(ns + 1, nc + 1, 3).transpose(1, 0, 2)
        return dict(verts=verts.copy(), vels=vels.copy())



# ── Ptera-exact ring kernel ────────────────────────────────────────────
# Biot-Savart with PteraSoftware's exact regularization (source-verified):
#   r_c^2 = r_c0^2 + 4*1.25643*(nu + 1e-4*|Gamma|)*age   (Lamb-Oseen + Squire)
#   v += (G/4pi)*(r1+r2)*(r1*r2 - r1.r2)/(r1*r2*(|r1xr2|^2 + |seg|^2*r_c^2)) * r1xr2
_LAMB4 = 4.0 * 1.25643
_SQUIRE = 1.0e-4


def _seg_vel_pt(p, a, b, rc_sq):
    """Ptera-form segment velocity, unit circulation; rc_sq broadcast (S,)."""
    r1 = p - a
    r2 = p - b
    cr = np.cross(r1, r2)
    r3sq = np.einsum('...c,...c->...', cr, cr)
    seg2 = np.einsum('...c,...c->...', b - a, b - a)
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    c3 = np.einsum('...c,...c->...', r1, r2)
    n1n2 = np.maximum(n1 * n2, 1e-30)
    den = n1n2 * (r3sq + seg2 * rc_sq)
    coef = np.where((r3sq > (1e-10 * n1n2) ** 2) & (n1 > 1e-12) & (n2 > 1e-12),
                    (n1 + n2) * (n1n2 - c3) / (4.0 * np.pi * np.maximum(den, 1e-30)),
                    0.0)
    return cr * coef[..., None]


def _rings_vel(targets, c0, c1, c2, c3, gamma=None, rc_sq=0.0):
    """Velocity at targets (T,3) from rings (S,3)x4; rc_sq scalar or (S,)."""
    T = targets[:, None, :]
    rc = np.broadcast_to(np.atleast_1d(rc_sq), (c0.shape[0],))[None, :]
    v = (_seg_vel_pt(T, c0[None], c1[None], rc) + _seg_vel_pt(T, c1[None], c2[None], rc)
         + _seg_vel_pt(T, c2[None], c3[None], rc) + _seg_vel_pt(T, c3[None], c0[None], rc))
    if gamma is None:
        return v
    return np.einsum('tsc,s->tc', v, gamma)


FIXED_WAKE_CORE = None   # set to a float (e.g. 0.12) to override the growth law


def _wake_rc_sq(rc0, gam_rows, ages, nu):
    """Per-row core radius squared via Ptera's growth law (or fixed override)."""
    if FIXED_WAKE_CORE is not None:
        return np.full(len(gam_rows), FIXED_WAKE_CORE ** 2)
    return np.array([rc0 ** 2 + _LAMB4 * (nu + _SQUIRE * np.abs(g).mean()) * a
                     for g, a in zip(gam_rows, ages)])


# ── hybrid-wake UVLM provider ────────────────────────────────────────────
def _particle_induce(targets, pos, alpha, sigma):
    """Polynomial-kernel particle induction, MATLAB -V sign (FLUXVortex)."""
    if len(pos) == 0:
        return np.zeros_like(targets)
    r = targets[:, None, :] - pos[None, :, :]              # (T,P,3)
    d = np.linalg.norm(r, axis=-1) + 1e-30
    rho = d / sigma[None, :]
    g = rho ** 3 / (rho ** 2 + 1.0) ** 1.5
    cross = np.cross(alpha[None, :, :], r)
    return -(g / (4.0 * np.pi * d ** 3))[..., None] * cross  # sum later


class FlapUVLMProvider:
    """ForceProvider: Ptera-exact connected-lattice free wake.

    Wake = point grid pts (R+1, ns+1, 3) shared between adjacent rings
    (Lagrangian sheet, exactly PteraSoftware's gridWrvp): the newest front row
    is the wing's TE ring back vertices at the CURRENT time, so TE motion
    never opens gaps. Ring (i,j) corners = pts[i,j], pts[i,j+1], pts[i+1,j+1],
    pts[i+1,j]; row circulations gam (R, ns); row ages for the Lamb-Oseen +
    Squire core growth (r_c0 = 0.03 * mean chord, Ptera's default).
    """

    def __init__(self, V_inf_vec, rho, dt_window, K=8, nu=15.06e-6,
                 chord=1.5, particles=True, max_particles=60000,
                 added_mass_operator=False, pop_scheme="drop",
                 merge_eps=1e-3, merge_protect=64):
        self.V_inf = np.asarray(V_inf_vec, dtype=float)
        self.rho = rho
        self.dtw = dt_window
        self.K = K
        self.nu = nu
        self.rc0 = 0.03 * chord
        self.particles = particles
        self.max_particles = max_particles
        self.added_mass_operator = added_mass_operator
        self.pop_scheme = pop_scheme        # none | drop | merge
        self.merge_eps = merge_eps          # at-wing rel. velocity threshold
        self.merge_protect = merge_protect  # newest particles excluded
        self.stats = dict(n_merged=0)       # cumulative merge count
        # committed wake state
        self.pts = None             # (R+1, ns+1, 3) near-field ring lattice
        self.gam = []               # list of (ns,) rows, newest first
        self.ages = []              # list of floats, newest first
        # far-field vortex particles (FLUXVortex hybrid)
        self.p_pos = np.zeros((0, 3))
        self.p_alpha = np.zeros((0, 3))
        self.p_sigma = np.zeros(0)
        self.gamma_prev = None
        self._gb_prev = None
        self.n_solves = 0

    def _particles_at(self, targets, chunk=256):
        """Particle induction (far field), target-chunked to bound memory
        at large populations ((chunk x P x 3) intermediates only)."""
        if len(self.p_pos) == 0:
            return np.zeros_like(targets)
        out = np.empty_like(targets)
        for s in range(0, len(targets), chunk):
            t = targets[s:s + chunk]
            r = t[:, None, :] - self.p_pos[None, :, :]
            d = np.linalg.norm(r, axis=-1) + 1e-30
            rho_ = d / self.p_sigma[None, :]
            g = rho_ ** 3 / (rho_ ** 2 + 1.0) ** 1.5
            cross = np.cross(self.p_alpha[None, :, :], r)
            out[s:s + chunk] = ((g / (4.0 * np.pi * d ** 3))[..., None]
                                * cross).sum(axis=1)
        return out

    # ------------------------------------------------------------------
    def _wake_rings(self):
        if self.pts is None or len(self.gam) == 0:
            return None
        p = self.pts
        c0 = p[:-1, :-1].reshape(-1, 3)
        c1 = p[:-1, 1:].reshape(-1, 3)
        c2 = p[1:, 1:].reshape(-1, 3)
        c3 = p[1:, :-1].reshape(-1, 3)
        gam = np.concatenate(self.gam)
        ns = p.shape[1] - 1
        rc_row = _wake_rc_sq(self.rc0, self.gam, self.ages, self.nu)
        rc = np.repeat(rc_row, ns)
        return c0, c1, c2, c3, gam, rc

    def _trial(self, state: dict) -> dict:
        verts, vels = state["verts"], state["vels"]
        nc, ns = verts.shape[0] - 1, verts.shape[1] - 1
        P = nc * ns
        # bound ring corners at quarter-panel offset
        vq = verts + 0.25 * (np.roll(verts, -1, axis=0) - verts)
        vq[-1] = verts[-1] + 0.25 * (verts[-1] - verts[-2])
        c0 = vq[:-1, :-1].reshape(P, 3); c1 = vq[:-1, 1:].reshape(P, 3)
        c2 = vq[1:, 1:].reshape(P, 3);  c3 = vq[1:, :-1].reshape(P, 3)
        vc = verts + 0.75 * (np.roll(verts, -1, axis=0) - verts)
        rc_pt = 0.25 * (vc[:-1, :-1] + vc[:-1, 1:] + vc[1:, 1:] + vc[1:, :-1])
        rc_pt = rc_pt.reshape(P, 3)
        d1 = (verts[1:, 1:] - verts[:-1, :-1]).reshape(P, 3)
        d2 = (verts[:-1, 1:] - verts[1:, :-1]).reshape(P, 3)
        crn = np.cross(d1, d2)
        area = 0.5 * np.linalg.norm(crn, axis=1)
        nrm = crn / (np.linalg.norm(crn, axis=1, keepdims=True) + 1e-30)
        vv = state["vels"]
        vvel = 0.25 * (vv[:-1, :-1] + vv[:-1, 1:]
                       + vv[1:, 1:] + vv[1:, :-1]).reshape(P, 3)
        # wake induction at colloc (Ptera core law)
        V_ext = np.zeros((P, 3))
        wr = self._wake_rings()
        if wr is not None:
            w0, w1, w2, w3, wg, wrc = wr
            V_ext += np.einsum('tsc,s->tc',
                               _seg_grouped(rc_pt, w0, w1, w2, w3, wrc), wg)
        V_ext += self._particles_at(rc_pt)
        # solve (bound rings regularized with rc0^2, age 0)
        Vq = _rings_vel(rc_pt, c0, c1, c2, c3, rc_sq=self.rc0 ** 2)
        A = np.einsum('tsc,tc->ts', Vq, nrm)
        rhs = -np.einsum('tc,tc->t', self.V_inf[None, :] + V_ext - vvel, nrm)
        gamma = np.linalg.solve(A, rhs).reshape(nc, ns)
        # forces (validated composition; V_loc excludes bound self-induction)
        V_loc = self.V_inf[None, :] + V_ext - vvel
        tau_c = (0.5 * (verts[1:, 1:] + verts[1:, :-1])
                 - 0.5 * (verts[:-1, 1:] + verts[:-1, :-1])).reshape(P, 3)
        tau_s = (0.5 * (verts[1:, 1:] + verts[:-1, 1:])
                 - 0.5 * (verts[1:, :-1] + verts[:-1, :-1])).reshape(P, 3)
        dc = np.linalg.norm(tau_c, axis=1); ds_ = np.linalg.norm(tau_s, axis=1)
        tch = tau_c / dc[:, None]; tsh = tau_s / ds_[:, None]
        g2 = gamma
        dgc = np.vstack([g2[:1], np.diff(g2, axis=0)]) / dc.reshape(nc, ns)
        dgs = np.hstack([g2[:, :1], np.diff(g2, axis=1)]) / ds_.reshape(nc, ns)
        gb = g2
        gb_prev = self._gb_prev if self._gb_prev is not None else gb
        # with the implicit added-mass operator active, the acceleration part
        # of dgamma/dt is carried by M_add - keep only it (no double count)
        dgb_dt = (0.0 * gb if self.added_mass_operator
                  else (gb - gb_prev) / self.dtw)
        dp = self.rho * (np.einsum('tc,tc->t', V_loc, tch).reshape(nc, ns) * dgc
                         + np.einsum('tc,tc->t', V_loc, tsh).reshape(nc, ns) * dgs
                         + dgb_dt)
        f_panel = (dp.reshape(P)[:, None] * area[:, None] * nrm).reshape(nc, ns, 3)
        # ---- wake evolution (Ptera order): convect grid, then prepend TE row
        te_back = np.vstack([c3[(nc - 1) * ns:],
                             c2[(nc - 1) * ns + ns - 1][None, :]])  # (ns+1,3)
        # shed strength: the departing bound TE ring carries its strength from
        # the PREVIOUS accepted solve (Ptera/classic delayed-Kutta timing)
        shed_gam = (self.gamma_prev[nc - 1].copy()
                    if self.gamma_prev is not None else gamma[nc - 1].copy())
        if self.pts is None:
            new_pts = np.stack([te_back, te_back + self.V_inf * self.dtw])
            new_gam = [shed_gam]
            new_ages = [0.0]
        else:
            pts_flat = self.pts.reshape(-1, 3)
            Vp = (self.V_inf[None, :]
                  + _rings_vel(pts_flat, c0, c1, c2, c3, gamma.reshape(-1),
                               rc_sq=self.rc0 ** 2))
            if wr is not None:
                Vp += np.einsum('tsc,s->tc',
                                _seg_grouped(pts_flat, w0, w1, w2, w3, wrc), wg)
            Vp += self._particles_at(pts_flat)
            moved = (pts_flat + Vp * self.dtw).reshape(self.pts.shape)
            new_pts = np.vstack([te_back[None], moved])
            new_gam = [shed_gam] + [g.copy() for g in self.gam]
            new_ages = [0.0] + [a + self.dtw for a in self.ages]
        madd = None
        if self.added_mass_operator:
            # dF/daccel: gamma responds as A^-1 N to colloc normal velocity;
            # the impulsive pressure rho*dgamma/dt gives F = M_add @ ddq_z.
            # Quarter lumping maps nodal z-accel -> colloc and panel force ->
            # nodes (same lumping as the force path). MATLAB Qf_p_mat analog.
            nn = (nc + 1) * (ns + 1)
            L_v = np.zeros((P, nn))           # nodal z -> colloc normal accel
            Lf = np.zeros((nn, P))            # panel z-force -> nodal z
            for i in range(nc):
                for j in range(ns):
                    p_ = i * ns + j
                    for (ii, jj) in ((i, j), (i + 1, j), (i, j + 1),
                                     (i + 1, j + 1)):
                        node = jj * (nc + 1) + ii
                        L_v[p_, node] += 0.25 * nrm[p_, 2]
                        Lf[node, p_] += 0.25
            Ainv_N = np.linalg.solve(A, L_v)          # dgamma/d(vz_nodal)
            # panel force per dgamma/dt: rho * area * n_z  (unsteady term)
            Fz = (self.rho * area * nrm[:, 2])[:, None] * Ainv_N
            M_zz = Lf @ Fz                            # (nn, nn): Fz per ddq_z
            madd = np.zeros((9 * nn, 9 * nn))
            zidx = 9 * np.arange(nn) + 2
            madd[np.ix_(zidx, zidx)] = M_zz
        ref_pts = rc_pt[:: max(1, P // 9)][:9].copy()
        return dict(f_panel=f_panel, gamma=gamma, gb=gb, madd=madd,
                    new_pts=new_pts, new_gam=new_gam, new_ages=new_ages,
                    ref_pts=ref_pts)

    # protocol -----------------------------------------------------------
    def solve(self, state: dict) -> NodalForceSet:
        out = self._trial(state)
        self.n_solves += 1
        panel_f = out["f_panel"]
        nc, ns = panel_f.shape[0], panel_f.shape[1]
        fgrid = np.zeros((nc + 1, ns + 1, 3))
        quarter = 0.25 * panel_f
        fgrid[:-1, :-1] += quarter
        fgrid[1:, :-1] += quarter
        fgrid[:-1, 1:] += quarter
        fgrid[1:, 1:] += quarter
        ndof = 9 * (nc + 1) * (ns + 1)
        f = np.zeros(ndof)
        f9 = f.reshape(-1, 9)
        f9[:, 0:3] = fgrid.transpose(1, 0, 2).reshape(-1, 3)
        madd = out.get("madd")
        return NodalForceSet(f, payload=out, madd=madd)

    def commit(self, forces: NodalForceSet) -> None:
        out = forces.payload
        if out is None:
            return
        self.gamma_prev = out["gamma"].copy()
        self._gb_prev = out["gb"].copy()
        pts, gam, ages = out["new_pts"], out["new_gam"], out["new_ages"]
        if len(gam) > self.K:
            if self.particles:                 # convert oldest rows to particles
                for ridx in range(self.K, len(gam)):
                    self._row_to_particles(pts[ridx], pts[ridx + 1],
                                           gam[ridx], ages[ridx])
            gam = gam[:self.K]
            ages = ages[:self.K]
            pts = pts[:self.K + 1]
        self.pts, self.gam, self.ages = pts, list(gam), list(ages)
        # convect particles with the freestream (far-field treatment)
        if len(self.p_pos):
            self.p_pos = self.p_pos + self.V_inf * self.dtw
            if self.pop_scheme == "drop" and len(self.p_pos) > self.max_particles:
                drop = len(self.p_pos) - self.max_particles
                self.p_pos = self.p_pos[drop:]
                self.p_alpha = self.p_alpha[drop:]
                self.p_sigma = self.p_sigma[drop:]
            elif self.pop_scheme == "merge":
                self._merge_pass(out["ref_pts"])

    # ── pairwise moment-conserving merging (at-wing error criterion) ────
    def _merge_pass(self, ref_pts):
        """Merge particle pairs whose replacement changes the induced
        velocity at the wing reference points by < merge_eps * |V_inf|.

        Merge formula conserves the 0th vorticity moment exactly
        (alpha_new = a1 + a2) and the 1st via the strength-weighted
        centroid (UAV-VPM scheme, arXiv 2307.02371)."""
        P = len(self.p_pos)
        if P < 2 * self.merge_protect:
            return
        wing_c = ref_pts.mean(axis=0)
        if getattr(self, "merge_protect_dist", None):
            # distance-based protection: never merge within this radius
            far = np.linalg.norm(self.p_pos - wing_c, axis=1) \
                > self.merge_protect_dist
            far_idx = np.nonzero(far)[0]
            if len(far_idx) < 4:
                return
            pos = self.p_pos[far_idx]
            alpha = self.p_alpha[far_idx]
            sig = self.p_sigma[far_idx]
            n_free = len(far_idx)
        else:
            far_idx = None
            n_free = P - self.merge_protect
            pos = self.p_pos[:n_free]
            alpha = self.p_alpha[:n_free]
            sig = self.p_sigma[:n_free]
        # distance-graded spatial hash (cells grow away from the wing)
        d_wing = np.linalg.norm(pos - wing_c, axis=1)
        cell = 0.1 + 0.15 * d_wing
        key = np.floor(pos / cell[:, None]).astype(np.int64)
        order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
        ks = key[order]
        same = np.all(ks[1:] == ks[:-1], axis=1)
        cand_i, cand_j = [], []
        used = np.zeros(n_free, bool)
        idx = order
        for a in np.nonzero(same)[0]:
            i, j = idx[a], idx[a + 1]
            if used[i] or used[j]:
                continue
            if alpha[i] @ alpha[j] <= 0.0:          # aligned pairs only
                continue
            used[i] = used[j] = True
            cand_i.append(i)
            cand_j.append(j)
        if not cand_i:
            return
        ci = np.array(cand_i)
        cj = np.array(cand_j)
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
            rho_ = d / S[None, :]
            g = rho_ ** 3 / (rho_ ** 2 + 1.0) ** 1.5
            return ((g / (4 * np.pi * d ** 3))[..., None]
                    * np.cross(A[None], r))          # (R, M, 3)

        dv = (vel(ref_pts, x1, a1, sig[ci]) + vel(ref_pts, x2, a2, sig[cj])
              - vel(ref_pts, xm, am, sm))
        err = np.abs(dv).max(axis=(0, 2))            # per candidate pair
        ok = err < self.merge_eps * np.linalg.norm(self.V_inf)
        if not ok.any():
            return
        ci, cj = ci[ok], cj[ok]
        if far_idx is not None:
            ci = far_idx[ci]
            cj = far_idx[cj]
        keep = np.ones(P, bool)
        keep[cj] = False
        self.p_pos[ci] = xm[ok]
        self.p_alpha[ci] = am[ok]
        self.p_sigma[ci] = sm[ok]
        self.p_pos = self.p_pos[keep]
        self.p_alpha = self.p_alpha[keep]
        self.p_sigma = self.p_sigma[keep]
        self.stats["n_merged"] += int(ok.sum())

    def circulation_ledger(self):
        """Total vector circulation of rings + particles (conservation audit)."""
        tot = self.p_alpha.sum(axis=0) if len(self.p_alpha) else np.zeros(3)
        if self.pts is not None:
            p = self.pts
            ns = p.shape[1] - 1
            for ridx, g in enumerate(self.gam):
                c0 = p[ridx, :-1]; c1 = p[ridx, 1:]
                c2 = p[ridx + 1, 1:]; c3 = p[ridx + 1, :-1]
                for jj in range(ns):
                    cc = [c0[jj], c1[jj], c2[jj], c3[jj]]
                    for a_, b_ in ((0, 1), (1, 2), (2, 3), (3, 0)):
                        tot = tot + g[jj] * (cc[b_] - cc[a_])
        return tot

    def _row_to_particles(self, front, back, gam_row, age):
        """4 segment-particles per lattice ring; sigma from the core law."""
        ns = len(gam_row)
        rc = np.sqrt(self.rc0 ** 2
                     + _LAMB4 * (self.nu + _SQUIRE * np.abs(gam_row).mean()) * age)
        sigma = max(rc, 0.5 * np.linalg.norm(self.V_inf) * self.dtw)
        pos, alpha = [], []
        for j in range(ns):
            g = gam_row[j]
            if abs(g) < 1e-12:
                continue
            c = [front[j], front[j + 1], back[j + 1], back[j]]
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                pos.append(0.5 * (c[a] + c[b]))
                alpha.append(g * (c[b] - c[a]))
        if pos:
            self.p_pos = np.vstack([self.p_pos, np.array(pos)])
            self.p_alpha = np.vstack([self.p_alpha, np.array(alpha)])
            self.p_sigma = np.concatenate([self.p_sigma,
                                           np.full(len(pos), sigma)])


def _seg_grouped(targets, c0, c1, c2, c3, rc_sq):
    """Ring velocities with PER-RING rc_sq array; returns (T,S,3) unit rings."""
    T = targets[:, None, :]
    rc = rc_sq[None, :]
    return (_seg_vel_pt(T, c0[None], c1[None], rc)
            + _seg_vel_pt(T, c1[None], c2[None], rc)
            + _seg_vel_pt(T, c2[None], c3[None], rc)
            + _seg_vel_pt(T, c3[None], c0[None], rc))
