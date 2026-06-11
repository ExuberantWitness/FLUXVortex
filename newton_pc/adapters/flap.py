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
    """Per-node (ndof,) structural force vector; affine/lincomb interpolable."""

    def __init__(self, f: np.ndarray, payload: dict | None = None):
        self.f = f
        self.payload = payload

    def affine(self, other: "NodalForceSet", beta: float) -> "NodalForceSet":
        return NodalForceSet(self.f + (other.f - self.f) * beta)

    def lincomb(self, pairs) -> "NodalForceSet":
        acc = None
        for fs, w in pairs:
            term = fs.f * w
            acc = term if acc is None else acc + term
        return NodalForceSet(acc)


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
                 thickness=2e-3, rho_s=1200.0, E0=5e9):
        self.kin = kin
        self.mode = mode
        self.nc, self.ns = nc, ns
        self.shell, self.nodes0 = build_plate_shell(
            chord, span, nc, ns, thickness * hscale, rho_s, E0 * kscale)
        self.t = 0.0
        if mode == "elastic":
            root = [n for n in range(self.shell.nn)
                    if abs(self.nodes0[n, 1]) < 1e-12]
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
            self.shell.step_newmark(forces.f, dt, t_end=t)
        self.t = t

    def state(self) -> dict:
        nc, ns = self.nc, self.ns
        q9 = self.shell.q.reshape(-1, 9)
        dq9 = self.shell.dq.reshape(-1, 9)
        verts = q9[:, 0:3].reshape(ns + 1, nc + 1, 3).transpose(1, 0, 2)
        vels = dq9[:, 0:3].reshape(ns + 1, nc + 1, 3).transpose(1, 0, 2)
        return dict(verts=verts.copy(), vels=vels.copy())



# ── physical-convention ring kernel (validated vs PteraSoftware steady) ───
def _seg_vel(p, a, b, core=0.0):
    """Biot-Savart segment a->b at p, unit circulation, PHYSICAL sign.

    ``core`` is a finite vortex-core radius (meters): the perpendicular
    distance h is regularized to sqrt(h^2 + core^2) via the denominator
    |r1 x r2|^2 + (core*|b-a|)^2. Coincident points contribute zero.
    """
    r1 = p - a; r2 = p - b
    cr = np.cross(r1, r2)
    d = np.einsum('...c,...c->...', cr, cr)
    seg2 = np.einsum('...c,...c->...', b - a, b - a)
    n1 = np.maximum(np.linalg.norm(r1, axis=-1), 1e-12)
    n2 = np.maximum(np.linalg.norm(r2, axis=-1), 1e-12)
    dot = np.einsum('...c,...c->...', b - a, r1 / n1[..., None] - r2 / n2[..., None])
    den = d + (core * core) * seg2 + 1e-12
    coef = np.where(d > 1e-12, dot / (4.0 * np.pi * den), 0.0)
    return cr * coef[..., None]


def _rings_vel(targets, c0, c1, c2, c3, gamma=None, core=0.0):
    """Velocity at targets (T,3) from rings with corners (S,3) x4, physical.
    Returns (T,S,3) unit-ring velocities, or (T,3) if gamma (S,) given."""
    T = targets[:, None, :]
    v = (_seg_vel(T, c0[None], c1[None], core) + _seg_vel(T, c1[None], c2[None], core)
         + _seg_vel(T, c2[None], c3[None], core) + _seg_vel(T, c3[None], c0[None], core))
    if gamma is None:
        return v
    return np.einsum('tsc,s->tc', v, gamma)


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
    """ForceProvider: free ring wake (newest K rows) + far-field particles."""

    def __init__(self, V_inf_vec, rho, dt_window, K=8, sigma_factor=1.0,
                 max_particles=20000, core=0.12):
        self.V_inf = np.asarray(V_inf_vec, dtype=float)
        self.rho = rho
        self.dtw = dt_window
        self.K = K
        self.sigma = sigma_factor * np.linalg.norm(self.V_inf) * dt_window
        self.core = core    # finite wake vortex core (m), ~panel/5
        self.max_particles = max_particles
        # persistent wake state (committed)
        self.wake_v: list = []      # ring corner arrays (ns,4,3)
        self.wake_g: list = []      # ring circulations (ns,)
        self.gamma_prev = None      # bound gamma of last accepted solve
        self.gamma_pprev = None
        self.p_pos = np.zeros((0, 3))
        self.p_alpha = np.zeros((0, 3))
        self._gb_prev = None     # cumulative bound circulation of last commit
        self.n_solves = 0

    # ------------------------------------------------------------------
    def _trial(self, state: dict) -> dict:
        """Repeatable physical-convention solve at `state` (no commit).

        Standard unsteady ring-VLM (Katz & Plotkin): rings at quarter-panel
        offset, collocation at 3/4 panel; free ring wake + far-field
        particles; pressure = rho*(V_loc . tau)*dGamma/ds + rho*d(Gamma_b)/dt.
        Validated: steady limit matches PteraSoftware exactly (37.6 N arena).
        """
        verts, vels = state["verts"], state["vels"]   # (nc+1, ns+1, 3)
        nc, ns = verts.shape[0] - 1, verts.shape[1] - 1
        P = nc * ns
        # panel ring corners (quarter-panel offset along chord index)
        vq = verts + 0.25 * (np.roll(verts, -1, axis=0) - verts)
        vq[-1] = verts[-1] + 0.25 * (verts[-1] - verts[-2])  # extrapolate TE row
        c0 = vq[:-1, :-1].reshape(P, 3); c1 = vq[:-1, 1:].reshape(P, 3)
        c2 = vq[1:, 1:].reshape(P, 3);  c3 = vq[1:, :-1].reshape(P, 3)
        # collocation at 3/4 panel, panel normal/area from corners
        vc = verts + 0.75 * (np.roll(verts, -1, axis=0) - verts)
        rc = 0.25 * (vc[:-1, :-1] + vc[:-1, 1:] + vc[1:, 1:] + vc[1:, :-1])
        rc = rc.reshape(P, 3)
        d1 = (verts[1:, 1:] - verts[:-1, :-1]).reshape(P, 3)
        d2 = (verts[:-1, 1:] - verts[1:, :-1]).reshape(P, 3)
        cr = np.cross(d1, d2)
        area = 0.5 * np.linalg.norm(cr, axis=1)
        nrm = cr / (np.linalg.norm(cr, axis=1, keepdims=True) + 1e-30)
        # structural velocity at colloc (corner average)
        vvel = 0.25 * (vels[:-1, :-1] + vels[:-1, 1:]
                       + vels[1:, 1:] + vels[1:, :-1]).reshape(P, 3)
        # external induction at colloc: committed wake rings + particles
        V_ext = np.zeros((P, 3))
        for wv, wg in zip(self.wake_v, self.wake_g):
            V_ext += _rings_vel(rc, wv[:, 0], wv[:, 1], wv[:, 2], wv[:, 3], wg,
                                core=self.core)
        V_ext += self._particles_at(rc)
        # AIC + solve (physical no-penetration)
        Vq = _rings_vel(rc, c0, c1, c2, c3)            # (P,P,3) unit rings
        A = np.einsum('tsc,tc->ts', Vq, nrm)
        rhs = -np.einsum('tc,tc->t', self.V_inf[None, :] + V_ext - vvel, nrm)
        gamma = np.linalg.solve(A, rhs).reshape(nc, ns)
        # forces: steady + unsteady Bernoulli (Katz-Plotkin 13.12)
        # Katz-Plotkin reference velocity: freestream + wake/particles -
        # structure motion. The bound sheet's own contribution is carried by
        # the gamma-gradient terms; including bound self-induction here
        # double-counts it (blows up under large flapping gamma swings).
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
        # unsteady Bernoulli: ring vortex == constant-doublet panel, so the
        # potential jump on panel ij is gamma_ij ITSELF (not the chordwise
        # cumulative). CCW ring ordering w.r.t. +n flips the sign (same
        # convention note as PteraSoftware issue #27).
        gb = g2
        gb_prev = self._gb_prev if self._gb_prev is not None else gb
        dgb_dt = (gb - gb_prev) / self.dtw
        dp = self.rho * (np.einsum('tc,tc->t', V_loc, tch).reshape(nc, ns) * dgc
                         + np.einsum('tc,tc->t', V_loc, tsh).reshape(nc, ns) * dgs
                         + dgb_dt)
        f_panel = dp.reshape(P)[:, None] * area[:, None] * nrm
        f_panel = f_panel.reshape(nc, ns, 3)
        # wake bookkeeping for commit: advect committed wake, shed new TE row
        wake_v_new = []
        for wv in self.wake_v:
            pts = wv.reshape(-1, 3)
            Vp = (self.V_inf[None, :]
                  + _rings_vel(pts, c0, c1, c2, c3, gamma.reshape(-1),
                               core=self.core))
            for wv2, wg2 in zip(self.wake_v, self.wake_g):
                Vp += _rings_vel(pts, wv2[:, 0], wv2[:, 1], wv2[:, 2],
                                 wv2[:, 3], wg2, core=self.core)
            wake_v_new.append((pts + Vp * self.dtw).reshape(wv.shape))
        # new TE row: front edge pinned at current TE ring back edge
        te0 = c3[(nc - 1) * ns:].copy(); te1 = c2[(nc - 1) * ns:].copy()
        back0 = te0 + self.V_inf * self.dtw
        back1 = te1 + self.V_inf * self.dtw
        new_row = np.stack([te0, te1, back1, back0], axis=1)   # (ns,4,3)
        new_gam = gamma[nc - 1].copy()
        return dict(f_panel=f_panel, gamma=gamma, gb=gb,
                    wake_v_new=[new_row] + wake_v_new,
                    wake_g_new=[new_gam] + [g.copy() for g in self.wake_g])

    def _particles_at(self, targets: np.ndarray) -> np.ndarray:
        if len(self.p_pos) == 0:
            return np.zeros_like(targets)
        sig = np.full(len(self.p_pos), self.sigma)
        contrib = _particle_induce(targets, self.p_pos, self.p_alpha, sig)
        return contrib.sum(axis=1)

    # protocol -----------------------------------------------------------
    def solve(self, state: dict) -> NodalForceSet:
        out = self._trial(state)
        self.n_solves += 1
        panel_f = out["f_panel"]
        nc, ns = panel_f.shape[0], panel_f.shape[1]
        # lump quarter of each panel force to its 4 corner nodes (grid (nc+1,ns+1))
        fgrid = np.zeros((nc + 1, ns + 1, 3))
        quarter = 0.25 * panel_f
        fgrid[:-1, :-1] += quarter
        fgrid[1:, :-1] += quarter
        fgrid[:-1, 1:] += quarter
        fgrid[1:, 1:] += quarter
        # map to shell DOF vector (node layout j-outer, position DOFs)
        ndof = 9 * (nc + 1) * (ns + 1)
        f = np.zeros(ndof)
        f9 = f.reshape(-1, 9)
        f9[:, 0:3] = fgrid.transpose(1, 0, 2).reshape(-1, 3)
        return NodalForceSet(f, payload=out)

    def commit(self, forces: NodalForceSet) -> None:
        out = forces.payload
        if out is None:
            return
        self.gamma_pprev = self.gamma_prev
        self.gamma_prev = out["gamma"].copy()
        self._gb_prev = out["gb"].copy()
        wv, wg = list(out["wake_v_new"]), list(out["wake_g_new"])
        while len(wv) > self.K:
            ring_v, ring_g = wv.pop(), wg.pop()    # oldest = last
            self._rings_to_particles(ring_v, ring_g)
        self.wake_v, self.wake_g = wv, wg
        if len(self.p_pos):
            self.p_pos = self.p_pos + self.V_inf * self.dtw
            if len(self.p_pos) > self.max_particles:
                drop = len(self.p_pos) - self.max_particles
                self.p_pos = self.p_pos[drop:]
                self.p_alpha = self.p_alpha[drop:]

    def _rings_to_particles(self, ring_v: np.ndarray, ring_g: np.ndarray):
        """4 segment-particles per ring: alpha = Gamma * (p2 - p1) at midpoint."""
        segs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        pos, alpha = [], []
        for js in range(ring_v.shape[0]):
            g = ring_g[js]
            if abs(g) < 1e-14:
                continue
            c = ring_v[js]
            for a, b in segs:
                pos.append(0.5 * (c[a] + c[b]))
                alpha.append(g * (c[b] - c[a]))
        if pos:
            self.p_pos = np.vstack([self.p_pos, np.array(pos)])
            self.p_alpha = np.vstack([self.p_alpha, np.array(alpha)])
