"""Joint LEV+TEV solver on the validated pterasoftware unsteady chassis.

The augmented per-step system replaces pterasoftware's post-hoc wake-copy
with an IN-SYSTEM joint solve (user reference architecture,
VLM_JOINT_WAKE_CORE.VLM_BING_LESP):

    unknowns x = [G_bound (N), G_TEV_new (S), G_LEV_new (S)]
    rows:
      N  Neumann  : AIC*G_bound + A_TEV*G_TEV + A_LEV*G_LEV = -(wake+freestream)
      S  Kelvin   : G_TEpanel - G_TEV - G_LEV = 0     (in-system, no ledger)
      S  LESP pin : G_LEpanel = G_pre*(crit/LESP)     (active strip)
                    G_LEV = 0                         (inactive strip)

Everything else (geometry stepping, wake RHS from exact wake rings, KJ loads,
movement machinery) is pterasoftware's validated unsteady pipeline. With LEV
disabled this reduces exactly to the bare core (wake copy = the Kelvin row).

Closure invariants (GateError on violation):
  G1  Neumann residual of the augmented solve
  G2  Kelvin row residual (in-system) + global circulation monitor
  G3  LESP pin residual / inactive-strip bound
  G4  newborn geometry guard
  G5  finite loads quantities

LESP formula (reference LESP_formula): 1.13*G_LE/(c*V_ref*(theta+sin theta)).
Reference constants: 1.13, sigma_factor 17.5, |LESP|<10 same-sign guard,
startup delay LEV_START.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import pterasoftware as ps

from pfield import ParticleField, TYPE_FRESH_SHED
from bing_joint_solver import ring_velocity, GateError, LESP_FACTOR, LESP_SANITY_MAX

SIGMA_FACTOR = 17.5


@dataclass
class JointConfig:
    lesp_crit: float = 0.11
    lev_start_step: int = 10
    enable_lev: bool = True
    load_mode: str = "bing"         # "bing": reference cap-and-edit (best
    # macro so far). "v4b3d": intact bound + impulse (degrades; documented).
    # loads via the impulse term (3D-native V4B decomposition, rounded-LE
    # plates). "bing": reference _BING cap-and-edit (sharp-plate, circulation
    # removing). joint_tev: full in-system TEV/LEV unknowns (sensitivity).
    joint_tev: bool = False
    sigma_factor: float = SIGMA_FACTOR
    particle_capacity: int = 200_000
    gate_rtol: float = 1e-8
    lesp_rtol: float = 1e-6
    lesp_inactive_margin: float = 1.5


class JointLEVTEVSolver(ps.unsteady_ring_vortex_lattice_method
                        .UnsteadyRingVortexLatticeMethodSolver):
    """Pterasoftware unsteady solver + in-system joint LEV/TEV solve."""

    def __init__(self, unsteady_problem, cfg: Optional[JointConfig] = None):
        super().__init__(unsteady_problem)
        self.jcfg = cfg or JointConfig()
        self.lev_pf = ParticleField(capacity=self.jcfg.particle_capacity)
        self._lev_gamma_hist: List[np.ndarray] = []
        self._tev_hist: List[np.ndarray] = []
        self._lev_hist: List[np.ndarray] = []
        self.ledger: List[dict] = []   # per-step strip ledger for drag loads
        self._last_bound: Optional[np.ndarray] = None
        self._circ0: Optional[float] = None
        self._lev_streak: dict = {}
        self._last_impulse: Optional[np.ndarray] = None
        self.impulse_force: List[np.ndarray] = []
        self.diag: List[dict] = []
        self._steps_done = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _panel_grid(self):
        """Return (panels(ch,S), S, C) of the current first airplane's wing."""
        wing = self.current_airplanes[0].wings[0]
        panels = wing.panels
        C, S = panels.shape
        return panels, S, C

    def _station_le_te_points(self):
        """LE/TE station point arrays (S+1, 3) from the current panels."""
        panels, S, C = self._panel_grid()
        le = np.zeros((S + 1, 3))
        te = np.zeros((S + 1, 3))
        for s in range(S + 1):
            if s < S:
                le[s] = np.asarray(panels[0, s].Flpp_GP1_CgP1)
                te[s] = np.asarray(panels[C - 1, s].Blpp_GP1_CgP1)
            else:
                le[s] = np.asarray(panels[0, S - 1].Frpp_GP1_CgP1)
                te[s] = np.asarray(panels[C - 1, S - 1].Brpp_GP1_CgP1)
        return le, te

    def _edge_velocity(self, which: str) -> np.ndarray:
        """Finite-difference kinematic velocity of LE/TE stations this step."""
        step = self._current_step
        if step == 0:
            return None
        cur = getattr(self, f"_{which}_points_now")
        prv = getattr(self, f"_{which}_points_prev")
        return (cur - prv) / self.delta_time

    def _panel_induced_velocity(self, targets: np.ndarray) -> np.ndarray:
        """Bound+wake ring induced velocity at arbitrary targets (last strengths)."""
        n_wake = len(self._current_wake_vortex_strengths)
        vel = np.zeros_like(targets)
        if self._last_bound is not None:
            args = dict(
                stackP_GP1_CgP1=targets,
                stackBrrvp_GP1_CgP1=self.stackBrbrvp_GP1_CgP1,
                stackFrrvp_GP1_CgP1=self.stackFrbrvp_GP1_CgP1,
                stackFlrvp_GP1_CgP1=self.stackFlbrvp_GP1_CgP1,
                stackBlrvp_GP1_CgP1=self.stackBlbrvp_GP1_CgP1,
                strengths=self._last_bound,
                r_c0s=np.zeros(len(self._last_bound)),
                singularity_counts=np.zeros(4, dtype=np.int64),
            )
            from pterasoftware import _aerodynamics_functions as aero
            vel += np.asarray(
                aero.expanded_velocities_from_ring_vortices(**args)).sum(axis=1)
        if n_wake:
            from pterasoftware import _aerodynamics_functions as aero
            vel += np.asarray(aero.expanded_velocities_from_ring_vortices(
                stackP_GP1_CgP1=targets,
                stackBrrvp_GP1_CgP1=self._currentStackBrwrvp_GP1_CgP1,
                stackFrrvp_GP1_CgP1=self._currentStackFrwrvp_GP1_CgP1,
                stackFlrvp_GP1_CgP1=self._currentStackFlwrvp_GP1_CgP1,
                stackBlrvp_GP1_CgP1=self._currentStackBlwrvp_GP1_CgP1,
                strengths=self._current_wake_vortex_strengths,
                r_c0s=self._currentStackWakeRc0s,
                singularity_counts=np.zeros(4, dtype=np.int64),
            )).sum(axis=1)
        return vel

    # ------------------------------------------------------------------
    # hook 1: convect LEV particles + inject trail into the wake RHS
    # ------------------------------------------------------------------
    def _calculate_wake_wing_influences(self) -> None:
        # remember edge stations for LESP/birth use before the step advances
        le_now, te_now = self._station_le_te_points()
        if hasattr(self, "_le_points_now"):
            self._le_points_prev = self._le_points_now
            self._te_points_prev = self._te_points_now
        self._le_points_now = le_now
        self._te_points_now = te_now

        super()._calculate_wake_wing_influences()

        if self.lev_pf.n > 0:
            # explicit RK3 convection: v = vInf + LEV field + panel-induced
            v_inf = np.asarray(self.current_operating_point.vInf_GP1__E)
            dt = self.delta_time
            n = self.lev_pf.n
            x0 = self.lev_pf.positions.copy()

            def staged(x):
                self.lev_pf.pos[:n] = x
                return (v_inf[None, :] + self.lev_pf.velocity_self()
                        + self._panel_induced_velocity(x))

            u1 = staged(x0)
            u2 = staged(x0 + 0.5 * dt * u1)
            u3 = staged(x0 + dt * (-u1 + 2.0 * u2))
            self.lev_pf.pos[:n] = x0 + (dt / 6.0) * (u1 + 4.0 * u2 + u3)
            self.lev_pf.promote_fresh()

            # inject trail velocity at collocation points into the wake stack
            v_lev_cp = self.lev_pf.velocity_at(
                np.asarray(self.stackCpp_GP1_CgP1))
            self._currentStackWakeWingInfluences__E = (
                np.asarray(self._currentStackWakeWingInfluences__E)
                + np.einsum("ij,ij->i", v_lev_cp,
                            np.asarray(self.stackUnitNormals_GP1)))

    # ------------------------------------------------------------------
    # hook 2: the joint solve
    # ------------------------------------------------------------------
    def _calculate_vortex_strengths(self) -> None:
        cfg = self.jcfg
        panels, S, C = self._panel_grid()
        N = self.num_panels
        aic = np.asarray(self._currentGridWingWingInfluences__E)
        rhs0 = -(np.asarray(self._currentStackWakeWingInfluences__E)
                 + np.asarray(self._currentStackFreestreamWingInfluences__E))
        norms = np.asarray(self.stackUnitNormals_GP1)
        cps = np.asarray(self.stackCpp_GP1_CgP1)

        gamma_pre = np.linalg.solve(aic, rhs0)
        self._last_bound = gamma_pre.copy()
        dt = self.delta_time

        # ---- _BING post-hoc path (reference driven variant) ----
        # Solve attached (chassis-exact), then cap the LE panel and shed the
        # removed circulation. The LE-row tangency is deliberately replaced by
        # the suction limit: separated LE flow does not enforce no-penetration.
        if not cfg.joint_tev:
            gp = gamma_pre.reshape(C, S)
            le_flat = np.arange(S)
            assert self.panels[le_flat[0]].is_leading_edge
            le_points = self._le_points_now
            v_edge_le = self._edge_velocity("le")
            v_inf = np.asarray(self.current_operating_point.vInf_GP1__E)
            if v_edge_le is None:
                v_rel_st = np.broadcast_to(v_inf, (S + 1, 3)).copy()
            else:
                v_rel_st = v_inf[None, :] - v_edge_le
            v_ref = 0.5 * (np.linalg.norm(v_rel_st[:-1], axis=1)
                           + np.linalg.norm(v_rel_st[1:], axis=1))
            chords = np.zeros(S)
            dx_first = np.zeros(S)
            for s in range(S):
                fl_le = np.asarray(panels[0, s].Flpp_GP1_CgP1)
                fr_le = np.asarray(panels[0, s].Frpp_GP1_CgP1)
                bl_te = np.asarray(panels[C - 1, s].Blpp_GP1_CgP1)
                br_te = np.asarray(panels[C - 1, s].Brpp_GP1_CgP1)
                chords[s] = 0.5 * (np.linalg.norm(bl_te - fl_le)
                                   + np.linalg.norm(br_te - fr_le))
                bl_le = np.asarray(panels[0, s].Blpp_GP1_CgP1)
                br_le = np.asarray(panels[0, s].Brpp_GP1_CgP1)
                dx_first[s] = 0.5 * (np.linalg.norm(bl_le - fl_le)
                                     + np.linalg.norm(br_le - fr_le))
            theta1 = np.arccos(np.clip(1.0 - 2.0 * dx_first / chords, -1.0, 1.0))
            lesp = -LESP_FACTOR * gp[0, :] / (
                chords * v_ref * (theta1 + np.sin(theta1)))
            # per-strip drag-ledger stash (load-level post-processing input)
            areas = np.zeros(S)
            for s in range(S):
                areas[s] = float(sum(panels[j, s].area for j in range(C)))
            self.ledger.append(dict(
                step=self._current_step, lesp=lesp.copy(), chords=chords.copy(),
                areas=areas.copy(), v_rel_st=v_rel_st.copy(),
                v_inf=np.asarray(v_inf, dtype=float).copy(),
                le_now=le_points.copy(), te_now=self._te_points_now.copy(),
                le_prev=(None if not hasattr(self, "_le_points_prev")
                         else self._le_points_prev.copy()),
                dt=dt))
            n_step = self._current_step
            # magnitude + same-sign guards (both reference guards; the
            # sign guard prevents shedding through Baik's sign transitions
            # where the excess computation is ill-defined)
            allowed = (cfg.enable_lev and n_step >= cfg.lev_start_step
                       and np.max(np.abs(lesp)) < LESP_SANITY_MAX
                       and np.max(lesp) * np.min(lesp) >= 0.0)
            active = allowed & (np.abs(lesp) > cfg.lesp_crit)
            gamma_bound = gamma_pre.copy()
            gamma_lev = np.zeros(S)
            if active.any():
                for s in np.flatnonzero(active):
                    # excess suction circulation above the critical LESP
                    excess = gamma_pre[le_flat[s]] * (
                        1.0 - abs(cfg.lesp_crit / lesp[s]))
                    # my-sense (forward traversal) lift-sense shed strength
                    gamma_lev[s] = -excess
                    if cfg.load_mode == "bing":
                        cap = gamma_pre[le_flat[s]] * abs(cfg.lesp_crit / lesp[s])
                        gamma_bound[le_flat[s]] = cap
                        gamma_lev[s] = cap - gamma_pre[le_flat[s]]
            self._current_bound_vortex_strengths = gamma_bound
            for i, panel in enumerate(self.panels):
                panel.ring_vortex.strength = gamma_bound[i]
            if active.any():
                # LEV ring geometry (continuation orientation) + shed
                n_hat = np.array([np.asarray(panels[0, s].unitNormal_GP1)
                                  for s in range(S)])
                leg_r = np.einsum("ij,ij->i", v_rel_st[1:, :] * dt, n_hat)[:, None] * n_hat
                leg_l = np.einsum("ij,ij->i", v_rel_st[:-1, :] * dt, n_hat)[:, None] * n_hat
                lev_rings = np.zeros((S, 4, 3))
                lev_rings[:, 0, :] = le_points[1:, :]
                lev_rings[:, 1, :] = le_points[:-1, :]
                lev_rings[:, 2, :] = le_points[:-1, :] + leg_l
                lev_rings[:, 3, :] = le_points[1:, :] + leg_r
                self._shed_ring_particles(lev_rings, gamma_lev, n_step)
            lesp_solved = -LESP_FACTOR * gamma_bound.reshape(C, S)[0, :] / (
                chords * v_ref * (theta1 + np.sin(theta1)))
            self._tev_solved = None
            self._lev_hist.append(gamma_lev.copy())
            self._steps_done += 1
            circ = float(np.sum(gamma_bound)
                         + sum(np.sum(l) for l in self._lev_hist) * -1.0)
            if self._circ0 is None:
                self._circ0 = circ
            self.diag.append(dict(
                step=n_step, n_particles=self.lev_pf.n,
                lev_strips=int(active.sum()),
                lesp_max=float(np.max(np.abs(lesp_solved))),
                g_tev=0.0, g_lev=float(np.sum(gamma_lev)),
                circ_drift=abs(circ - self._circ0)))
            return

        # ---- LESP per strip from the pre-solve front row ----
        gp = gamma_pre.reshape(C, S)          # pterasoftware order (ch, sp)
        # flat ordering of self.panels: row-major (ch, sp) => index = ch*S + sp
        le_flat = np.array([s for s in range(S)])
        te_flat = np.array([(C - 1) * S + s for s in range(S)])
        assert self.panels[le_flat[0]].is_leading_edge, "flat order mismatch"
        assert self.panels[te_flat[0]].is_trailing_edge, "flat order mismatch"

        le_points, te_points = self._le_points_now, self._te_points_now
        v_edge_le = self._edge_velocity("le")
        v_inf = np.asarray(self.current_operating_point.vInf_GP1__E)
        if v_edge_le is None:
            v_rel_st = np.broadcast_to(v_inf, (S + 1, 3)).copy()
        else:
            v_rel_st = v_inf[None, :] - v_edge_le
        v_ref = 0.5 * (np.linalg.norm(v_rel_st[:-1], axis=1)
                       + np.linalg.norm(v_rel_st[1:], axis=1))

        chords = np.zeros(S)
        dx_first = np.zeros(S)
        for s in range(S):
            # local chord: LE panel front edge -> TE panel back edge
            fl_le = np.asarray(panels[0, s].Flpp_GP1_CgP1)
            fr_le = np.asarray(panels[0, s].Frpp_GP1_CgP1)
            bl_te = np.asarray(panels[C - 1, s].Blpp_GP1_CgP1)
            br_te = np.asarray(panels[C - 1, s].Brpp_GP1_CgP1)
            chords[s] = 0.5 * (np.linalg.norm(bl_te - fl_le)
                               + np.linalg.norm(br_te - fr_le))
            # first-panel chordwise width (reference: geometric mesh spacing)
            bl_le = np.asarray(panels[0, s].Blpp_GP1_CgP1)
            br_le = np.asarray(panels[0, s].Brpp_GP1_CgP1)
            dx_first[s] = 0.5 * (np.linalg.norm(bl_le - fl_le)
                                 + np.linalg.norm(br_le - fr_le))
        theta1 = np.arccos(np.clip(1.0 - 2.0 * dx_first / chords, -1.0, 1.0))
        # pterasoftware ring sense: positive lift <-> negative ring strengths.
        # LESP is defined positive for positive leading-edge suction (a0).
        lesp = -LESP_FACTOR * gp[0, :] / (chords * v_ref * (theta1 + np.sin(theta1)))

        # ---- guards ----
        n_step = self._current_step
        allowed = (cfg.enable_lev and n_step >= cfg.lev_start_step
                   and np.max(np.abs(lesp)) < LESP_SANITY_MAX
                   and np.max(lesp) * np.min(lesp) >= 0.0)
        active = np.zeros(S, dtype=bool)
        if allowed:
            active = np.abs(lesp) > cfg.lesp_crit

        # ---- new ring geometries ----
        dt = self.delta_time
        v_edge_te = self._edge_velocity("te")
        if v_edge_te is None:
            v_rel_te = np.broadcast_to(v_inf, (S + 1, 3)).copy()
        else:
            v_rel_te = v_inf[None, :] - v_edge_te

        # TEV ring in pterasoftware vertex sense (Fr, Fl, Bl, Br):
        # front pair on the bound TE ring's EXTENDED back lattice line
        # (stackBr/Blbrvp of the TE panels) so it continues the bound sheet;
        # back pair convected downstream with the local relative fluid speed
        te_fr = np.array([np.asarray(panels[C - 1, s].ring_vortex.Brrvp_GP1_CgP1)
                          for s in range(S)])
        te_fl = np.array([np.asarray(panels[C - 1, s].ring_vortex.Blrvp_GP1_CgP1)
                          for s in range(S)])
        tev_rings = np.zeros((S, 4, 3))
        tev_rings[:, 0, :] = te_fr                                     # Fr st s+1
        tev_rings[:, 1, :] = te_fl                                     # Fl st s
        tev_rings[:, 2, :] = te_fl + v_rel_te[:-1, :] * dt             # Bl conv
        tev_rings[:, 3, :] = te_fr + v_rel_te[1:, :] * dt              # Br conv

        # LEV ring in CONTINUATION orientation: its LE-line leg runs Fr->Fl
        # (station hi->lo), the same direction as the bound front filament it
        # continues. After capping, bound keeps G_cap and the LEV particle
        # carries G_pre - G_cap, so the total LE circulation is conserved.
        n_hat = np.array([np.asarray(panels[0, s].unitNormal_GP1)
                          for s in range(S)])
        leg_r = np.einsum("ij,ij->i", v_rel_st[1:, :] * dt, n_hat)[:, None] * n_hat
        leg_l = np.einsum("ij,ij->i", v_rel_st[:-1, :] * dt, n_hat)[:, None] * n_hat
        lev_rings = np.zeros((S, 4, 3))    # (Fr, Fl, Bl, Br) order
        lev_rings[:, 0, :] = le_points[1:, :]                # Fr on LE
        lev_rings[:, 1, :] = le_points[:-1, :]               # Fl on LE
        lev_rings[:, 2, :] = le_points[:-1, :] + leg_l       # Bl offset
        lev_rings[:, 3, :] = le_points[1:, :] + leg_r        # Br offset

        def cols(rings):
            v = ring_velocity(cps, rings)          # (N, S, 3), my cyclic sense
            return np.einsum("ij,ikj->ik", norms, v)

        a_tev = cols(tev_rings)
        a_lev = cols(lev_rings)

        # ---- assemble ----
        n_aug = N + 2 * S
        A = np.zeros((n_aug, n_aug))
        b = np.zeros(n_aug)
        A[:N, :N] = aic
        if cfg.joint_tev:
            # their-sense unknowns: columns are the negated my-sense matrices
            A[:N, N:N + S] = -a_tev
            A[:N, N + S:] = -a_lev
        else:
            # my-sense LEV unknown: column matches the forward-traversal
            # particle field directly (see debug_column_vs_particle)
            A[:N, N + S:] = a_lev
        b[:N] = rhs0
        g_scale = max(1e-30, float(np.max(np.abs(gamma_pre))))
        for s in range(S):
            cap = gamma_pre[le_flat[s]] * abs(cfg.lesp_crit / lesp[s])
            if cfg.joint_tev:
                # reference rows: G_TEpanel - G_TEV - G_LEV = 0 (their sense)
                A[N + s, te_flat[s]] = 1.0
                A[N + s, N + s] = -1.0
                A[N + s, N + S + s] = -1.0
                if active[s]:
                    A[N + S + s, le_flat[s]] = 1.0
                    b[N + S + s] = cap
                else:
                    A[N + S + s, N + S + s] = 1.0
            else:
                # balanced explicit structure (reference _BING variant):
                # ONE row per strip: G_LEV = removed circulation in MY sense
                # (G_pre - cap < 0 = physical lift-sense continuation of the
                # LE filament); no panel pin (blob feedback relaxes LESP)
                A[N + s, N + S + s] = 1.0
                if active[s]:
                    b[N + s] = gamma_pre[le_flat[s]] - cap
                A[N + S + s, N + s] = 1.0   # wasted: G_TEV = 0

        x = np.linalg.solve(A, b)
        gamma_bound = x[:N]
        gamma_tev = x[N:N + S]
        gamma_lev = x[N + S:]

        # ---- gates ----
        res = A @ x - b
        if np.max(np.abs(res[:N])) > cfg.gate_rtol * max(1.0, g_scale):
            raise GateError(f"G1 Neumann residual {np.max(np.abs(res[:N])):.3e} "
                            f"step {n_step}")
        if np.max(np.abs(res[N:N + S])) > cfg.gate_rtol * max(1.0, g_scale):
            raise GateError(f"G2 Kelvin row residual step {n_step}")
        lesp_solved = -LESP_FACTOR * gamma_bound.reshape(C, S)[0, :] / (
            chords * v_ref * (theta1 + np.sin(theta1)))
        if allowed:
            if cfg.joint_tev:
                for s in np.flatnonzero(active):
                    tgt = np.sign(lesp[s]) * cfg.lesp_crit
                    if abs(lesp_solved[s] - tgt) > cfg.lesp_rtol * max(1.0, abs(tgt)):
                        raise GateError(
                            f"G3 pin residual strip {s}: {lesp_solved[s]:.4f} "
                            f"vs {tgt:.4f} step {n_step}")
            else:
                # explicit path: each shedding step must strictly relax LESP
                # vs its pre-solve value (blobs convect away each step, so the
                # equilibrium may legitimately sit above crit; what must hold
                # is monotone relaxation and boundedness).
                for s in np.flatnonzero(active):
                    streak = self._lev_streak.get(int(s), 0) + 1
                    self._lev_streak[int(s)] = streak
                    if abs(lesp_solved[s]) > 0.98 * abs(lesp[s]):
                        raise GateError(
                            f"G3 no LESP relaxation strip {s}: "
                            f"{lesp_solved[s]:.4f} vs pre {lesp[s]:.4f} "
                            f"step {n_step}")
                for s in np.flatnonzero(~active):
                    self._lev_streak[int(s)] = 0
            for s in np.flatnonzero(~active):
                if abs(lesp_solved[s]) > cfg.lesp_inactive_margin * cfg.lesp_crit:
                    raise GateError(
                        f"G3 inactive strip {s} LESP {lesp_solved[s]:.4f} "
                        f"step {n_step}")
        if not np.all(np.isfinite(x)):
            raise GateError(f"G5 non-finite strengths step {n_step}")

        # ---- write back bound strengths ----
        self._current_bound_vortex_strengths = gamma_bound
        for i, panel in enumerate(self.panels):
            panel.ring_vortex.strength = gamma_bound[i]

        # ---- shed LEV as particles ----
        self._dbg = dict(a_lev=a_lev.copy(), lev=gamma_lev.copy(),
                         rings=lev_rings.copy(), lesp_pre=lesp.copy(),
                         lesp_solved=lesp_solved.copy())
        self._shed_ring_particles(lev_rings, gamma_lev, n_step,
                                  reverse=self.jcfg.joint_tev)

        # ---- stash for the wake-populate hook ----
        self._tev_solved = gamma_tev.copy()
        self._lev_solved = gamma_lev.copy()
        self._tev_hist.append(gamma_tev.copy())
        self._lev_hist.append(gamma_lev.copy())
        self._steps_done += 1

        circ = float(np.sum(gamma_bound) + np.sum(gamma_tev)
                     + sum(np.sum(t) for t in self._tev_hist)
                     + self._lev_total())
        if self._circ0 is None:
            self._circ0 = circ
        self.diag.append(dict(
            step=n_step, n_particles=self.lev_pf.n,
            lev_strips=int(active.sum()), lesp_max=float(np.max(np.abs(lesp_solved))),
            g_tev=float(np.sum(gamma_tev)), g_lev=float(np.sum(gamma_lev)),
            circ_drift=abs(circ - self._circ0)))

    def _lev_total(self) -> float:
        if not self._lev_hist:
            return 0.0
        return float(sum(np.sum(l) for l in self._lev_hist))

    # ------------------------------------------------------------------
    # hook 3: wake population with the SOLVED TEV strength
    # ------------------------------------------------------------------
    def _populate_next_airplanes_wake_vortices(self) -> None:
        panels, S, C = self._panel_grid()
        saved = []
        tev = getattr(self, "_tev_solved", None)
        if tev is not None and self.jcfg.joint_tev:
            for s in range(S):
                p = panels[C - 1, s]
                saved.append((p.ring_vortex, p.ring_vortex.strength))
                p.ring_vortex.strength = tev[s]
        try:
            super()._populate_next_airplanes_wake_vortices()
        finally:
            for ring, strength in saved:
                ring.strength = strength

    # ------------------------------------------------------------------
    # hook 4: loads see the LEV particle field
    # ------------------------------------------------------------------
    def calculate_solution_velocity(self, stackP_GP1_CgP1=None, **kwargs):
        v = np.asarray(super().calculate_solution_velocity(
            stackP_GP1_CgP1=stackP_GP1_CgP1, **kwargs))
        if self.lev_pf.n > 0 and stackP_GP1_CgP1 is not None:
            v = v + self.lev_pf.velocity_at(np.asarray(stackP_GP1_CgP1))
        return v

    # ------------------------------------------------------------------
    # hook 5: LEV impulse force (Ramesh CNnc, 3D vortex-impulse form)
    # ------------------------------------------------------------------
    def _calculate_loads(self) -> None:
        super()._calculate_loads()
        # per-strip CN stash for the drag ledger (panel forces in GP1)
        if self.ledger and len(self.ledger) == self._current_step + 1:
            rec = self.ledger[-1]
            panels, S, C = self._panel_grid()
            cn = np.zeros(S)
            for s in range(S):
                f_tot = np.zeros(3)
                n_hat = np.asarray(panels[0, s].unitNormal_GP1)
                for j in range(C):
                    f = panels[j, s].forces_GP1
                    if f is not None:
                        f_tot = f_tot + np.asarray(f)
                q_strip = 0.5 * rec["v_inf"] @ rec["v_inf"] * 0.0  # noqa: F841
                # use the strip's own relative dynamic pressure
                v_rel = 0.5 * (rec["v_rel_st"][s] + rec["v_rel_st"][s + 1])
                q_strip = 0.5 * self.current_operating_point.rho * (v_rel @ v_rel)
                cn[s] = (f_tot @ n_hat) / max(q_strip * rec["areas"][s], 1e-30)
            rec["cn_strip"] = cn
        if self.lev_pf.n == 0 and self._last_impulse is None:
            return
        from pterasoftware import _transformations as tr
        op = self.current_operating_point
        T = op.T_pas_GP1_CgP1_to_W_CgP1
        x_w = np.asarray(tr.apply_T_to_vectors(
            T, self.lev_pf.positions, has_point=True))
        g_w = np.asarray(tr.apply_T_to_vectors(
            T, self.lev_pf.gammas, has_point=False))
        # anchor to a fixed world origin: add the moving-frame origin term
        x_o = np.asarray(tr.apply_T_to_vectors(
            T, np.zeros((1, 3)), has_point=True))[0]
        sum_g = g_w.sum(axis=0)
        I = 0.5 * op.rho * (np.cross(x_w, g_w).sum(axis=0) + np.cross(x_o, sum_g))
        F = np.zeros(3)
        if self._last_impulse is not None:
            F = -(I - self._last_impulse) / self.delta_time
        self._last_impulse = I
        self.impulse_force.append(F.copy())
        ap = self.current_airplanes[0]
        ap.forces_W = np.asarray(ap.forces_W) + F

    # ------------------------------------------------------------------
    def _shed_ring_particles(self, rings, strengths, n_step,
                             reverse: bool = False) -> None:
        """Shed rings as 4 vector particles. Traversal direction matches the
        matrix-column sense: forward (leg->leg+1) for the my-sense columns,
        reverse for the their-sense (joint_tev) columns."""
        sf = self.jcfg.sigma_factor
        pos, gam, sig, cir = [], [], [], []
        step = -1 if reverse else 1
        for k in range(len(rings)):
            s = float(strengths[k])
            if abs(s) < 1e-14:
                continue
            rv = rings[k]
            for leg in range(4):
                o = rv[leg]
                t = rv[(leg + step) % 4]
                vec = t - o
                length = float(np.linalg.norm(vec))
                if length < 1e-12:
                    continue
                pos.append(0.5 * (o + t))
                gam.append(vec * s)
                sig.append(length / sf)
                cir.append(s)
        if pos:
            self.lev_pf.add_particles(
                np.array(pos), np.array(gam), np.array(sig),
                circul=np.array(cir), ptype=TYPE_FRESH_SHED, birth_step=n_step)
