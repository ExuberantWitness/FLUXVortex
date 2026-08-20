"""BING joint LEV+TEV solver — reference architecture port onto Warp GPU backend.

Architecture (ported from the user's hand-written reference,
small-particle-computing-master: VLM_JOINT_WAKE_CORE.VLM_BING_LESP,
AERO_INERFACE_BING_LESP, VLM_CORE, VLM_VRING):

Per time step, ONE augmented linear system with unknowns
    x = [Gamma_bound (N=sp*ch), Gamma_TEV_new (sp), Gamma_LEV_new (sp)]
Rows:
    N   Neumann   : AIC*G_bound + A_TEV*G_TEV + A_LEV*G_LEV = -(V_ext - V_panel) . n
    sp  Kelvin    : G_TEpanel - G_TEV - G_LEV = 0        (in-system, no external ledger)
    sp  LESP pin  : G_LEpanel = G_pre*(crit/LESP)        (active strip; caps retained bound)
                    G_LEV = 0                            (inactive strip)

Discretization follows the reference: vortex lattice on the 1/4-chord line of
each panel, boundary condition at 3/4-chord midpoints, ring vertices ordered
cyclically (front_right -> back_right -> back_left -> front_left).

Loads: unsteady Bernoulli  p = rho * [ gamma_ch*(V.n_ch)/d_ch + dGamma/dt ]
with second-order backward dGamma/dt over three snapshots.

All free vorticity lives as particles (TEV and LEV shed every step); the bound
sheet is added as ephemeral TYPE_BOUND particles for wake convection only.

Closure invariants (checked every step, violation raises GateError):
    G1  Neumann residual of the augmented solve
    G2  Kelvin row residual + global circulation-drift monitor
    G3  LESP pin residual on active strips / no excess on inactive strips
    G4  birth geometry: newborn ring legs = edge velocity*dt; near-field guard
    G5  loads recomputed from the same Gamma history (finite dGamma/dt)

Reference constants kept verbatim: LESP factor 1.13, sigma_factor 17.5,
physics sanity gate (|LESP|<10, same sign across strips), startup delay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from pfield import ParticleField, TYPE_FREE, TYPE_FRESH_SHED, TYPE_BOUND

EPS_BS = 10000.0 * 2.2204e-16  # Biot-Savart core guard (reference: VLM_CORE)
LESP_FACTOR = 1.13             # reference: VLM_JOINT_WAKE_CORE.LESP_formula
SIGMA_FACTOR = 17.5            # reference: newborn particle core = leg/17.5
LESP_SANITY_MAX = 10.0         # reference: CHEAK_WHETHER_PHYSICS


class GateError(RuntimeError):
    """Closure-invariant violation. STOP the run; never tune parameters past it."""


# ----------------------------------------------------------------------------
# Geometry: reference mesh_2_ring_list_SINGLE port (vectorized)
# ----------------------------------------------------------------------------

@dataclass
class WingLattice:
    """Lattice on one wing snapshot.

    nodes: (sp+1, ch+1, 3) grid, span index i = 0..sp (root->tip),
    chord index j = 0..ch (LE->TE).  All rings (bound, TEV, LEV) use the
    cyclic vertex order
        (station i+1 @ front, station i+1 @ back, station i @ back, station i @ front)
    i.e. reference's front_right->back_right->back_left->front_left with
    "front" = lower chord index (LE side).
    """
    sp: int
    ch: int
    nodes: np.ndarray  # (sp+1, ch+1, 3)

    # 1/4-chord bound lattice rings, (sp, ch, 4, 3) in cyclic order above
    ring_vertices: np.ndarray = field(init=False)
    # 3/4-chord collocation points, (sp, ch, 3)
    collocation: np.ndarray = field(init=False)
    panel_area: np.ndarray = field(init=False)      # (sp, ch)
    panel_normal: np.ndarray = field(init=False)    # (sp, ch, 3) unit
    d_ch: np.ndarray = field(init=False)            # (sp, ch) chordwise leg length
    d_sp: np.ndarray = field(init=False)            # (sp, ch) spanwise leg length
    n_ch: np.ndarray = field(init=False)            # (sp, ch, 3) chordwise leg unit vector
    le_line: np.ndarray = field(init=False)         # (sp+1, 3) LE grid line (nodes[:,0])
    te_line: np.ndarray = field(init=False)         # (sp+1, 3) TE grid line (nodes[:,ch])
    le_normal: np.ndarray = field(init=False)       # (sp,) panel normal at LE row (per strip)
    chord: np.ndarray = field(init=False)           # (sp,) local chord (mean L/R)
    dx_first: np.ndarray = field(init=False)        # (sp,) geometric first-panel width

    def __post_init__(self):
        sp, ch, g = self.sp, self.ch, self.nodes
        # GEOM legs: chordwise from node (i,j)->(i,j+1); spanwise (i,j)->(i+1,j)
        leg_ch = g[:, 1:, :] - g[:, :-1, :]          # (sp+1, ch, 3) chordwise legs
        # AERO (1/4) lattice: front vertices at 25% of each panel's chordwise leg
        # back vertex of panel j = front vertex of panel j+1 (contiguous lattice);
        # for the TE panel the back edge extends one FULL panel chord beyond its
        # 1/4-line front edge (pterasoftware Brrvp and the reference SHORT_leg
        # both use this "super-horseshoe" TE extension).
        aero_front = g[:, :-1, :] + 0.25 * leg_ch    # (sp+1, ch, 3) stations x panels
        # 3/4 marks for collocation
        three_quarter = g[:, :-1, :] + 0.75 * leg_ch

        # ring vertices (sp, ch, 4, 3) cyclic:
        #   FR = aero_front[i+1, j], BR = back[i+1, j],
        #   BL = back[i, j],     FL = aero_front[i, j]
        back = np.empty_like(aero_front)
        back[:, :-1, :] = aero_front[:, 1:, :]
        back[:, -1, :] = aero_front[:, -1, :] + leg_ch[:, -1, :]

        fr = aero_front[1:, :, :]
        br = back[1:, :, :]
        bl = back[:-1, :, :]
        fl = aero_front[:-1, :, :]
        self.ring_vertices = np.stack((fr, br, bl, fl), axis=2)  # (sp, ch, 4, 3)

        # collocation: midpoint of the two 3/4 marks
        self.collocation = 0.5 * (three_quarter[1:, :, :] + three_quarter[:-1, :, :])

        # area + normal from the crossed diagonals (reference AREA_GEN)
        first_diag = fr - bl     # (sp, ch, 3)
        second_diag = fl - br
        cross = np.cross(first_diag, second_diag)
        cross_norm = np.linalg.norm(cross, axis=2)
        self.panel_area = 0.5 * cross_norm
        self.panel_normal = cross / np.maximum(cross_norm, 1e-300)[..., None]

        # leg lengths / unit vectors for the load terms
        chord_leg = br - fr                                  # (sp, ch, 3) chordwise
        span_leg = fl - fr                                   # (sp, ch, 3) spanwise
        self.d_ch = np.linalg.norm(chord_leg, axis=2)
        self.d_sp = np.linalg.norm(span_leg, axis=2)
        self.n_ch = chord_leg / np.maximum(self.d_ch, 1e-300)[..., None]

        self.le_line = g[:, 0, :].copy()
        self.te_line = g[:, -1, :].copy()
        self.le_normal = self.panel_normal[:, 0, :].mean(axis=1)  # unused placeholder
        # local chord and first-panel width (geometric grid, reference formula)
        c_l = np.linalg.norm(g[:-1, 0, :] - g[:-1, -1, :], axis=1)
        c_r = np.linalg.norm(g[1:, 0, :] - g[1:, -1, :], axis=1)
        self.chord = 0.5 * (c_l + c_r)
        dxl = np.linalg.norm(g[:-1, 1, :] - g[:-1, 0, :], axis=1)
        dxr = np.linalg.norm(g[1:, 1, :] - g[1:, 0, :], axis=1)
        self.dx_first = 0.5 * (dxl + dxr)

    # ---- indexing helpers (flatten order: strip-major, i*ch + j) ----
    def idx(self, i_strip: int, j_chord: int) -> int:
        return i_strip * self.ch + j_chord

    def le_panel_indices(self) -> np.ndarray:
        return np.array([self.idx(i, 0) for i in range(self.sp)])

    def te_panel_indices(self) -> np.ndarray:
        return np.array([self.idx(i, self.ch - 1) for i in range(self.sp)])


# ----------------------------------------------------------------------------
# Biot-Savart (reference Biot_Savart_Law_HY / RING_Biot_Savart, vectorized)
# ----------------------------------------------------------------------------

def line_velocity(targets: np.ndarray, origin: np.ndarray, termination: np.ndarray):
    """Velocity at targets (m,3) from line vortices origin->termination (n,3),
    broadcast (m,n,3), unit strength, with the reference eps core guard."""
    t = targets[:, None, :]            # (m,1,3)
    r1 = t - origin[None, :, :]        # (m,n,3)
    r2 = t - termination[None, :, :]
    r3 = np.cross(r1, r2)
    n1 = np.linalg.norm(r1, axis=2)
    n2 = np.linalg.norm(r2, axis=2)
    n3 = np.linalg.norm(r3, axis=2)
    ok = (n1 > EPS_BS) & (n2 > EPS_BS) & (n3 > EPS_BS)
    n1s = np.maximum(n1, EPS_BS)
    n2s = np.maximum(n2, EPS_BS)
    n3s = np.maximum(n3, EPS_BS)
    part1 = r3 / (n3s ** 2)[..., None]
    part2 = (n1s + n2s)[..., None]
    part3 = 1.0 - np.sum(r1 * r2, axis=2)[..., None] / (n1s * n2s)[..., None]
    v = (part1 * part2 * part3) / (4.0 * np.pi)
    return np.where(ok[..., None], v, 0.0)


def ring_velocity(targets: np.ndarray, rings: np.ndarray) -> np.ndarray:
    """Velocity at targets (m,3) from rings (k,4,3) unit strength -> (m,k,3).
    rings are (k,4,3) cyclic vertex arrays."""
    v = np.zeros((len(targets), len(rings), 3))
    for leg in range(4):
        o = rings[:, leg, :]
        t = rings[:, (leg + 1) % 4, :]
        v += line_velocity(targets, o, t)
    return v


# ----------------------------------------------------------------------------
# Loads (reference G_2_gamma_shift / d_t_gamma / Unsteady_Bernoulli port)
# ----------------------------------------------------------------------------

def gamma_ch_from_rings(gamma_ring: np.ndarray, sp: int, ch: int) -> np.ndarray:
    """Net spanwise filament strength per panel (chordwise difference).
    BOUND_strengths_ch[i,j] = G[i,j] - G[i,j-1], j=0 -> G[i,0]."""
    arr = gamma_ring.reshape(sp, ch)
    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0]
    out[:, 1:] = arr[:, 1:] - arr[:, :-1]
    return out


def dt_gamma_second_order(g_n: np.ndarray, g_1: np.ndarray, g_2: np.ndarray,
                          dt: float) -> np.ndarray:
    """Reference d_t_gamma: 2nd-order backward difference (3 snapshots)."""
    return (3.0 * g_n - 4.0 * g_1 + g_2) / (2.0 * dt)


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------

@dataclass
class BingConfig:
    dt: float
    rho: float
    v_freestream: np.ndarray                 # (3,) freestream velocity vector
    lesp_crit: float = 0.11                  # frozen from the 2D chain (Ramesh)
    lev_start_step: int = 10                 # reference LEV_START startup delay
    sigma_factor: float = SIGMA_FACTOR
    n_span: int = 12
    n_chord: int = 8
    particle_capacity: int = 400_000
    unsteady_loads: bool = True
    gate_neumann_rtol: float = 1e-8
    gate_kelvin_atol_scale: float = 1e-8
    gate_lesp_rtol: float = 1e-6
    gate_min_birth_distance: float = 1e-4   # m, newborn particle vs collocation
    enable_lev: bool = True


@dataclass
class StepDiagnostics:
    step: int
    n_particles: int
    lev_active_strips: int
    lesp_max: float
    gamma_tev_sum: float
    gamma_lev_sum: float
    kelvin_row_residual: float
    neumann_residual: float
    global_circulation: float
    force: np.ndarray


class BingJointSolver:
    """Joint LEV+TEV free-wake solver. One call to step() per time step."""

    def __init__(self, cfg: BingConfig):
        self.cfg = cfg
        self.pf = ParticleField(capacity=cfg.particle_capacity)
        self.gamma_hist: List[np.ndarray] = []      # ring strengths, newest last
        self.shed_tev: List[np.ndarray] = []
        self.shed_lev: List[np.ndarray] = []
        self.diag: List[StepDiagnostics] = []
        self._initial_circulation: Optional[float] = None

    # ------------------------------------------------------------------
    def _total_gamma_scale(self) -> float:
        if not self.gamma_hist:
            return 1.0
        return max(1e-30, float(np.max(np.abs(self.gamma_hist[-1]))))

    # ------------------------------------------------------------------
    def step(self, n_step: int, nodes_now: np.ndarray,
             nodes_prev: np.ndarray, nodes_next: np.ndarray) -> StepDiagnostics:
        cfg = self.cfg
        sp, ch = cfg.n_span, cfg.n_chord
        lat = WingLattice(sp, ch, nodes_now)
        N = sp * ch
        dt = cfg.dt

        # --- panel velocity (finite difference, reference convention) ---
        v_panel = -(lat.collocation.reshape(-1, 3)
                    - WingLattice(sp, ch, nodes_prev).collocation.reshape(-1, 3)) / dt

        # --- external field at collocation: freestream + particles ---
        targets = lat.collocation.reshape(-1, 3)
        if self.pf.n > 0:
            v_wake_at_cp = self.pf.velocity_at(targets)
        else:
            v_wake_at_cp = np.zeros_like(targets)
        v_ext = cfg.v_freestream[None, :] + v_wake_at_cp
        rhs_vec = v_ext - v_panel              # relative fluid velocity
        rhs = np.sum(rhs_vec * lat.panel_normal.reshape(-1, 3), axis=1)

        # --- bound AIC ---
        rings_flat = lat.ring_vertices.reshape(N, 4, 3)
        aic = np.einsum("ij,ikj->ik", lat.panel_normal.reshape(N, 3),
                        ring_velocity(targets, rings_flat))

        # --- attached pre-solve (pin values + LESP evaluation) ---
        gamma_pre = np.linalg.solve(aic, -rhs)
        gp = gamma_pre.reshape(sp, ch)

        # --- edge kinematic velocities (finite difference) ---
        v_edge_le = (nodes_now[:, 0, :] - nodes_prev[:, 0, :]) / dt
        v_edge_te = (nodes_now[:, -1, :] - nodes_prev[:, -1, :]) / dt

        # --- local relative fluid velocity at the LE/TE lines ---
        # (freestream + particle-induced - edge kinematics; the reference passes
        #  the freestream as "virtual velocity" for static wings, which is this
        #  same quantity with zero induced part)
        le_mid = 0.5 * (lat.le_line[:-1, :] + lat.le_line[1:, :])
        te_mid = 0.5 * (lat.te_line[:-1, :] + lat.te_line[1:, :])
        if self.pf.n > 0:
            v_part_le_mid = self.pf.velocity_at(le_mid)
            v_part_te_st = self.pf.velocity_at(lat.te_line)
            v_part_le_st = self.pf.velocity_at(lat.le_line)
        else:
            v_part_le_mid = np.zeros_like(le_mid)
            v_part_te_st = np.zeros_like(lat.te_line)
            v_part_le_st = np.zeros_like(lat.le_line)
        v_rel_le_mid = cfg.v_freestream[None, :] + v_part_le_mid \
            - 0.5 * (v_edge_le[:-1, :] + v_edge_le[1:, :])
        v_ref = np.linalg.norm(v_rel_le_mid, axis=1)
        # relative fluid velocity at each edge STATION (for ring birth geometry)
        v_rel_te_st = cfg.v_freestream[None, :] + v_part_te_st - v_edge_te
        v_rel_le_st = cfg.v_freestream[None, :] + v_part_le_st - v_edge_le

        theta1 = np.arccos(np.clip(1.0 - 2.0 * lat.dx_first / lat.chord, -1.0, 1.0))
        lesp = LESP_FACTOR * gp[:, 0] / (lat.chord * v_ref * (theta1 + np.sin(theta1)))

        # --- guards (reference CHEAK_WHETHER_PHYSICS + startup delay) ---
        lev_allowed = (cfg.enable_lev and n_step >= cfg.lev_start_step)
        physics_ok = True
        if lev_allowed:
            if np.max(np.abs(lesp)) > LESP_SANITY_MAX:
                physics_ok = False
            elif np.max(lesp) * np.min(lesp) < 0.0:
                physics_ok = False
        active = lev_allowed and physics_ok and np.any(np.abs(lesp) > cfg.lesp_crit)
        active_mask = np.zeros(sp, dtype=bool)
        if active:
            active_mask = np.abs(lesp) > cfg.lesp_crit

        # --- new TEV ring geometry (reference part 1) ---
        # front edge ON the wing TE line, back edge convected downstream with the
        # local relative fluid velocity. Cyclic order (FR, BR, BL, FL) keeps the
        # streamwise legs pointing downstream so the wake ring continues the
        # bound sheet's filaments (front/back per the reference convention).
        tev_rings = np.zeros((sp, 4, 3))
        tev_rings[:, 0, :] = lat.te_line[1:, :]                              # FR
        tev_rings[:, 1, :] = lat.te_line[1:, :] + v_rel_te_st[1:, :] * dt     # BR
        tev_rings[:, 2, :] = lat.te_line[:-1, :] + v_rel_te_st[:-1, :] * dt   # BL
        tev_rings[:, 3, :] = lat.te_line[:-1, :]                              # FL

        # --- new LEV ring geometry (reference part 2): LE line + normal-projected
        #     legs of the relative fluid velocity ---
        lev_rings = np.zeros((sp, 4, 3))
        n_hat = lat.panel_normal[:, 0, :]          # LE row panel normals per strip
        leg_proj_r = np.sum(v_rel_le_st[1:, :] * dt * n_hat, axis=1)[:, None] * n_hat
        leg_proj_l = np.sum(v_rel_le_st[:-1, :] * dt * n_hat, axis=1)[:, None] * n_hat
        lev_rings[:, 0, :] = lat.le_line[1:, :] + leg_proj_r          # FR
        lev_rings[:, 1, :] = lat.le_line[1:, :]                       # BR
        lev_rings[:, 2, :] = lat.le_line[:-1, :]                      # BL
        lev_rings[:, 3, :] = lat.le_line[:-1, :] + leg_proj_l         # FL

        a_tev = np.einsum("ij,ikj->ik", lat.panel_normal.reshape(N, 3),
                          ring_velocity(targets, tev_rings))
        a_lev = np.einsum("ij,ikj->ik", lat.panel_normal.reshape(N, 3),
                          ring_velocity(targets, lev_rings))

        # --- augmented system ---
        n_aug = N + 2 * sp
        A = np.zeros((n_aug, n_aug))
        b = np.zeros(n_aug)
        A[:N, :N] = aic
        A[:N, N:N + sp] = a_tev
        A[:N, N + sp:] = a_lev
        b[:N] = -rhs
        le_idx = lat.le_panel_indices()
        te_idx = lat.te_panel_indices()
        for i in range(sp):
            # Kelvin / filament-continuity row: G_TEpanel - G_TEV - G_LEV = 0
            A[N + i, te_idx[i]] += 1.0
            A[N + i, N + i] = -1.0
            A[N + i, N + sp + i] = -1.0
            # LESP row
            if active_mask[i]:
                A[N + sp + i, le_idx[i]] = 1.0
                b[N + sp + i] = gamma_pre[le_idx[i]] * abs(cfg.lesp_crit / lesp[i])
            else:
                A[N + sp + i, N + sp + i] = 1.0
                b[N + sp + i] = 0.0

        x = np.linalg.solve(A, b)
        gamma_bound = x[:N]
        gamma_tev = x[N:N + sp]
        gamma_lev = x[N + sp:]

        # ---------------- invariant gates ----------------
        g_scale = self._total_gamma_scale()
        res = A @ x - b
        neumann_res = float(np.max(np.abs(res[:N]))) if N else 0.0
        kelvin_res = float(np.max(np.abs(res[N:N + sp]))) if sp else 0.0
        if neumann_res > cfg.gate_neumann_rtol * max(1.0, g_scale):
            raise GateError(f"G1 Neumann residual {neumann_res:.3e} at step {n_step}")
        if kelvin_res > cfg.gate_kelvin_atol_scale * max(1.0, g_scale):
            raise GateError(f"G2 Kelvin row residual {kelvin_res:.3e} at step {n_step}")

        # G3: LESP from the solved front row
        gb = gamma_bound.reshape(sp, ch)
        lesp_solved = LESP_FACTOR * gb[:, 0] / (
            lat.chord * v_ref * (theta1 + np.sin(theta1)))
        for i in np.flatnonzero(active_mask):
            target = np.sign(lesp[i]) * cfg.lesp_crit
            if abs(lesp_solved[i] - target) > cfg.gate_lesp_rtol * max(1.0, abs(target)):
                raise GateError(
                    f"G3 LESP pin residual strip {i}: solved {lesp_solved[i]:.4f} "
                    f"vs pin {target:.4f} at step {n_step}")
        for i in np.flatnonzero(~active_mask):
            if abs(lesp_solved[i]) > 1.5 * cfg.lesp_crit:
                raise GateError(
                    f"G3 inactive strip {i} LESP exceeded 1.5*crit: {lesp_solved[i]:.4f} "
                    f"at step {n_step}")

        # global circulation monitor (bound + all shed rings, frozen strengths).
        # NOTE: in the ring-lattice discretization the Kelvin invariant is
        # filament continuity (the in-system row above), not the algebraic sum
        # of ring strengths; this scalar is logged to detect anomalous drift,
        # it is not a hard gate.
        total_circ = float(np.sum(gamma_bound)
                           + sum(np.sum(t) for t in self.shed_tev)
                           + sum(np.sum(l) for l in self.shed_lev))
        if self._initial_circulation is None:
            self._initial_circulation = total_circ
        self._circ_drift = abs(total_circ - self._initial_circulation)

        # ---------------- loads ----------------
        gamma_ch = gamma_ch_from_rings(gamma_bound, sp, ch)
        if len(self.gamma_hist) >= 2:
            g1 = self.gamma_hist[-1]
            g2 = self.gamma_hist[-2]
        else:
            g1 = np.zeros(N)
            g2 = np.zeros(N)
        if cfg.unsteady_loads:
            dgt = dt_gamma_second_order(gamma_bound, g1, g2, dt).reshape(sp, ch)
        else:
            dgt = np.zeros((sp, ch))
        # p = rho [ gamma_ch (V_rel . n_ch)/d_ch + dGamma/dt ]
        vdot = np.einsum("ijk,ijk->ij", rhs_vec.reshape(sp, ch, 3), lat.n_ch)
        pressure = cfg.rho * (gamma_ch * vdot / lat.d_ch + dgt)
        if not np.all(np.isfinite(pressure)):
            raise GateError(f"G5 non-finite pressure at step {n_step}")
        force = np.einsum("ij,ij,ijk->k", pressure, lat.panel_area,
                          lat.panel_normal)

        # ---------------- shed TEV/LEV as particles ----------------
        self._shed_rings(tev_rings, gamma_tev, n_step)
        self._shed_rings(lev_rings, gamma_lev, n_step)
        self.shed_tev.append(gamma_tev.copy())
        self.shed_lev.append(gamma_lev.copy())

        # G4: newborn guard — no newborn particle within guard distance of a CP
        if sp:
            new_pos = self.pf.positions[-8 * sp:]
            if len(new_pos):
                d2 = np.min(np.linalg.norm(
                    new_pos[:, None, :] - targets[None, :, :], axis=2), axis=1)
                if d2.min() < cfg.gate_min_birth_distance:
                    raise GateError(
                        f"G4 newborn particle {d2.min():.2e} m from collocation "
                        f"at step {n_step}")

        # ---------------- bound sheet as ephemeral particles ----------------
        self._shed_rings(rings_flat, gamma_bound, n_step, ptype=TYPE_BOUND)

        # ---------------- RK3 convection ----------------
        self._advance_particles(dt)

        # remove ephemeral bound particles, promote fresh shed to free
        self.pf.remove_type(TYPE_BOUND)
        self.pf.promote_fresh()

        # ---------------- bookkeeping ----------------
        self.gamma_hist.append(gamma_bound.copy())
        diag = StepDiagnostics(
            step=n_step, n_particles=self.pf.n,
            lev_active_strips=int(active_mask.sum()),
            lesp_max=float(np.max(np.abs(lesp_solved))),
            gamma_tev_sum=float(np.sum(gamma_tev)),
            gamma_lev_sum=float(np.sum(gamma_lev)),
            kelvin_row_residual=kelvin_res, neumann_residual=neumann_res,
            global_circulation=total_circ, force=force.copy())
        self.diag.append(diag)
        return diag

    # ------------------------------------------------------------------
    def _shed_rings(self, rings: np.ndarray, strengths: np.ndarray,
                    n_step: int, ptype: float = TYPE_FRESH_SHED) -> None:
        """Ring -> 4 vector particles (reference add_RING_particle)."""
        sf = self.cfg.sigma_factor
        pos_list, gam_list, sig_list, cir_list = [], [], [], []
        for k in range(len(rings)):
            s = float(strengths[k])
            if abs(s) < 1e-14:
                continue
            rv = rings[k]
            for leg in range(4):
                o = rv[leg]
                t = rv[(leg + 1) % 4]
                vec = t - o
                length = float(np.linalg.norm(vec))
                if length < 1e-12:
                    continue
                pos_list.append(0.5 * (o + t))
                gam_list.append(vec * s)
                sig_list.append(length / sf)
                cir_list.append(s)
        if pos_list:
            self.pf.add_particles(
                np.array(pos_list), np.array(gam_list), np.array(sig_list),
                circul=np.array(cir_list), ptype=ptype, birth_step=n_step)

    # ------------------------------------------------------------------
    def _advance_particles(self, dt: float) -> None:
        """RK3 convection; sources are staged together with targets each
        substep (velocity = freestream + full particle field, bound sheet
        present as ephemeral particles)."""
        v_inf = self.cfg.v_freestream
        n = self.pf.n
        if n == 0:
            return
        x0 = self.pf.positions.copy()

        def staged_velocity(x_stage):
            self.pf.pos[:n] = x_stage
            return v_inf[None, :] + self.pf.velocity_self()

        u1 = staged_velocity(x0)
        u2 = staged_velocity(x0 + 0.5 * dt * u1)
        u3 = staged_velocity(x0 + dt * (-u1 + 2.0 * u2))
        self.pf.pos[:n] = x0 + (dt / 6.0) * (u1 + 4.0 * u2 + u3)
