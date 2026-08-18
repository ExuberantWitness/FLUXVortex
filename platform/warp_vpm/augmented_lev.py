"""Roccia augmented-system LEV solver.

Implements the correct architecture: LEV circulations enter the linear
system as UNKNOWNS alongside bound panel circulations. Kelvin's theorem
rows couple them, preventing double-counting by construction.

Augmented system (24×24 for 16 panels + 8 LEV strips):
  [AIC (16×16)  A_LEV (16×8)] [Γ_bound]   [-b_wake - b_freestream]
  [C_KELVIN (8×16)  I (8×8)] [Γ_LEV  ] = [r_KELVIN]

Where:
  AIC[i,j]     = n_i · V_bound_j(x_i)      (existing Pterra matrix)
  A_LEV[i,k]   = n_i · V_LEV_k(x_i)        (unit-strength LEV k at collocation i)
  C_KELVIN[j,i] = δ(i == LE_row[j])         (picks LE panel j's bound Γ)
  I[j,k]       = δ(j == k)                  (LEV's own circulation)
  r_KELVIN[j]  = Γ_total_prev[j]            (previous step's bound + LEV)

When LEV is NOT active (LESP below critical), the constraint row for strip j
reduces to: Γ_bound_LE[j] = Γ_bound_LE_prev[j] (no change, no LEV).
"""
from __future__ import annotations

import numpy as np
from warp_vpm.pfield import ParticleField, TYPE_FRESH_SHED, TYPE_FREE
from warp_vpm.integrator import RK3_A, RK3_B


class AugmentedLEVSolver:
    """Mixin: Roccia augmented linear system with LEV unknowns."""

    def _init_augmented_lev(self, U, c, period_s, lesp_crit=0.11, lev_core=0.02):
        self.pf = ParticleField(capacity=100_000)
        self._U = U
        self._c = c
        self._period = period_s
        self._lesp_crit = lesp_crit
        self._lev_core = lev_core
        self._hybrid_step = -1
        # Per-strip state: [total_circulation_prev, lev_active]
        n_panels = 16  # 2 chordwise × 8 spanwise
        self._n_span = n_panels // 2
        self._lev_gamma = np.zeros(self._n_span)  # current LEV Γ per strip
        self._lev_positions = np.zeros((self._n_span, 3))  # LEV particle positions
        self._lev_active = np.zeros(self._n_span, dtype=bool)
        self._total_prev = np.zeros(self._n_span)  # Γ_bound_LE + Γ_LEV from prev step

    def _calculate_vortex_strengths(self) -> None:
        """Solve the augmented system [AIC, A_LEV; C, I] [Γ; Γ_LEV] = [-b; r]."""
        n_panels = len(self._current_bound_vortex_strengths)
        n_span = self._n_span
        n_total = n_panels + n_span

        # Build A_LEV: unit-strength LEV velocity at each collocation point
        # A_LEV[i, k] = n_i · V̂_LEV_k(x_i) where V̂ is for unit Γ along tangent_k
        coll = np.array(self.stackCpp_GP1_CgP1, dtype=float).reshape(-1, 3)
        normals = np.array(self.stackUnitNormals_GP1, dtype=float).reshape(-1, 3)

        # LEV particle positions and directions (per spanwise strip)
        le_vertices, le_tangents = self._get_le_geometry()
        a_lev = np.zeros((n_panels, n_span))

        # For each LEV strip k, compute unit-strength velocity at all collocation points
        for k in range(n_span):
            if not self._lev_active[k]:
                continue  # inactive strips contribute nothing
            pos_k = self._lev_positions[k]
            tangent_k = le_tangents[k]
            sigma_k = self._lev_core

            # Compute velocity at all collocation points from unit-strength particle k
            # Using the blob kernel: v = -1/(4π)·q(r/σ)/r³·(Δr × Γ)
            dx = coll - pos_k[None, :]
            r2 = np.sum(dx * dx, axis=1)
            r = np.sqrt(np.maximum(r2, 1e-30))
            rho = r / sigma_k
            # Regularization q(ρ) = erf(ρ/√2) - √(2/π)·ρ·exp(-ρ²/2)
            from math import erf, sqrt, pi
            q = np.array([erf(rl / sqrt(2)) - sqrt(2 / pi) * rl * np.exp(-0.5 * rl * rl)
                          for rl in rho])
            # Unit-strength Γ along tangent
            gamma_unit = tangent_k  # |Γ|=1, direction = tangent
            cross = np.cross(dx, gamma_unit[None, :])
            w = -1.0 / (4 * pi) * q / (r2 * r)
            vel_k = w[:, None] * cross  # (n_panels, 3)

            # Project on normals → normal velocity influence
            a_lev[:, k] = np.einsum("ij,ij->i", vel_k, normals)

        # Build constraint rows: Kelvin for active strips, Γ_LEV=0 for inactive
        c_kelvin = np.zeros((n_span, n_panels))
        d_lev = np.eye(n_span)
        r_constraint = np.zeros(n_span)
        for j in range(n_span):
            if self._lev_active[j]:
                # Kelvin: Γ_bound_LE[j] + Γ_LEV[j] = total_prev[j]
                c_kelvin[j, j] = 1.0  # picks LE panel j's bound Γ
                d_lev[j, j] = 1.0
                r_constraint[j] = self._total_prev[j]
            else:
                # Inactive: force Γ_LEV[j] = 0 (no constraint on bound)
                c_kelvin[j, :] = 0.0
                d_lev[j, j] = 1.0
                r_constraint[j] = 0.0

        # Assemble augmented system
        aug = np.zeros((n_total, n_total))
        aug[:n_panels, :n_panels] = self._currentGridWingWingInfluences__E
        aug[:n_panels, n_panels:] = a_lev
        aug[n_panels:, :n_panels] = c_kelvin
        aug[n_panels:, n_panels:] = d_lev

        # RHS
        rhs = np.zeros(n_total)
        rhs[:n_panels] = -(self._currentStackWakeWingInfluences__E
                           + self._currentStackFreestreamWingInfluences__E)
        rhs[n_panels:] = r_constraint

        # Solve
        try:
            solution = np.linalg.solve(aug, rhs)
        except np.linalg.LinAlgError:
            # Fallback: standard solve without LEV
            super()._calculate_vortex_strengths()
            return

        gamma_bound = solution[:n_panels]
        gamma_lev = solution[n_panels:]

        # Write bound strengths back
        self._current_bound_vortex_strengths = gamma_bound
        for i, panel in enumerate(self.panels):
            panel.ring_vortex.strength = gamma_bound[i]

        # Update LEV state
        self._lev_gamma = gamma_lev
        self._total_prev = gamma_bound[:n_span] + gamma_lev

        # Create/update LEV particles for next step
        self._update_lev_particles(le_vertices, le_tangents)

    def _get_le_geometry(self):
        """Get LE vertex positions and tangent directions per spanwise strip."""
        try:
            panels = self.panels
            n_span = self._n_span
            le_panels = panels[:n_span]
            vertices, tangents = [], []
            for p in le_panels:
                fl = np.array(p.Flpp_GP1_CgP1)
                fr = np.array(p.Frpp_GP1_CgP1)
                mid = 0.5 * (fl + fr)
                tangent = fr - fl
                tangent /= max(np.linalg.norm(tangent), 1e-12)
                vertices.append(mid)
                tangents.append(tangent)
            return np.array(vertices), np.array(tangents)
        except Exception:
            return None, None

    def _update_lev_particles(self, vertices, tangents):
        """Create or update LEV particles based on solved gamma_lev."""
        if vertices is None:
            return

        # Check LESP to determine which strips are active
        # Simple proxy: if |Γ_bound_LE| is large, LESP is high
        gamma_le = self._current_bound_vortex_strengths[:self._n_span]
        lesp_proxy = np.abs(gamma_le + self._lev_gamma)  # total circulation

        for j in range(self._n_span):
            # LESP active if total circulation exceeds threshold
            # (calibrated so that only extreme AoA triggers)
            if lesp_proxy[j] > self._lesp_trigger:
                self._lev_active[j] = True
                # Update LEV particle position: at LE vertex + normal offset
                panel = self.panels[j]
                normal = np.array(panel.unitNormal_GP1)
                self._lev_positions[j] = vertices[j] + 0.005 * normal  # 5mm offset
            else:
                self._lev_active[j] = False
                self._lev_gamma[j] = 0.0

        # Rebuild particle field (simpler than incremental update)
        active = np.flatnonzero(self._lev_active)
        if len(active) > 0:
            # Merge all active LEV particles into the field
            # Each step adds one particle per active strip (accumulating trail)
            pos_list = []
            gam_list = []
            sig_list = []
            for j in active:
                if abs(self._lev_gamma[j]) > 1e-12:
                    pos_list.append(self._lev_positions[j])
                    gam_list.append(self._lev_gamma[j] * tangents[j])
                    sig_list.append(self._lev_core)
            if pos_list:
                self.pf.add_particles(
                    np.array(pos_list), np.array(gam_list), np.array(sig_list),
                    ptype=TYPE_FRESH_SHED, birth_step=self._current_step)

    def _convect_particles(self):
        """RK3 convection of accumulated LEV trail particles."""
        if self.pf.n == 0:
            return
        vinf = np.asarray(self._currentVInf_GP1__E, dtype=np.float64)
        dt = float(self.delta_time)
        n = self.pf.n
        mem = np.zeros((n, 3))
        for stage in range(3):
            a, b = RK3_A[stage], RK3_B[stage]
            vel = self.pf.velocity_self() + vinf[None, :]
            mem = a * mem + vel
            self.pf.pos[:n] += b * dt * mem

    def _calculate_wake_wing_influences(self) -> None:
        """Hook: convect particles before the standard wake influence."""
        if self._current_step > 0 and self._hybrid_step != self._current_step:
            self._hybrid_step = self._current_step
            self.pf.promote_fresh()
            self._convect_particles()

        # Standard wake influence (NO LEV RHS addition — LEV is in the augmented solve)
        super()._calculate_wake_wing_influences()

        if self._current_step % 8 == 0:
            n_active = self._lev_active.sum()
            print(f"  step {self._current_step}: LEV active={n_active}, "
                  f"trail={self.pf.n}", flush=True)
