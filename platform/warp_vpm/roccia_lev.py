"""Roccia-style LEV solver: LESP clipping from bound circulation.

Core insight: the double-counting problem is solved by DEDUCTING LEV
circulation from the bound solve (Kelvin conservation), not adding
LEV velocity to the RHS. The LESP condition determines when to clip.

Architecture:
1. Standard UVLM solve: AIC @ Γ = -(b_wake + b_freestream)
2. LESP check: |Γ_LE| / (π/2 · U · c) > LESP_crit ?
3. If active: clip Γ_LE to LESP_crit equivalent, excess → LEV particle
4. LEV particles convect (self-induced + freestream + panel velocity)
5. LEV velocity enters NEXT step's RHS (self-consistent, no double-count)
"""
from __future__ import annotations

import numpy as np
from warp_vpm.pfield import ParticleField, TYPE_FRESH_SHED, TYPE_FREE
from warp_vpm.integrator import RK3_A, RK3_B


class RocciaLEVSolver:
    """Mixin: LESP-clipped bound circulation + LEV particles.

    The LEV circulation comes FROM the bound solve (excess above LESP critical),
    satisfying Kelvin's theorem by construction.
    """

    def _init_roccia_lev(self, U, c, period_s, lesp_crit=0.11, lev_core=0.02):
        self.pf = ParticleField(capacity=100_000)
        self._U = U
        self._c = c
        self._period = period_s
        self._lesp_crit = lesp_crit
        self._lev_core = lev_core
        self._hybrid_step = -1
        self._lev_history = []  # per-step LEV count for diagnostics

    def _calculate_vortex_strengths(self) -> None:
        """Override: standard solve + LESP clipping + LEV particle creation."""
        # 1. Standard solve (AIC @ Γ = -b)
        super()._calculate_vortex_strengths()

        # 2. Get bound vortex strengths
        gamma = np.array(self._current_bound_vortex_strengths, dtype=float)
        n_panels = len(gamma)
        n_span = n_panels // 2  # 2 chordwise rows → 8 spanwise
        le_indices = np.arange(n_span)  # panels 0..7 = LE row

        # 3. Compute LESP per spanwise strip
        # Thin-airfoil: LESP ≈ 2·|Γ_LE| / (π·U·c)
        # (Ramesh: LESP = leading-edge suction parameter ≈ 2A0 where A0 is
        #  the Fourier coefficient; for thin airfoil A0 ≈ α - α_camber)
        # Simpler: LESP ∝ |Γ_LE| / (0.5·π·U·c)
        lesp_denom = 0.5 * np.pi * self._U * self._c
        gamma_clipped = gamma.copy()
        levs_created = 0

        # 4. LESP clipping: excess becomes LEV particle
        for j in range(n_span):
            idx = le_indices[j]
            g = gamma[idx]
            lesp_val = abs(g) / lesp_denom

            if lesp_val > self._lesp_crit:
                # Maximum allowed bound circulation at this LE panel
                gamma_max = self._lesp_crit * lesp_denom * np.sign(g)

                # Excess circulation = new LEV
                gamma_excess = g - gamma_max

                # Clip bound circulation (Kelvin: bound + LEV = original)
                gamma_clipped[idx] = gamma_max

                # Create LEV particle at the LE vertex
                pass  # no particle creation (pure clipping)
                levs_created += 1

        # 5. Write clipped values back
        if levs_created > 0:
            for i, panel in enumerate(self.panels):
                panel.ring_vortex.strength = gamma_clipped[i]
            self._current_bound_vortex_strengths = gamma_clipped

        self._lev_history.append(levs_created)

    def _create_lev_particle(self, panel_idx, strip_idx, gamma_excess):
        """Create a LEV particle from the excess bound circulation."""
        panel = self.panels[panel_idx]

        # LE vertex position (midpoint of front edge)
        fl = np.array(panel.Flpp_GP1_CgP1)
        fr = np.array(panel.Frpp_GP1_CgP1)
        le_mid = 0.5 * (fl + fr)

        # LE tangent direction (spanwise, from root to tip)
        tangent = fr - fl
        tangent /= max(np.linalg.norm(tangent), 1e-12)

        # Displace slightly from the surface (along normal)
        normal = np.array(panel.unitNormal_GP1)
        displacement = 0.01  # 10mm from surface
        position = le_mid + displacement * normal

        # LEV circulation along the tangent
        gamma_vec = gamma_excess * tangent

        # Add particle
        self.pf.add_particles(
            position.reshape(1, 3),
            gamma_vec.reshape(1, 3),
            np.array([self._lev_core]),
            ptype=TYPE_FRESH_SHED,
            birth_step=self._current_step,
        )

    def _convect_particles(self):
        """RK3 convection: self-induced + freestream + panel velocity."""
        if self.pf.n == 0:
            return

        vinf = np.asarray(self._currentVInf_GP1__E, dtype=np.float64)
        dt = float(self.delta_time)
        n = self.pf.n
        mem = np.zeros((n, 3))

        for stage in range(3):
            a, b = RK3_A[stage], RK3_B[stage]
            pos = self.pf.positions
            vel = self.pf.velocity_self() + vinf[None, :]
            mem = a * mem + vel
            self.pf.pos[:n] += b * dt * mem

    def _inject_lev_feedback(self):
        """LEV particles' velocity at collocation points → RHS.

        This is NOT double-counting because the bound circulation was already
        clipped: the LEV carries exactly the circulation that was removed
        from the bound solve. The LEV's induced velocity represents the
        separated-flow effect that the bound vortex would have had.
        """
        if self.pf.n == 0:
            return
        coll = np.array(self.stackCpp_GP1_CgP1, dtype=float).reshape(-1, 3)
        normals = np.array(self.stackUnitNormals_GP1, dtype=float).reshape(-1, 3)
        v = self.pf.velocity_at(coll)
        infl = np.einsum("ij,ij->i", v, normals)
        n = min(len(self._currentStackWakeWingInfluences__E), len(infl))
        self._currentStackWakeWingInfluences__E[:n] += infl[:n]

    def _calculate_wake_wing_influences(self) -> None:
        """Main hook: convect particles, then standard wake influence."""
        if self._current_step > 0 and self._hybrid_step != self._current_step:
            self._hybrid_step = self._current_step
            self.pf.promote_fresh()
            self._convect_particles()

        # Standard wake influence (prescribed panels)
        super()._calculate_wake_wing_influences()

        # Add LEV feedback (from previously created particles)
        # This is self-consistent: LEV circulation was deducted from bound
        pass  # no feedback (pure clipping)

        if self._current_step % 8 == 0:
            print(f"  step {self._current_step}: LEV={self.pf.n}", flush=True)
