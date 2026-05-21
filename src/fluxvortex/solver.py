"""
UVPMHybridSolver — PteraSoftware UVLM wing solver + FLOWVPM-style vortex particle wake.

Inherits UnsteadyRingVortexLatticeMethodSolver and overrides only the wake-related
methods to use vortex particles instead of wake ring vortex panels.

Compatible with PteraSoftware 5.x (installed version with private method names).
"""
import numpy as np
import sys

# numpy compat
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pterasoftware as ps
from .particles import VortexParticleField


class UVPMHybridSolver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """
    UVLM + VPM hybrid solver. Uses ring vortex panels on the wing and vortex
    particles for the wake.

    Overrides:
      - _calculate_wake_wing_influences: particle BS instead of ring BS
      - _populate_next_airplanes_wake: shed particles instead of rings
    """

    def __init__(self, unsteady_problem, max_particles=50000, nu=0.0, rlxf=0.3):
        super().__init__(unsteady_problem)
        self._vpm_field = VortexParticleField(
            max_particles=max_particles, nu=nu, rlxf=rlxf
        )
        self._vpm_sigma_factor = 0.1  # sigma = factor * panel_chord

    def _calculate_wake_wing_influences(self):
        """
        Override: compute wake influence on wing using vortex particles
        instead of wake ring vortices.
        """
        if self._current_step == 0 or self._vpm_field.np == 0:
            self._currentStackWakeWingInfluences__E = np.zeros(self.num_panels)
            return

        # Use global-frame collocation points from the parent class
        collocation_points = self.stackCpp_GP1_CgP1  # (N_panels, 3)

        U_wake = self._vpm_field.induce_velocity_at(collocation_points)

        self._currentStackWakeWingInfluences__E = np.einsum(
            "ij,ij->i", U_wake, self.stackUnitNormals_GP1
        )

    def _populate_next_airplanes_wake(self):
        """
        Override: shed vortex particles from trailing edge instead of
        creating wake ring vortex panels. Then advect existing particles.

        Also calls parent's wake point management to keep the wake ring vortex
        data structures populated (needed for force/moment calculations).
        """
        # Shed new particles from the just-solved trailing edge
        if self._current_step > 0:
            self._shed_particles_from_trailing_edge()
            self._advect_wake_particles()

        # Use prescribed wake for the parent's ring vortex management.
        # We don't need free-wake ring vortex advection since we use particles
        # for wake-wing influences. This avoids the expensive O(N^2) ring
        # vortex free-wake computation.
        self._prescribed_wake = True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _shed_particles_from_trailing_edge(self):
        """
        Create new vortex particles from trailing edge panels.

        Each TE panel's ring vortex strength (scalar gamma) is converted to a
        vectorial circulation Gamma = gamma * back_leg_direction, placed at the
        midpoint of the trailing edge.
        """
        strength = self._current_bound_vortex_strengths
        if strength is None:
            return

        new_positions = []
        new_gammas = []
        new_sigmas = []

        panel_idx = 0
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels

                for i in range(nc):
                    for j in range(ns):
                        panel = panels[i, j]
                        gamma_scalar = strength[panel_idx]
                        panel_idx += 1

                        # Only shed from trailing edge panels
                        if not panel.is_trailing_edge:
                            continue

                        if abs(gamma_scalar) < 1e-15:
                            continue

                        # Trailing edge midpoint (back edge of panel)
                        bl = panel.Blpp_GP1_CgP1
                        br = panel.Brpp_GP1_CgP1
                        pos = 0.5 * (bl + br)

                        # Back leg direction for circulation vector
                        back_vec = br - bl
                        back_len = np.linalg.norm(back_vec)
                        if back_len < 1e-12:
                            continue
                        back_dir = back_vec / back_len

                        Gamma_vec = gamma_scalar * back_dir

                        # Core size based on panel chord
                        fr = panel.Frpp_GP1_CgP1
                        br_pt = panel.Brpp_GP1_CgP1
                        panel_chord = np.linalg.norm(fr - br_pt)
                        sigma = max(self._vpm_sigma_factor * panel_chord, 1e-4)

                        new_positions.append(pos)
                        new_gammas.append(Gamma_vec)
                        new_sigmas.append(sigma)

        if new_positions:
            self._vpm_field.add_particles_batch(
                np.array(new_positions),
                np.array(new_gammas),
                np.array(new_sigmas),
            )

    def _advect_wake_particles(self):
        """Advect wake particles using RK3 + rVPM."""
        dt = self.delta_time

        op = self.current_operating_point
        V_inf = op.vCg__E * np.array([
            np.cos(np.radians(op.beta)) * np.cos(np.radians(op.alpha)),
            np.sin(np.radians(op.beta)),
            -np.cos(np.radians(op.beta)) * np.sin(np.radians(op.alpha)),
        ])

        def U_inf_func(positions):
            return np.broadcast_to(V_inf, positions.shape).copy()

        self._vpm_field.advect_rk3(dt, U_inf_func, bound_velocity_func=None)
