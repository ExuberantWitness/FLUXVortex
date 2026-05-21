"""
UVPMHybridSolver — PteraSoftware UVLM wing solver + vortex particle wake.

Inherits UnsteadyRingVortexLatticeMethodSolver and overrides only the wake-related
methods to use vortex particles instead of wake ring vortex panels.

Shedding combines two mechanisms:
  1. Trailing vortex (FLOWVLM-style): spanwise circulation gradient at boundaries
  2. TE bound vortex: back-edge ring vortex contribution at panel midpoints

Compatible with PteraSoftware 5.x.
"""
import numpy as np

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pterasoftware as ps
from .particles import VortexParticleField


class UVPMHybridSolver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """
    UVLM + VPM hybrid solver. Uses ring vortex panels on the wing and vortex
    particles for the wake.
    """

    def __init__(self, unsteady_problem, max_particles=50000, nu=0.0, rlxf=0.3):
        super().__init__(unsteady_problem)
        self._vpm_field = VortexParticleField(
            max_particles=max_particles, nu=nu, rlxf=rlxf
        )

    def _calculate_wake_wing_influences(self):
        """Compute wake influence on wing using vortex particles."""
        if self._current_step == 0 or self._vpm_field.np == 0:
            self._currentStackWakeWingInfluences__E = np.zeros(self.num_panels)
            return

        collocation_points = self.stackCpp_GP1_CgP1
        U_wake = self._vpm_field.induce_velocity_at(collocation_points)
        self._currentStackWakeWingInfluences__E = np.einsum(
            "ij,ij->i", U_wake, self.stackUnitNormals_GP1
        )

    def _populate_next_airplanes_wake(self):
        """Shed particles from TE, advect them, maintain parent wake data."""
        if self._current_step > 0:
            self._shed_particles_from_trailing_edge()
            self._advect_wake_particles()

        self._prescribed_wake = True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _shed_particles_from_trailing_edge(self):
        """
        Shed vortex particles from trailing edge panels.

        Each TE panel sheds one particle representing its back-edge ring vortex.
        The wake ring front leg opposes the bound ring back leg (Kutta condition):
          Gamma_vec = -gamma * back_edge_vector

        The edge vector (not unit direction) is used so Gamma has correct units
        (m^3/s) for the Gaussian-erf Biot-Savart kernel.
        """
        strength = self._current_bound_vortex_strengths
        if strength is None:
            return

        op = self.current_operating_point
        V_inf_vec = op.vCg__E * np.array([
            np.cos(np.radians(op.beta)) * np.cos(np.radians(op.alpha)),
            np.sin(np.radians(op.beta)),
            -np.cos(np.radians(op.beta)) * np.sin(np.radians(op.alpha)),
        ])
        V_inf = np.linalg.norm(V_inf_vec)
        if V_inf < 1e-10:
            return
        infD = V_inf_vec / V_inf

        dt = self.delta_time
        dl = V_inf * dt
        sigma = dl * 0.5  # core radius ~ half shedding length
        te_offset = infD * dl * 0.25

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
                        gamma = strength[panel_idx]
                        panel_idx += 1

                        if not panel.is_trailing_edge:
                            continue

                        if abs(gamma) < 1e-15:
                            continue

                        bl = panel.Blpp_GP1_CgP1
                        br = panel.Brpp_GP1_CgP1
                        back_vec = br - bl

                        pos = 0.5 * (bl + br) + te_offset
                        Gamma_vec = -back_vec * gamma

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
