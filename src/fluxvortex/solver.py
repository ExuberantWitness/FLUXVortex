"""
UVPMHybridSolver — PteraSoftware UVLM wing solver + vortex particle wake.

Architecture (FLOWVLM-style one-way coupling):
  1. Wing aerodynamics: solved with PteraSoftware's ring-vortex UVLM (parent)
     — wake-wing influence comes from ring vortex panels (accurate, stable)
  2. VPM particles: shed from TE and advected independently
     — trailing-vortex particles (spanwise Gamma gradient × dl × freestream dir)
     — used for wake visualization and free-wake roll-up
     — do NOT feed back into the wing aerodynamics

Shedding strategy (after FLOWVLM flappingwing.jl / adds_particles_from_vlm):
  - One particle per spanwise boundary at the trailing edge
  - Circulation = (Gamma_inboard - Gamma_outboard) * dl (trailing vortex × length)
  - Direction = freestream (downstream)
  - sigma = dl (same as streamwise spacing)

Compatible with PteraSoftware 5.x.
"""
import numpy as np

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pterasoftware as ps
from .particles import VortexParticleField


class UVPMHybridSolver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """
    UVLM + VPM hybrid solver.

    Wing aerodynamics use the parent's ring-vortex UVLM (accurate wake-wing
    influence). VPM particles are shed alongside for wake visualization and
    free-wake dynamics, following FLOWVLM's one-way coupling approach.
    """

    def __init__(self, unsteady_problem, max_particles=50000, nu=0.0, rlxf=0.3,
                 stretch=True, free_wake=True):
        super().__init__(unsteady_problem)
        self._vpm_field = VortexParticleField(
            max_particles=max_particles, nu=nu, rlxf=rlxf
        )
        self._stretch = stretch
        self._free_wake = free_wake

    def _calculate_wake_wing_influences(self):
        """Delegate to parent: ring vortex panels for wake-wing influence."""
        super()._calculate_wake_wing_influences()

    def _populate_next_airplanes_wake(self):
        """Shed VPM particles, advect them, then let parent handle ring vortex wake."""
        if self._current_step > 0:
            self._shed_particles_from_trailing_edge()
            self._advect_wake_particles()

        self._prescribed_wake = True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _shed_particles_from_trailing_edge(self):
        """
        Shed trailing-vortex particles from TE spanwise boundaries.

        FLOWVLM approach: each spanwise boundary between TE panels sheds one
        particle whose circulation equals the spanwise Gamma gradient × dl,
        directed along the freestream. This represents the trailing vortex
        filament shed from that boundary.

        For a symmetric wing, the root boundary has zero gradient (symmetric
        image carries the same Gamma), so no particle is shed there.
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
        sigma = dl
        te_offset = infD * dl * 0.5

        new_positions = []
        new_gammas = []
        new_sigmas = []

        panel_idx = 0
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels

                is_symmetric = getattr(wing, 'symmetric', False)

                # Collect TE panels in spanwise order
                te_data = []
                for i in range(nc):
                    for j in range(ns):
                        panel = panels[i, j]
                        gamma = strength[panel_idx]
                        panel_idx += 1

                        if panel.is_trailing_edge:
                            te_data.append((gamma,
                                            panel.Blpp_GP1_CgP1,
                                            panel.Brpp_GP1_CgP1))

                n_te = len(te_data)
                if n_te == 0:
                    continue

                # Shed one trailing-vortex particle at each spanwise boundary
                for b in range(n_te + 1):
                    if b == 0:
                        gamma_in = te_data[0][0] if is_symmetric else 0.0
                        gamma_out = te_data[0][0]
                        pos = te_data[0][1]  # back-left of first TE panel
                    elif b == n_te:
                        gamma_in = te_data[-1][0]
                        gamma_out = 0.0
                        pos = te_data[-1][2]  # back-right of last TE panel
                    else:
                        gamma_in = te_data[b - 1][0]
                        gamma_out = te_data[b][0]
                        pos = 0.5 * (te_data[b - 1][2] + te_data[b][1])

                    mag_gamma = (gamma_in - gamma_out) * dl
                    if abs(mag_gamma) < 1e-15:
                        continue

                    Gamma_vec = mag_gamma * infD
                    new_positions.append(pos + te_offset)
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

        self._vpm_field.advect_rk3(dt, U_inf_func, bound_velocity_func=None,
                                    stretch=self._stretch, free_wake=self._free_wake)
