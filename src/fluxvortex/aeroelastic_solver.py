"""
AeroelasticSolver — Coupled UVLM aerodynamics + Euler-Bernoulli beam structure.

Fast version using direct panel vertex mutation instead of full geometry rebuild.
Deforms panel vertices in-place for the next timestep, avoiding expensive
PteraSoftware object creation at each step.
"""
import numpy as np

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

import pterasoftware as ps
from .beam_fe import BeamFE


class AeroelasticSolver(
    ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver
):
    """
    Coupled aeroelastic UVLM + beam FE solver.

    At each timestep:
      1. Solve aerodynamics (parent UVLM)
      2. Extract per-panel forces per spanwise station
      3. Map to beam nodal forces
      4. Solve beam FE for deformations
      5. Directly mutate panel vertices for the next timestep
    """

    def __init__(self, unsteady_problem, beam_params, relaxation=1.0,
                 x_ea_chord=0.33):
        super().__init__(unsteady_problem)
        self._relaxation = relaxation
        self._x_ea_chord = x_ea_chord  # elastic axis at fraction of chord from LE

        self.beam = BeamFE(**beam_params)
        self._prev_w = None
        self._prev_theta = None

        # Cached spanwise station positions (set on first call)
        self._station_y = None
        self._station_indices = None

        # Tip displacement history for flutter detection
        self.tip_w_history = []
        self.tip_theta_history = []
        self.force_history = []

    def run(self, prescribed_wake=True, calculate_streamlines=False,
            show_progress=True):
        self.steady_problems = list(self.steady_problems)
        super().run(
            prescribed_wake=prescribed_wake,
            calculate_streamlines=calculate_streamlines,
            show_progress=show_progress,
        )

    def _calculate_loads(self):
        super()._calculate_loads()
        if self._current_step >= 1:
            self._structural_coupling()

    def _structural_coupling(self):
        step = self._current_step
        if step >= self.num_steps - 1:
            return

        # 1. Extract distributed forces
        y_forces, lift_forces, moment_forces = self._extract_distributed_forces()

        if len(y_forces) == 0:
            self.tip_w_history.append(0.0)
            self.tip_theta_history.append(0.0)
            return

        # Record total force magnitude for diagnostics
        self.force_history.append(np.sum(np.abs(lift_forces)))

        # 2. Map to beam nodal forces
        F_beam = BeamFE.distribute_force_to_nodes(
            self.beam.y_nodes, y_forces, lift_forces, moment_forces
        )

        # 3. Step beam FE
        dt = self.delta_time
        self.beam.step(F_beam, dt)

        # 4. Get deformations with relaxation
        w_new, theta_new = self.beam.get_nodal_displacements()
        if self._prev_w is not None:
            w_new = self._relaxation * w_new + (1 - self._relaxation) * self._prev_w
            theta_new = self._relaxation * theta_new + (1 - self._relaxation) * self._prev_theta
        self._prev_w = w_new.copy()
        self._prev_theta = theta_new.copy()

        # Record tip displacement
        self.tip_w_history.append(w_new[-1])
        self.tip_theta_history.append(theta_new[-1])

        # 5. Deform panel vertices for next timestep
        self._deform_panels(step + 1, w_new, theta_new)

    def _extract_distributed_forces(self):
        """Extract spanwise-distributed lift and moment from panel forces."""
        all_y = []
        all_lift = []
        all_moment = []

        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels

                for j in range(ns):
                    station_lift = 0.0
                    station_moment = 0.0
                    station_y = 0.0
                    count = 0

                    for i in range(nc):
                        p = panels[i, j]
                        f = getattr(p, 'forces_GP1', None)
                        if f is not None:
                            # Lift = -Fz (PteraSoftware convention: z down)
                            station_lift += -f[2]
                            # Moment about quarter-chord (x-arm * lift)
                            cpp = p.Cpp_GP1_CgP1
                            station_moment += cpp[0] * (-f[2])
                            if count == 0:
                                station_y = cpp[1]
                            count += 1

                    if count > 0 and abs(station_y) > 1e-6:
                        all_y.append(station_y)
                        all_lift.append(station_lift)
                        all_moment.append(station_moment)

        return np.array(all_y), np.array(all_lift), np.array(all_moment)

    def _deform_panels(self, step, w_deform, theta_deform):
        """Directly mutate panel vertices of steady_problems[step].

        Deforms all 4 corner vertices + ring vortex vertices of each panel
        by interpolating beam heave/twist at the panel's spanwise position.
        Twist is applied as rotation about the elastic axis (at x_ea_chord * chord).
        """
        beam_y = self.beam.y_nodes
        problem = self.steady_problems[step]

        for airplane in problem.airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels

                # Compute elastic axis x-position from leading edge panel
                x_le = panels[0, 0].Frpp_GP1_CgP1[0]
                x_te = panels[nc-1, 0].Brpp_GP1_CgP1[0]
                chord = x_te - x_le
                x_ea = x_le + self._x_ea_chord * chord

                for i in range(nc):
                    for j in range(ns):
                        p = panels[i, j]

                        # Panel spanwise position from collocation point
                        cpp = p.Cpp_GP1_CgP1
                        y_pos = abs(cpp[1])

                        # Interpolate beam deformation at this y
                        w_here = np.interp(y_pos, beam_y, w_deform)
                        theta_here = np.interp(y_pos, beam_y, theta_deform)

                        # Heave: uniform z-offset at elastic axis
                        dz = w_here
                        sin_t = np.sin(theta_here)
                        cos_t = np.cos(theta_here) - 1.0

                        # Apply deformation to all 4 corner vertices
                        for attr in ['_Frpp_GP1_CgP1', '_Flpp_GP1_CgP1',
                                     '_Blpp_GP1_CgP1', '_Brpp_GP1_CgP1']:
                            try:
                                v = getattr(p, attr)
                                if v is not None and v.flags.writeable:
                                    x_rel = v[0] - x_ea
                                    v[2] += dz + x_rel * sin_t
                                    v[0] += x_rel * cos_t
                            except (AttributeError, ValueError):
                                pass

                        # Also deform ring vortex vertices
                        if p.ring_vortex is not None:
                            rv = p.ring_vortex
                            for attr in ['_Frrvp_GP1_CgP1', '_Flrvp_GP1_CgP1',
                                         '_Blrvp_GP1_CgP1', '_Brrvp_GP1_CgP1']:
                                try:
                                    v = getattr(rv, attr)
                                    if v is not None and v.flags.writeable:
                                        x_rel = v[0] - x_ea
                                        v[2] += dz + x_rel * sin_t
                                        v[0] += x_rel * cos_t
                                except (AttributeError, ValueError):
                                    pass

                        # Update collocation point (3/4 chord midpoint)
                        try:
                            if p._Cpp_GP1_CgP1 is not None and p._Cpp_GP1_CgP1.flags.writeable:
                                x_rel = p._Cpp_GP1_CgP1[0] - x_ea
                                p._Cpp_GP1_CgP1[2] += dz + x_rel * sin_t
                        except (AttributeError, ValueError):
                            pass
