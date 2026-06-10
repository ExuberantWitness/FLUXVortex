"""ANCFShell + UVPMHybridSolver (hybrid panel-particle wake) aeroelastic coupling.

Extends UVPMHybridSolver with ANCF structural coupling, enabling:
  - Near-field ring vortex panels + far-field VPM free-wake particles
  - Consistent load/displacement transfer between ANCF shell and UVLM
  - Visualization data capture for comparison with Yamano et al.

Reference parameters (Yamano single sheet, clamped LE):
  L=1.0, W=1.0 (AR=1), h=1e-3, M*=1.0, U*=15
"""

import numpy as np
import os
import time as time_mod

from .solver import UVPMHybridSolver
from .ancf_shell import ANCFShell, NDOF_NODE, NDOF_ELEM, _shape_funcs


class ANCFHybridAeroelasticSolver(UVPMHybridSolver):
    """ANCF shell + hybrid panel-particle VPM aeroelastic solver.

    Inherits UVPMHybridSolver for:
      - Ring-vortex panel wake (UVLM, accurate near-field)
      - VPM particle shedding, advection, stretching (free-wake far-field)
      - GPU Biot-Savart via Warp (when patched)

    Adds ANCF structural coupling:
      - Consistent force transfer (virtual work on shape functions)
      - Accurate displacement transfer (ANCF shape function interpolation)
      - Implicit Newmark-β or explicit Velocity-Verlet structural solve
    """

    def __init__(self, unsteady_problem, shell, integrator='implicit',
                 relaxation=0.5, structural_dt_ratio=2,
                 newton_tol=1e-6, max_newton=15,
                 max_particles=50000, nu=0.0, rlxf=0.3):
        UVPMHybridSolver.__init__(self, unsteady_problem,
                                   max_particles=max_particles,
                                   nu=nu, rlxf=rlxf)
        self.shell = shell
        self._integrator = integrator
        self._relaxation = relaxation
        self._struct_ratio = structural_dt_ratio
        self._newton_tol = newton_tol
        self._max_newton = max_newton

        # ── Mesh mapping ──
        self._n_aero_chord = None
        self._n_aero_span = None
        self._panel_to_elem = None
        self._panel_xi_eta = None
        self._corner_xi_eta = {}
        self._build_panel_mapping()

        # ── Reference ANCF DOFs (cached for displacement transfer) ──
        self._q_ref = np.zeros(self.shell.ndof)
        for n in range(self.shell.nn):
            base = n * NDOF_NODE
            self._q_ref[base:base + 3] = self.shell.nodes[n]
            self._q_ref[base + 3:base + 6] = [1.0, 0.0, 0.0]
            self._q_ref[base + 6:base + 9] = [0.0, 1.0, 0.0]

        # ── Initial pulse (Yamano half-sine) ──
        self._pulse_amplitude = None
        self._pulse_duration = 0.02
        self._pulse_start_time = None

        # ── Results tracking ──
        self.tip_w_history = []
        self.tip_theta_history = []
        self.force_history = []
        self.snapshots = {}  # step → {nodes, particles, velocities, ...}

    # ─── Mesh mapping (same as ANCFAeroelasticSolver) ──────────────────

    def _build_panel_mapping(self):
        if len(self.steady_problems) == 0:
            raise RuntimeError("No steady problems available.")
        prob0 = self.steady_problems[0]
        for airplane in prob0.airplanes:
            for wing in airplane.wings:
                self._n_aero_chord = wing.num_chordwise_panels
                self._n_aero_span = wing.num_spanwise_panels
                break
            break
        nc, ns = self._n_aero_chord, self._n_aero_span
        if nc is None or ns is None:
            raise RuntimeError("Could not determine UVLM panel layout.")

        self._panel_to_elem = np.full((nc, ns), -1, dtype=np.int32)
        self._panel_xi_eta = np.zeros((nc, ns, 2))

        elem_bbox = []
        for e in range(self.shell.ne):
            nd = self.shell.quads[e]
            elem_bbox.append((self.shell.nodes[nd, 0].min(),
                              self.shell.nodes[nd, 0].max(),
                              self.shell.nodes[nd, 1].min(),
                              self.shell.nodes[nd, 1].max()))

        p0_airplanes = prob0.airplanes
        for i in range(nc):
            for j in range(ns):
                p = None
                for airplane in p0_airplanes:
                    for wing in airplane.wings:
                        try:
                            p = wing.panels[i, j]
                        except (IndexError, TypeError):
                            pass
                if p is None:
                    continue
                cx, cy = p.Cpp_GP1_CgP1[0], abs(p.Cpp_GP1_CgP1[1])
                for e, (xmin, xmax, ymin, ymax) in enumerate(elem_bbox):
                    if xmin <= cx <= xmax and ymin <= cy <= ymax:
                        self._panel_to_elem[i, j] = e
                        dL = self.shell._dL[e]
                        dW = self.shell._dW[e]
                        self._panel_xi_eta[i, j] = [
                            (cx - xmin) / dL if dL > 1e-15 else 0.5,
                            (cy - ymin) / dW if dW > 1e-15 else 0.5]
                        break

        for name in ['Frpp', 'Flpp', 'Blpp', 'Brpp',
                     'Frrvp', 'Flrvp', 'Blrvp', 'Brrvp']:
            self._corner_xi_eta[name] = np.zeros((nc, ns, 2))
        self._corner_xi_eta['Frpp'][:, :] = [1.0, 1.0]
        self._corner_xi_eta['Flpp'][:, :] = [1.0, 0.0]
        self._corner_xi_eta['Blpp'][:, :] = [0.0, 0.0]
        self._corner_xi_eta['Brpp'][:, :] = [0.0, 1.0]
        self._corner_xi_eta['Frrvp'][:, :] = [1.0, 1.0]
        self._corner_xi_eta['Flrvp'][:, :] = [1.0, 0.0]
        self._corner_xi_eta['Blrvp'][:, :] = [0.0, 0.0]
        self._corner_xi_eta['Brrvp'][:, :] = [0.0, 1.0]

        n_mapped = np.sum(self._panel_to_elem >= 0)
        print(f"[ancf_hybrid] {n_mapped}/{nc*ns} panels mapped")

    def _get_panel(self, i, j):
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                try:
                    return wing.panels[i, j]
                except (IndexError, TypeError):
                    pass
        return None

    # ─── Override _calculate_loads ─────────────────────────────────────

    def _calculate_loads(self):
        super()._calculate_loads()
        if self._current_step >= 1:
            self._structural_coupling()

    # ─── Main coupling logic ───────────────────────────────────────────

    def _structural_coupling(self):
        step = self._current_step
        if step >= self.num_steps - 1:
            return

        dt_struct = self.delta_time / self._struct_ratio
        prev_q = self.shell.q.copy()

        for sub in range(self._struct_ratio):
            panel_forces = self._extract_panel_forces()
            F_struct = self._load_transfer(panel_forces)
            F_struct = self._apply_initial_perturbation(F_struct)

            if self._integrator == 'implicit':
                self.shell.step_newmark(F_struct, dt_struct,
                                         newton_tol=self._newton_tol,
                                         max_newton=self._max_newton)
            else:
                self.shell.step(F_struct, dt_struct)

            if self._pulse_start_time is not None:
                self._pulse_start_time += dt_struct

        if self._relaxation < 1.0:
            self.shell.q = prev_q + self._relaxation * (self.shell.q - prev_q)
            self.shell.dq *= self._relaxation

        if step + 1 < self.num_steps:
            self._displacement_transfer(step + 1)

        self._record_history()
        self.force_history.append(
            np.sum(np.abs(panel_forces)) if len(panel_forces) > 0 else 0.0)

    # ─── Force extraction & load transfer ─────────────────────────────

    def _extract_panel_forces(self):
        nc, ns = self._n_aero_chord, self._n_aero_span
        forces = np.zeros((nc, ns, 3))
        for i in range(nc):
            for j in range(ns):
                p = self._get_panel(i, j)
                if p is not None:
                    f = getattr(p, 'forces_GP1', None)
                    if f is not None:
                        forces[i, j] = f
        return forces

    def _load_transfer(self, panel_forces):
        F_struct = np.zeros(self.shell.ndof)
        nc, ns = self._n_aero_chord, self._n_aero_span
        for i in range(nc):
            for j in range(ns):
                f_panel = panel_forces[i, j]
                if np.all(f_panel == 0.0):
                    continue
                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue
                xi, eta = self._panel_xi_eta[i, j]
                dL, dW = self.shell._dL[e], self.shell._dW[e]
                S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
                Q_e = S.T @ f_panel
                dofs = self.shell._elem_dofs(e)
                F_struct[dofs] += Q_e
        return F_struct

    # ─── Displacement transfer ─────────────────────────────────────────

    def _displacement_transfer(self, next_step):
        problem = self.steady_problems[next_step]
        q_ref = self._q_ref

        for airplane in problem.airplanes:
            for wing in airplane.wings:
                nc, ns = wing.num_chordwise_panels, wing.num_spanwise_panels
                for i in range(nc):
                    for j in range(ns):
                        p = wing.panels[i, j]
                        e = self._panel_to_elem[i, j]
                        if e < 0:
                            continue
                        dL = self.shell._dL[e]
                        dW = self.shell._dW[e]
                        dofs = self.shell._elem_dofs(e)
                        q_e = self.shell.q[dofs]
                        q_ref_e = q_ref[dofs]

                        for corner_attr, rv_attr in [
                            ('_Frpp_GP1_CgP1', '_Frrvp_GP1_CgP1'),
                            ('_Flpp_GP1_CgP1', '_Flrvp_GP1_CgP1'),
                            ('_Blpp_GP1_CgP1', '_Blrvp_GP1_CgP1'),
                            ('_Brpp_GP1_CgP1', '_Brrvp_GP1_CgP1'),
                        ]:
                            name = corner_attr.lstrip('_').split('_')[0]
                            xi_c, eta_c = self._corner_xi_eta[name][i, j]
                            S = np.kron(_shape_funcs(xi_c, eta_c, dL, dW), np.eye(3))
                            disp = S @ q_e - S @ q_ref_e
                            try:
                                v = getattr(p, corner_attr)
                                if v is not None:
                                    v.flags.writeable = True
                                    v[:] += disp
                                    v.flags.writeable = False
                            except (AttributeError, ValueError, TypeError):
                                pass
                            try:
                                rv_obj = p.ring_vortex
                                if rv_obj is not None:
                                    v_rv = getattr(rv_obj, rv_attr)
                                    if v_rv is not None:
                                        v_rv.flags.writeable = True
                                        v_rv[:] += disp
                                        v_rv.flags.writeable = False
                            except (AttributeError, ValueError, TypeError):
                                pass

                        try:
                            cpp = p._Cpp_GP1_CgP1
                            if cpp is not None:
                                xi_c, eta_c = self._panel_xi_eta[i, j]
                                S = np.kron(_shape_funcs(xi_c, eta_c, dL, dW), np.eye(3))
                                disp_c = S @ q_e - S @ q_ref_e
                                cpp.flags.writeable = True
                                cpp[:] += disp_c
                                cpp.flags.writeable = False
                        except (AttributeError, ValueError, TypeError):
                            pass

    # ─── Initial perturbation (Yamano half-sine pulse) ─────────────────

    def set_initial_pulse(self, amplitude=1.0, duration=0.02):
        """Configure half-sine force pulse matching Yamano's initial disturbance.

        Yamano uses: q_in(t) = 0.5 * sin(pi * t / 0.2) * (t < 0.2)  [nondim]
        Dimensional: amplitude at tip (N), duration in seconds.
        """
        self._pulse_amplitude = amplitude
        self._pulse_duration = duration
        self._pulse_start_time = None  # set on first structural coupling

    def _apply_initial_perturbation(self, F_struct):
        """Add half-sine force pulse to tip during the initial transient window."""
        if self._pulse_amplitude is None or self._pulse_amplitude == 0:
            return F_struct

        # Track elapsed structural time
        if self._pulse_start_time is None:
            self._pulse_start_time = 0.0

        # Compute current cumulative structural time
        elapsed = self._pulse_start_time
        if elapsed >= self._pulse_duration:
            self._pulse_amplitude = 0  # pulse finished
            return F_struct

        # Half-sine pulse: F(t) = A * sin(pi * t / T) for t < T
        t = elapsed
        T = self._pulse_duration
        force_scale = np.sin(np.pi * t / T)

        # Apply to all trailing-edge nodes
        y_max = self.shell.nodes[:, 1].max()
        tip_nodes = np.where(np.abs(self.shell.nodes[:, 1] - y_max) < 1e-6)[0]
        x_max_tip = self.shell.nodes[tip_nodes, 0].max()

        for n in tip_nodes:
            base = n * NDOF_NODE
            if abs(self.shell.nodes[n, 0] - x_max_tip) < 1e-6:
                F_struct[base + 2] += self._pulse_amplitude * force_scale

        return F_struct

    # ─── Diagnostics ───────────────────────────────────────────────────

    def _record_history(self):
        y_max = self.shell.nodes[:, 1].max()
        x_max = self.shell.nodes[:, 0].max()
        tip_mask = (np.abs(self.shell.nodes[:, 1] - y_max) < 1e-6) & \
                   (np.abs(self.shell.nodes[:, 0] - x_max) < 1e-6)
        if np.any(tip_mask):
            tip_idx = np.where(tip_mask)[0][0]
            base = tip_idx * NDOF_NODE
            ref_z = self.shell.nodes[tip_idx, 2]
            self.tip_w_history.append(self.shell.q[base + 2] - ref_z)
            # Tip pitch angle from dy_r_z slope
            dy_rz = self.shell.q[base + 8]
            self.tip_theta_history.append(np.arctan2(dy_rz, 1.0))
        else:
            self.tip_w_history.append(0.0)
            self.tip_theta_history.append(0.0)

    def get_vpm_particles(self):
        """Return current VPM particle state for visualization."""
        return {
            'positions': self._vpm_field._pos[:self._vpm_field.np].copy(),
            'gamma': self._vpm_field._gamma[:self._vpm_field.np].copy(),
            'sigma': self._vpm_field._sigma[:self._vpm_field.np].copy(),
            'age': self._vpm_field._age[:self._vpm_field.np].copy(),
            'np': self._vpm_field.np,
        }

    def get_sheet_surface(self):
        """Return ANCF sheet surface mesh for visualization."""
        nodes = self.shell.positions()
        quads = self.shell.quads
        return nodes, quads

    def get_wake_ring_geometry(self):
        """Return ring vortex wake panel corner positions."""
        wake_data = []
        if hasattr(self, '_current_wake_vortex_points'):
            wake_data = self._current_wake_vortex_points
        return wake_data

    # ─── Snapshot capture ──────────────────────────────────────────────

    def capture_snapshot(self):
        """Save full visualization state at current step."""
        step = self._current_step
        nodes, quads = self.get_sheet_surface()
        vpm = self.get_vpm_particles()
        self.snapshots[step] = {
            'step': step,
            'time': step * self.delta_time,
            'nodes': nodes.copy(),
            'quads': quads.copy(),
            'tip_w': self.tip_w_history[-1] if self.tip_w_history else 0.0,
            'vpm_positions': vpm['positions'].copy(),
            'vpm_gamma': vpm['gamma'].copy(),
            'vpm_np': vpm['np'],
        }

    def save_snapshots(self, output_dir):
        """Save all captured snapshots to disk."""
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(output_dir, 'snapshots.npz'),
            steps=np.array(list(self.snapshots.keys())),
            snapshots=np.array(list(self.snapshots.values()), dtype=object),
            tip_w_history=np.array(self.tip_w_history),
            tip_theta_history=np.array(self.tip_theta_history),
            force_history=np.array(self.force_history),
        )
        print(f"[ancf_hybrid] Saved {len(self.snapshots)} snapshots to {output_dir}")


# ─── Yamano single-sheet parameter set ─────────────────────────────────

def yamano_single_sheet_params(U_star=25.0, M_star=1.0, AR=1.0,
                                V_inf=10.0, rho_fluid=1.225):
    """Compute dimensional parameters matching Yamano's nondimensional setup.

    Yamano et al. (J Sound Vib 2020) single sheet, clamped LE:
      M* = ρ_f * L / (ρ_m * h) = 1.0
      U* = V_inf * sqrt(ρ_m * h / (E * I / L³))  = 25.0 (flutter condition)

    Nondimensional stiffness: eta_m = mu_m / U*² where mu_m = 1/M* = ρ_m*h/(ρ_f*L)

    Returns dict of physical parameters for ANCF shell + UVLM setup.
    """
    L = 1.0  # chord length [m]
    W = L * AR  # span [m]
    h = 1e-3  # thickness [m]

    # Density from mass ratio: M* = ρ_f * L / (ρ_m * h)
    rho_struct = rho_fluid * L / (M_star * h)

    # Young's modulus from reduced velocity:
    # U*² = 12 * ρ_struct * L² * V² / (E * h²)
    # → E = 12 * ρ_struct * L² * V² / (U*² * h²)
    E = 12.0 * rho_struct * L**2 * V_inf**2 / (U_star**2 * h**2)

    # Nondimensional parameters (for comparison with Yamano's param_setting.m)
    mu_m = rho_struct * h / (rho_fluid * L)  # = 1/M*
    I_sec = W * h**3 / 12.0
    EI = E * I_sec
    eta_m = EI / (rho_fluid * V_inf**2 * L**3 * W)

    # Structural natural frequency (beam approximation)
    m_per_length = rho_struct * W * h
    omega1_beam = 1.875**2 * np.sqrt(EI / (m_per_length * L**4))
    freq1_beam = omega1_beam / (2 * np.pi)

    params = {
        'Length': L, 'Width': W, 'thickness': h,
        'rho': rho_struct, 'E': E, 'nu': 0.3,
        'V_inf': V_inf, 'rho_fluid': rho_fluid,
        'U_star': U_star, 'M_star': M_star,
        'mu_m': mu_m, 'eta_m': eta_m,
        'freq1_beam': freq1_beam,
    }
    return params


def print_yamano_params(params):
    """Print parameter summary for comparison with Yamano."""
    print("=" * 60)
    print("Yamano Single-Sheet Parameter Set (J Sound Vib 2020, Mech Eng J 2021)")
    print("=" * 60)
    key_params = ['U_star', 'M_star', 'mu_m', 'eta_m', 'V_inf', 'rho_fluid',
                  'Length', 'Width', 'thickness', 'rho', 'E', 'freq1_beam']
    for k in key_params:
        if k in params:
            v = params[k]
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4e}")
            else:
                print(f"  {k:20s}: {v}")
    print(f"  {'Target U*':20s}: 25.0  (single sheet, clamped LE, AR=1)")
    print(f"  {'Target M*':20s}: 1.0")
    print(f"  {'Target alpha':20s}: 0°  (flat plate aligned with flow)")
    print(f"  {'Wake truncation':20s}: 5.5 chords")
    print("=" * 60)


# ─── Velocity field computation for streamlines ────────────────────────

def compute_velocity_field(solver, x_grid, y_grid, z_plane=0.0,
                            include_bound=True, include_wake=True,
                            include_particles=True, include_freestream=True):
    """Compute velocity field on a 2D slice plane for streamlines.

    Uses the solver's current state (bound circulation + wake geometry + VPM particles).

    Parameters
    ----------
    solver : ANCFHybridAeroelasticSolver
        Active solver with current aerodynamic state.
    x_grid, y_grid : ndarray
        1D coordinate arrays defining the sampling grid.
    z_plane : float
        z-coordinate of the slice plane.
    include_bound, include_wake, include_particles, include_freestream : bool
        Which velocity components to include.

    Returns
    -------
    X, Y : ndarray
        Meshgrid coordinates.
    U, V, W : ndarray
        Velocity components at each grid point.
    """
    from .kernel import velocity_from_particles

    X, Y = np.meshgrid(x_grid, y_grid)
    n_pts = X.size
    points = np.column_stack([X.ravel(), Y.ravel(), np.full(n_pts, z_plane)])

    V_total = np.zeros((n_pts, 3))

    # Freestream
    if include_freestream:
        op = solver.current_operating_point
        V_inf_vec = np.array([
            op.vCg__E * np.cos(np.radians(op.beta)) * np.cos(np.radians(op.alpha)),
            op.vCg__E * np.sin(np.radians(op.beta)),
            -op.vCg__E * np.cos(np.radians(op.beta)) * np.sin(np.radians(op.alpha)),
        ])
        V_total += V_inf_vec

    # VPM particle contribution
    if include_particles and solver._vpm_field.np > 0:
        vpm = solver.get_vpm_particles()
        V_particles = velocity_from_particles(
            points, vpm['positions'][:vpm['np']],
            vpm['gamma'][:vpm['np']], vpm['sigma'][:vpm['np']])
        V_total += V_particles

    # Note: ring-vortex contributions require calling into PteraSoftware's
    # Biot-Savart functions. For now, we capture freestream + VPM particles
    # as the dominant far-field contributors. Ring vortex panels contribute
    # primarily in the near-field and require the collapsed/expanded BS kernels.

    U = V_total[:, 0].reshape(X.shape)
    V = V_total[:, 1].reshape(X.shape)
    W = V_total[:, 2].reshape(X.shape)

    return X, Y, U, V, W
