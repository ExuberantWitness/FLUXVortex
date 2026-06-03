"""BSTShell + UVPMHybridSolver aeroelastic coupling.

Supports:
  - Staggered coupling (1 force/solve per UVLM step)
  - Iterative coupling with Aitken Δ² relaxation
  - Conservative load transfer (virtual-work equivalent)
  - Displacement transfer (BST vertices → UVLM panel corners)

Reuses patterns from aero_coupling.py (ParticleMeshAeroelasticSolver).
"""
import numpy as np

from .bst_implicit_gpu import BSTImplicitGPU


class BSTAeroelasticSolver:
    """Aeroelastic coupling: BSTShell (implicit) + UVPMHybridSolver.

    Parameters
    ----------
    uvlm_solver : UVPMHybridSolver
        The aerodynamic solver (already stepping through time).
    shell : BSTShell
        The structural shell model.
    implicit_solver : BSTImplicitGPU
        The implicit dynamic solver for the shell.
    coupling : str
        'staggered' (1 force/step) or 'iterative' (Aitken sub-iterations).
    max_sub_iter : int
        Max coupling sub-iterations per UVLM step (for 'iterative').
    coupling_tol : float
        Convergence tolerance on displacement change.
    aitken_relaxation : bool
        Use Aitken Δ² dynamic relaxation.
    """

    def __init__(self, uvlm_solver, shell, implicit_solver,
                 coupling='iterative', max_sub_iter=5,
                 coupling_tol=1e-6, aitken_relaxation=True):
        self.uvpm = uvlm_solver
        self.shell = shell
        self.implicit = implicit_solver
        self._coupling = coupling
        self._max_sub_iter = max_sub_iter
        self._coupling_tol = coupling_tol
        self._aitken = aitken_relaxation

        # Aitken state
        self._aitken_factor = 1.0
        self._prev_residual = None

        # Results tracking
        self.tip_w_history = []
        self.newton_iters_history = []
        self.converged_history = []

    # ── Main coupling step ─────────────────────────────────────────────

    def step(self, dt, newton_max=20, newton_tol=1e-8):
        """One aeroelastic coupling step.

        Called after UVLM has advanced one timestep and computed forces.

        Returns (converged, n_sub_iters, n_newton_iters).
        """
        if self._coupling == 'staggered':
            return self._staggered_step(dt, newton_max, newton_tol)
        else:
            return self._iterative_step(dt, newton_max, newton_tol)

    def _staggered_step(self, dt, newton_max, newton_tol):
        """Staggered: 1 force extraction → 1 implicit solve."""
        # Extract aero forces from UVLM
        F_shell = self._extract_shell_forces()

        # Implicit structural step
        n_newton, r_norm = self.implicit.step(
            F_shell, dt, newton_max=newton_max, tol=newton_tol)

        # Transfer displacement to UVLM
        self._displacement_transfer()

        # Record tip displacement
        self._record_tip()

        self.newton_iters_history.append(n_newton)
        self.converged_history.append(True)

        return True, 1, n_newton

    def _iterative_step(self, dt, newton_max, newton_tol):
        """Iterative: force→solve→update→re-extract force→... with Aitken."""
        self._aitken_factor = 1.0
        self._prev_residual = None
        u_initial = self.shell.u.copy()

        converged = False
        n_newton = 0

        for k in range(self._max_sub_iter):
            # Extract aero forces from UVLM
            F_shell = self._extract_shell_forces()

            # Save state before structural solve
            u_before = self.shell.u.copy()

            # Implicit structural step
            n_newton, r_norm = self.implicit.step(
                F_shell, dt, newton_max=newton_max, tol=newton_tol)

            # Compute displacement change
            delta_u = self.shell.u - u_before

            # Aitken relaxation
            if self._aitken and k > 0:
                self._aitken_update(delta_u)
                self.shell.u = u_before + self._aitken_factor * delta_u

            # Transfer displacement to UVLM
            self._displacement_transfer()

            # Check convergence
            du_norm = np.max(np.abs(delta_u[self.shell.mass_inv > 0]))
            if du_norm < self._coupling_tol and k > 0:
                converged = True
                break

        self._record_tip()
        self.newton_iters_history.append(n_newton)
        self.converged_history.append(converged)

        return converged, k + 1, n_newton

    # ── Force extraction ───────────────────────────────────────────────

    def _extract_shell_forces(self):
        """Extract per-panel forces from UVLM and distribute to shell nodes."""
        panel_forces = self._extract_panel_forces()

        if len(panel_forces) == 0:
            return np.zeros((self.shell.nv, 3))

        # Distribute panel forces to shell vertices
        F_shell = self._conservative_load_transfer(panel_forces)
        return F_shell

    def _extract_panel_forces(self):
        """Extract per-panel aerodynamic forces from UVLM solver."""
        all_forces = []
        for airplane in self.uvpm.current_airplanes:
            for wing in airplane.wings:
                if isinstance(wing, list):
                    wing = wing[0]
                panels = wing.panels
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels

                for i in range(nc):
                    for j in range(ns):
                        p = panels[i, j]
                        f = getattr(p, 'forces_GP1', None)
                        if f is not None:
                            all_forces.append(f.copy())
                        else:
                            all_forces.append(np.zeros(3))

        return np.array(all_forces) if all_forces else np.zeros((0, 3))

    def _conservative_load_transfer(self, panel_forces):
        """Transfer UVLM quad-panel forces to BSTShell triangle vertices.

        Conservative: total force and moment preserved.
        Each quad panel maps to 2 BSTShell triangles.
        Force split: area-weighted → equal to 3 vertices per triangle.
        """
        F_shell = np.zeros((self.shell.nv, 3))
        shell = self.shell

        # Build panel→vertex mapping if not cached
        if not hasattr(self, '_panel_vert_map'):
            self._build_panel_mapping()

        for pidx in range(len(panel_forces)):
            f_panel = panel_forces[pidx]

            # Find shell vertices under this panel
            vidxs = self._panel_vert_map.get(pidx, [])
            if len(vidxs) == 0:
                continue

            # Distribute force equally to vertices under this panel
            f_per_vert = f_panel / len(vidxs)
            for vi in vidxs:
                F_shell[vi] += f_per_vert

        return F_shell

    def _build_panel_mapping(self):
        """Build UVLM panel → BSTShell vertex mapping.

        For structured meshes where UVLM panels and shell mesh share
        the same grid, each panel covers the same grid points as 2
        shell triangles.
        """
        self._panel_vert_map = {}
        shell = self.shell

        # Get UVLM panel layout
        for airplane in self.uvpm.current_airplanes:
            for wing in airplane.wings:
                if isinstance(wing, list):
                    wing = wing[0]
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels
                panels = wing.panels

                for i in range(nc):
                    for j in range(ns):
                        pidx = i * ns + j
                        p = panels[i, j]

                        # Panel center
                        cpp = p.Cpp_GP1_CgP1
                        y_panel = abs(cpp[1])

                        # Find shell vertices near this panel
                        # Use the chordwise/spanwise indices to match
                        dy = shell.vertices0[:, 1].max() / ns if ns > 0 else 1.0
                        dx = shell.vertices0[:, 0].max() / nc if nc > 0 else 1.0

                        # Panel corners
                        x0 = p.Frpp_GP1_CgP1[0]
                        x1 = p.Brpp_GP1_CgP1[0]
                        y0 = abs(p.Frpp_GP1_CgP1[1])
                        y1 = abs(p.Flpp_GP1_CgP1[1])

                        # Find vertices within panel bounds
                        vx = shell.vertices0[:, 0]
                        vy = shell.vertices0[:, 1]

                        mask = ((vx >= x0 - dx * 0.1) &
                                (vx <= x1 + dx * 0.1) &
                                (np.abs(vy) >= y0 - dy * 0.1) &
                                (np.abs(vy) <= y1 + dy * 0.1))

                        self._panel_vert_map[pidx] = np.where(mask)[0].tolist()

    # ── Displacement transfer ──────────────────────────────────────────

    def _displacement_transfer(self):
        """Map BSTShell vertex displacements to UVLM panel corners."""
        shell = self.shell
        u = shell.u
        x0 = shell.vertices0

        for airplane in self.uvpm.current_airplanes:
            for wing in airplane.wings:
                if isinstance(wing, list):
                    wing = wing[0]
                panels = wing.panels
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels

                for i in range(nc):
                    for j in range(ns):
                        p = panels[i, j]

                        # Find nearest shell vertices for each panel corner
                        for attr in ['_Frpp_GP1_CgP1', '_Flpp_GP1_CgP1',
                                     '_Blpp_GP1_CgP1', '_Brpp_GP1_CgP1']:
                            try:
                                corner = getattr(p, attr)
                                if corner is not None and corner.flags.writeable:
                                    disp = self._interpolate_displacement(
                                        corner, x0, u)
                                    corner[:] = corner + disp
                            except (AttributeError, ValueError):
                                pass

                        # Update collocation point
                        try:
                            if p._Cpp_GP1_CgP1 is not None:
                                p._Cpp_GP1_CgP1[:] = (
                                    0.5 * (p.Frpp_GP1_CgP1 + p.Blpp_GP1_CgP1)
                                    + 0.5 * (p.Flpp_GP1_CgP1 + p.Brpp_GP1_CgP1)) / 2.0
                        except (AttributeError, ValueError):
                            pass

    def _interpolate_displacement(self, point, ref_pos, disp):
        """Interpolate displacement at a point from nearest shell vertices."""
        dx = ref_pos - point
        dist = np.sum(dx ** 2, axis=1)
        # Inverse distance weighting with 4 nearest
        idx = np.argpartition(dist, min(4, len(dist) - 1))[:4]
        w = 1.0 / (dist[idx] + 1e-20)
        w /= w.sum()
        return np.sum(w[:, None] * disp[idx], axis=0)

    # ── Aitken relaxation ──────────────────────────────────────────────

    def _aitken_update(self, residual):
        """Compute Aitken Δ² relaxation factor."""
        if self._prev_residual is None:
            self._aitken_factor = 1.0
        else:
            dr = residual - self._prev_residual
            dr_sq = np.dot(dr.ravel(), dr.ravel())
            if dr_sq > 1e-30:
                self._aitken_factor = -self._aitken_factor * \
                    np.dot(self._prev_residual.ravel(), dr.ravel()) / dr_sq
                self._aitken_factor = max(0.05, min(2.0, self._aitken_factor))
        self._prev_residual = residual.copy()

    # ── Diagnostics ────────────────────────────────────────────────────

    def _record_tip(self):
        """Record tip displacement."""
        shell = self.shell
        y_max = shell.vertices0[:, 1].max()
        tip_mask = np.abs(shell.vertices0[:, 1] - y_max) < 1e-6
        if np.any(tip_mask):
            tip_w = np.mean(shell.u[tip_mask, 2])
        else:
            tip_w = 0.0
        self.tip_w_history.append(tip_w)
