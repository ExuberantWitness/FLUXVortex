"""
Conservative aeroelastic coupling between ParticleMesh (XPBD) and UVLM.

Implements:
  - Conservative load transfer (area-weighted, virtual-work-equivalent)
  - Displacement transfer (ParticleMesh vertices → UVLM panel corners)
  - Implicit fixed-point iteration with Aitken relaxation
  - Energy monitoring (kinetic, potential, aero work)

References:
  - Farhat, Lesoinne & LeTallec (1998), "Load and motion transfer mechanisms"
  - Piperno & Farhat (2001), "Partitioned procedures for coupled aeroelastic problems"
  - Macklin, Müller & Chentanez (2016), "XPBD"
"""
import numpy as np

from .particle_mesh import ParticleMesh


class ParticleMeshAeroelasticSolver:
    """Implicit bidirectional aeroelastic coupling: ParticleMesh ↔ UVLM.

    The coupling strategy is partitioned implicit:
      1. UVLM computes panel aerodynamic forces from current geometry
      2. Forces are conservatively transferred to ParticleMesh vertices
      3. ParticleMesh advances one structural timestep (XPBD)
      4. Deformed mesh is mapped back to UVLM panel geometry
      5. Steps 1-4 iterate until convergence (fixed-point + Aitken relaxation)

    Parameters
    ----------
    uvpm_solver : PteraSoftware UVLMSolver instance
        The aerodynamic solver (already configured with problem).
    particle_mesh : ParticleMesh
        The structural solver.
    max_sub_iter : int
        Maximum coupling sub-iterations per timestep.
    coupling_tol : float
        Convergence tolerance on displacement change between sub-iterations.
    aitken_relaxation : bool
        Use Aitken Δ² dynamic relaxation for faster convergence.
    energy_monitor : bool
        Track energy balance at each step.
    """

    def __init__(self, uvpm_solver, particle_mesh, max_sub_iter=5,
                 coupling_tol=1e-6, aitken_relaxation=True,
                 energy_monitor=True):
        self.uvpm = uvpm_solver
        self.pm = particle_mesh
        self.max_sub_iter = max_sub_iter
        self.coupling_tol = coupling_tol
        self.aitken = aitken_relaxation
        self.energy_monitor = energy_monitor

        # Build vertex ↔ panel mapping
        self._build_panel_mapping()

        # Aitken relaxation state
        self._aitken_factor = 1.0
        self._prev_residual = None

        # Energy tracking
        self.energy_history = []
        self._cumulative_aero_work = 0.0

        # Displacement history for flutter detection
        self.displacement_history = []

    def _build_panel_mapping(self):
        """Build correspondence between UVLM quad panels and ParticleMesh triangles.

        UVLM uses quadrilateral panels on a structured grid (nc × ns).
        ParticleMesh uses triangles on the same surface.
        Each UVLM quad panel maps to exactly 2 ParticleMesh triangles.
        """
        # Get UVLM panel layout
        airplane = self.uvpm.current_airplanes[0]
        wing = airplane.wings[0]
        self._nc = wing.num_chordwise_panels
        self._ns = wing.num_spanwise_panels
        n_panels = self._nc * self._ns

        # UVLM panel collocation points and corners
        panels = wing.panels
        self._panel_y = np.zeros(n_panels)
        self._panel_x_ea = np.zeros(n_panels)
        self._panel_chord = np.zeros(n_panels)

        # Store original panel corner positions (reference for mapping)
        # UVLM panels are accessed as panels[i_chord, j_span]
        self._panel_corners = []  # list of (4, 3) arrays

        for i in range(self._nc):
            for j in range(self._ns):
                p = panels[i, j]
                idx = i * self._ns + j
                cpp = p.Cpp_GP1_CgP1
                self._panel_y[idx] = abs(cpp[1])

                # Leading and trailing edge x from panel corners
                x_le = p.Frpp_GP1_CgP1[0]
                x_te = p.Brpp_GP1_CgP1[0]
                self._panel_chord[idx] = x_te - x_le
                self._panel_x_ea[idx] = x_le + 0.25 * self._panel_chord[idx]

                # Store original corners
                corners = np.array([
                    p.Frpp_GP1_CgP1.copy(),
                    p.Flpp_GP1_CgP1.copy(),
                    p.Blpp_GP1_CgP1.copy(),
                    p.Brpp_GP1_CgP1.copy(),
                ])
                self._panel_corners.append(corners)

        self._panel_corners = np.array(self._panel_corners)  # (n_panels, 4, 3)

        # Build ParticleMesh vertex ↔ UVLM panel corner correspondence
        # The ParticleMesh grid has (nc+1)*(ns+1) vertices on a structured grid
        # UVLM panels have corners at the same grid points
        # Panel(i,j) corners → grid vertices:
        #   FR = vertex(j*(nc+1) + i+1)
        #   FL = vertex((j+1)*(nc+1) + i+1)
        #   BL = vertex((j+1)*(nc+1) + i)
        #   BR = vertex(j*(nc+1) + i)
        n_x = self._nc + 1  # chordwise vertices
        n_y = self._ns + 1  # spanwise vertices

        # Panel-to-vertex mapping: panel_idx → [fr_idx, fl_idx, bl_idx, br_idx]
        self._panel_vert_idx = np.zeros((n_panels, 4), dtype=np.int32)
        for i in range(self._nc):
            for j in range(self._ns):
                pidx = i * self._ns + j
                # FR (front-right = chord i+1, span j)
                self._panel_vert_idx[pidx, 0] = j * n_x + (i + 1)
                # FL (front-left = chord i+1, span j+1)
                self._panel_vert_idx[pidx, 1] = (j + 1) * n_x + (i + 1)
                # BL (back-left = chord i, span j+1)
                self._panel_vert_idx[pidx, 2] = (j + 1) * n_x + i
                # BR (back-right = chord i, span j)
                self._panel_vert_idx[pidx, 3] = j * n_x + i

        # Identify which triangles correspond to which panel
        # create_wing_mesh splits each quad into 2 triangles:
        #   tri_A = [p00, p10, p11] or [p00, p10, p01]
        #   tri_B = [p00, p11, p01] or [p10, p11, p01]
        # We need to find the 2 triangles whose vertices are within the 4 corners
        self._panel_tri_idx = np.zeros((n_panels, 2), dtype=np.int32)
        tri_verts = self.pm.tri_indices  # (T, 3)

        for i in range(self._nc):
            for j in range(self._ns):
                pidx = i * self._ns + j
                v_fr, v_fl, v_bl, v_br = self._panel_vert_idx[pidx]
                panel_verts = {v_fr, v_fl, v_bl, v_br}

                # Find triangles whose vertices are subset of panel_verts
                found = 0
                for t in range(len(tri_verts)):
                    tverts = set(tri_verts[t])
                    if tverts.issubset(panel_verts) and len(tverts) >= 3:
                        self._panel_tri_idx[pidx, found] = t
                        found += 1
                        if found == 2:
                            break

    # ── Conservative Load Transfer ──────────────────────────────────────

    def conservative_load_transfer(self, panel_forces):
        """Transfer UVLM panel forces to ParticleMesh vertices.

        Conservative scheme based on virtual work equivalence:
          Σ F_aero · δx_aero = Σ F_struct · δx_struct

        Each quad panel is split into 2 triangles. Forces are distributed
        using area-weighted barycentric interpolation within each triangle,
        which is equivalent to consistent (shape-function-based) force
        projection for linear triangular elements.

        Parameters
        ----------
        panel_forces : (n_panels, 3) array
            Aerodynamic force on each UVLM panel.

        Returns
        -------
        F_vert : (N, 3) array
            Conservative force on each ParticleMesh vertex.
        """
        F_vert = np.zeros((self.pm.n_particles, 3), dtype=np.float64)

        for pidx in range(len(panel_forces)):
            f_panel = panel_forces[pidx]
            tri_a, tri_b = self._panel_tri_idx[pidx]

            # Area of each triangle
            area_a = self.pm.rest_area[tri_a]
            area_b = self.pm.rest_area[tri_b]
            total_area = area_a + area_b

            if total_area < 1e-20:
                continue

            # Distribute force proportional to triangle area
            # (piecewise-constant pressure assumption within the quad)
            f_a = f_panel * (area_a / total_area)
            f_b = f_panel * (area_b / total_area)

            # Each triangle force splits equally to its 3 vertices
            # This IS the consistent force vector for constant strain triangles
            for tri_idx, f_tri in [(tri_a, f_a), (tri_b, f_b)]:
                f3 = f_tri / 3.0
                i, j, k = self.pm.tri_indices[tri_idx]
                F_vert[i] += f3
                F_vert[j] += f3
                F_vert[k] += f3

        return F_vert

    def verify_load_conservation(self, panel_forces, F_vert):
        """Verify force balance and virtual work equivalence.

        Returns (force_error, vwork_error).
        """
        # Force balance: sum of aero forces should equal sum of structural forces
        f_aero_total = np.sum(panel_forces, axis=0)
        f_struct_total = np.sum(F_vert, axis=0)
        force_error = np.linalg.norm(f_aero_total - f_struct_total)

        # Moment balance about origin
        panel_centers = np.zeros((len(panel_forces), 3))
        for pidx in range(len(panel_forces)):
            tri_a, tri_b = self._panel_tri_idx[pidx]
            c_a = (self.pm.pos[self.pm.tri_indices[tri_a, 0]] +
                   self.pm.pos[self.pm.tri_indices[tri_a, 1]] +
                   self.pm.pos[self.pm.tri_indices[tri_a, 2]]) / 3.0
            c_b = (self.pm.pos[self.pm.tri_indices[tri_b, 0]] +
                   self.pm.pos[self.pm.tri_indices[tri_b, 1]] +
                   self.pm.pos[self.pm.tri_indices[tri_b, 2]]) / 3.0
            area_a = self.pm.rest_area[tri_a]
            area_b = self.pm.rest_area[tri_b]
            panel_centers[pidx] = (c_a * area_a + c_b * area_b) / (area_a + area_b)

        m_aero = np.sum(np.cross(panel_centers, panel_forces), axis=0)
        m_struct = np.sum(np.cross(self.pm.pos, F_vert), axis=0)
        moment_error = np.linalg.norm(m_aero - m_struct)

        return force_error, moment_error

    # ── Displacement Transfer ───────────────────────────────────────────

    def displacement_transfer(self, step):
        """Map ParticleMesh vertex positions to UVLM panel corners.

        For structured meshes where ParticleMesh vertices correspond
        one-to-one with UVLM panel corners, this is a direct copy.
        """
        airplane = self.uvpm.current_airplanes[0]
        wing = airplane.wings
        if isinstance(wing, list):
            wing = wing[0]
        panels = wing.panels

        for pidx in range(len(self._panel_vert_idx)):
            i_chord = pidx // self._ns
            j_span = pidx % self._ns
            p = panels[i_chord, j_span]

            vidx = self._panel_vert_idx[pidx]
            # 4 corners: FR, FL, BL, BR
            corner_attrs = [
                ('_Frpp_GP1_CgP1', '_Frrvp_GP1_CgP1'),
                ('_Flpp_GP1_CgP1', '_Flrvp_GP1_CgP1'),
                ('_Blpp_GP1_CgP1', '_Blrvp_GP1_CgP1'),
                ('_Brpp_GP1_CgP1', '_Brrvp_GP1_CgP1'),
            ]

            for c, (pp_attr, rv_attr) in zip(vidx, corner_attrs):
                new_pos = self.pm.pos[c]

                # Update panel corner
                try:
                    v = getattr(p, pp_attr)
                    if v is not None and v.flags.writeable:
                        v[:] = new_pos
                except (AttributeError, ValueError):
                    pass

                # Update ring vortex corner
                if p.ring_vortex is not None:
                    try:
                        v = getattr(p.ring_vortex, rv_attr)
                        if v is not None and v.flags.writeable:
                            v[:] = new_pos
                    except (AttributeError, ValueError):
                        pass

            # Update collocation point (3/4 chord midpoint)
            try:
                fr = self.pm.pos[vidx[0]]
                fl = self.pm.pos[vidx[1]]
                bl = self.pm.pos[vidx[2]]
                br = self.pm.pos[vidx[3]]
                # Collocation at 3/4 chord, mid-span of the panel
                p._Cpp_GP1_CgP1[:] = 0.75 * (fr + br) / 2 + 0.25 * (fl + bl) / 2
            except (AttributeError, ValueError):
                pass

    # ── Energy Monitor ──────────────────────────────────────────────────

    def compute_energy(self):
        """Compute total structural energy.

        Returns (E_kinetic, E_spring, E_bend, E_total).
        """
        # Kinetic energy
        E_kin = 0.5 * np.sum(self.pm.particle_mass[:, None] * self.pm.vel ** 2)

        # Spring potential energy: 0.5 * ke * (|Δx| - L_rest)²
        dx = self.pm.pos[self.pm.spring_j] - self.pm.pos[self.pm.spring_i]
        lengths = np.linalg.norm(dx, axis=1)
        E_spring = 0.5 * np.sum(self.pm.spring_ke * (lengths - self.pm.spring_rest) ** 2)

        # Bending potential energy: 0.5 * edge_ke * (θ - θ_rest)²
        E_bend = 0.0
        if len(self.pm.bend_i) > 0:
            x1 = self.pm.pos[self.pm.bend_i]
            x2 = self.pm.pos[self.pm.bend_j]
            x3 = self.pm.pos[self.pm.bend_k]
            x4 = self.pm.pos[self.pm.bend_l]
            n1 = np.cross(x3 - x1, x4 - x1)
            n2 = np.cross(x4 - x2, x3 - x2)
            e = x4 - x3
            n1_len = np.linalg.norm(n1, axis=1, keepdims=True)
            n2_len = np.linalg.norm(n2, axis=1, keepdims=True)
            n1_hat = n1 / np.maximum(n1_len, 1e-12)
            n2_hat = n2 / np.maximum(n2_len, 1e-12)
            e_hat = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            cos_t = np.sum(n1_hat * n2_hat, axis=1)
            sin_t = np.sum(np.cross(n1_hat, n2_hat) * e_hat, axis=1)
            theta = np.arctan2(sin_t, cos_t)
            E_bend = 0.5 * np.sum(self.pm.edge_ke * (theta - self.pm.bend_rest_angle) ** 2)

        E_total = E_kin + E_spring + E_bend
        return E_kin, E_spring, E_bend, E_total

    # ── Aitken Relaxation ───────────────────────────────────────────────

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
                # Clamp to [0.1, 2.0] for stability
                self._aitken_factor = max(0.1, min(2.0, self._aitken_factor))
        self._prev_residual = residual.copy()

    # ── Main Coupling Step ──────────────────────────────────────────────

    def step(self, dt, n_xpbd_iter=15, gravity=True):
        """One implicit coupling timestep.

        Parameters
        ----------
        dt : float
            Timestep size.
        n_xpbd_iter : int
            XPBD iterations per structural sub-step.
        gravity : bool
            Include gravity in external forces.

        Returns
        -------
        converged : bool
            Whether coupling iteration converged.
        n_iter : int
            Number of sub-iterations used.
        """
        # Reset Aitken for new timestep
        self._aitken_factor = 1.0
        self._prev_residual = None

        pos_initial = self.pm.pos.copy()
        converged = False
        n_iter = 0

        for k in range(self.max_sub_iter):
            n_iter = k + 1

            # 1. Compute external forces
            F_ext = np.zeros((self.pm.n_particles, 3), dtype=np.float64)
            if gravity:
                F_ext += self.pm.compute_gravity_forces()

            # 2. Get panel aero forces from UVLM
            #    (UVLM must have been solved for current geometry before calling step)
            panel_forces = self._extract_panel_forces()

            # 3. Conservative load transfer
            F_aero = self.conservative_load_transfer(panel_forces)
            F_ext += F_aero

            # 4. Structural step (XPBD)
            pos_before = self.pm.pos.copy()
            self.pm.step(F_ext, dt, n_iterations=n_xpbd_iter)

            # 5. Aitken relaxation on position change
            delta_pos = self.pm.pos - pos_before
            if self.aitken and k > 0:
                self._aitken_update(delta_pos)
                # Relax: blend relaxed position
                self.pm.pos = pos_before + self._aitken_factor * delta_pos
                self.pm.vel = (self.pm.pos - pos_initial) / dt  # fixme: should use pos_old

            # 6. Transfer displacement back to UVLM
            self.displacement_transfer(self.uvpm._current_step)

            # 7. Check convergence
            residual = delta_pos
            res_norm = np.linalg.norm(residual)
            if res_norm < self.coupling_tol and k > 0:
                converged = True
                break

        # Energy monitoring
        if self.energy_monitor:
            E_kin, E_spring, E_bend, E_total = self.compute_energy()
            # Aero work increment: F_aero · Δx
            panel_forces = self._extract_panel_forces()
            F_aero = self.conservative_load_transfer(panel_forces)
            dx = self.pm.pos - pos_initial
            aero_work = np.sum(F_aero * dx)
            self._cumulative_aero_work += aero_work

            self.energy_history.append({
                'E_kinetic': E_kin,
                'E_spring': E_spring,
                'E_bend': E_bend,
                'E_total': E_total,
                'W_aero': self._cumulative_aero_work,
                'E_total_plus_W_aero': E_total - self._cumulative_aero_work,
                'converged': converged,
                'n_sub_iter': n_iter,
            })

        # Record displacement for flutter detection
        # Tip = max span, average over chord
        max_y = self.pm.pos[:, 1].max()
        tip_mask = np.abs(self.pm.pos[:, 1] - max_y) < 1e-6
        if np.any(tip_mask):
            tip_disp = self.pm.pos[tip_mask, 2].mean() - pos_initial[tip_mask, 2].mean()
            self.displacement_history.append(tip_disp)
        else:
            self.displacement_history.append(0.0)

        return converged, n_iter

    def _extract_panel_forces(self):
        """Extract per-panel forces from UVLM solver state.

        Returns (n_panels, 3) array of forces.
        """
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

        return np.array(all_forces)

    # ── Standalone mode (without PteraSoftware UVLM) ───────────────────

    def step_standalone(self, F_aero_tri, dt, n_xpbd_iter=15, gravity=True):
        """One coupling step with externally-provided aero forces.

        For testing without a running UVLM solver.
        Uses direct triangle-to-vertex force distribution.

        Parameters
        ----------
        F_aero_tri : (T, 3) array
            Aerodynamic force on each triangle.
        dt : float
            Timestep.
        n_xpbd_iter : int
            XPBD iterations.
        gravity : bool
            Include gravity.
        """
        F_ext = np.zeros((self.pm.n_particles, 3), dtype=np.float64)
        if gravity:
            F_ext += self.pm.compute_gravity_forces()
        F_ext += self.pm.distribute_force_to_vertices(F_aero_tri)

        pos_old = self.pm.pos.copy()
        self.pm.step(F_ext, dt, n_iterations=n_xpbd_iter)

        # Energy
        if self.energy_monitor:
            E_kin, E_spring, E_bend, E_total = self.compute_energy()
            dx = self.pm.pos - pos_old
            F_aero_vert = self.pm.distribute_force_to_vertices(F_aero_tri)
            aero_work = np.sum(F_aero_vert * dx)
            self._cumulative_aero_work += aero_work
            self.energy_history.append({
                'E_kinetic': E_kin,
                'E_spring': E_spring,
                'E_bend': E_bend,
                'E_total': E_total,
                'W_aero': self._cumulative_aero_work,
                'E_total_plus_W_aero': E_total - self._cumulative_aero_work,
            })

        # Tip displacement
        max_y = self.pm.pos[:, 1].max()
        tip_mask = np.abs(self.pm.pos[:, 1] - max_y) < 1e-6
        if np.any(tip_mask):
            tip_disp = self.pm.pos[tip_mask, 2].mean() - pos_old[tip_mask, 2].mean()
            self.displacement_history.append(tip_disp)
        else:
            self.displacement_history.append(0.0)
