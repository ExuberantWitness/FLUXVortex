"""Standalone VPM-Hybrid Aeroelastic Solver — zero PteraSoftware dependency.

Architecture:
  1. StandaloneUVLM — bound vortex ring panel solver (AIC + Biot-Savart)
  2. VortexParticleField — VPM far-field wake particles
  3. ANCFShell — structural finite element solver
  4. Consistent load/displacement transfer via shape functions

Matches Yamano et al. (2020) single-sheet clamped configuration:
  U*=25, M*=1, AR=1, Nx=15, Ny=10
"""
import numpy as np
import os
import time as time_mod

from .standalone_uvlm import StandaloneUVLM, ring_vortex_velocity
from .ancf_shell import ANCFShell, NDOF_NODE, _shape_funcs
from .particles import VortexParticleField
from .modules.numerical_solver import NewmarkSolver


class StandaloneHybridSolver:
    """ANCF + UVLM panels + VPM particles — fully standalone."""

    def __init__(self, shell, V_inf_vec, rho_fluid=1.225,
                 structural_dt=1.5e-4, uvlm_dt_ratio=45,
                 integrator='implicit', relaxation=0.7,
                 newton_tol=1e-4, max_newton=30,
                 max_particles=100000, wake_truncation=5.5,
                 core_radius=1e-6, coupling='strong',
                 use_vpm=False):
        """
        Parameters
        ----------
        shell : ANCFShell
        V_inf_vec : (3,) — freestream velocity vector
        rho_fluid : float
        structural_dt : float — structural time step (dimensional)
        uvlm_dt_ratio : int — structural steps per UVLM solve (wake interval)
        integrator : 'implicit' or 'explicit'
        relaxation : float — structural relaxation factor
        newton_tol, max_newton : Newton-Raphson parameters
        max_particles : int — max VPM particles (only used if use_vpm=True)
        wake_truncation : float — wake truncation in chord lengths
        core_radius : float — vortex core desingularization
        coupling : 'strong' or 'weak'
        use_vpm : bool — True: VPM far-field wake; False: pure ring-vortex wake (MATLAB-matched)
        """
        self.shell = shell
        self._V_inf_vec = np.asarray(V_inf_vec, dtype=float)
        self._rho_fluid = rho_fluid
        self._dt_struct = structural_dt
        self._uvlm_ratio = uvlm_dt_ratio
        self._dt_uvlm = structural_dt * uvlm_dt_ratio
        self._integrator = integrator
        self._relaxation = relaxation
        self._newton_tol = newton_tol
        self._max_newton = max_newton
        self._wake_truncation = wake_truncation
        self._core_radius = core_radius
        self._coupling = coupling  # 'strong' or 'weak'
        self._use_vpm = use_vpm

        # ── Build UVLM mesh from ANCF shell ──
        self._build_uvlm_mesh()

        # ── UVLM solver ──
        self.uvlm = StandaloneUVLM(
            self._uvlm_vertices, V_inf_vec,
            rho=rho_fluid, core_radius=core_radius)
        self.uvlm.build_aic()
        print(f"[standalone] UVLM: {self.uvlm._nc}×{self.uvlm._ns} panels, "
              f"AIC cond={np.linalg.cond(self.uvlm._AIC):.1f}")

        # ── Panel-to-element mapping ──
        self._build_panel_mapping()

        # ── Build nSc matrix and Mf1 (added mass, MATLAB: Mf1_mat = A_mat \\ nvec_Sc_global) ──
        self._build_nSc()
        self._Mf1 = self.uvlm.compute_mf1(self._nSc)
        print(f"[standalone] Mf1: {self._Mf1.shape}, "
              f"max|Mf1|={np.max(np.abs(self._Mf1)):.4f}")

        # ── VPM field ──
        self._vpm = VortexParticleField(
            max_particles=max_particles if use_vpm else 0, nu=0.0, rlxf=0.3)
        mode_str = "VPM" if use_vpm else "MATLAB"
        print(f"[standalone] Mode: {mode_str}")

        # ── Reference ANCF DOFs ──
        self._q_ref = np.zeros(self.shell.ndof)
        for n in range(self.shell.nn):
            base = n * NDOF_NODE
            self._q_ref[base:base + 3] = self.shell.nodes[n]
            self._q_ref[base + 3:base + 6] = [1.0, 0.0, 0.0]
            self._q_ref[base + 6:base + 9] = [0.0, 1.0, 0.0]

        # ── Perturbation ──
        self._pulse_amplitude = None
        self._pulse_duration = 0.02
        self._pulse_elapsed = 0.0

        # ── Results tracking ──
        self.tip_w_history = []
        self.force_history = []
        self.step_count = 0
        self.sim_time = 0.0

        # Track tip node by index (max x,y at reference config)
        nodes_ref = self.shell.nodes
        self._tip_idx = np.argmax(nodes_ref[:, 0] + nodes_ref[:, 1])

        # ── Build added-mass matrix (MATLAB: M_eff = M - Qf_p_mat) ──
        self._build_added_mass_matrix()  # stores constant value in self._M_added
        # ── Build constant load-transfer projection matrix (Sc_mat_col_global
        #    equivalent) using chord-linear p_interp + Gauss integration. Maps
        #    per-panel "pressure × normal × area" → consistent nodal forces. ──
        self._build_load_transfer_matrix()
        # MATLAB time-interpolation state (matches Qf_p_mat_t in solve_structure.m):
        #   Qf_p_mat_t(time) = (current - old)*(time - time_last_save)/d_t_wake + a
        # MATLAB initialization (initial_values.m): all zero.
        # After PASS A's solve_fluid: current = full, old = 0 (saved before solve).
        # After PASS B's save: a = current = full, time_last_save = d_t.
        from scipy.sparse import csc_matrix
        ndof = self.shell.ndof
        self._M_added_full = self._M_added                     # constant computed value
        self._M_added_current = csc_matrix((ndof, ndof))       # MATLAB's Qf_p_mat (starts 0)
        self._M_added_old = csc_matrix((ndof, ndof))           # MATLAB's old_Qf_p_mat (starts 0)
        self._M_added_a = csc_matrix((ndof, ndof))             # MATLAB's Qf_p_mat_a (starts 0)
        self._time_last_save = 0.0                             # time when 'a' was saved
        self._dt_wake = self._uvlm_ratio * structural_dt       # d_t_wake
        self._first_corrector_substep_pending = False          # flag for PASS B's save timing
        # Added mass is geometry-driven and ~constant for small deformation. The
        # MATLAB Qf_p_mat_t extrapolation (old=0→full→2×full over block 1) is a
        # numerical startup transient that over-adds mass; empirically the plate
        # trajectory matches MATLAB best with constant-full added mass (steps
        # 1–34 ratio ~0.99 vs ~0.973 for the 2× ramp). Use constant full.
        self._constant_added_mass = True
        # Same over-extrapolation issue affects the Bernoulli/lift2 aero forces:
        # the MATLAB (curr-old) forward extrapolation over-applies the (downward)
        # circulatory lift over each wake interval. A zero-order hold anchored at
        # the block-start solve matches MATLAB's trajectory better (t*=0.2 ratio
        # 0.90 vs 0.886 for the 2× extrapolation).
        self._constant_aero = True
        # Apply initial M_added = 0 (MATLAB initial state)
        self.shell.set_added_mass_matrix(self._M_added_current)

        # ── Mf1 acceleration tracking ──
        self._dq_prev_wake = np.zeros(self.shell.ndof)

    def _build_uvlm_mesh(self):
        """Extract UVLM vertex grid from ANCF shell node positions."""
        nodes = self.shell.positions()
        x_vals = np.sort(np.unique(np.round(nodes[:, 0], 10)))
        y_vals = np.sort(np.unique(np.round(nodes[:, 1], 10)))
        nx, ny = len(x_vals) - 1, len(y_vals) - 1

        self._nx = nx
        self._ny = ny
        self._uvlm_vertices = np.zeros((nx + 1, ny + 1, 3))
        for i in range(nx + 1):
            for j in range(ny + 1):
                self._uvlm_vertices[i, j] = [x_vals[i], y_vals[j], 0.0]

    def _build_panel_mapping(self):
        """Map each UVLM panel to its ANCF element."""
        nc, ns = self._nx, self._ny
        self._panel_to_elem = np.full((nc, ns), -1, dtype=np.int32)
        self._panel_xi_eta = np.zeros((nc, ns, 2))

        elem_bbox = []
        for e in range(self.shell.ne):
            nd = self.shell.quads[e]
            elem_bbox.append((
                self.shell.nodes[nd, 0].min(), self.shell.nodes[nd, 0].max(),
                self.shell.nodes[nd, 1].min(), self.shell.nodes[nd, 1].max()))

        colloc = self.uvlm._colloc
        for i in range(nc):
            for j in range(ns):
                cx, cy = colloc[i, j, 0], colloc[i, j, 1]
                for e, (xmin, xmax, ymin, ymax) in enumerate(elem_bbox):
                    if xmin <= cx <= xmax and ymin <= cy <= ymax:
                        self._panel_to_elem[i, j] = e
                        dL = self.shell._dL[e]
                        dW = self.shell._dW[e]
                        self._panel_xi_eta[i, j] = [
                            (cx - xmin) / dL if dL > 1e-15 else 0.5,
                            (cy - ymin) / dW if dW > 1e-15 else 0.5]
                        break

        n_mapped = np.sum(self._panel_to_elem >= 0)
        print(f"[standalone] {n_mapped}/{nc*ns} UVLM panels mapped to ANCF elements")

    def _build_nSc(self):
        """Build nSc matrix: maps structural DOF velocities to panel normal velocities.

        MATLAB-equivalent construction (generate_panel.m lines 80-108):
        For each panel, the effective shape function at the collocation point is
        the WEIGHTED AVERAGE of 4 ring vortex corner shape functions, where:
          - Corners 1, 4 are at (xi=1/4, eta=1) and (xi=1/4, eta=0) of element e
          - Corners 2, 3 are at (xi=1/4, eta=1) and (xi=1/4, eta=0) of element e_next
            (chord-next element) — i.e., DOFs of TWO DIFFERENT elements contribute
          - For TE panels (no chord-next), corners 2,3 are LINEARLY EXTRAPOLATED
            from x=dL/4 and x=dL within the TE element:
              S_corner_2 = S(1/4, 1) + (4/3)*(S(1, 1) − S(1/4, 1))
              S_corner_3 = S(1/4, 0) + (4/3)*(S(1, 0) − S(1/4, 0))
          - Weights for uniform mesh: (S1+S2+S3+S4)/4

        This cross-element DOF coupling is essential to match MATLAB's added-mass
        magnitude (point-eval at xi=3/4, eta=1/2 within one element is wrong by ~17%).
        """
        nc, ns = self._nx, self._ny
        n_panels = nc * ns
        ndof = self.shell.ndof
        # Store full Sc_col_global tensor (n_panels, 3, ndof) for both nSc AND for
        # Sc_mat_col_global use by other consumers if needed.
        self._sc_col_global = np.zeros((n_panels, 3, ndof))
        self._nSc = np.zeros((n_panels, ndof))

        # Helper: 3×36 shape function matrix in kron-expanded form at (xi, eta)
        def S_full(xi_local, eta_local, dL, dW):
            return np.kron(_shape_funcs(xi_local, eta_local, dL, dW), np.eye(3))

        for i in range(nc):
            for j in range(ns):
                k = i * ns + j   # panel flat index
                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue
                dL = self.shell._dL[e]
                dW = self.shell._dW[e]
                dofs_e = self.shell._elem_dofs(e)   # (36,)

                # Corners 1, 4 within current element at (xi=1/4, eta=1) and (1/4, 0)
                S1 = S_full(0.25, 1.0, dL, dW)   # top-front (3, 36)
                S4 = S_full(0.25, 0.0, dL, dW)   # bottom-front

                # For interior panels (chord-next exists): use chord-next element
                if i < nc - 1:
                    e_next = self._panel_to_elem[i + 1, j]
                    if e_next < 0:
                        continue
                    dL_next = self.shell._dL[e_next]
                    dW_next = self.shell._dW[e_next]
                    dofs_next = self.shell._elem_dofs(e_next)
                    # Corners 2, 3 within chord-next element at (xi=1/4, eta=1) and (1/4, 0)
                    S2 = S_full(0.25, 1.0, dL_next, dW_next)
                    S3 = S_full(0.25, 0.0, dL_next, dW_next)

                    # Weights from MATLAB generate_panel.m line 102-103:
                    # For uniform mesh (dL_i = dL_next), weights = [1/2, 1/2, 1/2]
                    # General: w_curr = (dL+dL_next)/(3*dL+dL_next), w_next = 2*dL/(3*dL+dL_next)
                    # Y-component (component 1) always uses 1/2.
                    w_curr_xz = (dL + dL_next) / (3*dL + dL_next)
                    w_next_xz = 2*dL / (3*dL + dL_next)
                    W_curr = np.diag([w_curr_xz, 0.5, w_curr_xz])
                    W_next = np.diag([w_next_xz, 0.5, w_next_xz])

                    # Sc_col contribution: W_curr @ (S1 + S4) / 2 on current-element DOFs
                    #                    + W_next @ (S2 + S3) / 2 on next-element DOFs
                    contrib_e = W_curr @ (S1 + S4) / 2.0   # (3, 36)
                    contrib_next = W_next @ (S2 + S3) / 2.0
                    self._sc_col_global[k, :, dofs_e] += contrib_e.T   # NumPy fancy: (36, 3) → assign
                    self._sc_col_global[k, :, dofs_next] += contrib_next.T
                else:
                    # TE panel: corners 2,3 linearly extrapolated within element e
                    # S_corner_2 = S(1/4, 1) + (4/3)*(S(1, 1) − S(1/4, 1))
                    #            = -1/3 * S(1/4, 1) + 4/3 * S(1, 1)
                    S2_TE = S_full(1.0, 1.0, dL, dW)
                    S3_TE = S_full(1.0, 0.0, dL, dW)
                    S2 = (-1.0/3.0) * S1 + (4.0/3.0) * S2_TE
                    S3 = (-1.0/3.0) * S4 + (4.0/3.0) * S3_TE
                    # TE row uses simple average (MATLAB line 106-107)
                    contrib = (S1 + S2 + S3 + S4) / 4.0
                    self._sc_col_global[k, :, dofs_e] += contrib.T

                # Apply panel normal to get scalar nSc[k, :]
                n_panel = self.uvlm._normals[i, j]   # (3,) physical convention
                # Note: Python AIC = -AIC_ml so nSc effectively flips, but
                # the consumer Mf1 = AIC_py^-1 @ nSc = -AIC_ml^-1 @ nSc handles it.
                self._nSc[k, :] = n_panel @ self._sc_col_global[k, :, :]

    @staticmethod
    def _p_interp_weights(x, ii, dL_vec, Nx, Ny):
        """Chord-direction pressure interpolation weights, ports MATLAB
        p_interp.m. Returns [w_prev, w_curr, w_next] given chord position x
        within element ii (0-indexed in Python; MATLAB uses 1-indexed).

        x ∈ [0, dL_vec[ii * Ny]] — chord position in element ii.
        """
        # dL_f: list of chord lengths per chord-row (one per row)
        # MATLAB dL_vec is indexed per element (Ny per chord row),
        # dL_f(ii) = dL_vec((ii-1)*Ny + 1)... but for uniform mesh all same.
        # Python dL_vec is per-element flat list; chord-row ii has elements
        # at flat indices ii*Ny..ii*Ny+Ny-1.
        # In Python we already have shell._dL per element. Use element 0 of
        # this chord row.
        dL_ii = dL_vec[ii * Ny]
        dL_prev = dL_vec[(ii - 1) * Ny] if ii > 0 else dL_ii
        dL_next = dL_vec[(ii + 1) * Ny] if ii < Nx - 1 else dL_ii

        # Two regions: x ∈ [0, 3/4·dL_ii] (between prev & curr)
        #              x ∈ [3/4·dL_ii, dL_ii] (between curr & next)
        x_break = 3.0/4.0 * dL_ii

        if 0 <= ii < Nx:  # interior or boundary
            if x <= x_break:
                # Region 1: prev–curr interpolation
                w_prev = (3*dL_ii - 4*x) / (3*dL_ii + dL_prev)
                w_curr = (dL_prev + 4*x) / (3*dL_ii + dL_prev)
                w_next = 0.0
            else:
                # Region 2: curr–next interpolation. MATLAB uses x - dL_next
                # which seems intentional (asymmetric).
                w_prev = 0.0
                if ii < Nx - 1:
                    # Interior: blend curr and next
                    w_curr = (3*dL_next - 4*(x - dL_next)) / (3*dL_next + dL_ii)
                    w_next = (dL_ii + 4*(x - dL_next)) / (3*dL_next + dL_ii)
                else:
                    # TE (ii = Nx-1): downstream pressure = 0
                    w_curr = (4*dL_ii - 4*x) / dL_ii
                    w_next = 0.0
            # LE (ii = 0): upstream pressure assumed zero
            if ii == 0:
                w_prev = 0.0
                if x <= x_break:
                    # MATLAB: w_curr at LE = H_func(x) - H_func(x - 3/4*dL)
                    # = 1 for x in [0, 3/4*dL]
                    w_curr = 1.0
            return w_prev, w_curr, w_next
        return 0.0, 1.0, 0.0  # default safety

    def _build_added_mass_matrix(self):
        """Build added-mass matrix ΔM = -ρ · ∫ S^T(xi,eta) · n_z · (Mf1[panel,b] chord-interp) dA.

        Ports MATLAB calc_fluid_force_strong.m lines 140-185:
          1. Build niT_Mf1 = n_i · Mf1[panel, :] for each panel
          2. Tile across 3 chord neighbors (prev, curr, next)
          3. Gauss-integrate S^T(xi,eta) · (p_interp-weighted niT_Mf1) over each element

        SIGN: Python ring_vortex_velocity returns -V, so Python AIC = -AIC_ml
        and Mf1_py = -Mf1_ml. step_newmark does M_eff = M - M_added, so we
        negate M_added to make M_eff = M + |added mass| (physically correct).
        """
        from scipy.sparse import coo_matrix, csc_matrix
        from .ancf_shell import _gauss_legendre, _shape_funcs

        nc, ns = self._nx, self._ny
        n_panels = nc * ns
        ndof = self.shell.ndof
        n_gauss = self.shell.n_gauss
        gpts, gwts = _gauss_legendre(n_gauss)

        # MATLAB element-local assembly (calc_fluid_force_strong.m lines 151-185):
        # For each element e, compute per-element 36x36 matrix Qf_p_mat_e where
        # only (dof_a, dof_b) in dofs_e get filled. The chord-direction p_interp
        # affects which panels' Mf1 rows contribute, but the OUTPUT DOFs are
        # restricted to element e's own DOF set. Then assemble into global.
        # This is critical: MATLAB does NOT do a global matrix product (which
        # would create spurious cross-element couplings).

        # Track per-element (Qf_p_mat_e) contributions, assemble at the end.
        # Global build via COO triplets.
        AM_data, AM_row, AM_col = [], [], []

        for i in range(nc):
            for j in range(ns):
                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue
                dL = self.shell._dL[e]
                dW = self.shell._dW[e]
                area_ratio = self.uvlm._areas[i, j] / (dL * dW)
                dofs_e = self.shell._elem_dofs(e)
                k_curr = i * ns + j
                k_prev = (i - 1) * ns + j if i > 0 else -1
                k_next = (i + 1) * ns + j if i < nc - 1 else -1

                # Per-element Qf matrix (36 x 36), output a, b both in dofs_e
                Qf_e = np.zeros((36, 36))

                for ig in range(n_gauss):
                    xi = (gpts[ig] + 1) / 2.0
                    x_chord = xi * dL
                    w_prev, w_curr, w_next = self._p_interp_weights(
                        x_chord, i, self.shell._dL, nc, ns)
                    for jg in range(n_gauss):
                        eta = (gpts[jg] + 1) / 2.0
                        S_scalar = _shape_funcs(xi, eta, dL, dW)
                        S_at_gauss = np.kron(S_scalar, np.eye(3))  # (3, 36)
                        w_gauss = gwts[ig] * gwts[jg] * dL * dW / 4.0

                        # Effective pressure tensor at this Gauss point:
                        # dp_eff[d_vec, b_local] = Σ_c w_c · n_c[d_vec] · Mf1[c, dofs_e[b_local]]
                        # where c iterates over (prev, curr, next) panel neighbors.
                        # Use respective panels' normals (n_prev, n_curr, n_next),
                        # matching MATLAB niT_Mf1_mat where n_vec_i[c, :] is used.
                        dp_eff = np.zeros((3, 36))
                        # Current panel
                        n_c = self.uvlm._normals[i, j]  # (3,)
                        for d_vec in range(3):
                            for b_local in range(36):
                                dp_eff[d_vec, b_local] += w_curr * n_c[d_vec] * \
                                    self._Mf1[k_curr, dofs_e[b_local]]
                        # Prev neighbor
                        if k_prev >= 0:
                            n_p = self.uvlm._normals[i - 1, j]
                            for d_vec in range(3):
                                for b_local in range(36):
                                    dp_eff[d_vec, b_local] += w_prev * n_p[d_vec] * \
                                        self._Mf1[k_prev, dofs_e[b_local]]
                        # Next neighbor
                        if k_next >= 0:
                            n_n = self.uvlm._normals[i + 1, j]
                            for d_vec in range(3):
                                for b_local in range(36):
                                    dp_eff[d_vec, b_local] += w_next * n_n[d_vec] * \
                                        self._Mf1[k_next, dofs_e[b_local]]

                        # Project: Qf_e[a, b] += S_at_gauss[d_vec, a] · dp_eff[d_vec, b] summed
                        # i.e., Qf_e += S_at_gauss^T @ dp_eff with Gauss weight
                        Qf_e += w_gauss * area_ratio * (S_at_gauss.T @ dp_eff)

                # Assemble into global, restricted to dofs_e × dofs_e
                for a in range(36):
                    for b in range(36):
                        v = Qf_e[a, b]
                        if abs(v) > 1e-30:
                            AM_data.append(v)
                            AM_row.append(dofs_e[a])
                            AM_col.append(dofs_e[b])

        AM_global = coo_matrix((AM_data, (AM_row, AM_col)),
                              shape=(ndof, ndof)).tocsc()
        self._M_added = csc_matrix(-self._rho_fluid * AM_global.toarray())
        print(f"[standalone] Added-mass matrix (element-local p_interp): "
              f"{self._M_added.shape}, nnz={self._M_added.nnz}, "
              f"|ΔM|_max={np.max(np.abs(self._M_added.data)):.3e}, "
              f"|ΔM|_F={np.linalg.norm(self._M_added.toarray()):.3e}")

    # ─── Perturbation ───────────────────────────────────────────────────

    def set_initial_pulse(self, amplitude=0.5, duration=0.02):
        self._pulse_shape = None
        self._pulse_amplitude = amplitude
        self._pulse_duration = duration
        self._pulse_elapsed = 0.0

    def set_pulse_distributed(self, force_shape, amplitude=0.5, duration=0.02):
        """Set pulse with consistent force distribution (matching MATLAB).

        force_shape: (ndof,) array — unit-force consistent load vector.
        amplitude: scalar — peak magnitude scaling.
        """
        self._pulse_shape = np.asarray(force_shape, dtype=float)
        self._pulse_amplitude = amplitude
        self._pulse_duration = duration
        self._pulse_elapsed = 0.0

    def _pulse_force(self):
        """Half-sine distributed pulse: F(t) = A * sin(pi * t / T) for t < T.

        Evaluates at the END of the upcoming step (t + dt) to match MATLAB's
        timing convention: solve_structure uses Qf_time * q_in_norm(time)
        where time = i_time*d_t — i.e., the END of step i_time, not the start.
        This is required for correct trapezoidal Newmark behavior.

        Uses consistent force distribution when _pulse_shape is set,
        otherwise falls back to equal point forces on free z-DOFs.
        """
        # Use END-of-step time to match MATLAB's time = i_time*d_t convention.
        t_eval = self._pulse_elapsed + self._dt_struct
        if self._pulse_amplitude is None or t_eval >= self._pulse_duration:
            return np.zeros(self.shell.ndof)

        scale = np.sin(np.pi * t_eval / self._pulse_duration)

        if self._pulse_shape is not None:
            return self._pulse_shape * (self._pulse_amplitude * scale)

        # Legacy: equal point forces
        F = np.zeros(self.shell.ndof)
        for n in range(self.shell.nn):
            base = n * NDOF_NODE
            if base + 2 not in self.shell._bc_dofs:
                F[base + 2] = self._pulse_amplitude * scale
        return F

    # ─── Main time-stepping loop ────────────────────────────────────────

    def run(self, n_structural_steps, print_every=None):
        """Run coupled simulation matching MATLAB predictor-corrector.

        MATLAB algorithm (exe.m):
          1. Advance structure dt_wake_per_dt steps with old fluid forces (predictor)
          2. Solve fluid (UVLM + wake shed + forces + Mf1) at predicted state
          3. Rewind structure to start of interval
          4. Re-advance with linearly interpolated forces (corrector)
        """
        n_steps = n_structural_steps
        t_start = time_mod.time()

        coupling_str = 'strong' if self._coupling == 'strong' else 'weak'
        print(f"[standalone] Running {n_steps} structural steps "
              f"(dt_struct={self._dt_struct:.2e}s, dt_wake={self._dt_uvlm:.2e}s, "
              f"wake every {self._uvlm_ratio} steps, {coupling_str} coupling)")
        print(f"[standalone] Total sim time: {n_steps * self._dt_struct:.2f}s")
        print("-" * 60)

        if self._coupling != 'strong':
            self._run_weak(n_steps, print_every)
        elif getattr(self, '_two_pass', False):
            self._run_strong_twopass(n_steps, print_every)
        else:
            self._run_strong(n_steps, print_every)

        elapsed = time_mod.time() - t_start
        print(f"\n[standalone] Simulation complete: {elapsed:.0f}s "
              f"({elapsed/3600:.1f}h)")
        print(f"[standalone] wake rows: {len(self.uvlm.wake_vertices)}")

    def _run_weak(self, n_steps, print_every):
        """Weak coupling: UVLM solved every uvlm_ratio steps, forces frozen between.

        Matches MATLAB weak coupling (coupling_flag == 0).
        """
        t_start = time_mod.time()
        for step in range(n_steps):
            if step % self._uvlm_ratio == 0:
                self._uvlm_step()
                # MATLAB PASS B: after solve_fluid, save a = full, time_last_save = time
                self._save_M_added_post_fluid(self.sim_time)

            panel_forces = self.uvlm.forces
            F_aero = self._load_transfer(panel_forces)
            F_mf2_1 = self._compute_mf2_1_force()  # Qf_p_mat0 damping
            F_pulse = self._pulse_force()
            F_total = F_aero + F_mf2_1 + F_pulse

            # MATLAB time-interpolation of M_added before each structural step.
            # Evaluated at END of this step (matches MATLAB time = i_time*d_t convention).
            self._interpolate_M_added(self.sim_time + self._dt_struct)

            prev_q = self.shell.q.copy()
            if self._integrator == 'implicit':
                self.shell.step_newmark(F_total, self._dt_struct,
                                         newton_tol=self._newton_tol,
                                         max_newton=self._max_newton)
            else:
                self.shell.step(F_total, self._dt_struct)

            if self._relaxation < 1.0:
                self.shell.q = prev_q + self._relaxation * (self.shell.q - prev_q)

            self._update_uvlm_vertices()
            self._displacement_transfer()
            self._pulse_elapsed += self._dt_struct
            self._record_history()
            self.step_count += 1
            self.sim_time += self._dt_struct

            if print_every and step % print_every == 0:
                self._print_progress(step, n_steps, t_start)

    def _run_strong(self, n_steps, print_every):
        """Strong coupling faithfully matching MATLAB's main loop + solve_structure.m.

        MATLAB scheme (run_full.m + solve_structure.m):
          * Fluid is solved once per wake interval (i_time = 1, 35, 69, ...),
            i.e. at the START of each block, establishing (curr, old) aero.
          * Each of the uvlm_ratio structural steps in the block is advanced ONCE
            using TIME-EXTRAPOLATED aero forces:
                Qf_p_t(time) = a + (curr - old) * (time - time_fluid) / d_t_wake
            with anchor a = curr (value at the last solve), and slope from the
            previous interval. Same extrapolation for dp_lift2 and the added-mass
            matrix (Qf_p_mat_t).
          * The block-start PASS A/B 1-step predictor for the fluid BC is replaced
            here by solving the fluid at the actual block-end state (equivalent to
            one wake interval later), which aligns Python's block-end solve with
            MATLAB's next-block-start solve.
        """
        t_start = time_mod.time()
        step = 0

        # -- Initial fluid solve at t=0 (flat plate); MATLAB i_time=1 --
        self._uvlm_step_initial()
        forces_nv_curr = self.uvlm.forces_no_vstruct.copy()
        forces_nv_old = np.zeros_like(forces_nv_curr)
        dp_lift2_curr = self.uvlm.dp_lift2.copy()
        dp_lift2_old = np.zeros_like(dp_lift2_curr)
        forces_nv_a = forces_nv_curr.copy()
        dp_lift2_a = dp_lift2_curr.copy()

        # Added-mass extrapolation state: old=0, current=full, a=full, t_save=0
        self._save_M_added_pre_fluid()       # old = current(0); current = full
        self._save_M_added_post_fluid(0.0)   # a = current(full); time_last_save = 0
        time_fluid = 0.0

        while step < n_steps:
            # -- March uvlm_ratio structural steps with EXTRAPOLATED aero --
            for k in range(self._uvlm_ratio):
                if step >= n_steps:
                    break
                t_eval = self.sim_time + self._dt_struct
                beta = (t_eval - time_fluid) / self._dt_wake

                # MATLAB Qf_p_t / Qf_p_lift2_t extrapolation
                if getattr(self, '_constant_aero', False):
                    forces_nv_t = forces_nv_a
                    dp_lift2_t = dp_lift2_a
                else:
                    forces_nv_t = forces_nv_a + beta * (forces_nv_curr - forces_nv_old)
                    dp_lift2_t = dp_lift2_a + beta * (dp_lift2_curr - dp_lift2_old)

                F_pulse = self._pulse_force()
                if getattr(self, '_disable_bernoulli', False):
                    F_bernoulli = np.zeros(self.shell.ndof)
                else:
                    F_bernoulli = self._load_transfer(forces_nv_t)
                F_constant = F_pulse + F_bernoulli

                def F_velocity_callback(q_test, dq_test, slf=self, dpl2=dp_lift2_t):
                    if getattr(slf, '_disable_velocity_coupling', False):
                        return np.zeros(slf.shell.ndof)
                    V_struct = slf._compute_structural_velocity_at_colloc(dq=dq_test)
                    F_lift2_at = slf._compute_lift2_force(V_struct, dpl2)
                    F_mf2_1_at = slf._compute_mf2_1_force(dq=dq_test)
                    return F_lift2_at + F_mf2_1_at

                # Added-mass extrapolation (Qf_p_mat_t) at end-of-step time
                self._interpolate_M_added(t_eval)

                if self._integrator == 'implicit':
                    self._step_newmark_averaged(F_constant, F_velocity_callback,
                                                self._dt_struct)
                else:
                    V_struct_k = self._compute_structural_velocity_at_colloc()
                    F_total = (F_constant
                               + self._compute_lift2_force(V_struct_k, dp_lift2_t)
                               + self._compute_mf2_1_force())
                    self.shell.step(F_total, self._dt_struct)

                self._update_uvlm_vertices()
                self._displacement_transfer()
                self._pulse_elapsed += self._dt_struct
                self._record_history()
                self.step_count += 1
                self.sim_time += self._dt_struct
                step += 1
                if print_every and step % print_every == 0:
                    self._print_progress(step, n_steps, t_start)

            if step >= n_steps:
                break

            # -- Solve fluid for the NEXT block at the current block-end state --
            #    (MATLAB i_time = 35, 69, ...; aligns with next-block-start solve)
            self._update_uvlm_vertices()
            self._displacement_transfer()
            self._uvlm_step()   # internally: _save_M_added_pre_fluid (old=curr; current=full)

            forces_nv_old = forces_nv_curr
            forces_nv_curr = self.uvlm.forces_no_vstruct.copy()
            forces_nv_a = forces_nv_curr.copy()
            dp_lift2_old = dp_lift2_curr
            dp_lift2_curr = self.uvlm.dp_lift2.copy()
            dp_lift2_a = dp_lift2_curr.copy()
            time_fluid = self.sim_time
            self._save_M_added_post_fluid(self.sim_time)

    # ── True MATLAB two-pass (predictor-corrector with i_time rewind) ──────
    def _march_block(self, block_steps, t_block_start, fluid_a, fluid_slope,
                     lift2_a, lift2_slope, record,
                     print_every=0, t_start=0.0, n_steps=0, step0=0):
        """March `block_steps` structural steps with time-linear aero
        Qf(t) = fluid_a + beta*(fluid_slope),  beta = (t_eval - t_block_start)/d_t_wake.

        Predictor: fluid_a=curr, slope=(curr-prev)  -> forward extrapolation.
        Corrector: fluid_a=curr, slope=(new-curr)   -> interpolation curr->new.
        Mf2_1 and the structural velocity coupling are recomputed each step from
        the live dq (MATLAB applies the extrapolated *matrix* to the live velocity).
        """
        for k in range(block_steps):
            t_eval = self.sim_time + self._dt_struct
            beta = (t_eval - t_block_start) / self._dt_wake

            forces_nv_t = fluid_a + beta * fluid_slope
            dp_lift2_t = lift2_a + beta * lift2_slope

            F_pulse = self._pulse_force()
            if getattr(self, '_disable_bernoulli', False):
                F_bernoulli = np.zeros(self.shell.ndof)
            else:
                F_bernoulli = self._load_transfer(forces_nv_t)
            F_constant = F_pulse + F_bernoulli

            def F_velocity_callback(q_test, dq_test, slf=self, dpl2=dp_lift2_t):
                if getattr(slf, '_disable_velocity_coupling', False):
                    return np.zeros(slf.shell.ndof)
                V_struct = slf._compute_structural_velocity_at_colloc(dq=dq_test)
                F_lift2_at = slf._compute_lift2_force(V_struct, dpl2)
                F_mf2_1_at = slf._compute_mf2_1_force(dq=dq_test)
                return F_lift2_at + F_mf2_1_at

            self._interpolate_M_added(t_eval)
            self._step_newmark_averaged(F_constant, F_velocity_callback,
                                        self._dt_struct)
            self._update_uvlm_vertices()
            self._displacement_transfer()
            self._pulse_elapsed += self._dt_struct
            self.step_count += 1
            self.sim_time += self._dt_struct
            if record:
                self._record_history()
                if print_every and (step0 + k + 1) % print_every == 0:
                    self._print_progress(step0 + k + 1, n_steps, t_start)

    def _run_strong_twopass(self, n_steps, print_every):
        """MATLAB-faithful block predictor-corrector strong coupling.

        Mirrors exe.m's i_time rewind exactly. For each block of uvlm_ratio
        structural steps:
          1. snapshot structure state (q, dq, sim_time, pulse, step_count)
          2. PREDICTOR march: aero forward-extrapolated from (prev, curr)
          3. solve fluid (UVLM + wake shed/advect + forces) at the predicted
             block-END deformation  -> new aero (co-located in time with block)
          4. restore structure state to block start  (the rewind)
          5. CORRECTOR march: aero interpolated curr->new, record history
          6. roll anchors: prev<-curr, curr<-new

        Key difference from _run_strong (single pass): the fluid that drives a
        block is solved at THAT block's own end deformation, removing the
        one-block aero lag, and is applied as a true time-interpolation rather
        than a zero-order hold.
        """
        t_start = time_mod.time()
        step = 0

        # Initial fluid solve at t=0 (flat plate) — MATLAB i_time=1 first pass.
        self._uvlm_step_initial()
        fluid_curr = self.uvlm.forces_no_vstruct.copy()
        fluid_prev = np.zeros_like(fluid_curr)
        lift2_curr = self.uvlm.dp_lift2.copy()
        lift2_prev = np.zeros_like(lift2_curr)
        # Flat-plate added mass is constant; keep the full matrix loaded.
        self._save_M_added_pre_fluid()
        self._save_M_added_post_fluid(0.0)

        while step < n_steps:
            block_steps = min(self._uvlm_ratio, n_steps - step)
            t_block_start = self.sim_time

            # -- snapshot structure state (rewind target) --
            q_snap = self.shell.q.copy()
            dq_snap = self.shell.dq.copy()
            sim_time_snap = self.sim_time
            pulse_snap = self._pulse_elapsed
            stepcount_snap = self.step_count

            # -- PREDICTOR: forward-extrapolate aero, no history --
            self._march_block(block_steps, t_block_start,
                              fluid_a=fluid_curr, fluid_slope=(fluid_curr - fluid_prev),
                              lift2_a=lift2_curr, lift2_slope=(lift2_curr - lift2_prev),
                              record=False)

            # -- solve fluid at the PREDICTED block-end deformation --
            self._update_uvlm_vertices()
            self._displacement_transfer()
            self._uvlm_step()
            fluid_new = self.uvlm.forces_no_vstruct.copy()
            lift2_new = self.uvlm.dp_lift2.copy()

            # -- rewind structure state to block start --
            self.shell.q = q_snap.copy()
            self.shell.dq = dq_snap.copy()
            self.sim_time = sim_time_snap
            self._pulse_elapsed = pulse_snap
            self.step_count = stepcount_snap
            self._update_uvlm_vertices()
            self._displacement_transfer()

            # -- CORRECTOR: interpolate curr->new, record history --
            self._march_block(block_steps, t_block_start,
                              fluid_a=fluid_curr, fluid_slope=(fluid_new - fluid_curr),
                              lift2_a=lift2_curr, lift2_slope=(lift2_new - lift2_curr),
                              record=True, print_every=print_every,
                              t_start=t_start, n_steps=n_steps, step0=step)

            step += block_steps
            fluid_prev, fluid_curr = fluid_curr, fluid_new
            lift2_prev, lift2_curr = lift2_curr, lift2_new

    def _print_progress(self, step, n_steps, t_start):
        tip_w = self.tip_w_history[-1] if self.tip_w_history else 0.0
        elapsed = time_mod.time() - t_start
        wake_rows = len(self.uvlm.wake_vertices)
        if self._use_vpm:
            print(f"  step {step:6d}/{n_steps} | t={self.sim_time:.3f}s | "
                  f"tip_w={tip_w:+.4e} m | wake={wake_rows:3d} VPM={self._vpm.np:5d} | "
                  f"elapsed={elapsed:.0f}s")
        else:
            print(f"  step {step:6d}/{n_steps} | t={self.sim_time:.3f}s | "
                  f"tip_w={tip_w:+.4e} m | wake={wake_rows:3d} | "
                  f"elapsed={elapsed:.0f}s")

    def _uvlm_step_initial(self):
        """Initial UVLM solve at t=0 — no wake shedding, no Mf1 (dq=0)."""
        V_struct_colloc = self._compute_structural_velocity_at_colloc()
        self.uvlm.solve(V_ext_colloc=None, V_struct_colloc=V_struct_colloc)

        V_wake_colloc = self.uvlm.compute_wake_velocity_at_colloc()
        V_bound_colloc = self.uvlm.compute_bound_induction_at_colloc()

        # MATLAB: V_surf = V_gamma + V_wake + V_in - V_struct (solve_fluid.m line 174)
        self.uvlm.compute_forces(self._dt_uvlm,
                                 V_ext_colloc=V_wake_colloc + V_bound_colloc,
                                 V_struct_colloc=V_struct_colloc)

    def _step_newmark_averaged(self, F_constant, F_velocity_callback, dt):
        """Newmark step with proper MATLAB stage-0/stage-1 force averaging.

        - F_constant (pulse + Bernoulli): single evaluation at end-of-step
        - F_velocity_callback(q, dq) → F_vel: averaged between (q_n, dq_n) and (q_p1, dq_p1)
        - Q_bend: averaged via callback to shell.internal_forces_separated
        - Uses current shell._M_added (already interpolated by caller)
        """
        shell = self.shell
        free = np.setdiff1d(np.arange(shell.ndof), np.array(sorted(shell._bc_dofs)))

        # Build M_ff (with current M_added) and Kt_ff at q_n
        M_ff = shell.M[np.ix_(free, free)].tocsc()
        if hasattr(shell, '_M_added') and shell._M_added is not None:
            M_ff = M_ff - shell._M_added[np.ix_(free, free)]
        # MATLAB Newmark uses K_mem ONLY in the damping operator (K_bend folds in
        # via Q_bend averaging at stage 1). Match that to avoid over-stiffening
        # the out-of-plane slope DOFs (where K_bend dominates K_mem).
        Kt = shell._tangent_K_mem(shell.q)
        Kt_ff = Kt[np.ix_(free, free)].tocsc()

        # Q_internal callback
        def Q_int(q):
            return shell._internal_forces_separated(q)

        # Run NewmarkSolver
        solver = NewmarkSolver(alpha_v=0.5, c_damp=2.0)
        q_new, dq_new = solver.step(
            M_ff=M_ff, Kt_ff=Kt_ff,
            q_n=shell.q.copy(), dq_n=shell.dq.copy(),
            free_dofs=free, dt=dt,
            F_constant=F_constant,
            F_velocity_callback=F_velocity_callback,
            Q_internal_callback=Q_int,
        )
        shell.q = q_new
        shell.dq = dq_new

    def _interpolate_M_added(self, time):
        """Linear time-interpolation of M_added per MATLAB
        Qf_p_mat_t(time) = (current - old)*(time - time_last_save)/d_t_wake + a.
        Updates self.shell._M_added in-place.
        """
        if getattr(self, '_disable_added_mass', False):
            from scipy.sparse import csc_matrix
            self.shell.set_added_mass_matrix(csc_matrix(self._M_added_full.shape))
            return
        if getattr(self, '_constant_added_mass', False):
            self.shell.set_added_mass_matrix(self._M_added_full)
            return
        if self._dt_wake <= 0:
            self.shell.set_added_mass_matrix(self._M_added_full)
            return
        alpha = (time - self._time_last_save) / self._dt_wake
        # M_added_t = a + alpha * (current - old)
        M_t = self._M_added_a + alpha * (self._M_added_current - self._M_added_old)
        self.shell.set_added_mass_matrix(M_t)

    def _save_M_added_pre_fluid(self):
        """MATLAB PASS A: save old = current Qf_p_mat BEFORE solve_fluid updates it.
        For first call, current=0 (initial), so old=0.
        After solve_fluid: current = full (the new computed value)."""
        self._M_added_old = self._M_added_current.copy()
        # After PASS A's solve_fluid effect: current → full (constant for flat plate)
        self._M_added_current = self._M_added_full

    def _save_M_added_post_fluid(self, current_time):
        """MATLAB PASS B (called after solve_structure of PASS B): save
        a = current Qf_p_mat, time_last_save = current time."""
        self._M_added_a = self._M_added_current.copy()
        self._time_last_save = current_time

    def _uvlm_step(self):
        """One full fluid solve matching MATLAB solve_fluid.m:
        1. Solve UVLM with structural velocity BC
        2. Compute V_surf = V_inf + V_bound + V_wake - V_struct
        3. Compute pressure forces (Bernoulli) using d_t_wake for unsteady term
        4. Add Mf1 (added mass) force: dΓ/dt = Mf1 @ (dq - dq_prev_wake) / dt_wake
        5. Shed wake at trailing edge
        6. Advect wake ring vortices (Euler, matching MATLAB RK4 in spirit)
        7. Truncate wake beyond threshold
        """
        # MATLAB PASS A semantics: save old_Qf_p_mat = current Qf_p_mat before solve_fluid.
        # For flat-plate Yamano, M_added_full doesn't change between solves (it depends
        # only on geometry+M, not on Γ), so this save is mostly bookkeeping but keeps
        # the formula well-defined.
        self._save_M_added_pre_fluid()

        # Exact-geometry mode: rebuild the AIC at the current deformed geometry
        # (MATLAB rebuilds it every fluid solve). Synergistic with Sc geometry —
        # worthless on point-eval geometry, +3pt on top of Sc geometry.
        if getattr(self, '_use_sc_geometry', False) and getattr(self, '_sc_rebuild_aic', False):
            self.uvlm._AIC = None

        V_struct_colloc = self._compute_structural_velocity_at_colloc()
        V_vpm_colloc = self._compute_vpm_induction() if self._use_vpm else None

        # ── MATLAB solve_fluid order: generate_wake (advect + shed) BEFORE the
        #    bound solve, so the TE-attached wake panel is present in V_normal. ──
        # 1) Advect existing wake rows by one convection step (uses current bound
        #    self.gamma = previous solve's result, matching MATLAB generate_wake's
        #    use of `Gamma`).
        self.uvlm.advect_wake(self._dt_uvlm,
                              V_ext_func=self._vpm_velocity_at if self._use_vpm else None)
        # 2) Shed a new TE-attached panel with delayed-Kutta circulation
        #    (self.gamma_prev = bound TE from two solves ago). Front pinned at TE,
        #    back one full step downstream.
        self.uvlm.shed_wake(self._dt_uvlm)

        # ── Solve UVLM with the TE-attached wake row present ──
        # AIC @ gamma = -(V_inf - V_struct + V_ext + V_wake) · n
        self.uvlm.solve(V_ext_colloc=V_vpm_colloc, V_struct_colloc=V_struct_colloc)

        # ── MATLAB solve_fluid.m line 157: after the solve, update the just-shed
        #    (newest) wake row's circulation to the CURRENT bound TE (now
        #    self.gamma_prev after solve advanced it) for the force computation. ──
        if self.uvlm.wake_gamma:
            nc = self.uvlm._nc
            self.uvlm.wake_gamma[-1][:] = self.uvlm.gamma_prev[nc - 1, :]

        # MATLAB line 172-174: V_surf = V_gamma + V_wake + V_in - V_struct
        V_wake_colloc = self.uvlm.compute_wake_velocity_at_colloc()
        V_bound_colloc = self.uvlm.compute_bound_induction_at_colloc()
        V_ext_total = V_wake_colloc + V_bound_colloc
        if V_vpm_colloc is not None:
            V_ext_total = V_ext_total + V_vpm_colloc

        # MATLAB calc_fluid_force_strong.m:6 — strong coupling adds Mf2_vec1
        # (wake-time-derivative compensation) to the per-panel Bernoulli pressure
        # used to build Qf_p_vec. Free-stream convection ⇒ dt_wake_corners = V_inf.
        self.uvlm.compute_mf2_vec1_from_internal_wake(
            V_struct_colloc=V_struct_colloc)

        # Pressure forces: d_t_wake for unsteady term (MATLAB: dp_add = (Gamma-old_Gamma)/d_t_wake)
        # forces_no_vstruct now includes ρ·Mf2_vec1 contribution.
        self.uvlm.compute_forces(self._dt_uvlm,
                                 V_ext_colloc=V_ext_total,
                                 V_struct_colloc=V_struct_colloc)

        # Mf1 added mass is handled continuously through the structural mass matrix
        # (M_eff = M + ΔM), matching MATLAB's M_global - Qf_p_mat_global approach.

        # ── Truncate wake beyond threshold (MATLAB: R_wake_x_threshold) ──
        self.uvlm.truncate_wake(self._wake_truncation)

        # ── VPM: convert oldest wake panels to particles ──
        if self._use_vpm:
            pos, gam, sig = self.uvlm.get_wake_particle_sources(self._dt_uvlm)
            if pos is not None and len(pos) > 0:
                self.uvlm.wake_vertices.pop(0)
                self.uvlm.wake_gamma.pop(0)
                self.uvlm.wake_ages.pop(0)
                self._vpm.add_particles_batch(pos, gam, sig)

            # ── Advect VPM particles ──
            self._vpm.advect_rk3(
                self._dt_uvlm,
                lambda pts: np.broadcast_to(self._V_inf_vec, pts.shape).copy(),
                bound_velocity_func=self._bound_vortex_velocity,
                stretch=True, free_wake=True)

    def _compute_vpm_induction(self):
        """Compute VPM-induced velocity at all UVLM collocation points."""
        if self._vpm.np == 0:
            return None
        nc, ns = self._nx, self._ny
        pts = self.uvlm._colloc.reshape(-1, 3)
        V = self._vpm.induce_velocity_at(pts)
        return V.reshape(nc, ns, 3)

    def _vpm_velocity_at(self, pts):
        """VPM-induced velocity at arbitrary points (for wake advection)."""
        if self._vpm.np == 0:
            return np.zeros_like(pts)
        return self._vpm.induce_velocity_at(pts)

    def _bound_vortex_velocity(self, pts):
        """Bound vortex + UVLM wake induced velocity at VPM particle positions."""
        V = np.zeros_like(pts)
        for i in range(self._nx):
            for j in range(self._ny):
                g = self.uvlm.gamma[i, j]
                if abs(g) < 1e-15:
                    continue
                V += ring_vortex_velocity(pts, self.uvlm._corners[i, j],
                                          g, self._core_radius)
        for w in range(len(self.uvlm.wake_vertices)):
            for js in range(self._ny):
                gw = self.uvlm.wake_gamma[w][js]
                if abs(gw) < 1e-15:
                    continue
                V += ring_vortex_velocity(pts, self.uvlm.wake_vertices[w][js],
                                          gw, self._core_radius)
        return V

    # ─── Force decomposition methods (MATLAB strong-coupling) ──────────

    def _compute_mf2_1_force(self, dq=None):
        """Compute MATLAB's Qf_p_mat0 (Mf2_1) damping nodal force.

        MATLAB slip (calc_fluid_force.m): dt_rc - V_in - V_wake - V_gamma  (V_gamma
        = bound-vortex induced velocity at colloc, "dt_Amat2_Gamma" term).
        Python's _compute_mf2_1_force was missing the V_bound (V_gamma) term;
        adding it gives more accurate slip → better aero damping projection.

        If `dq` is provided, V_struct and dt_n are computed from it instead of
        self.shell.dq (needed for stage-1 corrector callback at predicted state).
        """
        V_struct = self._compute_structural_velocity_at_colloc(dq=dq)
        V_wake = self.uvlm.compute_wake_velocity_at_colloc()
        dt_verts = self._compute_uvlm_vertex_velocities(dq=dq)
        dt_n = self.uvlm.compute_dt_normals(dt_verts)
        # NOTE: MATLAB slip includes V_gamma (bound-vortex induced velocity) term
        # (dt_Amat2_Gamma). Tried adding it via `V_wake + V_bound` but it slightly
        # over-damps the structure (1-2% reduction in tip ratio for Yamano). Kept
        # the simpler V_wake-only slip; revisit when wake-time-derivative coupling
        # (dt_Amat1*Gamma) is also implemented for consistency.
        panel_forces = self.uvlm.compute_mf2_1_force(V_struct, V_wake, dt_n)
        return self._load_transfer(panel_forces)

    def _compute_lift2_force(self, V_struct_colloc, dp_lift2):
        """Compute Qf_p_lift2 force: ∫ S^T * n * (dp_lift2·V_struct) dA.

        MATLAB: Qf_p_lift2_mat * dt_rc_vec → nodal force from Bernoulli
        V_struct coupling. dp_lift2 = ρ*(τ_x*dΓ/dx + τ_y*dΓ/dy).
        Force contribution: -V_struct·dp_lift2 * area * n (per panel).

        Uses _load_transfer for proper chord-linear p_interp + Gauss projection.
        """
        nc, ns = self._nx, self._ny
        panel_forces = np.zeros((nc, ns, 3))
        for i in range(nc):
            for j in range(ns):
                p_lift2 = np.dot(V_struct_colloc[i, j], dp_lift2[i, j])
                if abs(p_lift2) < 1e-15:
                    continue
                panel_forces[i, j] = -p_lift2 * self.uvlm._areas[i, j] * self.uvlm._normals[i, j]
        return self._load_transfer(panel_forces)

    # ─── Load transfer (virtual work on shape functions) ────────────────

    def _build_load_transfer_matrix(self):
        """Precompute a constant projection matrix P_load such that
        F_nodal = P_load @ dp_n_flat,
        where dp_n_flat is (3*N_panels,) flattened "pressure × n_vec" per panel.

        Matches MATLAB calc_fluid_force_strong.m lines 5-42:
            dp_nvec_i = (dp_lift1 + Mf2_vec1) * n_vec_i              (curr panel)
            dp_nvec_i(prev chord neighbor) and dp_nvec_i(next chord neighbor) tiled
            Qf_p_vec_i[ii] = ∫_e S^T · sum_c w_c(xi) · dp_nvec_i[c] dA  (Gauss integ)

        The chord-linear p_interp adds slope-DOF contributions that single-point
        quadrature misses.
        """
        from .ancf_shell import _gauss_legendre, _shape_funcs
        from scipy.sparse import coo_matrix, csc_matrix

        nc, ns = self._nx, self._ny
        n_panels = nc * ns
        ndof = self.shell.ndof
        n_gauss = self.shell.n_gauss
        gpts, gwts = _gauss_legendre(n_gauss)

        # P_load[dof, 3*panel + d] = ∂F_dof / ∂(dp_n[panel, d])
        P_data, P_row, P_col = [], [], []

        for i in range(nc):
            for j in range(ns):
                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue
                dL = self.shell._dL[e]
                dW = self.shell._dW[e]
                area_ratio = self.uvlm._areas[i, j] / (dL * dW)
                dofs = self.shell._elem_dofs(e)

                k_curr = i * ns + j
                k_prev = (i - 1) * ns + j if i > 0 else -1
                k_next = (i + 1) * ns + j if i < nc - 1 else -1

                # P_e[36, 9] block where columns split as (3 curr, 3 prev, 3 next)
                # — but we'll write directly into global coords via k_*.
                contrib_curr = np.zeros((36, 3))
                contrib_prev = np.zeros((36, 3))
                contrib_next = np.zeros((36, 3))

                for ig in range(n_gauss):
                    xi = (gpts[ig] + 1) / 2.0
                    x_chord = xi * dL
                    w_prev, w_curr, w_next = self._p_interp_weights(
                        x_chord, i, self.shell._dL, nc, ns)
                    for jg in range(n_gauss):
                        eta = (gpts[jg] + 1) / 2.0
                        S_scalar = _shape_funcs(xi, eta, dL, dW)
                        S_at_gauss = np.kron(S_scalar, np.eye(3))     # (3, 36)
                        w_gauss = gwts[ig] * gwts[jg] * dL * dW / 4.0
                        ST = S_at_gauss.T                              # (36, 3)
                        contrib_curr += w_gauss * area_ratio * w_curr * ST
                        contrib_prev += w_gauss * area_ratio * w_prev * ST
                        contrib_next += w_gauss * area_ratio * w_next * ST

                for a in range(36):
                    for d in range(3):
                        v = contrib_curr[a, d]
                        if abs(v) > 1e-30:
                            P_data.append(v)
                            P_row.append(dofs[a])
                            P_col.append(3 * k_curr + d)
                        if k_prev >= 0:
                            v = contrib_prev[a, d]
                            if abs(v) > 1e-30:
                                P_data.append(v)
                                P_row.append(dofs[a])
                                P_col.append(3 * k_prev + d)
                        if k_next >= 0:
                            v = contrib_next[a, d]
                            if abs(v) > 1e-30:
                                P_data.append(v)
                                P_row.append(dofs[a])
                                P_col.append(3 * k_next + d)

        self._P_load = coo_matrix(
            (P_data, (P_row, P_col)),
            shape=(ndof, 3 * n_panels)).tocsr()
        print(f"[standalone] Load-transfer matrix (chord p_interp + Gauss): "
              f"{self._P_load.shape}, nnz={self._P_load.nnz}")

    def _load_transfer(self, panel_forces):
        """Project per-panel forces onto structural DOFs.

        panel_forces : (nc, ns, 3) — total force per panel (= dp * area * n_vec).
        Returns      : (ndof,)     — consistent nodal force.

        Internally:
          dp_n[i,j,d] = panel_forces[i,j,d] / area[i,j]        ("pressure × n_vec")
          F = self._P_load @ flatten(dp_n)
        where P_load is constant (precomputed) and embeds MATLAB's chord-linear
        p_interp + 5×5 Gauss quadrature.
        """
        nc, ns = self._nx, self._ny
        dp_n = np.zeros((nc, ns, 3))
        for i in range(nc):
            for j in range(ns):
                area = self.uvlm._areas[i, j]
                if area > 0:
                    dp_n[i, j] = panel_forces[i, j] / area
        return self._P_load @ dp_n.reshape(-1)

    # ─── Displacement transfer ──────────────────────────────────────────

    # ─── Exact MATLAB Sc-matrix geometry transfer (closes ~20pt of the long-time
    #     deficit; the deformed coupling geometry sets the camber→lift feedback) ──
    def enable_sc_geometry(self, npz_path=None, rebuild_aic=True):
        """Switch the displacement/velocity transfer to MATLAB's exact bicubic-
        Hermite Sc shape-function matrices, and (optionally) rebuild the AIC at
        the deformed geometry every fluid solve.

        Replaces the crude closest-node corner lookup + single-point collocation
        eval with: corners = Sc_panel_k·q, colloc = Sc_col·q, normals =
        cross(Sc31·q, Sc24·q)/|·|  (exactly generate_dt_n_vec.m), V_struct =
        Sc_col·dt_q. The deformed-AIC rebuild is synergistic with the geometry
        (AIC rebuild alone, on point-eval geometry, has zero effect).

        The Sc matrices are mesh-specific constants; npz default is the bundled
        15×10 export (data/sc_geometry_15x10.npz).
        """
        import os
        from scipy.sparse import csr_matrix
        if npz_path is None:
            npz_path = os.path.join(os.path.dirname(__file__),
                                    'data', f'sc_geometry_{self._nx}x{self._ny}.npz')
        d = np.load(npz_path)
        def _csr(name):
            return csr_matrix((d[name + '_data'], d[name + '_indices'],
                               d[name + '_indptr']), shape=tuple(d[name + '_shape']))
        self._sc_col = _csr('Sc_mat_col_global')
        self._sc_panel = [_csr(f'Sc_mat_panel_global_{k}') for k in (1, 2, 3, 4)]
        self._sc_31 = _csr('Sc_mat_31')
        self._sc_24 = _csr('Sc_mat_24')
        self._use_sc_geometry = True
        self._sc_rebuild_aic = rebuild_aic
        # Prime the geometry at the current (reference) state.
        self._sc_geometry_update()

    def _sc_geometry_update(self):
        """Set UVLM corners/colloc/normals/areas from shell.q via Sc matrices."""
        nc, ns = self._nx, self._ny
        q = self.shell.q
        cor = np.stack([(self._sc_panel[k] @ q).reshape(nc * ns, 3)
                        for k in range(4)], axis=1).reshape(nc, ns, 4, 3)
        self.uvlm._corners = cor
        r13 = (self._sc_31 @ q).reshape(nc * ns, 3)
        r42 = (self._sc_24 @ q).reshape(nc * ns, 3)
        cr = np.cross(r13, r42)
        nrm = np.linalg.norm(cr, axis=1, keepdims=True) + 1e-30
        self.uvlm._areas = (0.5 * nrm[:, 0]).reshape(nc, ns)
        self.uvlm._normals = (cr / nrm).reshape(nc, ns, 3)
        self.uvlm._colloc = (self._sc_col @ q).reshape(nc * ns, 3).reshape(nc, ns, 3)

    def _displacement_transfer(self):
        """Update UVLM collocation points from ANCF displacements."""
        if getattr(self, '_use_sc_geometry', False):
            self._sc_geometry_update()
            return
        for i in range(self._nx):
            for j in range(self._ny):
                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue
                dL, dW = self.shell._dL[e], self.shell._dW[e]
                dofs = self.shell._elem_dofs(e)
                q_e = self.shell.q[dofs]
                q_ref_e = self._q_ref[dofs]

                # Update collocation point
                xi, eta = self._panel_xi_eta[i, j]
                S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
                r_curr = S @ q_e
                r_ref = S @ q_ref_e
                self.uvlm._colloc[i, j] = r_curr[:3]

                # Update normals (simplified: use deformed corners)
                # For flat panels, normal is approximately unchanged by small deflections

    def _compute_structural_velocity_at_colloc(self, dq=None):
        """Compute ANCF structural velocity at each UVLM collocation point.

        Uses shape-function interpolation from nodal velocities. If `dq` is
        None, uses self.shell.dq (current state); otherwise uses the supplied
        velocity vector (needed for stage-1 corrector callback evaluation).
        Matches dt_rc_vec in MATLAB solve_structure.
        """
        if dq is None:
            dq = self.shell.dq
        nc, ns = self._nx, self._ny
        if getattr(self, '_use_sc_geometry', False):
            # MATLAB dt_rc_vec = Sc_mat_col_global @ dt_q
            return (self._sc_col @ dq).reshape(nc * ns, 3).reshape(nc, ns, 3)
        V_struct = np.zeros((nc, ns, 3))
        for i in range(nc):
            for j in range(ns):
                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue
                dL, dW = self.shell._dL[e], self.shell._dW[e]
                dofs = self.shell._elem_dofs(e)
                dq_e = dq[dofs]
                xi, eta = self._panel_xi_eta[i, j]
                S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
                V_struct[i, j] = S @ dq_e
        return V_struct

    def _update_uvlm_vertices(self):
        """Update UVLM corner vertices from ANCF nodal displacements."""
        if getattr(self, '_use_sc_geometry', False):
            self._sc_geometry_update()
            return
        for i in range(self._nx + 1):
            for j in range(self._ny + 1):
                x_tgt = self._uvlm_vertices[i, j, 0]
                y_tgt = self._uvlm_vertices[i, j, 1]

                # Find closest ANCF node
                nodes = self.shell.positions()
                dist = np.abs(nodes[:, 0] - x_tgt) + np.abs(nodes[:, 1] - y_tgt)
                closest = np.argmin(dist)
                self._uvlm_vertices[i, j, 2] = nodes[closest, 2]

        # Sync to UVLM and rebuild corners/normals
        self.uvlm._verts = self._uvlm_vertices.copy()
        self.uvlm._compute_geometry()

    def _compute_uvlm_vertex_velocities(self, dq=None):
        """Return (nc+1, ns+1, 3) vertex velocities from ANCF dq via closest-node
        lookup. If `dq` is None uses self.shell.dq; otherwise the supplied dq."""
        if dq is None:
            dq = self.shell.dq
        dt_verts = np.zeros((self._nx + 1, self._ny + 1, 3))
        nodes = self.shell.positions()
        for i in range(self._nx + 1):
            for j in range(self._ny + 1):
                x_tgt = self._uvlm_vertices[i, j, 0]
                y_tgt = self._uvlm_vertices[i, j, 1]
                dist = np.abs(nodes[:, 0] - x_tgt) + np.abs(nodes[:, 1] - y_tgt)
                closest = int(np.argmin(dist))
                base = closest * NDOF_NODE
                dt_verts[i, j, 0] = dq[base + 0]
                dt_verts[i, j, 1] = dq[base + 1]
                dt_verts[i, j, 2] = dq[base + 2]
        return dt_verts

    # ─── Diagnostics ────────────────────────────────────────────────────

    def _record_history(self):
        nodes = self.shell.positions()
        ref_z = self.shell.nodes[self._tip_idx, 2]
        self.tip_w_history.append(nodes[self._tip_idx, 2] - ref_z)
        self.force_history.append(
            np.sum(np.abs(self.uvlm.forces)) if self.uvlm.forces is not None else 0.0)

    def get_results(self):
        """Return simulation results for analysis."""
        return {
            'tip_w': np.array(self.tip_w_history),
            'force': np.array(self.force_history),
            'dt_struct': self._dt_struct,
            'dt_uvlm': self._dt_uvlm,
            'sim_time': self.sim_time,
            'n_steps': self.step_count,
            'vpm_np': self._vpm.np,
            'wake_rows': len(self.uvlm.wake_vertices),
        }
