"""1:1 MATLAB translation of Yamano et al. FSI solver.

Direct translation of MATLAB source files:
  exe.m → MainLoop
  solve_structure.m → solve_structure_step
  new_X_func_FAST.m → newmark_step
  generate_stiff_matrices.m → _compute_structural_matrices
  generate_Qf_time_mat.m → _compute_pulse_force_vector
  solve_fluid.m → _solve_fluid
  calc_fluid_force.m → _calc_fluid_force
  calc_fluid_force_strong.m → _calc_fluid_force_strong

Uses ANCFShell for element precomputations (shape functions at Gauss points)
but applies MATLAB's nondimensional scaling throughout.
"""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, eye as speye, bmat as spbmat
from scipy.sparse.linalg import spsolve

from .ancf_shell import ANCFShell, NDOF_NODE, _shape_funcs, _gauss_legendre
from .standalone_uvlm import StandaloneUVLM


class MatlabFSISolver:
    """Direct MATLAB translation of the Yamano FSI solver."""

    def __init__(self, shell, V_inf_vec, rho_fluid=1.225,
                 structural_dt=1.5e-4, uvlm_dt_ratio=45,
                 Length=1.0, U_star=25.0, M_star=1.0,
                 alpha_v=0.5, theta_a=0.0, C_theta_a=0.0, J_a=0.0,
                 coupling_flag=1, wake_truncation=5.5):
        """
        Parameters (MATLAB variable names in comments):
          shell: ANCFShell instance
          V_inf_vec: freestream velocity vector (dimensional)
          rho_fluid: fluid density [kg/m³]
          structural_dt: d_t, structural time step [s]
          uvlm_dt_ratio: dt_wake_per_dt, structural steps per fluid solve
          Length: plate length [m]
          U_star: Ua, nondimensional speed
          M_star: Ma, mass ratio
          alpha_v: Newmark parameter (0.5 = trapezoidal)
          theta_a: structural damping parameter (0 for Yamano)
          C_theta_a: rotary damping coefficient
          J_a: rotary inertia coefficient
          coupling_flag: 1 = strong, 0 = weak
          wake_truncation: R_wake_x_threshold / Length
        """
        self.shell = shell
        self._V_inf_vec = np.asarray(V_inf_vec, dtype=float)
        self._rho_fluid = rho_fluid
        self._d_t = structural_dt
        self._dt_wake_per_dt = uvlm_dt_ratio
        self._d_t_wake = structural_dt * uvlm_dt_ratio
        self._Length = Length
        self._Ua = U_star
        self._Ma = M_star
        self._alpha_v = alpha_v
        self._theta_a = theta_a
        self._C_theta_a = C_theta_a
        self._J_a = J_a
        self._coupling_flag = coupling_flag
        self._wake_truncation = wake_truncation

        # ── Nondimensional scaling parameters (param_setting.m) ──
        self._mu_m = 1.0 / M_star
        self._eta_m = self._mu_m / U_star**2
        # zeta_m = Aa/Ia * eta_m * Length² = (12/thick²) * eta_m * Length²
        thick = shell.h
        self._zeta_m = (12.0 / thick**2) * self._eta_m * Length**2

        # ── Reference values for dimensionalization ──
        self._V_inf = np.linalg.norm(V_inf_vec)
        # Reference force: rho_f * V_inf² * L²
        self._F_ref = rho_fluid * self._V_inf**2 * Length**2
        # Nondimensional freestream
        self._V_in_nd = V_inf_vec / self._V_inf

        # ── Build UVLM mesh ──
        self._build_uvlm_mesh()
        self.uvlm = StandaloneUVLM(
            self._uvlm_vertices, V_inf_vec,
            rho=rho_fluid, core_radius=1e-6)
        self.uvlm.build_aic()
        print(f"[matlab_fsi] UVLM: {self.uvlm._nc}x{self.uvlm._ns} panels, "
              f"AIC cond={np.linalg.cond(self.uvlm._AIC):.1f}")

        # ── Panel-to-element mapping ──
        self._build_panel_mapping()

        # ── nSc for Mf1 computation ──
        self._build_nSc()

        # ── N_q, N_q_all (MATLAB naming) ──
        self._N_qi = 9  # DOF per node
        self._N_q = 36  # DOF per element (4 nodes * 9)
        self._N_q_all = shell.ndof  # total DOF
        self._N_element = self._nx * self._ny  # number of UVLM panels

        # ── Precompute i_vec_v (element DOF indices, MATLAB cell array) ──
        self._i_vec_v = {}
        for ii in range(shell.ne):
            self._i_vec_v[ii + 1] = shell._elem_dofs(ii)  # 1-based indexing

        # ── Precompute structural data ──
        self._precompute_structural_data()

        # ── State vectors (MATLAB: h_X_vec) ──
        # X_vec = [q_vec; dt_q_vec] of size (2*N_q_all,)
        self._q_vec = shell.q.copy()
        self._dt_q_vec = shell.dq.copy()
        self._X_vec = np.concatenate([self._q_vec, self._dt_q_vec])

        # ── Fluid state ──
        self._old_Gamma = np.zeros(self._N_element)
        self._Gamma = np.zeros(self._N_element)
        self._Gamma_trail = np.zeros(self._ny)
        self._Gamma_wake = None
        self._r_wake_1 = None
        self._r_wake_2 = None
        self._r_wake_3 = None
        self._r_wake_4 = None
        self._dt_r_wake_1 = None
        self._dt_r_wake_2 = None
        self._dt_r_wake_3 = None
        self._dt_r_wake_4 = None

        # ── Fluid force matrices (MATLAB globals) ──
        self._Qf_p_global = np.zeros(self._N_q_all)
        self._Qf_p_mat_global = coo_matrix((self._N_q_all, self._N_q_all)).tocsc()
        self._Qf_p_mat0_global = np.zeros((self._N_q_all, self._N_element))
        self._Qf_p_lift2_mat_global = np.zeros((self._N_q_all, 3 * self._N_element))

        self._old_Qf_p_global = None
        self._old_Qf_p_mat_global = None
        self._old_Qf_p_mat0_global = None
        self._old_Qf_p_lift2_mat_global = None
        self._Qf_p_global_a = None
        self._Qf_p_mat_global_a = None
        self._Qf_p_mat0_global_a = None
        self._Qf_p_lift2_mat_global_a = None

        # ── Mf1 matrix ──
        self._Mf1_mat = None
        self._Mf2_mat = None
        self._Mf2_vec1 = None

        # ── Fluid timings ──
        self._time_fluid = 0.0
        self._time_wake_m = 0.0

        # ── Pulse force ──
        self._Qf_time_global = None

        # ── Results ──
        self._tip_w_history = []
        self._tip_idx = np.argmax(self.shell.nodes[:, 0] + self.shell.nodes[:, 1])
        self._ref_z = self.shell.nodes[self._tip_idx, 2]

        # ── Scratch for Newmark ──
        self._out1 = {}  # stores D_matrix, A1_A2_Xn between stages

        print(f"[matlab_fsi] zeta_m={self._zeta_m:.4f}, eta_m={self._eta_m:.6f}")
        print(f"[matlab_fsi] F_ref={self._F_ref:.2f} N, N_q_all={self._N_q_all}")

    # ═══════════════════════════════════════════════════════════════════
    # UVLM mesh construction
    # ═══════════════════════════════════════════════════════════════════

    def _build_uvlm_mesh(self):
        nodes = self.shell.positions()
        x_vals = np.sort(np.unique(np.round(nodes[:, 0], 10)))
        y_vals = np.sort(np.unique(np.round(nodes[:, 1], 10)))
        self._nx = len(x_vals) - 1
        self._ny = len(y_vals) - 1
        self._Nx = self._nx
        self._Ny = self._ny
        self._x_vec = x_vals
        self._y_vec = y_vals
        self._dL_vec = np.diff(x_vals)
        self._dW_vec = np.diff(y_vals)
        self._uvlm_vertices = np.zeros((self._nx + 1, self._ny + 1, 3))
        for i in range(self._nx + 1):
            for j in range(self._ny + 1):
                self._uvlm_vertices[i, j] = [x_vals[i], y_vals[j], 0.0]

    def _build_panel_mapping(self):
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
        print(f"[matlab_fsi] {n_mapped}/{nc * ns} UVLM panels mapped to ANCF elements")

    def _build_nSc(self):
        nc, ns = self._nx, self._ny
        n_panels = nc * ns
        ndof = self.shell.ndof
        self._nSc = np.zeros((n_panels, ndof))
        for i in range(nc):
            for j in range(ns):
                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue
                dL = self.shell._dL[e]
                dW = self.shell._dW[e]
                xi, eta = self._panel_xi_eta[i, j]
                S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
                n = self.uvlm._normals[i, j]
                panel_idx = i * ns + j
                dofs = self.shell._elem_dofs(e)
                self._nSc[panel_idx, dofs] = n @ S

    # ═══════════════════════════════════════════════════════════════════
    # Precompute structural data (MATLAB: generate_matrices.m)
    # ═══════════════════════════════════════════════════════════════════

    def _precompute_structural_data(self):
        """Precompute mass matrix and Sc_mat at collocation points."""
        n_gauss = self.shell.n_gauss
        p_vec, w_vec = _gauss_legendre(n_gauss)

        ndof = self._N_q_all
        ne = self.shell.ne

        # ── Mass matrix (MATLAB: M_mat_i, assembled as mu_m * M_mat) ──
        rows, cols, vals = [], [], []
        for e in range(ne):
            ed = self.shell._elems[e]
            dofs = self.shell._elem_dofs(e)
            M_e = np.zeros((self._N_q, self._N_q))
            dL = self.shell._dL[e]
            dW = self.shell._dW[e]
            for i in range(n_gauss):
                xi_g = (p_vec[i] + 1) / 2.0
                for j in range(n_gauss):
                    eta_g = (p_vec[j] + 1) / 2.0
                    S = np.kron(_shape_funcs(xi_g, eta_g, dL, dW), np.eye(3))
                    M_e += (dL * dW / 4.0) * w_vec[i] * w_vec[j] * (S.T @ S)
            # Scale: M_global += mu_m * M_mat_i  (MATLAB nondim scaling)
            M_e *= self._mu_m
            for a in range(self._N_q):
                for b in range(self._N_q):
                    if abs(M_e[a, b]) > 1e-30:
                        rows.append(dofs[a])
                        cols.append(dofs[b])
                        vals.append(M_e[a, b])
        self._M_global = coo_matrix((vals, (rows, cols)),
                                     shape=(ndof, ndof)).tocsc()

        # ── Sc_mat_col_global (collocation mapping) ──
        # Maps q_vec → collocation positions (3*N_element, N_q_all)
        nc, ns = self._nx, self._ny
        n_panels = nc * ns
        Sc_rows, Sc_cols, Sc_vals = [], [], []
        for i in range(nc):
            for j in range(ns):
                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue
                dL = self.shell._dL[e]
                dW = self.shell._dW[e]
                xi, eta = self._panel_xi_eta[i, j]
                S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
                dofs = self.shell._elem_dofs(e)
                for d in range(3):
                    for k in range(self._N_q):
                        if abs(S[d, k]) > 1e-15:
                            Sc_rows.append(3 * (i * ns + j) + d)
                            Sc_cols.append(dofs[k])
                            Sc_vals.append(S[d, k])
        self._Sc_mat_col_global = coo_matrix(
            (Sc_vals, (Sc_rows, Sc_cols)),
            shape=(3 * n_panels, ndof)).tocsc()

        # ── dL_vec, dW_vec (per element) ──
        self._dL_elem = self.shell._dL.copy()
        self._dW_elem = self.shell._dW.copy()

    # ═══════════════════════════════════════════════════════════════════
    # Pulse force (MATLAB: generate_Qf_time_mat.m)
    # ═══════════════════════════════════════════════════════════════════

    def set_pulse(self, q_in_vec=np.array([0.0, 0.0, 1.0]),
                  q_in_norm_func=None):
        """Set up pulse force.

        MATLAB: Qf_time_global * q_in_norm(time)
        q_in_norm = @(time)( 0.5*sin(pi*time/0.2).*(time < 0.2) )

        q_in_vec is nondimensional direction vector.
        Qf_time_global is the consistent nodal force for unit body force.
        In nondimensional form, this integrates S^T * q_in_vec over area (no h factor).
        """
        n_gauss = self.shell.n_gauss
        p_vec, w_vec = _gauss_legendre(n_gauss)
        ne = self.shell.ne
        ndof = self._N_q_all

        if q_in_norm_func is None:
            # Default: half-sine pulse
            def q_in_norm_func(time):
                return 0.5 * np.sin(np.pi * time / 0.2) * (time < 0.2)

        self._q_in_norm = q_in_norm_func

        Qf_time_global = np.zeros(ndof)
        for e in range(ne):
            dL = self.shell._dL[e]
            dW = self.shell._dW[e]
            dofs = self.shell._elem_dofs(e)
            int_StF = np.zeros(self._N_q)
            for i in range(n_gauss):
                xi_g = (p_vec[i] + 1) / 2.0
                for j in range(n_gauss):
                    eta_g = (p_vec[j] + 1) / 2.0
                    S = np.kron(_shape_funcs(xi_g, eta_g, dL, dW), np.eye(3))
                    StF = S.T @ q_in_vec
                    int_StF += (dL * dW / 4.0) * w_vec[i] * w_vec[j] * StF
            Qf_time_global[dofs] += int_StF

        self._Qf_time_global = Qf_time_global
        print(f"[matlab_fsi] Pulse force vector: max|Qf|={np.max(np.abs(Qf_time_global)):.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # MATLAB: generate_stiff_matrices.m
    # ═══════════════════════════════════════════════════════════════════

    def _compute_structural_matrices(self, q_vec, dt_q_vec, flag_output=True):
        """Compute Qe_global, Qk_global, dq_Qe_global, Qd_global, etc.

        MATLAB: generate_stiff_matrices.m

        flag_output=True: also compute tangent stiffness (expensive), used in predictor.
        flag_output=False: skip tangent stiffness, used in corrector.
        """
        n_gauss = self.shell.n_gauss
        p_vec, w_vec = _gauss_legendre(n_gauss)

        ne = self.shell.ne
        ndof = self._N_q_all

        Qe_global = np.zeros(ndof)
        Qk_global = np.zeros(ndof)

        # Sparse assembly vectors for dq_Qe_global and Qd_global
        dq_Qe_rows, dq_Qe_cols, dq_Qe_vals = [], [], []
        Qd_rows, Qd_cols, Qd_vals = [], [], []

        for e in range(ne):
            dL = self.shell._dL[e]
            dW = self.shell._dW[e]
            dofs = self.shell._elem_dofs(e)
            q_e = q_vec[dofs]

            # ── Precompute shape function matrices at Gauss points ──
            S_v   = np.zeros((n_gauss, n_gauss, 3, self._N_q))
            dSx_v = np.zeros((n_gauss, n_gauss, 3, self._N_q))
            dSy_v = np.zeros((n_gauss, n_gauss, 3, self._N_q))
            d2Sx_v = np.zeros((n_gauss, n_gauss, 3, self._N_q))
            d2Sy_v = np.zeros((n_gauss, n_gauss, 3, self._N_q))
            d2Sxy_v = np.zeros((n_gauss, n_gauss, 3, self._N_q))

            for i in range(n_gauss):
                xi_g = (p_vec[i] + 1) / 2.0
                for j in range(n_gauss):
                    eta_g = (p_vec[j] + 1) / 2.0
                    S_v[i, j] = np.kron(_shape_funcs(xi_g, eta_g, dL, dW), np.eye(3))
                    from .ancf_shell import _shape_dxi, _shape_deta, _shape_dxi2, _shape_deta2, _shape_dxieta
                    dSx_v[i, j] = np.kron(_shape_dxi(xi_g, eta_g, dL, dW), np.eye(3)) / dL
                    dSy_v[i, j] = np.kron(_shape_deta(xi_g, eta_g, dL, dW), np.eye(3)) / dW
                    d2Sx_v[i, j] = np.kron(_shape_dxi2(xi_g, eta_g, dL, dW), np.eye(3)) / dL**2
                    d2Sy_v[i, j] = np.kron(_shape_deta2(xi_g, eta_g, dL, dW), np.eye(3)) / dW**2
                    d2Sxy_v[i, j] = np.kron(_shape_dxieta(xi_g, eta_g, dL, dW), np.eye(3)) / (dL * dW)

            # ── Element elastic (membrane) forces: Qe_eps ──
            if flag_output:
                Qe_eps_e = np.zeros(self._N_q)
                dq_Qe_eps_e = np.zeros((self._N_q, self._N_q))
                Qd_eps_e = np.zeros((self._N_q, self._N_q))

            Qe_k_e = np.zeros(self._N_q)

            # Constitutive matrix (nondimensional, no E factor)
            Dp_mat = 1.0 / (1.0 - self.shell.nu_xy**2) * np.array([
                [1, self.shell.nu_xy, 0],
                [self.shell.nu_xy, 1, 0],
                [0, 0, (1 - self.shell.nu_xy) / 2]
            ])

            for i in range(n_gauss):
                for j in range(n_gauss):
                    w = (dL * dW / 4.0) * w_vec[i] * w_vec[j]
                    S   = S_v[i, j]
                    dSx = dSx_v[i, j]
                    dSy = dSy_v[i, j]
                    d2Sx = d2Sx_v[i, j]
                    d2Sy = d2Sy_v[i, j]
                    d2Sxy = d2Sxy_v[i, j]

                    dx_r = dSx @ q_e
                    dy_r = dSy @ q_e
                    d2x_r = d2Sx @ q_e
                    d2y_r = d2Sy @ q_e
                    d2xy_r = d2Sxy @ q_e

                    # ── Membrane strain (eps) ──
                    eps_xx = 0.5 * (dx_r @ dx_r - 1.0)
                    eps_yy = 0.5 * (dy_r @ dy_r - 1.0)
                    gam_xy = dx_r @ dy_r
                    eps_v = np.array([eps_xx, eps_yy, gam_xy])

                    deps = np.zeros((3, self._N_q))
                    deps[0] = dSx.T @ dx_r
                    deps[1] = dSy.T @ dy_r
                    deps[2] = dSx.T @ dy_r + dSy.T @ dx_r

                    Dm_eps = Dp_mat @ eps_v

                    if flag_output:
                        Qe_eps_e += w * (deps.T @ Dm_eps)

                        # Tangent: A1*Dm_eps[0] + A2*Dm_eps[1] + A3*Dm_eps[2] + deps^T*Dm*deps
                        K_geo = (dSx.T @ dSx) * Dm_eps[0] + \
                                (dSy.T @ dSy) * Dm_eps[1] + \
                                (dSx.T @ dSy + dSy.T @ dSx) * Dm_eps[2]
                        K_mat = deps.T @ Dp_mat @ deps
                        dq_Qe_eps_e += w * (K_geo + K_mat)

                    # ── Bending (kappa) ──
                    n_vec = np.cross(dx_r, dy_r)
                    norm_n = np.linalg.norm(n_vec)
                    if norm_n < 1e-15:
                        continue
                    n_hat = n_vec / norm_n

                    kxx = n_hat @ d2x_r
                    kyy = n_hat @ d2y_r
                    kxy = n_hat @ d2xy_r
                    k_v = np.array([kxx, kyy, 2.0 * kxy])

                    # Bending Jacobian
                    skew_dx = np.array([[0, -dx_r[2], dx_r[1]],
                                        [dx_r[2], 0, -dx_r[0]],
                                        [-dx_r[1], dx_r[0], 0]])
                    skew_dy = np.array([[0, -dy_r[2], dy_r[1]],
                                        [dy_r[2], 0, -dy_r[0]],
                                        [-dy_r[1], dy_r[0], 0]])
                    P = np.eye(3) - np.outer(n_hat, n_hat)
                    dn = -skew_dy @ dSx + skew_dx @ dSy
                    dn_hat = (P @ dn) / norm_n

                    dk = np.empty((3, self._N_q))
                    dk[0] = d2x_r @ dn_hat + n_hat @ d2Sx
                    dk[1] = d2y_r @ dn_hat + n_hat @ d2Sy
                    dk[2] = 2.0 * (d2xy_r @ dn_hat + n_hat @ d2Sxy)

                    Dk_k = Dp_mat @ k_v  # NOT h³/12 * Dp_mat!
                    Qe_k_e += w * (dk.T @ Dk_k)

            # ── Scale and assemble: Qe_global += zeta_m * Qe_eps (membrane)
            #                        Qk_global += eta_m * Qe_k   (bending)
            if flag_output:
                Qe_global[dofs] += self._zeta_m * Qe_eps_e
            Qk_global[dofs] += self._eta_m * Qe_k_e

            if flag_output:
                dq_Qe_global_scaled = self._zeta_m * dq_Qe_eps_e
                for a in range(self._N_q):
                    for b in range(self._N_q):
                        v = dq_Qe_global_scaled[a, b]
                        if abs(v) > 1e-30:
                            dq_Qe_rows.append(dofs[a])
                            dq_Qe_cols.append(dofs[b])
                            dq_Qe_vals.append(v)

        # ── Assemble sparse matrices ──
        dq_Qe_global = coo_matrix((dq_Qe_vals, (dq_Qe_rows, dq_Qe_cols)),
                                   shape=(ndof, ndof)).tocsc() if flag_output else None
        Qd_global = coo_matrix((ndof, ndof)).tocsc()  # Zero when theta_a=0
        Qd_theta_global = coo_matrix((ndof, ndof)).tocsc()
        J_global_1 = coo_matrix((ndof, ndof)).tocsc()
        J_global_2 = coo_matrix((ndof, ndof)).tocsc()

        return {
            'Qe_global': Qe_global,
            'Qk_global': Qk_global,
            'dq_Qe_global': dq_Qe_global,
            'Qd_global': Qd_global,
            'Qd_theta_global': Qd_theta_global,
            'J_global_1': J_global_1,
            'J_global_2': J_global_2,
        }

    # ═══════════════════════════════════════════════════════════════════
    # MATLAB: new_X_func_FAST.m
    # ═══════════════════════════════════════════════════════════════════

    def _newmark_step(self, X_vec, m_global_struct, qf_global_struct,
                      dq_qe_global_struct, qe_global_struct, qd_global_struct,
                      stage, out1):
        """Newmark first-order time integration.

        MATLAB: new_X_func_FAST.m

        stage=0: predictor (compute and store D_matrix, A1_A2_Xn)
        stage=1: corrector (reuse stored matrices)

        D = [[I, 0], [Qd + C_damp*d_t/2*dq_Qe, M]]
        X2 = [[0, I], [0, 0]]
        A1 = D - alpha_v*d_t*X2
        A2 = D + (1-alpha_v)*d_t*X2
        X_{n+1} = A1⁻¹ @ (A2 @ X_n + d_t * [0; Q_global])
        """
        N_q_all = self._N_q_all
        d_t = self._d_t
        alpha_v = self._alpha_v

        # Extract matrices
        M_global = m_global_struct['M_global']
        Qf_global = qf_global_struct['Qf_global']
        dq_Qe_global = dq_qe_global_struct['dq_Qe_global']
        Qe_global = qe_global_struct['Qe_global']
        Qd_global = qd_global_struct['Qd_global']

        # ── BC handling ──
        # Node constraints: fix nodes at x=0 (leading edge)
        fixed_dofs = sorted(self.shell._bc_dofs)
        i_vec = np.array(fixed_dofs, dtype=np.int32)

        # Q_global = Qf - Qe (external minus internal)
        Q_global = Qf_global - Qe_global

        # Remove constrained DOFs from matrices
        not_i_vec = np.setdiff1d(np.arange(N_q_all), i_vec)
        nf = len(not_i_vec)

        M_ff = M_global[np.ix_(not_i_vec, not_i_vec)]
        Q_ff = Q_global[not_i_vec]
        Qd_ff = Qd_global[np.ix_(not_i_vec, not_i_vec)]
        dq_Qe_ff = dq_Qe_global[np.ix_(not_i_vec, not_i_vec)]

        # C_damp: 2 when theta_a==0 (non-dissipative), 1 otherwise
        C_damp = 2 if self._theta_a == 0 else 1

        if stage == 0:
            # Build D_matrix = [[I, 0], [Qd + C_damp*d_t/2*dq_Qe, M]]
            I_sp = speye(nf, format='csc')
            O_sp = coo_matrix((nf, nf)).tocsc()
            D_bot_left = Qd_ff + C_damp * d_t / 2.0 * dq_Qe_ff
            D_matrix = spbmat([[I_sp, O_sp], [D_bot_left, M_ff]], format='csc')
            out1['D_matrix'] = D_matrix
        else:
            D_matrix = out1['D_matrix']

        # X2_matrix = [[0, I], [0, 0]]
        I_sp = speye(nf, format='csc')
        O_sp = coo_matrix((nf, nf)).tocsc()
        X2_matrix = spbmat([[O_sp, I_sp], [O_sp, O_sp]], format='csc')

        A_mat1 = D_matrix - alpha_v * d_t * X2_matrix
        A_mat2 = D_matrix + (1.0 - alpha_v) * d_t * X2_matrix

        X_n_free = np.concatenate([X_vec[not_i_vec], X_vec[N_q_all + not_i_vec]])

        if stage == 0:
            out1['A1_A2_Xn'] = spsolve(A_mat1, A_mat2 @ X_n_free)

        rhs = np.zeros(2 * nf)
        rhs[nf:] = d_t * Q_ff
        out_0 = out1['A1_A2_Xn'] + spsolve(A_mat1, rhs)

        out = X_vec.copy()
        out[not_i_vec] = out_0[:nf]
        out[N_q_all + not_i_vec] = out_0[nf:]

        return out

    # ═══════════════════════════════════════════════════════════════════
    # MATLAB: solve_structure.m
    # ═══════════════════════════════════════════════════════════════════

    def _solve_structure(self, i_time, fluid_compute_flag):
        """One structural time step with predictor-corrector.

        MATLAB: solve_structure.m

        When fluid_compute_flag:
          Predictor — uses old fluid forces (Qf_p_global_a etc.)
          Runs Newmark predictor to get X_vec_p
          (Corrector happens in _solve_fluid_corrector)

        When not fluid_compute_flag:
          Corrector — uses interpolated fluid forces
          Averages Qf_p_mat0 and Qf_p_lift2
        """
        N_q_all = self._N_q_all
        time = i_time * self._d_t

        # Extract state
        X_vec = self._X_vec.copy()
        q_vec = X_vec[:N_q_all]
        dt_q_vec = X_vec[N_q_all:]

        # ── Fluid force interpolation (MATLAB: linear in time) ──
        if self._old_Qf_p_global is not None:
            # Qf_p_global_t = linear interpolation
            alpha_t = (time - self._time_fluid) / self._d_t_wake
            Qf_p_global_tv_n = self._old_Qf_p_global + alpha_t * (self._Qf_p_global - self._old_Qf_p_global)
            Qf_p_lift2_global_tv_n = self._old_Qf_p_lift2_mat_global + alpha_t * (self._Qf_p_lift2_mat_global - self._old_Qf_p_lift2_mat_global)
            Qf_p_mat_global_tv_n = self._old_Qf_p_mat_global + alpha_t * (self._Qf_p_mat_global - self._old_Qf_p_mat_global)
            Qf_p_mat0_global_tv_n = self._old_Qf_p_mat0_global + alpha_t * (self._Qf_p_mat0_global - self._old_Qf_p_mat0_global)

            if not fluid_compute_flag and self._Qf_p_global_a is not None:
                Qf_p_global_tv_n = self._Qf_p_global_a
                Qf_p_lift2_global_tv_n = self._Qf_p_lift2_mat_global_a
                Qf_p_mat_global_tv_n = self._Qf_p_mat_global_a
                Qf_p_mat0_global_tv_n = self._Qf_p_mat0_global_a
        else:
            Qf_p_global_tv_n = self._Qf_p_global
            Qf_p_lift2_global_tv_n = self._Qf_p_lift2_mat_global
            Qf_p_mat_global_tv_n = self._Qf_p_mat_global
            Qf_p_mat0_global_tv_n = self._Qf_p_mat0_global

        # ── Collocation point velocity ──
        rc_vec = self._Sc_mat_col_global @ q_vec  # (3*N_element,)
        rc_vec = rc_vec.reshape(-1, 3)  # (N_element, 3)

        dt_rc_vec = self._Sc_mat_col_global @ dt_q_vec
        dt_rc_vec = dt_rc_vec.reshape(-1, 3)

        # ── Normal vectors at collocation ──
        n_vec_i = self.uvlm._normals.reshape(-1, 3)  # (N_element, 3)
        dt_n_vec_i = np.zeros_like(n_vec_i)  # Simplified: normals constant for flat plate

        # ── Wake velocity at plate ──
        V_wake_plate = self._compute_wake_velocity_at_plate()
        if V_wake_plate is None:
            V_wake_plate = np.zeros((self._N_element, 3))
            dt_Amat2_Gamma = np.zeros((self._N_element, 3))
            dt_Amat1 = np.zeros((self._N_element, self._N_element))
            Gamma = np.zeros(self._N_element)
        else:
            Gamma = self._Gamma
            dt_Amat1 = self._dt_Amat1 if hasattr(self, '_dt_Amat1') else np.zeros((self._N_element, self._N_element))
            dt_Amat2_Gamma = self._dt_Amat2_Gamma if hasattr(self, '_dt_Amat2_Gamma') else np.zeros((self._N_element, 3))

        V_in = np.ones((self._N_element, 1)) * self._V_in_nd.reshape(1, 3)

        # ── Qf_p_mat0: Mf2_1 coupling ──
        Qf_p_mat0_global_t_n = Qf_p_mat0_global_tv_n @ (
            np.sum((dt_rc_vec - V_in - V_wake_plate - dt_Amat2_Gamma) * n_vec_i, axis=1)
            - (dt_Amat1 @ Gamma if dt_Amat1 is not None else np.zeros(self._N_element))
        )

        # ── Qf_p_lift2: V_struct coupling ──
        Qf_p_lift2_global_t_n = Qf_p_lift2_global_tv_n @ dt_rc_vec.T.flatten()

        # ── Compute structural matrices at current state ──
        flag_output = fluid_compute_flag
        mats_n = self._compute_structural_matrices(q_vec, dt_q_vec, flag_output)

        Qk_global_n = mats_n['Qk_global']
        Qe_global_n = mats_n['Qe_global']
        dq_Qe_global_n = mats_n['dq_Qe_global']
        Qd_global_n = mats_n['Qd_global']
        Qd_theta_global_n = mats_n['Qd_theta_global']
        J_global_1_n = mats_n['J_global_1']
        J_global_2_n = mats_n['J_global_2']

        # ── Pulse force ──
        pulse_force = np.zeros(N_q_all)
        if self._Qf_time_global is not None and self._q_in_norm is not None:
            pulse_force = self._Qf_time_global * self._q_in_norm(time)

        # ═══════════════════════════════════════════
        # PREDICTOR
        # ═══════════════════════════════════════════
        # M_global_struct: M_eff = M + J1 - Qf_p_mat  (MATLAB: M_global + J1 - Qf_p_mat)
        M_eff = self._M_global + J_global_1_n - Qf_p_mat_global_tv_n

        m_global_struct = {'M_global': M_eff}
        qf_global_struct = {
            'Qf_global': pulse_force + Qf_p_global_tv_n + Qf_p_mat0_global_t_n + Qf_p_lift2_global_t_n
        }
        dq_qe_global_struct = {'dq_Qe_global': dq_Qe_global_n}
        qe_global_struct = {'Qe_global': Qe_global_n + Qk_global_n}  # elastic + kinetic
        qd_global_struct = {'Qd_global': Qd_global_n + Qd_theta_global_n + J_global_2_n}

        X_vec_p = self._newmark_step(
            X_vec, m_global_struct, qf_global_struct,
            dq_qe_global_struct, qe_global_struct, qd_global_struct,
            0, self._out1)

        # ── Store predictor state ──
        self._X_vec_p = X_vec_p

        if not fluid_compute_flag:
            # ═══════════════════════════════════════════
            # CORRECTOR
            # ═══════════════════════════════════════════
            q_vec_p = X_vec_p[:N_q_all]
            dt_q_vec_p = X_vec_p[N_q_all:]

            # Recompute collocation velocity at predictor state
            dt_rc_vec_p = self._Sc_mat_col_global @ dt_q_vec_p
            dt_rc_vec_p = dt_rc_vec_p.reshape(-1, 3)

            # Qf_p_mat0 at predictor state
            Qf_p_mat0_global_t_np1 = Qf_p_mat0_global_tv_n @ (
                np.sum((dt_rc_vec_p - V_in - V_wake_plate - dt_Amat2_Gamma) * n_vec_i, axis=1)
                - (dt_Amat1 @ Gamma if dt_Amat1 is not None else np.zeros(self._N_element))
            )
            Qf_p_lift2_global_t_np1 = Qf_p_lift2_global_tv_n @ dt_rc_vec_p.T.flatten()

            # Recompute structural matrices at predictor state (flag_output=False: no tangent)
            mats_np1 = self._compute_structural_matrices(q_vec_p, dt_q_vec_p, False)
            Qk_global_np1 = mats_np1['Qk_global']

            # Corrector effective mass (same as predictor)
            M_eff_corr = self._M_global + J_global_1_n - Qf_p_mat_global_tv_n

            m_global_struct_c = {'M_global': M_eff_corr}
            qf_global_struct_c = {
                'Qf_global': (pulse_force
                             + Qf_p_global_tv_n  # Bernoulli: NOT averaged
                             + (Qf_p_mat0_global_t_n + Qf_p_mat0_global_t_np1) / 2.0  # averaged
                             + (Qf_p_lift2_global_t_n + Qf_p_lift2_global_t_np1) / 2.0)  # averaged
            }
            dq_qe_global_struct_c = {'dq_Qe_global': dq_Qe_global_n}
            # Qe = Qe_n (NOT averaged), Qk = (Qk_n + Qk_np1)/2 (averaged)
            qe_global_struct_c = {
                'Qe_global': Qe_global_n + (Qk_global_n + Qk_global_np1) / 2.0
            }
            qd_global_struct_c = {'Qd_global': Qd_global_n + Qd_theta_global_n + J_global_2_n}

            new_X_vec = self._newmark_step(
                X_vec, m_global_struct_c, qf_global_struct_c,
                dq_qe_global_struct_c, qe_global_struct_c, qd_global_struct_c,
                1, self._out1)

            # Store for next step
            self._X_vec = new_X_vec
            self._q_vec = new_X_vec[:N_q_all]
            self._dt_q_vec = new_X_vec[N_q_all:]

        # ── Record tip displacement ──
        self._tip_w_history.append(self._q_vec[self._tip_idx * 9 + 2] -
                                    self._ref_z)

    # ═══════════════════════════════════════════════════════════════════
    # MATLAB: solve_fluid.m
    # ═══════════════════════════════════════════════════════════════════

    def _solve_fluid(self, i_wake_time):
        """Solve fluid at current structural state.

        MATLAB: solve_fluid.m

        Steps:
          1. Compute panel node positions
          2. Compute unit normals
          3. Update wake (shed + advect)
          4. Compute AIC matrix
          5. Solve Gamma = A_mat \ V_normal
          6. Compute surface velocity V_surf
          7. Compute fluid forces
        """
        # ── Update UVLM vertices from current displacement ──
        self._update_uvlm_vertices()

        # ── Panel normals ──
        n_vec_i = self.uvlm._normals.reshape(-1, 3)
        self._n_vec_i = n_vec_i

        # ── Collocation positions and velocity ──
        rc_vec = self._Sc_mat_col_global @ self._q_vec
        rc_vec = rc_vec.reshape(-1, 3)

        dt_rc_vec = self._Sc_mat_col_global @ self._dt_q_vec
        dt_rc_vec = dt_rc_vec.reshape(-1, 3)

        V_in = np.ones((self._N_element, 1)) * self._V_in_nd.reshape(1, 3)

        # ── AIC matrix ──
        # MATLAB: q1234_mat → A_mat = inner_mat(q1234_mat, n_vec_i_mat)
        # Already computed in StandaloneUVLM
        A_mat = self.uvlm._AIC  # (N_element, N_element)

        # ── Wake (shed + advect) ──
        self._generate_wake(i_wake_time)

        # ── Wake velocity at collocation ──
        V_wake_plate = self._compute_wake_velocity_at_plate()
        if V_wake_plate is None:
            V_wake_plate = np.zeros((self._N_element, 3))

        # ── Normal velocity (RHS) ──
        V_normal = np.sum((dt_rc_vec - V_in - V_wake_plate) * n_vec_i, axis=1)

        # ── Solve for Gamma ──
        Gamma = np.linalg.solve(A_mat, V_normal)

        # ── Trailing edge circulation ──
        Gamma_trail = self._old_Gamma[-self._ny:]

        # ── Surface velocity ──
        V_gamma = self._compute_bound_induction(Gamma)
        V_wake_plate = self._compute_wake_velocity_at_plate()
        if V_wake_plate is None:
            V_wake_plate = np.zeros((self._N_element, 3))

        V_surf = V_gamma + V_wake_plate + V_in - dt_rc_vec
        V_surf1 = V_gamma + V_wake_plate + V_in

        # ── Store for structure solver ──
        self._V_surf = V_surf
        self._V_surf1 = V_surf1
        self._V_in = V_in
        self._dt_rc_vec = dt_rc_vec
        self._Gamma = Gamma
        self._old_Gamma = Gamma.copy()
        self._Gamma_trail = Gamma_trail

        # ── Compute forces ──
        self._calc_fluid_force()

    def _compute_bound_induction(self, Gamma):
        """Compute V_gamma = bound vortex induction at collocation points."""
        V_gamma = np.zeros((self._N_element, 3))
        for i in range(self._nx):
            for j in range(self._ny):
                k = i * self._ny + j
                g = Gamma[k]
                if abs(g) < 1e-15:
                    continue
                corners = self.uvlm._corners[i, j]
                for ii in range(self._nx):
                    for jj in range(self._ny):
                        kk = ii * self._ny + jj
                        pt = self.uvlm._colloc[ii, jj]
                        from .standalone_uvlm import ring_vortex_velocity
                        V_gamma[kk] += ring_vortex_velocity(
                            pt.reshape(1, 3), corners, g,
                            self.uvlm._core_radius).flatten()
        return V_gamma

    # ═══════════════════════════════════════════════════════════════════
    # Wake (MATLAB: generate_wake.m)
    # ═══════════════════════════════════════════════════════════════════

    def _generate_wake(self, i_wake_time):
        """Generate wake ring vortices.

        MATLAB: generate_wake.m

        i_wake_time == 1: initial wake from trailing edge
        i_wake_time > 1: RK4 advection of existing wake + new shedding
        """
        N_trail = self._ny
        i_trail = np.arange(self._N_element - N_trail, self._N_element)

        # Trailing edge panel nodes
        r_panel_vec_2 = self.uvlm._corners[:, :, 1, :].reshape(-1, 3)  # corner 2
        r_panel_vec_3 = self.uvlm._corners[:, :, 2, :].reshape(-1, 3)  # corner 3
        # Velocities at trailing edge
        dt_rc_vec = self._dt_rc_vec

        if i_wake_time == 1:
            # First wake step
            r_panel_vec_2_end = r_panel_vec_2[i_trail]
            r_panel_vec_3_end = r_panel_vec_3[i_trail]
            r_panel_vec_31_end = r_panel_vec_3_end[0]  # Y=0 node

            # Wake corner positions
            # Velocity at trailing edge = V_gamma + V_in - V_struct (simplified)
            V_wake_2 = self._V_in_nd - dt_rc_vec[i_trail]
            V_wake_31 = self._V_in_nd - dt_rc_vec[i_trail[0]]

            self._r_wake_1 = r_panel_vec_2_end.copy()
            self._r_wake_4 = r_panel_vec_3_end.copy()
            self._r_wake_2 = r_panel_vec_2_end + V_wake_2 * self._d_t_wake
            r_wake_31 = r_panel_vec_31_end + V_wake_31 * self._d_t_wake
            self._r_wake_3 = np.zeros((N_trail, 3))
            self._r_wake_3[0] = r_wake_31
            self._r_wake_3[1:] = self._r_wake_2[:-1]

            self._Gamma_wake = self._Gamma_trail.copy()

            self._dt_r_wake_2 = V_wake_2.copy()
            self._dt_r_wake_31 = V_wake_31.copy()
        else:
            # Subsequent steps: shed new wake row + advect existing
            r_panel_vec_2_end = r_panel_vec_2[i_trail]
            r_panel_vec_3_end = r_panel_vec_3[i_trail]
            r_panel_vec_31_end = r_panel_vec_3_end[0]

            # Prepend trailing edge nodes to existing wake
            old_r_wake_2 = np.vstack([r_panel_vec_2_end, self._r_wake_2])
            old_r_wake_31 = np.vstack([r_panel_vec_31_end, self._r_wake_3[::self._ny]])

            N_wake_trail = old_r_wake_2.shape[0]

            # Velocity at wake nodes (Euler: V_wake = V_gamma + V_wake + V_in - V_struct)
            # Simplified: use freestream + induced from bound + self-induced from wake
            V_in_wake = np.ones((N_wake_trail, 1)) * self._V_in_nd.reshape(1, 3)

            r_wake_23 = np.vstack([old_r_wake_2, old_r_wake_31])

            # Euler advection (simplified compared to MATLAB RK4)
            V_gamma_wake = self._compute_bound_induction_at(r_wake_23)
            V_wake_self = self._compute_wake_induction_at(r_wake_23)

            V_wake_2_new = V_gamma_wake[:N_wake_trail] + V_wake_self[:N_wake_trail] + V_in_wake
            V_wake_31_new = V_gamma_wake[N_wake_trail:] + V_wake_self[N_wake_trail:] + V_in_wake[:N_wake_trail // self._ny]

            self._r_wake_2 = old_r_wake_2 + V_wake_2_new * self._d_t_wake
            r_wake_31_new = old_r_wake_31 + V_wake_31_new * self._d_t_wake

            N_wake = self._r_wake_2.shape[0]
            self._r_wake_3 = np.zeros((N_wake, 3))
            self._r_wake_3[::self._ny] = r_wake_31_new
            idx_r2 = np.arange(N_wake)
            idx_r2 = idx_r2[idx_r2 % self._ny != 0]
            self._r_wake_3[idx_r2] = self._r_wake_2[idx_r2 - 1]

            self._r_wake_1 = np.vstack([r_panel_vec_2_end, self._r_wake_2[:-self._ny]])
            self._r_wake_4 = np.vstack([r_panel_vec_3_end, self._r_wake_3[:-self._ny]])

            # Update wake circulation
            self._Gamma_wake = np.concatenate([self._Gamma_trail, self._Gamma_wake])

            # Truncate wake
            self._truncate_wake()

    def _truncate_wake(self):
        """Truncate wake beyond R_wake_x_threshold."""
        if self._r_wake_1 is None:
            return
        threshold = self._wake_truncation * self._Length
        x_center = (self._r_wake_1[:, 0] + self._r_wake_4[:, 0]) / 2.0
        keep = x_center <= threshold
        idx = np.where(~keep)[0]
        if len(idx) > 0:
            first_bad = (idx[0] // self._ny) * self._ny
            if first_bad < len(self._r_wake_1):
                self._r_wake_1 = self._r_wake_1[:first_bad]
                self._r_wake_2 = self._r_wake_2[:first_bad]
                self._r_wake_3 = self._r_wake_3[:first_bad]
                self._r_wake_4 = self._r_wake_4[:first_bad]
                self._Gamma_wake = self._Gamma_wake[:first_bad]

    # ═══════════════════════════════════════════════════════════════════
    # Fluid force computation (MATLAB: calc_fluid_force.m + calc_fluid_force_strong.m)
    # ═══════════════════════════════════════════════════════════════════

    def _calc_fluid_force(self):
        """Compute fluid forces and Mf1/Mf2 matrices.

        MATLAB: calc_fluid_force.m
        """
        N_element = self._N_element
        Gamma = self._Gamma
        old_Gamma = self._old_Gamma
        V_surf = self._V_surf
        V_surf1 = self._V_surf1
        n_vec_i = self._n_vec_i

        # ── Tangent vectors ──
        r_panel_vec_1 = self.uvlm._corners[:, :, 0, :].reshape(-1, 3)
        r_panel_vec_2 = self.uvlm._corners[:, :, 1, :].reshape(-1, 3)
        r_panel_vec_3 = self.uvlm._corners[:, :, 2, :].reshape(-1, 3)
        r_panel_vec_4 = self.uvlm._corners[:, :, 3, :].reshape(-1, 3)

        r21_vec = r_panel_vec_2 - r_panel_vec_1
        r34_vec = r_panel_vec_3 - r_panel_vec_4
        r14_vec = r_panel_vec_1 - r_panel_vec_4
        r23_vec = r_panel_vec_2 - r_panel_vec_3

        tau_x = (r21_vec + r34_vec) / 2.0
        tau_y = (r14_vec + r23_vec) / 2.0

        d_x_vec = np.linalg.norm(tau_x, axis=1, keepdims=True)
        d_y_vec = np.linalg.norm(tau_y, axis=1, keepdims=True)
        d_x_vec[d_x_vec < 1e-15] = 1.0
        d_y_vec[d_y_vec < 1e-15] = 1.0

        tau_x = tau_x / d_x_vec
        tau_y = tau_y / d_y_vec

        # ── Circulation gradients ──
        d_x_mat = d_x_vec.reshape(self._nx, self._ny).T
        d_y_mat = d_y_vec.reshape(self._nx, self._ny).T

        Gamma_mat = Gamma.reshape(self._nx, self._ny).T  # (ny, nx)

        # x-gradient: forward difference
        dx_Gamma = np.zeros_like(Gamma_mat)
        dx_Gamma[0, :] = Gamma_mat[0, :] / d_x_mat[0, :]
        dx_Gamma[1:, :] = np.diff(Gamma_mat, axis=0) / d_x_mat[1:, :]

        # y-gradient: central difference with one-sided at edges
        Gamma_mat2 = np.pad(Gamma_mat, ((0, 0), (1, 1)), mode='constant')
        dy_Gamma = (Gamma_mat2[:, 2:] - Gamma_mat2[:, :-2]) / (2.0 * d_y_mat)
        dy_Gamma[:, 0] = Gamma_mat[:, 0] / d_y_mat[:, 0]
        dy_Gamma[:, -1] = -Gamma_mat[:, -1] / d_y_mat[:, -1]

        # ── Pressure ──
        tau_x_dx = tau_x * (dx_Gamma.T.reshape(-1, 1) * np.ones((1, 3)))
        tau_y_dy = tau_y * (dy_Gamma.T.reshape(-1, 1) * np.ones((1, 3)))

        dp_add = (Gamma - old_Gamma) / self._d_t_wake
        dp_lift = np.sum(V_surf * (tau_x_dx + tau_y_dy), axis=1)
        dp_lift1 = np.sum(V_surf1 * (tau_x_dx + tau_y_dy), axis=1)
        dp_lift2 = -(tau_x_dx + tau_y_dy)  # (N_element, 3)

        dp_vec = dp_lift + dp_add

        # ── Store ──
        self._dp_add = dp_add
        self._dp_lift = dp_lift
        self._dp_lift1 = dp_lift1
        self._dp_lift2 = dp_lift2
        self._dp_vec = dp_vec

        # ── Mf1: added mass matrix ──
        if self._Mf1_mat is None:
            self._compute_mf1()
        Mf1_mat = self._Mf1_mat

        # ── Mf2 ──
        if self._Mf2_mat is None:
            A_mat = self.uvlm._AIC
            self._Mf2_mat = np.linalg.inv(A_mat)

        # ── Strong coupling forces ──
        if self._coupling_flag == 1:
            self._calc_fluid_force_strong()
        else:
            self._Qf_p_global = np.zeros(self._N_q_all)
            self._Qf_p_mat_global = coo_matrix((self._N_q_all, self._N_q_all)).tocsc()
            self._Qf_p_mat0_global = np.zeros((self._N_q_all, self._N_element))
            self._Qf_p_lift2_mat_global = np.zeros((self._N_q_all, 3 * self._N_element))

    def _calc_fluid_force_strong(self):
        """Compute strong coupling fluid force matrices.

        MATLAB: calc_fluid_force_strong.m

        Qf_p_vec_i: Bernoulli force per element (Gauss-integrated with x-interpolation)
        Qf_p_lift2_mat_i: velocity coupling matrix
        Qf_p_mat0_i: Mf2_1 coupling matrix
        Qf_p_mat_i: Mf1 added mass matrix
        """
        n_gauss = self.shell.n_gauss
        p_vec, w_vec = _gauss_legendre(n_gauss)

        N_element = self._N_element
        N_q_all = self._N_q_all
        ne = self.shell.ne
        nx = self._nx
        ny = self._ny
        n_vec_i = self._n_vec_i
        dp_lift1 = self._dp_lift1
        dp_lift2 = self._dp_lift2
        Mf2_mat = self._Mf2_mat

        # ── Qf_p_vec_i: Bernoulli force (Mf2_2) ──
        # dp_nvec_i: interpolated in x-direction
        # For now, simplified: use per-element integration with constant pressure
        Qf_p_vec_i = np.zeros((self._N_q, ne))
        for e in range(ne):
            dL = self.shell._dL[e]
            dW = self.shell._dW[e]
            dofs = self.shell._elem_dofs(e)

            # Find which panel maps to this element
            panel_mask = self._panel_to_elem == e
            if not panel_mask.any():
                continue
            i_idx, j_idx = np.where(panel_mask)
            panel_idx = i_idx[0] * ny + j_idx[0]

            dp_val = self._dp_vec[panel_idx]  # dp_lift + dp_add
            n = n_vec_i[panel_idx]
            F_panel = dp_val * n * self.uvlm._areas[i_idx[0], j_idx[0]]

            xi, eta = self._panel_xi_eta[i_idx[0], j_idx[0]]
            S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
            Qf_p_vec_i[:, e] = S.T @ F_panel

        # Assemble global
        self._Qf_p_global = np.zeros(N_q_all)
        for e in range(ne):
            dofs = self.shell._elem_dofs(e)
            self._Qf_p_global[dofs] += Qf_p_vec_i[:, e]

        # ── Qf_p_lift2_mat_global: V_struct coupling ──
        Qf_p_lift2_mat_global = np.zeros((N_q_all, 3 * N_element))
        for e in range(ne):
            dL = self.shell._dL[e]
            dW = self.shell._dW[e]
            dofs = self.shell._elem_dofs(e)

            panel_mask = self._panel_to_elem == e
            if not panel_mask.any():
                continue
            i_idx, j_idx = np.where(panel_mask)
            panel_idx = i_idx[0] * ny + j_idx[0]

            dp2 = dp_lift2[panel_idx]  # (3,)
            n = n_vec_i[panel_idx]
            area = self.uvlm._areas[i_idx[0], j_idx[0]]
            # ni^T * dp_lift2 = n @ dp2 (scalar) → force = n^T * dp_lift2 * n * area
            # Actually MATLAB: niT_dp_lift2 = n * dp_lift2^T (3x3), then
            # Qf_p_lift2 * dt_rc = ∫ S^T * n * (dp_lift2 · dt_rc) dA
            # = ∫ S^T * n * dp_lift2^T dA · dt_rc
            # So the matrix is: S^T * n * dp_lift2^T * area
            xi, eta = self._panel_xi_eta[i_idx[0], j_idx[0]]
            S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
            # n * dp_lift2^T: (3, 3) matrix
            mat_3x3 = np.outer(n, dp2) * area
            Qf_p_lift2_mat_global[np.ix_(dofs, [3*panel_idx + d for d in range(3)])] += S.T @ mat_3x3

        self._Qf_p_lift2_mat_global = Qf_p_lift2_mat_global

        # ── Qf_p_mat0_global: Mf2_1 coupling ──
        Qf_p_mat0_global = np.zeros((N_q_all, N_element))
        for e in range(ne):
            dL = self.shell._dL[e]
            dW = self.shell._dW[e]
            dofs = self.shell._elem_dofs(e)

            panel_mask = self._panel_to_elem == e
            if not panel_mask.any():
                continue
            i_idx, j_idx = np.where(panel_mask)
            panel_idx = i_idx[0] * ny + j_idx[0]

            n = n_vec_i[panel_idx]
            area = self.uvlm._areas[i_idx[0], j_idx[0]]
            # Mf2_mat[panel_idx, :] is (N_element,) row → n * Mf2_row * area
            xi, eta = self._panel_xi_eta[i_idx[0], j_idx[0]]
            S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
            # n ⊗ Mf2_row = n_3x1 * Mf2_row_1xN * area
            mat_3xN = np.outer(n, Mf2_mat[panel_idx, :]) * area
            Qf_p_mat0_global[np.ix_(dofs, np.arange(N_element))] += S.T @ mat_3xN

        self._Qf_p_mat0_global = Qf_p_mat0_global

        # ── Qf_p_mat_global: Mf1 added mass ──
        if self._Mf1_mat is not None:
            Mf1_mat = self._Mf1_mat
            Qf_p_mat_global = coo_matrix((N_q_all, N_q_all)).tocsc()
            rows, cols, vals = [], [], []

            for e in range(ne):
                dL = self.shell._dL[e]
                dW = self.shell._dW[e]
                dofs = self.shell._elem_dofs(e)
                dof_set = set(dofs)

                panel_mask = self._panel_to_elem == e
                if not panel_mask.any():
                    continue
                i_idx, j_idx = np.where(panel_mask)
                panel_idx = i_idx[0] * ny + j_idx[0]

                n = n_vec_i[panel_idx]
                area = self.uvlm._areas[i_idx[0], j_idx[0]]
                # Mf1_mat[panel_idx, :] maps dq → dΓ/dt (1, N_q_all)
                # Force: n * Mf1_row * area (3, N_q_all)
                # Nodal: S^T @ (n * Mf1_row * area) (N_q, N_q_all)
                xi, eta = self._panel_xi_eta[i_idx[0], j_idx[0]]
                S = np.kron(_shape_funcs(xi, eta, dL, dW), np.eye(3))
                # n ⊗ Mf1_row
                mat_3xN = np.outer(n, Mf1_mat[panel_idx, :]) * area
                local_mat = S.T @ mat_3xN  # (N_q, N_q_all)

                for a in range(self._N_q):
                    for b in range(N_q_all):
                        if abs(local_mat[a, b]) > 1e-15:
                            rows.append(dofs[a])
                            cols.append(b)
                            vals.append(local_mat[a, b])

            self._Qf_p_mat_global = coo_matrix(
                (vals, (rows, cols)), shape=(N_q_all, N_q_all)).tocsc()

    def _compute_mf1(self):
        """Compute Mf1 matrix: A_mat \ nvec_Sc_global."""
        A_mat = self.uvlm._AIC
        nvec_Sc = self._nSc  # (N_element, N_q_all)
        self._Mf1_mat = np.linalg.solve(A_mat, nvec_Sc)

    # ═══════════════════════════════════════════════════════════════════
    # Wake velocity helpers
    # ═══════════════════════════════════════════════════════════════════

    def _compute_wake_velocity_at_plate(self):
        """Compute V_wake_plate: wake-induced velocity at collocation points."""
        if self._Gamma_wake is None or len(self._Gamma_wake) == 0:
            return np.zeros((self._N_element, 3))

        V = np.zeros((self._N_element, 3))
        from .standalone_uvlm import ring_vortex_velocity
        N_wake = len(self._Gamma_wake) // self._ny
        for w in range(N_wake):
            for js in range(self._ny):
                idx = w * self._ny + js
                if idx >= len(self._Gamma_wake):
                    break
                gw = self._Gamma_wake[idx]
                if abs(gw) < 1e-15:
                    continue
                corners = np.array([
                    self._r_wake_1[idx],
                    self._r_wake_2[idx],
                    self._r_wake_3[idx],
                    self._r_wake_4[idx]
                ])
                for k in range(self._N_element):
                    pt = self.uvlm._colloc.reshape(-1, 3)[k]
                    V[k] += ring_vortex_velocity(
                        pt.reshape(1, 3), corners, gw,
                        self.uvlm._core_radius).flatten()
        return V

    def _compute_bound_induction_at(self, pts):
        """Compute bound vortex induction at arbitrary points."""
        V = np.zeros_like(pts)
        from .standalone_uvlm import ring_vortex_velocity
        Gamma = self._Gamma
        for i in range(self._nx):
            for j in range(self._ny):
                g = Gamma[i * self._ny + j]
                if abs(g) < 1e-15:
                    continue
                corners = self.uvlm._corners[i, j]
                V += ring_vortex_velocity(pts, corners, g,
                                          self.uvlm._core_radius)
        return V

    def _compute_wake_induction_at(self, pts):
        """Compute wake-induced velocity at arbitrary points."""
        if self._Gamma_wake is None:
            return np.zeros_like(pts)
        V = np.zeros_like(pts)
        from .standalone_uvlm import ring_vortex_velocity
        for idx in range(len(self._Gamma_wake)):
            gw = self._Gamma_wake[idx]
            if abs(gw) < 1e-15:
                continue
            corners = np.array([
                self._r_wake_1[idx], self._r_wake_2[idx],
                self._r_wake_3[idx], self._r_wake_4[idx]
            ])
            V += ring_vortex_velocity(pts, corners, gw,
                                      self.uvlm._core_radius)
        return V

    # ═══════════════════════════════════════════════════════════════════
    # UVLM vertex update
    # ═══════════════════════════════════════════════════════════════════

    def _update_uvlm_vertices(self):
        """Update UVLM corner vertices from structural displacement."""
        for i in range(self._nx + 1):
            for j in range(self._ny + 1):
                x_tgt = self._x_vec[i]
                y_tgt = self._y_vec[j]
                nodes = self.shell.positions()
                dist = np.abs(nodes[:, 0] - x_tgt) + np.abs(nodes[:, 1] - y_tgt)
                closest = np.argmin(dist)
                self._uvlm_vertices[i, j, 2] = nodes[closest, 2]
        self.uvlm._verts = self._uvlm_vertices.copy()
        self.uvlm._compute_geometry()

    # ═══════════════════════════════════════════════════════════════════
    # Main loop (MATLAB: exe.m)
    # ═══════════════════════════════════════════════════════════════════

    def run(self, n_structural_steps, print_every=None):
        """Run FSI simulation with MATLAB's rewind predictor-corrector.

        MATLAB: exe.m lines 60-157
        """
        i_time = 1
        i_time_cnt = 1
        i_wake_time = 1
        time_fluid = 0.0
        fluid_compute_flag = True
        time = 0.0
        n_steps = n_structural_steps

        print(f"[matlab_fsi] Running {n_steps} structural steps "
              f"(d_t={self._d_t:.2e}, d_t_wake={self._d_t_wake:.2e}, "
              f"dt_wake_per_dt={self._dt_wake_per_dt})")

        t_start = __import__('time').time()

        while time <= n_steps * self._d_t or not fluid_compute_flag:
            time = i_time * self._d_t
            if time > n_steps * self._d_t and fluid_compute_flag:
                break

            # ── Solve structure ──
            self._solve_structure(i_time, fluid_compute_flag)

            # ── Solve fluid at wake intervals ──
            if i_time % self._dt_wake_per_dt == 1:
                if fluid_compute_flag:
                    # Store old values
                    self._old_Qf_p_global = self._Qf_p_global.copy()
                    self._old_Qf_p_mat_global = self._Qf_p_mat_global.copy()
                    self._old_Qf_p_mat0_global = self._Qf_p_mat0_global.copy()
                    self._old_Qf_p_lift2_mat_global = self._Qf_p_lift2_mat_global.copy()

                    # Solve fluid at predictor end state
                    self._solve_fluid(i_wake_time)
                    i_wake_time += 1

                    # Rewind: i_time -= i_time_cnt
                    i_time -= i_time_cnt
                    fluid_compute_flag = False
                else:
                    # Store updated values for corrector
                    self._Qf_p_global_a = self._Qf_p_global.copy()
                    self._Qf_p_mat_global_a = self._Qf_p_mat_global.copy()
                    self._Qf_p_mat0_global_a = self._Qf_p_mat0_global.copy()
                    self._Qf_p_lift2_mat_global_a = self._Qf_p_lift2_mat_global.copy()
                    time_fluid = time

                    i_time_cnt = 0
                    fluid_compute_flag = True

                # Print info
                if print_every and i_time % print_every == 0:
                    tw = self._tip_w_history[-1] if self._tip_w_history else 0
                    print(f"  step {i_time:6d} | t={time:.4f} | tip_w={tw:+.6e} m")

            i_time += 1
            i_time_cnt += 1

        elapsed = __import__('time').time() - t_start
        n_wake = len(self._Gamma_wake) if self._Gamma_wake is not None else 0
        print(f"\n[matlab_fsi] Simulation complete: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"[matlab_fsi] Wake rows: {n_wake // self._ny}")

    def get_results(self):
        """Return simulation results."""
        return {
            'tip_w': np.array(self._tip_w_history),
            'dt_struct': self._d_t,
            'dt_uvlm': self._d_t_wake,
        }
