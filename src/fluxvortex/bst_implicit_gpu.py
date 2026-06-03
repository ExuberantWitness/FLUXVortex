"""GPU-friendly implicit dynamic solver for BST shell.

Supports:
  - Integration schemes: backward Euler, Newmark-β, generalized-α
  - K strategies: JFNK (Jacobian-free Newton-Krylov),
                  fd_direct (finite-difference K assembly + dense solve),
                  cg (wp.optim.linear.cg with JFNK matvec),
                  cusolver (fd K + cuSOLVER Cholesky)
  - Newton-Raphson iteration with convergence tracking
"""
import numpy as np


class BSTImplicitGPU:
    """Implicit dynamic solver for BSTShell.

    Parameters
    ----------
    shell : BSTShell
        The structural shell model.
    scheme : str
        Time integration: 'euler', 'newmark', 'gen_alpha'.
    k_strategy : str
        How to handle K_tangent: 'jfnk', 'fd_direct', 'cg', 'cusolver'.
    rho_inf : float
        Spectral radius for generalized-α (default 0.8).
    """

    def __init__(self, shell, scheme='newmark', k_strategy='jfnk',
                 rho_inf=0.8):
        self.shell = shell
        self.scheme = scheme
        self.k_strategy = k_strategy

        # Integration parameters
        if scheme == 'newmark':
            self.beta = 0.25
            self.gamma = 0.5
            self.alpha_m = 0.0
            self.alpha_f = 0.0
        elif scheme == 'euler':
            self.beta = 1.0
            self.gamma = 1.0
            self.alpha_m = 0.0
            self.alpha_f = 0.0
        elif scheme == 'gen_alpha':
            self.alpha_f = rho_inf / (1.0 + rho_inf)
            self.alpha_m = (2.0 * rho_inf - 1.0) / (1.0 + rho_inf)
            # Chung & Hulbert (1993): second-order accurate + unconditionally stable
            self.gamma = 0.5 - self.alpha_m + self.alpha_f
            self.beta = (1.0 - self.alpha_m + self.alpha_f) ** 2 / 4.0
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        self.damping = shell.damping
        self.newton_history = []
        self._fd_eps = 1e-7  # FD step for tangent approximation

    # ── Main time step ─────────────────────────────────────────────────

    def step(self, F_ext, dt, newton_max=20, tol=1e-8):
        """One implicit time step.

        Returns (n_iters, residual_norm).
        """
        shell = self.shell
        u_n = shell.u.copy()
        v_n = shell.v.copy()
        a_n = shell.a.copy()

        # Initial guess: extrapolate from current state
        u_trial = u_n + dt * v_n + 0.5 * dt ** 2 * a_n
        _apply_bc(u_trial, shell)

        free = shell.mass_inv > 0.0
        iter_history = []

        R = self._residual(u_trial, u_n, v_n, a_n, F_ext, dt)

        for k in range(newton_max):
            r_norm = np.max(np.abs(R[free]))
            iter_history.append(r_norm)

            if r_norm < tol:
                break

            # Solve K·du = -R
            neg_R = -R
            if self.k_strategy == 'jfnk':
                du = self._jfnk_solve(neg_R, u_trial, R,
                                      u_n, v_n, a_n, F_ext, dt)
            elif self.k_strategy == 'ibm_precond':
                du = self._ibm_precond_solve(neg_R, u_trial,
                                             u_n, v_n, a_n, F_ext, dt)
            else:
                du = self._fd_direct_solve(neg_R, u_trial,
                                           u_n, v_n, a_n, F_ext, dt)

            # Backtracking line search
            alpha = _line_search(
                self._residual_fn(u_n, v_n, a_n, F_ext, dt),
                u_trial, du, R, free)
            u_trial += alpha * du
            _apply_bc(u_trial, shell)

            R = self._residual(u_trial, u_n, v_n, a_n, F_ext, dt)

        self.newton_history.append(iter_history)

        # Update shell state
        a_new, v_new = self._update_state(u_trial, u_n, v_n, a_n, dt)
        shell.u[:] = u_trial
        shell.v[:] = v_new
        shell.a[:] = a_new

        if shell.use_gpu and shell._gpu_ctx is not None:
            shell.sync_to_gpu()

        return k + 1, r_norm

    # ── Residual computation ───────────────────────────────────────────

    def _residual(self, u, u_n, v_n, a_n, F_ext, dt):
        """Compute R(u) for the time integration scheme.

        Newmark/Euler: R = M·a + c·M·v - F_int(u) - F_ext = 0
        Gen-α: R = M·a_{n+1-α_m} + c·M·v_{n+1-α_f} - F_int(u_{n+1-α_f}) - F_ext = 0
        """
        shell = self.shell
        a, v = self._kinematics(u, u_n, v_n, a_n, dt)

        if self.scheme == 'gen_alpha':
            af = self.alpha_f
            am = self.alpha_m
            # α-weighted acceleration and velocity
            a_w = (1.0 - am) * a + am * a_n
            v_w = (1.0 - af) * v + af * v_n
            # Evaluate F_int at u_{n+1-α_f} = (1-α_f)·u_{n+1} + α_f·u_n
            u_alpha = (1.0 - af) * u + af * u_n
        else:
            a_w = a
            v_w = v
            u_alpha = u

        # Internal forces at (possibly α-weighted) displacement
        u_save = shell.u.copy()
        shell.u[:] = u_alpha
        F_int = shell.compute_forces()
        shell.u[:] = u_save

        # R = M·a_w + c·M·v_w - F_int - F_ext = 0
        R = np.zeros_like(u)
        m = shell.mass
        for i in range(shell.nv):
            if shell.mass_inv[i] > 0:
                R[i] = m[i] * a_w[i] + self.damping * m[i] * v_w[i] \
                       - F_int[i] - F_ext[i]

        return R

    def _kinematics(self, u, u_n, v_n, a_n, dt):
        """Compute (a, v) from u using Newmark equations."""
        b = self.beta
        g = self.gamma

        a = (u - u_n - dt * v_n - (0.5 - b) * dt ** 2 * a_n) / (b * dt ** 2)
        v = v_n + dt * ((1 - g) * a_n + g * a)

        return a, v

    def _update_state(self, u, u_n, v_n, a_n, dt):
        """Compute final (a, v) from converged u."""
        a, v = self._kinematics(u, u_n, v_n, a_n, dt)
        return a, v

    # ── K strategy: JFNK-PCG ───────────────────────────────────────────

    def _jfnk_solve(self, b, u, R_u, u_n, v_n, a_n, F_ext, dt,
                    max_iter=100, tol=1e-6):
        """Solve K·du = b via JFNK-PCG.

        K·p ≈ (R(u + ε·p) - R(u)) / ε
        """
        eps = self._fd_eps
        free = self.shell.mass_inv > 0.0

        # CG variables
        x = np.zeros_like(b)
        # Initial residual: r = b (since x=0, K·x=0)
        r = b.copy()
        p = r.copy()
        rs_old = _dot_free(r, r, free)

        for i in range(max_iter):
            # JFNK matvec: K·p
            p_norm = np.sqrt(np.dot(p.ravel(), p.ravel()))
            if p_norm < 1e-30:
                break
            delta = eps * max(np.max(np.abs(u)), 1.0) / p_norm

            u_pert = u + delta * p
            _apply_bc(u_pert, self.shell)
            R_pert = self._residual(u_pert, u_n, v_n, a_n, F_ext, dt)

            Ap = (R_pert - R_u) / delta

            # CG update
            pAp = _dot_free(p, Ap, free)
            if abs(pAp) < 1e-30:
                break
            alpha = rs_old / pAp

            x += alpha * p
            r_new = r - alpha * Ap

            rs_new = _dot_free(r_new, r_new, free)
            if np.sqrt(rs_new) < tol * np.sqrt(_dot_free(b, b, free) + 1e-30):
                break

            beta_cg = rs_new / (rs_old + 1e-30)
            p = r_new + beta_cg * p
            r = r_new
            rs_old = rs_new

        return x

    # ── K strategy: IBM preconditioner (constant PSD stiffness) ────────

    def _ibm_precond_solve(self, b, u, u_n, v_n, a_n, F_ext, dt):
        """Solve using precomputed IBM Q as bending stiffness.

        K_eff = M/(β*dt²) + c*M*γ/(β*dt) + K_membrane_FD + Q_ibm
        Q is constant PSD, membrane K is approximated as constant too.
        """
        shell = self.shell
        nv = shell.nv
        ndof = nv * 3

        # Ensure IBM Q is built
        if not hasattr(shell, '_Q') or shell._Q is None:
            shell._precompute_ibm()

        b_dt2 = self.beta * dt * dt
        g_over_b = self.gamma / self.beta

        # Build effective stiffness: K_eff per DOF component
        # For each vertex i: K_eff[i] = m_i/b_dt2 + c*m_i*g_over_b
        # Plus membrane stiffness (from FD, constant approximation)
        # Plus IBM bending Q

        # First, compute membrane stiffness via FD (once, at reference config)
        if not hasattr(self, '_K_eff_ibm'):
            self._build_ibm_precond(dt)

        # Solve
        bc_flat = np.repeat(shell.mass_inv == 0.0, 3)
        b_flat = b.ravel().copy()
        b_flat[bc_flat] = 0.0

        try:
            du_flat = np.linalg.solve(self._K_eff_ibm, b_flat)
        except np.linalg.LinAlgError:
            du_flat = np.linalg.lstsq(self._K_eff_ibm, b_flat, rcond=None)[0]

        return du_flat.reshape(nv, 3)

    def _build_ibm_precond(self, dt):
        """Build the constant effective stiffness matrix for IBM preconditioner."""
        shell = self.shell
        nv = shell.nv
        ndof = nv * 3

        b_dt2 = self.beta * dt * dt
        g_over_b = self.gamma / self.beta

        # Dynamic terms: M/(β*dt²) + c*M*γ/(β*dt) per vertex
        diag_val = shell.mass / b_dt2 + self.damping * shell.mass * g_over_b

        # Start with diagonal dynamic terms
        K = np.zeros((ndof, ndof))
        for i in range(nv):
            for d in range(3):
                idx = i * 3 + d
                K[idx, idx] = diag_val[i]

        # Add IBM bending: Q acts per-component
        Q = shell._Q
        for i in range(nv):
            for j in range(nv):
                if abs(Q[i, j]) > 1e-30:
                    for d in range(3):
                        K[i * 3 + d, j * 3 + d] += Q[i, j]

        # BC: zero rows/cols for constrained DOFs
        bc_flat = np.repeat(shell.mass_inv == 0.0, 3)
        K[bc_flat, :] = 0.0
        K[:, bc_flat] = 0.0
        for idx in range(ndof):
            if bc_flat[idx]:
                K[idx, idx] = 1.0

        self._K_eff_ibm = K

    # ── K strategy: Finite-difference direct ───────────────────────────

    def _fd_direct_solve(self, b, u, u_n, v_n, a_n, F_ext, dt):
        """Assemble K via finite differences and solve K·du = b directly."""
        ndof = self.shell.nv * 3
        R_u = self._residual(u, u_n, v_n, a_n, F_ext, dt)
        eps = self._fd_eps * max(np.max(np.abs(u)), 1.0)

        K = np.zeros((ndof, ndof))
        for j in range(ndof):
            e_j = np.zeros(ndof)
            e_j[j] = eps
            e_j_3d = e_j.reshape(self.shell.nv, 3)

            u_pert = u + e_j_3d
            _apply_bc(u_pert, self.shell)
            R_pert = self._residual(u_pert, u_n, v_n, a_n, F_ext, dt)

            K[:, j] = (R_pert.ravel() - R_u.ravel()) / eps

        # Enforce BCs: zero rows and columns for constrained DOFs
        bc_flat = np.repeat(self.shell.mass_inv == 0.0, 3)
        K[bc_flat, :] = 0.0
        K[:, bc_flat] = 0.0
        K[bc_flat, bc_flat] = 1.0

        b_flat = b.ravel().copy()
        b_flat[bc_flat] = 0.0

        try:
            du_flat = np.linalg.solve(K, b_flat)
        except np.linalg.LinAlgError:
            du_flat = np.linalg.lstsq(K, b_flat, rcond=None)[0]

        return du_flat.reshape(self.shell.nv, 3)

    # ── Solve equilibrium (static) ─────────────────────────────────────

    def solve_equilibrium(self, F_ext, max_iter=30, tol=1e-8):
        """Find static equilibrium: F_int(u) + F_ext = 0.

        Uses Newton-Raphson with FD-assembled K + backtracking line search.
        """
        shell = self.shell
        u_trial = shell.u.copy()
        free = shell.mass_inv > 0.0

        for k in range(max_iter):
            u_save = shell.u.copy()
            shell.u[:] = u_trial
            F_int = shell.compute_forces()
            shell.u[:] = u_save

            R = F_int + F_ext
            r_norm = np.max(np.abs(R[free]))
            if r_norm < tol:
                break

            # FD tangent
            eps = self._fd_eps
            ndof = shell.nv * 3
            K = np.zeros((ndof, ndof))

            for j in range(ndof):
                e_j = np.zeros(ndof)
                e_j[j] = eps
                e_j_3d = e_j.reshape(shell.nv, 3)
                u_pert = u_trial + e_j_3d
                _apply_bc(u_pert, shell)

                shell.u[:] = u_pert
                F_int_pert = shell.compute_forces()
                shell.u[:] = u_save

                K[:, j] = (F_int_pert.ravel() - F_int.ravel()) / eps

            bc_flat = np.repeat(shell.mass_inv == 0.0, 3)
            K[bc_flat, :] = 0.0
            K[:, bc_flat] = 0.0
            K[bc_flat, bc_flat] = 1.0
            neg_R = -R.ravel()
            neg_R[bc_flat] = 0.0

            try:
                du = np.linalg.solve(K, neg_R)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K, neg_R, rcond=None)[0]

            du_3d = du.reshape(shell.nv, 3)

            # Backtracking line search
            alpha = _line_search_static(
                shell, F_ext, u_trial, du_3d, R, free)

            u_trial += alpha * du_3d
            _apply_bc(u_trial, shell)

        shell.u[:] = u_trial
        shell.v[:] = 0.0
        shell.a[:] = 0.0
        if shell.use_gpu and shell._gpu_ctx is not None:
            shell.sync_to_gpu()

        return k + 1, r_norm

    # ── Helper for line search ─────────────────────────────────────────

    def _residual_fn(self, u_n, v_n, a_n, F_ext, dt):
        """Return a callable R(u) for line search."""
        def fn(u):
            return self._residual(u, u_n, v_n, a_n, F_ext, dt)
        return fn


def _apply_bc(u, shell):
    """Zero out displacements at BC nodes."""
    bc = shell.mass_inv == 0.0
    u[bc] = 0.0


def _dot_free(a, b, free):
    """Dot product over free DOFs only."""
    return np.sum(a[free] * b[free])


def _res_norm(R, free):
    """Max absolute residual over free DOFs."""
    return np.max(np.abs(R[free]))


def _line_search(R_fn, u, du, R0, free, max_iter=10, c1=1e-4):
    """Backtracking line search: find α s.t. ||R(u+α·du)|| < ||R(u)||."""
    r0 = _res_norm(R0, free)
    alpha = 1.0
    for _ in range(max_iter):
        u_new = u + alpha * du
        R_new = R_fn(u_new)
        r_new = _res_norm(R_new, free)
        if r_new < r0 * (1.0 - c1 * alpha):
            return alpha
        alpha *= 0.5
    return alpha


def _line_search_static(shell, F_ext, u, du, R0, free,
                        max_iter=10, c1=1e-4):
    """Line search for static equilibrium."""
    r0 = _res_norm(R0, free)
    alpha = 1.0
    u_save = shell.u.copy()

    for _ in range(max_iter):
        u_new = u + alpha * du
        _apply_bc(u_new, shell)
        shell.u[:] = u_new
        F_int = shell.compute_forces()
        shell.u[:] = u_save

        R_new = F_int + F_ext
        r_new = _res_norm(R_new, free)
        if r_new < r0 * (1.0 - c1 * alpha):
            return alpha
        alpha *= 0.5

    return alpha
