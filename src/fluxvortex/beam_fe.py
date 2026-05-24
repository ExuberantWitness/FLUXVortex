"""
Euler-Bernoulli Beam FE with coupled bending-torsion for aeroelastic simulation.

Each node has 3 DOF: w (heave), dw/dy (bending slope), theta (twist).
Hermite cubic shape functions for bending, linear for torsion.
Newmark-beta time integration (average acceleration, unconditionally stable).

References:
  - Hodges & Pierce, "Introduction to Structural Dynamics and Aeroelasticity"
  - Bisplinghoff, Ashley & Halfman, "Aeroelasticity"
  - Goland & Luke (1948), "The Flutter of a Uniform Wing"
"""
import numpy as np


class BeamFE:
    """
    1D Euler-Bernoulli beam finite element model with coupled bending-torsion.

    Designed for cantilevered wing aeroelasticity: clamped at root (y=0),
    free at tip (y=L). The elastic axis is the beam reference line.

    DOF ordering per node: [w, dw/dy, theta]
    Global DOF vector: [w0, (dw/dy)0, theta0, w1, (dw/dy)1, theta1, ...]
    """

    def __init__(self, length, n_elements, EI, GJ, m_per_length, Ip,
                 x_ea_cg=0.0, structural_damping=0.0):
        """
        Args:
            length: Beam length (semi-span) [m]
            n_elements: Number of finite elements
            EI: Bending stiffness [N·m²]
            GJ: Torsional stiffness [N·m²]
            m_per_length: Mass per unit length [kg/m]
            Ip: Polar moment of inertia per unit length about EA [kg·m]
            x_ea_cg: Distance from elastic axis to CG (positive = CG aft of EA) [m]
            structural_damping: Rayleigh damping ratio
        """
        self.L = length
        self.nelem = n_elements
        self.nnodes = n_elements + 1
        self.n_dof_per_node = 3
        self.ndof = self.nnodes * self.n_dof_per_node

        # Material properties (can be arrays of length n_elements for non-uniform beam)
        self.EI = np.atleast_1d(np.float64(EI)) * np.ones(self.nelem)
        self.GJ = np.atleast_1d(np.float64(GJ)) * np.ones(self.nelem)
        self.m = np.atleast_1d(np.float64(m_per_length)) * np.ones(self.nelem)
        self.Ip = np.atleast_1d(np.float64(Ip)) * np.ones(self.nelem)
        self.x_ea_cg = np.atleast_1d(np.float64(x_ea_cg)) * np.ones(self.nelem)
        self.zeta = structural_damping

        # Element length
        self.Le = length / n_elements
        # Node positions
        self.y_nodes = np.linspace(0, length, self.nnodes)

        # Assemble global matrices
        self.K = np.zeros((self.ndof, self.ndof))
        self.M = np.zeros((self.ndof, self.ndof))
        self._assemble()

        # Rayleigh damping: C = beta*K (stiffness-proportional)
        # beta = 2*zeta/omega_1
        self.C = np.zeros((self.ndof, self.ndof))
        if self.zeta > 0:
            # Use generalized eigenvalue problem on BC-reduced matrices
            # to get correct omega_1 (M^-1K is not symmetric, eigvalsh fails)
            from scipy.linalg import eigh as _eigh
            K_r, M_r, _, _ = self.apply_bc(self.K, self.M)
            eigvals, _ = _eigh(K_r, M_r)
            omega1 = np.sqrt(max(eigvals[0], 1e-6))
            self.C = 2 * self.zeta / omega1 * self.K

        # State vectors
        self.d = np.zeros(self.ndof)   # displacement
        self.v = np.zeros(self.ndof)   # velocity
        self.a = np.zeros(self.ndof)   # acceleration

        # Newmark parameters (average acceleration)
        self.beta_nm = 0.25
        self.gamma_nm = 0.5

    def _element_stiffness_bending(self, EI_e, Le):
        """Hermite cubic element stiffness matrix for bending (4x4)."""
        L = Le
        return EI_e / L**3 * np.array([
            [12,    6*L,   -12,    6*L   ],
            [6*L,   4*L**2, -6*L,  2*L**2],
            [-12,  -6*L,    12,   -6*L   ],
            [6*L,   2*L**2, -6*L,  4*L**2],
        ])

    def _element_mass_bending(self, m_e, Le):
        """Consistent mass matrix for bending (4x4)."""
        L = Le
        return m_e * L / 420 * np.array([
            [156,    22*L,    54,    -13*L  ],
            [22*L,   4*L**2,  13*L,  -3*L**2],
            [54,     13*L,    156,   -22*L  ],
            [-13*L, -3*L**2, -22*L,   4*L**2],
        ])

    def _element_stiffness_torsion(self, GJ_e, Le):
        """Linear element stiffness matrix for torsion (2x2)."""
        return GJ_e / Le * np.array([
            [1, -1],
            [-1, 1],
        ])

    def _element_mass_torsion(self, Ip_e, Le):
        """Consistent mass matrix for torsion (2x2)."""
        return Ip_e * Le / 6 * np.array([
            [2, 1],
            [1, 2],
        ])

    def _element_coupling_mass(self, x_ea_cg_e, m_e, Le):
        """Bending-torsion coupling mass matrix from CG offset.

        Mc[i_b, j_θ] = m * x_α * ∫₀ᴸ N_bending_i · N_torsion_j dy

        Hermite cubic: N1=1-3ξ²+2ξ³, N2=L(ξ-2ξ²+ξ³),
                       N3=3ξ²-2ξ³,    N4=L(-ξ²+ξ³)
        Linear torsion: P1=1-ξ, P2=ξ

        Integrand max degree 4, so 3-point Gauss quadrature is exact.
        """
        return self._coupling_numerical(x_ea_cg_e, m_e, Le)

    @staticmethod
    def _coupling_numerical(xc, m, L):
        """Compute coupling mass matrix via 3-point Gauss quadrature on [0,1]."""
        # 3-point Gauss quadrature points and weights on [0,1]
        gpts = np.array([0.5 - 0.5 * np.sqrt(3.0/5.0),
                         0.5,
                         0.5 + 0.5 * np.sqrt(3.0/5.0)])
        gwts = np.array([5.0/18.0, 8.0/18.0, 5.0/18.0])

        # Shape functions evaluated at quadrature points
        xi = gpts
        # Hermite cubic bending
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L * (xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L * (-xi**2 + xi**3)
        # Linear torsion
        P1 = 1 - xi
        P2 = xi

        N_bend = np.array([N1, N2, N3, N4])  # (4, 3)
        N_tors = np.array([P1, P2])           # (2, 3)

        M_couple = np.zeros((6, 6))
        bend_idx = [0, 1, 3, 4]
        tors_idx = [2, 5]

        for i, bi in enumerate(bend_idx):
            for j, tj in enumerate(tors_idx):
                integral = np.sum(gwts * N_bend[i] * N_tors[j]) * L
                val = m * xc * integral
                M_couple[bi, tj] = val
                M_couple[tj, bi] = val  # symmetric

        return M_couple

    def _assemble(self):
        """Assemble global stiffness and mass matrices."""
        Le = self.Le
        for e in range(self.nelem):
            EI_e = self.EI[e]
            GJ_e = self.GJ[e]
            m_e = self.m[e]
            Ip_e = self.Ip[e]
            xc_e = self.x_ea_cg[e]

            # Bending (Hermite cubic, 4x4)
            Kb = self._element_stiffness_bending(EI_e, Le)
            Mb = self._element_mass_bending(m_e, Le)

            # Torsion (linear, 2x2)
            Kt = self._element_stiffness_torsion(GJ_e, Le)
            Mt = self._element_mass_torsion(Ip_e, Le)

            # Coupling (6x6)
            Mc = self._element_coupling_mass(xc_e, m_e, Le)

            # Combine into 6x6 element matrix
            Ke = np.zeros((6, 6))
            Me = np.zeros((6, 6))
            # DOF order: [w1, (dw/dy)1, theta1, w2, (dw/dy)2, theta2]
            # Bending: DOF 0,1,3,4
            bend_idx = [0, 1, 3, 4]
            for i, gi in enumerate(bend_idx):
                for j, gj in enumerate(bend_idx):
                    Ke[gi, gj] += Kb[i, j]
                    Me[gi, gj] += Mb[i, j]
            # Torsion: DOF 2,5
            tors_idx = [2, 5]
            for i, gi in enumerate(tors_idx):
                for j, gj in enumerate(tors_idx):
                    Ke[gi, gj] += Kt[i, j]
                    Me[gi, gj] += Mt[i, j]
            # Coupling
            Me += Mc

            # Assemble into global
            dofs = [e*3, e*3+1, e*3+2, (e+1)*3, (e+1)*3+1, (e+1)*3+2]
            for i in range(6):
                for j in range(6):
                    self.K[dofs[i], dofs[j]] += Ke[i, j]
                    self.M[dofs[i], dofs[j]] += Me[i, j]

    def apply_bc(self, K, M, C=None):
        """Apply cantilever BC: fix all DOFs at root node (y=0).

        Returns reduced matrices with root DOFs removed.
        """
        # Root node DOFs: indices 0, 1, 2
        fixed = [0, 1, 2]
        free = [i for i in range(self.ndof) if i not in fixed]

        K_r = K[np.ix_(free, free)]
        M_r = M[np.ix_(free, free)]
        C_r = None
        if C is not None:
            C_r = C[np.ix_(free, free)]
        return K_r, M_r, C_r, free

    def step(self, F_full, dt):
        """Advance one timestep using Newmark-beta.

        Args:
            F_full: Full DOF force vector (ndof,). Gravity and aerodynamic forces.
            dt: Timestep size [s]
        """
        K_r, M_r, C_r, free = self.apply_bc(self.K, self.M, self.C)
        F_r = F_full[free]

        d_r = self.d[free]
        v_r = self.v[free]
        a_r = self.a[free]

        beta = self.beta_nm
        gamma = self.gamma_nm

        # Effective stiffness
        K_eff = K_r + gamma / (beta * dt) * C_r + 1 / (beta * dt**2) * M_r

        # Effective force
        F_eff = F_r + M_r @ (
            1 / (beta * dt**2) * d_r
            + 1 / (beta * dt) * v_r
            + (1 / (2 * beta) - 1) * a_r
        ) + C_r @ (
            gamma / (beta * dt) * d_r
            + (gamma / beta - 1) * v_r
            + dt * (gamma / (2 * beta) - 1) * a_r
        )

        # Solve
        d_new_r = np.linalg.solve(K_eff, F_eff)

        # Update acceleration and velocity
        a_new_r = (1 / (beta * dt**2) * (d_new_r - d_r)
                   - 1 / (beta * dt) * v_r
                   - (1 / (2 * beta) - 1) * a_r)
        v_new_r = v_r + dt * ((1 - gamma) * a_r + gamma * a_new_r)

        # Map back to full DOF
        self.d[:] = 0.0
        self.v[:] = 0.0
        self.a[:] = 0.0
        self.d[free] = d_new_r
        self.v[free] = v_new_r
        self.a[free] = a_new_r

    def get_nodal_displacements(self):
        """Return (w, theta) at each node.

        Returns:
            w: (nnodes,) heave displacement [m]
            theta: (nnodes,) twist angle [rad]
        """
        w = self.d[0::3]
        theta = self.d[2::3]
        return w, theta

    def get_nodal_positions(self):
        """Return node positions along the span."""
        return self.y_nodes.copy()

    def compute_natural_frequencies(self, n_modes=None):
        """Compute natural frequencies of the cantilevered beam.

        Returns:
            frequencies: (n_modes,) in Hz
            mode_shapes: (ndof_free, n_modes) mode shape matrix
        """
        from scipy.linalg import eigh as sp_eigh

        K_r, M_r, _, _ = self.apply_bc(self.K, self.M)

        eigenvalues, eigenvectors = sp_eigh(K_r, M_r)
        eigenvalues = np.maximum(eigenvalues, 0)
        frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

        if n_modes is not None:
            frequencies = frequencies[:n_modes]
            eigenvectors = eigenvectors[:, :n_modes]

        return frequencies, eigenvectors

    @staticmethod
    def distribute_force_to_nodes(y_nodes, y_forces, f_forces, m_forces=None):
        """Distribute forces from arbitrary spanwise locations to beam nodes.

        Uses linear interpolation for force distribution.

        Args:
            y_nodes: (N,) beam node positions
            y_forces: (M,) force application points
            f_forces: (M,) forces at those points (e.g., lift per unit span)
            m_forces: (M,) moments at those points (e.g., pitching moment)

        Returns:
            F_nodal: (3*N,) force vector [w-forces, dw/dy-moments, theta-torques]
        """
        N = len(y_nodes)
        F = np.zeros(3 * N)

        for k in range(len(y_forces)):
            y = y_forces[k]
            # Find bracketing nodes
            idx = np.searchsorted(y_nodes, y, side='right') - 1
            idx = max(0, min(idx, N - 2))
            Le = y_nodes[idx + 1] - y_nodes[idx]
            xi = (y - y_nodes[idx]) / Le
            xi = max(0, min(xi, 1))

            # Distribute force to nodes (linear interpolation)
            w_force = f_forces[k]
            F[3 * idx] += w_force * (1 - xi)
            F[3 * (idx + 1)] += w_force * xi

            if m_forces is not None:
                F[3 * idx + 2] += m_forces[k] * (1 - xi)
                F[3 * (idx + 1) + 2] += m_forces[k] * xi

        return F

    def reset(self):
        """Reset beam state to zero."""
        self.d[:] = 0.0
        self.v[:] = 0.0
        self.a[:] = 0.0
