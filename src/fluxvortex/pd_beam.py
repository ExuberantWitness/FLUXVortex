"""PD Micro-Beam Bond Model (Zheng 2022) — standalone 1D beam solver.

3 DOFs per node: w (heave), ψ (bending slope = dw/dy), θ (twist).
Standard Euler-Bernoulli beam element stiffness for bending,
linear element for torsion, coupled mass from CG offset.

Uses Velocity-Verlet integration: real velocity state variable,
symplectic, energy-conserving.

Designed as a drop-in replacement for BeamFE in AeroelasticSolver.
"""
import numpy as np


class PDBeam:
    """1D beam with pairwise micro-beam bond forces + Velocity-Verlet.

    Compatible with AeroelasticSolver interface:
      - y_nodes: node positions along span
      - step(F_beam, dt): advance one timestep
      - get_nodal_displacements(): returns (w, theta) arrays
      - nnodes: number of nodes
    """

    def __init__(self, length, n_elements, EI, GJ, m_per_length, Ip,
                 x_ea_cg=0.0, structural_damping=0.0):
        self.L = length
        self.nelem = n_elements
        self.nnodes = n_elements + 1
        self.Le = length / n_elements
        self.y_nodes = np.linspace(0, length, self.nnodes)

        self.EI = EI
        self.GJ = GJ
        self.damping = structural_damping

        # Per-element mass properties
        Le = self.Le
        self.m_node = m_per_length * Le          # mass per node (lumped)
        self.Ip_node = Ip * Le                   # torsion inertia per node
        self.J_node = self.m_node * Le**2 / 12   # slope inertia per node
        self.S_node = m_per_length * x_ea_cg * Le  # static moment per node

        # DOF layout: [w0, ψ0, θ0, w1, ψ1, θ1, ...]
        self.nd = 3  # DOFs per node
        n_dof = self.nd * self.nnodes
        self.n_dof = n_dof
        self.d = np.zeros(n_dof)  # displacement
        self.v = np.zeros(n_dof)  # velocity

        # Precompute per-node inverse mass matrix (3x3)
        m = self.m_node
        Ip = self.Ip_node
        J = self.J_node
        S = self.S_node
        det = m * Ip - S**2
        self.M_inv = np.array([
            [Ip / det, 0, -S / det],
            [0, 1.0 / J, 0],
            [-S / det, 0, m / det]
        ])

    def _compute_internal_forces(self):
        """Bending + torsion restoring forces from micro-beam bonds."""
        nd = self.nd
        Le = self.Le
        n = self.nnodes
        u = self.d
        F = np.zeros(nd * n)

        for e in range(n - 1):
            ii, jj = nd * e, nd * (e + 1)
            wi, si, ti = u[ii], u[ii+1], u[ii+2]
            wj, sj, tj = u[jj], u[jj+1], u[jj+2]

            # Bending: standard Hermite beam element stiffness
            cb = self.EI / Le**3
            Kb_u = cb * np.array([
                12*wi + 6*Le*si - 12*wj + 6*Le*sj,
                6*Le*wi + 4*Le**2*si - 6*Le*wj + 2*Le**2*sj,
                -12*wi - 6*Le*si + 12*wj - 6*Le*sj,
                6*Le*wi + 2*Le**2*si - 6*Le*wj + 4*Le**2*sj
            ])
            F[ii]   -= Kb_u[0]
            F[ii+1] -= Kb_u[1]
            F[jj]   -= Kb_u[2]
            F[jj+1] -= Kb_u[3]

            # Torsion: linear element
            ct = self.GJ / Le
            F[ii+2] -= ct * (ti - tj)
            F[jj+2] -= ct * (-ti + tj)

        return F

    def _accel(self, F_total):
        a = np.zeros_like(F_total)
        nd = self.nd
        for i in range(self.nnodes):
            k = nd * i
            a[k:k+3] = self.M_inv @ F_total[k:k+3]
        return a

    def step(self, F_beam, dt):
        """Velocity-Verlet timestep.

        Args:
            F_beam: (n_dof,) external force vector (from UVLM).
                     DOF layout: [w0, ψ0, θ0, w1, ψ1, θ1, ...]
            dt: timestep
        """
        nd = self.nd

        # Internal + damping forces
        F_int = self._compute_internal_forces()
        if self.damping > 0:
            for i in range(self.nnodes):
                k = nd * i
                F_int[k]   -= self.damping * self.m_node * self.v[k]
                F_int[k+1] -= self.damping * self.J_node * self.v[k+1]
                F_int[k+2] -= self.damping * self.Ip_node * self.v[k+2]

        # Step 1: acceleration
        a = self._accel(F_int + F_beam)

        # Step 2: position update
        self.d += self.v * dt + 0.5 * a * dt**2

        # Step 3: new acceleration
        F_int_new = self._compute_internal_forces()
        if self.damping > 0:
            for i in range(self.nnodes):
                k = nd * i
                F_int_new[k]   -= self.damping * self.m_node * self.v[k]
                F_int_new[k+1] -= self.damping * self.J_node * self.v[k+1]
                F_int_new[k+2] -= self.damping * self.Ip_node * self.v[k+2]
        a_new = self._accel(F_int_new + F_beam)

        # Step 4: velocity update
        self.v += 0.5 * (a + a_new) * dt

        # Step 5: clamped BC at root
        for k in range(3):
            self.d[k] = 0.0
            self.v[k] = 0.0

    def get_nodal_displacements(self):
        """Return (w, theta) per node — same interface as BeamFE."""
        nd = self.nd
        w = self.d[0::nd].copy()
        theta = self.d[2::nd].copy()
        return w, theta

    def reset(self):
        self.d[:] = 0.0
        self.v[:] = 0.0
