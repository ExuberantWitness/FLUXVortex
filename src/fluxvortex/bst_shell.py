"""BST (Basic Shell Triangle) rotation-free shell element.

3-node triangle, each node has only 3 displacement DOFs (no rotations).
Membrane: standard CST (Constant Strain Triangle).
Bending: dihedral angle model (Bridson 2002) on interior edges.

Uses explicit Velocity-Verlet integration with real velocity state variable.
GPU-friendly: per-element force computation -> scatter-add to nodes, no global matrix.

Parameters (E, nu, h) unify membrane/plate/beam through one constitutive model:
  - Membrane stiffness: E*h
  - Bending stiffness:  E*h^3 / 12(1-nu^2)
"""
import numpy as np


class BSTShell:
    """Rotation-free triangular shell: CST membrane + dihedral bending.

    Parameters
    ----------
    vertices : (N, 3) array
        Initial vertex positions. x=chordwise, y=spanwise, z=up.
    triangles : (T, 3) array
        Triangle connectivity (vertex indices).
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    h : float
        Thickness [m].
    rho : float
        Density [kg/m^3].
    structural_damping : float
        Rayleigh damping coefficient (velocity-proportional).
    """

    def __init__(self, vertices, triangles, E, nu, h, rho,
                 structural_damping=0.0):
        self.vertices0 = np.array(vertices, dtype=np.float64)
        self.triangles = np.array(triangles, dtype=np.int32)
        self.nv = len(self.vertices0)
        self.nt = len(self.triangles)

        # Material
        self.E = E
        self.nu = nu
        self.h = float(h)
        self.rho = rho
        self.damping = structural_damping

        # Flexural rigidity D = E*h^3 / (12*(1-nu^2))
        self.D = E * h**3 / (12.0 * (1.0 - nu * nu))

        # Plane stress constitutive matrix D (3x3) for membrane
        fac = E / (1.0 - nu * nu)
        self.D_mem = fac * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

        # Precompute per-triangle reference geometry (for membrane CST)
        self._precompute_ref_geometry()

        # Build interior edge list for dihedral bending
        self._build_interior_edges()

        # Lumped mass: triangle area * h * rho / 3 per vertex
        self.mass = np.zeros(self.nv)
        for t in range(self.nt):
            i0, i1, i2 = self.triangles[t]
            area = self._ref_area[t]
            m_tri = area * h * rho
            self.mass[i0] += m_tri / 3.0
            self.mass[i1] += m_tri / 3.0
            self.mass[i2] += m_tri / 3.0
        self.mass_inv = np.where(self.mass > 1e-30, 1.0 / self.mass, 0.0)

        # State: displacement and velocity (real state variables)
        self.u = np.zeros((self.nv, 3))
        self.v = np.zeros((self.nv, 3))
        self.a = np.zeros((self.nv, 3))

    # ------------------------------------------------------------------
    # Reference geometry (membrane CST)
    # ------------------------------------------------------------------
    def _precompute_ref_geometry(self):
        """Precompute reference area, shape function derivatives per triangle."""
        self._ref_area = np.zeros(self.nt)
        self._dNdx = np.zeros((self.nt, 3))
        self._dNdy = np.zeros((self.nt, 3))

        for t in range(self.nt):
            i0, i1, i2 = self.triangles[t]
            x0 = self.vertices0[i0]
            x1 = self.vertices0[i1]
            x2 = self.vertices0[i2]

            e1 = x1 - x0
            e2 = x2 - x0

            cross = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(cross)
            self._ref_area[t] = max(area, 1e-30)

            J = np.array([[e1[0], e2[0]],
                          [e1[1], e2[1]]])
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(detJ) < 1e-30:
                detJ = 1e-30
            Jinv = np.array([[J[1, 1], -J[0, 1]],
                             [-J[1, 0], J[0, 0]]]) / detJ

            # Physical derivatives via J^{-T}: use COLUMNS of Jinv
            self._dNdx[t] = np.array([
                -(Jinv[0, 0] + Jinv[1, 0]),
                Jinv[0, 0],
                Jinv[1, 0],
            ])
            self._dNdy[t] = np.array([
                -(Jinv[0, 1] + Jinv[1, 1]),
                Jinv[0, 1],
                Jinv[1, 1],
            ])

    # ------------------------------------------------------------------
    # Interior edges for dihedral bending
    # ------------------------------------------------------------------
    def _build_interior_edges(self):
        """Build list of interior edges (shared by 2 triangles) for bending.

        For each interior edge, store the 4 vertices:
          - ea, eb: edge endpoints
          - ec: opposite vertex in triangle 0
          - ed: opposite vertex in triangle 1

        Also compute reference dihedral angle and edge length.
        """
        edge_to_tris = {}
        tri_edge_opp = {}  # (tri, edge_idx) -> opposite vertex in that triangle

        for t in range(self.nt):
            i0, i1, i2 = self.triangles[t]
            edges = [(i0, i1, i2), (i1, i2, i0), (i0, i2, i1)]
            for va, vb, opp in edges:
                key = (min(va, vb), max(va, vb))
                edge_to_tris.setdefault(key, []).append(t)
                tri_edge_opp[(t, key)] = opp

        interior_edges = []
        for key, tris in edge_to_tris.items():
            if len(tris) == 2:
                t0, t1 = tris
                va, vb = key
                ec = tri_edge_opp[(t0, key)]
                ed = tri_edge_opp[(t1, key)]

                # Compute reference dihedral angle
                pa = self.vertices0[va]
                pb = self.vertices0[vb]
                pc = self.vertices0[ec]
                pd = self.vertices0[ed]

                e_vec = pb - pa
                L = np.linalg.norm(e_vec)
                if L < 1e-30:
                    continue

                # Face normals (consistent orientation: both using edge a->b)
                n0 = np.cross(pb - pa, pc - pa)  # tri 0: (a,b,c)
                n1 = np.cross(pd - pa, pb - pa)  # tri 1: (a,d,b) — reversed edge for consistency

                n0_len = np.linalg.norm(n0)
                n1_len = np.linalg.norm(n1)
                if n0_len < 1e-30 or n1_len < 1e-30:
                    continue

                n0_hat = n0 / n0_len
                n1_hat = n1 / n1_len

                cos_theta = np.clip(np.dot(n0_hat, n1_hat), -1.0, 1.0)
                sin_theta = np.dot(np.cross(n0_hat, n1_hat), e_vec / L)
                theta_ref = np.arctan2(sin_theta, cos_theta)

                # Triangle areas and heights from edge
                A0 = 0.5 * n0_len
                A1 = 0.5 * n1_len

                interior_edges.append({
                    'ea': va, 'eb': vb, 'ec': ec, 'ed': ed,
                    'L': L, 'theta_ref': theta_ref,
                    'A0': A0, 'A1': A1,
                })

        self._interior_edges = interior_edges
        self.n_interior_edges = len(interior_edges)

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------
    def compute_forces(self):
        """Compute total internal forces: membrane (CST) + bending (dihedral)."""
        F = np.zeros((self.nv, 3))
        self._compute_membrane_forces(F)
        self._compute_bending_forces(F)
        return F

    def _compute_membrane_forces(self, F):
        """CST membrane forces: constant strain per triangle."""
        D = self.D_mem
        h = self.h

        for t in range(self.nt):
            i0, i1, i2 = self.triangles[t]
            area = self._ref_area[t]
            dNdx = self._dNdx[t]
            dNdy = self._dNdy[t]

            u0, u1, u2 = self.u[i0], self.u[i1], self.u[i2]

            ux = np.array([u0[0], u1[0], u2[0]])
            uy = np.array([u0[1], u1[1], u2[1]])

            eps = np.array([
                dNdx @ ux,
                dNdy @ uy,
                dNdx @ uy + dNdy @ ux,
            ])

            sigma = D @ eps
            coeff = -area * h

            nodes = [i0, i1, i2]
            for k in range(3):
                fx = coeff * (dNdx[k] * sigma[0] + dNdy[k] * sigma[2])
                fy = coeff * (dNdx[k] * sigma[2] + dNdy[k] * sigma[1])
                F[nodes[k], 0] += fx
                F[nodes[k], 1] += fy

    def _compute_bending_forces(self, F):
        """Dihedral angle bending forces on interior edges.

        For each interior edge (ea, eb) with opposite vertices ec, ed:
          1. Compute current dihedral angle theta
          2. dtheta = theta - theta_ref
          3. Force = -D * L * dtheta * grad_theta
        """
        x = self.vertices0 + self.u  # current positions
        D = self.D

        for edge in self._interior_edges:
            ea = edge['ea']
            eb = edge['eb']
            ec = edge['ec']
            ed = edge['ed']
            L = edge['L']
            theta_ref = edge['theta_ref']
            A0 = edge['A0']
            A1 = edge['A1']

            pa, pb = x[ea], x[eb]
            pc, pd = x[ec], x[ed]

            e_vec = pb - pa
            L_cur = np.linalg.norm(e_vec)
            if L_cur < 1e-30:
                continue
            e_hat = e_vec / L_cur

            # Face normals (same orientation convention as reference)
            n0 = np.cross(pb - pa, pc - pa)
            n1 = np.cross(pd - pa, pb - pa)

            n0_len = np.linalg.norm(n0)
            n1_len = np.linalg.norm(n1)
            if n0_len < 1e-30 or n1_len < 1e-30:
                continue

            n0_hat = n0 / n0_len
            n1_hat = n1 / n1_len

            # Current dihedral angle
            cos_theta = np.clip(np.dot(n0_hat, n1_hat), -1.0, 1.0)
            sin_theta = np.dot(np.cross(n0_hat, n1_hat), e_hat)
            theta = np.arctan2(sin_theta, cos_theta)

            dtheta = theta - theta_ref

            if abs(dtheta) < 1e-15:
                continue

            # Gradients of theta w.r.t. opposite vertices
            # Note: negative sign because theta = angle between normals (0=flat),
            # not the exterior dihedral angle (pi=flat).
            grad_c = -(L_cur / (2.0 * A0)) * n0_hat
            grad_d = -(L_cur / (2.0 * A1)) * n1_hat

            # Gradients w.r.t. edge vertices (from translational invariance)
            grad_a = -(np.dot(pc - pa, e_vec) / (L_cur * L_cur)) * grad_c \
                     -(np.dot(pd - pa, e_vec) / (L_cur * L_cur)) * grad_d
            grad_b = (np.dot(pc - pb, e_vec) / (L_cur * L_cur)) * grad_c \
                     + (np.dot(pd - pb, e_vec) / (L_cur * L_cur)) * grad_d

            # Bending force: F = -D * L * dtheta * grad
            # Use reference L for consistent stiffness (or current L_cur)
            coeff = -D * L * dtheta

            F[ea] += coeff * grad_a
            F[eb] += coeff * grad_b
            F[ec] += coeff * grad_c
            F[ed] += coeff * grad_d

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def step(self, F_ext, dt):
        """Velocity-Verlet timestep with real velocity state."""
        F_int = self.compute_forces()

        F_damp = np.zeros_like(self.v)
        if self.damping > 0:
            F_damp = -self.damping * self.v * self.mass[:, None]

        F_total = F_int + F_ext + F_damp
        a_new = F_total * self.mass_inv[:, None]

        # Velocity-Verlet
        self.u += self.v * dt + 0.5 * a_new * dt * dt
        self.v += 0.5 * (self.a + a_new) * dt
        self.a = a_new

        # Apply boundary conditions
        mask = self.mass_inv > 0
        self.u[~mask] = 0.0
        self.v[~mask] = 0.0
        self.a[~mask] = 0.0

    def set_bc(self, node_indices):
        """Set clamped boundary conditions at given nodes."""
        for i in node_indices:
            self.mass_inv[i] = 0.0

    def get_nodal_positions(self):
        """Return current vertex positions."""
        return self.vertices0 + self.u

    def get_nodal_displacements(self):
        """Return displacement array (N, 3)."""
        return self.u.copy()

    def reset(self):
        """Reset displacement and velocity to zero."""
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.a[:] = 0.0
