"""
Unified 2D particle-mesh structural dynamics.

Triangle mesh with particles at vertices. Stiffness/mass distributions
are per-element parameters, enabling continuous rigid<->flexible transitions
without switching element types.

Two solvers available:
  - step_force(): Explicit force-based (Verlet). ke directly = physical spring
    stiffness. Independent of dt. Use for quantitative validation.
  - step(): XPBD constraint projection. For rigid constraints (ke >> 1/dt²).
    Better stability for extreme stiffness ratios.

Bending model: Laplacian-based (biharmonic) bending.
  Energy: E = 0.5 * bending_D * sum_i |L(x_i)|^2
  Force:  F = -bending_D * L(L(x))  (two Laplacian passes)
  Where L is the cotangent-weighted Laplacian operator.
  Parameter: bending_D = EI / w  (flexural rigidity per unit width).
  For a beam strip: bending_D = EI / w, total stiffness = bending_D * w = EI.
"""
import numpy as np


class ParticleMesh:
    """Unified 2D particle-mesh structural model with XPBD solver.

    Triangle mesh with particles at vertices. By adjusting per-edge spring_ke
    and the bending_D parameter, the same mesh can represent rigid plates,
    flexible membranes, rods, or any mixture — without switching element types.

    Parameters
    ----------
    vertices : (N, 3) array
        Initial particle positions.
    triangles : (T, 3) array
        Triangle connectivity (vertex indices).
    spring_ke : float or (E,) array
        Spring stiffness for all triangle edges.
    spring_kd : float or (E,) array
        Spring damping for all triangle edges.
    bending_D : float
        Bending rigidity per unit width (D = EI/w for a beam strip).
        The Laplacian bending model uses this as the stiffness parameter.
        For a beam: D = EI / strip_width.
    bending_damp : float
        Bending damping coefficient.
    mass : float or (N,) array
        Particle mass. Total mass distributed uniformly if scalar.
    density : float, optional
        If provided, compute mass from triangle area × density instead.
    gravity : (3,) array
        Gravitational acceleration. Default [0, 0, -9.81].
    """

    def __init__(self, vertices, triangles, spring_ke=1e4, spring_kd=1.0,
                 bending_D=0.0, bending_damp=0.0, mass=0.01, density=None,
                 gravity=None):
        self.pos = np.array(vertices, dtype=np.float64).copy()
        self.vel = np.zeros_like(self.pos)
        self.tri_indices = np.array(triangles, dtype=np.int32)

        self.n_particles = len(self.pos)
        self.n_triangles = len(self.tri_indices)

        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])
        self.gravity = np.array(gravity, dtype=np.float64)

        # Build edge topology from triangles
        self._build_edges()

        # Reference configuration
        self._compute_reference_config()

        # Spring parameters (for all edges — triangle boundary + internal)
        n_edges = len(self.spring_i)
        self.spring_ke = np.full(n_edges, spring_ke, dtype=np.float64) \
            if np.isscalar(spring_ke) else np.array(spring_ke, dtype=np.float64)
        self.spring_kd = np.full(n_edges, spring_kd, dtype=np.float64) \
            if np.isscalar(spring_kd) else np.array(spring_kd, dtype=np.float64)

        # Bending parameters (Laplacian-based)
        self.bending_D = float(bending_D)
        self.bending_damp = float(bending_damp)

        # Build cotangent-weighted Laplacian operator
        self._build_laplacian()

        # Mass
        if density is not None:
            self.particle_mass = self._compute_mass_from_density(density)
        elif np.isscalar(mass):
            self.particle_mass = np.full(self.n_particles, mass, dtype=np.float64)
        else:
            self.particle_mass = np.array(mass, dtype=np.float64)

        # Inverse mass (fixed particles will be set to 0)
        self.inv_mass = 1.0 / self.particle_mass

        # Fixed particles (wing root, etc.)
        self.fixed_mask = np.zeros(self.n_particles, dtype=bool)

        # Extra springs (actuators, structural reinforcements)
        self._extra_spring_i = []
        self._extra_spring_j = []
        self._extra_spring_ke = []
        self._extra_spring_kd = []
        self._extra_spring_rest = []
        self._extra_springs_integrated = False

        # History for animation / analysis
        self.pos_history = []

        # Store initial (reference) positions
        self._pos_ref = self.pos.copy()

    def _build_edges(self):
        """Extract all edges and identify internal (bending) edges."""
        edge_to_tris = {}
        for t_idx, (i, j, k) in enumerate(self.tri_indices):
            for a, b in [(i, j), (j, k), (k, i)]:
                key = (min(a, b), max(a, b))
                if key not in edge_to_tris:
                    edge_to_tris[key] = []
                edge_to_tris[key].append(t_idx)

        # All edges (for spring constraints)
        edges = list(edge_to_tris.keys())
        self.spring_i = np.array([e[0] for e in edges], dtype=np.int32)
        self.spring_j = np.array([e[1] for e in edges], dtype=np.int32)

        # Internal edges (shared by 2 triangles → bending edges)
        # For bending: 4 vertices (i_opp0, i_opp1, edge_a, edge_b)
        bend_i, bend_j, bend_k, bend_l = [], [], [], []
        for (a, b), tris in edge_to_tris.items():
            if len(tris) == 2:
                t0, t1 = tris
                # Find the opposite vertex in each triangle
                opp0 = [v for v in self.tri_indices[t0] if v != a and v != b][0]
                opp1 = [v for v in self.tri_indices[t1] if v != a and v != b][0]
                bend_i.append(opp0)
                bend_j.append(opp1)
                bend_k.append(a)
                bend_l.append(b)

        self.bend_i = np.array(bend_i, dtype=np.int32)
        self.bend_j = np.array(bend_j, dtype=np.int32)
        self.bend_k = np.array(bend_k, dtype=np.int32)
        self.bend_l = np.array(bend_l, dtype=np.int32)

    def _compute_reference_config(self):
        """Compute rest lengths, rest angles, rest areas."""
        # Spring rest lengths
        dx = self.pos[self.spring_j] - self.pos[self.spring_i]
        self.spring_rest = np.linalg.norm(dx, axis=1)

        # Bending rest angles (dihedral angle in reference config)
        if len(self.bend_i) > 0:
            x1 = self.pos[self.bend_i]
            x2 = self.pos[self.bend_j]
            x3 = self.pos[self.bend_k]
            x4 = self.pos[self.bend_l]

            n1 = np.cross(x3 - x1, x4 - x1)
            n2 = np.cross(x4 - x2, x3 - x2)
            e = x4 - x3

            n1_len = np.linalg.norm(n1, axis=1, keepdims=True)
            n2_len = np.linalg.norm(n2, axis=1, keepdims=True)
            e_len = np.linalg.norm(e, axis=1, keepdims=True)

            safe_n1 = np.where(n1_len > 1e-12, n1_len, 1.0)
            safe_n2 = np.where(n2_len > 1e-12, n2_len, 1.0)

            n1_hat = n1 / safe_n1
            n2_hat = n2 / safe_n2
            e_hat = e / np.maximum(e_len, 1e-12)

            cos_t = np.sum(n1_hat * n2_hat, axis=1)
            sin_t = np.sum(np.cross(n1_hat, n2_hat) * e_hat, axis=1)
            self.bend_rest_angle = np.arctan2(sin_t, cos_t)
            self.bend_rest_length = e_len.ravel()
        else:
            self.bend_rest_angle = np.array([], dtype=np.float64)
            self.bend_rest_length = np.array([], dtype=np.float64)

        # Triangle rest areas
        v0 = self.pos[self.tri_indices[:, 0]]
        v1 = self.pos[self.tri_indices[:, 1]]
        v2 = self.pos[self.tri_indices[:, 2]]
        self.rest_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        # Vertex-to-triangle adjacency (for force distribution)
        self.vert_adj_tris = [[] for _ in range(self.n_particles)]
        for t_idx, (i, j, k) in enumerate(self.tri_indices):
            self.vert_adj_tris[i].append(t_idx)
            self.vert_adj_tris[k].append(t_idx)

    def _build_laplacian(self):
        """Build cotangent-weighted Laplacian operator for bending.

        For each edge (i,j), the cotangent weight is:
          w_ij = (cot(alpha) + cot(beta)) / 2
        where alpha, beta are angles opposite to edge (i,j) in the two
        adjacent triangles. For boundary edges, only one cotangent is used.

        Stored as edge-based arrays (same structure as spring edges) for
        efficient GPU-style parallel computation.
        """
        pos = self.pos

        # Use all mesh edges (not just internal) for the Laplacian
        n_edges = len(self.spring_i)
        weights = np.zeros(n_edges, dtype=np.float64)

        # Build edge-to-triangle map (reuse from _build_edges)
        edge_to_tris = {}
        for t_idx, (i, j, k) in enumerate(self.tri_indices):
            for a, b in [(i, j), (j, k), (k, i)]:
                key = (min(a, b), max(a, b))
                if key not in edge_to_tris:
                    edge_to_tris[key] = []
                edge_to_tris[key].append(t_idx)

        # Build edge-key to spring-index map
        edge_key_to_idx = {}
        for e_idx in range(n_edges):
            a, b = int(self.spring_i[e_idx]), int(self.spring_j[e_idx])
            key = (min(a, b), max(a, b))
            edge_key_to_idx[key] = e_idx

        for e_idx in range(n_edges):
            a = int(self.spring_i[e_idx])
            b = int(self.spring_j[e_idx])
            key = (min(a, b), max(a, b))
            tris = edge_to_tris.get(key, [])

            w = 0.0
            for t_idx in tris:
                tri = self.tri_indices[t_idx]
                # Find the opposite vertex (not a or b)
                opp = [v for v in tri if v != a and v != b]
                if len(opp) != 1:
                    continue
                opp = opp[0]

                # Cotangent of angle at opposite vertex
                # Vectors from opp to edge endpoints
                va = pos[a] - pos[opp]
                vb = pos[b] - pos[opp]
                cross = np.cross(va, vb)
                dot = np.dot(va, vb)
                cross_norm = np.linalg.norm(cross)
                if cross_norm > 1e-12:
                    w += dot / cross_norm  # cot(angle)

            weights[e_idx] = w / 2.0

        self.lap_weights = weights  # (E,) cotangent weights per edge
        self.lap_i = self.spring_i  # same edge endpoints
        self.lap_j = self.spring_j

        # Identify boundary vertices and compute ghost-vertex info
        # For each boundary edge endpoint, find the correct mirror vertex by
        # looking inward from the boundary (perpendicular to boundary edge,
        # toward mesh interior). The mirror is the Laplacian neighbor most
        # aligned with that inward direction.
        boundary_verts = set()

        # Build per-vertex neighbor list with edge weights (from Laplacian)
        vert_neighbors = [[] for _ in range(self.n_particles)]
        for e_idx in range(n_edges):
            i, j = int(self.spring_i[e_idx]), int(self.spring_j[e_idx])
            w = weights[e_idx]
            vert_neighbors[i].append((j, w))
            vert_neighbors[j].append((i, w))

        ghost_v_list = []
        ghost_w_list = []
        ghost_mirror_list = []

        for (a, b), tris in edge_to_tris.items():
            if len(tris) == 1:
                boundary_verts.add(a)
                boundary_verts.add(b)
                t_idx = tris[0]
                tri = self.tri_indices[t_idx]
                c = [v for v in tri if v != a and v != b][0]

                # Inward direction: from edge midpoint toward opposite vertex c,
                # projected perpendicular to the boundary edge
                edge_vec = pos[b] - pos[a]
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-12:
                    continue
                edge_dir = edge_vec / edge_len
                mid_to_c = pos[c] - 0.5 * (pos[a] + pos[b])
                # Perpendicular component (inward normal of this boundary edge)
                inward = mid_to_c - np.dot(mid_to_c, edge_dir) * edge_dir
                inward_len = np.linalg.norm(inward)
                if inward_len < 1e-12:
                    continue
                inward_dir = inward / inward_len

                # For each endpoint, find best mirror neighbor
                for v in (a, b):
                    best_neighbor = -1
                    best_score = -1.0  # alignment * weight (prefer aligned AND strong)
                    for nbr, w in vert_neighbors[v]:
                        if w < 1e-12:
                            continue
                        d = pos[nbr] - pos[v]
                        d_len = np.linalg.norm(d)
                        if d_len < 1e-12:
                            continue
                        alignment = np.dot(d / d_len, inward_dir)
                        # Score: alignment (must be positive = inward) weighted by edge weight
                        score = alignment * w
                        if score > best_score:
                            best_score = score
                            best_neighbor = nbr

                    if best_neighbor >= 0:
                        # Find edge weight to best_neighbor
                        key = (min(v, best_neighbor), max(v, best_neighbor))
                        w_ghost = weights[edge_key_to_idx[key]] if key in edge_key_to_idx else 0.0
                        ghost_v_list.append(v)
                        ghost_w_list.append(w_ghost)
                        ghost_mirror_list.append(best_neighbor)

        self._is_boundary = np.zeros(self.n_particles, dtype=bool)
        for v in boundary_verts:
            self._is_boundary[v] = True

        # Deduplicate: for each vertex, keep only the ghost entry with
        # the largest weight. Corner vertices (on 2 boundary edges) would
        # get double ghost corrections that cancel their entire L1 row.
        per_vertex_best = {}  # vertex -> (weight, mirror)
        for i in range(len(ghost_v_list)):
            v = ghost_v_list[i]
            w = ghost_w_list[i]
            m = ghost_mirror_list[i]
            if v not in per_vertex_best or w > per_vertex_best[v][0]:
                per_vertex_best[v] = (w, m)

        dedup_v = list(per_vertex_best.keys())
        dedup_w = [per_vertex_best[v][0] for v in dedup_v]
        dedup_m = [per_vertex_best[v][1] for v in dedup_v]

        # Store ghost info as arrays for vectorized correction
        self._ghost_idx = np.array(dedup_v, dtype=np.int32)
        self._ghost_w = np.array(dedup_w, dtype=np.float64)
        self._ghost_mirror = np.array(dedup_m, dtype=np.int32)

        # Compute Voronoi area per vertex for correct energy normalization
        self.vertex_area = np.zeros(self.n_particles, dtype=np.float64)
        for t_idx, (i, j, k) in enumerate(self.tri_indices):
            A = self.rest_area[t_idx] / 3.0
            self.vertex_area[i] += A
            self.vertex_area[j] += A
            self.vertex_area[k] += A

    def _apply_laplacian(self, x):
        """Apply cotangent Laplacian to a (N, 3) field.

        L[i] = sum_j w_ij * (x[j] - x[i])

        This is a single SpMV-like operation over the edge list.
        Ghost-vertex boundary correction is handled separately in
        _compute_bending_elastic_forces for the first Laplacian pass.
        """
        diff = x[self.lap_j] - x[self.lap_i]  # (E, 3)
        weighted = self.lap_weights[:, None] * diff  # (E, 3)

        L = np.zeros_like(x)
        np.add.at(L, self.lap_i, weighted)
        np.add.at(L, self.lap_j, -weighted)
        return L

    def _apply_laplacian_scalar(self, w):
        """Apply cotangent Laplacian to scalar field w (N,).

        Same as _apply_laplacian but for scalar values — avoids (N,3) allocation.
        """
        diff = w[self.lap_j] - w[self.lap_i]
        weighted = self.lap_weights * diff
        L = np.zeros(self.n_particles, dtype=np.float64)
        np.add.at(L, self.lap_i, weighted)
        np.add.at(L, self.lap_j, -weighted)
        return L

    def _apply_ghost_scalar(self, L, w):
        """Apply ghost correction to scalar Laplacian result L for field w."""
        if len(self._ghost_idx) == 0:
            return
        gv = self._ghost_idx
        gm = self._ghost_mirror
        gw = self._ghost_w
        w_v = w[gv]
        w_m = w[gm]
        clamped = self.fixed_mask[gv]
        free = ~clamped
        correction = np.zeros(len(gv), dtype=np.float64)
        correction[clamped] = gw[clamped] * (w_m[clamped] - w_v[clamped])
        correction[free] = gw[free] * (w_v[free] - w_m[free])
        np.add.at(L, gv, correction)

    # ── Co-rotational Biharmonic Solver ──────────────────────────────────

    def _compute_vertex_normals(self):
        """Compute area-weighted vertex normals from adjacent triangles."""
        normals = np.zeros((self.n_particles, 3), dtype=np.float64)
        v0 = self.pos[self.tri_indices[:, 0]]
        v1 = self.pos[self.tri_indices[:, 1]]
        v2 = self.pos[self.tri_indices[:, 2]]
        tri_normals = np.cross(v1 - v0, v2 - v0)  # (T, 3), magnitude = 2*area
        for t_idx, (i, j, k) in enumerate(self.tri_indices):
            normals[i] += tri_normals[t_idx]
            normals[j] += tri_normals[t_idx]
            normals[k] += tri_normals[t_idx]
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.maximum(lengths, 1e-12)

    def solve_static_biharmonic(self, F_ext, n_corot_iters=3):
        """Solve static equilibrium using row-averaged 1D biharmonic.

        For beam-like strips: groups vertices by spanwise (y) coordinate,
        averages z-displacement per row, then solves the 1D Euler-Bernoulli
        beam equation EI*d⁴w/dy⁴=q using Hermite cubic FEM stiffness.

        This gives exact results for beam bending with proper clamped/free BCs.
        """
        # Build row structure (cached)
        if not hasattr(self, '_rows'):
            self._build_rows()

        rows = self._rows
        n_rows = len(rows)

        free = ~self.fixed_mask

        for corot_iter in range(n_corot_iters):
            normals = self._compute_vertex_normals()

            # Average z-displacement and force per row
            w_row = np.zeros(n_rows)
            f_row = np.zeros(n_rows)
            for r, verts_in_row in enumerate(rows):
                mask = free[verts_in_row]
                free_verts = verts_in_row[mask]
                if len(free_verts) == 0:
                    continue
                # Average displacement along normal
                disp = self.pos[free_verts] - self._pos_ref[free_verts]
                w_row[r] = np.mean(np.sum(disp * normals[free_verts], axis=1))
                # Sum forces along normal
                f_row[r] = np.sum(np.sum(F_ext[free_verts] * normals[free_verts], axis=1))

            # 1D Hermite cubic beam FEM stiffness
            # DOFs per node: [w, dw/dy], total 2*n_rows DOFs
            # Clamped root: remove DOFs 0,1 (w₀=0, w'₀=0)
            n_dof = 2 * n_rows
            dy = self._row_spacing

            # Assemble 1D beam stiffness K_1d and force vector
            K_1d = np.zeros((n_dof, n_dof))
            F_1d = np.zeros(n_dof)

            # Hermite cubic element stiffness (standard Euler-Bernoulli)
            for e in range(n_rows - 1):
                Le = dy
                ke = self.bending_D / (Le * Le * Le) * np.array([
                    [ 12,    6*Le,  -12,    6*Le  ],
                    [ 6*Le,  4*Le*Le, -6*Le, 2*Le*Le],
                    [-12,  -6*Le,   12,   -6*Le  ],
                    [ 6*Le,  2*Le*Le, -6*Le, 4*Le*Le],
                ])
                # Actually for beam: EI * K_hermite / L³
                # We use D = EI/w as bending_D, and distribute force as f_total per row
                # The 1D equation: EI * d⁴w/dy⁴ = f_per_length
                # With EI = D * w, f_per_length = f_row / w ... no.
                # Actually: f_row is the TOTAL force on the row (sum of per-vertex forces)
                # The beam equation per unit length: EI * d⁴w/dy⁴ = f_per_length
                # f_per_length = f_row / dy (force per row / spacing = force per unit length)
                # No wait, f_row already IS the total force on the row.
                # For FEM: the consistent force vector distributes f_row to the DOFs.
                # For uniform load q (N/m): f_e = q * [L/2, L²/12, L/2, -L²/12]
                # Here f_row acts as a point force at the row position.

                # Scale stiffness by strip width: EI = D * w where w is strip width
                EI_eff = self.bending_D * self._strip_width

                dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
                for i in range(4):
                    for j in range(4):
                        K_1d[dofs[i], dofs[j]] += EI_eff * ke[i, j] / (Le**3)
                        # Wait, ke already has the 1/L³ factor. Let me redo.

            # Redo with correct Hermite stiffness
            K_1d = np.zeros((n_dof, n_dof))
            EI_eff = self.bending_D * self._strip_width  # EI = D * w

            for e in range(n_rows - 1):
                Le = dy
                # Standard Hermite beam element stiffness: EI/L³ * [12, 6L, -12, 6L; ...]
                k = EI_eff / (Le**3) * np.array([
                    [ 12,    6*Le,   -12,    6*Le   ],
                    [ 6*Le,  4*Le**2, -6*Le,  2*Le**2],
                    [-12,   -6*Le,    12,   -6*Le   ],
                    [ 6*Le,  2*Le**2, -6*Le,  4*Le**2],
                ])
                dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                for i in range(4):
                    for j in range(4):
                        K_1d[dofs[i], dofs[j]] += k[i, j]

            # Force vector: f_row[i] is total force at row i.
            # For FEM with Hermite cubics, point load at node i:
            # F_{2i} = f_row[i], F_{2i+1} = 0 (no moment)
            for r in range(n_rows):
                F_1d[2*r] = f_row[r]

            # Apply BCs: clamped root (row 0) — remove DOFs 0,1
            fixed_dofs = [0, 1]  # w₀ = 0, w'₀ = 0
            free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]

            K_ff = K_1d[np.ix_(free_dofs, free_dofs)]
            F_ff = F_1d[free_dofs]

            # Solve
            w_1d = np.zeros(n_dof)
            w_1d[free_dofs] = np.linalg.solve(K_ff, F_ff)

            # Extract w and dw/dy per row
            w_per_row = w_1d[0::2]  # displacement at each row
            # dw_per_row = w_1d[1::2]  # slope at each row (not needed for update)

            # Map back to 3D vertices
            self.pos = self._pos_ref.copy()
            for r, verts_in_row in enumerate(rows):
                mask = free[verts_in_row]
                free_verts = verts_in_row[mask]
                if len(free_verts) > 0:
                    self.pos[free_verts] += w_per_row[r] * normals[free_verts]

        return w_per_row

    def _build_rows(self):
        """Group vertices by spanwise (y) coordinate into rows."""
        y_coords = self._pos_ref[:, 1]
        unique_y = np.unique(np.round(y_coords, 10))
        self._rows = []
        for y_val in unique_y:
            row = np.where(np.abs(y_coords - y_val) < 1e-8)[0]
            self._rows.append(row)

        # Compute row spacing and strip width
        if len(unique_y) > 1:
            self._row_spacing = unique_y[1] - unique_y[0]
        else:
            self._row_spacing = 1.0

        x_coords = self._pos_ref[:, 0]
        self._strip_width = x_coords.max() - x_coords.min()

        # Per-row EA position (for torsion DOF extraction)
        # The EA is at the minimum-x vertex (leading edge) or at a user-specified fraction
        if not hasattr(self, '_ea_chord_frac'):
            self._ea_chord_frac = 0.0  # default: LE

    # ── 1D Beam Dynamics (Bending + Torsion + CG Coupling) ─────────────

    def setup_beam(self, EI, GJ, m_per_length, Ip, x_ea_cg=0.0,
                   structural_damping=0.0, ea_chord_frac=0.33):
        """Enable 1D beam dynamics mode with bending-torsion coupling.

        Assembles Hermite cubic bending + linear torsion FEM matrices
        from the mesh row structure. Uses Newmark-beta time integration.

        Parameters
        ----------
        EI : float
            Bending stiffness [N·m²].
        GJ : float
            Torsional stiffness [N·m²].
        m_per_length : float
            Mass per unit span [kg/m].
        Ip : float
            Polar moment of inertia per unit length about EA [kg·m].
        x_ea_cg : float
            Distance from elastic axis to CG (positive = CG aft of EA) [m].
        structural_damping : float
            Rayleigh damping ratio.
        ea_chord_frac : float
            Elastic axis position as fraction of chord from LE [0, 1].
        """
        if not hasattr(self, '_rows'):
            self._build_rows()

        self._ea_chord_frac = ea_chord_frac
        n_rows = len(self._rows)
        n_dof = 3 * n_rows  # [w, dw/dy, theta] per row
        Le = self._row_spacing

        # Assemble stiffness and mass matrices (same as BeamFE)
        K = np.zeros((n_dof, n_dof))
        M = np.zeros((n_dof, n_dof))

        for e in range(n_rows - 1):
            # Bending (Hermite cubic, 4x4)
            Kb = EI / Le**3 * np.array([
                [ 12,    6*Le,   -12,    6*Le   ],
                [ 6*Le,  4*Le**2, -6*Le,  2*Le**2],
                [-12,   -6*Le,    12,   -6*Le   ],
                [ 6*Le,  2*Le**2, -6*Le,  4*Le**2],
            ])
            Mb = m_per_length * Le / 420 * np.array([
                [156,    22*Le,    54,    -13*Le  ],
                [22*Le,   4*Le**2,  13*Le,  -3*Le**2],
                [54,     13*Le,    156,   -22*Le  ],
                [-13*Le, -3*Le**2, -22*Le,   4*Le**2],
            ])

            # Torsion (linear, 2x2)
            Kt = GJ / Le * np.array([[1, -1], [-1, 1]])
            Mt = Ip * Le / 6 * np.array([[2, 1], [1, 2]])

            # Coupling mass (CG offset, 6x6)
            Mc = self._beam_coupling_mass(x_ea_cg, m_per_length, Le)

            # Combine into 6x6 element: DOFs [w1, w'1, θ1, w2, w'2, θ2]
            Ke = np.zeros((6, 6))
            Me = np.zeros((6, 6))
            bend_idx = [0, 1, 3, 4]
            for i, gi in enumerate(bend_idx):
                for j, gj in enumerate(bend_idx):
                    Ke[gi, gj] += Kb[i, j]
                    Me[gi, gj] += Mb[i, j]
            tors_idx = [2, 5]
            for i, gi in enumerate(tors_idx):
                for j, gj in enumerate(tors_idx):
                    Ke[gi, gj] += Kt[i, j]
                    Me[gi, gj] += Mt[i, j]
            Me += Mc

            # Assemble into global
            dofs = [e*3, e*3+1, e*3+2, (e+1)*3, (e+1)*3+1, (e+1)*3+2]
            for i in range(6):
                for j in range(6):
                    K[dofs[i], dofs[j]] += Ke[i, j]
                    M[dofs[i], dofs[j]] += Me[i, j]

        # Rayleigh damping
        C = np.zeros((n_dof, n_dof))
        if structural_damping > 0:
            from scipy.linalg import eigh
            fixed_dofs = [0, 1, 2]  # root: w=0, w'=0, θ=0
            free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
            K_r = K[np.ix_(free_dofs, free_dofs)]
            M_r = M[np.ix_(free_dofs, free_dofs)]
            eigvals, _ = eigh(K_r, M_r)
            omega1 = np.sqrt(max(eigvals[0], 1e-6))
            C = 2 * structural_damping / omega1 * K

        # Store beam state
        self._beam_K = K
        self._beam_M = M
        self._beam_C = C
        self._beam_d = np.zeros(n_dof)   # displacement
        self._beam_v = np.zeros(n_dof)   # velocity
        self._beam_a = np.zeros(n_dof)   # acceleration
        self._beam_n_dof = n_dof
        self._beam_n_rows = n_rows
        self._beam_Le = Le

        # Newmark parameters (average acceleration)
        self._beam_beta = 0.25
        self._beam_gamma = 0.5

    @staticmethod
    def _beam_coupling_mass(x_ea_cg, m, L):
        """Bending-torsion coupling mass matrix from CG offset.

        Same as BeamFE._coupling_numerical: 3-point Gauss quadrature.
        """
        gpts = np.array([0.5 - 0.5*np.sqrt(3/5), 0.5, 0.5 + 0.5*np.sqrt(3/5)])
        gwts = np.array([5/18, 8/18, 5/18])
        xi = gpts
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L * (xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L * (-xi**2 + xi**3)
        P1 = 1 - xi
        P2 = xi
        N_bend = np.array([N1, N2, N3, N4])
        N_tors = np.array([P1, P2])

        Mc = np.zeros((6, 6))
        bend_idx = [0, 1, 3, 4]
        tors_idx = [2, 5]
        for i, bi in enumerate(bend_idx):
            for j, tj in enumerate(tors_idx):
                integral = np.sum(gwts * N_bend[i] * N_tors[j]) * L
                val = m * x_ea_cg * integral
                Mc[bi, tj] = val
                Mc[tj, bi] = val
        return Mc

    def step_beam(self, F_ext, dt):
        """One timestep of beam dynamics (Newmark-beta).

        Extracts per-row forces from F_ext, integrates beam DOFs,
        maps updated DOFs back to 3D vertex positions.

        Parameters
        ----------
        F_ext : (N, 3) array
            External forces on mesh vertices.
        dt : float
            Timestep size.
        """
        if not hasattr(self, '_beam_K'):
            raise RuntimeError("Call setup_beam() first")

        K = self._beam_K
        M = self._beam_M
        C = self._beam_C
        n_dof = self._beam_n_dof
        n_rows = self._beam_n_rows
        Le = self._beam_Le
        rows = self._rows

        # 1. Map F_ext → beam force vector
        F_beam = np.zeros(n_dof)
        ea_frac = self._ea_chord_frac
        free_mask = ~self.fixed_mask

        for r in range(n_rows):
            row_verts = rows[r]
            free_in_row = row_verts[free_mask[row_verts]]
            if len(free_in_row) == 0:
                continue
            # Shear force (sum of z-forces)
            F_beam[3*r] = np.sum(F_ext[free_in_row, 2])
            # Moment about EA (from x-offset × Fz)
            x_local = self._pos_ref[free_in_row, 0] - self._pos_ref[free_in_row[0], 0]
            # More robust: use EA position
            x_min = self._pos_ref[row_verts, 0].min()
            x_max = self._pos_ref[row_verts, 0].max()
            chord = x_max - x_min
            x_ea = x_min + ea_frac * chord
            x_from_ea = self._pos_ref[free_in_row, 0] - x_ea
            F_beam[3*r + 2] = np.sum(F_ext[free_in_row, 2] * x_from_ea)

        # 2. Apply BC: clamped root (row 0) → remove DOFs 0,1,2
        fixed_dofs = [0, 1, 2]
        free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]

        K_r = K[np.ix_(free_dofs, free_dofs)]
        M_r = M[np.ix_(free_dofs, free_dofs)]
        C_r = C[np.ix_(free_dofs, free_dofs)]
        F_r = F_beam[free_dofs]

        d_r = self._beam_d[free_dofs]
        v_r = self._beam_v[free_dofs]
        a_r = self._beam_a[free_dofs]

        beta = self._beam_beta
        gamma = self._beam_gamma

        # Effective stiffness
        K_eff = K_r + gamma/(beta*dt)*C_r + 1/(beta*dt**2)*M_r

        # Effective force
        F_eff = F_r + M_r @ (
            1/(beta*dt**2)*d_r + 1/(beta*dt)*v_r + (1/(2*beta)-1)*a_r
        ) + C_r @ (
            gamma/(beta*dt)*d_r + (gamma/beta-1)*v_r + dt*(gamma/(2*beta)-1)*a_r
        )

        # Solve
        d_new_r = np.linalg.solve(K_eff, F_eff)

        # Update acceleration and velocity
        a_new_r = (1/(beta*dt**2)*(d_new_r - d_r)
                   - 1/(beta*dt)*v_r - (1/(2*beta)-1)*a_r)
        v_new_r = v_r + dt*((1-gamma)*a_r + gamma*a_new_r)

        # Map back
        self._beam_d[:] = 0.0
        self._beam_v[:] = 0.0
        self._beam_a[:] = 0.0
        self._beam_d[free_dofs] = d_new_r
        self._beam_v[free_dofs] = v_new_r
        self._beam_a[free_dofs] = a_new_r

        # 3. Map beam DOFs → vertex positions
        self._beam_dofs_to_vertices()

        # Zero velocity for fixed particles
        self.vel[self.fixed_mask] = 0.0

    def step_beam_dofs(self, F_beam, dt):
        """One timestep with beam-level forces directly (no vertex mapping).

        Use this when forces are naturally per-row (shear + moment)
        to avoid the vertex → beam round-trip.

        Parameters
        ----------
        F_beam : (n_beam_dof,) array
            Beam force vector. DOF order: [w0, w'0, θ0, w1, w'1, θ1, ...]
            F_beam[3*r] = shear force on row r (positive = +z)
            F_beam[3*r+1] = bending moment on row r (usually 0)
            F_beam[3*r+2] = torque on row r (positive = nose-up)
        dt : float
            Timestep size.
        """
        if not hasattr(self, '_beam_K'):
            raise RuntimeError("Call setup_beam() first")

        K = self._beam_K
        M = self._beam_M
        C = self._beam_C
        n_dof = self._beam_n_dof

        fixed_dofs = [0, 1, 2]
        free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]

        K_r = K[np.ix_(free_dofs, free_dofs)]
        M_r = M[np.ix_(free_dofs, free_dofs)]
        C_r = C[np.ix_(free_dofs, free_dofs)]
        F_r = F_beam[free_dofs]

        d_r = self._beam_d[free_dofs]
        v_r = self._beam_v[free_dofs]
        a_r = self._beam_a[free_dofs]

        beta = self._beam_beta
        gamma = self._beam_gamma

        K_eff = K_r + gamma/(beta*dt)*C_r + 1/(beta*dt**2)*M_r

        F_eff = F_r + M_r @ (
            1/(beta*dt**2)*d_r + 1/(beta*dt)*v_r + (1/(2*beta)-1)*a_r
        ) + C_r @ (
            gamma/(beta*dt)*d_r + (gamma/beta-1)*v_r + dt*(gamma/(2*beta)-1)*a_r
        )

        d_new_r = np.linalg.solve(K_eff, F_eff)

        a_new_r = (1/(beta*dt**2)*(d_new_r - d_r)
                   - 1/(beta*dt)*v_r - (1/(2*beta)-1)*a_r)
        v_new_r = v_r + dt*((1-gamma)*a_r + gamma*a_new_r)

        self._beam_d[:] = 0.0
        self._beam_v[:] = 0.0
        self._beam_a[:] = 0.0
        self._beam_d[free_dofs] = d_new_r
        self._beam_v[free_dofs] = v_new_r
        self._beam_a[free_dofs] = a_new_r

        self._beam_dofs_to_vertices()
        self.vel[self.fixed_mask] = 0.0

    def _beam_dofs_to_vertices(self):
        """Map beam DOFs (w, w', θ per row) to 3D vertex positions."""
        rows = self._rows
        ea_frac = self._ea_chord_frac
        self.pos = self._pos_ref.copy()

        for r, row_verts in enumerate(rows):
            w_r = self._beam_d[3*r]        # heave
            theta_r = self._beam_d[3*r+2]  # twist

            if len(row_verts) == 0:
                continue

            # Chord geometry
            x_min = self._pos_ref[row_verts, 0].min()
            x_max = self._pos_ref[row_verts, 0].max()
            chord = x_max - x_min
            if chord < 1e-12:
                continue
            x_ea = x_min + ea_frac * chord

            for v in row_verts:
                x_local = self._pos_ref[v, 0] - x_ea  # distance from EA
                y = self._pos_ref[v, 1]
                # Bending + twist
                self.pos[v, 0] = self._pos_ref[v, 0] + x_local * (np.cos(theta_r) - 1)
                self.pos[v, 2] = w_r + x_local * np.sin(theta_r)

    def get_beam_dofs(self):
        """Return current beam DOFs: w (n_rows,), theta (n_rows,)."""
        n_rows = self._beam_n_rows
        w = self._beam_d[0::3]
        theta = self._beam_d[2::3]
        return w, theta

    def perturb_beam_tip(self, w_tip=0.0, theta_tip=0.0):
        """Apply initial perturbation to tip beam DOFs."""
        tip = self._beam_n_rows - 1
        self._beam_d[3*tip] = w_tip
        self._beam_d[3*tip + 2] = theta_tip
        self._beam_dofs_to_vertices()

    def compute_beam_natural_frequencies(self, n_modes=None):
        """Compute natural frequencies of the beam."""
        from scipy.linalg import eigh
        fixed_dofs = [0, 1, 2]
        free_dofs = [i for i in range(self._beam_n_dof) if i not in fixed_dofs]
        K_r = self._beam_K[np.ix_(free_dofs, free_dofs)]
        M_r = self._beam_M[np.ix_(free_dofs, free_dofs)]
        eigvals, eigvecs = eigh(K_r, M_r)
        eigvals = np.maximum(eigvals, 0)
        freqs = np.sqrt(eigvals) / (2*np.pi)
        if n_modes is not None:
            freqs = freqs[:n_modes]
            eigvecs = eigvecs[:, :n_modes]
        return freqs, eigvecs

    def solve_static_plate(self, F_ext, n_corot_iters=3):
        """Solve 2D thin plate bending using cotangent Laplacian biharmonic.

        For fully clamped plates (all boundary vertices fixed), the biharmonic
        K = L @ A_inv @ L on interior DOFs is correct without ghost correction.
        The continuous equation is D * Δ²w = q (Kirchhoff plate theory).

        Parameters
        ----------
        F_ext : (N, 3) external forces (pressure × vertex_area)
        n_corot_iters : co-rotation iterations for large deformations
        """
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        # Build biharmonic stiffness on interior DOFs (cached)
        if not hasattr(self, '_K_plate'):
            self._assemble_plate_stiffness()

        K = self._K_plate
        free = ~self.fixed_mask
        free_idx = np.where(free)[0]
        K_ff = K[np.ix_(free_idx, free_idx)]

        # Factorize once
        if not hasattr(self, '_K_plate_lu'):
            from scipy.sparse.linalg import splu
            self._K_plate_lu = splu(K_ff.tocsc())

        for corot_iter in range(n_corot_iters):
            normals = self._compute_vertex_normals()
            f_n = np.sum(F_ext * normals, axis=1)

            # Solve: D * K_ff * w = f_n_free
            rhs = f_n[free_idx] / self.bending_D
            w_free = self._K_plate_lu.solve(rhs)

            w_new = np.zeros(self.n_particles, dtype=np.float64)
            w_new[free_idx] = w_free

            self.pos = self._pos_ref.copy()
            self.pos[free_idx] += w_new[free_idx, None] * normals[free_idx]
            self.pos[self.fixed_mask] = self._pos_ref[self.fixed_mask]

        return w_new

    def _assemble_plate_stiffness(self):
        """Build K = L @ diag(1/A) @ L for 2D plate bending.

        Uses the standard cotangent Laplacian (no ghost correction).
        Only valid when ALL boundary vertices are fixed (clamped plate).
        """
        from scipy import sparse

        n = self.n_particles
        rows, cols, vals = [], [], []

        for e in range(len(self.lap_i)):
            i, j = int(self.lap_i[e]), int(self.lap_j[e])
            w_e = self.lap_weights[e]
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w_e, w_e])
        for e in range(len(self.lap_i)):
            i, j = int(self.lap_i[e]), int(self.lap_j[e])
            w_e = self.lap_weights[e]
            rows.extend([i, j])
            cols.extend([i, j])
            vals.extend([-w_e, -w_e])

        L_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        A_vec = np.maximum(self.vertex_area, 1e-12)
        A_inv = sparse.diags(1.0 / A_vec)

        self._K_plate = (L_mat @ A_inv @ L_mat).tocsc()

    def _build_L1(self):
        """Build ghost-corrected cotangent Laplacian as sparse matrix."""
        from scipy import sparse

        n = self.n_particles
        rows, cols, vals = [], [], []

        # Standard Laplacian entries
        for e in range(len(self.lap_i)):
            i, j = int(self.lap_i[e]), int(self.lap_j[e])
            w_e = self.lap_weights[e]
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w_e, w_e])
        for e in range(len(self.lap_i)):
            i, j = int(self.lap_i[e]), int(self.lap_j[e])
            w_e = self.lap_weights[e]
            rows.extend([i, j])
            cols.extend([i, j])
            vals.extend([-w_e, -w_e])

        # Ghost correction entries
        for k in range(len(self._ghost_idx)):
            gv = int(self._ghost_idx[k])
            gm = int(self._ghost_mirror[k])
            gw = self._ghost_w[k]
            if self.fixed_mask[gv]:
                # clamped: even extension (mirror = self)
                rows.extend([gv, gv])
                cols.extend([gm, gv])
                vals.extend([gw, -gw])
            else:
                # free: odd extension (mirror = reflection)
                rows.extend([gv, gv])
                cols.extend([gm, gv])
                vals.extend([-gw, gw])

        self._L1_csc = sparse.csr_matrix((vals, (rows, cols)),
                                          shape=(n, n)).tocsc()

    def _assemble_biharmonic_stiffness(self):
        """Assemble scalar biharmonic stiffness K = L1 @ diag(1/A) @ L1.

        L1 is the ghost-corrected cotangent Laplacian from reference config.
        K operates on scalar fields (N,) and is assembled once.
        """
        from scipy import sparse

        n = self.n_particles

        # Build Laplacian sparse matrix L_mat: L[i,j] = w_ij, L[i,i] = -sum_j w_ij
        rows, cols, vals = [], [], []
        for e in range(len(self.lap_i)):
            i, j = int(self.lap_i[e]), int(self.lap_j[e])
            w_e = self.lap_weights[e]
            # Off-diagonal: L[i,j] = +w, L[j,i] = +w
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w_e, w_e])
        # Diagonal: L[i,i] = -sum_j w_ij
        for e in range(len(self.lap_i)):
            i, j = int(self.lap_i[e]), int(self.lap_j[e])
            w_e = self.lap_weights[e]
            rows.extend([i, j])
            cols.extend([i, j])
            vals.extend([-w_e, -w_e])

        L_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

        # Ghost correction matrix G
        g_rows, g_cols, g_vals = [], [], []
        for k in range(len(self._ghost_idx)):
            gv = int(self._ghost_idx[k])
            gm = int(self._ghost_mirror[k])
            gw = self._ghost_w[k]
            if self.fixed_mask[gv]:
                # clamped: L[gv, gm] += gw, L[gv, gv] -= gw
                g_rows.extend([gv, gv])
                g_cols.extend([gm, gv])
                g_vals.extend([gw, -gw])
            else:
                # free: L[gv, gm] -= gw, L[gv, gv] += gw
                g_rows.extend([gv, gv])
                g_cols.extend([gm, gv])
                g_vals.extend([-gw, gw])

        G_mat = sparse.csr_matrix((g_vals, (g_rows, g_cols)), shape=(n, n))
        L1 = L_mat + G_mat

        # Area inverse diagonal
        A_vec = np.maximum(self.vertex_area, 1e-12)
        A_inv = sparse.diags(1.0 / A_vec)

        # Biharmonic: K = L1 @ A_inv @ L1 (ghost-corrected).
        # L1 is non-symmetric due to different ghost corrections for
        # clamped vs free boundaries, but splu handles non-symmetric systems.
        self._K_biharm = (L1 @ A_inv @ L1).tocsc()
        self._L1 = L1.tocsc()
        self._A_inv = A_inv

    def _compute_mass_from_density(self, density):
        """Distribute mass to vertices based on triangle areas."""
        mass = np.zeros(self.n_particles, dtype=np.float64)
        for t_idx, (i, j, k) in enumerate(self.tri_indices):
            tri_mass = density * self.rest_area[t_idx]
            mass[i] += tri_mass / 3.0
            mass[j] += tri_mass / 3.0
            mass[k] += tri_mass / 3.0
        return mass

    # ── Public API: constraints ────────────────────────────────────────

    def fix_particles(self, indices):
        """Fix particles at their current position (zero inv_mass)."""
        indices = np.atleast_1d(indices)
        self.fixed_mask[indices] = True
        self.inv_mass[indices] = 0.0

    def add_spring(self, i, j, ke, kd, rest_length=None):
        """Add an extra spring between particles i and j.

        Used for actuators, structural reinforcements, or hinge models.
        If rest_length is None, uses current distance.
        """
        self._extra_spring_i.append(i)
        self._extra_spring_j.append(j)
        self._extra_spring_ke.append(ke)
        self._extra_spring_kd.append(kd)
        if rest_length is None:
            rest_length = np.linalg.norm(self.pos[i] - self.pos[j])
        self._extra_spring_rest.append(rest_length)
        self._extra_springs_integrated = False

    def _integrate_extra_springs(self):
        """Merge extra springs into the main spring arrays."""
        if self._extra_springs_integrated or not self._extra_spring_i:
            return
        n_extra = len(self._extra_spring_i)
        self.spring_i = np.concatenate([self.spring_i, self._extra_spring_i])
        self.spring_j = np.concatenate([self.spring_j, self._extra_spring_j])
        self.spring_ke = np.concatenate([self.spring_ke, self._extra_spring_ke])
        self.spring_kd = np.concatenate([self.spring_kd, self._extra_spring_kd])
        self.spring_rest = np.concatenate([self.spring_rest, self._extra_spring_rest])
        self._extra_springs_integrated = True

    # ── Force-Based Solver ─────────────────────────────────────────────

    def _compute_spring_forces(self):
        """Compute spring forces: F = -ke*(|Δx| - L₀)*n - kd*(v_rel·n)*n.

        Returns (N, 3) force array.
        """
        self._integrate_extra_springs()

        xi = self.pos[self.spring_i]
        xj = self.pos[self.spring_j]
        vi = self.vel[self.spring_i]
        vj = self.vel[self.spring_j]

        dx = xj - xi
        lengths = np.linalg.norm(dx, axis=1)
        safe_len = np.maximum(lengths, 1e-12)
        n = dx / safe_len[:, None]

        # Elastic force: F = ke * (L - L₀) along n
        stretch = lengths - self.spring_rest
        f_elastic = self.spring_ke * stretch

        # Damping: F = kd * (v_rel · n) along n
        v_rel = vi - vj
        f_damp = self.spring_kd * np.sum(v_rel * n, axis=1)

        f_total = (f_elastic + f_damp)[:, None] * n

        # Accumulate to vertices
        F = np.zeros((self.n_particles, 3), dtype=np.float64)
        np.add.at(F, self.spring_i, f_total)
        np.add.at(F, self.spring_j, -f_total)

        return F

    def _compute_bending_forces(self):
        """Compute bending forces from Laplacian energy (combined elastic + damp).

        Energy: E = 0.5 * bending_D * sum_i |L(x_i)|^2 / A_i
        Force:  F = -bending_D * L(L(x))  (biharmonic via two Laplacian passes)

        Plus velocity-dependent damping: F_damp = -bending_damp * L(v)

        Returns (N, 3) force array.
        """
        if self.bending_D <= 0 and self.bending_damp <= 0:
            return np.zeros((self.n_particles, 3), dtype=np.float64)

        F = np.zeros((self.n_particles, 3), dtype=np.float64)

        if self.bending_D > 0:
            # Elastic bending: F = -bending_D * L(L(x))
            L1 = self._apply_laplacian(self.pos)
            L2 = self._apply_laplacian(L1)
            F -= self.bending_D * L2

        if self.bending_damp > 0:
            # Damping: F = -bending_damp * L(v)
            Lv = self._apply_laplacian(self.vel)
            F -= self.bending_damp * Lv

        return F

    def step_force(self, F_ext, dt):
        """Advance one timestep using explicit force integration (Symplectic Euler).

        ke/kd directly represent physical spring stiffness/damping.
        No compliance mapping — frequency is independent of dt.

        Uses semi-implicit damping for stability: the damping force on velocity
        is treated implicitly, preventing the c*dt/m > 2 instability of explicit
        damping in Symplectic Euler.

        Parameters
        ----------
        F_ext : (N, 3) array
            External forces (gravity, aerodynamic).
        dt : float
            Timestep size.
        """
        # Compute elastic (position-dependent) forces
        F_elastic = self._compute_elastic_forces()

        # Compute damping (velocity-dependent) forces
        F_damp = self._compute_damping_forces()

        # Total force
        F_total = F_ext + F_elastic + F_damp

        # Semi-implicit velocity update with implicit damping:
        #   v_new = v + (F_elastic + F_ext - c*v) / m * dt
        #   v_new * (1 + c_eff*dt/m) = v + (F_elastic + F_ext) / m * dt
        #   v_new = (v + F_elastic_dt + F_ext_dt) / (1 + c_eff * dt / m)
        # We approximate c_eff per-particle from |F_damp| / |v| when |v| > 0.
        acc_elastic = (F_ext + F_elastic) * self.inv_mass[:, None]

        # Per-particle effective damping ratio: c_eff * dt / m
        # Approximate from |F_damp| ≈ c_eff * |v| → c_eff ≈ |F_damp| / max(|v|, eps)
        vel_norm = np.linalg.norm(self.vel, axis=1)
        damp_norm = np.linalg.norm(F_damp, axis=1)
        # c_eff * dt / m = |F_damp| * inv_mass * dt / max(|v|, eps)
        damp_ratio = damp_norm * self.inv_mass * dt / np.maximum(vel_norm, 1e-10)
        # Clamp to reasonable range
        damp_ratio = np.minimum(damp_ratio, 10.0)

        self.vel = (self.vel + acc_elastic * dt) / (1.0 + damp_ratio[:, None])
        self.pos += self.vel * dt

        # Zero velocity and position change for fixed particles
        self.vel[self.fixed_mask] = 0.0

    def step_hybrid(self, F_ext, dt, n_iterations=10):
        """Hybrid solver: XPBD for stretch constraints + force-based bending.

        Stretch edges are enforced via XPBD constraint projection (handles
        extreme stiffness naturally). Bending is computed as Laplacian-based
        explicit forces (biharmonic operator).

        This gives dt-independent bending dynamics while maintaining stretch
        rigidity through constraint projection.

        Parameters
        ----------
        F_ext : (N, 3) array
            External forces (gravity, aerodynamic).
        dt : float
            Timestep size.
        n_iterations : int
            XPBD iterations for stretch constraint projection.
        """
        self._integrate_extra_springs()

        pos_old = self.pos.copy()

        # 1. Compute bending forces (force-based, dt-independent physics)
        F_bend = self._compute_bending_elastic_forces()
        F_bend += self._compute_bending_damping_forces()

        # 2. Predict positions with external + bending forces
        F_total = F_ext + F_bend
        acc = F_total * self.inv_mass[:, None]
        self.pos = pos_old + self.vel * dt + acc * (dt * dt)

        # 3. XPBD stretch constraint projection (handles rigid stretch)
        n_springs = len(self.spring_i)

        if not hasattr(self, '_constraint_count'):
            self._constraint_count = np.zeros(self.n_particles, dtype=np.float64)
            np.add.at(self._constraint_count, self.spring_i, 1.0)
            np.add.at(self._constraint_count, self.spring_j, 1.0)
            self._constraint_count = np.maximum(self._constraint_count, 1.0)

        lambdas_spring = np.zeros(n_springs)

        for _ in range(n_iterations):
            delta = np.zeros_like(self.pos)
            self._solve_springs(dt, lambdas_spring, delta)
            self.pos += delta / self._constraint_count[:, None]

        # 4. Derive velocity
        self.vel = (self.pos - pos_old) / dt

    def solve_static(self, F_ext, n_iterations=100, stretch_iters=5,
                     relaxation=0.01):
        """Find static equilibrium iteratively (no time integration).

        Computes bending forces and applies XPBD stretch constraints
        alternately until equilibrium. No velocity, no timestep, no CFL issues.

        Parameters
        ----------
        F_ext : (N, 3) array
            External forces (applied loads, gravity).
        n_iterations : int
            Number of outer iterations.
        stretch_iters : int
            XPBD iterations per outer step for stretch constraints.
        relaxation : float
            Position update step size (small = stable, large = fast).
        """
        self._integrate_extra_springs()

        free = ~self.fixed_mask

        for it in range(n_iterations):
            # Compute bending forces (Laplacian-based)
            F_bend = self._compute_bending_elastic_forces()
            F_total = F_ext + F_bend

            # Move free particles toward equilibrium (with displacement clamping)
            displacement = F_total * self.inv_mass[:, None] * relaxation
            disp_norm = np.linalg.norm(displacement, axis=1, keepdims=True)
            max_disp = 5e-3  # max 5mm per iteration
            scale = np.minimum(1.0, max_disp / np.maximum(disp_norm, 1e-20))
            displacement *= scale
            self.pos[free] += displacement[free]

            # Position-based stretch projection (keep edges at rest length)
            for _ in range(stretch_iters):
                xi = self.pos[self.spring_i]
                xj = self.pos[self.spring_j]
                diff = xj - xi
                lengths = np.linalg.norm(diff, axis=1)
                safe_len = np.maximum(lengths, 1e-12)
                n = diff / safe_len[:, None]
                # Correction: push/pull to rest length
                correction = (lengths - self.spring_rest)[:, None] * n * 0.5
                wi = self.inv_mass[self.spring_i]
                wj = self.inv_mass[self.spring_j]
                ws = wi + wj
                ws_safe = np.maximum(ws, 1e-12)
                self.pos[self.spring_i] += correction * (wi / ws_safe)[:, None]
                self.pos[self.spring_j] -= correction * (wj / ws_safe)[:, None]

            # Fix root positions
            self.pos[self.fixed_mask] = self._pos_ref[self.fixed_mask]

            if it % 50 == 0:
                f_net = np.linalg.norm(F_total[free])
                if f_net < 1e-8:
                    break

    def _compute_elastic_forces(self):
        """Compute position-dependent (elastic) forces: springs + bending.

        Returns (N, 3) force array. No velocity-dependent terms.
        """
        self._integrate_extra_springs()

        # Spring elastic forces: F = ke * (|dx| - L0) * n
        F = np.zeros((self.n_particles, 3), dtype=np.float64)

        xi = self.pos[self.spring_i]
        xj = self.pos[self.spring_j]
        dx = xj - xi
        lengths = np.linalg.norm(dx, axis=1)
        safe_len = np.maximum(lengths, 1e-12)
        n = dx / safe_len[:, None]

        stretch = lengths - self.spring_rest
        f_spring = (self.spring_ke * stretch)[:, None] * n

        np.add.at(F, self.spring_i, f_spring)
        np.add.at(F, self.spring_j, -f_spring)

        # Bending elastic forces (Laplacian-based)
        if self.bending_D > 0:
            F += self._compute_bending_elastic_forces()

        return F

    def _compute_damping_forces(self):
        """Compute velocity-dependent (damping) forces.

        Returns (N, 3) force array.
        """
        F = np.zeros((self.n_particles, 3), dtype=np.float64)

        # Spring damping: F = -kd * (v_rel · n) * n
        xi = self.pos[self.spring_i]
        xj = self.pos[self.spring_j]
        vi = self.vel[self.spring_i]
        vj = self.vel[self.spring_j]

        dx = xj - xi
        lengths = np.linalg.norm(dx, axis=1)
        safe_len = np.maximum(lengths, 1e-12)
        n = dx / safe_len[:, None]

        v_rel = vi - vj
        f_damp = (self.spring_kd * np.sum(v_rel * n, axis=1))[:, None] * n

        np.add.at(F, self.spring_i, f_damp)
        np.add.at(F, self.spring_j, -f_damp)

        # Bending damping (Laplacian-based)
        if self.bending_damp > 0:
            F += self._compute_bending_damping_forces()

        return F

    def _compute_bending_elastic_forces(self):
        """Compute elastic bending forces via Laplacian biharmonic.

        Uses displacement from reference config to isolate bending:
          delta = pos - pos_ref  (out-of-plane displacement)
          Hn_i = L(delta_i) / A_i  (Laplacian per unit area)
          F = -bending_D * L(Hn)

        This correctly discretizes the thin-plate biharmonic F = -D * Δ²w.
        Ghost-vertex boundary correction on both Laplacian passes.
        """
        if self.bending_D <= 0:
            return np.zeros((self.n_particles, 3), dtype=np.float64)

        area = np.maximum(self.vertex_area, 1e-12)
        displacement = self.pos - self._pos_ref
        L1 = self._apply_laplacian(displacement)
        self._apply_ghost_correction(L1, displacement)

        Hn = L1 / area[:, None]
        L2 = self._apply_laplacian(Hn)
        self._apply_ghost_correction(L2, Hn)

        return -self.bending_D * L2

    def _apply_ghost_correction(self, L, x):
        """Apply ghost-vertex correction to Laplacian result L for field x."""
        if len(self._ghost_idx) == 0:
            return
        gv = self._ghost_idx
        gm = self._ghost_mirror
        gw = self._ghost_w
        x_v = x[gv]
        x_m = x[gm]
        clamped = self.fixed_mask[gv]
        free = ~clamped
        correction = np.zeros((len(gv), 3), dtype=np.float64)
        correction[clamped] = gw[clamped, None] * (x_m[clamped] - x_v[clamped])
        correction[free] = gw[free, None] * (x_v[free] - x_m[free])
        np.add.at(L, gv, correction)

    def _compute_bending_damping_forces(self):
        """Compute velocity-dependent bending damping.

        F = -bending_damp * L(v)
        """
        if self.bending_damp <= 0:
            return np.zeros((self.n_particles, 3), dtype=np.float64)

        Lv = self._apply_laplacian(self.vel)
        return -self.bending_damp * Lv

    # ── XPBD Beam Mode (Dihedral Bending + Torsion) ──────────────────

    def setup_xpbd_beam(self, EI, GJ, m_per_length, Ip, x_ea_cg=0.0,
                        structural_damping=0.0, ea_chord_frac=0.33):
        """Configure XPBD beam mode with dihedral bending + torsion constraints.

        Maps beam section properties to per-edge XPBD constraint stiffnesses
        and per-vertex mass distribution for CG coupling.

        Parameters
        ----------
        EI : float
            Bending stiffness [N·m²].
        GJ : float
            Torsional stiffness [N·m²].
        m_per_length : float
            Mass per unit span [kg/m].
        Ip : float
            Polar moment of inertia per unit length about EA [kg·m].
        x_ea_cg : float
            Distance from EA to CG (positive = CG aft of EA) [m].
        structural_damping : float
            Damping ratio for Rayleigh-type damping.
        ea_chord_frac : float
            Elastic axis position as fraction of chord from LE.
        """
        if not hasattr(self, '_rows'):
            self._build_rows()

        self._xpbd_beam_ea_frac = ea_chord_frac
        rows = self._rows
        n_rows = len(rows)
        Le = self._row_spacing

        # Count internal edges that contribute to bending per row gap
        n_bend_internal = len(self.bend_i)

        # Dihedral bending stiffness per internal edge.
        # Total bending energy per element = 0.5 * (EI/Le) * φ²
        # Distributed across all internal edges in the strip.
        # For n_chord=2 strip: ~4-5 internal edges per row gap contribute.
        # Row-based curvature bending stiffness (calibrated for n_chord=2 strips).
        # Physical bending energy: U = 0.5 * (EI/Le^3) * C^2, where C = second finite diff.
        # Empirical calibration: ke_bend ≈ 0.105 * EI/Le for n_chord=2, n_iter=15.
        # Factor accounts for XPBD iteration count and constraint coupling.
        n_gaps = n_rows - 1
        self._dihedral_ke = 0.0  # disable dihedral angle bending
        self._dihedral_kd = 0.0

        # Torsion topology: for each pair of adjacent rows, find LE and TE vertices
        self._torsion_le = []  # LE vertex index per row
        self._torsion_te = []  # TE vertex index per row
        for r, row_verts in enumerate(rows):
            x_coords = self._pos_ref[row_verts, 0]
            le_idx = row_verts[np.argmin(x_coords)]
            te_idx = row_verts[np.argmax(x_coords)]
            self._torsion_le.append(le_idx)
            self._torsion_te.append(te_idx)
        self._torsion_le = np.array(self._torsion_le, dtype=np.int32)
        self._torsion_te = np.array(self._torsion_te, dtype=np.int32)

        # Torsion constraint stiffness (calibrated for n_chord=2, n_iter=15).
        # Empirical: ke_tors ≈ 0.409 * GJ / Le
        self._torsion_ke = 0.409 * GJ / Le
        self._torsion_kd = structural_damping * 2.0 * self._torsion_ke
        self._torsion_chord = self._strip_width

        # Asymmetric mass distribution for CG coupling
        # CG offset from EA → more mass on CG side
        chord = self._strip_width
        x_ea = ea_chord_frac * chord
        for r, row_verts in enumerate(rows):
            n_v = len(row_verts)
            mass_per_row = m_per_length * Le
            # Distribute mass with CG offset:
            # m_i = m_base * (1 + x_cg_offset_factor * x_from_center)
            x_positions = self._pos_ref[row_verts, 0]
            x_from_ea = x_positions - x_ea
            # Mass moment about EA should equal m_per_length * x_ea_cg * Le
            # Simple: shift mass proportionally to x_from_ea
            mass_base = mass_per_row / n_v
            # Moment from base distribution:
            moment_base = np.sum(mass_base * x_from_ea)
            # Desired moment: m_per_length * x_ea_cg * Le
            moment_desired = m_per_length * x_ea_cg * Le
            # Correction: shift mass proportional to x_from_ea
            # Δm_i = α * x_from_ea_i, Σ Δm_i = 0 (conservation)
            # Σ Δm_i * x_from_ea_i = moment_desired - moment_base
            sum_x2 = np.sum(x_from_ea**2)
            if sum_x2 > 1e-12:
                alpha = (moment_desired - moment_base) / sum_x2
            else:
                alpha = 0.0
            self.particle_mass[row_verts] = mass_base + alpha * x_from_ea
            # Ensure positive mass
            self.particle_mass[row_verts] = np.maximum(self.particle_mass[row_verts], 1e-6)

        # Recompute inv_mass
        self.inv_mass = 1.0 / self.particle_mass
        self.inv_mass[self.fixed_mask] = 0.0

        # Row-based bending stiffness (replaces dihedral angle approach).
        # Constraint: C_r = w_{r-1} - 2*w_r + w_{r+1} (curvature)
        # Empirical calibration for n_chord=2, n_iter=15:
        #   ke_bend ≈ 0.105 * EI / Le  (gives correct Euler-Bernoulli frequency)
        self._row_bend_ke = 0.105 * EI / Le
        self._row_bend_kd = structural_damping * 2.0 * self._row_bend_ke

        # Damping: simple velocity damping factor
        self._xpbd_beam_damping = structural_damping

        self._xpbd_beam_configured = True

    def step_xpbd_beam(self, F_ext, dt, n_iterations=15):
        """XPBD step with row-based curvature bending + torsion + stretch constraints.

        Uses position-based prediction (not velocity accumulation) to avoid
        oscillatory feedback from constraint overshoot. The external force is
        applied as a position correction (F/m*dt^2), not a velocity impulse,
        so the derived velocity is self-consistent with the constraint projection.
        """
        if not getattr(self, '_xpbd_beam_configured', False):
            raise RuntimeError("Call setup_xpbd_beam() first")

        self._integrate_extra_springs()
        pos_old = self.pos.copy()

        # 1. Predict: inertial + force → position (NO velocity accumulation)
        self.pos = pos_old + self.vel * dt + F_ext * self.inv_mass[:, None] * (dt * dt)

        # 2. XPBD constraint projection
        n_springs = len(self.spring_i)
        n_rows = len(self._rows)
        n_curvature = max(n_rows - 2, 0)
        n_torsion = len(self._torsion_le) - 1

        lambdas_spring = np.zeros(n_springs)
        lambdas_curv = np.zeros(n_curvature)
        lambdas_torsion = np.zeros(n_torsion)

        if not hasattr(self, '_constraint_count'):
            self._constraint_count = np.zeros(self.n_particles, dtype=np.float64)
            np.add.at(self._constraint_count, self.spring_i, 1.0)
            np.add.at(self._constraint_count, self.spring_j, 1.0)
            self._constraint_count = np.maximum(self._constraint_count, 1.0)

        for _ in range(n_iterations):
            # Stretch constraints (distance)
            delta = np.zeros_like(self.pos)
            self._solve_springs(dt, lambdas_spring, delta)
            self.pos += delta / self._constraint_count[:, None]

            # Bending constraints (row-based curvature)
            self._solve_xpbd_curvature(dt, lambdas_curv)

            # Torsion constraints (relative twist)
            self._solve_xpbd_torsion(dt, lambdas_torsion)

        # 3. Derive velocity from position change
        self.vel = (self.pos - pos_old) / dt

        # 4. Apply damping
        if self._xpbd_beam_damping > 0:
            self.vel *= (1.0 - self._xpbd_beam_damping)

        # 5. Fix particles
        self.vel[self.fixed_mask] = 0.0

    def step_xpbd_aero(self, V, rho=1.225, CL_alpha=2*np.pi,
                        ea_chord_frac=None, dt=0.001, n_iterations=15,
                        aero_compliance_factor=1.0):
        """XPBD step with aero as a soft constraint inside the iteration loop.

        Divergence cause: aero as F_ext creates position impulse → conflicts
        with constraint projection → exponential overshoot.

        Fix: aero target becomes an XPBD constraint with soft compliance.
        Each iteration balances structural constraints against aero target.
        Structural constraints always win (stiffer), aero gently steers.
        """
        if not getattr(self, '_xpbd_beam_configured', False):
            raise RuntimeError("Call setup_xpbd_beam() first")

        if ea_chord_frac is None:
            ea_chord_frac = getattr(self, '_xpbd_beam_ea_frac', 0.33)
        chord = getattr(self, '_strip_width', 1.0)
        e_ac = (ea_chord_frac - 0.25) * chord
        q = 0.5 * rho * V * V
        Q = q * chord * CL_alpha
        rows = self._rows
        n_rows = len(rows)
        Le = self._row_spacing

        self._integrate_extra_springs()
        pos_old = self.pos.copy()

        # Predict: inertial only (aero handled as constraint)
        self.pos = pos_old + self.vel * dt

        n_springs = len(self.spring_i)
        n_curvature = max(n_rows - 2, 0)
        n_torsion = len(self._torsion_le) - 1

        lambdas_spring = np.zeros(n_springs)
        lambdas_curv = np.zeros(n_curvature)
        lambdas_torsion = np.zeros(n_torsion)
        lambdas_aero = np.zeros(n_rows)

        if not hasattr(self, '_constraint_count'):
            self._constraint_count = np.zeros(self.n_particles, dtype=np.float64)
            np.add.at(self._constraint_count, self.spring_i, 1.0)
            np.add.at(self._constraint_count, self.spring_j, 1.0)
            self._constraint_count = np.maximum(self._constraint_count, 1.0)

        # Aero constraint stiffness: soft relative to structural
        ke_aero = Q * Le * aero_compliance_factor
        alpha_aero = 1.0 / (max(ke_aero, 1e-6) * dt * dt)

        for _ in range(n_iterations):
            # Structural constraints
            delta = np.zeros_like(self.pos)
            self._solve_springs(dt, lambdas_spring, delta)
            self.pos += delta / self._constraint_count[:, None]

            self._solve_xpbd_curvature(dt, lambdas_curv)
            self._solve_xpbd_torsion(dt, lambdas_torsion)

            # Aero constraint: soft push toward aero target per row
            for r, rv in enumerate(rows):
                free = rv[~self.fixed_mask[rv]]
                if len(free) == 0:
                    continue
                nf = len(free)

                # Current row state
                dz = self.pos[free, 2] - self._pos_ref[free, 2]
                w_r = np.mean(dz)
                w_dot_r = (np.mean(self.pos[free, 2]) -
                           np.mean(pos_old[free, 2])) / dt

                x_from_ea = self._pos_ref[free, 0] - ea_chord_frac * chord
                sum_x2 = np.sum(x_from_ea**2)
                theta = (np.sum((dz - w_r) * x_from_ea) / sum_x2
                         if sum_x2 > 1e-12 else 0.0)

                # Aero force per vertex
                alpha_eff = theta - w_dot_r / max(V, 1.0)
                lift_ps = Q * alpha_eff
                shear = lift_ps * Le / nf
                moment_dist = (lift_ps * e_ac * Le * x_from_ea / sum_x2
                               if sum_x2 > 1e-12 else 0.0)
                Fz = shear + moment_dist

                # Desired position shift from aero
                dz_target = Fz * self.inv_mass[free] * dt * dt
                dz_target_mean = np.mean(dz_target)

                # Constraint violation
                C = dz_target_mean
                if abs(C) < 1e-18:
                    continue

                # Denominator: sum of (inv_mass * grad^2)
                denom = np.sum(self.inv_mass[free]) / (nf * nf)

                dlambda = -(C + alpha_aero * lambdas[r]) / (denom + alpha_aero)
                lambdas[r] += dlambda

                # Apply correction
                self.pos[free, 2] += self.inv_mass[free] * dlambda / nf

        # Derive velocity
        self.vel = (self.pos - pos_old) / dt

        if self._xpbd_beam_damping > 0:
            self.vel *= (1.0 - self._xpbd_beam_damping)

        self.vel[self.fixed_mask] = 0.0

    def step_xpbd_beam_coupled(self, aero_force_fn, dt, n_iterations=15,
                                 aero_damping_coeff=None):
        """Semi-implicit aero-structural coupling for XPBD beam.

        Applies aero force explicitly but treats aerodynamic damping
        implicitly to prevent the phase-shift instability that causes
        spurious divergence below flutter speed.

        The aero force F = q*c*CLa*(theta - w_dot/V) has two parts:
          - Position-dependent: q*c*CLa*theta (stiffness, applied explicitly)
          - Velocity-dependent: -q*c*CLa/V * w_dot (damping, applied implicitly)

        The implicit damping prevents the positive feedback where constraint
        overshoot creates wrong-sign velocity that excites instead of damps.

        Parameters
        ----------
        aero_force_fn : callable(mesh) -> (N, 3) array
            Returns total aero force (includes velocity-dependent part).
        dt : float
            Timestep.
        n_iterations : int
            XPBD iterations.
        aero_damping_coeff : (N,) array or None
            Per-vertex aerodynamic damping c_aero = q*c*CLa/V * Le/n_free.
            If None, estimated from the aero force magnitude.
        """
        if not getattr(self, '_xpbd_beam_configured', False):
            raise RuntimeError("Call setup_xpbd_beam() first")

        self._integrate_extra_springs()
        pos_old = self.pos.copy()

        # 1. Compute aero force from current state
        F_aero = aero_force_fn(self)

        # 2. Semi-implicit velocity update:
        #    vel_new = vel_old + F_aero * inv_mass * dt
        #    with implicit aero damping: vel_new /= (1 + c_aero * inv_mass * dt)
        #    where c_aero absorbs the velocity-dependent part of F_aero.
        vel_pred = self.vel + F_aero * self.inv_mass[:, None] * dt

        if aero_damping_coeff is not None:
            # User-provided per-vertex damping coefficients
            damp_factor = 1.0 / (1.0 + aero_damping_coeff * self.inv_mass * dt)
            vel_pred *= damp_factor[:, None]
        else:
            # Estimate damping from force/velocity ratio
            vel_z = np.abs(self.vel[:, 2])
            f_z = np.abs(F_aero[:, 2])
            # c_aero ≈ F_z / max(|v_z|, eps) for vertices with nonzero force
            has_force = f_z > 1e-12
            c_aero_est = np.zeros(self.n_particles)
            c_aero_est[has_force] = f_z[has_force] / np.maximum(vel_z[has_force], 0.01)
            damp_factor = 1.0 / (1.0 + c_aero_est * self.inv_mass * dt)
            vel_pred *= damp_factor[:, None]

        self.vel = vel_pred
        self.pos = pos_old + self.vel * dt

        # 3. XPBD structural constraint projection
        n_springs = len(self.spring_i)
        n_rows = len(self._rows)
        n_curvature = max(n_rows - 2, 0)
        n_torsion = len(self._torsion_le) - 1

        lambdas_spring = np.zeros(n_springs)
        lambdas_curv = np.zeros(n_curvature)
        lambdas_torsion = np.zeros(n_torsion)

        if not hasattr(self, '_constraint_count'):
            self._constraint_count = np.zeros(self.n_particles, dtype=np.float64)
            np.add.at(self._constraint_count, self.spring_i, 1.0)
            np.add.at(self._constraint_count, self.spring_j, 1.0)
            self._constraint_count = np.maximum(self._constraint_count, 1.0)

        for _ in range(n_iterations):
            delta = np.zeros_like(self.pos)
            self._solve_springs(dt, lambdas_spring, delta)
            self.pos += delta / self._constraint_count[:, None]

            self._solve_xpbd_curvature(dt, lambdas_curv)
            self._solve_xpbd_torsion(dt, lambdas_torsion)

        # 4. Derive velocity from position change
        self.vel = (self.pos - pos_old) / dt

        # 5. Structural damping
        if self._xpbd_beam_damping > 0:
            self.vel *= (1.0 - self._xpbd_beam_damping)

        self.vel[self.fixed_mask] = 0.0

    def step_xpbd_aeroelastic(self, V, rho=1.225, CL_alpha=2*np.pi,
                               chord=None, ea_chord_frac=None, dt=0.001,
                               n_iterations=15):
        """XPBD aeroelastic step: velocity-only aero coupling.

        XPBD constraint projection overshoots position corrections, creating
        wrong-sign derived velocities. Applying aero as position corrections
        feeds back through constraint projection → divergence.

        Solution: apply aero forces ONLY as velocity corrections. The next
        timestep's prediction uses these velocities, naturally incorporating
        the aero coupling without any constraint interaction.

          1. XPBD structural step (no aero) → positions, derived velocities
          2. Compute row DOFs (w, theta) and row-averaged velocities
          3. Apply aero stiffness as velocity impulse: vel += F*inv_mass*dt
          4. Apply aero damping implicitly: vel /= (1 + c*inv_mass*dt)
        """
        if not getattr(self, '_xpbd_beam_configured', False):
            raise RuntimeError("Call setup_xpbd_beam() first")

        if chord is None:
            chord = getattr(self, '_strip_width', 1.0)
        if ea_chord_frac is None:
            ea_chord_frac = getattr(self, '_xpbd_beam_ea_frac', 0.33)

        e_ac = (ea_chord_frac - 0.25) * chord
        q = 0.5 * rho * V * V
        Q = q * chord * CL_alpha

        rows = self._rows
        n_rows = len(rows)
        Le = self._row_spacing

        # 1. Pure structural XPBD step
        F_zero = np.zeros_like(self.pos)
        self.step_xpbd_beam(F_zero, dt, n_iterations)

        # 2. Extract row DOFs and velocities
        w_dof = np.zeros(n_rows)
        theta_dof = np.zeros(n_rows)
        w_dot = np.zeros(n_rows)
        for r, rv in enumerate(rows):
            free = rv[~self.fixed_mask[rv]]
            if len(free) == 0:
                continue
            dz = self.pos[free, 2] - self._pos_ref[free, 2]
            w_dof[r] = np.mean(dz)
            w_dot[r] = np.mean(self.vel[free, 2])
            x_from_ea = self._pos_ref[free, 0] - ea_chord_frac * chord
            sum_x2 = np.sum(x_from_ea**2)
            if sum_x2 > 1e-12:
                theta_dof[r] = np.sum((dz - w_dof[r]) * x_from_ea) / sum_x2

        # 3. Aero force per row
        alpha_eff = theta_dof - w_dot / max(V, 1.0)
        lift_per_span = Q * alpha_eff

        # 4. Apply as velocity corrections only
        c_aero_w = Q / max(V, 1.0) * Le

        for r, rv in enumerate(rows):
            free = rv[~self.fixed_mask[rv]]
            if len(free) == 0:
                continue
            nf = len(free)
            x_from_ea = self._pos_ref[free, 0] - ea_chord_frac * chord
            sum_x2 = np.sum(x_from_ea**2)

            shear = lift_per_span[r] * Le
            moment = lift_per_span[r] * e_ac * Le

            Fz = shear / nf
            if sum_x2 > 1e-12:
                Fz = Fz + moment * x_from_ea / sum_x2

            # Velocity impulse from aero
            self.vel[free, 2] += Fz * self.inv_mass[free] * dt

            # Implicit aero damping
            c_per_vert = c_aero_w / nf
            damp_factor = 1.0 / (1.0 + c_per_vert * self.inv_mass[free] * dt)
            self.vel[free, 2] *= damp_factor

        self.vel[self.fixed_mask] = 0.0

    def _solve_xpbd_curvature(self, dt, lambdas):
        """Row-based curvature bending constraint (replaces dihedral angle).

        For each triple of adjacent rows (r-1, r, r+1):
          w_r = mean z-displacement of row r vertices
          C_r = w_{r-1} - 2*w_r + w_{r+1}  (second finite difference = curvature * Le²)

        This is linear in z-coordinates, with clear stiffness mapping ke = EI / Le.
        Distributes corrections equally to all vertices in each row.
        """
        rows = self._rows
        n_rows = len(rows)
        n_curv = n_rows - 2
        if n_curv <= 0 or not hasattr(self, '_row_bend_ke') or self._row_bend_ke < 1e-12:
            return

        # Row-averaged z displacements (from reference)
        w = np.zeros(n_rows)
        for r, rv in enumerate(rows):
            w[r] = np.mean(self.pos[rv, 2] - self._pos_ref[rv, 2])

        # Curvature constraint: C_r = w[r-1] - 2*w[r] + w[r+1]
        C = w[:-2] - 2.0 * w[1:-1] + w[2:]  # (n_curv,)

        # Gradient: each vertex in row r contributes to C_r via w_r
        # ∂C_r/∂z[v_in_row_{r-1}] = 1/n_v_{r-1}
        # ∂C_r/∂z[v_in_row_{r}]   = -2/n_v_{r}
        # ∂C_r/∂z[v_in_row_{r+1}] = 1/n_v_{r+1}
        # For the XPBD denominator, sum over all affected vertices:
        # denom_r = Σ w_i |∂C/∂z_i|² for all vertices in rows r-1,r,r+1

        n_v = np.array([len(rv) for rv in rows], dtype=np.float64)

        # Inverse mass per row (average)
        inv_mass_per_row = np.zeros(n_rows)
        for r, rv in enumerate(rows):
            inv_mass_per_row[r] = np.sum(self.inv_mass[rv])

        # |∂C/∂z|² for each row's contribution:
        # Row r-1: (1/n_v[r-1])² * n_v[r-1] = 1/n_v[r-1]  ... summed over all verts in row
        # Actually: each vertex v in row r-1 has ∂C/∂z_v = 1/n_v[r-1]
        # Σ_v w_v |∂C/∂z_v|² = Σ_v (1/m_v) * (1/n_v[r-1])² = (1/n_v[r-1])² * inv_mass_row[r-1]
        denom = ((1.0 / n_v[:-2])**2 * inv_mass_per_row[:-2] +
                 (2.0 / n_v[1:-1])**2 * inv_mass_per_row[1:-1] +
                 (1.0 / n_v[2:])**2 * inv_mass_per_row[2:])

        # Compliance
        alpha = 1.0 / (self._row_bend_ke * dt * dt)

        # Damping
        gamma = self._row_bend_kd / (self._row_bend_ke * dt) if self._row_bend_ke > 0 else 0.0

        # Velocity contribution for damping
        v_w = np.zeros(n_rows)
        for r, rv in enumerate(rows):
            v_w[r] = np.mean(self.vel[rv, 2])
        dC_dt = v_w[:-2] - 2.0 * v_w[1:-1] + v_w[2:]

        # XPBD correction
        dlambda = -(C + alpha * lambdas + gamma * dC_dt * dt) / ((1.0 + gamma) * denom + alpha)
        lambdas += dlambda

        # Apply position corrections to z-coordinates of each row's vertices
        # Δz_v = inv_mass[v] * dlambda * (∂C/∂z_v) = inv_mass[v] * dlambda * coeff/n_v[r]
        for r in range(n_curv):
            rv0 = rows[r]
            rv1 = rows[r + 1]
            rv2 = rows[r + 2]

            dz0 = self.inv_mass[rv0] * dlambda[r] * (1.0 / n_v[r])
            dz1 = self.inv_mass[rv1] * dlambda[r] * (-2.0 / n_v[r + 1])
            dz2 = self.inv_mass[rv2] * dlambda[r] * (1.0 / n_v[r + 2])

            self.pos[rv0, 2] += dz0
            self.pos[rv1, 2] += dz1
            self.pos[rv2, 2] += dz2

    def _solve_xpbd_dihedral(self, dt, lambdas):
        """XPBD dihedral angle bending constraint projection.

        For each internal edge (shared by 2 triangles), computes the dihedral
        angle and applies position corrections to satisfy the constraint.

        Constraint: C = θ - θ_rest
        Gradient follows Newton XPBD bending_constraint formulation.
        """
        # 4 vertices per bending edge
        x1 = self.pos[self.bend_i]  # opposite vertex 1
        x2 = self.pos[self.bend_j]  # opposite vertex 2
        x3 = self.pos[self.bend_k]  # edge endpoint a
        x4 = self.pos[self.bend_l]  # edge endpoint b

        # Edge vector and face normals
        e = x4 - x3
        n1 = np.cross(x3 - x1, x4 - x1)
        n2 = np.cross(x4 - x2, x3 - x2)

        n1_sq = np.sum(n1 * n1, axis=1)
        n2_sq = np.sum(n2 * n2, axis=1)
        e_sq = np.sum(e * e, axis=1)

        # Avoid degenerate cases
        valid = (n1_sq > 1e-20) & (n2_sq > 1e-20) & (e_sq > 1e-20)
        if not np.any(valid):
            return

        n1_len = np.sqrt(n1_sq)
        n2_len = np.sqrt(n2_sq)
        e_len = np.sqrt(e_sq)

        n1_hat = n1 / n1_len[:, None]
        n2_hat = n2 / n2_len[:, None]
        e_hat = e / e_len[:, None]

        # Dihedral angle
        cos_theta = np.sum(n1_hat * n2_hat, axis=1)
        sin_theta = np.sum(np.cross(n1_hat, n2_hat) * e_hat, axis=1)
        theta = np.arctan2(sin_theta, cos_theta)

        # Constraint violation
        C = theta - self.bend_rest_angle

        # Gradients (Newton XPBD formulation)
        # grad_x1 = -n1_hat * e_len / n1_len  (scaled by edge length / normal length)
        # grad_x2 = -n2_hat * e_len / n2_len
        # grad_x3, grad_x4: more complex, involve e_hat projection

        # Simplified gradients (works well in practice):
        # Following Bridson/Müller formulation for dihedral angle gradient
        d1 = e_len / n1_len
        d2 = e_len / n2_len

        grad_x1 = -n1_hat * d1[:, None]
        grad_x2 = -n2_hat * d2[:, None]

        # grad_x3 = n1_hat * d1 * dot(n1_hat, e_hat) - n2_hat * d2 * dot(n2_hat, e_hat) + (e_hat * (d1 - d2))
        # Simplified: use the standard 4-vertex gradient
        n1_dot_e = np.sum(n1_hat * e_hat, axis=1)
        n2_dot_e = np.sum(n2_hat * e_hat, axis=1)

        grad_x3 = (n1_hat * (n1_dot_e[:, None] * d1[:, None])
                   + n2_hat * (n2_dot_e[:, None] * d2[:, None])
                   - e_hat * ((d1 - d2)[:, None]))

        grad_x4 = (-n1_hat * (n1_dot_e[:, None] * d1[:, None])
                   - n2_hat * (n2_dot_e[:, None] * d2[:, None])
                   + e_hat * ((d1 - d2)[:, None]))

        # Actually, let me use the standard formulation more carefully.
        # From "A Survey on Position-Based Dynamics" (Bender et al. 2017):
        # For dihedral angle constraint with vertices p1,p2,p3,p4 (p3-p4 is shared edge):
        # ∇_{p1} θ = -e_len / |n1| * n1_hat
        # ∇_{p2} θ = -e_len / |n2| * n2_hat
        # ∇_{p3} θ = (n1·e)/|n1| * ∇_{p1}θ - (n2·e)/|n2| * ∇_{p2}θ
        # Wait, this needs the correct sign.

        # Let me use the simpler Newton formulation directly:
        # grad magnitude squared per vertex:
        w1 = self.inv_mass[self.bend_i]
        w2 = self.inv_mass[self.bend_j]
        w3 = self.inv_mass[self.bend_k]
        w4 = self.inv_mass[self.bend_l]

        # |grad|² per vertex (simplified: use n_hat * e_len / n_len squared)
        g1_sq = np.sum(grad_x1 * grad_x1, axis=1)
        g2_sq = np.sum(grad_x2 * grad_x2, axis=1)
        g3_sq = np.sum(grad_x3 * grad_x3, axis=1)
        g4_sq = np.sum(grad_x4 * grad_x4, axis=1)

        denom = w1 * g1_sq + w2 * g2_sq + w3 * g3_sq + w4 * g4_sq

        # Compliance
        alpha = 1.0 / (self._dihedral_ke * dt * dt)

        # XPBD correction
        dlambda = -(C + alpha * lambdas) / (denom + alpha)
        dlambda *= valid  # zero out degenerate edges

        lambdas += dlambda

        # Position corrections
        corr1 = (w1 * dlambda)[:, None] * grad_x1
        corr2 = (w2 * dlambda)[:, None] * grad_x2
        corr3 = (w3 * dlambda)[:, None] * grad_x3
        corr4 = (w4 * dlambda)[:, None] * grad_x4

        np.add.at(self.pos, self.bend_i, corr1)
        np.add.at(self.pos, self.bend_j, corr2)
        np.add.at(self.pos, self.bend_k, corr3)
        np.add.at(self.pos, self.bend_l, corr4)

    def _solve_xpbd_torsion(self, dt, lambdas):
        """XPBD torsion constraint: relative twist between adjacent rows.

        For each pair of adjacent rows, the twist angle is:
          θ_j = (z[TE_j] - z[LE_j]) / chord

        Constraint: C = θ_{j+1} - θ_j (rest = 0 for untwisted)
        Linear constraint in z-coordinates, simple gradient.
        """
        le = self._torsion_le
        te = self._torsion_te
        chord = self._torsion_chord
        n_torsion = len(le) - 1

        if n_torsion <= 0 or chord < 1e-12 or self._torsion_ke < 1e-12:
            return

        # Twist angle at each row (small angle approximation)
        theta = (self.pos[te, 2] - self.pos[le, 2]) / chord

        # Constraint: relative twist between adjacent rows
        C = theta[1:] - theta[:-1]  # (n_torsion,)

        # Gradient: ∂C_j/∂z for 4 vertices (LE_j, TE_j, LE_{j+1}, TE_{j+1})
        # C_j = (z[TE_{j+1}] - z[LE_{j+1}]) / chord - (z[TE_j] - z[LE_j]) / chord
        # ∂C_j/∂z[LE_j] = 1/chord
        # ∂C_j/∂z[TE_j] = -1/chord
        # ∂C_j/∂z[LE_{j+1}] = -1/chord
        # ∂C_j/∂z[TE_{j+1}] = 1/chord
        grad_mag_sq = 4.0 / (chord * chord)  # same for all 4 vertices

        # Inverse masses for the 4 vertices
        w_le = self.inv_mass[le[:-1]]   # LE_j
        w_te = self.inv_mass[te[:-1]]   # TE_j
        w_le1 = self.inv_mass[le[1:]]   # LE_{j+1}
        w_te1 = self.inv_mass[te[1:]]   # TE_{j+1}

        denom = (w_le + w_te + w_le1 + w_te1) * grad_mag_sq

        # Compliance
        alpha = 1.0 / (self._torsion_ke * dt * dt)

        # XPBD correction
        dlambda = -(C + alpha * lambdas) / (denom + alpha)

        lambdas += dlambda

        # Position corrections (z only)
        dz_le = w_le * dlambda / chord
        dz_te = -w_te * dlambda / chord
        dz_le1 = -w_le1 * dlambda / chord
        dz_te1 = w_te1 * dlambda / chord

        np.add.at(self.pos[:, 2], le[:-1], dz_le)
        np.add.at(self.pos[:, 2], te[:-1], dz_te)
        np.add.at(self.pos[:, 2], le[1:], dz_le1)
        np.add.at(self.pos[:, 2], te[1:], dz_te1)

    def get_beam_dofs_xpbd(self):
        """Extract beam DOFs from particle positions.

        Returns w (heave) and theta (twist) per row.
        """
        if not hasattr(self, '_rows'):
            self._build_rows()
        rows = self._rows
        n_rows = len(rows)
        w = np.zeros(n_rows)
        theta = np.zeros(n_rows)

        for r, row_verts in enumerate(rows):
            dz = self.pos[row_verts, 2] - self._pos_ref[row_verts, 2]
            w[r] = np.mean(dz)
            if hasattr(self, '_xpbd_beam_ea_frac'):
                x_ea = self._xpbd_beam_ea_frac * self._strip_width
            else:
                x_ea = 0.33 * self._strip_width
            x_from_ea = self._pos_ref[row_verts, 0] - x_ea
            sum_x2 = np.sum(x_from_ea**2)
            if sum_x2 > 1e-12:
                theta[r] = np.sum((dz - w[r]) * x_from_ea) / sum_x2

        return w, theta

    def step_xpbd_beam_forces(self, F_ext, dt, EI, GJ, n_iterations=15,
                               damping_ratio=0.005):
        """XPBD stretch + 1D force-based bending + torsion.

        No FEM matrices. Bending and torsion forces computed from
        row-averaged displacements using finite differences.
        CG coupling from asymmetric mass (set up via setup_xpbd_beam).

        Parameters
        ----------
        F_ext : (N, 3) external forces
        dt : timestep
        EI : bending stiffness [N·m²]
        GJ : torsional stiffness [N·m²]
        n_iterations : XPBD iterations for stretch
        damping_ratio : velocity damping factor
        """
        self._integrate_extra_springs()
        pos_old = self.pos.copy()

        # 1. Compute beam bending + torsion forces from current positions
        F_beam = self._compute_beam_row_forces(EI, GJ)

        # 2. Total external + beam forces
        F_total = F_ext + F_beam

        # 3. Predict positions
        acc = F_total * self.inv_mass[:, None]
        self.pos = pos_old + self.vel * dt + acc * (dt * dt)

        # 4. XPBD stretch constraint projection
        if not hasattr(self, '_constraint_count'):
            self._constraint_count = np.zeros(self.n_particles, dtype=np.float64)
            np.add.at(self._constraint_count, self.spring_i, 1.0)
            np.add.at(self._constraint_count, self.spring_j, 1.0)
            self._constraint_count = np.maximum(self._constraint_count, 1.0)

        lambdas_spring = np.zeros(len(self.spring_i))

        for _ in range(n_iterations):
            delta = np.zeros_like(self.pos)
            self._solve_springs(dt, lambdas_spring, delta)
            self.pos += delta / self._constraint_count[:, None]

        # 5. Derive velocity
        self.vel = (self.pos - pos_old) / dt

        # 6. Damping
        if damping_ratio > 0:
            self.vel *= (1.0 - damping_ratio)

        # 7. Fixed particles
        self.vel[self.fixed_mask] = 0.0

    def _compute_beam_row_forces(self, EI, GJ):
        """Compute bending + torsion forces from row-averaged DOFs.

        Bending uses curvature constraint: C = w_{r-1} - 2w_r + w_{r+1}
        with stiffness ke_bend calibrated to match Euler-Bernoulli frequency.
        The force is the energy gradient: F = -ke_bend * C * dC/dw.

        Torsion uses relative twist: C = theta_{r+1} - theta_r
        with stiffness ke_tors calibrated to match Saint-Venant frequency.

        Returns (N, 3) force array.
        """
        if not hasattr(self, '_rows'):
            self._build_rows()
        rows = self._rows
        n_rows = len(rows)
        Le = self._row_spacing
        free_mask = ~self.fixed_mask

        # Stiffness from continuous beam theory:
        # Bending: E = EI/(2*Le³) * Σ C_r²  → ke_bend = EI/Le³
        # Torsion: E = GJ/(2*Le) * Σ (Δθ)²  → ke_tors = GJ/Le
        ke_bend = EI / Le**3
        ke_tors = GJ / Le

        # Row-averaged displacements
        w = np.zeros(n_rows)
        theta = np.zeros(n_rows)
        for r, row_verts in enumerate(rows):
            free_in_row = row_verts[free_mask[row_verts]]
            if len(free_in_row) > 0:
                w[r] = np.mean(self.pos[free_in_row, 2] - self._pos_ref[free_in_row, 2])
                if hasattr(self, '_torsion_le') and len(self._torsion_le) > r:
                    dz_le = self.pos[self._torsion_le[r], 2] - self._pos_ref[self._torsion_le[r], 2]
                    dz_te = self.pos[self._torsion_te[r], 2] - self._pos_ref[self._torsion_te[r], 2]
                    theta[r] = (dz_te - dz_le) / self._strip_width

        # Bending force from curvature energy: E = 0.5 * ke_bend * sum(C_r^2)
        # F_r = -dE/dw_r = -ke_bend * sum_s C_s * dC_s/dw_r
        # For C_r = w[r-1] - 2w[r] + w[r+1], the contribution to row r is:
        # From C_{r-1}: dC_{r-1}/dw_r = -2 → force contribution: -ke_bend * C_{r-1} * (-2)
        # From C_r:     dC_r/dw_r = +1 → but wait, this is for w[r-1]
        # Actually: C_s = w[s-1] - 2w[s] + w[s+1]
        # dC_s/dw_r = 1 if r=s-1, -2 if r=s, 1 if r=s+1, 0 otherwise
        F_w = np.zeros(n_rows)
        for s in range(1, n_rows - 1):
            C_s = w[s-1] - 2*w[s] + w[s+1]
            F_w[s-1] -= ke_bend * C_s * 1.0
            F_w[s]   -= ke_bend * C_s * (-2.0)
            F_w[s+1] -= ke_bend * C_s * 1.0

        # Torsion force from twist energy: E = 0.5 * ke_tors * sum((theta[r+1]-theta[r])^2)
        # F_theta_r = -dE/dtheta_r
        F_th = np.zeros(n_rows)
        for s in range(n_rows - 1):
            C_t = theta[s+1] - theta[s]
            F_th[s]   += ke_tors * C_t   # dC/dtheta_s = -1 → -ke*C*(-1) = +ke*C
            F_th[s+1] -= ke_tors * C_t   # dC/dtheta_{s+1} = +1 → -ke*C*(+1) = -ke*C

        # Map row forces to particle z-forces
        F = np.zeros_like(self.pos)
        if hasattr(self, '_xpbd_beam_ea_frac'):
            x_ea = self._xpbd_beam_ea_frac * self._strip_width
        else:
            x_ea = 0.33 * self._strip_width

        for r, row_verts in enumerate(rows):
            free_in_row = row_verts[free_mask[row_verts]]
            if len(free_in_row) == 0:
                continue
            n_free = len(free_in_row)

            # Bending: equal z-force per free vertex in row
            Fz_shear = F_w[r] / n_free

            # Torsion: z-force proportional to x-distance from EA
            x_from_ea = self._pos_ref[free_in_row, 0] - x_ea
            sum_x2 = np.sum(x_from_ea**2)
            if sum_x2 > 1e-12:
                Fz_torsion = F_th[r] * x_from_ea / sum_x2
            else:
                Fz_torsion = np.zeros(n_free)

            F[free_in_row, 2] += Fz_shear + Fz_torsion

        return F

    # ── RID Quasi-Explicit Solver (Lu & Hu 2025) ──────────────────────

    def step_rid_beam(self, F_ext, dt, EI, GJ, n_iterations=10, damping_ratio=0.005):
        """RID quasi-explicit solver: explicit bend/torsion + RID spring correction.

        Bending and torsion forces computed explicitly from row-averaged DOFs.
        Spring stretch corrected via force-direction decomposition with
        Hessian-based adaptive step size (Lu & Hu 2025).

        Key difference from XPBD: no Lagrange multiplier accumulation,
        forces recomputed from current positions each iteration.
        Velocity is a real state variable.
        """
        if not hasattr(self, '_rows'):
            self._build_rows()

        self._integrate_extra_springs()
        pos_old = self.pos.copy()

        # 1. Explicit forces: bending + torsion (from current state)
        F_beam = self._compute_beam_row_forces(EI, GJ)
        F_total = F_ext + F_beam

        # 2. Predict: inertia + explicit forces
        acc = F_total * self.inv_mass[:, None]
        self.pos = pos_old + self.vel * dt + acc * (dt * dt)

        # 3. RID spring correction (force-direction decomposition)
        if not hasattr(self, '_constraint_count'):
            self._constraint_count = np.zeros(self.n_particles, dtype=np.float64)
            np.add.at(self._constraint_count, self.spring_i, 1.0)
            np.add.at(self._constraint_count, self.spring_j, 1.0)
            self._constraint_count = np.maximum(self._constraint_count, 1.0)

        for _ in range(n_iterations):
            xi = self.pos[self.spring_i]
            xj = self.pos[self.spring_j]
            dx = xi - xj
            lengths = np.linalg.norm(dx, axis=1)
            safe_len = np.maximum(lengths, 1e-12)
            n = dx / safe_len[:, None]

            stretch = lengths - self.spring_rest
            wi = self.inv_mass[self.spring_i]
            wj = self.inv_mass[self.spring_j]
            w_sum = wi + wj

            # RID quasi-explicit: ||f|| / (||H|| * w_sum + 1/dt²)
            # For springs: ||f|| = ke*|stretch|, ||H|| = ke
            f_mag = self.spring_ke * np.abs(stretch)
            H_norm = self.spring_ke
            denom = H_norm * w_sum + 1.0 / (dt * dt)

            valid = (self.spring_ke > 0) & (w_sum > 0) & (np.abs(stretch) > 1e-15)
            s = np.where(valid, f_mag / denom, 0.0)

            # Correction toward rest length
            sign_s = np.sign(stretch)
            dxi = -(wi * s * sign_s)[:, None] * n
            dxj = (wj * s * sign_s)[:, None] * n

            delta = np.zeros_like(self.pos)
            np.add.at(delta, self.spring_i, dxi)
            np.add.at(delta, self.spring_j, dxj)

            self.pos += delta / self._constraint_count[:, None]

        # 4. Update velocity (real state variable)
        self.vel = (self.pos - pos_old) / dt

        # 5. Damping + fixed particles
        if damping_ratio > 0:
            self.vel *= (1.0 - damping_ratio)
        self.vel[self.fixed_mask] = 0.0

    # ── XPBD Solver ────────────────────────────────────────────────────

    def step(self, F_ext, dt, n_iterations=10):
        """Advance one timestep using XPBD.

        Standard XPBD (Macklin & Müller 2016):
          1. Predict positions: x_pred = x + v*dt + (F_ext/m)*dt²
          2. Constraint projection on x_pred (n_iterations rounds)
          3. Derive velocity: v = (x_new - x_old) / dt

        Parameters
        ----------
        F_ext : (N, 3) array
            External forces on each particle (aerodynamic, gravity, etc.).
        dt : float
            Timestep size.
        n_iterations : int
            Number of XPBD constraint projection iterations.
        """
        self._integrate_extra_springs()

        # 1. Predict positions (Verlet-style, no velocity accumulation)
        pos_old = self.pos.copy()
        acc = F_ext * self.inv_mass[:, None]
        self.pos = pos_old + self.vel * dt + acc * (dt * dt)

        # 2. Constraint projection iterations (Jacobi with averaging)
        n_springs = len(self.spring_i)

        # Precompute constraint count per vertex (for Jacobi averaging)
        if not hasattr(self, '_constraint_count'):
            self._constraint_count = np.zeros(self.n_particles, dtype=np.float64)
            np.add.at(self._constraint_count, self.spring_i, 1.0)
            np.add.at(self._constraint_count, self.spring_j, 1.0)
            self._constraint_count = np.maximum(self._constraint_count, 1.0)

        lambdas_spring = np.zeros(n_springs)

        for _ in range(n_iterations):
            delta = np.zeros_like(self.pos)
            self._solve_springs(dt, lambdas_spring, delta)
            # Average corrections by constraint count per vertex
            self.pos += delta / self._constraint_count[:, None]

        # 3. Apply bending forces (Laplacian-based, force mode)
        if self.bending_D > 0 or self.bending_damp > 0:
            F_bend = self._compute_bending_elastic_forces()
            F_bend += self._compute_bending_damping_forces()
            self.pos += F_bend * self.inv_mass[:, None] * (dt * dt)

        # 3. Derive velocity from position change
        self.vel = (self.pos - pos_old) / dt

    def _solve_springs(self, dt, lambdas, delta):
        """XPBD spring constraint projection.

        From Newton XPBD kernels.py: solve_springs.
        Constraint: C = |x_i - x_j| - L_rest = 0
        Correction: Δλ = -(C + α·λ + γ·∇C·Δx) / ((1+γ)·(w_i+w_j) + α)
        Writes corrections into delta array (not directly to pos).
        """
        xi = self.pos[self.spring_i]
        xj = self.pos[self.spring_j]
        vi = self.vel[self.spring_i]
        vj = self.vel[self.spring_j]

        xij = xi - xj
        lengths = np.linalg.norm(xij, axis=1)
        safe_len = np.maximum(lengths, 1e-12)
        n = xij / safe_len[:, None]

        # Constraint violation
        C = lengths - self.spring_rest

        # Compliance and damping
        alpha = 1.0 / (self.spring_ke * dt * dt)
        gamma = self.spring_kd / (self.spring_ke * dt)

        # Inverse masses
        wi = self.inv_mass[self.spring_i]
        wj = self.inv_mass[self.spring_j]
        denom = wi + wj  # |∇C|² = 1 for distance constraint

        # Damping term: ∇C · (v_i - v_j) · dt
        grad_dot_v = dt * np.sum(n * (vi - vj), axis=1)

        # Lagrange multiplier update
        dlambda = -(C + alpha * lambdas + gamma * grad_dot_v) / ((1 + gamma) * denom + alpha)

        # Skip zero-stiffness / zero-mass constraints
        valid = (self.spring_ke > 0) & (denom > 0)
        dlambda *= valid

        lambdas += dlambda

        # Position corrections → accumulate into delta
        dxi = (wi * dlambda)[:, None] * n
        dxj = -(wj * dlambda)[:, None] * n

        np.add.at(delta, self.spring_i, dxi)
        np.add.at(delta, self.spring_j, dxj)

    # ── UVLM Coupling Interface ────────────────────────────────────────

    def get_surface_mesh(self):
        """Return deformed surface mesh for UVLM.

        Returns
        -------
        pos : (N, 3) array
            Current vertex positions.
        tri_indices : (T, 3) array
            Triangle connectivity.
        """
        return self.pos.copy(), self.tri_indices

    def distribute_force_to_vertices(self, tri_forces):
        """Distribute per-triangle forces to vertices (1/3 each).

        Parameters
        ----------
        tri_forces : (T, 3) array
            Force on each triangle (e.g., from UVLM aerodynamics).

        Returns
        -------
        F_vert : (N, 3) array
            Force on each vertex.
        """
        F = np.zeros((self.n_particles, 3), dtype=np.float64)
        for t_idx, (i, j, k) in enumerate(self.tri_indices):
            f3 = tri_forces[t_idx] / 3.0
            F[i] += f3
            F[j] += f3
            F[k] += f3
        return F

    def compute_gravity_forces(self):
        """Compute gravitational force on each particle."""
        return self.particle_mass[:, None] * self.gravity[None, :]

    # ── Actuator Interface ─────────────────────────────────────────────

    def set_actuator_rest_length(self, spring_index, rest_length):
        """Set the rest length of a spring (for actuation).

        The spring_index refers to the index in the merged spring arrays
        (mesh edges first, then extra springs).
        """
        self._integrate_extra_springs()
        self.spring_rest[spring_index] = rest_length

    # ── Query ──────────────────────────────────────────────────────────

    def get_triangle_centers(self):
        """Return (T, 3) array of triangle centroid positions."""
        v0 = self.pos[self.tri_indices[:, 0]]
        v1 = self.pos[self.tri_indices[:, 1]]
        v2 = self.pos[self.tri_indices[:, 2]]
        return (v0 + v1 + v2) / 3.0

    def get_triangle_areas(self):
        """Return (T,) array of current triangle areas."""
        v0 = self.pos[self.tri_indices[:, 0]]
        v1 = self.pos[self.tri_indices[:, 1]]
        v2 = self.pos[self.tri_indices[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    def get_triangle_normals(self):
        """Return (T, 3) array of unit triangle normals."""
        v0 = self.pos[self.tri_indices[:, 0]]
        v1 = self.pos[self.tri_indices[:, 1]]
        v2 = self.pos[self.tri_indices[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(lengths, 1e-12)

    def record_state(self):
        """Save current positions to history."""
        self.pos_history.append(self.pos.copy())

    def reset(self):
        """Reset velocities and history (positions keep current values)."""
        self.vel[:] = 0.0
        self.pos_history = []


# ── Wing Mesh Generator ────────────────────────────────────────────────

def create_wing_mesh(chord, span, n_chord, n_span,
                     chord_distribution=None, sweep_angle=0.0,
                     twist_distribution=None, dihedral_angle=0.0):
    """Generate a rectangular wing triangle mesh.

    Parameters
    ----------
    chord : float
        Root chord length.
    span : float
        Semi-span length.
    n_chord : int
        Number of chordwise panels.
    n_span : int
        Number of spanwise panels.
    chord_distribution : callable or None
        f(y/span) -> chord at that span station. Default: constant.
    sweep_angle : float
        Sweep angle in radians. LE sweeps back.
    twist_distribution : callable or None
        f(y/span) -> twist angle in radians. Default: none.
    dihedral_angle : float
        Dihedral angle in radians.

    Returns
    -------
    vertices : (N, 3) array
        Mesh vertex positions. x=chordwise, y=spanwise, z=up.
    triangles : (T, 3) array
        Triangle vertex indices.
    """
    n_x = n_chord + 1
    n_y = n_span + 1

    vertices = np.zeros((n_x * n_y, 3), dtype=np.float64)

    for j in range(n_y):
        eta = j / n_span  # normalized span [0, 1]
        y = eta * span

        # Local chord
        if chord_distribution is not None:
            c = chord * chord_distribution(eta)
        else:
            c = chord

        # Sweep offset
        x_sweep = y * np.tan(sweep_angle)

        # Twist
        twist = 0.0
        if twist_distribution is not None:
            twist = twist_distribution(eta)

        # Dihedral
        z_dihedral = y * np.tan(dihedral_angle)

        for i in range(n_x):
            xi = i / n_chord  # normalized chord [0, 1]
            idx = j * n_x + i

            x = x_sweep + xi * c
            z = z_dihedral + xi * c * np.sin(twist) * 0  # simplified: no camber
            # Apply twist as rotation about leading edge
            x_local = xi * c
            z_local = x_local * np.sin(twist)
            x_local_twisted = x_local * np.cos(twist)

            vertices[idx, 0] = x_sweep + x_local_twisted
            vertices[idx, 1] = y
            vertices[idx, 2] = z_dihedral + z_local

    # Triangulate: split each quad into 2 triangles
    triangles = []
    for j in range(n_span):
        for i in range(n_chord):
            p00 = j * n_x + i
            p10 = j * n_x + (i + 1)
            p01 = (j + 1) * n_x + i
            p11 = (j + 1) * n_x + (i + 1)

            # Alternating diagonal for better quality
            if (i + j) % 2 == 0:
                triangles.append([p00, p10, p11])
                triangles.append([p00, p11, p01])
            else:
                triangles.append([p00, p10, p01])
                triangles.append([p10, p11, p01])

    return vertices, np.array(triangles, dtype=np.int32)
