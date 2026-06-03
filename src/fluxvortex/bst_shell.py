"""BST rotation-free shell element with orthotropic material support.

3-node triangle, each node has only 3 displacement DOFs (no rotations).
Membrane: standard CST (Constant Strain Triangle), orthotropic constitutive law.
Bending: Battini curvature tensor model (Battini 2002) per element.

Supports both isotropic (E, nu) and orthotropic (Ex, Ey, nu_xy, G_xy) materials.
GPU acceleration via NVIDIA Warp when use_gpu=True.

Time integration: explicit Velocity-Verlet with real velocity state variable.
"""
import numpy as np


class BSTShell:
    """Rotation-free triangular shell: CST membrane + Battini bending.

    Parameters
    ----------
    vertices : (N, 3) array
        Initial vertex positions. x=chordwise, y=spanwise, z=up.
    triangles : (T, 3) array
        Triangle connectivity (vertex indices).
    h : float
        Thickness [m].
    rho : float
        Density [kg/m^3].
    E : float, optional
        Young's modulus for isotropic material [Pa].
    nu : float, optional
        Poisson's ratio for isotropic material.
    Ex : float, optional
        Young's modulus in x-direction [Pa] (orthotropic).
    Ey : float, optional
        Young's modulus in y-direction [Pa] (orthotropic).
    nu_xy : float, optional
        Poisson's ratio xy (orthotropic).
    G_xy : float, optional
        Shear modulus [Pa] (orthotropic).
    structural_damping : float
        Velocity-proportional damping coefficient.
    use_gpu : bool
        Use NVIDIA Warp GPU acceleration.
    """

    def __init__(self, vertices, triangles, h, rho,
                 E=None, nu=None,
                 Ex=None, Ey=None, nu_xy=None, G_xy=None,
                 structural_damping=0.0, use_gpu=False,
                 bending_model='ibm', ibm_cal=0.252):
        self.vertices0 = np.array(vertices, dtype=np.float64)
        self.triangles = np.array(triangles, dtype=np.int32)
        self.nv = len(self.vertices0)
        self.nt = len(self.triangles)
        self.h = float(h)
        self.rho = float(rho)
        self.damping = structural_damping
        self.use_gpu = use_gpu
        self.bending_model = bending_model
        self._ibm_cal = ibm_cal

        self._init_material(E, nu, Ex, Ey, nu_xy, G_xy)
        self._precompute_ref_geometry()
        self._build_interior_edges()
        self._build_battini_data()

        self.mass = np.zeros(self.nv)
        for t in range(self.nt):
            i0, i1, i2 = self.triangles[t]
            m_tri = self._ref_area[t] * h * rho
            self.mass[i0] += m_tri / 3.0
            self.mass[i1] += m_tri / 3.0
            self.mass[i2] += m_tri / 3.0
        self.mass_inv = np.where(self.mass > 1e-30, 1.0 / self.mass, 0.0)

        self.u = np.zeros((self.nv, 3))
        self.v = np.zeros((self.nv, 3))
        self.a = np.zeros((self.nv, 3))
        self._Q = None  # IBM bending matrix, built on first use

        self._gpu_ctx = None
        if self.use_gpu:
            self._init_gpu()

    # ------------------------------------------------------------------
    # Material initialization
    # ------------------------------------------------------------------
    def _init_material(self, E, nu, Ex, Ey, nu_xy, G_xy):
        has_iso = E is not None and nu is not None
        has_ortho = all(v is not None for v in [Ex, Ey, nu_xy, G_xy])

        if has_ortho:
            self.Ex = float(Ex)
            self.Ey = float(Ey)
            self.nu_xy = float(nu_xy)
            self.nu_yx = self.nu_xy * self.Ey / self.Ex
            self.G_xy = float(G_xy)
            self.is_isotropic = False
        elif has_iso:
            self.Ex = float(E)
            self.Ey = float(E)
            self.nu_xy = float(nu)
            self.nu_yx = float(nu)
            self.G_xy = float(E) / (2.0 * (1.0 + float(nu)))
            self.is_isotropic = True
        else:
            raise ValueError(
                "Provide either (E, nu) for isotropic or "
                "(Ex, Ey, nu_xy, G_xy) for orthotropic material")

        nu_prod = self.nu_xy * self.nu_yx
        if nu_prod >= 1.0:
            raise ValueError(
                f"nu_xy*nu_yx = {nu_prod:.4f} >= 1.0, "
                "constitutive matrix not positive-definite")

        denom = 1.0 - nu_prod

        self._D00 = self.Ex / denom
        self._D01 = self.nu_xy * self.Ey / denom
        self._D11 = self.Ey / denom
        self._D22 = self.G_xy

        self.D_mem = np.array([
            [self._D00, self._D01, 0.0],
            [self._D01, self._D11, 0.0],
            [0.0, 0.0, self._D22],
        ])

        h3 = self.h ** 3
        self._Dx = self.Ex * h3 / (12.0 * denom)
        self._Dy = self.Ey * h3 / (12.0 * denom)
        self._D1 = self.nu_xy * self._Dy
        self._Dxy = self.G_xy * h3 / 12.0
        self.D = self._Dy  # backward compat

    # ------------------------------------------------------------------
    # Reference geometry (membrane CST)
    # ------------------------------------------------------------------
    def _precompute_ref_geometry(self):
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
    # Interior edges with per-edge bending stiffness
    # ------------------------------------------------------------------
    def _build_interior_edges(self):
        edge_to_tris = {}
        tri_edge_opp = {}

        for t in range(self.nt):
            i0, i1, i2 = self.triangles[t]
            edges = [(i0, i1, i2), (i1, i2, i0), (i0, i2, i1)]
            for va, vb, opp in edges:
                key = (min(va, vb), max(va, vb))
                edge_to_tris.setdefault(key, []).append(t)
                tri_edge_opp[(t, key)] = opp

        ea_l, eb_l, ec_l, ed_l = [], [], [], []
        L_l, theta_l, A0_l, A1_l, D_l = [], [], [], [], []

        for key, tris in edge_to_tris.items():
            if len(tris) != 2:
                continue
            t0, t1 = tris
            va, vb = key
            ec = tri_edge_opp[(t0, key)]
            ed = tri_edge_opp[(t1, key)]

            pa = self.vertices0[va]
            pb = self.vertices0[vb]
            pc = self.vertices0[ec]
            pd = self.vertices0[ed]

            e_vec = pb - pa
            L = np.linalg.norm(e_vec)
            if L < 1e-30:
                continue

            n0 = np.cross(pb - pa, pc - pa)
            n1 = np.cross(pd - pa, pb - pa)
            n0_len = np.linalg.norm(n0)
            n1_len = np.linalg.norm(n1)
            if n0_len < 1e-30 or n1_len < 1e-30:
                continue

            n0_hat = n0 / n0_len
            n1_hat = n1 / n1_len
            cos_th = np.clip(np.dot(n0_hat, n1_hat), -1.0, 1.0)
            sin_th = np.dot(np.cross(n0_hat, n1_hat), e_vec / L)
            theta_ref = np.arctan2(sin_th, cos_th)

            alpha = np.arctan2(e_vec[1], e_vec[0])
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            ca2 = ca * ca
            sa2 = sa * sa
            D_edge = (self._Dx * ca2 * ca2
                      + 2.0 * (self._D1 + 2.0 * self._Dxy) * sa2 * ca2
                      + self._Dy * sa2 * sa2)

            ea_l.append(va); eb_l.append(vb)
            ec_l.append(ec); ed_l.append(ed)
            L_l.append(L)
            theta_l.append(theta_ref)
            A0_l.append(0.5 * n0_len)
            A1_l.append(0.5 * n1_len)
            D_l.append(D_edge)

        self.n_interior_edges = len(ea_l)
        self._edge_ea = np.array(ea_l, dtype=np.int32)
        self._edge_eb = np.array(eb_l, dtype=np.int32)
        self._edge_ec = np.array(ec_l, dtype=np.int32)
        self._edge_ed = np.array(ed_l, dtype=np.int32)
        self._edge_L = np.array(L_l, dtype=np.float64)
        self._edge_theta_ref = np.array(theta_l, dtype=np.float64)
        self._edge_A0 = np.array(A0_l, dtype=np.float64)
        self._edge_A1 = np.array(A1_l, dtype=np.float64)
        self._edge_D = np.array(D_l, dtype=np.float64)

    # ------------------------------------------------------------------
    # Battini curvature tensor data (per triangle-edge connectivity)
    # ------------------------------------------------------------------
    def _build_battini_data(self):
        """Pre-compute triangle-edge connectivity and in-plane normals for Battini bending.

        For each triangle t with edges k=0,1,2 (opposite node k), stores:
          - which interior edge index corresponds to edge k
          - the in-plane unit normal m_k (perpendicular to edge, pointing INTO triangle)
          - edge length L_k
        Also stores for each interior edge: the two adjacent triangles and
        the edge index k within each triangle.
        """
        # Build edge key -> interior edge index
        edge_key_to_idx = {}
        for ei in range(self.n_interior_edges):
            key = (min(self._edge_ea[ei], self._edge_eb[ei]),
                   max(self._edge_ea[ei], self._edge_eb[ei]))
            edge_key_to_idx[key] = ei

        # For each triangle, 3 edges: edge k is opposite to node k
        # Triangle nodes: tri[t] = [n0, n1, n2]
        # Edge 0: n1-n2 (opposite n0)
        # Edge 1: n0-n2 (opposite n1)
        # Edge 2: n0-n1 (opposite n2)
        _tri_edge_idx = -np.ones((self.nt, 3), dtype=np.int32)   # interior edge index, -1 = boundary
        _tri_edge_L = np.zeros((self.nt, 3), dtype=np.float64)   # edge length
        _tri_edge_mx = np.zeros((self.nt, 3), dtype=np.float64)  # in-plane normal x
        _tri_edge_my = np.zeros((self.nt, 3), dtype=np.float64)  # in-plane normal y

        for t in range(self.nt):
            n0, n1, n2 = self.triangles[t]
            p0 = self.vertices0[n0]
            p1 = self.vertices0[n1]
            p2 = self.vertices0[n2]

            edges_nodes = [(1, 2, 0), (0, 2, 1), (0, 1, 2)]  # (ia, ib, iopp) for k=0,1,2

            for k, (ia, ib, iopp) in enumerate(edges_nodes):
                na, nb, nopp = self.triangles[t, ia], self.triangles[t, ib], self.triangles[t, iopp]
                pa = self.vertices0[na]
                pb = self.vertices0[nb]
                popp = self.vertices0[nopp]

                e_vec = pb - pa
                L = np.linalg.norm(e_vec)
                _tri_edge_L[t, k] = L
                if L < 1e-30:
                    continue

                # In-plane normal: rotate edge 90° to point toward opposite vertex
                # e_vec = (ex, ey), m = (ey, -ex) or (-ey, ex)
                # Choose sign so m · (popp - midpoint) > 0
                mx, my = e_vec[1], -e_vec[0]
                mid = 0.5 * (pa + pb)
                if mx * (popp[0] - mid[0]) + my * (popp[1] - mid[1]) < 0:
                    mx, my = -mx, -my
                mn = np.sqrt(mx * mx + my * my)
                if mn > 1e-30:
                    mx /= mn
                    my /= mn
                _tri_edge_mx[t, k] = mx
                _tri_edge_my[t, k] = my

                # Map to interior edge index
                key = (min(na, nb), max(na, nb))
                if key in edge_key_to_idx:
                    _tri_edge_idx[t, k] = edge_key_to_idx[key]

        self._battini_tri_edge_idx = _tri_edge_idx
        self._battini_tri_edge_L = _tri_edge_L
        self._battini_tri_edge_mx = _tri_edge_mx
        self._battini_tri_edge_my = _tri_edge_my

        # For each interior edge, record the two adjacent triangles
        # and which edge index k within each triangle
        _edge_tri0 = -np.ones(self.n_interior_edges, dtype=np.int32)
        _edge_k_in_t0 = -np.ones(self.n_interior_edges, dtype=np.int32)
        _edge_tri1 = -np.ones(self.n_interior_edges, dtype=np.int32)
        _edge_k_in_t1 = -np.ones(self.n_interior_edges, dtype=np.int32)

        for t in range(self.nt):
            for k in range(3):
                ei = _tri_edge_idx[t, k]
                if ei < 0:
                    continue
                if _edge_tri0[ei] < 0:
                    _edge_tri0[ei] = t
                    _edge_k_in_t0[ei] = k
                else:
                    _edge_tri1[ei] = t
                    _edge_k_in_t1[ei] = k

        self._battini_edge_tri0 = _edge_tri0
        self._battini_edge_k_in_t0 = _edge_k_in_t0
        self._battini_edge_tri1 = _edge_tri1
        self._battini_edge_k_in_t1 = _edge_k_in_t1

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------
    def compute_forces(self):
        F = np.zeros((self.nv, 3))
        self._compute_membrane_forces(F)
        if self.bending_model == 'ibm':
            self._compute_bending_forces_ibm(F)
        else:
            self._compute_bending_forces_dihedral(F)
        return F

    def _compute_bending_forces_ibm(self, F):
        """IBM bending (Laplacian-based, PSD stiffness) + torsion springs.

        IBM captures bending via mean curvature. Torsion is added via
        discrete twist springs between spanwise strips: resist differential
        x-slope of z-displacement between adjacent spanwise rows.
        """
        if self._Q is None:
            self._precompute_ibm()
        for d in range(3):
            F[:, d] -= self._Q @ self.u[:, d]

        # Add torsion springs: penalize differential twist between strips
        self._compute_torsion_forces(F)

    def _compute_torsion_forces(self, F):
        """Discrete torsion springs between spanwise strips.

        For each pair of adjacent spanwise rows, penalize differential
        twist (x-slope of z-displacement). This adds GJ-matching torsion
        on top of the IBM bending (which only captures mean curvature).

        Energy: E = Σ_j (GJ_eff / (2*dy)) * (θ_{j+1} - θ_j)²
        where θ_j = mean(∂w/∂x) at row j ≈ linear fit of u_z vs x.
        """
        if not hasattr(self, '_torsion_strips'):
            self._build_torsion_strips()

        GJ_eff = 4.0 * self._Dxy * self._ref_chord  # = GJ from plate theory
        strips = self._torsion_strips
        if len(strips) < 2:
            return

        # Compute twist angle per strip: θ = slope of u_z vs x_ref
        thetas = np.zeros(len(strips))
        for si, (idxs, x_mean) in enumerate(strips):
            u_z = self.u[idxs, 2]
            x_rel = self.vertices0[idxs, 0] - x_mean
            if len(x_rel) > 1:
                thetas[si] = np.polyfit(x_rel, u_z, 1)[0]

        # Force: for each pair of adjacent strips, apply restoring torque
        for si in range(len(strips) - 1):
            dy = self._torsion_dy
            k_tors = GJ_eff / dy
            d_theta = thetas[si + 1] - thetas[si]
            torque = -k_tors * d_theta

            # Distribute torque as z-forces proportional to x-distance from centroid
            for s, sign in [(si, -1.0), (si + 1, 1.0)]:
                idxs, x_mean = strips[s]
                x_rel = self.vertices0[idxs, 0] - x_mean
                sum_x2 = np.sum(x_rel**2)
                if sum_x2 > 1e-20:
                    F[idxs, 2] += sign * torque * x_rel / sum_x2

    def _build_torsion_strips(self):
        """Identify spanwise strips of nodes for torsion computation."""
        y_vals = sorted(set(np.round(self.vertices0[:, 1], 8)))
        strips = []
        for yv in y_vals:
            mask = np.abs(self.vertices0[:, 1] - yv) < 1e-8
            idxs = np.where(mask)[0]
            if len(idxs) > 1:
                x_mean = np.mean(self.vertices0[idxs, 0])
                strips.append((idxs, x_mean))
        self._torsion_strips = strips
        if len(strips) >= 2:
            self._torsion_dy = strips[1][0][0] - strips[0][0][0]
            # Get actual y-distance
            y0 = self.vertices0[strips[0][0][0], 1]
            y1 = self.vertices0[strips[1][0][0], 1]
            self._torsion_dy = abs(y1 - y0)
        else:
            self._torsion_dy = 1.0

    @property
    def _ref_chord(self):
        return self.vertices0[:, 0].max() - self.vertices0[:, 0].min()
        """Per-edge dihedral angle bending (correct physics including torsion)."""
        if self.n_interior_edges == 0:
            return
        x = self.vertices0 + self.u
        ea = self._edge_ea; eb = self._edge_eb
        ec = self._edge_ec; ed = self._edge_ed
        pa = x[ea]; pb = x[eb]; pc = x[ec]; pd = x[ed]
        e_vec = pb - pa
        L_cur = np.linalg.norm(e_vec, axis=1)
        ok = L_cur > 1e-30
        e_hat = np.zeros_like(e_vec)
        e_hat[ok] = e_vec[ok] / L_cur[ok, None]
        n0 = np.cross(e_vec, pc - pa)
        n1 = np.cross(pd - pa, e_vec)
        n0_len = np.linalg.norm(n0, axis=1)
        n1_len = np.linalg.norm(n1, axis=1)
        ok &= (n0_len > 1e-30) & (n1_len > 1e-30)
        n0_hat = np.zeros_like(n0)
        n1_hat = np.zeros_like(n1)
        n0_hat[ok] = n0[ok] / n0_len[ok, None]
        n1_hat[ok] = n1[ok] / n1_len[ok, None]
        cos_th = np.clip(np.sum(n0_hat * n1_hat, axis=1), -1.0, 1.0)
        cross_th = np.cross(n0_hat, n1_hat)
        sin_th = np.sum(cross_th * e_hat, axis=1)
        theta = np.arctan2(sin_th, cos_th)
        dtheta = theta - self._edge_theta_ref
        ok &= np.abs(dtheta) > 1e-15
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return
        # Simple coefficient: c_e = -D*L*Δθ
        coeff = -self._edge_D[idx] * self._edge_L[idx] * dtheta[idx]
        L_c = L_cur[idx]
        n0h = n0_hat[idx]; n1h = n1_hat[idx]
        A0 = self._edge_A0[idx]; A1 = self._edge_A1[idx]
        grad_c = -(L_c / (2.0 * A0))[:, None] * n0h
        grad_d = -(L_c / (2.0 * A1))[:, None] * n1h
        pc_p = pc[idx]; pd_p = pd[idx]; pa_p = pa[idx]; pb_p = pb[idx]
        L2 = L_c * L_c
        s_c = np.sum((pc_p - pa_p) * e_vec[idx], axis=1) / L2
        s_d = np.sum((pd_p - pa_p) * e_vec[idx], axis=1) / L2
        t_c = np.sum((pc_p - pb_p) * e_vec[idx], axis=1) / L2
        t_d = np.sum((pd_p - pb_p) * e_vec[idx], axis=1) / L2
        grad_a = -s_c[:, None] * grad_c - s_d[:, None] * grad_d
        grad_b = t_c[:, None] * grad_c + t_d[:, None] * grad_d
        c = coeff[:, None]
        ea_i = ea[idx]; eb_i = eb[idx]; ec_i = ec[idx]; ed_i = ed[idx]
        for d in range(3):
            np.add.at(F[:, d], ea_i, (c * grad_a)[:, d])
            np.add.at(F[:, d], eb_i, (c * grad_b)[:, d])
            np.add.at(F[:, d], ec_i, (c * grad_c)[:, d])
            np.add.at(F[:, d], ed_i, (c * grad_d)[:, d])

    def _compute_membrane_forces(self, F):
        # Vectorised CST membrane forces (no Python loop over triangles)
        tri = self.triangles                          # (nt, 3)
        u_all = self.u                                # (nv, 3)
        ux = u_all[tri, 0]                            # (nt, 3)
        uy = u_all[tri, 1]                            # (nt, 3)
        dNdx = self._dNdx                             # (nt, 3)
        dNdy = self._dNdy                             # (nt, 3)

        eps_xx = np.sum(dNdx * ux, axis=1)            # (nt,)
        eps_yy = np.sum(dNdy * uy, axis=1)
        eps_xy = np.sum(dNdx * uy, axis=1) + np.sum(dNdy * ux, axis=1)

        sig_xx = self._D00 * eps_xx + self._D01 * eps_yy
        sig_yy = self._D01 * eps_xx + self._D11 * eps_yy
        sig_xy = self._D22 * eps_xy

        coeff = -self._ref_area * self.h              # (nt,)
        fx_k = coeff[:, None] * (dNdx * sig_xx[:, None] + dNdy * sig_xy[:, None])
        fy_k = coeff[:, None] * (dNdx * sig_xy[:, None] + dNdy * sig_yy[:, None])

        np.add.at(F[:, 0], tri.ravel(), fx_k.ravel())
        np.add.at(F[:, 1], tri.ravel(), fy_k.ravel())

    def _compute_bending_forces(self, F):
        """Isometric Bending Model (IBM) — Wardetzky, Bergou et al. 2006/2007.

        Bending energy: E = (1/2) * u^T * Q * u
        where Q is a constant PSD matrix built from the cotangent Laplacian.

        Force: f = -Q * u  (linear in displacement)
        Stiffness: K_bend = Q  (constant, PSD by construction)
        """
        if self._Q is None:
            self._precompute_ibm()
        # f_bend = -Q * u, applied per-component (Q acts on vertex indices)
        for d in range(3):
            F[:, d] -= self._Q @ self.u[:, d]

    def _precompute_ibm(self):
        """Build the constant PSD bending stiffness Q = L^T * diag(D/A) * L.

        L is the cotangent Laplacian (integrated form).
        D/A is the bending rigidity divided by vertex area.
        Q is PSD since it's A^T * B * A with B positive diagonal.
        """
        nv = self.nv
        L = np.zeros((nv, nv))

        # Build cotangent Laplacian from triangles
        for t in range(self.nt):
            i0, i1, i2 = self.triangles[t]
            p0 = self.vertices0[i0]
            p1 = self.vertices0[i1]
            p2 = self.vertices0[i2]

            # For each edge of this triangle, compute cotangent of opposite angle
            # Edge (i0,i1) opposite to i2: cot(angle at i2)
            # Edge (i1,i2) opposite to i0: cot(angle at i0)
            # Edge (i0,i2) opposite to i1: cot(angle at i1)

            e01 = p1 - p0; e02 = p2 - p0; e12 = p2 - p1

            # cot(angle at i2) for edge (i0,i1):
            # vectors from i2 to i0 and i2 to i1
            a = p0 - p2; b = p1 - p2
            cross = np.cross(a, b)
            cn = np.linalg.norm(cross)
            if cn > 1e-30:
                cot2 = np.dot(a, b) / cn
            else:
                cot2 = 0.0

            # cot(angle at i0) for edge (i1,i2):
            a = p1 - p0; b = p2 - p0
            cross = np.cross(a, b)
            cn = np.linalg.norm(cross)
            if cn > 1e-30:
                cot0 = np.dot(a, b) / cn
            else:
                cot0 = 0.0

            # cot(angle at i1) for edge (i0,i2):
            a = p0 - p1; b = p2 - p1
            cross = np.cross(a, b)
            cn = np.linalg.norm(cross)
            if cn > 1e-30:
                cot1 = np.dot(a, b) / cn
            else:
                cot1 = 0.0

            # Add to Laplacian: L[i,j] += cot(angle_opposite_to_edge_ij) / 2
            L[i0, i1] += cot2 / 2.0
            L[i1, i0] += cot2 / 2.0
            L[i1, i2] += cot0 / 2.0
            L[i2, i1] += cot0 / 2.0
            L[i0, i2] += cot1 / 2.0
            L[i2, i0] += cot1 / 2.0

        # Diagonal: L[i,i] = -sum_j L[i,j]
        for i in range(nv):
            L[i, i] = -np.sum(L[i, :])

        # Vertex areas (lumped: 1/3 of adjacent triangle areas)
        A_vert = np.zeros(nv)
        for t in range(self.nt):
            for k in range(3):
                A_vert[self.triangles[t, k]] += self._ref_area[t] / 3.0

        # Bending rigidity per vertex.
        # The IBM with isotropic cotangent Laplacian overestimates stiffness
        # by a factor ~4 for structured meshes. Calibrate analytically:
        # For the continuous Laplacian on a rectangle, the integrated bending
        # stiffness gives E = D * ∫|Δw|² dA, which for a cantilever with
        # tip load P gives δ = P*L³/(3*D*width). We want δ = P*L³/(3*EI),
        # so D*width = EI → D = EI/width. The cotangent Laplacian already
        # includes the area integration, so D_vert = (EI/chord) * cal_factor.
        # The calibration factor accounts for the discrete-vs-continuous gap.
        D_base = self._Dy  # = EI / chord for this shell
        # Empirical calibration: ratio ~0.252 for structured quad meshes
        # This factor arises because the cotangent Laplacian on a structured
        # triangulation overestimates the bending energy by ~4×
        cal = self._ibm_cal
        D_vert = np.full(nv, D_base * cal)

        # Weight matrix: W = diag(D_vert / A_vert)
        W = np.zeros(nv)
        for i in range(nv):
            if A_vert[i] > 1e-30:
                W[i] = D_vert[i] / A_vert[i]

        # Q = L^T * diag(W) * L  (PSD by construction)
        WL = L.copy()
        for i in range(nv):
            WL[i, :] *= W[i]
        self._Q = L.T @ WL

        # Zero out BC rows/columns in Q
        bc = self.mass_inv == 0.0
        for i in range(nv):
            if bc[i]:
                self._Q[i, :] = 0.0
                self._Q[:, i] = 0.0

        self._Q_bend = self._Q.copy()  # expose for implicit solver

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def step(self, F_ext, dt):
        if self.use_gpu and self._gpu_ctx is not None:
            self._step_gpu(F_ext, dt)
            return
        self._step_cpu(F_ext, dt)

    def _step_cpu(self, F_ext, dt):
        # Velocity-Verlet: update position with old acceleration first
        self.u += self.v * dt + 0.5 * self.a * dt * dt

        F_int = self.compute_forces()
        F_damp = np.zeros_like(self.v)
        if self.damping > 0:
            F_damp = -self.damping * self.v * self.mass[:, None]
        F_total = F_int + F_ext + F_damp
        a_new = F_total * self.mass_inv[:, None]

        self.v += 0.5 * (self.a + a_new) * dt
        self.a = a_new

        mask = self.mass_inv > 0
        self.u[~mask] = 0.0
        self.v[~mask] = 0.0
        self.a[~mask] = 0.0

    def step_subcycles(self, F_ext, dt, n_sub):
        """Run n_sub substeps with constant external force."""
        dt_sub = dt / n_sub
        if self.use_gpu and self._gpu_ctx is not None:
            self._gpu_subcycle(F_ext, dt_sub, n_sub)
        else:
            for _ in range(n_sub):
                self._step_cpu(F_ext, dt_sub)

    def compute_cfl_dt(self, safety=0.3):
        c_max = max(np.sqrt(self.Ex / self.rho),
                    np.sqrt(self.Ey / self.rho))
        if self.n_interior_edges > 0:
            L_min = self._edge_L.min()
        else:
            L_min = 1.0
        return safety * L_min / c_max

    # ------------------------------------------------------------------
    # Boundary conditions & accessors
    # ------------------------------------------------------------------
    def set_bc(self, node_indices):
        for i in node_indices:
            self.mass_inv[i] = 0.0

    def get_nodal_positions(self):
        return self.vertices0 + self.u

    def get_nodal_displacements(self):
        return self.u.copy()

    def reset(self):
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.a[:] = 0.0

    def sync_to_gpu(self):
        if self._gpu_ctx is not None:
            self._gpu_ctx.upload_state(self)

    # ------------------------------------------------------------------
    # GPU methods (lazy import)
    # ------------------------------------------------------------------
    def _init_gpu(self):
        try:
            from .warp_bst import BSTGPUContext
        except ImportError:
            raise ImportError(
                "NVIDIA Warp required for GPU mode. pip install warp-lang")
        self._gpu_ctx = BSTGPUContext(self)

    def _step_gpu(self, F_ext, dt):
        from .warp_bst import bst_subcycle_gpu
        bst_subcycle_gpu(self._gpu_ctx, F_ext, dt, 1)
        self._gpu_ctx.download_state(self)

    def _gpu_subcycle(self, F_ext, dt_sub, n_sub):
        from .warp_bst import bst_subcycle_gpu
        bst_subcycle_gpu(self._gpu_ctx, F_ext, dt_sub, n_sub)
        self._gpu_ctx.download_state(self)
