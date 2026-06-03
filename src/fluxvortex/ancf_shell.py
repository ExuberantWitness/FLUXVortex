"""ANCF (Absolute Nodal Coordinate Formulation) thin shell element.

Port of MATLAB_ANCF_shell (Yamano et al.) — 4-node quadrilateral with
bicubic Hermite shape functions, 9 DOF/node, Kirchhoff-Love plate theory.

Node coordinates: [r(3), dr/dx(3), dr/dy(3)] = 9 DOF per node
Element DOFs: 4 nodes x 9 DOF = 36

References:
  - Yamano et al., J. Sound and Vibration (2020)
  - https://github.com/KRproject-tech/MATLAB_ANCF_shell
"""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# ─── Constants ───
NDOF_NODE = 9
NDOF_ELEM = 4 * NDOF_NODE  # 36


# ─── Gauss-Legendre quadrature ───
def _gauss_legendre(n):
    """Return (points, weights) for n-point Gauss-Legendre quadrature on [-1,1]."""
    x = np.zeros(n)
    w = np.zeros(n)
    m = (n + 1) // 2
    for i in range(m):
        z = np.cos(np.pi * (i + 0.75) / (n + 0.5))
        z1 = z + 1.0
        while abs(z - z1) > 1e-15:
            p1, p2 = 1.0, 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
        x[i] = -z
        x[n - 1 - i] = z
        w[i] = 2.0 / ((1 - z * z) * pp * pp)
        w[n - 1 - i] = w[i]
    return x, w


# ─── Bicubic Hermite shape functions ───
# 12 scalar shape functions for a 4-node quad with position + 2 slope DOFs.
# Parameter space: xi in [0,1], eta in [0,1]
# Node ordering: 1=(0,0), 2=(1,0), 3=(1,1), 4=(0,1)
# Physical coords: x = xi*dL, y = eta*dW

def _shape_funcs(xi, eta, dL, dW):
    S = np.empty(12)
    S[0]  = -(xi-1)*(eta-1)*(2*eta**2 - eta + 2*xi**2 - xi - 1)
    S[1]  = -dL*xi*(xi-1)**2*(eta-1)
    S[2]  = -dW*eta*(eta-1)**2*(xi-1)
    S[3]  = xi*(2*eta**2 - eta - 3*xi + 2*xi**2)*(eta-1)
    S[4]  = -dL*xi**2*(xi-1)*(eta-1)
    S[5]  = dW*xi*eta*(eta-1)**2
    S[6]  = -xi*eta*(1 - 3*xi - 3*eta + 2*eta**2 + 2*xi**2)
    S[7]  = dL*xi**2*eta*(xi-1)
    S[8]  = dW*xi*eta**2*(eta-1)
    S[9]  = eta*(xi-1)*(2*xi**2 - xi - 3*eta + 2*eta**2)
    S[10] = dL*xi*eta*(xi-1)**2
    S[11] = -dW*eta**2*(xi-1)*(eta-1)
    return S


def _shape_dxi(xi, eta, dL, dW):
    dS = np.empty(12)
    dS[0]  = -(eta-1)*(2*eta**2 - eta + 2*xi**2 - xi - 1) - (xi-1)*(eta-1)*(4*xi - 1)
    dS[1]  = -dL*(xi-1)**2*(eta-1) - dL*xi*2*(xi-1)*(eta-1)
    dS[2]  = -dW*eta*(eta-1)**2
    dS[3]  = (2*eta**2 - eta - 3*xi + 2*xi**2)*(eta-1) + xi*(-3 + 4*xi)*(eta-1)
    dS[4]  = -dL*2*xi*(xi-1)*(eta-1) - dL*xi**2*(eta-1)
    dS[5]  = dW*eta*(eta-1)**2
    dS[6]  = -eta*(1 - 3*xi - 3*eta + 2*eta**2 + 2*xi**2) - xi*eta*(-3 + 4*xi)
    dS[7]  = dL*2*xi*eta*(xi-1) + dL*xi**2*eta
    dS[8]  = dW*eta**2*(eta-1)
    dS[9]  = eta*(2*xi**2 - xi - 3*eta + 2*eta**2) + eta*(xi-1)*(4*xi - 1)
    dS[10] = dL*eta*(xi-1)**2 + dL*xi*eta*2*(xi-1)
    dS[11] = -dW*eta**2*(eta-1)
    return dS


def _shape_deta(xi, eta, dL, dW):
    dS = np.empty(12)
    dS[0]  = -(xi-1)*(2*eta**2 - eta + 2*xi**2 - xi - 1) - (xi-1)*(eta-1)*(4*eta - 1)
    dS[1]  = -dL*xi*(xi-1)**2
    dS[2]  = -dW*(eta-1)**2*(xi-1) - dW*eta*2*(eta-1)*(xi-1)
    dS[3]  = xi*(4*eta - 1)*(eta-1) + xi*(2*eta**2 - eta - 3*xi + 2*xi**2)
    dS[4]  = -dL*xi**2*(xi-1)
    dS[5]  = dW*xi*(eta-1)**2 + dW*xi*eta*2*(eta-1)
    dS[6]  = -xi*(1 - 3*xi - 3*eta + 2*eta**2 + 2*xi**2) - xi*eta*(-3 + 4*eta)
    dS[7]  = dL*xi**2*(xi-1)
    dS[8]  = dW*xi*2*eta*(eta-1) + dW*xi*eta**2
    dS[9]  = (xi-1)*(2*xi**2 - xi - 3*eta + 2*eta**2) + eta*(xi-1)*(-3 + 4*eta)
    dS[10] = dL*xi*(xi-1)**2
    dS[11] = -dW*2*eta*(xi-1)*(eta-1) - dW*eta**2*(xi-1)
    return dS


def _shape_dxi2(xi, eta, dL, dW):
    d2S = np.empty(12)
    d2S[0]  = -(eta-1)*(4*xi-1) - (eta-1)*(4*xi-1) - (xi-1)*(eta-1)*4
    d2S[1]  = -dL*2*(xi-1)*(eta-1) - dL*2*(xi-1)*(eta-1) - dL*xi*2*(eta-1)
    d2S[2]  = 0.0
    d2S[3]  = (-3+4*xi)*(eta-1) + (-3+4*xi)*(eta-1) + xi*4*(eta-1)
    d2S[4]  = -dL*2*(xi-1)*(eta-1) - dL*2*xi*(eta-1) - dL*2*xi*(eta-1)
    d2S[5]  = 0.0
    d2S[6]  = -eta*(-3+4*xi) - eta*(-3+4*xi) - xi*eta*4
    d2S[7]  = dL*2*eta*(xi-1) + dL*2*xi*eta + dL*2*xi*eta
    d2S[8]  = 0.0
    d2S[9]  = eta*(4*xi-1) + eta*(4*xi-1) + eta*(xi-1)*4
    d2S[10] = dL*eta*2*(xi-1) + dL*eta*2*(xi-1) + dL*xi*eta*2
    d2S[11] = 0.0
    return d2S


def _shape_deta2(xi, eta, dL, dW):
    d2S = np.empty(12)
    d2S[0]  = -(xi-1)*(4*eta-1) - (xi-1)*(4*eta-1) - (xi-1)*(eta-1)*4
    d2S[1]  = 0.0
    d2S[2]  = -dW*2*(eta-1)*(xi-1) - dW*2*(eta-1)*(xi-1) - dW*eta*2*(xi-1)
    d2S[3]  = xi*4*(eta-1) + xi*(4*eta-1) + xi*(4*eta-1)
    d2S[4]  = 0.0
    d2S[5]  = dW*xi*2*(eta-1) + dW*xi*2*(eta-1) + dW*xi*eta*2
    d2S[6]  = -xi*(-3+4*eta) - xi*(-3+4*eta) - xi*eta*4
    d2S[7]  = 0.0
    d2S[8]  = dW*xi*2*(eta-1) + dW*xi*2*eta + dW*xi*2*eta
    d2S[9]  = (xi-1)*(-3+4*eta) + (xi-1)*(-3+4*eta) + eta*(xi-1)*4
    d2S[10] = 0.0
    d2S[11] = -dW*2*(xi-1)*(eta-1) - dW*2*eta*(xi-1) - dW*2*eta*(xi-1)
    return d2S


def _shape_dxieta(xi, eta, dL, dW):
    d2S = np.empty(12)
    d2S[0]  = -(2*eta**2 - eta + 2*xi**2 - xi - 1) - (eta-1)*(4*eta-1) - (xi-1)*(4*xi-1)
    d2S[1]  = -dL*(xi-1)**2 - dL*xi*2*(xi-1)
    d2S[2]  = -dW*(eta-1)**2 - dW*eta*2*(eta-1)
    d2S[3]  = (4*eta-1)*(eta-1) + (2*eta**2 - eta - 3*xi + 2*xi**2) + xi*(-3+4*xi)
    d2S[4]  = -dL*2*xi*(xi-1) - dL*xi**2
    d2S[5]  = dW*(eta-1)**2 + dW*eta*2*(eta-1)
    d2S[6]  = -(1-3*xi-3*eta+2*eta**2+2*xi**2) - eta*(-3+4*eta) - xi*(-3+4*xi)
    d2S[7]  = dL*2*xi*(xi-1) + dL*xi**2
    d2S[8]  = dW*2*eta*(eta-1) + dW*eta**2
    d2S[9]  = (2*xi**2 - xi - 3*eta + 2*eta**2) + eta*(-3+4*eta) + (xi-1)*(4*xi-1)
    d2S[10] = dL*(xi-1)**2 + dL*xi*2*(xi-1)
    d2S[11] = -dW*2*eta*(eta-1) - dW*eta**2
    return d2S


# ─── Element precomputation ───
class _ElementData:
    """Precomputed shape function matrices at Gauss points for one element."""
    __slots__ = ('S', 'dSx', 'dSy', 'd2Sx', 'd2Sy', 'd2Sxy',
                 'dL', 'dW', 'detJ')

    def __init__(self, dL, dW, n_gauss=5):
        self.dL = dL
        self.dW = dW
        self.detJ = dL * dW / 4.0

        pts, wts = _gauss_legendre(n_gauss)
        xi_pts = (pts + 1) / 2.0
        ng = n_gauss

        self.S     = np.zeros((ng, ng, 3, 36))
        self.dSx   = np.zeros((ng, ng, 3, 36))
        self.dSy   = np.zeros((ng, ng, 3, 36))
        self.d2Sx  = np.zeros((ng, ng, 3, 36))
        self.d2Sy  = np.zeros((ng, ng, 3, 36))
        self.d2Sxy = np.zeros((ng, ng, 3, 36))

        I3 = np.eye(3)
        for i in range(ng):
            xi = xi_pts[i]
            for j in range(ng):
                eta = xi_pts[j]
                self.S[i, j]     = np.kron(_shape_funcs(xi, eta, dL, dW), I3)
                self.dSx[i, j]   = np.kron(_shape_dxi(xi, eta, dL, dW), I3) / dL
                self.dSy[i, j]   = np.kron(_shape_deta(xi, eta, dL, dW), I3) / dW
                self.d2Sx[i, j]  = np.kron(_shape_dxi2(xi, eta, dL, dW), I3) / dL**2
                self.d2Sy[i, j]  = np.kron(_shape_deta2(xi, eta, dL, dW), I3) / dW**2
                self.d2Sxy[i, j] = np.kron(_shape_dxieta(xi, eta, dL, dW), I3) / (dL * dW)


# ─── ANCF Shell ───
class ANCFShell:
    """ANCF thin shell element (9 DOF/node, Kirchhoff-Love).

    Parameters
    ----------
    nodes : (nn, 3) nodal reference positions
    quads : (ne, 4) quad element connectivity [n0,n1,n2,n3]
            Node order: n0=(0,0), n1=(1,0), n2=(1,1), n3=(0,1)
    h     : thickness
    rho   : density
    Ex, Ey : Young's moduli
    nu_xy : Poisson's ratio
    G_xy  : shear modulus (if None, computed from Ex, nu_xy)
    mode  : 'full' (membrane+bending), 'membrane' (bending stiffness=0)
    n_gauss : Gauss quadrature order (default 5)
    structural_damping : Rayleigh damping coefficient
    """

    NDOF_NODE = NDOF_NODE

    def __init__(self, nodes, quads, h, rho, Ex, Ey, nu_xy, G_xy=None,
                 mode='full', n_gauss=5, structural_damping=0.0):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.quads = np.asarray(quads, dtype=np.int32)
        self.h = h
        self.rho = rho
        self.nu_xy = nu_xy
        self.mode = mode
        self.n_gauss = n_gauss
        self.structural_damping = structural_damping

        self.nn = len(self.nodes)
        self.ne = len(self.quads)
        self.ndof = self.nn * NDOF_NODE

        if G_xy is None:
            G_xy = Ex / (2.0 * (1.0 + nu_xy))

        # Membrane stiffness matrix (stress resultants per unit thickness)
        self.Dm = Ex / (1 - nu_xy**2) * np.array([
            [1,     nu_xy, 0],
            [nu_xy, 1,     0],
            [0,     0,     (1-nu_xy)/2]
        ])
        # Bending stiffness matrix
        if mode == 'membrane':
            self.Dk = np.zeros((3, 3))
        else:
            self.Dk = h**3 / 12.0 * self.Dm

        # Generalized coordinates: [q; dq/dt]
        # q per node: [rx,ry,rz, dx_rx,dx_ry,dx_rz, dy_rx,dy_ry,dy_rz]
        self.q = np.zeros(self.ndof)
        self.dq = np.zeros(self.ndof)

        # Initialize q with reference config + unit slopes
        for n in range(self.nn):
            base = n * NDOF_NODE
            self.q[base:base+3] = self.nodes[n]
            self.q[base+3:base+6] = [1.0, 0.0, 0.0]
            self.q[base+6:base+9] = [0.0, 1.0, 0.0]

        # Precompute element data
        self._elems = []
        self._dL = np.zeros(self.ne)
        self._dW = np.zeros(self.ne)
        for e in range(self.ne):
            n0, n1, n2, n3 = self.quads[e]
            dL = abs(self.nodes[n1, 0] - self.nodes[n0, 0])
            dW = abs(self.nodes[n3, 1] - self.nodes[n0, 1])
            if dL < 1e-15:
                dL = abs(self.nodes[n2, 0] - self.nodes[n3, 0])
            if dW < 1e-15:
                dW = abs(self.nodes[n2, 1] - self.nodes[n1, 1])
            self._dL[e] = dL
            self._dW[e] = dW
            self._elems.append(_ElementData(dL, dW, n_gauss))

        # Boundary conditions
        self._bc_dofs = set()

        # Assemble constant mass matrix
        self.M = self._assemble_mass()

    def _elem_dofs(self, e):
        nd = self.quads[e]
        dofs = np.empty(NDOF_ELEM, dtype=np.int32)
        for k in range(4):
            dofs[k*9:(k+1)*9] = nd[k]*NDOF_NODE + np.arange(9)
        return dofs

    # ─── Mass matrix (constant) ───
    def _assemble_mass(self):
        rows, cols, vals = [], [], []
        pts, wts = _gauss_legendre(self.n_gauss)

        for e in range(self.ne):
            ed = self._elems[e]
            dofs = self._elem_dofs(e)
            M_e = np.zeros((NDOF_ELEM, NDOF_ELEM))
            for i in range(self.n_gauss):
                for j in range(self.n_gauss):
                    S = ed.S[i, j]
                    StS = S.T @ S
                    w = wts[i] * wts[j] * ed.detJ
                    M_e += w * StS
            M_e *= self.rho * self.h

            for a in range(NDOF_ELEM):
                for b in range(NDOF_ELEM):
                    if abs(M_e[a, b]) > 1e-30:
                        rows.append(dofs[a])
                        cols.append(dofs[b])
                        vals.append(M_e[a, b])

        return coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()

    # ─── Element internal forces + analytical tangent ───
    def _elem_forces_and_tangent(self, e, q_global):
        """Compute internal forces Qe and tangent stiffness dQe/dq for element e.

        Returns (Qe, Kt_e) where Qe is (36,) and Kt_e is (36, 36).

        Following MATLAB reference:
          membrane force: Q_mem = h * integral(deps^T * Dm * eps) dA
          membrane tangent: K_mem = h * integral(dqdq_eps^T * Dm * eps + deps^T * Dm * deps) dA
          bending force: Q_bend = integral(dk^T * Dk * kappa) dA
          bending tangent: K_bend = integral(dk_dq^T * Dk * dk_dq + ...) dA
        """
        ed = self._elems[e]
        dofs = self._elem_dofs(e)
        q_e = q_global[dofs]

        pts, wts = _gauss_legendre(self.n_gauss)
        ng = self.n_gauss

        Q_mem = np.zeros(NDOF_ELEM)
        Q_bend = np.zeros(NDOF_ELEM)
        K_mem = np.zeros((NDOF_ELEM, NDOF_ELEM))
        K_bend = np.zeros((NDOF_ELEM, NDOF_ELEM))

        for i in range(ng):
            for j in range(ng):
                w = wts[i] * wts[j] * ed.detJ

                dSx   = ed.dSx[i, j]   # (3, 36)
                dSy   = ed.dSy[i, j]   # (3, 36)
                d2Sx  = ed.d2Sx[i, j]  # (3, 36)
                d2Sy  = ed.d2Sy[i, j]  # (3, 36)
                d2Sxy = ed.d2Sxy[i, j] # (3, 36)

                dx_r   = dSx @ q_e     # (3,)
                dy_r   = dSy @ q_e     # (3,)
                d2x_r  = d2Sx @ q_e    # (3,)
                d2y_r  = d2Sy @ q_e    # (3,)
                d2xy_r = d2Sxy @ q_e   # (3,)

                # ── Membrane (Green-Lagrange strain) ──
                eps_xx = 0.5 * (dx_r @ dx_r - 1.0)
                eps_yy = 0.5 * (dy_r @ dy_r - 1.0)
                gam_xy = dx_r @ dy_r
                eps_v = np.array([eps_xx, eps_yy, gam_xy])  # (3,)

                # deps/dq: (3, 36)
                deps = np.zeros((3, NDOF_ELEM))
                deps[0] = dSx.T @ dx_r
                deps[1] = dSy.T @ dy_r
                deps[2] = dSx.T @ dy_r + dSy.T @ dx_r

                # Precomputed A matrices (dqdq_eps in MATLAB):
                # A1 = dx_S^T * dx_S,  A2 = dy_S^T * dy_S,  A3 = dx_S^T * dy_S + dy_S^T * dx_S
                A1 = dSx.T @ dSx   # (36, 36)
                A2 = dSy.T @ dSy   # (36, 36)
                A3 = dSx.T @ dSy + dSy.T @ dSx  # (36, 36)

                Dm_eps = self.Dm @ eps_v  # (3,)
                Q_mem += w * (deps.T @ Dm_eps)

                # Tangent: K = h * integral(dqdq^T * Dm * eps + deps^T * Dm * deps)
                # dqdq_eps^T * Dm_eps: (36,36) weighted by strain
                K_mem += w * (A1 * Dm_eps[0] + A2 * Dm_eps[1] + A3 * Dm_eps[2])
                # deps^T * Dm * deps: (36,36) material stiffness
                K_mem += w * (deps.T @ self.Dm @ deps)

                # ── Bending (Kirchhoff-Love curvature) ──
                if self.mode != 'membrane':
                    n_vec = np.cross(dx_r, dy_r)
                    norm_n = np.linalg.norm(n_vec)
                    if norm_n < 1e-15:
                        continue
                    n_hat = n_vec / norm_n

                    kxx = n_hat @ d2x_r
                    kyy = n_hat @ d2y_r
                    kxy = n_hat @ d2xy_r
                    k_v = np.array([kxx, kyy, 2.0 * kxy])

                    # dk/dq: (3, 36) analytical
                    dk = np.zeros((3, NDOF_ELEM))
                    P = np.eye(3) - np.outer(n_hat, n_hat)

                    # Also compute d2k/dq2 for bending tangent
                    # dn_hat/dq_j = (P @ dn_j) / norm_n
                    # d2n_hat/dq_j dq_k is complex but MATLAB omits the second-order
                    # normal derivative in the tangent, using only first-order terms.

                    for jj in range(NDOF_ELEM):
                        dSx_j = dSx[:, jj]
                        dSy_j = dSy[:, jj]
                        dn_j = np.cross(dSx_j, dy_r) + np.cross(dx_r, dSy_j)
                        dn_hat_j = (P @ dn_j) / norm_n

                        dk[0, jj] = dn_hat_j @ d2x_r + n_hat @ d2Sx[:, jj]
                        dk[1, jj] = dn_hat_j @ d2y_r + n_hat @ d2Sy[:, jj]
                        dk[2, jj] = 2.0 * (dn_hat_j @ d2xy_r + n_hat @ d2Sxy[:, jj])

                    Dk_k = self.Dk @ k_v  # (3,)
                    Q_bend += w * (dk.T @ Dk_k)

                    # Bending tangent: K_bend = integral(dk^T * Dk * dk) dA
                    # Following MATLAB: only first-order tangent (dk_dq^T * Dk * dk_dq)
                    K_bend += w * (dk.T @ self.Dk @ dk)

        Qe = Q_mem * self.h + Q_bend
        Kt = K_mem * self.h + K_bend
        return Qe, Kt

    def _elem_forces(self, e, q_global):
        Qe, _ = self._elem_forces_and_tangent(e, q_global)
        return Qe

    def _internal_forces(self, q=None):
        if q is None:
            q = self.q
        Qe = np.zeros(self.ndof)
        for e in range(self.ne):
            Qe_e = self._elem_forces(e, q)
            dofs = self._elem_dofs(e)
            Qe[dofs] += Qe_e
        return Qe

    def _internal_forces_and_tangent(self, q=None):
        """Assemble global internal forces and tangent stiffness."""
        if q is None:
            q = self.q
        Qe = np.zeros(self.ndof)
        rows, cols, vals = [], [], []
        for e in range(self.ne):
            Qe_e, Kt_e = self._elem_forces_and_tangent(e, q)
            dofs = self._elem_dofs(e)
            Qe[dofs] += Qe_e
            for a_idx, a in enumerate(dofs):
                for b_idx, b in enumerate(dofs):
                    if abs(Kt_e[a_idx, b_idx]) > 1e-30:
                        rows.append(a)
                        cols.append(b)
                        vals.append(Kt_e[a_idx, b_idx])
        Kt = coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()
        return Qe, Kt

    # ─── Boundary conditions ───
    def set_bc(self, nodes_bc, fix_slopes=True):
        for n in nodes_bc:
            base = n * NDOF_NODE
            for k in range(NDOF_NODE):
                self._bc_dofs.add(base + k)

    def set_bc_translations(self, nodes_bc):
        """Fix only translational DOFs (rx, ry, rz), leave slopes free."""
        for n in nodes_bc:
            base = n * NDOF_NODE
            for k in range(3):
                self._bc_dofs.add(base + k)

    def gravity_load(self, g=9.81):
        """Consistent gravity body force: Qf = integral S^T * F_in * h dA."""
        F_in = np.array([0.0, 0.0, -self.rho * g])
        Qf = np.zeros(self.ndof)
        pts, wts = _gauss_legendre(self.n_gauss)
        for e in range(self.ne):
            ed = self._elems[e]
            dofs = self._elem_dofs(e)
            for i in range(self.n_gauss):
                for j in range(self.n_gauss):
                    w = wts[i] * wts[j] * ed.detJ * self.h
                    S = ed.S[i, j]  # (3, 36)
                    Qf[dofs] += w * (S.T @ F_in)
        return Qf

    # ─── Implicit Newmark-beta time step ───
    def step_newmark(self, F_ext, dt, alpha_v=0.5, newton_tol=1e-8, max_newton=20):
        """One implicit Newmark-beta step (trapezoidal rule when alpha_v=0.5).

        Following MATLAB reference: new_X_func_FAST.m + solve_structure.m
        Uses analytical tangent stiffness from _elem_forces_and_tangent.

        Two-stage solve:
          Stage 0: tangent at q(n), solve for predictor
          Stage 1: update forces at q(n+1), re-solve with averaged forces
        """
        bc = np.array(sorted(self._bc_dofs), dtype=np.int32)
        free = np.setdiff1d(np.arange(self.ndof), bc)
        nf = len(free)

        q_n = self.q.copy()
        dq_n = self.dq.copy()

        # Stage 0: assemble Qe(q_n) and tangent Kt(q_n)
        Qe_n, Kt_n = self._internal_forces_and_tangent(q_n)

        c_damp = 2.0 if self.structural_damping == 0 else 1.0

        # Extract free-free submatrices (sparse)
        Kt_ff = Kt_n[np.ix_(free, free)].tocsc()
        M_ff = self.M[np.ix_(free, free)].tocsc()
        Qd_ff = (self.structural_damping * Kt_ff).tocsc() if self.structural_damping > 0 else None

        # Build D_matrix (sparse): [[I, 0], [Qd + c*dt/2*Kt, M]]
        from scipy.sparse import eye as speye, bmat as spbmat, csc_matrix as spcsc
        I_sp = speye(nf, format='csc')
        O_sp = spcsc((nf, nf))

        D_bot_left = (Qd_ff + c_damp * dt / 2.0 * Kt_ff) if Qd_ff is not None else c_damp * dt / 2.0 * Kt_ff
        D_mat = spbmat([[I_sp, O_sp], [D_bot_left, M_ff]], format='csc')

        # X2_matrix: [[0, I], [0, 0]]
        X2_mat = spbmat([[O_sp, I_sp], [O_sp, O_sp]], format='csc')

        A1 = D_mat - alpha_v * dt * X2_mat
        A2 = D_mat + (1.0 - alpha_v) * dt * X2_mat

        X_n_free = np.concatenate([q_n[free], dq_n[free]])
        A2Xn = A2 @ X_n_free

        # Stage 0: Q_global = F - Qe(q_n)
        Q_global = F_ext - Qe_n
        rhs0 = np.zeros(2 * nf)
        rhs0[nf:] = Q_global[free]

        A1_inv_A2Xn = spsolve(A1, A2Xn)
        X_p1_free = A1_inv_A2Xn + dt * spsolve(A1, rhs0)

        # Stage 1: compute Qe at predicted q_{n+1}
        q_p1 = q_n.copy()
        q_p1[free] = X_p1_free[:nf]
        Qe_p1 = self._internal_forces(q_p1)

        # Use averaged force: (Qe_n + Qe_p1)/2
        Q_global2 = F_ext - (Qe_n + Qe_p1) / 2.0
        rhs1 = np.zeros(2 * nf)
        rhs1[nf:] = Q_global2[free]

        X_new_free = A1_inv_A2Xn + dt * spsolve(A1, rhs1)

        self.q[free] = X_new_free[:nf]
        self.dq[free] = X_new_free[nf:]

    # ─── Time stepping (Velocity-Verlet explicit) ───
    def step(self, F_ext, dt, n_sub=1):
        dt_sub = dt / n_sub
        bc = np.array(sorted(self._bc_dofs), dtype=np.int32)
        free = np.setdiff1d(np.arange(self.ndof), bc)

        M_free = self.M[np.ix_(free, free)].tocsc()

        for _ in range(n_sub):
            Qe = self._internal_forces()
            rhs = F_ext - Qe
            if self.structural_damping > 0:
                rhs -= self.structural_damping * self.dq

            a = np.zeros(self.ndof)
            a[free] = spsolve(M_free, rhs[free])

            self.q[free] += dt_sub * self.dq[free] + 0.5 * dt_sub**2 * a[free]
            self.dq[free] += dt_sub * a[free]

    # ─── Accessors ───
    def positions(self):
        pos = np.zeros((self.nn, 3))
        for n in range(self.nn):
            base = n * NDOF_NODE
            pos[n] = self.q[base:base+3]
        return pos

    @property
    def u(self):
        u = np.zeros(self.ndof)
        for n in range(self.nn):
            base = n * NDOF_NODE
            u[base:base+3] = self.q[base:base+3] - self.nodes[n]
        return u

    def get_displacement(self):
        return self.positions() - self.nodes
