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
                 'dL', 'dW', 'detJ', 'gw', 'A1', 'A2', 'A3')

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
        self.gw    = np.zeros((ng, ng))  # Gauss weights * detJ

        # Precomputed A matrices (constant, depend only on shape fns)
        self.A1 = np.zeros((ng, ng, 36, 36))
        self.A2 = np.zeros((ng, ng, 36, 36))
        self.A3 = np.zeros((ng, ng, 36, 36))

        I3 = np.eye(3)
        for i in range(ng):
            xi = xi_pts[i]
            for j in range(ng):
                eta = xi_pts[j]
                self.S[i, j]     = np.kron(_shape_funcs(xi, eta, dL, dW), I3)
                dSx_ij = np.kron(_shape_dxi(xi, eta, dL, dW), I3) / dL
                dSy_ij = np.kron(_shape_deta(xi, eta, dL, dW), I3) / dW
                self.dSx[i, j]   = dSx_ij
                self.dSy[i, j]   = dSy_ij
                self.d2Sx[i, j]  = np.kron(_shape_dxi2(xi, eta, dL, dW), I3) / dL**2
                self.d2Sy[i, j]  = np.kron(_shape_deta2(xi, eta, dL, dW), I3) / dW**2
                self.d2Sxy[i, j] = np.kron(_shape_dxieta(xi, eta, dL, dW), I3) / (dL * dW)
                self.gw[i, j]    = wts[i] * wts[j] * self.detJ
                self.A1[i, j]    = dSx_ij.T @ dSx_ij
                self.A2[i, j]    = dSy_ij.T @ dSy_ij
                self.A3[i, j]    = dSx_ij.T @ dSy_ij + dSy_ij.T @ dSx_ij


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

        self.Ex, self.Ey = Ex, Ey

        # Membrane plane-stress constitutive (orthotropic; reduces to isotropic).
        # The isotropic convention (Ey==Ex and G_xy unset) keeps the exact legacy
        # expression so the bit-exact warp_fsi golden is preserved by construction.
        if G_xy is None and Ey == Ex:
            G_xy = Ex / (2.0 * (1.0 + nu_xy))
            self.Dm = Ex / (1 - nu_xy**2) * np.array([
                [1,     nu_xy, 0],
                [nu_xy, 1,     0],
                [0,     0,     (1-nu_xy)/2]
            ])
        else:
            # orthotropic plane stress: nu_yx from reciprocity nu_xy/Ex = nu_yx/Ey
            if G_xy is None:
                G_xy = Ex / (2.0 * (1.0 + nu_xy))
            nu_yx = nu_xy * Ey / Ex
            denom = 1.0 - nu_xy * nu_yx
            self.Dm = np.array([
                [Ex / denom,         nu_yx * Ex / denom, 0.0],
                [nu_xy * Ey / denom, Ey / denom,         0.0],
                [0.0,                0.0,                G_xy]
            ])
        self.G_xy = G_xy
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
        """
        ed = self._elems[e]
        dofs = self._elem_dofs(e)
        q_e = q_global[dofs]
        ng = self.n_gauss

        Q_mem = np.zeros(NDOF_ELEM)
        Q_bend = np.zeros(NDOF_ELEM)
        K_mem = np.zeros((NDOF_ELEM, NDOF_ELEM))
        K_bend = np.zeros((NDOF_ELEM, NDOF_ELEM))

        for i in range(ng):
            for j in range(ng):
                w = ed.gw[i, j]

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
                eps_v = np.array([eps_xx, eps_yy, gam_xy])

                deps = np.zeros((3, NDOF_ELEM))
                deps[0] = dSx.T @ dx_r
                deps[1] = dSy.T @ dy_r
                deps[2] = dSx.T @ dy_r + dSy.T @ dx_r

                A1 = ed.A1[i, j]
                A2 = ed.A2[i, j]
                A3 = ed.A3[i, j]

                Dm_eps = self.Dm @ eps_v
                Q_mem += w * (deps.T @ Dm_eps)

                K_mem += w * (A1 * Dm_eps[0] + A2 * Dm_eps[1] + A3 * Dm_eps[2])
                K_mem += w * (deps.T @ self.Dm @ deps)

                # ── Bending (Kirchhoff-Love curvature) ──
                if self.mode != 'membrane':
                    n_vec = np.cross(dx_r, dy_r)
                    norm_n = np.linalg.norm(n_vec)
                    if norm_n < 1e-15:
                        continue
                    n_hat = n_vec / norm_n
                    inv_norm = 1.0 / norm_n

                    kxx = n_hat @ d2x_r
                    kyy = n_hat @ d2y_r
                    kxy = n_hat @ d2xy_r
                    k_v = np.array([kxx, kyy, 2.0 * kxy])

                    P = np.eye(3) - np.outer(n_hat, n_hat)

                    # Vectorized bending Jacobian:
                    # dn = -skew(dy_r) @ dSx + skew(dx_r) @ dSy
                    skew_dx = np.array([[0, -dx_r[2], dx_r[1]],
                                        [dx_r[2], 0, -dx_r[0]],
                                        [-dx_r[1], dx_r[0], 0]])
                    skew_dy = np.array([[0, -dy_r[2], dy_r[1]],
                                        [dy_r[2], 0, -dy_r[0]],
                                        [-dy_r[1], dy_r[0], 0]])
                    dn = -skew_dy @ dSx + skew_dx @ dSy  # (3, 36)
                    dn_hat = (P @ dn) * inv_norm  # (3, 36)

                    dk = np.empty((3, NDOF_ELEM))
                    dk[0] = d2x_r @ dn_hat + n_hat @ d2Sx
                    dk[1] = d2y_r @ dn_hat + n_hat @ d2Sy
                    dk[2] = 2.0 * (d2xy_r @ dn_hat + n_hat @ d2Sxy)

                    Dk_k = self.Dk @ k_v
                    Q_bend += w * (dk.T @ Dk_k)

                    K_bend += w * (dk.T @ self.Dk @ dk)

        Qe = Q_mem * self.h + Q_bend
        Kt = K_mem * self.h + K_bend
        return Qe, Kt

    def _elem_forces(self, e, q_global):
        Qe, _ = self._elem_forces_and_tangent(e, q_global)
        return Qe

    def _elem_forces_separated(self, e, q_global):
        """Return (Q_mem, Q_bend, Kt) — separated membrane and bending forces.

        Q_mem includes h factor (dimensional membrane force).
        Q_bend is dimensional bending force.
        """
        ed = self._elems[e]
        q_e = q_global[self._elem_dofs(e)]
        ng = self.n_gauss

        Q_mem_e = np.zeros(NDOF_ELEM)
        Q_bend_e = np.zeros(NDOF_ELEM)
        K_mem = np.zeros((NDOF_ELEM, NDOF_ELEM))
        K_bend = np.zeros((NDOF_ELEM, NDOF_ELEM))

        for i in range(ng):
            for j in range(ng):
                w = ed.gw[i, j]
                dSx = ed.dSx[i, j]
                dSy = ed.dSy[i, j]
                d2Sx = ed.d2Sx[i, j]
                d2Sy = ed.d2Sy[i, j]
                d2Sxy = ed.d2Sxy[i, j]

                dx_r = dSx @ q_e
                dy_r = dSy @ q_e

                # Membrane
                eps_xx = 0.5 * (dx_r @ dx_r - 1.0)
                eps_yy = 0.5 * (dy_r @ dy_r - 1.0)
                gam_xy = dx_r @ dy_r
                eps_v = np.array([eps_xx, eps_yy, gam_xy])

                deps = np.zeros((3, NDOF_ELEM))
                deps[0] = dSx.T @ dx_r
                deps[1] = dSy.T @ dy_r
                deps[2] = dSx.T @ dy_r + dSy.T @ dx_r

                A1 = ed.A1[i, j]
                A2 = ed.A2[i, j]
                A3 = ed.A3[i, j]

                Dm_eps = self.Dm @ eps_v
                Q_mem_e += w * (deps.T @ Dm_eps)
                K_mem += w * (A1 * Dm_eps[0] + A2 * Dm_eps[1] + A3 * Dm_eps[2])
                K_mem += w * (deps.T @ self.Dm @ deps)

                # Bending
                if self.mode != 'membrane':
                    d2x_r = d2Sx @ q_e
                    d2y_r = d2Sy @ q_e
                    d2xy_r = d2Sxy @ q_e

                    n_vec = np.cross(dx_r, dy_r)
                    norm_n = np.linalg.norm(n_vec)
                    if norm_n < 1e-15:
                        continue
                    n_hat = n_vec / norm_n
                    inv_norm = 1.0 / norm_n

                    kxx = n_hat @ d2x_r
                    kyy = n_hat @ d2y_r
                    kxy = n_hat @ d2xy_r
                    k_v = np.array([kxx, kyy, 2.0 * kxy])

                    P = np.eye(3) - np.outer(n_hat, n_hat)

                    skew_dx = np.array([[0, -dx_r[2], dx_r[1]],
                                        [dx_r[2], 0, -dx_r[0]],
                                        [-dx_r[1], dx_r[0], 0]])
                    skew_dy = np.array([[0, -dy_r[2], dy_r[1]],
                                        [dy_r[2], 0, -dy_r[0]],
                                        [-dy_r[1], dy_r[0], 0]])
                    dn = -skew_dy @ dSx + skew_dx @ dSy
                    dn_hat = (P @ dn) * inv_norm

                    dk = np.empty((3, NDOF_ELEM))
                    dk[0] = d2x_r @ dn_hat + n_hat @ d2Sx
                    dk[1] = d2y_r @ dn_hat + n_hat @ d2Sy
                    dk[2] = 2.0 * (d2xy_r @ dn_hat + n_hat @ d2Sxy)

                    Dk_k = self.Dk @ k_v
                    Q_bend_e += w * (dk.T @ Dk_k)
                    K_bend += w * (dk.T @ self.Dk @ dk)

        Q_mem_e *= self.h
        K_mem_scaled = K_mem * self.h
        Kt = K_mem_scaled + K_bend
        return Q_mem_e, Q_bend_e, Kt, K_mem_scaled

    def _elem_forces_sep(self, e, q_global):
        """Forces-only element assembly (Q_mem, Q_bend) — skips the O(ndof²)
        tangent-stiffness work. Used by the 2×/step Newmark force callback."""
        ed = self._elems[e]
        q_e = q_global[self._elem_dofs(e)]
        ng = self.n_gauss
        Q_mem_e = np.zeros(NDOF_ELEM)
        Q_bend_e = np.zeros(NDOF_ELEM)
        membrane_only = (self.mode == 'membrane')
        for i in range(ng):
            for j in range(ng):
                w = ed.gw[i, j]
                dSx = ed.dSx[i, j]; dSy = ed.dSy[i, j]
                dx_r = dSx @ q_e
                dy_r = dSy @ q_e
                eps_v = np.array([0.5 * (dx_r @ dx_r - 1.0),
                                  0.5 * (dy_r @ dy_r - 1.0),
                                  dx_r @ dy_r])
                deps0 = dSx.T @ dx_r
                deps1 = dSy.T @ dy_r
                deps2 = dSx.T @ dy_r + dSy.T @ dx_r
                Dm_eps = self.Dm @ eps_v
                Q_mem_e += w * (deps0 * Dm_eps[0] + deps1 * Dm_eps[1] + deps2 * Dm_eps[2])

                if not membrane_only:
                    d2Sx = ed.d2Sx[i, j]; d2Sy = ed.d2Sy[i, j]; d2Sxy = ed.d2Sxy[i, j]
                    d2x_r = d2Sx @ q_e; d2y_r = d2Sy @ q_e; d2xy_r = d2Sxy @ q_e
                    n_vec = np.cross(dx_r, dy_r)
                    norm_n = np.linalg.norm(n_vec)
                    if norm_n < 1e-15:
                        continue
                    n_hat = n_vec / norm_n
                    k_v = np.array([n_hat @ d2x_r, n_hat @ d2y_r, 2.0 * (n_hat @ d2xy_r)])
                    P = np.eye(3) - np.outer(n_hat, n_hat)
                    skew_dx = np.array([[0, -dx_r[2], dx_r[1]],
                                        [dx_r[2], 0, -dx_r[0]],
                                        [-dx_r[1], dx_r[0], 0]])
                    skew_dy = np.array([[0, -dy_r[2], dy_r[1]],
                                        [dy_r[2], 0, -dy_r[0]],
                                        [-dy_r[1], dy_r[0], 0]])
                    dn = -skew_dy @ dSx + skew_dx @ dSy
                    dn_hat = (P @ dn) / norm_n
                    dk0 = d2x_r @ dn_hat + n_hat @ d2Sx
                    dk1 = d2y_r @ dn_hat + n_hat @ d2Sy
                    dk2 = 2.0 * (d2xy_r @ dn_hat + n_hat @ d2Sxy)
                    Dk_k = self.Dk @ k_v
                    Q_bend_e += w * (dk0 * Dk_k[0] + dk1 * Dk_k[1] + dk2 * Dk_k[2])
        Q_mem_e *= self.h
        return Q_mem_e, Q_bend_e

    def _internal_forces_separated(self, q=None):
        """Return (Q_mem_global, Q_bend_global) — separated global forces."""
        if q is None:
            q = self.q
        Q_mem = np.zeros(self.ndof)
        Q_bend = np.zeros(self.ndof)
        for e in range(self.ne):
            Qm_e, Qb_e = self._elem_forces_sep(e, q)
            dofs = self._elem_dofs(e)
            Q_mem[dofs] += Qm_e
            Q_bend[dofs] += Qb_e
        return Q_mem, Q_bend

    def _tangent_K_mem(self, q=None):
        """Assemble GLOBAL membrane-only tangent stiffness K_mem.

        MATLAB's solve_structure uses dq_Qe_global = membrane K only in the
        Newmark operator (K_bend appears only via Q_bend averaging at stage 1).
        Python's step_newmark mirrors this — use K_mem (not K_mem + K_bend) as
        the damping operator stiffness.
        """
        if q is None:
            q = self.q
        nb = self.ne
        all_rows = np.empty(nb * NDOF_ELEM * NDOF_ELEM, dtype=np.int64)
        all_cols = np.empty_like(all_rows)
        all_vals = np.empty(nb * NDOF_ELEM * NDOF_ELEM, dtype=float)
        ptr = 0
        ng = self.n_gauss
        for e in range(self.ne):
            ed = self._elems[e]
            q_e = q[self._elem_dofs(e)]
            K_mem = np.zeros((NDOF_ELEM, NDOF_ELEM))
            for i in range(ng):
                for j in range(ng):
                    w = ed.gw[i, j]
                    dSx = ed.dSx[i, j]; dSy = ed.dSy[i, j]
                    dx_r = dSx @ q_e; dy_r = dSy @ q_e
                    eps_v = np.array([0.5 * (dx_r @ dx_r - 1.0),
                                      0.5 * (dy_r @ dy_r - 1.0),
                                      dx_r @ dy_r])
                    deps = np.empty((3, NDOF_ELEM))
                    deps[0] = dSx.T @ dx_r
                    deps[1] = dSy.T @ dy_r
                    deps[2] = dSx.T @ dy_r + dSy.T @ dx_r
                    Dm_eps = self.Dm @ eps_v
                    K_mem += w * (ed.A1[i, j] * Dm_eps[0] + ed.A2[i, j] * Dm_eps[1]
                                  + ed.A3[i, j] * Dm_eps[2])
                    K_mem += w * (deps.T @ self.Dm @ deps)
            K_mem *= self.h
            dofs = self._elem_dofs(e)
            n = NDOF_ELEM
            block = n * n
            all_rows[ptr:ptr + block] = np.repeat(dofs, n)
            all_cols[ptr:ptr + block] = np.tile(dofs, n)
            all_vals[ptr:ptr + block] = K_mem.ravel()
            ptr += block
        return coo_matrix((all_vals, (all_rows, all_cols)),
                          shape=(self.ndof, self.ndof)).tocsc()

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
            # Unrolled assembly: only store non-zero entries
            Kt_flat = Kt_e.ravel()
            mask = np.abs(Kt_flat) > 1e-30
            idx = np.where(mask)[0]
            a_idx = idx // NDOF_ELEM
            b_idx = idx % NDOF_ELEM
            rows.extend(dofs[a_idx].tolist())
            cols.extend(dofs[b_idx].tolist())
            vals.extend(Kt_flat[idx].tolist())
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

    def set_bc_pinned(self, nodes_bc):
        """Pinned BC: fix position + y-gradient, leave x-gradient free.

        Matches Yamano et al. pinned leading-edge condition:
          node_r_0  → fix rx, ry, rz
          node_dxr_0 → free (x-gradient unconstrained)
          node_dyr_0 → fix dy_rx, dy_ry, dy_rz
        """
        for n in nodes_bc:
            base = n * NDOF_NODE
            for k in range(3):          # rx, ry, rz
                self._bc_dofs.add(base + k)
            for k in range(6, 9):       # dy_rx, dy_ry, dy_rz
                self._bc_dofs.add(base + k)

    def gravity_load(self, g=9.81):
        """Consistent gravity body force: Qf = integral S^T * F_in * h dA."""
        F_in = np.array([0.0, 0.0, -self.rho * g])
        return self.distributed_load(F_in)

    def distributed_load(self, F_body):
        """Consistent nodal force from uniform body force density F_body (N/m³).

        Qf = integral S^T * F_body * h dA  (Gauss-integrated shape functions).
        """
        Qf = np.zeros(self.ndof)
        for e in range(self.ne):
            ed = self._elems[e]
            dofs = self._elem_dofs(e)
            for i in range(self.n_gauss):
                for j in range(self.n_gauss):
                    w = ed.gw[i, j] * self.h
                    S = ed.S[i, j]
                    Qf[dofs] += w * (S.T @ F_body)
        return Qf

    # ─── Prescribed (time-varying) boundary motion ───
    def set_prescribed_motion(self, nodes_presc, callback):
        """Drive boundary nodes kinematically: callback(t) -> (q_b, dq_b, ddq_b)
        over the prescribed DOFs (9 per node, node-major order of
        ``sorted(nodes_presc)``).

        Implementation: the prescribed DOFs are eliminated like static BCs;
        elastic coupling into free DOFs flows exactly through Qe(q_full); the
        inertial coupling is added to the RHS as -M[free, presc] @ ddq_b(t).
        ``step_newmark`` must then be called with its ``t_end`` argument.
        With ``callback=None`` semantics are identical to ``set_bc``.
        """
        nodes = sorted(int(n) for n in nodes_presc)
        self.set_bc(nodes, fix_slopes=True)
        self._presc_dofs = np.array([9 * n + d for n in nodes
                                     for d in range(9)], dtype=np.int64)
        self._presc_cb = callback
        self._M_fp = None  # lazy: M[free, presc] (depends on final BC set)

    def set_added_mass_matrix(self, M_added):
        """Set the fluid added-mass matrix (Mf1 contribution).

        The effective mass used in time integration is M_eff = M - M_added.
        This matches MATLAB: m_global_struct.M_global = M + J_1 - Qf_p_mat
        """
        self._M_added = M_added

    # ─── Implicit Newmark-beta time step ───
    def step_newmark(self, F_ext, dt, alpha_v=0.5, newton_tol=1e-8, max_newton=20,
                     t_end=None):
        """One implicit Newmark-beta step (trapezoidal rule when alpha_v=0.5).

        Following MATLAB reference: new_X_func_FAST.m + solve_structure.m
        Uses analytical tangent stiffness from _elem_forces_and_tangent.

        Two-stage solve:
          Stage 0: tangent at q(n), solve for predictor (Qe_n = Q_mem_n + Q_bend_n)
          Stage 1: MATCHES MATLAB: Qe = Q_mem_n + (Q_bend_n + Q_bend_np1)/2
                   (membrane NOT averaged, bending IS averaged)
        """
        bc = np.array(sorted(self._bc_dofs), dtype=np.int32)
        free = np.setdiff1d(np.arange(self.ndof), bc)
        nf = len(free)

        q_n = self.q.copy()
        dq_n = self.dq.copy()

        # Prescribed boundary motion: inertial coupling enters the RHS as
        # -M[free, presc] @ ddq_b(t_end); elastic coupling flows through
        # Qe(q_full) below (q_p1 / final state carry the new boundary values).
        _presc = getattr(self, '_presc_cb', None)
        _qb = _dqb = None
        if _presc is not None and t_end is not None:
            _qb, _dqb, _ddqb = _presc(t_end)
            pd = self._presc_dofs
            if self._M_fp is None:
                self._M_fp = self.M[np.ix_(free, pd)].tocsc()
            F_ext = F_ext.copy()
            F_ext[free] -= self._M_fp @ np.asarray(_ddqb, dtype=float)

        # Stage 0: separate membrane and bending forces
        Q_mem_n, Q_bend_n = self._internal_forces_separated(q_n)
        Qe_n = Q_mem_n + Q_bend_n
        # Newmark damping operator uses K_mem ONLY (matches MATLAB dq_Qe_global).
        # K_bend influences the result via Q_bend averaging at stage 1.
        Kt_n = self._tangent_K_mem(q_n)

        c_damp = 2.0 if self.structural_damping == 0 else 1.0

        # Extract free-free submatrices (sparse)
        Kt_ff = Kt_n[np.ix_(free, free)].tocsc()
        M_ff = self.M[np.ix_(free, free)].tocsc()
        # Add fluid added-mass if available (MATLAB: M_eff = M - Qf_p_mat)
        if hasattr(self, '_M_added') and self._M_added is not None:
            M_ff = M_ff - self._M_added[np.ix_(free, free)]
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

        # Stage 0: Q_global = F - (Q_mem_n + Q_bend_n)
        Q_global = F_ext - Qe_n
        rhs0 = np.zeros(2 * nf)
        rhs0[nf:] = Q_global[free]

        A1_inv_A2Xn = spsolve(A1, A2Xn)
        X_p1_free = A1_inv_A2Xn + dt * spsolve(A1, rhs0)

        # Stage 1: compute Q_bend at predicted q_{n+1}
        # MATLAB corrector: Qe = Q_mem_n + (Q_bend_n + Q_bend_np1)/2
        q_p1 = q_n.copy()
        q_p1[free] = X_p1_free[:nf]
        if _qb is not None:
            q_p1[self._presc_dofs] = _qb   # boundary at end-of-step for averaging
        _, Q_bend_p1 = self._internal_forces_separated(q_p1)

        # MATLAB: Q_global = Qf - (Q_mem_n + (Q_bend_n + Q_bend_np1)/2)
        Qe_corr = Q_mem_n + (Q_bend_n + Q_bend_p1) / 2.0
        Q_global2 = F_ext - Qe_corr
        rhs1 = np.zeros(2 * nf)
        rhs1[nf:] = Q_global2[free]

        X_new_free = A1_inv_A2Xn + dt * spsolve(A1, rhs1)

        self.q[free] = X_new_free[:nf]
        self.dq[free] = X_new_free[nf:]
        if _qb is not None:
            self.q[self._presc_dofs] = _qb
            self.dq[self._presc_dofs] = _dqb

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
