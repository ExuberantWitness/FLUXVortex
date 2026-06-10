"""Batched aero<->structure coupling kernels.

- csr_matvec: batched y = A·x with a SHARED constant sparse A (CSR). Used for the
  load transfer F_nodal = _P_load · dp_n (panel pressures -> consistent nodal force;
  _P_load is the chord-p_interp + Gauss projection, CPU-validated bit-exact to
  MATLAB Qf_p_global).
- struct_vel_kernel: V_struct[panel] = S(xi,eta)·dq_elem (structural velocity at
  UVLM collocation points), mirrors _compute_structural_velocity_at_colloc.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from . import config

DTYPE = config.DTYPE
VEC3 = config.VEC3


@wp.kernel
def csr_matvec_kernel(rowptr: wp.array(dtype=wp.int32, ndim=1),   # (nrows+1,)
                      colidx: wp.array(dtype=wp.int32, ndim=1),   # (nnz,)
                      vals: wp.array(dtype=DTYPE, ndim=1),        # (nnz,) shared
                      x: wp.array(dtype=DTYPE, ndim=2),           # (B, ncols)
                      y: wp.array(dtype=DTYPE, ndim=2)):          # (B, nrows) out
    e, row = wp.tid()
    s = DTYPE(0.0)
    for k in range(rowptr[row], rowptr[row + 1]):
        s = s + vals[k] * x[e, colidx[k]]
    y[e, row] = s


class CSR:
    """Shared constant CSR matrix on device (rows×cols)."""
    def __init__(self, A_scipy, device=None):
        device = device or config.DEVICE
        A = A_scipy.tocsr()
        NP = config.NP_DTYPE
        self.nrows, self.ncols = A.shape
        self.rowptr = wp.array(A.indptr.astype(np.int32), dtype=wp.int32, device=device)
        self.colidx = wp.array(A.indices.astype(np.int32), dtype=wp.int32, device=device)
        self.vals = wp.array(A.data.astype(NP), dtype=DTYPE, device=device)
        self.device = device

    def matvec(self, x_wp, out=None):
        B = x_wp.shape[0]
        if out is None:
            out = wp.zeros((B, self.nrows), dtype=DTYPE, device=self.device)
        wp.launch(csr_matvec_kernel, dim=(B, self.nrows),
                  inputs=[self.rowptr, self.colidx, self.vals, x_wp],
                  outputs=[out], device=self.device)
        return out


@wp.kernel
def struct_vel_kernel(dq: wp.array(dtype=DTYPE, ndim=2),          # (B, ndof)
                      pSmat: wp.array(dtype=DTYPE, ndim=3),       # (P, 3, 36) panel shape mats
                      pedofs: wp.array(dtype=wp.int32, ndim=2),   # (P, 36) elem DOFs per panel
                      Vout: wp.array(dtype=VEC3, ndim=2)):        # (B, P) out
    e, p = wp.tid()
    v = wp.vector(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    for a in range(36):
        qa = dq[e, pedofs[p, a]]
        for d in range(3):
            v[d] = v[d] + pSmat[p, d, a] * qa
    Vout[e, p] = v


class CouplingConstants:
    """Per-panel shape-function maps for structural velocity / displacement transfer."""
    def __init__(self, solver, device=None):
        device = device or config.DEVICE
        NP = config.NP_DTYPE
        from ..ancf_shell import _shape_funcs
        nc, ns = solver._nx, solver._ny
        P = nc * ns
        pSmat = np.zeros((P, 3, 36))
        pedofs = np.zeros((P, 36), dtype=np.int32)
        I3 = np.eye(3)
        for i in range(nc):
            for j in range(ns):
                p = i * ns + j
                e = solver._panel_to_elem[i, j]
                xi, eta = solver._panel_xi_eta[i, j]
                dL, dW = solver.shell._dL[e], solver.shell._dW[e]
                S = np.kron(_shape_funcs(xi, eta, dL, dW), I3)   # (3,36)
                pSmat[p] = S
                pedofs[p] = solver.shell._elem_dofs(e)
        self.P = P
        self.pSmat = wp.array(pSmat.astype(NP), dtype=DTYPE, device=device)
        self.pedofs = wp.array(pedofs, dtype=wp.int32, device=device)
        self.device = device

    def struct_velocity(self, dq_wp, out=None):
        B = dq_wp.shape[0]
        if out is None:
            out = wp.zeros((B, self.P), dtype=VEC3, device=self.device)
        wp.launch(struct_vel_kernel, dim=(B, self.P),
                  inputs=[dq_wp, self.pSmat, self.pedofs], outputs=[out], device=self.device)
        return out
