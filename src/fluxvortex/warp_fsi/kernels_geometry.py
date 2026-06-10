"""Exact MATLAB Sc-matrix geometry transfer on device (the geom+AIC fix).

The long-time CPU deficit (t*=1.0 ratio 0.579->0.776) closes when the UVLM
coupling geometry is built from MATLAB's exact bicubic-Hermite shape-function
matrices instead of closest-node corners + point-eval collocation:

    corners_k = Sc_panel_k · q   (k=1..4)          (B,P,4) VEC3
    colloc    = Sc_col · q                          (B,P)   VEC3
    normals   = cross(Sc31·q, Sc24·q) / |·|         (B,P)   VEC3   (=generate_dt_n_vec.m)
    V_struct  = Sc_col · dq                         (B,P)   VEC3

All Sc matrices are SHARED constants (same mesh -> one upload). The deformed
geometry then feeds build_aic_batched every fluid solve, so the AIC rebuild is
automatic on GPU (the kernel takes geometry as input). Validated bit-exact vs
the CPU `standalone_hybrid_solver._sc_geometry_update`.
"""
from __future__ import annotations
import os
import numpy as np
import warp as wp
from . import config
from .kernels_coupling import CSR

DTYPE = config.DTYPE
VEC3 = config.VEC3


@wp.kernel
def flat_to_vec3_kernel(flat: wp.array(dtype=DTYPE, ndim=2),   # (B, 3P)
                        out: wp.array(dtype=VEC3, ndim=2)):    # (B, P)
    e, p = wp.tid()
    out[e, p] = wp.vector(flat[e, 3 * p], flat[e, 3 * p + 1], flat[e, 3 * p + 2])


@wp.kernel
def pack_corners_kernel(f0: wp.array(dtype=DTYPE, ndim=2),     # 4× (B, 3P)
                        f1: wp.array(dtype=DTYPE, ndim=2),
                        f2: wp.array(dtype=DTYPE, ndim=2),
                        f3: wp.array(dtype=DTYPE, ndim=2),
                        out: wp.array(dtype=VEC3, ndim=3)):     # (B, P, 4)
    e, p = wp.tid()
    out[e, p, 0] = wp.vector(f0[e, 3 * p], f0[e, 3 * p + 1], f0[e, 3 * p + 2])
    out[e, p, 1] = wp.vector(f1[e, 3 * p], f1[e, 3 * p + 1], f1[e, 3 * p + 2])
    out[e, p, 2] = wp.vector(f2[e, 3 * p], f2[e, 3 * p + 1], f2[e, 3 * p + 2])
    out[e, p, 3] = wp.vector(f3[e, 3 * p], f3[e, 3 * p + 1], f3[e, 3 * p + 2])


@wp.kernel
def normals_kernel(r13: wp.array(dtype=DTYPE, ndim=2),         # (B, 3P)
                   r42: wp.array(dtype=DTYPE, ndim=2),         # (B, 3P)
                   nout: wp.array(dtype=VEC3, ndim=2),         # (B, P)
                   aout: wp.array(dtype=DTYPE, ndim=2)):       # (B, P) areas
    e, p = wp.tid()
    a = wp.vector(r13[e, 3 * p], r13[e, 3 * p + 1], r13[e, 3 * p + 2])
    b = wp.vector(r42[e, 3 * p], r42[e, 3 * p + 1], r42[e, 3 * p + 2])
    c = wp.cross(a, b)
    L = wp.length(c) + DTYPE(1.0e-30)
    nout[e, p] = c / L
    aout[e, p] = DTYPE(0.5) * L


class ScGeometry:
    """Device-side exact Sc-matrix geometry transfer (the geom+AIC fix).

    Holds the constant Sc CSR matrices and produces the deformed UVLM geometry
    (corners/colloc/normals/areas) and structural velocity at collocation from
    the batched structural state q/dq.
    """

    def __init__(self, nc, ns, npz_path=None, device=None):
        device = device or config.DEVICE
        if npz_path is None:
            npz_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'data', f'sc_geometry_{nc}x{ns}.npz')
        d = np.load(npz_path)
        from scipy.sparse import csr_matrix

        def _csr(name):
            return csr_matrix((d[name + '_data'], d[name + '_indices'],
                               d[name + '_indptr']), shape=tuple(d[name + '_shape']))

        self.nc, self.ns, self.P = nc, ns, nc * ns
        self.device = device
        self._col = CSR(_csr('Sc_mat_col_global'), device)
        self._panel = [CSR(_csr(f'Sc_mat_panel_global_{k}'), device) for k in (1, 2, 3, 4)]
        self._s31 = CSR(_csr('Sc_mat_31'), device)
        self._s24 = CSR(_csr('Sc_mat_24'), device)
        # scratch flat buffers are (re)allocated lazily per batch size
        self._B = None

    def _alloc(self, B):
        if self._B == B:
            return
        dev = self.device
        f = lambda: wp.zeros((B, 3 * self.P), dtype=DTYPE, device=dev)
        self._f = [f() for _ in range(4)]
        self._fcol = f()
        self._fr13 = f()
        self._fr42 = f()
        self.corners = wp.zeros((B, self.P, 4), dtype=VEC3, device=dev)
        self.colloc = wp.zeros((B, self.P), dtype=VEC3, device=dev)
        self.normals = wp.zeros((B, self.P), dtype=VEC3, device=dev)
        self.areas = wp.zeros((B, self.P), dtype=DTYPE, device=dev)
        self.vstruct = wp.zeros((B, self.P), dtype=VEC3, device=dev)
        self._B = B

    def update(self, q_wp):
        """Compute corners/colloc/normals/areas from q (B, ndof). Returns self."""
        B = q_wp.shape[0]
        self._alloc(B)
        dev = self.device
        for k in range(4):
            self._panel[k].matvec(q_wp, out=self._f[k])
        self._col.matvec(q_wp, out=self._fcol)
        self._s31.matvec(q_wp, out=self._fr13)
        self._s24.matvec(q_wp, out=self._fr42)
        wp.launch(pack_corners_kernel, dim=(B, self.P),
                  inputs=[self._f[0], self._f[1], self._f[2], self._f[3]],
                  outputs=[self.corners], device=dev)
        wp.launch(flat_to_vec3_kernel, dim=(B, self.P),
                  inputs=[self._fcol], outputs=[self.colloc], device=dev)
        wp.launch(normals_kernel, dim=(B, self.P),
                  inputs=[self._fr13, self._fr42],
                  outputs=[self.normals, self.areas], device=dev)
        return self

    def struct_velocity(self, dq_wp, out=None):
        """V_struct = Sc_col · dq, as (B, P) VEC3."""
        B = dq_wp.shape[0]
        self._alloc(B)
        self._col.matvec(dq_wp, out=self._fcol)
        if out is None:
            out = self.vstruct
        wp.launch(flat_to_vec3_kernel, dim=(B, self.P),
                  inputs=[self._fcol], outputs=[out], device=self.device)
        return out
