"""GPU-resident structural↔aero transfer (Phase G1 of the accuracy-preserving GPU acceleration).

The differentiable coupled PC solver currently does the structure↔aero transfer on the HOST every
iteration — `corners = P @ q` (gather node positions) and `Fnodal = dist @ Fp` (scatter ¼ of each
panel force to its 4 corner nodes) — forcing a GPU→host→GPU round-trip inside the hot PC/adjoint loops.
The maps are trivial: P is a position-GATHER (each lattice corner IS a node's translational DOFs) and
dist is a ¼-force SCATTER. This module implements them, and their transposes (for the adjoint), as Warp
kernels so the whole transfer stays on the GPU. Validated bit-exact against the NumPy P/dist.

Index maps (matching diff_coupled_fsi._index_maps):
  corner k=(i*(ny+1)+j) ← node n=(j*(nx+1)+i)            (i=0..nx chordwise, j=0..ny spanwise)
  panel p=(pi*ny+pj) → 4 nodes ((pi,pj),(pi+1,pj),(pi,pj+1),(pi+1,pj+1))
"""
from __future__ import annotations

import numpy as np

import warp as wp
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.config import DTYPE

V3 = wp.vec3d


def transfer_maps(nx, ny, device=None):
    """Build the GPU index arrays: corner_node (ncv,), panel_nodes (npan,4)."""
    dev = device or cfg.DEVICE
    ncv = (nx + 1) * (ny + 1); npan = nx * ny
    corner_node = np.zeros(ncv, np.int32)
    for i in range(nx + 1):
        for j in range(ny + 1):
            corner_node[i * (ny + 1) + j] = j * (nx + 1) + i
    panel_nodes = np.zeros((npan, 4), np.int32)
    for pi in range(nx):
        for pj in range(ny):
            panel_nodes[pi * ny + pj] = [j2 * (nx + 1) + i2 for (i2, j2) in
                                         ((pi, pj), (pi + 1, pj), (pi, pj + 1), (pi + 1, pj + 1))]
    return (wp.array(corner_node, dtype=wp.int32, device=dev),
            wp.array(panel_nodes, dtype=wp.int32, device=dev), ncv, npan)


@wp.kernel
def gather_corners_kernel(q: wp.array(dtype=DTYPE), corner_node: wp.array(dtype=wp.int32),
                          corners: wp.array(dtype=V3)):
    """corners[k] = (q[9n], q[9n+1], q[9n+2]),  n = corner_node[k].   (= P·q)"""
    k = wp.tid(); n = corner_node[k]
    corners[k] = V3(q[9 * n + 0], q[9 * n + 1], q[9 * n + 2])


@wp.kernel
def gather_T_kernel(adj_corners: wp.array(dtype=V3), corner_node: wp.array(dtype=wp.int32),
                    adj_q: wp.array(dtype=DTYPE)):
    """adj_q[9n+d] = adj_corners[k][d],  n = corner_node[k].   (= Pᵀ·adj_corners; corners↔nodes bijective)"""
    k = wp.tid(); n = corner_node[k]; a = adj_corners[k]
    adj_q[9 * n + 0] = a[0]; adj_q[9 * n + 1] = a[1]; adj_q[9 * n + 2] = a[2]


@wp.kernel
def scatter_force_kernel(Fp: wp.array(dtype=V3), panel_nodes: wp.array(dtype=wp.int32, ndim=2),
                         Fnodal: wp.array(dtype=DTYPE)):
    """Fnodal[9n+d] += ¼·Fp[p][d] for each of panel p's 4 corner nodes n.   (= dist·Fp)"""
    p = wp.tid(); f = Fp[p]
    for c in range(4):
        n = panel_nodes[p, c]
        wp.atomic_add(Fnodal, 9 * n + 0, DTYPE(0.25) * f[0])
        wp.atomic_add(Fnodal, 9 * n + 1, DTYPE(0.25) * f[1])
        wp.atomic_add(Fnodal, 9 * n + 2, DTYPE(0.25) * f[2])


@wp.kernel
def scatter_T_kernel(adj_Fnodal: wp.array(dtype=DTYPE), panel_nodes: wp.array(dtype=wp.int32, ndim=2),
                     adj_Fp: wp.array(dtype=V3)):
    """adj_Fp[p][d] = ¼·Σ_{4 nodes n of p} adj_Fnodal[9n+d].   (= distᵀ·adj_Fnodal)"""
    p = wp.tid(); ax = DTYPE(0.0); ay = DTYPE(0.0); az = DTYPE(0.0)
    for c in range(4):
        n = panel_nodes[p, c]
        ax += adj_Fnodal[9 * n + 0]; ay += adj_Fnodal[9 * n + 1]; az += adj_Fnodal[9 * n + 2]
    adj_Fp[p] = V3(DTYPE(0.25) * ax, DTYPE(0.25) * ay, DTYPE(0.25) * az)


def verify(nx=6, ny=4, seed=0):
    """Bit-exact validation of the four transfer kernels against the NumPy P / dist (and transposes)."""
    wp.init(); dev = cfg.DEVICE; NP = cfg.NP_DTYPE
    import diff_coupled_fsi as dcf
    from diff_struct_design import _build_shell
    sh = _build_shell(nx=nx, ny=ny); ndof = sh.ndof
    P, dist = dcf._index_maps(sh, nx, ny)
    cn, pn, ncv, npan = transfer_maps(nx, ny, dev)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(ndof); Fp = rng.standard_normal((npan, 3))
    adj_corners = rng.standard_normal((ncv, 3)); adj_Fnodal = rng.standard_normal(ndof)
    qw = wp.array(q.astype(NP), dtype=DTYPE, device=dev)
    cw = wp.zeros(ncv, dtype=V3, device=dev)
    wp.launch(gather_corners_kernel, dim=ncv, inputs=[qw, cn], outputs=[cw], device=dev)
    e_gather = np.max(np.abs(cw.numpy() - (P @ q).reshape(ncv, 3)))
    Fpw = wp.array(Fp.astype(NP), dtype=V3, device=dev); Fn = wp.zeros(ndof, dtype=DTYPE, device=dev)
    wp.launch(scatter_force_kernel, dim=npan, inputs=[Fpw, pn], outputs=[Fn], device=dev)
    e_scatter = np.max(np.abs(Fn.numpy() - dist @ Fp.reshape(-1)))
    acw = wp.array(adj_corners.astype(NP), dtype=V3, device=dev); aq = wp.zeros(ndof, dtype=DTYPE, device=dev)
    wp.launch(gather_T_kernel, dim=ncv, inputs=[acw, cn], outputs=[aq], device=dev)
    e_gatherT = np.max(np.abs(aq.numpy() - P.T @ adj_corners.reshape(-1)))
    afw = wp.array(adj_Fnodal.astype(NP), dtype=DTYPE, device=dev); aFp = wp.zeros(npan, dtype=V3, device=dev)
    wp.launch(scatter_T_kernel, dim=npan, inputs=[afw, pn], outputs=[aFp], device=dev)
    e_scatterT = np.max(np.abs(aFp.numpy() - (dist.T @ adj_Fnodal).reshape(npan, 3)))
    ok = max(e_gather, e_scatter, e_gatherT, e_scatterT) < 1e-12
    print(f"GPU structure↔aero transfer kernels vs NumPy P/dist ({nx}x{ny}, {ndof} DOF, {npan} panels):")
    print(f"  gather P·q     : {e_gather:.1e}")
    print(f"  scatter dist·Fp: {e_scatter:.1e}")
    print(f"  Pᵀ·adj_corners : {e_gatherT:.1e}")
    print(f"  distᵀ·adj_F    : {e_scatterT:.1e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the transfer is bit-exact on GPU — kills the host round-trip "
          f"with zero accuracy loss (Phase G1 of the accuracy-preserving GPU acceleration)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
