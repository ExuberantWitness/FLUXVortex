"""Batched ANCF shell element assembly kernels (membrane tangent K_mem).

Ports the math of ancf_shell._tangent_K_mem (the membrane tangent used by the
Newmark damping operator). 36x36 element blocks exceed Warp's matrix types, so:
  kernel A (env, elem, gauss): compute deps (3x36) and Dm_eps (3) -> scratch
  kernel B (env, elem, a, b):  K_mem_block[e,a,b] = Σ_g gw_g·h·(
        A1_g[a,b]·Dm_eps_g0 + A2_g[a,b]·Dm_eps_g1 + A3_g[a,b]·Dm_eps_g2
        + Σ_mn deps_g[m,a]·Dm[m,n]·deps_g[n,b])

Element constants (shared across envs, uploaded once) come from _ElementData:
  dSx,dSy : (ne, ngg, 3, 36)     A1,A2,A3 : (ne, ngg, 36, 36)     gw : (ne, ngg)
  edofs   : (ne, 36) int         Dm : (3,3)   h : scalar
State per env: q (B, ndof).
"""
from __future__ import annotations
import numpy as np
import warp as wp
from . import config

DTYPE = config.DTYPE
MAT33 = config.MAT33


@wp.kernel
def ancf_deps_kernel(q: wp.array(dtype=DTYPE, ndim=2),         # (B, ndof)
                     dSx: wp.array(dtype=DTYPE, ndim=4),        # (ne, ngg, 3, 36)
                     dSy: wp.array(dtype=DTYPE, ndim=4),
                     edofs: wp.array(dtype=wp.int32, ndim=2),   # (ne, 36)
                     Dm: MAT33,
                     deps: wp.array(dtype=DTYPE, ndim=4),       # (B, ne, ngg, 108=3*36) out
                     Dm_eps: wp.array(dtype=DTYPE, ndim=4)):    # (B, ne, ngg, 3) out
    e, el, g = wp.tid()
    # dx_r, dy_r = dSx@q_e, dSy@q_e
    dxr = wp.vector(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    dyr = wp.vector(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    for a in range(36):
        qa = q[e, edofs[el, a]]
        for d in range(3):
            dxr[d] = dxr[d] + dSx[el, g, d, a] * qa
            dyr[d] = dyr[d] + dSy[el, g, d, a] * qa
    eps0 = DTYPE(0.5) * (wp.dot(dxr, dxr) - DTYPE(1.0))
    eps1 = DTYPE(0.5) * (wp.dot(dyr, dyr) - DTYPE(1.0))
    eps2 = wp.dot(dxr, dyr)
    me0 = Dm[0, 0] * eps0 + Dm[0, 1] * eps1 + Dm[0, 2] * eps2
    me1 = Dm[1, 0] * eps0 + Dm[1, 1] * eps1 + Dm[1, 2] * eps2
    me2 = Dm[2, 0] * eps0 + Dm[2, 1] * eps1 + Dm[2, 2] * eps2
    Dm_eps[e, el, g, 0] = me0
    Dm_eps[e, el, g, 1] = me1
    Dm_eps[e, el, g, 2] = me2
    for a in range(36):
        d0 = DTYPE(0.0); d1 = DTYPE(0.0); d2 = DTYPE(0.0)
        for d in range(3):
            sx = dSx[el, g, d, a]
            sy = dSy[el, g, d, a]
            d0 = d0 + sx * dxr[d]
            d1 = d1 + sy * dyr[d]
            d2 = d2 + sx * dyr[d] + sy * dxr[d]
        deps[e, el, g, a] = d0           # row 0: [0..36)
        deps[e, el, g, 36 + a] = d1       # row 1: [36..72)
        deps[e, el, g, 72 + a] = d2       # row 2: [72..108)


@wp.kernel
def ancf_kmem_kernel(A1: wp.array(dtype=DTYPE, ndim=4),        # (ne, ngg, 36, 36)
                     A2: wp.array(dtype=DTYPE, ndim=4),
                     A3: wp.array(dtype=DTYPE, ndim=4),
                     gw: wp.array(dtype=DTYPE, ndim=2),         # (ne, ngg)
                     deps: wp.array(dtype=DTYPE, ndim=4),       # (B, ne, ngg, 108)
                     Dm_eps: wp.array(dtype=DTYPE, ndim=4),     # (B, ne, ngg, 3)
                     Dm: MAT33,
                     h: DTYPE,
                     ngg: int,
                     Kblk: wp.array(dtype=DTYPE, ndim=4)):      # (B, ne, 36, 36) out
    e, el, a, b = wp.tid()
    acc = DTYPE(0.0)
    for g in range(ngg):
        w = gw[el, g] * h
        # geometric: A_g[a,b]·Dm_eps_g
        geo = (A1[el, g, a, b] * Dm_eps[e, el, g, 0]
               + A2[el, g, a, b] * Dm_eps[e, el, g, 1]
               + A3[el, g, a, b] * Dm_eps[e, el, g, 2])
        # material: Σ_mn deps[m,a]·Dm[m,n]·deps[n,b]
        da0 = deps[e, el, g, a]; da1 = deps[e, el, g, 36 + a]; da2 = deps[e, el, g, 72 + a]
        db0 = deps[e, el, g, b]; db1 = deps[e, el, g, 36 + b]; db2 = deps[e, el, g, 72 + b]
        mat = (da0 * (Dm[0, 0] * db0 + Dm[0, 1] * db1 + Dm[0, 2] * db2)
               + da1 * (Dm[1, 0] * db0 + Dm[1, 1] * db1 + Dm[1, 2] * db2)
               + da2 * (Dm[2, 0] * db0 + Dm[2, 1] * db1 + Dm[2, 2] * db2))
        acc = acc + w * (geo + mat)
    Kblk[e, el, a, b] = acc


@wp.func
def _col3(arr: wp.array(dtype=DTYPE, ndim=4), el: int, g: int, a: int):
    return wp.vector(arr[el, g, 0, a], arr[el, g, 1, a], arr[el, g, 2, a])


@wp.kernel
def ancf_force_gauss_kernel(q: wp.array(dtype=DTYPE, ndim=2),
                            dSx: wp.array(dtype=DTYPE, ndim=4), dSy: wp.array(dtype=DTYPE, ndim=4),
                            d2Sx: wp.array(dtype=DTYPE, ndim=4), d2Sy: wp.array(dtype=DTYPE, ndim=4),
                            d2Sxy: wp.array(dtype=DTYPE, ndim=4),
                            edofs: wp.array(dtype=wp.int32, ndim=2),
                            Dm: MAT33, Dk: MAT33,
                            deps: wp.array(dtype=DTYPE, ndim=4),   # (B,ne,ngg,108) out
                            dk: wp.array(dtype=DTYPE, ndim=4),     # (B,ne,ngg,108) out
                            Dm_eps: wp.array(dtype=DTYPE, ndim=4), # (B,ne,ngg,3) out
                            Dk_k: wp.array(dtype=DTYPE, ndim=4)):  # (B,ne,ngg,3) out
    e, el, g = wp.tid()
    z = wp.vector(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    dxr = z; dyr = z; d2x = z; d2y = z; d2xy = z
    for a in range(36):
        qa = q[e, edofs[el, a]]
        dxr = dxr + _col3(dSx, el, g, a) * qa
        dyr = dyr + _col3(dSy, el, g, a) * qa
        d2x = d2x + _col3(d2Sx, el, g, a) * qa
        d2y = d2y + _col3(d2Sy, el, g, a) * qa
        d2xy = d2xy + _col3(d2Sxy, el, g, a) * qa
    # membrane stress
    eps0 = DTYPE(0.5) * (wp.dot(dxr, dxr) - DTYPE(1.0))
    eps1 = DTYPE(0.5) * (wp.dot(dyr, dyr) - DTYPE(1.0))
    eps2 = wp.dot(dxr, dyr)
    Dm_eps[e, el, g, 0] = Dm[0, 0]*eps0 + Dm[0, 1]*eps1 + Dm[0, 2]*eps2
    Dm_eps[e, el, g, 1] = Dm[1, 0]*eps0 + Dm[1, 1]*eps1 + Dm[1, 2]*eps2
    Dm_eps[e, el, g, 2] = Dm[2, 0]*eps0 + Dm[2, 1]*eps1 + Dm[2, 2]*eps2
    # bending: normal + curvature stress resultant
    nvec = wp.cross(dxr, dyr)
    nn = wp.length(nvec)
    nhat = nvec / nn
    k0 = wp.dot(nhat, d2x); k1 = wp.dot(nhat, d2y); k2 = DTYPE(2.0) * wp.dot(nhat, d2xy)
    Dk_k[e, el, g, 0] = Dk[0, 0]*k0 + Dk[0, 1]*k1 + Dk[0, 2]*k2
    Dk_k[e, el, g, 1] = Dk[1, 0]*k0 + Dk[1, 1]*k1 + Dk[1, 2]*k2
    Dk_k[e, el, g, 2] = Dk[2, 0]*k0 + Dk[2, 1]*k1 + Dk[2, 2]*k2
    for a in range(36):
        sxa = _col3(dSx, el, g, a); sya = _col3(dSy, el, g, a)
        # membrane strain gradient
        deps[e, el, g, a] = wp.dot(sxa, dxr)
        deps[e, el, g, 36 + a] = wp.dot(sya, dyr)
        deps[e, el, g, 72 + a] = wp.dot(sxa, dyr) + wp.dot(sya, dxr)
        # bending curvature gradient: dn_a = dxr×sya - dyr×sxa ; dn_hat=P·dn/nn
        dn = wp.cross(dxr, sya) - wp.cross(dyr, sxa)
        dnh = (dn - nhat * wp.dot(nhat, dn)) / nn
        dxx = _col3(d2Sx, el, g, a); dyy = _col3(d2Sy, el, g, a); dxy = _col3(d2Sxy, el, g, a)
        dk[e, el, g, a] = wp.dot(d2x, dnh) + wp.dot(nhat, dxx)
        dk[e, el, g, 36 + a] = wp.dot(d2y, dnh) + wp.dot(nhat, dyy)
        dk[e, el, g, 72 + a] = DTYPE(2.0) * (wp.dot(d2xy, dnh) + wp.dot(nhat, dxy))


@wp.kernel
def ancf_force_assemble_kernel(gw: wp.array(dtype=DTYPE, ndim=2),
                               deps: wp.array(dtype=DTYPE, ndim=4),
                               dk: wp.array(dtype=DTYPE, ndim=4),
                               Dm_eps: wp.array(dtype=DTYPE, ndim=4),
                               Dk_k: wp.array(dtype=DTYPE, ndim=4),
                               edofs: wp.array(dtype=wp.int32, ndim=2),
                               h: DTYPE, ngg: int,
                               Qmem: wp.array(dtype=DTYPE, ndim=2),    # (B,ndof) accumulate
                               Qbend: wp.array(dtype=DTYPE, ndim=2)):  # (B,ndof) accumulate
    e, el, a = wp.tid()
    am = DTYPE(0.0)
    ab = DTYPE(0.0)
    for g in range(ngg):
        w = gw[el, g]
        mem = (deps[e, el, g, a] * Dm_eps[e, el, g, 0]
               + deps[e, el, g, 36 + a] * Dm_eps[e, el, g, 1]
               + deps[e, el, g, 72 + a] * Dm_eps[e, el, g, 2])
        ben = (dk[e, el, g, a] * Dk_k[e, el, g, 0]
               + dk[e, el, g, 36 + a] * Dk_k[e, el, g, 1]
               + dk[e, el, g, 72 + a] * Dk_k[e, el, g, 2])
        am = am + w * h * mem
        ab = ab + w * ben
    wp.atomic_add(Qmem, e, edofs[el, a], am)
    wp.atomic_add(Qbend, e, edofs[el, a], ab)


# ─── Host helpers ──────────────────────────────────────────────────────────

class ANCFConstants:
    """Uploaded element constants + DOF map (shared across envs)."""
    def __init__(self, shell, device=None):
        device = device or config.DEVICE
        NP = config.NP_DTYPE
        ne = shell.ne
        ng = shell.n_gauss
        ngg = ng * ng
        # gather element constants into (ne, ngg, ...) arrays
        dSx = np.zeros((ne, ngg, 3, 36)); dSy = np.zeros_like(dSx)
        d2Sx = np.zeros((ne, ngg, 3, 36)); d2Sy = np.zeros_like(d2Sx); d2Sxy = np.zeros_like(d2Sx)
        A1 = np.zeros((ne, ngg, 36, 36)); A2 = np.zeros_like(A1); A3 = np.zeros_like(A1)
        gw = np.zeros((ne, ngg))
        Me = np.zeros((ne, 36, 36))   # constant element mass blocks (ρh·∫SᵀS)
        edofs = np.zeros((ne, 36), dtype=np.int32)
        for el in range(ne):
            ed = shell._elems[el]
            Mblk = np.zeros((36, 36))
            for i in range(ng):
                for j in range(ng):
                    g = i * ng + j
                    dSx[el, g] = ed.dSx[i, j]
                    dSy[el, g] = ed.dSy[i, j]
                    d2Sx[el, g] = ed.d2Sx[i, j]; d2Sy[el, g] = ed.d2Sy[i, j]
                    d2Sxy[el, g] = ed.d2Sxy[i, j]
                    A1[el, g] = ed.A1[i, j]; A2[el, g] = ed.A2[i, j]; A3[el, g] = ed.A3[i, j]
                    gw[el, g] = ed.gw[i, j]
                    Mblk += ed.gw[i, j] * (ed.S[i, j].T @ ed.S[i, j])
            Me[el] = shell.rho * shell.h * Mblk
            edofs[el] = shell._elem_dofs(el)
        self.ne = ne; self.ngg = ngg; self.ndof = shell.ndof
        self.h = float(shell.h)
        self.mode = shell.mode
        self.Dm_np = np.asarray(shell.Dm, dtype=NP)
        self.Dk_np = np.asarray(shell.Dk, dtype=NP)
        self.edofs_np = edofs
        self.dSx = wp.array(dSx.astype(NP), dtype=DTYPE, device=device)
        self.dSy = wp.array(dSy.astype(NP), dtype=DTYPE, device=device)
        self.d2Sx = wp.array(d2Sx.astype(NP), dtype=DTYPE, device=device)
        self.d2Sy = wp.array(d2Sy.astype(NP), dtype=DTYPE, device=device)
        self.d2Sxy = wp.array(d2Sxy.astype(NP), dtype=DTYPE, device=device)
        self.A1 = wp.array(A1.astype(NP), dtype=DTYPE, device=device)
        self.A2 = wp.array(A2.astype(NP), dtype=DTYPE, device=device)
        self.A3 = wp.array(A3.astype(NP), dtype=DTYPE, device=device)
        self.gw = wp.array(gw.astype(NP), dtype=DTYPE, device=device)
        self.edofs = wp.array(edofs, dtype=wp.int32, device=device)
        self.Me = wp.array(Me.astype(NP), dtype=DTYPE, device=device)   # (ne,36,36) shared
        # free-DOF mask (1.0 free, 0.0 Dirichlet/BC)
        free = np.ones(self.ndof, dtype=NP)
        for d in shell._bc_dofs:
            free[d] = 0.0
        self.free_np = free
        self.free = wp.array(free, dtype=DTYPE, device=device)


def assemble_kmem_blocks(q_wp, C: ANCFConstants, device=None):
    """Return per-element membrane tangent blocks Kblk (B, ne, 36, 36)."""
    device = device or config.DEVICE
    B = q_wp.shape[0]
    NP = config.NP_DTYPE
    Dm = MAT33(*[DTYPE(NP(v)) for v in C.Dm_np.ravel()])
    deps = wp.zeros((B, C.ne, C.ngg, 108), dtype=DTYPE, device=device)  # 108 = 3*36
    Dm_eps = wp.zeros((B, C.ne, C.ngg, 3), dtype=DTYPE, device=device)
    wp.launch(ancf_deps_kernel, dim=(B, C.ne, C.ngg),
              inputs=[q_wp, C.dSx, C.dSy, C.edofs, Dm],
              outputs=[deps, Dm_eps], device=device)
    Kblk = wp.zeros((B, C.ne, 36, 36), dtype=DTYPE, device=device)
    wp.launch(ancf_kmem_kernel, dim=(B, C.ne, 36, 36),
              inputs=[C.A1, C.A2, C.A3, C.gw, deps, Dm_eps, Dm, DTYPE(NP(C.h)), C.ngg],
              outputs=[Kblk], device=device)
    return Kblk


def assemble_internal_force_sep(q_wp, C: ANCFConstants, device=None):
    """Assemble ANCF internal forces, membrane and bending separately: (Qmem, Qbend),
    each (B, ndof). Needed by the Newmark stage-1 bending averaging."""
    device = device or config.DEVICE
    B = q_wp.shape[0]
    NP = config.NP_DTYPE
    Dm = MAT33(*[DTYPE(NP(v)) for v in C.Dm_np.ravel()])
    Dk = MAT33(*[DTYPE(NP(v)) for v in C.Dk_np.ravel()])
    deps = wp.zeros((B, C.ne, C.ngg, 108), dtype=DTYPE, device=device)
    dk = wp.zeros((B, C.ne, C.ngg, 108), dtype=DTYPE, device=device)
    Dm_eps = wp.zeros((B, C.ne, C.ngg, 3), dtype=DTYPE, device=device)
    Dk_k = wp.zeros((B, C.ne, C.ngg, 3), dtype=DTYPE, device=device)
    wp.launch(ancf_force_gauss_kernel, dim=(B, C.ne, C.ngg),
              inputs=[q_wp, C.dSx, C.dSy, C.d2Sx, C.d2Sy, C.d2Sxy, C.edofs, Dm, Dk],
              outputs=[deps, dk, Dm_eps, Dk_k], device=device)
    Qmem = wp.zeros((B, C.ndof), dtype=DTYPE, device=device)
    Qbend = wp.zeros((B, C.ndof), dtype=DTYPE, device=device)
    wp.launch(ancf_force_assemble_kernel, dim=(B, C.ne, 36),
              inputs=[C.gw, deps, dk, Dm_eps, Dk_k, C.edofs, DTYPE(NP(C.h)), C.ngg],
              outputs=[Qmem, Qbend], device=device)
    return Qmem, Qbend


def assemble_internal_force(q_wp, C: ANCFConstants, device=None):
    """Q_int = Q_mem + Q_bend, (B, ndof)."""
    Qmem, Qbend = assemble_internal_force_sep(q_wp, C, device)
    wp.launch(_add_kernel, dim=Qmem.shape, inputs=[Qmem, Qbend], device=device or config.DEVICE)
    return Qmem


@wp.kernel
def _add_kernel(a: wp.array(dtype=DTYPE, ndim=2), b: wp.array(dtype=DTYPE, ndim=2)):
    e, d = wp.tid()
    a[e, d] = a[e, d] + b[e, d]


def scatter_kmem_global(Kblk_np, edofs_np, ndof):
    """Host scatter of per-element blocks (ne,36,36) -> dense (ndof,ndof)."""
    K = np.zeros((ndof, ndof))
    ne = Kblk_np.shape[0]
    for el in range(ne):
        d = edofs_np[el]
        K[np.ix_(d, d)] += Kblk_np[el]
    return K
