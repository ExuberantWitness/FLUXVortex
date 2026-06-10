"""MATLAB-exact UVLM induction kernels (generate_q1234_mat.m semantics).

Differences from the legacy `device.ring_vel`:
  - denominator regularization |r1 x r2|^2 + eps_v   (eps_v = 1e-9)
  - algebraic vortex core Kv = h^2/(h^(2Nc)+r_core^(2Nc))^(1/Nc), Nc = 2,
    with PER-SOURCE-RING r_core = max(4 segment lengths, Length/Nx) * r_eps
  - segment traversal (1->4),(2->1),(3->2),(4->3); returns -(sum) (MATLAB sign)

Validated against /tmp/ml_uvlm.py (numpy, bit-exact vs MATLAB A_mat 1.5e-16).
"""
from __future__ import annotations
import numpy as np
import warp as wp
from . import config

DTYPE = config.DTYPE
VEC3 = config.VEC3


@wp.func
def ml_seg_vel(rc: VEC3, pa: VEC3, pb: VEC3, r_core: DTYPE, eps_v: DTYPE) -> VEC3:
    """One segment a->b of a unit ring, MATLAB formula (incl. eps_v + Kv core)."""
    r1 = rc - pa
    r2 = rc - pb
    r0 = r1 - r2
    cr = wp.cross(r1, r2)
    ncr2 = wp.dot(cr, cr)
    n1 = wp.max(wp.length(r1), DTYPE(2.220446049250313e-16))
    n2 = wp.max(wp.length(r2), DTYPE(2.220446049250313e-16))
    dot = wp.dot(r0, r1 / n1 - r2 / n2)
    q = cr * (dot / (DTYPE(4.0) * DTYPE(3.141592653589793) * (ncr2 + eps_v)))
    # algebraic core: Kv = h^2/sqrt(h^4 + r_core^4)   (Ncore = 2)
    n0 = wp.length(r0)
    h2 = ncr2 / (n0 * n0)
    Kv = h2 / wp.sqrt(h2 * h2 + r_core * r_core * r_core * r_core)
    return q * Kv


@wp.func
def ml_ring_vel(rc: VEC3, x1: VEC3, x2: VEC3, x3: VEC3, x4: VEC3,
                gamma: DTYPE, r_core: DTYPE, eps_v: DTYPE) -> VEC3:
    """Unit-ring induced velocity, MATLAB -(q1+q2+q3+q4) sign, scaled by gamma.
    Segments: (1->4),(2->1),(3->2),(4->3)."""
    v = ml_seg_vel(rc, x1, x4, r_core, eps_v)
    v = v + ml_seg_vel(rc, x2, x1, r_core, eps_v)
    v = v + ml_seg_vel(rc, x3, x2, r_core, eps_v)
    v = v + ml_seg_vel(rc, x4, x3, r_core, eps_v)
    return -v * gamma


@wp.kernel
def ml_rcore_kernel(c1: wp.array(dtype=VEC3, ndim=2),   # (B,S) ring corners
                    c2: wp.array(dtype=VEC3, ndim=2),
                    c3: wp.array(dtype=VEC3, ndim=2),
                    c4: wp.array(dtype=VEC3, ndim=2),
                    L_over_Nx: DTYPE, r_eps: DTYPE,
                    rcore: wp.array(dtype=DTYPE, ndim=2)):  # (B,S) out
    """Per-source-ring core radius = max(4 segment lengths, L/Nx) * r_eps."""
    e, s = wp.tid()
    l1 = wp.length(c4[e, s] - c1[e, s])
    l2 = wp.length(c1[e, s] - c2[e, s])
    l3 = wp.length(c2[e, s] - c3[e, s])
    l4 = wp.length(c3[e, s] - c4[e, s])
    m = wp.max(wp.max(l1, l2), wp.max(l3, l4))
    rcore[e, s] = wp.max(m, L_over_Nx) * r_eps


@wp.kernel
def ml_aic_kernel(colloc: wp.array(dtype=VEC3, ndim=2),    # (B,T)
                  normals: wp.array(dtype=VEC3, ndim=2),   # (B,T)
                  c1: wp.array(dtype=VEC3, ndim=2),        # (B,S)
                  c2: wp.array(dtype=VEC3, ndim=2),
                  c3: wp.array(dtype=VEC3, ndim=2),
                  c4: wp.array(dtype=VEC3, ndim=2),
                  rcore: wp.array(dtype=DTYPE, ndim=2),    # (B,S)
                  eps_v: DTYPE,
                  AIC: wp.array(dtype=DTYPE, ndim=3)):     # (B,T,S)
    e, i, j = wp.tid()
    v = ml_ring_vel(colloc[e, i], c1[e, j], c2[e, j], c3[e, j], c4[e, j],
                    DTYPE(1.0), rcore[e, j], eps_v)
    AIC[e, i, j] = wp.dot(v, normals[e, i])


@wp.kernel
def ml_induce_kernel(targets: wp.array(dtype=VEC3, ndim=2),  # (B,T)
                     c1: wp.array(dtype=VEC3, ndim=2),       # (B,S)
                     c2: wp.array(dtype=VEC3, ndim=2),
                     c3: wp.array(dtype=VEC3, ndim=2),
                     c4: wp.array(dtype=VEC3, ndim=2),
                     gamma: wp.array(dtype=DTYPE, ndim=2),   # (B,S)
                     rcore: wp.array(dtype=DTYPE, ndim=2),
                     eps_v: DTYPE,
                     V: wp.array(dtype=VEC3, ndim=2)):       # (B,T) accumulated
    e, i, j = wp.tid()
    g = gamma[e, j]
    if g == DTYPE(0.0):
        return
    v = ml_ring_vel(targets[e, i], c1[e, j], c2[e, j], c3[e, j], c4[e, j],
                    g, rcore[e, j], eps_v)
    wp.atomic_add(V, e, i, v)


@wp.func
def ml_dt_seg(rc: VEC3, pa: VEC3, pb: VEC3,
              drc: VEC3, dpa: VEC3, dpb: VEC3) -> VEC3:
    """Analytic d/dt of one UNregularized segment (dt_generate_q1234_mat.m)."""
    r1 = rc - pa
    r2 = rc - pb
    d1 = drc - dpa
    d2 = drc - dpb
    r0 = r1 - r2
    d0 = d1 - d2
    cr = wp.cross(r1, r2)
    ncr2 = wp.dot(cr, cr)
    dcr = wp.cross(d1, r2) + wp.cross(r1, d2)
    term1 = dcr / ncr2 - cr * (DTYPE(2.0) * wp.dot(dcr, cr) / (ncr2 * ncr2))
    n1 = wp.length(r1)
    n2 = wp.length(r2)
    u = r1 / n1 - r2 / n2
    crn = cr / ncr2
    du = (d1 / n1 - r1 * (wp.dot(d1, r1) / (n1 * n1 * n1))
          - d2 / n2 + r2 * (wp.dot(d2, r2) / (n2 * n2 * n2)))
    return term1 * wp.dot(r0, u) + crn * wp.dot(d0, u) + crn * wp.dot(r0, du)


@wp.func
def ml_dt_ring(rc: VEC3, x1: VEC3, x2: VEC3, x3: VEC3, x4: VEC3,
               drc: VEC3, d1: VEC3, d2: VEC3, d3: VEC3, d4: VEC3,
               gamma: DTYPE) -> VEC3:
    v = ml_dt_seg(rc, x1, x4, drc, d1, d4)
    v = v + ml_dt_seg(rc, x2, x1, drc, d2, d1)
    v = v + ml_dt_seg(rc, x3, x2, drc, d3, d2)
    v = v + ml_dt_seg(rc, x4, x3, drc, d4, d3)
    return -v * (gamma / (DTYPE(4.0) * DTYPE(3.141592653589793)))


@wp.kernel
def ml_dt_induce_kernel(colloc: wp.array(dtype=VEC3, ndim=2),    # (B,T)
                        dt_colloc: wp.array(dtype=VEC3, ndim=2),
                        c1: wp.array(dtype=VEC3, ndim=2),        # (B,S)
                        c2: wp.array(dtype=VEC3, ndim=2),
                        c3: wp.array(dtype=VEC3, ndim=2),
                        c4: wp.array(dtype=VEC3, ndim=2),
                        d1: wp.array(dtype=VEC3, ndim=2),
                        d2: wp.array(dtype=VEC3, ndim=2),
                        d3: wp.array(dtype=VEC3, ndim=2),
                        d4: wp.array(dtype=VEC3, ndim=2),
                        gamma: wp.array(dtype=DTYPE, ndim=2),
                        V: wp.array(dtype=VEC3, ndim=2)):        # (B,T)
    e, i, j = wp.tid()
    g = gamma[e, j]
    if g == DTYPE(0.0):
        return
    v = ml_dt_ring(colloc[e, i], c1[e, j], c2[e, j], c3[e, j], c4[e, j],
                   dt_colloc[e, i], d1[e, j], d2[e, j], d3[e, j], d4[e, j], g)
    wp.atomic_add(V, e, i, v)


@wp.kernel
def ml_dt_aic_kernel(colloc: wp.array(dtype=VEC3, ndim=2),
                     dt_colloc: wp.array(dtype=VEC3, ndim=2),
                     normals: wp.array(dtype=VEC3, ndim=2),
                     c1: wp.array(dtype=VEC3, ndim=2),
                     c2: wp.array(dtype=VEC3, ndim=2),
                     c3: wp.array(dtype=VEC3, ndim=2),
                     c4: wp.array(dtype=VEC3, ndim=2),
                     d1: wp.array(dtype=VEC3, ndim=2),
                     d2: wp.array(dtype=VEC3, ndim=2),
                     d3: wp.array(dtype=VEC3, ndim=2),
                     d4: wp.array(dtype=VEC3, ndim=2),
                     dtA: wp.array(dtype=DTYPE, ndim=3)):        # (B,T,S) dt_Amat1
    e, i, j = wp.tid()
    v = ml_dt_ring(colloc[e, i], c1[e, j], c2[e, j], c3[e, j], c4[e, j],
                   dt_colloc[e, i], d1[e, j], d2[e, j], d3[e, j], d4[e, j],
                   DTYPE(1.0))
    dtA[e, i, j] = wp.dot(v, normals[e, i])
