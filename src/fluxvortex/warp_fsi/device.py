"""Warp @wp.func device functions — the reused per-thread math.

Biot-Savart segment + ring vortex, ported bit-for-bit from the CPU reference
`standalone_uvlm._vortex_segment_velocity` / `ring_vortex_velocity`, INCLUDING
the `-V` sign convention (matches MATLAB `q1234_mat = -(q1+q2+q3+q4)`).

dtype is baked in at import from `config` (set_dtype before importing this).
"""
from __future__ import annotations
import math
import warp as wp
from . import config

DTYPE = config.DTYPE
VEC3 = config.VEC3
_INV_4PI = 1.0 / (4.0 * math.pi)


@wp.func
def seg_vel(p: VEC3, A: VEC3, B: VEC3, gamma: DTYPE, eps2: DTYPE):
    """Induced velocity at point p from straight vortex segment A->B.

    Mirrors `_vortex_segment_velocity`: core desingularization on the squared
    cross-product norm and on the |r| tangent denominators.
    """
    r1 = p - A
    r2 = p - B
    r0 = B - A
    cr = wp.cross(r1, r2)
    cr_n2 = wp.max(wp.dot(cr, cr), eps2)
    r1n = wp.sqrt(wp.max(wp.dot(r1, r1), eps2))
    r2n = wp.sqrt(wp.max(wp.dot(r2, r2), eps2))
    dotv = wp.dot(r1, r0) / r1n - wp.dot(r2, r0) / r2n
    coeff = gamma * DTYPE(_INV_4PI) * dotv / cr_n2
    return cr * coeff


@wp.func
def ring_vel(p: VEC3, c0: VEC3, c1: VEC3, c2: VEC3, c3: VEC3,
             gamma: DTYPE, eps2: DTYPE):
    """Induced velocity at p from a closed 4-segment ring [c0,c1,c2,c3].

    Returns -V to match the CPU `ring_vortex_velocity` / MATLAB sign convention.
    """
    v = seg_vel(p, c0, c1, gamma, eps2)
    v = v + seg_vel(p, c1, c2, gamma, eps2)
    v = v + seg_vel(p, c2, c3, gamma, eps2)
    v = v + seg_vel(p, c3, c0, gamma, eps2)
    return -v


@wp.func
def dt_seg_vel(p: VEC3, A: VEC3, B: VEC3, gamma: DTYPE,
               dtp: VEC3, dtA: VEC3, dtB: VEC3, eps2: DTYPE):
    """d/dt of seg_vel as p, A, B move — mirrors `_dt_vortex_segment_velocity`
    (product rule on (r1×r2)/|r1×r2|² · ⟨r0, r1/|r1|−r2/|r2|⟩)."""
    r1 = p - A
    r2 = p - B
    r0 = B - A
    dr1 = dtp - dtA
    dr2 = dtp - dtB
    dr0 = dtB - dtA
    cr = wp.cross(r1, r2)
    dcr = wp.cross(dr1, r2) + wp.cross(r1, dr2)
    cr_n2 = wp.max(wp.dot(cr, cr), eps2)
    cr_n4 = cr_n2 * cr_n2
    r1n = wp.sqrt(wp.max(wp.dot(r1, r1), eps2))
    r2n = wp.sqrt(wp.max(wp.dot(r2, r2), eps2))
    r1n3 = r1n * r1n * r1n
    r2n3 = r2n * r2n * r2n
    dcr_over_n2 = dcr / cr_n2 - cr * (DTYPE(2.0) * wp.dot(dcr, cr) / cr_n4)
    d_rhat = (dr1 / r1n - r1 * (wp.dot(dr1, r1) / r1n3)
              - dr2 / r2n + r2 * (wp.dot(dr2, r2) / r2n3))
    rhat = r1 / r1n - r2 / r2n
    cr_over_n2 = cr / cr_n2
    dt_V = (dcr_over_n2 * wp.dot(r0, rhat)
            + cr_over_n2 * wp.dot(dr0, rhat)
            + cr_over_n2 * wp.dot(r0, d_rhat))
    return dt_V * (gamma * DTYPE(_INV_4PI))


@wp.func
def dt_ring_vel(p: VEC3, c0: VEC3, c1: VEC3, c2: VEC3, c3: VEC3, gamma: DTYPE,
                dtp: VEC3, d0: VEC3, d1: VEC3, d2: VEC3, d3: VEC3, eps2: DTYPE):
    """d/dt of ring_vel — returns -Σ to match the CPU/MATLAB sign convention."""
    v = dt_seg_vel(p, c0, c1, gamma, dtp, d0, d1, eps2)
    v = v + dt_seg_vel(p, c1, c2, gamma, dtp, d1, d2, eps2)
    v = v + dt_seg_vel(p, c2, c3, gamma, dtp, d2, d3, eps2)
    v = v + dt_seg_vel(p, c3, c0, gamma, dtp, d3, d0, eps2)
    return -v
