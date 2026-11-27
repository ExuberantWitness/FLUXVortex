"""Fused warp kernel for the finite-ring Biot-Savart influence sum.

Replaces the 64 MB point-tiling loop in ``Q16NativeV5MSolver._ring_velocity``
for large wake systems (9k rings x 12k points and up).  The tiled torch path
materializes ~15 intermediate (points_tile, rings, 3) tensors per call —
pure memory traffic.  This kernel gives each (target, source-block) thread a
private accumulator and finishes with one torch reduction, mirroring the
fused particle kernel in pfield_torch_gpu.py.

The MATH is the frozen paper model (native_ring_velocity_expanded): the
fourth-order core kv = h^2 / sqrt(h^4 + core^4) with per-ring core =
max(edge lengths, reference_length) * core_fraction, and the filament
base = cross/(cross^2 + EPS_V) * scalar / 4pi.  v2 keeps the same model
with reciprocal-multiplies instead of repeated divides and a precomputed
core^4 (fp64 rounding differences only).
"""
from __future__ import annotations

import math
import os

import torch
import warp as wp

RING_BS_BLOCK_SIZE = 256

_C_INV_4PI = 1.0 / (4.0 * math.pi)
_C_FP64_EPS = 2.220446049250313e-16  # torch.finfo(float64).eps
#: Denominator regularisation of the filament law, |a x b|^2 -> |a x b|^2 + EPS_V.
#: The author's MATLAB froze it at 1e-9 (Yamano: "eps_v = 1e-9, in squared denominator").
#:
#: It carries UNITS: |a x b| is m^2, so EPS_V is m^4, and the exact law is recovered only
#: where |a x b|^2 >> EPS_V.  1e-9 holds on metre-scale wings.  On the Rojratsirikul et al.
#: (2011) wing (chord 68.8 mm, panels 4.6 mm) the near field has |a x b|^2 ~ 1e-10 and that
#: value dominates it: the self term is 10% of Biot-Savart and the nearest-neighbour term
#: changes sign (platform/warp_vpm/diagnose_ring_kernel_scale_invariance.py, note A19).
#:
#: The default is 1e-12 since 2026-11-25 (the owner's decision).  The cut-off length
#: sqrt(EPS_V)/L is then 0.22 mm on a 4.6 mm panel, the steady lattice is within 0.4% of the
#: unregularised one (0.2203 against 0.2194 at alpha = 5 deg), and a point that leaves a
#: filament line under a Runge-Kutta stage still meets the regularised law at the
#: trailing-edge junction.  It must stay positive: it is what keeps the denominator off zero.
#:
#: `FLUXV_RING_EPS_V` overrides the constant for BOTH copies on this path: this one, behind
#: the fused kernel, and the torch function `native_ring_velocity_expanded` behind the
#: influence matrix.  A value for another wing is chosen relative to its panel size, never
#: copied from another model; `diagnose_ring_kernel_scale_invariance.py` shows where a
#: candidate becomes inert.
_EPS_V = float(os.environ.get("FLUXV_RING_EPS_V", "1.0e-12"))
if not (math.isfinite(_EPS_V) and _EPS_V > 0.0):
    raise ValueError(
        f"FLUXV_RING_EPS_V must be a finite positive number, got {_EPS_V!r}"
    )
_C_EPS_V = _EPS_V                     # the same constant, as the fused kernel sees it

#: FLUXV_RING_KERNEL selects the filament law (plan V3, P0-A).
#:
#:   legacy  |a x b|^2 + EPS_V in the denominator, EPS_V = FLUXV_RING_EPS_V (1e-12 m^4 by
#:           default, 1e-9 before 2026-11-25).  The form the author wrote; what the
#:           constant does is documented above.
#:   v3      exact Biot-Savart with a dimensionless collinear guard.  |a x b| = |e| h, the
#:           filament length times the perpendicular distance, so cross_sq / |e|^4 =
#:           (h/|e|)^2 is dimensionless.  A filament whose evaluation point lies on its line
#:           (that ratio at or below FLUXV_RING_COLLINEAR_REL^2, default 1e-12) contributes
#:           zero, which IS its Biot-Savart value there; every other filament uses its exact
#:           |a x b|^2.  In float64 the rounding noise of the cross product of two
#:           panel-length vectors corresponds to h/|e| ~ 1e-16, so 1e-12 is four decades
#:           above it, and it is far below the closest approach in a ring lattice (a
#:           collocation point sits half a panel from its own ring's sides).  In float32 the
#:           same noise sits near 1e-9 (99th percentile 4e-8 for 4.6 mm panels), so the
#:           threshold has to be raised before this kernel runs in single precision.
#:
#:           The fourth-order core kv = h^2 / sqrt(h^4 + r_c^4) is kept apart from the guard:
#:           a caller that passes `core_radius` gets a core of that fixed radius on every
#:           filament of the call, whatever the filament's length (the legacy core scaled
#:           with the longest edge of each ring and with a reference length that assumed a
#:           one-metre chord); a caller that passes none gets the exact law.  That is the
#:           plan's split (FLUX_V5M_PLAN_AUTHOR_REPLY_AND_V3_20261125.md, question 6): no
#:           core for the influence matrix and for the wake's action on the collocation
#:           points, a core tied to the panel size for the free wake's self-convection and
#:           for the load-velocity evaluations.
_KERNEL_CHOICE = os.environ.get("FLUXV_RING_KERNEL", "legacy").strip().lower()
if _KERNEL_CHOICE not in {"legacy", "v3"}:
    raise ValueError(
        f"FLUXV_RING_KERNEL must be 'legacy' or 'v3', got {_KERNEL_CHOICE!r}"
    )
_KERNEL_V3 = _KERNEL_CHOICE == "v3"
_COLLINEAR_REL = float(os.environ.get("FLUXV_RING_COLLINEAR_REL", "1e-12"))
if not (math.isfinite(_COLLINEAR_REL) and 0.0 < _COLLINEAR_REL < 1.0):
    raise ValueError(
        f"FLUXV_RING_COLLINEAR_REL must be finite and in (0, 1), got {_COLLINEAR_REL!r}"
    )
#: Stage 3 (S5-1) opt-in mixed-precision Biot-Savart for free-wake/LEV convection and load
#: midpoints under FLUXV_RING_KERNEL=v3: origin-shifted cancellation-free FP32 in the far field
#: (r > 4*max(|e|, r_c)), exact FP64 in the near field / near-collinear cone, FP64 accumulation.
#: AIC and collocation Neumann solves stay pure FP64.  Default OFF (bit-identical production).
_MIXED_PRECISION = os.environ.get("FLUXV_RING_MIXED_PRECISION", "0").strip().lower() in {
    "1", "true", "on", "yes",
}


@wp.kernel
def _ring_bs_partials(
    targets: wp.array2d(dtype=wp.float64),
    starts: wp.array2d(dtype=wp.float64),
    ends: wp.array2d(dtype=wp.float64),
    filament_gamma: wp.array1d(dtype=wp.float64),
    filament_core4: wp.array1d(dtype=wp.float64),
    block_size: int,
    partials: wp.array3d(dtype=wp.float64),
):
    t, b = wp.tid()
    n_s = int(filament_core4.shape[0])
    start = b * block_size
    stop = wp.min(start + block_size, n_s)
    inv_4pi = wp.float64(_C_INV_4PI)
    eps = wp.float64(_C_FP64_EPS)
    eps_v = wp.float64(_C_EPS_V)
    zero = wp.float64(0.0)
    accx = zero
    accy = zero
    accz = zero
    tx = targets[t, 0]
    ty = targets[t, 1]
    tz = targets[t, 2]
    for s in range(start, stop):
        ax = tx - starts[s, 0]
        ay = ty - starts[s, 1]
        az = tz - starts[s, 2]
        bx = tx - ends[s, 0]
        by = ty - ends[s, 1]
        bz = tz - ends[s, 2]
        ex = ax - bx
        ey = ay - by
        ez = az - bz
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        cross_sq = cx * cx + cy * cy + cz * cz
        one = wp.float64(1.0)
        inv_a = one / wp.max(wp.sqrt(ax * ax + ay * ay + az * az), eps)
        inv_b = one / wp.max(wp.sqrt(bx * bx + by * by + bz * bz), eps)
        inv_e = one / wp.max(wp.sqrt(ex * ex + ey * ey + ez * ez), eps)
        ux = ax * inv_a - bx * inv_b
        uy = ay * inv_a - by * inv_b
        uz = az * inv_a - bz * inv_b
        scalar = ex * ux + ey * uy + ez * uz
        inv_den = scalar * inv_4pi / (cross_sq + eps_v)
        h2 = cross_sq * inv_e * inv_e
        kv = h2 / wp.sqrt(h2 * h2 + filament_core4[s])
        w = kv * filament_gamma[s] * inv_den
        accx += w * cx
        accy += w * cy
        accz += w * cz
    partials[t, b, 0] = accx
    partials[t, b, 1] = accy
    partials[t, b, 2] = accz


@wp.kernel
def _ring_bs_partials_v3(
    targets: wp.array2d(dtype=wp.float64),
    starts: wp.array2d(dtype=wp.float64),
    ends: wp.array2d(dtype=wp.float64),
    filament_gamma: wp.array1d(dtype=wp.float64),
    filament_core4: wp.array1d(dtype=wp.float64),
    collinear_rel: float,
    block_size: int,
    partials: wp.array3d(dtype=wp.float64),
):
    # Plan V3 (P0-A) filament law; `_ring_bs_partials` above is left untouched so the
    # legacy path stays bit-identical by construction.  Same accumulation structure,
    # same core factor; only the denominator differs: the exact |a x b|^2 with a
    # dimensionless collinear guard instead of |a x b|^2 + eps.
    #
    # Warp compiles every kernel of a module when the module loads, so this kernel is
    # parsed even under the default flag: a scalar kernel argument declared as `float`
    # arrives as float32 and must be widened explicitly before it meets a float64 value.
    t, b = wp.tid()
    n_s = int(filament_core4.shape[0])
    start = b * block_size
    stop = wp.min(start + block_size, n_s)
    inv_4pi = wp.float64(_C_INV_4PI)
    eps = wp.float64(_C_FP64_EPS)
    rel = wp.float64(collinear_rel)
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    tiny = wp.float64(1.0e-300)
    accx = zero
    accy = zero
    accz = zero
    tx = targets[t, 0]
    ty = targets[t, 1]
    tz = targets[t, 2]
    for s in range(start, stop):
        ax = tx - starts[s, 0]
        ay = ty - starts[s, 1]
        az = tz - starts[s, 2]
        bx = tx - ends[s, 0]
        by = ty - ends[s, 1]
        bz = tz - ends[s, 2]
        ex = ax - bx
        ey = ay - by
        ez = az - bz
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        cross_sq = cx * cx + cy * cy + cz * cz
        inv_a = one / wp.max(wp.sqrt(ax * ax + ay * ay + az * az), eps)
        inv_b = one / wp.max(wp.sqrt(bx * bx + by * by + bz * bz), eps)
        inv_e = one / wp.max(wp.sqrt(ex * ex + ey * ey + ez * ez), eps)
        ux = ax * inv_a - bx * inv_b
        uy = ay * inv_a - by * inv_b
        uz = az * inv_a - bz * inv_b
        scalar = ex * ux + ey * uy + ez * uz
        edge_sq = ex * ex + ey * ey + ez * ez
        keep = cross_sq > rel * rel * edge_sq * edge_sq
        # |a x b|^2 on kept filaments; a harmless positive value elsewhere so the
        # unselected branch of the select is never inf.
        den = wp.max(cross_sq, tiny)
        h_sq = cross_sq * inv_e * inv_e
        kv = h_sq / wp.sqrt(h_sq * h_sq + filament_core4[s])
        w = wp.where(
            keep, scalar * inv_4pi / den * kv * filament_gamma[s], zero
        )
        accx += w * cx
        accy += w * cy
        accz += w * cz
    partials[t, b, 0] = accx
    partials[t, b, 1] = accy
    partials[t, b, 2] = accz


@wp.kernel
def _ring_bs_partials_v3_mixed(
    targets64: wp.array2d(dtype=wp.float64),
    starts64: wp.array2d(dtype=wp.float64),
    ends64: wp.array2d(dtype=wp.float64),
    gamma64: wp.array1d(dtype=wp.float64),
    core4_64: wp.array1d(dtype=wp.float64),
    targets32: wp.array2d(dtype=wp.float32),
    starts32: wp.array2d(dtype=wp.float32),
    edges32: wp.array2d(dtype=wp.float32),
    edge_sq32: wp.array1d(dtype=wp.float32),
    inv_edge_sq32: wp.array1d(dtype=wp.float32),
    near_r2_32: wp.array1d(dtype=wp.float32),
    gamma32: wp.array1d(dtype=wp.float32),
    core4_32: wp.array1d(dtype=wp.float32),
    collinear_rel: float,
    block_size: int,
    partials: wp.array3d(dtype=wp.float64),
):
    """Stage 3 (S5-0 + S5-1) mixed-precision v3 ring Biot-Savart kernel.

    Launched with dim=(n_blocks, n_targets) so consecutive warp lanes share the same
    source block `b` (100% warp-broadcast on filament loads, coalesced target/partial IO).
    """

    b, t = wp.tid()
    n_s = int(core4_32.shape[0])
    start = b * block_size
    stop = wp.min(start + block_size, n_s)
    inv_4pi32 = wp.float32(_C_INV_4PI)
    one32 = wp.float32(1.0)
    col_cone32 = wp.float32(1.0e-5)

    inv_4pi64 = wp.float64(_C_INV_4PI)
    eps64 = wp.float64(_C_FP64_EPS)
    rel64 = wp.float64(collinear_rel)
    rel_sq64 = rel64 * rel64
    one64 = wp.float64(1.0)
    zero64 = wp.float64(0.0)
    tiny64 = wp.float64(1.0e-300)

    accx = zero64
    accy = zero64
    accz = zero64
    accx32 = wp.float32(0.0)
    accy32 = wp.float32(0.0)
    accz32 = wp.float32(0.0)
    tx32 = targets32[t, 0]
    ty32 = targets32[t, 1]
    tz32 = targets32[t, 2]
    tx64 = targets64[t, 0]
    ty64 = targets64[t, 1]
    tz64 = targets64[t, 2]

    for s in range(start, stop):
        ax32 = tx32 - starts32[s, 0]
        ay32 = ty32 - starts32[s, 1]
        az32 = tz32 - starts32[s, 2]
        ex32 = edges32[s, 0]
        ey32 = edges32[s, 1]
        ez32 = edges32[s, 2]
        bx32 = ax32 - ex32
        by32 = ay32 - ey32
        bz32 = az32 - ez32
        a2_32 = ax32 * ax32 + ay32 * ay32 + az32 * az32
        b2_32 = bx32 * bx32 + by32 * by32 + bz32 * bz32
        cx32 = ey32 * az32 - ez32 * ay32
        cy32 = ez32 * ax32 - ex32 * az32
        cz32 = ex32 * ay32 - ey32 * ax32
        csq32 = cx32 * cx32 + cy32 * cy32 + cz32 * cz32
        e2_32 = edge_sq32[s]

        if wp.min(a2_32, b2_32) > near_r2_32[s] and csq32 > col_cone32 * e2_32 * a2_32:
            ra32 = wp.sqrt(a2_32)
            rb32 = wp.sqrt(b2_32)
            inv_a32 = one32 / ra32
            inv_b32 = one32 / rb32
            ea32 = ex32 * ax32 + ey32 * ay32 + ez32 * az32
            eb32 = ea32 - e2_32
            scalar32 = (e2_32 - ea32 * (ea32 + eb32) * inv_a32 / (ra32 + rb32)) * inv_b32
            h_sq32 = csq32 * inv_edge_sq32[s]
            kv32 = h_sq32 / wp.sqrt(h_sq32 * h_sq32 + core4_32[s])
            w32 = scalar32 * inv_4pi32 / csq32 * kv32 * gamma32[s]
            accx32 += w32 * cx32
            accy32 += w32 * cy32
            accz32 += w32 * cz32
        else:
            ax = tx64 - starts64[s, 0]
            ay = ty64 - starts64[s, 1]
            az = tz64 - starts64[s, 2]
            bx = tx64 - ends64[s, 0]
            by = ty64 - ends64[s, 1]
            bz = tz64 - ends64[s, 2]
            ex = ax - bx
            ey = ay - by
            ez = az - bz
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            cross_sq = cx * cx + cy * cy + cz * cz
            inv_a = one64 / wp.max(wp.sqrt(ax * ax + ay * ay + az * az), eps64)
            inv_b = one64 / wp.max(wp.sqrt(bx * bx + by * by + bz * bz), eps64)
            edge_sq = ex * ex + ey * ey + ez * ez
            inv_e = one64 / wp.max(wp.sqrt(edge_sq), eps64)
            ux = ax * inv_a - bx * inv_b
            uy = ay * inv_a - by * inv_b
            uz = az * inv_a - bz * inv_b
            scalar = ex * ux + ey * uy + ez * uz
            keep = cross_sq > rel_sq64 * edge_sq * edge_sq
            den = wp.max(cross_sq, tiny64)
            h_sq = cross_sq * inv_e * inv_e
            kv = h_sq / wp.sqrt(h_sq * h_sq + core4_64[s])
            w = wp.where(keep, scalar * inv_4pi64 / den * kv * gamma64[s], zero64)
            accx += w * cx
            accy += w * cy
            accz += w * cz
    partials[b, t, 0] = accx + wp.float64(accx32)
    partials[b, t, 1] = accy + wp.float64(accy32)
    partials[b, t, 2] = accz + wp.float64(accz32)


@wp.kernel
def _ring_bs_partials_v3_cored_fp32(
    targets32: wp.array2d(dtype=wp.float32),
    starts32: wp.array2d(dtype=wp.float32),
    edges32: wp.array2d(dtype=wp.float32),
    edge_sq32: wp.array1d(dtype=wp.float32),
    inv_edge_sq32: wp.array1d(dtype=wp.float32),
    gamma32: wp.array1d(dtype=wp.float32),
    core4_32: wp.array1d(dtype=wp.float32),
    block_size: int,
    partials: wp.array3d(dtype=wp.float64),
):
    """Branch-free, singularity-free FP32 Biot-Savart kernel for cored v3 filaments.

    Supports both uniform and per-filament Squire viscous core radii (`core4_32[s]`).
    Launched with dim=(n_blocks, n_targets): all 32 lanes of a CUDA warp share the exact
    same filament index `s` (100% L1 broadcast on source loads, coalesced target/partial IO).
    """

    b, t = wp.tid()
    n_s = int(gamma32.shape[0])
    start = b * block_size
    stop = wp.min(start + block_size, n_s)
    inv_4pi = wp.float32(_C_INV_4PI)
    eps = wp.float32(1.0e-12)
    accx = wp.float32(0.0)
    accy = wp.float32(0.0)
    accz = wp.float32(0.0)
    tx = targets32[t, 0]
    ty = targets32[t, 1]
    tz = targets32[t, 2]
    for s in range(start, stop):
        ax = tx - starts32[s, 0]
        ay = ty - starts32[s, 1]
        az = tz - starts32[s, 2]
        ex = edges32[s, 0]
        ey = edges32[s, 1]
        ez = edges32[s, 2]
        bx = ax - ex
        by = ay - ey
        bz = az - ez
        a2 = ax * ax + ay * ay + az * az
        b2 = bx * bx + by * by + bz * bz
        ra = wp.sqrt(a2) + eps
        rb = wp.sqrt(b2) + eps
        e2 = edge_sq32[s]
        inv_e2 = inv_edge_sq32[s]
        ea = ex * ax + ey * ay + ez * az
        eb = ea - e2
        scalar = (e2 - ea * (ea + eb) / (ra * (ra + rb))) / rb
        cx = ey * az - ez * ay
        cy = ez * ax - ex * az
        cz = ex * ay - ey * ax
        h_sq = (cx * cx + cy * cy + cz * cz) * inv_e2
        w = (scalar * inv_4pi * inv_e2 * gamma32[s]) / wp.sqrt(h_sq * h_sq + core4_32[s])
        accx += w * cx
        accy += w * cy
        accz += w * cz
    partials[b, t, 0] = wp.float64(accx)
    partials[b, t, 1] = wp.float64(accy)
    partials[b, t, 2] = wp.float64(accz)


def mean_chordwise_panel(rings: torch.Tensor) -> float:
    """Mean chordwise length, in metres, of rings stored (R, 4, 3).

    The corners are front-left, front-right, back-right, back-left, so the length of a ring
    is half of |back-left + back-right - front-left - front-right|.  This is the reference
    length of the legacy core, `core_fraction * reference_length`.  The author's code wrote
    `1 / chordwise_panels`, which is this length for a chord of one metre and is wrong by the
    chord for any other wing (plan V3, Bug-4).  One device synchronisation per call.
    """

    if rings.shape[0] == 0:
        return 0.0
    length = 0.5 * torch.linalg.vector_norm(
        rings[:, 2] + rings[:, 3] - rings[:, 0] - rings[:, 1], dim=1
    )
    return float(length.mean().item())


def native_ring_velocity_fused(
    points: torch.Tensor,
    rings: torch.Tensor,
    gamma: torch.Tensor,
    *,
    core_fraction: float,
    reference_length: float,
    block_size: int = RING_BS_BLOCK_SIZE,
    core_radius: float | torch.Tensor | None = None,
    mixed_precision: bool = False,
) -> torch.Tensor:
    """Velocity at ``points`` induced by ``rings`` with per-ring ``gamma``.

    Same model as native_ring_velocity_expanded followed by the caller's
    gamma-weighted ring sum, fused into one kernel launch plus one torch
    reduction.  fp64 summation order differs from the tiled torch path
    (rounding level only, like the fused particle kernel).

    Under FLUXV_RING_KERNEL=v3 the launch uses the v3 kernel (see _KERNEL_V3):
    `core_fraction` and `reference_length` are ignored, `core_radius` is the
    fixed or per-ring core radius (None = exact law).  Under the default
    the historical kernel and arguments are used unchanged.
    """

    if rings.shape[0] == 0:
        return torch.zeros_like(points)
    device = points.device
    starts = rings.reshape(-1, 3)
    ends = torch.roll(rings, shifts=-1, dims=1).reshape(-1, 3)
    filament_gamma = gamma.repeat_interleave(4)
    n_filament = int(filament_gamma.shape[0])
    if _KERNEL_V3:
        is_tensor_core = isinstance(core_radius, torch.Tensor)
        if is_tensor_core:
            rc_ring = core_radius.to(device=device, dtype=torch.float64).reshape(-1)
            core4 = (rc_ring * rc_ring * rc_ring * rc_ring).repeat_interleave(4)
            has_pos_core = True
        elif core_radius is not None and float(core_radius) > 0.0:
            core4 = torch.full(
                (n_filament,), float(core_radius) ** 4, device=device, dtype=torch.float64
            )
            has_pos_core = True
        else:
            core4 = torch.zeros((n_filament,), device=device, dtype=torch.float64)
            has_pos_core = False
        if mixed_precision:
            bs_mx = max(block_size, 512)
            n_blocks_mx = (n_filament + bs_mx - 1) // bs_mx
            partials_mx = torch.empty(
                (n_blocks_mx, points.shape[0], 3), device=device, dtype=torch.float64
            )
            origin = points.mean(dim=0, keepdim=True)
            pts32 = (points - origin).to(torch.float32).contiguous()
            starts32 = (starts - origin).to(torch.float32).contiguous()
            edges64 = ends - starts
            edges32 = edges64.to(torch.float32).contiguous()
            edge_sq64 = torch.sum(edges64 * edges64, dim=1)
            edge_sq32 = edge_sq64.to(torch.float32).contiguous()
            inv_edge_sq32 = (1.0 / torch.clamp(edge_sq64, min=1.0e-300)).to(torch.float32).contiguous()
            gamma32 = filament_gamma.to(torch.float32).contiguous()
            core4_32 = core4.to(torch.float32).contiguous()
            if has_pos_core:
                wp.launch(
                    _ring_bs_partials_v3_cored_fp32,
                    dim=(n_blocks_mx, points.shape[0]),
                    inputs=[
                        wp.from_torch(pts32, dtype=wp.float32),
                        wp.from_torch(starts32, dtype=wp.float32),
                        wp.from_torch(edges32, dtype=wp.float32),
                        wp.from_torch(edge_sq32, dtype=wp.float32),
                        wp.from_torch(inv_edge_sq32, dtype=wp.float32),
                        wp.from_torch(gamma32, dtype=wp.float32),
                        wp.from_torch(core4_32, dtype=wp.float32),
                        bs_mx,
                    ],
                    outputs=[wp.from_torch(partials_mx, dtype=wp.float64)],
                    device="cuda:0",
                )
                return torch.sum(partials_mx, dim=0)
            near_r2_32 = torch.clamp(
                16.0 * edge_sq64,
                min=1.0e-12,
            ).to(torch.float32).contiguous()
            wp.launch(
                _ring_bs_partials_v3_mixed,
                dim=(n_blocks_mx, points.shape[0]),
                inputs=[
                    wp.from_torch(points.contiguous(), dtype=wp.float64),
                    wp.from_torch(starts.contiguous(), dtype=wp.float64),
                    wp.from_torch(ends.contiguous(), dtype=wp.float64),
                    wp.from_torch(filament_gamma.contiguous(), dtype=wp.float64),
                    wp.from_torch(core4.contiguous(), dtype=wp.float64),
                    wp.from_torch(pts32, dtype=wp.float32),
                    wp.from_torch(starts32, dtype=wp.float32),
                    wp.from_torch(edges32, dtype=wp.float32),
                    wp.from_torch(edge_sq32, dtype=wp.float32),
                    wp.from_torch(inv_edge_sq32, dtype=wp.float32),
                    wp.from_torch(near_r2_32, dtype=wp.float32),
                    wp.from_torch(gamma32, dtype=wp.float32),
                    wp.from_torch(core4_32, dtype=wp.float32),
                    float(_COLLINEAR_REL),
                    bs_mx,
                ],
                outputs=[wp.from_torch(partials_mx, dtype=wp.float64)],
                device="cuda:0",
            )
            return torch.sum(partials_mx, dim=0)
        n_blocks = (n_filament + block_size - 1) // block_size
        partials = torch.zeros(
            (points.shape[0], n_blocks, 3), device=device, dtype=torch.float64
        )
        wp.launch(
            _ring_bs_partials_v3,
            dim=(points.shape[0], n_blocks),
            inputs=[
                wp.from_torch(points.contiguous(), dtype=wp.float64),
                wp.from_torch(starts.contiguous(), dtype=wp.float64),
                wp.from_torch(ends.contiguous(), dtype=wp.float64),
                wp.from_torch(filament_gamma.contiguous(), dtype=wp.float64),
                wp.from_torch(core4.contiguous(), dtype=wp.float64),
                float(_COLLINEAR_REL),
                block_size,
            ],
            outputs=[wp.from_torch(partials, dtype=wp.float64)],
            device="cuda:0",
        )
        return torch.sum(partials, dim=1)
    n_blocks = (n_filament + block_size - 1) // block_size
    partials = torch.zeros(
        (points.shape[0], n_blocks, 3), device=device, dtype=torch.float64
    )
    edge_lengths = torch.linalg.vector_norm(
        torch.roll(rings, shifts=-1, dims=1) - rings, dim=2
    )
    source_scale = torch.maximum(
        torch.max(edge_lengths, dim=1).values,
        torch.full(
            (rings.shape[0],),
            float(reference_length),
            device=device,
            dtype=torch.float64,
        ),
    )
    core = source_scale * float(core_fraction)
    core4 = (core * core * core * core).repeat_interleave(4)
    wp.launch(
        _ring_bs_partials,
        dim=(points.shape[0], n_blocks),
        inputs=[
            wp.from_torch(points.contiguous(), dtype=wp.float64),
            wp.from_torch(starts.contiguous(), dtype=wp.float64),
            wp.from_torch(ends.contiguous(), dtype=wp.float64),
            wp.from_torch(filament_gamma.contiguous(), dtype=wp.float64),
            wp.from_torch(core4.contiguous(), dtype=wp.float64),
            block_size,
        ],
        outputs=[wp.from_torch(partials, dtype=wp.float64)],
        device="cuda:0",
    )
    return torch.sum(partials, dim=1)


__all__ = ["native_ring_velocity_fused", "RING_BS_BLOCK_SIZE"]


# ── Expanded (per-leg) finite-ring influence, shared load path (M1-2) ────
_FOUR_PI = 4.0 * math.pi
# `_EPS_V` is defined once at the top of the module, next to the fused kernel's copy,
# so the influence matrix and every other ring-velocity evaluation share one constant.


def native_ring_velocity_expanded(
    points: torch.Tensor,
    rings: torch.Tensor,
    *,
    core_fraction: float,
    reference_length: float,
    core_radius: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Finite-ring influence (per leg, before the gamma-weighted sum).

    Under FLUXV_RING_KERNEL=v3 this is the exact law with the dimensionless collinear
    guard and, if `core_radius` is given, the fourth-order core of that fixed radius
    (see _KERNEL_V3); `core_fraction` and `reference_length` are then ignored.  Under the
    default it is the paper's fourth-order core model over |a x b|^2 + eps, unchanged.
    """

    if rings.shape[0] == 0:
        return torch.zeros(
            (points.shape[0], 0, 3), device=points.device, dtype=torch.float64
        )
    starts = rings.reshape(-1, 3)
    ends = torch.roll(rings, shifts=-1, dims=1).reshape(-1, 3)
    a = points[:, None, :] - starts[None, :, :]
    b = points[:, None, :] - ends[None, :, :]
    edge = a - b
    cross = torch.linalg.cross(a, b, dim=2)
    cross_sq = torch.sum(cross * cross, dim=2)
    cross_norm = torch.sqrt(cross_sq)
    norm_a = torch.linalg.vector_norm(a, dim=2)
    norm_b = torch.linalg.vector_norm(b, dim=2)
    edge_norm = torch.linalg.vector_norm(edge, dim=2)
    eps = torch.finfo(torch.float64).eps
    unit_difference = a / torch.clamp(norm_a, min=eps)[:, :, None] - b / torch.clamp(
        norm_b, min=eps
    )[:, :, None]
    scalar = torch.sum(edge * unit_difference, dim=2)
    if _KERNEL_V3:
        edge_sq = edge_norm * edge_norm
        keep = cross_sq > (_COLLINEAR_REL * _COLLINEAR_REL) * edge_sq * edge_sq
        denominator = torch.where(keep, cross_sq, torch.ones_like(cross_sq))
        factor = torch.where(
            keep, scalar / (_FOUR_PI * denominator), torch.zeros_like(scalar)
        )
        base = cross * factor[:, :, None]
        if core_radius is None:
            return base.reshape(points.shape[0], rings.shape[0], 4, 3).sum(dim=2)
        h_sq = cross_sq / torch.clamp(edge_sq, min=eps)
        if isinstance(core_radius, torch.Tensor):
            rc4 = (
                core_radius.to(device=points.device, dtype=torch.float64).reshape(-1) ** 4
            ).repeat_interleave(4)[None, :]
        else:
            rc4 = float(core_radius) ** 4
        kv = h_sq / torch.sqrt(h_sq * h_sq + rc4)
        return (kv[:, :, None] * base).reshape(
            points.shape[0], rings.shape[0], 4, 3
        ).sum(dim=2)
    base = (
        cross
        / (cross_sq + _EPS_V)[:, :, None]
        * scalar[:, :, None]
        / _FOUR_PI
    )
    source_edge = torch.linalg.vector_norm(
        torch.roll(rings, shifts=-1, dims=1) - rings, dim=2
    )
    source_scale = torch.maximum(
        torch.max(source_edge, dim=1).values,
        torch.full(
            (rings.shape[0],),
            float(reference_length),
            device=points.device,
            dtype=torch.float64,
        ),
    )
    core = source_scale * float(core_fraction)
    h = cross_norm / torch.clamp(edge_norm, min=eps)
    core_leg = core.repeat_interleave(4)[None, :]
    kv = h * h / torch.sqrt(h**4 + core_leg**4)
    velocity = (kv[:, :, None] * base).reshape(
        points.shape[0], rings.shape[0], 4, 3
    ).sum(dim=2)
    if not bool(torch.isfinite(velocity).all().item()):
        raise FloatingPointError("native ring influence is non-finite")
    return velocity


__all__ = ["RING_BS_BLOCK_SIZE", "native_ring_velocity_expanded", "native_ring_velocity_fused"]
