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
#: Frozen at 1e-9 from the author's MATLAB (Yamano: "eps_v = 1e-9, in squared
#: denominator").
#:
#: It carries UNITS: |a x b| is m^2, so EPS_V is m^4, and the exact law is recovered only
#: where |a x b|^2 >> EPS_V.  That holds on metre-scale wings.  On the Rojratsirikul et
#: al. (2011) wing (chord 68.8 mm, panels 4.6 mm) the near field has |a x b|^2 ~ 1e-10
#: and the constant dominates it -- measured with
#: platform/warp_vpm/diagnose_ring_kernel_scale_invariance.py, the self term is 10% of
#: Biot-Savart and the nearest-neighbour term changes sign (note A19).
#:
#: `FLUXV_RING_EPS_V` overrides the constant for BOTH copies on this path: this one,
#: behind the fused kernel, and the torch function `native_ring_velocity_expanded`
#: behind the influence matrix.  The default is the frozen value, so the production
#: path is unchanged.  A value for a small wing should be set relative to its panel
#: size, not copied from another model; `diagnose_ring_kernel_scale_invariance.py`
#: shows where a candidate becomes inert.
_EPS_V = float(os.environ.get("FLUXV_RING_EPS_V", "1.0e-9"))
if not (math.isfinite(_EPS_V) and _EPS_V > 0.0):
    raise ValueError(
        f"FLUXV_RING_EPS_V must be a finite positive number, got {_EPS_V!r}"
    )
_C_EPS_V = _EPS_V                     # the same constant, as the fused kernel sees it

#: FLUXV_RING_KERNEL selects the filament law (plan V3, P0-A).
#:
#:   legacy  |a x b|^2 + EPS_V in the denominator, EPS_V = FLUXV_RING_EPS_V (1e-9 m^4 by
#:           default).  Kept bit-for-bit; what that constant does is documented above.
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


def native_ring_velocity_fused(
    points: torch.Tensor,
    rings: torch.Tensor,
    gamma: torch.Tensor,
    *,
    core_fraction: float,
    reference_length: float,
    block_size: int = RING_BS_BLOCK_SIZE,
    core_radius: float | None = None,
) -> torch.Tensor:
    """Velocity at ``points`` induced by ``rings`` with per-ring ``gamma``.

    Same model as native_ring_velocity_expanded followed by the caller's
    gamma-weighted ring sum, fused into one kernel launch plus one torch
    reduction.  fp64 summation order differs from the tiled torch path
    (rounding level only, like the fused particle kernel).

    Under FLUXV_RING_KERNEL=v3 the launch uses the v3 kernel (see _KERNEL_V3):
    `core_fraction` and `reference_length` are ignored, `core_radius` is the
    fixed core radius of every filament (None = exact law).  Under the default
    the historical kernel and arguments are used unchanged.
    """

    if rings.shape[0] == 0:
        return torch.zeros_like(points)
    device = points.device
    starts = rings.reshape(-1, 3)
    ends = torch.roll(rings, shifts=-1, dims=1).reshape(-1, 3)
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
    filament_gamma = gamma.repeat_interleave(4)
    n_filament = int(core4.shape[0])
    n_blocks = (n_filament + block_size - 1) // block_size
    partials = torch.zeros(
        (points.shape[0], n_blocks, 3), device=device, dtype=torch.float64
    )
    if _KERNEL_V3:
        core4 = (
            torch.zeros_like(core4)
            if core_radius is None
            else torch.full_like(core4, float(core_radius) ** 4)
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
    core_radius: float | None = None,
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
        kv = h_sq / torch.sqrt(h_sq * h_sq + float(core_radius) ** 4)
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
