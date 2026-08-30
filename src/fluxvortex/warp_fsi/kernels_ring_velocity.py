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

import torch
import warp as wp

RING_BS_BLOCK_SIZE = 256

_C_INV_4PI = 1.0 / (4.0 * math.pi)
_C_FP64_EPS = 2.220446049250313e-16  # torch.finfo(float64).eps
_C_EPS_V = 1.0e-9                     # frozen _EPS_V regularization


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


def native_ring_velocity_fused(
    points: torch.Tensor,
    rings: torch.Tensor,
    gamma: torch.Tensor,
    *,
    core_fraction: float,
    reference_length: float,
    block_size: int = RING_BS_BLOCK_SIZE,
) -> torch.Tensor:
    """Velocity at ``points`` induced by ``rings`` with per-ring ``gamma``.

    Same model as native_ring_velocity_expanded followed by the caller's
    gamma-weighted ring sum, fused into one kernel launch plus one torch
    reduction.  fp64 summation order differs from the tiled torch path
    (rounding level only, like the fused particle kernel).
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
_EPS_V = 1.0e-9


def native_ring_velocity_expanded(
    points: torch.Tensor,
    rings: torch.Tensor,
    *,
    core_fraction: float,
    reference_length: float,
) -> torch.Tensor:
    """Finite-ring influence with the paper's fourth-order core model."""

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
