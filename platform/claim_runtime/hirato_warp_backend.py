"""Warp execution backend for the no-force Hirato spatial-state shadow.

This module changes only how the registered Biot--Savart sums are evaluated.
It does not own circulation, topology, convection history, pressure, or force.
The NumPy implementation in :mod:`hirato_shadow` remains the equation oracle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from fluxvortex.warp_fsi.config import DTYPE


V3 = wp.vec3d


@wp.func
def _mirror_y(point: V3) -> V3:
    return V3(point[0], -point[1], point[2])


@wp.func
def _vseg_lamb_oseen(
    point: V3,
    start: V3,
    end: V3,
    core_radius: DTYPE,
) -> V3:
    r_start = point - start
    r_end = point - end
    filament = end - start
    cross = wp.cross(r_start, r_end)
    cross_sq = wp.dot(cross, cross)
    filament_sq = wp.dot(filament, filament)
    exponent = cross_sq / (
        filament_sq * core_radius * core_radius + wp.float64(1.0e-300)
    )
    cutoff = wp.float64(1.0) - wp.exp(-exponent)
    if exponent < wp.float64(1.0e-6):
        cutoff = exponent - wp.float64(0.5) * exponent * exponent
    start_norm = wp.sqrt(wp.dot(r_start, r_start) + wp.float64(1.0e-20))
    end_norm = wp.sqrt(wp.dot(r_end, r_end) + wp.float64(1.0e-20))
    coefficient = wp.dot(
        filament,
        r_start / start_norm - r_end / end_norm,
    )
    return (
        wp.float64(1.0)
        / (wp.float64(4.0) * wp.float64(3.141592653589793))
        * coefficient
        * cutoff
        / (cross_sq + wp.float64(1.0e-300))
        * cross
    )


@wp.func
def _ring_vel_lamb_oseen(
    point: V3,
    c0: V3,
    c1: V3,
    c2: V3,
    c3: V3,
    core_radius: DTYPE,
) -> V3:
    return (
        _vseg_lamb_oseen(point, c0, c1, core_radius)
        + _vseg_lamb_oseen(point, c1, c2, core_radius)
        + _vseg_lamb_oseen(point, c2, c3, core_radius)
        + _vseg_lamb_oseen(point, c3, c0, core_radius)
    )


@wp.func
def _ring_velocity_with_image(
    point: V3,
    rings: wp.array(dtype=V3, ndim=2),
    index: int,
    core_radius: DTYPE,
    mirror_symmetry: bool,
) -> V3:
    c0 = rings[index, 0]
    c1 = rings[index, 1]
    c2 = rings[index, 2]
    c3 = rings[index, 3]
    velocity = _ring_vel_lamb_oseen(
        point,
        c0,
        c1,
        c2,
        c3,
        core_radius,
    )
    if mirror_symmetry:
        m0 = _mirror_y(c0)
        m1 = _mirror_y(c1)
        m2 = _mirror_y(c2)
        m3 = _mirror_y(c3)
        velocity = velocity + _ring_vel_lamb_oseen(
            point,
            m0,
            m3,
            m2,
            m1,
            core_radius,
        )
    return velocity


@wp.kernel
def _velocity_channels_kernel(
    points: wp.array(dtype=V3),
    bound_rings: wp.array(dtype=V3, ndim=2),
    bound_gamma: wp.array(dtype=DTYPE),
    bound_count: int,
    tev_rings: wp.array(dtype=V3, ndim=2),
    tev_gamma: wp.array(dtype=DTYPE),
    tev_count: int,
    lev_rings: wp.array(dtype=V3, ndim=2),
    lev_gamma: wp.array(dtype=DTYPE),
    lev_count: int,
    u_infinity: V3,
    core_radius: DTYPE,
    mirror_symmetry: bool,
    bound_out: wp.array(dtype=V3),
    tev_out: wp.array(dtype=V3),
    lev_out: wp.array(dtype=V3),
    total_out: wp.array(dtype=V3),
):
    point_index = wp.tid()
    point = points[point_index]
    bound_velocity = V3(0.0, 0.0, 0.0)
    tev_velocity = V3(0.0, 0.0, 0.0)
    lev_velocity = V3(0.0, 0.0, 0.0)
    for ring_index in range(bound_count):
        bound_velocity = (
            bound_velocity
            + bound_gamma[ring_index]
            * _ring_velocity_with_image(
                point,
                bound_rings,
                ring_index,
                core_radius,
                mirror_symmetry,
            )
        )
    for ring_index in range(tev_count):
        tev_velocity = (
            tev_velocity
            + tev_gamma[ring_index]
            * _ring_velocity_with_image(
                point,
                tev_rings,
                ring_index,
                core_radius,
                mirror_symmetry,
            )
        )
    for ring_index in range(lev_count):
        lev_velocity = (
            lev_velocity
            + lev_gamma[ring_index]
            * _ring_velocity_with_image(
                point,
                lev_rings,
                ring_index,
                core_radius,
                mirror_symmetry,
            )
        )
    bound_out[point_index] = bound_velocity
    tev_out[point_index] = tev_velocity
    lev_out[point_index] = lev_velocity
    total_out[point_index] = (
        u_infinity + bound_velocity + tev_velocity + lev_velocity
    )


@dataclass(frozen=True)
class WarpVelocityChannels:
    freestream: np.ndarray
    bound: np.ndarray
    tev: np.ndarray
    lev: np.ndarray
    total: np.ndarray


def _ring_array(value, *, device: str):
    array = np.asarray(value, dtype=np.float64)
    if array.shape[1:] != (4, 3):
        raise ValueError(f"ring array must have shape (n,4,3), got {array.shape}")
    return wp.array(array, dtype=V3, device=device)


def _gamma_array(value, count: int, *, device: str):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"gamma must have shape {(count,)}, got {array.shape}")
    return wp.array(array, dtype=DTYPE, device=device)


def velocity_channels(
    points,
    *,
    bound_rings,
    bound_gamma,
    tev_rings,
    tev_gamma,
    lev_rings,
    lev_gamma,
    u_infinity,
    core_radius: float,
    mirror_symmetry: bool,
    device: str,
) -> WarpVelocityChannels:
    """Evaluate the same five velocity channels as the NumPy ledger."""
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"points must have shape (n,3), got {point_array.shape}")
    if len(point_array) == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return WarpVelocityChannels(empty, empty, empty, empty, empty)
    u_inf = np.asarray(u_infinity, dtype=np.float64)
    if u_inf.shape != (3,):
        raise ValueError(f"u_infinity must have shape (3,), got {u_inf.shape}")
    if not np.isfinite(core_radius) or core_radius <= 0.0:
        raise ValueError("core_radius must be positive and finite")

    bound_count = len(bound_rings)
    tev_count = len(tev_rings)
    lev_count = len(lev_rings)
    points_wp = wp.array(point_array, dtype=V3, device=device)
    bound_rings_wp = _ring_array(bound_rings, device=device)
    tev_rings_wp = _ring_array(tev_rings, device=device)
    lev_rings_wp = _ring_array(lev_rings, device=device)
    bound_gamma_wp = _gamma_array(
        bound_gamma,
        bound_count,
        device=device,
    )
    tev_gamma_wp = _gamma_array(tev_gamma, tev_count, device=device)
    lev_gamma_wp = _gamma_array(lev_gamma, lev_count, device=device)
    bound_out = wp.zeros(len(point_array), dtype=V3, device=device)
    tev_out = wp.zeros(len(point_array), dtype=V3, device=device)
    lev_out = wp.zeros(len(point_array), dtype=V3, device=device)
    total_out = wp.zeros(len(point_array), dtype=V3, device=device)
    wp.launch(
        _velocity_channels_kernel,
        dim=len(point_array),
        inputs=[
            points_wp,
            bound_rings_wp,
            bound_gamma_wp,
            bound_count,
            tev_rings_wp,
            tev_gamma_wp,
            tev_count,
            lev_rings_wp,
            lev_gamma_wp,
            lev_count,
            V3(*[float(value) for value in u_inf]),
            float(core_radius),
            bool(mirror_symmetry),
        ],
        outputs=[bound_out, tev_out, lev_out, total_out],
        device=device,
    )
    freestream = np.broadcast_to(u_inf, point_array.shape).copy()
    return WarpVelocityChannels(
        freestream=freestream,
        bound=bound_out.numpy(),
        tev=tev_out.numpy(),
        lev=lev_out.numpy(),
        total=total_out.numpy(),
    )
