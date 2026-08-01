"""No-force ALE identity for a material wake potential jump.

The physical sheet-average velocity and a computational surface velocity may
have different tangential components.  If ``mu`` is the material potential
jump, its rate in the computational mesh coordinates is

    d(mu)/dt|mesh + (u_bar - w_mesh).grad_s(mu) = 0,

provided the two velocities have the same normal component.  This module
checks that continuum identity only.  It does not select a discrete transport
scheme, evaluate an induced velocity, or calculate pressure or force.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .distributed_doublet import DistributedDoubletError


def _array(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise DistributedDoubletError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise DistributedDoubletError(
            f"{name} contains non-finite values"
        )
    return array


@dataclass(frozen=True)
class MaterialGaugeTransportIdentity:
    relative_tangential_velocity: np.ndarray
    transport_residual: np.ndarray
    maximum_normal_velocity_mismatch: float
    maximum_surface_gradient_normal_component: float
    maximum_absolute_transport_residual: float
    passed: bool


def material_potential_jump_ale_identity(
    *,
    normal: Any,
    fluid_sheet_average_velocity: Any,
    mesh_velocity: Any,
    surface_gradient: Any,
    mesh_time_derivative: Any,
    normal_velocity_tolerance: float = 1.0e-12,
    surface_gradient_tolerance: float = 1.0e-12,
    transport_residual_tolerance: float = 1.0e-12,
) -> MaterialGaugeTransportIdentity:
    """Evaluate the scalar material-invariance identity in an ALE gauge."""
    normal_array = _array("normal", normal)
    if normal_array.ndim != 2 or normal_array.shape[1] != 3:
        raise DistributedDoubletError(
            "normal must have shape (n,3)"
        )
    shape = normal_array.shape
    fluid = _array(
        "fluid_sheet_average_velocity",
        fluid_sheet_average_velocity,
        shape=shape,
    )
    mesh = _array("mesh_velocity", mesh_velocity, shape=shape)
    gradient = _array("surface_gradient", surface_gradient, shape=shape)
    rate = _array(
        "mesh_time_derivative",
        mesh_time_derivative,
        shape=(len(normal_array),),
    )
    tolerances = (
        normal_velocity_tolerance,
        surface_gradient_tolerance,
        transport_residual_tolerance,
    )
    if any(
        not np.isfinite(value) or value < 0.0
        for value in tolerances
    ):
        raise DistributedDoubletError(
            "ALE identity tolerances must be finite and non-negative"
        )
    normal_norm = np.linalg.norm(normal_array, axis=1)
    if np.any(
        np.abs(normal_norm - 1.0)
        > 64.0 * np.finfo(float).eps
    ):
        raise DistributedDoubletError(
            "normal rows must be unit vectors"
        )
    relative = fluid - mesh
    normal_mismatch = np.einsum(
        "ij,ij->i",
        relative,
        normal_array,
    )
    gradient_normal = np.einsum(
        "ij,ij->i",
        gradient,
        normal_array,
    )
    relative_tangential = (
        relative - normal_mismatch[:, None] * normal_array
    )
    residual = rate + np.einsum(
        "ij,ij->i",
        relative_tangential,
        gradient,
    )
    maximum_normal = float(
        np.max(np.abs(normal_mismatch), initial=0.0)
    )
    maximum_gradient_normal = float(
        np.max(np.abs(gradient_normal), initial=0.0)
    )
    maximum_residual = float(
        np.max(np.abs(residual), initial=0.0)
    )
    passed = (
        maximum_normal <= normal_velocity_tolerance
        and maximum_gradient_normal <= surface_gradient_tolerance
        and maximum_residual <= transport_residual_tolerance
    )
    return MaterialGaugeTransportIdentity(
        relative_tangential_velocity=relative_tangential,
        transport_residual=residual,
        maximum_normal_velocity_mismatch=maximum_normal,
        maximum_surface_gradient_normal_component=(
            maximum_gradient_normal
        ),
        maximum_absolute_transport_residual=maximum_residual,
        passed=passed,
    )
