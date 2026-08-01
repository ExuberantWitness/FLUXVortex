"""Conservative aerodynamic-point to structural-coordinate load transfer.

The only admissible transfer is the transpose of the current kinematic map:

    delta_x_a = J(q_s) delta_q_s
    Q_s       = J(q_s)^T f_a

This module intentionally has no stiffness, mass, area weighting, target
strain, or aerodynamic closure. Aerodynamic forces remain attached to their
physical application points and are transferred by discrete virtual work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class LoadTransferError(ValueError):
    """Invalid point load or kinematic Jacobian."""


def _finite(name: str, value, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise LoadTransferError(
            f"{name} must have ndim={ndim}, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise LoadTransferError(f"{name} contains non-finite values")
    return array


def skew(vector) -> np.ndarray:
    """Matrix ``[r]_x`` such that ``[r]_x v = r cross v``."""
    value = _finite("vector", vector, ndim=1)
    if value.shape != (3,):
        raise LoadTransferError(
            f"vector must have shape (3,), got {value.shape}"
        )
    x, y, z = value
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def rigid_body_jacobian(points, *, origin=None) -> np.ndarray:
    """Rigid translations/rotations at point locations.

    Generalized-coordinate order is ``(tx,ty,tz,rx,ry,rz)`` and
    ``delta_x = delta_t + delta_theta cross (x-origin)``.
    """
    point_array = _finite("points", points, ndim=2)
    if point_array.shape[1] != 3:
        raise LoadTransferError(
            f"points must have shape (n,3), got {point_array.shape}"
        )
    if origin is None:
        origin_array = np.zeros(3)
    else:
        origin_array = _finite("origin", origin, ndim=1)
        if origin_array.shape != (3,):
            raise LoadTransferError(
                f"origin must have shape (3,), got {origin_array.shape}"
            )
    jacobian = np.zeros((len(point_array), 3, 6))
    jacobian[:, :, :3] = np.eye(3)
    for index, arm in enumerate(point_array - origin_array):
        jacobian[index, :, 3:] = -skew(arm)
    return jacobian


@dataclass(frozen=True)
class VirtualWorkReport:
    max_absolute_residual: float
    max_relative_residual: float
    probes: int
    passed: bool


@dataclass(frozen=True)
class RigidResultantReport:
    force_residual: np.ndarray
    moment_residual: np.ndarray
    max_absolute_residual: float
    passed: bool


@dataclass(frozen=True)
class GeneralizedLoad:
    values: np.ndarray
    point_force: np.ndarray
    kinematic_jacobian: np.ndarray

    def virtual_work_report(
        self,
        generalized_variations,
        *,
        absolute_tolerance: float = 1.0e-12,
        relative_tolerance: float = 1.0e-12,
    ) -> VirtualWorkReport:
        variations = _finite(
            "generalized_variations",
            generalized_variations,
            ndim=2,
        )
        if variations.shape[1] != len(self.values):
            raise LoadTransferError(
                "generalized variation width does not match generalized load"
            )
        if (
            absolute_tolerance < 0.0
            or relative_tolerance < 0.0
            or not np.isfinite(absolute_tolerance)
            or not np.isfinite(relative_tolerance)
        ):
            raise LoadTransferError(
                "virtual-work tolerances must be finite and non-negative"
            )
        structural_work = variations @ self.values
        aerodynamic_variation = np.einsum(
            "ijk,pk->pij",
            self.kinematic_jacobian,
            variations,
        )
        aerodynamic_work = np.einsum(
            "pij,ij->p",
            aerodynamic_variation,
            self.point_force,
        )
        residual = structural_work - aerodynamic_work
        absolute = np.abs(residual)
        scale = np.maximum(
            np.maximum(np.abs(structural_work), np.abs(aerodynamic_work)),
            absolute_tolerance,
        )
        relative = absolute / scale
        max_absolute = float(np.max(absolute, initial=0.0))
        max_relative = float(np.max(relative, initial=0.0))
        return VirtualWorkReport(
            max_absolute_residual=max_absolute,
            max_relative_residual=max_relative,
            probes=len(variations),
            passed=(
                max_absolute <= absolute_tolerance
                or max_relative <= relative_tolerance
            ),
        )


def transfer_generalized(point_force, kinematic_jacobian) -> GeneralizedLoad:
    """Apply ``Q=J^T f`` without any post-load redistribution."""
    force = _finite("point_force", point_force, ndim=2)
    jacobian = _finite(
        "kinematic_jacobian",
        kinematic_jacobian,
        ndim=3,
    )
    if force.shape[1] != 3:
        raise LoadTransferError(
            f"point_force must have shape (n,3), got {force.shape}"
        )
    if jacobian.shape[:2] != force.shape:
        raise LoadTransferError(
            "kinematic_jacobian must have shape (npoint,3,ndof)"
        )
    values = np.einsum("ijk,ij->k", jacobian, force)
    if not np.all(np.isfinite(values)):
        raise LoadTransferError("generalized load contains non-finite values")
    return GeneralizedLoad(
        values=values,
        point_force=force.copy(),
        kinematic_jacobian=jacobian.copy(),
    )


def resultant(point_position, point_force, *, origin=None) -> tuple[np.ndarray, np.ndarray]:
    """Return physical total force and moment about ``origin``."""
    position = _finite("point_position", point_position, ndim=2)
    force = _finite("point_force", point_force, ndim=2)
    if position.shape != force.shape or position.shape[1] != 3:
        raise LoadTransferError(
            "point_position and point_force must both have shape (n,3)"
        )
    if origin is None:
        origin_array = np.zeros(3)
    else:
        origin_array = _finite("origin", origin, ndim=1)
        if origin_array.shape != (3,):
            raise LoadTransferError(
                f"origin must have shape (3,), got {origin_array.shape}"
            )
    total_force = np.sum(force, axis=0)
    total_moment = np.sum(
        np.cross(position - origin_array, force),
        axis=0,
    )
    return total_force, total_moment


def rigid_resultant_report(
    point_position,
    point_force,
    generalized_load,
    *,
    origin=None,
    translation_columns=(0, 1, 2),
    rotation_columns=(3, 4, 5),
    tolerance: float = 1.0e-12,
) -> RigidResultantReport:
    """Audit embedded rigid-body columns of an arbitrary structural map."""
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise LoadTransferError("tolerance must be finite and non-negative")
    values = _finite("generalized_load", generalized_load, ndim=1)
    translation_columns = tuple(int(index) for index in translation_columns)
    rotation_columns = tuple(int(index) for index in rotation_columns)
    if len(translation_columns) != 3 or len(rotation_columns) != 3:
        raise LoadTransferError(
            "translation_columns and rotation_columns must each have length 3"
        )
    selected = translation_columns + rotation_columns
    if min(selected) < 0 or max(selected) >= len(values):
        raise LoadTransferError("rigid-body column index is out of range")
    force, moment = resultant(
        point_position,
        point_force,
        origin=origin,
    )
    force_residual = values[list(translation_columns)] - force
    moment_residual = values[list(rotation_columns)] - moment
    maximum = float(
        max(
            np.max(np.abs(force_residual), initial=0.0),
            np.max(np.abs(moment_residual), initial=0.0),
        )
    )
    return RigidResultantReport(
        force_residual=force_residual,
        moment_residual=moment_residual,
        max_absolute_residual=maximum,
        passed=maximum <= tolerance,
    )


def transfer_linear_nodal(aero_to_structure_displacement, point_force) -> np.ndarray:
    """Linear special case ``f_s=H^T f_a``."""
    mapping = _finite(
        "aero_to_structure_displacement",
        aero_to_structure_displacement,
        ndim=2,
    )
    force = _finite("point_force", point_force, ndim=2)
    if force.shape[1] != 3 or mapping.shape[0] != force.size:
        raise LoadTransferError(
            "mapping rows must equal 3*npoint and point_force must be (n,3)"
        )
    return mapping.T @ force.reshape(-1)
