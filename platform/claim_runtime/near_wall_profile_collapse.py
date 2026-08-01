"""Common profile projection for surface IBL and VES states.

This module exposes an information audit, not a boundary-layer closure.  A
caller supplies the finite-thickness profile and an explicit edge convention.
The same profile is then projected to:

* the velocity-defect moments used by the incompressible surface IBL; and
* the actual mass, momentum and velocity jumps stored by a
  vortex-entrainment sheet (VES).

No edge location, pressure, force, LESP or release amplitude is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NearWallProfileError(ValueError):
    """Invalid finite-thickness near-wall profile or collapse geometry."""


def _vector(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise NearWallProfileError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise NearWallProfileError(f"{name} contains non-finite values")
    return array.copy()


def _unit_vector(name: str, value) -> np.ndarray:
    array = _vector(name, value)
    if abs(np.linalg.norm(array)-1.0) > 1.0e-12:
        raise NearWallProfileError(f"{name} must be a unit vector")
    return array


def _trapezoid(value: np.ndarray, coordinate: np.ndarray, *, axis: int = 0):
    return np.trapezoid(value, coordinate, axis=axis)


@dataclass(frozen=True)
class NearWallProfile:
    """Explicit finite-thickness layer data at one surface location."""

    normal_coordinate: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    external_tangential_velocity: np.ndarray
    outer_velocity_plus: np.ndarray
    outer_velocity_minus: np.ndarray
    sheet_normal: np.ndarray
    edge_convention: str

    def __post_init__(self) -> None:
        coordinate = np.asarray(self.normal_coordinate, dtype=float)
        if coordinate.ndim != 1 or len(coordinate) < 2:
            raise NearWallProfileError(
                "normal_coordinate must be a one-dimensional array with at least two points"
            )
        if not np.all(np.isfinite(coordinate)):
            raise NearWallProfileError("normal_coordinate contains non-finite values")
        if np.any(np.diff(coordinate) <= 0.0):
            raise NearWallProfileError("normal_coordinate must be strictly increasing")

        density = np.asarray(self.density, dtype=float)
        if density.shape != coordinate.shape:
            raise NearWallProfileError(
                f"density must have shape {coordinate.shape}, got {density.shape}"
            )
        if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
            raise NearWallProfileError("density must be finite and strictly positive")

        velocity = np.asarray(self.velocity, dtype=float)
        if velocity.shape != (len(coordinate), 3):
            raise NearWallProfileError(
                f"velocity must have shape ({len(coordinate)},3), got {velocity.shape}"
            )
        if not np.all(np.isfinite(velocity)):
            raise NearWallProfileError("velocity contains non-finite values")

        normal = _unit_vector("sheet_normal", self.sheet_normal)
        external = _vector(
            "external_tangential_velocity",
            self.external_tangential_velocity,
        )
        if abs(external@normal) > 1.0e-11:
            raise NearWallProfileError(
                "external_tangential_velocity must lie in the sheet tangent plane"
            )
        plus = _vector("outer_velocity_plus", self.outer_velocity_plus)
        minus = _vector("outer_velocity_minus", self.outer_velocity_minus)
        if not isinstance(self.edge_convention, str) or not self.edge_convention.strip():
            raise NearWallProfileError("edge_convention must be a non-empty string")

        object.__setattr__(self, "normal_coordinate", coordinate.copy())
        object.__setattr__(self, "density", density.copy())
        object.__setattr__(self, "velocity", velocity.copy())
        object.__setattr__(self, "external_tangential_velocity", external)
        object.__setattr__(self, "outer_velocity_plus", plus)
        object.__setattr__(self, "outer_velocity_minus", minus)
        object.__setattr__(self, "sheet_normal", normal)
        object.__setattr__(self, "edge_convention", self.edge_convention.strip())


@dataclass(frozen=True)
class IBLVESProfileProjection:
    """Co-projected IBL deficit moments and VES actual-layer state."""

    layer_thickness: float
    mass_flux_defect: np.ndarray
    momentum_flux_defect: np.ndarray
    momentum_flux_trace: float
    surface_mass_density: float
    surface_momentum: np.ndarray
    intrinsic_velocity: np.ndarray
    vortex_sheet_strength: np.ndarray
    entrainment_strength: float
    edge_convention: str


def project_near_wall_profile(profile: NearWallProfile) -> IBLVESProfileProjection:
    """Project one explicit layer to both IBL and VES representations.

    The IBL moments use only velocity components tangent to the supplied sheet.
    VES mass and momentum integrate the actual supplied layer fields.
    """
    if not isinstance(profile, NearWallProfile):
        raise NearWallProfileError("profile must be a NearWallProfile")

    coordinate = profile.normal_coordinate
    normal = profile.sheet_normal
    tangent_projector = np.eye(3)-np.outer(normal, normal)
    tangent_velocity = profile.velocity@tangent_projector.T
    defect = profile.external_tangential_velocity-tangent_velocity

    mass_flux_defect = _trapezoid(defect, coordinate, axis=0)
    momentum_integrand = np.einsum("ni,nj->nij", defect, tangent_velocity)
    momentum_flux_defect = _trapezoid(
        momentum_integrand,
        coordinate,
        axis=0,
    )

    surface_mass_density = float(_trapezoid(profile.density, coordinate))
    surface_momentum = _trapezoid(
        profile.density[:, None]*profile.velocity,
        coordinate,
        axis=0,
    )
    intrinsic_velocity = surface_momentum/surface_mass_density

    velocity_jump = profile.outer_velocity_plus-profile.outer_velocity_minus
    vortex_sheet_strength = np.cross(normal, velocity_jump)
    entrainment_strength = -float(normal@velocity_jump)

    return IBLVESProfileProjection(
        layer_thickness=float(coordinate[-1]-coordinate[0]),
        mass_flux_defect=np.asarray(mass_flux_defect, dtype=float),
        momentum_flux_defect=np.asarray(momentum_flux_defect, dtype=float),
        momentum_flux_trace=float(np.trace(momentum_flux_defect)),
        surface_mass_density=surface_mass_density,
        surface_momentum=np.asarray(surface_momentum, dtype=float),
        intrinsic_velocity=np.asarray(intrinsic_velocity, dtype=float),
        vortex_sheet_strength=np.asarray(vortex_sheet_strength, dtype=float),
        entrainment_strength=entrainment_strength,
        edge_convention=profile.edge_convention,
    )
