"""Equation-level vortex-entrainment-sheet (VES) reference operators.

The spatial influence follows DeVoria & Mohseni (JFM 866, 2019), Eq. (15):

    u(x) = 1/(4*pi) integral_S
           [gamma x (x-xs) - q (x-xs)] / |x-xs|^3 dA.

``gamma`` is the tangential vortex-sheet strength and ``q=-[[u_n]]`` is the
independent entrainment/source-sheet strength.  The module also exposes the
local mass and momentum residuals of their Eqs. (6)-(7), plus a named release
junction ledger.

This is an off-sheet CPU equation oracle.  It intentionally has no core
radius, on-sheet self-advection, LESP, aerodynamic force, or target-load input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class VortexEntrainmentError(ValueError):
    """Invalid VES state, quadrature, or conservation input."""


def _finite(name: str, value, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise VortexEntrainmentError(
            f"{name} must have ndim={ndim}, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise VortexEntrainmentError(f"{name} contains non-finite values")
    return array


def _vectors(name: str, value, *, length: int | None = None) -> np.ndarray:
    array = _finite(name, value, ndim=2)
    if array.shape[1] != 3:
        raise VortexEntrainmentError(
            f"{name} must have shape (n,3), got {array.shape}"
        )
    if length is not None and len(array) != length:
        raise VortexEntrainmentError(
            f"{name} must have length {length}, got {len(array)}"
        )
    return array


def _scalars(name: str, value, *, length: int) -> np.ndarray:
    array = _finite(name, value, ndim=1)
    if array.shape != (length,):
        raise VortexEntrainmentError(
            f"{name} must have shape ({length},), got {array.shape}"
        )
    return array


@dataclass(frozen=True)
class VortexEntrainmentInfluence:
    velocity: np.ndarray
    vortex_velocity: np.ndarray
    entrainment_velocity: np.ndarray
    total_circulation_vector: np.ndarray
    total_entrainment_rate: float


@dataclass(frozen=True)
class VortexEntrainmentBalance:
    mass_residual: float
    momentum_residual: np.ndarray
    maximum_absolute_residual: float


@dataclass(frozen=True)
class ReleaseJunctionReport:
    circulation_residual: np.ndarray
    mass_residual: float
    momentum_residual: np.ndarray
    entrainment_residual: float
    maximum_absolute_residual: float


def vortex_entrainment_velocity(
    target_points,
    source_points,
    source_area_weights,
    vortex_sheet_strength,
    entrainment_strength,
    source_normals,
) -> VortexEntrainmentInfluence:
    """Evaluate the off-sheet VES influence without regularization.

    ``source_area_weights`` already includes surface quadrature weights.
    Targets coincident with a source quadrature point are rejected; this
    routine never replaces the singular distance with a core radius.
    """
    target = _vectors("target_points", target_points)
    source = _vectors("source_points", source_points)
    count = len(source)
    area = _scalars("source_area_weights", source_area_weights, length=count)
    gamma = _vectors(
        "vortex_sheet_strength",
        vortex_sheet_strength,
        length=count,
    )
    q = _scalars(
        "entrainment_strength",
        entrainment_strength,
        length=count,
    )
    normals = _vectors("source_normals", source_normals, length=count)
    if np.any(area <= 0.0):
        raise VortexEntrainmentError(
            "source_area_weights must be strictly positive"
        )
    normal_length = np.linalg.norm(normals, axis=1)
    if np.any(normal_length <= 0.0):
        raise VortexEntrainmentError("source_normals must be non-zero")
    normals = normals / normal_length[:, None]
    scale = np.maximum(np.linalg.norm(gamma, axis=1), 1.0)
    tangent_error = np.abs(np.einsum("ij,ij->i", gamma, normals))
    if np.any(tangent_error > 1.0e-12*scale):
        raise VortexEntrainmentError(
            "vortex_sheet_strength must be tangent to the source surface"
        )

    separation = target[:, None, :] - source[None, :, :]
    radius_squared = np.einsum(
        "tqj,tqj->tq",
        separation,
        separation,
    )
    geometry_scale = max(
        float(np.max(np.linalg.norm(source, axis=1), initial=0.0)),
        float(np.max(np.linalg.norm(target, axis=1), initial=0.0)),
        1.0,
    )
    singular_limit = (
        64.0*np.finfo(float).eps*geometry_scale
    ) ** 2
    if np.any(radius_squared <= singular_limit):
        raise VortexEntrainmentError(
            "off-sheet oracle received a target on a source point"
        )
    inverse_radius_cubed = radius_squared ** -1.5
    weighted_kernel = (
        inverse_radius_cubed*area[None, :]/(4.0*np.pi)
    )
    vortex_integrand = np.cross(
        gamma[None, :, :],
        separation,
    )
    entrainment_integrand = -q[None, :, None]*separation
    vortex_velocity = np.einsum(
        "tq,tqj->tj",
        weighted_kernel,
        vortex_integrand,
    )
    entrainment_velocity = np.einsum(
        "tq,tqj->tj",
        weighted_kernel,
        entrainment_integrand,
    )
    velocity = vortex_velocity+entrainment_velocity
    return VortexEntrainmentInfluence(
        velocity=velocity,
        vortex_velocity=vortex_velocity,
        entrainment_velocity=entrainment_velocity,
        total_circulation_vector=np.einsum(
            "q,qj->j",
            area,
            gamma,
        ),
        total_entrainment_rate=float(area@q),
    )


def equal_density_mass_flux_jump(
    bulk_density: float,
    entrainment_strength,
) -> np.ndarray:
    """Return ``[[rho (u-v).n]]=-rho*q`` for equal outer densities."""
    density = float(bulk_density)
    if not np.isfinite(density) or density <= 0.0:
        raise VortexEntrainmentError(
            "bulk_density must be finite and strictly positive"
        )
    q = _finite("entrainment_strength", entrainment_strength)
    return -density*q


def vortex_entrainment_local_balance(
    *,
    surface_mass_density: float,
    material_surface_mass_rate: float,
    surface_velocity_divergence: float,
    outer_mass_flux_jump: float,
    material_surface_acceleration,
    surface_stress_divergence,
    surface_body_force,
    outer_momentum_flux_jump,
    pressure_jump: float,
    normal,
    shear_stress_jump,
) -> VortexEntrainmentBalance:
    """Evaluate the local residuals of VES mass and momentum.

    Sign convention:

    ``outer_mass_flux_jump = [[rho (u-v).n]]`` and
    ``outer_momentum_flux_jump = [[rho (u-v)((u-v).n)]]``.
    """
    rho_s = float(surface_mass_density)
    drho_s = float(material_surface_mass_rate)
    divergence = float(surface_velocity_divergence)
    mass_jump = float(outer_mass_flux_jump)
    pressure = float(pressure_jump)
    scalar_values = (rho_s, drho_s, divergence, mass_jump, pressure)
    if not np.all(np.isfinite(scalar_values)) or rho_s < 0.0:
        raise VortexEntrainmentError(
            "surface mass inputs must be finite and rho_s non-negative"
        )
    acceleration = _finite(
        "material_surface_acceleration",
        material_surface_acceleration,
        ndim=1,
    )
    surface_stress = _finite(
        "surface_stress_divergence",
        surface_stress_divergence,
        ndim=1,
    )
    body_force = _finite(
        "surface_body_force",
        surface_body_force,
        ndim=1,
    )
    momentum_jump = _finite(
        "outer_momentum_flux_jump",
        outer_momentum_flux_jump,
        ndim=1,
    )
    normal_vector = _finite("normal", normal, ndim=1)
    shear_jump = _finite(
        "shear_stress_jump",
        shear_stress_jump,
        ndim=1,
    )
    for name, value in (
        ("material_surface_acceleration", acceleration),
        ("surface_stress_divergence", surface_stress),
        ("surface_body_force", body_force),
        ("outer_momentum_flux_jump", momentum_jump),
        ("normal", normal_vector),
        ("shear_stress_jump", shear_jump),
    ):
        if value.shape != (3,):
            raise VortexEntrainmentError(
                f"{name} must have shape (3,), got {value.shape}"
            )
    normal_norm = np.linalg.norm(normal_vector)
    if normal_norm <= 0.0:
        raise VortexEntrainmentError("normal must be non-zero")
    normal_vector = normal_vector/normal_norm

    mass_residual = drho_s+rho_s*divergence+mass_jump
    momentum_residual = (
        rho_s*acceleration
        - (surface_stress+rho_s*body_force)
        + momentum_jump
        + pressure*normal_vector
        - shear_jump
    )
    maximum = float(max(
        abs(mass_residual),
        np.max(np.abs(momentum_residual), initial=0.0),
    ))
    return VortexEntrainmentBalance(
        mass_residual=float(mass_residual),
        momentum_residual=momentum_residual,
        maximum_absolute_residual=maximum,
    )


def release_junction_report(
    *,
    incoming_circulation_rate,
    newborn_circulation_rate,
    incoming_mass_rate,
    newborn_mass_rate: float,
    incoming_momentum_rate,
    newborn_momentum_rate,
    incoming_entrainment_rate,
    newborn_entrainment_rate: float,
) -> ReleaseJunctionReport:
    """Audit a named IBL-to-newborn-VES conservation junction.

    Incoming arrays may contain contributions from multiple surface cells or
    sides.  No missing channel is inferred from another residual.
    """
    circulation_in = _vectors(
        "incoming_circulation_rate",
        incoming_circulation_rate,
    )
    circulation_out = _finite(
        "newborn_circulation_rate",
        newborn_circulation_rate,
        ndim=1,
    )
    momentum_in = _vectors(
        "incoming_momentum_rate",
        incoming_momentum_rate,
    )
    momentum_out = _finite(
        "newborn_momentum_rate",
        newborn_momentum_rate,
        ndim=1,
    )
    mass_in = _finite("incoming_mass_rate", incoming_mass_rate, ndim=1)
    entrainment_in = _finite(
        "incoming_entrainment_rate",
        incoming_entrainment_rate,
        ndim=1,
    )
    if circulation_out.shape != (3,) or momentum_out.shape != (3,):
        raise VortexEntrainmentError(
            "newborn circulation and momentum rates must have shape (3,)"
        )
    if len(mass_in) == 0 or len(entrainment_in) == 0:
        raise VortexEntrainmentError(
            "release junction requires explicit mass and entrainment inputs"
        )
    newborn_mass = float(newborn_mass_rate)
    newborn_entrainment = float(newborn_entrainment_rate)
    if not np.isfinite(newborn_mass) or not np.isfinite(newborn_entrainment):
        raise VortexEntrainmentError(
            "newborn scalar rates must be finite"
        )

    circulation_residual = (
        circulation_out-np.sum(circulation_in, axis=0)
    )
    mass_residual = newborn_mass-float(np.sum(mass_in))
    momentum_residual = momentum_out-np.sum(momentum_in, axis=0)
    entrainment_residual = (
        newborn_entrainment-float(np.sum(entrainment_in))
    )
    maximum = float(max(
        np.max(np.abs(circulation_residual), initial=0.0),
        abs(mass_residual),
        np.max(np.abs(momentum_residual), initial=0.0),
        abs(entrainment_residual),
    ))
    return ReleaseJunctionReport(
        circulation_residual=circulation_residual,
        mass_residual=mass_residual,
        momentum_residual=momentum_residual,
        entrainment_residual=entrainment_residual,
        maximum_absolute_residual=maximum,
    )

