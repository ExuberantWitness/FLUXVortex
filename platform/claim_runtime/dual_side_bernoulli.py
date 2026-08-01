"""Dual-side moving-surface Bernoulli observer for a thin doublet sheet.

This module does not introduce a second pressure model.  It reconstructs the
two limiting pressures whose difference is exactly the already validated
FLUXV potential-jump pressure:

    chi = phi_plus - phi_minus
    u_plus/minus = u_bar +/- 0.5 grad_s(chi)
    B_plus/minus = D_wall(phi_plus/minus)/Dt
                   - v_wall . u_plus/minus
                   + 0.5 |u_plus/minus|^2
    p_plus/minus / rho = C(t) - B_plus/minus

Therefore

    (p_minus-p_plus)/rho
      = D_wall(chi)/Dt + (u_bar-v_wall).grad_s(chi).

The common Bernoulli gauge ``C(t)`` cancels from the pressure jump and from
all surface pressure gradients.  Side-resolved pressure is needed only to
form the moving-wall circulation-source boundary condition; aerodynamic
force is still assembled once from the pressure jump.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DualSideBernoulliError(ValueError):
    """Invalid dual-side Bernoulli field or orientation."""


def _finite(name: str, value, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise DualSideBernoulliError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise DualSideBernoulliError(f"{name} contains non-finite values")
    return array


def _scalar_field(name: str, value, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(count, float(array))
    if array.shape != (count,):
        raise DualSideBernoulliError(
            f"{name} must be scalar or shape ({count},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise DualSideBernoulliError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class DualSideBernoulli:
    velocity_plus: np.ndarray
    velocity_minus: np.ndarray
    bernoulli_plus: np.ndarray
    bernoulli_minus: np.ndarray
    pressure_plus: np.ndarray
    pressure_minus: np.ndarray
    pressure_jump: np.ndarray
    unified_pressure_jump: np.ndarray
    max_jump_identity_residual: float


@dataclass(frozen=True)
class PairedWallSourceReport:
    source_plus: np.ndarray
    source_minus: np.ndarray
    paired_source: np.ndarray
    expected_paired_source: np.ndarray
    max_identity_residual: float
    passed: bool


def dual_side_moving_bernoulli(
    *,
    density: float,
    mean_potential_wall_rate,
    potential_jump_wall_rate,
    mean_velocity,
    potential_jump_surface_gradient,
    wall_velocity,
    bernoulli_gauge=0.0,
) -> DualSideBernoulli:
    """Recover the two Bernoulli limits without changing the pressure jump.

    ``potential_jump_surface_gradient`` is ``grad_s(phi_plus-phi_minus)``.
    ``mean_velocity`` is the principal-value/sheet-average inertial velocity,
    not the velocity relative to the wall.  ``bernoulli_gauge`` has pressure
    divided by density units and must be common to both sides.
    """
    if not np.isfinite(density) or density <= 0.0:
        raise DualSideBernoulliError(
            f"density must be positive and finite, got {density}"
        )
    velocity_bar = np.asarray(mean_velocity, dtype=float)
    if velocity_bar.ndim != 2 or velocity_bar.shape[1] != 3:
        raise DualSideBernoulliError(
            "mean_velocity must have shape (n,3)"
        )
    count = len(velocity_bar)
    if not np.all(np.isfinite(velocity_bar)):
        raise DualSideBernoulliError(
            "mean_velocity contains non-finite values"
        )
    gradient = _finite(
        "potential_jump_surface_gradient",
        potential_jump_surface_gradient,
        shape=(count, 3),
    )
    wall = _finite(
        "wall_velocity",
        wall_velocity,
        shape=(count, 3),
    )
    mean_rate = _scalar_field(
        "mean_potential_wall_rate",
        mean_potential_wall_rate,
        count,
    )
    jump_rate = _scalar_field(
        "potential_jump_wall_rate",
        potential_jump_wall_rate,
        count,
    )
    gauge = _scalar_field("bernoulli_gauge", bernoulli_gauge, count)

    velocity_plus = velocity_bar + 0.5 * gradient
    velocity_minus = velocity_bar - 0.5 * gradient
    rate_plus = mean_rate + 0.5 * jump_rate
    rate_minus = mean_rate - 0.5 * jump_rate
    bernoulli_plus = (
        rate_plus
        - np.einsum("ij,ij->i", wall, velocity_plus)
        + 0.5 * np.einsum("ij,ij->i", velocity_plus, velocity_plus)
    )
    bernoulli_minus = (
        rate_minus
        - np.einsum("ij,ij->i", wall, velocity_minus)
        + 0.5 * np.einsum("ij,ij->i", velocity_minus, velocity_minus)
    )
    pressure_plus = density * (gauge - bernoulli_plus)
    pressure_minus = density * (gauge - bernoulli_minus)
    pressure_jump = pressure_minus - pressure_plus
    unified = density * (
        jump_rate
        + np.einsum("ij,ij->i", velocity_bar - wall, gradient)
    )
    residual = float(
        np.max(np.abs(pressure_jump - unified), initial=0.0)
    )
    return DualSideBernoulli(
        velocity_plus=velocity_plus,
        velocity_minus=velocity_minus,
        bernoulli_plus=bernoulli_plus,
        bernoulli_minus=bernoulli_minus,
        pressure_plus=pressure_plus,
        pressure_minus=pressure_minus,
        pressure_jump=pressure_jump,
        unified_pressure_jump=unified,
        max_jump_identity_residual=residual,
    )


def moving_wall_circulation_source(
    *,
    normal_into_fluid,
    wall_acceleration,
    specific_pressure_gradient,
    body_force_potential_gradient=None,
) -> np.ndarray:
    """Terrington et al. (2022), Eq. 4.30, with explicit orientation.

    The returned vector has acceleration units.  ``normal_into_fluid`` is the
    oriented normal used for that physical wall side.  The caller must not
    silently reuse one normal for both sides of a thin sheet.
    """
    normal = np.asarray(normal_into_fluid, dtype=float)
    if normal.ndim != 2 or normal.shape[1] != 3:
        raise DualSideBernoulliError(
            "normal_into_fluid must have shape (n,3)"
        )
    count = len(normal)
    if not np.all(np.isfinite(normal)):
        raise DualSideBernoulliError(
            "normal_into_fluid contains non-finite values"
        )
    norm = np.linalg.norm(normal, axis=1)
    if np.any(norm <= 0.0) or np.max(
        np.abs(norm - 1.0), initial=0.0
    ) > 1.0e-10:
        raise DualSideBernoulliError(
            "normal_into_fluid must contain unit vectors"
        )
    acceleration = _finite(
        "wall_acceleration",
        wall_acceleration,
        shape=(count, 3),
    )
    pressure_gradient = _finite(
        "specific_pressure_gradient",
        specific_pressure_gradient,
        shape=(count, 3),
    )
    if body_force_potential_gradient is None:
        body_gradient = np.zeros((count, 3), dtype=float)
    else:
        body_gradient = _finite(
            "body_force_potential_gradient",
            body_force_potential_gradient,
            shape=(count, 3),
        )
    return np.cross(
        normal,
        acceleration + pressure_gradient + body_gradient,
    )


def paired_thin_sheet_source_report(
    *,
    normal_plus,
    wall_acceleration,
    specific_pressure_gradient_plus,
    specific_pressure_gradient_minus,
    specific_pressure_jump_gradient,
    tolerance: float = 1.0e-12,
) -> PairedWallSourceReport:
    """Check the two-sided wall-source identity for a zero-thickness sheet.

    The pressure jump convention is ``p_minus-p_plus``.  With the minus-side
    normal equal to ``-normal_plus``, wall acceleration cancels and

        sigma_plus + sigma_minus
          = -normal_plus x grad_s[(p_minus-p_plus)/rho].
    """
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise DualSideBernoulliError(
            "tolerance must be finite and non-negative"
        )
    normal = np.asarray(normal_plus, dtype=float)
    if normal.ndim != 2 or normal.shape[1] != 3:
        raise DualSideBernoulliError(
            "normal_plus must have shape (n,3)"
        )
    count = len(normal)
    acceleration = _finite(
        "wall_acceleration",
        wall_acceleration,
        shape=(count, 3),
    )
    gradient_plus = _finite(
        "specific_pressure_gradient_plus",
        specific_pressure_gradient_plus,
        shape=(count, 3),
    )
    gradient_minus = _finite(
        "specific_pressure_gradient_minus",
        specific_pressure_gradient_minus,
        shape=(count, 3),
    )
    jump_gradient = _finite(
        "specific_pressure_jump_gradient",
        specific_pressure_jump_gradient,
        shape=(count, 3),
    )
    source_plus = moving_wall_circulation_source(
        normal_into_fluid=normal,
        wall_acceleration=acceleration,
        specific_pressure_gradient=gradient_plus,
    )
    source_minus = moving_wall_circulation_source(
        normal_into_fluid=-normal,
        wall_acceleration=acceleration,
        specific_pressure_gradient=gradient_minus,
    )
    paired = source_plus + source_minus
    expected = -np.cross(normal, jump_gradient)
    residual = float(
        np.max(np.linalg.norm(paired - expected, axis=1), initial=0.0)
    )
    return PairedWallSourceReport(
        source_plus=source_plus,
        source_minus=source_minus,
        paired_source=paired,
        expected_paired_source=expected,
        max_identity_residual=residual,
        passed=residual <= tolerance,
    )
