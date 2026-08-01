"""Source-exact interval discretization for the Riziotis 2-D IBL equations.

Riziotis (2003), Eqs. (7.111)--(7.113), rewrites the momentum and
kinetic-energy integral equations over two adjacent control points using
logarithmic increments and arithmetic interval-centre values.  This module
implements that algebra only.  It does not:

* select a laminar/turbulent closure;
* initialize the stagnation point;
* choose the source-documented backward-space stability branch;
* move a separation point or remap state; or
* solve a viscous--inviscid Newton system.

Those omissions are deliberate claim boundaries.  The functions here are
equation oracles that later assembly code can call without changing the
frozen source equations or consulting any target-force data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .svi_dw_types import SVIDWValidationError


def _finite_scalar(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise SVIDWValidationError(f"{name} must be finite")
    return result


def _positive_scalar(name: str, value: Any) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise SVIDWValidationError(f"{name} must be positive")
    return result


def _nonnegative_scalar(name: str, value: Any) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise SVIDWValidationError(f"{name} must be non-negative")
    return result


def bdf2_time_derivative(
    current: Any,
    previous: Any,
    previous_previous: Any,
    *,
    time_step: float,
) -> float | np.ndarray:
    """Return the fixed-step BDF2 derivative in thesis Eq. (7.113)."""
    dt = _positive_scalar("time_step", time_step)
    current_array = np.asarray(current, dtype=float)
    previous_array = np.asarray(previous, dtype=float)
    older_array = np.asarray(previous_previous, dtype=float)
    if (
        current_array.shape != previous_array.shape
        or current_array.shape != older_array.shape
    ):
        raise SVIDWValidationError("BDF2 time levels must have identical shapes")
    if not (
        np.all(np.isfinite(current_array))
        and np.all(np.isfinite(previous_array))
        and np.all(np.isfinite(older_array))
    ):
        raise SVIDWValidationError("BDF2 time levels must be finite")
    derivative = (3.0 * current_array - 4.0 * previous_array + older_array) / (2.0 * dt)
    if derivative.ndim == 0:
        return float(derivative)
    derivative = np.array(derivative, dtype=float, copy=True)
    derivative.setflags(write=False)
    return derivative


@dataclass(frozen=True)
class IBLIntervalEndpoint2D:
    """One positive-log-domain endpoint in Eqs. (7.111)--(7.112)."""

    arc_length_from_stagnation: float
    edge_density: float
    edge_tangential_velocity: float
    displacement_thickness: float
    momentum_thickness: float
    kinetic_energy_shape_factor: float
    density_shape_factor: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "arc_length_from_stagnation",
            "edge_density",
            "edge_tangential_velocity",
            "displacement_thickness",
            "momentum_thickness",
            "kinetic_energy_shape_factor",
        ):
            object.__setattr__(self, name, _positive_scalar(name, getattr(self, name)))
        object.__setattr__(
            self,
            "density_shape_factor",
            _nonnegative_scalar("density_shape_factor", self.density_shape_factor),
        )
        if self.displacement_thickness < self.momentum_thickness:
            raise SVIDWValidationError(
                "displacement_thickness must not be below momentum_thickness"
            )

    @property
    def shape_factor(self) -> float:
        return self.displacement_thickness / self.momentum_thickness


@dataclass(frozen=True)
class IBLIntervalTemporalDerivatives2D:
    """BDF2 derivatives of the four composites printed in Eq. (7.111/112).

    The caller owns the time-level storage.  Keeping these named composites
    explicit prevents the second Eq. (7.112) term from being silently
    changed from ``d(rho*delta*)/dt`` to density-thickness storage.
    """

    dt_rho_edge_tangential_velocity_displacement: float
    dt_rho_edge_tangential_velocity_squared_momentum: float
    dt_rho_displacement_thickness: float
    dt_edge_tangential_velocity: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite_scalar(name, getattr(self, name)))


@dataclass(frozen=True)
class IBLIntervalClosure2D:
    """Interval-centre source/closure values in Eq. (7.111/112)."""

    skin_friction_coefficient: float
    dissipation_coefficient: float
    angular_velocity: float = 0.0
    local_flow_acceleration: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "skin_friction_coefficient",
            _finite_scalar(
                "skin_friction_coefficient",
                self.skin_friction_coefficient,
            ),
        )
        object.__setattr__(
            self,
            "dissipation_coefficient",
            _nonnegative_scalar(
                "dissipation_coefficient", self.dissipation_coefficient
            ),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _finite_scalar("angular_velocity", self.angular_velocity),
        )
        object.__setattr__(
            self,
            "local_flow_acceleration",
            _finite_scalar("local_flow_acceleration", self.local_flow_acceleration),
        )


@dataclass(frozen=True)
class IBLIntervalResidualTerms2D:
    """Named dimensionless terms of thesis Eqs. (7.111)--(7.112)."""

    log_arc_increment: float
    momentum_unsteady: float
    momentum_log_theta: float
    momentum_log_edge_velocity: float
    momentum_log_density: float
    momentum_skin_friction: float
    momentum_residual: float
    energy_momentum_storage: float
    energy_displacement_storage: float
    energy_edge_acceleration: float
    energy_mass_deficit_storage: float
    energy_log_shape: float
    energy_log_edge_velocity: float
    energy_rotation_east: float
    energy_skin_friction: float
    energy_dissipation: float
    energy_local_acceleration: float
    energy_residual: float


def _log_increment(
    name: str,
    upstream: float,
    downstream: float,
) -> float:
    first = _positive_scalar(f"{name}.upstream", upstream)
    second = _positive_scalar(f"{name}.downstream", downstream)
    return float(np.log(second) - np.log(first))


def evaluate_riziotis_interval_residuals(
    upstream: IBLIntervalEndpoint2D,
    downstream: IBLIntervalEndpoint2D,
    temporal: IBLIntervalTemporalDerivatives2D,
    closure: IBLIntervalClosure2D,
) -> IBLIntervalResidualTerms2D:
    """Evaluate thesis Eqs. (7.111)--(7.112), left side equal to zero.

    Superscript ``m`` values are the arithmetic means of the two endpoint
    values, exactly as defined immediately after Eq. (7.112).  The rotation
    term contains the signed East approximation
    ``Theta_n=(theta+delta*) d(delta*)/ds`` already used in the source's
    logarithmic interval equation.
    """
    if not isinstance(upstream, IBLIntervalEndpoint2D):
        raise SVIDWValidationError("upstream must be IBLIntervalEndpoint2D")
    if not isinstance(downstream, IBLIntervalEndpoint2D):
        raise SVIDWValidationError("downstream must be IBLIntervalEndpoint2D")
    if not isinstance(temporal, IBLIntervalTemporalDerivatives2D):
        raise SVIDWValidationError("temporal must be IBLIntervalTemporalDerivatives2D")
    if not isinstance(closure, IBLIntervalClosure2D):
        raise SVIDWValidationError("closure must be IBLIntervalClosure2D")
    if downstream.arc_length_from_stagnation <= upstream.arc_length_from_stagnation:
        raise SVIDWValidationError("IBL interval arc length must increase downstream")

    arc = 0.5 * (
        upstream.arc_length_from_stagnation + downstream.arc_length_from_stagnation
    )
    rho = 0.5 * (upstream.edge_density + downstream.edge_density)
    velocity = 0.5 * (
        upstream.edge_tangential_velocity + downstream.edge_tangential_velocity
    )
    delta = 0.5 * (upstream.displacement_thickness + downstream.displacement_thickness)
    theta = 0.5 * (upstream.momentum_thickness + downstream.momentum_thickness)
    shape = 0.5 * (upstream.shape_factor + downstream.shape_factor)
    shape_star = 0.5 * (
        upstream.kinetic_energy_shape_factor + downstream.kinetic_energy_shape_factor
    )
    shape_starstar = 0.5 * (
        upstream.density_shape_factor + downstream.density_shape_factor
    )

    log_arc = _log_increment(
        "arc_length_from_stagnation",
        upstream.arc_length_from_stagnation,
        downstream.arc_length_from_stagnation,
    )
    log_theta = _log_increment(
        "momentum_thickness",
        upstream.momentum_thickness,
        downstream.momentum_thickness,
    )
    log_velocity = _log_increment(
        "edge_tangential_velocity",
        upstream.edge_tangential_velocity,
        downstream.edge_tangential_velocity,
    )
    log_density = _log_increment(
        "edge_density", upstream.edge_density, downstream.edge_density
    )
    log_shape_star = _log_increment(
        "kinetic_energy_shape_factor",
        upstream.kinetic_energy_shape_factor,
        downstream.kinetic_energy_shape_factor,
    )
    log_delta = _log_increment(
        "displacement_thickness",
        upstream.displacement_thickness,
        downstream.displacement_thickness,
    )

    momentum_unsteady = (
        arc
        / (rho * velocity**2 * theta)
        * log_arc
        * temporal.dt_rho_edge_tangential_velocity_displacement
    )
    momentum_velocity = (2.0 + shape) * log_velocity
    momentum_skin_friction = (
        -closure.skin_friction_coefficient * arc / (2.0 * theta) * log_arc
    )
    momentum_residual = (
        momentum_unsteady
        + log_theta
        + momentum_velocity
        + log_density
        + momentum_skin_friction
    )

    energy_momentum_storage = (
        arc
        / (rho * velocity**3 * shape_star * theta)
        * log_arc
        * temporal.dt_rho_edge_tangential_velocity_squared_momentum
    )
    energy_displacement_storage = (
        arc
        / (rho * velocity * shape_star * theta)
        * log_arc
        * temporal.dt_rho_displacement_thickness
    )
    energy_edge_acceleration = (
        2.0
        * shape_starstar
        * arc
        / (velocity**2 * shape_star)
        * log_arc
        * temporal.dt_edge_tangential_velocity
    )
    energy_mass_deficit_storage = (
        -arc
        / (rho * velocity**2 * theta)
        * log_arc
        * temporal.dt_rho_edge_tangential_velocity_displacement
    )
    energy_velocity = (2.0 * shape_starstar / shape_star + (1.0 - shape)) * log_velocity
    energy_rotation = (
        -4.0
        * closure.angular_velocity
        / velocity
        * (theta + delta)
        * shape
        / shape_star
        * log_delta
    )
    energy_skin_friction = (
        closure.skin_friction_coefficient * arc / (2.0 * theta) * log_arc
    )
    energy_dissipation = (
        -2.0 * closure.dissipation_coefficient * arc / (theta * shape_star) * log_arc
    )
    energy_local_acceleration = (
        -2.0
        * closure.local_flow_acceleration
        * shape
        * arc
        / (velocity**2 * shape_star)
        * log_arc
    )
    energy_residual = (
        energy_momentum_storage
        + energy_displacement_storage
        + energy_edge_acceleration
        + energy_mass_deficit_storage
        + log_shape_star
        + energy_velocity
        + energy_rotation
        + energy_skin_friction
        + energy_dissipation
        + energy_local_acceleration
    )

    return IBLIntervalResidualTerms2D(
        log_arc_increment=float(log_arc),
        momentum_unsteady=float(momentum_unsteady),
        momentum_log_theta=float(log_theta),
        momentum_log_edge_velocity=float(momentum_velocity),
        momentum_log_density=float(log_density),
        momentum_skin_friction=float(momentum_skin_friction),
        momentum_residual=float(momentum_residual),
        energy_momentum_storage=float(energy_momentum_storage),
        energy_displacement_storage=float(energy_displacement_storage),
        energy_edge_acceleration=float(energy_edge_acceleration),
        energy_mass_deficit_storage=float(energy_mass_deficit_storage),
        energy_log_shape=float(log_shape_star),
        energy_log_edge_velocity=float(energy_velocity),
        energy_rotation_east=float(energy_rotation),
        energy_skin_friction=float(energy_skin_friction),
        energy_dissipation=float(energy_dissipation),
        energy_local_acceleration=float(energy_local_acceleration),
        energy_residual=float(energy_residual),
    )
