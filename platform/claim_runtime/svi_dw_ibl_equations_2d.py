"""Term-by-term oracle for Riziotis--Voutsinas IBL Eqs. (9)--(10).

This module implements only the published differential residuals at one
attached boundary-layer station.  It deliberately does not choose:

* a laminar, transition, turbulent, or dissipation closure;
* a time-integration rule or spatial finite-difference stencil;
* a stagnation initialization or separation criterion; or
* any target-force calibration.

The caller supplies already-discretized time and streamwise derivatives.
Keeping those derivatives explicit makes the source equation auditable before
the later simultaneous viscous--inviscid Newton system is assembled.
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


@dataclass(frozen=True)
class IBLStationState2D:
    """Physical state needed by the two published integral equations.

    ``edge_tangential_velocity`` is the positive
    stagnation-to-downstream tangential component ``w_e_tau`` used by the IBL
    march, not the signed clockwise body-contour trace.  ``edge_speed`` is the
    distinct relative-speed magnitude ``w_e`` printed in the local-acceleration
    term of Eq. (10).  They coincide only in the zero-transpiration limit and
    therefore must not share one field in a strong-interaction implementation.

    The three shape factors are derived from thickness inventories so
    inconsistent duplicated values cannot enter the residual.
    """

    edge_density: float
    edge_tangential_velocity: float
    edge_speed: float
    displacement_thickness: float
    momentum_thickness: float
    kinetic_energy_thickness: float
    density_thickness: float = 0.0
    normal_momentum_transport: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "edge_density",
            _positive_scalar("edge_density", self.edge_density),
        )
        object.__setattr__(
            self,
            "edge_tangential_velocity",
            _positive_scalar(
                "edge_tangential_velocity",
                self.edge_tangential_velocity,
            ),
        )
        object.__setattr__(
            self,
            "edge_speed",
            _positive_scalar("edge_speed", self.edge_speed),
        )
        object.__setattr__(
            self,
            "displacement_thickness",
            _positive_scalar(
                "displacement_thickness",
                self.displacement_thickness,
            ),
        )
        object.__setattr__(
            self,
            "momentum_thickness",
            _positive_scalar("momentum_thickness", self.momentum_thickness),
        )
        object.__setattr__(
            self,
            "kinetic_energy_thickness",
            _positive_scalar(
                "kinetic_energy_thickness",
                self.kinetic_energy_thickness,
            ),
        )
        object.__setattr__(
            self,
            "density_thickness",
            _nonnegative_scalar("density_thickness", self.density_thickness),
        )
        object.__setattr__(
            self,
            "normal_momentum_transport",
            _finite_scalar(
                "normal_momentum_transport",
                self.normal_momentum_transport,
            ),
        )
        if self.displacement_thickness < self.momentum_thickness:
            raise SVIDWValidationError(
                "displacement_thickness must not be below momentum_thickness"
            )
        if self.kinetic_energy_thickness < self.momentum_thickness:
            raise SVIDWValidationError(
                "kinetic_energy_thickness must not be below "
                "momentum_thickness"
            )

    @property
    def shape_factor(self) -> float:
        return self.displacement_thickness / self.momentum_thickness

    @property
    def kinetic_energy_shape_factor(self) -> float:
        return self.kinetic_energy_thickness / self.momentum_thickness

    @property
    def density_shape_factor(self) -> float:
        return self.density_thickness / self.momentum_thickness


@dataclass(frozen=True)
class IBLStationDerivatives2D:
    """Discrete derivative values; the numerical stencil is caller-owned."""

    dt_rho_edge_tangential_velocity_displacement: float
    ds_momentum_thickness: float
    ds_edge_tangential_velocity: float
    ds_edge_density: float
    dt_rho_edge_tangential_velocity_squared_momentum: float
    dt_rho_displacement_thickness: float
    dt_edge_tangential_velocity: float
    ds_kinetic_energy_shape_factor: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _finite_scalar(name, getattr(self, name)),
            )


@dataclass(frozen=True)
class IBLClosureTerms2D:
    """Named closure/source terms on the right sides of Eqs. (9)--(10)."""

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
                "dissipation_coefficient",
                self.dissipation_coefficient,
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
            _finite_scalar(
                "local_flow_acceleration",
                self.local_flow_acceleration,
            ),
        )


@dataclass(frozen=True)
class IBLResidualTerms2D:
    """Every signed term and the final left-minus-right residual."""

    momentum_unsteady_storage: float
    momentum_thickness_gradient: float
    momentum_edge_tangential_velocity_gradient: float
    momentum_density_gradient: float
    momentum_skin_friction_rhs: float
    momentum_residual: float
    energy_momentum_storage: float
    energy_displacement_storage: float
    energy_edge_acceleration_storage: float
    energy_mass_deficit_storage: float
    energy_rotation_transport: float
    energy_shape_gradient: float
    energy_edge_tangential_velocity_gradient: float
    energy_dissipation_rhs: float
    energy_local_acceleration_rhs: float
    energy_skin_friction_rhs: float
    energy_residual: float


def evaluate_riziotis_ibl_residuals(
    state: IBLStationState2D,
    derivatives: IBLStationDerivatives2D,
    closure: IBLClosureTerms2D,
) -> IBLResidualTerms2D:
    """Evaluate Riziotis--Voutsinas (2008) Eqs. (9) and (10).

    Residual sign is always ``published left side - published right side``.
    In particular, the local-acceleration term in Eq. (10) multiplies
    ``displacement_thickness``.  It must not be replaced by the distinct
    density thickness.
    """
    if not isinstance(state, IBLStationState2D):
        raise SVIDWValidationError("state must be IBLStationState2D")
    if not isinstance(derivatives, IBLStationDerivatives2D):
        raise SVIDWValidationError(
            "derivatives must be IBLStationDerivatives2D"
        )
    if not isinstance(closure, IBLClosureTerms2D):
        raise SVIDWValidationError("closure must be IBLClosureTerms2D")

    rho = state.edge_density
    tangential_velocity = state.edge_tangential_velocity
    edge_speed = state.edge_speed
    delta_star = state.displacement_thickness
    theta = state.momentum_thickness
    shape = state.shape_factor
    shape_star = state.kinetic_energy_shape_factor
    shape_starstar = state.density_shape_factor

    momentum_unsteady = (
        derivatives.dt_rho_edge_tangential_velocity_displacement
        / (rho * tangential_velocity**2)
    )
    momentum_theta_gradient = derivatives.ds_momentum_thickness
    momentum_velocity_gradient = (
        (2.0 + shape)
        * theta
        / tangential_velocity
        * derivatives.ds_edge_tangential_velocity
    )
    momentum_density_gradient = (
        theta / rho * derivatives.ds_edge_density
    )
    momentum_rhs = 0.5 * closure.skin_friction_coefficient
    momentum_residual = (
        momentum_unsteady
        + momentum_theta_gradient
        + momentum_velocity_gradient
        + momentum_density_gradient
        - momentum_rhs
    )

    energy_momentum_storage = (
        derivatives.dt_rho_edge_tangential_velocity_squared_momentum
        / (rho * tangential_velocity**3)
    )
    energy_displacement_storage = (
        derivatives.dt_rho_displacement_thickness
        / (rho * tangential_velocity)
    )
    energy_edge_acceleration = (
        2.0
        * shape_starstar
        * theta
        / tangential_velocity**2
        * derivatives.dt_edge_tangential_velocity
    )
    energy_mass_deficit_storage = (
        -shape_star
        / (rho * tangential_velocity**2)
        * derivatives.dt_rho_edge_tangential_velocity_displacement
    )
    energy_rotation = (
        -4.0
        * closure.angular_velocity
        / tangential_velocity
        * state.normal_momentum_transport
    )
    energy_shape_gradient = (
        theta * derivatives.ds_kinetic_energy_shape_factor
    )
    energy_velocity_gradient = (
        (
            2.0 * shape_starstar
            + shape_star * (1.0 - shape)
        )
        * theta
        / tangential_velocity
        * derivatives.ds_edge_tangential_velocity
    )
    energy_dissipation_rhs = 2.0 * closure.dissipation_coefficient
    energy_local_acceleration_rhs = (
        2.0
        * closure.local_flow_acceleration
        / edge_speed**2
        * delta_star
    )
    energy_skin_friction_rhs = (
        -0.5 * shape_star * closure.skin_friction_coefficient
    )
    energy_residual = (
        energy_momentum_storage
        + energy_displacement_storage
        + energy_edge_acceleration
        + energy_mass_deficit_storage
        + energy_rotation
        + energy_shape_gradient
        + energy_velocity_gradient
        - energy_dissipation_rhs
        - energy_local_acceleration_rhs
        - energy_skin_friction_rhs
    )

    return IBLResidualTerms2D(
        momentum_unsteady_storage=float(momentum_unsteady),
        momentum_thickness_gradient=float(momentum_theta_gradient),
        momentum_edge_tangential_velocity_gradient=float(
            momentum_velocity_gradient
        ),
        momentum_density_gradient=float(momentum_density_gradient),
        momentum_skin_friction_rhs=float(momentum_rhs),
        momentum_residual=float(momentum_residual),
        energy_momentum_storage=float(energy_momentum_storage),
        energy_displacement_storage=float(energy_displacement_storage),
        energy_edge_acceleration_storage=float(energy_edge_acceleration),
        energy_mass_deficit_storage=float(energy_mass_deficit_storage),
        energy_rotation_transport=float(energy_rotation),
        energy_shape_gradient=float(energy_shape_gradient),
        energy_edge_tangential_velocity_gradient=float(
            energy_velocity_gradient
        ),
        energy_dissipation_rhs=float(energy_dissipation_rhs),
        energy_local_acceleration_rhs=float(
            energy_local_acceleration_rhs
        ),
        energy_skin_friction_rhs=float(energy_skin_friction_rhs),
        energy_residual=float(energy_residual),
    )
