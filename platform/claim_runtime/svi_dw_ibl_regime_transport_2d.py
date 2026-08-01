"""Source-form laminar/turbulent regime-transport oracles.

This module transcribes only Riziotis (2003), Eqs. (7.120)--(7.121), and
the transition construction described immediately below those equations.
It intentionally contains no spatial marcher, nonlinear solver, Newton
iteration, empirical retuning, or force-target input.

The transported turbulent state is ``sqrt(C_tau)``.  Consequently,
Eq. (7.121) retains both the published ``log(sqrt(C_tau))`` increment and
arithmetic interval-midpoint values.  The edge-velocity log increment remains
signed.  No absolute value, epsilon floor, or hidden clipping is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Final

from .svi_dw_types import SVIDWValidationError


RIZIOTIS_N_CRIT: Final[float] = 9.0
RIZIOTIS_CTAU_TRANSITION_SQRT_RATIO: Final[float] = 0.7


def _finite_scalar(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SVIDWValidationError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
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


def _finite_result(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise SVIDWValidationError(
            f"{name} is non-finite for the supplied source-form inputs"
        )
    return float(value)


def arithmetic_midpoint(upstream: Any, downstream: Any) -> float:
    """Return the arithmetic interval midpoint used by superscript ``m``."""
    first = _finite_scalar("upstream", upstream)
    second = _finite_scalar("downstream", downstream)
    return _finite_result("arithmetic_midpoint", 0.5 * (first + second))


@dataclass(frozen=True)
class LaminarENIntervalResidual2D:
    """Named terms of the laminar amplification residual, Eq. (7.120)."""

    amplification_gradient: float
    amplification_source: float
    residual: float


def evaluate_laminar_en_interval_residual(
    *,
    arc_length_upstream: float,
    arc_length_downstream: float,
    amplification_upstream: float,
    amplification_downstream: float,
    growth_per_momentum_reynolds: float,
    velocity_gradient_parameter: float,
    reynolds_length_factor: float,
    momentum_thickness: float,
) -> LaminarENIntervalResidual2D:
    """Evaluate Riziotis thesis Eq. (7.120), left side minus right side.

    ``velocity_gradient_parameter`` is the signed quantity ``m(H_k)`` from
    Eq. (7.118).  The four source values correspond to superscript ``ib`` in
    Eq. (7.120); they are deliberately *not* replaced by interval midpoints.
    """
    arc_up = _nonnegative_scalar("arc_length_upstream", arc_length_upstream)
    arc_down = _nonnegative_scalar(
        "arc_length_downstream",
        arc_length_downstream,
    )
    if arc_down <= arc_up:
        raise SVIDWValidationError(
            "laminar e^N interval arc length must increase downstream"
        )
    n_up = _finite_scalar("amplification_upstream", amplification_upstream)
    n_down = _finite_scalar(
        "amplification_downstream",
        amplification_downstream,
    )
    growth = _positive_scalar(
        "growth_per_momentum_reynolds",
        growth_per_momentum_reynolds,
    )
    velocity_gradient = _finite_scalar(
        "velocity_gradient_parameter",
        velocity_gradient_parameter,
    )
    length_factor = _positive_scalar(
        "reynolds_length_factor",
        reynolds_length_factor,
    )
    theta = _positive_scalar("momentum_thickness", momentum_thickness)

    amplification_gradient = (n_down - n_up) / (arc_down - arc_up)
    amplification_source = (
        growth
        * 0.5
        * (velocity_gradient + 1.0)
        * length_factor
        / theta
    )
    residual = amplification_gradient - amplification_source
    return LaminarENIntervalResidual2D(
        amplification_gradient=_finite_result(
            "amplification_gradient",
            amplification_gradient,
        ),
        amplification_source=_finite_result(
            "amplification_source",
            amplification_source,
        ),
        residual=_finite_result("laminar_en_interval_residual", residual),
    )


@dataclass(frozen=True)
class TurbulentShearEndpoint2D:
    """One endpoint of the turbulent ``sqrt(C_tau)`` interval equation."""

    arc_length_from_stagnation: float
    edge_tangential_velocity: float
    boundary_layer_thickness: float
    displacement_thickness: float
    kinematic_shape_factor: float
    skin_friction_coefficient: float
    sqrt_shear_coefficient: float
    equilibrium_sqrt_shear_coefficient: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arc_length_from_stagnation",
            _nonnegative_scalar(
                "arc_length_from_stagnation",
                self.arc_length_from_stagnation,
            ),
        )
        for name in (
            "edge_tangential_velocity",
            "boundary_layer_thickness",
            "displacement_thickness",
            "sqrt_shear_coefficient",
        ):
            object.__setattr__(
                self,
                name,
                _positive_scalar(name, getattr(self, name)),
            )
        if self.boundary_layer_thickness <= self.displacement_thickness:
            raise SVIDWValidationError(
                "boundary_layer_thickness must exceed "
                "displacement_thickness"
            )
        shape = _finite_scalar(
            "kinematic_shape_factor",
            self.kinematic_shape_factor,
        )
        if shape <= 1.0:
            raise SVIDWValidationError(
                "kinematic_shape_factor must be greater than one"
            )
        object.__setattr__(self, "kinematic_shape_factor", shape)
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
            "equilibrium_sqrt_shear_coefficient",
            _nonnegative_scalar(
                "equilibrium_sqrt_shear_coefficient",
                self.equilibrium_sqrt_shear_coefficient,
            ),
        )


@dataclass(frozen=True)
class TurbulentShearIntervalResidual2D:
    """Named terms of the turbulent shear residual, Eq. (7.121)."""

    interval_length: float
    boundary_layer_thickness_midpoint: float
    displacement_thickness_midpoint: float
    sqrt_shear_midpoint: float
    equilibrium_sqrt_shear_midpoint: float
    kinematic_shape_midpoint: float
    skin_friction_midpoint: float
    log_sqrt_shear_increment: float
    log_edge_velocity_increment: float
    log_shear_transport: float
    equilibrium_relaxation: float
    wall_shear_drive: float
    signed_edge_velocity_transport: float
    residual: float


def evaluate_turbulent_shear_interval_residual(
    upstream: TurbulentShearEndpoint2D,
    downstream: TurbulentShearEndpoint2D,
) -> TurbulentShearIntervalResidual2D:
    """Evaluate Riziotis thesis Eq. (7.121), left side minus right side.

    Every superscript-``m`` quantity is the arithmetic mean of its two
    endpoint values.  In particular, ``H_k`` is averaged before evaluating
    ``((H_k - 1)/(6.7 H_k))**2``.  The final source term uses the signed
    increment ``log(U_e_down)-log(U_e_up)`` exactly as printed.
    """
    if not isinstance(upstream, TurbulentShearEndpoint2D):
        raise SVIDWValidationError(
            "upstream must be TurbulentShearEndpoint2D"
        )
    if not isinstance(downstream, TurbulentShearEndpoint2D):
        raise SVIDWValidationError(
            "downstream must be TurbulentShearEndpoint2D"
        )

    interval_length = (
        downstream.arc_length_from_stagnation
        - upstream.arc_length_from_stagnation
    )
    if interval_length <= 0.0:
        raise SVIDWValidationError(
            "turbulent shear interval arc length must increase downstream"
        )

    delta = arithmetic_midpoint(
        upstream.boundary_layer_thickness,
        downstream.boundary_layer_thickness,
    )
    delta_star = arithmetic_midpoint(
        upstream.displacement_thickness,
        downstream.displacement_thickness,
    )
    sqrt_shear = arithmetic_midpoint(
        upstream.sqrt_shear_coefficient,
        downstream.sqrt_shear_coefficient,
    )
    equilibrium_sqrt_shear = arithmetic_midpoint(
        upstream.equilibrium_sqrt_shear_coefficient,
        downstream.equilibrium_sqrt_shear_coefficient,
    )
    h_k = arithmetic_midpoint(
        upstream.kinematic_shape_factor,
        downstream.kinematic_shape_factor,
    )
    skin_friction = arithmetic_midpoint(
        upstream.skin_friction_coefficient,
        downstream.skin_friction_coefficient,
    )

    log_sqrt_shear = (
        math.log(downstream.sqrt_shear_coefficient)
        - math.log(upstream.sqrt_shear_coefficient)
    )
    log_edge_velocity = (
        math.log(downstream.edge_tangential_velocity)
        - math.log(upstream.edge_tangential_velocity)
    )
    log_shear_transport = 2.0 * delta * log_sqrt_shear
    equilibrium_relaxation = (
        5.6
        * (equilibrium_sqrt_shear - sqrt_shear)
        * interval_length
    )
    stress_defect = (
        0.5 * skin_friction
        - ((h_k - 1.0) / (6.7 * h_k)) ** 2
    )
    wall_shear_drive = (
        2.0
        * delta
        * 4.0
        / (3.0 * delta_star)
        * stress_defect
        * interval_length
    )
    signed_edge_velocity_transport = -2.0 * delta * log_edge_velocity
    residual = (
        log_shear_transport
        - equilibrium_relaxation
        - wall_shear_drive
        - signed_edge_velocity_transport
    )

    values = {
        "interval_length": interval_length,
        "boundary_layer_thickness_midpoint": delta,
        "displacement_thickness_midpoint": delta_star,
        "sqrt_shear_midpoint": sqrt_shear,
        "equilibrium_sqrt_shear_midpoint": equilibrium_sqrt_shear,
        "kinematic_shape_midpoint": h_k,
        "skin_friction_midpoint": skin_friction,
        "log_sqrt_shear_increment": log_sqrt_shear,
        "log_edge_velocity_increment": log_edge_velocity,
        "log_shear_transport": log_shear_transport,
        "equilibrium_relaxation": equilibrium_relaxation,
        "wall_shear_drive": wall_shear_drive,
        "signed_edge_velocity_transport": signed_edge_velocity_transport,
        "residual": residual,
    }
    return TurbulentShearIntervalResidual2D(
        **{
            name: _finite_result(name, value)
            for name, value in values.items()
        }
    )


def n9_transition_event(amplification: float) -> float:
    """Return the signed transition event ``n_tilde - 9``."""
    value = _finite_scalar("amplification", amplification)
    return _finite_result(
        "n9_transition_event",
        value - RIZIOTIS_N_CRIT,
    )


def n9_transition_fraction(
    amplification_upstream: float,
    amplification_downstream: float,
) -> float:
    """Return the unique downstream interval fraction where ``n_tilde=9``.

    The source march reaches transition from below, so the function accepts
    only a non-decreasing bracket with the upstream event non-positive and
    the downstream event non-negative.  An interval identically equal to
    nine has no unique transition point and fails closed.
    """
    event_up = n9_transition_event(amplification_upstream)
    event_down = n9_transition_event(amplification_downstream)
    if event_up > 0.0 or event_down < 0.0:
        raise SVIDWValidationError(
            "n=9 transition must be bracketed from below downstream"
        )
    denominator = event_down - event_up
    if denominator <= 0.0:
        raise SVIDWValidationError(
            "n=9 transition bracket must be unique and non-decreasing"
        )
    fraction = -event_up / denominator
    if fraction < 0.0 or fraction > 1.0:
        raise SVIDWValidationError(
            "n=9 transition fraction lies outside the interval"
        )
    return _finite_result("n9_transition_fraction", fraction)


@dataclass(frozen=True)
class TransitionInterpolationEndpoint2D:
    """Endpoint state used by the source's transition-interval interpolation."""

    arc_length_from_stagnation: float
    amplification: float
    displacement_thickness: float
    momentum_thickness: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arc_length_from_stagnation",
            _nonnegative_scalar(
                "arc_length_from_stagnation",
                self.arc_length_from_stagnation,
            ),
        )
        object.__setattr__(
            self,
            "amplification",
            _finite_scalar("amplification", self.amplification),
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
            _positive_scalar(
                "momentum_thickness",
                self.momentum_thickness,
            ),
        )
        if self.displacement_thickness < self.momentum_thickness:
            raise SVIDWValidationError(
                "displacement_thickness must not be below momentum_thickness"
            )


@dataclass(frozen=True)
class TransitionState2D:
    """Linearly interpolated ``n=9`` location and thickness state."""

    interval_fraction: float
    arc_length_from_stagnation: float
    amplification: float
    displacement_thickness: float
    momentum_thickness: float


def interpolate_n9_transition_state(
    upstream: TransitionInterpolationEndpoint2D,
    downstream: TransitionInterpolationEndpoint2D,
) -> TransitionState2D:
    """Linearly interpolate ``s``, ``delta*``, and ``theta`` at ``n=9``."""
    if not isinstance(upstream, TransitionInterpolationEndpoint2D):
        raise SVIDWValidationError(
            "upstream must be TransitionInterpolationEndpoint2D"
        )
    if not isinstance(downstream, TransitionInterpolationEndpoint2D):
        raise SVIDWValidationError(
            "downstream must be TransitionInterpolationEndpoint2D"
        )
    if (
        downstream.arc_length_from_stagnation
        <= upstream.arc_length_from_stagnation
    ):
        raise SVIDWValidationError(
            "transition interpolation arc length must increase downstream"
        )

    fraction = n9_transition_fraction(
        upstream.amplification,
        downstream.amplification,
    )

    def interpolate(first: float, second: float) -> float:
        return _finite_result(
            "linearly_interpolated_transition_state",
            first + fraction * (second - first),
        )

    return TransitionState2D(
        interval_fraction=fraction,
        arc_length_from_stagnation=interpolate(
            upstream.arc_length_from_stagnation,
            downstream.arc_length_from_stagnation,
        ),
        amplification=interpolate(
            upstream.amplification,
            downstream.amplification,
        ),
        displacement_thickness=interpolate(
            upstream.displacement_thickness,
            downstream.displacement_thickness,
        ),
        momentum_thickness=interpolate(
            upstream.momentum_thickness,
            downstream.momentum_thickness,
        ),
    )


def initialize_transition_sqrt_shear_coefficient(
    equilibrium_sqrt_shear_coefficient: float,
) -> float:
    """Return ``sqrt(C_tau,tr)=0.7 sqrt(C_tau,eq)`` at transition."""
    equilibrium = _positive_scalar(
        "equilibrium_sqrt_shear_coefficient",
        equilibrium_sqrt_shear_coefficient,
    )
    return _finite_result(
        "transition_sqrt_shear_coefficient",
        RIZIOTIS_CTAU_TRANSITION_SQRT_RATIO * equilibrium,
    )


__all__ = [
    "LaminarENIntervalResidual2D",
    "RIZIOTIS_CTAU_TRANSITION_SQRT_RATIO",
    "RIZIOTIS_N_CRIT",
    "TransitionInterpolationEndpoint2D",
    "TransitionState2D",
    "TurbulentShearEndpoint2D",
    "TurbulentShearIntervalResidual2D",
    "arithmetic_midpoint",
    "evaluate_laminar_en_interval_residual",
    "evaluate_turbulent_shear_interval_residual",
    "initialize_transition_sqrt_shear_coefficient",
    "interpolate_n9_transition_state",
    "n9_transition_event",
    "n9_transition_fraction",
]
