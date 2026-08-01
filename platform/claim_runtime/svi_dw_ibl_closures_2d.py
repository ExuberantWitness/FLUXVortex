"""Source-form closure oracle for the Riziotis (2003) two-dimensional IBL.

The functions in this module transcribe Eqs. (7.33)--(7.35),
(7.41)--(7.45), (7.48), and (7.54)--(7.62) of Riziotis' 2003 thesis.
They are scalar, stateless, and contain no response data, fitted parameter,
finite-difference stencil, Newton iteration, or target-force correction.

The one source correction is ``H0 = 3 + 400 / Re_theta`` at and above
``Re_theta = 400``.  The thesis prints ``4 / Re_theta``; continuity at the
branch point, the XFOIL 6.99 developer source, and Agrawal et al. (2024)
identify the missing two zeroes.  No other XFOIL closure, clamp, or transition
rule is used here.

Every singular or non-real input domain fails closed.  In particular, this
module does not add an epsilon floor at stagnation, an XFOIL ``RTZ`` floor, a
``0.995 - Us`` limiter, or a positivity clip to any published expression.
"""
from __future__ import annotations

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


def _shape_above_one(name: str, value: Any) -> float:
    result = _finite_scalar(name, value)
    if result <= 1.0:
        raise SVIDWValidationError(f"{name} must be greater than one")
    return result


def _finite_result(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise SVIDWValidationError(
            f"{name} is non-finite for the supplied source-form inputs"
        )
    return value


def kinematic_shape_factor(
    shape_factor: float,
    edge_mach: float,
) -> float:
    """Return ``H_k`` from thesis Eq. (7.55).

    ``edge_mach`` is the non-negative local Mach-number magnitude.  This
    transformation alone only requires a positive resulting ``H_k``; the
    laminar and turbulent closure functions impose their stricter ``H_k > 1``
    physical domains.
    """
    shape = _positive_scalar("shape_factor", shape_factor)
    mach = _nonnegative_scalar("edge_mach", edge_mach)
    mach_squared = mach * mach
    result = (shape - 0.290 * mach_squared) / (
        1.0 + 0.113 * mach_squared
    )
    result = _finite_result("kinematic_shape_factor", result)
    if result <= 0.0:
        raise SVIDWValidationError(
            "kinematic_shape_factor must be positive"
        )
    return result


def density_shape_factor(
    kinematic_shape: float,
    edge_mach: float,
) -> float:
    """Return ``H**`` from thesis Eq. (7.56)."""
    h_k = _finite_scalar("kinematic_shape", kinematic_shape)
    if h_k <= 0.8:
        raise SVIDWValidationError(
            "kinematic_shape must be greater than 0.8 for Eq. (7.56)"
        )
    mach = _nonnegative_scalar("edge_mach", edge_mach)
    result = (0.064 / (h_k - 0.8) + 0.251) * mach * mach
    result = _finite_result("density_shape_factor", result)
    if result < 0.0:
        raise SVIDWValidationError("density_shape_factor became negative")
    return result


def laminar_energy_shape_factor(kinematic_shape: float) -> float:
    """Return laminar ``H*`` from thesis Eq. (7.57)."""
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    if h_k <= 4.0:
        result = 1.515 + 0.076 * (4.0 - h_k) ** 2 / h_k
    else:
        result = 1.515 + 0.040 * (h_k - 4.0) ** 2 / h_k
    return _positive_scalar("laminar_energy_shape_factor", result)


def laminar_skin_friction_coefficient(
    kinematic_shape: float,
    momentum_reynolds: float,
) -> float:
    """Return laminar ``C_f`` from thesis Eq. (7.58)."""
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    reynolds = _positive_scalar("momentum_reynolds", momentum_reynolds)
    if h_k <= 7.4:
        scaled_half_cf = (
            -0.067
            + 0.01977 * (7.4 - h_k) ** 2 / (h_k - 1.0)
        )
    else:
        scaled_half_cf = (
            -0.067
            + 0.022 * (1.0 - 1.4 / (h_k - 6.0)) ** 2
        )
    return _finite_result(
        "laminar_skin_friction_coefficient",
        2.0 * scaled_half_cf / reynolds,
    )


def laminar_dissipation_ratio(
    kinematic_shape: float,
    momentum_reynolds: float,
) -> float:
    """Return ``2 C_D / H*`` from thesis Eq. (7.59).

    The separated branch deliberately retains the thesis coefficient
    ``-0.003``.  The different coefficient in later XFOIL sources is outside
    this source oracle.
    """
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    reynolds = _positive_scalar("momentum_reynolds", momentum_reynolds)
    if h_k <= 4.0:
        scaled_ratio = 0.207 + 0.00205 * (4.0 - h_k) ** 5.5
    else:
        difference_squared = (h_k - 4.0) ** 2
        scaled_ratio = (
            0.207
            - 0.003
            * difference_squared
            / (1.0 + 0.02 * difference_squared)
        )
    result = _finite_result(
        "laminar_dissipation_ratio",
        scaled_ratio / reynolds,
    )
    if result < 0.0:
        raise SVIDWValidationError(
            "laminar_dissipation_ratio became negative"
        )
    return result


def laminar_dissipation_coefficient(
    kinematic_shape: float,
    momentum_reynolds: float,
) -> float:
    """Return laminar ``C_D`` by combining thesis Eqs. (7.57), (7.59)."""
    energy_shape = laminar_energy_shape_factor(kinematic_shape)
    ratio = laminar_dissipation_ratio(
        kinematic_shape,
        momentum_reynolds,
    )
    return _positive_scalar(
        "laminar_dissipation_coefficient",
        0.5 * energy_shape * ratio,
    )


def turbulent_skin_friction_coefficient(
    kinematic_shape: float,
    momentum_reynolds: float,
    edge_mach: float,
) -> float:
    """Return turbulent ``C_f`` from thesis Eq. (7.61)."""
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    reynolds = _positive_scalar("momentum_reynolds", momentum_reynolds)
    mach = _nonnegative_scalar("edge_mach", edge_mach)
    compressibility_factor = math.sqrt(1.0 + 0.2 * mach * mach)
    log_base = math.log10(reynolds / compressibility_factor)
    if not math.isfinite(log_base) or log_base <= 0.0:
        raise SVIDWValidationError(
            "Eq. (7.61) requires log10(momentum_reynolds / Fc) > 0"
        )
    numerator = (
        0.3
        * math.exp(-1.33 * h_k)
        * log_base ** (-1.74 - 0.31 * h_k)
        + 0.00011 * (math.tanh(4.0 - h_k / 0.875) - 1.0)
    )
    return _finite_result(
        "turbulent_skin_friction_coefficient",
        numerator / compressibility_factor,
    )


def turbulent_h0(momentum_reynolds: float) -> float:
    """Return the uniquely corrected branch parameter ``H0``.

    The two formulas agree at ``Re_theta = 400``; the corrected upper branch
    owns that equality.  This is the sole place where the XFOIL developer
    source is used to resolve the thesis' ``4`` versus ``400`` print conflict.
    """
    reynolds = _positive_scalar("momentum_reynolds", momentum_reynolds)
    if reynolds < 400.0:
        return 4.0
    return _finite_result("turbulent_h0", 3.0 + 400.0 / reynolds)


def turbulent_energy_shape_factor(
    kinematic_shape: float,
    momentum_reynolds: float,
) -> float:
    """Return turbulent ``H*`` from the old thesis Eq. (7.62).

    This is intentionally not the newer XFOIL ``HST`` closure and contains no
    XFOIL Reynolds-number clamp.
    """
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    reynolds = _positive_scalar("momentum_reynolds", momentum_reynolds)
    h_0 = turbulent_h0(reynolds)
    baseline = 1.505 + 4.0 / reynolds
    if h_k < h_0:
        result = (
            baseline
            + (0.165 - 1.6 / math.sqrt(reynolds))
            * (h_0 - h_k) ** 1.6
            / h_k
        )
    else:
        logarithm = math.log(reynolds)
        if logarithm == 0.0:
            raise SVIDWValidationError(
                "Eq. (7.62) separated branch requires ln(Re_theta) != 0"
            )
        compound_denominator = h_k - h_0 + 4.0 / logarithm
        if compound_denominator == 0.0:
            raise SVIDWValidationError(
                "Eq. (7.62) separated-branch denominator is singular"
            )
        result = (
            baseline
            + (h_k - h_0) ** 2
            * (
                0.04 / h_k
                + 0.007
                * logarithm
                / compound_denominator**2
            )
        )
    return _positive_scalar("turbulent_energy_shape_factor", result)


def turbulent_slip_velocity_ratio(
    energy_shape: float,
    kinematic_shape: float,
    shape_factor: float,
) -> float:
    """Return the equivalent wall-slip ratio ``U_s`` from Eq. (7.42)."""
    h_star = _positive_scalar("energy_shape", energy_shape)
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    shape = _positive_scalar("shape_factor", shape_factor)
    result = 0.5 * h_star * (
        1.0 - (4.0 / 3.0) * (h_k - 1.0) / shape
    )
    return _finite_result("turbulent_slip_velocity_ratio", result)


def turbulent_nominal_thickness(
    momentum_thickness: float,
    displacement_thickness: float,
    kinematic_shape: float,
) -> float:
    """Return nominal boundary-layer thickness ``delta`` from Eq. (7.44)."""
    theta = _positive_scalar("momentum_thickness", momentum_thickness)
    delta_star = _positive_scalar(
        "displacement_thickness",
        displacement_thickness,
    )
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    result = theta * (3.15 + 1.72 / (h_k - 1.0)) + delta_star
    return _positive_scalar("turbulent_nominal_thickness", result)


def turbulent_equilibrium_shear_coefficient(
    energy_shape: float,
    slip_velocity_ratio: float,
    kinematic_shape: float,
    shape_factor: float,
) -> float:
    """Return equilibrium ``C_tau`` from thesis Eq. (7.45)."""
    h_star = _positive_scalar("energy_shape", energy_shape)
    slip = _finite_scalar("slip_velocity_ratio", slip_velocity_ratio)
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    shape = _positive_scalar("shape_factor", shape_factor)
    if 1.0 - slip <= 0.0:
        raise SVIDWValidationError(
            "Eq. (7.45) requires 1 - slip_velocity_ratio > 0"
        )
    result = (
        h_star
        * 0.015
        / (1.0 - slip)
        * (h_k - 1.0) ** 3
        / (h_k * h_k * shape)
    )
    return _positive_scalar(
        "turbulent_equilibrium_shear_coefficient",
        result,
    )


def turbulent_dissipation_coefficient(
    skin_friction_coefficient: float,
    slip_velocity_ratio: float,
    shear_stress_coefficient: float,
) -> float:
    """Return turbulent ``C_D`` from thesis Eq. (7.41)."""
    skin_friction = _finite_scalar(
        "skin_friction_coefficient",
        skin_friction_coefficient,
    )
    slip = _finite_scalar("slip_velocity_ratio", slip_velocity_ratio)
    shear = _nonnegative_scalar(
        "shear_stress_coefficient",
        shear_stress_coefficient,
    )
    result = (
        0.5 * skin_friction * slip
        + shear * (1.0 - slip)
    )
    result = _finite_result("turbulent_dissipation_coefficient", result)
    if result < 0.0:
        raise SVIDWValidationError(
            "turbulent_dissipation_coefficient became negative"
        )
    return result


def east_normal_momentum_thickness(
    momentum_thickness: float,
    displacement_thickness: float,
    displacement_thickness_gradient: float,
) -> float:
    """Return signed East ``Theta_n`` from thesis Eq. (7.48).

    The sign of ``d(delta*)/ds`` is preserved exactly.  No absolute value or
    positivity limiter is part of the source closure.
    """
    theta = _positive_scalar("momentum_thickness", momentum_thickness)
    delta_star = _positive_scalar(
        "displacement_thickness",
        displacement_thickness,
    )
    gradient = _finite_scalar(
        "displacement_thickness_gradient",
        displacement_thickness_gradient,
    )
    return _finite_result(
        "east_normal_momentum_thickness",
        (theta + delta_star) * gradient,
    )


def body_tangential_acceleration(
    angular_velocity: float,
    angular_acceleration: float,
    point_from_origin_tangent: float,
    point_from_origin_normal: float,
    origin_tangential_acceleration: float,
) -> float:
    """Return moving-frame ``a`` from the thesis momentum equations.

    ``a = Omega**2 R_OP,s + dOmega/dt R_OP,n - d2 R_O,s/dt2``.
    All signs are coordinate signs; no magnitude or response-dependent
    convention is introduced.
    """
    omega = _finite_scalar("angular_velocity", angular_velocity)
    omega_dot = _finite_scalar(
        "angular_acceleration",
        angular_acceleration,
    )
    radius_tangent = _finite_scalar(
        "point_from_origin_tangent",
        point_from_origin_tangent,
    )
    radius_normal = _finite_scalar(
        "point_from_origin_normal",
        point_from_origin_normal,
    )
    origin_acceleration = _finite_scalar(
        "origin_tangential_acceleration",
        origin_tangential_acceleration,
    )
    return _finite_result(
        "body_tangential_acceleration",
        omega * omega * radius_tangent
        + omega_dot * radius_normal
        - origin_acceleration,
    )


def en_growth_per_momentum_reynolds(kinematic_shape: float) -> float:
    """Return ``d(n_tilde)/d(Re_theta)`` from thesis Eq. (7.34)."""
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    inner = (
        2.4 * h_k
        - 3.7
        + 2.5 * math.tanh(1.5 * h_k - 4.65)
    )
    result = 0.01 * math.sqrt(inner * inner + 0.25)
    return _positive_scalar("en_growth_per_momentum_reynolds", result)


def en_onset_momentum_reynolds(kinematic_shape: float) -> float:
    """Return ``Re_theta0`` from thesis Eq. (7.35)."""
    h_k = _shape_above_one("kinematic_shape", kinematic_shape)
    inverse_offset = 1.0 / (h_k - 1.0)
    exponent = (
        (1.415 * inverse_offset - 0.489)
        * math.tanh(20.0 * inverse_offset - 12.9)
        + 3.295 * inverse_offset
        + 0.44
    )
    try:
        result = 10.0**exponent
    except OverflowError as exc:
        raise SVIDWValidationError(
            "en_onset_momentum_reynolds overflowed near H_k = 1"
        ) from exc
    return _positive_scalar("en_onset_momentum_reynolds", result)


def en_similar_flow_amplification(
    kinematic_shape: float,
    momentum_reynolds: float,
) -> float:
    """Return similar-flow ``n_tilde`` from thesis Eq. (7.33).

    The algebraic value is intentionally not clipped to zero below
    ``Re_theta0``.  The later transition march owns instability onset and
    integration; this function is only the published scalar relation.
    """
    reynolds = _positive_scalar("momentum_reynolds", momentum_reynolds)
    result = en_growth_per_momentum_reynolds(kinematic_shape) * (
        reynolds - en_onset_momentum_reynolds(kinematic_shape)
    )
    return _finite_result("en_similar_flow_amplification", result)


def en_spatial_growth(
    kinematic_shape: float,
    momentum_reynolds_gradient: float,
) -> float:
    """Return ``d(n_tilde)/ds`` from thesis Eq. (7.37)."""
    reynolds_gradient = _finite_scalar(
        "momentum_reynolds_gradient",
        momentum_reynolds_gradient,
    )
    return _finite_result(
        "en_spatial_growth",
        en_growth_per_momentum_reynolds(kinematic_shape)
        * reynolds_gradient,
    )


def transition_amplification_threshold() -> float:
    """Return the source-fixed transition threshold ``n_crit = 9``."""
    return RIZIOTIS_N_CRIT


def transition_shear_coefficient(
    equilibrium_shear_coefficient: float,
) -> float:
    """Return ``C_tau,tr`` from ``sqrt(C_tau,tr)=0.7 sqrt(C_tau,eq)``."""
    equilibrium = _nonnegative_scalar(
        "equilibrium_shear_coefficient",
        equilibrium_shear_coefficient,
    )
    return _finite_result(
        "transition_shear_coefficient",
        RIZIOTIS_CTAU_TRANSITION_SQRT_RATIO**2 * equilibrium,
    )


__all__ = [
    "RIZIOTIS_CTAU_TRANSITION_SQRT_RATIO",
    "RIZIOTIS_N_CRIT",
    "body_tangential_acceleration",
    "density_shape_factor",
    "east_normal_momentum_thickness",
    "en_growth_per_momentum_reynolds",
    "en_onset_momentum_reynolds",
    "en_similar_flow_amplification",
    "en_spatial_growth",
    "kinematic_shape_factor",
    "laminar_dissipation_coefficient",
    "laminar_dissipation_ratio",
    "laminar_energy_shape_factor",
    "laminar_skin_friction_coefficient",
    "transition_amplification_threshold",
    "transition_shear_coefficient",
    "turbulent_dissipation_coefficient",
    "turbulent_energy_shape_factor",
    "turbulent_equilibrium_shear_coefficient",
    "turbulent_h0",
    "turbulent_nominal_thickness",
    "turbulent_skin_friction_coefficient",
    "turbulent_slip_velocity_ratio",
]
