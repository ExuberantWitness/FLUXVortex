"""Typed span-parity primitives for the frozen S3ai-v2.2 protocol.

This module contains no body/wake solve and no aerodynamic closure.  It
implements only the active seven-coefficient reflection algebra and the
zero-safe, same-space observation rules preregistered in S3ai-v2.2.

Two details are deliberately fail-closed:

* the reflection is the declared canonical reversal, never a fitted or
  coordinate-inferred permutation; and
* the active mass must be exactly symmetric, positive definite, and
  persymmetric, so primal and dual even/odd decompositions cannot silently
  mix incompatible spaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


ACTIVE_TRACE_SIZE = 7
MetricRole = Literal["primal", "dual"]
Parity = Literal["even", "odd"]

MANUFACTURED_EVEN_ACTIVE = np.array(
    (1.0, -2.0, 3.0, 4.0, 3.0, -2.0, 1.0),
    dtype=float,
)
MANUFACTURED_ODD_ACTIVE = np.array(
    (1.0, -2.0, 3.0, 0.0, -3.0, 2.0, -1.0),
    dtype=float,
)
MANUFACTURED_ODD_AMPLITUDE = 2.0 ** -10


class ReachablePressureSymmetryError(ValueError):
    """The frozen active-space reflection or parity family is invalid."""


@dataclass(frozen=True)
class SpanParityOperators:
    """Canonical reversal and its exact even/odd projectors."""

    reflection: np.ndarray
    even_projector: np.ndarray
    odd_projector: np.ndarray
    mass: np.ndarray


@dataclass(frozen=True)
class ParityDecompositionReport:
    """One typed orthogonal even/odd decomposition."""

    metric_role: MetricRole
    value: np.ndarray
    even: np.ndarray
    odd: np.ndarray
    value_norm: float
    even_norm: float
    odd_norm: float
    pythagorean_abs_residual: float


@dataclass(frozen=True)
class ProjectedQuadratureInterval:
    """Zero-safe q8/q10/q12 interval for one projected component."""

    parity: Parity
    metric_role: MetricRole
    q8_projected: np.ndarray
    q10_projected: np.ndarray
    q12_projected: np.ndarray
    coarse_medium_change: float
    medium_fine_change: float
    contraction_ratio: float | None
    floating_round_term: float
    floating_plateau: bool
    tail_allowance: float
    uncertainty: float
    observed_norm: float
    lower: float | None
    upper: float | None
    passed: bool


@dataclass(frozen=True)
class ParityDecisionReport:
    """Combined even/odd interval and the frozen v2.2 decision semantics."""

    even: ProjectedQuadratureInterval
    odd: ProjectedQuadratureInterval
    protocol_no_go: bool
    resolved_symmetry_failure: bool
    no_resolved_symmetry_violation: bool
    relative_odd_to_even_upper: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class StagewiseOddNoncancellation:
    """Sum of local odd dual norms before any window cancellation."""

    stage_count: int
    stage_odd_norms: tuple[float, ...]
    value: float
    window_residual: np.ndarray
    window_odd_norm: float


def _finite_active_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if (
        vector.shape != (ACTIVE_TRACE_SIZE,)
        or not np.all(np.isfinite(vector))
    ):
        raise ReachablePressureSymmetryError(
            f"{name} must be a finite active vector with shape "
            f"{(ACTIVE_TRACE_SIZE,)}"
        )
    return vector.copy()


def canonical_active_reflection() -> np.ndarray:
    """Return the frozen active reflection ``J7*v == v[::-1]``."""

    return np.eye(ACTIVE_TRACE_SIZE, dtype=float)[::-1].copy()


def _canonical_reflection(value: Any | None) -> np.ndarray:
    expected = canonical_active_reflection()
    if value is None:
        return expected
    reflection = np.asarray(value, dtype=float)
    if (
        reflection.shape != expected.shape
        or not np.all(np.isfinite(reflection))
        or not np.array_equal(reflection, expected)
    ):
        raise ReachablePressureSymmetryError(
            "active reflection must be the declared canonical reversal J7"
        )
    return reflection.copy()


def _active_mass(value: Any, reflection: np.ndarray) -> np.ndarray:
    mass = np.asarray(value, dtype=float)
    shape = (ACTIVE_TRACE_SIZE, ACTIVE_TRACE_SIZE)
    if (
        mass.shape != shape
        or not np.all(np.isfinite(mass))
        or not np.array_equal(mass, mass.T)
    ):
        raise ReachablePressureSymmetryError(
            "active mass must be a finite exactly symmetric 7x7 matrix"
        )
    try:
        np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as error:
        raise ReachablePressureSymmetryError(
            "active mass must be positive definite"
        ) from error
    if not np.array_equal(reflection @ mass @ reflection, mass):
        raise ReachablePressureSymmetryError(
            "active mass must be exactly persymmetric under J7"
        )
    if not np.array_equal(reflection @ mass, mass @ reflection):
        raise ReachablePressureSymmetryError(
            "active mass must commute exactly with J7"
        )
    return mass.copy()


def span_parity_operators(
    mass: Any,
    *,
    reflection: Any | None = None,
) -> SpanParityOperators:
    """Validate and return the exact S3ai-v2.2 active parity algebra."""

    reverse = _canonical_reflection(reflection)
    metric = _active_mass(mass, reverse)
    identity = np.eye(ACTIVE_TRACE_SIZE, dtype=float)
    even = 0.5 * (identity + reverse)
    odd = 0.5 * (identity - reverse)
    exact_checks = (
        np.array_equal(even @ even, even),
        np.array_equal(odd @ odd, odd),
        np.array_equal(even + odd, identity),
        np.array_equal(even @ odd, np.zeros_like(identity)),
        np.array_equal(odd @ even, np.zeros_like(identity)),
    )
    if not all(exact_checks):
        raise ReachablePressureSymmetryError(
            "canonical parity projectors failed their exact algebra"
        )
    return SpanParityOperators(
        reflection=reverse,
        even_projector=even,
        odd_projector=odd,
        mass=metric,
    )


def typed_mass_norm(
    value: Any,
    mass: Any,
    *,
    metric_role: MetricRole,
    reflection: Any | None = None,
) -> float:
    """Return the primal ``M`` norm or dual ``M^-1`` norm."""

    vector = _finite_active_vector("value", value)
    operators = span_parity_operators(mass, reflection=reflection)
    if metric_role == "primal":
        square = float(vector @ operators.mass @ vector)
    elif metric_role == "dual":
        square = float(
            vector @ np.linalg.solve(operators.mass, vector)
        )
    else:
        raise ReachablePressureSymmetryError(
            "metric_role must be exactly 'primal' or 'dual'"
        )
    scale = max(
        1.0,
        float(np.linalg.norm(vector)) ** 2
        * float(np.linalg.norm(operators.mass, ord=2)),
    )
    if square < -64.0 * np.finfo(float).eps * scale:
        raise ReachablePressureSymmetryError(
            "typed mass norm received a non-positive metric"
        )
    return float(np.sqrt(max(square, 0.0)))


def project_active_parity(
    value: Any,
    mass: Any,
    *,
    parity: Parity,
    reflection: Any | None = None,
) -> np.ndarray:
    """Project an active vector with the frozen ``Pi+`` or ``Pi-``."""

    vector = _finite_active_vector("value", value)
    operators = span_parity_operators(mass, reflection=reflection)
    if parity == "even":
        return operators.even_projector @ vector
    if parity == "odd":
        return operators.odd_projector @ vector
    raise ReachablePressureSymmetryError(
        "parity must be exactly 'even' or 'odd'"
    )


def parity_decomposition(
    value: Any,
    mass: Any,
    *,
    metric_role: MetricRole,
    reflection: Any | None = None,
) -> ParityDecompositionReport:
    """Return a typed decomposition and its Pythagorean residual."""

    vector = _finite_active_vector("value", value)
    operators = span_parity_operators(mass, reflection=reflection)
    even = operators.even_projector @ vector
    odd = operators.odd_projector @ vector
    value_norm = typed_mass_norm(
        vector,
        operators.mass,
        metric_role=metric_role,
    )
    even_norm = typed_mass_norm(
        even,
        operators.mass,
        metric_role=metric_role,
    )
    odd_norm = typed_mass_norm(
        odd,
        operators.mass,
        metric_role=metric_role,
    )
    residual = abs(
        value_norm * value_norm
        - even_norm * even_norm
        - odd_norm * odd_norm
    )
    return ParityDecompositionReport(
        metric_role=metric_role,
        value=vector,
        even=even,
        odd=odd,
        value_norm=value_norm,
        even_norm=even_norm,
        odd_norm=odd_norm,
        pythagorean_abs_residual=float(residual),
    )


def unprojected_operand_round_term(
    operands: Sequence[Any],
    mass: Any,
    *,
    metric_role: MetricRole,
    floating_factor: float = 4096.0,
    reflection: Any | None = None,
) -> float:
    """Scale one float64 round term from pre-projection typed operands."""

    factor = float(floating_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ReachablePressureSymmetryError(
            "floating_factor must be finite and positive"
        )
    if not operands:
        raise ReachablePressureSymmetryError(
            "unprojected round scaling requires at least one operand"
        )
    operators = span_parity_operators(mass, reflection=reflection)
    vectors = [
        _finite_active_vector(f"operand_{index}", operand)
        for index, operand in enumerate(operands)
    ]
    scale = max(
        typed_mass_norm(
            vector,
            operators.mass,
            metric_role=metric_role,
        )
        for vector in vectors
    )
    return float(factor * np.finfo(float).eps * scale)


def projected_quadrature_interval(
    *,
    q8: Any,
    q10: Any,
    q12: Any,
    mass: Any,
    parity: Parity,
    metric_role: MetricRole,
    unprojected_round_operands: Sequence[Any],
    floating_factor: float = 4096.0,
    reflection: Any | None = None,
) -> ProjectedQuadratureInterval:
    """Form the frozen projected q8/q10/q12 geometric-tail interval.

    Projection precedes both differences.  The float64 term is instead
    scaled from the explicitly supplied *unprojected* operands and is added
    once, after the geometric tail is formed.
    """

    operators = span_parity_operators(mass, reflection=reflection)
    vectors = {
        "q8": _finite_active_vector("q8", q8),
        "q10": _finite_active_vector("q10", q10),
        "q12": _finite_active_vector("q12", q12),
    }
    projector = (
        operators.even_projector
        if parity == "even"
        else operators.odd_projector
        if parity == "odd"
        else None
    )
    if projector is None:
        raise ReachablePressureSymmetryError(
            "parity must be exactly 'even' or 'odd'"
        )
    projected = {
        name: projector @ vector
        for name, vector in vectors.items()
    }
    delta1 = typed_mass_norm(
        projected["q10"] - projected["q8"],
        operators.mass,
        metric_role=metric_role,
    )
    delta2 = typed_mass_norm(
        projected["q12"] - projected["q10"],
        operators.mass,
        metric_role=metric_role,
    )
    round_term = unprojected_operand_round_term(
        unprojected_round_operands,
        operators.mass,
        metric_role=metric_role,
        floating_factor=floating_factor,
    )
    plateau = max(delta1, delta2) <= round_term
    if plateau:
        ratio = None if delta1 == 0.0 else float(delta2 / delta1)
        tail = delta2
        passed = True
    elif delta1 <= round_term:
        ratio = None
        tail = np.inf
        passed = False
    else:
        ratio = float(delta2 / delta1)
        passed = bool(ratio < 1.0)
        tail = float(delta2 / (1.0 - ratio)) if passed else np.inf

    observed = typed_mass_norm(
        projected["q12"],
        operators.mass,
        metric_role=metric_role,
    )
    if passed:
        uncertainty = float(tail + round_term)
        lower: float | None = max(0.0, observed - uncertainty)
        upper: float | None = observed + uncertainty
    else:
        uncertainty = np.inf
        lower = None
        upper = None
    return ProjectedQuadratureInterval(
        parity=parity,
        metric_role=metric_role,
        q8_projected=projected["q8"].copy(),
        q10_projected=projected["q10"].copy(),
        q12_projected=projected["q12"].copy(),
        coarse_medium_change=delta1,
        medium_fine_change=delta2,
        contraction_ratio=ratio,
        floating_round_term=round_term,
        floating_plateau=plateau,
        tail_allowance=float(tail),
        uncertainty=uncertainty,
        observed_norm=observed,
        lower=lower,
        upper=upper,
        passed=passed,
    )


def parity_quadrature_decision(
    *,
    q8: Any,
    q10: Any,
    q12: Any,
    mass: Any,
    metric_role: MetricRole,
    unprojected_round_operands: Sequence[Any],
    floating_factor: float = 4096.0,
    reflection: Any | None = None,
) -> ParityDecisionReport:
    """Apply the v2.2 ``L_->0`` failure and zero-safe ratio semantics."""

    even = projected_quadrature_interval(
        q8=q8,
        q10=q10,
        q12=q12,
        mass=mass,
        parity="even",
        metric_role=metric_role,
        unprojected_round_operands=unprojected_round_operands,
        floating_factor=floating_factor,
        reflection=reflection,
    )
    odd = projected_quadrature_interval(
        q8=q8,
        q10=q10,
        q12=q12,
        mass=mass,
        parity="odd",
        metric_role=metric_role,
        unprojected_round_operands=unprojected_round_operands,
        floating_factor=floating_factor,
        reflection=reflection,
    )
    reasons: list[str] = []
    if not even.passed:
        reasons.append("even projected q family did not contract or plateau")
    if not odd.passed:
        reasons.append("odd projected q family did not contract or plateau")
    resolved_failure = bool(
        odd.passed
        and odd.lower is not None
        and odd.lower > 0.0
    )
    if resolved_failure:
        reasons.append("odd lower bound L_minus is positive")
    protocol_no_go = bool(reasons)
    no_resolved = bool(
        even.passed
        and odd.passed
        and odd.lower == 0.0
    )
    relative: float | None = None
    if (
        even.passed
        and odd.passed
        and even.lower is not None
        and even.lower > 0.0
        and odd.upper is not None
    ):
        relative = float(odd.upper / even.lower)
    return ParityDecisionReport(
        even=even,
        odd=odd,
        protocol_no_go=protocol_no_go,
        resolved_symmetry_failure=resolved_failure,
        no_resolved_symmetry_violation=no_resolved,
        relative_odd_to_even_upper=relative,
        reasons=tuple(reasons),
    )


def stagewise_odd_noncancellation(
    residuals: Any,
    mass: Any,
    *,
    reflection: Any | None = None,
) -> StagewiseOddNoncancellation:
    """Return ``sum_n ||Pi_minus R_n||_(M^-1)`` for dual residuals."""

    stages = np.asarray(residuals, dtype=float)
    if (
        stages.ndim != 2
        or stages.shape[0] < 1
        or stages.shape[1] != ACTIVE_TRACE_SIZE
        or not np.all(np.isfinite(stages))
    ):
        raise ReachablePressureSymmetryError(
            "residuals must be a nonempty finite array with shape (n,7)"
        )
    operators = span_parity_operators(mass, reflection=reflection)
    odd_stages = stages @ operators.odd_projector.T
    norms = tuple(
        typed_mass_norm(
            stage,
            operators.mass,
            metric_role="dual",
        )
        for stage in odd_stages
    )
    window = np.sum(stages, axis=0)
    window_odd = operators.odd_projector @ window
    return StagewiseOddNoncancellation(
        stage_count=int(stages.shape[0]),
        stage_odd_norms=norms,
        value=float(sum(norms)),
        window_residual=window.copy(),
        window_odd_norm=typed_mass_norm(
            window_odd,
            operators.mass,
            metric_role="dual",
        ),
    )
