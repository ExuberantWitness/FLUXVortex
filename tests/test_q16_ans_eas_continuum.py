"""Q16 MITC16 plus transverse-normal ANS/EAS condensation oracle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluxvortex.q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_PARAMETRIC_NODES,
    Q16ReferenceElement,
)
from fluxvortex.q16_ans_eas_continuum import Q16MITC16EASContinuumElement


def _element(*, thickness: float = 0.02) -> Q16MITC16EASContinuumElement:
    rows = np.asarray(
        [
            [
                xi + 1.0,
                0.5 * (eta + 1.0),
                0.0,
                0.0,
                0.0,
                0.5 * thickness,
            ]
            for xi, eta in Q16_PARAMETRIC_NODES
        ],
        dtype=np.float64,
    )
    reference = Q16ReferenceElement(np.ascontiguousarray(rows))
    return Q16MITC16EASContinuumElement(
        reference,
        young_modulus=7.0e7,
        poisson_ratio=0.3,
        density=1120.0,
    )


def _cylindrical_state(
    element: Q16MITC16EASContinuumElement, *, radius: float = 3.0
) -> np.ndarray:
    rows = element.reference.reference_rows.copy()
    x = rows[:, 0].copy()
    angle = x / radius
    half_thickness = float(element.reference.reference_rows[0, 5])
    rows[:, :3] = np.column_stack(
        [
            radius * np.sin(angle),
            element.reference.reference_rows[:, 1],
            radius * (1.0 - np.cos(angle)),
        ]
    )
    rows[:, 3:] = np.column_stack(
        [
            -half_thickness * np.sin(angle),
            np.zeros(16, dtype=np.float64),
            half_thickness * np.cos(angle),
        ]
    )
    return np.ascontiguousarray(rows.reshape(Q16_DOF_PER_ELEMENT))


def _rigid_state(element: Q16MITC16EASContinuumElement) -> np.ndarray:
    angle = 0.38
    rotation = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float64,
    )
    rows = element.reference.reference_rows.copy()
    rows[:, :3] = rows[:, :3] @ rotation.T + [0.2, -0.1, 0.7]
    rows[:, 3:] = rows[:, 3:] @ rotation.T
    return np.ascontiguousarray(rows.reshape(Q16_DOF_PER_ELEMENT))


def test_q16_ans_eas_rigid_motion_is_null_and_stationary() -> None:
    element = _element()
    state = _rigid_state(element)
    alpha = element.solve_enhanced_parameter(state)
    assert abs(alpha) <= 2.0e-16
    assert abs(element.enhanced_stationarity_residual(state, alpha)) <= 2.0e-12
    assert element.strain_energy(state) <= 2.0e-20
    assert float(np.linalg.norm(element.internal_force(state))) <= 3.0e-7


def test_q16_eas_mode_is_orthogonal_to_constant_stress_on_distorted_geometry() -> None:
    rows = _element().reference.reference_rows.copy()
    x = rows[:, 0].copy()
    y = rows[:, 1].copy()
    rows[:, 2] = 0.04 * x * y + 0.015 * x**2
    rows[:, 3] = 0.002 * y
    rows[:, 4] = -0.003 * x
    rows[:, 5] *= 1.0 + 0.04 * x - 0.03 * y
    element = Q16MITC16EASContinuumElement(
        Q16ReferenceElement(np.ascontiguousarray(rows)),
        young_modulus=7.0e7,
        poisson_ratio=0.3,
        density=1120.0,
    )
    scale = element.reference_volume * float(
        np.max(np.abs(element.quadrature.enhanced_modes))
    )
    assert float(np.max(np.abs(element.enhanced_mode_volume_integral))) <= (
        512.0 * np.finfo(np.float64).eps * max(scale, 1.0)
    )


def test_q16_ans_eas_condensation_is_stationary_and_lowers_bending_energy() -> None:
    element = _element()
    state = _cylindrical_state(element)
    alpha = element.solve_enhanced_parameter(state)
    assert abs(alpha) > 1.0e-8
    residual = element.enhanced_stationarity_residual(state, alpha)
    assert math.isfinite(residual)
    assert (
        element.enhanced_stationarity_relative_residual(state, alpha)
        <= 512.0 * np.finfo(np.float64).eps
    )
    condensed = element.strain_energy(state)
    compatible = element.strain_energy_at_parameter(state, 0.0)
    assert condensed < compatible
    assert condensed > 0.0


def test_q16_ans_eas_internal_force_is_condensed_energy_derivative() -> None:
    element = _element()
    state = _cylindrical_state(element, radius=2.7)
    direction = np.random.default_rng(601).normal(size=Q16_DOF_PER_ELEMENT)
    direction = np.ascontiguousarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    # Director coordinates are O(thickness); use a small absolute perturbation
    # so the centered derivative remains in its asymptotic range.
    step = 2.0e-8
    difference = (
        element.strain_energy(np.ascontiguousarray(state + step * direction))
        - element.strain_energy(np.ascontiguousarray(state - step * direction))
    ) / (2.0 * step)
    analytic = float(element.internal_force(state) @ direction)
    assert analytic == pytest.approx(difference, rel=5.0e-7, abs=5.0e-5)


def test_q16_ans_eas_analytic_condensed_jv_matches_force_difference() -> None:
    element = _element()
    state = _cylindrical_state(element, radius=2.4)
    direction = np.random.default_rng(1701).normal(size=Q16_DOF_PER_ELEMENT)
    direction = np.ascontiguousarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    step = 8.0e-7
    difference = (
        element.internal_force(np.ascontiguousarray(state + step * direction))
        - element.internal_force(np.ascontiguousarray(state - step * direction))
    ) / (2.0 * step)
    analytic = element.tangent_action(state, direction)
    relative = float(
        np.linalg.norm(analytic - difference) / max(np.linalg.norm(difference), 1.0)
    )
    assert relative <= 8.0e-7


@pytest.mark.parametrize("thickness_ratio", [1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1])
def test_q16_ans_eas_registered_thickness_ratios_remain_finite(
    thickness_ratio: float,
) -> None:
    element = _element(thickness=2.0 * thickness_ratio)
    state = _cylindrical_state(element)
    alpha = element.solve_enhanced_parameter(state)
    energy = element.strain_energy(state)
    force = element.internal_force(state)
    assert math.isfinite(alpha)
    assert math.isfinite(energy) and energy > 0.0
    assert bool(np.isfinite(force).all())


def test_q16_ans_eas_rejects_malformed_state_and_parameter() -> None:
    element = _element()
    state = _cylindrical_state(element)
    with pytest.raises(TypeError, match="exact numpy.ndarray"):
        element.strain_energy(list(state))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="float64"):
        element.internal_force(state.astype(np.float32))
    bad = state.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        element.solve_enhanced_parameter(bad)
    with pytest.raises(ValueError, match="finite"):
        element.strain_energy_at_parameter(state, float("nan"))

    inverted = state.copy().reshape(16, 6)
    inverted[:, 3:] *= -1.0
    with pytest.raises(FloatingPointError, match="orientation reversing"):
        element.strain_energy(np.ascontiguousarray(inverted.reshape(96)))
