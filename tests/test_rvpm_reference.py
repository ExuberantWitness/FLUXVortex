from __future__ import annotations

import numpy as np
import pytest

from fluxvortex.rvpm_reference import (
    direct_gaussian_erf_velocity_jacobian,
    pack_julia_column_major,
    unpack_julia_column_major,
)
from fluxvortex.rvpm_transport import (
    corrected_pedrizzetti,
    lsrk3_step_direct,
    make_particle_state,
    reformulated_vpm_rhs,
)


def test_single_particle_self_interaction_is_exact_zero() -> None:
    result = direct_gaussian_erf_velocity_jacobian(
        [[0.2, -0.1, 0.4]],
        [[0.3, 0.2, -0.5]],
        [0.17],
    )
    assert np.array_equal(result.velocity, np.zeros((1, 3)))
    assert np.array_equal(result.jacobian, np.zeros((1, 3, 3)))


def test_julia_column_major_layout_round_trips() -> None:
    jacobian = np.array([[[11.0, 12.0, 13.0], [21.0, 22.0, 23.0], [31.0, 32.0, 33.0]]])
    packed = pack_julia_column_major(jacobian)
    assert np.array_equal(
        packed,
        [[11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0]],
    )
    assert np.array_equal(unpack_julia_column_major(packed), jacobian)


def test_direct_jacobian_matches_velocity_central_difference() -> None:
    source_positions = np.array(
        [[0.0, 0.0, 0.0], [0.31, -0.14, 0.22], [-0.27, 0.38, -0.19]]
    )
    source_gamma = np.array(
        [[0.12, -0.07, 0.05], [-0.03, 0.11, 0.08], [0.09, 0.04, -0.06]]
    )
    source_sigma = np.array([0.16, 0.21, 0.13])
    target = np.array([[0.17, 0.09, -0.11]])
    reference = direct_gaussian_erf_velocity_jacobian(
        source_positions,
        source_gamma,
        source_sigma,
        target_positions=target,
    )

    step = 2.0e-7
    finite_difference = np.zeros((3, 3))
    for coordinate in range(3):
        offset = np.zeros_like(target)
        offset[0, coordinate] = step
        positive = direct_gaussian_erf_velocity_jacobian(
            source_positions,
            source_gamma,
            source_sigma,
            target_positions=target + offset,
        ).velocity[0]
        negative = direct_gaussian_erf_velocity_jacobian(
            source_positions,
            source_gamma,
            source_sigma,
            target_positions=target - offset,
        ).velocity[0]
        finite_difference[:, coordinate] = (positive - negative) / (2.0 * step)

    assert np.allclose(reference.jacobian[0], finite_difference, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize(
    ("positions", "gamma", "sigma", "message"),
    [
        ([[0.0, 0.0]], [[1.0, 0.0, 0.0]], [0.1], "positions"),
        ([[0.0, 0.0, 0.0]], [[1.0, 0.0]], [0.1], "gamma"),
        ([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [0.0], "positive"),
        ([[np.nan, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [0.1], "finite"),
        ([[False, False, False]], [[1.0, 0.0, 0.0]], [0.1], "numeric dtype"),
    ],
)
def test_direct_interaction_fails_closed(
    positions: list[list[float]],
    gamma: list[list[float]],
    sigma: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        direct_gaussian_erf_velocity_jacobian(positions, gamma, sigma)


def test_reformulated_rhs_uses_j_transpose_and_preserves_gamma_sigma_invariant() -> (
    None
):
    gamma = np.array([[0.4, -0.2, 0.3]])
    sigma = np.array([0.17])
    jacobian = np.array([[[0.1, 0.7, -0.2], [-0.3, 0.4, 0.8], [0.6, -0.5, -0.1]]])
    stretching, z_rate, gamma_rate, sigma_rate = reformulated_vpm_rhs(
        gamma,
        sigma,
        jacobian,
    )
    expected_stretching = jacobian[0].T @ gamma[0]
    assert np.allclose(stretching[0], expected_stretching, rtol=0.0, atol=1e-16)
    assert not np.allclose(stretching[0], jacobian[0] @ gamma[0])

    logarithmic_invariant_rate = (
        np.dot(gamma[0], gamma_rate[0]) / np.dot(gamma[0], gamma[0])
        + 2.0 * sigma_rate[0] / sigma[0]
    )
    assert abs(logarithmic_invariant_rate) < 2e-16
    assert np.all(np.isfinite(z_rate))


def test_lsrk3_constant_freestream_is_exact_and_has_three_stages() -> None:
    state = make_particle_state([[0.1, -0.2, 0.3]], [[0.4, 0.2, -0.1]], [0.15])
    velocity = np.array([0.7, -0.4, 0.2])
    advanced, stages = lsrk3_step_direct(
        state,
        0.025,
        freestream_velocity=velocity,
    )
    assert len(stages) == 3
    assert np.allclose(
        advanced.positions,
        state.positions + 0.025 * velocity[None, :],
        rtol=0.0,
        atol=6e-17,
    )
    assert np.array_equal(advanced.gamma, state.gamma)
    assert np.array_equal(advanced.sigma, state.sigma)
    assert all(np.array_equal(stage.rhs.velocity, np.zeros((1, 3))) for stage in stages)
    assert all(
        np.array_equal(stage.rhs.jacobian, np.zeros((1, 3, 3))) for stage in stages
    )


@pytest.mark.parametrize(
    ("delta_time", "freestream", "message"),
    [
        (0.0, [0.0, 0.0, 0.0], "positive"),
        (True, [0.0, 0.0, 0.0], "boolean"),
        (0.1, [0.0, 0.0], "length-3"),
        (0.1, [False, False, False], "numeric dtype"),
    ],
)
def test_lsrk3_invalid_control_inputs_fail_closed(
    delta_time: float,
    freestream: list[float],
    message: str,
) -> None:
    state = make_particle_state([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [0.2])
    with pytest.raises(ValueError, match=message):
        lsrk3_step_direct(state, delta_time, freestream_velocity=freestream)


def test_corrected_pedrizzetti_preserves_old_gamma_norm() -> None:
    gamma = np.array([0.31, -0.22, 0.17])
    jacobian = np.array([[0.1, 0.0, -0.4], [0.7, -0.2, 0.0], [0.0, 0.2, 0.3]])
    relaxed = corrected_pedrizzetti(gamma, jacobian, 0.3)
    assert np.isclose(
        np.linalg.norm(relaxed), np.linalg.norm(gamma), rtol=0.0, atol=1e-16
    )
    assert not np.array_equal(relaxed, gamma)


def test_corrected_pedrizzetti_zero_vorticity_is_bitwise_noop() -> None:
    gamma = np.array([0.31, -0.22, 0.17])
    symmetric_jacobian = np.diag([0.1, -0.2, 0.3])
    relaxed = corrected_pedrizzetti(gamma, symmetric_jacobian, 0.3)
    assert np.array_equal(relaxed, gamma)


def test_corrected_pedrizzetti_rejects_undefined_zero_strength_case() -> None:
    jacobian = np.array([[0.1, 0.0, -0.4], [0.7, -0.2, 0.0], [0.0, 0.2, 0.3]])
    with pytest.raises(ValueError, match="zero gamma"):
        corrected_pedrizzetti(np.zeros(3), jacobian, 0.3)


def test_corrected_pedrizzetti_rejects_antiparallel_singular_mix() -> None:
    # curl(J) = (-1, 0, 0), exactly opposite to Gamma at alpha=1/2.
    jacobian = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    with pytest.raises(FloatingPointError, match="singular"):
        corrected_pedrizzetti(np.array([1.0, 0.0, 0.0]), jacobian, 0.5)


def test_public_primitives_do_not_mutate_inputs() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]])
    gamma = np.array([[0.2, 0.1, -0.1], [-0.1, 0.3, 0.2]])
    sigma = np.array([0.15, 0.18])
    originals = (positions.copy(), gamma.copy(), sigma.copy())
    direct_gaussian_erf_velocity_jacobian(positions, gamma, sigma)
    state = make_particle_state(positions, gamma, sigma)
    lsrk3_step_direct(state, 0.01, freestream_velocity=[0.2, 0.0, 0.0])
    corrected_pedrizzetti(gamma[0], np.diag([0.1, -0.2, 0.3]), 0.3)
    assert np.array_equal(positions, originals[0])
    assert np.array_equal(gamma, originals[1])
    assert np.array_equal(sigma, originals[2])
