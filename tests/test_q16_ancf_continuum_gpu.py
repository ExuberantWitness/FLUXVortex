"""Q16 continuum residual, consistent mass and Jv CPU/CUDA parity."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from fluxvortex.q16_ancf_continuum import Q16ContinuumElement
from fluxvortex.q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_PARAMETRIC_NODES,
    Q16ReferenceElement,
)
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_ancf import Q16CudaContinuumOperator


pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def _reference_rows(
    length: float = 1.8, width: float = 1.2, thickness: float = 0.16
) -> np.ndarray:
    return np.asarray(
        [
            [
                0.5 * length * (xi + 1.0),
                0.5 * width * (eta + 1.0),
                0.025 * xi * eta,
                0.006 * eta,
                -0.004 * xi,
                0.5 * thickness,
            ]
            for xi, eta in Q16_PARAMETRIC_NODES
        ],
        dtype=np.float64,
    )


def _element() -> Q16ContinuumElement:
    return Q16ContinuumElement(
        Q16ReferenceElement(_reference_rows()),
        young_modulus=7.2e7,
        poisson_ratio=0.31,
        density=1180.0,
    )


def _deformed(element: Q16ContinuumElement, scale: float = 1.0) -> np.ndarray:
    rows = element.reference.reference_rows.copy()
    x = rows[:, 0].copy()
    y = rows[:, 1].copy()
    rows[:, 0] += scale * (0.012 * x + 0.006 * x * y)
    rows[:, 1] += scale * (-0.007 * y + 0.004 * x**2)
    rows[:, 2] += scale * (0.018 * x * y + 0.009 * x**2)
    rows[:, 3:] += scale * np.column_stack(
        [0.001 * y, -0.0015 * x, 0.003 * np.ones(16)]
    )
    return np.ascontiguousarray(rows.reshape(Q16_DOF_PER_ELEMENT))


def _rigid_state(element: Q16ContinuumElement) -> np.ndarray:
    angle = 0.47
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rows = element.reference.reference_rows.copy()
    rows[:, :3] = rows[:, :3] @ rotation.T + [0.3, -0.2, 0.7]
    rows[:, 3:] = rows[:, 3:] @ rotation.T
    return np.ascontiguousarray(rows.reshape(Q16_DOF_PER_ELEMENT))


def test_q16_continuum_finite_rigid_motion_has_zero_energy_and_force() -> None:
    element = _element()
    rigid = _rigid_state(element)
    assert element.strain_energy(rigid) <= 2.0e-20
    assert float(np.linalg.norm(element.internal_force(rigid))) <= 2.0e-7


def test_q16_internal_force_is_energy_directional_derivative() -> None:
    element = _element()
    state = _deformed(element)
    direction = np.random.default_rng(9201).normal(size=Q16_DOF_PER_ELEMENT)
    direction = np.ascontiguousarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    step = 2.0e-7
    plus = np.ascontiguousarray(state + step * direction)
    minus = np.ascontiguousarray(state - step * direction)
    finite_difference = (element.strain_energy(plus) - element.strain_energy(minus)) / (
        2.0 * step
    )
    analytic = float(element.internal_force(state) @ direction)
    assert analytic == pytest.approx(finite_difference, rel=2.0e-7, abs=2.0e-5)


def test_q16_analytic_tangent_action_matches_centered_force_difference() -> None:
    element = _element()
    state = _deformed(element)
    direction = np.random.default_rng(142).normal(size=Q16_DOF_PER_ELEMENT)
    direction = np.ascontiguousarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    step = 1.0e-6
    finite_difference = (
        element.internal_force(np.ascontiguousarray(state + step * direction))
        - element.internal_force(np.ascontiguousarray(state - step * direction))
    ) / (2.0 * step)
    analytic = element.tangent_action(state, direction)
    relative = np.linalg.norm(analytic - finite_difference) / max(
        np.linalg.norm(finite_difference), 1.0
    )
    assert relative <= 2.0e-7


def test_q16_consistent_mass_action_is_symmetric_positive_and_has_exact_mass() -> None:
    element = _element()
    rng = np.random.default_rng(731)
    first = np.ascontiguousarray(rng.normal(size=Q16_DOF_PER_ELEMENT), dtype=np.float64)
    second = np.ascontiguousarray(
        rng.normal(size=Q16_DOF_PER_ELEMENT), dtype=np.float64
    )
    mass_first = element.mass_action(first)
    mass_second = element.mass_action(second)
    assert float(first @ mass_first) > 0.0
    assert float(first @ mass_second) == pytest.approx(
        float(second @ mass_first), rel=2.0e-13, abs=2.0e-12
    )

    rigid_x = np.zeros(Q16_DOF_PER_ELEMENT, dtype=np.float64)
    rigid_x.reshape(16, 6)[:, 0] = 1.0
    recovered_mass = float(rigid_x @ element.mass_action(rigid_x))
    assert recovered_mass == pytest.approx(
        element.reference_volume * element.density, rel=3.0e-13
    )


def test_q16_cuda_residual_mass_and_jv_match_independent_oracle() -> None:
    assert config.dtype_name() == "float64"
    element = _element()
    cuda = Q16CudaContinuumOperator((element,), device=config.DEVICE)
    states_np = np.ascontiguousarray(
        np.stack([_deformed(element, 0.7), _deformed(element, 1.1)])[:, None, :]
    )
    directions_np = np.random.default_rng(83).normal(size=states_np.shape)
    directions_np = np.ascontiguousarray(directions_np, dtype=np.float64)
    states = wp.array(states_np, dtype=config.DTYPE, device=config.DEVICE)
    directions = wp.array(directions_np, dtype=config.DTYPE, device=config.DEVICE)

    force = cuda.internal_force(states)
    force_repeat = cuda.internal_force(states)
    tangent = cuda.tangent_action(states, directions)
    mass = cuda.mass_action(directions)
    wp.synchronize_device(config.DEVICE)

    expected_force = np.stack(
        [element.internal_force(state[0]) for state in states_np]
    )[:, None, :]
    expected_tangent = np.stack(
        [
            element.tangent_action(state[0], direction[0])
            for state, direction in zip(states_np, directions_np, strict=True)
        ]
    )[:, None, :]
    expected_mass = np.stack(
        [element.mass_action(direction[0]) for direction in directions_np]
    )[:, None, :]
    np.testing.assert_allclose(force.numpy(), expected_force, rtol=1.0e-10, atol=2e-7)
    np.testing.assert_allclose(
        tangent.numpy(), expected_tangent, rtol=1.0e-10, atol=4e-7
    )
    np.testing.assert_allclose(mass.numpy(), expected_mass, rtol=1.0e-11, atol=2e-9)
    np.testing.assert_array_equal(force_repeat.numpy(), force.numpy())


def test_q16_cuda_operator_rejects_host_and_nonfinite_states() -> None:
    element = _element()
    cuda = Q16CudaContinuumOperator((element,), device=config.DEVICE)
    host = wp.zeros((1, 1, 96), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="must reside on CUDA device"):
        cuda.internal_force(host)

    bad = _deformed(element)[None, None, :]
    bad[0, 0, 4] = np.nan
    bad_cuda = wp.array(bad, dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(FloatingPointError, match="state contains non-finite"):
        cuda.internal_force(bad_cuda)
