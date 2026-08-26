"""CUDA parity for the fixed Q16 MITC16+ANS/EAS condensed operator."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from fluxvortex.q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_PARAMETRIC_NODES,
    Q16ReferenceElement,
)
from fluxvortex.q16_ans_eas_continuum import Q16MITC16EASContinuumElement
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_ans_eas import Q16CudaMITC16EASOperator


pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def _element() -> Q16MITC16EASContinuumElement:
    rows = np.asarray(
        [
            [xi + 1.0, 0.5 * (eta + 1.0), 0.0, 0.0, 0.0, 0.01]
            for xi, eta in Q16_PARAMETRIC_NODES
        ],
        dtype=np.float64,
    )
    return Q16MITC16EASContinuumElement(
        Q16ReferenceElement(np.ascontiguousarray(rows)),
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


def test_q16_ans_eas_cuda_alpha_force_and_jv_match_independent_oracle() -> None:
    element = _element()
    cuda = Q16CudaMITC16EASOperator((element,), device=config.DEVICE)
    states_np = np.ascontiguousarray(
        np.stack(
            [
                _cylindrical_state(element, radius=2.6),
                _cylindrical_state(element, radius=3.2),
            ]
        )[:, None, :]
    )
    directions_np = np.ascontiguousarray(
        np.random.default_rng(2208).normal(size=states_np.shape), dtype=np.float64
    )
    states = wp.array(states_np, dtype=config.DTYPE, device=config.DEVICE)
    directions = wp.array(directions_np, dtype=config.DTYPE, device=config.DEVICE)

    alpha = cuda.enhanced_parameter(states)
    force = cuda.internal_force(states)
    force_repeat = cuda.internal_force(states)
    tangent = cuda.tangent_action(states, directions)
    wp.synchronize_device(config.DEVICE)

    expected_alpha = np.asarray(
        [[element.solve_enhanced_parameter(state[0])] for state in states_np]
    )
    expected_force = np.stack(
        [element.internal_force(state[0]) for state in states_np]
    )[:, None, :]
    expected_tangent = np.stack(
        [
            element.tangent_action(state[0], direction[0])
            for state, direction in zip(states_np, directions_np, strict=True)
        ]
    )[:, None, :]
    np.testing.assert_allclose(alpha.numpy(), expected_alpha, rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(force.numpy(), expected_force, rtol=2e-10, atol=2e-6)
    np.testing.assert_allclose(tangent.numpy(), expected_tangent, rtol=3e-10, atol=8e-6)
    np.testing.assert_array_equal(force_repeat.numpy(), force.numpy())


def test_q16_ans_eas_cuda_rejects_host_wrong_dtype_shape_and_nonfinite() -> None:
    element = _element()
    cuda = Q16CudaMITC16EASOperator((element,), device=config.DEVICE)
    host = wp.zeros((1, 1, Q16_DOF_PER_ELEMENT), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="must reside on CUDA device"):
        cuda.internal_force(host)

    wrong_dtype = wp.zeros(
        (1, 1, Q16_DOF_PER_ELEMENT), dtype=wp.float32, device=config.DEVICE
    )
    with pytest.raises(TypeError, match="float64"):
        cuda.enhanced_parameter(wrong_dtype)

    wrong_shape = wp.zeros((1, 1, 95), dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(ValueError, match="end with shape"):
        cuda.internal_force(wrong_shape)

    bad = _cylindrical_state(element)[None, None, :]
    bad[0, 0, 7] = np.nan
    bad_cuda = wp.array(bad, dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(FloatingPointError, match="contains non-finite"):
        cuda.internal_force(bad_cuda)
