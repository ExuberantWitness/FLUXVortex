"""Focused mathematical contract for the fixed Q16 ANCF macro-shell."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluxvortex.q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_DOF_PER_NODE,
    Q16_NODE_COUNT,
    Q16_PARAMETRIC_NODES,
    Q16_PARAMETRIC_NODES_1D,
    Q16ReferenceElement,
    q16_interpolate,
    q16_shape,
)


def _flat_reference(length: float = 2.0, width: float = 1.5, thickness: float = 0.2):
    rows = []
    for xi, eta in Q16_PARAMETRIC_NODES:
        rows.append(
            [
                0.5 * length * (xi + 1.0),
                0.5 * width * (eta + 1.0),
                0.0,
                0.0,
                0.0,
                0.5 * thickness,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _state_from_rows(rows: np.ndarray) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(Q16_DOF_PER_ELEMENT)


def test_q16_contract_is_exactly_sixteen_nodes_and_ninety_six_dofs() -> None:
    assert Q16_NODE_COUNT == 16
    assert Q16_DOF_PER_NODE == 6
    assert Q16_DOF_PER_ELEMENT == 96
    expected = np.array(
        [-1.0, -1.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 1.0],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(Q16_PARAMETRIC_NODES_1D, expected)
    assert Q16_PARAMETRIC_NODES.shape == (16, 2)
    assert Q16_PARAMETRIC_NODES.flags.writeable is False


def test_q16_shape_is_kronecker_at_all_sixteen_nodes() -> None:
    for active, (xi, eta) in enumerate(Q16_PARAMETRIC_NODES):
        shape, dxi, deta = q16_shape(float(xi), float(eta))
        expected = np.zeros(16, dtype=np.float64)
        expected[active] = 1.0
        np.testing.assert_allclose(shape, expected, rtol=0.0, atol=8.0e-15)
        assert shape.flags.writeable is False
        assert dxi.flags.writeable is False
        assert deta.flags.writeable is False


@pytest.mark.parametrize(
    ("xi", "eta"),
    [(-0.83, -0.41), (-0.2, 0.7), (0.0, 0.0), (0.63, -0.52), (0.91, 0.88)],
)
def test_q16_partition_of_unity_and_derivative_sums(xi: float, eta: float) -> None:
    shape, dxi, deta = q16_shape(xi, eta)
    assert abs(float(np.sum(shape)) - 1.0) <= 64.0 * np.finfo(np.float64).eps
    assert abs(float(np.sum(dxi))) <= 64.0 * np.finfo(np.float64).eps
    assert abs(float(np.sum(deta))) <= 64.0 * np.finfo(np.float64).eps


def test_q16_reproduces_a_tensor_cubic_field_without_lower_order_oracle() -> None:
    xi_nodes = Q16_PARAMETRIC_NODES[:, 0]
    eta_nodes = Q16_PARAMETRIC_NODES[:, 1]

    def polynomial(xi: np.ndarray | float, eta: np.ndarray | float):
        return (
            0.7
            - 0.2 * xi
            + 0.4 * eta
            + 0.13 * xi**3
            - 0.31 * eta**3
            + 0.27 * xi**2 * eta
            - 0.19 * xi * eta**2
            + 0.11 * xi**3 * eta**3
        )

    nodal = polynomial(xi_nodes, eta_nodes)
    for xi, eta in [(-0.73, 0.21), (-0.11, -0.67), (0.37, 0.49), (0.82, -0.35)]:
        actual = q16_interpolate(nodal, xi, eta)
        assert actual == pytest.approx(polynomial(xi, eta), rel=0.0, abs=2.0e-14)


def test_q16_flat_reference_maps_position_and_covariant_bases_exactly() -> None:
    length, width, thickness = 2.0, 1.5, 0.2
    reference = Q16ReferenceElement(_flat_reference(length, width, thickness))
    q = reference.reference_state
    xi, eta, zeta = 0.23, -0.41, 0.37
    np.testing.assert_allclose(
        reference.position(q, xi, eta, zeta),
        [
            0.5 * length * (xi + 1.0),
            0.5 * width * (eta + 1.0),
            0.5 * thickness * zeta,
        ],
        rtol=0.0,
        atol=2.0e-14,
    )
    a_xi, a_eta, a_zeta = reference.covariant_bases(q, xi, eta, zeta)
    np.testing.assert_allclose(a_xi, [0.5 * length, 0.0, 0.0], atol=2.0e-14)
    np.testing.assert_allclose(a_eta, [0.0, 0.5 * width, 0.0], atol=2.0e-14)
    np.testing.assert_allclose(a_zeta, [0.0, 0.0, 0.5 * thickness], atol=2.0e-14)


def test_q16_green_lagrange_strain_is_zero_under_finite_rigid_motion() -> None:
    reference = Q16ReferenceElement(_flat_reference())
    angle = 0.61
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transformed = reference.reference_rows.copy()
    transformed[:, :3] = transformed[:, :3] @ rotation.T + [0.4, -0.7, 1.2]
    transformed[:, 3:] = transformed[:, 3:] @ rotation.T
    q = _state_from_rows(transformed)
    for xi, eta, zeta in [(-0.7, 0.2, -0.8), (0.0, 0.0, 0.0), (0.61, -0.33, 0.9)]:
        strain = reference.green_lagrange_strain(q, xi, eta, zeta)
        np.testing.assert_allclose(strain, np.zeros(6), rtol=0.0, atol=2.0e-14)


def test_q16_thickness_stretch_patch_matches_metric_definition() -> None:
    reference = Q16ReferenceElement(_flat_reference(thickness=0.2))
    rows = reference.reference_rows.copy()
    rows[:, 3:] *= 1.1
    strain = reference.green_lagrange_strain(_state_from_rows(rows), 0.17, -0.44, 0.31)
    g2 = (0.1) ** 2
    expected_e33 = 0.5 * ((1.1**2) * g2 - g2)
    assert strain[2] == pytest.approx(expected_e33, rel=0.0, abs=3.0e-16)
    np.testing.assert_allclose(strain[[0, 1, 3, 4, 5]], 0.0, atol=2.0e-14)


def test_q16_consistent_mass_is_symmetric_positive_and_recovers_total_mass() -> None:
    density = 7.5
    length, width, thickness = 2.0, 1.5, 0.2
    reference = Q16ReferenceElement(_flat_reference(length, width, thickness))
    mass = reference.consistent_mass_matrix(density=density)
    np.testing.assert_allclose(mass, mass.T, rtol=0.0, atol=3.0e-14)
    eigenvalues = np.linalg.eigvalsh(mass)
    assert float(np.min(eigenvalues)) > 0.0
    velocity = np.zeros(Q16_DOF_PER_ELEMENT, dtype=np.float64)
    velocity.reshape(16, 6)[:, 0] = 1.0
    recovered = float(velocity @ mass @ velocity)
    assert recovered == pytest.approx(density * length * width * thickness, rel=2.0e-13)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((15, 6), dtype=np.float64),
        np.zeros((16, 6), dtype=np.float32),
        np.full((16, 6), np.nan, dtype=np.float64),
    ],
)
def test_q16_reference_rejects_wrong_shape_dtype_and_nonfinite_state(
    bad: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Q16ReferenceElement(bad)
