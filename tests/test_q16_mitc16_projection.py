"""MITC16 assumed-covariant-strain projection for the fixed Q16 shell."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluxvortex.q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_PARAMETRIC_NODES,
    Q16ReferenceElement,
)
from fluxvortex.q16_mitc16_projection import (
    MITC16_TYING_POINTS_3,
    MITC16_TYING_POINTS_4,
    Q16MITC16Projector,
)


def _reference(
    *, length: float = 2.0, width: float = 1.0, thickness: float = 0.02
) -> Q16ReferenceElement:
    rows = np.asarray(
        [
            [
                0.5 * length * (xi + 1.0),
                0.5 * width * (eta + 1.0),
                0.0,
                0.0,
                0.0,
                0.5 * thickness,
            ]
            for xi, eta in Q16_PARAMETRIC_NODES
        ],
        dtype=np.float64,
    )
    return Q16ReferenceElement(np.ascontiguousarray(rows))


def _smooth_state(reference: Q16ReferenceElement) -> np.ndarray:
    rows = reference.reference_rows.copy()
    x = rows[:, 0].copy()
    y = rows[:, 1].copy()
    rows[:, 0] += 0.013 * x + 0.004 * x * y
    rows[:, 1] += -0.009 * y + 0.003 * x**2
    rows[:, 2] += 0.021 * x * y + 0.006 * x**2 - 0.004 * y**2
    rows[:, 3:] += np.column_stack([0.0012 * y, -0.0017 * x, 0.0025 + 0.0008 * x])
    return np.ascontiguousarray(rows.reshape(Q16_DOF_PER_ELEMENT))


def _rigid_state(reference: Q16ReferenceElement) -> np.ndarray:
    axis = np.asarray([0.3, -0.4, 0.8], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.63
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    )
    rows = reference.reference_rows.copy()
    rows[:, :3] = rows[:, :3] @ rotation.T + [0.7, -0.3, 0.2]
    rows[:, 3:] = rows[:, 3:] @ rotation.T
    return np.ascontiguousarray(rows.reshape(Q16_DOF_PER_ELEMENT))


def _cylindrical_kirchhoff_state(
    reference: Q16ReferenceElement, *, radius: float = 3.0
) -> np.ndarray:
    rows = reference.reference_rows.copy()
    x = rows[:, 0].copy()
    angle = x / radius
    half_thickness = float(reference.reference_rows[0, 5])
    rows[:, :3] = np.column_stack(
        [
            radius * np.sin(angle),
            reference.reference_rows[:, 1],
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


def test_mitc16_tying_sets_are_the_published_fixed_gauss_layout() -> None:
    expected_three = np.asarray(
        [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)],
        dtype=np.float64,
    )
    expected_four = np.polynomial.legendre.leggauss(4)[0]
    np.testing.assert_array_equal(MITC16_TYING_POINTS_3, expected_three)
    np.testing.assert_array_equal(MITC16_TYING_POINTS_4, expected_four)
    assert not MITC16_TYING_POINTS_3.flags.writeable
    assert not MITC16_TYING_POINTS_4.flags.writeable


@pytest.mark.parametrize("zeta", [-0.61, 0.0, 0.47])
def test_mitc16_projected_components_reproduce_all_tying_values(zeta: float) -> None:
    reference = _reference()
    state = _smooth_state(reference)
    projector = Q16MITC16Projector(reference)

    for eta in MITC16_TYING_POINTS_4:
        for xi in MITC16_TYING_POINTS_3:
            projected = projector.covariant_strain(
                state, float(xi), float(eta), float(zeta)
            )
            direct = reference.green_lagrange_strain(
                state, float(xi), float(eta), float(zeta)
            )
            assert projected[0] == pytest.approx(direct[0], abs=3.0e-15)
            midplane = reference.green_lagrange_strain(
                state, float(xi), float(eta), 0.0
            )
            assert projected[5] == pytest.approx(midplane[5], abs=3.0e-15)

    for eta in MITC16_TYING_POINTS_3:
        for xi in MITC16_TYING_POINTS_4:
            projected = projector.covariant_strain(
                state, float(xi), float(eta), float(zeta)
            )
            direct = reference.green_lagrange_strain(
                state, float(xi), float(eta), float(zeta)
            )
            assert projected[1] == pytest.approx(direct[1], abs=3.0e-15)
            midplane = reference.green_lagrange_strain(
                state, float(xi), float(eta), 0.0
            )
            assert projected[4] == pytest.approx(midplane[4], abs=3.0e-15)

    for eta in MITC16_TYING_POINTS_3:
        for xi in MITC16_TYING_POINTS_3:
            projected = projector.covariant_strain(
                state, float(xi), float(eta), float(zeta)
            )
            direct = reference.green_lagrange_strain(
                state, float(xi), float(eta), float(zeta)
            )
            assert projected[3] == pytest.approx(direct[3], abs=3.0e-15)


def test_mitc16_keeps_transverse_normal_strain_explicitly_compatible() -> None:
    reference = _reference()
    state = _smooth_state(reference)
    projector = Q16MITC16Projector(reference)
    for xi, eta, zeta in ((-0.81, 0.23, -0.4), (0.14, -0.55, 0.0), (0.7, 0.8, 0.6)):
        projected = projector.covariant_strain(state, xi, eta, zeta)
        direct = reference.green_lagrange_strain(state, xi, eta, zeta)
        assert projected[2] == direct[2]
    assert projector.transverse_normal_mode == "compatible-not-ans-eas"


def test_mitc16_rigid_motion_has_zero_projected_strain() -> None:
    reference = _reference()
    state = _rigid_state(reference)
    projector = Q16MITC16Projector(reference)
    for xi, eta, zeta in ((-0.9, -0.8, -1.0), (-0.2, 0.4, 0.1), (0.8, 0.7, 1.0)):
        strain = projector.covariant_strain(state, xi, eta, zeta)
        assert float(np.max(np.abs(strain))) <= 2.0e-15


def test_mitc16_reduces_cylindrical_bending_parasitic_shear_by_fifty_fold() -> None:
    reference = _reference()
    state = _cylindrical_kirchhoff_state(reference)
    projector = Q16MITC16Projector(reference)
    sample = np.linspace(-1.0, 1.0, 21)
    direct_shear: list[float] = []
    projected_shear: list[float] = []
    for eta in sample:
        for xi in sample:
            direct = reference.green_lagrange_strain(state, float(xi), float(eta), 0.0)
            projected = projector.covariant_strain(state, float(xi), float(eta), 0.0)
            direct_shear.extend([float(direct[4]), float(direct[5])])
            projected_shear.extend([float(projected[4]), float(projected[5])])
    direct_norm = float(np.linalg.norm(direct_shear))
    projected_norm = float(np.linalg.norm(projected_shear))
    assert direct_norm > 0.0
    assert projected_norm / direct_norm <= 0.02


def test_mitc16_rejects_wrong_state_and_coordinate_domains() -> None:
    reference = _reference()
    projector = Q16MITC16Projector(reference)
    state = _smooth_state(reference)
    with pytest.raises(TypeError, match="exact numpy.ndarray"):
        projector.covariant_strain(list(state), 0.0, 0.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="float64"):
        projector.covariant_strain(state.astype(np.float32), 0.0, 0.0, 0.0)
    bad = state.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        projector.covariant_strain(bad, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match=r"xi must lie in \[-1, 1\]"):
        projector.covariant_strain(state, 1.01, 0.0, 0.0)
