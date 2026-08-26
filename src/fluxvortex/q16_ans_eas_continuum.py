"""Q16 MITC16 continuum oracle with transverse-normal ANS/EAS.

This module combines the fixed MITC16 assumed covariant strain spaces with a
Q16 nodal assumed transverse-normal strain and one thickness-linear enhanced
strain mode.  The enhanced parameter is solved and statically condensed at the
element level.  Energy, residual and tangent action use the same condensed
field.  The implementation is an independent NumPy oracle for later CUDA
translation; it is not a CPU fallback for the production FSI loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_NODE_COUNT,
    Q16_PARAMETRIC_NODES,
    Q16ReferenceElement,
    q16_shape,
)
from .q16_mitc16_projection import (
    MITC16_TYING_POINTS_3,
    MITC16_TYING_POINTS_4,
)

_FLOAT64 = np.dtype(np.float64)
Q16_EAS_IN_PLANE_QUADRATURE_ORDER = 6
Q16_EAS_THICKNESS_QUADRATURE_ORDER = 3
Q16_EAS_QUADRATURE_POINT_COUNT = (
    Q16_EAS_IN_PLANE_QUADRATURE_ORDER**2 * Q16_EAS_THICKNESS_QUADRATURE_ORDER
)


def _readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


def _positive_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_state(name: str, value: np.ndarray) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != _FLOAT64:
        raise TypeError(f"{name} must use float64")
    if value.shape != (Q16_DOF_PER_ELEMENT,):
        raise ValueError(f"{name} must have shape ({Q16_DOF_PER_ELEMENT},)")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.reshape(Q16_NODE_COUNT, 6)


def _lagrange_values(nodes: np.ndarray, coordinate: float) -> np.ndarray:
    values = np.empty(nodes.size, dtype=np.float64)
    for active in range(nodes.size):
        value = 1.0
        for other in range(nodes.size):
            if other != active:
                value *= (coordinate - nodes[other]) / (nodes[active] - nodes[other])
        values[active] = value
    return values


def _strain_matrix(vector: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [vector[0], 0.5 * vector[3], 0.5 * vector[5]],
            [0.5 * vector[3], vector[1], 0.5 * vector[4]],
            [0.5 * vector[5], 0.5 * vector[4], vector[2]],
        ],
        dtype=np.float64,
    )


def _strain_vector(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            2.0 * matrix[0, 1],
            2.0 * matrix[1, 2],
            2.0 * matrix[0, 2],
        ],
        dtype=np.float64,
    )


def _covariant_to_cartesian_transform(reference_jacobian: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(reference_jacobian)
    transform = np.empty((6, 6), dtype=np.float64)
    for component in range(6):
        unit = np.zeros(6, dtype=np.float64)
        unit[component] = 1.0
        cartesian = inverse.T @ _strain_matrix(unit) @ inverse
        transform[:, component] = _strain_vector(cartesian)
    return transform


@dataclass(frozen=True, slots=True, eq=False)
class Q16EASQuadratureData:
    coordinates: np.ndarray
    reference_weights: np.ndarray
    covariant_to_cartesian: np.ndarray
    enhanced_modes: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.reference_weights.size)


def _make_quadrature(reference: Q16ReferenceElement) -> Q16EASQuadratureData:
    plane_points, plane_weights = np.polynomial.legendre.leggauss(
        Q16_EAS_IN_PLANE_QUADRATURE_ORDER
    )
    thickness_points, thickness_weights = np.polynomial.legendre.leggauss(
        Q16_EAS_THICKNESS_QUADRATURE_ORDER
    )
    coordinates = np.empty((Q16_EAS_QUADRATURE_POINT_COUNT, 3), dtype=np.float64)
    weights = np.empty(Q16_EAS_QUADRATURE_POINT_COUNT, dtype=np.float64)
    transforms = np.empty((Q16_EAS_QUADRATURE_POINT_COUNT, 6, 6), dtype=np.float64)
    enhanced = np.empty((Q16_EAS_QUADRATURE_POINT_COUNT, 6), dtype=np.float64)
    center_bases = reference.covariant_bases(reference.reference_state, 0.0, 0.0, 0.0)
    center_jacobian = np.column_stack(center_bases)
    center_determinant = float(np.linalg.det(center_jacobian))
    if not math.isfinite(center_determinant) or center_determinant <= 0.0:
        raise ValueError("reference center Jacobian is non-positive")
    center_transform = _covariant_to_cartesian_transform(center_jacobian)
    point = 0
    for zeta, wz in zip(thickness_points, thickness_weights, strict=True):
        for xi, wx in zip(plane_points, plane_weights, strict=True):
            for eta, wy in zip(plane_points, plane_weights, strict=True):
                bases = reference.covariant_bases(
                    reference.reference_state,
                    float(xi),
                    float(eta),
                    float(zeta),
                )
                jacobian = np.column_stack(bases)
                determinant = float(np.linalg.det(jacobian))
                if not math.isfinite(determinant) or determinant <= 0.0:
                    raise ValueError("reference quadrature Jacobian is non-positive")
                transform = _covariant_to_cartesian_transform(jacobian)
                covariant_mode = np.zeros(6, dtype=np.float64)
                covariant_mode[2] = float(zeta)
                coordinates[point] = [float(xi), float(eta), float(zeta)]
                weights[point] = float(wx) * float(wy) * float(wz) * determinant
                transforms[point] = transform
                # Simo--Rifai/Yamashita EAS mapping: a center transformation
                # and determinant ratio make the enhanced mode orthogonal to
                # any constant Cartesian stress on distorted geometries.
                enhanced[point] = (
                    center_determinant
                    / determinant
                    * (center_transform @ covariant_mode)
                )
                point += 1
    if point != Q16_EAS_QUADRATURE_POINT_COUNT:
        raise AssertionError("fixed Q16 ANS/EAS quadrature count drifted")
    return Q16EASQuadratureData(
        coordinates=_readonly(coordinates),
        reference_weights=_readonly(weights),
        covariant_to_cartesian=_readonly(transforms),
        enhanced_modes=_readonly(enhanced),
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16MITC16EASContinuumElement:
    """Fixed Q16 StVK element with MITC16 and condensed thickness EAS."""

    reference: Q16ReferenceElement
    young_modulus: float
    poisson_ratio: float
    density: float
    lame_lambda: float
    lame_mu: float
    constitutive: np.ndarray
    quadrature: Q16EASQuadratureData
    reference_volume: float
    enhanced_mode_volume_integral: np.ndarray
    transverse_normal_mode: str

    def __init__(
        self,
        reference: Q16ReferenceElement,
        *,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
    ) -> None:
        if type(reference) is not Q16ReferenceElement:
            raise TypeError("reference must be an exact Q16ReferenceElement")
        young = _positive_scalar("young_modulus", young_modulus)
        rho = _positive_scalar("density", density)
        if isinstance(poisson_ratio, bool) or not isinstance(
            poisson_ratio, (int, float, np.integer, np.floating)
        ):
            raise TypeError("poisson_ratio must be a real scalar")
        nu = float(poisson_ratio)
        if not math.isfinite(nu) or not (-1.0 < nu < 0.5):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5)")
        lame_lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        lame_mu = young / (2.0 * (1.0 + nu))
        constitutive = np.zeros((6, 6), dtype=np.float64)
        constitutive[:3, :3] = lame_lambda
        constitutive[0, 0] += 2.0 * lame_mu
        constitutive[1, 1] += 2.0 * lame_mu
        constitutive[2, 2] += 2.0 * lame_mu
        constitutive[3, 3] = lame_mu
        constitutive[4, 4] = lame_mu
        constitutive[5, 5] = lame_mu
        quadrature = _make_quadrature(reference)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "young_modulus", young)
        object.__setattr__(self, "poisson_ratio", nu)
        object.__setattr__(self, "density", rho)
        object.__setattr__(self, "lame_lambda", lame_lambda)
        object.__setattr__(self, "lame_mu", lame_mu)
        object.__setattr__(self, "constitutive", _readonly(constitutive))
        object.__setattr__(self, "quadrature", quadrature)
        object.__setattr__(
            self, "reference_volume", float(np.sum(quadrature.reference_weights))
        )
        object.__setattr__(
            self,
            "enhanced_mode_volume_integral",
            _readonly(
                np.sum(
                    quadrature.reference_weights[:, None] * quadrature.enhanced_modes,
                    axis=0,
                )
            ),
        )
        object.__setattr__(self, "transverse_normal_mode", "q16-ans-plus-eas-1")

    def _direct_strain_b(
        self,
        rows: np.ndarray,
        xi: float,
        eta: float,
        zeta: float,
        direction_rows: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        shape, dxi, deta = q16_shape(float(xi), float(eta))
        positions = rows[:, :3]
        directors = rows[:, 3:]
        a_xi = dxi @ positions + zeta * (dxi @ directors)
        a_eta = deta @ positions + zeta * (deta @ directors)
        a_zeta = shape @ directors
        reference_bases = self.reference.covariant_bases(
            self.reference.reference_state,
            float(xi),
            float(eta),
            float(zeta),
        )
        strain = np.asarray(
            [
                0.5 * (a_xi @ a_xi - reference_bases[0] @ reference_bases[0]),
                0.5 * (a_eta @ a_eta - reference_bases[1] @ reference_bases[1]),
                0.5 * (a_zeta @ a_zeta - reference_bases[2] @ reference_bases[2]),
                a_xi @ a_eta - reference_bases[0] @ reference_bases[1],
                a_eta @ a_zeta - reference_bases[1] @ reference_bases[2],
                a_xi @ a_zeta - reference_bases[0] @ reference_bases[2],
            ],
            dtype=np.float64,
        )
        b_matrix = np.zeros((6, Q16_DOF_PER_ELEMENT), dtype=np.float64)
        delta_b = np.zeros_like(b_matrix) if direction_rows is not None else None
        if direction_rows is not None:
            delta_positions = direction_rows[:, :3]
            delta_directors = direction_rows[:, 3:]
            delta_xi = dxi @ delta_positions + zeta * (dxi @ delta_directors)
            delta_eta = deta @ delta_positions + zeta * (deta @ delta_directors)
            delta_zeta = shape @ delta_directors
        for node in range(Q16_NODE_COUNT):
            for coordinate in range(3):
                position_dof = node * 6 + coordinate
                director_dof = position_dof + 3
                b_matrix[0, position_dof] = dxi[node] * a_xi[coordinate]
                b_matrix[1, position_dof] = deta[node] * a_eta[coordinate]
                b_matrix[3, position_dof] = (
                    dxi[node] * a_eta[coordinate] + deta[node] * a_xi[coordinate]
                )
                b_matrix[4, position_dof] = deta[node] * a_zeta[coordinate]
                b_matrix[5, position_dof] = dxi[node] * a_zeta[coordinate]

                zeta_dxi = zeta * dxi[node]
                zeta_deta = zeta * deta[node]
                b_matrix[0, director_dof] = zeta_dxi * a_xi[coordinate]
                b_matrix[1, director_dof] = zeta_deta * a_eta[coordinate]
                b_matrix[2, director_dof] = shape[node] * a_zeta[coordinate]
                b_matrix[3, director_dof] = (
                    zeta_dxi * a_eta[coordinate] + zeta_deta * a_xi[coordinate]
                )
                b_matrix[4, director_dof] = (
                    zeta_deta * a_zeta[coordinate] + shape[node] * a_eta[coordinate]
                )
                b_matrix[5, director_dof] = (
                    zeta_dxi * a_zeta[coordinate] + shape[node] * a_xi[coordinate]
                )
                if delta_b is not None:
                    delta_b[0, position_dof] = dxi[node] * delta_xi[coordinate]
                    delta_b[1, position_dof] = deta[node] * delta_eta[coordinate]
                    delta_b[3, position_dof] = (
                        dxi[node] * delta_eta[coordinate]
                        + deta[node] * delta_xi[coordinate]
                    )
                    delta_b[4, position_dof] = deta[node] * delta_zeta[coordinate]
                    delta_b[5, position_dof] = dxi[node] * delta_zeta[coordinate]
                    delta_b[0, director_dof] = zeta_dxi * delta_xi[coordinate]
                    delta_b[1, director_dof] = zeta_deta * delta_eta[coordinate]
                    delta_b[2, director_dof] = shape[node] * delta_zeta[coordinate]
                    delta_b[3, director_dof] = (
                        zeta_dxi * delta_eta[coordinate]
                        + zeta_deta * delta_xi[coordinate]
                    )
                    delta_b[4, director_dof] = (
                        zeta_deta * delta_zeta[coordinate]
                        + shape[node] * delta_eta[coordinate]
                    )
                    delta_b[5, director_dof] = (
                        zeta_dxi * delta_zeta[coordinate]
                        + shape[node] * delta_xi[coordinate]
                    )
        return strain, b_matrix, delta_b

    @staticmethod
    def _interpolate(
        samples: np.ndarray, eta_weights: np.ndarray, xi_weights: np.ndarray
    ) -> np.ndarray:
        return np.tensordot(
            eta_weights,
            np.tensordot(samples, xi_weights, axes=(1, 0)),
            axes=(0, 0),
        )

    def _projected_fields(
        self,
        state: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        rows = _exact_state("state", state)
        direction_rows = (
            None if direction is None else _exact_state("direction", direction)
        )
        for xi, eta, zeta in self.quadrature.coordinates:
            current_bases = self.reference.covariant_bases(
                state, float(xi), float(eta), float(zeta)
            )
            determinant = float(np.linalg.det(np.column_stack(current_bases)))
            scale = max(
                float(np.linalg.norm(current_bases[0]))
                * float(np.linalg.norm(current_bases[1]))
                * float(np.linalg.norm(current_bases[2])),
                np.finfo(np.float64).tiny,
            )
            if (
                not math.isfinite(determinant)
                or determinant <= 512.0 * np.finfo(np.float64).eps * scale
            ):
                raise FloatingPointError(
                    "Q16 current geometry is orientation reversing or singular"
                )
        point_count = self.quadrature.point_count
        strains = np.empty((point_count, 6), dtype=np.float64)
        b_matrices = np.empty((point_count, 6, Q16_DOF_PER_ELEMENT), dtype=np.float64)
        delta_b_matrices = (
            np.empty_like(b_matrices) if direction_rows is not None else None
        )

        shear_13 = np.empty((4, 3), dtype=np.float64)
        shear_13_b = np.empty((4, 3, 6, Q16_DOF_PER_ELEMENT), dtype=np.float64)
        shear_23 = np.empty((3, 4), dtype=np.float64)
        shear_23_b = np.empty((3, 4, 6, Q16_DOF_PER_ELEMENT), dtype=np.float64)
        shear_13_db = np.empty_like(shear_13_b) if direction_rows is not None else None
        shear_23_db = np.empty_like(shear_23_b) if direction_rows is not None else None
        for eta_index, eta_tie in enumerate(MITC16_TYING_POINTS_4):
            for xi_index, xi_tie in enumerate(MITC16_TYING_POINTS_3):
                value, b_matrix, delta_b = self._direct_strain_b(
                    rows,
                    float(xi_tie),
                    float(eta_tie),
                    0.0,
                    direction_rows,
                )
                shear_13[eta_index, xi_index] = value[5]
                shear_13_b[eta_index, xi_index] = b_matrix
                if shear_13_db is not None and delta_b is not None:
                    shear_13_db[eta_index, xi_index] = delta_b
        for eta_index, eta_tie in enumerate(MITC16_TYING_POINTS_3):
            for xi_index, xi_tie in enumerate(MITC16_TYING_POINTS_4):
                value, b_matrix, delta_b = self._direct_strain_b(
                    rows,
                    float(xi_tie),
                    float(eta_tie),
                    0.0,
                    direction_rows,
                )
                shear_23[eta_index, xi_index] = value[4]
                shear_23_b[eta_index, xi_index] = b_matrix
                if shear_23_db is not None and delta_b is not None:
                    shear_23_db[eta_index, xi_index] = delta_b

        normal_33 = np.empty(Q16_NODE_COUNT, dtype=np.float64)
        normal_33_b = np.empty((Q16_NODE_COUNT, 6, Q16_DOF_PER_ELEMENT))
        normal_33_db = (
            np.empty_like(normal_33_b) if direction_rows is not None else None
        )
        for node, (xi_node, eta_node) in enumerate(Q16_PARAMETRIC_NODES):
            value, b_matrix, delta_b = self._direct_strain_b(
                rows,
                float(xi_node),
                float(eta_node),
                0.0,
                direction_rows,
            )
            normal_33[node] = value[2]
            normal_33_b[node] = b_matrix
            if normal_33_db is not None and delta_b is not None:
                normal_33_db[node] = delta_b

        thickness_points = np.polynomial.legendre.leggauss(
            Q16_EAS_THICKNESS_QUADRATURE_ORDER
        )[0]
        plane_points = np.polynomial.legendre.leggauss(
            Q16_EAS_IN_PLANE_QUADRATURE_ORDER
        )[0]
        point = 0
        for zeta in thickness_points:
            normal_11 = np.empty((4, 3), dtype=np.float64)
            normal_11_b = np.empty_like(shear_13_b)
            normal_22 = np.empty((3, 4), dtype=np.float64)
            normal_22_b = np.empty_like(shear_23_b)
            shear_12 = np.empty((3, 3), dtype=np.float64)
            shear_12_b = np.empty((3, 3, 6, Q16_DOF_PER_ELEMENT))
            normal_11_db = (
                np.empty_like(normal_11_b) if direction_rows is not None else None
            )
            normal_22_db = (
                np.empty_like(normal_22_b) if direction_rows is not None else None
            )
            shear_12_db = (
                np.empty_like(shear_12_b) if direction_rows is not None else None
            )
            for eta_index, eta_tie in enumerate(MITC16_TYING_POINTS_4):
                for xi_index, xi_tie in enumerate(MITC16_TYING_POINTS_3):
                    value, b_matrix, delta_b = self._direct_strain_b(
                        rows,
                        float(xi_tie),
                        float(eta_tie),
                        float(zeta),
                        direction_rows,
                    )
                    normal_11[eta_index, xi_index] = value[0]
                    normal_11_b[eta_index, xi_index] = b_matrix
                    if normal_11_db is not None and delta_b is not None:
                        normal_11_db[eta_index, xi_index] = delta_b
            for eta_index, eta_tie in enumerate(MITC16_TYING_POINTS_3):
                for xi_index, xi_tie in enumerate(MITC16_TYING_POINTS_4):
                    value, b_matrix, delta_b = self._direct_strain_b(
                        rows,
                        float(xi_tie),
                        float(eta_tie),
                        float(zeta),
                        direction_rows,
                    )
                    normal_22[eta_index, xi_index] = value[1]
                    normal_22_b[eta_index, xi_index] = b_matrix
                    if normal_22_db is not None and delta_b is not None:
                        normal_22_db[eta_index, xi_index] = delta_b
            for eta_index, eta_tie in enumerate(MITC16_TYING_POINTS_3):
                for xi_index, xi_tie in enumerate(MITC16_TYING_POINTS_3):
                    value, b_matrix, delta_b = self._direct_strain_b(
                        rows,
                        float(xi_tie),
                        float(eta_tie),
                        float(zeta),
                        direction_rows,
                    )
                    shear_12[eta_index, xi_index] = value[3]
                    shear_12_b[eta_index, xi_index] = b_matrix
                    if shear_12_db is not None and delta_b is not None:
                        shear_12_db[eta_index, xi_index] = delta_b

            for xi in plane_points:
                xi_three = _lagrange_values(MITC16_TYING_POINTS_3, float(xi))
                xi_four = _lagrange_values(MITC16_TYING_POINTS_4, float(xi))
                for eta in plane_points:
                    eta_three = _lagrange_values(MITC16_TYING_POINTS_3, float(eta))
                    eta_four = _lagrange_values(MITC16_TYING_POINTS_4, float(eta))
                    shape, _, _ = q16_shape(float(xi), float(eta))
                    covariant = np.asarray(
                        [
                            self._interpolate(normal_11, eta_four, xi_three),
                            self._interpolate(normal_22, eta_three, xi_four),
                            shape @ normal_33,
                            self._interpolate(shear_12, eta_three, xi_three),
                            self._interpolate(shear_23, eta_three, xi_four),
                            self._interpolate(shear_13, eta_four, xi_three),
                        ],
                        dtype=np.float64,
                    )
                    covariant_b = np.stack(
                        [
                            self._interpolate(normal_11_b, eta_four, xi_three)[0],
                            self._interpolate(normal_22_b, eta_three, xi_four)[1],
                            np.tensordot(shape, normal_33_b[:, 2], axes=(0, 0)),
                            self._interpolate(shear_12_b, eta_three, xi_three)[3],
                            self._interpolate(shear_23_b, eta_three, xi_four)[4],
                            self._interpolate(shear_13_b, eta_four, xi_three)[5],
                        ]
                    )
                    transform = self.quadrature.covariant_to_cartesian[point]
                    strains[point] = transform @ covariant
                    b_matrices[point] = transform @ covariant_b
                    if delta_b_matrices is not None:
                        if (
                            normal_11_db is None
                            or normal_22_db is None
                            or shear_12_db is None
                            or shear_13_db is None
                            or shear_23_db is None
                            or normal_33_db is None
                        ):
                            raise AssertionError(
                                "directional strain data is incomplete"
                            )
                        covariant_db = np.stack(
                            [
                                self._interpolate(normal_11_db, eta_four, xi_three)[0],
                                self._interpolate(normal_22_db, eta_three, xi_four)[1],
                                np.tensordot(shape, normal_33_db[:, 2], axes=(0, 0)),
                                self._interpolate(shear_12_db, eta_three, xi_three)[3],
                                self._interpolate(shear_23_db, eta_three, xi_four)[4],
                                self._interpolate(shear_13_db, eta_four, xi_three)[5],
                            ]
                        )
                        delta_b_matrices[point] = transform @ covariant_db
                    point += 1
        if point != point_count:
            raise AssertionError("projected Q16 field count drifted")
        if not bool(np.isfinite(strains).all()) or not bool(
            np.isfinite(b_matrices).all()
        ):
            raise FloatingPointError("projected Q16 fields became non-finite")
        return strains, b_matrices, delta_b_matrices

    def _eas_stiffness(self) -> float:
        value = 0.0
        for point in range(self.quadrature.point_count):
            mode = self.quadrature.enhanced_modes[point]
            value += self.quadrature.reference_weights[point] * float(
                mode @ self.constitutive @ mode
            )
        if not math.isfinite(value) or value <= 0.0:
            raise FloatingPointError("Q16 EAS local stiffness is non-positive")
        return float(value)

    def solve_enhanced_parameter(self, state: np.ndarray) -> float:
        strains, _, _ = self._projected_fields(state)
        residual = 0.0
        for point in range(self.quadrature.point_count):
            residual += self.quadrature.reference_weights[point] * float(
                self.quadrature.enhanced_modes[point]
                @ self.constitutive
                @ strains[point]
            )
        alpha = -residual / self._eas_stiffness()
        if not math.isfinite(alpha):
            raise FloatingPointError("Q16 EAS parameter became non-finite")
        return float(alpha)

    def enhanced_stationarity_residual(
        self, state: np.ndarray, enhanced_parameter: float
    ) -> float:
        alpha = _finite_scalar("enhanced_parameter", enhanced_parameter)
        strains, _, _ = self._projected_fields(state)
        residual = 0.0
        for point in range(self.quadrature.point_count):
            mode = self.quadrature.enhanced_modes[point]
            stress = self.constitutive @ (strains[point] + mode * alpha)
            residual += self.quadrature.reference_weights[point] * float(mode @ stress)
        if not math.isfinite(residual):
            raise FloatingPointError("Q16 EAS residual became non-finite")
        return float(residual)

    def enhanced_stationarity_relative_residual(
        self, state: np.ndarray, enhanced_parameter: float
    ) -> float:
        """Return the EAS residual normalized by its absolute term ledger."""

        alpha = _finite_scalar("enhanced_parameter", enhanced_parameter)
        strains, _, _ = self._projected_fields(state)
        residual = 0.0
        scale = 0.0
        for point in range(self.quadrature.point_count):
            mode = self.quadrature.enhanced_modes[point]
            stress = self.constitutive @ (strains[point] + mode * alpha)
            term = self.quadrature.reference_weights[point] * float(mode @ stress)
            residual += term
            scale += abs(term)
        relative = abs(residual) / max(scale, np.finfo(np.float64).tiny)
        if not math.isfinite(relative):
            raise FloatingPointError("Q16 EAS relative residual became non-finite")
        return float(relative)

    def strain_energy_at_parameter(
        self, state: np.ndarray, enhanced_parameter: float
    ) -> float:
        alpha = _finite_scalar("enhanced_parameter", enhanced_parameter)
        strains, _, _ = self._projected_fields(state)
        energy = 0.0
        for point in range(self.quadrature.point_count):
            strain = strains[point] + self.quadrature.enhanced_modes[point] * alpha
            energy += (
                0.5
                * self.quadrature.reference_weights[point]
                * float(strain @ self.constitutive @ strain)
            )
        if not math.isfinite(energy):
            raise FloatingPointError("Q16 ANS/EAS energy became non-finite")
        return float(energy)

    def strain_energy(self, state: np.ndarray) -> float:
        alpha = self.solve_enhanced_parameter(state)
        return self.strain_energy_at_parameter(state, alpha)

    def internal_force(self, state: np.ndarray) -> np.ndarray:
        alpha = self.solve_enhanced_parameter(state)
        strains, b_matrices, _ = self._projected_fields(state)
        force = np.zeros(Q16_DOF_PER_ELEMENT, dtype=np.float64)
        for point in range(self.quadrature.point_count):
            strain = strains[point] + self.quadrature.enhanced_modes[point] * alpha
            stress = self.constitutive @ strain
            force += self.quadrature.reference_weights[point] * (
                b_matrices[point].T @ stress
            )
        if not bool(np.isfinite(force).all()):
            raise FloatingPointError("Q16 ANS/EAS force became non-finite")
        return _readonly(force)

    def tangent_action(self, state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        alpha = self.solve_enhanced_parameter(state)
        strains, b_matrices, delta_b_matrices = self._projected_fields(state, direction)
        direction_vector = _exact_state("direction", direction).reshape(
            Q16_DOF_PER_ELEMENT
        )
        if delta_b_matrices is None:
            raise AssertionError("directional B matrices were not constructed")
        alpha_rhs = 0.0
        for point in range(self.quadrature.point_count):
            delta_compatible = b_matrices[point] @ direction_vector
            alpha_rhs += self.quadrature.reference_weights[point] * float(
                self.quadrature.enhanced_modes[point]
                @ self.constitutive
                @ delta_compatible
            )
        delta_alpha = -alpha_rhs / self._eas_stiffness()
        action = np.zeros(Q16_DOF_PER_ELEMENT, dtype=np.float64)
        for point in range(self.quadrature.point_count):
            mode = self.quadrature.enhanced_modes[point]
            strain = strains[point] + mode * alpha
            stress = self.constitutive @ strain
            delta_strain = b_matrices[point] @ direction_vector + mode * delta_alpha
            delta_stress = self.constitutive @ delta_strain
            action += self.quadrature.reference_weights[point] * (
                delta_b_matrices[point].T @ stress + b_matrices[point].T @ delta_stress
            )
        if not bool(np.isfinite(action).all()):
            raise FloatingPointError("Q16 ANS/EAS tangent action became non-finite")
        return _readonly(action)


__all__ = [
    "Q16EASQuadratureData",
    "Q16MITC16EASContinuumElement",
    "Q16_EAS_IN_PLANE_QUADRATURE_ORDER",
    "Q16_EAS_QUADRATURE_POINT_COUNT",
    "Q16_EAS_THICKNESS_QUADRATURE_ORDER",
]
