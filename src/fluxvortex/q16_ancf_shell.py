"""Fixed Q16 shear/thickness-deformable ANCF macro-shell reference operators.

This module freezes the mathematical state used by the CUDA implementation:
sixteen tensor-product cubic Gauss--Lobatto nodes, with position and physical
half-thickness director coordinates at every node.  It intentionally contains
only the fixed production element; it is not a runtime-order element factory.

The NumPy operations here are an independent small-problem oracle.  They are
not a CPU fallback for the production FSI time loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

Q16_NODE_COUNT = 16
Q16_DOF_PER_NODE = 6
Q16_DOF_PER_ELEMENT = Q16_NODE_COUNT * Q16_DOF_PER_NODE
_FLOAT64 = np.dtype(np.float64)
_DOMAIN = "flux-v5m-q16-ancf-reference-v1"


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


Q16_PARAMETRIC_NODES_1D = _readonly_float64(
    np.array(
        [-1.0, -1.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 1.0],
        dtype=np.float64,
    )
)
Q16_PARAMETRIC_NODES = _readonly_float64(
    np.array(
        [
            (xi, eta)
            for eta in Q16_PARAMETRIC_NODES_1D
            for xi in Q16_PARAMETRIC_NODES_1D
        ],
        dtype=np.float64,
    )
)


def _coordinate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        raise TypeError(f"{name} must be a real floating-point scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < -1.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [-1, 1]")
    return result


def _lagrange_1d(coordinate: float) -> tuple[np.ndarray, np.ndarray]:
    nodes = Q16_PARAMETRIC_NODES_1D
    values = np.empty(4, dtype=np.float64)
    derivatives = np.empty(4, dtype=np.float64)
    for active in range(4):
        denominator = 1.0
        numerator = 1.0
        for other in range(4):
            if other == active:
                continue
            denominator *= nodes[active] - nodes[other]
            numerator *= coordinate - nodes[other]
        values[active] = numerator / denominator

        derivative = 0.0
        for omitted in range(4):
            if omitted == active:
                continue
            term = 1.0
            for other in range(4):
                if other == active or other == omitted:
                    continue
                term *= coordinate - nodes[other]
            derivative += term
        derivatives[active] = derivative / denominator
    return values, derivatives


def q16_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed Q16 shape values and parametric derivatives.

    Node order is row-major in parameter space: ``xi`` varies fastest inside
    each constant-``eta`` row.
    """

    xi_value = _coordinate("xi", xi)
    eta_value = _coordinate("eta", eta)
    lx, dlx = _lagrange_1d(xi_value)
    ly, dly = _lagrange_1d(eta_value)
    shape = np.empty(Q16_NODE_COUNT, dtype=np.float64)
    dxi = np.empty(Q16_NODE_COUNT, dtype=np.float64)
    deta = np.empty(Q16_NODE_COUNT, dtype=np.float64)
    for eta_index in range(4):
        for xi_index in range(4):
            node = 4 * eta_index + xi_index
            shape[node] = lx[xi_index] * ly[eta_index]
            dxi[node] = dlx[xi_index] * ly[eta_index]
            deta[node] = lx[xi_index] * dly[eta_index]
    return (
        _readonly_float64(shape),
        _readonly_float64(dxi),
        _readonly_float64(deta),
    )


def _exact_float64_array(
    name: str, value: np.ndarray, expected_shape: tuple[int, ...]
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != _FLOAT64:
        raise TypeError(f"{name} must use float64")
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return _readonly_float64(value)


def q16_interpolate(nodal_values: np.ndarray, xi: float, eta: float):
    """Interpolate scalar or vector nodal values with the fixed Q16 basis."""

    if type(nodal_values) is not np.ndarray:
        raise TypeError("nodal_values must be an exact numpy.ndarray")
    if nodal_values.dtype != _FLOAT64:
        raise TypeError("nodal_values must use float64")
    if nodal_values.ndim < 1 or nodal_values.shape[0] != Q16_NODE_COUNT:
        raise ValueError("nodal_values leading dimension must be sixteen")
    if not nodal_values.flags.c_contiguous:
        raise ValueError("nodal_values must be C-contiguous")
    if not bool(np.isfinite(nodal_values).all()):
        raise ValueError("nodal_values must contain only finite values")
    shape, _, _ = q16_shape(xi, eta)
    output = np.tensordot(shape, nodal_values, axes=(0, 0))
    if np.ndim(output) == 0:
        return float(output)
    return _readonly_float64(np.asarray(output, dtype=np.float64))


def _state_payload(rows: np.ndarray) -> bytes:
    header = json.dumps(
        {
            "domain": _DOMAIN,
            "dtype": "float64",
            "shape": [Q16_NODE_COUNT, Q16_DOF_PER_NODE],
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return header + b"\0" + rows.tobytes(order="C")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16ReferenceElement:
    """Immutable reference geometry for the fixed 96-DOF Q16 element.

    Per-node rows are ``[r_x, r_y, r_z, g_x, g_y, g_z]``.  ``g`` is the
    physical half-thickness vector.  The through-thickness coordinate therefore
    spans ``x = r + zeta*g`` for ``zeta in [-1, 1]``.
    """

    reference_rows: np.ndarray
    reference_state: np.ndarray
    reference_sha256: str

    def __init__(self, reference_rows: np.ndarray) -> None:
        rows = _exact_float64_array(
            "reference_rows",
            reference_rows,
            (Q16_NODE_COUNT, Q16_DOF_PER_NODE),
        )
        state = rows.reshape(Q16_DOF_PER_ELEMENT)
        object.__setattr__(self, "reference_rows", rows)
        object.__setattr__(self, "reference_state", state)
        object.__setattr__(
            self,
            "reference_sha256",
            hashlib.sha256(_state_payload(rows)).hexdigest(),
        )
        self._assert_positive_reference_jacobian()

    @staticmethod
    def _validated_state(state: np.ndarray) -> np.ndarray:
        return _exact_float64_array("state", state, (Q16_DOF_PER_ELEMENT,)).reshape(
            Q16_NODE_COUNT, Q16_DOF_PER_NODE
        )

    @staticmethod
    def _bases_from_rows(
        rows: np.ndarray, xi: float, eta: float, zeta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zeta_value = _coordinate("zeta", zeta)
        shape, dxi, deta = q16_shape(xi, eta)
        positions = rows[:, :3]
        directors = rows[:, 3:]
        a_xi = dxi @ positions + zeta_value * (dxi @ directors)
        a_eta = deta @ positions + zeta_value * (deta @ directors)
        a_zeta = shape @ directors
        return (
            np.asarray(a_xi, dtype=np.float64),
            np.asarray(a_eta, dtype=np.float64),
            np.asarray(a_zeta, dtype=np.float64),
        )

    def _assert_positive_reference_jacobian(self) -> None:
        quadrature, _ = np.polynomial.legendre.leggauss(6)
        for xi in quadrature:
            for eta in quadrature:
                for zeta in (-1.0, 0.0, 1.0):
                    bases = self._bases_from_rows(
                        self.reference_rows, float(xi), float(eta), zeta
                    )
                    determinant = float(np.linalg.det(np.column_stack(bases)))
                    scale = max(
                        float(np.linalg.norm(bases[0]))
                        * float(np.linalg.norm(bases[1]))
                        * float(np.linalg.norm(bases[2])),
                        np.finfo(np.float64).tiny,
                    )
                    if (
                        not math.isfinite(determinant)
                        or determinant <= 512.0 * np.finfo(np.float64).eps * scale
                    ):
                        raise ValueError(
                            "reference geometry has a non-positive or singular Jacobian"
                        )

    def position(
        self, state: np.ndarray, xi: float, eta: float, zeta: float
    ) -> np.ndarray:
        rows = self._validated_state(state)
        zeta_value = _coordinate("zeta", zeta)
        shape, _, _ = q16_shape(xi, eta)
        output = shape @ rows[:, :3] + zeta_value * (shape @ rows[:, 3:])
        return _readonly_float64(np.asarray(output, dtype=np.float64))

    def covariant_bases(
        self, state: np.ndarray, xi: float, eta: float, zeta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self._validated_state(state)
        return tuple(
            _readonly_float64(np.asarray(value, dtype=np.float64))
            for value in self._bases_from_rows(rows, xi, eta, zeta)
        )

    def green_lagrange_strain(
        self, state: np.ndarray, xi: float, eta: float, zeta: float
    ) -> np.ndarray:
        """Return ``[E11,E22,E33,2E12,2E23,2E13]`` in covariant form."""

        current = self.covariant_bases(state, xi, eta, zeta)
        reference = self._bases_from_rows(self.reference_rows, xi, eta, zeta)
        metric = np.empty(6, dtype=np.float64)
        metric[0] = 0.5 * (
            np.dot(current[0], current[0]) - np.dot(reference[0], reference[0])
        )
        metric[1] = 0.5 * (
            np.dot(current[1], current[1]) - np.dot(reference[1], reference[1])
        )
        metric[2] = 0.5 * (
            np.dot(current[2], current[2]) - np.dot(reference[2], reference[2])
        )
        metric[3] = np.dot(current[0], current[1]) - np.dot(reference[0], reference[1])
        metric[4] = np.dot(current[1], current[2]) - np.dot(reference[1], reference[2])
        metric[5] = np.dot(current[0], current[2]) - np.dot(reference[0], reference[2])
        return _readonly_float64(metric)

    def consistent_mass_matrix(
        self,
        *,
        density: float,
        in_plane_order: int = 6,
        thickness_order: int = 3,
    ) -> np.ndarray:
        """Integrate the consistent 96x96 reference mass matrix.

        This is a small-problem oracle; the production CUDA path will apply mass
        matrix-free and will not assemble this dense matrix.
        """

        if isinstance(density, bool) or not isinstance(
            density, (int, float, np.integer, np.floating)
        ):
            raise TypeError("density must be a real scalar")
        rho = float(density)
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("density must be finite and positive")
        if type(in_plane_order) is not int or in_plane_order < 4:
            raise ValueError("in_plane_order must be an exact int at least four")
        if type(thickness_order) is not int or thickness_order < 2:
            raise ValueError("thickness_order must be an exact int at least two")

        plane_points, plane_weights = np.polynomial.legendre.leggauss(in_plane_order)
        thick_points, thick_weights = np.polynomial.legendre.leggauss(thickness_order)
        mass = np.zeros((Q16_DOF_PER_ELEMENT, Q16_DOF_PER_ELEMENT), dtype=np.float64)
        interpolation = np.zeros((3, Q16_DOF_PER_ELEMENT), dtype=np.float64)
        for xi, wx in zip(plane_points, plane_weights, strict=True):
            for eta, wy in zip(plane_points, plane_weights, strict=True):
                shape, _, _ = q16_shape(float(xi), float(eta))
                for zeta, wz in zip(thick_points, thick_weights, strict=True):
                    interpolation.fill(0.0)
                    for node in range(Q16_NODE_COUNT):
                        for component in range(3):
                            interpolation[component, node * 6 + component] = shape[node]
                            interpolation[component, node * 6 + 3 + component] = (
                                float(zeta) * shape[node]
                            )
                    bases = self._bases_from_rows(
                        self.reference_rows, float(xi), float(eta), float(zeta)
                    )
                    jacobian = float(np.linalg.det(np.column_stack(bases)))
                    if not math.isfinite(jacobian) or jacobian <= 0.0:
                        raise ValueError("reference Jacobian became non-positive")
                    weight = rho * float(wx) * float(wy) * float(wz) * jacobian
                    mass += weight * (interpolation.T @ interpolation)
        return _readonly_float64(mass)


__all__ = [
    "Q16_DOF_PER_ELEMENT",
    "Q16_DOF_PER_NODE",
    "Q16_NODE_COUNT",
    "Q16_PARAMETRIC_NODES",
    "Q16_PARAMETRIC_NODES_1D",
    "Q16ReferenceElement",
    "q16_interpolate",
    "q16_shape",
]
