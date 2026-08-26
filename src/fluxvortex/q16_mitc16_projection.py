"""MITC16 assumed-covariant-strain projection for the fixed Q16 shell.

The tying layout follows Bucalem and Bathe's MITC16 general shell element.
This module is deliberately a NumPy formulation oracle.  It projects the two
in-plane normal, in-plane shear and transverse shear components while leaving
the transverse normal component compatible.  The latter must be replaced only
after the separate ANS/EAS thickness formulation and condensation gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .q16_ancf_shell import Q16ReferenceElement

_FLOAT64 = np.dtype(np.float64)


def _readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


MITC16_TYING_POINTS_3 = _readonly(
    np.asarray(
        [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)],
        dtype=np.float64,
    )
)
MITC16_TYING_POINTS_4 = _readonly(
    np.asarray(np.polynomial.legendre.leggauss(4)[0], dtype=np.float64)
)


def _coordinate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        raise TypeError(f"{name} must be a floating-point scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < -1.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [-1, 1]")
    return result


def _lagrange_values(nodes: np.ndarray, coordinate: float) -> np.ndarray:
    values = np.empty(nodes.size, dtype=np.float64)
    for active in range(nodes.size):
        value = 1.0
        for other in range(nodes.size):
            if other != active:
                value *= (coordinate - nodes[other]) / (nodes[active] - nodes[other])
        values[active] = value
    return values


def _tensor_interpolate(
    samples: np.ndarray, eta_values: np.ndarray, xi_values: np.ndarray
) -> float:
    return float(eta_values @ samples @ xi_values)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16MITC16Projector:
    """Independent fixed-order MITC16 covariant strain projector."""

    reference: Q16ReferenceElement
    transverse_normal_mode: str

    def __init__(self, reference: Q16ReferenceElement) -> None:
        if type(reference) is not Q16ReferenceElement:
            raise TypeError("reference must be an exact Q16ReferenceElement")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "transverse_normal_mode", "compatible-not-ans-eas")

    def covariant_strain(
        self,
        state: np.ndarray,
        xi: float,
        eta: float,
        zeta: float,
    ) -> np.ndarray:
        """Return `[E11,E22,E33,2E12,2E23,2E13]` after MITC16 projection.

        In-plane strains are tied at the requested thickness coordinate.
        Transverse shear strains are tied on the middle surface and are
        therefore independent of the requested thickness coordinate.
        """

        xi_value = _coordinate("xi", xi)
        eta_value = _coordinate("eta", eta)
        zeta_value = _coordinate("zeta", zeta)

        # Validate the complete state before constructing any tying samples.
        direct = self.reference.green_lagrange_strain(
            state, xi_value, eta_value, zeta_value
        )
        xi_three = _lagrange_values(MITC16_TYING_POINTS_3, xi_value)
        eta_three = _lagrange_values(MITC16_TYING_POINTS_3, eta_value)
        xi_four = _lagrange_values(MITC16_TYING_POINTS_4, xi_value)
        eta_four = _lagrange_values(MITC16_TYING_POINTS_4, eta_value)

        e11 = np.empty((4, 3), dtype=np.float64)
        e13 = np.empty((4, 3), dtype=np.float64)
        for eta_index, eta_tie in enumerate(MITC16_TYING_POINTS_4):
            for xi_index, xi_tie in enumerate(MITC16_TYING_POINTS_3):
                in_plane = self.reference.green_lagrange_strain(
                    state, float(xi_tie), float(eta_tie), zeta_value
                )
                middle = self.reference.green_lagrange_strain(
                    state, float(xi_tie), float(eta_tie), 0.0
                )
                e11[eta_index, xi_index] = in_plane[0]
                e13[eta_index, xi_index] = middle[5]

        e22 = np.empty((3, 4), dtype=np.float64)
        e23 = np.empty((3, 4), dtype=np.float64)
        for eta_index, eta_tie in enumerate(MITC16_TYING_POINTS_3):
            for xi_index, xi_tie in enumerate(MITC16_TYING_POINTS_4):
                in_plane = self.reference.green_lagrange_strain(
                    state, float(xi_tie), float(eta_tie), zeta_value
                )
                middle = self.reference.green_lagrange_strain(
                    state, float(xi_tie), float(eta_tie), 0.0
                )
                e22[eta_index, xi_index] = in_plane[1]
                e23[eta_index, xi_index] = middle[4]

        e12 = np.empty((3, 3), dtype=np.float64)
        for eta_index, eta_tie in enumerate(MITC16_TYING_POINTS_3):
            for xi_index, xi_tie in enumerate(MITC16_TYING_POINTS_3):
                in_plane = self.reference.green_lagrange_strain(
                    state, float(xi_tie), float(eta_tie), zeta_value
                )
                e12[eta_index, xi_index] = in_plane[3]

        projected = np.asarray(
            [
                _tensor_interpolate(e11, eta_four, xi_three),
                _tensor_interpolate(e22, eta_three, xi_four),
                float(direct[2]),
                _tensor_interpolate(e12, eta_three, xi_three),
                _tensor_interpolate(e23, eta_three, xi_four),
                _tensor_interpolate(e13, eta_four, xi_three),
            ],
            dtype=np.float64,
        )
        if not bool(np.isfinite(projected).all()):
            raise FloatingPointError("MITC16 projected strain became non-finite")
        return _readonly(projected)


__all__ = [
    "MITC16_TYING_POINTS_3",
    "MITC16_TYING_POINTS_4",
    "Q16MITC16Projector",
]
