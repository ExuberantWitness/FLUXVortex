"""Finite-blunt-base topology and identifiability oracle.

This S2d oracle asks only whether one material sheet origin can represent
both separation corners of an open NACA trailing-edge base.  It does not
choose a Kutta closure or calculate pressure, circulation, sheet strength,
base drag, confluence motion, or production loads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BluntBaseTopologyError(ValueError):
    """Invalid geometry for the blunt-base topology canonical."""


@dataclass(frozen=True)
class BluntBaseTopology:
    base_fraction: float
    thickness_coefficient: float
    upper_corner: np.ndarray
    lower_corner: np.ndarray
    upper_tangent: np.ndarray
    lower_tangent: np.ndarray
    half_base_thickness: float
    base_thickness: float
    geometry_identity_residual: float
    optimal_single_origin: np.ndarray
    optimal_single_origin_residual: float
    normalized_single_origin_residual: float
    two_origin_attachment_residual: float
    tangent_gap_angle_rad: float
    optimal_B2_direction: np.ndarray
    optimal_B2_direction_mismatch_rad: float
    single_junction_topologically_admissible: bool


def _unit(name: str, vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise BluntBaseTopologyError(
            f"{name} must be finite and nonzero"
        )
    return vector / norm


def naca4_blunt_base_topology(
    *,
    base_fraction: float,
    maximum_camber: float = 0.02,
    camber_location: float = 0.40,
    thickness_ratio: float = 0.06,
    chord: float = 1.0,
) -> BluntBaseTopology:
    """Evaluate the preregistered open-to-closed NACA trailing edge."""

    fraction = float(base_fraction)
    m = float(maximum_camber)
    p = float(camber_location)
    thickness = float(thickness_ratio)
    section_chord = float(chord)
    values = np.array(
        [fraction, m, p, thickness, section_chord], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise BluntBaseTopologyError("all inputs must be finite")
    if not 0.0 <= fraction <= 1.0:
        raise BluntBaseTopologyError(
            "base_fraction must lie in [0,1]"
        )
    if m < 0.0 or not 0.0 < p < 1.0:
        raise BluntBaseTopologyError(
            "invalid NACA four-digit camber parameters"
        )
    if not 0.0 < thickness < 1.0 or section_chord <= 0.0:
        raise BluntBaseTopologyError(
            "thickness_ratio and chord must be positive"
        )

    # Standard NACA open/closed fourth-order coefficients.  The endpoint
    # continuation is a geometry identity, not an aerodynamic parameter.
    coefficient = -0.1036 + fraction * (-0.1015 + 0.1036)
    x = 1.0
    mean_ordinate = (
        m / (1.0 - p) ** 2
        * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2)
    )
    mean_slope = 2.0 * m / (1.0 - p) ** 2 * (p - x)
    mean_curvature_parameter = -2.0 * m / (1.0 - p) ** 2
    theta = float(np.arctan(mean_slope))
    theta_derivative = float(
        mean_curvature_parameter / (1.0 + mean_slope**2)
    )

    half_base = 5.0 * thickness * (
        0.2969
        - 0.1260
        - 0.3516
        + 0.2843
        + coefficient
    )
    if fraction == 0.0:
        # The standard closed coefficient is defined to close the endpoint;
        # remove only its floating-point polynomial cancellation residue.
        half_base = 0.0
    half_base_derivative = 5.0 * thickness * (
        0.5 * 0.2969
        - 0.1260
        - 2.0 * 0.3516
        + 3.0 * 0.2843
        + 4.0 * coefficient
    )
    normal = np.array([-np.sin(theta), np.cos(theta)])
    normal_derivative = np.array(
        [
            -np.cos(theta) * theta_derivative,
            -np.sin(theta) * theta_derivative,
        ]
    )
    mean_corner = section_chord * np.array([x, mean_ordinate])
    mean_tangent = np.array([1.0, mean_slope])
    offset = section_chord * half_base * normal
    offset_derivative = (
        half_base_derivative * normal
        + half_base * normal_derivative
    )
    upper_corner = mean_corner + offset
    lower_corner = mean_corner - offset
    upper_tangent = _unit(
        "upper tangent", mean_tangent + offset_derivative
    )
    lower_tangent = _unit(
        "lower tangent", mean_tangent - offset_derivative
    )

    base = float(np.linalg.norm(upper_corner - lower_corner))
    expected_base = 2.0 * section_chord * abs(half_base)
    origin = 0.5 * (upper_corner + lower_corner)
    single_origin_residual = 0.5 * base
    normalized_origin = (
        single_origin_residual / base if base > 0.0 else 0.0
    )
    tangent_dot = float(
        np.clip(np.dot(upper_tangent, lower_tangent), -1.0, 1.0)
    )
    tangent_gap = float(np.arccos(tangent_dot))
    best_direction = _unit(
        "B2 minimax direction", upper_tangent + lower_tangent
    )
    mismatch = 0.5 * tangent_gap
    return BluntBaseTopology(
        base_fraction=fraction,
        thickness_coefficient=float(coefficient),
        upper_corner=upper_corner,
        lower_corner=lower_corner,
        upper_tangent=upper_tangent,
        lower_tangent=lower_tangent,
        half_base_thickness=float(section_chord * half_base),
        base_thickness=base,
        geometry_identity_residual=float(base - expected_base),
        optimal_single_origin=origin,
        optimal_single_origin_residual=single_origin_residual,
        normalized_single_origin_residual=float(normalized_origin),
        two_origin_attachment_residual=0.0,
        tangent_gap_angle_rad=tangent_gap,
        optimal_B2_direction=best_direction,
        optimal_B2_direction_mismatch_rad=mismatch,
        single_junction_topologically_admissible=(base == 0.0),
    )
