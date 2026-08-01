"""Curved-P2 Joukowski airfoil Kutta and pressure oracle.

Stage S2b of claim ``N3.1j3b6d`` bridges the S2a prescribed-circulation
circle to a finite-thickness body whose circulation is selected analytically
by a sharp trailing-edge Kutta condition.  The classified potential cut is
located only at the mapped cusp.  One trace derivative, one Bernoulli
pressure and one body-pressure integral are used.

This is not an unsteady wake solver, a finite-base closure, a three-
dimensional model, an LEV model, or a production load path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circulation_cut_oracle import CirculationCutError


@dataclass(frozen=True)
class JoukowskiP2Evaluation:
    panel_count: int
    quadrature_order: int
    mapping_parameter_b: float
    circle_center: np.ndarray
    circle_radius: float
    freestream_speed: float
    density: float
    angle_of_attack_rad: float
    kutta_circulation: float
    potential_gauge: float
    node_theta: np.ndarray
    node_coordinates: np.ndarray
    node_potential_gauge_free: np.ndarray
    node_potential: np.ndarray
    quadrature_theta: np.ndarray
    quadrature_coordinates: np.ndarray
    quadrature_arc_weights: np.ndarray
    tangential_velocity: np.ndarray
    exact_tangential_velocity: np.ndarray
    pressure_coefficient: np.ndarray
    exact_pressure_coefficient: np.ndarray
    potential_jump: float
    circulation: float
    kutta_numerator_residual: float
    pressure_force: np.ndarray
    drag: float
    lift: float
    expected_lift: float
    surface_velocity_rms_error: float
    surface_cp_rms_error: float
    lift_relative_error: float
    trailing_side_cp_mismatch: float


def _positive_finite(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise CirculationCutError(f"{name} must be positive and finite")
    return scalar


def _integer_at_least(name: str, value: int, minimum: int) -> int:
    integer = int(value)
    if integer != value or integer < minimum:
        raise CirculationCutError(
            f"{name} must be an integer >= {minimum}"
        )
    return integer


def _p2_shape(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = np.column_stack((
        2.0 * tau * tau - 3.0 * tau + 1.0,
        4.0 * tau * (1.0 - tau),
        2.0 * tau * tau - tau,
    ))
    derivative = np.column_stack((
        4.0 * tau - 3.0,
        4.0 - 8.0 * tau,
        4.0 * tau - 1.0,
    ))
    return shape, derivative


def _cusp_enriched_shape(
    tau: np.ndarray,
    *,
    cusp_at_start: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint/midpoint/far-end trace with the cusp derivative fixed zero."""

    coordinate = tau if cusp_at_start else 1.0 - tau
    cusp = (
        1.0 - 7.0 * coordinate**2 + 6.0 * coordinate**3
    )
    midpoint = 8.0 * coordinate**2 - 8.0 * coordinate**3
    far = -coordinate**2 + 2.0 * coordinate**3
    cusp_derivative = -14.0 * coordinate + 18.0 * coordinate**2
    midpoint_derivative = (
        16.0 * coordinate - 24.0 * coordinate**2
    )
    far_derivative = -2.0 * coordinate + 6.0 * coordinate**2
    if cusp_at_start:
        return (
            np.column_stack((cusp, midpoint, far)),
            np.column_stack((
                cusp_derivative,
                midpoint_derivative,
                far_derivative,
            )),
        )
    return (
        np.column_stack((far, midpoint, cusp)),
        -np.column_stack((
            far_derivative,
            midpoint_derivative,
            cusp_derivative,
        )),
    )


def _mapped_geometry(
    theta: np.ndarray,
    *,
    center: complex,
    radius: float,
    mapping_parameter_b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = radius * np.exp(1j * theta)
    zeta = center + w
    if np.any(np.abs(zeta) <= np.finfo(float).tiny):
        raise CirculationCutError(
            "the generating circle may not pass through zeta=0"
        )
    coordinate = zeta + mapping_parameter_b**2 / zeta
    derivative_theta = (
        1.0 - mapping_parameter_b**2 / zeta**2
    ) * 1j * w
    return w, zeta, coordinate, derivative_theta


def joukowski_p2_kutta_trace(
    *,
    panel_count: int,
    mapping_parameter_b: float = 1.0,
    circle_center: tuple[float, float] = (-0.08, 0.0),
    freestream_speed: float = 1.0,
    density: float = 1.0,
    angle_of_attack_deg: float = 5.0,
    quadrature_order: int = 16,
    potential_gauge: float = 0.0,
) -> JoukowskiP2Evaluation:
    """Evaluate the preregistered symmetric Joukowski Kutta canonical."""

    panels = _integer_at_least("panel_count", panel_count, 4)
    order = _integer_at_least("quadrature_order", quadrature_order, 2)
    b = _positive_finite("mapping_parameter_b", mapping_parameter_b)
    speed = _positive_finite("freestream_speed", freestream_speed)
    rho = _positive_finite("density", density)
    center_array = np.asarray(circle_center, dtype=float)
    if (
        center_array.shape != (2,)
        or not np.all(np.isfinite(center_array))
    ):
        raise CirculationCutError(
            "circle_center must contain two finite coordinates"
        )
    if abs(float(center_array[1])) > 1.0e-15:
        raise CirculationCutError(
            "S2b canonical requires a real circle center"
        )
    center = complex(*center_array)
    radius = b - float(center_array[0])
    if radius <= 0.0:
        raise CirculationCutError(
            "circle center must lie upstream of the mapped cusp"
        )
    alpha = np.deg2rad(float(angle_of_attack_deg))
    gauge = float(potential_gauge)
    if not np.isfinite(alpha) or not np.isfinite(gauge):
        raise CirculationCutError(
            "angle_of_attack_deg and potential_gauge must be finite"
        )
    gamma = -4.0 * np.pi * radius * speed * np.sin(alpha)

    node_theta = np.linspace(0.0, 2.0 * np.pi, 2 * panels + 1)
    node_w, _, node_coordinate_complex, _ = _mapped_geometry(
        node_theta,
        center=center,
        radius=radius,
        mapping_parameter_b=b,
    )
    node_complex_potential = (
        speed * np.exp(-1j * alpha) * node_w
        + speed * np.exp(1j * alpha) * radius**2 / node_w
    )
    node_potential_gauge_free = (
        np.real(node_complex_potential)
        + gamma * node_theta / (2.0 * np.pi)
    )
    node_potential = node_potential_gauge_free + gauge

    gauss_coordinate, gauss_weight = np.polynomial.legendre.leggauss(
        order
    )
    tau = 0.5 * (gauss_coordinate + 1.0)
    tau_weight = 0.5 * gauss_weight
    ordinary_shape, ordinary_derivative = _p2_shape(tau)
    delta_theta = 2.0 * np.pi / panels

    theta_blocks: list[np.ndarray] = []
    coordinate_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    velocity_blocks: list[np.ndarray] = []
    exact_velocity_blocks: list[np.ndarray] = []
    cp_blocks: list[np.ndarray] = []
    exact_cp_blocks: list[np.ndarray] = []
    force = np.zeros(2, dtype=float)
    circulation = 0.0
    first_cp: float | None = None
    last_cp: float | None = None
    dynamic_pressure = 0.5 * rho * speed * speed

    for panel in range(panels):
        if panel == 0:
            shape, derivative = _cusp_enriched_shape(
                tau, cusp_at_start=True
            )
        elif panel == panels - 1:
            shape, derivative = _cusp_enriched_shape(
                tau, cusp_at_start=False
            )
        else:
            shape, derivative = ordinary_shape, ordinary_derivative
        local_indices = [2 * panel, 2 * panel + 1, 2 * panel + 2]
        local_potential = node_potential_gauge_free[local_indices]
        local_coordinate = node_coordinate_complex[local_indices]
        theta = panel * delta_theta + tau * delta_theta
        coordinate = shape @ local_coordinate
        coordinate_derivative = derivative @ (
            local_coordinate - local_coordinate[0]
        )
        arc_jacobian = np.abs(coordinate_derivative)
        if np.any(arc_jacobian <= np.finfo(float).tiny):
            raise CirculationCutError(
                "degenerate P2 geometry derivative inside quadrature"
            )
        potential_derivative = derivative @ (
            local_potential - local_potential[0]
        )
        tangential_velocity = potential_derivative / arc_jacobian
        cp = 1.0 - (tangential_velocity / speed) ** 2

        w, zeta, _, exact_coordinate_derivative_theta = (
            _mapped_geometry(
                theta,
                center=center,
                radius=radius,
                mapping_parameter_b=b,
            )
        )
        complex_velocity = (
            speed * np.exp(-1j * alpha)
            - speed * np.exp(1j * alpha) * radius**2 / w**2
            - 1j * gamma / (2.0 * np.pi * w)
        ) / (1.0 - b**2 / zeta**2)
        exact_tangent = (
            exact_coordinate_derivative_theta
            / np.abs(exact_coordinate_derivative_theta)
        )
        exact_tangential_velocity = np.real(
            complex_velocity * exact_tangent
        )
        exact_cp = 1.0 - (
            exact_tangential_velocity / speed
        ) ** 2

        physical_weight = arc_jacobian * tau_weight
        tangent = coordinate_derivative / arc_jacobian
        outward_normal = np.column_stack((
            np.imag(tangent),
            -np.real(tangent),
        ))
        force -= dynamic_pressure * np.sum(
            cp[:, None] * outward_normal * physical_weight[:, None],
            axis=0,
        )
        circulation += float(np.dot(
            tangential_velocity, physical_weight
        ))
        if panel == 0:
            first_cp = float(cp[0])
        if panel == panels - 1:
            last_cp = float(cp[-1])

        theta_blocks.append(theta)
        coordinate_blocks.append(np.column_stack((
            np.real(coordinate),
            np.imag(coordinate),
        )))
        weight_blocks.append(physical_weight)
        velocity_blocks.append(tangential_velocity)
        exact_velocity_blocks.append(exact_tangential_velocity)
        cp_blocks.append(cp)
        exact_cp_blocks.append(exact_cp)

    quadrature_theta = np.concatenate(theta_blocks)
    quadrature_coordinates = np.concatenate(coordinate_blocks)
    quadrature_arc_weights = np.concatenate(weight_blocks)
    tangential_velocity = np.concatenate(velocity_blocks)
    exact_tangential_velocity = np.concatenate(exact_velocity_blocks)
    pressure_coefficient = np.concatenate(cp_blocks)
    exact_pressure_coefficient = np.concatenate(exact_cp_blocks)

    flow_axis = np.array([np.cos(alpha), np.sin(alpha)])
    lift_axis = np.array([-np.sin(alpha), np.cos(alpha)])
    drag = float(np.dot(force, flow_axis))
    lift = float(np.dot(force, lift_axis))
    expected_lift = float(-rho * speed * gamma)
    lift_relative_error = float(
        abs(lift - expected_lift)
        / max(abs(expected_lift), np.finfo(float).tiny)
    )
    kutta_numerator = (
        speed * np.exp(-1j * alpha)
        - speed * np.exp(1j * alpha)
        - 1j * gamma / (2.0 * np.pi * radius)
    )

    return JoukowskiP2Evaluation(
        panel_count=panels,
        quadrature_order=order,
        mapping_parameter_b=b,
        circle_center=center_array,
        circle_radius=radius,
        freestream_speed=speed,
        density=rho,
        angle_of_attack_rad=float(alpha),
        kutta_circulation=float(gamma),
        potential_gauge=gauge,
        node_theta=node_theta,
        node_coordinates=np.column_stack((
            np.real(node_coordinate_complex),
            np.imag(node_coordinate_complex),
        )),
        node_potential_gauge_free=node_potential_gauge_free,
        node_potential=node_potential,
        quadrature_theta=quadrature_theta,
        quadrature_coordinates=quadrature_coordinates,
        quadrature_arc_weights=quadrature_arc_weights,
        tangential_velocity=tangential_velocity,
        exact_tangential_velocity=exact_tangential_velocity,
        pressure_coefficient=pressure_coefficient,
        exact_pressure_coefficient=exact_pressure_coefficient,
        potential_jump=float(node_potential[-1] - node_potential[0]),
        circulation=float(circulation),
        kutta_numerator_residual=float(abs(kutta_numerator)),
        pressure_force=force,
        drag=drag,
        lift=lift,
        expected_lift=expected_lift,
        surface_velocity_rms_error=float(np.sqrt(np.mean(
            (tangential_velocity - exact_tangential_velocity) ** 2
        ))),
        surface_cp_rms_error=float(np.sqrt(np.mean(
            (pressure_coefficient - exact_pressure_coefficient) ** 2
        ))),
        lift_relative_error=lift_relative_error,
        trailing_side_cp_mismatch=float(abs(first_cp - last_cp)),
    )
