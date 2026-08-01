"""P2 potential-cut topology and pressure oracle on a circular cylinder.

This module is a deliberately narrow canonical for claim ``N3.1j3b6d``.
It distinguishes two trace spaces on the same closed material curve:

* ``closed`` identifies the first and last potential degree of freedom, so
  the integral of the discrete tangential gradient telescopes to zero.
* ``cut`` duplicates those two degrees of freedom and prescribes their jump
  as circulation.  This is the two-dimensional analogue of a classified
  doublet-wake cut.

The cut trace is differentiated once and passed through one Bernoulli
pressure integration.  There is no force fit, wake model, Kutta parameter,
LEV closure, production activation, or structural model here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


class CirculationCutError(ValueError):
    """Invalid input to the circulation-cut canonical."""


TraceTopology = Literal["closed", "cut"]


@dataclass(frozen=True)
class CircularP2TraceEvaluation:
    """One curved quadratic trace and its pressure integral."""

    topology: TraceTopology
    panel_count: int
    quadrature_order: int
    radius: float
    freestream_speed: float
    density: float
    prescribed_circulation: float
    potential_gauge: float
    node_theta: np.ndarray
    node_potential: np.ndarray
    quadrature_theta: np.ndarray
    quadrature_arc_weights: np.ndarray
    tangential_velocity: np.ndarray
    exact_tangential_velocity: np.ndarray
    pressure_coefficient: np.ndarray
    potential_jump: float
    circulation: float
    telescoping_circulation: float
    pressure_force: np.ndarray
    expected_pressure_force: np.ndarray
    tangential_velocity_rms_error: float
    lift_relative_error: float

    @property
    def drag(self) -> float:
        return float(self.pressure_force[0])

    @property
    def lift(self) -> float:
        return float(self.pressure_force[1])


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
    """Quadratic endpoint/midpoint Lagrange basis and tau derivatives."""

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


def circular_p2_trace(
    *,
    panel_count: int,
    topology: TraceTopology,
    radius: float = 1.0,
    freestream_speed: float = 1.0,
    density: float = 1.0,
    prescribed_circulation: float = 0.8,
    quadrature_order: int = 16,
    potential_gauge: float = 0.0,
) -> CircularP2TraceEvaluation:
    """Evaluate a curved P2 potential trace and its pressure force.

    The analytic cut potential and surface tangential velocity are

    ``phi = 2 U a cos(theta) + Gamma theta/(2 pi)``,
    ``u_theta = -2 U sin(theta) + Gamma/(2 pi a)``.

    For ``closed`` topology, the final trace degree of freedom is identified
    with the first one.  The resulting last-panel gradient may be large, but
    its closed-loop integral is exactly zero by construction; that behavior
    is the topology test rather than a candidate flow solution.
    """

    panels = _integer_at_least("panel_count", panel_count, 2)
    order = _integer_at_least("quadrature_order", quadrature_order, 2)
    if topology not in ("closed", "cut"):
        raise CirculationCutError("topology must be 'closed' or 'cut'")
    a = _positive_finite("radius", radius)
    speed = _positive_finite("freestream_speed", freestream_speed)
    rho = _positive_finite("density", density)
    gamma = float(prescribed_circulation)
    gauge = float(potential_gauge)
    if not np.isfinite(gamma):
        raise CirculationCutError(
            "prescribed_circulation must be finite"
        )
    if not np.isfinite(gauge):
        raise CirculationCutError("potential_gauge must be finite")

    node_theta = np.linspace(0.0, 2.0 * np.pi, 2 * panels + 1)
    node_potential = (
        2.0 * speed * a * np.cos(node_theta)
        + gamma * node_theta / (2.0 * np.pi)
        + gauge
    )
    if topology == "closed":
        node_potential[-1] = node_potential[0]

    gauss_coordinate, gauss_weight = np.polynomial.legendre.leggauss(
        order
    )
    tau = 0.5 * (gauss_coordinate + 1.0)
    tau_weight = 0.5 * gauss_weight
    shape, derivative = _p2_shape(tau)
    delta_theta = 2.0 * np.pi / panels

    theta_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    velocity_blocks: list[np.ndarray] = []
    exact_velocity_blocks: list[np.ndarray] = []
    cp_blocks: list[np.ndarray] = []
    endpoint_differences: list[float] = []
    for panel in range(panels):
        local_values = node_potential[
            [2 * panel, 2 * panel + 1, 2 * panel + 2]
        ]
        theta = panel * delta_theta + tau * delta_theta
        velocity = (
            derivative @ local_values
        ) / (a * delta_theta)
        exact_velocity = (
            -2.0 * speed * np.sin(theta)
            + gamma / (2.0 * np.pi * a)
        )
        theta_blocks.append(theta)
        weight_blocks.append(a * delta_theta * tau_weight)
        velocity_blocks.append(velocity)
        exact_velocity_blocks.append(exact_velocity)
        cp_blocks.append(1.0 - (velocity / speed) ** 2)
        endpoint_differences.append(
            float(local_values[2] - local_values[0])
        )

    quadrature_theta = np.concatenate(theta_blocks)
    arc_weights = np.concatenate(weight_blocks)
    tangential_velocity = np.concatenate(velocity_blocks)
    exact_tangential_velocity = np.concatenate(exact_velocity_blocks)
    pressure_coefficient = np.concatenate(cp_blocks)
    normal = np.column_stack((
        np.cos(quadrature_theta),
        np.sin(quadrature_theta),
    ))
    dynamic_pressure = 0.5 * rho * speed * speed
    pressure_force = -dynamic_pressure * np.sum(
        pressure_coefficient[:, None]
        * normal
        * arc_weights[:, None],
        axis=0,
    )
    expected_force = np.array([0.0, -rho * speed * gamma])
    circulation = float(np.dot(tangential_velocity, arc_weights))
    telescoping = float(np.sum(endpoint_differences))
    velocity_rms_error = float(np.sqrt(np.mean(
        (tangential_velocity - exact_tangential_velocity) ** 2
    )))
    lift_scale = max(abs(expected_force[1]), np.finfo(float).tiny)
    lift_relative_error = float(
        abs(pressure_force[1] - expected_force[1]) / lift_scale
    )

    return CircularP2TraceEvaluation(
        topology=topology,
        panel_count=panels,
        quadrature_order=order,
        radius=a,
        freestream_speed=speed,
        density=rho,
        prescribed_circulation=gamma,
        potential_gauge=gauge,
        node_theta=node_theta,
        node_potential=node_potential,
        quadrature_theta=quadrature_theta,
        quadrature_arc_weights=arc_weights,
        tangential_velocity=tangential_velocity,
        exact_tangential_velocity=exact_tangential_velocity,
        pressure_coefficient=pressure_coefficient,
        potential_jump=float(node_potential[-1] - node_potential[0]),
        circulation=circulation,
        telescoping_circulation=telescoping,
        pressure_force=pressure_force,
        expected_pressure_force=expected_force,
        tangential_velocity_rms_error=velocity_rms_error,
        lift_relative_error=lift_relative_error,
    )
