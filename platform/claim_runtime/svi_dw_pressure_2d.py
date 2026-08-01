"""Unified surface pressure and traction oracle for N2.6e1.

The pressure equation is the moving-body Bernoulli relation disclosed in
Riziotis (2003), Eqs. (7.143)--(7.147).  Its state is the gauge-invariant
surface quantity ``Phi-Phi_inf``.  Points inside the separated bubble receive
the same source-owned total-pressure deficit ``Delta h``; no independent
dynamic-stall force is added.

The resulting pressure and signed wall shear are integrated exactly once on
the actual two-sided panel surface.  This module deliberately does not infer
potential, ``Delta h``, separation, or skin friction from a force target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .svi_dw_types import ActualSurface2D, SVIDWValidationError


def _finite_scalar(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise SVIDWValidationError(f"{name} must be finite")
    return result


def _positive_scalar(name: str, value: Any) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise SVIDWValidationError(f"{name} must be positive")
    return result


def _finite_array(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise SVIDWValidationError(
            f"{name} must be finite with shape {shape}, got {result.shape}"
        )
    result.setflags(write=False)
    return result


def first_order_backward_time_derivative(
    current: Any,
    previous: Any,
    *,
    time_step: float,
) -> float | np.ndarray:
    """Return the source-owned first-order potential derivative."""
    dt = _positive_scalar("time_step", time_step)
    current_array = np.asarray(current, dtype=float)
    previous_array = np.asarray(previous, dtype=float)
    if current_array.shape != previous_array.shape:
        raise SVIDWValidationError("potential time levels must have identical shapes")
    if not (np.all(np.isfinite(current_array)) and np.all(np.isfinite(previous_array))):
        raise SVIDWValidationError("potential time levels must be finite")
    result = (current_array - previous_array) / dt
    if result.ndim == 0:
        return float(result)
    result = np.array(result, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BernoulliPressureInput2D:
    """Typed fields for thesis Eqs. (7.146)--(7.147)."""

    density: float
    freestream_velocity_inertial: Any
    relative_surface_velocity: Any
    body_surface_velocity: Any
    phi_minus_phi_infinity_current: Any
    phi_minus_phi_infinity_previous: Any
    time_step: float
    inside_separated_bubble: Any
    total_pressure_deficit: float

    def __post_init__(self) -> None:
        density = _positive_scalar("density", self.density)
        freestream = _finite_array(
            "freestream_velocity_inertial",
            self.freestream_velocity_inertial,
            shape=(2,),
        )
        relative_raw = np.asarray(self.relative_surface_velocity, dtype=float)
        if relative_raw.ndim != 2 or relative_raw.shape[1:] != (2,):
            raise SVIDWValidationError(
                "relative_surface_velocity must have shape (n,2)"
            )
        count = relative_raw.shape[0]
        relative = _finite_array(
            "relative_surface_velocity",
            relative_raw,
            shape=(count, 2),
        )
        body = _finite_array(
            "body_surface_velocity",
            self.body_surface_velocity,
            shape=(count, 2),
        )
        phi_current = _finite_array(
            "phi_minus_phi_infinity_current",
            self.phi_minus_phi_infinity_current,
            shape=(count,),
        )
        phi_previous = _finite_array(
            "phi_minus_phi_infinity_previous",
            self.phi_minus_phi_infinity_previous,
            shape=(count,),
        )
        bubble = np.array(self.inside_separated_bubble, dtype=bool, copy=True)
        if bubble.shape != (count,):
            raise SVIDWValidationError(
                "inside_separated_bubble must have one flag per surface point"
            )
        bubble.setflags(write=False)
        time_step = _positive_scalar("time_step", self.time_step)
        deficit = _finite_scalar("total_pressure_deficit", self.total_pressure_deficit)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "freestream_velocity_inertial", freestream)
        object.__setattr__(self, "relative_surface_velocity", relative)
        object.__setattr__(self, "body_surface_velocity", body)
        object.__setattr__(self, "phi_minus_phi_infinity_current", phi_current)
        object.__setattr__(self, "phi_minus_phi_infinity_previous", phi_previous)
        object.__setattr__(self, "time_step", time_step)
        object.__setattr__(self, "inside_separated_bubble", bubble)
        object.__setattr__(self, "total_pressure_deficit", deficit)


@dataclass(frozen=True)
class BernoulliPressureResult2D:
    pressure_difference: np.ndarray
    potential_time_derivative: np.ndarray
    freestream_kinetic_head: float
    relative_kinetic_head: np.ndarray
    body_kinetic_head: np.ndarray
    bubble_pressure_deficit: np.ndarray


def evaluate_moving_body_bernoulli_pressure(
    inputs: BernoulliPressureInput2D,
) -> BernoulliPressureResult2D:
    """Evaluate ``p-p_inf`` with one gauge and one bubble deficit."""
    if not isinstance(inputs, BernoulliPressureInput2D):
        raise SVIDWValidationError("inputs must be BernoulliPressureInput2D")
    potential_dt = first_order_backward_time_derivative(
        inputs.phi_minus_phi_infinity_current,
        inputs.phi_minus_phi_infinity_previous,
        time_step=inputs.time_step,
    )
    freestream_head = 0.5 * float(
        np.dot(
            inputs.freestream_velocity_inertial,
            inputs.freestream_velocity_inertial,
        )
    )
    relative_head = 0.5 * np.einsum(
        "ij,ij->i",
        inputs.relative_surface_velocity,
        inputs.relative_surface_velocity,
    )
    body_head = 0.5 * np.einsum(
        "ij,ij->i",
        inputs.body_surface_velocity,
        inputs.body_surface_velocity,
    )
    bubble_deficit = (
        inputs.total_pressure_deficit * inputs.inside_separated_bubble.astype(float)
    )
    pressure_head = (
        freestream_head - relative_head + body_head - potential_dt - bubble_deficit
    )
    pressure = inputs.density * pressure_head
    arrays = []
    for value in (
        pressure,
        potential_dt,
        relative_head,
        body_head,
        bubble_deficit,
    ):
        array = np.array(value, dtype=float, copy=True)
        array.setflags(write=False)
        arrays.append(array)
    return BernoulliPressureResult2D(
        pressure_difference=arrays[0],
        potential_time_derivative=arrays[1],
        freestream_kinetic_head=float(freestream_head),
        relative_kinetic_head=arrays[2],
        body_kinetic_head=arrays[3],
        bubble_pressure_deficit=arrays[4],
    )


@dataclass(frozen=True)
class SurfaceTractionLedger2D:
    """One panel-resolved pressure/shear force and moment ledger."""

    pressure_difference: np.ndarray
    signed_wall_shear: np.ndarray
    pressure_force_per_panel: np.ndarray
    shear_force_per_panel: np.ndarray
    total_force_per_panel: np.ndarray
    total_force: np.ndarray
    moment_per_panel: np.ndarray
    total_moment: float
    reference_point_body: np.ndarray


def integrate_surface_traction_once(
    surface: ActualSurface2D,
    *,
    pressure_difference: Any,
    signed_wall_shear: Any,
    reference_point_body: Any = (0.0, 0.0),
) -> SurfaceTractionLedger2D:
    """Integrate ``[-(p-p_inf)n + tau_w t] ds`` exactly once.

    ``signed_wall_shear`` is positive along the clockwise body-contour
    tangent.  Conversion from each IBL side's stagnation-to-trailing-edge
    convention must happen before this ledger.
    """
    if not isinstance(surface, ActualSurface2D):
        raise SVIDWValidationError("surface must be ActualSurface2D")
    count = surface.panel_count
    pressure = _finite_array("pressure_difference", pressure_difference, shape=(count,))
    shear = _finite_array("signed_wall_shear", signed_wall_shear, shape=(count,))
    reference = _finite_array("reference_point_body", reference_point_body, shape=(2,))
    pressure_force = (
        -pressure[:, None]
        * surface.panel_outward_normals
        * surface.panel_lengths[:, None]
    )
    shear_force = (
        shear[:, None] * surface.panel_tangents * surface.panel_lengths[:, None]
    )
    total_per_panel = pressure_force + shear_force
    total_force = np.sum(total_per_panel, axis=0)
    arm = surface.panel_midpoints - reference
    moment_per_panel = (
        arm[:, 0] * total_per_panel[:, 1] - arm[:, 1] * total_per_panel[:, 0]
    )
    total_moment = float(np.sum(moment_per_panel))

    arrays = []
    for value in (
        pressure,
        shear,
        pressure_force,
        shear_force,
        total_per_panel,
        total_force,
        moment_per_panel,
        reference,
    ):
        array = np.array(value, dtype=float, copy=True)
        array.setflags(write=False)
        arrays.append(array)
    return SurfaceTractionLedger2D(
        pressure_difference=arrays[0],
        signed_wall_shear=arrays[1],
        pressure_force_per_panel=arrays[2],
        shear_force_per_panel=arrays[3],
        total_force_per_panel=arrays[4],
        total_force=arrays[5],
        moment_per_panel=arrays[6],
        total_moment=total_moment,
        reference_point_body=arrays[7],
    )
