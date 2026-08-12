"""Auditable one-state unsteady lifting-line attached-flow prototype.

This module implements the reduced model in Izraelevitz, Zhu and
Triantafyllou (2017), equations (1)--(40), as far as those equations close
without the authors' MATLAB/Drake implementation.  It is intentionally kept
separate from :mod:`uvlm_polar_correction`: the latter remains the production
UVLM-preserving nonlinear-polar experiment, while this file is an independent
attached-flow model that can be compared with, or blended against, it.

The implemented ledger is

* infinite tilted horseshoe trailers, equations (1)--(2);
* the one-pole lift/circulation states, equations (19)--(24), (30)--(31);
* vector lift and induced drag, equations (32)--(34);
* flat-plate strip added mass, equations (35)--(39); and
* the lifting-line-to-lifting-surface gain in equation (42).

Only source-published constants are defaults.  In particular, no digitised
Figure 11 load enters this module.  The public ``movement_one_state_ullt``
function accepts a Ptera ``Movement`` and physical reference values; it has no
paper name, case identifier, or observation-dependent branch.

Known closure choices are explicit rather than hidden:

* The paper does not specify its ODE integrator.  A stable exponential update
  with a predictor/corrector evaluation of effective normalwash is used.
* Equation (42) is applied as the paper describes: a common post-solve gain on
  lift, circulation, and induced velocity.  It is not fitted to a transient.
* Added-mass body rates are periodic centred differences of the moving
  three-quarter-chord strip frame.  This is the direct discrete analogue of
  the body-frame rates in equation (39).
* The optional hybrid helper performs a transparent load-level blend.  It is
  not advertised as a decomposition of Ptera's UVLM circulatory and
  non-circulatory pressures, which are unavailable in its exported history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OneStateULLTParameters:
    """Source-published constants for the reduced attached-flow model."""

    lift_indicial_amplitude: float = -0.5
    lift_indicial_decay: float = -0.25
    circulation_indicial_amplitude: float = -0.8
    circulation_indicial_decay: float = -0.25
    section_lift_slope_per_rad: float = 2.0 * np.pi
    lifting_surface_correction_k: float = 13.5
    flat_plate_added_mass_factor: float = 0.85

    def __post_init__(self) -> None:
        if self.lift_indicial_amplitude == 0.0:
            raise ValueError("lift indicial amplitude cannot be zero")
        if self.lift_indicial_decay >= 0.0:
            raise ValueError("lift indicial pole must be stable")
        if self.circulation_indicial_decay != self.lift_indicial_decay:
            raise ValueError("one-state model requires the shared published pole")
        if self.section_lift_slope_per_rad <= 0.0:
            raise ValueError("section lift slope must be positive")
        if self.lifting_surface_correction_k < 0.0:
            raise ValueError("surface correction K cannot be negative")
        if self.flat_plate_added_mass_factor <= 0.0:
            raise ValueError("added-mass factor must be positive")

    @property
    def lift_initial_value(self) -> float:
        """Return ``phi(0)`` from paper equation (30)."""

        return 1.0 + self.lift_indicial_amplitude

    @property
    def circulation_initial_value(self) -> float:
        """Return ``Gamma_tilde(0)`` from paper equation (31)."""

        return 1.0 + self.circulation_indicial_amplitude

    @property
    def circulation_to_lift_state_gain(self) -> float:
        """Return ``A_Gamma/A_phi`` from paper equation (22)."""

        return self.circulation_indicial_amplitude / self.lift_indicial_amplitude

    def manifest(self) -> dict[str, float | str]:
        out: dict[str, float | str] = asdict(self)
        out.update(
            indicial_source=(
                "Izraelevitz et al. 2017 Table 1 and equations (30)-(31)"
            ),
            surface_correction_source="Izraelevitz et al. 2017 equation (42)",
            added_mass_source=(
                "Izraelevitz et al. 2017 equations (35)-(39); "
                "Munk Kam=0.85 value reported for AR=3"
            ),
            observation_fit="none",
        )
        return out


DEFAULT_ULLT_PARAMETERS = OneStateULLTParameters()


def lifting_surface_correction_gain(
    aspect_ratio: float,
    parameters: OneStateULLTParameters = DEFAULT_ULLT_PARAMETERS,
) -> float:
    """Return the paper's equation-(42) line-to-surface correction."""

    if aspect_ratio <= 0.0:
        raise ValueError("aspect ratio must be positive")
    numerator = 1.0 / (2.0 * np.pi) + 1.0 / (np.pi * aspect_ratio)
    denominator = numerator + (
        parameters.lifting_surface_correction_k
        * np.pi
        / (180.0 * aspect_ratio**2)
    )
    return float(numerator / denominator)


def _unit(vector: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(length, floor)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _strip_geometry(airplane: Any) -> dict[str, np.ndarray]:
    """Consolidate Ptera chordwise panels into lifting-line blade strips."""

    quarter_centres: list[np.ndarray] = []
    three_quarter_centres: list[np.ndarray] = []
    quarter_left: list[np.ndarray] = []
    quarter_right: list[np.ndarray] = []
    chord_axes: list[np.ndarray] = []
    span_axes: list[np.ndarray] = []
    normal_axes: list[np.ndarray] = []
    chord_lengths: list[float] = []
    strip_widths: list[float] = []
    areas: list[float] = []

    for wing in airplane.wings:
        panels = wing.panels
        for span_index in range(panels.shape[1]):
            leading = panels[0, span_index]
            trailing = panels[-1, span_index]
            left_le = np.asarray(leading.Flpp_G_Cg, dtype=float)
            right_le = np.asarray(leading.Frpp_G_Cg, dtype=float)
            left_te = np.asarray(trailing.Blpp_G_Cg, dtype=float)
            right_te = np.asarray(trailing.Brpp_G_Cg, dtype=float)
            left_quarter = left_le + 0.25 * (left_te - left_le)
            right_quarter = right_le + 0.25 * (right_te - right_le)
            left_three_quarter = left_le + 0.75 * (left_te - left_le)
            right_three_quarter = right_le + 0.75 * (right_te - right_le)

            strip_panels = panels[:, span_index]
            panel_areas = np.asarray(
                [float(panel.area) for panel in strip_panels], dtype=float
            )
            panel_normals = np.asarray(
                [np.asarray(panel.unitNormal_G, dtype=float) for panel in strip_panels]
            )
            area = float(np.sum(panel_areas))
            span_vector = right_quarter - left_quarter
            width = float(np.linalg.norm(span_vector))
            if area <= 0.0 or width <= 0.0:
                raise ValueError("movement contains a degenerate aerodynamic strip")

            leading_mid = 0.5 * (left_le + right_le)
            trailing_mid = 0.5 * (left_te + right_te)
            chord_axis = _unit(trailing_mid - leading_mid)
            span_axis = _unit(span_vector - np.dot(span_vector, chord_axis) * chord_axis)
            panel_normal = _unit(
                np.sum(panel_areas[:, None] * panel_normals, axis=0)
            )
            normal_axis = _unit(np.cross(chord_axis, span_axis))
            if np.dot(normal_axis, panel_normal) < 0.0:
                normal_axis = -normal_axis

            quarter_centres.append(0.5 * (left_quarter + right_quarter))
            three_quarter_centres.append(
                0.5 * (left_three_quarter + right_three_quarter)
            )
            quarter_left.append(left_quarter)
            quarter_right.append(right_quarter)
            chord_axes.append(chord_axis)
            span_axes.append(span_axis)
            normal_axes.append(normal_axis)
            chord_lengths.append(area / width)
            strip_widths.append(width)
            areas.append(area)

    if not quarter_centres:
        raise ValueError("movement airplane contains no aerodynamic strips")
    return {
        "quarter_centres": np.asarray(quarter_centres),
        "three_quarter_centres": np.asarray(three_quarter_centres),
        "quarter_left": np.asarray(quarter_left),
        "quarter_right": np.asarray(quarter_right),
        "chord_axes": np.asarray(chord_axes),
        "span_axes": np.asarray(span_axes),
        "normal_axes": np.asarray(normal_axes),
        "chord_lengths": np.asarray(chord_lengths),
        "strip_widths": np.asarray(strip_widths),
        "areas": np.asarray(areas),
    }


def _periodic_derivative(
    values: np.ndarray,
    delta_time: float,
    steps_per_cycle: int,
) -> np.ndarray:
    """Return a centred derivative with phase-correct movement boundaries.

    Ptera movements may contain either ``cycles * steps_per_cycle`` samples or
    the same history plus a duplicated phase-zero endpoint.  A plain global
    ``np.roll`` incorrectly pairs that duplicate endpoint with the first
    phase-zero sample instead of a phase-one sample.  That bad endpoint then
    contaminates the penultimate sample when this helper is applied a second
    time for added-mass acceleration.  Interior samples retain their true
    chronological neighbours; only the two array boundaries are completed
    from the matching phase of a complete cycle.
    """

    samples = np.asarray(values, dtype=float)
    if delta_time <= 0.0 or steps_per_cycle < 3:
        raise ValueError("delta_time must be positive and a cycle needs >=3 steps")
    if samples.shape[0] < steps_per_cycle:
        raise ValueError("history is shorter than one complete cycle")

    forward = np.roll(samples, -1, axis=0)
    backward = np.roll(samples, 1, axis=0)
    # At the initial phase-zero sample, the previous phase is the final sample
    # of the first complete cycle, not an optional duplicate endpoint.
    backward[0] = samples[steps_per_cycle - 1]

    last = samples.shape[0] - 1
    last_phase = last % steps_per_cycle
    next_phase = (last_phase + 1) % steps_per_cycle
    last_block_start = last - last_phase
    next_index = last_block_start + next_phase
    if next_index >= samples.shape[0]:
        next_index -= steps_per_cycle
    forward[-1] = samples[next_index]
    return (forward - backward) / (2.0 * delta_time)


def _trailer_influence(
    control_points: np.ndarray,
    left_origins: np.ndarray,
    right_origins: np.ndarray,
    left_directions: np.ndarray,
    right_directions: np.ndarray,
) -> np.ndarray:
    """Return vector velocity per unit circulation from paper equation (1)."""

    r_left = control_points[:, None, :] - left_origins[None, :, :]
    r_right = control_points[:, None, :] - right_origins[None, :, :]
    distance_left = np.linalg.norm(r_left, axis=-1)
    distance_right = np.linalg.norm(r_right, axis=-1)
    denominator_left = distance_left * (
        distance_left - np.sum(left_directions[None, :, :] * r_left, axis=-1)
    )
    denominator_right = distance_right * (
        distance_right - np.sum(right_directions[None, :, :] * r_right, axis=-1)
    )
    floor = 1.0e-12
    denominator_left = np.maximum(denominator_left, floor)
    denominator_right = np.maximum(denominator_right, floor)
    left = -np.cross(left_directions[None, :, :], r_left) / (
        4.0 * np.pi * denominator_left[..., None]
    )
    right = np.cross(right_directions[None, :, :], r_right) / (
        4.0 * np.pi * denominator_right[..., None]
    )
    return left + right


def _solve_one_state_step(
    geometry: dict[str, np.ndarray],
    relative_velocity_three_quarter: np.ndarray,
    relative_velocity_left_quarter: np.ndarray,
    relative_velocity_right_quarter: np.ndarray,
    state_lift: np.ndarray,
    parameters: OneStateULLTParameters,
) -> dict[str, np.ndarray]:
    span = geometry["span_axes"]
    normal = geometry["normal_axes"]
    chord = geometry["chord_lengths"]
    velocity_perpendicular = relative_velocity_three_quarter - (
        np.sum(relative_velocity_three_quarter * span, axis=-1, keepdims=True) * span
    )
    speed_perpendicular = np.linalg.norm(velocity_perpendicular, axis=-1)
    normalwash = np.sum(relative_velocity_three_quarter * normal, axis=-1)

    influence_vector = _trailer_influence(
        geometry["quarter_centres"],
        geometry["quarter_left"],
        geometry["quarter_right"],
        _unit(relative_velocity_left_quarter),
        _unit(relative_velocity_right_quarter),
    )
    influence_normal = np.sum(
        influence_vector * normal[:, None, :], axis=-1
    )
    gamma_initial = parameters.circulation_initial_value
    state_circulation = (
        parameters.circulation_to_lift_state_gain * state_lift
    )
    system = np.diag(
        2.0 / (chord * parameters.section_lift_slope_per_rad)
    ) - gamma_initial * influence_normal
    circulation_line = np.linalg.solve(
        system,
        gamma_initial * normalwash + state_circulation,
    )
    induced_vector_line = np.einsum(
        "ijc,j->ic", influence_vector, circulation_line
    )
    induced_normalwash_line = np.sum(induced_vector_line * normal, axis=-1)
    effective_normalwash_line = normalwash + induced_normalwash_line
    return {
        "velocity_perpendicular": velocity_perpendicular,
        "speed_perpendicular": speed_perpendicular,
        "normalwash": normalwash,
        "influence_vector": influence_vector,
        "circulation_line": circulation_line,
        "induced_vector_line": induced_vector_line,
        "induced_normalwash_line": induced_normalwash_line,
        "effective_normalwash_line": effective_normalwash_line,
    }


def _added_mass_force_history(
    geometry_history: list[dict[str, np.ndarray]],
    body_velocity_three_quarter: np.ndarray,
    freestream_vector: np.ndarray,
    rho_kg_m3: float,
    delta_time: float,
    steps_per_cycle: int,
    parameters: OneStateULLTParameters,
) -> np.ndarray:
    """Return strip forces from paper equations (35)--(39)."""

    chord_axis = np.stack([item["chord_axes"] for item in geometry_history])
    span_axis = np.stack([item["span_axes"] for item in geometry_history])
    normal_axis = np.stack([item["normal_axes"] for item in geometry_history])
    rotation = np.stack((chord_axis, span_axis, normal_axis), axis=-1)
    rotation_rate = _periodic_derivative(rotation, delta_time, steps_per_cycle)
    omega_global = np.empty(chord_axis.shape, dtype=float)
    for time_index in range(rotation.shape[0]):
        for strip_index in range(rotation.shape[1]):
            omega_matrix = (
                rotation_rate[time_index, strip_index]
                @ rotation[time_index, strip_index].T
            )
            omega_matrix = 0.5 * (omega_matrix - omega_matrix.T)
            omega_global[time_index, strip_index] = (
                omega_matrix[2, 1],
                omega_matrix[0, 2],
                omega_matrix[1, 0],
            )

    body_velocity_relative_fluid = (
        body_velocity_three_quarter - freestream_vector[None, None, :]
    )
    body_velocity_b = np.einsum(
        "tsji,tsj->tsi", rotation, body_velocity_relative_fluid
    )
    omega_b = np.einsum("tsji,tsj->tsi", rotation, omega_global)
    velocity_b = np.concatenate((body_velocity_b, omega_b), axis=-1)
    acceleration_b = _periodic_derivative(
        velocity_b, delta_time, steps_per_cycle
    )

    output = np.zeros_like(body_velocity_three_quarter)
    for time_index, geometry in enumerate(geometry_history):
        chord = geometry["chord_lengths"]
        width = geometry["strip_widths"]
        for strip_index in range(chord.size):
            c = float(chord[strip_index])
            ds = float(width[strip_index])
            m33 = (
                rho_kg_m3
                * np.pi
                * parameters.flat_plate_added_mass_factor
                * (c / 2.0) ** 2
                * ds
            )
            m35 = m33 * c / 4.0
            m55 = rho_kg_m3 * np.pi / 8.0 * (c / 2.0) ** 4 * ds + (
                c / 4.0
            ) ** 2 * m33
            mass = np.zeros((6, 6), dtype=float)
            mass[2, 2] = m33
            mass[2, 4] = m35
            mass[4, 2] = m35
            mass[4, 4] = m55
            block11 = mass[:3, :3]
            block12 = mass[:3, 3:]
            block21 = mass[3:, :3]
            block22 = mass[3:, 3:]
            vb = body_velocity_b[time_index, strip_index]
            omega = omega_b[time_index, strip_index]
            coriolis = np.zeros((6, 6), dtype=float)
            momentum_translation = block11 @ vb + block12 @ omega
            momentum_rotation = block21 @ vb + block22 @ omega
            coriolis[:3, 3:] = -_skew(momentum_translation)
            coriolis[3:, :3] = -_skew(momentum_translation)
            coriolis[3:, 3:] = -_skew(momentum_rotation)
            force_moment_b = -mass @ acceleration_b[time_index, strip_index]
            force_moment_b -= coriolis @ velocity_b[time_index, strip_index]
            output[time_index, strip_index] = (
                rotation[time_index, strip_index] @ force_moment_b[:3]
            )
    return output


def _periodic_resample(
    phase: np.ndarray,
    values: np.ndarray,
    output_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if output_samples < 2:
        raise ValueError("output_samples must be at least two")
    target = np.arange(output_samples, dtype=float) / output_samples
    order = np.argsort(phase)
    output = np.empty((output_samples,) + values.shape[1:], dtype=float)
    values_flat = values.reshape(values.shape[0], -1)
    output_flat = output.reshape(output_samples, -1)
    for column in range(values_flat.shape[1]):
        output_flat[:, column] = np.interp(
            target,
            phase[order],
            values_flat[order, column],
            period=1.0,
        )
    return target, output


def smooth_separation_fraction(
    alpha_rad: np.ndarray,
    *,
    attached_limit_deg: float,
    fully_separated_deg: float,
) -> np.ndarray:
    """Return the shared smoothstep separation gate from local incidence.

    Gate limits are required arguments so they remain owned by the external
    nonlinear-polar model rather than becoming additional ULLT calibration
    constants.  Passing the limits from ``FullAnglePolarParameters`` makes
    this bit-for-bit the same local gate used by ``uvlm_polar_correction``.
    """

    if not 0.0 <= attached_limit_deg < fully_separated_deg <= 90.0:
        raise ValueError("invalid attached-to-separated angle interval")
    incidence_deg = np.abs(np.rad2deg(np.asarray(alpha_rad, dtype=float)))
    fraction = np.clip(
        (incidence_deg - attached_limit_deg)
        / (fully_separated_deg - attached_limit_deg),
        0.0,
        1.0,
    )
    return fraction**2 * (3.0 - 2.0 * fraction)


def movement_one_state_ullt(
    movement: Any,
    *,
    source_cycle_step_range: tuple[int, int] | list[int],
    period_s: float,
    freestream_m_s: float,
    rho_kg_m3: float,
    aspect_ratio: float,
    area_m2: float,
    output_samples: int = 128,
    parameters: OneStateULLTParameters = DEFAULT_ULLT_PARAMETERS,
) -> dict[str, Any]:
    """Run the source-constrained one-state ULLT on a Ptera movement.

    The selected source cycle is inclusive and is used only for reporting.
    States are marched from movement step zero through the selected cycle so
    that preceding cycles provide the wake-state spin-up.
    """

    if min(period_s, freestream_m_s, rho_kg_m3, aspect_ratio, area_m2) <= 0.0:
        raise ValueError("all physical reference values must be positive")
    start, stop = (int(value) for value in source_cycle_step_range)
    airplanes = movement.airplanes[0]
    if start < 0 or stop < start or stop >= len(airplanes):
        raise ValueError("source cycle step range lies outside movement history")
    expected_steps = int(round(period_s / movement.delta_time))
    if stop - start + 1 != expected_steps:
        raise ValueError("selected cycle does not match period/delta_time")

    geometry_history = [_strip_geometry(airplane) for airplane in airplanes]
    strip_counts = {item["areas"].size for item in geometry_history}
    if len(strip_counts) != 1:
        raise ValueError("strip topology changes during the movement")
    positions_three_quarter = np.stack(
        [item["three_quarter_centres"] for item in geometry_history]
    )
    positions_left_quarter = np.stack(
        [item["quarter_left"] for item in geometry_history]
    )
    positions_right_quarter = np.stack(
        [item["quarter_right"] for item in geometry_history]
    )
    body_velocity_three_quarter = _periodic_derivative(
        positions_three_quarter, movement.delta_time, expected_steps
    )
    body_velocity_left_quarter = _periodic_derivative(
        positions_left_quarter, movement.delta_time, expected_steps
    )
    body_velocity_right_quarter = _periodic_derivative(
        positions_right_quarter, movement.delta_time, expected_steps
    )
    freestream_vector = np.array([freestream_m_s, 0.0, 0.0])
    relative_velocity_three_quarter = (
        freestream_vector[None, None, :] - body_velocity_three_quarter
    )
    relative_velocity_left_quarter = (
        freestream_vector[None, None, :] - body_velocity_left_quarter
    )
    relative_velocity_right_quarter = (
        freestream_vector[None, None, :] - body_velocity_right_quarter
    )

    time_count, strip_count = positions_three_quarter.shape[:2]
    state = np.zeros(strip_count, dtype=float)
    circulation_history = np.empty((time_count, strip_count), dtype=float)
    state_history = np.empty_like(circulation_history)
    alpha_history = np.empty_like(circulation_history)
    circulatory_force = np.empty((time_count, strip_count, 3), dtype=float)
    surface_gain = lifting_surface_correction_gain(aspect_ratio, parameters)

    for time_index, geometry in enumerate(geometry_history):
        step = _solve_one_state_step(
            geometry,
            relative_velocity_three_quarter[time_index],
            relative_velocity_left_quarter[time_index],
            relative_velocity_right_quarter[time_index],
            state,
            parameters,
        )
        state_history[time_index] = state
        circulation = surface_gain * step["circulation_line"]
        circulation_history[time_index] = circulation
        chordwise_velocity = np.sum(
            relative_velocity_three_quarter[time_index]
            * geometry["chord_axes"],
            axis=-1,
        )
        alpha_history[time_index] = np.arctan2(
            step["normalwash"], chordwise_velocity
        )

        effective_normalwash = step["effective_normalwash_line"]
        lift_per_span_line = (
            0.5
            * rho_kg_m3
            * geometry["chord_lengths"]
            * parameters.section_lift_slope_per_rad
            * step["speed_perpendicular"]
            * (parameters.lift_initial_value * effective_normalwash + state)
        )
        lift_per_span = surface_gain * lift_per_span_line
        induced_vector = surface_gain * step["induced_vector_line"]
        # Equation (33): the unsteady infinite-wing downwash is the lift
        # response deficit relative to effective normalwash.
        downwash_2d = (
            parameters.lift_initial_value * effective_normalwash
            + state
            - effective_normalwash
        )
        base_lift_direction = _unit(
            np.cross(
                step["velocity_perpendicular"], geometry["span_axes"]
            )
        )
        velocity_total = (
            relative_velocity_three_quarter[time_index]
            + induced_vector
            + downwash_2d[:, None] * base_lift_direction
        )
        force_direction = _unit(
            np.cross(velocity_total, geometry["span_axes"])
        )
        circulatory_force[time_index] = (
            lift_per_span[:, None]
            * geometry["strip_widths"][:, None]
            * force_direction
        )

        decay_rate = (
            2.0
            * step["speed_perpendicular"]
            / geometry["chord_lengths"]
            * parameters.lift_indicial_decay
        )
        exponential = np.exp(decay_rate * movement.delta_time)
        state_predictor = exponential * state + (
            parameters.lift_indicial_amplitude
            * effective_normalwash
            * (exponential - 1.0)
        )
        predicted = _solve_one_state_step(
            geometry,
            relative_velocity_three_quarter[time_index],
            relative_velocity_left_quarter[time_index],
            relative_velocity_right_quarter[time_index],
            state_predictor,
            parameters,
        )
        mean_normalwash = 0.5 * (
            effective_normalwash + predicted["effective_normalwash_line"]
        )
        state = exponential * state + (
            parameters.lift_indicial_amplitude
            * mean_normalwash
            * (exponential - 1.0)
        )

    added_mass_force = _added_mass_force_history(
        geometry_history,
        body_velocity_three_quarter,
        freestream_vector,
        rho_kg_m3,
        movement.delta_time,
        expected_steps,
        parameters,
    )
    circulatory_force_total = np.sum(circulatory_force, axis=1)
    added_mass_force_total = np.sum(added_mass_force, axis=1)
    force_total = circulatory_force_total + added_mass_force_total

    selected = np.arange(start, stop + 1, dtype=int)
    raw_phase = np.mod(selected * movement.delta_time / period_s, 1.0)
    phase, force_output = _periodic_resample(
        raw_phase, force_total[selected], output_samples
    )
    _, circulatory_output = _periodic_resample(
        raw_phase, circulatory_force_total[selected], output_samples
    )
    _, added_mass_output = _periodic_resample(
        raw_phase, added_mass_force_total[selected], output_samples
    )
    _, circulation_output = _periodic_resample(
        raw_phase, circulation_history[selected], output_samples
    )
    _, state_output = _periodic_resample(
        raw_phase, state_history[selected], output_samples
    )
    _, alpha_output = _periodic_resample(
        raw_phase, alpha_history[selected], output_samples
    )
    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    # This global frame follows the paper's Figure 2: +x is drag and +z is
    # lift.  Hence thrust is -Fx.
    lift = force_output[:, 2]
    drag = force_output[:, 0]
    return {
        "phase": phase,
        "force_g_n": force_output,
        "circulatory_force_g_n": circulatory_output,
        "added_mass_force_g_n": added_mass_output,
        "lift_n": lift,
        "drag_n": drag,
        "thrust_n": -drag,
        "CL": lift / q_area,
        "CD": drag / q_area,
        "CT": -drag / q_area,
        "mean_lift_n": float(np.mean(force_total[selected, 2])),
        "mean_drag_n": float(np.mean(force_total[selected, 0])),
        "mean_thrust_n": float(-np.mean(force_total[selected, 0])),
        "mean_CL": float(np.mean(force_total[selected, 2]) / q_area),
        "mean_CD": float(np.mean(force_total[selected, 0]) / q_area),
        "mean_CT": float(-np.mean(force_total[selected, 0]) / q_area),
        "circulation_m2_s": circulation_output,
        "lift_state_m_s": state_output,
        "alpha_rad": alpha_output,
        "strip_area_m2": np.mean(
            np.stack([item["areas"] for item in geometry_history])[selected],
            axis=0,
        ),
        "max_abs_alpha_deg": float(
            np.max(np.abs(np.rad2deg(alpha_history[selected])))
        ),
        "surface_correction_gain": surface_gain,
        "strip_count": strip_count,
        "source_cycle_step_range": [start, stop],
        "parameters": parameters.manifest(),
        "model_semantics": (
            "source-constrained one-state ULLT attached-flow prototype plus "
            "flat-plate added mass; no digitized-load fit"
        ),
    }


def blend_attached_with_uvlm_polar(
    ullt_history: dict[str, Any],
    uvlm_polar_history: dict[str, Any],
    separation_fraction: np.ndarray,
) -> dict[str, Any]:
    """Blend ULLT attached loads into an externally computed UVLM/polar load.

    ``separation_fraction=0`` selects ULLT; ``1`` selects the UVLM plus
    nonlinear-polar history.  This helper is deliberately load-level and
    labels that limitation.  It does not claim access to Ptera's internal
    circulatory/non-circulatory pressure decomposition.
    """

    phase = np.asarray(ullt_history["phase"], dtype=float)
    other_phase = np.asarray(uvlm_polar_history["phase"], dtype=float)
    if phase.shape != other_phase.shape or not np.allclose(
        phase, other_phase, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("ULLT and UVLM/polar phases are not aligned")
    separation = np.asarray(separation_fraction, dtype=float)
    if separation.shape != phase.shape or np.any(~np.isfinite(separation)):
        raise ValueError("separation fraction must be a finite phase vector")
    if np.any((separation < 0.0) | (separation > 1.0)):
        raise ValueError("separation fraction must lie in [0, 1]")
    ullt_lift = np.asarray(ullt_history["lift_n"], dtype=float)
    ullt_drag = np.asarray(ullt_history["drag_n"], dtype=float)
    other_lift = np.asarray(uvlm_polar_history["lift_n"], dtype=float)
    if "drag_n" in uvlm_polar_history:
        other_drag = np.asarray(uvlm_polar_history["drag_n"], dtype=float)
    else:
        other_drag = -np.asarray(uvlm_polar_history["thrust_n"], dtype=float)
    lift = (1.0 - separation) * ullt_lift + separation * other_lift
    drag = (1.0 - separation) * ullt_drag + separation * other_drag
    return {
        "phase": phase.copy(),
        "lift_n": lift,
        "drag_n": drag,
        "thrust_n": -drag,
        "separation_fraction": separation.copy(),
        "model_semantics": (
            "auditable load-level ULLT-attached / UVLM-polar-separated blend; "
            "not a UVLM pressure-channel decomposition"
        ),
    }
