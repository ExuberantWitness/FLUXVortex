"""Ptera-native surface-release load corrections for the FluxV v5f prototype.

This module is deliberately a pure, post-solve mechanical kernel.  It accepts the
four LineVortex vectors, centers, and velocities already evaluated by the Ptera
parent load path.  It does not evaluate a pressure model, an impulse model, a
section polar, or any alternate induced velocity.

The surface-release ledger is intentionally not represented as an increment to
every parent bound RingVortex.  That construction would create a spurious front
leg.  Instead, its only effective LineVortex-strength corrections are

* ``front``: identically zero;
* ``right``/``left``: the current surface release at a span boundary and one
  half of its difference from the adjacent strip on an interior edge;
* ``back``: ``surface_release_current - wake_release_previous`` on trailing-edge
  Panels and zero elsewhere.

At step zero no previous wake exists, so the kernel forcibly uses zero for the
effective ``wake_release_previous`` regardless of the supplied finite audit
value.  The Kutta--Joukowski correction uses Ptera's exact operand order,
``rho * delta_gamma * cross(V, l)``.  Hirato Eq. 17 is a separate material-LEV
state term and is present only on strips active at the current step::

    delta_F_dGamma = (
        -rho * area * normal
        * (gamma_lev_current - gamma_lev_previous) / dt
        if active_current else 0
    )

This kernel must not be attached to a solver until an integration test proves
that every supplied leg vector, center, and velocity is the corresponding array
from the live parent ``_calculate_loads`` invocation.  Passing a velocity from a
different wake, pressure, or movement-velocity owner would invalidate the
mechanical parity established here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


LEG_NAMES = ("right", "front", "left", "back")


def _finite_vector_array(
    value: np.ndarray, *, name: str, panel_count: int
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (panel_count, 3):
        raise ValueError(f"{name} must have shape (panel_count, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _finite_scalar_array(
    value: np.ndarray, *, name: str, panel_count: int
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (panel_count,):
        raise ValueError(f"{name} must have shape (panel_count,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _leg_arrays(
    values: Mapping[str, np.ndarray], *, name: str, panel_count: int
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by {LEG_NAMES}")
    if set(values) != set(LEG_NAMES):
        raise ValueError(f"{name} must contain exactly the keys {LEG_NAMES}")
    return {
        leg: _finite_vector_array(
            values[leg], name=f"{name}[{leg!r}]", panel_count=panel_count
        )
        for leg in LEG_NAMES
    }


def _topology_columns(
    topology: Mapping[str, np.ndarray], *, panel_count: int
) -> dict[str, np.ndarray]:
    if not isinstance(topology, Mapping):
        raise TypeError("topology must be a mapping")
    required = (
        "airplane_index",
        "wing_index",
        "chordwise_index",
        "spanwise_index",
    )
    columns: dict[str, np.ndarray] = {}
    for name in required:
        if name not in topology:
            raise ValueError(f"topology is missing {name!r}")
        raw = np.asarray(topology[name])
        if raw.shape != (panel_count,):
            raise ValueError(f"topology[{name!r}] must have shape (panel_count,)")
        if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
            raw.dtype, np.bool_
        ):
            raise TypeError(f"topology[{name!r}] must contain integers")
        column = raw.astype(np.int64, copy=True)
        if np.any(column < 0):
            raise ValueError(f"topology[{name!r}] must be nonnegative")
        columns[name] = column
    return columns


def _wing_index_maps(
    topology: dict[str, np.ndarray],
) -> list[tuple[np.ndarray, dict[tuple[int, int], int], int, int]]:
    airplane = topology["airplane_index"]
    wing = topology["wing_index"]
    chord = topology["chordwise_index"]
    span = topology["spanwise_index"]
    wing_keys = list(dict.fromkeys(zip(airplane, wing, strict=True)))
    index_maps: list[tuple[np.ndarray, dict[tuple[int, int], int], int, int]] = []
    for airplane_index, wing_index in wing_keys:
        indices = np.flatnonzero((airplane == airplane_index) & (wing == wing_index))
        chord_count = int(np.max(chord[indices], initial=-1)) + 1
        span_count = int(np.max(span[indices], initial=-1)) + 1
        if chord_count <= 0 or span_count <= 0:
            raise ValueError("each wing must contain a nonempty panel grid")
        lookup: dict[tuple[int, int], int] = {}
        for panel_index in indices:
            key = (int(chord[panel_index]), int(span[panel_index]))
            if key in lookup:
                raise ValueError("wing topology contains a duplicate panel cell")
            lookup[key] = int(panel_index)
        expected = {
            (chordwise_index, spanwise_index)
            for chordwise_index in range(chord_count)
            for spanwise_index in range(span_count)
        }
        if set(lookup) != expected:
            raise ValueError("each wing topology must be a complete rectangular grid")
        index_maps.append((indices, lookup, chord_count, span_count))
    if sum(indices.size for indices, *_ in index_maps) != airplane.size:
        raise ValueError("panel topology does not assign every panel to one wing")
    return index_maps


def _validate_stripwise_field(
    field: np.ndarray,
    *,
    name: str,
    wing_maps: list[tuple[np.ndarray, dict[tuple[int, int], int], int, int]],
) -> None:
    for _, lookup, chord_count, span_count in wing_maps:
        for spanwise_index in range(span_count):
            reference = field[lookup[(0, spanwise_index)]]
            chord_values = np.asarray(
                [
                    field[lookup[(chordwise_index, spanwise_index)]]
                    for chordwise_index in range(chord_count)
                ]
            )
            if not np.array_equal(chord_values, np.full(chord_count, reference)):
                raise ValueError(f"{name} must be constant over each strip chord")


def _effective_strength_deltas(
    surface_release_current: np.ndarray,
    wake_release_previous: np.ndarray,
    *,
    wing_maps: list[tuple[np.ndarray, dict[tuple[int, int], int], int, int]],
    panel_count: int,
) -> dict[str, np.ndarray]:
    strengths = {leg: np.zeros(panel_count, dtype=float) for leg in LEG_NAMES}
    for _, lookup, chord_count, span_count in wing_maps:
        for chordwise_index in range(chord_count):
            for spanwise_index in range(span_count):
                panel_index = lookup[(chordwise_index, spanwise_index)]
                release_here = surface_release_current[panel_index]
                if spanwise_index == span_count - 1:
                    strengths["right"][panel_index] = release_here
                else:
                    panel_to_right = lookup[(chordwise_index, spanwise_index + 1)]
                    strengths["right"][panel_index] = (
                        release_here - surface_release_current[panel_to_right]
                    ) / 2.0

                if spanwise_index == 0:
                    strengths["left"][panel_index] = release_here
                else:
                    panel_to_left = lookup[(chordwise_index, spanwise_index - 1)]
                    strengths["left"][panel_index] = (
                        release_here - surface_release_current[panel_to_left]
                    ) / 2.0

                if chordwise_index == chord_count - 1:
                    strengths["back"][panel_index] = (
                        release_here - wake_release_previous[panel_index]
                    )
    return strengths


def _active_panel_array(value: bool | np.ndarray, *, panel_count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape == ():
        if not np.issubdtype(raw.dtype, np.bool_):
            raise TypeError("active_current must be boolean")
        return np.full(panel_count, bool(raw), dtype=bool)
    if raw.shape != (panel_count,):
        raise ValueError("active_current must be scalar or have shape (panel_count,)")
    if not np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("active_current must contain booleans")
    return raw.astype(bool, copy=True)


def _canonicalize_zeros(array: np.ndarray) -> np.ndarray:
    """Return a copy with every numerical zero represented as positive zero."""

    result = np.asarray(array, dtype=float).copy()
    result[result == 0.0] = 0.0
    return result


def calculate_native_surface_release_load_correction(
    *,
    rho_kg_m3: float,
    delta_time_s: float,
    current_step: int,
    topology: Mapping[str, np.ndarray],
    surface_release_current_m2_s: np.ndarray,
    wake_release_previous_m2_s: np.ndarray,
    gamma_lev_current_m2_s: np.ndarray,
    gamma_lev_previous_m2_s: np.ndarray,
    active_current: bool | np.ndarray,
    panel_area_m2: np.ndarray,
    panel_normal_gp1: np.ndarray,
    panel_collocation_gp1_cgp1_m: np.ndarray,
    leg_vectors_gp1_m: Mapping[str, np.ndarray],
    leg_velocities_gp1_m_s: Mapping[str, np.ndarray],
    leg_centers_gp1_cgp1_m: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Calculate only the Ptera-native force and moment corrections.

    The velocity arrays must be the exact values used by the parent Ptera load
    calculation: parent ``calculate_solution_velocity`` at each corresponding
    leg center plus the parent's movement velocity at that same center.  Centers
    and the panel collocation points must already be relative to the first
    Airplane's center of gravity.

    The four circulation-state inputs are panel-order arrays, but each value must
    be identical along the chord of its ``(airplane, wing, span)`` strip.
    ``active_current`` may be one boolean or a similarly broadcast panel-order
    boolean array.  Requiring these broadcasts explicitly prevents an
    accidental, unreviewed chordwise state model.
    """

    rho = float(rho_kg_m3)
    delta_time = float(delta_time_s)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho_kg_m3 must be finite and positive")
    if not np.isfinite(delta_time) or delta_time <= 0.0:
        raise ValueError("delta_time_s must be finite and positive")

    if isinstance(current_step, (bool, np.bool_)) or not isinstance(
        current_step, (int, np.integer)
    ):
        raise TypeError("current_step must be an integer")
    step = int(current_step)
    if step < 0:
        raise ValueError("current_step must be nonnegative")

    surface_release_raw = np.asarray(surface_release_current_m2_s)
    if surface_release_raw.ndim != 1 or surface_release_raw.size == 0:
        raise ValueError(
            "surface_release_current_m2_s must be a nonempty one-dimensional array"
        )
    panel_count = int(surface_release_raw.size)
    surface_release_current = _finite_scalar_array(
        surface_release_current_m2_s,
        name="surface_release_current_m2_s",
        panel_count=panel_count,
    )
    wake_release_previous_supplied = _finite_scalar_array(
        wake_release_previous_m2_s,
        name="wake_release_previous_m2_s",
        panel_count=panel_count,
    )
    gamma_lev_current = _finite_scalar_array(
        gamma_lev_current_m2_s,
        name="gamma_lev_current_m2_s",
        panel_count=panel_count,
    )
    gamma_lev_previous = _finite_scalar_array(
        gamma_lev_previous_m2_s,
        name="gamma_lev_previous_m2_s",
        panel_count=panel_count,
    )
    area = _finite_scalar_array(
        panel_area_m2, name="panel_area_m2", panel_count=panel_count
    )
    if np.any(area <= 0.0):
        raise ValueError("panel_area_m2 must be positive")
    normal = _finite_vector_array(
        panel_normal_gp1, name="panel_normal_gp1", panel_count=panel_count
    )
    collocation = _finite_vector_array(
        panel_collocation_gp1_cgp1_m,
        name="panel_collocation_gp1_cgp1_m",
        panel_count=panel_count,
    )
    vectors = _leg_arrays(
        leg_vectors_gp1_m, name="leg_vectors_gp1_m", panel_count=panel_count
    )
    velocities = _leg_arrays(
        leg_velocities_gp1_m_s,
        name="leg_velocities_gp1_m_s",
        panel_count=panel_count,
    )
    centers = _leg_arrays(
        leg_centers_gp1_cgp1_m,
        name="leg_centers_gp1_cgp1_m",
        panel_count=panel_count,
    )
    topology_columns = _topology_columns(topology, panel_count=panel_count)
    wing_maps = _wing_index_maps(topology_columns)
    active = _active_panel_array(active_current, panel_count=panel_count)
    stripwise_fields = (
        (surface_release_current, "surface_release_current_m2_s"),
        (wake_release_previous_supplied, "wake_release_previous_m2_s"),
        (gamma_lev_current, "gamma_lev_current_m2_s"),
        (gamma_lev_previous, "gamma_lev_previous_m2_s"),
        (active, "active_current"),
    )
    for field, name in stripwise_fields:
        _validate_stripwise_field(field, name=name, wing_maps=wing_maps)

    wake_release_previous = (
        np.zeros(panel_count, dtype=float)
        if step == 0
        else wake_release_previous_supplied
    )

    strength_deltas = _effective_strength_deltas(
        surface_release_current,
        wake_release_previous,
        wing_maps=wing_maps,
        panel_count=panel_count,
    )
    leg_forces: dict[str, np.ndarray] = {}
    leg_moments: dict[str, np.ndarray] = {}
    for leg in LEG_NAMES:
        leg_forces[leg] = _canonicalize_zeros(
            rho
            * strength_deltas[leg][:, None]
            * np.cross(velocities[leg], vectors[leg], axis=-1)
        )
        leg_moments[leg] = _canonicalize_zeros(
            np.cross(centers[leg], leg_forces[leg], axis=-1)
        )

    delta_kj_force = _canonicalize_zeros(
        sum((leg_forces[leg] for leg in LEG_NAMES), np.zeros((panel_count, 3)))
    )
    gamma_lev_rate = np.where(
        active,
        (gamma_lev_current - gamma_lev_previous) / delta_time,
        0.0,
    )
    delta_eq17_force = _canonicalize_zeros(
        -(rho * area[:, None] * normal * gamma_lev_rate[:, None])
    )
    delta_total_force = _canonicalize_zeros(delta_kj_force + delta_eq17_force)
    delta_kj_moment = _canonicalize_zeros(
        sum((leg_moments[leg] for leg in LEG_NAMES), np.zeros((panel_count, 3)))
    )
    delta_eq17_moment = _canonicalize_zeros(
        np.cross(collocation, delta_eq17_force, axis=-1)
    )
    delta_total_moment = _canonicalize_zeros(delta_kj_moment + delta_eq17_moment)

    force_closure = float(
        np.max(
            np.abs(delta_total_force - delta_kj_force - delta_eq17_force),
            initial=0.0,
        )
    )
    moment_closure = float(
        np.max(
            np.abs(delta_total_moment - delta_kj_moment - delta_eq17_moment),
            initial=0.0,
        )
    )
    leg_force_closure = float(
        np.max(
            np.abs(
                delta_kj_force
                - sum(
                    (leg_forces[leg] for leg in LEG_NAMES),
                    np.zeros((panel_count, 3)),
                )
            ),
            initial=0.0,
        )
    )
    leg_moment_closure = float(
        np.max(
            np.abs(
                delta_kj_moment
                - sum(
                    (leg_moments[leg] for leg in LEG_NAMES),
                    np.zeros((panel_count, 3)),
                )
            ),
            initial=0.0,
        )
    )

    return {
        "delta_kutta_joukowski_force_gp1_n": delta_kj_force,
        "delta_eq17_force_gp1_n": delta_eq17_force,
        "delta_total_force_gp1_n": delta_total_force,
        "delta_kutta_joukowski_moment_gp1_cgp1_nm": delta_kj_moment,
        "delta_eq17_moment_gp1_cgp1_nm": delta_eq17_moment,
        "delta_total_moment_gp1_cgp1_nm": delta_total_moment,
        "delta_kutta_joukowski_force_total_gp1_n": _canonicalize_zeros(
            np.sum(delta_kj_force, axis=0)
        ),
        "delta_eq17_force_total_gp1_n": _canonicalize_zeros(
            np.sum(delta_eq17_force, axis=0)
        ),
        "delta_total_force_total_gp1_n": _canonicalize_zeros(
            np.sum(delta_total_force, axis=0)
        ),
        "delta_kutta_joukowski_moment_total_gp1_cgp1_nm": _canonicalize_zeros(
            np.sum(delta_kj_moment, axis=0)
        ),
        "delta_eq17_moment_total_gp1_cgp1_nm": _canonicalize_zeros(
            np.sum(delta_eq17_moment, axis=0)
        ),
        "delta_total_moment_total_gp1_cgp1_nm": _canonicalize_zeros(
            np.sum(delta_total_moment, axis=0)
        ),
        "leg_ledger": {
            leg: {
                "effective_strength_delta_m2_s": _canonicalize_zeros(
                    strength_deltas[leg]
                ),
                "delta_force_gp1_n": leg_forces[leg].copy(),
                "delta_moment_gp1_cgp1_nm": leg_moments[leg].copy(),
            }
            for leg in LEG_NAMES
        },
        "closure": {
            "force_component_max_abs_n": force_closure,
            "moment_component_max_abs_nm": moment_closure,
            "kj_leg_force_max_abs_n": leg_force_closure,
            "kj_leg_moment_max_abs_nm": leg_moment_closure,
        },
    }


__all__ = ["LEG_NAMES", "calculate_native_surface_release_load_correction"]
