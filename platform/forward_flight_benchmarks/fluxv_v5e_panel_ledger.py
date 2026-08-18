"""Read-only per-panel UVLM load ledger for the FluxV v5e shadow model.

The ledger isolates the two line items already present in PteraSoftware's
unsteady UVLM force calculation without changing either one::

    F_total = F_KJ + F_dGamma
    F_dGamma = -rho * (Gamma_n - Gamma_{n-1}) * area * normal / dt

``PanelLedgerUVPMHybridSolver`` calls the existing augmented solver first and
only then copies its solved state into a ledger.  Setting
``record_panel_ledger=False`` is therefore an exact parent-solver path.  This
module deliberately does not choose an equivalent strip circulation and does
not implement an ULLT closure; strip output retains every chordwise circulation
for the caller to make that choice explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pterasoftware import _transformations

from .augmented_uvpm import AugmentedUVPMHybridSolver


PANEL_LEDGER_CLOSURE_ATOL_N = 1.0e-12


_PANEL_HISTORY_KEYS = (
    "panel_total_force_gp1_n",
    "ptera_dgamma_force_gp1_n",
    "kutta_joukowski_force_gp1_n",
    "panel_total_force_w_n",
    "ptera_dgamma_force_w_n",
    "kutta_joukowski_force_w_n",
    "gamma_current_m2_s",
    "gamma_previous_m2_s",
    "panel_area_m2",
    "panel_normal_gp1",
    "panel_normal_w",
    "collocation_gp1_m",
    "collocation_w_m",
    "frame_transform_gp1_to_w",
)


def _copy_topology(airplanes: list[Any]) -> dict[str, np.ndarray]:
    """Return global-panel topology in Ptera's flattened panel order."""

    airplane_indices: list[int] = []
    wing_indices: list[int] = []
    chordwise_indices: list[int] = []
    spanwise_indices: list[int] = []
    global_indices: list[int] = []
    global_position = 0
    for airplane_index, airplane in enumerate(airplanes):
        for wing_index, wing in enumerate(airplane.wings):
            panels = np.asarray(wing.panels, dtype=object)
            if panels.ndim != 2 or panels.size == 0:
                raise ValueError(
                    "each aerodynamic wing must have a nonempty panel grid"
                )
            for chordwise_index in range(panels.shape[0]):
                for spanwise_index in range(panels.shape[1]):
                    panel = panels[chordwise_index, spanwise_index]
                    local_chord = getattr(panel, "local_chordwise_position", None)
                    local_span = getattr(panel, "local_spanwise_position", None)
                    if local_chord is not None and int(local_chord) != chordwise_index:
                        raise ValueError(
                            "panel chordwise topology disagrees with its grid"
                        )
                    if local_span is not None and int(local_span) != spanwise_index:
                        raise ValueError(
                            "panel spanwise topology disagrees with its grid"
                        )
                    global_indices.append(global_position)
                    airplane_indices.append(airplane_index)
                    wing_indices.append(wing_index)
                    chordwise_indices.append(chordwise_index)
                    spanwise_indices.append(spanwise_index)
                    global_position += 1
    return {
        "panel_global_index": np.asarray(global_indices, dtype=int),
        "airplane_index": np.asarray(airplane_indices, dtype=int),
        "wing_index": np.asarray(wing_indices, dtype=int),
        "chordwise_index": np.asarray(chordwise_indices, dtype=int),
        "spanwise_index": np.asarray(spanwise_indices, dtype=int),
    }


def _panel_force_array(airplanes: list[Any]) -> np.ndarray:
    values: list[np.ndarray] = []
    for airplane in airplanes:
        for wing in airplane.wings:
            for panel in np.ravel(wing.panels):
                force = np.asarray(panel.forces_GP1, dtype=float)
                if force.shape != (3,):
                    raise ValueError("each solved panel force must have shape (3,)")
                values.append(force.copy())
    if not values:
        raise ValueError("solver contains no aerodynamic panels")
    return np.asarray(values, dtype=float)


def _maximum_component_identity_error(
    total: np.ndarray,
    dgamma: np.ndarray,
    kutta_joukowski: np.ndarray,
) -> float:
    return float(np.max(np.abs(total - dgamma - kutta_joukowski), initial=0.0))


def _maximum_group_identity_error(
    total: np.ndarray,
    dgamma: np.ndarray,
    kutta_joukowski: np.ndarray,
    group_keys: list[tuple[int, ...]],
) -> float:
    maximum = 0.0
    for key in dict.fromkeys(group_keys):
        mask = np.asarray([candidate == key for candidate in group_keys], dtype=bool)
        maximum = max(
            maximum,
            _maximum_component_identity_error(
                np.sum(total[mask], axis=0),
                np.sum(dgamma[mask], axis=0),
                np.sum(kutta_joukowski[mask], axis=0),
            ),
        )
    return float(maximum)


def _closure_diagnostics(
    total_gp1: np.ndarray,
    dgamma_gp1: np.ndarray,
    kutta_joukowski_gp1: np.ndarray,
    total_w: np.ndarray,
    dgamma_w: np.ndarray,
    kutta_joukowski_w: np.ndarray,
    topology: dict[str, np.ndarray],
) -> dict[str, float]:
    airplanes = np.asarray(topology["airplane_index"], dtype=int)
    wings = np.asarray(topology["wing_index"], dtype=int)
    spans = np.asarray(topology["spanwise_index"], dtype=int)
    strip_keys = list(zip(airplanes, wings, spans, strict=True))
    airplane_keys = [(int(index),) for index in airplanes]
    diagnostics = {
        "per_panel_gp1_max_abs_n": _maximum_component_identity_error(
            total_gp1, dgamma_gp1, kutta_joukowski_gp1
        ),
        "per_panel_w_max_abs_n": _maximum_component_identity_error(
            total_w, dgamma_w, kutta_joukowski_w
        ),
        "per_strip_gp1_max_abs_n": _maximum_group_identity_error(
            total_gp1, dgamma_gp1, kutta_joukowski_gp1, strip_keys
        ),
        "per_strip_w_max_abs_n": _maximum_group_identity_error(
            total_w, dgamma_w, kutta_joukowski_w, strip_keys
        ),
        "per_airplane_gp1_max_abs_n": _maximum_group_identity_error(
            total_gp1, dgamma_gp1, kutta_joukowski_gp1, airplane_keys
        ),
        "per_airplane_w_max_abs_n": _maximum_group_identity_error(
            total_w, dgamma_w, kutta_joukowski_w, airplane_keys
        ),
    }
    diagnostics["all_levels_max_abs_n"] = float(max(diagnostics.values()))
    return diagnostics


def _validate_panel_inputs(
    *,
    current_gamma: np.ndarray,
    previous_gamma: np.ndarray,
    panel_area: np.ndarray,
    panel_normal: np.ndarray,
    collocation: np.ndarray,
    transform: np.ndarray,
    panel_count: int,
    rho_kg_m3: float,
    delta_time_s: float,
) -> None:
    expected_vectors = (panel_count, 3)
    if current_gamma.shape != (panel_count,) or previous_gamma.shape != (panel_count,):
        raise ValueError("bound-circulation arrays do not match panel topology")
    if panel_area.shape != (panel_count,):
        raise ValueError("panel-area array does not match panel topology")
    if panel_normal.shape != expected_vectors or collocation.shape != expected_vectors:
        raise ValueError("panel geometry arrays do not match panel topology")
    if transform.shape != (4, 4):
        raise ValueError("GP1-to-wind frame transform must have shape (4, 4)")
    arrays = (
        current_gamma,
        previous_gamma,
        panel_area,
        panel_normal,
        collocation,
        transform,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("panel ledger inputs must be finite")
    if np.any(panel_area <= 0.0):
        raise ValueError("panel areas must be positive")
    if not np.isfinite(rho_kg_m3) or rho_kg_m3 <= 0.0:
        raise ValueError("air density must be finite and positive")
    if not np.isfinite(delta_time_s) or delta_time_s <= 0.0:
        raise ValueError("solver time step must be finite and positive")


class PanelLedgerUVPMHybridSolver(AugmentedUVPMHybridSolver):
    """Augmented UVPM solver with an optional read-only panel load ledger."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._record_panel_ledger = bool(kwargs.pop("record_panel_ledger", True))
        super().__init__(*args, **kwargs)
        self.panel_load_ledger: list[dict[str, Any]] = []

    def _calculate_loads(self) -> None:
        """Run the parent solver, then copy its exact solved panel line items."""

        if not self._record_panel_ledger:
            super()._calculate_loads()
            return

        current_gamma = np.asarray(
            self._current_bound_vortex_strengths, dtype=float
        ).copy()
        previous_gamma = np.asarray(
            self._last_bound_vortex_strengths, dtype=float
        ).copy()
        panel_area = np.asarray(self.panel_areas, dtype=float).copy()
        panel_normal_gp1 = np.asarray(self.stackUnitNormals_GP1, dtype=float).copy()
        collocation_gp1 = np.asarray(self.stackCpp_GP1_CgP1, dtype=float).copy()
        topology = _copy_topology(self.current_airplanes)
        transform_gp1_to_w = np.asarray(
            self.current_operating_point.T_pas_GP1_CgP1_to_W_CgP1, dtype=float
        ).copy()
        rho_kg_m3 = float(self.current_operating_point.rho)
        delta_time_s = float(self.delta_time)

        super()._calculate_loads()

        total_force_gp1 = _panel_force_array(self.current_airplanes)
        panel_count = total_force_gp1.shape[0]
        _validate_panel_inputs(
            current_gamma=current_gamma,
            previous_gamma=previous_gamma,
            panel_area=panel_area,
            panel_normal=panel_normal_gp1,
            collocation=collocation_gp1,
            transform=transform_gp1_to_w,
            panel_count=panel_count,
            rho_kg_m3=rho_kg_m3,
            delta_time_s=delta_time_s,
        )
        if topology["panel_global_index"].size != panel_count:
            raise ValueError("panel topology does not match solved panel forces")

        ptera_dgamma_force_gp1 = -(
            rho_kg_m3
            * (current_gamma - previous_gamma)[:, None]
            * panel_area[:, None]
            * panel_normal_gp1
            / delta_time_s
        )
        kutta_joukowski_force_gp1 = total_force_gp1 - ptera_dgamma_force_gp1
        total_force_w = _transformations.apply_T_to_vectors(
            transform_gp1_to_w, total_force_gp1, has_point=False
        )
        ptera_dgamma_force_w = _transformations.apply_T_to_vectors(
            transform_gp1_to_w, ptera_dgamma_force_gp1, has_point=False
        )
        kutta_joukowski_force_w = _transformations.apply_T_to_vectors(
            transform_gp1_to_w, kutta_joukowski_force_gp1, has_point=False
        )
        panel_normal_w = _transformations.apply_T_to_vectors(
            transform_gp1_to_w, panel_normal_gp1, has_point=False
        )
        collocation_w = _transformations.apply_T_to_vectors(
            transform_gp1_to_w, collocation_gp1, has_point=True
        )
        closure = _closure_diagnostics(
            total_force_gp1,
            ptera_dgamma_force_gp1,
            kutta_joukowski_force_gp1,
            total_force_w,
            ptera_dgamma_force_w,
            kutta_joukowski_force_w,
            topology,
        )
        if closure["all_levels_max_abs_n"] >= PANEL_LEDGER_CLOSURE_ATOL_N:
            raise FloatingPointError("panel load-ledger component identity failed")

        self.panel_load_ledger.append(
            {
                "step": int(self._current_step),
                "rho_kg_m3": rho_kg_m3,
                "delta_time_s": delta_time_s,
                "panel_total_force_gp1_n": total_force_gp1.copy(),
                "ptera_dgamma_force_gp1_n": ptera_dgamma_force_gp1.copy(),
                "kutta_joukowski_force_gp1_n": kutta_joukowski_force_gp1.copy(),
                "panel_total_force_w_n": total_force_w.copy(),
                "ptera_dgamma_force_w_n": ptera_dgamma_force_w.copy(),
                "kutta_joukowski_force_w_n": kutta_joukowski_force_w.copy(),
                "gamma_current_m2_s": current_gamma.copy(),
                "gamma_previous_m2_s": previous_gamma.copy(),
                "panel_area_m2": panel_area.copy(),
                "panel_normal_gp1": panel_normal_gp1.copy(),
                "panel_normal_w": panel_normal_w.copy(),
                "collocation_gp1_m": collocation_gp1.copy(),
                "collocation_w_m": collocation_w.copy(),
                "frame_transform_gp1_to_w": transform_gp1_to_w.copy(),
                "topology": {key: value.copy() for key, value in topology.items()},
                "closure": closure,
            }
        )


def _coherent_cycle_bounds(
    rows: list[dict[str, Any]], *, delta_time_s: float, period_s: float
) -> tuple[list[dict[str, Any]], int, int, int]:
    if not rows:
        raise FloatingPointError("solver produced no panel load ledger")
    if not np.isfinite(delta_time_s) or delta_time_s <= 0.0:
        raise ValueError("delta_time_s must be finite and positive")
    if not np.isfinite(period_s) or period_s <= 0.0:
        raise ValueError("period_s must be finite and positive")
    samples_float = period_s / delta_time_s
    samples_per_cycle = int(round(samples_float))
    if samples_per_cycle < 2 or not np.isclose(
        samples_float, samples_per_cycle, atol=1.0e-10, rtol=0.0
    ):
        raise ValueError("period must contain an integer number of solver steps")

    ordered = sorted(rows, key=lambda row: int(row["step"]))
    steps_all = np.asarray([int(row["step"]) for row in ordered], dtype=int)
    if np.unique(steps_all).size != steps_all.size:
        raise FloatingPointError("panel ledger contains duplicate solver steps")
    last_step = int(steps_all[-1])
    phase_at_last = last_step * delta_time_s / period_s
    cycle_end = (
        last_step
        if np.isclose(phase_at_last, round(phase_at_last), atol=1.0e-10, rtol=0.0)
        else last_step + 1
    )
    cycle_start = cycle_end - samples_per_cycle
    selected = [row for row in ordered if cycle_start <= int(row["step"]) < cycle_end]
    expected_steps = np.arange(cycle_start, cycle_end, dtype=int)
    selected_steps = np.asarray([int(row["step"]) for row in selected], dtype=int)
    if not np.array_equal(selected_steps, expected_steps):
        raise FloatingPointError("panel ledger does not contain one coherent cycle")
    return selected, cycle_start, cycle_end, samples_per_cycle


def extract_last_coherent_panel_cycle(
    solver: PanelLedgerUVPMHybridSolver,
    *,
    period_s: float,
    movement: Any | None = None,
    delta_time_s: float | None = None,
) -> dict[str, Any]:
    """Stack the last complete, endpoint-safe panel-ledger cycle.

    ``movement.delta_time`` is accepted for parity with the existing benchmark
    extractors.  Passing ``delta_time_s`` directly is useful for offline ledger
    audits; exactly one source of the time step must be supplied.
    """

    if (movement is None) == (delta_time_s is None):
        raise ValueError("provide exactly one of movement or delta_time_s")
    if movement is not None:
        delta_time_s = float(movement.delta_time)
    assert delta_time_s is not None
    selected, cycle_start, cycle_end, samples_per_cycle = _coherent_cycle_bounds(
        solver.panel_load_ledger,
        delta_time_s=float(delta_time_s),
        period_s=period_s,
    )

    recorded_delta_times = np.asarray(
        [float(row["delta_time_s"]) for row in selected], dtype=float
    )
    recorded_densities = np.asarray(
        [float(row["rho_kg_m3"]) for row in selected], dtype=float
    )
    if not np.all(np.isfinite(recorded_delta_times)) or not np.allclose(
        recorded_delta_times, float(delta_time_s), atol=1.0e-15, rtol=0.0
    ):
        raise FloatingPointError("recorded solver time step changed within the cycle")
    if not np.all(np.isfinite(recorded_densities)) or np.any(recorded_densities <= 0.0):
        raise FloatingPointError("recorded air density is invalid")

    reference_topology = selected[0]["topology"]
    topology: dict[str, np.ndarray] = {}
    for key, reference in reference_topology.items():
        reference_array = np.asarray(reference, dtype=int)
        if not all(
            np.array_equal(np.asarray(row["topology"][key], dtype=int), reference_array)
            for row in selected
        ):
            raise FloatingPointError(
                "panel topology changed within the extracted cycle"
            )
        topology[key] = reference_array.copy()

    stacked: dict[str, np.ndarray] = {}
    for key in _PANEL_HISTORY_KEYS:
        values = [np.asarray(row[key]) for row in selected]
        shape = values[0].shape
        if not all(value.shape == shape for value in values):
            raise FloatingPointError(f"panel ledger field {key} changed shape")
        stacked[key] = np.stack(values, axis=0)
    if not all(np.all(np.isfinite(value)) for value in stacked.values()):
        raise FloatingPointError("panel cycle contains non-finite ledger values")

    steps = np.arange(cycle_start, cycle_end, dtype=int)
    phase = np.mod(steps * float(delta_time_s) / period_s, 1.0)
    maximum_closure = float(
        max(float(row["closure"]["all_levels_max_abs_n"]) for row in selected)
    )
    if maximum_closure >= PANEL_LEDGER_CLOSURE_ATOL_N:
        raise FloatingPointError("extracted panel cycle violates component identity")
    return {
        "step": steps,
        "phase": phase,
        "source_cycle_step_range": [cycle_start, cycle_end - 1],
        "samples_per_cycle": samples_per_cycle,
        "delta_time_s": float(delta_time_s),
        "period_s": float(period_s),
        "rho_kg_m3": recorded_densities,
        "topology": topology,
        "closure_all_levels_max_abs_n": maximum_closure,
        **stacked,
    }


def _ordered_topology_groups(
    topology: dict[str, np.ndarray], keys: tuple[str, ...]
) -> list[tuple[tuple[int, ...], np.ndarray]]:
    columns = [np.asarray(topology[key], dtype=int) for key in keys]
    if not columns or any(column.shape != columns[0].shape for column in columns):
        raise ValueError("invalid panel topology columns")
    rows = list(zip(*columns, strict=True))
    groups: list[tuple[tuple[int, ...], np.ndarray]] = []
    for key in dict.fromkeys(rows):
        indices = np.asarray(
            [index for index, candidate in enumerate(rows) if candidate == key],
            dtype=int,
        )
        groups.append((tuple(int(value) for value in key), indices))
    return groups


def aggregate_panel_cycle_to_strips(cycle: dict[str, Any]) -> dict[str, Any]:
    """Aggregate force line items while retaining chordwise circulation.

    A strip is ``(airplane, wing, spanwise panel index)``.  Force components are
    summed over chord; geometry numerators and all chordwise Gamma values remain
    explicit.  No area-weighted, trailing-edge, or summed equivalent Gamma is
    selected here.
    """

    topology = cycle["topology"]
    time_count = np.asarray(cycle["phase"]).size
    total_gp1 = np.asarray(cycle["panel_total_force_gp1_n"], dtype=float)
    dgamma_gp1 = np.asarray(cycle["ptera_dgamma_force_gp1_n"], dtype=float)
    kutta_joukowski_gp1 = np.asarray(cycle["kutta_joukowski_force_gp1_n"], dtype=float)
    total_w = np.asarray(cycle["panel_total_force_w_n"], dtype=float)
    dgamma_w = np.asarray(cycle["ptera_dgamma_force_w_n"], dtype=float)
    kutta_joukowski_w = np.asarray(cycle["kutta_joukowski_force_w_n"], dtype=float)
    if total_gp1.ndim != 3 or total_gp1.shape[0] != time_count:
        raise ValueError("panel cycle forces must have shape (time, panel, 3)")
    force_arrays = (
        total_gp1,
        dgamma_gp1,
        kutta_joukowski_gp1,
        total_w,
        dgamma_w,
        kutta_joukowski_w,
    )
    if any(array.shape != total_gp1.shape for array in force_arrays):
        raise ValueError("panel force histories are not aligned")
    panel_closure = max(
        _maximum_component_identity_error(total_gp1, dgamma_gp1, kutta_joukowski_gp1),
        _maximum_component_identity_error(total_w, dgamma_w, kutta_joukowski_w),
    )
    if panel_closure >= PANEL_LEDGER_CLOSURE_ATOL_N:
        raise FloatingPointError("per-panel component identity failed")

    chordwise = np.asarray(topology["chordwise_index"], dtype=int)
    strip_rows: list[dict[str, Any]] = []
    strip_closure = 0.0
    strip_groups = _ordered_topology_groups(
        topology, ("airplane_index", "wing_index", "spanwise_index")
    )
    for (airplane_index, wing_index, spanwise_index), unsorted_indices in strip_groups:
        order = np.argsort(chordwise[unsorted_indices], kind="stable")
        indices = unsorted_indices[order]
        chord_indices = chordwise[indices]
        if not np.array_equal(chord_indices, np.arange(chord_indices.size)):
            raise ValueError("strip chordwise topology is incomplete or duplicated")
        area = np.asarray(cycle["panel_area_m2"], dtype=float)[:, indices]
        normal_gp1 = np.asarray(cycle["panel_normal_gp1"], dtype=float)[:, indices]
        normal_w = np.asarray(cycle["panel_normal_w"], dtype=float)[:, indices]
        collocation_gp1 = np.asarray(cycle["collocation_gp1_m"], dtype=float)[
            :, indices
        ]
        collocation_w = np.asarray(cycle["collocation_w_m"], dtype=float)[:, indices]

        strip_total_gp1 = np.sum(total_gp1[:, indices], axis=1)
        strip_dgamma_gp1 = np.sum(dgamma_gp1[:, indices], axis=1)
        strip_kj_gp1 = np.sum(kutta_joukowski_gp1[:, indices], axis=1)
        strip_total_w = np.sum(total_w[:, indices], axis=1)
        strip_dgamma_w = np.sum(dgamma_w[:, indices], axis=1)
        strip_kj_w = np.sum(kutta_joukowski_w[:, indices], axis=1)
        closure_gp1 = _maximum_component_identity_error(
            strip_total_gp1, strip_dgamma_gp1, strip_kj_gp1
        )
        closure_w = _maximum_component_identity_error(
            strip_total_w, strip_dgamma_w, strip_kj_w
        )
        strip_closure = max(strip_closure, closure_gp1, closure_w)
        strip_rows.append(
            {
                "airplane_index": airplane_index,
                "wing_index": wing_index,
                "spanwise_index": spanwise_index,
                "panel_global_indices": np.asarray(
                    topology["panel_global_index"], dtype=int
                )[indices].copy(),
                "chordwise_indices": chord_indices.copy(),
                "gamma_current_by_chord_m2_s": np.asarray(
                    cycle["gamma_current_m2_s"], dtype=float
                )[:, indices].copy(),
                "gamma_previous_by_chord_m2_s": np.asarray(
                    cycle["gamma_previous_m2_s"], dtype=float
                )[:, indices].copy(),
                "panel_area_by_chord_m2": area.copy(),
                "panel_normal_gp1_by_chord": normal_gp1.copy(),
                "panel_normal_w_by_chord": normal_w.copy(),
                "collocation_gp1_by_chord_m": collocation_gp1.copy(),
                "collocation_w_by_chord_m": collocation_w.copy(),
                "strip_area_m2": np.sum(area, axis=1),
                "area_normal_sum_gp1_m2": np.sum(area[..., None] * normal_gp1, axis=1),
                "area_normal_sum_w_m2": np.sum(area[..., None] * normal_w, axis=1),
                "area_collocation_sum_gp1_m3": np.sum(
                    area[..., None] * collocation_gp1, axis=1
                ),
                "area_collocation_sum_w_m3": np.sum(
                    area[..., None] * collocation_w, axis=1
                ),
                "total_force_gp1_n": strip_total_gp1,
                "ptera_dgamma_force_gp1_n": strip_dgamma_gp1,
                "kutta_joukowski_force_gp1_n": strip_kj_gp1,
                "total_force_w_n": strip_total_w,
                "ptera_dgamma_force_w_n": strip_dgamma_w,
                "kutta_joukowski_force_w_n": strip_kj_w,
                "closure_gp1_max_abs_n": closure_gp1,
                "closure_w_max_abs_n": closure_w,
            }
        )

    airplane_rows: list[dict[str, Any]] = []
    airplane_closure = 0.0
    for (airplane_index,), indices in _ordered_topology_groups(
        topology, ("airplane_index",)
    ):
        airplane_total_gp1 = np.sum(total_gp1[:, indices], axis=1)
        airplane_dgamma_gp1 = np.sum(dgamma_gp1[:, indices], axis=1)
        airplane_kj_gp1 = np.sum(kutta_joukowski_gp1[:, indices], axis=1)
        airplane_total_w = np.sum(total_w[:, indices], axis=1)
        airplane_dgamma_w = np.sum(dgamma_w[:, indices], axis=1)
        airplane_kj_w = np.sum(kutta_joukowski_w[:, indices], axis=1)
        closure_gp1 = _maximum_component_identity_error(
            airplane_total_gp1, airplane_dgamma_gp1, airplane_kj_gp1
        )
        closure_w = _maximum_component_identity_error(
            airplane_total_w, airplane_dgamma_w, airplane_kj_w
        )
        airplane_closure = max(airplane_closure, closure_gp1, closure_w)
        airplane_rows.append(
            {
                "airplane_index": airplane_index,
                "total_force_gp1_n": airplane_total_gp1,
                "ptera_dgamma_force_gp1_n": airplane_dgamma_gp1,
                "kutta_joukowski_force_gp1_n": airplane_kj_gp1,
                "total_force_w_n": airplane_total_w,
                "ptera_dgamma_force_w_n": airplane_dgamma_w,
                "kutta_joukowski_force_w_n": airplane_kj_w,
                "closure_gp1_max_abs_n": closure_gp1,
                "closure_w_max_abs_n": closure_w,
            }
        )
    closure = max(panel_closure, strip_closure, airplane_closure)
    if closure >= PANEL_LEDGER_CLOSURE_ATOL_N:
        raise FloatingPointError("strip or airplane component identity failed")
    return {
        "step": np.asarray(cycle["step"], dtype=int).copy(),
        "phase": np.asarray(cycle["phase"], dtype=float).copy(),
        "frame_transform_gp1_to_w": np.asarray(
            cycle["frame_transform_gp1_to_w"], dtype=float
        ).copy(),
        "strips": strip_rows,
        "airplanes": airplane_rows,
        "closure_per_panel_max_abs_n": float(panel_closure),
        "closure_per_strip_max_abs_n": float(strip_closure),
        "closure_per_airplane_max_abs_n": float(airplane_closure),
        "equivalent_gamma_rule": "not_selected_chordwise_values_retained",
    }
