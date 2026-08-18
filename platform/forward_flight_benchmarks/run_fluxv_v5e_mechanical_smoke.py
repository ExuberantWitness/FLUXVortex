"""Run observation-free mechanical smoke gates for the FluxV v5e shadow.

The runner executes the actual prescribed-wake UVLM load path for one frozen
representative condition from each current benchmark family:

* Yang 2025 at 15 degrees installation angle;
* Izraelevitz/Scherer Figure 14 at theta=15 degrees and psi=60 degrees; and
* Baik 2012 W2.

It then replaces exactly Ptera's old ``dGamma/dt`` strip-force line item by
the source-frozen one-state phi/Gamma mismatch and an independently computed
kinematic added-mass force.  No experimental file is opened and no prediction
is scored.  The emitted force histories are mechanical evidence only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pterasoftware as ps

from forward_flight_benchmarks.augmented_uvpm import (
    added_mass_aspect_ratio_factor,
)
from forward_flight_benchmarks.baik2012 import (
    BAIK_2012_CASES,
    build_baik_movement,
)
from forward_flight_benchmarks.cases import (
    IZRAELEVITZ_2017_FIG14_SCHERER,
    YANG_2025,
)
from forward_flight_benchmarks.fluxv_v5e_line_item import (
    CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE,
    ullt_to_uvlm_line_item_shadow,
)
from forward_flight_benchmarks.fluxv_v5e_panel_ledger import (
    PanelLedgerUVPMHybridSolver,
    aggregate_panel_cycle_to_strips,
    extract_last_coherent_panel_cycle,
)
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_scherer_movement,
    build_yang2025_movement,
)
from forward_flight_benchmarks.ullt_attached import (
    DEFAULT_ULLT_PARAMETERS,
    _added_mass_force_history,
    _periodic_derivative,
    _strip_geometry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "fluxv_v5c_nextgen_20260814"
)
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5e_mechanical_smoke"

WARMUP_CYCLES = 80
CLOSURE_TOLERANCE_N = 1.0e-12
STATE_PERIODIC_TOLERANCE = 1.0e-8
GEOMETRY_TOLERANCE = 1.0e-12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty smoke CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _case_specs() -> tuple[dict[str, Any], ...]:
    baik = BAIK_2012_CASES["W2"]
    return (
        {
            "benchmark": "yang2025",
            "condition": "aoa15_smoke",
            "period_s": YANG_2025.period_s,
            "rho_kg_m3": YANG_2025.rho_kg_m3,
            "freestream_m_s": YANG_2025.freestream_m_s,
            "area_m2": YANG_2025.area_m2,
            "aspect_ratio": YANG_2025.aspect_ratio,
            "case_manifest": YANG_2025.manifest(),
            "builder": lambda: build_yang2025_movement(15.0, "smoke"),
        },
        {
            "benchmark": "izraelevitz2017_fig14",
            "condition": "theta15_psi60_smoke",
            "period_s": IZRAELEVITZ_2017_FIG14_SCHERER.period_s,
            "rho_kg_m3": IZRAELEVITZ_2017_FIG14_SCHERER.rho_kg_m3,
            "freestream_m_s": IZRAELEVITZ_2017_FIG14_SCHERER.freestream_m_s,
            "area_m2": IZRAELEVITZ_2017_FIG14_SCHERER.area_m2,
            "aspect_ratio": IZRAELEVITZ_2017_FIG14_SCHERER.aspect_ratio,
            "case_manifest": IZRAELEVITZ_2017_FIG14_SCHERER.manifest(),
            "builder": lambda: build_izraelevitz_scherer_movement(15.0, 60.0, "smoke"),
        },
        {
            "benchmark": "baik2012",
            "condition": "W2_smoke",
            "period_s": baik.period_s,
            "rho_kg_m3": baik.rho_kg_m3,
            "freestream_m_s": baik.freestream_m_s,
            "area_m2": baik.area_m2,
            "aspect_ratio": baik.geometric_aspect_ratio,
            "case_manifest": baik.manifest(),
            "builder": lambda: build_baik_movement(baik, "smoke"),
        },
    )


def _rotation_history(transform_history: np.ndarray) -> np.ndarray:
    transforms = np.asarray(transform_history, dtype=float)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("wind transforms must have shape (time, 4, 4)")
    rotations = transforms[:, :3, :3]
    identity = np.einsum("tji,tjk->tik", rotations, rotations)
    error = float(np.max(np.abs(identity - np.eye(3))))
    if error >= GEOMETRY_TOLERANCE:
        raise FloatingPointError("GP1-to-wind transform is not orthonormal")
    return rotations


def _vectors_to_wind(rotations: np.ndarray, values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=float)
    if vectors.ndim != 3 or vectors.shape[0] != rotations.shape[0]:
        raise ValueError("force history and wind transforms do not align")
    return np.einsum("tij,tsj->tsi", rotations, vectors)


def _geometry_histories(
    movement: Any,
    cycle: dict[str, Any],
    aggregation: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return selected strip geometry after strict ledger-order validation."""

    if len(movement.airplanes) != 1:
        raise ValueError("v5e smoke supports exactly one airplane movement family")
    airplane_history = movement.airplanes[0]
    geometry_all = [_strip_geometry(airplane) for airplane in airplane_history]
    strip_counts = {int(item["areas"].size) for item in geometry_all}
    if len(strip_counts) != 1:
        raise FloatingPointError("movement strip topology changed over time")
    strip_count = strip_counts.pop()
    if strip_count != len(aggregation["strips"]):
        raise FloatingPointError("movement and UVLM ledger strip counts disagree")

    first_airplane = airplane_history[0]
    expected_keys = [
        (0, wing_index, spanwise_index)
        for wing_index, wing in enumerate(first_airplane.wings)
        for spanwise_index in range(wing.panels.shape[1])
    ]
    observed_keys = [
        (
            int(row["airplane_index"]),
            int(row["wing_index"]),
            int(row["spanwise_index"]),
        )
        for row in aggregation["strips"]
    ]
    order_match = expected_keys == observed_keys

    steps = np.asarray(cycle["step"], dtype=int)
    if np.any(steps < 0) or np.any(steps >= len(geometry_all)):
        raise FloatingPointError("ledger cycle steps lie outside movement geometry")
    selected = [geometry_all[int(step)] for step in steps]
    geometry = {
        key: np.stack([item[key] for item in selected], axis=0)
        for key in (
            "chord_lengths",
            "strip_widths",
            "normal_axes",
            "span_axes",
            "three_quarter_centres",
            "areas",
        )
    }

    ledger_area = np.stack(
        [
            np.asarray(row["strip_area_m2"], dtype=float)
            for row in aggregation["strips"]
        ],
        axis=1,
    )
    ledger_normal_gp1 = np.stack(
        [
            np.asarray(row["area_normal_sum_gp1_m2"], dtype=float)
            / np.asarray(row["strip_area_m2"], dtype=float)[:, None]
            for row in aggregation["strips"]
        ],
        axis=1,
    )
    area_error = float(np.max(np.abs(geometry["areas"] - ledger_area)))
    normal_error = float(np.max(np.abs(geometry["normal_axes"] - ledger_normal_gp1)))
    normal_unit_error = float(
        np.max(np.abs(np.linalg.norm(geometry["normal_axes"], axis=2) - 1.0))
    )
    diagnostics = {
        "strip_order_exact": bool(order_match),
        "strip_area_max_abs_m2": area_error,
        "strip_normal_max_abs": normal_error,
        "strip_normal_unit_max_abs": normal_unit_error,
    }
    if not order_match:
        raise FloatingPointError("movement and ledger strip orders disagree")
    if max(area_error, normal_error, normal_unit_error) >= GEOMETRY_TOLERANCE:
        raise FloatingPointError("movement and ledger strip geometry disagree")

    geometry["all_geometry"] = np.asarray(geometry_all, dtype=object)
    return geometry, diagnostics


def _run_case(
    spec: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    movement, movement_metadata = spec["builder"]()
    if len(movement.airplane_movements) != 1:
        raise ValueError("v5e smoke is fail-closed for multi-airplane problems")
    problem = ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)
    solver = PanelLedgerUVPMHybridSolver(
        problem,
        max_particles=20000,
        stretch=False,
        free_wake=False,
        record_vpm_particles=False,
        record_panel_ledger=True,
    )
    started = time.perf_counter()
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    runtime_s = time.perf_counter() - started

    period_s = float(spec["period_s"])
    delta_time_s = float(movement.delta_time)
    steps_per_cycle = int(round(period_s / delta_time_s))
    cycle = extract_last_coherent_panel_cycle(
        solver,
        period_s=period_s,
        movement=movement,
    )
    aggregation = aggregate_panel_cycle_to_strips(cycle)
    geometry, geometry_diagnostics = _geometry_histories(movement, cycle, aggregation)

    baseline_gp1 = np.stack(
        [
            np.asarray(row["total_force_gp1_n"], dtype=float)
            for row in aggregation["strips"]
        ],
        axis=1,
    )
    f_kj_gp1 = np.stack(
        [
            np.asarray(row["kutta_joukowski_force_gp1_n"], dtype=float)
            for row in aggregation["strips"]
        ],
        axis=1,
    )
    old_fd_gamma_gp1 = np.stack(
        [
            np.asarray(row["ptera_dgamma_force_gp1_n"], dtype=float)
            for row in aggregation["strips"]
        ],
        axis=1,
    )

    geometry_all = list(geometry.pop("all_geometry"))
    positions_all = np.stack(
        [item["three_quarter_centres"] for item in geometry_all], axis=0
    )
    body_velocity_all = _periodic_derivative(
        positions_all, delta_time_s, steps_per_cycle
    )
    selected_steps = np.asarray(cycle["step"], dtype=int)
    body_velocity = body_velocity_all[selected_steps]
    freestream_vector = np.array([float(spec["freestream_m_s"]), 0.0, 0.0], dtype=float)
    relative_velocity = freestream_vector[None, None, :] - body_velocity
    span_axis = np.asarray(geometry["span_axes"], dtype=float)
    relative_perpendicular = relative_velocity - (
        np.sum(relative_velocity * span_axis, axis=2, keepdims=True) * span_axis
    )
    v_perp = np.linalg.norm(relative_perpendicular, axis=2)
    if np.any(~np.isfinite(v_perp)) or np.any(v_perp <= 0.0):
        raise FloatingPointError("kinematic three-quarter-chord V_perp is invalid")
    lift_direction = np.cross(relative_perpendicular, span_axis) / v_perp[:, :, None]
    lift_direction_unit_error = float(
        np.max(np.abs(np.linalg.norm(lift_direction, axis=2) - 1.0))
    )
    if lift_direction_unit_error >= GEOMETRY_TOLERANCE:
        raise FloatingPointError("kinematic Kutta--Joukowski direction is not unit")
    lift_normal_projection = np.sum(
        lift_direction * np.asarray(geometry["normal_axes"], dtype=float), axis=2
    )
    lift_normal_angle_deg = np.rad2deg(
        np.arccos(np.clip(lift_normal_projection, -1.0, 1.0))
    )

    aspect_ratio = float(spec["aspect_ratio"])
    added_mass_factor = added_mass_aspect_ratio_factor(aspect_ratio)
    added_mass_parameters = replace(
        DEFAULT_ULLT_PARAMETERS,
        flat_plate_added_mass_factor=added_mass_factor,
    )
    added_mass_all_gp1 = _added_mass_force_history(
        geometry_all,
        body_velocity_all,
        freestream_vector,
        float(spec["rho_kg_m3"]),
        delta_time_s,
        steps_per_cycle,
        added_mass_parameters,
    )
    added_mass_gp1 = added_mass_all_gp1[selected_steps]

    closure_inputs = {
        "baseline_total_force_history": baseline_gp1,
        "f_kj_history": f_kj_gp1,
        "strip_lift_direction": lift_direction,
        "chord": geometry["chord_lengths"],
        "strip_width": geometry["strip_widths"],
        "v_perp": v_perp,
        "density": float(spec["rho_kg_m3"]),
        "delta_time": delta_time_s,
        "kinematic_added_mass_force_history": added_mass_gp1,
        "kinematic_added_mass_provenance": (CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE),
    }
    disabled = ullt_to_uvlm_line_item_shadow(
        baseline_gp1,
        f_kj_history=None,
        strip_lift_direction=None,
        chord=None,
        strip_width=None,
        v_perp=None,
        density=None,
        delta_time=None,
        kinematic_added_mass_force_history=None,
        kinematic_added_mass_provenance=None,
        initial_state=None,
        enabled=False,
    )
    disabled_force = np.asarray(disabled["new_force_history"])
    disabled_exact = (
        disabled_force.dtype == baseline_gp1.dtype
        and disabled_force.tobytes() == baseline_gp1.tobytes()
    )

    state: np.ndarray | None = None
    for _ in range(WARMUP_CYCLES):
        warm = ullt_to_uvlm_line_item_shadow(**closure_inputs, initial_state=state)
        state = np.asarray(warm["state"]["final_state"], dtype=float)
    production_initial_state = state.copy()
    line_item = ullt_to_uvlm_line_item_shadow(
        **closure_inputs, initial_state=production_initial_state
    )
    state_periodic_error = float(
        np.max(
            np.abs(
                np.asarray(line_item["state"]["final_state"], dtype=float)
                - production_initial_state
            )
        )
    )

    new_gp1 = np.asarray(line_item["new_force_history"], dtype=float)
    delta_gp1 = new_gp1 - baseline_gp1
    rotations = _rotation_history(cycle["frame_transform_gp1_to_w"])
    baseline_w = _vectors_to_wind(rotations, baseline_gp1)
    f_kj_w = _vectors_to_wind(rotations, f_kj_gp1)
    old_fd_gamma_w = _vectors_to_wind(rotations, old_fd_gamma_gp1)
    added_mass_w = _vectors_to_wind(rotations, added_mass_gp1)
    delta_phi_gamma_w = _vectors_to_wind(
        rotations,
        np.asarray(
            line_item["components"]["delta_phi_gamma_force_history"], dtype=float
        ),
    )
    new_w = _vectors_to_wind(rotations, new_gp1)
    delta_w = _vectors_to_wind(rotations, delta_gp1)

    baseline_airplane_w = np.asarray(
        aggregation["airplanes"][0]["total_force_w_n"], dtype=float
    )
    baseline_transform_error = float(
        np.max(np.abs(np.sum(baseline_w, axis=1) - baseline_airplane_w))
    )
    replacement_identity_error = float(
        line_item["ledger"]["replacement_identity_max_abs_residual"]
    )
    new_closure_error = float(line_item["ledger"]["new_force_closure_max_abs_residual"])
    span_sum_error = float(
        np.max(
            np.abs(
                np.asarray(line_item["span_summed_force_history"], dtype=float)
                - np.sum(new_gp1, axis=1)
            )
        )
    )
    finite_arrays = (
        baseline_gp1,
        f_kj_gp1,
        old_fd_gamma_gp1,
        added_mass_gp1,
        v_perp,
        lift_direction,
        lift_normal_angle_deg,
        new_gp1,
        delta_gp1,
        baseline_w,
        new_w,
        delta_w,
        np.asarray(line_item["state"]["x_history"]),
    )
    finite = all(np.all(np.isfinite(value)) for value in finite_arrays)

    metrics = {
        "panel_closure_max_abs_n": float(cycle["closure_all_levels_max_abs_n"]),
        "strip_closure_max_abs_n": float(aggregation["closure_per_strip_max_abs_n"]),
        "airplane_closure_max_abs_n": float(
            aggregation["closure_per_airplane_max_abs_n"]
        ),
        "baseline_gp1_to_w_max_abs_n": baseline_transform_error,
        "disabled_max_abs_n": float(np.max(np.abs(disabled_force - baseline_gp1))),
        "state_periodic_max_abs": state_periodic_error,
        "replacement_identity_max_abs_n": replacement_identity_error,
        "new_force_closure_max_abs_n": new_closure_error,
        "span_sum_max_abs_n": span_sum_error,
        "v_perp_min_m_s": float(np.min(v_perp)),
        "v_perp_max_m_s": float(np.max(v_perp)),
        "added_mass_factor": added_mass_factor,
        "lift_direction_unit_max_abs": lift_direction_unit_error,
        "lift_direction_surface_normal_projection_min": float(
            np.min(lift_normal_projection)
        ),
        "lift_direction_surface_normal_projection_max": float(
            np.max(lift_normal_projection)
        ),
        "lift_direction_surface_normal_angle_max_deg": float(
            np.max(lift_normal_angle_deg)
        ),
        "kj_lift_direction_identity_max_abs_n": float(
            line_item["ledger"]["kj_lift_direction_identity_max_abs_residual"]
        ),
        "delta_lift_direction_max_abs_n": float(
            line_item["ledger"]["delta_lift_direction_max_abs_residual"]
        ),
        **geometry_diagnostics,
    }
    gates = {
        "panel_closure": metrics["panel_closure_max_abs_n"] < CLOSURE_TOLERANCE_N,
        "strip_closure": metrics["strip_closure_max_abs_n"] < CLOSURE_TOLERANCE_N,
        "airplane_closure": metrics["airplane_closure_max_abs_n"] < CLOSURE_TOLERANCE_N,
        "baseline_wind_transform": metrics["baseline_gp1_to_w_max_abs_n"]
        < CLOSURE_TOLERANCE_N,
        "disabled_bitwise_exact": bool(disabled_exact),
        "state_periodic": metrics["state_periodic_max_abs"] < STATE_PERIODIC_TOLERANCE,
        "finite": bool(finite),
        "replacement_ledger": max(
            metrics["replacement_identity_max_abs_n"],
            metrics["new_force_closure_max_abs_n"],
            metrics["span_sum_max_abs_n"],
            metrics["kj_lift_direction_identity_max_abs_n"],
            metrics["delta_lift_direction_max_abs_n"],
        )
        < CLOSURE_TOLERANCE_N,
        "lift_direction_unit": metrics["lift_direction_unit_max_abs"]
        < GEOMETRY_TOLERANCE,
        "geometry_strip_order": bool(metrics["strip_order_exact"]),
        "geometry_strip_area_normal": max(
            metrics["strip_area_max_abs_m2"],
            metrics["strip_normal_max_abs"],
            metrics["strip_normal_unit_max_abs"],
        )
        < GEOMETRY_TOLERANCE,
    }

    q_area = (
        0.5
        * float(spec["rho_kg_m3"])
        * float(spec["freestream_m_s"]) ** 2
        * float(spec["area_m2"])
    )
    old_airplane_w = np.sum(baseline_w, axis=1)
    new_airplane_w = np.sum(new_w, axis=1)
    delta_airplane_w = np.sum(delta_w, axis=1)
    component_airplane_w = {
        "f_kj": np.sum(f_kj_w, axis=1),
        "old_fd_gamma": np.sum(old_fd_gamma_w, axis=1),
        "delta_phi_gamma": np.sum(delta_phi_gamma_w, axis=1),
        "kinematic_added_mass": np.sum(added_mass_w, axis=1),
    }

    history_rows: list[dict[str, Any]] = []
    phase = np.asarray(cycle["phase"], dtype=float)
    steps = np.asarray(cycle["step"], dtype=int)
    for index in range(phase.size):
        row: dict[str, Any] = {
            "benchmark": spec["benchmark"],
            "condition": spec["condition"],
            "step": int(steps[index]),
            "phase": float(phase[index]),
        }
        for label, force in (
            ("old", old_airplane_w),
            ("lineitem", new_airplane_w),
            ("delta", delta_airplane_w),
            ("f_kj", component_airplane_w["f_kj"]),
            ("old_fd_gamma", component_airplane_w["old_fd_gamma"]),
            ("delta_phi_gamma", component_airplane_w["delta_phi_gamma"]),
            ("kinematic_added_mass", component_airplane_w["kinematic_added_mass"]),
        ):
            row[f"{label}_fx_w_n"] = float(force[index, 0])
            row[f"{label}_fy_w_n"] = float(force[index, 1])
            row[f"{label}_fz_w_n"] = float(force[index, 2])
        row.update(
            old_lift_n=float(-old_airplane_w[index, 2]),
            old_drag_n=float(-old_airplane_w[index, 0]),
            old_CL=float(-old_airplane_w[index, 2] / q_area),
            old_CD=float(-old_airplane_w[index, 0] / q_area),
            lineitem_lift_n=float(-new_airplane_w[index, 2]),
            lineitem_drag_n=float(-new_airplane_w[index, 0]),
            lineitem_CL=float(-new_airplane_w[index, 2] / q_area),
            lineitem_CD=float(-new_airplane_w[index, 0] / q_area),
        )
        history_rows.append(row)

    strip_rows: list[dict[str, Any]] = []
    gamma_eq = np.asarray(line_item["state"]["gamma_eq_history"], dtype=float)
    delta_t_tilde = np.asarray(line_item["state"]["delta_t_tilde_history"], dtype=float)
    y_gamma = np.asarray(line_item["state"]["y_gamma_history"], dtype=float)
    x_history = np.asarray(line_item["state"]["x_history"], dtype=float)
    y_phi = np.asarray(line_item["state"]["y_phi_history"], dtype=float)
    for time_index in range(phase.size):
        for strip_index, strip in enumerate(aggregation["strips"]):
            strip_rows.append(
                {
                    "benchmark": spec["benchmark"],
                    "condition": spec["condition"],
                    "step": int(steps[time_index]),
                    "phase": float(phase[time_index]),
                    "strip_index": strip_index,
                    "airplane_index": int(strip["airplane_index"]),
                    "wing_index": int(strip["wing_index"]),
                    "spanwise_index": int(strip["spanwise_index"]),
                    "chord_m": float(
                        geometry["chord_lengths"][time_index, strip_index]
                    ),
                    "strip_width_m": float(
                        geometry["strip_widths"][time_index, strip_index]
                    ),
                    "v_perp_m_s": float(v_perp[time_index, strip_index]),
                    "lift_direction_x_gp1": float(
                        lift_direction[time_index, strip_index, 0]
                    ),
                    "lift_direction_y_gp1": float(
                        lift_direction[time_index, strip_index, 1]
                    ),
                    "lift_direction_z_gp1": float(
                        lift_direction[time_index, strip_index, 2]
                    ),
                    "lift_direction_surface_normal_projection": float(
                        lift_normal_projection[time_index, strip_index]
                    ),
                    "lift_direction_surface_normal_angle_deg": float(
                        lift_normal_angle_deg[time_index, strip_index]
                    ),
                    "gamma_eq_m2_s": float(gamma_eq[time_index, strip_index]),
                    "delta_t_tilde": float(delta_t_tilde[time_index, strip_index]),
                    "y_gamma": float(y_gamma[time_index, strip_index]),
                    "x_state": float(x_history[time_index, strip_index]),
                    "y_phi": float(y_phi[time_index, strip_index]),
                    "normal_x_gp1": float(
                        geometry["normal_axes"][time_index, strip_index, 0]
                    ),
                    "normal_y_gp1": float(
                        geometry["normal_axes"][time_index, strip_index, 1]
                    ),
                    "normal_z_gp1": float(
                        geometry["normal_axes"][time_index, strip_index, 2]
                    ),
                }
            )

    summary = {
        "benchmark": spec["benchmark"],
        "condition": spec["condition"],
        "status": "mechanical_gates_passed" if all(gates.values()) else "failed",
        "runtime_s": runtime_s,
        "samples_per_cycle": int(cycle["samples_per_cycle"]),
        "source_cycle_step_range": list(cycle["source_cycle_step_range"]),
        "strip_count": len(aggregation["strips"]),
        "panel_count": int(cycle["topology"]["panel_global_index"].size),
        "movement": movement_metadata,
        "case": spec["case_manifest"],
        "metrics": metrics,
        "gates": gates,
        "unscored_force_summary": {
            "old_mean_CL": float(np.mean(-old_airplane_w[:, 2] / q_area)),
            "old_mean_CD": float(np.mean(-old_airplane_w[:, 0] / q_area)),
            "lineitem_mean_CL": float(np.mean(-new_airplane_w[:, 2] / q_area)),
            "lineitem_mean_CD": float(np.mean(-new_airplane_w[:, 0] / q_area)),
            "delta_mean_CL": float(np.mean(-delta_airplane_w[:, 2] / q_area)),
            "delta_mean_CD": float(np.mean(-delta_airplane_w[:, 0] / q_area)),
        },
        "added_mass": {
            "aspect_ratio": aspect_ratio,
            "interpolated_factor": added_mass_factor,
            "parameters": added_mass_parameters.manifest(),
        },
        "line_item_parameters": line_item["parameters"],
        "line_item_provenance": line_item["provenance"],
        "time_layer_contract": {
            "one_state": "right-end zero-order hold; post-update force",
            "body_velocity_and_added_mass": (
                "periodic centered movement derivatives; offline smoke only"
            ),
            "lift_direction": (
                "unit(cross(V_rel,0.75c perpendicular to span, span_axis))"
            ),
        },
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(
            f"v5e mechanical smoke gate failure for {spec['benchmark']}: {failed}"
        )
    return summary, history_rows, strip_rows


def run(output: Path) -> dict[str, Any]:
    """Execute all three frozen smoke cases and write a hashed artifact."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    strip_rows: list[dict[str, Any]] = []
    for spec in _case_specs():
        case_summary, case_force_rows, case_strip_rows = _run_case(spec)
        case_summaries.append(case_summary)
        force_rows.extend(case_force_rows)
        strip_rows.extend(case_strip_rows)

    all_gates_passed = all(
        all(bool(value) for value in case["gates"].values()) for case in case_summaries
    )
    summary = {
        "run_id": output.name,
        "status": (
            "v5e_mechanical_smoke_gates_passed"
            if all_gates_passed
            else "v5e_mechanical_smoke_gate_failure"
        ),
        "promotion_status": "mechanical_only_not_experimentally_scored",
        "model_identity": "FluxV-v5e-ULLT-UVLM-line-item-shadow",
        "conditions": [
            "Yang2025 AoA=15 deg smoke",
            "Izraelevitz Figure14 theta=15 deg psi=60 deg smoke",
            "Baik2012 W2 smoke",
        ],
        "warmup_cycles": WARMUP_CYCLES,
        "observation_files_read": [],
        "experimental_scoring_performed": False,
        "force_axis_contract": (
            "Ptera wind axes: +x is thrust and +z is down; "
            "reported lift=-Fz and drag=-Fx"
        ),
        "cases": case_summaries,
        "limitations": [
            "These smoke histories are mechanical diagnostics, not accuracy evidence.",
            "The local V_perp uses movement kinematics at 0.75c and freestream; same-time-layer UVLM induced velocity is not exported by this adapter.",
            "No LDVM/LEV, polar, separation selector, profile drag, or experimental force correction is included.",
            "The runner is fail-closed for multi-airplane problems because movement G and solver GP1 frames are only identical for these single-airplane cases.",
        ],
    }
    if not all_gates_passed:
        raise RuntimeError("one or more v5e mechanical smoke gates failed")

    force_path = output / "airplane_force_history.csv"
    strip_path = output / "strip_state_history.csv"
    summary_path = output / "summary.json"
    _write_csv(force_path, force_rows)
    _write_csv(strip_path, strip_rows)
    _write_json(summary_path, summary)

    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("fluxv_v5e_panel_ledger.py").resolve(),
        Path(__file__).with_name("fluxv_v5e_line_item.py").resolve(),
        Path(__file__).with_name("ullt_attached.py").resolve(),
        Path(__file__).with_name("augmented_uvpm.py").resolve(),
        Path(__file__).with_name("ptera_adapter.py").resolve(),
        Path(__file__).with_name("baik2012.py").resolve(),
        Path(__file__).with_name("cases.py").resolve(),
        REPO_ROOT / "pyproject.toml",
    )
    manifest = {
        "run_id": output.name,
        "source_hashes": {
            path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
            for path in source_paths
        },
        "result_hashes": {
            path.name: _sha256(path) for path in (force_path, strip_path, summary_path)
        },
        "packages": {
            "numpy": _package_version("numpy"),
            "PteraSoftware": _package_version("PteraSoftware"),
            "fluxvortex": _package_version("fluxvortex"),
        },
        "observation_files": [],
        "experimental_scoring": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(_jsonable(run(args.output_dir.resolve())), indent=2))


if __name__ == "__main__":
    main()
