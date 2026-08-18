"""Protocol-only tests for the v5h11 Baik-W2 no-observation runner.

These tests use synthetic primitive records.  They intentionally do not import
or execute Ptera, source generation, the paper target, or any scorer.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import numpy as np
import pytest


# Load the standalone runner file without executing the package __init__, whose
# legacy exports import Ptera.  The protocol boundary under test must remain
# solver-free.
_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "forward_flight_benchmarks"
    / "run_fluxv_v5h11_baik_w2.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "fluxv_v5h11_w2_runner_under_test", _RUNNER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


_H = "0" * 64
_ERROR = {32: 4.0e-8, 64: 1.0e-8, 128: 0.0}


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _source_placement(
    *, family: str, mode: str, source_step: int, cell: int
) -> dict[str, object]:
    first = mode in {"first", "restart"}
    inactive = mode == "inactive"
    return {
        "schema_id": runner.SOURCE_PLACEMENT_SCHEMA_ID,
        "vortex_family": family,
        "placement_mode": mode,
        "edge_anchor_position_over_chord_backend_world": [0.0, 0.0],
        "birth_position_over_chord_backend_world": (None if inactive else [0.0, 0.0]),
        "birth_displacement_from_edge_over_chord_backend_world": (
            None if inactive else [0.0, 0.0]
        ),
        "q_birth_over_u_backend_world": [0.0, 0.0] if first else None,
        "q_kinematic_over_u_backend_world": [0.0, 0.0] if first else None,
        "q_old_wake_over_u_backend_world": [0.0, 0.0] if first else None,
        "q_provisional_tev_over_u_backend_world": [0.0, 0.0] if first else None,
        "continuous_parent_source_id": (
            None
            if first or inactive
            else f"{_source_lineage_id(cell)}:step:{source_step - 1}:{family}-newborn"
        ),
        "continuous_parent_position_over_chord_backend_world": (
            None if first or inactive else [0.0, 0.0]
        ),
        "used_for_topology_eligible": family == "lev" and first,
    }


def _source_provenance(cell: int) -> dict[str, object]:
    return {
        "interface_id": runner.SOURCE_INTERFACE_ID,
        "backend_id": runner.SOURCE_BACKEND_ID,
        "physical_section_id": f"section:{cell}",
        "physical_strip_id": f"strip:{cell}",
        "section_family": "synthetic-section",
        "reynolds": 1000.0,
        "lesp_critical": 0.2,
        "threshold_source": "synthetic protocol fixture",
        "threshold_source_role": "published_model_parameter",
        "delta_time_convective_nominal": 0.1,
        "pivot_fraction_chord": 0.25,
        "ndiv": 12,
        "naterm": 3,
        "resolved_core_radius_chord": 0.02,
        "max_wake_steps": 8,
        "geometry_identity": f"synthetic-geometry:{cell}",
        "geometry_hash_sha256": _digest(f"synthetic-geometry:{cell}"),
        "geometry_role": "explicit_paired_camber_ordinate_and_slope",
        "geometry_station_count": 12,
        "circulation_units": "Gamma/(U_ref*c_ref)",
        "circulation_scale_u_times_c_m2_per_s": 1.0,
        "position_units": "(x/c_ref,z/c_ref)",
        "position_scale_chord_m": 1.0,
        "position_frame": runner.SOURCE_POSITION_FRAME,
        "circulation_sign": runner.SOURCE_CIRCULATION_SIGN,
        "tev_birth_law": runner.SOURCE_TEV_BIRTH_LAW,
        "lev_birth_law": runner.SOURCE_LEV_BIRTH_LAW,
        "birth_time_layer": runner.SOURCE_BIRTH_TIME_LAYER,
        "dimensionalization_limitations": (
            runner.SOURCE_DIMENSIONALIZATION_LIMITATIONS
        ),
        "source_parity": True,
        "source_solver": "clean_linear",
        "canonical": False,
        "canonical_blocker": runner.SOURCE_CANONICAL_BLOCKER,
        "bottom_model_parity": runner.SOURCE_BOTTOM_MODEL_PARITY,
        "ownership_scope": runner.SOURCE_OWNERSHIP_SCOPE,
        "observation_access": "none",
        "target_case_branch": "none",
    }


def _source_lineage_id(cell: int) -> str:
    provenance = _source_provenance(cell)
    threshold = {
        "value": provenance["lesp_critical"],
        "section_family": provenance["section_family"],
        "reynolds": provenance["reynolds"],
        "source": provenance["threshold_source"],
        "source_role": provenance["threshold_source_role"],
    }
    return (
        "dvm-section-"
        + sha256(
            runner._json_bytes({"provenance": provenance, "threshold": threshold})
        ).hexdigest()[:20]
    )


def _source_kelvin_ledger(*, source_step: int, active: bool) -> dict[str, object]:
    return {
        "circulation_units": "Gamma/(U_ref*c_ref)",
        "gamma_bound_post": 0.0,
        "gamma_old_tev_persisted": 0.0,
        "gamma_old_lev_persisted": 0.0,
        "gamma_deleted_before": 0.0,
        # This is deliberately different from the final coupled TEV value.
        # It is placement audit evidence and must not enter SOURCE_VECTOR_FIELDS.
        "gamma_tev_new_te_only_provisional": 0.125 if active else 0.0,
        "gamma_tev_new_solved": 0.0,
        "gamma_tev_new_persisted": 0.0,
        "gamma_lev_new_solved": 0.0,
        "gamma_lev_new_persisted": 0.0,
        "gamma_deleted_after": 0.0,
        "gamma_deleted_delta": 0.0,
        "gamma_tev_persisted_after": 0.0,
        "gamma_lev_persisted_after": 0.0,
        "tev_solved_to_persisted_delta": 0.0,
        "first_tev_zeroed": source_step == 1,
        "kelvin_solve_residual": 0.0,
        "persistence_residual": 0.0,
    }


def _raw_step(level: int, layer: int) -> dict[str, object]:
    row: dict[str, object] = {name: 0 for name in runner.RAW_STEP_FIELDS}
    source_step = runner.SOURCE_STEPS[layer - 1]
    ptera_step = runner.PTERA_STEPS[layer - 1]
    row.update(
        {
            "transport_substeps": level,
            "layer": layer,
            "source_step_index": source_step,
            "ptera_step_index": ptera_step,
            "status": "completed",
            "particle_count": layer + 1,
            "material_tracer_count": 10,
            "material_support_tracer_count": 1,
            "frontier_node_tracer_count": 9,
            "no_penetration_max_abs": 0.0,
            "kelvin_residual_max_abs": 0.0,
            "raw_cl": -(3.0 + _ERROR[level]) / 10.0,
            "raw_cd": -(float(layer) + _ERROR[level]) / 10.0,
            "direct_field_call_count": 6 * level,
            "ptera_center_call_count": 6 * level,
            "ptera_offset_call_count": 18 * level,
            "fd_physical_evaluation_count": 3 * level,
            "fd_tracer_evaluation_count": 3 * level,
            "fd_evaluator_call_count": 24 * level,
            "transport_substep_count": level,
            "transport_stage_count": 3 * level,
            "physical_field_call_count": 3 * level,
            "tracer_field_call_count": 3 * level,
            "observer_call_count": 3 * level,
            "stage_pre_reconstruction_count": 3 * level,
            "stage_post_reconstruction_count": 3 * level,
            "physical_rhs_call_count": 3 * level,
            "storage_reset_count": level,
            "tracer_storage_reset_count": level,
            "invariant_reference_freeze_count": 1,
            "sigma_storage_update_count": 0,
            "relaxation_call_count": 0,
            "compact_stage_record_count": 3 * level,
            "retained_stage_array_count": 0,
            "evidence_byte_count": 128,
            "native_collocation_evaluation_count": 1,
            "native_load_batch_evaluation_count": 4,
            "native_load_call_count": 1,
            "transport_macro_call_count": 1,
            "row_commit_count": 1,
        }
    )
    for name in runner.RAW_STEP_FIELDS:
        if name.endswith("sha256"):
            row[name] = _digest(f"raw:{level}:{layer}:{name}")
    row["row_owner_before_sha256"] = _H
    row["advanced_owner_sha256"] = _H
    row["ptera_parent_sha256_before"] = _H
    row["ptera_parent_sha256_after"] = _H
    row["stream_stage_chain_sha256"] = _stages(level, layer)[-1]["chain_sha256"]
    return row


def _source_manifest_history(through_step: int) -> list[list[dict[str, object]]]:
    parents = [
        sha256(
            runner._json_bytes(
                [runner.SOURCE_EVENT_CHAIN_DOMAIN, _source_lineage_id(cell)]
            )
        ).hexdigest()
        for cell in range(8)
    ]
    history: list[list[dict[str, object]]] = []
    for source_step in range(1, through_step + 1):
        active = source_step >= runner.SOURCE_STEPS[0]
        mode = (
            "none"
            if not active
            else ("first" if source_step == runner.SOURCE_STEPS[0] else "continuous")
        )
        current: list[dict[str, object]] = []
        for cell in range(8):
            provenance = _source_provenance(cell)
            lineage_id = _source_lineage_id(cell)
            lev_before = max(0, source_step - runner.SOURCE_STEPS[0])
            manifest: dict[str, object] = {
                "enabled": True,
                "status": "evaluated_source_only_noncanonical_d0_unconsumed",
                "delta_time_convective": 0.1,
                "a0_pre": 0.3 if active else 0.0,
                "a0_post": 0.2 if active else 0.0,
                "lesp_critical": 0.2,
                "lesp_active": active,
                "lesp_signed_target": 0.2 if active else 0.0,
                "lesp_constraint_residual": 0.0,
                "lev_birth_mode": mode,
                "restart": False,
                "gamma_lev_new_over_u_c": 0.0,
                "gamma_tev_new_solved_over_u_c": 0.0,
                "gamma_tev_new_persisted_over_u_c": 0.0,
                "lev_placement": _source_placement(
                    family="lev",
                    mode=("inactive" if mode == "none" else mode),
                    source_step=source_step,
                    cell=cell,
                ),
                "tev_placement": _source_placement(
                    family="tev",
                    mode="first" if source_step == 1 else "continuous",
                    source_step=source_step,
                    cell=cell,
                ),
                "lev_birth_position_over_chord_backend_world": (
                    [0.0, 0.0] if active else None
                ),
                "tev_birth_position_over_chord_backend_world": [0.0, 0.0],
                "kelvin_residual_over_u_c": 0.0,
                "kelvin_ledger": _source_kelvin_ledger(
                    source_step=source_step, active=active
                ),
                "lineage": {
                    "physical_section_id": provenance["physical_section_id"],
                    "physical_strip_id": provenance["physical_strip_id"],
                    "section_lineage_id": lineage_id,
                    "source_step_index": source_step,
                    "parent_state_step_index": source_step - 1,
                    "newborn_tev_source_id": (
                        f"{lineage_id}:step:{source_step}:tev-newborn"
                    ),
                    "newborn_lev_source_id": (
                        f"{lineage_id}:step:{source_step}:lev-newborn"
                        if active
                        else None
                    ),
                    "newborn_tev_role": (
                        "constraint_column_solved_then_zeroed_before_persistence"
                        if source_step == 1
                        else "coupled_newest_persisted_step_source"
                    ),
                    "newborn_lev_role": (
                        "lesp_gated_newborn_persisted_step_source"
                        if active
                        else "inactive_no_newborn_source"
                    ),
                    "persistent_tev_history_role": (
                        "backend-owned convected TE history; not re-exported as newborn"
                    ),
                    "persistent_lev_history_role": (
                        "backend-owned convected LE history; not re-exported as newborn"
                    ),
                    "persistent_tev_count_before": source_step - 1,
                    "persistent_tev_count_after": source_step,
                    "persistent_lev_count_before": lev_before,
                    "persistent_lev_count_after": lev_before + int(active),
                    "persistent_history_exported": False,
                },
                "provenance": provenance,
                "parent_event_manifest_sha256": parents[cell],
                "producer_manifest_sha256": "",
            }
            unsigned = dict(manifest)
            del unsigned["producer_manifest_sha256"]
            manifest["producer_manifest_sha256"] = sha256(
                runner.SOURCE_EVENT_DIGEST_PREFIX + runner._json_bytes(unsigned)
            ).hexdigest()
            current.append(manifest)
        parents = [str(item["producer_manifest_sha256"]) for item in current]
        history.append(current)
    return history


def _source_cell_manifests(level: int, layer: int) -> list[dict[str, object]]:
    del level
    source_step = runner.SOURCE_STEPS[layer - 1]
    return _source_manifest_history(source_step)[-1]


def _source_prehistory_manifests() -> list[list[dict[str, object]]]:
    return _source_manifest_history(runner.SOURCE_STEPS[0] - 1)


def _source_event(level: int, source_step: int) -> dict[str, object]:
    ptera_step = source_step - 1
    layer = runner.SOURCE_STEPS.index(source_step) + 1
    manifests = _source_cell_manifests(level, layer)
    row: dict[str, object] = {name: 0 for name in runner.SOURCE_EVENT_FIELDS}
    row.update(
        {
            "transport_substeps": level,
            "source_step_index": source_step,
            "ptera_step_index": ptera_step,
            "source_time_s": ptera_step * runner.DELTA_TIME_S,
            "cell_count": 8,
            "status": "completed",
            "birth_modes": [
                "first" if source_step == runner.SOURCE_STEPS[0] else "continuous"
            ]
            * 8,
            "event_sha256": runner._payload_sha256(
                "fluxv-v5h11-source-event-aggregate-v1",
                [item["producer_manifest_sha256"] for item in manifests],
            ),
            "parent_event_sha256": runner._payload_sha256(
                "fluxv-v5h11-source-parent-aggregate-v1",
                [item["parent_event_manifest_sha256"] for item in manifests],
            ),
        }
    )
    row["a0_pre"] = [item["a0_pre"] for item in manifests]
    row["a0_post"] = [item["a0_post"] for item in manifests]
    row["gamma_lev_new_m2_s"] = [item["gamma_lev_new_over_u_c"] for item in manifests]
    row["gamma_lev_persisted_m2_s"] = [
        item["kelvin_ledger"]["gamma_lev_persisted_after"] for item in manifests
    ]
    row["gamma_tev_new_m2_s"] = [
        item["gamma_tev_new_solved_over_u_c"] for item in manifests
    ]
    row["kelvin_residual_m2_s"] = [
        item["kelvin_residual_over_u_c"] for item in manifests
    ]
    return row


def _owner_event(level: int, layer: int) -> dict[str, object]:
    arrays = _trajectory_arrays(level, layer)
    previous_owner = _owner_event(level, layer - 1) if layer > 1 else None
    row: dict[str, object] = {name: _H for name in runner.OWNER_EVENT_FIELDS}
    row.update(
        {
            "schema_id": "fluxv-v5h11-baik-w2-owner-event-v1",
            "transport_substeps": level,
            "layer": layer,
            "source_step_index": runner.SOURCE_STEPS[layer - 1],
            "ptera_step_index": runner.PTERA_STEPS[layer - 1],
            "status": "completed",
            "changed_particle_ids": [] if layer == 1 else [f"P{layer - 2}"],
            "appended_particle_ids": (["P0", "P1"] if layer == 1 else [f"P{layer}"]),
        }
    )
    if layer == 1:
        row["commit_event"] = None
    else:
        assert previous_owner is not None
        previous_arrays = _trajectory_arrays(level, layer - 1)
        changed_indices = [layer - 2]
        changed = np.asarray(changed_indices, dtype=np.int64)
        before_gamma = previous_arrays["end_gamma"][changed]
        after_gamma = arrays["start_gamma"][changed]
        commit: dict[str, object] = {name: _H for name in runner.COMMIT_EVENT_FIELDS}
        commit.update(
            {
                "proposal_id": f"proposal:{level}:{layer}",
                "release_index": layer,
                "changed_indices": changed_indices,
                "appended_indices": [layer],
                "parent_transport_event_sha256": previous_owner["transport_event"][
                    "transport_event_sha256"
                ],
                "upstream_nodes_sha256": runner._v5h10_digest(
                    "fluxv-v5h10-transported-live-nodes-v1",
                    previous_arrays["frontier_tracer_positions"],
                ),
                "before_gamma_sha256": runner._v5h10_digest(
                    "v5h10-before-gamma", before_gamma
                ),
                "added_gamma_sha256": runner._v5h10_digest(
                    "v5h10-added-gamma", after_gamma - before_gamma
                ),
                "after_gamma_sha256": runner._v5h10_digest(
                    "v5h10-after-gamma", after_gamma
                ),
                "operator_order": "global_row_commit_before_ptera_before_lsrk3",
                "global_graph_build_count": 1,
                "clone_count": 0,
                "counter_particle_count": 0,
                "fresh_upstream_particle_count": 0,
                "remesh_count": 0,
                "ptera_call_count": 0,
                "load_call_count": 0,
                "feedback_call_count": 0,
                "transport_call_count": 0,
            }
        )
        previous_commit = previous_owner["commit_event"]
        commit["previous_event_sha256"] = (
            "0" * 64 if previous_commit is None else previous_commit["event_sha256"]
        )
        commit["event_sha256"] = runner._v5h10_event_digest(
            "fluxv-v5h10-row-event-v1",
            commit,
            runner.COMMIT_EVENT_FIELDS,
            "event_sha256",
            tuple_fields=("changed_indices", "appended_indices"),
        )
        row["commit_event"] = commit
    transport: dict[str, object] = {name: _H for name in runner.TRANSPORT_EVENT_FIELDS}
    transport.update(
        {
            "release_index": layer,
            "source_step_index": runner.SOURCE_STEPS[layer - 1],
            "transport_end_time_s": (
                runner.SOURCE_STEPS[layer - 1] * runner.DELTA_TIME_S
            ),
            "transported_arrays_sha256": runner._v5h10_digest(
                "fluxv-v5h10-transported-arrays-v1",
                arrays["end_positions"],
                arrays["end_gamma"],
                arrays["end_sigma"],
            ),
            "live_boundary_nodes_sha256": runner._v5h10_digest(
                "fluxv-v5h10-transported-live-nodes-v1",
                arrays["frontier_tracer_positions"],
            ),
            "previous_transport_event_sha256": (
                "0" * 64
                if previous_owner is None
                else previous_owner["transport_event"]["transport_event_sha256"]
            ),
        }
    )
    transport["transport_event_sha256"] = runner._v5h10_event_digest(
        "fluxv-v5h10-row-transport-event-v1",
        transport,
        runner.TRANSPORT_EVENT_FIELDS,
        "transport_event_sha256",
    )
    row["transport_event"] = transport
    return row


def _particle_count(level: int, layer: int) -> dict[str, object]:
    return {
        "transport_substeps": level,
        "layer": layer,
        "status": "completed",
        "particle_count": layer + 1,
        "material_tracer_count": 10,
        "material_support_tracer_count": 1,
        "frontier_node_tracer_count": 9,
        "changed_particle_count": 0 if layer == 1 else 1,
        "appended_particle_count": 2 if layer == 1 else 1,
    }


def _loads(level: int, layer: int) -> list[dict[str, object]]:
    error = _ERROR[level]
    force = np.asarray([float(layer) + error, 2.0 + error, 3.0 + error])
    moment = np.asarray([0.2 * layer + error, 0.3 + error, 0.4 + error])
    rows: list[dict[str, object]] = []
    for index, panel_id in enumerate(runner.PANEL_IDS):
        panel_force = force if index == len(runner.PANEL_IDS) - 1 else np.zeros(3)
        panel_moment = moment if index == len(runner.PANEL_IDS) - 1 else np.zeros(3)
        rows.append(
            _load_row(level, layer, "panel", panel_id, panel_force, panel_moment)
        )
    rows.append(
        _load_row(
            level,
            layer,
            "total",
            runner.TOTAL_PANEL_ID,
            force,
            moment,
        )
    )
    return rows


def _load_row(
    level: int,
    layer: int,
    scope: str,
    panel_id: str,
    force: np.ndarray,
    moment: np.ndarray,
) -> dict[str, object]:
    return {
        "transport_substeps": level,
        "layer": layer,
        "scope": scope,
        "panel_id": panel_id,
        "force_x_n": float(force[0]),
        "force_y_n": float(force[1]),
        "force_z_n": float(force[2]),
        "moment_x_nm": float(moment[0]),
        "moment_y_nm": float(moment[1]),
        "moment_z_nm": float(moment[2]),
        "force_coefficient_x": (float(force[0]) / 10.0 if scope == "total" else None),
        "force_coefficient_y": (float(force[1]) / 10.0 if scope == "total" else None),
        "force_coefficient_z": (float(force[2]) / 10.0 if scope == "total" else None),
        "raw_cl": -float(force[2]) / 10.0 if scope == "total" else None,
        "raw_cd": -float(force[0]) / 10.0 if scope == "total" else None,
    }


def _trajectory_arrays(level: int, layer: int) -> dict[str, np.ndarray]:
    error = _ERROR[level]
    if layer == 1:
        start_positions = np.asarray(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.25]], dtype=np.float64
        )
        start_gamma = np.asarray([[1.0, 0.1, 0.2], [1.1, 0.15, 0.22]], dtype=np.float64)
        start_sigma = np.asarray([1.0, 1.1], dtype=np.float64)
    else:
        previous = _trajectory_arrays(level, layer - 1)
        start_positions = np.vstack(
            (
                previous["end_positions"],
                np.asarray([[float(layer), float(layer), 0.25 * layer]]),
            )
        )
        start_gamma = np.vstack(
            (
                previous["end_gamma"],
                np.asarray([[1.0 + 0.1 * layer, 0.1, 0.2]]),
            )
        )
        start_gamma[layer - 2] += 0.005 * layer
        start_sigma = np.concatenate(
            (previous["end_sigma"], np.asarray([1.0 + 0.1 * layer]))
        )
    end_positions = start_positions + 0.1 + error
    end_gamma = start_gamma + 0.05 + error
    end_sigma = start_sigma * np.sqrt(
        runner._stable_row_norms(start_gamma) / runner._stable_row_norms(end_gamma)
    )
    force = np.asarray([float(layer) + error, 2.0 + error, 3.0 + error])
    moment = np.asarray([0.2 * layer + error, 0.3 + error, 0.4 + error])
    probe_velocity, probe_jacobian = runner._direct_gaussian_erf_probe_oracle(
        end_positions, end_gamma, end_sigma
    )
    return {
        "start_positions": start_positions,
        "start_gamma": start_gamma,
        "start_sigma": start_sigma,
        "end_positions": end_positions,
        "end_gamma": end_gamma,
        "end_sigma": end_sigma,
        "material_tracer_positions": end_positions[[0]].copy(),
        "frontier_tracer_positions": np.asarray(
            [[float(index), 0.1 * index, float(layer)] for index in range(9)],
            dtype=np.float64,
        )
        + error,
        "probe_velocity": probe_velocity,
        "probe_jacobian": probe_jacobian,
        "force": force,
        "moment": moment,
        "invariant_start": runner._stable_row_norms(start_gamma) * start_sigma**2,
        "invariant_end": runner._stable_row_norms(end_gamma) * end_sigma**2,
        "no_penetration_residual": np.zeros(16, dtype=np.float64),
    }


def _stages(level: int, layer: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    arrays = _trajectory_arrays(level, layer)
    start_state_sha256 = runner._stream_state_sha256_from_arrays(
        arrays["start_positions"], arrays["start_gamma"], arrays["start_sigma"]
    )
    end_state_sha256 = runner._stream_state_sha256_from_arrays(
        arrays["end_positions"], arrays["end_gamma"], arrays["end_sigma"]
    )
    final_tracer_sha256 = runner._array_sha256(
        np.concatenate(
            [
                arrays["material_tracer_positions"],
                arrays["frontier_tracer_positions"],
            ],
            axis=0,
        )
    )
    previous = runner.STREAM_STAGE_CHAIN_GENESIS
    rk_a = (0.0, -5.0 / 9.0, -153.0 / 128.0)
    rk_b = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)
    for substep in range(1, level + 1):
        for stage in (1, 2, 3):
            record_sha = _digest(f"stage:{level}:{layer}:{substep}:{stage}")
            chain = sha256(
                ("fluxv-ir-wrk3-stream-stage-link-v1\0" + previous + record_sha).encode(
                    "ascii"
                )
            ).hexdigest()
            row: dict[str, object] = {name: _H for name in runner.STAGE_FIELDS}
            row.update(
                {
                    "transport_substeps": level,
                    "layer": layer,
                    "source_step_index": runner.SOURCE_STEPS[layer - 1],
                    "ptera_step_index": runner.PTERA_STEPS[layer - 1],
                    "substep": substep,
                    "stage": stage,
                    "status": "completed",
                    "substep_delta_time": runner.DELTA_TIME_S / level,
                    "rk_a": rk_a[stage - 1],
                    "rk_b": rk_b[stage - 1],
                    "invariant_residual_over_slog_max": 0.0,
                    "h_jacobian_frobenius": 0.1,
                    "h_convective_over_sigma": 0.01,
                    "direct_field_call_count": 2,
                    "ptera_center_call_count": 2,
                    "ptera_offset_call_count": 6,
                    "stream_record_sha256": record_sha,
                    "previous_chain_sha256": previous,
                    "chain_sha256": chain,
                    "failure_type": "",
                    "failure_message": "",
                }
            )
            row["pre_state_sha256"] = (
                start_state_sha256 if not rows else rows[-1]["post_state_sha256"]
            )
            row["post_state_sha256"] = _H
            row["tracer_pre_sha256"] = (
                _H if not rows else rows[-1]["tracer_post_sha256"]
            )
            row["tracer_post_sha256"] = _H
            rows.append(row)
            previous = chain
    rows[-1]["post_state_sha256"] = end_state_sha256
    rows[-1]["tracer_post_sha256"] = final_tracer_sha256
    return rows


def _failed_stage(
    completed_shape: dict[str, object] | object,
    *,
    failure_type: str = "SyntheticStageFailure",
    failure_message: str = "protocol-only terminal stage",
) -> dict[str, object]:
    assert isinstance(completed_shape, dict)
    row: dict[str, object] = {name: None for name in runner.STAGE_FIELDS}
    for name in (
        "transport_substeps",
        "layer",
        "source_step_index",
        "ptera_step_index",
        "substep",
        "stage",
        "previous_chain_sha256",
    ):
        row[name] = completed_shape[name]
    row.update(
        {
            "status": "failed",
            "chain_sha256": completed_shape["previous_chain_sha256"],
            "failure_type": failure_type,
            "failure_message": failure_message,
        }
    )
    return row


def _append_compact_stage(
    sink: runner.ArtifactSink, row: dict[str, object] | object
) -> None:
    assert isinstance(row, dict)
    sink.add_transport_stage_from_compact_evidence(
        row,
        {
            "invariant_residual_over_slog_max": row["invariant_residual_over_slog_max"],
            "h_jacobian_frobenius": row["h_jacobian_frobenius"],
            "h_convective_over_sigma": row["h_convective_over_sigma"],
        },
    )


def _trajectory(level: int, layer: int) -> dict[str, object]:
    arrays = _trajectory_arrays(level, layer)
    particle_ids = tuple(f"P{index}" for index in range(layer + 1))
    material_ids = ("P0",)
    manifests = _source_cell_manifests(level, layer)
    prehistory = _source_prehistory_manifests() if layer == 1 else []
    return {
        "transport_substeps": level,
        "layer": layer,
        "status": "completed",
        "particle_ids": particle_ids,
        "material_tracer_ids": material_ids,
        "frontier_node_ids": runner.FRONTIER_NODE_IDS,
        "arrays": arrays,
        "metadata": {
            "source_cell_manifest_sha256": runner._payload_sha256(
                "fluxv-v5h11-source-cell-manifests-v1", manifests
            ),
            "source_cell_manifests": manifests,
            "source_prehistory_manifest_sha256": runner._payload_sha256(
                "fluxv-v5h11-source-prehistory-manifests-v1", prehistory
            ),
            "source_prehistory_manifests": prehistory,
            "source_kelvin_ledger_sha256": runner._payload_sha256(
                "fluxv-v5h11-source-kelvin-ledgers-v1",
                [item["kelvin_ledger"] for item in manifests],
            ),
            "source_kelvin_evidence_sha256": runner._payload_sha256(
                "fluxv-v5h11-source-kelvin-v1",
                {
                    "atol_m2_s": float(runner.KELVIN_RESIDUAL_MAX).hex(),
                    "kelvin_ledger_sha256": runner._payload_sha256(
                        "fluxv-v5h11-source-kelvin-ledgers-v1",
                        [item["kelvin_ledger"] for item in manifests],
                    ),
                    "residual_m2_s": float(0.0).hex(),
                    "row_owner_sha256": _H,
                    "source_event_sha256": _source_event(
                        level, runner.SOURCE_STEPS[layer - 1]
                    )["event_sha256"],
                    "source_step_index": runner.SOURCE_STEPS[layer - 1],
                },
            ),
            "particle_id_sequence_sha256": runner._payload_sha256(
                "fluxv-v5h11-particle-id-sequence-v1", list(particle_ids)
            ),
            "material_tracer_id_sequence_sha256": runner._payload_sha256(
                "fluxv-v5h11-material-id-sequence-v1", list(material_ids)
            ),
            "frontier_start_positions_sha256": _digest(
                f"frontier-start:{level}:{layer}"
            ),
            "fixed_probe_contract": [
                list(point) for point in runner.FIXED_PROBES_GP1_M
            ],
        },
    }


def _pass_records() -> runner.ArtifactRecords:
    raw_steps: list[dict[str, object]] = []
    source_events: list[dict[str, object]] = []
    owner_events: list[dict[str, object]] = []
    particle_counts: list[dict[str, object]] = []
    raw_loads: list[dict[str, object]] = []
    transport_stages: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    for level in runner.FORMAL_LEVELS:
        for source_step in runner.SOURCE_STEPS:
            source_events.append(_source_event(level, source_step))
        for layer in runner.LAYERS:
            raw_steps.append(_raw_step(level, layer))
            owner_events.append(_owner_event(level, layer))
            particle_counts.append(_particle_count(level, layer))
            raw_loads.extend(_loads(level, layer))
            transport_stages.extend(_stages(level, layer))
            trajectories.append(_trajectory(level, layer))
    return runner.ArtifactRecords(
        execution_mode=runner.SYNTHETIC_EXECUTION_MODE,
        status="PASS",
        raw_steps=tuple(raw_steps),
        source_events=tuple(source_events),
        owner_events=tuple(owner_events),
        particle_counts=tuple(particle_counts),
        raw_loads=tuple(raw_loads),
        transport_stages=tuple(transport_stages),
        trajectories=tuple(trajectories),
    )


def _with_force_convergence_failure(
    records: runner.ArtifactRecords,
) -> runner.ArtifactRecords:
    """Keep every layer cross-link valid while making N=64 a bad load candidate."""

    key = (64, 2)
    layer_index = runner.EXPECTED_LAYER_KEYS.index(key)
    raw_steps = list(records.raw_steps)
    trajectories = list(records.trajectories)
    raw_loads = list(records.raw_loads)

    trajectory = dict(trajectories[layer_index])
    arrays = dict(trajectory["arrays"])
    force = np.asarray(arrays["force"], dtype=np.float64).copy()
    force[0] = 1.0e308
    arrays["force"] = force
    trajectory["arrays"] = arrays
    trajectories[layer_index] = trajectory

    load_offset = 17 * layer_index
    for row_index in (load_offset + len(runner.PANEL_IDS) - 1, load_offset + 16):
        load = dict(raw_loads[row_index])
        load["force_x_n"] = float(force[0])
        if load["scope"] == "total":
            load["force_coefficient_x"] = float(force[0]) / 10.0
            load["raw_cd"] = -float(force[0]) / 10.0
        raw_loads[row_index] = load

    raw = dict(raw_steps[layer_index])
    raw["raw_cd"] = -float(force[0]) / 10.0
    raw_steps[layer_index] = raw
    return replace(
        records,
        raw_steps=tuple(raw_steps),
        raw_loads=tuple(raw_loads),
        trajectories=tuple(trajectories),
    )


def _with_forged_first_source_genesis(
    records: runner.ArtifactRecords,
) -> runner.ArtifactRecords:
    source_rows = list(records.source_events)
    trajectories = list(records.trajectories)
    for index in (0, 3, 6):
        trajectory = dict(trajectories[index])
        metadata = dict(trajectory["metadata"])
        manifests = [dict(item) for item in metadata["source_cell_manifests"]]
        manifests[0]["parent_event_manifest_sha256"] = sha256(
            runner._json_bytes(
                [
                    runner.SOURCE_EVENT_CHAIN_DOMAIN,
                    manifests[0]["lineage"]["section_lineage_id"],
                ]
            )
        ).hexdigest()
        unsigned = dict(manifests[0])
        del unsigned["producer_manifest_sha256"]
        manifests[0]["producer_manifest_sha256"] = sha256(
            runner.SOURCE_EVENT_DIGEST_PREFIX + runner._json_bytes(unsigned)
        ).hexdigest()
        metadata["source_cell_manifests"] = manifests
        metadata["source_cell_manifest_sha256"] = runner._payload_sha256(
            "fluxv-v5h11-source-cell-manifests-v1", manifests
        )
        source = dict(source_rows[index])
        source["event_sha256"] = runner._payload_sha256(
            "fluxv-v5h11-source-event-aggregate-v1",
            [item["producer_manifest_sha256"] for item in manifests],
        )
        source["parent_event_sha256"] = runner._payload_sha256(
            "fluxv-v5h11-source-parent-aggregate-v1",
            [item["parent_event_manifest_sha256"] for item in manifests],
        )
        metadata["source_kelvin_evidence_sha256"] = runner._payload_sha256(
            "fluxv-v5h11-source-kelvin-v1",
            {
                "atol_m2_s": float(runner.KELVIN_RESIDUAL_MAX).hex(),
                "kelvin_ledger_sha256": metadata["source_kelvin_ledger_sha256"],
                "residual_m2_s": float(0.0).hex(),
                "row_owner_sha256": records.raw_steps[index]["row_owner_before_sha256"],
                "source_event_sha256": source["event_sha256"],
                "source_step_index": runner.SOURCE_STEPS[0],
            },
        )
        trajectory["metadata"] = metadata
        source_rows[index] = source
        trajectories[index] = trajectory
    return replace(
        records,
        source_events=tuple(source_rows),
        trajectories=tuple(trajectories),
    )


def _provenance(output: Path, replicate: str) -> dict[str, object]:
    return {
        "replicate": replicate,
        "output_path": str(output.resolve()),
        "run_uuid": f"00000000-0000-0000-0000-00000000000{1 if replicate == 'A' else 2}",
        "start_utc": f"2026-08-16T00:00:0{1 if replicate == 'A' else 2}+00:00",
    }


def _write_checksum_file(directory: Path) -> None:
    names = sorted(set(runner.ARTIFACT_FILES) - {"SHA256SUMS"})
    (directory / "SHA256SUMS").write_text(
        "".join(f"{runner._sha256_file(directory / name)}  {name}\n" for name in names),
        encoding="ascii",
    )


def _rewrite_external_audit(token_path: Path, manifest: dict[str, object]) -> None:
    manifest_path = Path(json.loads(token_path.read_text())["dependency_manifest_path"])
    manifest_path.write_bytes(runner._json_bytes(manifest) + b"\n")
    token = json.loads(token_path.read_text())
    token["dependency_manifest_sha256"] = runner._sha256_file(manifest_path)
    token_path.write_bytes(runner._json_bytes(token) + b"\n")


def _make_external_audit_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    executor_source: str | None = None,
) -> tuple[Path, dict[str, object], Path]:
    leaves: list[dict[str, object]] = []
    mutable_leaf: Path | None = None
    repository_root = _RUNNER_PATH.parents[2]
    canonical_paths = dict(runner.CANONICAL_PROJECT_LEAF_RELATIVE_PATH)
    for name in runner.REQUIRED_DEPENDENCY_LEAVES:
        if name == "formal_executor_module":
            path = (tmp_path / "fluxv_v5h11_baik_w2_executor.py").resolve()
            path.write_text(
                executor_source
                or "def build_fluxv_v5h11_w2_executor(api):\n    return object()\n",
                encoding="utf-8",
            )
            canonical_paths[name] = str(path)
        elif name == "formal_executor_test":
            path = (tmp_path / "test_fluxv_v5h11_baik_w2_executor.py").resolve()
            path.write_text(
                "# synthetic audited executor test leaf\n", encoding="utf-8"
            )
            canonical_paths[name] = str(path)
            mutable_leaf = path
        elif name in canonical_paths:
            configured = Path(canonical_paths[name])
            path = (
                configured if configured.is_absolute() else repository_root / configured
            ).resolve()
            if not path.exists():
                path = (tmp_path / Path(canonical_paths[name]).name).resolve()
                path.write_text(f"synthetic audit leaf: {name}\n", encoding="utf-8")
                canonical_paths[name] = str(path)
        elif name == "pterasoftware_solver_source":
            path = runner._installed_distribution_file(
                "pterasoftware",
                "pterasoftware/unsteady_ring_vortex_lattice_method.py",
            )
        elif name in {
            "pterasoftware_distribution_metadata",
            "fluxvortex_distribution_metadata",
            "numpy_distribution_metadata",
            "scipy_distribution_metadata",
        }:
            distribution = name.removesuffix("_distribution_metadata")
            path = runner._installed_distribution_identity(distribution)[0]
        else:
            path = (tmp_path / f"{name}.txt").resolve()
            path.write_text(f"synthetic audit leaf: {name}\n", encoding="utf-8")
        leaves.append(
            {"name": name, "path": str(path), "sha256": runner._sha256_file(path)}
        )
    monkeypatch.setattr(runner, "CANONICAL_PROJECT_LEAF_RELATIVE_PATH", canonical_paths)
    manifest_path = (tmp_path / "dependency-manifest.json").resolve()
    manifest = {
        "schema_id": runner.DEPENDENCY_MANIFEST_SCHEMA_ID,
        "dependency_set_scope": runner.DEPENDENCY_SET_SCOPE,
        "leaf_files": leaves,
        "runtime_module_files": [],
    }
    manifest_path.write_bytes(runner._json_bytes(manifest) + b"\n")
    token_path = (tmp_path / "audit-token.json").resolve()
    token = {
        "schema_id": runner.DEPENDENCY_AUDIT_TOKEN_SCHEMA_ID,
        "audit_id": "synthetic-audit-only",
        "audit_scope": runner.FORMAL_AUDIT_SCOPE,
        "audit_verdict": "PASS",
        "dependency_manifest_path": str(manifest_path),
        "dependency_manifest_sha256": runner._sha256_file(manifest_path),
        "observation_access": runner.OBSERVATION_ACCESS,
    }
    token_path.write_bytes(runner._json_bytes(token) + b"\n")
    assert mutable_leaf is not None
    return token_path, manifest, mutable_leaf


@pytest.fixture(scope="module")
def pass_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("v5h11_w2_pass")
    left, right = root / "A", root / "B"
    records = _pass_records()
    runner.publish_artifact(
        records, left, replicate="A", run_provenance=_provenance(left, "A")
    )
    runner.publish_artifact(
        records, right, replicate="B", run_provenance=_provenance(right, "B")
    )
    return left, right


def test_pass_bundle_has_exact_counts_base64_and_independent_convergence(
    pass_pair: tuple[Path, Path],
) -> None:
    artifact, _ = pass_pair
    result = runner.verify_artifact(artifact)
    assert result["status"] == "PASS"
    assert result["passed"] is True
    assert {path.name for path in artifact.iterdir()} == set(runner.ARTIFACT_FILES)

    summary = json.loads((artifact / "summary.json").read_text())
    assert summary["execution_mode"] == runner.SYNTHETIC_EXECUTION_MODE
    assert (
        summary["schema_id"]
        == runner.SUMMARY_SCHEMA_BY_MODE[runner.SYNTHETIC_EXECUTION_MODE]
    )
    assert summary["semantic_input_file_sha256"] == {
        name: runner._sha256_file(artifact / name) for name in runner.ARTIFACT_FILES[:8]
    }
    assert summary["fixed_probes_gp1_m"] == [
        list(point) for point in runner.FIXED_PROBES_GP1_M
    ]
    assert summary["fixed_probe_source_sha256"] == runner.FIXED_PROBE_SOURCE_SHA256
    assert summary["row_counts"] == {
        "owner_events": 9,
        "particle_counts": 9,
        "raw_loads": 153,
        "raw_steps": 9,
        "source_events": 9,
        "trajectories": 9,
        "transport_stages": 2016,
    }
    convergence = json.loads((artifact / "convergence.json").read_text())
    assert convergence["metric_count"] == 27
    assert convergence["passed"] is True

    trajectory = json.loads((artifact / "trajectory_arrays.json").read_text())
    assert len(trajectory["records"]) == 9
    encoded = trajectory["records"][0]["arrays"]
    assert trajectory["records"][0]["frontier_node_ids"]["items"] == list(
        runner.FRONTIER_NODE_IDS
    )
    assert len(trajectory["records"][0]["material_tracer_ids"]["items"]) == 1
    assert set(encoded) == set(runner.REQUIRED_TRAJECTORY_ARRAYS)
    assert all(
        set(value) == {"data_base64", "dtype", "order", "sha256", "shape"}
        for value in encoded.values()
    )
    assert np.array_equal(
        runner.decode_array(encoded["end_positions"]),
        _trajectory(32, 1)["arrays"]["end_positions"],
    )
    assert runner.decode_array(encoded["probe_velocity"]).shape == (3, 3)
    assert runner.decode_array(encoded["probe_jacobian"]).shape == (3, 3, 3)

    checksum_lines = (artifact / "SHA256SUMS").read_text("ascii").splitlines()
    assert len(checksum_lines) == 11
    assert [line.split("  ", 1)[1] for line in checksum_lines] == sorted(
        set(runner.ARTIFACT_FILES) - {"SHA256SUMS"}
    )


def test_a_b_compare_exactly_nine_semantic_files(pass_pair: tuple[Path, Path]) -> None:
    left, right = pass_pair
    root = runner.compare_semantic_artifacts(left, right)
    assert len(root) == 64
    assert len(runner.SEMANTIC_FILES) == 9
    assert all(
        (left / name).read_bytes() == (right / name).read_bytes()
        for name in runner.SEMANTIC_FILES
    )
    assert (left / "run_manifest.json").read_bytes() != (
        right / "run_manifest.json"
    ).read_bytes()
    assert (left / "run.log").read_bytes() != (right / "run.log").read_bytes()


def test_convergence_is_recomputed_from_raw_durable_inputs(
    pass_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    source, _ = pass_pair
    scratch = tmp_path / "tampered"
    shutil.copytree(source, scratch)
    document = json.loads((scratch / "trajectory_arrays.json").read_text())
    record = next(
        item
        for item in document["records"]
        if item["transport_substeps"] == 64 and item["layer"] == 1
    )
    array = runner.decode_array(record["arrays"]["end_positions"]).copy()
    array += 0.1
    record["arrays"]["end_positions"] = runner.encode_array(array)
    (scratch / "trajectory_arrays.json").write_bytes(
        runner._json_bytes(document) + b"\n"
    )
    with pytest.raises(ValueError, match="(direct replay oracle|ID-aligned particle)"):
        runner.recompute_convergence_from_artifacts(scratch)


def test_convergence_rejects_cross_level_id_reordering(
    pass_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    source, _ = pass_pair
    scratch = tmp_path / "reordered-ids"
    shutil.copytree(source, scratch)
    document = json.loads((scratch / "trajectory_arrays.json").read_text())
    record = next(
        item
        for item in document["records"]
        if item["transport_substeps"] == 64 and item["layer"] == 1
    )
    record["particle_ids"] = runner._id_sequence(
        list(reversed(record["particle_ids"]["items"]))
    )
    (scratch / "trajectory_arrays.json").write_bytes(
        runner._json_bytes(document) + b"\n"
    )
    with pytest.raises(ValueError, match="ID"):
        runner.recompute_convergence_from_artifacts(scratch)


def test_complete_matrix_convergence_failure_publishes_full_stop(
    tmp_path: Path,
) -> None:
    records = _with_force_convergence_failure(_pass_records())
    runner._validate_records(records)
    output = tmp_path / "full-matrix-convergence-stop"
    verification = runner.publish_artifact(
        records,
        output,
        replicate="A",
        run_provenance=_provenance(output, "A"),
    )

    assert verification["status"] == "STOP"
    assert verification["passed"] is False
    assert {path.name for path in output.iterdir()} == set(runner.ARTIFACT_FILES)
    summary = json.loads((output / "summary.json").read_text())
    convergence = json.loads((output / "convergence.json").read_text())
    trajectory = json.loads((output / "trajectory_arrays.json").read_text())
    assert summary["status"] == trajectory["status"] == convergence["status"] == "STOP"
    assert summary["stop_code"] == "convergence_gate_failed"
    assert summary["row_counts"] == {
        "owner_events": 9,
        "particle_counts": 9,
        "raw_loads": 153,
        "raw_steps": 9,
        "source_events": 9,
        "trajectories": 9,
        "transport_stages": 2016,
    }
    assert convergence["complete_matrix"] is True
    assert convergence["passed"] is False
    assert all(
        metric["failure_reason"] is None or isinstance(metric["failure_reason"], str)
        for metric in convergence["metrics"]
    )
    overflow_metric = runner._l2_metrics(
        np.zeros(1, dtype=np.float64),
        np.asarray([1.0e308], dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        relative_max=1.0,
    )
    assert overflow_metric["failure_reason"] == "unrepresentable_relative_magnitude"
    assert overflow_metric["relative_64_128"] is None
    runner._json_bytes(overflow_metric)
    assert all(row["status"] == "completed" for row in trajectory["records"])
    assert summary["terminal_coordinate"] == {
        "transport_substeps": 128,
        "layer": 3,
        "source_step_index": 6,
        "ptera_step_index": 5,
        "substep": 128,
        "stage": 3,
        "phase": "convergence_replay",
        "stage_began": False,
    }
    assert runner.verify_artifact(output)["status"] == "STOP"


def test_complete_matrix_runtime_stop_keeps_passing_convergence(
    tmp_path: Path,
) -> None:
    full = _pass_records()
    final = full.transport_stages[-1]
    records = replace(
        full,
        status="STOP",
        terminal_coordinate={
            "transport_substeps": final["transport_substeps"],
            "layer": final["layer"],
            "source_step_index": final["source_step_index"],
            "ptera_step_index": final["ptera_step_index"],
            "substep": final["substep"],
            "stage": final["stage"],
            "phase": "coupling_callback",
            "stage_began": False,
        },
        stop_code="coupling_callback_error",
        stop_message="independent synthetic runtime post-matrix failure",
    )
    runner._validate_records(records)
    output = tmp_path / "post-matrix-runtime-stop"
    verification = runner.publish_artifact(
        records,
        output,
        replicate="A",
        run_provenance=_provenance(output, "A"),
    )
    assert verification["status"] == "STOP"
    summary = json.loads((output / "summary.json").read_text())
    convergence = json.loads((output / "convergence.json").read_text())
    assert summary["status"] == convergence["status"] == "STOP"
    assert summary["convergence_passed"] is True
    assert convergence["passed"] is True
    assert summary["stop_code"] == "coupling_callback_error"
    assert runner.verify_artifact(output)["status"] == "STOP"
    assert not runner._passing_convergence_stop_is_allowed(
        status="STOP",
        stop_code="convergence_gate_failed",
        terminal_coordinate={
            **records.terminal_coordinate,
            "phase": "convergence_replay",
        },
        row_counts=summary["row_counts"],
    )


def test_stop_bundle_is_exact_prefix_with_terminal_failed_coordinate(
    tmp_path: Path,
) -> None:
    full = _pass_records()
    completed = [dict(row) for row in full.transport_stages[:2]]
    failed = _failed_stage(dict(full.transport_stages[2]))
    terminal = {
        "transport_substeps": 32,
        "layer": 1,
        "source_step_index": 4,
        "ptera_step_index": 3,
        "substep": 1,
        "stage": 3,
        "phase": "transport_stage",
        "stage_began": True,
    }
    stop = runner.ArtifactRecords(
        execution_mode=runner.SYNTHETIC_EXECUTION_MODE,
        status="STOP",
        raw_steps=(),
        source_events=full.source_events[:1],
        owner_events=(),
        particle_counts=(),
        raw_loads=(),
        transport_stages=tuple(completed + [failed]),
        trajectories=(),
        terminal_coordinate=terminal,
        stop_code="synthetic_stage_failure",
        stop_message="synthetic protocol STOP",
    )
    output = tmp_path / "STOP"
    runner.publish_artifact(
        stop, output, replicate="A", run_provenance=_provenance(output, "A")
    )
    verification = runner.verify_artifact(output)
    assert verification["status"] == "STOP"
    assert {path.name for path in output.iterdir()} == set(runner.ARTIFACT_FILES)
    with (output / "transport_stages.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["substep"]), int(row["stage"])) for row in rows] == [
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    assert [row["status"] for row in rows] == ["completed", "completed", "failed"]
    assert rows[-1]["stream_record_sha256"] == ""
    assert rows[-1]["invariant_residual_over_slog_max"] == ""
    assert rows[-1]["direct_field_call_count"] == ""
    summary = json.loads((output / "summary.json").read_text())
    assert summary["terminal_coordinate"] == terminal
    assert len((output / "SHA256SUMS").read_text("ascii").splitlines()) == 11


def test_renameat2_noreplace_has_one_winner_under_race(tmp_path: Path) -> None:
    destination = tmp_path / "winner"
    staging = [tmp_path / "staging-a", tmp_path / "staging-b"]
    for index, path in enumerate(staging):
        path.mkdir()
        (path / "identity").write_text(str(index), encoding="ascii")

    def publish(path: Path) -> str:
        try:
            runner._publish_directory_noreplace(path, destination)
        except FileExistsError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, staging))
    assert sorted(outcomes) == ["lost", "won"]
    assert (destination / "identity").read_text("ascii") in {"0", "1"}


def test_formal_entry_stops_before_executor_module_load_or_ptera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader_calls = 0

    def bomb_loader(dependency_audit: object) -> object:
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("dependency preflight must stop before module load")

    monkeypatch.setattr(runner, "_load_formal_executor", bomb_loader)

    output = tmp_path / "preflight-stop"
    result = runner.run_formal_attempt(output, replicate="A")
    assert result["status"] == "STOP"
    assert loader_calls == 0
    assert "pterasoftware" not in sys.modules
    assert "forward_flight_benchmarks" not in sys.modules
    assert runner.DEFAULT_DEPENDENCY_AUDIT_TOKEN is None
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stop_code"] == "dependencies_unbound"
    assert summary["terminal_coordinate"] == {
        "layer": None,
        "phase": "dependency_preflight",
        "ptera_step_index": None,
        "source_step_index": None,
        "stage": None,
        "stage_began": False,
        "substep": None,
        "transport_substeps": None,
    }
    assert {path.name for path in output.iterdir()} == set(runner.ARTIFACT_FILES)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["execution_mode"] == runner.FORMAL_EXECUTION_MODE
    assert manifest["dependency_audit"]["status"] == "unbound"


def test_array_base64_and_publish_are_fail_closed(pass_pair: tuple[Path, Path]) -> None:
    artifact, _ = pass_pair
    payload = runner.encode_array(np.asarray([[1.0, 2.0]], dtype=np.float64))
    payload["data_base64"] = payload["data_base64"][:-1] + "!"
    with pytest.raises(ValueError, match="base64"):
        runner.decode_array(payload)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.publish_artifact(_pass_records(), artifact, replicate="A")


def test_external_dependency_audit_is_acyclic_and_runtime_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_path, manifest, mutable_leaf = _make_external_audit_token(
        tmp_path, monkeypatch
    )
    evidence = runner._verified_dependency_audit(token_path)
    assert evidence["status"] == "verified"
    assert tuple(evidence["leaf_file_sha256"]) == runner.REQUIRED_DEPENDENCY_LEAVES
    assert set(manifest) == {
        "schema_id",
        "dependency_set_scope",
        "leaf_files",
        "runtime_module_files",
    }
    assert all(
        leaf["path"] not in {str(token_path), evidence["dependency_manifest_path"]}
        for leaf in manifest["leaf_files"]
    )
    bad_token_path = (tmp_path / "bad-verdict-token.json").resolve()
    bad_token = json.loads(token_path.read_text())
    bad_token["audit_verdict"] = "FAIL"
    bad_token_path.write_bytes(runner._json_bytes(bad_token) + b"\n")
    with pytest.raises(runner.DependencyFreezeError, match="verdict/scope"):
        runner._verified_dependency_audit(bad_token_path)
    mutable_leaf.write_text("mutated after audit\n", encoding="utf-8")
    with pytest.raises(runner.DependencyFreezeError, match="runtime hash mismatch"):
        runner._runtime_reverify_dependency_audit(evidence)


def test_dependency_leaf_alias_and_loaded_module_substitute_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "alias-audit"
    audit_root.mkdir()
    token_path, manifest, _ = _make_external_audit_token(audit_root, monkeypatch)
    leaves = manifest["leaf_files"]
    assert isinstance(leaves, list)
    baik_leaf = next(item for item in leaves if item["name"] == "baik2012")
    alias = (audit_root / "baik2012.txt").resolve()
    alias.write_text("not the canonical Baik implementation\n", encoding="utf-8")
    baik_leaf.update(path=str(alias), sha256=runner._sha256_file(alias))
    _rewrite_external_audit(token_path, manifest)
    with pytest.raises(runner.DependencyFreezeError, match="non-canonical identity"):
        runner._verified_dependency_audit(token_path)

    good_kernel = (_RUNNER_PATH.parents[2] / "src/fluxvortex/kernel.py").resolve()
    token_path, manifest, _ = _make_external_audit_token(audit_root, monkeypatch)
    substitute = (audit_root / "substitute_kernel.py").resolve()
    substitute.write_text("# sys.path substitute\n", encoding="utf-8")
    manifest["runtime_module_files"] = [
        {
            "module_name": "fluxvortex.kernel",
            "path": str(substitute),
            "sha256": runner._sha256_file(substitute),
        }
    ]
    _rewrite_external_audit(token_path, manifest)
    with pytest.raises(runner.DependencyFreezeError, match="non-canonical origin"):
        runner._verified_dependency_audit(token_path)

    token_path, manifest, _ = _make_external_audit_token(audit_root, monkeypatch)
    manifest["runtime_module_files"] = [
        {
            "module_name": "ldvm_fourier",
            "path": str(substitute),
            "sha256": runner._sha256_file(substitute),
        }
    ]
    _rewrite_external_audit(token_path, manifest)
    with pytest.raises(runner.DependencyFreezeError, match="project path"):
        runner._verified_dependency_audit(token_path)

    token_path, manifest, _ = _make_external_audit_token(audit_root, monkeypatch)
    manifest["runtime_module_files"] = [
        {
            "module_name": "fluxvortex.kernel",
            "path": str(good_kernel),
            "sha256": runner._sha256_file(good_kernel),
        }
    ]
    _rewrite_external_audit(token_path, manifest)
    evidence = runner._verified_dependency_audit(token_path)
    fake_module = ModuleType("fluxvortex.kernel")
    fake_module.__file__ = str(substitute)
    monkeypatch.setitem(sys.modules, "fluxvortex.kernel", fake_module)
    with pytest.raises(runner.DependencyFreezeError, match="unmanifested or drifted"):
        runner._capture_observed_runtime_modules(evidence)


@pytest.mark.parametrize(
    ("leaf_name", "distribution_name"),
    tuple(runner.DISTRIBUTION_LEAF_TO_NAME.items()),
)
def test_fake_same_name_distribution_metadata_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaf_name: str,
    distribution_name: str,
) -> None:
    token_path, manifest, _ = _make_external_audit_token(tmp_path, monkeypatch)
    fake_metadata = (
        tmp_path / "site-packages" / f"{distribution_name}-0.0.dist-info" / "METADATA"
    ).resolve()
    fake_metadata.parent.mkdir(parents=True, exist_ok=True)
    fake_metadata.write_text(
        f"Name: {distribution_name}\nVersion: 0.0\n", encoding="utf-8"
    )
    leaves = manifest["leaf_files"]
    assert isinstance(leaves, list)
    leaf = next(item for item in leaves if item["name"] == leaf_name)
    leaf.update(path=str(fake_metadata), sha256=runner._sha256_file(fake_metadata))
    _rewrite_external_audit(token_path, manifest)
    with pytest.raises(
        runner.DependencyFreezeError, match="distribution metadata identity"
    ):
        runner._verified_dependency_audit(token_path)


def test_dependency_drift_after_preflight_publishes_explicit_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    executor_source = """
from pathlib import Path
def build_fluxv_v5h11_w2_executor(api):
    class SyntheticDriftExecutor:
        def attest_dependency_origins(self):
            return None
        def run_formal_matrix(self, *, levels, sink):
            Path(api.dependency_leaf_file_path["formal_executor_test"]).write_text(
                "drift during synthetic long run\\n", encoding="utf-8"
            )
            raise api.stop_constructor(
                "synthetic_stop",
                {
                    "transport_substeps": None, "layer": None,
                    "source_step_index": None, "ptera_step_index": None,
                    "substep": None, "stage": None,
                    "phase": "synthetic_callback", "stage_began": False,
                },
                "protocol-only dependency drift",
            )
    return SyntheticDriftExecutor()
""".lstrip()
    token_path, _, _ = _make_external_audit_token(
        audit_root, monkeypatch, executor_source=executor_source
    )
    output = tmp_path / "dependency-drift-stop"
    result = runner.run_formal_attempt(
        output,
        replicate="A",
        dependency_audit_token=token_path,
        invocation_argv=("synthetic-drift-test",),
    )
    assert result["status"] == "STOP"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stop_code"] == "dependency_drift"
    assert summary["terminal_coordinate"]["phase"] == "dependency_postflight"
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["dependency_audit"]["status"] == "unbound"


def test_spec_loaded_executor_uses_injected_stop_class_and_preserves_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_source = """
def build_fluxv_v5h11_w2_executor(api):
    class SyntheticExecutor:
        def attest_dependency_origins(self):
            assert api.dependency_leaf_file_path["runner"].endswith(
                "run_fluxv_v5h11_baik_w2.py"
            )
            assert len(api.dependency_leaf_file_sha256["runner"]) == 64
            try:
                api.dependency_leaf_file_path["runner"] = "mutated"
            except TypeError:
                pass
            else:
                raise AssertionError("dependency leaf maps must be read-only")

        def run_formal_matrix(self, *, levels, sink):
            assert levels == api.formal_levels
            assert isinstance(sink, api.artifact_sink_type)
            row = {name: 0 for name in api.source_event_fields}
            row.update({
                "transport_substeps": 32,
                "source_step_index": 4,
                "ptera_step_index": 3,
                "source_time_s": 3 * 0.11125,
                "cell_count": 8,
                "status": "completed",
                "birth_modes": ["first"] * 8,
                "event_sha256": "0" * 64,
                "parent_event_sha256": "1" * 64,
            })
            for name in (
                "a0_pre", "a0_post", "gamma_lev_new_m2_s",
                "gamma_lev_persisted_m2_s", "gamma_tev_new_m2_s",
                "kelvin_residual_m2_s",
            ):
                row[name] = [0.0] * 8
            sink.add_source_event(row)
            raise api.stop_constructor(
                "synthetic_controlled_stop",
                {
                    "transport_substeps": 32,
                    "layer": None,
                    "source_step_index": 4,
                    "ptera_step_index": 3,
                    "substep": None,
                    "stage": None,
                    "phase": "source_post",
                    "stage_began": False,
                },
                "protocol-only controlled STOP",
            )
    return SyntheticExecutor()
""".lstrip()
    audit_root = tmp_path / "abi-audit"
    audit_root.mkdir()
    token_path, _, _ = _make_external_audit_token(
        audit_root, monkeypatch, executor_source=executor_source
    )
    output = tmp_path / "abi-stop"
    result = runner.run_formal_attempt(
        output,
        replicate="A",
        dependency_audit_token=token_path,
        invocation_argv=("synthetic-abi-test",),
    )
    assert result["status"] == "STOP"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stop_code"] == "synthetic_controlled_stop"
    assert summary["row_counts"]["source_events"] == 1
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["dependency_audit"]["status"] == "verified"


def test_generic_callback_error_preserves_source_terminal_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor_source = """
def build_fluxv_v5h11_w2_executor(api):
    class SyntheticExecutor:
        def attest_dependency_origins(self):
            return None
        def run_formal_matrix(self, *, levels, sink):
            row = {name: 0 for name in api.source_event_fields}
            row.update({
                "transport_substeps": 32,
                "source_step_index": 4,
                "ptera_step_index": 3,
                "source_time_s": 3 * 0.11125,
                "cell_count": 8,
                "status": "completed",
                "birth_modes": ["first"] * 8,
                "event_sha256": "0" * 64,
                "parent_event_sha256": "1" * 64,
            })
            for name in (
                "a0_pre", "a0_post", "gamma_lev_new_m2_s",
                "gamma_lev_persisted_m2_s", "gamma_tev_new_m2_s",
                "kelvin_residual_m2_s",
            ):
                row[name] = [0.0] * 8
            sink.add_source_event(row)
            raise RuntimeError("synthetic callback failure")
    return SyntheticExecutor()
""".lstrip()
    audit_root = tmp_path / "generic-audit"
    audit_root.mkdir()
    token_path, _, _ = _make_external_audit_token(
        audit_root, monkeypatch, executor_source=executor_source
    )
    output = tmp_path / "generic-stop"
    result = runner.run_formal_attempt(
        output, replicate="A", dependency_audit_token=token_path
    )
    assert result["status"] == "STOP"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stop_code"] == "coupling_callback_error"
    assert summary["terminal_coordinate"] == {
        "transport_substeps": 32,
        "layer": None,
        "source_step_index": 4,
        "ptera_step_index": 3,
        "substep": None,
        "stage": None,
        "phase": "coupling_callback",
        "stage_began": False,
    }


def test_formal_early_return_and_bad_stop_terminal_publish_canonical_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    early_root = tmp_path / "early-return-audit"
    early_root.mkdir()
    early_token, _, _ = _make_external_audit_token(
        early_root,
        monkeypatch,
        executor_source=(
            "def build_fluxv_v5h11_w2_executor(api):\n"
            "    class EarlyReturnExecutor:\n"
            "        def attest_dependency_origins(self):\n"
            "            return None\n"
            "        def run_formal_matrix(self, *, levels, sink):\n"
            "            return None\n"
            "    return EarlyReturnExecutor()\n"
        ),
    )
    early_output = tmp_path / "early-return-stop"
    early = runner.run_formal_attempt(
        early_output,
        replicate="A",
        dependency_audit_token=early_token,
    )
    assert early["status"] == "STOP"
    early_summary = json.loads((early_output / "summary.json").read_text())
    assert early_summary["stop_code"] == "coupling_incomplete"
    assert early_summary["terminal_coordinate"] == {
        "transport_substeps": None,
        "layer": None,
        "source_step_index": None,
        "ptera_step_index": None,
        "substep": None,
        "stage": None,
        "phase": "coupling_completion",
        "stage_began": False,
    }
    assert {path.name for path in early_output.iterdir()} == set(runner.ARTIFACT_FILES)

    terminal_root = tmp_path / "bad-terminal-audit"
    terminal_root.mkdir()
    terminal_token, _, _ = _make_external_audit_token(
        terminal_root,
        monkeypatch,
        executor_source=(
            "def build_fluxv_v5h11_w2_executor(api):\n"
            "    class BadTerminalExecutor:\n"
            "        def attest_dependency_origins(self):\n"
            "            return None\n"
            "        def run_formal_matrix(self, *, levels, sink):\n"
            "            raise api.stop_constructor(\n"
            "                'untrusted_terminal',\n"
            "                {\n"
            "                    'transport_substeps': 128, 'layer': 3,\n"
            "                    'source_step_index': 6, 'ptera_step_index': 5,\n"
            "                    'substep': 128, 'stage': 3,\n"
            "                    'phase': 'forged', 'stage_began': False,\n"
            "                },\n"
            "                'synthetic malformed terminal',\n"
            "            )\n"
            "    return BadTerminalExecutor()\n"
        ),
    )
    terminal_output = tmp_path / "bad-terminal-stop"
    terminal = runner.run_formal_attempt(
        terminal_output,
        replicate="B",
        dependency_audit_token=terminal_token,
    )
    assert terminal["status"] == "STOP"
    terminal_summary = json.loads((terminal_output / "summary.json").read_text())
    assert terminal_summary["stop_code"] == "coupling_stop_contract_error"
    assert terminal_summary["terminal_coordinate"] == {
        "transport_substeps": None,
        "layer": None,
        "source_step_index": None,
        "ptera_step_index": None,
        "substep": None,
        "stage": None,
        "phase": "coupling_callback",
        "stage_began": False,
    }
    assert {path.name for path in terminal_output.iterdir()} == set(
        runner.ARTIFACT_FILES
    )


def test_audited_executor_must_not_import_runner_class_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_path, _, _ = _make_external_audit_token(
        tmp_path,
        monkeypatch,
        executor_source=(
            "import forward_flight_benchmarks.run_fluxv_v5h11_baik_w2\n"
            "def build_fluxv_v5h11_w2_executor(api):\n"
            "    return object()\n"
        ),
    )
    evidence = runner._verified_dependency_audit(token_path)
    with pytest.raises(runner.DependencyFreezeError, match="cannot import the runner"):
        runner._load_formal_executor(evidence)


def test_audited_executor_importfrom_alias_cannot_import_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_path, _, _ = _make_external_audit_token(
        tmp_path,
        monkeypatch,
        executor_source=(
            "from forward_flight_benchmarks import "
            "run_fluxv_v5h11_baik_w2 as runner\n"
            "def build_fluxv_v5h11_w2_executor(api):\n"
            "    return object()\n"
        ),
    )
    evidence = runner._verified_dependency_audit(token_path)
    with pytest.raises(runner.DependencyFreezeError, match="cannot import the runner"):
        runner._load_formal_executor(evidence)


def test_synthetic_fixture_cannot_masquerade_as_formal_pass(tmp_path: Path) -> None:
    formal = replace(_pass_records(), execution_mode=runner.FORMAL_EXECUTION_MODE)
    assert runner.FORMAL_W2_CONVECTIVE_DT == float(np.pi / 32.0)
    with pytest.raises(ValueError, match="formal W2 source provenance"):
        runner._validate_records(formal)
    with pytest.raises(PermissionError, match="audited formal entry"):
        runner.publish_artifact(formal, tmp_path / "formal", replicate="A")
    with pytest.raises(TypeError, match="unexpected keyword"):
        runner.publish_artifact(
            formal,
            tmp_path / "formal-authorized-but-unbound",
            replicate="A",
            _formal_publication_capability=object(),
        )
    assert not hasattr(runner, "_FORMAL_PUBLICATION_CAPABILITY")
    assert not hasattr(runner, "_publish_artifact_core")
    assert not hasattr(runner, "_publish_formal_after_dependency_rehash")
    assert not hasattr(runner, "_publish_formal_after_dependency_rehash_impl")
    with pytest.raises(ValueError, match="synthetic fixture"):
        runner.publish_artifact(
            _pass_records(),
            tmp_path / "synthetic-with-audit",
            replicate="A",
            dependency_audit={"status": "verified"},
        )


def test_strict_json_jsonl_and_csv_reject_duplicates_or_schema_drift(
    tmp_path: Path,
) -> None:
    duplicate_json = tmp_path / "duplicate.json"
    duplicate_json.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner._load_json(duplicate_json)

    duplicate_jsonl = tmp_path / "owner_events.jsonl"
    duplicate_jsonl.write_text(
        '{"status":"completed","status":"failed"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner._load_jsonl(duplicate_jsonl)

    duplicate_csv = tmp_path / "raw_steps.csv"
    duplicate_csv.write_text("status,status\ncompleted,completed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header/schema"):
        runner._read_csv(duplicate_csv)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("invariant_residual_over_slog_max", "normalized invariant-residual"),
        ("h_jacobian_frobenius", "h\\|\\|J_total\\|\\|F"),
        ("h_convective_over_sigma", "Galilean/sigma"),
    ),
)
def test_completed_stage_stability_gates_reject_hostile_finite_values(
    field: str, message: str
) -> None:
    records = _pass_records()
    first = dict(records.transport_stages[0])
    first[field] = 1.0e200
    hostile = replace(records, transport_stages=(first, *records.transport_stages[1:]))
    with pytest.raises(ValueError, match=message):
        runner._validate_records(hostile)


def test_stage_fields_bind_directly_to_compact_coupling_evidence() -> None:
    row = _stages(32, 1)[0]
    compact = {
        "schema_id": "synthetic-real-shape-compact-stage",
        "invariant_residual_over_slog_max": row["invariant_residual_over_slog_max"],
        "h_jacobian_frobenius": row["h_jacobian_frobenius"],
        "h_convective_over_sigma": row["h_convective_over_sigma"],
        "unrelated_digest": _H,
    }
    assert runner.compact_stage_stability_fields(compact) == {
        "invariant_residual_over_slog_max": 0.0,
        "h_jacobian_frobenius": 0.1,
        "h_convective_over_sigma": 0.01,
    }
    sink = runner.ArtifactSink()
    sink.add_source_event(_source_event(32, 4))
    sink.add_transport_stage_from_compact_evidence(row, compact)
    assert len(sink.transport_stages) == 1
    invented = dict(row)
    invented["h_convective_over_sigma"] = 0.02
    with pytest.raises(ValueError, match="differ from compact evidence"):
        sink.add_transport_stage_from_compact_evidence(invented, compact)


def test_stop_prefix_rejects_terminal_coordinate_and_commit_order_drift() -> None:
    full = _pass_records()
    failed = _failed_stage(
        dict(full.transport_stages[2]),
        failure_type="SyntheticFailure",
        failure_message="terminal",
    )
    terminal = {
        "transport_substeps": 32,
        "layer": 1,
        "source_step_index": 4,
        "ptera_step_index": 4,
        "substep": 1,
        "stage": 3,
        "phase": "transport_stage",
        "stage_began": True,
    }
    wrong_terminal = runner.ArtifactRecords(
        execution_mode=runner.SYNTHETIC_EXECUTION_MODE,
        status="STOP",
        raw_steps=(),
        source_events=full.source_events[:1],
        owner_events=(),
        particle_counts=(),
        raw_loads=(),
        transport_stages=(*full.transport_stages[:2], failed),
        trajectories=(),
        terminal_coordinate=terminal,
        stop_code="synthetic",
        stop_message="synthetic",
    )
    with pytest.raises(ValueError, match="coordinates disagree"):
        runner._validate_records(wrong_terminal)

    premature = replace(
        wrong_terminal,
        raw_steps=full.raw_steps[:1],
        owner_events=full.owner_events[:1],
        particle_counts=full.particle_counts[:1],
        raw_loads=full.raw_loads[:17],
        trajectories=full.trajectories[:1],
        terminal_coordinate={**terminal, "ptera_step_index": 3},
    )
    with pytest.raises(ValueError, match="before all 3N stages"):
        runner._validate_records(premature)

    incomplete_load = runner.ArtifactRecords(
        execution_mode=runner.SYNTHETIC_EXECUTION_MODE,
        status="STOP",
        raw_steps=full.raw_steps[:1],
        source_events=full.source_events[:1],
        owner_events=full.owner_events[:1],
        particle_counts=full.particle_counts[:1],
        raw_loads=full.raw_loads[:16],
        transport_stages=full.transport_stages[: 3 * 32],
        trajectories=full.trajectories[:1],
        terminal_coordinate={
            "transport_substeps": 32,
            "layer": 1,
            "source_step_index": 4,
            "ptera_step_index": 3,
            "substep": 32,
            "stage": 3,
            "phase": "load_commit",
            "stage_began": False,
        },
        stop_code="synthetic",
        stop_message="synthetic",
    )
    with pytest.raises(ValueError, match="17-row load block"):
        runner._validate_records(incomplete_load)


def test_unbegun_stop_binds_all_six_fields_to_next_durable_stage() -> None:
    full = _pass_records()
    terminal = {
        "transport_substeps": 32,
        "layer": 1,
        "source_step_index": 4,
        "ptera_step_index": 3,
        "substep": 1,
        "stage": 3,
        "phase": "coupling_callback",
        "stage_began": False,
    }
    stopped = runner.ArtifactRecords(
        execution_mode=runner.SYNTHETIC_EXECUTION_MODE,
        status="STOP",
        raw_steps=(),
        source_events=full.source_events[:1],
        owner_events=(),
        particle_counts=(),
        raw_loads=(),
        transport_stages=full.transport_stages[:2],
        trajectories=(),
        terminal_coordinate=terminal,
        stop_code="synthetic",
        stop_message="synthetic",
    )
    runner._validate_records(stopped)
    for field, wrong in (
        ("transport_substeps", 64),
        ("layer", 2),
        ("source_step_index", 5),
        ("ptera_step_index", 4),
        ("substep", 2),
        ("stage", 1),
    ):
        hostile = replace(
            stopped,
            terminal_coordinate={**terminal, field: wrong},
        )
        with pytest.raises(ValueError, match="durable-prefix coordinate"):
            runner._validate_records(hostile)


def test_source_prefix_cannot_skip_ahead_of_completed_layers(
    tmp_path: Path,
) -> None:
    full = _pass_records()
    sink = runner.ArtifactSink()
    sink.add_source_event(full.source_events[0])
    with pytest.raises(ValueError, match="single current uncommitted layer"):
        sink.add_source_event(full.source_events[1])

    terminal = {
        "transport_substeps": 32,
        "layer": None,
        "source_step_index": 5,
        "ptera_step_index": 4,
        "substep": None,
        "stage": None,
        "phase": "source_post",
        "stage_began": False,
    }
    ahead = runner.ArtifactRecords(
        execution_mode=runner.SYNTHETIC_EXECUTION_MODE,
        status="STOP",
        raw_steps=(),
        source_events=full.source_events[:2],
        owner_events=(),
        particle_counts=(),
        raw_loads=(),
        transport_stages=(),
        trajectories=(),
        terminal_coordinate=terminal,
        stop_code="synthetic",
        stop_message="synthetic",
    )
    with pytest.raises(ValueError, match="single current uncommitted layer"):
        runner._validate_records(ahead)

    one_source = replace(
        ahead,
        source_events=full.source_events[:1],
        terminal_coordinate={
            **terminal,
            "source_step_index": 4,
            "ptera_step_index": 3,
        },
    )
    output = tmp_path / "one-source-stop"
    runner.publish_artifact(
        one_source,
        output,
        replicate="A",
        run_provenance=_provenance(output, "A"),
    )
    (output / "source_events.csv").write_bytes(
        runner._csv_bytes(runner.SOURCE_EVENT_FIELDS, full.source_events[:2])
    )
    summary = json.loads((output / "summary.json").read_text())
    summary["terminal_coordinate"] = terminal
    summary["row_counts"]["source_events"] = 2
    with pytest.raises(ValueError, match="single current uncommitted layer"):
        runner._validate_serialized_counts(output, summary)


def test_sink_atomic_layer_commit_waits_for_full_stage_block() -> None:
    full = _pass_records()
    sink = runner.ArtifactSink()
    sink.add_source_event(full.source_events[0])
    for row in full.transport_stages[:2]:
        _append_compact_stage(sink, dict(row))
    with pytest.raises(ValueError, match="all 3N completed stages"):
        sink.commit_completed_layer(
            raw_step=full.raw_steps[0],
            source_event=full.source_events[0],
            owner_event=full.owner_events[0],
            particle_count=full.particle_counts[0],
            raw_loads=full.raw_loads[:17],
            trajectory_array_record=full.trajectories[0],
        )
    assert not sink.raw_steps
    assert not sink.raw_loads

    sink = runner.ArtifactSink()
    sink.add_source_event(full.source_events[0])
    for row in full.transport_stages[: 3 * 32]:
        _append_compact_stage(sink, dict(row))
    sink.commit_completed_layer(
        raw_step=full.raw_steps[0],
        source_event=full.source_events[0],
        owner_event=full.owner_events[0],
        particle_count=full.particle_counts[0],
        raw_loads=full.raw_loads[:17],
        trajectory_array_record=full.trajectories[0],
    )
    assert len(sink.raw_steps) == 1
    assert len(sink.source_events) == 1
    assert len(sink.owner_events) == 1
    assert len(sink.particle_counts) == 1
    assert len(sink.raw_loads) == 17
    assert len(sink.trajectories) == 1


def test_sink_rejects_bad_source_and_stage_before_durable_prefix(
    tmp_path: Path,
) -> None:
    full = _pass_records()
    sink = runner.ArtifactSink()
    bad_source = dict(full.source_events[0])
    bad_source["a0_pre"] = [1.0]
    with pytest.raises(ValueError, match="source vector"):
        sink.add_source_event(bad_source)
    assert sink.source_events == ()
    sink.add_source_event(full.source_events[0])

    bad_stage = dict(full.transport_stages[0])
    bad_stage["pre_state_sha256"] = "bad"
    with pytest.raises(ValueError, match="transport-stage completed digest"):
        _append_compact_stage(sink, bad_stage)
    assert sink.transport_stages == ()
    _append_compact_stage(sink, dict(full.transport_stages[0]))

    stop = sink.freeze(
        execution_mode=runner.SYNTHETIC_EXECUTION_MODE,
        status="STOP",
        terminal_coordinate=runner._terminal_coordinate_from_sink(
            sink, phase="synthetic_callback"
        ),
        stop_code="synthetic_callback_error",
        stop_message="bad callback evidence was rejected before append",
    )
    output = tmp_path / "last-good-prefix-stop"
    runner.publish_artifact(
        stop,
        output,
        replicate="A",
        run_provenance=_provenance(output, "A"),
    )
    assert runner.verify_artifact(output)["status"] == "STOP"
    assert len(runner._read_csv(output / "source_events.csv")) == 1
    assert len(runner._read_csv(output / "transport_stages.csv")) == 1


def test_sink_commits_three_source_preappended_layers_in_order() -> None:
    full = _pass_records()
    sink = runner.ArtifactSink()
    stage_offset = 0
    for layer_index in range(3):
        sink.add_source_event(full.source_events[layer_index])
        stage_count = 3 * 32
        for row in full.transport_stages[stage_offset : stage_offset + stage_count]:
            _append_compact_stage(sink, dict(row))
        stage_offset += stage_count
        load_offset = 17 * layer_index
        sink.commit_completed_layer(
            raw_step=full.raw_steps[layer_index],
            source_event=full.source_events[layer_index],
            owner_event=full.owner_events[layer_index],
            particle_count=full.particle_counts[layer_index],
            raw_loads=full.raw_loads[load_offset : load_offset + 17],
            trajectory_array_record=full.trajectories[layer_index],
        )
    assert len(sink.raw_steps) == 3
    assert len(sink.source_events) == 3
    assert len(sink.raw_loads) == 51


def test_producer_rejects_hostile_completed_crosslink_evidence() -> None:
    full = _pass_records()

    owner = dict(full.owner_events[0])
    transport = dict(owner["transport_event"])
    transport["previous_transport_event_sha256"] = "f" * 64
    transport["transport_event_sha256"] = runner._v5h10_event_digest(
        "fluxv-v5h10-row-transport-event-v1",
        transport,
        runner.TRANSPORT_EVENT_FIELDS,
        "transport_event_sha256",
    )
    owner["transport_event"] = transport
    with pytest.raises(ValueError, match="transport genesis"):
        runner._validate_records(
            replace(full, owner_events=(owner, *full.owner_events[1:]))
        )

    with pytest.raises(ValueError, match="exact prehistory parent"):
        runner._validate_records(_with_forged_first_source_genesis(full))

    owner = dict(full.owner_events[0])
    transport = dict(owner["transport_event"])
    transport["transport_event_sha256"] = "f" * 64
    owner["transport_event"] = transport
    with pytest.raises(ValueError, match="transport-event replay digest"):
        runner._validate_records(
            replace(full, owner_events=(owner, *full.owner_events[1:]))
        )

    owner = dict(full.owner_events[1])
    commit = dict(owner["commit_event"])
    commit["upstream_nodes_sha256"] = "f" * 64
    owner["commit_event"] = commit
    with pytest.raises(ValueError, match="commit-event replay digest"):
        runner._validate_records(
            replace(
                full,
                owner_events=(full.owner_events[0], owner, *full.owner_events[2:]),
            )
        )

    trajectory = dict(full.trajectories[1])
    arrays = dict(trajectory["arrays"])
    start_positions = np.asarray(arrays["start_positions"]).copy()
    start_positions[0, 0] += 7.0
    arrays["start_positions"] = start_positions
    trajectory["arrays"] = arrays
    stages = list(full.transport_stages)
    first_stage_index = next(
        index
        for index, row in enumerate(stages)
        if runner._stage_key(row) == (32, 2, 1, 1)
    )
    first_stage = dict(stages[first_stage_index])
    first_stage["pre_state_sha256"] = runner._stream_state_sha256_from_arrays(
        start_positions,
        np.asarray(arrays["start_gamma"]),
        np.asarray(arrays["start_sigma"]),
    )
    stages[first_stage_index] = first_stage
    with pytest.raises(ValueError, match="reset cumulative position/sigma"):
        runner._validate_records(
            replace(
                full,
                trajectories=(full.trajectories[0], trajectory, *full.trajectories[2:]),
                transport_stages=tuple(stages),
            )
        )

    raw = dict(full.raw_steps[0])
    raw["raw_cl"] = 999.0
    with pytest.raises(ValueError, match="CL/CD"):
        runner._validate_records(replace(full, raw_steps=(raw, *full.raw_steps[1:])))

    raw = dict(full.raw_steps[0])
    raw["stream_stage_chain_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="stage-chain"):
        runner._validate_records(replace(full, raw_steps=(raw, *full.raw_steps[1:])))

    raw = dict(full.raw_steps[0])
    raw["no_penetration_max_abs"] = 1.0e-13
    with pytest.raises(ValueError, match="no-penetration maximum"):
        runner._validate_records(replace(full, raw_steps=(raw, *full.raw_steps[1:])))

    source = dict(full.source_events[0])
    source["a0_pre"] = [1.0, *([0.0] * 7)]
    with pytest.raises(ValueError, match="source (generation|aggregate vector)"):
        runner._validate_records(
            replace(full, source_events=(source, *full.source_events[1:]))
        )

    source = dict(full.source_events[0])
    source["event_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source (generation|aggregate event)"):
        runner._validate_records(
            replace(full, source_events=(source, *full.source_events[1:]))
        )

    stage = dict(full.transport_stages[1])
    stage["pre_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="state/tracer prefix"):
        runner._validate_records(
            replace(
                full,
                transport_stages=(
                    full.transport_stages[0],
                    stage,
                    *full.transport_stages[2:],
                ),
            )
        )

    owner = dict(full.owner_events[1])
    owner["appended_particle_ids"] = ["P1"]
    commit = dict(owner["commit_event"])
    commit["appended_indices"] = [1]
    owner["commit_event"] = commit
    with pytest.raises(ValueError, match="append ancestry"):
        runner._validate_records(
            replace(
                full,
                owner_events=(full.owner_events[0], owner, *full.owner_events[2:]),
            )
        )

    owner = dict(full.owner_events[1])
    commit = dict(owner["commit_event"])
    commit["clone_count"] = -1
    owner["commit_event"] = commit
    with pytest.raises(ValueError, match="owner clone_count"):
        runner._validate_records(
            replace(
                full,
                owner_events=(full.owner_events[0], owner, *full.owner_events[2:]),
            )
        )

    trajectory = dict(full.trajectories[0])
    arrays = dict(trajectory["arrays"])
    arrays["probe_velocity"] = np.zeros((3, 3), dtype=np.float64)
    trajectory["arrays"] = arrays
    with pytest.raises(ValueError, match="direct replay oracle"):
        runner._validate_records(
            replace(full, trajectories=(trajectory, *full.trajectories[1:]))
        )

    trajectory = dict(full.trajectories[0])
    arrays = dict(trajectory["arrays"])
    arrays["force"] = np.zeros(3, dtype=np.float64)
    trajectory["arrays"] = arrays
    with pytest.raises(ValueError, match="TOTAL load"):
        runner._validate_records(
            replace(full, trajectories=(trajectory, *full.trajectories[1:]))
        )

    trajectory = dict(full.trajectories[0])
    arrays = dict(trajectory["arrays"])
    material = np.asarray(arrays["material_tracer_positions"]).copy()
    material[0, 0] += 1.0e-6
    arrays["material_tracer_positions"] = material
    trajectory["arrays"] = arrays
    with pytest.raises(ValueError, match="ID-aligned particle support"):
        runner._validate_records(
            replace(full, trajectories=(trajectory, *full.trajectories[1:]))
        )

    trajectory = dict(full.trajectories[0])
    arrays = dict(trajectory["arrays"])
    arrays["invariant_start"] = np.full(2, 123.0, dtype=np.float64)
    trajectory["arrays"] = arrays
    with pytest.raises(ValueError, match="invariant_start is not replayable"):
        runner._validate_records(
            replace(full, trajectories=(trajectory, *full.trajectories[1:]))
        )

    trajectory = dict(full.trajectories[0])
    metadata = dict(trajectory["metadata"])
    manifests = [dict(item) for item in metadata["source_cell_manifests"]]
    manifests[0]["producer_manifest_sha256"] = "f" * 64
    metadata["source_cell_manifests"] = manifests
    metadata["source_cell_manifest_sha256"] = runner._payload_sha256(
        "fluxv-v5h11-source-cell-manifests-v1", manifests
    )
    trajectory["metadata"] = metadata
    with pytest.raises(ValueError, match="producer manifest digest"):
        runner._validate_records(
            replace(full, trajectories=(trajectory, *full.trajectories[1:]))
        )

    load_rows = [dict(row) for row in full.raw_loads[:17]]
    for index, row in enumerate(load_rows[:16]):
        row["force_x_n"] = 0.0
        row["force_y_n"] = 1.0e308 if index % 2 == 0 else -1.0e308
    load_rows[-1]["force_x_n"] = 1.0e294
    load_rows[-1]["force_y_n"] = 0.0
    with pytest.raises(ValueError, match="panel force"):
        runner._validate_load_block(full.raw_steps[0], load_rows)

    trajectory = _trajectory(32, 1)
    arrays = trajectory["arrays"]
    assert isinstance(arrays, dict)
    for gamma_name, sigma_name, invariant_name in (
        ("start_gamma", "start_sigma", "invariant_start"),
        ("end_gamma", "end_sigma", "invariant_end"),
    ):
        gamma = np.asarray(arrays[gamma_name]).copy()
        sigma = np.asarray(arrays[sigma_name]).copy()
        invariant = np.asarray(arrays[invariant_name]).copy()
        gamma[0] = (1.0e308, 0.0, 0.0)
        sigma[0] = 1.0
        invariant[0] = 1.0e308
        invariant[1] = 1.0e294
        arrays[gamma_name] = gamma
        arrays[sigma_name] = sigma
        arrays[invariant_name] = invariant
    with pytest.raises(ValueError, match="invariant_start is not replayable"):
        runner._validate_completed_trajectory_semantics(trajectory)


def test_owner_replay_rejects_nonfinite_derived_gamma_delta() -> None:
    full = _pass_records()
    trajectories = [dict(full.trajectories[index]) for index in range(2)]
    previous_arrays = dict(trajectories[0]["arrays"])
    previous_gamma = np.asarray(previous_arrays["end_gamma"]).copy()
    previous_gamma[0] = (-1.0e308, 0.0, 0.0)
    previous_arrays["end_gamma"] = previous_gamma
    trajectories[0]["arrays"] = previous_arrays

    current_arrays = dict(trajectories[1]["arrays"])
    current_gamma = np.asarray(current_arrays["start_gamma"]).copy()
    current_gamma[0] = (1.0e308, 0.0, 0.0)
    current_arrays["start_gamma"] = current_gamma
    trajectories[1]["arrays"] = current_arrays

    owners = [dict(full.owner_events[index]) for index in range(2)]
    previous_transport = dict(owners[0]["transport_event"])
    previous_transport["transported_arrays_sha256"] = runner._v5h10_digest(
        "fluxv-v5h10-transported-arrays-v1",
        previous_arrays["end_positions"],
        previous_gamma,
        previous_arrays["end_sigma"],
    )
    previous_transport["transport_event_sha256"] = runner._v5h10_event_digest(
        "fluxv-v5h10-row-transport-event-v1",
        previous_transport,
        runner.TRANSPORT_EVENT_FIELDS,
        "transport_event_sha256",
    )
    owners[0]["transport_event"] = previous_transport

    commit = dict(owners[1]["commit_event"])
    before = previous_gamma[np.asarray(commit["changed_indices"], dtype=np.int64)]
    after = current_gamma[np.asarray(commit["changed_indices"], dtype=np.int64)]
    with np.errstate(over="ignore"):
        added = after - before
    commit["parent_transport_event_sha256"] = previous_transport[
        "transport_event_sha256"
    ]
    commit["before_gamma_sha256"] = runner._v5h10_digest("v5h10-before-gamma", before)
    commit["after_gamma_sha256"] = runner._v5h10_digest("v5h10-after-gamma", after)
    commit["added_gamma_sha256"] = runner._v5h10_digest("v5h10-added-gamma", added)
    commit["event_sha256"] = runner._v5h10_event_digest(
        "fluxv-v5h10-row-event-v1",
        commit,
        runner.COMMIT_EVENT_FIELDS,
        "event_sha256",
        tuple_fields=("changed_indices", "appended_indices"),
    )
    owners[1]["commit_event"] = commit
    current_transport = dict(owners[1]["transport_event"])
    current_transport["previous_transport_event_sha256"] = previous_transport[
        "transport_event_sha256"
    ]
    current_transport["transport_event_sha256"] = runner._v5h10_event_digest(
        "fluxv-v5h10-row-transport-event-v1",
        current_transport,
        runner.TRANSPORT_EVENT_FIELDS,
        "transport_event_sha256",
    )
    owners[1]["transport_event"] = current_transport

    with pytest.raises(ValueError, match="Gamma delta is non-finite"):
        runner._validate_owner_semantics(
            owners,
            full.particle_counts[:2],
            trajectories,
        )


def test_trajectory_macro_invariant_and_b1_gamma_domain_are_replayed() -> None:
    runner._validate_completed_trajectory_semantics(_trajectory(32, 1))
    gate = np.float64(runner.STAGE_INVARIANT_RESIDUAL_OVER_SLOG_MAX)
    runner._require_invariant_log_gate(
        np.asarray([np.nextafter(gate, np.float64(0.0))]),
        np.ones(1, dtype=np.float64),
    )
    runner._require_invariant_log_gate(np.asarray([gate]), np.ones(1, dtype=np.float64))
    with pytest.raises(ValueError, match="log residual exceeds frozen gate"):
        runner._require_invariant_log_gate(
            np.asarray([np.nextafter(gate, np.float64(np.inf))]),
            np.ones(1, dtype=np.float64),
        )

    drifted = _trajectory(32, 1)
    arrays = dict(drifted["arrays"])
    end_sigma = np.asarray(arrays["end_sigma"]).copy()
    end_sigma[0] *= 1.0 + 2.0e-7
    arrays["end_sigma"] = end_sigma
    arrays["invariant_end"] = (
        runner._stable_row_norms(np.asarray(arrays["end_gamma"])) * end_sigma**2
    )
    drifted["arrays"] = arrays
    with pytest.raises(ValueError, match="macro invariant log residual"):
        runner._validate_completed_trajectory_semantics(drifted)

    underflow = _trajectory(32, 1)
    arrays = dict(underflow["arrays"])
    for gamma_name, sigma_name, invariant_name in (
        ("start_gamma", "start_sigma", "invariant_start"),
        ("end_gamma", "end_sigma", "invariant_end"),
    ):
        gamma = np.asarray(arrays[gamma_name]).copy()
        gamma[0] = (1.0e-200, 0.0, 0.0)
        arrays[gamma_name] = gamma
        arrays[invariant_name] = (
            runner._stable_row_norms(gamma) * np.asarray(arrays[sigma_name]) ** 2
        )
    underflow["arrays"] = arrays
    with pytest.raises(ValueError, match="frozen B1 threshold"):
        runner._validate_completed_trajectory_semantics(underflow)

    zero_sigma_drift = _trajectory(32, 1)
    arrays = dict(zero_sigma_drift["arrays"])
    signed_zero = np.asarray((-0.0, 0.0, -0.0), dtype=np.float64)
    start_gamma = np.asarray(arrays["start_gamma"]).copy()
    end_gamma = np.asarray(arrays["end_gamma"]).copy()
    start_gamma[0] = signed_zero
    end_gamma[0] = signed_zero
    end_sigma = np.asarray(arrays["end_sigma"]).copy()
    end_sigma[0] = 12345.0
    arrays["start_gamma"] = start_gamma
    arrays["end_gamma"] = end_gamma
    arrays["end_sigma"] = end_sigma
    arrays["invariant_start"] = (
        runner._stable_row_norms(start_gamma) * np.asarray(arrays["start_sigma"]) ** 2
    )
    arrays["invariant_end"] = runner._stable_row_norms(end_gamma) * end_sigma**2
    zero_sigma_drift["arrays"] = arrays
    with pytest.raises(ValueError, match="exact-zero Gamma/sigma bits"):
        runner._validate_completed_trajectory_semantics(zero_sigma_drift)

    signed_zero_drift = _trajectory(32, 1)
    arrays = dict(signed_zero_drift["arrays"])
    start_gamma = np.asarray(arrays["start_gamma"]).copy()
    end_gamma = np.asarray(arrays["end_gamma"]).copy()
    start_gamma[0] = (-0.0, 0.0, 0.0)
    end_gamma[0] = (0.0, 0.0, 0.0)
    end_sigma = np.asarray(arrays["end_sigma"]).copy()
    end_sigma[0] = np.asarray(arrays["start_sigma"])[0]
    arrays["start_gamma"] = start_gamma
    arrays["end_gamma"] = end_gamma
    arrays["end_sigma"] = end_sigma
    arrays["invariant_start"] = (
        runner._stable_row_norms(start_gamma) * np.asarray(arrays["start_sigma"]) ** 2
    )
    arrays["invariant_end"] = runner._stable_row_norms(end_gamma) * end_sigma**2
    signed_zero_drift["arrays"] = arrays
    with pytest.raises(ValueError, match="exact-zero Gamma/sigma bits"):
        runner._validate_completed_trajectory_semantics(signed_zero_drift)


def test_source_v3_manifest_replays_new_audit_only_evidence() -> None:
    records = _pass_records()
    runner._validate_records(records)
    source = records.source_events[0]
    trajectory = records.trajectories[0]
    metadata = trajectory["metadata"]
    assert isinstance(metadata, dict)
    manifests = metadata["source_cell_manifests"]
    assert isinstance(manifests, list)
    manifest = manifests[0]
    assert manifest["provenance"]["interface_id"] == runner.SOURCE_INTERFACE_ID
    assert manifest["provenance"]["backend_id"] == runner.SOURCE_BACKEND_ID
    assert set(manifest["kelvin_ledger"]) == set(runner.SOURCE_KELVIN_LEDGER_FIELDS)
    assert manifest["kelvin_ledger"]["gamma_tev_new_te_only_provisional"] == 0.125
    assert source["gamma_tev_new_m2_s"] == [0.0] * 8

    placement = dict(manifest["lev_placement"])
    placement["q_old_wake_over_u_backend_world"] = [1.0, 0.0]
    placement["q_provisional_tev_over_u_backend_world"] = [-1.0, 0.0]
    with pytest.raises(ValueError, match="provisional-TEV velocity"):
        runner._validate_source_placement_v3(
            placement,
            family="lev",
            expected_mode="first",
            legacy_birth=manifest["lev_birth_position_over_chord_backend_world"],
            delta_time_convective=manifest["delta_time_convective"],
            section_lineage_id=manifest["lineage"]["section_lineage_id"],
            source_step=manifest["lineage"]["source_step_index"],
            provisional_tev_gamma=manifest["kelvin_ledger"][
                "gamma_tev_new_te_only_provisional"
            ],
            provisional_tev_birth=manifest[
                "tev_birth_position_over_chord_backend_world"
            ],
            resolved_core_radius=manifest["provenance"]["resolved_core_radius_chord"],
        )

    continuous_manifest = records.trajectories[1]["metadata"]["source_cell_manifests"][
        0
    ]
    continuous = dict(continuous_manifest["lev_placement"])
    continuous["continuous_parent_source_id"] = "forged-parent"
    continuous["continuous_parent_position_over_chord_backend_world"] = [0.125, 0.0]
    with pytest.raises(ValueError, match="continuous placement is not replayable"):
        runner._validate_source_placement_v3(
            continuous,
            family="lev",
            expected_mode="continuous",
            legacy_birth=continuous_manifest[
                "lev_birth_position_over_chord_backend_world"
            ],
            delta_time_convective=continuous_manifest["delta_time_convective"],
            section_lineage_id=continuous_manifest["lineage"]["section_lineage_id"],
            source_step=continuous_manifest["lineage"]["source_step_index"],
            provisional_tev_gamma=continuous_manifest["kelvin_ledger"][
                "gamma_tev_new_te_only_provisional"
            ],
            provisional_tev_birth=continuous_manifest[
                "tev_birth_position_over_chord_backend_world"
            ],
            resolved_core_radius=continuous_manifest["provenance"][
                "resolved_core_radius_chord"
            ],
        )

    malformed_trajectory = dict(trajectory)
    malformed_metadata = json.loads(runner._json_bytes(metadata))
    malformed_trajectory["metadata"] = malformed_metadata
    malformed_manifest = malformed_metadata["source_cell_manifests"][0]
    del malformed_manifest["kelvin_ledger"]["gamma_tev_new_te_only_provisional"]
    malformed_unsigned = dict(malformed_manifest)
    del malformed_unsigned["producer_manifest_sha256"]
    malformed_manifest["producer_manifest_sha256"] = sha256(
        runner.SOURCE_EVENT_DIGEST_PREFIX + runner._json_bytes(malformed_unsigned)
    ).hexdigest()
    with pytest.raises(ValueError, match="Kelvin-ledger schema"):
        runner._validate_source_manifest_binding(source, malformed_trajectory, None)

    hostile_ledger = dict(manifest["kelvin_ledger"])
    for name in (
        "gamma_old_tev_persisted",
        "gamma_old_lev_persisted",
        "gamma_deleted_before",
        "gamma_tev_persisted_after",
        "gamma_lev_persisted_after",
        "gamma_deleted_after",
    ):
        hostile_ledger[name] = 9.0e307
    with pytest.raises(ValueError, match="overflowed replay"):
        runner._validate_source_kelvin_ledger_v3(
            hostile_ledger,
            manifest=manifest,
            source_step=4,
            active=True,
        )


def test_serialized_replay_rejects_probe_source_and_load_crosslink_tampering(
    pass_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    source_artifact, _ = pass_pair

    probe_scratch = tmp_path / "zero-probes"
    shutil.copytree(source_artifact, probe_scratch)
    trajectory_document = json.loads(
        (probe_scratch / "trajectory_arrays.json").read_text()
    )
    trajectory_document["records"][0]["arrays"]["probe_velocity"] = runner.encode_array(
        np.zeros((3, 3), dtype=np.float64)
    )
    (probe_scratch / "trajectory_arrays.json").write_bytes(
        runner._json_bytes(trajectory_document) + b"\n"
    )
    with pytest.raises(ValueError, match="direct replay oracle"):
        runner._validate_serialized_counts(
            probe_scratch,
            json.loads((probe_scratch / "summary.json").read_text()),
        )

    source_scratch = tmp_path / "source-vector"
    shutil.copytree(source_artifact, source_scratch)
    source_rows = runner._read_csv(source_scratch / "source_events.csv")
    source_rows[0]["a0_pre"] = runner._json_bytes([1.0, *([0.0] * 7)]).decode("utf-8")
    (source_scratch / "source_events.csv").write_bytes(
        runner._csv_bytes(runner.SOURCE_EVENT_FIELDS, source_rows)
    )
    with pytest.raises(ValueError, match="source (generation|aggregate vector)"):
        runner._validate_serialized_counts(
            source_scratch,
            json.loads((source_scratch / "summary.json").read_text()),
        )

    load_scratch = tmp_path / "raw-cl"
    shutil.copytree(source_artifact, load_scratch)
    raw_rows = runner._read_csv(load_scratch / "raw_steps.csv")
    raw_rows[0]["raw_cl"] = "999"
    (load_scratch / "raw_steps.csv").write_bytes(
        runner._csv_bytes(runner.RAW_STEP_FIELDS, raw_rows)
    )
    with pytest.raises(ValueError, match="CL/CD"):
        runner._validate_serialized_counts(
            load_scratch,
            json.loads((load_scratch / "summary.json").read_text()),
        )

    load_scale_scratch = tmp_path / "cross-axis-load-scale"
    shutil.copytree(source_artifact, load_scale_scratch)
    load_rows = runner._read_csv(load_scale_scratch / "raw_loads.csv")
    for index, row in enumerate(load_rows[:16]):
        row["force_x_n"] = "0"
        row["force_y_n"] = "1e308" if index % 2 == 0 else "-1e308"
    load_rows[16]["force_x_n"] = "1e294"
    load_rows[16]["force_y_n"] = "0"
    (load_scale_scratch / "raw_loads.csv").write_bytes(
        runner._csv_bytes(runner.RAW_LOAD_FIELDS, load_rows)
    )
    trajectory_document = json.loads(
        (load_scale_scratch / "trajectory_arrays.json").read_text()
    )
    total_force = runner.decode_array(
        trajectory_document["records"][0]["arrays"]["force"]
    ).copy()
    total_force[0] = 1.0e294
    total_force[1] = 0.0
    trajectory_document["records"][0]["arrays"]["force"] = runner.encode_array(
        total_force
    )
    (load_scale_scratch / "trajectory_arrays.json").write_bytes(
        runner._json_bytes(trajectory_document) + b"\n"
    )
    with pytest.raises(ValueError, match="panel force"):
        runner._validate_serialized_counts(
            load_scale_scratch,
            json.loads((load_scale_scratch / "summary.json").read_text()),
        )

    invariant_scratch = tmp_path / "cross-particle-invariant-scale"
    shutil.copytree(source_artifact, invariant_scratch)
    trajectory_document = json.loads(
        (invariant_scratch / "trajectory_arrays.json").read_text()
    )
    first = trajectory_document["records"][0]["arrays"]
    for gamma_name, sigma_name, invariant_name in (
        ("start_gamma", "start_sigma", "invariant_start"),
        ("end_gamma", "end_sigma", "invariant_end"),
    ):
        gamma = runner.decode_array(first[gamma_name]).copy()
        sigma = runner.decode_array(first[sigma_name]).copy()
        invariant = runner.decode_array(first[invariant_name]).copy()
        gamma[0] = (1.0e308, 0.0, 0.0)
        sigma[0] = 1.0
        invariant[0] = 1.0e308
        invariant[1] = 1.0e294
        first[gamma_name] = runner.encode_array(gamma)
        first[sigma_name] = runner.encode_array(sigma)
        first[invariant_name] = runner.encode_array(invariant)
    (invariant_scratch / "trajectory_arrays.json").write_bytes(
        runner._json_bytes(trajectory_document) + b"\n"
    )
    with pytest.raises(ValueError, match="invariant_start is not replayable"):
        runner._validate_serialized_counts(
            invariant_scratch,
            json.loads((invariant_scratch / "summary.json").read_text()),
        )

    macro_scratch = tmp_path / "macro-invariant-drift"
    shutil.copytree(source_artifact, macro_scratch)
    trajectory_document = json.loads(
        (macro_scratch / "trajectory_arrays.json").read_text()
    )
    first = trajectory_document["records"][0]["arrays"]
    end_gamma = runner.decode_array(first["end_gamma"])
    end_sigma = runner.decode_array(first["end_sigma"]).copy()
    end_sigma[0] *= 1.0 + 2.0e-7
    first["end_sigma"] = runner.encode_array(end_sigma)
    first["invariant_end"] = runner.encode_array(
        runner._stable_row_norms(end_gamma) * end_sigma**2
    )
    probe_velocity, probe_jacobian = runner._direct_gaussian_erf_probe_oracle(
        runner.decode_array(first["end_positions"]), end_gamma, end_sigma
    )
    first["probe_velocity"] = runner.encode_array(probe_velocity)
    first["probe_jacobian"] = runner.encode_array(probe_jacobian)
    (macro_scratch / "trajectory_arrays.json").write_bytes(
        runner._json_bytes(trajectory_document) + b"\n"
    )
    with pytest.raises(ValueError, match="macro invariant log residual"):
        runner._validate_serialized_counts(
            macro_scratch,
            json.loads((macro_scratch / "summary.json").read_text()),
        )

    genesis_scratch = tmp_path / "source-genesis"
    shutil.copytree(source_artifact, genesis_scratch)
    forged = _with_forged_first_source_genesis(_pass_records())
    (genesis_scratch / "source_events.csv").write_bytes(
        runner._csv_bytes(runner.SOURCE_EVENT_FIELDS, forged.source_events)
    )
    (genesis_scratch / "trajectory_arrays.json").write_bytes(
        runner._json_bytes(runner._trajectory_document(forged)) + b"\n"
    )
    with pytest.raises(ValueError, match="exact prehistory parent"):
        runner._validate_serialized_counts(
            genesis_scratch,
            json.loads((genesis_scratch / "summary.json").read_text()),
        )

    cloud_scratch = tmp_path / "cloud-reset"
    shutil.copytree(source_artifact, cloud_scratch)
    trajectory_document = json.loads(
        (cloud_scratch / "trajectory_arrays.json").read_text()
    )
    trajectory_row = trajectory_document["records"][1]
    start_positions = runner.decode_array(
        trajectory_row["arrays"]["start_positions"]
    ).copy()
    start_positions[0, 0] += 7.0
    trajectory_row["arrays"]["start_positions"] = runner.encode_array(start_positions)
    (cloud_scratch / "trajectory_arrays.json").write_bytes(
        runner._json_bytes(trajectory_document) + b"\n"
    )
    stage_rows = runner._read_csv(cloud_scratch / "transport_stages.csv")
    first_layer_two_stage = next(
        row
        for row in stage_rows
        if (
            int(row["transport_substeps"]),
            int(row["layer"]),
            int(row["substep"]),
            int(row["stage"]),
        )
        == (32, 2, 1, 1)
    )
    first_layer_two_stage["pre_state_sha256"] = runner._stream_state_sha256_from_arrays(
        start_positions,
        runner.decode_array(trajectory_row["arrays"]["start_gamma"]),
        runner.decode_array(trajectory_row["arrays"]["start_sigma"]),
    )
    (cloud_scratch / "transport_stages.csv").write_bytes(
        runner._csv_bytes(runner.STAGE_FIELDS, stage_rows)
    )
    with pytest.raises(ValueError, match="reset cumulative position/sigma"):
        runner._validate_serialized_counts(
            cloud_scratch,
            json.loads((cloud_scratch / "summary.json").read_text()),
        )

    owner_scratch = tmp_path / "owner-self-digest"
    shutil.copytree(source_artifact, owner_scratch)
    owner_rows = runner._load_jsonl(owner_scratch / "owner_events.jsonl")
    transport = dict(owner_rows[0]["transport_event"])
    transport["transport_event_sha256"] = "f" * 64
    owner_rows[0]["transport_event"] = transport
    (owner_scratch / "owner_events.jsonl").write_bytes(
        b"".join(runner._json_bytes(row) + b"\n" for row in owner_rows)
    )
    with pytest.raises(ValueError, match="transport-event replay digest"):
        runner._validate_serialized_counts(
            owner_scratch,
            json.loads((owner_scratch / "summary.json").read_text()),
        )


def test_verify_recomputes_summary_semantic_input_map(
    pass_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    source, _ = pass_pair
    scratch = tmp_path / "stale-summary-map"
    shutil.copytree(source, scratch)
    summary = json.loads((scratch / "summary.json").read_text())
    summary["semantic_input_file_sha256"]["raw_steps.csv"] = "f" * 64
    summary_without_hash = dict(summary)
    summary_without_hash.pop("summary_payload_sha256")
    summary["summary_payload_sha256"] = runner._payload_sha256(
        "fluxv-v5h11-baik-w2-summary-v1", summary_without_hash
    )
    (scratch / "summary.json").write_bytes(runner._json_bytes(summary) + b"\n")
    _write_checksum_file(scratch)
    with pytest.raises(ValueError, match="semantic-input file map"):
        runner.verify_artifact(scratch)


def test_sink_freezes_nested_values_and_rejects_non_json_objects() -> None:
    sink = runner.ArtifactSink()
    row = _source_event(32, 4)
    expected_event_sha256 = row["event_sha256"]
    birth_modes = row["birth_modes"]
    assert isinstance(birth_modes, list)
    sink.add_source_event(row)
    birth_modes.append("MUTATED")
    row["event_sha256"] = "f" * 64
    frozen = sink.source_events[0]
    assert frozen["birth_modes"] == ["first"] * 8
    assert frozen["event_sha256"] == expected_event_sha256

    foreign = _source_event(32, 4)
    foreign["birth_modes"] = np.asarray([1.0])
    with pytest.raises(TypeError, match="non-JSON primitive"):
        sink.add_source_event(foreign)
    nonfinite = _source_event(32, 4)
    nonfinite["a0_pre"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        sink.add_source_event(nonfinite)
    wrong_key = _owner_event(32, 1)
    wrong_key["commit_event"] = {1: "integer", "1": "string"}
    with pytest.raises(TypeError, match="non-string mapping key"):
        sink.add_owner_event(wrong_key)


def test_trajectory_sink_freezes_exact_encoded_array_bytes() -> None:
    sink = runner.ArtifactSink()
    source64 = np.asarray([-0.0, 1.25, -2.5], dtype=np.float64)
    expected64 = source64.tobytes(order="C")
    record = {
        "transport_substeps": 32,
        "layer": 1,
        "status": "failed",
        "particle_ids": (),
        "material_tracer_ids": (),
        "frontier_node_ids": (),
        "arrays": {"end_positions": source64},
        "metadata": {"synthetic": True},
    }
    sink.add_trajectory_array_record(record)
    source64[:] = 99.0

    frozen = sink.trajectories[0]
    arrays = frozen["arrays"]
    assert isinstance(arrays, dict)
    decoded64 = runner.decode_array(arrays["end_positions"])
    assert decoded64.dtype == np.dtype("float64")
    assert decoded64.tobytes(order="C") == expected64
    assert np.signbit(decoded64[0])

    completed_float32 = _trajectory(32, 1)
    completed_float32["arrays"]["probe_velocity"] = np.asarray(
        completed_float32["arrays"]["probe_velocity"], dtype=np.float32
    )
    with pytest.raises(ValueError, match="exact float64 dtype"):
        sink.add_trajectory_array_record(completed_float32)

    malformed = runner.encode_array(np.asarray([1.0], dtype=np.float64))
    malformed["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        sink.add_trajectory(
            {
                "transport_substeps": 32,
                "layer": 1,
                "status": "failed",
                "particle_ids": (),
                "material_tracer_ids": (),
                "frontier_node_ids": (),
                "arrays": {"end_positions": malformed},
                "metadata": {},
            }
        )
