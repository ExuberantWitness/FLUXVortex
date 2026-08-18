"""Run the observation-free v5h2 dyadic cumulative-cloud numerical gate.

The gate changes only the newborn edge quadrature: each retained edge uses
``n(level) = n0 * 2**level`` with one fixed SI smoothing radius.  DVM remains
the source-event owner, the node ribbon remains the topology owner, and rVPM
remains the post-birth transport owner.  No surface solver, force, target
observation, or paper-score path is called here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import radians
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from fluxvortex.rvpm_dyadic_edge_bridge import (
    deposit_edge_graph_prescribed_sigma_dyadic_panels,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from forward_flight_benchmarks.run_v5h_cumulative_cloud_gate import (
    CASE_GEOMETRIES,
    FIXED_PROBES_GP1_M,
    FREESTREAM_GP1_M_PER_S,
    MAX_PARTICLE_INVARIANT_REL,
    MAX_RIBBON_RESIDUAL,
    PHYSICAL_RELEASE_DT_S,
    SIGMA_BIRTH_M,
    TRANSPORT_SUBSTEPS,
    _ConfigurationResult,
    _array_sha256,
    _bridge_metrics,
    _cell_strength_ownership_passed,
    _continuous_one_third_passed,
    _fact_physical_identity,
    _frontier_observable,
    _geometry_nodes,
    _h_refinement_gate,
    _map_node_layer,
    _placement_metrics,
    _report_snapshot,
    _ribbon_cells,
    _ribbon_half_step_matches_placement,
    _ribbon_handoff_integrity,
    _source_layer_passed,
    _source_sets,
    _step_sources,
    _time_refinement_gate,
    _transport_invariants,
)
from forward_flight_benchmarks.v5h_dvm_node_placement import (
    NodeLocalDVMPlacementAdapter,
)
from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
    NodeOwnedDVMRibbonShadow,
)
from forward_flight_benchmarks.v5h2_dyadic_cumulative_cloud_transport import (
    DyadicCumulativeCloudTransportReport,
    attest_dyadic_cumulative_ribbon_handoff,
    materialize_dyadic_cumulative_particle_state,
    transport_dyadic_accumulated_particle_cloud,
    validate_dyadic_cumulative_cloud_transport_report,
)


FloatArray = NDArray[np.float64]
GateGeometry = Literal["straight", "taper", "twist"]

RUN_SCHEMA_ID = "fluxv-v5h2-dyadic-cumulative-cloud-gate-v1"
BASE_TARGET_SPACING_M = 0.04
REFINEMENT_LEVELS = (0, 1, 2, 3)
NOMINAL_TARGET_SPACINGS_M = tuple(
    BASE_TARGET_SPACING_M / (2**level) for level in REFINEMENT_LEVELS
)
FIRST_ALPHA_RAD = radians(35.0)
CONTINUOUS_ALPHA_RAD = radians(55.0)
SPAN_CELLS = 4
RELEASE_COUNT = 3
CONFIGURATION_COUNT = 36
ARTIFACT_SCHEMA_ID = "fluxv-v5h2-dyadic-cumulative-artifact-v1"
ARTIFACT_FILENAMES = frozenset(
    {
        "README.md",
        "summary.json",
        "raw_refinement.json",
        "recomputed_gates.json",
        "source_manifest.json",
        "semantic_manifest.json",
        "run_manifest.json",
        "result_manifest.json",
        "SHA256SUMS",
    }
)


@dataclass(frozen=True, slots=True)
class DyadicGateConfig:
    geometries: tuple[GateGeometry, ...] = CASE_GEOMETRIES
    refinement_levels: tuple[int, ...] = REFINEMENT_LEVELS
    transport_substeps: tuple[int, ...] = TRANSPORT_SUBSTEPS
    span_cells: int = SPAN_CELLS
    release_count: int = RELEASE_COUNT
    base_target_spacing_m: float = BASE_TARGET_SPACING_M
    smoothing_radius_m: float = SIGMA_BIRTH_M


def _validate_config(config: DyadicGateConfig) -> None:
    if type(config) is not DyadicGateConfig or config != DyadicGateConfig():
        raise ValueError("the v5h2 dyadic gate grid is frozen")
    if (
        len(config.geometries)
        * len(config.refinement_levels)
        * len(config.transport_substeps)
        != CONFIGURATION_COUNT
    ):
        raise ValueError("the v5h2 grid must contain exactly 36 configurations")


def _run_release_sequence(
    geometry: GateGeometry,
    *,
    refinement_level: int,
    transport_substeps: int,
    wing_id: str,
) -> _ConfigurationResult:
    nodes = _geometry_nodes(geometry, SPAN_CELLS)
    node_sources, cell_sources = _source_sets(geometry, nodes)
    placement_adapter = NodeLocalDVMPlacementAdapter(wing_id=wing_id)
    ribbon_mapper = NodeOwnedDVMRibbonShadow(wing_id=wing_id, source_family="lev")
    previous: DyadicCumulativeCloudTransportReport | None = None
    release_summaries: list[dict[str, Any]] = []
    release_raw: list[dict[str, Any]] = []
    additive_count = 0
    final_frontier = np.empty((0, 3), dtype=np.float64)
    final_probe = np.empty((0, 3), dtype=np.float64)
    final_state: object | None = None
    final_report: DyadicCumulativeCloudTransportReport | None = None
    realized_spacing_max = 0.0
    expected_modes = ("first", "continuous", "continuous")
    alphas = (FIRST_ALPHA_RAD, CONTINUOUS_ALPHA_RAD, CONTINUOUS_ALPHA_RAD)

    for step, (alpha, expected_mode) in enumerate(
        zip(alphas, expected_modes, strict=True), start=1
    ):
        source_time = (step - 1) * PHYSICAL_RELEASE_DT_S
        node_events = _step_sources(node_sources, alpha)
        cell_events = _step_sources(cell_sources, alpha)
        placement = _map_node_layer(
            placement_adapter,
            wing_id,
            geometry,
            nodes,
            node_events,
            cell_events,
            source_time_s=source_time,
        )
        ribbon = ribbon_mapper.map_step(
            _ribbon_cells(geometry, nodes, cell_events),
            placement.kinematics,
            delta_time_s=PHYSICAL_RELEASE_DT_S,
            transport_enabled=True,
            source_time_s=source_time,
            frontier_transport_report=previous if step > 1 else None,
            node_placement_result=placement,
        )
        if ribbon.edge_graph is None:
            raise RuntimeError("active dyadic release produced no edge graph")
        previous_snapshot = None if previous is None else _report_snapshot(previous)
        previous_state = (
            None
            if previous is None
            else materialize_dyadic_cumulative_particle_state(previous)
        )
        previous_count = 0 if previous is None else previous.total_particle_count
        handoff = attest_dyadic_cumulative_ribbon_handoff(
            ribbon_mapper,
            ribbon,
            wing_id=wing_id,
            source_time_s=source_time,
            previous_report=previous,
        )
        report = transport_dyadic_accumulated_particle_cloud(
            handoff,
            smoothing_radius_m=SIGMA_BIRTH_M,
            base_target_spacing_m=BASE_TARGET_SPACING_M,
            refinement_level=refinement_level,
            transport_end_time_s=source_time + PHYSICAL_RELEASE_DT_S,
            transport_substeps=transport_substeps,
            freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
        )
        if validate_dyadic_cumulative_cloud_transport_report(report) is not report:
            raise RuntimeError("dyadic report attestation changed identity")
        state = materialize_dyadic_cumulative_particle_state(report)
        latest_sidecar = report.release_plan_sidecars[-1]
        diagnostic_bridge = deposit_edge_graph_prescribed_sigma_dyadic_panels(
            ribbon.edge_graph,
            smoothing_radius=SIGMA_BIRTH_M,
            base_target_spacing=BASE_TARGET_SPACING_M,
            refinement_level=refinement_level,
            step=step,
        ).bridge
        bridge_metrics = _bridge_metrics(
            diagnostic_bridge,
            requested_spacing_m=(BASE_TARGET_SPACING_M / (2**refinement_level)),
            smoothing_radius_m=SIGMA_BIRTH_M,
        )
        realized_spacing_max = max(
            realized_spacing_max,
            max(panel.realized_spacing_m for panel in latest_sidecar.plan.edge_panels),
        )
        additive_count += report.new_particle_count
        cloud = report.transported_particle_cloud
        prefix_passed = bool(
            previous is None
            or (
                cloud.particle_ids[:previous_count]
                == previous.transported_particle_cloud.particle_ids
                and cloud.lineage[:previous_count]
                == previous.transported_particle_cloud.lineage
            )
        )
        immutable_passed = bool(
            previous is None or _report_snapshot(previous) == previous_snapshot
        )
        latest_slice = cloud.release_slices[-1]
        sidecar_passed = bool(
            latest_sidecar.plan.refinement_level == refinement_level
            and latest_sidecar.plan.predicted_total_particle_count
            == report.new_particle_count
            and latest_slice.particle_count == report.new_particle_count
            and latest_slice.deposition_target_spacing_m
            == BASE_TARGET_SPACING_M / (2**refinement_level)
            and all(
                panel.panel_count == panel.base_panel_count * (2**refinement_level)
                for panel in latest_sidecar.plan.edge_panels
            )
        )
        placement_metrics = _placement_metrics(placement, expected_mode=expected_mode)
        handoff_metrics = _ribbon_handoff_integrity(ribbon, placement)
        source_passed = bool(
            _source_layer_passed(
                node_events, expected_step=step, expected_mode=expected_mode
            )
            and _source_layer_passed(
                cell_events, expected_step=step, expected_mode=expected_mode
            )
        )
        if step == 1:
            birth_passed = _ribbon_half_step_matches_placement(ribbon, placement)
            continuous_passed, continuous_residual = True, 0.0
        else:
            birth_passed = True
            continuous_passed, continuous_residual = _continuous_one_third_passed(
                previous, placement, ribbon
            )
        ribbon_passed = bool(
            ribbon.diagnostics.seam_count == 0
            and ribbon.diagnostics.incidence_residual <= MAX_RIBBON_RESIDUAL
            and ribbon.diagnostics.edge_reconstruction_residual <= MAX_RIBBON_RESIDUAL
            and ribbon.diagnostics.feedback_call_count == 0
            and handoff_metrics["passed"]
            and _cell_strength_ownership_passed(ribbon, cell_events)
        )
        invariants = _transport_invariants(previous_state, diagnostic_bridge, state)
        if invariants["particle_invariant_relative_drift_max"] > (
            MAX_PARTICLE_INVARIANT_REL
        ):
            raise RuntimeError("dyadic particle invariant gate failed")
        frontier_delta, frontier_positions, birth_positions = _frontier_observable(
            report, ribbon
        )
        probe = direct_gaussian_erf_velocity_jacobian(
            state.positions,
            state.gamma,
            state.sigma,
            target_positions=FIXED_PROBES_GP1_M,
        ).velocity
        release_passed = bool(
            source_passed
            and placement_metrics["passed"]
            and birth_passed
            and continuous_passed
            and ribbon_passed
            and bridge_metrics["passed"]
            and invariants["passed"]
            and prefix_passed
            and immutable_passed
            and sidecar_passed
            and report.total_particle_count == additive_count
            and report.exact_append_passed
            and report.one_combined_field_passed
            and report.stage_pre_replay_passed
            and report.dyadic_plan_passed
            and report.observation_access == "none"
            and report.target_case_branch == "none"
        )
        release_summaries.append(
            {
                "source_step_index": step,
                "source_time_s": source_time,
                "expected_mode": expected_mode,
                "new_particle_count": report.new_particle_count,
                "total_particle_count": report.total_particle_count,
                "plan_sha256": latest_sidecar.plan.plan_sha256,
                "sidecar_sha256": latest_sidecar.sidecar_sha256,
                "sidecar_passed": sidecar_passed,
                "source_passed": source_passed,
                "placement_passed": placement_metrics["passed"],
                "continuous_one_third_residual_max_m": continuous_residual,
                "ribbon_passed": ribbon_passed,
                "edge_passed": bridge_metrics["passed"],
                "transport_invariants_passed": invariants["passed"],
                "prefix_identity_passed": prefix_passed,
                "previous_report_immutable_passed": immutable_passed,
                "passed": release_passed,
            }
        )
        release_raw.append(
            {
                "source_step_index": step,
                "positions_gp1_m": state.positions.copy(),
                "gamma_vector_m3_per_s": state.gamma.copy(),
                "sigma_m": state.sigma.copy(),
                "frontier_positions_gp1_m": frontier_positions.copy(),
                "latest_birth_positions_gp1_m": birth_positions.copy(),
                "frontier_minus_latest_birth_gp1_m": frontier_delta.copy(),
                "fixed_probe_induced_velocity_gp1_m_per_s": probe.copy(),
                "frontier_fact_identity": tuple(
                    row[:-1] for row in _fact_physical_identity(report)
                ),
                "particle_ids": cloud.particle_ids,
                "lineage": cloud.lineage,
                "release_slices": cloud.release_slices,
                "release_plan_sidecars": report.release_plan_sidecars,
                "cloud_sha256": cloud.cloud_sha256,
                "report_sha256": report.report_sha256,
            }
        )
        previous = report
        final_report = report
        final_state = state
        final_frontier = frontier_delta
        final_probe = probe

    if final_report is None or final_state is None:
        raise RuntimeError("dyadic release sequence produced no state")
    nominal_spacing = BASE_TARGET_SPACING_M / (2**refinement_level)
    summary = {
        "geometry": geometry,
        "span_cells": SPAN_CELLS,
        "refinement_level": refinement_level,
        "target_spacing_m": nominal_spacing,
        "base_target_spacing_m": BASE_TARGET_SPACING_M,
        "transport_substeps": transport_substeps,
        "smoothing_radius_m": SIGMA_BIRTH_M,
        "release_count": RELEASE_COUNT,
        "final_particle_count": final_report.total_particle_count,
        "realized_spacing_max_m": realized_spacing_max,
        "final_cloud_sha256": final_report.transported_particle_cloud.cloud_sha256,
        "final_report_sha256": final_report.report_sha256,
        "final_state_sha256": _array_sha256(
            final_state.positions,
            final_state.gamma,
            final_state.sigma,
            final_frontier,
            final_probe,
        ),
        "releases": release_summaries,
        "passed": bool(
            release_summaries and all(row["passed"] for row in release_summaries)
        ),
    }
    raw = {
        "positions_gp1_m": final_state.positions.copy(),
        "gamma_vector_m3_per_s": final_state.gamma.copy(),
        "sigma_m": final_state.sigma.copy(),
        "frontier_minus_latest_birth_gp1_m": final_frontier.copy(),
        "fixed_probe_induced_velocity_gp1_m_per_s": final_probe.copy(),
        "particle_ids": final_report.transported_particle_cloud.particle_ids,
        "lineage": final_report.transported_particle_cloud.lineage,
        "release_slices": final_report.transported_particle_cloud.release_slices,
        "release_plan_sidecars": final_report.release_plan_sidecars,
        "cloud_sha256": final_report.transported_particle_cloud.cloud_sha256,
        "report_sha256": final_report.report_sha256,
        "release_raw": release_raw,
    }
    return _ConfigurationResult(
        summary=summary,
        raw=raw,
        frontier_minus_latest_birth=final_frontier,
        fixed_probe_induced_velocity=final_probe,
        particle_count=final_report.total_particle_count,
        realized_spacing_max_m=realized_spacing_max,
    )


def run_minimal_smoke() -> dict[str, Any]:
    row = _run_release_sequence(
        "straight",
        refinement_level=0,
        transport_substeps=1,
        wing_id="v5h2-smoke:straight:l0:s1:wing",
    )
    return {
        "schema_id": RUN_SCHEMA_ID,
        "scope": "minimal_smoke_only",
        "configuration": row.summary,
        "passed": bool(row.summary["passed"]),
    }


def _sidecar_record(sidecar: object) -> dict[str, Any]:
    plan = sidecar.plan
    return {
        "release_index": sidecar.release_index,
        "source_step_index": sidecar.source_step_index,
        "source_time_s": sidecar.source_time_s,
        "sidecar_sha256": sidecar.sidecar_sha256,
        "bridge_manifest_sha256": sidecar.bridge_manifest_sha256,
        "plan_sha256": plan.plan_sha256,
        "parent_edge_graph_sha256": plan.parent_edge_graph_sha256,
        "smoothing_radius_m": plan.smoothing_radius_m,
        "base_target_spacing_m": plan.base_target_spacing_m,
        "refinement_level": plan.refinement_level,
        "refinement_multiplier": plan.refinement_multiplier,
        "predicted_total_particle_count": plan.predicted_total_particle_count,
        "edge_panels": [
            {
                "edge_key": repr(panel.edge_key),
                "edge_length_m": panel.edge_length_m,
                "base_panel_count": panel.base_panel_count,
                "refinement_level": panel.refinement_level,
                "panel_count": panel.panel_count,
                "realized_spacing_m": panel.realized_spacing_m,
                "smoothing_radius_m": panel.smoothing_radius_m,
                "parent_edge_graph_sha256": panel.parent_edge_graph_sha256,
            }
            for panel in plan.edge_panels
        ],
    }


def _raw_configuration_record(row: _ConfigurationResult) -> dict[str, Any]:
    frontier = np.ascontiguousarray(row.frontier_minus_latest_birth, dtype=np.float64)
    probes = np.ascontiguousarray(row.fixed_probe_induced_velocity, dtype=np.float64)
    positions = np.ascontiguousarray(row.raw["positions_gp1_m"], dtype=np.float64)
    gamma = np.ascontiguousarray(row.raw["gamma_vector_m3_per_s"], dtype=np.float64)
    sigma = np.ascontiguousarray(row.raw["sigma_m"], dtype=np.float64)
    releases = []
    for summary, raw in zip(
        row.summary["releases"], row.raw["release_raw"], strict=True
    ):
        latest_sidecar = raw["release_plan_sidecars"][-1]
        releases.append(
            {
                **summary,
                "cloud_sha256": raw["cloud_sha256"],
                "report_sha256": raw["report_sha256"],
                "sidecar": _sidecar_record(latest_sidecar),
            }
        )
    return {
        "geometry": row.summary["geometry"],
        "span_cells": row.summary["span_cells"],
        "refinement_level": row.summary["refinement_level"],
        "base_target_spacing_m": row.summary["base_target_spacing_m"],
        "nominal_target_spacing_m": row.summary["target_spacing_m"],
        "transport_substeps": row.summary["transport_substeps"],
        "smoothing_radius_m": row.summary["smoothing_radius_m"],
        "release_count": row.summary["release_count"],
        "particle_count": row.particle_count,
        "realized_spacing_max_m": row.realized_spacing_max_m,
        "frontier_minus_latest_birth_gp1_m": frontier.tolist(),
        "fixed_probe_induced_velocity_gp1_m_per_s": probes.tolist(),
        "positions_gp1_m": positions.tolist(),
        "gamma_vector_m3_per_s": gamma.tolist(),
        "sigma_m": sigma.tolist(),
        "state_and_observable_sha256": _array_sha256(
            positions, gamma, sigma, frontier, probes
        ),
        "final_cloud_sha256": row.summary["final_cloud_sha256"],
        "final_report_sha256": row.summary["final_report_sha256"],
        "releases": releases,
        "mechanics_passed": row.summary["passed"],
    }


def _strict_array(name: str, value: object, shape_tail: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} has a foreign shape")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def recompute_dyadic_gate_from_raw(
    raw_payload: object,
    reported_summary: object | None = None,
) -> dict[str, Any]:
    """Rebuild all v5h2 numerical gates from strict JSON-safe raw records."""

    if not isinstance(raw_payload, dict) or raw_payload.get("schema_id") != (
        "fluxv-v5h2-dyadic-raw-refinement-v1"
    ):
        raise ValueError("raw v5h2 refinement payload has a foreign schema")
    records = raw_payload.get("configurations")
    if not isinstance(records, list) or len(records) != CONFIGURATION_COUNT:
        raise ValueError("raw v5h2 payload must contain exactly 36 configurations")
    rows: dict[tuple[str, int, int], _ConfigurationResult] = {}
    mechanics_by_key: dict[tuple[str, int, int], bool] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("raw v5h2 configuration has a foreign schema")
        geometry = record.get("geometry")
        level = record.get("refinement_level")
        substeps = record.get("transport_substeps")
        if (
            geometry not in CASE_GEOMETRIES
            or type(level) is not int
            or level not in REFINEMENT_LEVELS
            or type(substeps) is not int
            or substeps not in TRANSPORT_SUBSTEPS
        ):
            raise ValueError("raw v5h2 configuration key is outside the frozen grid")
        key = geometry, level, substeps
        if key in rows:
            raise ValueError("raw v5h2 configuration grid contains a duplicate")
        nominal = float(record["nominal_target_spacing_m"])
        if nominal != BASE_TARGET_SPACING_M / (2**level):
            raise ValueError("raw nominal spacing disagrees with dyadic level")
        frontier = _strict_array(
            "frontier observable",
            record["frontier_minus_latest_birth_gp1_m"],
            (3,),
        )
        probes = _strict_array(
            "probe observable",
            record["fixed_probe_induced_velocity_gp1_m_per_s"],
            (3,),
        )
        positions = _strict_array("positions", record["positions_gp1_m"], (3,))
        gamma = _strict_array("gamma vectors", record["gamma_vector_m3_per_s"], (3,))
        sigma = np.asarray(record["sigma_m"], dtype=np.float64)
        if (
            frontier.shape != (SPAN_CELLS + 1, 3)
            or probes.shape != FIXED_PROBES_GP1_M.shape
            or positions.shape != gamma.shape
            or sigma.shape != (positions.shape[0],)
            or not np.all(np.isfinite(sigma))
            or np.any(sigma <= 0.0)
            or positions.shape[0] != record["particle_count"]
        ):
            raise ValueError("raw v5h2 state/observable shape or sigma is invalid")
        expected_state_hash = _array_sha256(positions, gamma, sigma, frontier, probes)
        if expected_state_hash != record["state_and_observable_sha256"]:
            raise ValueError("raw v5h2 state/observable digest is inconsistent")
        releases = record.get("releases")
        mechanics = bool(
            isinstance(releases, list)
            and len(releases) == RELEASE_COUNT
            and record.get("release_count") == RELEASE_COUNT
            and record.get("span_cells") == SPAN_CELLS
            and record.get("base_target_spacing_m") == BASE_TARGET_SPACING_M
            and record.get("smoothing_radius_m") == SIGMA_BIRTH_M
        )
        additive = 0
        if isinstance(releases, list):
            for step, release in enumerate(releases, start=1):
                if not isinstance(release, dict):
                    mechanics = False
                    continue
                sidecar = release.get("sidecar")
                panels = (
                    None
                    if not isinstance(sidecar, dict)
                    else sidecar.get("edge_panels")
                )
                new_count = release.get("new_particle_count")
                if type(new_count) is int:
                    additive += new_count
                panel_counts_passed = bool(
                    isinstance(panels, list)
                    and panels
                    and all(
                        type(panel.get("base_panel_count")) is int
                        and type(panel.get("panel_count")) is int
                        and panel.get("refinement_level") == level
                        and panel["panel_count"]
                        == panel["base_panel_count"] * (2**level)
                        and panel.get("smoothing_radius_m") == SIGMA_BIRTH_M
                        and 0.0 < panel.get("realized_spacing_m", 0.0) <= nominal
                        for panel in panels
                        if isinstance(panel, dict)
                    )
                    and len(panels) == sum(isinstance(panel, dict) for panel in panels)
                )
                mechanics = bool(
                    mechanics
                    and release.get("source_step_index") == step
                    and release.get("passed") is True
                    and release.get("sidecar_passed") is True
                    and isinstance(sidecar, dict)
                    and sidecar.get("release_index") == step
                    and sidecar.get("source_step_index") == step
                    and sidecar.get("refinement_level") == level
                    and sidecar.get("refinement_multiplier") == 2**level
                    and sidecar.get("predicted_total_particle_count") == new_count
                    and panel_counts_passed
                )
        mechanics = bool(
            mechanics
            and additive == record["particle_count"]
            and record.get("mechanics_passed") is True
        )
        proxy_summary = {
            "geometry": geometry,
            "target_spacing_m": nominal,
            "transport_substeps": substeps,
            "passed": mechanics,
        }
        row = _ConfigurationResult(
            summary=proxy_summary,
            raw={},
            frontier_minus_latest_birth=frontier,
            fixed_probe_induced_velocity=probes,
            particle_count=int(record["particle_count"]),
            realized_spacing_max_m=float(record["realized_spacing_max_m"]),
        )
        rows[key] = row
        mechanics_by_key[key] = mechanics
    expected_keys = {
        (geometry, level, substeps)
        for geometry in CASE_GEOMETRIES
        for level in REFINEMENT_LEVELS
        for substeps in TRANSPORT_SUBSTEPS
    }
    if set(rows) != expected_keys:
        raise ValueError("raw v5h2 configuration grid is incomplete")

    geometries: list[dict[str, Any]] = []
    for geometry in CASE_GEOMETRIES:
        by_level = [
            [rows[(geometry, level, substeps)] for substeps in TRANSPORT_SUBSTEPS]
            for level in REFINEMENT_LEVELS
        ]
        time_gates = [
            {
                "refinement_level": level,
                "nominal_target_spacing_m": BASE_TARGET_SPACING_M / (2**level),
                **_time_refinement_gate(level_rows),
            }
            for level, level_rows in zip(REFINEMENT_LEVELS, by_level, strict=True)
        ]
        h_gate = _h_refinement_gate([level_rows[-1] for level_rows in by_level])
        mechanics = all(
            mechanics_by_key[(geometry, level, substeps)]
            for level in REFINEMENT_LEVELS
            for substeps in TRANSPORT_SUBSTEPS
        )
        passed = bool(
            mechanics
            and all(gate["passed"] for gate in time_gates)
            and h_gate["passed"]
        )
        geometries.append(
            {
                "geometry": geometry,
                "configuration_mechanics_passed": mechanics,
                "time_refinement_by_level": time_gates,
                "h_refinement_at_finest_time": h_gate,
                "passed": passed,
            }
        )
    passed = bool(all(row["passed"] for row in geometries))
    recomputed = {
        "schema_id": "fluxv-v5h2-dyadic-recomputed-gates-v1",
        "configuration_count": len(rows),
        "release_count": len(rows) * RELEASE_COUNT,
        "geometries": geometries,
        "passed": passed,
        "status": "go_v5h2_mechanics_only" if passed else "stop_v5h2_spatial_gate",
    }
    if reported_summary is not None:
        if not isinstance(reported_summary, dict):
            raise ValueError("reported v5h2 summary has a foreign schema")
        reported_by_geometry = {
            item["geometry"]: item for item in reported_summary.get("geometries", [])
        }
        reported_match = bool(
            reported_summary.get("configuration_count") == len(rows)
            and reported_summary.get("release_count") == len(rows) * RELEASE_COUNT
            and reported_summary.get("passed") is passed
            and reported_summary.get("status") == recomputed["status"]
            and set(reported_by_geometry) == set(CASE_GEOMETRIES)
            and all(
                reported_by_geometry[item["geometry"]].get("passed") is item["passed"]
                and reported_by_geometry[item["geometry"]].get(
                    "configuration_mechanics_passed"
                )
                is item["configuration_mechanics_passed"]
                and reported_by_geometry[item["geometry"]].get(
                    "time_refinement_by_level"
                )
                == item["time_refinement_by_level"]
                and reported_by_geometry[item["geometry"]].get(
                    "h_refinement_at_finest_time"
                )
                == item["h_refinement_at_finest_time"]
                for item in geometries
            )
        )
        recomputed["reported_values_match"] = reported_match
        recomputed["passed"] = bool(passed and reported_match)
        if not recomputed["passed"]:
            recomputed["status"] = "stop_v5h2_artifact_recomputation"
    return recomputed


def run_full_dyadic_gate(
    config: DyadicGateConfig = DyadicGateConfig(),
) -> dict[str, Any]:
    """Run 36 non-target configurations and the frozen convergence gates."""

    _validate_config(config)
    configurations: list[_ConfigurationResult] = []
    geometries: list[dict[str, Any]] = []
    for geometry in config.geometries:
        by_level: list[list[_ConfigurationResult]] = []
        for level in config.refinement_levels:
            rows = [
                _run_release_sequence(
                    geometry,
                    refinement_level=level,
                    transport_substeps=substeps,
                    wing_id=(
                        f"v5h2:{geometry}:level={level}:" f"substeps={substeps}:wing"
                    ),
                )
                for substeps in config.transport_substeps
            ]
            configurations.extend(rows)
            by_level.append(rows)
        time_gates = [
            {
                "refinement_level": level,
                "nominal_target_spacing_m": (
                    config.base_target_spacing_m / (2**level)
                ),
                **_time_refinement_gate(rows),
            }
            for level, rows in zip(config.refinement_levels, by_level, strict=True)
        ]
        h_gate = _h_refinement_gate([rows[-1] for rows in by_level])
        mechanics = all(row.summary["passed"] for rows in by_level for row in rows)
        exact_doubling = all(
            all(
                panel.panel_count == panel.base_panel_count * (2**level)
                for sidecar in row.raw["release_plan_sidecars"]
                for panel in sidecar.plan.edge_panels
            )
            for level, rows in zip(config.refinement_levels, by_level, strict=True)
            for row in rows
        )
        geometry_passed = bool(
            mechanics
            and exact_doubling
            and all(gate["passed"] for gate in time_gates)
            and h_gate["passed"]
        )
        geometries.append(
            {
                "geometry": geometry,
                "configuration_count": sum(len(rows) for rows in by_level),
                "configurations": [row.summary for rows in by_level for row in rows],
                "time_refinement_by_level": time_gates,
                "h_refinement_at_finest_time": h_gate,
                "configuration_mechanics_passed": mechanics,
                "exact_per_edge_doubling_passed": exact_doubling,
                "passed": geometry_passed,
            }
        )
    if len(configurations) != CONFIGURATION_COUNT:
        raise RuntimeError("v5h2 did not execute exactly 36 configurations")
    passed = bool(geometries and all(row["passed"] for row in geometries))
    summary = {
        "schema_id": RUN_SCHEMA_ID,
        "evaluation_mode": "simulation_only",
        "target_observation_access": "none",
        "paper_scoring_performed": False,
        "configuration_count": len(configurations),
        "release_count": len(configurations) * RELEASE_COUNT,
        "base_target_spacing_m": BASE_TARGET_SPACING_M,
        "refinement_levels": list(REFINEMENT_LEVELS),
        "nominal_target_spacings_m": list(NOMINAL_TARGET_SPACINGS_M),
        "transport_substeps": list(TRANSPORT_SUBSTEPS),
        "geometries": geometries,
        "passed": passed,
        "status": "go_v5h2_mechanics_only" if passed else "stop_v5h2_spatial_gate",
    }
    raw_payload = {
        "schema_id": "fluxv-v5h2-dyadic-raw-refinement-v1",
        "configuration_count": len(configurations),
        "release_count": len(configurations) * RELEASE_COUNT,
        "configurations": [_raw_configuration_record(row) for row in configurations],
    }
    recomputed = recompute_dyadic_gate_from_raw(raw_payload, summary)
    if not recomputed["passed"]:
        summary["passed"] = False
        summary["status"] = "stop_v5h2_artifact_recomputation"
    return {
        "summary": summary,
        "raw_refinement": raw_payload,
        "recomputed_gates": recomputed,
        # Convenience mirrors retain the original public result surface.
        **summary,
    }


def _json_text(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _declared_source_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "platform/forward_flight_benchmarks/run_v5h2_dyadic_cumulative_cloud_gate.py",
        "platform/forward_flight_benchmarks/v5h2_dyadic_cumulative_cloud_transport.py",
        "platform/forward_flight_benchmarks/v5h_cumulative_cloud_transport.py",
        "platform/forward_flight_benchmarks/v5h_dvm_node_ribbon.py",
        "platform/forward_flight_benchmarks/v5h_dvm_node_placement.py",
        "platform/forward_flight_benchmarks/v5h_dvm_source.py",
        "platform/ldvm_fourier.py",
        "platform/forward_flight_benchmarks/ldvm_uvlm_correction.py",
        "src/fluxvortex/rvpm_dyadic_edge_bridge.py",
        "src/fluxvortex/rvpm_edge_bridge.py",
        "src/fluxvortex/rvpm_reference.py",
        "src/fluxvortex/rvpm_transport.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"declared v5h2 source is absent: {relative}")
        files.append({"path": relative, "sha256": _file_sha256(path)})
    return {
        "schema_id": "fluxv-v5h2-dyadic-source-manifest-v1",
        "files": files,
    }


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> object:
        raise ValueError(f"non-finite JSON token in {path.name}: {token}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def verify_dyadic_gate_artifacts(output_directory: Path | str) -> dict[str, Any]:
    """Verify strict JSON, source/result hashes, and disk recomputation."""

    output = Path(output_directory).resolve()
    if {path.name for path in output.iterdir()} != set(ARTIFACT_FILENAMES):
        raise ValueError("v5h2 artifact file set is incomplete or contains extras")
    sha_rows = output.joinpath("SHA256SUMS").read_text(encoding="utf-8").splitlines()
    expected_names = sorted(ARTIFACT_FILENAMES - {"SHA256SUMS"})
    parsed: dict[str, str] = {}
    for row in sha_rows:
        digest, separator, name = row.partition("  ")
        if separator != "  " or name in parsed:
            raise ValueError("SHA256SUMS has a malformed or duplicate row")
        parsed[name] = digest
    if sorted(parsed) != expected_names:
        raise ValueError("SHA256SUMS does not cover the exact artifact file set")
    if any(_file_sha256(output / name) != parsed[name] for name in expected_names):
        raise ValueError("v5h2 artifact SHA256SUMS verification failed")
    summary = _strict_json(output / "summary.json")
    raw = _strict_json(output / "raw_refinement.json")
    recomputed = _strict_json(output / "recomputed_gates.json")
    source_manifest = _strict_json(output / "source_manifest.json")
    result_manifest = _strict_json(output / "result_manifest.json")
    semantic_manifest = _strict_json(output / "semantic_manifest.json")
    _strict_json(output / "run_manifest.json")
    source_now = _declared_source_manifest()
    if source_manifest != source_now:
        raise ValueError("declared v5h2 source closure is stale")
    disk_recomputed = recompute_dyadic_gate_from_raw(raw, summary)
    if disk_recomputed != recomputed or not disk_recomputed["passed"]:
        raise ValueError("v5h2 disk gate recomputation disagrees or stops")
    result_files = result_manifest.get("files")
    if not isinstance(result_files, list):
        raise ValueError("v5h2 result manifest has a foreign schema")
    for item in result_files:
        if (
            not isinstance(item, dict)
            or item.get("path") not in expected_names
            or item.get("sha256") != _file_sha256(output / item["path"])
        ):
            raise ValueError("v5h2 result manifest contains a stale entry")
    semantic_files = semantic_manifest.get("files")
    if not isinstance(semantic_files, list):
        raise ValueError("v5h2 semantic manifest has a foreign schema")
    semantic_digest = sha256(b"fluxv-v5h2-semantic-artifact-v1\0")
    for item in semantic_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("v5h2 semantic manifest entry is malformed")
        path = output / item["path"]
        digest = _file_sha256(path)
        if item.get("sha256") != digest:
            raise ValueError("v5h2 semantic manifest contains a stale entry")
        semantic_digest.update(item["path"].encode("utf-8"))
        semantic_digest.update(b"\0")
        semantic_digest.update(path.read_bytes())
        semantic_digest.update(b"\0")
    if semantic_manifest.get("semantic_digest_sha256") != semantic_digest.hexdigest():
        raise ValueError("v5h2 semantic artifact digest is inconsistent")
    return {
        "passed": True,
        "semantic_digest_sha256": semantic_digest.hexdigest(),
        "configuration_count": disk_recomputed["configuration_count"],
        "release_count": disk_recomputed["release_count"],
    }


def write_dyadic_gate_artifacts(
    result: dict[str, Any], output_directory: Path | str
) -> dict[str, Any]:
    """Write one complete, strict, content-addressed v5h2 artifact bundle."""

    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty artifact directory")
    output.mkdir(parents=True, exist_ok=True)
    summary = result.get("summary")
    raw = result.get("raw_refinement")
    recomputed = result.get("recomputed_gates")
    if (
        not isinstance(summary, dict)
        or not isinstance(raw, dict)
        or not isinstance(recomputed, dict)
    ):
        raise ValueError("v5h2 result does not contain the three artifact payloads")
    if not summary.get("passed") or not recomputed.get("passed"):
        raise ValueError("a stopped v5h2 result cannot be published as GO")
    source_manifest = _declared_source_manifest()
    readme = (
        "# FluxV v5h2 dyadic cumulative-cloud gate\n\n"
        "This non-target mechanical artifact records 36 configurations and 108 "
        "releases. It uses fixed birth sigma, exact per-edge dyadic panel doubling, "
        "and one combined rVPM transport field. It contains no target observations, "
        "Ptera surface loads, or paper scoring. A GO here is mechanics-only.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "summary.json").write_text(_json_text(summary), encoding="utf-8")
    (output / "raw_refinement.json").write_text(_json_text(raw), encoding="utf-8")
    (output / "recomputed_gates.json").write_text(
        _json_text(recomputed), encoding="utf-8"
    )
    (output / "source_manifest.json").write_text(
        _json_text(source_manifest), encoding="utf-8"
    )
    semantic_names = (
        "README.md",
        "raw_refinement.json",
        "recomputed_gates.json",
        "source_manifest.json",
        "summary.json",
    )
    semantic_digest = sha256(b"fluxv-v5h2-semantic-artifact-v1\0")
    semantic_files = []
    for name in semantic_names:
        path = output / name
        semantic_files.append({"path": name, "sha256": _file_sha256(path)})
        semantic_digest.update(name.encode("utf-8"))
        semantic_digest.update(b"\0")
        semantic_digest.update(path.read_bytes())
        semantic_digest.update(b"\0")
    semantic_manifest = {
        "schema_id": "fluxv-v5h2-dyadic-semantic-manifest-v1",
        "files": semantic_files,
        "semantic_digest_sha256": semantic_digest.hexdigest(),
    }
    (output / "semantic_manifest.json").write_text(
        _json_text(semantic_manifest), encoding="utf-8"
    )
    run_manifest = {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "run_uuid": str(uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output),
        "evaluation_mode": "simulation_only",
        "target_observation_access": "none",
        "paper_scoring_performed": False,
    }
    (output / "run_manifest.json").write_text(
        _json_text(run_manifest), encoding="utf-8"
    )
    pre_result_names = sorted(
        ARTIFACT_FILENAMES - {"result_manifest.json", "SHA256SUMS"}
    )
    result_manifest = {
        "schema_id": "fluxv-v5h2-dyadic-result-manifest-v1",
        "files": [
            {"path": name, "sha256": _file_sha256(output / name)}
            for name in pre_result_names
        ],
    }
    (output / "result_manifest.json").write_text(
        _json_text(result_manifest), encoding="utf-8"
    )
    checksum_names = sorted(ARTIFACT_FILENAMES - {"SHA256SUMS"})
    checksum_text = "".join(
        f"{_file_sha256(output / name)}  {name}\n" for name in checksum_names
    )
    (output / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")
    return verify_dyadic_gate_artifacts(output)


def run_and_write_dyadic_gate(output_directory: Path | str) -> dict[str, Any]:
    result = run_full_dyadic_gate()
    verification = write_dyadic_gate_artifacts(result, output_directory)
    return {"result": result["summary"], "verification": verification}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    outcome = run_and_write_dyadic_gate(args.output)
    print(_json_text(outcome))


if __name__ == "__main__":
    main()


__all__ = [
    "BASE_TARGET_SPACING_M",
    "CONFIGURATION_COUNT",
    "DyadicGateConfig",
    "NOMINAL_TARGET_SPACINGS_M",
    "REFINEMENT_LEVELS",
    "RELEASE_COUNT",
    "RUN_SCHEMA_ID",
    "recompute_dyadic_gate_from_raw",
    "run_and_write_dyadic_gate",
    "run_full_dyadic_gate",
    "run_minimal_smoke",
    "verify_dyadic_gate_artifacts",
    "write_dyadic_gate_artifacts",
]
