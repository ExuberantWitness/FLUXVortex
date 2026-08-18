"""Run the observation-free v5h two-release frontier vertical-slice gate.

This runner proves only the bounded step-1 to step-2 chain

``DVM source -> node-owned ribbon -> fixed-sigma rVPM transport
             -> attested NodeFrontierFact -> continuous second ribbon``.

It never merges release layers, writes to a Ptera parent, evaluates a load, or
reads a target-paper observation.  A third release is deliberately blocked by
the passive-frontier v1 contract until an exact-once cumulative-cloud API is
implemented and audited.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from fluxvortex.rvpm_edge_bridge import (
    DIAGNOSTIC_SHADOW_OWNER,
    FROZEN_OVERLAP_LAMBDA,
    RING_PHYSICAL_OWNER,
    ShadowBridgeResult,
    deposit_edge_graph_fixed_sigma,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.rvpm_transport import lsrk3_step_direct, make_particle_state
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
from forward_flight_benchmarks.run_v5h_manufactured_shadow_gate import (
    ALPHA_RAD,
    CASE_GEOMETRIES,
    FIXED_PROBES_GP1_M,
    LESP_CRITICAL,
    PHYSICAL_RELEASE_DT_S,
    PIVOT_FRACTION_CHORD,
    REFERENCE_SPEED_M_PER_S,
    REYNOLDS,
    SOURCE_SETTINGS,
    SPAN_M,
    GateGeometry,
    _cell_plane_map,
    _node_geometry,
)
from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
    DVMPlaneToGP1Map,
    DVMSpanCellSource,
    NodeOwnedDVMRibbonShadow,
)
from forward_flight_benchmarks.v5h_dvm_source import (
    DVMSourceEvent,
    V5hDVMSource,
)
from forward_flight_benchmarks.v5h_passive_frontier_transport import (
    PASSIVE_FRONTIER_CONTINUATION_SCOPE,
    materialize_transported_particle_state,
    transport_passive_node_frontiers,
    validate_passive_frontier_transport_report,
)


RUN_SCHEMA_ID = "fluxv-v5h-two-release-frontier-vertical-gate-v1"
DEFAULT_SPAN_CELLS = 4
DEFAULT_SPAN_SUBDIVISIONS = (2, 4, 8, 16)
DEFAULT_TRANSPORT_SUBSTEPS = (1, 2, 4)
MAX_SOURCE_KELVIN_M2_PER_S = 1.0e-10
MAX_RIBBON_RESIDUAL = 1.0e-12
MAX_VECTOR_ABS = 1.0e-14
MAX_VECTOR_REL = 1.0e-6
MAX_PARTICLE_INVARIANT_REL = 1.0e-6
MAX_STAGE_DT_JACOBIAN = 1.5
MAX_STAGE_DT_SELF_SPEED_OVER_SIGMA = 0.5
MAX_FINE_TIME_RELATIVE_DIFFERENCE = 1.0e-6
MIN_TIME_REFINEMENT_RATIO = 1.5
MAX_FINE_SPATIAL_RELATIVE_DIFFERENCE = 0.05
MAX_SPATIAL_DIFFERENCE_GROWTH = 1.25

LIMITATIONS = (
    "non-target two-release mechanical shadow only; all span cells remain "
    "active; step1-to-step2 passive-frontier v1; no cumulative-cloud merge, "
    "third release, Ptera parent write, feedback, load, force, target "
    "observation, or performance claim; sigma=2.125*h is a fixed "
    "FLOWUnsteady-static-sheet-inspired development transfer, not a "
    "validated dynamic LEV core law"
)


@dataclass(frozen=True)
class GateConfig:
    """Frozen mesh and time-refinement family for the bounded gate."""

    span_cells: int = DEFAULT_SPAN_CELLS
    span_subdivisions: tuple[int, ...] = DEFAULT_SPAN_SUBDIVISIONS
    transport_substeps: tuple[int, ...] = DEFAULT_TRANSPORT_SUBSTEPS


@dataclass(frozen=True)
class _ConfigurationResult:
    summary: dict[str, Any]
    state_vector: np.ndarray
    frontier_positions: np.ndarray
    transported_probe_velocity: np.ndarray


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_config(config: GateConfig) -> None:
    if isinstance(config.span_cells, bool) or config.span_cells < 2:
        raise ValueError("span_cells must be an integer >= 2")
    for name, values in (
        ("span_subdivisions", config.span_subdivisions),
        ("transport_substeps", config.transport_substeps),
    ):
        if not values:
            raise ValueError(f"{name} must not be empty")
        if any(isinstance(value, bool) or value < 1 for value in values):
            raise ValueError(f"{name} must contain positive integers")
        if tuple(sorted(values)) != tuple(values) or len(set(values)) != len(values):
            raise ValueError(f"{name} must be strictly increasing")
    if len(config.span_subdivisions) < 3:
        raise ValueError("spatial convergence requires at least three levels")
    if len(config.transport_substeps) < 3:
        raise ValueError("time convergence requires at least three levels")


def _source_for_cell(
    geometry: GateGeometry,
    cell_index: int,
    chord_m: float,
) -> V5hDVMSource:
    convective_dt = PHYSICAL_RELEASE_DT_S * REFERENCE_SPEED_M_PER_S / chord_m
    return V5hDVMSource(
        physical_section_id=f"frontier:{geometry}:section:{cell_index}",
        physical_strip_id=f"frontier:{geometry}:strip:{cell_index}",
        geometry_identity="generic explicit zero camber flat plate",
        reference_speed_m_per_s=REFERENCE_SPEED_M_PER_S,
        reference_chord_m=chord_m,
        zero_camber_surrogate=True,
        delta_time_convective=convective_dt,
        pivot_fraction_chord=PIVOT_FRACTION_CHORD,
        threshold=LESPThreshold(
            value=LESP_CRITICAL,
            section_family="generic thin flat plate",
            reynolds=REYNOLDS,
            source="Ramesh LDVM v2.5 published source input Lcrit=0.18",
            source_role="published_source_input",
        ),
        settings=LDVMSectionSettings(
            ndiv=SOURCE_SETTINGS.ndiv,
            naterm=SOURCE_SETTINGS.naterm,
            max_wake_steps=SOURCE_SETTINGS.max_wake_steps,
            core_radius_chord=SOURCE_SETTINGS.core_radius_chord,
        ),
    )


def _events_for_step(
    sources: tuple[V5hDVMSource, ...],
) -> tuple[DVMSourceEvent, ...]:
    return tuple(source.step(ALPHA_RAD, 0.0, 0.0) for source in sources)


def _cells_for_events(
    geometry: GateGeometry,
    events: tuple[DVMSourceEvent, ...],
    anchors: tuple[np.ndarray, ...],
    twists: tuple[float, ...],
) -> tuple[DVMSpanCellSource, ...]:
    cells: list[DVMSpanCellSource] = []
    for cell_index, event in enumerate(events):
        plane_map: DVMPlaneToGP1Map = _cell_plane_map(
            geometry,
            cell_index,
            anchors[cell_index],
            anchors[cell_index + 1],
            0.5 * (twists[cell_index] + twists[cell_index + 1]),
        )
        cells.append(
            DVMSpanCellSource(
                cell_id=f"{geometry}:cell:{cell_index}",
                left_node_id=f"{geometry}:node:{cell_index}",
                right_node_id=f"{geometry}:node:{cell_index + 1}",
                event=event,
                plane_to_gp1=plane_map,
            )
        )
    return tuple(cells)


def _scaled_kelvin(event: DVMSourceEvent) -> float:
    return abs(float(event.kelvin_residual_over_u_c)) * float(
        event.provenance.circulation_scale_u_times_c_m2_per_s
    )


def _source_events_pass(
    first_events: tuple[DVMSourceEvent, ...],
    second_events: tuple[DVMSourceEvent, ...],
) -> bool:
    if len(first_events) != len(second_events):
        return False
    for first, second in zip(first_events, second_events, strict=True):
        if not (
            first.lineage.source_step_index == 1
            and second.lineage.source_step_index == 2
            and first.lesp_active
            and second.lesp_active
            and first.lev_birth_mode == "first"
            and second.lev_birth_mode == "continuous"
            and second.parent_event_manifest_sha256 == first.producer_manifest_sha256
            and _scaled_kelvin(first) <= MAX_SOURCE_KELVIN_M2_PER_S
            and _scaled_kelvin(second) <= MAX_SOURCE_KELVIN_M2_PER_S
        ):
            return False
    return True


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(1.0e-15, float(np.linalg.norm(right)))
    return float(np.linalg.norm(left - right) / scale)


def _lineage_cell_keys(lineage: Any) -> frozenset[str]:
    keys: set[str] = set()
    for incidence in lineage.ring_incidences:
        ring_id = str(incidence.ring_id)
        keys.add(re.sub(r":step:[0-9]+$", "", ring_id))
    if not keys:
        raise RuntimeError("a retained particle has no source-cell incidence")
    return frozenset(keys)


def _temporal_overlap_metrics(
    transported_positions: np.ndarray,
    transported_sigma: np.ndarray,
    transported_lineage: tuple[Any, ...],
    second_bridge: ShadowBridgeResult,
    *,
    target_spacing_m: float,
    birth_sigma_m: float,
) -> dict[str, Any]:
    old_keys = tuple(_lineage_cell_keys(item) for item in transported_lineage)
    new_keys = tuple(_lineage_cell_keys(item) for item in second_bridge.lineage)
    max_distance = 0.0
    min_birth_overlap = float("inf")
    min_two_sided_overlap = float("inf")
    missing_group_count = 0
    for index, keys in enumerate(new_keys):
        candidates = [
            old_index
            for old_index, old_group in enumerate(old_keys)
            if keys.intersection(old_group)
        ]
        if not candidates:
            missing_group_count += 1
            continue
        distances = np.linalg.norm(
            transported_positions[candidates] - second_bridge.positions[index],
            axis=1,
        )
        local = int(np.argmin(distances))
        distance = float(distances[local])
        old_index = candidates[local]
        max_distance = max(max_distance, distance)
        if distance == 0.0:
            birth_overlap = float("inf")
            two_sided = float("inf")
        else:
            birth_overlap = birth_sigma_m / distance
            two_sided = (
                min(birth_sigma_m, float(transported_sigma[old_index])) / distance
            )
        min_birth_overlap = min(min_birth_overlap, birth_overlap)
        min_two_sided_overlap = min(min_two_sided_overlap, two_sided)
    passed = bool(
        missing_group_count == 0
        and max_distance <= target_spacing_m * (1.0 + 1.0e-12)
        and min_birth_overlap >= FROZEN_OVERLAP_LAMBDA * (1.0 - 1.0e-12)
        and min_two_sided_overlap >= 2.0 * (1.0 - 1.0e-12)
    )
    return {
        "particle_count": int(second_bridge.positions.shape[0]),
        "missing_same_cell_group_count": missing_group_count,
        "max_nearest_same_cell_distance_m": max_distance,
        "max_distance_over_target_spacing": max_distance / target_spacing_m,
        "min_birth_sigma_over_distance": min_birth_overlap,
        "min_two_sided_sigma_over_distance": min_two_sided_overlap,
        "passed": passed,
    }


def _transport_diagnostics(
    first_bridge: ShadowBridgeResult,
    transport_substeps: int,
) -> dict[str, float]:
    state = make_particle_state(
        first_bridge.positions,
        first_bridge.gamma,
        first_bridge.sigma,
    )
    delta_time = PHYSICAL_RELEASE_DT_S / transport_substeps
    freestream = np.asarray((REFERENCE_SPEED_M_PER_S, 0.0, 0.0), dtype=float)
    max_dt_jacobian = 0.0
    max_dt_self_speed_over_sigma = 0.0
    for _ in range(transport_substeps):
        state, stages = lsrk3_step_direct(
            state,
            delta_time,
            freestream_velocity=freestream,
        )
        for stage in stages:
            jacobian_norm = np.linalg.norm(
                stage.rhs.jacobian.reshape(stage.rhs.jacobian.shape[0], -1),
                axis=1,
            )
            self_speed = np.linalg.norm(stage.rhs.velocity, axis=1)
            max_dt_jacobian = max(
                max_dt_jacobian,
                float(np.max(delta_time * jacobian_norm, initial=0.0)),
            )
            max_dt_self_speed_over_sigma = max(
                max_dt_self_speed_over_sigma,
                float(
                    np.max(
                        delta_time * self_speed / stage.pre.sigma,
                        initial=0.0,
                    )
                ),
            )
    return {
        "max_stage_dt_jacobian_frobenius": max_dt_jacobian,
        "max_stage_dt_self_speed_over_sigma": max_dt_self_speed_over_sigma,
    }


def _run_configuration(
    geometry: GateGeometry,
    *,
    span_cells: int,
    span_subdivisions: int,
    transport_substeps: int,
) -> _ConfigurationResult:
    nodes, anchors, chords, twists = _node_geometry(geometry, span_cells)
    cell_chords = tuple(
        0.5 * (chords[index] + chords[index + 1]) for index in range(span_cells)
    )
    sources = tuple(
        _source_for_cell(geometry, index, cell_chords[index])
        for index in range(span_cells)
    )
    mapper = NodeOwnedDVMRibbonShadow(
        wing_id=f"frontier:{geometry}:wing",
        source_family="lev",
    )

    first_events = _events_for_step(sources)
    first = mapper.map_step(
        _cells_for_events(geometry, first_events, anchors, twists),
        nodes,
        delta_time_s=PHYSICAL_RELEASE_DT_S,
        transport_enabled=True,
    )
    if first.edge_graph is None:
        raise RuntimeError("first release produced no ribbon graph")
    target_spacing = SPAN_M / span_cells / span_subdivisions
    smoothing_radius = FROZEN_OVERLAP_LAMBDA * target_spacing
    first_bridge = deposit_edge_graph_fixed_sigma(
        first.edge_graph,
        smoothing_radius=smoothing_radius,
        step=1,
        physical_owner=RING_PHYSICAL_OWNER,
        owner_state=DIAGNOSTIC_SHADOW_OWNER,
    )
    report = transport_passive_node_frontiers(
        first,
        first_bridge,
        wing_id=f"frontier:{geometry}:wing",
        transport_start_time_s=0.0,
        transport_end_time_s=PHYSICAL_RELEASE_DT_S,
        transport_substeps=transport_substeps,
        freestream_velocity_gp1_m_per_s=(REFERENCE_SPEED_M_PER_S, 0.0, 0.0),
    )
    validate_passive_frontier_transport_report(report)
    transported = materialize_transported_particle_state(report)

    second_events = _events_for_step(sources)
    second = mapper.map_step(
        _cells_for_events(geometry, second_events, anchors, twists),
        nodes,
        delta_time_s=PHYSICAL_RELEASE_DT_S,
        transport_enabled=True,
        source_time_s=PHYSICAL_RELEASE_DT_S,
        frontier_transport_report=report,
    )
    if second.edge_graph is None:
        raise RuntimeError("continuous second release produced no ribbon graph")
    second_bridge = deposit_edge_graph_fixed_sigma(
        second.edge_graph,
        smoothing_radius=smoothing_radius,
        step=2,
        physical_owner=RING_PHYSICAL_OWNER,
        owner_state=DIAGNOSTIC_SHADOW_OWNER,
    )

    initial_invariant = (
        np.linalg.norm(first_bridge.gamma, axis=1) * first_bridge.sigma**2
    )
    final_invariant = np.linalg.norm(transported.gamma, axis=1) * transported.sigma**2
    nonzero = initial_invariant > 0.0
    invariant_relative = float(
        np.max(
            np.abs(final_invariant[nonzero] - initial_invariant[nonzero])
            / initial_invariant[nonzero],
            initial=0.0,
        )
    )
    vector_scale = max(
        1.0e-30,
        float(np.sum(np.linalg.norm(first_bridge.gamma, axis=1))),
    )
    vector_drift_abs = float(np.linalg.norm(np.sum(transported.gamma, axis=0)))
    vector_drift_rel = vector_drift_abs / vector_scale
    overlap = _temporal_overlap_metrics(
        transported.positions,
        transported.sigma,
        report.transported_particle_cloud.lineage,
        second_bridge,
        target_spacing_m=target_spacing,
        birth_sigma_m=smoothing_radius,
    )
    indicators = _transport_diagnostics(first_bridge, transport_substeps)
    frontier_positions = np.asarray(
        [fact.advected_position_gp1_m for fact in report.facts], dtype=float
    )
    probe_velocity = direct_gaussian_erf_velocity_jacobian(
        transported.positions,
        transported.gamma,
        transported.sigma,
        target_positions=FIXED_PROBES_GP1_M,
    ).velocity
    second_birth_positions = np.asarray(
        [
            birth.birth_position_gp1_m
            for birth in second.node_births
            if birth.birth_position_gp1_m is not None
        ],
        dtype=float,
    )
    state_vector = np.concatenate(
        (
            transported.positions.ravel(),
            transported.gamma.ravel(),
            transported.sigma.ravel(),
            frontier_positions.ravel(),
            second_birth_positions.ravel(),
            probe_velocity.ravel(),
        )
    )
    first_diag = first.diagnostics
    second_diag = second.diagnostics
    source_passed = _source_events_pass(first_events, second_events)
    ribbon_passed = bool(
        first_diag.first_node_count == span_cells + 1
        and first_diag.continuous_node_count == 0
        and first_diag.transport_advance_count == 0
        and second_diag.continuous_node_count == span_cells + 1
        and second_diag.transport_advance_count == span_cells + 1
        and first_diag.incidence_residual <= MAX_RIBBON_RESIDUAL
        and first_diag.edge_reconstruction_residual <= MAX_RIBBON_RESIDUAL
        and second_diag.incidence_residual <= MAX_RIBBON_RESIDUAL
        and second_diag.edge_reconstruction_residual <= MAX_RIBBON_RESIDUAL
        and first_diag.seam_count == 0
        and second_diag.seam_count == 0
        and first_diag.feedback_call_count == 0
        and second_diag.feedback_call_count == 0
    )
    bridge_passed = bool(
        first_bridge.diagnostics.max_edge_conservation_abs <= MAX_VECTOR_ABS
        and second_bridge.diagnostics.max_edge_conservation_abs <= MAX_VECTOR_ABS
        and first_bridge.diagnostics.global_conservation_abs <= MAX_VECTOR_ABS
        and second_bridge.diagnostics.global_conservation_abs <= MAX_VECTOR_ABS
        and np.all(first_bridge.sigma == smoothing_radius)
        and np.all(second_bridge.sigma == smoothing_radius)
    )
    transport_passed = bool(
        np.all(np.isfinite(transported.positions))
        and np.all(np.isfinite(transported.gamma))
        and np.all(np.isfinite(transported.sigma))
        and np.all(transported.sigma > 0.0)
        and invariant_relative <= MAX_PARTICLE_INVARIANT_REL
        and vector_drift_rel <= MAX_VECTOR_REL
        and indicators["max_stage_dt_jacobian_frobenius"] <= MAX_STAGE_DT_JACOBIAN
        and indicators["max_stage_dt_self_speed_over_sigma"]
        <= MAX_STAGE_DT_SELF_SPEED_OVER_SIGMA
    )
    passed = bool(
        source_passed
        and ribbon_passed
        and bridge_passed
        and transport_passed
        and overlap["passed"]
    )
    summary = {
        "span_subdivisions": span_subdivisions,
        "transport_substeps": transport_substeps,
        "target_spacing_m": target_spacing,
        "smoothing_radius_m": smoothing_radius,
        "first_particle_count": int(first_bridge.positions.shape[0]),
        "second_particle_count": int(second_bridge.positions.shape[0]),
        "source_kelvin_max_m2_per_s": max(
            [_scaled_kelvin(event) for event in first_events + second_events],
            default=0.0,
        ),
        "first_modes": [birth.mode for birth in first.node_births],
        "second_modes": [birth.mode for birth in second.node_births],
        "frontier_fact_count": len(report.facts),
        "accepted_frontier_count": second_diag.transport_advance_count,
        "report_sha256": report.report_sha256,
        "parent_ribbon_sha256": report.parent_ribbon_digest_sha256,
        "deposited_cloud_before_sha256": (report.deposited_cloud_digest_before_sha256),
        "transported_cloud_after_sha256": (
            report.transported_cloud_digest_after_sha256
        ),
        "particle_invariant_relative_drift_max": invariant_relative,
        "global_vector_drift_abs_m3_per_s": vector_drift_abs,
        "global_vector_drift_relative": vector_drift_rel,
        "max_stage_dt_jacobian_frobenius": indicators[
            "max_stage_dt_jacobian_frobenius"
        ],
        "max_stage_dt_self_speed_over_sigma": indicators[
            "max_stage_dt_self_speed_over_sigma"
        ],
        "temporal_layer_overlap": overlap,
        "source_passed": source_passed,
        "ribbon_passed": ribbon_passed,
        "fixed_sigma_bridge_passed": bridge_passed,
        "transport_passed": transport_passed,
        "passed": passed,
    }
    return _ConfigurationResult(
        summary=summary,
        state_vector=state_vector,
        frontier_positions=frontier_positions,
        transported_probe_velocity=probe_velocity,
    )


def _time_refinement_gate(
    rows: list[_ConfigurationResult],
) -> dict[str, Any]:
    differences = [
        _relative_l2(left.state_vector, right.state_vector)
        for left, right in zip(rows[:-1], rows[1:])
    ]
    fine = differences[-1]
    coarse = differences[-2]
    ratio = None if fine <= 1.0e-14 else coarse / fine
    passed = bool(
        fine <= MAX_FINE_TIME_RELATIVE_DIFFERENCE
        and (
            fine <= 1.0e-14
            or (ratio is not None and ratio >= MIN_TIME_REFINEMENT_RATIO)
        )
    )
    return {
        "consecutive_state_relative_l2": differences,
        "coarse_to_fine_error_ratio": ratio,
        "fine_relative_l2_limit": MAX_FINE_TIME_RELATIVE_DIFFERENCE,
        "minimum_refinement_ratio": MIN_TIME_REFINEMENT_RATIO,
        "passed": passed,
    }


def _spatial_refinement_gate(
    rows: list[_ConfigurationResult],
) -> dict[str, Any]:
    frontier_differences = [
        _relative_l2(left.frontier_positions, right.frontier_positions)
        for left, right in zip(rows[:-1], rows[1:])
    ]
    probe_differences = [
        _relative_l2(
            left.transported_probe_velocity,
            right.transported_probe_velocity,
        )
        for left, right in zip(rows[:-1], rows[1:])
    ]

    def family_passed(values: list[float]) -> bool:
        return bool(
            values[-1] <= MAX_FINE_SPATIAL_RELATIVE_DIFFERENCE
            and values[-1] <= MAX_SPATIAL_DIFFERENCE_GROWTH * max(values[-2], 1.0e-15)
        )

    return {
        "consecutive_frontier_position_relative_l2": frontier_differences,
        "consecutive_probe_velocity_relative_l2": probe_differences,
        "fine_relative_l2_limit": MAX_FINE_SPATIAL_RELATIVE_DIFFERENCE,
        "maximum_difference_growth": MAX_SPATIAL_DIFFERENCE_GROWTH,
        "frontier_passed": family_passed(frontier_differences),
        "probe_velocity_passed": family_passed(probe_differences),
        "passed": bool(
            family_passed(frontier_differences) and family_passed(probe_differences)
        ),
    }


def run_frontier_vertical_gate(config: GateConfig = GateConfig()) -> dict[str, Any]:
    """Run the two-release mechanical gate and return JSON-compatible evidence."""

    _validate_config(config)
    geometry_rows: list[dict[str, Any]] = []
    all_configuration_passed = True
    all_time_passed = True
    all_spatial_passed = True
    for geometry in CASE_GEOMETRIES:
        by_space: list[list[_ConfigurationResult]] = []
        for span_subdivisions in config.span_subdivisions:
            time_rows = [
                _run_configuration(
                    geometry,
                    span_cells=config.span_cells,
                    span_subdivisions=span_subdivisions,
                    transport_substeps=transport_substeps,
                )
                for transport_substeps in config.transport_substeps
            ]
            by_space.append(time_rows)
        time_gates = [_time_refinement_gate(rows) for rows in by_space]
        spatial_rows = [rows[-1] for rows in by_space]
        spatial_gate = _spatial_refinement_gate(spatial_rows)
        configuration_passed = all(
            row.summary["passed"] for rows in by_space for row in rows
        )
        time_passed = all(gate["passed"] for gate in time_gates)
        all_configuration_passed &= configuration_passed
        all_time_passed &= time_passed
        all_spatial_passed &= spatial_gate["passed"]
        geometry_rows.append(
            {
                "geometry": geometry,
                "configurations": [row.summary for rows in by_space for row in rows],
                "time_refinement_by_span_subdivision": [
                    {
                        "span_subdivisions": span_subdivisions,
                        **gate,
                    }
                    for span_subdivisions, gate in zip(
                        config.span_subdivisions, time_gates, strict=True
                    )
                ],
                "spatial_refinement_at_finest_time": spatial_gate,
                "configuration_passed": configuration_passed,
                "time_refinement_passed": time_passed,
                "spatial_refinement_passed": spatial_gate["passed"],
                "passed": bool(
                    configuration_passed and time_passed and spatial_gate["passed"]
                ),
            }
        )

    passed = bool(all_configuration_passed and all_time_passed and all_spatial_passed)
    repo_root = Path(__file__).resolve().parents[2]
    code_paths = {
        "runner": Path(__file__),
        "manufactured_geometry_provider": Path(__file__).with_name(
            "run_v5h_manufactured_shadow_gate.py"
        ),
        "dvm_source": Path(__file__).with_name("v5h_dvm_source.py"),
        "dvm_node_ribbon": Path(__file__).with_name("v5h_dvm_node_ribbon.py"),
        "passive_frontier_transport": Path(__file__).with_name(
            "v5h_passive_frontier_transport.py"
        ),
        "ldvm_fourier": repo_root / "platform/ldvm_fourier.py",
        "ldvm_uvlm_correction": repo_root
        / "platform/forward_flight_benchmarks/ldvm_uvlm_correction.py",
        "rvpm_edge_bridge": repo_root / "src/fluxvortex/rvpm_edge_bridge.py",
        "rvpm_reference": repo_root / "src/fluxvortex/rvpm_reference.py",
        "rvpm_transport": repo_root / "src/fluxvortex/rvpm_transport.py",
    }
    return {
        "schema_id": RUN_SCHEMA_ID,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scope": "non_target_two_release_frontier_vertical_slice_only",
        "observation_access": "none",
        "target_case_branch": "none",
        "canonical_eligible": False,
        "feedback_call_count": 0,
        "parent_write_count": 0,
        "load_write_count": 0,
        "force_channel": "not_implemented_and_not_evaluated",
        "passive_frontier_scope": PASSIVE_FRONTIER_CONTINUATION_SCOPE,
        "config": {
            "span_cells": config.span_cells,
            "span_subdivisions": list(config.span_subdivisions),
            "transport_substeps": list(config.transport_substeps),
            "physical_release_dt_s": PHYSICAL_RELEASE_DT_S,
            "frozen_overlap_lambda": FROZEN_OVERLAP_LAMBDA,
        },
        "thresholds": {
            "source_kelvin_m2_per_s": MAX_SOURCE_KELVIN_M2_PER_S,
            "ribbon_residual": MAX_RIBBON_RESIDUAL,
            "vector_absolute": MAX_VECTOR_ABS,
            "vector_relative": MAX_VECTOR_REL,
            "particle_invariant_relative": MAX_PARTICLE_INVARIANT_REL,
            "stage_dt_jacobian": MAX_STAGE_DT_JACOBIAN,
            "stage_dt_self_speed_over_sigma": (MAX_STAGE_DT_SELF_SPEED_OVER_SIGMA),
            "fine_time_relative_difference": (MAX_FINE_TIME_RELATIVE_DIFFERENCE),
            "minimum_time_refinement_ratio": MIN_TIME_REFINEMENT_RATIO,
            "fine_spatial_relative_difference": (MAX_FINE_SPATIAL_RELATIVE_DIFFERENCE),
            "maximum_spatial_difference_growth": (MAX_SPATIAL_DIFFERENCE_GROWTH),
        },
        "gate_summary": {
            "configuration_mechanics_passed": all_configuration_passed,
            "time_refinement_passed": all_time_passed,
            "spatial_refinement_passed": all_spatial_passed,
            "stop_required": not passed,
        },
        "geometries": geometry_rows,
        "limitations": LIMITATIONS,
        "blocked": {
            "third_release": "requires cumulative-cloud exact-once merge v2",
            "target_scoring": "prohibited by this mechanical gate",
        },
        "code_sha256": {name: _sha256_file(path) for name, path in code_paths.items()},
    }


def write_artifact(result: dict[str, Any], output: Path) -> Path:
    """Write one JSON artifact without overwriting existing evidence."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_frontier_vertical_gate()
    path = write_artifact(result, args.output)
    print(json.dumps({"artifact": str(path), "status": result["status"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
