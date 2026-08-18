"""Observation-free FluxV v5h6 generic geometry/refinement gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from math import cos, radians, sin
from pathlib import Path
from time import monotonic
from typing import Any, Final, Literal

import numpy as np
import pterasoftware as ps

from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian

from .fluxv_v5h5_synchronized_coupling import (
    SynchronizedPteraRVPMCouplingSolver,
    SynchronizedReleaseLayer,
    make_fluxv_v5h5_synchronized_solver,
    validate_synchronized_cloud_transport_report,
)
from .ldvm_uvlm_correction import LDVMSectionSettings, LESPThreshold
from .v5h_dvm_node_placement import (
    DVMNodePlacementCell,
    GP1NodeSectionFact,
    NodeLocalDVMPlacementAdapter,
)
from .v5h_dvm_node_ribbon import DVMPlaneToGP1Map, DVMSpanCellSource
from .v5h_dvm_source import V5hDVMSource


GateGeometry = Literal["straight", "taper", "twist"]

RUN_ID: Final = "20260815_fluxv_v5h6_geometry_refinement"
RUN_SCHEMA_ID: Final = "fluxv-v5h6-geometry-refinement-v1"
GEOMETRIES: Final = ("straight", "taper", "twist")
SPAN_CELLS: Final = 4
SPAN_M: Final = 0.60
ROOT_CHORD_M: Final = 0.20
TIP_TAPER_RATIO: Final = 0.60
TIP_TWIST_DEG: Final = 20.0
SPEED_M_PER_S: Final = 2.0
SOURCE_INCIDENCE_DEG: Final = 60.0
PHYSICAL_HORIZON_S: Final = 0.06
BIRTH_SIGMA_M: Final = 0.085
BASE_SPACING_M: Final = 0.04
SPATIAL_LEVELS: Final = (0, 1, 2)
TEMPORAL_DT_S: Final = (0.02, 0.01, 0.005)
FINE_RELATIVE_LIMIT: Final = 0.02
MIN_DIFFERENCE_RATIO: Final = 1.25
MAX_PARTICLES_PER_CONFIGURATION: Final = 20_000
MAX_RUNTIME_S: Final = 30.0 * 60.0
FIXED_PROBES_GP1_M: Final = np.asarray(
    (
        (0.05, 0.075, 0.35),
        (0.10, 0.225, 0.45),
        (-0.05, 0.375, 0.40),
        (0.15, 0.525, 0.50),
        (0.20, 0.300, 0.65),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class GeometryNode:
    node_id: str
    leading_edge_gp1_m: tuple[float, float, float]
    chord_m: float
    twist_deg: float
    x_axis_gp1: tuple[float, float, float]
    z_axis_gp1: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ConfigurationResult:
    geometry: GateGeometry
    refinement_level: int
    delta_time_s: float
    num_steps: int
    particle_count: int
    report_sha256: tuple[str, ...]
    frontier_displacement_gp1_m: np.ndarray
    probe_velocity_gp1_m_per_s: np.ndarray
    final_cloud_sha256: str
    max_no_penetration_abs: float
    max_exact_append_abs: float
    max_frontier_replay_abs: float
    mechanics_passed: bool


def _strict_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        value = np.asarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _chord_twist(geometry: GateGeometry, eta: float) -> tuple[float, float]:
    if geometry == "straight":
        return ROOT_CHORD_M, 0.0
    if geometry == "taper":
        return ROOT_CHORD_M * (1.0 - (1.0 - TIP_TAPER_RATIO) * eta), 0.0
    if geometry == "twist":
        return ROOT_CHORD_M, TIP_TWIST_DEG * eta
    raise ValueError(f"unknown geometry {geometry!r}")


def _geometry_nodes(geometry: GateGeometry) -> tuple[GeometryNode, ...]:
    nodes: list[GeometryNode] = []
    for index in range(SPAN_CELLS + 1):
        eta = index / SPAN_CELLS
        chord, twist_deg = _chord_twist(geometry, eta)
        twist = radians(twist_deg)
        x_axis = (cos(twist), 0.0, -sin(twist))
        z_axis = (-sin(twist), 0.0, -cos(twist))
        nodes.append(
            GeometryNode(
                node_id=f"{geometry}:node:{index}",
                leading_edge_gp1_m=(0.0, SPAN_M * eta, 0.0),
                chord_m=chord,
                twist_deg=twist_deg,
                x_axis_gp1=x_axis,
                z_axis_gp1=z_axis,
            )
        )
    return tuple(nodes)


def _threshold() -> LESPThreshold:
    return LESPThreshold(
        value=0.18,
        section_family="generic thin flat plate",
        reynolds=30_000.0,
        source="published source input used only by a non-target mechanical gate",
        source_role="published_source_input",
    )


def _source(
    geometry: GateGeometry,
    role: str,
    index: int,
    chord_m: float,
    delta_time_s: float,
) -> V5hDVMSource:
    return V5hDVMSource(
        physical_section_id=f"v5h6:{geometry}:{role}:section:{index}",
        physical_strip_id=f"v5h6:{geometry}:{role}:strip:{index}",
        geometry_identity=(
            f"generic zero-camber {geometry} flat-plate section chord={chord_m.hex()}"
        ),
        reference_speed_m_per_s=SPEED_M_PER_S,
        reference_chord_m=chord_m,
        zero_camber_surrogate=True,
        delta_time_convective=delta_time_s * SPEED_M_PER_S / chord_m,
        pivot_fraction_chord=0.25,
        threshold=_threshold(),
        settings=LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48),
    )


def _node_fact(
    *,
    geometry: GateGeometry,
    node: GeometryNode,
    event: Any,
    source_time_s: float,
) -> GP1NodeSectionFact:
    x_axis = np.asarray(node.x_axis_gp1)
    z_axis = np.asarray(node.z_axis_gp1)
    lev_2d = np.asarray(
        event.lev_placement.edge_anchor_position_over_chord_backend_world
    )
    tev_2d = np.asarray(
        event.tev_placement.edge_anchor_position_over_chord_backend_world
    )
    lev_gp1 = np.asarray(node.leading_edge_gp1_m)
    tev_gp1 = lev_gp1 + node.chord_m * (
        (tev_2d[0] - lev_2d[0]) * x_axis + (tev_2d[1] - lev_2d[1]) * z_axis
    )
    return GP1NodeSectionFact(
        wing_id=f"v5h6:{geometry}:wing",
        node_id=node.node_id,
        source_step_index=int(event.lineage.source_step_index),
        source_time_s=source_time_s,
        event=event,
        lev_edge_anchor_gp1_m=tuple(lev_gp1),
        tev_edge_anchor_gp1_m=tuple(tev_gp1),
        reference_chord_m=node.chord_m,
        reference_speed_m_per_s=SPEED_M_PER_S,
        dvm_x_axis_gp1=node.x_axis_gp1,
        dvm_z_axis_gp1=node.z_axis_gp1,
        positive_span_axis_gp1=(0.0, 1.0, 0.0),
        topology_patch_id=f"v5h6:{geometry}:main-patch",
        coordinate_frame_id="ptera-gp1-v5h6",
        node_lineage_id=event.lineage.section_lineage_id,
        geometry_token=event.provenance.geometry_hash_sha256,
    )


def _cell_plane_map(
    geometry: GateGeometry,
    cell_index: int,
    left: GeometryNode,
    right: GeometryNode,
) -> DVMPlaneToGP1Map:
    twist = radians(0.5 * (left.twist_deg + right.twist_deg))
    x_axis = np.asarray((cos(twist), 0.0, -sin(twist)))
    span = np.asarray(right.leading_edge_gp1_m) - np.asarray(left.leading_edge_gp1_m)
    span /= np.linalg.norm(span)
    x_axis -= np.dot(x_axis, span) * span
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(span, x_axis)
    z_axis /= np.linalg.norm(z_axis)
    return DVMPlaneToGP1Map(
        origin_gp1_m=0.5
        * (np.asarray(left.leading_edge_gp1_m) + np.asarray(right.leading_edge_gp1_m)),
        x_axis_gp1=x_axis,
        z_axis_gp1=z_axis,
        positive_circulation_axis_gp1=span,
        circulation_to_ring_traversal_sign=1,
        provenance=f"v5h6 shared {geometry} geometry cell {cell_index}",
    )


def _release_layers(
    geometry: GateGeometry,
    *,
    delta_time_s: float,
    num_steps: int,
) -> tuple[SynchronizedReleaseLayer, ...]:
    nodes = _geometry_nodes(geometry)
    node_sources = tuple(
        _source(geometry, "node", index, node.chord_m, delta_time_s)
        for index, node in enumerate(nodes)
    )
    cell_chords = tuple(
        0.5 * (nodes[index].chord_m + nodes[index + 1].chord_m)
        for index in range(SPAN_CELLS)
    )
    cell_sources = tuple(
        _source(geometry, "cell", index, chord, delta_time_s)
        for index, chord in enumerate(cell_chords)
    )
    placement = NodeLocalDVMPlacementAdapter(wing_id=f"v5h6:{geometry}:wing")
    layers: list[SynchronizedReleaseLayer] = []
    for step in range(1, num_steps + 1):
        source_time = (step - 1) * delta_time_s
        node_events = tuple(
            source.step(np.deg2rad(SOURCE_INCIDENCE_DEG), 0.0, 0.0)
            for source in node_sources
        )
        cell_events = tuple(
            source.step(np.deg2rad(SOURCE_INCIDENCE_DEG), 0.0, 0.0)
            for source in cell_sources
        )
        facts = tuple(
            _node_fact(
                geometry=geometry,
                node=node,
                event=event,
                source_time_s=source_time,
            )
            for node, event in zip(nodes, node_events, strict=True)
        )
        placement_cells = tuple(
            DVMNodePlacementCell(
                cell_id=f"{geometry}:cell:{index}",
                left_node_fact=facts[index],
                right_node_fact=facts[index + 1],
                cell_source_event=event,
            )
            for index, event in enumerate(cell_events)
        )
        placement_result = placement.map_step(
            placement_cells, delta_time_s=delta_time_s
        )
        cells = tuple(
            DVMSpanCellSource(
                cell_id=f"{geometry}:cell:{index}",
                left_node_id=nodes[index].node_id,
                right_node_id=nodes[index + 1].node_id,
                event=event,
                plane_to_gp1=_cell_plane_map(
                    geometry, index, nodes[index], nodes[index + 1]
                ),
            )
            for index, event in enumerate(cell_events)
        )
        layers.append(
            SynchronizedReleaseLayer(
                source_step_index=step,
                source_time_s=source_time,
                cells=cells,
                node_placement_result=placement_result,
            )
        )
    return tuple(layers)


def _problem(
    geometry: GateGeometry, *, delta_time_s: float, num_steps: int
) -> ps.problems.UnsteadyProblem:
    root_chord, _ = _chord_twist(geometry, 0.0)
    tip_chord, tip_twist = _chord_twist(geometry, 1.0)
    airfoil = ps.geometry.airfoil.Airfoil(name="naca0001")
    sections = [
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=root_chord,
            Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            num_spanwise_panels=SPAN_CELLS,
            spanwise_spacing="uniform",
        ),
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=tip_chord,
            Lp_Wcsp_Lpp=(0.0, SPAN_M, 0.0),
            angles_Wcsp_to_Wcs_ixyz=(0.0, tip_twist, 0.0),
            num_spanwise_panels=None,
            spanwise_spacing=None,
        ),
    ]
    wing = ps.geometry.wing.Wing(
        name=f"v5h6 generic {geometry}",
        wing_cross_sections=sections,
        symmetric=False,
        num_chordwise_panels=1,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name=f"v5h6 generic {geometry} airplane",
        s_ref=0.5 * (root_chord + tip_chord) * SPAN_M,
        c_ref=root_chord,
        b_ref=SPAN_M,
    )
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=[
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=section
            )
            for section in sections
        ],
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wing_movement]
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=1.2, vCg__E=SPEED_M_PER_S, alpha=4.0, beta=0.0, nu=1.5e-5
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=(
            ps.movements.operating_point_movement.OperatingPointMovement(
                base_operating_point=operating_point
            )
        ),
        delta_time=delta_time_s,
        num_steps=num_steps,
        max_wake_rows=2,
    )
    return ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)


def _run_configuration(
    geometry: GateGeometry,
    *,
    refinement_level: int,
    delta_time_s: float,
) -> ConfigurationResult:
    raw_steps = PHYSICAL_HORIZON_S / delta_time_s
    num_steps = int(round(raw_steps))
    if num_steps < 1 or abs(num_steps * delta_time_s - PHYSICAL_HORIZON_S) > 1e-15:
        raise ValueError("time step must divide the fixed physical horizon")
    solver = make_fluxv_v5h5_synchronized_solver(
        _problem(geometry, delta_time_s=delta_time_s, num_steps=num_steps),
        enabled=True,
        release_layers=_release_layers(
            geometry, delta_time_s=delta_time_s, num_steps=num_steps
        ),
        wing_id=f"v5h6:{geometry}:wing",
        smoothing_radius_m=BIRTH_SIGMA_M,
        base_target_spacing_m=BASE_SPACING_M,
        refinement_level=refinement_level,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    if not isinstance(solver, SynchronizedPteraRVPMCouplingSolver):
        raise RuntimeError("v5h6 enabled factory returned a foreign solver")
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    reports = tuple(
        validate_synchronized_cloud_transport_report(report)
        for report in solver.v5h5_transport_reports
    )
    if len(reports) != num_steps:
        raise RuntimeError("v5h6 did not close every synchronized layer")
    final = reports[-1]
    if final.total_particle_count > MAX_PARTICLES_PER_CONFIGURATION:
        raise RuntimeError("v5h6 configuration exceeded its particle cap")
    nodes = _geometry_nodes(geometry)
    anchors = {node.node_id: np.asarray(node.leading_edge_gp1_m) for node in nodes}
    facts = sorted(final.facts, key=lambda fact: fact.node_id)
    frontier = np.asarray(
        [
            np.asarray(fact.advected_position_gp1_m) - anchors[fact.node_id]
            for fact in facts
        ],
        dtype=np.float64,
    )
    state = final.transported_state
    probe_velocity = direct_gaussian_erf_velocity_jacobian(
        state.positions,
        state.gamma,
        state.sigma,
        target_positions=FIXED_PROBES_GP1_M,
    ).velocity
    max_residual = max(
        float(report.ptera_feedback_report.no_penetration_max_abs)
        for report in reports
        if report.ptera_feedback_report is not None
    )
    max_append = max(report.exact_append_prefix_max_abs for report in reports)
    max_replay = max(report.frontier_stage_replay_max_abs for report in reports)
    mechanics = (
        np.all(np.isfinite(frontier))
        and np.all(np.isfinite(probe_velocity))
        and np.all(np.isfinite(state.positions))
        and np.all(np.isfinite(state.gamma))
        and np.all(np.isfinite(state.sigma))
        and np.all(state.sigma > 0.0)
        and max_residual <= 1.0e-12
        and max_append == 0.0
        and max_replay == 0.0
        and all(report.parent_state_unchanged for report in reports)
        and all(
            report.feedback_write_count
            == report.parent_write_count
            == report.load_write_count
            == 0
            for report in reports
        )
    )
    return ConfigurationResult(
        geometry=geometry,
        refinement_level=refinement_level,
        delta_time_s=delta_time_s,
        num_steps=num_steps,
        particle_count=final.total_particle_count,
        report_sha256=tuple(report.report_sha256 for report in reports),
        frontier_displacement_gp1_m=frontier,
        probe_velocity_gp1_m_per_s=np.asarray(probe_velocity),
        final_cloud_sha256=_array_sha256(state.positions, state.gamma, state.sigma),
        max_no_penetration_abs=max_residual,
        max_exact_append_abs=max_append,
        max_frontier_replay_abs=max_replay,
        mechanics_passed=bool(mechanics),
    )


def _relative_difference(coarse: np.ndarray, fine: np.ndarray) -> float:
    if coarse.shape != fine.shape:
        return float("nan")
    denominator = float(np.linalg.norm(fine.ravel()))
    if not np.isfinite(denominator) or denominator <= 1.0e-12:
        return float("nan")
    return float(np.linalg.norm((coarse - fine).ravel()) / denominator)


def _convergence_gate(
    rows: tuple[ConfigurationResult, ConfigurationResult, ConfigurationResult],
    attribute: str,
) -> dict[str, Any]:
    arrays = tuple(np.asarray(getattr(row, attribute)) for row in rows)
    first = _relative_difference(arrays[0], arrays[1])
    second = _relative_difference(arrays[1], arrays[2])
    ratio = first / second if np.isfinite(second) and second > 0.0 else None
    exact = first == 0.0 and second == 0.0
    passed = (
        all(np.all(np.isfinite(array)) for array in arrays)
        and np.isfinite(first)
        and np.isfinite(second)
        and second <= FINE_RELATIVE_LIMIT
        and (exact or (ratio is not None and ratio >= MIN_DIFFERENCE_RATIO))
    )
    return {
        "observable": attribute,
        "coarse_to_middle_relative_difference": first,
        "middle_to_fine_relative_difference": second,
        "difference_ratio": ratio,
        "exact_across_levels": exact,
        "passed": bool(passed),
    }


def _row_payload(row: ConfigurationResult) -> dict[str, Any]:
    return {
        "geometry": row.geometry,
        "refinement_level": row.refinement_level,
        "delta_time_s": row.delta_time_s,
        "num_steps": row.num_steps,
        "particle_count": row.particle_count,
        "report_sha256": list(row.report_sha256),
        "frontier_displacement_gp1_m": row.frontier_displacement_gp1_m.tolist(),
        "probe_velocity_gp1_m_per_s": row.probe_velocity_gp1_m_per_s.tolist(),
        "final_cloud_sha256": row.final_cloud_sha256,
        "max_no_penetration_abs": row.max_no_penetration_abs,
        "max_exact_append_abs": row.max_exact_append_abs,
        "max_frontier_replay_abs": row.max_frontier_replay_abs,
        "mechanics_passed": row.mechanics_passed,
    }


def run_minimal_smoke() -> dict[str, Any]:
    row = _run_configuration("straight", refinement_level=0, delta_time_s=0.02)
    return {
        "schema_id": "fluxv-v5h6-minimal-smoke-v1",
        "passed": row.mechanics_passed,
        "row": _row_payload(row),
    }


def run_gate() -> dict[str, Any]:
    start = monotonic()
    rows: dict[tuple[GateGeometry, int, float], ConfigurationResult] = {}
    for geometry in GEOMETRIES:
        for level in SPATIAL_LEVELS:
            rows[(geometry, level, 0.02)] = _run_configuration(
                geometry, refinement_level=level, delta_time_s=0.02
            )
        for delta_time in TEMPORAL_DT_S[1:]:
            rows[(geometry, 0, delta_time)] = _run_configuration(
                geometry, refinement_level=0, delta_time_s=delta_time
            )
        if monotonic() - start > MAX_RUNTIME_S:
            raise TimeoutError("v5h6 exceeded its preregistered runtime budget")

    spatial: list[dict[str, Any]] = []
    temporal: list[dict[str, Any]] = []
    for geometry in GEOMETRIES:
        spatial_rows = tuple(rows[(geometry, level, 0.02)] for level in SPATIAL_LEVELS)
        temporal_rows = tuple(rows[(geometry, 0, dt)] for dt in TEMPORAL_DT_S)
        spatial.append(
            {
                "geometry": geometry,
                "frontier": _convergence_gate(
                    spatial_rows, "frontier_displacement_gp1_m"
                ),
                "probe": _convergence_gate(spatial_rows, "probe_velocity_gp1_m_per_s"),
            }
        )
        temporal.append(
            {
                "geometry": geometry,
                "frontier": _convergence_gate(
                    temporal_rows, "frontier_displacement_gp1_m"
                ),
                "probe": _convergence_gate(temporal_rows, "probe_velocity_gp1_m_per_s"),
            }
        )
    mechanics = all(row.mechanics_passed for row in rows.values())
    convergence = all(
        family[observable]["passed"]
        for family in (*spatial, *temporal)
        for observable in ("frontier", "probe")
    )
    passed = mechanics and convergence
    ordered_rows = [
        rows[key] for key in sorted(rows, key=lambda item: (item[0], item[1], item[2]))
    ]
    return {
        "schema_id": RUN_SCHEMA_ID,
        "run_id": RUN_ID,
        "status": (
            "go_generic_geometry_refinement_mechanics_only"
            if passed
            else "stop_v5h6_refinement_gate_failed"
        ),
        "passed": passed,
        "scope": {
            "evaluation": "simulation_only_self_supervised_mechanical",
            "observation_access": "none",
            "target_case_branch": "none",
            "paper_scoring": False,
            "claim_limit": (
                "generic geometry/refinement mechanics only; no aerodynamic "
                "accuracy or general 3-D stability claim"
            ),
        },
        "contract": {
            "geometries": list(GEOMETRIES),
            "span_cells": SPAN_CELLS,
            "physical_horizon_s": PHYSICAL_HORIZON_S,
            "source_incidence_deg": SOURCE_INCIDENCE_DEG,
            "birth_sigma_m": BIRTH_SIGMA_M,
            "base_spacing_m": BASE_SPACING_M,
            "spatial_levels": list(SPATIAL_LEVELS),
            "temporal_dt_s": list(TEMPORAL_DT_S),
            "fine_relative_limit": FINE_RELATIVE_LIMIT,
            "minimum_difference_ratio": MIN_DIFFERENCE_RATIO,
            "particle_cap": MAX_PARTICLES_PER_CONFIGURATION,
            "runtime_budget_s": MAX_RUNTIME_S,
        },
        "unique_configuration_count": len(rows),
        "spatial_row_count": len(GEOMETRIES) * len(SPATIAL_LEVELS),
        "temporal_row_count": len(GEOMETRIES) * len(TEMPORAL_DT_S),
        "rows": [_row_payload(row) for row in ordered_rows],
        "spatial_convergence": spatial,
        "temporal_convergence": temporal,
        "gates": {
            "all_configuration_mechanics_passed": mechanics,
            "all_spatial_and_temporal_convergence_passed": convergence,
            "target_access_count": 0,
            "paper_scoring_call_count": 0,
        },
    }


def _source_paths(root: Path) -> tuple[Path, ...]:
    relative = (
        "platform/forward_flight_benchmarks/run_fluxv_v5h6_geometry_refinement.py",
        "platform/tests/test_run_fluxv_v5h6_geometry_refinement.py",
        "platform/forward_flight_benchmarks/fluxv_v5h5_synchronized_coupling.py",
        "platform/forward_flight_benchmarks/fluxv_v5h3_native_feedback.py",
        "platform/forward_flight_benchmarks/fluxv_v5h4_ptera_rvpm_transport.py",
        "platform/forward_flight_benchmarks/v5h_dvm_source.py",
        "platform/forward_flight_benchmarks/v5h_dvm_node_placement.py",
        "platform/forward_flight_benchmarks/v5h_dvm_node_ribbon.py",
        "platform/ldvm_fourier.py",
        "platform/forward_flight_benchmarks/ldvm_uvlm_correction.py",
        "src/fluxvortex/rvpm_dyadic_edge_bridge.py",
        "src/fluxvortex/rvpm_edge_bridge.py",
        "src/fluxvortex/rvpm_reference.py",
        "src/fluxvortex/rvpm_transport.py",
    )
    return tuple(root / item for item in relative)


def write_artifact(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_dir}")
    output_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    summary = run_gate()
    source_manifest = {
        "schema_id": "source-sha256-v1",
        "files": {
            str(path.relative_to(root)): _file_sha256(path)
            for path in _source_paths(root)
        },
    }
    (output_dir / "summary.json").write_text(_strict_json(summary), encoding="utf-8")
    (output_dir / "source_manifest.json").write_text(
        _strict_json(source_manifest), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# FluxV v5h6 generic geometry/refinement gate\n\n"
        "Observation-free mechanical convergence evidence only. No target-paper "
        "data, scoring, or aerodynamic-accuracy claim is present.\n",
        encoding="utf-8",
    )
    payloads = ("README.md", "source_manifest.json", "summary.json")
    result_manifest = {name: _file_sha256(output_dir / name) for name in payloads}
    (output_dir / "result_manifest.json").write_text(
        _strict_json({"schema_id": "result-sha256-v1", "files": result_manifest}),
        encoding="utf-8",
    )
    checksum_files = (*payloads, "result_manifest.json")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{_file_sha256(output_dir / name)}  {name}" for name in checksum_files
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        summary = run_minimal_smoke()
    else:
        if arguments.output_dir is None:
            parser.error("--output-dir is required unless --smoke is used")
        summary = write_artifact(arguments.output_dir)
    print(_strict_json(summary), end="")
    return 0 if bool(summary["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
