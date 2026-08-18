"""Run and archive the non-target FluxV v5h5 synchronized mechanical gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pterasoftware as ps

from fluxvortex.solver import UVPMHybridSolver

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
from .v5h_dvm_node_ribbon import (
    DVMPlaneToGP1Map,
    DVMSpanCellSource,
)
from .v5h_dvm_source import V5hDVMSource


DT_S = 0.02
SPEED_M_PER_S = 2.0
CHORD_M = 0.25
SIGMA_BIRTH_M = 0.085
BASE_SPACING_M = 0.04


def _strict_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _threshold() -> LESPThreshold:
    return LESPThreshold(
        value=0.18,
        section_family="generic thin flat plate",
        reynolds=30_000.0,
        source="published source input used only by a non-target mechanical gate",
        source_role="published_source_input",
    )


def _source(role: str, index: int) -> V5hDVMSource:
    return V5hDVMSource(
        physical_section_id=f"v5h5:{role}:section:{index}",
        physical_strip_id=f"v5h5:{role}:strip:{index}",
        geometry_identity="explicit zero-camber flat-plate surrogate",
        reference_speed_m_per_s=SPEED_M_PER_S,
        reference_chord_m=CHORD_M,
        zero_camber_surrogate=True,
        delta_time_convective=DT_S * SPEED_M_PER_S / CHORD_M,
        pivot_fraction_chord=0.25,
        threshold=_threshold(),
        settings=LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48),
    )


def _plane_map(index: int) -> DVMPlaneToGP1Map:
    return DVMPlaneToGP1Map(
        origin_gp1_m=np.asarray((float(index), 0.0, 0.0)),
        x_axis_gp1=np.asarray((1.0, 0.0, 0.0)),
        z_axis_gp1=np.asarray((0.0, 0.0, -1.0)),
        positive_circulation_axis_gp1=np.asarray((0.0, 1.0, 0.0)),
        circulation_to_ring_traversal_sign=1,
        provenance="v5h5 non-target DVM x-z to Ptera GP1 map",
    )


def _node_fact(index: int, event: Any, source_time_s: float) -> GP1NodeSectionFact:
    x_axis = np.asarray((1.0, 0.0, 0.0))
    z_axis = np.asarray((0.0, 0.0, -1.0))
    lev_2d = np.asarray(
        event.lev_placement.edge_anchor_position_over_chord_backend_world
    )
    tev_2d = np.asarray(
        event.tev_placement.edge_anchor_position_over_chord_backend_world
    )
    lev_gp1 = np.asarray((0.0, float(index), 0.0))
    chord = float(event.provenance.position_scale_chord_m)
    tev_gp1 = lev_gp1 + chord * (
        (tev_2d[0] - lev_2d[0]) * x_axis + (tev_2d[1] - lev_2d[1]) * z_axis
    )
    return GP1NodeSectionFact(
        wing_id="wing",
        node_id=f"node-{index}",
        source_step_index=int(event.lineage.source_step_index),
        source_time_s=source_time_s,
        event=event,
        lev_edge_anchor_gp1_m=tuple(lev_gp1),
        tev_edge_anchor_gp1_m=tuple(tev_gp1),
        reference_chord_m=chord,
        reference_speed_m_per_s=float(
            event.provenance.circulation_scale_u_times_c_m2_per_s / chord
        ),
        dvm_x_axis_gp1=tuple(x_axis),
        dvm_z_axis_gp1=tuple(z_axis),
        positive_span_axis_gp1=(0.0, 1.0, 0.0),
        topology_patch_id="v5h5-main-patch",
        coordinate_frame_id="ptera-gp1-v5h5-wing",
        node_lineage_id=event.lineage.section_lineage_id,
        geometry_token=event.provenance.geometry_hash_sha256,
    )


def _release_layers(num_steps: int) -> tuple[SynchronizedReleaseLayer, ...]:
    node_sources = tuple(_source("node", index) for index in range(3))
    cell_sources = tuple(_source("cell", index) for index in range(2))
    placement = NodeLocalDVMPlacementAdapter(wing_id="wing")
    layers: list[SynchronizedReleaseLayer] = []
    for step in range(1, num_steps + 1):
        source_time = (step - 1) * DT_S
        node_events = tuple(
            source.step(np.deg2rad(35.0), 0.0, 0.0) for source in node_sources
        )
        cell_events = tuple(
            source.step(np.deg2rad(35.0), 0.0, 0.0) for source in cell_sources
        )
        facts = tuple(
            _node_fact(index, event, source_time)
            for index, event in enumerate(node_events)
        )
        placement_cells = tuple(
            DVMNodePlacementCell(
                cell_id=f"cell-{index}",
                left_node_fact=facts[index],
                right_node_fact=facts[index + 1],
                cell_source_event=event,
            )
            for index, event in enumerate(cell_events)
        )
        placement_result = placement.map_step(
            placement_cells,
            delta_time_s=DT_S,
        )
        cells = tuple(
            DVMSpanCellSource(
                cell_id=f"cell-{index}",
                left_node_id=f"node-{index}",
                right_node_id=f"node-{index + 1}",
                event=event,
                plane_to_gp1=_plane_map(index),
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


def _problem(num_steps: int) -> ps.problems.UnsteadyProblem:
    airfoil = ps.geometry.airfoil.Airfoil(name="naca0001")
    sections = [
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=0.2,
            Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            num_spanwise_panels=2,
            spanwise_spacing="uniform",
        ),
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=0.2,
            Lp_Wcsp_Lpp=(0.0, 0.4, 0.0),
            num_spanwise_panels=None,
            spanwise_spacing=None,
        ),
    ]
    wing = ps.geometry.wing.Wing(
        name="v5h5 generic API rectangle",
        wing_cross_sections=sections,
        symmetric=False,
        num_chordwise_panels=1,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing], name="v5h5 generic airplane", s_ref=0.08, c_ref=0.2, b_ref=0.4
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
        base_airplane=airplane,
        wing_movements=[wing_movement],
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=1.2, vCg__E=2.0, alpha=4.0, beta=0.0, nu=1.5e-5
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=(
            ps.movements.operating_point_movement.OperatingPointMovement(
                base_operating_point=operating_point
            )
        ),
        delta_time=DT_S,
        num_steps=num_steps,
        max_wake_rows=2,
    )
    return ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)


def _run_solver(
    *, num_steps: int, feedback_enabled: bool, transport_enabled: bool
) -> SynchronizedPteraRVPMCouplingSolver:
    solver = make_fluxv_v5h5_synchronized_solver(
        _problem(num_steps),
        enabled=True,
        release_layers=_release_layers(num_steps),
        wing_id="wing",
        smoothing_radius_m=SIGMA_BIRTH_M,
        base_target_spacing_m=BASE_SPACING_M,
        refinement_level=0,
        feedback_enabled=feedback_enabled,
        transport_enabled=transport_enabled,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    if not isinstance(solver, SynchronizedPteraRVPMCouplingSolver):
        raise RuntimeError("v5h5 enabled factory returned the wrong solver")
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def _run_parent(num_steps: int) -> UVPMHybridSolver:
    solver = UVPMHybridSolver(
        _problem(num_steps), max_particles=100, stretch=False, free_wake=False
    )
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def _native_state_sha256(solver: UVPMHybridSolver) -> str:
    digest = sha256()
    for name in (
        "_current_bound_vortex_strengths",
        "_current_wake_vortex_strengths",
        "_current_wake_vortex_ages",
        "_currentStackBrwrvp_GP1_CgP1",
        "_currentStackFrwrvp_GP1_CgP1",
        "_currentStackFlwrvp_GP1_CgP1",
        "_currentStackBlwrvp_GP1_CgP1",
    ):
        array = np.asarray(getattr(solver, name))
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    for step, problem in enumerate(solver.steady_problems):
        for airplane_index, airplane in enumerate(problem.airplanes):
            for name in (
                "forces_W",
                "forceCoefficients_W",
                "moments_W_CgP1",
                "momentCoefficients_W_CgP1",
            ):
                array = np.asarray(getattr(airplane, name))
                digest.update(f"{step}:{airplane_index}:{name}".encode("ascii"))
                digest.update(array.dtype.str.encode("ascii"))
                digest.update(repr(array.shape).encode("ascii"))
                digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _run_summary() -> dict[str, Any]:
    full = _run_solver(num_steps=3, feedback_enabled=True, transport_enabled=True)
    replay = _run_solver(num_steps=3, feedback_enabled=True, transport_enabled=True)
    transport_only = _run_solver(
        num_steps=3, feedback_enabled=False, transport_enabled=True
    )
    parent = _run_parent(3)
    full_one = _run_solver(num_steps=1, feedback_enabled=True, transport_enabled=True)
    feedback_only = _run_solver(
        num_steps=1, feedback_enabled=True, transport_enabled=False
    )

    reports = tuple(
        validate_synchronized_cloud_transport_report(report)
        for report in full.v5h5_transport_reports
    )
    replay_reports = tuple(
        validate_synchronized_cloud_transport_report(report)
        for report in replay.v5h5_transport_reports
    )
    layers = [
        {
            "ptera_step_index": report.ptera_feedback_report.ptera_step_index,
            "dvm_source_step_index": report.parent_source_step_index,
            "previous_particle_count": report.previous_particle_count,
            "new_particle_count": report.new_particle_count,
            "total_particle_count": report.total_particle_count,
            "node_birth_modes": [
                birth.mode for birth in report.prepared_cloud.node_births
            ],
            "no_penetration_max_abs": (
                report.ptera_feedback_report.no_penetration_max_abs
            ),
            "exact_append_prefix_max_abs": report.exact_append_prefix_max_abs,
            "frontier_stage_replay_max_abs": report.frontier_stage_replay_max_abs,
            "ptera_center_calls": report.ptera_center_call_count,
            "ptera_finite_difference_calls": (
                report.ptera_finite_difference_call_count
            ),
            "frontier_transport_parent_calls": (
                report.frontier_transport_parent_call_count
            ),
            "frontier_replay_parent_calls": report.frontier_replay_parent_call_count,
            "parent_state_unchanged": report.parent_state_unchanged,
            "feedback_write_count": report.feedback_write_count,
            "parent_write_count": report.parent_write_count,
            "surface_channel_write_count": report.load_write_count,
            "report_sha256": report.report_sha256,
        }
        for report in reports
        if report.ptera_feedback_report is not None
    ]
    deterministic = [item.report_sha256 for item in reports] == [
        item.report_sha256 for item in replay_reports
    ]
    transport_reduction = _native_state_sha256(transport_only) == _native_state_sha256(
        parent
    )
    feedback_reduction = _native_state_sha256(full_one) == _native_state_sha256(
        feedback_only
    )
    main_pass = (
        len(layers) == 3
        and [item["node_birth_modes"] for item in layers]
        == [["first"] * 3, ["continuous"] * 3, ["continuous"] * 3]
        and all(item["new_particle_count"] == 102 for item in layers)
        and [item["total_particle_count"] for item in layers] == [102, 204, 306]
        and all(item["no_penetration_max_abs"] <= 1.0e-12 for item in layers)
        and all(item["exact_append_prefix_max_abs"] == 0.0 for item in layers)
        and all(item["frontier_stage_replay_max_abs"] == 0.0 for item in layers)
        and all(item["parent_state_unchanged"] for item in layers)
        and all(
            item["feedback_write_count"]
            == item["parent_write_count"]
            == item["surface_channel_write_count"]
            == 0
            for item in layers
        )
    )
    passed = main_pass and deterministic and transport_reduction and feedback_reduction
    return {
        "schema_id": "fluxv-v5h5-synchronized-smoke-summary-v1",
        "status": (
            "go_three_layer_synchronized_mechanics_only"
            if passed
            else "stop_v5h5_mechanical_gate_failed"
        ),
        "passed": passed,
        "scope": {
            "geometry": "generic NACA0001 straight rectangular API wing",
            "source_family": "LEV only",
            "evaluation": "simulation_only_self_supervised_mechanical",
            "observation_access": "none",
            "target_case_branch": "none",
            "paper_scoring": False,
            "claim_limit": "no aerodynamic accuracy or general 3-D stability claim",
        },
        "configuration": {
            "delta_time_s": DT_S,
            "birth_sigma_m": SIGMA_BIRTH_M,
            "base_spacing_m": BASE_SPACING_M,
            "release_count": 3,
            "ptera_step_count": 3,
        },
        "layers": layers,
        "gates": {
            "three_layer_main_path_passed": main_pass,
            "deterministic_replay_passed": deterministic,
            "transport_only_native_ptera_exact_reduction_passed": (transport_reduction),
            "feedback_only_preserves_full_mode_ptera_state_passed": (
                feedback_reduction
            ),
            "target_access_count": 0,
            "paper_scoring_call_count": 0,
        },
        "state_sha256": {
            "full_three_layer": _native_state_sha256(full),
            "full_three_layer_replay": _native_state_sha256(replay),
            "transport_only": _native_state_sha256(transport_only),
            "native_parent": _native_state_sha256(parent),
            "full_one_layer": _native_state_sha256(full_one),
            "feedback_only_one_layer": _native_state_sha256(feedback_only),
        },
    }


def _source_paths(root: Path) -> tuple[Path, ...]:
    relative = (
        "platform/forward_flight_benchmarks/run_fluxv_v5h5_synchronized_smoke.py",
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
    """Run the gate and write a new, non-overwriting evidence bundle."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_dir}")
    output_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    summary = _run_summary()
    sources = {
        str(path.relative_to(root)): _file_sha256(path) for path in _source_paths(root)
    }
    (output_dir / "summary.json").write_text(_strict_json(summary), encoding="utf-8")
    (output_dir / "source_manifest.json").write_text(
        _strict_json({"schema_id": "source-sha256-v1", "files": sources}),
        encoding="utf-8",
    )
    readme = (
        "# FluxV v5h5 synchronized mechanical smoke\n\n"
        "This artifact covers a generic straight-wing, LEV-only, three-layer "
        "mechanical gate. It does not read target observations, score a paper "
        "case, or support an aerodynamic-accuracy claim.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    result_files = ("README.md", "source_manifest.json", "summary.json")
    result_manifest = {name: _file_sha256(output_dir / name) for name in result_files}
    (output_dir / "result_manifest.json").write_text(
        _strict_json({"schema_id": "result-sha256-v1", "files": result_manifest}),
        encoding="utf-8",
    )
    checksum_files = (*result_files, "result_manifest.json")
    checksum_lines = [
        f"{_file_sha256(output_dir / name)}  {name}" for name in checksum_files
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    summary = write_artifact(arguments.output_dir)
    print(_strict_json(summary), end="")
    return 0 if bool(summary["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
