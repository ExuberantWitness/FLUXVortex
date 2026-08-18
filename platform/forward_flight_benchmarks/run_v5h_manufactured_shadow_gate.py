"""Run bounded, observation-free v5h DVM ribbon shadow gates.

This runner exercises only a single, directly evaluated first-release source
layer.  It maps that layer to a node-owned ribbon, deposits conservative
Gaussian-erf diagnostic particles with the frozen overlap ratio, and advances
copies of the resulting cloud without adding another source layer.  It has no
Ptera parent, load channel, feedback path, target-case branch, or observation
input.

The run is deliberately not a repeated-release or aerodynamic-performance
test.  In particular, it cannot qualify the current mapper's continuous-node
frontier for material transport; that requires a separately audited advected
frontier handoff.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from math import cos, pi, radians, sin
from pathlib import Path
from typing import Any, Literal

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
from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
    INTERFACE_ID as RIBBON_INTERFACE_ID,
    DVMPlaneToGP1Map,
    DVMSpanCellSource,
    NodeOwnedDVMRibbonShadow,
    SpanNodeKinematics,
)
from forward_flight_benchmarks.v5h_dvm_source import (
    SOURCE_BACKEND_ID,
    SOURCE_INTERFACE_ID,
    DVMSourceEvent,
    V5hDVMSource,
)


GateGeometry = Literal["straight", "taper", "twist"]

RUN_SCHEMA_ID = "fluxv-v5h-single-release-mechanical-shadow-v2"
CASE_GEOMETRIES: tuple[GateGeometry, ...] = ("straight", "taper", "twist")
REFERENCE_SPEED_M_PER_S = 2.0
ROOT_CHORD_M = 0.20
SPAN_M = 0.60
PHYSICAL_RELEASE_DT_S = 0.002
PIVOT_FRACTION_CHORD = 0.25
ALPHA_RAD = radians(35.0)
LESP_CRITICAL = 0.18
REYNOLDS = 30_000.0
SOURCE_SETTINGS = LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48)
DEFAULT_SPAN_CELLS = 4
DEFAULT_SPAN_SUBDIVISIONS = (2, 4, 8, 16)
DEFAULT_TRANSPORT_SUBSTEPS = (1, 2, 4)
TRANSPORT_CLOUD_SPAN_SUBDIVISIONS = 4
FLOWVPM_COMMIT = "4f433fb09f6baad25db65c9905e0d9cbb09663ce"
FLOWUNSTEADY_COMMIT = "b7283db2e94a5f44a7ef2d57f223b0bdb8d0dec7"
FIXED_PROBES_GP1_M = np.asarray(
    (
        (0.05, 0.075, 0.35),
        (0.10, 0.225, 0.45),
        (-0.05, 0.375, 0.40),
        (0.15, 0.525, 0.50),
        (0.20, 0.300, 0.65),
    ),
    dtype=np.float64,
)

LIMITATIONS = (
    "single directly evaluated first LEV release only; diagnostic shadow; "
    "flat-plate generic non-target geometry; published Lcrit=0.18 used as a "
    "fixed source input; no repeated release, advected-frontier handoff, Ptera "
    "feedback, surface load, target observation, or performance claim; the "
    "particle smoothing radius is prescribed from the span mesh and is "
    "independent of the O(dt) newborn-edge length; fixed-sigma deposition is a "
    "FLOWUnsteady static-sheet-inspired development transfer, not a validated "
    "dynamic LEV wake-core law"
)


@dataclass(frozen=True)
class GateConfig:
    """Frozen numerical controls for the bounded mechanical gate."""

    span_cells: int = DEFAULT_SPAN_CELLS
    span_subdivisions: tuple[int, ...] = DEFAULT_SPAN_SUBDIVISIONS
    transport_substeps: tuple[int, ...] = DEFAULT_TRANSPORT_SUBSTEPS


class _ExplodingInput:
    """Sentinel proving that exact-off paths do not inspect their inputs."""

    def __iter__(self) -> Any:
        raise AssertionError("disabled mapper iterated an input")

    def __float__(self) -> float:
        raise AssertionError("disabled mapper converted a numeric input")


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
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
    if TRANSPORT_CLOUD_SPAN_SUBDIVISIONS not in config.span_subdivisions:
        raise ValueError(
            "span_subdivisions must include the frozen transport-cloud level "
            f"{TRANSPORT_CLOUD_SPAN_SUBDIVISIONS}"
        )


def _chord_and_twist(geometry: GateGeometry, eta: float) -> tuple[float, float]:
    if geometry == "straight":
        return ROOT_CHORD_M, 0.0
    if geometry == "taper":
        return ROOT_CHORD_M * (1.0 - 0.4 * eta), 0.0
    if geometry == "twist":
        return ROOT_CHORD_M, radians(20.0) * eta
    raise ValueError(f"unsupported gate geometry {geometry!r}")


def _chord_direction(twist_rad: float) -> np.ndarray:
    return np.asarray((cos(twist_rad), 0.0, -sin(twist_rad)), dtype=np.float64)


def _node_geometry(
    geometry: GateGeometry,
    span_cells: int,
) -> tuple[
    tuple[SpanNodeKinematics, ...],
    tuple[np.ndarray, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    anchors: list[np.ndarray] = []
    chords: list[float] = []
    twists: list[float] = []
    nodes: list[SpanNodeKinematics] = []
    for node_index in range(span_cells + 1):
        eta = node_index / span_cells
        chord, twist = _chord_and_twist(geometry, eta)
        pivot = np.asarray((0.0, SPAN_M * eta, 0.0), dtype=np.float64)
        anchor = pivot - PIVOT_FRACTION_CHORD * chord * _chord_direction(twist)
        velocity = np.asarray(
            (0.30 + 0.05 * eta, 0.0, -0.04 * (1.0 + eta)),
            dtype=np.float64,
        )
        anchors.append(anchor)
        chords.append(chord)
        twists.append(twist)
        nodes.append(
            SpanNodeKinematics(
                node_id=f"{geometry}:node:{node_index}",
                anchor_position_gp1_m=anchor,
                edge_velocity_gp1_m_per_s=velocity,
            )
        )
    return tuple(nodes), tuple(anchors), tuple(chords), tuple(twists)


def _cell_plane_map(
    geometry: GateGeometry,
    cell_index: int,
    left_anchor: np.ndarray,
    right_anchor: np.ndarray,
    midpoint_twist: float,
) -> DVMPlaneToGP1Map:
    span_delta = right_anchor - left_anchor
    span_axis = span_delta / np.linalg.norm(span_delta)
    chord_guess = _chord_direction(midpoint_twist)
    chord_projection = chord_guess - np.dot(chord_guess, span_axis) * span_axis
    x_axis = chord_projection / np.linalg.norm(chord_projection)
    z_axis = np.cross(span_axis, x_axis)
    z_axis /= np.linalg.norm(z_axis)
    return DVMPlaneToGP1Map(
        origin_gp1_m=0.5 * (left_anchor + right_anchor),
        x_axis_gp1=x_axis,
        z_axis_gp1=z_axis,
        positive_circulation_axis_gp1=span_axis,
        circulation_to_ring_traversal_sign=1,
        provenance=(
            f"generic non-target {geometry} cell {cell_index}; explicit "
            "orthonormal DVM x-z to GP1 map"
        ),
    )


def _direct_first_event(
    *,
    geometry: GateGeometry,
    cell_index: int,
    chord_m: float,
) -> DVMSourceEvent:
    convective_dt = PHYSICAL_RELEASE_DT_S * REFERENCE_SPEED_M_PER_S / chord_m
    source = V5hDVMSource(
        physical_section_id=f"manufactured:{geometry}:section:{cell_index}",
        physical_strip_id=f"manufactured:{geometry}:strip:{cell_index}",
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
        settings=SOURCE_SETTINGS,
    )
    event = source.step(ALPHA_RAD, 0.0, 0.0)
    _attest_event(event)
    return event


def _attest_event(event: DVMSourceEvent) -> None:
    checks = {
        "enabled": event.enabled,
        "evaluated_source_only": event.status.startswith("evaluated_source_only_"),
        "first_source_step": event.lineage.source_step_index == 1,
        "first_lev_release": event.lesp_active and event.lev_birth_mode == "first",
        "not_restart": not event.restart,
        "source_interface": event.provenance.interface_id == SOURCE_INTERFACE_ID,
        "source_backend": event.provenance.backend_id == SOURCE_BACKEND_ID,
        "source_parity": event.provenance.source_parity,
        "noncanonical": not event.provenance.canonical,
        "no_observation": event.provenance.observation_access == "none",
        "no_target_branch": event.provenance.target_case_branch == "none",
        "source_only_owner": "source quantities only"
        in event.provenance.ownership_scope,
        "kelvin_ledger": event.kelvin_ledger is not None,
        "finite_lev": np.isfinite(event.gamma_lev_new_over_u_c),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"DVM source attestation failed: {failed}")


def _build_first_release_case(
    geometry: GateGeometry,
    span_cells: int,
) -> tuple[Any, tuple[DVMSourceEvent, ...]]:
    nodes, anchors, node_chords, node_twists = _node_geometry(geometry, span_cells)
    events: list[DVMSourceEvent] = []
    cells: list[DVMSpanCellSource] = []
    for cell_index in range(span_cells):
        chord = 0.5 * (node_chords[cell_index] + node_chords[cell_index + 1])
        event = _direct_first_event(
            geometry=geometry,
            cell_index=cell_index,
            chord_m=chord,
        )
        events.append(event)
        cells.append(
            DVMSpanCellSource(
                cell_id=f"{geometry}:cell:{cell_index}",
                left_node_id=f"{geometry}:node:{cell_index}",
                right_node_id=f"{geometry}:node:{cell_index + 1}",
                event=event,
                plane_to_gp1=_cell_plane_map(
                    geometry,
                    cell_index,
                    anchors[cell_index],
                    anchors[cell_index + 1],
                    0.5 * (node_twists[cell_index] + node_twists[cell_index + 1]),
                ),
            )
        )
    mapper = NodeOwnedDVMRibbonShadow(
        wing_id=f"manufactured:{geometry}:wing",
        source_family="lev",
    )
    result = mapper.map_step(
        tuple(cells),
        nodes,
        delta_time_s=PHYSICAL_RELEASE_DT_S,
    )
    if result.edge_graph is None:
        raise RuntimeError("direct first-release source produced no ribbon graph")
    if any(record.mode != "first" for record in result.node_births):
        raise RuntimeError("single release unexpectedly left first-node mode")
    return result, tuple(events)


def _bridge_metrics(
    bridge: ShadowBridgeResult,
    *,
    span_subdivisions: int,
    target_max_spacing_m: float,
    smoothing_radius_m: float,
    reference_probe_velocity_m_per_s: np.ndarray,
) -> dict[str, Any]:
    diagnostics = bridge.diagnostics
    finite = bool(
        np.all(np.isfinite(bridge.positions))
        and np.all(np.isfinite(bridge.gamma))
        and np.all(np.isfinite(bridge.sigma))
        and np.all(bridge.sigma > 0.0)
    )
    lambda_values: list[float] = []
    if bridge.edge_graph is None:
        raise RuntimeError("enabled deposition lost its edge graph")
    edge_by_key = {edge.key: edge for edge in bridge.edge_graph.retained_edges}
    for index, lineage in enumerate(bridge.lineage):
        edge = edge_by_key[lineage.source_edge]
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        h = float(np.linalg.norm(end - start) / lineage.subdivision_count)
        lambda_values.append(float(bridge.sigma[index] / h))
    overlap_lower_bound = bool(
        lambda_values and min(lambda_values) >= FROZEN_OVERLAP_LAMBDA * (1.0 - 1.0e-14)
    )
    fixed_sigma = bool(bridge.sigma.size and np.all(bridge.sigma == smoothing_radius_m))
    probe_velocity = direct_gaussian_erf_velocity_jacobian(
        bridge.positions,
        bridge.gamma,
        bridge.sigma,
        target_positions=FIXED_PROBES_GP1_M,
    ).velocity
    reference_norm = float(np.linalg.norm(reference_probe_velocity_m_per_s))
    if not np.isfinite(reference_norm) or reference_norm <= 0.0:
        raise FloatingPointError("analytic probe reference has nonpositive norm")
    probe_relative_l2 = float(
        np.linalg.norm(probe_velocity - reference_probe_velocity_m_per_s)
        / reference_norm
    )
    passed = bool(
        finite
        and overlap_lower_bound
        and fixed_sigma
        and bridge.feedback_velocity is None
        and diagnostics.feedback_call_count == 0
        and diagnostics.clip_count == 0
        and diagnostics.nonfinite_count == 0
        and diagnostics.owner_conflict_count == 0
        and diagnostics.incidence_residual <= 1.0e-12
        and diagnostics.edge_reconstruction_residual <= 1.0e-12
        and diagnostics.max_edge_conservation_abs <= 1.0e-14
        and diagnostics.max_edge_conservation_rel <= 1.0e-12
        and diagnostics.global_conservation_abs <= 1.0e-14
        and diagnostics.global_conservation_rel <= 1.0e-12
    )
    return {
        "span_subdivisions": span_subdivisions,
        "target_max_spacing_m": target_max_spacing_m,
        "smoothing_radius_m": smoothing_radius_m,
        "particle_count": diagnostics.particle_count,
        "source_edge_count": diagnostics.source_edge_count,
        "retained_edge_count": diagnostics.retained_edge_count,
        "incidence_residual": diagnostics.incidence_residual,
        "edge_reconstruction_residual": diagnostics.edge_reconstruction_residual,
        "max_edge_conservation_abs": diagnostics.max_edge_conservation_abs,
        "max_edge_conservation_rel": diagnostics.max_edge_conservation_rel,
        "global_conservation_abs": diagnostics.global_conservation_abs,
        "global_conservation_rel": diagnostics.global_conservation_rel,
        "sigma_min_m": float(np.min(bridge.sigma)),
        "sigma_max_m": float(np.max(bridge.sigma)),
        "realized_overlap_min": min(lambda_values),
        "realized_overlap_max": max(lambda_values),
        "overlap_lower_bound_2p125": overlap_lower_bound,
        "fixed_sigma_across_edges": fixed_sigma,
        "analytic_finite_segment_probe_relative_l2": probe_relative_l2,
        "finite": finite,
        "clip_count": diagnostics.clip_count,
        "nonfinite_count": diagnostics.nonfinite_count,
        "owner_conflict_count": diagnostics.owner_conflict_count,
        "feedback_call_count": diagnostics.feedback_call_count,
        "passed": passed,
    }


def _finite_segment_velocity(
    start: np.ndarray,
    end: np.ndarray,
    circulation: float,
    probes: np.ndarray,
) -> np.ndarray:
    """Independent singular finite-segment Biot--Savart reference field."""

    r1 = probes - start
    r2 = probes - end
    segment = end - start
    cross = np.cross(r1, r2)
    cross_norm_squared = np.einsum("ij,ij->i", cross, cross)
    if np.any(cross_norm_squared == 0.0):
        raise ValueError("a fixed probe is collinear with a source edge")
    endpoint_factor = np.einsum(
        "j,ij->i",
        segment,
        r1 / np.linalg.norm(r1, axis=1)[:, None]
        - r2 / np.linalg.norm(r2, axis=1)[:, None],
    )
    return (
        circulation
        / (4.0 * pi)
        * cross
        * (endpoint_factor / cross_norm_squared)[:, None]
    )


def _analytic_graph_probe_velocity(graph: Any) -> np.ndarray:
    velocity = np.zeros_like(FIXED_PROBES_GP1_M)
    for edge in graph.retained_edges:
        velocity += _finite_segment_velocity(
            np.asarray(edge.start_position, dtype=np.float64),
            np.asarray(edge.end_position, dtype=np.float64),
            float(edge.circulation),
            FIXED_PROBES_GP1_M,
        )
    if not np.all(np.isfinite(velocity)):
        raise FloatingPointError("analytic graph probe field is non-finite")
    return velocity


def _transport_metrics(
    bridge: ShadowBridgeResult,
    transport_substeps: int,
) -> dict[str, Any]:
    state = make_particle_state(bridge.positions, bridge.gamma, bridge.sigma)
    initial_invariant = np.linalg.norm(state.gamma, axis=1) * state.sigma**2
    stage_count = 0
    max_stage_dt_velocity_over_sigma = 0.0
    max_stage_dt_self_velocity_over_sigma = 0.0
    max_stage_dt_jacobian_frobenius = 0.0
    transport_dt = PHYSICAL_RELEASE_DT_S / transport_substeps
    freestream = np.asarray((REFERENCE_SPEED_M_PER_S, 0.0, 0.0), dtype=np.float64)
    for _ in range(transport_substeps):
        state, stages = lsrk3_step_direct(
            state,
            transport_dt,
            freestream_velocity=freestream,
        )
        stage_count += len(stages)
        for stage in stages:
            self_speed = np.linalg.norm(stage.rhs.velocity, axis=1)
            max_stage_dt_self_velocity_over_sigma = max(
                max_stage_dt_self_velocity_over_sigma,
                float(
                    np.max(
                        transport_dt * self_speed / stage.pre.sigma,
                        initial=0.0,
                    )
                ),
            )
            speed = np.linalg.norm(stage.rhs.velocity + freestream[None, :], axis=1)
            max_stage_dt_velocity_over_sigma = max(
                max_stage_dt_velocity_over_sigma,
                float(np.max(transport_dt * speed / stage.pre.sigma, initial=0.0)),
            )
            jacobian_norm = np.linalg.norm(
                stage.rhs.jacobian.reshape(stage.rhs.jacobian.shape[0], -1),
                axis=1,
            )
            max_stage_dt_jacobian_frobenius = max(
                max_stage_dt_jacobian_frobenius,
                float(np.max(transport_dt * jacobian_norm, initial=0.0)),
            )
    final_invariant = np.linalg.norm(state.gamma, axis=1) * state.sigma**2
    nonzero = initial_invariant > 0.0
    invariant_relative_max = float(
        np.max(
            np.abs(final_invariant[nonzero] - initial_invariant[nonzero])
            / initial_invariant[nonzero],
            initial=0.0,
        )
    )
    finite = bool(
        np.all(np.isfinite(state.positions))
        and np.all(np.isfinite(state.gamma))
        and np.all(np.isfinite(state.sigma))
        and np.all(state.sigma > 0.0)
    )
    return {
        "transport_substeps": transport_substeps,
        "transport_dt_s": transport_dt,
        "stage_count": stage_count,
        "particle_count": int(state.positions.shape[0]),
        "finite": finite,
        "sigma_min_m": float(np.min(state.sigma)),
        "sigma_max_m": float(np.max(state.sigma)),
        "rvpm_gamma_sigma2_relative_drift_max": invariant_relative_max,
        "max_stage_dt_self_velocity_over_sigma": (
            max_stage_dt_self_velocity_over_sigma
        ),
        "max_stage_dt_velocity_over_sigma": max_stage_dt_velocity_over_sigma,
        "max_stage_dt_jacobian_frobenius": max_stage_dt_jacobian_frobenius,
        "position_l2_m": float(np.linalg.norm(state.positions)),
        "gamma_l2_m3_per_s": float(np.linalg.norm(state.gamma)),
        "passed": bool(finite and invariant_relative_max <= 1.0e-6),
    }


def _disabled_gate() -> dict[str, Any]:
    mapper = NodeOwnedDVMRibbonShadow(
        wing_id="manufactured:disabled:wing",
        source_family="lev",
    )
    before = mapper.state_snapshot
    exploding = _ExplodingInput()
    result = mapper.map_step(
        exploding,  # type: ignore[arg-type]
        exploding,  # type: ignore[arg-type]
        delta_time_s=exploding,
        enabled=False,
    )
    passed = bool(
        result.edge_graph is None
        and result.feedback_velocity is None
        and result.diagnostics.feedback_call_count == 0
        and result.diagnostics.transport_advance_count == 0
        and mapper.state_snapshot == before
    )
    return {
        "input_blind": passed,
        "feedback_call_count": result.diagnostics.feedback_call_count,
        "transport_advance_count": result.diagnostics.transport_advance_count,
        "state_unchanged": mapper.state_snapshot == before,
        "passed": passed,
    }


def run_mechanical_shadow_gate(config: GateConfig = GateConfig()) -> dict[str, Any]:
    """Run S0/S1/S3 in memory and return a JSON-compatible result."""

    _validate_config(config)
    source_path = Path(__file__).with_name("v5h_dvm_source.py")
    ribbon_path = Path(__file__).with_name("v5h_dvm_node_ribbon.py")
    runner_path = Path(__file__)
    repo_root = Path(__file__).resolve().parents[2]
    bridge_path = repo_root / "src/fluxvortex/rvpm_edge_bridge.py"
    reference_path = repo_root / "src/fluxvortex/rvpm_reference.py"
    transport_path = repo_root / "src/fluxvortex/rvpm_transport.py"
    ldvm_path = repo_root / "platform/ldvm_fourier.py"
    correction_path = (
        repo_root / "platform/forward_flight_benchmarks/ldvm_uvlm_correction.py"
    )
    cases: list[dict[str, Any]] = []
    for geometry in CASE_GEOMETRIES:
        ribbon, events = _build_first_release_case(geometry, config.span_cells)
        graph = ribbon.edge_graph
        assert graph is not None
        reference_probe_velocity = _analytic_graph_probe_velocity(graph)
        subdivision_rows: list[dict[str, Any]] = []
        transport_bridge: ShadowBridgeResult | None = None
        span_cell_width = SPAN_M / config.span_cells
        for span_subdivisions in config.span_subdivisions:
            target_max_spacing = span_cell_width / span_subdivisions
            smoothing_radius = FROZEN_OVERLAP_LAMBDA * target_max_spacing
            bridge = deposit_edge_graph_fixed_sigma(
                graph,
                smoothing_radius=smoothing_radius,
                step=1,
                physical_owner=RING_PHYSICAL_OWNER,
                owner_state=DIAGNOSTIC_SHADOW_OWNER,
                overlap_lambda=FROZEN_OVERLAP_LAMBDA,
            )
            subdivision_rows.append(
                _bridge_metrics(
                    bridge,
                    span_subdivisions=span_subdivisions,
                    target_max_spacing_m=target_max_spacing,
                    smoothing_radius_m=smoothing_radius,
                    reference_probe_velocity_m_per_s=reference_probe_velocity,
                )
            )
            if span_subdivisions == TRANSPORT_CLOUD_SPAN_SUBDIVISIONS:
                transport_bridge = bridge
        assert transport_bridge is not None
        transport_rows = [
            _transport_metrics(transport_bridge, substeps)
            for substeps in config.transport_substeps
        ]
        source_kelvin = [
            abs(float(event.kelvin_residual_over_u_c))
            * float(event.provenance.circulation_scale_u_times_c_m2_per_s)
            for event in events
        ]
        ribbon_diag = ribbon.diagnostics
        probe_errors = [
            row["analytic_finite_segment_probe_relative_l2"] for row in subdivision_rows
        ]
        spatial_probe_convergence = bool(
            probe_errors[-1] <= 0.01
            and all(
                coarse / fine >= 1.5
                for coarse, fine in zip(probe_errors[:-1], probe_errors[1:])
                if fine > 1.0e-14
            )
        )
        mapping_passed = bool(
            all(row["passed"] for row in subdivision_rows)
            and spatial_probe_convergence
            and max(source_kelvin, default=0.0) <= 1.0e-10
            and ribbon.feedback_velocity is None
            and ribbon_diag.feedback_call_count == 0
            and ribbon_diag.transport_advance_count == 0
            and ribbon_diag.seam_count == 0
            and ribbon_diag.nonfinite_count == 0
            and ribbon_diag.source_reuse_count == 0
            and ribbon_diag.incidence_residual <= 1.0e-12
            and ribbon_diag.edge_reconstruction_residual <= 1.0e-12
        )
        transport_passed = bool(all(row["passed"] for row in transport_rows))
        case_passed = bool(mapping_passed and transport_passed)
        cases.append(
            {
                "case_id": f"generic_non_target_{geometry}",
                "geometry": geometry,
                "source_event_count": len(events),
                "source_step_indices": [
                    event.lineage.source_step_index for event in events
                ],
                "source_statuses": [event.status for event in events],
                "all_direct_first_lev_release": all(
                    event.lesp_active and event.lev_birth_mode == "first"
                    for event in events
                ),
                "geometry_hashes": [
                    event.provenance.geometry_hash_sha256 for event in events
                ],
                "max_kelvin_residual_m2_per_s": max(source_kelvin, default=0.0),
                "ribbon": {
                    "interface_id": ribbon_diag.interface_id,
                    "active_cell_count": ribbon_diag.active_cell_count,
                    "shared_node_count": ribbon_diag.shared_node_count,
                    "first_node_count": ribbon_diag.first_node_count,
                    "continuous_node_count": ribbon_diag.continuous_node_count,
                    "restart_node_count": ribbon_diag.restart_node_count,
                    "incidence_residual": ribbon_diag.incidence_residual,
                    "edge_reconstruction_residual": (
                        ribbon_diag.edge_reconstruction_residual
                    ),
                    "seam_count": ribbon_diag.seam_count,
                    "nonfinite_count": ribbon_diag.nonfinite_count,
                    "source_reuse_count": ribbon_diag.source_reuse_count,
                    "feedback_call_count": ribbon_diag.feedback_call_count,
                    "transport_advance_count": ribbon_diag.transport_advance_count,
                },
                "subdivision_refinement": subdivision_rows,
                "analytic_probe_positions_gp1_m": FIXED_PROBES_GP1_M.tolist(),
                "spatial_probe_convergence_passed": spatial_probe_convergence,
                "single_cloud_transport_substeps": transport_rows,
                "parent_write_count": 0,
                "load_write_count": 0,
                "feedback_call_count": 0,
                "mapping_passed": mapping_passed,
                "single_cloud_transport_passed": transport_passed,
                "passed": case_passed,
            }
        )
    disabled = _disabled_gate()
    s1_passed = bool(all(case["mapping_passed"] for case in cases))
    s3_passed = bool(all(case["single_cloud_transport_passed"] for case in cases))
    passed = bool(disabled["passed"] and s1_passed and s3_passed)
    return {
        "schema_id": RUN_SCHEMA_ID,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scope": "non_target_single_release_mechanical_shadow_only",
        "observation_access": "none",
        "target_case_branch": "none",
        "source_kind": "direct_V5hDVMSource_event",
        "source_interface_id": SOURCE_INTERFACE_ID,
        "source_backend_id": SOURCE_BACKEND_ID,
        "ribbon_interface_id": RIBBON_INTERFACE_ID,
        "source_canonical_eligible": False,
        "frozen_overlap_lambda": FROZEN_OVERLAP_LAMBDA,
        "spatial_deposition_contract": {
            "status": "development_transfer_not_canonical",
            "flowvpm_commit": FLOWVPM_COMMIT,
            "flowunsteady_commit": FLOWUNSTEADY_COMMIT,
            "source_location": (
                "FLOWUnsteady src/FLOWUnsteady_vehicle_vlm_unsteady.jl "
                "_static_particles: sigma is fixed before "
                "np=ceil(edge_length/(sigma/overlap))"
            ),
            "overlap_role": (
                "minimum particles-per-smoothing-radius bound for the companion "
                "static VLM vortex-sheet discretization; not a universal rVPM "
                "wake-core law"
            ),
            "project_spatial_scale": (
                "target spacing = nominal span-cell width / span_subdivisions; "
                "sigma = 2.125 * target spacing"
            ),
            "release_clock_decoupled": True,
            "observation_fit": "none",
        },
        "transport_stability_contract": {
            "primary_explicit_integrator_indicator": "max(dt * ||Jacobian||_F)",
            "self_induced_advection_indicator": "max(dt * |u_self| / sigma)",
            "total_advection_indicator": (
                "max(dt * |u_self + U_inf| / sigma); diagnostic only because "
                "uniform U_inf is removable by a Galilean transformation"
            ),
            "clip_or_limiter": "none",
        },
        "physical_release_dt_s": PHYSICAL_RELEASE_DT_S,
        "reference_speed_m_per_s": REFERENCE_SPEED_M_PER_S,
        "alpha_deg": 35.0,
        "lesp_critical": LESP_CRITICAL,
        "config": {
            "span_cells": config.span_cells,
            "span_subdivisions": list(config.span_subdivisions),
            "transport_substeps": list(config.transport_substeps),
        },
        "disabled_gate": disabled,
        "gate_summary": {
            "s0_exact_off_and_attestation_passed": disabled["passed"],
            "s1_single_release_ribbon_and_deposition_passed": s1_passed,
            "s3_single_cloud_transport_passed": s3_passed,
            "stop_required": not passed,
        },
        "cases": cases,
        "parent_write_count": 0,
        "load_write_count": 0,
        "feedback_call_count": 0,
        "limitations": LIMITATIONS,
        "blocked": {
            "repeated_release_transport": (
                "requires an audited advected NodeFrontierFact handoff"
            ),
            "target_scoring": "prohibited by this mechanical gate",
            "subdivision_claim": (
                "fixed-sigma FLOWUnsteady-style deposition uses a span-mesh "
                "spacing bound; a full transported spatial-convergence claim "
                "still requires the repeated-release frontier contract"
            ),
        },
        "code_sha256": {
            "runner": _sha256_file(runner_path),
            "dvm_source": _sha256_file(source_path),
            "dvm_node_ribbon": _sha256_file(ribbon_path),
            "ldvm_fourier": _sha256_file(ldvm_path),
            "ldvm_uvlm_correction": _sha256_file(correction_path),
            "rvpm_edge_bridge": _sha256_file(bridge_path),
            "rvpm_reference": _sha256_file(reference_path),
            "rvpm_transport": _sha256_file(transport_path),
        },
    }


def write_smoke_artifact(result: dict[str, Any], output: Path) -> Path:
    """Write one JSON smoke artifact, refusing to overwrite an existing path."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new JSON path; use /tmp for the bounded smoke run",
    )
    args = parser.parse_args()
    result = run_mechanical_shadow_gate()
    path = write_smoke_artifact(result, args.output)
    print(json.dumps({"artifact": str(path), "status": result["status"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
