"""Superseded S3ai-v1 reachable-pressure obstruction protocol draft.

The implementation audit rejected the v1 protocol before formal execution.
There is no command-line entry point and no result artifact.  The public
runner fails closed so the frozen-but-superseded protocol cannot accidentally
produce a physical classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_boundary_3d_cut_wake_junction_guard import (  # noqa: E402
    build_canonical_diamond_wing,
)
from claim_runtime.actual_wake_reachable_pressure import (  # noqa: E402
    DirectIndependentStageObservation,
    centered_tangent,
    dual_mass_norm,
    observe_direct_independent_stage,
    weak_pressure_step_residual,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.material_wake_time_march import (  # noqa: E402
    ExplicitMidpointWakeMarch,
    march_actual_boundary_material_wake_explicit_midpoint,
)


CASES = HERE / "docs" / "diag" / "actual_wake_reachable_pressure_obstruction_cases_20260728_125228.yaml"
AUDIT = HERE / "docs" / "diag" / "research_n3_actual_wake_reachable_pressure_obstruction_audit_20260728_130355.md"


class ReachablePressureGuardError(RuntimeError):
    """The frozen S3ai contract or its read-only observation is invalid."""


@dataclass(frozen=True)
class MarchPressureObservation:
    """Read-only window observation and all representation diagnostics."""

    residual: np.ndarray
    mass_active: np.ndarray
    direct_bie_relative_residual: float
    compatibility_abs_residual: float
    zero_tip_abs_residual: float
    direct_w_factorization_abs_residual: float
    direct_w_rank_deficiency: int
    old_state_mutation_abs_residual: float
    history_time_gap_abs_residual: float
    history_geometry_gap_abs_residual: float
    history_trace_jump_abs_residual: float
    midpoint_identity_abs_residual: float
    current_attachment_abs_residual: float
    kelvin_ledger_abs_residual: float


def _load_frozen_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("artifact") != (
        "actual_wake_reachable_pressure_obstruction_preregistration"
    ):
        raise ReachablePressureGuardError("S3ai frozen preregistration is invalid")
    if contract.get("stage") != "S3ai_reachable_material_history_pressure_obstruction":
        raise ReachablePressureGuardError("unexpected S3ai stage")
    if contract.get("status") != "preregistered_before_any_formal_execution":
        raise ReachablePressureGuardError("S3ai contract is no longer preregistered")
    if contract.get("decision", {}).get("production_activation_allowed") is not False:
        raise ReachablePressureGuardError("S3ai must not activate production")
    return contract


def _maximum_abs(value: Any) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=float)), initial=0.0))


def _cauchy(values: list[np.ndarray], mass_active: np.ndarray) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ReachablePressureGuardError("S3ai Cauchy families require exactly three levels")
    coarse_medium = dual_mass_norm(values[0] - values[1], mass_active)
    medium_fine = dual_mass_norm(values[1] - values[2], mass_active)
    ratio = coarse_medium / max(medium_fine, np.finfo(float).tiny)
    return coarse_medium, medium_fine, ratio


def _incident_history(time: float, *, epsilon: float, start: float, end: float, face_count: int) -> np.ndarray:
    alpha = epsilon * np.sin(np.pi * (time - start) / (end - start))
    vector = np.array((np.cos(alpha), 0.0, np.sin(alpha)))
    return np.repeat(vector[None, :], face_count, axis=0)


def observe_material_march(
    march: ExplicitMidpointWakeMarch,
    *,
    upper_face_indices: np.ndarray,
    lower_face_indices: np.ndarray,
    direct_w_quadrature_order: int,
    pressure_line_quadrature_order: int,
) -> MarchPressureObservation:
    """Evaluate S3ai only on the S3e half/full coupled material stages."""
    if not march.steps:
        raise ReachablePressureGuardError("S3e march contains no stages")
    window: np.ndarray | None = None
    mass: np.ndarray | None = None
    diagnostics = {
        "direct_bie_relative_residual": 0.0,
        "compatibility_abs_residual": 0.0,
        "zero_tip_abs_residual": 0.0,
        "direct_w_factorization_abs_residual": 0.0,
        "direct_w_rank_deficiency": 0,
        "old_state_mutation_abs_residual": 0.0,
        "history_time_gap_abs_residual": 0.0,
        "history_geometry_gap_abs_residual": 0.0,
        "history_trace_jump_abs_residual": 0.0,
        "midpoint_identity_abs_residual": 0.0,
        "current_attachment_abs_residual": 0.0,
        "kelvin_ledger_abs_residual": 0.0,
    }
    cumulative_release = np.zeros_like(march.steps[0].body_jump_previous)
    for step in march.steps:
        # Both midpoint and full stages independently assemble W_direct.  No
        # right inverse is called in this physical observation path.
        midpoint = observe_direct_independent_stage(
            step.half_stage,
            upper_face_indices=upper_face_indices,
            lower_face_indices=lower_face_indices,
            direct_w_quadrature_order=direct_w_quadrature_order,
            pressure_line_quadrature_order=pressure_line_quadrature_order,
        )
        current = observe_direct_independent_stage(
            step.full_stage,
            upper_face_indices=upper_face_indices,
            lower_face_indices=lower_face_indices,
            direct_w_quadrature_order=direct_w_quadrature_order,
            pressure_line_quadrature_order=pressure_line_quadrature_order,
        )
        if mass is None:
            mass = midpoint.active_mass
        if not np.array_equal(mass, midpoint.active_mass) or not np.array_equal(mass, current.active_mass):
            raise ReachablePressureGuardError("active P2 mass changed along frozen path")
        step_residual = weak_pressure_step_residual(
            mass,
            step.body_jump_previous[midpoint.direct_assembly.active_row_indices],
            step.body_jump_current[current.direct_assembly.active_row_indices],
            step.time_current - step.time_previous,
            midpoint.weak_pressure,
        )
        window = step_residual if window is None else window + step_residual
        for observation in (midpoint, current):
            diagnostics["direct_bie_relative_residual"] = max(diagnostics["direct_bie_relative_residual"], observation.direct_full_bie_relative_residual)
            diagnostics["compatibility_abs_residual"] = max(diagnostics["compatibility_abs_residual"], observation.compatibility_abs_residual)
            diagnostics["zero_tip_abs_residual"] = max(diagnostics["zero_tip_abs_residual"], observation.zero_tip_abs_residual)
            diagnostics["direct_w_factorization_abs_residual"] = max(diagnostics["direct_w_factorization_abs_residual"], observation.direct_w_factorization_abs_residual)
            diagnostics["direct_w_rank_deficiency"] = max(diagnostics["direct_w_rank_deficiency"], observation.direct_w_rank_deficiency)
        history = step.history_after.continuity_report()
        diagnostics["old_state_mutation_abs_residual"] = max(diagnostics["old_state_mutation_abs_residual"], step.old_strength_mutation)
        diagnostics["history_time_gap_abs_residual"] = max(diagnostics["history_time_gap_abs_residual"], history.max_time_gap)
        diagnostics["history_geometry_gap_abs_residual"] = max(
            diagnostics["history_geometry_gap_abs_residual"],
            history.max_geometry_gap,
            step.old_geometry_convection_error,
        )
        diagnostics["history_trace_jump_abs_residual"] = max(diagnostics["history_trace_jump_abs_residual"], history.max_trace_jump)
        diagnostics["midpoint_identity_abs_residual"] = max(diagnostics["midpoint_identity_abs_residual"], step.midpoint_row_identity_error)
        diagnostics["current_attachment_abs_residual"] = max(diagnostics["current_attachment_abs_residual"], step.current_attachment_error)
        # Gamma_bound=-mu and birth sign +1: retain the typed full P2 trace.
        cumulative_release += (
            step.body_jump_midpoint - step.body_jump_previous
            + step.body_jump_current - step.body_jump_midpoint
        )
        diagnostics["kelvin_ledger_abs_residual"] = max(
            diagnostics["kelvin_ledger_abs_residual"],
            _maximum_abs(
                -step.body_jump_current
                + cumulative_release
                + march.steps[0].body_jump_previous
            ),
        )
    if window is None or mass is None:  # pragma: no cover - protected above
        raise ReachablePressureGuardError("S3ai observation did not produce a window")
    return MarchPressureObservation(residual=window, mass_active=mass, **diagnostics)


def _gates(observation: MarchPressureObservation, thresholds: dict[str, Any]) -> dict[str, bool]:
    return {
        "direct_full_bie": observation.direct_bie_relative_residual <= float(thresholds["direct_full_bie_relative_residual_max"]),
        "compatibility": observation.compatibility_abs_residual <= float(thresholds["compatibility_abs_max"]),
        "zero_tip": observation.zero_tip_abs_residual <= float(thresholds["zero_tip_abs_max"]),
        "direct_w": observation.direct_w_factorization_abs_residual <= float(thresholds["direct_w_factorization_abs_max"]) and observation.direct_w_rank_deficiency <= int(thresholds["direct_w_rank_deficiency_max"]),
        "old_state": observation.old_state_mutation_abs_residual <= float(thresholds["old_state_mutation_abs_max"]),
        "history": observation.history_time_gap_abs_residual <= float(thresholds["history_time_gap_abs_max"]) and observation.history_geometry_gap_abs_residual <= float(thresholds["history_geometry_gap_abs_max"]) and observation.history_trace_jump_abs_residual <= float(thresholds["history_trace_jump_abs_max"]),
        "midpoint_attachment": observation.midpoint_identity_abs_residual <= float(thresholds["midpoint_identity_abs_max"]) and observation.current_attachment_abs_residual <= float(thresholds["current_attachment_abs_max"]),
        "kelvin": observation.kelvin_ledger_abs_residual <= float(thresholds["kelvin_ledger_abs_max"]),
    }


def run_preregistered_observation() -> dict[str, Any]:
    """Fail closed because the frozen S3ai-v1 protocol was superseded.

    No formal execution occurred before the audit.  A v2 protocol must be
    frozen and independently audited rather than repairing v1 after seeing
    numerical outputs.
    """
    raise ReachablePressureGuardError(
        "S3ai-v1 was aborted before formal execution; see "
        f"{AUDIT.name}. Freeze and audit v2 before running a physical gate."
    )


def _superseded_v1_draft() -> dict[str, Any]:
    """Retain the unexecuted implementation draft for v2 code review only."""
    contract = _load_frozen_contract()
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    mesh, upper, lower, cut_edges, endpoints = build_canonical_diamond_wing()
    topology = classified_p2_cut_topology(mesh, upper_face_indices=upper, lower_face_indices=lower, cut_edges=cut_edges, zero_jump_end_vertices=endpoints)
    start = float(canonical["physical_time"]["start"])
    end = float(canonical["physical_time"]["end"])
    original_vertices = mesh.vertices.copy()
    original_upper = upper.copy()
    original_lower = lower.copy()
    original_cut_coordinates = topology.cut_node_coordinates.copy()
    original_cut_vertices = topology.ordered_cut_vertex_indices.copy()

    def march_for(epsilon: float, timestep: float, boundary_order: int) -> ExplicitMidpointWakeMarch:
        return march_actual_boundary_material_wake_explicit_midpoint(
            mesh, topology,
            incident_velocity_at_time=lambda time: _incident_history(time, epsilon=epsilon, start=start, end=end, face_count=len(mesh.faces)),
            initial_body_cut_jump=np.zeros(len(topology.cut_node_coordinates)),
            time_start=start, time_end=end, timestep=timestep,
            trailing_edge_x=float(canonical["trailing_edge_x"]),
            convection_speed=float(canonical["prescribed_material_convection_speed"]),
            target_quadrature_order=boundary_order, source_quadrature_order=boundary_order,
        )

    def observe(epsilon: float, timestep: float, boundary_order: int) -> MarchPressureObservation:
        return observe_material_march(march_for(epsilon, timestep, boundary_order), upper_face_indices=upper, lower_face_indices=lower, direct_w_quadrature_order=int(canonical["direct_w_quadrature_order"]), pressure_line_quadrature_order=int(canonical["pressure_line_quadrature_order"]))

    epsilons = [float(value) for value in contract["families"]["epsilon_centered"]["positive_epsilon_rad"]]
    epsilon_dt = float(contract["families"]["epsilon_centered"]["timestep"])
    epsilon_observations = [(observe(value, epsilon_dt, int(canonical["body_boundary_quadrature_order"])), observe(-value, epsilon_dt, int(canonical["body_boundary_quadrature_order"]))) for value in epsilons]
    mass = epsilon_observations[0][0].mass_active
    epsilon_tangents = [centered_tangent(plus.residual, minus.residual, eps) for eps, (plus, minus) in zip(epsilons, epsilon_observations, strict=True)]
    time_values = [float(value) for value in contract["families"]["timestep_centered"]["timestep"]]
    time_epsilon = float(contract["families"]["timestep_centered"]["epsilon_rad"])
    timestep_observations = [
        (
            observe(time_epsilon, dt, int(canonical["body_boundary_quadrature_order"])),
            observe(-time_epsilon, dt, int(canonical["body_boundary_quadrature_order"])),
        )
        for dt in time_values
    ]
    timestep_tangents = [
        centered_tangent(plus.residual, minus.residual, time_epsilon)
        for plus, minus in timestep_observations
    ]
    zero = [observe(0.0, epsilon_dt, int(order)) for order in contract["families"]["zero_reference_quadrature"]["boundary_orders"]]
    zero_norms = [dual_mass_norm(value.residual, mass) for value in zero]
    epsilon_cauchy = _cauchy(epsilon_tangents[-3:], mass)
    timestep_cauchy = _cauchy(timestep_tangents, mass)
    # A repeat is read-only and uses an already frozen epsilon/dt member;
    # its difference enters uncertainty but is never used as an offset.
    repeat_plus = observe(
        epsilons[1], epsilon_dt,
        int(canonical["body_boundary_quadrature_order"]),
    )
    repeat_minus = observe(
        -epsilons[1], epsilon_dt,
        int(canonical["body_boundary_quadrature_order"]),
    )
    repeat_tangent = centered_tangent(
        repeat_plus.residual, repeat_minus.residual, epsilons[1]
    )
    deterministic_repeat_difference = dual_mass_norm(
        repeat_tangent - epsilon_tangents[1], mass
    )
    partial_uncertainty = max(
        zero_norms[-1], epsilon_cauchy[1], timestep_cauchy[1],
        deterministic_repeat_difference,
    )
    mutation = max(
        _maximum_abs(mesh.vertices - original_vertices),
        _maximum_abs(upper - original_upper),
        _maximum_abs(lower - original_lower),
        _maximum_abs(topology.cut_node_coordinates - original_cut_coordinates),
        _maximum_abs(
            topology.ordered_cut_vertex_indices - original_cut_vertices
        ),
    )
    all_observations = (
        [item for pair in epsilon_observations for item in pair]
        + [item for pair in timestep_observations for item in pair]
        + zero
    )
    gates = {name: all(_gates(value, thresholds)[name] for value in all_observations) for name in _gates(all_observations[0], thresholds)}
    gates["zero_reference"] = zero_norms[-1] <= float(thresholds["zero_reference_pressure_finest_abs_max"]) and all(right < left for left, right in zip(zero_norms, zero_norms[1:]))
    gates["epsilon_cauchy"] = epsilon_cauchy[2] >= float(thresholds["epsilon_tangent_cauchy_ratio_min"])
    gates["timestep_cauchy"] = timestep_cauchy[2] >= float(thresholds["timestep_tangent_cauchy_ratio_min"])
    gates["input_immutable"] = mutation <= float(thresholds["input_mutation_abs_max"])
    gate_failure = not all(gates.values())
    return {
        "stage": contract["stage"],
        "stage_decision": "NO-GO" if gate_failure else "UNRESOLVED",
        "production_activation_allowed": False,
        "reason": (
            "One or more preregistered representation/Cauchy gates failed."
            if gate_failure
            else "No frozen mapping converts body BIE/compatibility/Kelvin "
            "residuals into the required dual-mass uncertainty scale."
        ),
        "checks": gates,
        "epsilon_cauchy": epsilon_cauchy,
        "timestep_cauchy": timestep_cauchy,
        "zero_reference_dual_mass_norms": zero_norms,
        "finest_tangent_dual_mass_norm": dual_mass_norm(epsilon_tangents[-1], mass),
        "deterministic_repeat_dual_mass_difference": deterministic_repeat_difference,
        "partial_numerical_uncertainty": partial_uncertainty,
    }
