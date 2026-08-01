"""Run the preregistered S3v fixed-geometry actual-DAE time gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_body_wake_velocity_ledger_guard import (  # noqa: E402
    _canonical_state,
)
from actual_wake_owned_stage_velocity_guard import (  # noqa: E402
    _actual_ledger,
)
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    wake_sheet_interior_query,
)
from claim_runtime.actual_wake_linearized_dae import (  # noqa: E402
    build_actual_wake_linearized_dae,
    solve_actual_wake_affine_counterfactual,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.actual_wake_stage_velocity import (  # noqa: E402
    actual_wake_owned_quadrature,
    assemble_owned_actual_wake_p2_transport,
    project_actual_wake_vertex_star_normal_velocity,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_linearized_dae_time_cases.yaml"
)
S3T_CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_owned_stage_velocity_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_linearized_dae_time_results.json"
)


def _relative_error(
    value: np.ndarray,
    reference: np.ndarray,
    initial: np.ndarray,
) -> float:
    scale = max(
        float(np.linalg.norm(reference, ord=np.inf)),
        float(np.linalg.norm(initial, ord=np.inf)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(value - reference, ord=np.inf) / scale)


def _cauchy(finals: list[np.ndarray]) -> tuple[list[float], float]:
    changes = [
        float(np.linalg.norm(right - left, ord=np.inf))
        for left, right in zip(finals, finals[1:])
    ]
    ratios = [
        changes[index] / max(changes[index + 1], np.finfo(float).tiny)
        for index in range(len(changes) - 1)
    ]
    return changes, min(ratios)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    s3t_contract = yaml.safe_load(S3T_CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    (
        mesh,
        body_topology,
        _upper,
        _lower,
        _cut_edges,
        _endpoints,
        _pre_solve_history,
        attachment,
        solution,
    ) = _canonical_state()
    history = solution.wake_history
    topology = actual_wake_stage_topology(
        history,
        body_attachment_id="canonical-body-cut",
    )
    state_before = topology.global_p2_state(history)
    geometry_before = tuple(
        band.surface.vertices.copy() for band in history.bands
    )

    geometry_query = wake_sheet_interior_query(history)
    geometry_ledger = _actual_ledger(
        solution,
        geometry_query,
        s3t_contract,
    )
    body_geometry_velocity = np.zeros(
        (
            len(topology.boundary_roles.body_attachment_p1_dofs),
            3,
        )
    )
    projection = project_actual_wake_vertex_star_normal_velocity(
        topology,
        history,
        geometry_query,
        geometry_ledger.total,
        body_attachment_velocity=body_geometry_velocity,
    )
    quadrature_order = int(
        canonical["physical_operator"]["triangle_quadrature_order"]
    )
    quadrature = actual_wake_owned_quadrature(
        topology,
        history,
        quadrature_order=quadrature_order,
        query_id="S3v-actual-P2-query",
    )
    transport_ledger = _actual_ledger(
        solution,
        quadrature.query,
        s3t_contract,
    )
    mesh_velocity = projection.evaluate_query(
        history,
        quadrature.query,
    )
    transport = assemble_owned_actual_wake_p2_transport(
        topology,
        history,
        quadrature,
        transport_ledger.total - mesh_velocity,
    )
    dae = build_actual_wake_linearized_dae(
        mesh,
        body_topology,
        topology,
        solution,
        attachment,
        transport,
        boundary_quadrature_order=10,
    )

    free = dae.free_dof_indices
    counter_direction = np.cos(
        (np.arange(len(free), dtype=float) + 1.0) * np.sqrt(2.0)
    )
    counter_state = dae.base_global_state.copy()
    counter_state[free] += counter_direction
    counter_solution = solve_actual_wake_affine_counterfactual(
        mesh,
        body_topology,
        topology,
        solution,
        attachment,
        counter_state,
        quadrature_order=10,
    )
    counter_solved = topology.global_p2_state(
        counter_solution.wake_history
    )
    counter_error = float(
        np.max(
            np.abs(
                counter_solved[dae.body_dof_indices]
                - dae.body_trace(counter_state[free])
            ),
            initial=0.0,
        )
    )
    maximum_boundary_weak = max(
        dae.report.maximum_relative_weak_residual,
        float(counter_solution.relative_weak_residual),
    )

    time = canonical["time"]
    duration = float(time["end"]) - float(time["start"])
    steps = [int(value) for value in time["step_families"]]
    initial = dae.base_global_state[free]
    exact = dae.exact_affine_step(initial, duration)
    explicit_records = [
        dae.explicit_midpoint(initial, duration=duration, steps=count)
        for count in steps
    ]
    implicit_records = [
        dae.implicit_trapezoidal(
            initial,
            duration=duration,
            steps=count,
        )
        for count in steps
    ]
    explicit_finals = [record[0] for record in explicit_records]
    implicit_finals = [record[0] for record in implicit_records]
    explicit_changes, explicit_cauchy = _cauchy(explicit_finals)
    implicit_changes, implicit_cauchy = _cauchy(implicit_finals)
    explicit_error = _relative_error(
        explicit_finals[-1],
        exact,
        initial,
    )
    implicit_error = _relative_error(
        implicit_finals[-1],
        exact,
        initial,
    )
    explicit_implicit = _relative_error(
        explicit_finals[-1],
        implicit_finals[-1],
        initial,
    )
    algebraic_error = max(
        [record[2] for record in explicit_records + implicit_records]
    )
    weak_residual = max(
        [record[1] for record in explicit_records + implicit_records]
    )
    polynomial_defect = dae.explicit_polynomial_defect(
        duration / steps[0]
    )

    input_mutation = max(
        float(
            np.max(
                np.abs(
                    topology.global_p2_state(history) - state_before
                ),
                initial=0.0,
            )
        ),
        max(
            float(
                np.max(
                    np.abs(band.surface.vertices - before),
                    initial=0.0,
                )
            )
            for band, before in zip(
                history.bands,
                geometry_before,
                strict=True,
            )
        ),
    )

    map_checks = {
        "actual_affine_trace_map_is_exact": (
            dae.report.basis_solve_count == len(free)
            and dae.report.base_reconstruction_error
            <= float(
                thresholds["algebraic_base_reconstruction_abs_max"]
            )
            and counter_error
            <= float(
                thresholds[
                    "algebraic_counterfactual_reconstruction_abs_max"
                ]
            )
            and maximum_boundary_weak
            <= float(
                thresholds[
                    "actual_boundary_relative_weak_residual_max"
                ]
            )
        ),
        "effective_differential_mass_is_valid": (
            len(free) - dae.report.effective_mass_rank
            <= int(
                thresholds["effective_mass_rank_deficiency_max"]
            )
            and dae.report.effective_mass_condition_number
            <= float(
                thresholds["effective_mass_condition_number_max"]
            )
        ),
        "both_methods_preserve_named_constraints": (
            algebraic_error
            <= float(thresholds["algebraic_constraint_abs_max"])
            and weak_residual
            <= float(thresholds["free_weak_normalized_residual_max"])
        ),
    }
    explicit_checks = {
        "explicit_midpoint_is_second_order": (
            explicit_cauchy
            >= float(thresholds["explicit_time_cauchy_ratio_min"])
        ),
        "explicit_midpoint_matches_exact_reference": (
            explicit_error
            <= float(thresholds["explicit_finest_relative_error_max"])
        ),
        "explicit_local_stability_defect_is_bounded": (
            polynomial_defect
            <= float(
                thresholds[
                    "coarsest_explicit_polynomial_defect_max"
                ]
            )
        ),
    }
    implicit_checks = {
        "implicit_trapezoidal_is_second_order": (
            implicit_cauchy
            >= float(thresholds["implicit_time_cauchy_ratio_min"])
        ),
        "implicit_trapezoidal_matches_exact_reference": (
            implicit_error
            <= float(thresholds["implicit_finest_relative_error_max"])
        ),
    }
    common_checks = {
        "finest_explicit_and_implicit_agree": (
            explicit_implicit
            <= float(
                thresholds[
                    "finest_explicit_implicit_relative_difference_max"
                ]
            )
        ),
        "gate_does_not_mutate_actual_state": (
            input_mutation
            <= float(thresholds["input_state_mutation_abs_max"])
        ),
    }
    checks = {
        **map_checks,
        **explicit_checks,
        **implicit_checks,
        **common_checks,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    map_ok = all(map_checks.values()) and all(common_checks.values())
    explicit_ok = all(explicit_checks.values())
    implicit_ok = all(implicit_checks.values())
    if map_ok and explicit_ok and implicit_ok:
        decision = "EXPLICIT-GO"
    elif map_ok and implicit_ok:
        decision = "IMPLICIT-ONLY"
    else:
        decision = "NO-GO"
    result = {
        "artifact": "actual_wake_linearized_dae_time_oracle",
        "claim_node": contract["claim_node"],
        "counterfactual_claim_node": contract[
            "counterfactual_claim_node"
        ],
        "stage": contract["stage"],
        "stage_decision": decision,
        "checks": checks,
        "aggregate_metrics": {
            "basis_solve_count": dae.report.basis_solve_count,
            "algebraic_base_reconstruction_abs_max": (
                dae.report.base_reconstruction_error
            ),
            "algebraic_counterfactual_reconstruction_abs_max": (
                counter_error
            ),
            "actual_boundary_relative_weak_residual_max": (
                maximum_boundary_weak
            ),
            "effective_mass_rank_deficiency": (
                len(free) - dae.report.effective_mass_rank
            ),
            "effective_mass_condition_number": (
                dae.report.effective_mass_condition_number
            ),
            "reduced_spectral_radius": float(
                np.max(
                    np.abs(np.linalg.eigvals(dae.reduced_matrix)),
                    initial=0.0,
                )
            ),
            "algebraic_constraint_abs_max": algebraic_error,
            "free_weak_normalized_residual_max": weak_residual,
            "explicit_final_changes": explicit_changes,
            "explicit_time_cauchy_ratio_min": explicit_cauchy,
            "implicit_final_changes": implicit_changes,
            "implicit_time_cauchy_ratio_min": implicit_cauchy,
            "explicit_finest_relative_error": explicit_error,
            "implicit_finest_relative_error": implicit_error,
            "coarsest_explicit_polynomial_defect": polynomial_defect,
            "finest_explicit_implicit_relative_difference": (
                explicit_implicit
            ),
            "input_state_mutation_abs_max": input_mutation,
            "geometry_projection_residual_fraction": (
                projection.report.maximum_absolute_residual
                / max(
                    projection.report.maximum_input_normal_speed,
                    np.finfo(float).eps,
                )
            ),
            "actual_relative_velocity_normal_component_max": (
                transport.maximum_relative_velocity_normal_component
            ),
        },
        "interpretation": {
            "EXPLICIT-GO": (
                "The fixed-geometry actual DAE supplies no local evidence "
                "requiring a nonlinear geometry fixed point. Advance only "
                "to a nonlinear previous-time/stage-resolve prototype."
            ),
            "IMPLICIT-ONLY": (
                "The explicit local time gate failed while the implicit "
                "reference passed; preregister the nonlinear implicit "
                "residual branch."
            ),
            "NO-GO": (
                "The local actual DAE time-method identity was not "
                "established."
            ),
        }[decision],
        "forbidden_quantities_absent": [
            "geometry_update",
            "velocity_update",
            "fitted_damping",
            "pressure",
            "force",
            "LESP",
            "target_load",
            "structural_dynamics",
        ],
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(
        0 if payload["stage_decision"] == "EXPLICIT-GO" else 1
    )
