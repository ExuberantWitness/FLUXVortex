"""Run the preregistered S3u actual-wake DAE coupling audit.

The audit is deliberately read-only.  It measures the two algebraic arrows
that a time algorithm must preserve:

    interior material P2 state -> solved body-attachment trace
    attachment trace rate      -> free P2 weak rate through M_fb

No velocity field, time step, geometry update, pressure or force is used.
"""
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
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.actual_wake_stage_velocity import (  # noqa: E402
    actual_wake_owned_quadrature,
    assemble_owned_actual_wake_p2_transport,
)
from claim_runtime.p2_surface_material_transport import (  # noqa: E402
    P2EssentialScalarTrace,
    p2_essential_trace_transport_rate,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_time_coupling_audit_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_time_coupling_audit_results.json"
)


def _normalized_inf_residual(
    residual: np.ndarray,
    *terms: np.ndarray,
) -> float:
    scale = max(
        (
            float(np.linalg.norm(term, ord=np.inf))
            for term in terms
        ),
        default=0.0,
    )
    return float(
        np.linalg.norm(residual, ord=np.inf)
        / max(scale, np.finfo(float).tiny)
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
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
        base_solution,
    ) = _canonical_state()
    # The preregistered actual state is the algebraically consistent output,
    # not the geometrically identical pre-solve history used to construct it.
    history = base_solution.wake_history
    topology = actual_wake_stage_topology(
        history,
        body_attachment_id="canonical-body-cut",
    )
    state_before = topology.global_p2_state(history)
    geometry_before = tuple(
        band.surface.vertices.copy() for band in history.bands
    )

    quadrature = actual_wake_owned_quadrature(
        topology,
        history,
        quadrature_order=int(
            canonical["mass_assembly"]["quadrature_order"]
        ),
        query_id="S3u-zero-advection-mass-query",
    )
    zero_velocity = np.zeros_like(quadrature.query.points)
    mass_operator = assemble_owned_actual_wake_p2_transport(
        topology,
        history,
        quadrature,
        zero_velocity,
    )
    body = topology.boundary_roles.body_attachment_p2_dofs
    all_boundary = topology.boundary_roles.all_boundary_p2_dofs
    count = topology.p2_topology.degree_of_freedom_count
    free = np.setdiff1d(
        np.arange(count, dtype=np.int64),
        body,
    )
    strict_interior = np.setdiff1d(
        np.arange(count, dtype=np.int64),
        all_boundary,
    )
    mass_ff = mass_operator.mass_matrix[np.ix_(free, free)]
    mass_fb = mass_operator.mass_matrix[np.ix_(free, body)]
    free_mass_rank = int(np.linalg.matrix_rank(mass_ff))
    mass_boundary_coupling = float(
        np.linalg.norm(mass_fb, ord=np.inf)
        / max(
            float(np.linalg.norm(mass_ff, ord=np.inf)),
            np.finfo(float).tiny,
        )
    )

    direction = np.zeros(count, dtype=float)
    phase = np.arange(1, len(strict_interior) + 1, dtype=float)
    direction[strict_interior] = np.sin(
        np.pi * phase / (len(strict_interior) + 1)
    )
    direction /= np.max(np.abs(direction), initial=0.0)
    amplitudes = tuple(
        float(value)
        for value in canonical["interior_counterfactual"]["amplitudes"]
    )
    if (
        len(amplitudes) != 2
        or amplitudes[0] <= 0.0
        or amplitudes[1] != 2.0 * amplitudes[0]
    ):
        raise RuntimeError(
            "S3u amplitudes must be positive and exactly doubled"
        )

    solutions = []
    responses = []
    for amplitude in amplitudes:
        perturbed_history = topology.rebuild_history(
            history,
            state_before + amplitude * direction,
        )
        solution = solve_actual_boundary_body_wake_p2(
            mesh,
            body_topology,
            incident_velocity=base_solution.incident_velocity,
            wall_velocity=base_solution.wall_velocity,
            downstream_edge_x=None,
            prescribed_wake_history=perturbed_history,
            prescribed_wake_attachment=attachment,
            target_quadrature_order=10,
            source_quadrature_order=10,
        )
        solutions.append(solution)
        responses.append(
            solution.body_cut_jump - base_solution.body_cut_jump
        )
    response_scale = float(
        np.max(np.abs(responses[0]), initial=0.0)
    )
    response_linearity = float(
        np.max(
            np.abs(responses[1] - 2.0 * responses[0]),
            initial=0.0,
        )
        / max(
            float(np.max(np.abs(responses[1]), initial=0.0)),
            np.finfo(float).tiny,
        )
    )
    body_trace_rate = responses[0] / amplitudes[0]
    base_attachment_error = float(
        np.max(
            np.abs(
                state_before[body] - base_solution.body_cut_jump
            ),
            initial=0.0,
        )
    )

    essential = P2EssentialScalarTrace(
        body,
        "S3u:actual-body-inflow",
    )
    zero_state = np.zeros(count, dtype=float)
    zero_body = np.zeros(len(body), dtype=float)
    correct = p2_essential_trace_transport_rate(
        mass_operator,
        zero_state,
        essential_trace=essential,
        prescribed_value=zero_body,
        prescribed_time_derivative=body_trace_rate,
        prescribed_value_tolerance=0.0,
    )
    correct_mass_free = mass_ff @ correct.rate[free]
    boundary_mass_rate = mass_fb @ body_trace_rate
    correct_residual = correct_mass_free + boundary_mass_rate
    correct_normalized = _normalized_inf_residual(
        correct_residual,
        correct_mass_free,
        boundary_mass_rate,
    )

    sequential_rate = mass_operator.rate(zero_state)
    sequential_rate[body] = body_trace_rate
    sequential_mass_free = mass_ff @ sequential_rate[free]
    sequential_residual = (
        sequential_mass_free + boundary_mass_rate
    )
    sequential_normalized = _normalized_inf_residual(
        sequential_residual,
        sequential_mass_free,
        boundary_mass_rate,
    )

    maximum_weak_residual = max(
        [
            float(base_solution.relative_weak_residual),
            *[
                float(solution.relative_weak_residual)
                for solution in solutions
            ],
        ]
    )
    state_mutation = float(
        np.max(
            np.abs(topology.global_p2_state(history) - state_before),
            initial=0.0,
        )
    )
    geometry_mutation = max(
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
    )
    input_mutation = max(state_mutation, geometry_mutation)

    checks = {
        "typed_body_trace_leaves_expected_free_state": (
            count
            == int(thresholds["expected_global_p2_dof_count"])
            and len(body)
            == int(thresholds["expected_body_p2_dof_count"])
            and len(free)
            == int(thresholds["expected_free_p2_dof_count"])
            and base_attachment_error == 0.0
        ),
        "free_mass_is_full_rank_and_boundary_coupled": (
            len(free) - free_mass_rank
            <= int(thresholds["free_mass_rank_deficiency_max"])
            and mass_boundary_coupling
            >= float(
                thresholds["normalized_mass_boundary_coupling_min"]
            )
        ),
        "interior_state_changes_algebraic_body_trace": (
            response_scale
            >= float(
                thresholds[
                    "algebraic_body_trace_response_abs_min"
                ]
            )
        ),
        "algebraic_response_is_linear": (
            response_linearity
            <= float(
                thresholds[
                    "algebraic_response_linearity_relative_max"
                ]
            )
        ),
        "correct_free_constrained_block_closes": (
            correct_normalized
            <= float(
                thresholds[
                    "correct_free_block_normalized_residual_max"
                ]
            )
        ),
        "sequential_clamp_leaves_order_one_residual": (
            sequential_normalized
            >= float(
                thresholds[
                    "sequential_clamp_normalized_residual_min"
                ]
            )
        ),
        "all_actual_boundary_solves_remain_valid": (
            maximum_weak_residual
            <= float(
                thresholds[
                    "actual_boundary_relative_weak_residual_max"
                ]
            )
        ),
        "audit_does_not_mutate_input_state": (
            input_mutation
            <= float(thresholds["input_state_mutation_abs_max"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_time_coupling_audit",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": (
            "COUPLED" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "global_p2_dof_count": count,
            "body_p2_dof_count": len(body),
            "free_p2_dof_count": len(free),
            "strict_interior_p2_dof_count": len(strict_interior),
            "base_attachment_identity_abs_max": (
                base_attachment_error
            ),
            "free_mass_rank_deficiency": len(free) - free_mass_rank,
            "normalized_mass_boundary_coupling": (
                mass_boundary_coupling
            ),
            "algebraic_body_trace_response_abs_max": response_scale,
            "algebraic_body_trace_directional_rate_abs_max": float(
                np.max(np.abs(body_trace_rate), initial=0.0)
            ),
            "algebraic_response_linearity_relative": (
                response_linearity
            ),
            "correct_free_block_normalized_residual": (
                correct_normalized
            ),
            "sequential_clamp_normalized_residual": (
                sequential_normalized
            ),
            "actual_boundary_relative_weak_residual_max": (
                maximum_weak_residual
            ),
            "input_state_mutation_abs_max": input_mutation,
        },
        "interpretation": (
            "The actual stage is a two-way differential-algebraic "
            "coupling; endpoint clamping is not a constraint-consistent "
            "second-order composition."
            if all(checks.values())
            else "The preregistered DAE identity was not established."
        ),
        "next_gate": (
            "Preregister constraint-consistent half-explicit RK and "
            "present/averaged-time implicit residual candidates using "
            "named stage, weak, attachment, transport and time-Cauchy "
            "residuals."
        ),
        "forbidden_quantities_absent": [
            "actual_velocity_evaluation",
            "geometry_update",
            "P2_scalar_update",
            "timestep",
            "nonlinear_iteration",
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
        0 if payload["stage_decision"] == "COUPLED" else 1
    )
