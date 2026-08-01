"""Run the preregistered S3ac0 direct/frozen trace equivalence gate."""
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
from claim_runtime.actual_wake_direct_trace import (  # noqa: E402
    solve_actual_wake_coupled_newborn_trace_direct,
)
from claim_runtime.actual_wake_newborn_coupled_stage import (  # noqa: E402
    solve_actual_wake_coupled_newborn_trace,
)
from claim_runtime.actual_wake_newborn_transition import (  # noqa: E402
    augment_actual_wake_with_newborn_band,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.actual_wake_stage_velocity import (  # noqa: E402
    actual_wake_owned_quadrature,
)
from claim_runtime.actual_wake_weak_geometry_velocity import (  # noqa: E402
    project_actual_wake_global_weak_normal_velocity,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_direct_trace_equivalence_cases.yaml"
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
    / "actual_wake_direct_trace_equivalence_results.json"
)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    s3t_contract = yaml.safe_load(
        S3T_CASES.read_text(encoding="utf-8")
    )
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
    geometry_snapshot = tuple(
        band.surface.vertices.copy() for band in history.bands
    )
    scalar_snapshot = tuple(
        band.potential_jump_rows.copy() for band in history.bands
    )
    quadrature = actual_wake_owned_quadrature(
        topology,
        history,
        quadrature_order=7,
        query_id="S3ac0-q7-release",
    )
    ledger = _actual_ledger(solution, quadrature.query, s3t_contract)
    weak = project_actual_wake_global_weak_normal_velocity(
        topology,
        history,
        quadrature,
        ledger.total,
    )
    body_p1 = topology.boundary_roles.body_attachment_p1_dofs
    body_edge = topology.p1_vertices[body_p1]
    old_trace = topology.chronological_rows(history)[-1]
    timestep = 0.01
    start_time = float(history.bands[-1].time_nodes[-1])
    transition = augment_actual_wake_with_newborn_band(
        history,
        topology,
        released_edge=(
            body_edge
            + 0.5 * timestep * weak.dof_velocity[body_p1]
        ),
        current_body_edge=body_edge,
        time_nodes=np.array(
            (
                start_time,
                start_time + 0.25 * timestep,
                start_time + 0.5 * timestep,
            )
        ),
        midpoint_trace=old_trace,
        current_trace=old_trace,
        sheet_id="S3ac0-half-equivalence",
    )
    frozen = solve_actual_wake_coupled_newborn_trace(
        mesh,
        body_topology,
        transition,
        attachment,
        incident_velocity=solution.incident_velocity,
        wall_velocity=solution.wall_velocity,
        copy_counterfactual_trace=old_trace,
        boundary_quadrature_order=10,
    )
    direct = solve_actual_wake_coupled_newborn_trace_direct(
        mesh,
        body_topology,
        transition,
        attachment,
        incident_velocity=solution.incident_velocity,
        wall_velocity=solution.wall_velocity,
        copy_counterfactual_trace=old_trace,
        boundary_quadrature_order=10,
    )
    body_potential = float(
        np.max(
            np.abs(
                frozen.solution.global_body_potential
                - direct.solution.global_body_potential
            ),
            initial=0.0,
        )
    )
    wake_p2 = float(
        np.max(
            np.abs(
                frozen.global_p2_state - direct.global_p2_state
            ),
            initial=0.0,
        )
    )
    trace = float(
        np.max(
            np.abs(frozen.solved_trace - direct.solved_trace),
            initial=0.0,
        )
    )
    diagnostic_names = (
        "free_state_preservation_error",
        "maximum_actual_boundary_relative_weak_residual",
        "maximum_wake_attachment_error",
        "copy_counterfactual_residual",
        "input_state_mutation",
    )
    diagnostic = max(
        abs(
            float(getattr(frozen.report, name))
            - float(getattr(direct.report, name))
        )
        for name in diagnostic_names
    )
    mutation = max(
        *(
            float(
                np.max(
                    np.abs(band.surface.vertices - geometry),
                    initial=0.0,
                )
            )
            for band, geometry in zip(
                history.bands,
                geometry_snapshot,
                strict=True,
            )
        ),
        *(
            float(
                np.max(
                    np.abs(band.potential_jump_rows - scalar),
                    initial=0.0,
                )
            )
            for band, scalar in zip(
                history.bands,
                scalar_snapshot,
                strict=True,
            )
        ),
    )
    checks = {
        "global_body_potential_is_identical": (
            body_potential
            <= float(thresholds["global_body_potential_abs_max"])
        ),
        "global_wake_P2_state_is_identical": (
            wake_p2
            <= float(thresholds["global_wake_P2_abs_max"])
        ),
        "solved_current_trace_is_identical": (
            trace <= float(thresholds["solved_trace_abs_max"])
        ),
        "all_shared_diagnostics_are_identical": (
            diagnostic <= float(thresholds["diagnostic_abs_max"])
        ),
        "input_state_is_immutable": (
            mutation
            <= float(thresholds["input_state_mutation_abs_max"])
        ),
        "direct_path_uses_one_actual_solve": (
            direct.report.basis_solve_count
            == int(
                thresholds["direct_basis_solve_count_expected"]
            )
            and frozen.report.basis_solve_count > 1
        ),
        "matrix_rhs_and_quadrature_are_identical": (
            np.array_equal(
                frozen.solution.matrix,
                direct.solution.matrix,
            )
            and np.array_equal(
                frozen.solution.right_hand_side,
                direct.solution.right_hand_side,
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_direct_trace_equivalence_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "global_body_potential_abs_max": body_potential,
            "global_wake_P2_abs_max": wake_p2,
            "solved_trace_abs_max": trace,
            "diagnostic_abs_max": diagnostic,
            "input_state_mutation_abs_max": mutation,
            "frozen_basis_solve_count": (
                frozen.report.basis_solve_count
            ),
            "direct_basis_solve_count": (
                direct.report.basis_solve_count
            ),
            "matrix_bitwise_equal": bool(
                np.array_equal(
                    frozen.solution.matrix,
                    direct.solution.matrix,
                )
            ),
            "rhs_bitwise_equal": bool(
                np.array_equal(
                    frozen.solution.right_hand_side,
                    direct.solution.right_hand_side,
                )
            ),
        },
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2))
    return 0 if result["stage_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
