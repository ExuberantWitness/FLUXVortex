"""Run preregistered implicit-ClaimGroup solver-semantic gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.implicit_claim_group import (
    ImplicitClaimGroup,
    ImplicitClaimGroupError,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "implicit_claim_group_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "implicit_claim_group_results.json"


MATRIX = np.array([
    [2.0, 0.1, 0.4, -0.2, 0.3, 0.0],
    [0.2, 1.8, -0.1, 0.3, 0.0, 0.4],
    [0.3, 0.0, 1.7, 0.2, -0.4, 0.1],
    [0.0, -0.2, 0.3, 1.9, 0.2, -0.3],
    [0.4, 0.1, -0.2, 0.0, 1.6, 0.2],
    [-0.1, 0.3, 0.0, -0.2, 0.1, 1.7],
])
TRUTH = {
    "n1": np.array([0.18, -0.07]),
    "n2_6": np.array([0.11, -0.04]),
    "n3": np.array([0.06, 0.02]),
}
STATE_NAMES = ("n1", "n2_6", "n3")
RESIDUAL_NAMES = ("no_penetration", "ibl", "kelvin")
SCALES = {"no_penetration": 8.0, "ibl": 0.4, "kelvin": 2.0}
STATE_COLUMNS = {
    name: slice(2*index, 2*index+2)
    for index, name in enumerate(STATE_NAMES)
}
RESIDUAL_ROWS = {
    name: slice(2*index, 2*index+2)
    for index, name in enumerate(RESIDUAL_NAMES)
}
TARGET = MATRIX@np.concatenate([TRUTH[name] for name in STATE_NAMES])


def _linear_solution(state_order, residual_order):
    group = ImplicitClaimGroup(
        state_block_sizes={name: 2 for name in state_order},
        residual_block_sizes={name: 2 for name in residual_order},
        residual_physical_scales={
            name: SCALES[name] for name in residual_order
        },
    )

    def residual(state):
        vector = np.concatenate([state[name] for name in STATE_NAMES])
        raw = MATRIX@vector-TARGET
        return {
            name: raw[RESIDUAL_ROWS[name]]
            for name in residual_order
        }

    def jacobian(_state):
        return {
            (residual_name, state_name):
                MATRIX[
                    RESIDUAL_ROWS[residual_name],
                    STATE_COLUMNS[state_name],
                ]
            for residual_name in residual_order
            for state_name in state_order
        }

    return group.solve(
        initial_state={name: np.zeros(2) for name in state_order},
        residual_function=residual,
        jacobian_function=jacobian,
        normalized_tolerance=1.0e-12,
    )


def _nonlinear_solution():
    group = ImplicitClaimGroup(
        state_block_sizes={"n1": 1, "n2_6": 1, "n3": 1},
        residual_block_sizes={
            "no_penetration": 1,
            "ibl": 1,
            "kelvin": 1,
        },
        residual_physical_scales={
            "no_penetration": 1.0,
            "ibl": 1.0,
            "kelvin": 1.0,
        },
    )

    def residual(state):
        x = state["n1"][0]
        y = state["n2_6"][0]
        z = state["n3"][0]
        return {
            "no_penetration": np.array([x+y+z-3.0]),
            "ibl": np.array([x*x+y-2.0]),
            "kelvin": np.array([z+y*y-2.0]),
        }

    def jacobian(state):
        x = state["n1"][0]
        y = state["n2_6"][0]
        return {
            ("no_penetration", "n1"): np.array([[1.0]]),
            ("no_penetration", "n2_6"): np.array([[1.0]]),
            ("no_penetration", "n3"): np.array([[1.0]]),
            ("ibl", "n1"): np.array([[2.0*x]]),
            ("ibl", "n2_6"): np.array([[1.0]]),
            ("ibl", "n3"): np.array([[0.0]]),
            ("kelvin", "n1"): np.array([[0.0]]),
            ("kelvin", "n2_6"): np.array([[2.0*y]]),
            ("kelvin", "n3"): np.array([[1.0]]),
        }

    return group.solve(
        initial_state={
            "n1": np.array([0.7]),
            "n2_6": np.array([1.2]),
            "n3": np.array([0.9]),
        },
        residual_function=residual,
        jacobian_function=jacobian,
        normalized_tolerance=1.0e-12,
        maximum_updates=12,
    )


def run(*, write: bool = False) -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())
    linear = _linear_solution(STATE_NAMES, RESIDUAL_NAMES)
    linear_error = float(max(
        np.max(np.abs(linear.state[name]-TRUTH[name]))
        for name in STATE_NAMES
    ))
    nonlinear = _nonlinear_solution()
    nonlinear_error = float(max(
        np.max(np.abs(value-1.0))
        for value in nonlinear.state.values()
    ))
    permuted = _linear_solution(
        ("n3", "n1", "n2_6"),
        ("kelvin", "no_penetration", "ibl"),
    )
    permutation_error = float(max(
        np.max(np.abs(linear.state[name]-permuted.state[name]))
        for name in STATE_NAMES
    ))

    forbidden_raised = False
    try:
        ImplicitClaimGroup(
            state_block_sizes={"n1": 1},
            residual_block_sizes={"target_lift": 1},
            residual_physical_scales={"target_lift": 1.0},
        )
    except ImplicitClaimGroupError:
        forbidden_raised = True
    rank_raised = False
    rank_group = ImplicitClaimGroup(
        state_block_sizes={"n1": 1},
        residual_block_sizes={"no_penetration": 1},
        residual_physical_scales={"no_penetration": 1.0},
    )
    try:
        rank_group.solve(
            initial_state={"n1": np.array([0.0])},
            residual_function=lambda state: {
                "no_penetration": np.array([1.0])
            },
            jacobian_function=lambda state: {
                ("no_penetration", "n1"): np.array([[0.0]])
            },
        )
    except ImplicitClaimGroupError:
        rank_raised = True

    metrics = {
        "exact_linear_three_block": {
            "state_error": linear_error,
            "max_normalized_residual": linear.max_normalized_residual,
        },
        "nonlinear_three_way_coupling": {
            "state_error": nonlinear_error,
            "max_normalized_residual":
                nonlinear.max_normalized_residual,
            "maximum_iterations": nonlinear.update_count,
        },
        "named_permutation_invariance": {
            "named_state_difference": permutation_error,
        },
        "failure_semantics": {
            "rank_deficiency_must_raise": rank_raised,
            "forbidden_residual_must_raise": forbidden_raised,
        },
    }
    thresholds = {
        case["id"]: case["gates"]
        for case in prereg["cases"]
    }
    passed = {}
    for case_id, gates in thresholds.items():
        passed[case_id] = {}
        for metric, threshold in gates.items():
            value = metrics[case_id][metric]
            if isinstance(threshold, bool):
                passed[case_id][metric] = value is threshold
            else:
                passed[case_id][metric] = value <= float(threshold)
    result = {
        "artifact": prereg["artifact"],
        "claim_node": prereg["claim_node"],
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": passed,
        "all_pass": all(
            value
            for case in passed.values()
            for value in case.values()
        ),
        "scope_limit": prereg["promotion"]["must_not_claim"],
        "interpretation": (
            "Named block Newton semantics pass manufactured coupled systems "
            "and explicit failure gates. No physical N1-N2.6-N3 residual has "
            "been assembled."
        ),
    }
    if write:
        RESULT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2)+"\n"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run(write=arguments.write), ensure_ascii=False, indent=2))

