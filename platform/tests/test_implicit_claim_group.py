import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.implicit_claim_group import (  # noqa: E402
    ImplicitClaimGroup,
    ImplicitClaimGroupError,
)


def linear_problem(state_order=None, residual_order=None):
    state_order = state_order or ["n1", "n2_6", "n3"]
    residual_order = residual_order or ["no_penetration", "ibl", "kelvin"]
    state_sizes = {name: 2 for name in state_order}
    residual_sizes = {name: 2 for name in residual_order}
    scales = {
        "no_penetration": 8.0,
        "ibl": 0.4,
        "kelvin": 2.0,
    }
    matrix = np.array([
        [2.0, 0.1, 0.4, -0.2, 0.3, 0.0],
        [0.2, 1.8, -0.1, 0.3, 0.0, 0.4],
        [0.3, 0.0, 1.7, 0.2, -0.4, 0.1],
        [0.0, -0.2, 0.3, 1.9, 0.2, -0.3],
        [0.4, 0.1, -0.2, 0.0, 1.6, 0.2],
        [-0.1, 0.3, 0.0, -0.2, 0.1, 1.7],
    ])
    canonical_state = ["n1", "n2_6", "n3"]
    canonical_residual = ["no_penetration", "ibl", "kelvin"]
    state_columns = {
        name: slice(2*index, 2*index+2)
        for index, name in enumerate(canonical_state)
    }
    residual_rows = {
        name: slice(2*index, 2*index+2)
        for index, name in enumerate(canonical_residual)
    }
    truth = {
        "n1": np.array([0.18, -0.07]),
        "n2_6": np.array([0.11, -0.04]),
        "n3": np.array([0.06, 0.02]),
    }
    truth_vector = np.concatenate([truth[name] for name in canonical_state])
    target = matrix@truth_vector

    def residual(state):
        vector = np.concatenate([state[name] for name in canonical_state])
        raw = matrix@vector-target
        return {
            name: raw[residual_rows[name]]
            for name in residual_order
        }

    def jacobian(_state):
        return {
            (residual_name, state_name):
                matrix[residual_rows[residual_name], state_columns[state_name]]
            for residual_name in residual_order
            for state_name in state_order
        }

    group = ImplicitClaimGroup(
        state_block_sizes=state_sizes,
        residual_block_sizes=residual_sizes,
        residual_physical_scales={
            name: scales[name] for name in residual_order
        },
    )
    initial = {name: np.zeros(2) for name in state_order}
    return group, initial, residual, jacobian, truth


class ImplicitClaimGroupTests(unittest.TestCase):
    def test_exact_linear_three_block_system(self):
        group, initial, residual, jacobian, truth = linear_problem()
        solution = group.solve(
            initial_state=initial,
            residual_function=residual,
            jacobian_function=jacobian,
            normalized_tolerance=1.0e-12,
        )
        self.assertTrue(solution.passed)
        self.assertEqual(solution.update_count, 1)
        for name in truth:
            np.testing.assert_allclose(
                solution.state[name],
                truth[name],
                atol=2.0e-15,
            )
        self.assertLessEqual(solution.max_normalized_residual, 1.0e-15)

    def test_nonlinear_three_way_coupling(self):
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

        solution = group.solve(
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
        self.assertTrue(solution.passed)
        self.assertLessEqual(solution.update_count, 12)
        for value in solution.state.values():
            np.testing.assert_allclose(value, 1.0, atol=2.0e-12)

    def test_named_permutation_does_not_change_solution(self):
        first = linear_problem()
        second = linear_problem(
            state_order=["n3", "n1", "n2_6"],
            residual_order=["kelvin", "no_penetration", "ibl"],
        )
        solutions = []
        for group, initial, residual, jacobian, _ in (first, second):
            solutions.append(group.solve(
                initial_state=initial,
                residual_function=residual,
                jacobian_function=jacobian,
                normalized_tolerance=1.0e-12,
            ))
        for name in ("n1", "n2_6", "n3"):
            np.testing.assert_allclose(
                solutions[0].state[name],
                solutions[1].state[name],
                atol=2.0e-15,
            )

    def test_rank_deficiency_and_forbidden_residual_fail(self):
        with self.assertRaises(ImplicitClaimGroupError):
            ImplicitClaimGroup(
                state_block_sizes={"n1": 1},
                residual_block_sizes={"target_lift": 1},
                residual_physical_scales={"target_lift": 1.0},
            )
        group = ImplicitClaimGroup(
            state_block_sizes={"n1": 1},
            residual_block_sizes={"no_penetration": 1},
            residual_physical_scales={"no_penetration": 1.0},
        )
        with self.assertRaises(ImplicitClaimGroupError):
            group.solve(
                initial_state={"n1": np.array([0.0])},
                residual_function=lambda state: {
                    "no_penetration": np.array([1.0])
                },
                jacobian_function=lambda state: {
                    ("no_penetration", "n1"): np.array([[0.0]])
                },
            )


if __name__ == "__main__":
    unittest.main()

