import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.coupled_lesp_dde import (  # noqa: E402
    solve_coupled_lesp_dde_stage,
)
from claim_runtime.hirato_equations import (  # noqa: E402
    lesp_eq6,
)


class CoupledLespDDETests(unittest.TestCase):
    def test_affine_bound_reaction_closes_lesp_and_linear_system(self):
        ns = 3
        aic = np.array(
            [
                [2.0, 0.1, 0.0, 0.0],
                [0.2, 1.8, 0.1, 0.0],
                [0.0, 0.1, 1.7, 0.2],
                [0.0, 0.0, 0.2, 1.5],
            ]
        )
        influence = np.array(
            [
                [0.7, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.1, 0.9],
                [0.2, 0.1, 0.3],
            ]
        )
        chord = np.ones(ns)
        delta_x = np.full(ns, 0.25)
        scale = lesp_eq6(
            np.ones(ns),
            1.0,
            chord,
            delta_x,
        )
        desired_pre = np.array([0.34, -0.32, 0.20])
        bound_pre = np.array(
            [
                desired_pre[0] / scale[0],
                desired_pre[1] / scale[1],
                desired_pre[2] / scale[2],
                0.05,
            ]
        )
        result = solve_coupled_lesp_dde_stage(
            aic=aic,
            rhs_without_nascent_lev=aic @ bound_pre,
            unit_lev_normal_influence=influence,
            u_infinity=1.0,
            chord=chord,
            delta_x_front=delta_x,
            lesp_crit=0.27,
        )
        np.testing.assert_array_equal(
            result.active,
            [True, True, False],
        )
        np.testing.assert_allclose(
            result.a0_post[:2],
            [0.27, -0.27],
            atol=2.0e-14,
        )
        self.assertTrue(result.passed)

    def test_no_active_strip_leaves_bound_solution_unchanged(self):
        result = solve_coupled_lesp_dde_stage(
            aic=np.eye(2),
            rhs_without_nascent_lev=[0.01, -0.01],
            unit_lev_normal_influence=np.eye(2),
            u_infinity=2.0,
            chord=[1.0, 1.0],
            delta_x_front=[0.25, 0.25],
            lesp_crit=0.27,
        )
        np.testing.assert_allclose(result.bound_pre, result.bound_post)
        np.testing.assert_allclose(result.gamma_lev, 0.0)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
