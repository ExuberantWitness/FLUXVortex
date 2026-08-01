import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_finite_angle_sheet_guard import run as run_guard  # noqa: E402
from claim_runtime.finite_angle_sheet_formation import (  # noqa: E402
    FiniteAngleSheetError,
    finite_angle_sheet_formation,
)


class ActualBoundaryFiniteAngleSheetTests(unittest.TestCase):
    def test_symmetric_case_identifies_only_bisector_and_zero_strength(self):
        case = finite_angle_sheet_formation(
            u1_plus=-1.0,
            u2_minus=-1.0,
            wedge_angle_deg=40.0,
        )
        self.assertFalse(case.state_identifiable)
        self.assertIsNone(case.relative_velocity)
        self.assertAlmostEqual(
            case.delta_theta1, 0.5 * case.wedge_angle_rad
        )
        self.assertAlmostEqual(case.sheet_strength, 0.0)
        self.assertAlmostEqual(case.circulation_rate, 0.0)

    def test_nondegenerate_state_closes_all_equations(self):
        case = finite_angle_sheet_formation(
            u1_plus=-2.0,
            u2_minus=-1.0,
            wedge_angle_deg=40.0,
        )
        self.assertTrue(case.state_identifiable)
        self.assertGreaterEqual(case.u_g_plus, 0.0)
        self.assertGreaterEqual(case.u_g_minus, 0.0)
        self.assertLess(case.normalized_direction_residual, 1.0e-13)
        self.assertLess(
            case.normalized_circulation_rate_residual, 1.0e-13
        )
        self.assertLess(case.normalized_momentum_residual, 1.0e-13)

    def test_invalid_shedding_sign_is_rejected(self):
        with self.assertRaises(FiniteAngleSheetError):
            finite_angle_sheet_formation(
                u1_plus=1.0,
                u2_minus=-1.0,
                wedge_angle_deg=40.0,
            )

    def test_preregistered_guard_passes_without_production_activation(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
