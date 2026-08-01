from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from analyze_n26e1b_corner_scaling import analyze


class N26E1BCornerScalingDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = analyze(
            PLATFORM
            / "docs"
            / "diag"
            / "n26e1b1_source_faithful_te_refinement_result_20260730.json"
        )

    def test_geometry_owned_exponent_matches_frozen_mean_trace(self) -> None:
        predicted = self.result["geometry"][
            "post_kutta_regular_velocity_exponent"
        ]
        observed = self.result["traces"]["mean"][
            "interval_exponents_64_128_and_128_256"
        ][1]
        self.assertAlmostEqual(predicted, 0.06068033385533589, places=14)
        self.assertLess(abs(predicted - observed), 0.005)

    def test_predicted_and_observed_changes_are_distinct_recorded_values(self) -> None:
        predicted = self.result["geometry"][
            "predicted_raw_trace_change_128_256"
        ]
        observed = self.result["traces"]["mean"][
            "raw_changes_64_128_and_128_256"
        ][1]
        self.assertAlmostEqual(predicted, 0.08775765251729473, places=14)
        self.assertAlmostEqual(observed, 0.08200721928964151, places=14)

    def test_diagnosis_cannot_promote_claim_or_candidate(self) -> None:
        self.assertFalse(self.result["claim_state_change_allowed"])
        self.assertFalse(self.result["candidate_promotion_allowed"])
        self.assertEqual(
            self.result["role"],
            "post_result_diagnosis_only",
        )

    def test_jump_is_not_hidden_by_regular_mode_normalization(self) -> None:
        jump_change = self.result["traces"]["jump"][
            "modal_coefficient_change_128_256"
        ]
        mean_change = self.result["traces"]["mean"][
            "modal_coefficient_change_128_256"
        ]
        self.assertGreater(jump_change, 0.10)
        self.assertLess(mean_change, 0.02)


if __name__ == "__main__":
    unittest.main()
