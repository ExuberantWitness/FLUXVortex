import unittest
from pathlib import Path

import yaml


class HiratoCanonicalCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "diag"
            / "hirato_canonical_cases.yaml"
        )
        cls.spec = yaml.safe_load(path.read_text())

    def test_shared_identity_is_frozen_from_paper(self):
        shared = self.spec["shared"]
        self.assertEqual(shared["airfoil"], "SD7003")
        self.assertEqual(shared["chord_reynolds"], 20000)
        self.assertAlmostEqual(shared["lesp_crit"], 0.27)
        self.assertAlmostEqual(shared["aspect_ratio"], 6.0)
        self.assertAlmostEqual(shared["reduced_pitch_rate_K"], 0.3)

    def test_case1_and_case2_onset_are_distinct_field_guards(self):
        c1 = self.spec["cases"]["case1"]["expected_onset"]
        c2 = self.spec["cases"]["case2"]["expected_onset"]
        self.assertEqual(c1["topology"], "root-first")
        self.assertEqual(c2["topology"], "midspan-first")
        self.assertAlmostEqual(c1["tstar"], 1.710)
        self.assertAlmostEqual(c2["tstar"], 1.575)
        self.assertAlmostEqual(c1["abs_y_over_halfspan"], 0.0)
        self.assertAlmostEqual(c2["abs_y_over_halfspan"], 0.5)

    def test_conservation_guards_are_stricter_than_field_tolerances(self):
        guards = self.spec["preregistered_numerical_guards"]
        self.assertLessEqual(guards["conservation"]["eq9_relative_residual"], 1e-10)
        self.assertLessEqual(
            guards["conservation"]["eq17_recomposition_abs_residual"],
            1e-12,
        )
        self.assertGreater(guards["onset_tstar_abs_tolerance"], 0.0)
        self.assertGreater(
            guards["onset_span_abs_y_over_halfspan_tolerance"],
            0.0,
        )

    def test_first_ring_candidates_are_preregistered_without_force_selection(self):
        registration = self.spec["first_ring_preregistration"]
        self.assertEqual(
            registration["historical_full_freestream_step"]["state"],
            "falsified",
        )
        self.assertEqual(set(registration["candidates"]), {"P-A", "P-R"})
        self.assertEqual(
            registration["core_sensitivity"]["rc_over_ell_min"],
            [0.10, 0.25, 0.49],
        )
        self.assertTrue(
            all(
                candidate["state"] == "partial"
                for candidate in registration["candidates"].values()
            )
        )
        prohibited = " ".join(registration["guards"]["no_go"]).lower()
        self.assertIn("target-force", prohibited)
        self.assertIn("v4.1 cache", prohibited)

    def test_read_only_sensitivity_preserves_registered_failure(self):
        observed = self.spec["current_read_only_observation"][
            "complete_symmetry_sensitivity"
        ]
        guards = self.spec["preregistered_numerical_guards"]
        records = observed["records"]
        self.assertEqual(len(records), 12)
        self.assertEqual(sum(row["gate"] == "pass" for row in records), 11)
        failure = [row for row in records if row["gate"] == "fail"]
        self.assertEqual(len(failure), 1)
        self.assertEqual(
            (failure[0]["case"], failure[0]["nc"], failure[0]["steps"]),
            ("case1", 4, 240),
        )
        expected = self.spec["cases"]["case1"]["expected_onset"]["tstar"]
        self.assertGreater(
            abs(failure[0]["tstar"] - expected),
            guards["onset_tstar_abs_tolerance"],
        )


if __name__ == "__main__":
    unittest.main()
