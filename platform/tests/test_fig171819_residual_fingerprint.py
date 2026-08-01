import json
import sys
import unittest
from pathlib import Path


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_residual_fingerprint as fingerprint  # noqa: E402


ARTIFACT = (
    PLATFORM
    / "docs"
    / "diag"
    / "baseline_residual_fingerprint_v41_confirmed42_20260729_125402.json"
)
ARTIFACT_SHA256 = (
    "5a51f5f6460fea8e39c8c87df008a2760c23afbf81331fa4cc992af14fcdecbe"
)


class Fig171819ResidualFingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        cls.report = fingerprint.build_fingerprint(
            cls.artifact,
            input_path=ARTIFACT,
            expected_input_sha256=ARTIFACT_SHA256,
        )

    def test_exact_confirmed_contract_and_no_premature_claim(self):
        report = self.report
        self.assertEqual(report["status"], "DESCRIPTIVE_FINGERPRINT_COMPLETE")
        self.assertTrue(all(report["validity_gates"].values()))
        self.assertEqual(len(report["official_curves"]), 42)
        self.assertEqual(len(report["samples"]), 434)
        self.assertEqual(len(report["physical_curve_families"]), 34)
        self.assertEqual(len(report["duplicate_aliases"]), 8)
        self.assertEqual(
            report["claim_attribution"],
            {
                "decision": "NO_DECISION",
                "reason": "NODE_ATTRIBUTION_REQUIRED",
                "required_next_evidence": (
                    "For each preregistered witness contrast, record wind-axis "
                    "L/T contributions from N1/N2/N3/N4/N6, graph numerical "
                    "reduction, ledger guards, and claim graph identity."
                ),
            },
        )

    def test_samples_preserve_measurements_and_solver_brackets(self):
        self.assertFalse(
            any(
                sample["curve"].startswith(("19|c|", "19|d|"))
                for sample in self.report["samples"]
            )
        )
        for sample in self.report["samples"]:
            self.assertAlmostEqual(
                sample["model_N"] - sample["measurement_N"],
                sample["error_N"],
            )
            self.assertAlmostEqual(
                sample["left_weight"] + sample["right_weight"],
                1.0,
            )
            self.assertGreaterEqual(sample["left_weight"], 0.0)
            self.assertGreaterEqual(sample["right_weight"], 0.0)

    def test_duplicate_curves_do_not_inflate_physical_family_count(self):
        alias_keys = {
            tuple(alias["official_curve_keys"])
            for alias in self.report["duplicate_aliases"]
        }
        self.assertIn(("17|a|2.0", "18|c|(8.0, 2.0)"), alias_keys)
        self.assertIn(("18|a|8.0", "19|a|5"), alias_keys)
        self.assertEqual(
            sum(
                family["n_official_curves"]
                for family in self.report["physical_curve_families"]
            ),
            42,
        )

    def test_family_equal_strata_merge_aliases_before_weighting(self):
        figure_channel = {
            (row["figure"], row["channel"]): row
            for row in self.report["strata"][
                "physical_family_equal_figure_channel"
            ]
        }
        self.assertEqual(
            {
                key: (
                    row["n_physical_families"],
                    row["n_official_curves"],
                )
                for key, row in figure_channel.items()
            },
            {
                ("17", "L"): (5, 5),
                ("17", "T"): (5, 5),
                ("18", "L"): (12, 12),
                ("18", "T"): (12, 12),
                ("19", "L"): (4, 4),
                ("19", "T"): (4, 4),
            },
        )

        abscissa_channel = {
            (row["abscissa"], row["channel"]): row
            for row in self.report["strata"][
                "physical_family_equal_abscissa_channel"
            ]
        }
        self.assertEqual(
            {
                key: (
                    row["n_physical_families"],
                    row["n_official_curves"],
                )
                for key, row in abscissa_channel.items()
            },
            {
                ("frequency_Hz", "L"): (6, 7),
                ("frequency_Hz", "T"): (6, 7),
                ("twist_deg", "L"): (11, 14),
                ("twist_deg", "T"): (11, 14),
            },
        )
        self.assertEqual(
            sum(
                row["n_physical_families"]
                for row in abscissa_channel.values()
            ),
            34,
        )
        self.assertEqual(
            sum(row["n_official_curves"] for row in abscissa_channel.values()),
            42,
        )

    def test_witness_plan_is_inside_confirmed_solver_support(self):
        confirmed_keys = {
            key
            for sample in self.report["samples"]
            for key in (
                sample["left_condition_key"],
                sample["right_condition_key"],
            )
        }
        witness_keys = set(
            self.report["witness_plan"]["unique_solver_condition_keys"]
        )
        self.assertTrue(witness_keys)
        self.assertLessEqual(witness_keys, confirmed_keys)

    def test_preregistered_input_hash_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "validity gates failed"):
            fingerprint.build_fingerprint(
                self.artifact,
                input_path=ARTIFACT,
                expected_input_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
