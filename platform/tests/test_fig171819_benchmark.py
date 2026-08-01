import json
import sys
import unittest
from collections import Counter
from pathlib import Path


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_benchmark as benchmark  # noqa: E402


class Fig171819BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repro = json.loads((PLATFORM / "docs" / "repro_data.json").read_text())
        cls.v41 = json.loads((PLATFORM / "docs" / "s6_sweep_v41.json").read_text())
        cls.measurements = benchmark.load_measurements()

    def test_complete_solver_contract_has_50_curves_and_184_conditions(self):
        self.assertEqual(len(benchmark.CURVES), 50)
        self.assertEqual(len(benchmark.CONDITIONS), 184)
        self.assertEqual(
            Counter(curve.figure for curve in benchmark.CURVES),
            {"17": 10, "18": 24, "19": 16},
        )
        confirmed_curves = benchmark.CURVES_BY_EVIDENCE_SCOPE[
            benchmark.EVIDENCE_CONFIRMED
        ]
        conditional_curves = benchmark.CURVES_BY_EVIDENCE_SCOPE[
            benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
        ]
        confirmed_conditions = set(
            benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[
                benchmark.EVIDENCE_CONFIRMED
            ]
        )
        conditional_conditions = set(
            benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[
                benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
            ]
        )
        self.assertEqual(len(confirmed_curves), 42)
        self.assertEqual(len(conditional_curves), 8)
        self.assertEqual(len(confirmed_conditions), 151)
        self.assertEqual(len(conditional_conditions), 48)
        self.assertEqual(
            len(confirmed_conditions & conditional_conditions), 15
        )
        self.assertEqual(
            len(conditional_conditions - confirmed_conditions), 33
        )
        self.assertEqual(
            {curve.key for curve in conditional_curves},
            {
                f"19|{panel}|{aoa:g}"
                for panel in ("c", "d")
                for aoa in benchmark.AOAS
            },
        )

    def test_data_md_contract_has_all_50_curves_and_530_raw_samples(self):
        report = benchmark.validate_measurement_contract(self.measurements)
        self.assertTrue(report["passed"], msg=report)
        self.assertEqual(report["actual_curve_count"], 50)
        self.assertEqual(report["actual_measurement_samples"], 530)
        self.assertEqual(report["channel_sample_counts"], {"T": 265, "L": 265})
        self.assertEqual(
            report["evidence_scope_sample_counts"],
            {
                benchmark.EVIDENCE_CONFIRMED: 434,
                benchmark.EVIDENCE_CONDITIONAL_FIG19_CD: 96,
            },
        )
        self.assertEqual(
            report["source"]["sha256"],
            benchmark.FROZEN_DATA_MD_SHA256,
        )
        self.assertEqual(report["source"]["path"], "platform/docs/data.md")
        self.assertEqual(report["endpoint_projection_candidates"], 18)

    def test_raw_measurement_spot_checks_and_evidence_based_identity(self):
        fig17 = self.measurements["17|a|1.4"]
        self.assertEqual(fig17.x[1], 4.875)
        self.assertAlmostEqual(fig17.values_g[0], -208.457711442786)
        self.assertAlmostEqual(
            fig17.values_N[0],
            -208.457711442786 * benchmark.GRAM_FORCE_TO_NEWTON,
        )

        fig19 = self.measurements["19|d|15"]
        self.assertAlmostEqual(fig19.x[-1], 44.9323308270677)
        self.assertAlmostEqual(fig19.values_g[-1], 1279.78723404255)

        # data.md has the old U labels; the physical identity follows the
        # source-PDF legend and is corrected in memory without rewriting data.
        u6 = self.measurements["18|a|6.0"]
        self.assertEqual(u6.source_key, "18|a|10.0")
        self.assertAlmostEqual(u6.values_g[0], -116.243654822335)
        self.assertIsNotNone(u6.identity_correction)

    def test_fig18_panel_channel_identity_follows_columns_not_captions(self):
        for key, curve in self.measurements.items():
            if key.startswith("18|c|"):
                self.assertEqual(curve.channel, "T")
                self.assertEqual(curve.abscissa, "twist_deg")
            if key.startswith("18|d|"):
                self.assertEqual(curve.channel, "L")
                self.assertEqual(curve.abscissa, "twist_deg")
        self.assertEqual(
            {
                curve.channel
                for curve in benchmark.CURVES
                if curve.panel == "c" and curve.figure == "18"
            },
            {"T"},
        )
        self.assertEqual(
            {
                curve.channel
                for curve in benchmark.CURVES
                if curve.panel == "d" and curve.figure == "18"
            },
            {"L"},
        )

    def test_model_is_interpolated_to_measurement_x(self):
        model, provenance = benchmark.interpolate_model_to_measurement_x(
            [1.4, 1.7, 2.0, 2.3, 2.6],
            [14.0, 17.0, 20.0, 23.0, 26.0],
            [1.5, 2.5],
            abscissa="frequency_Hz",
            curve_key="synthetic",
        )
        self.assertEqual(model.tolist(), [15.0, 25.0])
        self.assertEqual(provenance["direction"], "model_to_measurement_x")
        self.assertFalse(provenance["measurement_values_interpolated"])
        self.assertEqual(provenance["boundary_projection_count"], 0)

        report = benchmark.scorecard(
            self.v41,
            self.repro,
            sweep_name="s6_sweep_v41.json",
        )
        row = next(item for item in report["rows"] if item["curve"] == "18|a|8.0")
        self.assertEqual(row["expected_solver_points"], 5)
        self.assertEqual(row["measurement_points"], 7)
        self.assertAlmostEqual(row["measurement_x"][1], 1.50044052863436)
        self.assertEqual(
            row["interpolation"]["direction"],
            "model_to_measurement_x",
        )
        self.assertFalse(report["contract"]["measurement_values_interpolated"])

    def test_legacy_repro_argument_is_not_used_as_ground_truth(self):
        with_repro = benchmark.scorecard(self.v41, self.repro)
        without_repro = benchmark.scorecard(self.v41, {})
        self.assertEqual(
            with_repro["aggregates"],
            without_repro["aggregates"],
        )
        self.assertEqual(
            with_repro["contract"]["measurement_validation"]["source"],
            without_repro["contract"]["measurement_validation"]["source"],
        )

    def test_material_out_of_domain_measurement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside model interpolation domain"):
            benchmark.interpolate_model_to_measurement_x(
                benchmark.FS,
                [0.0] * len(benchmark.FS),
                [1.38, 2.6],
                abscissa="frequency_Hz",
                curve_key="outside",
            )

        # A digitizer endpoint offset inside the declared tolerance is
        # projected and recorded, not extrapolated.
        model, provenance = benchmark.interpolate_model_to_measurement_x(
            benchmark.FS,
            benchmark.FS,
            [1.39454545454545, 2.6],
            abscissa="frequency_Hz",
            curve_key="endpoint-jitter",
        )
        self.assertAlmostEqual(model[0], 1.4)
        self.assertEqual(provenance["boundary_projection_count"], 1)
        self.assertEqual(provenance["boundary_projections"][0]["evaluation_x"], 1.4)

    def test_missing_measurement_curve_is_rejected(self):
        incomplete = dict(self.measurements)
        incomplete.pop("19|d|15")
        validation = benchmark.validate_measurement_contract(incomplete)
        self.assertFalse(validation["passed"])
        self.assertEqual(validation["missing_curve_keys"], ["19|d|15"])
        with self.assertRaisesRegex(ValueError, "raw measurement contract"):
            benchmark.scorecard(
                self.v41,
                measurements=incomplete,
            )

    def test_frozen_118_coverage_and_scoped_scores(self):
        report = benchmark.coverage(self.v41)
        self.assertEqual(report["valid_unique_conditions"], 118)
        self.assertEqual(report["missing_unique_conditions"], 66)
        self.assertEqual(report["complete_curves"], 38)
        self.assertEqual(
            {
                figure: values["complete_curves"]
                for figure, values in report["figures"].items()
            },
            {"17": 10, "18": 12, "19": 16},
        )
        self.assertFalse(report["complete"])

        confirmed_coverage = benchmark.coverage(
            self.v41,
            evidence_scope=benchmark.EVIDENCE_CONFIRMED,
        )
        conditional_coverage = benchmark.coverage(
            self.v41,
            evidence_scope=benchmark.EVIDENCE_CONDITIONAL_FIG19_CD,
        )
        self.assertEqual(
            (
                confirmed_coverage["valid_unique_conditions"],
                confirmed_coverage["expected_unique_conditions"],
                confirmed_coverage["complete_curves"],
                confirmed_coverage["partial_curves"],
            ),
            (85, 151, 30, 12),
        )
        self.assertEqual(
            (
                conditional_coverage["valid_unique_conditions"],
                conditional_coverage["expected_unique_conditions"],
                conditional_coverage["complete_curves"],
                conditional_coverage["partial_curves"],
            ),
            (48, 48, 8, 0),
        )

        score = benchmark.scorecard(
            self.v41,
            self.repro,
            sweep_name="s6_sweep_v41.json",
        )
        self.assertEqual(score["schema_version"], 3)
        self.assertEqual(
            score["primary_evidence_scope"],
            benchmark.EVIDENCE_CONFIRMED,
        )
        self.assertEqual(score["aggregates"]["ALL"]["ALL"]["n_points"], 290)
        self.assertAlmostEqual(
            score["aggregates"]["ALL"]["L"]["mae_N"],
            0.8976018362015258,
        )
        self.assertAlmostEqual(
            score["aggregates"]["ALL"]["T"]["mae_N"],
            1.661049059699497,
        )
        self.assertAlmostEqual(
            score["aggregates"]["ALL"]["ALL"]["mae_N"],
            1.2793254479505114,
        )
        self.assertEqual(
            score["evidence_scopes"][
                benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
            ]["aggregates"]["ALL"]["ALL"]["n_points"],
            96,
        )
        self.assertEqual(
            score["diagnostic_all_scopes_not_for_claim_decision"][
                "aggregates"
            ]["ALL"]["ALL"]["n_points"],
            386,
        )
        self.assertFalse(score["promotion_eligible"])
        self.assertIn("solver_grid_incomplete", score["promotion_blockers"])

    def test_missing_66_are_only_fig18_cd_u6_u10_twist_points(self):
        report = benchmark.coverage(self.v41)
        signatures = Counter()
        for key in report["missing_condition_keys"]:
            U, freq, twist, aoa = map(float, key.split("_"))
            self.assertIn(U, (6.0, 10.0))
            self.assertIn(freq, benchmark.FIG18_TWIST_FREQS)
            self.assertNotEqual(twist, 22.5)
            self.assertEqual(aoa, 5.0)
            signatures[(U, freq)] += 1
        self.assertEqual(
            signatures,
            Counter(
                {
                    (U, freq): 11
                    for U in (6.0, 10.0)
                    for freq in benchmark.FIG18_TWIST_FREQS
                }
            ),
        )

    def test_full184_primary_rows_and_aggregates_are_confirmed_only(self):
        full = {
            benchmark.condition_key(condition): {"L": 0.0, "T": 0.0}
            for condition in benchmark.CONDITIONS
        }
        score = benchmark.scorecard(full)
        confirmed = score["evidence_scopes"][
            benchmark.EVIDENCE_CONFIRMED
        ]
        conditional = score["evidence_scopes"][
            benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
        ]
        self.assertEqual(len(score["rows"]), 42)
        self.assertEqual(
            score["aggregates"]["ALL"]["ALL"]["n_points"], 434
        )
        self.assertEqual(
            score["aggregates"]["ALL"]["ALL"]["n_complete_curves"], 42
        )
        self.assertFalse(
            any(
                row["curve"].startswith(("19|c|", "19|d|"))
                for row in score["rows"]
            )
        )
        self.assertEqual(
            (
                confirmed["coverage"]["valid_unique_conditions"],
                confirmed["coverage"]["expected_unique_conditions"],
                confirmed["coverage"]["complete_curves"],
            ),
            (151, 151, 42),
        )
        self.assertEqual(
            conditional["aggregates"]["ALL"]["ALL"]["n_points"], 96
        )
        self.assertEqual(
            (
                conditional["coverage"]["valid_unique_conditions"],
                conditional["coverage"]["expected_unique_conditions"],
                conditional["coverage"]["complete_curves"],
            ),
            (48, 48, 8),
        )
        self.assertTrue(confirmed["residual_fingerprint_ready"])
        self.assertFalse(conditional["residual_fingerprint_ready"])

    def test_conditional_only_perturbation_cannot_change_confirmed_metrics(self):
        baseline = {
            benchmark.condition_key(condition): {"L": 0.0, "T": 0.0}
            for condition in benchmark.CONDITIONS
        }
        perturbed = {
            key: dict(value) for key, value in baseline.items()
        }
        confirmed_conditions = set(
            benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[
                benchmark.EVIDENCE_CONFIRMED
            ]
        )
        conditional_conditions = set(
            benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[
                benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
            ]
        )
        conditional_only = conditional_conditions - confirmed_conditions
        self.assertEqual(len(conditional_only), 33)
        for condition in conditional_only:
            perturbed[benchmark.condition_key(condition)] = {
                "L": 1.0e6,
                "T": -1.0e6,
            }

        baseline_score = benchmark.scorecard(baseline)
        perturbed_score = benchmark.scorecard(perturbed)
        self.assertEqual(
            baseline_score["rows"],
            perturbed_score["rows"],
        )
        self.assertEqual(
            baseline_score["aggregates"],
            perturbed_score["aggregates"],
        )
        self.assertNotEqual(
            baseline_score["evidence_scopes"][
                benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
            ]["aggregates"],
            perturbed_score["evidence_scopes"][
                benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
            ]["aggregates"],
        )

    def test_shared_conditions_are_not_globally_conditional(self):
        confirmed = set(
            benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[
                benchmark.EVIDENCE_CONFIRMED
            ]
        )
        conditional = set(
            benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[
                benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
            ]
        )
        shared = confirmed & conditional
        self.assertEqual(len(shared), 15)
        self.assertEqual(
            {
                condition
                for condition in shared
                if condition[3] == 5.0
                and condition[1] == 2.6
            },
            {
                (8.0, 2.6, twist, 5.0)
                for twist in benchmark.TWS
            },
        )

    def test_confirmed_artifact_contains_no_conditional_curve_rows(self):
        full = {
            benchmark.condition_key(condition): {"L": 0.0, "T": 0.0}
            for condition in benchmark.CONDITIONS
        }
        report = benchmark.scorecard(full)
        artifact = benchmark.build_evidence_scope_artifact(
            report,
            provenance={"test": {"sha256": "0" * 64}},
        )
        self.assertEqual(artifact["status"], "ready_for_baseline_diagnosis")
        self.assertEqual(len(artifact["curve_rows"]), 42)
        self.assertEqual(artifact["residual_point_count"], 434)
        self.assertEqual(len(artifact["excluded_curve_keys"]), 8)
        self.assertEqual(
            artifact["coverage"]["valid_unique_conditions"], 151
        )
        self.assertFalse(artifact["global_promotion_eligible"])
        self.assertTrue(
            all(
                row["evidence_scope"] == benchmark.EVIDENCE_CONFIRMED
                for row in artifact["curve_rows"]
            )
        )

    def test_fig19_cd_frequency_assumption_blocks_promotion_even_at_184(self):
        full = {
            benchmark.condition_key(condition): {"L": 0.0, "T": 0.0}
            for condition in benchmark.CONDITIONS
        }
        report = benchmark.scorecard(full)
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(
            report["contract"]["fig19_cd_frequency"]["status"],
            "unresolved",
        )
        self.assertEqual(
            report["contract"]["fig19_cd_frequency"]["conditional_assumption_Hz"],
            2.6,
        )
        self.assertEqual(
            report["contract"]["status"],
            "conditional_diagnostic",
        )
        self.assertFalse(report["promotion_eligible"])
        self.assertEqual(
            report["promotion_blockers"],
            ["fig19_cd_fixed_frequency_unresolved"],
        )


if __name__ == "__main__":
    unittest.main()
