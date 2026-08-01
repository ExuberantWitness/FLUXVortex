from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_benchmark as benchmark  # noqa: E402
import score_n3_shadow_gates as scorer  # noqa: E402


def guarded_record(L: float, T: float, **extra):
    return {
        "L": float(L),
        "T": float(T),
        "L_wind_v41_counterfactual": float(L),
        "T_wind_v41_counterfactual": float(T),
        "claim_guards": {
            name: {"passed": True}
            for name in scorer.campaign_runner._N3_ONLY_REQUIRED_GUARDS
        },
        "claim_manifest": {
            "closure": scorer.CLOSURE,
            "internal_stages": [
                {
                    "id": scorer.campaign_runner.N3_ONLY_CLAIM,
                    "runtime_owner": "N3",
                    "runtime_binding": "internal_stage",
                }
            ],
        },
        "n3_spatial_n3only": {
            "closure": scorer.CLOSURE,
            "claim_node": scorer.campaign_runner.N3_ONLY_CLAIM,
        },
        **extra,
    }


class N3ShadowGateScorerTests(unittest.TestCase):
    def test_frozen_scope_shapes(self):
        self.assertEqual(len(scorer.SMOKE3), 3)
        self.assertEqual(len(scorer.REPRESENTATIVE32), 32)
        self.assertEqual(len(scorer.REPRESENTATIVE_COMPLETE_CURVES), 12)
        self.assertEqual(len(scorer.SLOPE_WITNESS_CURVES), 6)

    def test_refinement_is_normalized_to_fine_member(self):
        report = scorer._force_refinement_report(
            {"case": {"L": 10.0, "T": -2.0}},
            {"case": {"L": 8.0, "T": -2.5}},
            ("case",),
        )
        rows = {row["channel"]: row for row in report["rows"]}
        self.assertAlmostEqual(rows["L"]["relative_to_fine_percent"], 25.0)
        self.assertAlmostEqual(rows["T"]["relative_to_fine_percent"], 20.0)
        self.assertAlmostEqual(report["max_relative_percent"], 25.0)
        vector = report["wind_axis_vector_norm"]
        expected = (
            math.hypot(10.0 - 8.0, -2.0 - -2.5)
            / math.hypot(8.0, -2.5)
            * 100.0
        )
        self.assertAlmostEqual(
            vector["rows"][0]["relative_to_fine_percent"],
            expected,
        )
        self.assertAlmostEqual(vector["max_relative_percent"], expected)

    def test_force_pair_requires_nonempty_all_pass_guards(self):
        valid = guarded_record(1.0, 2.0)
        invalid_records = []
        missing_all = dict(valid)
        missing_all.pop("claim_guards")
        invalid_records.append(missing_all)
        empty = dict(valid, claim_guards={})
        invalid_records.append(empty)
        failed = dict(valid)
        failed["claim_guards"] = dict(valid["claim_guards"])
        failed["claim_guards"]["force_ledger"] = {"passed": False}
        invalid_records.append(failed)
        malformed = dict(valid)
        malformed["claim_guards"] = dict(valid["claim_guards"])
        malformed["claim_guards"]["force_ledger"] = {}
        invalid_records.append(malformed)
        non_mapping = dict(valid)
        non_mapping["claim_guards"] = dict(valid["claim_guards"])
        non_mapping["claim_guards"]["force_ledger"] = True
        invalid_records.append(non_mapping)
        for required_guard in (
            scorer.campaign_runner._N3_ONLY_REQUIRED_GUARDS
        ):
            missing_one = dict(valid)
            missing_one["claim_guards"] = dict(valid["claim_guards"])
            missing_one["claim_guards"].pop(required_guard)
            invalid_records.append(missing_one)
        for record in invalid_records:
            with self.subTest(record=record):
                with self.assertRaisesRegex(
                    ValueError,
                    "claim_guards|claim guards|required claim guards",
                ):
                    scorer._valid_force_pair(record, context="test")
        self.assertEqual(
            scorer._valid_force_pair(
                guarded_record(1.0, 2.0),
                context="test",
            ),
            (1.0, 2.0),
        )

    def test_force_pair_rejects_bare_or_unbound_claim_records(self):
        valid = guarded_record(1.0, 2.0)
        bare = {"L": 1.0, "T": 2.0}
        missing_manifest = dict(valid)
        missing_manifest.pop("claim_manifest")
        wrong_owner = dict(valid)
        wrong_owner["claim_manifest"] = {
            "closure": scorer.CLOSURE,
            "internal_stages": [
                {
                    "id": scorer.campaign_runner.N3_ONLY_CLAIM,
                    "runtime_owner": "N2",
                    "runtime_binding": "internal_stage",
                }
            ],
        }
        for record in (bare, missing_manifest, wrong_owner):
            with self.subTest(record=record):
                with self.assertRaisesRegex(
                    ValueError,
                    "strict campaign schema",
                ):
                    scorer._valid_force_pair(record, context="test")

    def test_same_call_counterfactual_is_atomic(self):
        with self.assertRaisesRegex(ValueError, "complete same-call"):
            scorer._same_call_counterfactual_pair(
                {
                    "L": 1.0,
                    "T": 2.0,
                    "L_wind_v41_counterfactual": 3.0,
                },
                context="condition",
            )

    def test_new_slope_reversal_requires_counterfactual_to_have_matched(self):
        base = {
            "curve": "witness",
            "measurement_x": [0.0, 1.0, 2.0],
            "measurement_N": [0.0, 1.0, 2.0],
        }
        candidate = dict(
            base,
            model_at_measurement_x_N=[2.0, 1.0, 0.0],
        )
        counterfactual = dict(
            base,
            model_at_measurement_x_N=[0.0, 1.0, 2.0],
        )
        report = scorer._slope_witness_report(
            {"witness": candidate},
            {"witness": counterfactual},
            witness_keys=("witness",),
        )
        self.assertEqual(report["new_reversal_curves"], ["witness"])
        counterfactual["model_at_measurement_x_N"] = [3.0, 2.0, 1.0]
        report = scorer._slope_witness_report(
            {"witness": candidate},
            {"witness": counterfactual},
            witness_keys=("witness",),
        )
        self.assertEqual(report["new_reversal_curves"], [])

    def test_equal_candidate_and_counterfactual_fails_strict_improvement(self):
        results = {}
        for condition in scorer.REPRESENTATIVE32:
            key = benchmark.condition_key(condition)
            results[key] = guarded_record(
                1.0,
                -1.0,
                L_wind_v41_counterfactual=1.0,
                T_wind_v41_counterfactual=-1.0,
            )
        bundle = self._write_bundle(
            results,
            scope="representative32",
            conditions=scorer.REPRESENTATIVE32,
        )
        report = scorer.score_g2(bundle)
        self.assertFalse(report["passed"])
        self.assertFalse(
            report["gates"]["overall_point_weighted_MAE_strictly_lower"]
        )
        self.assertEqual(report["candidate"]["ALL"]["curve_count"], 12)
        self.assertEqual(
            report["candidate"]["ALL"]["measurement_point_count"],
            104,
        )
        self.assertEqual(
            report["candidate"],
            report["same_call_v41_counterfactual"],
        )

    def test_g1_applies_both_channel_time_gates(self):
        q16_results = {}
        q24_results = {}
        dt_half_results = {}
        for index, condition in enumerate(scorer.SMOKE3, start=1):
            key = benchmark.condition_key(condition)
            q16_results[key] = guarded_record(
                10.0 + index,
                -5.0 - index,
            )
            q24_results[key] = guarded_record(
                (10.0 + index) * 1.001,
                (-5.0 - index) * 1.001,
            )
            dt_half_results[key] = dict(q16_results[key])
        high_key = benchmark.condition_key(scorer.SMOKE3[-1])
        dt_half_results[high_key] = guarded_record(
            q16_results[high_key]["L"] * 1.02,
            q16_results[high_key]["T"] * 0.98,
        )
        q16 = self._write_bundle(
            q16_results,
            scope="smoke3",
            conditions=scorer.SMOKE3,
        )
        q24 = self._write_bundle(
            q24_results,
            scope="smoke3",
            conditions=scorer.SMOKE3,
            model_args={"spatial_p2_quadrature": 24},
            candidate_id=f"{scorer.CANDIDATE_ID}_q24",
        )
        dt_half = self._write_bundle(
            dt_half_results,
            scope="smoke3",
            conditions=scorer.SMOKE3,
            grid={
                "mode": "quick",
                "nc": 4,
                "ns": 8,
                "n_cycle": 2,
                "steps_per_cycle": 120,
                "wake_rows": 120,
            },
        )
        report = scorer.score_g1(q16, q24, dt_half)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["gates"].values()))

    def test_quadrature_failure_is_terminal_without_dt_half(self):
        q16_results = {}
        q24_results = {}
        for condition in scorer.SMOKE3:
            key = benchmark.condition_key(condition)
            q16_results[key] = guarded_record(10.0, -5.0)
            q24_results[key] = guarded_record(9.9, -4.95)
        q16 = self._write_bundle(
            q16_results,
            scope="smoke3",
            conditions=scorer.SMOKE3,
        )
        q24 = self._write_bundle(
            q24_results,
            scope="smoke3",
            conditions=scorer.SMOKE3,
            model_args={"spatial_p2_quadrature": 24},
        )
        stage = scorer.score_g1_quadrature(q16, q24)
        self.assertFalse(stage["passed"])
        self.assertTrue(stage["terminal_nogo"])
        self.assertEqual(stage["decision"], "TERMINAL_NO_GO")
        self.assertEqual(
            stage["time_family"]["status"],
            "not_run_by_preregistered_early_stop",
        )
        self.assertIsNone(stage["inputs"]["dt_half"])

        full = scorer.score_g1(q16, q24)
        self.assertTrue(full["terminal_nogo"])
        self.assertEqual(full["decision"], "TERMINAL_NO_GO")
        self.assertEqual(
            full["time_family"]["status"],
            "not_run_by_preregistered_early_stop",
        )

    def test_dt_half_is_required_only_after_quadrature_passes(self):
        q16_results = {}
        q24_results = {}
        for condition in scorer.SMOKE3:
            key = benchmark.condition_key(condition)
            q16_results[key] = guarded_record(10.0, -5.0)
            q24_results[key] = guarded_record(10.01, -5.005)
        q16 = self._write_bundle(
            q16_results,
            scope="smoke3",
            conditions=scorer.SMOKE3,
        )
        q24 = self._write_bundle(
            q24_results,
            scope="smoke3",
            conditions=scorer.SMOKE3,
            model_args={"spatial_p2_quadrature": 24},
        )
        stage = scorer.score_g1_quadrature(q16, q24)
        self.assertTrue(stage["passed"])
        self.assertFalse(stage["terminal_nogo"])
        self.assertEqual(stage["decision"], "GO_TO_G1_TIME")
        with self.assertRaisesRegex(ValueError, "--dt-half is now required"):
            scorer.score_g1(q16, q24)

    def test_parser_exposes_quadrature_only_early_stop_command(self):
        args = scorer.build_parser().parse_args(
            ["g1-quadrature", "--q16", "q16", "--q24", "q24"]
        )
        self.assertEqual(args.gate, "g1-quadrature")
        full = scorer.build_parser().parse_args(
            ["g1", "--q16", "q16", "--q24", "q24"]
        )
        self.assertIsNone(full.dt_half)

    def test_resolve_run_locks_and_caches_hashes_of_parsed_bytes(self):
        condition = scorer.SMOKE3[0]
        key = benchmark.condition_key(condition)
        fixture = self._write_bundle(
            {key: guarded_record(1.0, 2.0)},
            scope="smoke3",
            conditions=(condition,),
        )
        with scorer.campaign_runner._RunDirectoryLock(fixture.run_dir):
            with self.assertRaisesRegex(RuntimeError, "already locked"):
                scorer._resolve_run(fixture.run_dir)

        resolved = scorer._resolve_run(fixture.run_dir)
        identity_before_mutation = scorer._bundle_identity(resolved)
        resolved.results_path.write_text("{}\n", encoding="utf-8")
        identity_after_mutation = scorer._bundle_identity(resolved)
        self.assertEqual(
            identity_after_mutation["results_sha256"],
            identity_before_mutation["results_sha256"],
        )
        self.assertNotEqual(
            identity_after_mutation["results_sha256"],
            scorer._sha256(resolved.results_path),
        )
        self.assertIn(key, resolved.results)

    def _write_bundle(
        self,
        results,
        *,
        scope,
        conditions,
        grid=None,
        model_args=None,
        candidate_id=scorer.CANDIDATE_ID,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run_dir = Path(temporary.name)
        model_args = {} if model_args is None else dict(model_args)
        resolved = dict(model_args)
        config = {
            "run_identity": {
                "candidate_id": candidate_id,
                "closure": scorer.CLOSURE,
                "scope": scope,
                "condition_count": len(conditions),
                "condition_keys": [
                    benchmark.condition_key(condition) for condition in conditions
                ],
                "grid": grid
                or {
                    "mode": "quick",
                    "nc": 4,
                    "ns": 8,
                    "n_cycle": 2,
                    "steps_per_cycle": 60,
                    "wake_rows": 60,
                },
                "model_args": model_args,
                "resolved_model_config_before_closure_profile": resolved,
            }
        }
        status = {"status": "complete"}
        config_path = run_dir / "config.json"
        results_path = run_dir / "candidate_results.json"
        status_path = run_dir / "status.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        results_path.write_text(json.dumps(results), encoding="utf-8")
        status_path.write_text(json.dumps(status), encoding="utf-8")
        return scorer.RunBundle(
            run_dir=run_dir,
            config_path=config_path,
            results_path=results_path,
            status_path=status_path,
            config=config,
            results=results,
            status=status,
            artifact_sha256={
                "config": scorer._sha256(config_path),
                "results": scorer._sha256(results_path),
                "status": scorer._sha256(status_path),
            },
        )


if __name__ == "__main__":
    unittest.main()
