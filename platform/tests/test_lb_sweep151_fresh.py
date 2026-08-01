from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import lb_sweep151_fresh as runner  # noqa: E402


def _guards(error: float = 0.0, tolerance: float = 1.0e-9) -> dict:
    return {
        "force_ledger": {
            "passed": True,
            "max_abs_error_N": error,
            "tolerance_N": tolerance,
        },
        "unclassified_force": {
            "passed": True,
            "max_abs_error_N": error,
            "tolerance_N": tolerance,
            "body_force_N": [0.0, 0.0, 0.0],
        },
        "unclassified_physical_force": {
            "passed": True,
            "max_abs_error_N": error,
            "tolerance_N": tolerance,
            "body_force_N": [0.0, 0.0, 0.0],
        },
        "cycle_reduction": {
            "passed": True,
            "max_abs_error_N": error,
            "tolerance_N": tolerance,
            "body_force_N": [0.0, 0.0, 0.0],
        },
        "aero_output_invariance": {
            "passed": True,
            "max_abs_error_N": error,
            "tolerance_N": tolerance,
            "checked_fields": ["L_wind", "T_wind"],
            "changed_fields": [],
        },
    }


def _manifest(guards: dict) -> dict:
    topology = ["N1", "N2", "N3", "N4", "N5", "N6", "R0"]
    return {
        "closure": "v41",
        "topology": topology,
        "nodes": [
            {
                "id": node_id,
                "state": "partial",
                "freeze": node_id in {"N1", "N4"},
                "runtime_role": "diagnostic" if node_id in {"N4", "N5", "R0"} else "physics",
                "implementation": f"test.{node_id}",
                "implementation_version": "test-v1",
                "implementation_hash": f"sha256:{index + 1:064x}",
            }
            for index, node_id in enumerate(topology)
        ],
        "parameter_sources": {"closure": "explicit"},
        "guards": copy.deepcopy(guards),
    }


def _contributions(body_force: tuple[float, float, float]) -> dict:
    contributions: dict[str, list[dict]] = {}
    for node_id, inventory in runner.EXPECTED_CONTRIBUTION_INVENTORY.items():
        contributions[node_id] = []
        for index, (channel, role) in enumerate(inventory):
            force = body_force if node_id == "N1" and index == 0 else (0.0, 0.0, 0.0)
            contributions[node_id].append(
                {
                    "channel": channel,
                    "role": role,
                    "body_force": list(force),
                    "metadata": {},
                }
            )
    return contributions


def _fixture(
    condition: tuple[float, float, float, float] = runner.ANCHOR,
) -> dict:
    key = runner.condition_key(condition)
    guards = _guards()
    common_manifest = _manifest(_guards(error=5.0e-13))
    graph_identity = runner.witness._claim_graph_identity_sha256(common_manifest)
    contributions = _contributions((-1.0, 0.0, 2.0))
    summary, ledger_total = runner.witness._contribution_summary(
        contributions,
        aoa_deg=condition[3],
    )
    wind = runner.witness._wind_force(ledger_total, condition[3])
    result = {"L": wind["L_N"], "T": wind["T_N"]}
    old_baseline = {
        key: {
            "L": result["L"] - 0.1,
            "T": result["T"] + 0.2,
        }
    }
    expected_call = {
        "U": condition[0],
        "freq": condition[1],
        "twist_amp_deg": condition[2] / 2.0,
        "aoa_deg": condition[3],
        "closure": "v41",
    }
    evidence = {
        "condition_key": key,
        "condition": runner._condition_record(condition),
        "resolved_call": copy.deepcopy(expected_call),
        "old_baseline": {
            "L_N": old_baseline[key]["L"],
            "T_N": old_baseline[key]["T"],
        },
        "signed_old_baseline_delta_N": {
            "L_N": result["L"] - old_baseline[key]["L"],
            "T_N": result["T"] - old_baseline[key]["T"],
        },
        "claim_graph_identity_sha256": graph_identity,
        "claim_manifest_sha256": runner._case_claim_manifest_hash(
            common_manifest,
            guards,
        ),
        "claim_guards": guards,
        "claim_contributions": contributions,
        "contribution_summary": summary,
        "recomputed_ledger": {
            "total_body_force_N": ledger_total.tolist(),
            "total_wind_force": wind,
            "max_body_error_N": 0.0,
            "max_wind_error_N": 0.0,
        },
        "wall_s": 1.0,
    }
    return {
        "key": key,
        "condition": condition,
        "result": result,
        "evidence": evidence,
        "manifest_guard": guards,
        "expected_call": expected_call,
        "graph_identity": graph_identity,
        "common_manifest": common_manifest,
        "old_baseline": old_baseline,
    }


def _error(fixture: dict) -> str | None:
    return runner._case_validation_error(
        key=fixture["key"],
        condition=fixture["condition"],
        result=fixture["result"],
        evidence=fixture["evidence"],
        manifest_guard=fixture["manifest_guard"],
        expected_call=fixture["expected_call"],
        graph_identity=fixture["graph_identity"],
        common_claim_manifest=fixture["common_manifest"],
        old_baseline=fixture["old_baseline"],
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_gpu_result(aoa_deg: float) -> dict:
    guards = _guards()
    contributions = _contributions((-1.0, 0.0, 2.0))
    wind = runner.witness._wind_force((-1.0, 0.0, 2.0), aoa_deg)
    return {
        "L_wind": wind["L_N"],
        "T_wind": wind["T_N"],
        "Fx_body": -1.0,
        "Fz_body": 2.0,
        "claim_guards": guards,
        "claim_manifest": _manifest(guards),
        "claim_contributions": contributions,
    }


class Fresh151SavedCaseTests(unittest.TestCase):
    def test_valid_saved_case_passes_with_case_specific_guards(self) -> None:
        fixture = _fixture()
        self.assertNotEqual(
            fixture["common_manifest"]["guards"],
            fixture["evidence"]["claim_guards"],
        )
        self.assertIsNone(_error(fixture))

    def test_rejects_tampered_lift_or_thrust_with_cached_zero_error(self) -> None:
        for field, value in (("L", 999999.0), ("T", -999999.0)):
            with self.subTest(field=field):
                fixture = _fixture()
                fixture["result"][field] = value
                self.assertIn("delta mismatch", _error(fixture))

    def test_rejects_bool_string_nan_inf_and_extra_result_fields(self) -> None:
        for field, value in (
            ("L", True),
            ("L", "1.2"),
            ("L", float("nan")),
            ("T", float("inf")),
        ):
            with self.subTest(field=field, value=value):
                fixture = _fixture()
                fixture["result"][field] = value
                self.assertIsNotNone(_error(fixture))
        fixture = _fixture()
        fixture["result"]["Fx"] = 0.0
        self.assertIn("exactly", _error(fixture))

    def test_rejects_empty_missing_or_extra_contribution_nodes(self) -> None:
        mutations = (
            lambda value: value.clear(),
            lambda value: value.pop("N2"),
            lambda value: value.__setitem__("N5", []),
        )
        for mutate in mutations:
            fixture = _fixture()
            mutate(fixture["evidence"]["claim_contributions"])
            self.assertIsNotNone(_error(fixture))

    def test_rejects_channel_role_duplicate_and_cross_node_rewire(self) -> None:
        fixture = _fixture()
        fixture["evidence"]["claim_contributions"]["N1"][0]["channel"] = "profile_drag"
        self.assertIn("inventory", _error(fixture))

        fixture = _fixture()
        fixture["evidence"]["claim_contributions"]["N2"][0]["role"] = "diagnostic"
        self.assertIn("inventory", _error(fixture))

        fixture = _fixture()
        fixture["evidence"]["claim_contributions"]["N1"].append(
            copy.deepcopy(fixture["evidence"]["claim_contributions"]["N1"][0])
        )
        self.assertIn("inventory", _error(fixture))

    def test_rejects_non_numeric_or_extra_raw_force_fields(self) -> None:
        for value in (True, "1.0"):
            fixture = _fixture()
            fixture["evidence"]["claim_contributions"]["N1"][0]["body_force"][0] = value
            self.assertIn("strict finite", _error(fixture))
        fixture = _fixture()
        fixture["evidence"]["claim_contributions"]["N1"][0]["unexpected"] = 1
        self.assertIn("schema", _error(fixture))

    def test_rejects_forged_summary_or_cached_ledger(self) -> None:
        fixture = _fixture()
        fixture["evidence"]["contribution_summary"]["N1"]["total_body_force_N"][0] += 1.0
        self.assertIn("summary", _error(fixture))

        fixture = _fixture()
        fixture["evidence"]["recomputed_ledger"]["total_wind_force"]["L_N"] += 1.0
        self.assertIn("stored wind", _error(fixture))

    def test_rejects_raw_force_mutation_when_cached_diagnostics_are_reforged(self) -> None:
        fixture = _fixture()
        contributions = fixture["evidence"]["claim_contributions"]
        contributions["N1"][0]["body_force"][0] += 1.0
        summary, ledger = runner.witness._contribution_summary(
            contributions,
            aoa_deg=fixture["condition"][3],
        )
        fixture["evidence"]["contribution_summary"] = summary
        fixture["evidence"]["recomputed_ledger"] = {
            "total_body_force_N": ledger.tolist(),
            "total_wind_force": runner.witness._wind_force(
                ledger,
                fixture["condition"][3],
            ),
            "max_body_error_N": 0.0,
            "max_wind_error_N": 0.0,
        }
        self.assertIn("does not close", _error(fixture))

    def test_rejects_cross_wired_condition_resolved_call_and_evidence(self) -> None:
        fixture = _fixture()
        fixture["evidence"]["condition"]["U_m_s"] = 10.0
        self.assertIn("condition mismatch", _error(fixture))

        fixture = _fixture()
        fixture["evidence"]["resolved_call"]["freq"] = 1.4
        self.assertIn("resolved-call", _error(fixture))

        first = _fixture(runner.ANCHOR)
        second = _fixture((8.0, 2.6, 22.5, 5.0))
        first["evidence"] = second["evidence"]
        self.assertIsNotNone(_error(first))

    def test_rejects_guard_error_over_tolerance_nonfinite_or_negative(self) -> None:
        for error, tolerance in (
            (2.0, 1.0),
            (float("nan"), 1.0),
            (0.0, -1.0),
        ):
            fixture = _fixture()
            guard = fixture["evidence"]["claim_guards"]["force_ledger"]
            guard["max_abs_error_N"] = error
            guard["tolerance_N"] = tolerance
            fixture["manifest_guard"] = fixture["evidence"]["claim_guards"]
            self.assertIn("guards", _error(fixture))

    def test_rejects_manifest_guard_hash_and_graph_identity_drift(self) -> None:
        fixture = _fixture()
        fixture["manifest_guard"] = _guards(error=1.0e-13)
        self.assertIn("guard mismatch", _error(fixture))

        fixture = _fixture()
        fixture["evidence"]["claim_manifest_sha256"] = "f" * 64
        self.assertIn("manifest hash", _error(fixture))

        fixture = _fixture()
        fixture["graph_identity"] = "f" * 64
        self.assertIn("graph identity", _error(fixture))


class Fresh151CheckpointTests(unittest.TestCase):
    def test_orphan_in_each_file_is_purged_from_all(self) -> None:
        fixture = _fixture()
        key = fixture["key"]
        for missing in ("result", "evidence", "guard"):
            with self.subTest(missing=missing):
                results = {key: copy.deepcopy(fixture["result"])}
                evidence = {key: copy.deepcopy(fixture["evidence"])}
                guards = {key: copy.deepcopy(fixture["manifest_guard"])}
                {"result": results, "evidence": evidence, "guard": guards}[
                    missing
                ].pop(key)
                discarded = runner._sanitize_resume_checkpoint(
                    results=results,
                    case_evidence=evidence,
                    case_guards=guards,
                    expected_calls={key: fixture["expected_call"]},
                    graph_identity=fixture["graph_identity"],
                    common_claim_manifest=fixture["common_manifest"],
                    old_baseline=fixture["old_baseline"],
                )
                self.assertEqual(discarded[key], "result/evidence/guard orphan")
                self.assertEqual(results, {})
                self.assertEqual(evidence, {})
                self.assertEqual(guards, {})

    def test_tampered_case_is_purged_from_all(self) -> None:
        fixture = _fixture()
        key = fixture["key"]
        fixture["result"]["L"] = 999999.0
        results = {key: fixture["result"]}
        evidence = {key: fixture["evidence"]}
        guards = {key: fixture["manifest_guard"]}
        discarded = runner._sanitize_resume_checkpoint(
            results=results,
            case_evidence=evidence,
            case_guards=guards,
            expected_calls={key: fixture["expected_call"]},
            graph_identity=fixture["graph_identity"],
            common_claim_manifest=fixture["common_manifest"],
            old_baseline=fixture["old_baseline"],
        )
        self.assertIn(key, discarded)
        self.assertEqual((results, evidence, guards), ({}, {}, {}))

    def test_equal_counts_with_different_key_sets_fail_completion(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "three-file key-set mismatch"):
            runner._validate_complete_checkpoint(
                results={"a": {"L": 1.0, "T": 2.0}},
                case_evidence={"b": {}},
                case_guards={"c": {}},
                expected_calls={},
                graph_identity=None,
                common_claim_manifest=None,
                old_baseline={},
            )

    def test_solver_source_add_remove_or_modify_fails_snapshot(self) -> None:
        expected = {"platform/a.py": "a" * 64}
        baseline = {"authoritative_source_hashes": {}}
        for actual in (
            {},
            {"platform/a.py": "b" * 64},
            {
                "platform/a.py": "a" * 64,
                "platform/b.py": "b" * 64,
            },
        ):
            with self.subTest(actual=actual):
                with mock.patch.object(
                    runner.witness,
                    "_validate_solver_sources",
                    return_value=actual,
                ):
                    with self.assertRaisesRegex(RuntimeError, "source-set drift"):
                        runner._validate_solver_sources(expected, baseline)

    def test_loaded_unregistered_or_modified_project_source_fails(self) -> None:
        with mock.patch.object(
            runner,
            "_snapshot_loaded_project_sources",
            return_value={"platform/unregistered.py": "a" * 64},
        ):
            with self.assertRaisesRegex(RuntimeError, "loaded source closure"):
                runner._validate_loaded_source_closure({}, {})
        with mock.patch.object(
            runner,
            "_snapshot_loaded_project_sources",
            return_value={"platform/a.py": "b" * 64},
        ):
            with self.assertRaisesRegex(RuntimeError, "loaded source closure"):
                runner._validate_loaded_source_closure(
                    {"platform/a.py": "a" * 64},
                    {},
                )

    def test_anchor_nogo_record_contains_both_attempts(self) -> None:
        manifest = {"failures": {}}
        error = RuntimeError("anchor failed")
        runner._record_anchor_failure(
            manifest=manifest,
            error=error,
            cold_value={"L": 1.0, "T": 2.0},
            cold_evidence={"claim_guards": _guards()},
            warm_value={"L": 3.0, "T": 4.0},
            warm_evidence={"claim_guards": _guards()},
        )
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("anchor failed", manifest["failures"]["__session_anchor__"])
        self.assertEqual(manifest["anchor_nogo"]["warm_value"]["L"], 3.0)

    def test_runtime_or_call_contract_drift_fails_resume(self) -> None:
        identity = {
            "python_executable": "/env/python",
            "warp_device": "cuda:0",
            "solver_dtype": "tracked_source_default",
        }
        manifest = {
            "runtime_identity": copy.deepcopy(identity),
            "resolved_call_contract_sha256": "a" * 64,
        }
        runner._validate_runtime_and_call_contract(
            manifest,
            runtime_identity=identity,
            call_contract_sha256="a" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "runtime identity drift"):
            runner._validate_runtime_and_call_contract(
                manifest,
                runtime_identity={**identity, "warp_device": "cuda:1"},
                call_contract_sha256="a" * 64,
            )
        with self.assertRaisesRegex(RuntimeError, "resolved-call contract drift"):
            runner._validate_runtime_and_call_contract(
                manifest,
                runtime_identity=identity,
                call_contract_sha256="b" * 64,
            )

    def test_strict_guard_rejects_bool_or_string_numeric_fields(self) -> None:
        for value in (True, "0.0"):
            guards = _guards()
            guards["force_ledger"]["max_abs_error_N"] = value
            self.assertFalse(runner._valid_guards_strict(guards))


class Fresh151MockCampaignTests(unittest.TestCase):
    def test_full_fresh_campaign_and_complete_resume_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="fresh151-mock-",
            dir=runner.ROOT,
        ) as directory:
            root = Path(directory)
            result_path = root / "result.json"
            manifest_path = root / "manifest.json"
            contributions_path = root / "contributions.json"
            old_baseline_path = root / "old_baseline.json"
            old_seed_path = root / "old_seed.json"
            baseline_manifest_path = root / "baseline_manifest.json"
            lock_path = root / "campaign.lock"
            timestamp = "20990101_000000"

            old_baseline = {}
            for condition in runner.CONDITIONS:
                key = runner.condition_key(condition)
                value = _fake_gpu_result(condition[3])
                old_baseline[key] = {
                    "L": value["L_wind"],
                    "T": value["T_wind"],
                }
            first_key = sorted(old_baseline)[0]
            old_seed = {first_key: old_baseline[first_key]}
            runner._write_json_atomic(old_baseline_path, old_baseline)
            runner._write_json_atomic(old_seed_path, old_seed)
            runner._write_json_atomic(baseline_manifest_path, {})

            calls: list[dict] = []

            def fake_gpu_run_twist(**kwargs):
                calls.append(dict(kwargs))
                return _fake_gpu_result(float(kwargs["aoa_deg"]))

            fake_warp = types.ModuleType("warp")
            fake_warp.__version__ = "test"
            fake_warp.init = lambda: None
            fake_robo = types.ModuleType("_v2_robo")
            fake_robo.gpu_run_twist = fake_gpu_run_twist
            solver_sources = {"platform/fake_solver.py": "a" * 64}
            control_sources = {"platform/fake_control.py": "b" * 64}
            runtime_identity = {
                "solver_config": {
                    "dtype_name": "float64",
                    "device": "cuda:0",
                },
                "warp_device": {
                    "name": "mock GPU",
                    "arch": 89,
                },
            }

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(runner, "OLD_MERGED_BASELINE", old_baseline_path)
                )
                stack.enter_context(
                    mock.patch.object(runner, "OLD_SEED", old_seed_path)
                )
                stack.enter_context(
                    mock.patch.object(
                        runner.witness,
                        "BASELINE_MANIFEST",
                        baseline_manifest_path,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner.witness,
                        "BASELINE_RESULT_SHA256",
                        _sha256(old_baseline_path),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner.witness,
                        "BASELINE_MANIFEST_SHA256",
                        _sha256(baseline_manifest_path),
                    )
                )
                stack.enter_context(mock.patch.object(runner, "RUN_LOCK", lock_path))
                stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_paths",
                        return_value=(
                            result_path,
                            manifest_path,
                            contributions_path,
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner.witness,
                        "_validate_solver_sources",
                        return_value=solver_sources,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_snapshot_control_sources",
                        return_value=control_sources,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_validate_execution_sources",
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_runtime_identity",
                        return_value=runtime_identity,
                    )
                )
                stack.enter_context(
                    mock.patch.dict(
                        sys.modules,
                        {"warp": fake_warp, "_v2_robo": fake_robo},
                    )
                )

                with redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.run(timestamp=timestamp), 0)
                self.assertEqual(len(calls), 152)
                self.assertEqual(len(json.loads(result_path.read_text())), 151)
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["completed_condition_count"], 151)
                self.assertEqual(
                    manifest["result_sha256"],
                    _sha256(result_path),
                )
                self.assertEqual(
                    manifest["contributions_sha256"],
                    _sha256(contributions_path),
                )

                call_count = len(calls)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        runner.run(timestamp=timestamp, resume=True),
                        0,
                    )
                self.assertEqual(len(calls), call_count)


if __name__ == "__main__":
    unittest.main()
