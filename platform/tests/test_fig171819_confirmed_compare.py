from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_confirmed_compare as compare  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guards(
    *,
    passed: bool = True,
    force_error: float = 0.0,
    unclassified_error: float = 0.0,
    unclassified_physical_error: float = 0.0,
    cycle_error: float = 0.0,
    aero_error: float = 0.0,
) -> dict:
    return {
        "force_ledger": {
            "passed": passed,
            "max_abs_error_N": force_error,
            "tolerance_N": 1.0e-9,
        },
        "unclassified_force": {
            "passed": passed,
            "max_abs_error_N": unclassified_error,
            "tolerance_N": 1.0e-9,
            "body_force_N": [unclassified_error, 0.0, 0.0],
        },
        "unclassified_physical_force": {
            "passed": passed,
            "max_abs_error_N": unclassified_physical_error,
            "tolerance_N": 1.0e-9,
            "body_force_N": [unclassified_physical_error, 0.0, 0.0],
        },
        "cycle_reduction": {
            "passed": passed,
            "max_abs_error_N": cycle_error,
            "tolerance_N": 1.0e-12,
            "body_force_N": [0.0, 0.0, 0.0],
        },
        "aero_output_invariance": {
            "passed": passed,
            "max_abs_error_N": aero_error,
            "tolerance_N": 0.0,
            "checked_fields": list(compare.AUTHORIZED_AERO_OUTPUT_FIELDS),
            "changed_fields": [],
        },
    }


def _claim_manifest(guards: dict) -> dict:
    manifest = copy.deepcopy(compare.AUTHORIZED_V41_GRAPH_CONTRACT)
    manifest["guards"] = copy.deepcopy(guards)
    return manifest


def _contributions(
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    output: dict[str, list[dict]] = {}
    for node_id, inventory in compare.EXPECTED_CONTRIBUTION_INVENTORY.items():
        output[node_id] = []
        for index, (channel, role) in enumerate(inventory):
            force = body_force if node_id == "N1" and index == 0 else (0.0, 0.0, 0.0)
            output[node_id].append(
                {
                    "channel": channel,
                    "role": role,
                    "body_force": list(force),
                    "metadata": {},
                }
            )
    return output


def _fake_source_hashes(prefix: str, count: int) -> dict[str, str]:
    return {
        f"platform/{prefix}_{index:03d}.py": f"{index + 1:064x}"
        for index in range(count)
    }


def _identity(label: str) -> dict[str, str]:
    return {"path": f"synthetic/{label}", "sha256": "a" * 64}


def _runtime_identity() -> dict:
    return {
        "python_executable": "/synthetic/python",
        "python_version": "3.synthetic",
        "python_implementation": "CPython",
        "platform": "synthetic-linux",
        "numpy_version": "synthetic",
        "warp_version": "synthetic",
        "solver_config": {
            "dtype_name": "float64",
            "dtype": "float64",
            "numpy_dtype": "float64",
            "device": "cuda:0",
            "CR_TOL": 1.0e-9,
            "NEWTON_TOL": 1.0e-9,
            "GEOM_ATOL": 1.0e-12,
            "PORT_ATOL": 1.0e-12,
        },
        "warp_device": {
            "text": "CUDA device synthetic",
            "alias": "cuda:0",
            "name": "synthetic GPU",
            "arch": 89,
        },
        "environment": {name: None for name in compare.RUNTIME_ENVIRONMENT_KEYS},
    }


def _write_authorization(
    path: Path,
    *,
    manifest: dict,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
) -> None:
    authorization = {
        "schema_version": 1,
        "artifact_type": "v41_fresh151_postprocess_authorization",
        "status": "AUTHORIZED_IDENTITY_ONLY_BEFORE_CAMPAIGN_COMPLETION",
        "run_id": manifest["run_id"],
        "scope": {
            "result_path": compare._display_path(result_path),
            "manifest_path": compare._display_path(manifest_path),
            "contributions_path": compare._display_path(contributions_path),
            "expected_condition_count": compare.EXPECTED_CONDITIONS,
            "confirmed_curve_count": compare.EXPECTED_CURVES,
            "confirmed_measurement_count": compare.EXPECTED_SAMPLES,
            "physical_family_count": compare.EXPECTED_PHYSICAL_FAMILIES,
            "alias_group_count": compare.EXPECTED_ALIAS_GROUPS,
        },
        "frozen_manifest_identity": {
            "solver_source_count": len(manifest["solver_source_hashes"]),
            "solver_source_hashes_canonical_sha256": compare._canonical_hash(
                manifest["solver_source_hashes"]
            ),
            "control_source_count": len(manifest["control_source_hashes"]),
            "control_source_hashes_canonical_sha256": compare._canonical_hash(
                manifest["control_source_hashes"]
            ),
            "runner_identity_canonical_sha256": compare._canonical_hash(
                manifest["runner"]
            ),
            "preregistration_identity_canonical_sha256": compare._canonical_hash(
                manifest["preregistration"]
            ),
            "integrity_addendum_identity_canonical_sha256": (
                compare._canonical_hash(manifest["integrity_addendum"])
            ),
            "launch_authorization_identity_canonical_sha256": (
                compare._canonical_hash(manifest["launch_authorization"])
            ),
            "expected_condition_keys_canonical_sha256": compare._canonical_hash(
                manifest["expected_condition_keys"]
            ),
            "resolved_call_contract_sha256": manifest["resolved_call_contract_sha256"],
            "runtime_identity_canonical_sha256": compare._canonical_hash(
                manifest["runtime_identity"]
            ),
            "claim_graph_identity_sha256": manifest["claim_graph_identity_sha256"],
            "common_claim_manifest_static_canonical_sha256": (
                compare._canonical_hash(
                    compare._claim_graph_static_payload(
                        manifest["common_claim_manifest"]
                    )
                )
            ),
        },
        "claim_graph_contract": copy.deepcopy(compare.AUTHORIZED_V41_GRAPH_CONTRACT),
        "guard_contract": copy.deepcopy(compare.AUTHORIZED_GUARD_CONTRACT),
        "frozen_postprocess_inputs": {
            "measurement_data_sha256": (compare.AUTHORIZED_MEASUREMENT_DATA_SHA256),
            "benchmark_source_sha256": (compare.AUTHORIZED_BENCHMARK_SOURCE_SHA256),
            "fingerprint_source_sha256": (compare.AUTHORIZED_FINGERPRINT_SOURCE_SHA256),
        },
        "interpretation": copy.deepcopy(compare.AUTHORIZED_INTERPRETATION),
    }
    _write_json(path, authorization)


class _CompletionSpy:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))
        if self.failure is not None:
            raise self.failure


def _build_complete_bundle(root: Path, *, status: str = "complete") -> dict[str, Path]:
    run_id = "20260729_000000"
    result_path = root / "fresh151.json"
    manifest_path = root / "fresh151_manifest.json"
    contributions_path = root / "fresh151_contributions.json"
    scoring_prereg_path = root / "postscore_prereg.md"
    authorization_path = root / "postprocess_authorization.json"
    scoring_prereg_path.write_text(
        "# Synthetic post-score preregistration\n", encoding="utf-8"
    )

    guards = _guards()
    common_manifest = _claim_manifest(_guards(cycle_error=5.0e-13))
    graph_identity = compare.witness._claim_graph_identity_sha256(common_manifest)
    case_manifest = dict(common_manifest)
    case_manifest["guards"] = copy.deepcopy(guards)
    case_manifest_sha256 = compare._canonical_hash(case_manifest)
    zero_contributions = _contributions()

    results: dict[str, dict[str, float]] = {}
    cases: dict[str, dict] = {}
    case_guards: dict[str, dict] = {}
    resolved_calls: dict[str, dict] = {}
    for key in sorted(compare.EXPECTED_KEYS):
        condition = compare.CONDITION_BY_KEY[key]
        summary, ledger_total = compare.witness._contribution_summary(
            zero_contributions,
            aoa_deg=condition[3],
        )
        wind = compare.witness._wind_force(ledger_total, condition[3])
        result = {"L": wind["L_N"], "T": wind["T_N"]}
        results[key] = result
        resolved_call = {
            "U": condition[0],
            "freq": condition[1],
            "twist_amp_deg": condition[2] / 2.0,
            "aoa_deg": condition[3],
            "closure": "v41",
            "flap_amp_deg": 22.5,
            "twist_phase_deg": 90.0,
            "nc": 4,
            "ns": 8,
            "n_cycle": 2,
            "steps_per_cycle": 240,
            "wake_rows": 60,
            "real_geom": True,
            "sym": True,
            "les_suction": True,
            "visc": True,
            "d_para": 0.5,
        }
        resolved_calls[key] = resolved_call
        cases[key] = {
            "condition_key": key,
            "condition": compare._condition_record(condition),
            "resolved_call": resolved_call,
            "old_baseline": {"L_N": 0.0, "T_N": 0.0},
            "signed_old_baseline_delta_N": {
                "L_N": result["L"],
                "T_N": result["T"],
            },
            "claim_graph_identity_sha256": graph_identity,
            "claim_manifest_sha256": case_manifest_sha256,
            "claim_guards": copy.deepcopy(guards),
            "claim_contributions": copy.deepcopy(zero_contributions),
            "contribution_summary": summary,
            "recomputed_ledger": {
                "total_body_force_N": ledger_total.tolist(),
                "total_wind_force": wind,
                "max_body_error_N": 0.0,
                "max_wind_error_N": 0.0,
            },
            "wall_s": 1.0,
        }
        case_guards[key] = copy.deepcopy(guards)

    contribution_file = {
        "schema_version": 2,
        "run_id": run_id,
        "result_path": compare._display_path(result_path),
        "manifest_path": compare._display_path(manifest_path),
        "contributions_path": compare._display_path(contributions_path),
        "cases": cases,
    }
    _write_json(result_path, results)
    _write_json(contributions_path, contribution_file)
    coverage = compare.benchmark.coverage(
        results,
        evidence_scope=compare.benchmark.EVIDENCE_CONFIRMED,
    )
    manifest = {
        "schema_version": 2,
        "status": status,
        "run_id": run_id,
        "preregistration": _identity("runner_prereg"),
        "integrity_addendum": _identity("integrity_addendum"),
        "launch_authorization": _identity("launch_authorization"),
        "runner": _identity("runner"),
        "solver_source_hashes": _fake_source_hashes(
            "solver", compare.EXPECTED_SOLVER_SOURCES
        ),
        "control_source_hashes": _fake_source_hashes(
            "control", compare.EXPECTED_CONTROL_SOURCES
        ),
        "expected_condition_keys": sorted(compare.EXPECTED_KEYS),
        "expected_condition_count": compare.EXPECTED_CONDITIONS,
        "completed_condition_count": compare.EXPECTED_CONDITIONS,
        "resolved_call_contract_sha256": compare._canonical_hash(resolved_calls),
        "runtime_identity": _runtime_identity(),
        "claim_graph_identity_sha256": graph_identity,
        "common_claim_manifest": common_manifest,
        "case_guards": case_guards,
        "failures": {},
        "result_path": compare._display_path(result_path),
        "manifest_path": compare._display_path(manifest_path),
        "contributions_path": compare._display_path(contributions_path),
        "final_confirmed_coverage": coverage,
        "result_sha256": _sha256(result_path),
        "contributions_sha256": _sha256(contributions_path),
    }
    anchor_key = compare.benchmark.condition_key((8.0, 2.6, 0.0, 5.0))
    anchor_value = copy.deepcopy(results[anchor_key])
    anchor_delta = copy.deepcopy(cases[anchor_key]["signed_old_baseline_delta_N"])
    manifest["cold_preconditioner"] = {
        "value": copy.deepcopy(anchor_value),
        "claim_graph_identity_sha256": graph_identity,
        "guards": copy.deepcopy(guards),
        "discarded_from_results": True,
    }
    manifest["formal_anchor"] = {
        "condition_key": anchor_key,
        "value": copy.deepcopy(anchor_value),
        "old_baseline_delta_N": copy.deepcopy(anchor_delta),
        "guards": copy.deepcopy(guards),
    }
    manifest["runtime_sessions"] = [
        {
            "started_at": "2026-07-29T00:00:00+08:00",
            "cold_preconditioner": {
                "value": copy.deepcopy(anchor_value),
                "graph_identity": graph_identity,
            },
            "warm_anchor": {
                "value": copy.deepcopy(anchor_value),
                "old_baseline_delta_N": copy.deepcopy(anchor_delta),
                "graph_identity": graph_identity,
            },
        }
    ]
    _write_json(manifest_path, manifest)
    _write_authorization(
        authorization_path,
        manifest=manifest,
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
    )
    return {
        "result": result_path,
        "manifest": manifest_path,
        "contributions": contributions_path,
        "prereg": scoring_prereg_path,
        "authorization": authorization_path,
    }


def _refresh_receipts(paths: dict[str, Path]) -> None:
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["result_sha256"] = _sha256(paths["result"])
    manifest["contributions_sha256"] = _sha256(paths["contributions"])
    _write_json(paths["manifest"], manifest)


def _build_baseline(
    paths: dict[str, Path],
    *,
    out_dir: Path,
    completion_validator: _CompletionSpy | None = None,
    expected_authorization_sha256: str | None = None,
) -> dict:
    validator = completion_validator or _CompletionSpy()
    return compare.build_baseline(
        result_path=paths["result"],
        manifest_path=paths["manifest"],
        contributions_path=paths["contributions"],
        scoring_prereg_path=paths["prereg"],
        authorization_path=paths["authorization"],
        out_dir=out_dir,
        completion_validator=validator,
        expected_authorization_path=paths["authorization"],
        expected_authorization_sha256=(
            expected_authorization_sha256
            if expected_authorization_sha256 is not None
            else _sha256(paths["authorization"])
        ),
        expected_prereg_path=paths["prereg"],
        expected_prereg_sha256=_sha256(paths["prereg"]),
    )


class ConfirmedCompareIntegrityTests(unittest.TestCase):
    def test_production_completion_validator_calls_exact_resume_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supplied = (
                root / "result.json",
                root / "manifest.json",
                root / "contributions.json",
            )
            run = mock.Mock(return_value=0)
            fake_runner = types.SimpleNamespace(
                _paths=lambda run_id: supplied,
                run=run,
            )
            with mock.patch.dict(
                sys.modules,
                {"lb_sweep151_fresh": fake_runner},
            ):
                compare._production_completion_validator(
                    run_id="20260729_000000",
                    result_path=supplied[0],
                    manifest_path=supplied[1],
                    contributions_path=supplied[2],
                )
            run.assert_called_once_with(
                timestamp="20260729_000000",
                resume=True,
            )

    def test_production_completion_validator_rejects_wrong_runner_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supplied = (
                root / "result.json",
                root / "manifest.json",
                root / "contributions.json",
            )
            run = mock.Mock(return_value=0)
            fake_runner = types.SimpleNamespace(
                _paths=lambda run_id: (
                    root / "wrong-result.json",
                    supplied[1],
                    supplied[2],
                ),
                run=run,
            )
            with mock.patch.dict(
                sys.modules,
                {"lb_sweep151_fresh": fake_runner},
            ):
                with self.assertRaisesRegex(
                    compare.BaselineContractError,
                    "not the frozen runner path set",
                ):
                    compare._production_completion_validator(
                        run_id="20260729_000000",
                        result_path=supplied[0],
                        manifest_path=supplied[1],
                        contributions_path=supplied[2],
                    )
            run.assert_not_called()

    def test_frozen_postprocess_hash_literals_and_files_are_exact(self) -> None:
        expected = {
            Path(compare.benchmark.DEFAULT_DATA_MD).resolve(): (
                "ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1"
            ),
            Path(compare.benchmark.__file__).resolve(): (
                # 2026-08-01 双 scope 契约终裁后版本（FIG19_CD_FREQUENCY_STATUS=conditional_scope）
                "45b93584550eea4b16969381c59fe1038f68b21cf7cd1371052a19315c2444da"
            ),
            Path(compare.residual_fingerprint.__file__).resolve(): (
                "127db39b6028f1be676a10f95dc932f35e29402fd590dc979d97e269f4bc14e8"
            ),
        }
        self.assertEqual(
            compare.AUTHORIZED_MEASUREMENT_DATA_SHA256,
            expected[Path(compare.benchmark.DEFAULT_DATA_MD).resolve()],
        )
        self.assertEqual(
            compare.AUTHORIZED_BENCHMARK_SOURCE_SHA256,
            expected[Path(compare.benchmark.__file__).resolve()],
        )
        self.assertEqual(
            compare.AUTHORIZED_FINGERPRINT_SOURCE_SHA256,
            expected[Path(compare.residual_fingerprint.__file__).resolve()],
        )
        for path, digest in expected.items():
            with self.subTest(path=path):
                self.assertEqual(_sha256(path), digest)

    def test_authorized_graph_rejects_self_consistent_non_v41_variants(self) -> None:
        for mutation in ("v4_legacy", "NX"):
            with self.subTest(mutation=mutation):
                manifest = _claim_manifest(_guards())
                if mutation == "v4_legacy":
                    manifest["closure"] = "v4_legacy"
                else:
                    manifest["topology"].append("NX")
                    manifest["nodes"].append(
                        {
                            "id": "NX",
                            "state": "partial",
                            "freeze": False,
                            "runtime_role": "physics",
                            "implementation": "synthetic:NX",
                            "implementation_version": "1",
                            "implementation_hash": f"sha256:{'f' * 64}",
                        }
                    )
                self_consistent_identity = compare.witness._claim_graph_identity_sha256(
                    manifest
                )
                with self.assertRaisesRegex(
                    compare.BaselineContractError,
                    "not the authorized V4.1 graph|not authorized V4.1",
                ):
                    compare._validate_authorized_graph(
                        manifest,
                        self_consistent_identity,
                    )

    def test_guard_tolerance_99_of_100_attack_is_rejected(self) -> None:
        guards = _guards()
        guards["force_ledger"]["max_abs_error_N"] = 99.0e-9
        guards["force_ledger"]["tolerance_N"] = 100.0e-9
        with self.assertRaisesRegex(
            compare.BaselineContractError,
            "differs from authorized V4.1",
        ):
            compare._validate_guard_set(guards, label="99-of-100 attack")

    def test_guard_body_force_must_reproduce_reported_error(self) -> None:
        guards = _guards(unclassified_error=1.0e-10)
        guards["unclassified_force"]["body_force_N"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(
            compare.BaselineContractError,
            "body-force maximum differs",
        ):
            compare._validate_guard_set(guards, label="body-force attack")

    def test_running_bundle_is_rejected_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root, status="running")
            out_dir = root / "out"
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "not complete",
            ):
                _build_baseline(paths, out_dir=out_dir)
            self.assertFalse(out_dir.exists())

    def test_authorization_rejects_source_mapping_drift_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            source = next(iter(sorted(manifest["solver_source_hashes"])))
            manifest["solver_source_hashes"][source] = "f" * 64
            _write_json(paths["manifest"], manifest)
            spy = _CompletionSpy()
            out_dir = root / "out"
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "identity differs from postprocess authorization",
            ):
                _build_baseline(
                    paths,
                    out_dir=out_dir,
                    completion_validator=spy,
                )
            self.assertEqual(spy.calls, [])
            self.assertFalse(out_dir.exists())

    def test_authorization_rejects_runtime_identity_drift_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["runtime_identity"]["python_version"] = "drifted"
            _write_json(paths["manifest"], manifest)
            spy = _CompletionSpy()
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "identity differs from postprocess authorization",
            ):
                _build_baseline(
                    paths,
                    out_dir=root / "out",
                    completion_validator=spy,
                )
            self.assertEqual(spy.calls, [])

    def test_authorization_rejects_guard_contract_drift_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            authorization = json.loads(
                paths["authorization"].read_text(encoding="utf-8")
            )
            authorization["guard_contract"]["force_ledger"]["tolerance_N"] = 100.0e-9
            _write_json(paths["authorization"], authorization)
            spy = _CompletionSpy()
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "graph/guard contract differs",
            ):
                _build_baseline(
                    paths,
                    out_dir=root / "out",
                    completion_validator=spy,
                )
            self.assertEqual(spy.calls, [])

    def test_authorization_file_hash_is_a_root_of_trust(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            authorized_sha256 = _sha256(paths["authorization"])
            paths["authorization"].write_text(
                paths["authorization"].read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            spy = _CompletionSpy()
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "authorization SHA-256 mismatch",
            ):
                _build_baseline(
                    paths,
                    out_dir=root / "out",
                    completion_validator=spy,
                    expected_authorization_sha256=authorized_sha256,
                )
            self.assertEqual(spy.calls, [])

    def test_completion_validator_failure_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            failure = RuntimeError("synthetic complete-resume failure")
            spy = _CompletionSpy(failure=failure)
            out_dir = root / "out"
            with self.assertRaisesRegex(
                RuntimeError,
                "complete-resume failure",
            ):
                _build_baseline(
                    paths,
                    out_dir=out_dir,
                    completion_validator=spy,
                )
            self.assertEqual(len(spy.calls), 1)
            self.assertEqual(
                spy.calls[0]["run_id"],
                "20260729_000000",
            )
            self.assertFalse(out_dir.exists())

    def test_complete_resume_may_not_mutate_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)

            def mutate_manifest(**kwargs: object) -> None:
                manifest_path = Path(kwargs["manifest_path"])
                manifest_path.write_text(
                    manifest_path.read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "changed the immutable fresh151 triplet",
            ):
                compare.build_baseline(
                    result_path=paths["result"],
                    manifest_path=paths["manifest"],
                    contributions_path=paths["contributions"],
                    scoring_prereg_path=paths["prereg"],
                    authorization_path=paths["authorization"],
                    out_dir=root / "out",
                    completion_validator=mutate_manifest,
                    expected_authorization_path=paths["authorization"],
                    expected_authorization_sha256=_sha256(paths["authorization"]),
                    expected_prereg_path=paths["prereg"],
                    expected_prereg_sha256=_sha256(paths["prereg"]),
                )
            self.assertFalse((root / "out").exists())

    def test_runtime_identity_and_session_shells_are_rejected(self) -> None:
        for mutation, pattern in (
            ("identity", "runtime identity schema mismatch"),
            ("session", "runtime session 0: schema mismatch"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    paths = _build_complete_bundle(root)
                    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                    if mutation == "identity":
                        manifest["runtime_identity"] = {"device": "synthetic"}
                    else:
                        manifest["runtime_sessions"] = [{"session": "synthetic"}]
                    _write_json(paths["manifest"], manifest)
                    with self.assertRaisesRegex(
                        compare.BaselineContractError,
                        pattern,
                    ):
                        compare.validate_fresh151_bundle(
                            result_path=paths["result"],
                            manifest_path=paths["manifest"],
                            contributions_path=paths["contributions"],
                        )

    def test_partial_or_conditional_extra_result_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            results = json.loads(paths["result"].read_text(encoding="utf-8"))
            results.pop(next(iter(sorted(results))))
            _write_json(paths["result"], results)
            _refresh_receipts(paths)
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "result.*key set",
            ):
                compare.validate_fresh151_bundle(
                    result_path=paths["result"],
                    manifest_path=paths["manifest"],
                    contributions_path=paths["contributions"],
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            results = json.loads(paths["result"].read_text(encoding="utf-8"))
            conditional = set(
                compare.benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[
                    compare.benchmark.EVIDENCE_CONDITIONAL_FIG19_CD
                ]
            )
            confirmed = set(compare.CONFIRMED_CONDITIONS)
            extra_condition = sorted(conditional - confirmed)[0]
            results[compare.benchmark.condition_key(extra_condition)] = {
                "L": 0.0,
                "T": 0.0,
            }
            _write_json(paths["result"], results)
            _refresh_receipts(paths)
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "result.*key set",
            ):
                compare.validate_fresh151_bundle(
                    result_path=paths["result"],
                    manifest_path=paths["manifest"],
                    contributions_path=paths["contributions"],
                )

    def test_file_receipt_path_and_run_id_tampering_are_rejected(self) -> None:
        for mutation, pattern in (
            ("hash", "result SHA-256"),
            ("path", "does not bind supplied file"),
            ("run_id", "run_id mismatch"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    paths = _build_complete_bundle(root)
                    if mutation == "hash":
                        results = json.loads(
                            paths["result"].read_text(encoding="utf-8")
                        )
                        first = next(iter(sorted(results)))
                        results[first]["L"] = 1.0
                        _write_json(paths["result"], results)
                    elif mutation == "path":
                        manifest = json.loads(
                            paths["manifest"].read_text(encoding="utf-8")
                        )
                        manifest["result_path"] = "wrong.json"
                        _write_json(paths["manifest"], manifest)
                    else:
                        contributions = json.loads(
                            paths["contributions"].read_text(encoding="utf-8")
                        )
                        contributions["run_id"] = "wrong"
                        _write_json(paths["contributions"], contributions)
                        _refresh_receipts(paths)
                    with self.assertRaisesRegex(
                        compare.BaselineContractError,
                        pattern,
                    ):
                        compare.validate_fresh151_bundle(
                            result_path=paths["result"],
                            manifest_path=paths["manifest"],
                            contributions_path=paths["contributions"],
                        )

    def test_guard_graph_and_inventory_tampering_are_rejected(self) -> None:
        for mutation, pattern in (
            ("guard", "guard failed"),
            ("graph", "graph identity mismatch"),
            ("inventory", "node inventory mismatch"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    paths = _build_complete_bundle(root)
                    contributions = json.loads(
                        paths["contributions"].read_text(encoding="utf-8")
                    )
                    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                    key = next(iter(sorted(compare.EXPECTED_KEYS)))
                    case = contributions["cases"][key]
                    if mutation == "guard":
                        case["claim_guards"]["force_ledger"]["passed"] = False
                        manifest["case_guards"][key]["force_ledger"]["passed"] = False
                    elif mutation == "graph":
                        case["claim_graph_identity_sha256"] = "f" * 64
                    else:
                        case["claim_contributions"].pop("N6")
                    _write_json(paths["contributions"], contributions)
                    _write_json(paths["manifest"], manifest)
                    _refresh_receipts(paths)
                    with self.assertRaisesRegex(
                        compare.BaselineContractError,
                        pattern,
                    ):
                        compare.validate_fresh151_bundle(
                            result_path=paths["result"],
                            manifest_path=paths["manifest"],
                            contributions_path=paths["contributions"],
                        )

    def test_semantically_self_consistent_nonclosing_ledger_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            contribution_file = json.loads(
                paths["contributions"].read_text(encoding="utf-8")
            )
            key = next(iter(sorted(compare.EXPECTED_KEYS)))
            condition = compare.CONDITION_BY_KEY[key]
            case = contribution_file["cases"][key]
            case["claim_contributions"]["N1"][0]["body_force"] = [1.0, 0.0, 0.0]
            summary, ledger_total = compare.witness._contribution_summary(
                case["claim_contributions"],
                aoa_deg=condition[3],
            )
            wind = compare.witness._wind_force(ledger_total, condition[3])
            case["contribution_summary"] = summary
            case["recomputed_ledger"] = {
                "total_body_force_N": ledger_total.tolist(),
                "total_wind_force": wind,
                "max_body_error_N": 0.0,
                "max_wind_error_N": 0.0,
            }
            _write_json(paths["contributions"], contribution_file)
            _refresh_receipts(paths)
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "does not close",
            ):
                compare.validate_fresh151_bundle(
                    result_path=paths["result"],
                    manifest_path=paths["manifest"],
                    contributions_path=paths["contributions"],
                )

    def test_full_resolved_call_dictionary_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            contribution_file = json.loads(
                paths["contributions"].read_text(encoding="utf-8")
            )
            key = next(iter(sorted(compare.EXPECTED_KEYS)))
            contribution_file["cases"][key]["resolved_call"]["nc"] = 5
            _write_json(paths["contributions"], contribution_file)
            _refresh_receipts(paths)
            with self.assertRaisesRegex(
                compare.BaselineContractError,
                "full resolved-call contract",
            ):
                compare.validate_fresh151_bundle(
                    result_path=paths["result"],
                    manifest_path=paths["manifest"],
                    contributions_path=paths["contributions"],
                )


class ConfirmedCompareBaselineOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.paths = _build_complete_bundle(cls.root)
        cls.out_dir = cls.root / "out"
        cls.completion_spy = _CompletionSpy()
        cls.receipt = _build_baseline(
            cls.paths,
            out_dir=cls.out_dir,
            completion_validator=cls.completion_spy,
        )
        cls.output_paths = compare._output_paths(
            cls.out_dir,
            str(cls.receipt["run_id"]),
        )
        cls.scorecard = json.loads(
            cls.output_paths["scorecard"].read_text(encoding="utf-8")
        )
        cls.artifact = json.loads(
            cls.output_paths["artifact"].read_text(encoding="utf-8")
        )
        cls.fingerprint = json.loads(
            cls.output_paths["fingerprint"].read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_baseline_receipt_binds_complete_triplet(self) -> None:
        receipt = self.receipt
        self.assertEqual(len(self.completion_spy.calls), 1)
        self.assertEqual(
            self.completion_spy.calls[0]["run_id"],
            "20260729_000000",
        )
        self.assertEqual(receipt["status"], "READY_FOR_CONFIRMED_BASELINE_DIAGNOSIS")
        self.assertEqual(
            receipt["baseline_bundle_id"],
            compare._canonical_hash(receipt["bundle_id_payload"]),
        )
        self.assertEqual(
            receipt["contract"],
            {
                "official_curves": 42,
                "raw_measurement_samples": 434,
                "solver_conditions": 151,
                "physical_families": 34,
                "duplicate_alias_groups": 8,
                "figure_curve_counts": {"17": 10, "18": 24, "19": 8},
                "conditional_fig19_cd_curves_excluded": sorted(
                    compare.EXPECTED_CONDITIONAL_CURVES
                ),
            },
        )
        self.assertEqual(receipt["validation"]["condition_count"], 151)
        self.assertEqual(receipt["validation"]["evidence_case_count"], 151)
        self.assertEqual(receipt["validation"]["case_guard_count"], 151)
        self.assertLessEqual(
            receipt["validation"]["maximum_recomputed_body_ledger_error_N"],
            1.0e-9,
        )
        self.assertFalse(receipt["global_promotion_eligible"])
        self.assertEqual(
            receipt["global_promotion_blockers"],
            [
                "Fig19(c,d) authoritative fixed-frequency identity unresolved",
                (
                    "authoritative global condition union unresolved; 184 only "
                    "if shared frequency, 217 possible if channels differ"
                ),
            ],
        )
        self.assertTrue(
            receipt["validation"]["same_timestamp_complete_resume_revalidation"]
        )
        self.assertIn(
            "postprocess_authorization",
            receipt["input_artifacts"],
        )
        for name, identity in receipt["outputs"].items():
            with self.subTest(name=name):
                raw_path = Path(identity["path"])
                path = raw_path if raw_path.is_absolute() else compare.ROOT / raw_path
                self.assertTrue(path.is_file())
                self.assertEqual(_sha256(path), identity["sha256"])

    def test_score_artifacts_are_exact_confirmed42_only(self) -> None:
        scorecard = self.scorecard
        self.assertEqual(scorecard["schema_version"], 3)
        self.assertEqual(len(scorecard["rows"]), 42)
        self.assertEqual(
            sum(len(row["error_N"]) for row in scorecard["rows"]),
            434,
        )
        self.assertEqual(
            scorecard["coverage"]["valid_unique_conditions"],
            151,
        )
        self.assertEqual(
            set(scorecard["evidence_scopes"]),
            {compare.benchmark.EVIDENCE_CONFIRMED},
        )
        self.assertEqual(
            scorecard["promotion_blockers"],
            ["fig19_cd_fixed_frequency_unresolved"],
        )
        self.assertEqual(
            scorecard["excluded_evidence_scope"]["rows_in_this_scorecard"],
            0,
        )
        self.assertFalse(
            any(
                row["curve"].startswith(("19|c|", "19|d|")) for row in scorecard["rows"]
            )
        )

        self.assertEqual(len(self.artifact["curve_rows"]), 42)
        self.assertEqual(self.artifact["residual_point_count"], 434)
        self.assertEqual(
            set(self.artifact["excluded_curve_keys"]),
            compare.EXPECTED_CONDITIONAL_CURVES,
        )
        self.assertEqual(len(self.fingerprint["official_curves"]), 42)
        self.assertEqual(len(self.fingerprint["samples"]), 434)
        self.assertEqual(
            len(self.fingerprint["physical_curve_families"]),
            34,
        )
        self.assertEqual(len(self.fingerprint["duplicate_aliases"]), 8)
        self.assertFalse(
            any(
                item["curve"].startswith(("19|c|", "19|d|"))
                for item in (
                    self.fingerprint["official_curves"] + self.fingerprint["samples"]
                )
            )
        )
        self.assertTrue(all(self.fingerprint["validity_gates"].values()))

    def test_pf_equal_overall_and_figure_channel_counts(self) -> None:
        strata = self.fingerprint["aggregates"][
            "physical_family_equal_by_figure_channel"
        ]
        for figure, channels in compare.EXPECTED_PF_CELL_COUNTS.items():
            for channel, (expected_pf, expected_curves) in channels.items():
                with self.subTest(figure=figure, channel=channel):
                    cell = strata[figure][channel]
                    self.assertEqual(
                        cell["n_physical_families"],
                        expected_pf,
                    )
                    self.assertEqual(
                        cell["n_official_curves"],
                        expected_curves,
                    )
        self.assertEqual(
            self.fingerprint["primary_metric"]["physical_family_count"],
            34,
        )
        self.assertTrue(
            math.isclose(
                self.fingerprint["primary_metric"]["value"],
                strata["ALL"]["ALL"]["mean_family_mae_N"],
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )

    def test_scope_aware_overlays_and_sidecars(self) -> None:
        expected = {
            "17": (10, {"a", "b"}),
            "18": (24, {"a", "b", "c", "d"}),
            "19": (8, {"a", "b"}),
        }
        for figure, (curve_count, panels) in expected.items():
            with self.subTest(figure=figure):
                image = self.output_paths[f"fig{figure}"]
                sidecar_path = self.output_paths[f"fig{figure}_sidecar"]
                self.assertGreater(image.stat().st_size, 1000)
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                self.assertEqual(sidecar["curve_count"], curve_count)
                self.assertEqual(set(sidecar["panels"]), panels)
                self.assertEqual(sidecar["conditional_fig19_cd_rows"], 0)
                self.assertFalse(
                    any(
                        key.startswith(("19|c|", "19|d|"))
                        for key in sidecar["curve_keys"]
                    )
                )
                self.assertEqual(
                    sidecar["image"]["sha256"],
                    _sha256(image),
                )

    def test_existing_output_set_is_not_overwritten(self) -> None:
        before = {name: _sha256(path) for name, path in self.output_paths.items()}
        with self.assertRaisesRegex(
            compare.BaselineContractError,
            "refusing to overwrite",
        ):
            _build_baseline(self.paths, out_dir=self.out_dir)
        after = {name: _sha256(path) for name, path in self.output_paths.items()}
        self.assertEqual(before, after)

    def test_plot_failure_rolls_back_and_clean_rerun_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _build_complete_bundle(root)
            out_dir = root / "out"
            planned = compare._output_paths(
                out_dir,
                "20260729_000000",
            )
            real_plot = compare._plot_figure

            def fail_after_fig17(**kwargs: object) -> dict:
                if kwargs["figure_id"] == "18":
                    raise RuntimeError("synthetic plot failure")
                return real_plot(**kwargs)

            first_spy = _CompletionSpy()
            with mock.patch.object(
                compare,
                "_plot_figure",
                side_effect=fail_after_fig17,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic plot failure",
                ):
                    _build_baseline(
                        paths,
                        out_dir=out_dir,
                        completion_validator=first_spy,
                    )
            self.assertEqual(len(first_spy.calls), 1)
            self.assertTrue(all(not path.exists() for path in planned.values()))

            rerun_spy = _CompletionSpy()
            receipt = _build_baseline(
                paths,
                out_dir=out_dir,
                completion_validator=rerun_spy,
            )
            self.assertEqual(len(rerun_spy.calls), 1)
            self.assertEqual(
                receipt["status"],
                "READY_FOR_CONFIRMED_BASELINE_DIAGNOSIS",
            )
            self.assertTrue(all(path.is_file() for path in planned.values()))


if __name__ == "__main__":
    unittest.main()
