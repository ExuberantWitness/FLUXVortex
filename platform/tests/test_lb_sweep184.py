from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import lb_sweep184 as runner  # noqa: E402


REAL_VALIDATE_SOURCE_SNAPSHOT = runner._validate_source_snapshot
MOCK_CLAIM_MANIFEST = {
    "closure": "v41",
    "topology": ["N1"],
    "nodes": [
        {
            "id": "N1",
            "state": "validated",
            "freeze": True,
            "runtime_role": "physics",
            "implementation": "claim_runtime.components:UVLMComponent",
            "implementation_version": "test-v1",
            "implementation_hash": "sha256:" + "1" * 64,
        }
    ],
    "parameter_sources": {"closure": "test"},
    "guards": {},
}
MOCK_GRAPH_IDENTITY = runner._claim_graph_identity_sha256(
    MOCK_CLAIM_MANIFEST
)


def _force_guard(
    *,
    passed: bool = True,
    error: float = 0.0,
    tolerance: float = 1.0e-9,
) -> dict[str, object]:
    return {
        "passed": passed,
        "max_abs_error_N": error,
        "tolerance_N": tolerance,
    }


def _case_guards() -> dict[str, object]:
    return {
        "force_ledger": _force_guard(),
        "unclassified_force": _force_guard(),
        "unclassified_physical_force": _force_guard(),
        "cycle_reduction": _force_guard(tolerance=1.0e-12),
        "aero_output_invariance": _force_guard(tolerance=0.0),
        "claim_manifest_sha256": runner._canonical_hash(
            MOCK_CLAIM_MANIFEST
        ),
        "claim_graph_identity_sha256": MOCK_GRAPH_IDENTITY,
    }


def _gpu_result(
    *,
    lift: float = 1.25,
    thrust: float = -0.75,
    guards: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "L_wind": lift,
        "T_wind": thrust,
        "claim_guards": _case_guards() if guards is None else guards,
        "claim_manifest": copy.deepcopy(MOCK_CLAIM_MANIFEST),
    }


class Sweep184Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = json.loads(runner.SEED.read_text(encoding="utf-8"))
        cls.expected_keys = {
            runner.condition_key(condition) for condition in runner.CONDITIONS
        }
        cls.missing_conditions = [
            condition
            for condition in runner.CONDITIONS
            if runner.condition_key(condition) not in cls.seed
        ]

    @contextlib.contextmanager
    def isolated_layout(self):
        with tempfile.TemporaryDirectory(prefix="fluxv-sweep184-test-") as raw:
            root = Path(raw)
            docs = root / "platform" / "docs"
            diag = docs / "diag"
            scorecards = docs / "scorecards"
            docs.mkdir(parents=True)
            diag.mkdir()
            scorecards.mkdir()

            seed_path = docs / "s6_sweep_v41.json"
            seed_path.write_text(
                json.dumps(self.seed, allow_nan=False), encoding="utf-8"
            )
            repro_path = docs / "repro_data.json"
            repro_path.write_text("{}\n", encoding="utf-8")

            layout = SimpleNamespace(
                root=root,
                docs=docs,
                diag=diag,
                scorecards=scorecards,
                seed=seed_path,
                repro=repro_path,
                result=docs / "s6_sweep_v41_full184_20260101_000000.json",
                manifest=diag
                / "fig171819_v41_baseline_manifest_20260101_000000.json",
                scorecard=scorecards
                / "scorecard_v41_full184_20260101_000000.json",
                fixed_result=docs / "s6_sweep_v41_full184.json",
                fixed_manifest=diag / "fig171819_v41_baseline_manifest.json",
                fixed_scorecard=scorecards / "scorecard_v41_full184.json",
                lock=diag / "fig171819_v41_baseline.lock",
                source_hashes={"frozen/runtime.py": "a" * 64},
                tracked_sources={"platform/lb_sweep184.py": "b" * 64},
                governed_source_members={
                    pattern: [] for pattern in runner.GOVERNED_SOURCE_GLOBS
                },
                runtime_identity={"runtime": "mock-stable"},
            )
            layout.authoritative_sources = {
                **layout.source_hashes,
                **layout.tracked_sources,
            }
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(runner, "ROOT", root))
                stack.enter_context(mock.patch.object(runner, "DOCS", docs))
                stack.enter_context(mock.patch.object(runner, "DIAG", diag))
                stack.enter_context(
                    mock.patch.object(runner, "SCORECARDS", scorecards)
                )
                stack.enter_context(mock.patch.object(runner, "SEED", seed_path))
                stack.enter_context(
                    mock.patch.object(runner, "DEFAULT_REPRO", repro_path)
                )
                stack.enter_context(
                    mock.patch.object(
                        runner, "FIXED_RESULT", layout.fixed_result
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner, "FIXED_MANIFEST", layout.fixed_manifest
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner, "FIXED_SCORECARD", layout.fixed_scorecard
                    )
                )
                stack.enter_context(
                    mock.patch.object(runner, "RUN_LOCK", layout.lock)
                )
                stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_validate_frozen_identity",
                        return_value=layout.source_hashes,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_snapshot_loaded_project_sources",
                        return_value=layout.tracked_sources,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_snapshot_authoritative_sources",
                        return_value=layout.authoritative_sources,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner, "_validate_authoritative_sources"
                    )
                )
                stack.enter_context(
                    mock.patch.object(runner, "_validate_source_snapshot")
                )
                stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_runtime_identity",
                        return_value=layout.runtime_identity,
                    )
                )
                yield layout

    def _fake_modules(self, gpu_run_twist):
        warp = types.ModuleType("warp")
        warp.init = mock.Mock()
        robo = types.ModuleType("_v2_robo")
        robo.gpu_run_twist = gpu_run_twist
        return mock.patch.dict(
            sys.modules, {"warp": warp, "_v2_robo": robo}
        )

    def _resume_manifest(
        self,
        layout,
        gpu_run_twist,
        *,
        checkpoint: dict[str, object] | None = None,
        case_guards: dict[str, object] | None = None,
    ) -> dict[str, object]:
        checkpoint = self.seed if checkpoint is None else checkpoint
        case_guards = {} if case_guards is None else case_guards
        manifest = {
            "schema_version": runner.MANIFEST_SCHEMA_VERSION,
            "campaign": "test",
            "run_id": "20260101_000000",
            "status": "running",
            "started_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "result_path": str(layout.result.relative_to(layout.root)),
            "scorecard_path": str(layout.scorecard.relative_to(layout.root)),
            "source_hashes": layout.source_hashes,
            "authoritative_source_hashes": layout.authoritative_sources,
            "tracked_source_hashes": layout.tracked_sources,
            "governed_source_members": layout.governed_source_members,
            "claim_graph_identity_sha256": MOCK_GRAPH_IDENTITY,
            "base_profile_sha256": runner._canonical_hash(runner.BASE),
            "target_contract_sha256": runner._canonical_hash(
                runner._target_contract_payload()
            ),
            "call_contract_sha256": runner._call_contract_sha256(
                gpu_run_twist
            ),
            "seed_coverage": runner.coverage(self.seed),
            "expected_unique_conditions": 184,
            "missing_condition_count_at_start": 66,
            "completed_new_conditions": runner._count_valid_new_results(
                checkpoint,
                self.seed,
                case_guards,
                MOCK_GRAPH_IDENTITY,
            ),
            "failures": {},
            "case_guards": case_guards,
            "runtime_identity": layout.runtime_identity,
            "runtime_sessions": [],
            "cold_preconditioner": None,
            "formal_anchor": None,
            "resume_events": [],
            "data_identity_gate": {
                "fig19_cd_fixed_frequency": "unresolved",
                "conditional_assumption_hz": 2.6,
                "publication_authorized": False,
            },
        }
        return manifest

    @staticmethod
    def _conditional_scorecard(checkpoint, _repro, *, sweep_name=""):
        sweep_coverage = runner.coverage(checkpoint)
        return {
            "sweep": sweep_name,
            "coverage": sweep_coverage,
            "contract": {
                "fig19_cd_frequency": {
                    "status": "unresolved",
                }
            },
            "promotion_eligible": bool(sweep_coverage["complete"]),
        }

    def test_dry_run_never_imports_warp_or_writes_artifacts(self):
        with tempfile.TemporaryDirectory(
            prefix="fluxv-sweep184-dry-"
        ) as raw:
            directory = Path(raw)
            result = directory / "result.json"
            manifest = directory / "manifest.json"
            scorecard = directory / "scorecard.json"
            before = set(directory.iterdir())
            with mock.patch.dict(sys.modules, {"warp": None}):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = runner.run(
                        result_path=result,
                        manifest_path=manifest,
                        scorecard_path=scorecard,
                        dry_run=True,
                    )
            self.assertEqual(status, 0)
            self.assertEqual(set(directory.iterdir()), before)
            self.assertFalse(result.exists())
            self.assertFalse(manifest.exists())
            self.assertFalse(scorecard.exists())

    def test_exact_call_contract_uses_half_nominal_twist_and_frozen_spc(self):
        calls: list[dict[str, object]] = []

        def fake_gpu_run_twist(**kwargs):
            calls.append(kwargs)
            return _gpu_result()

        condition = (10.0, 2.3, 45.0, 5.0)
        value, guards = runner._run_condition(fake_gpu_run_twist, condition)
        self.assertEqual(value, {"L": 1.25, "T": -0.75})
        self.assertTrue(runner._valid_case_guard(guards))
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["twist_amp_deg"], 22.5)
        self.assertEqual(call["flap_amp_deg"], 22.5)
        self.assertEqual(call["twist_phase_deg"], 90.0)
        self.assertEqual(call["steps_per_cycle"], 780)
        self.assertEqual(call["wake_rows"], 780)
        self.assertEqual(call["nc"], 12)
        self.assertEqual(call["ns"], 16)
        self.assertEqual(call["n_cycle"], 4)
        self.assertEqual(call["closure"], "v41")
        resolved = runner._resolved_call(fake_gpu_run_twist, condition)
        for name, expected in call.items():
            self.assertEqual(resolved[name], expected, msg=name)

    def test_resume_discards_and_reexecutes_bad_new_records(self):
        with self.isolated_layout() as layout:
            checkpoint = copy.deepcopy(self.seed)
            case_guards: dict[str, object] = {}
            for index, condition in enumerate(self.missing_conditions):
                key = runner.condition_key(condition)
                checkpoint[key] = {"L": float(index), "T": -float(index)}
                case_guards[key] = _case_guards()

            bad_conditions = self.missing_conditions[:3]
            malformed_key = runner.condition_key(bad_conditions[0])
            nonfinite_key = runner.condition_key(bad_conditions[1])
            missing_guard_key = runner.condition_key(bad_conditions[2])
            checkpoint[malformed_key] = {"L": 1.0}
            checkpoint[nonfinite_key] = {"L": math.nan, "T": 2.0}
            case_guards.pop(missing_guard_key)

            calls: list[dict[str, object]] = []

            def fake_gpu_run_twist(**kwargs):
                calls.append(kwargs)
                if len(calls) <= 2:
                    anchor = self.seed[runner.condition_key(runner.ANCHOR)]
                    return _gpu_result(
                        lift=float(anchor["L"]), thrust=float(anchor["T"])
                    )
                return _gpu_result(
                    lift=10.0 + len(calls), thrust=-10.0 - len(calls)
                )

            manifest = self._resume_manifest(
                layout,
                fake_gpu_run_twist,
                checkpoint=checkpoint,
                case_guards=case_guards,
            )
            layout.result.write_text(
                json.dumps(checkpoint, allow_nan=True), encoding="utf-8"
            )
            layout.manifest.write_text(
                json.dumps(manifest, allow_nan=False), encoding="utf-8"
            )
            with self._fake_modules(fake_gpu_run_twist):
                with mock.patch.object(
                    runner, "scorecard", side_effect=self._conditional_scorecard
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        status = runner._run_locked(
                            result_path=layout.result,
                            manifest_path=layout.manifest,
                            scorecard_path=layout.scorecard,
                            resume=True,
                        )

            self.assertEqual(status, 2)
            # Cold preconditioner + formal anchor + exactly the three discarded cases.
            self.assertEqual(len(calls), 5)
            rerun_signatures = {
                (
                    float(call["U"]),
                    float(call["freq"]),
                    float(call["twist_amp_deg"]) * 2.0,
                    float(call["aoa_deg"]),
                )
                for call in calls[2:]
            }
            self.assertEqual(rerun_signatures, set(bad_conditions))
            repaired = json.loads(layout.result.read_text(encoding="utf-8"))
            repaired_manifest = json.loads(
                layout.manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(set(repaired), self.expected_keys)
            for condition in bad_conditions:
                key = runner.condition_key(condition)
                self.assertTrue(runner._valid_result_record(repaired[key]))
                self.assertTrue(
                    runner._valid_case_guard(
                        repaired_manifest["case_guards"][key]
                    )
                )
            self.assertEqual(
                set(
                    repaired_manifest["resume_events"][-1][
                        "discarded_invalid_condition_keys"
                    ]
                ),
                {malformed_key, nonfinite_key, missing_guard_key},
            )

    def test_resume_rejects_obsolete_manifest_schema(self):
        with self.isolated_layout() as layout:
            def fake_gpu_run_twist(**_kwargs):
                return _gpu_result()

            manifest = self._resume_manifest(
                layout, fake_gpu_run_twist
            )
            manifest["schema_version"] = (
                runner.MANIFEST_SCHEMA_VERSION - 1
            )
            layout.result.write_text(
                json.dumps(self.seed, allow_nan=False), encoding="utf-8"
            )
            layout.manifest.write_text(
                json.dumps(manifest, allow_nan=False), encoding="utf-8"
            )
            with self._fake_modules(fake_gpu_run_twist):
                with self.assertRaisesRegex(
                    RuntimeError, "resume manifest schema mismatch"
                ):
                    runner._run_locked(
                        result_path=layout.result,
                        manifest_path=layout.manifest,
                        scorecard_path=layout.scorecard,
                        resume=True,
                    )

    def test_resume_rejects_changed_seed_and_unexpected_key(self):
        good = _case_guards()
        expected = set(self.expected_keys)

        changed_seed = copy.deepcopy(self.seed)
        first_seed_key = next(iter(changed_seed))
        changed_seed[first_seed_key] = {"L": 999.0, "T": 999.0}
        with self.assertRaisesRegex(RuntimeError, "changed frozen seed"):
            runner._sanitize_resume_checkpoint(
                changed_seed,
                seed=self.seed,
                expected_keys=expected,
                case_guards={},
                claim_graph_identity_sha256=MOCK_GRAPH_IDENTITY,
            )

        extra = copy.deepcopy(self.seed)
        extra["not_a_benchmark_condition"] = {"L": 0.0, "T": 0.0}
        with self.assertRaisesRegex(RuntimeError, "unexpected condition"):
            runner._sanitize_resume_checkpoint(
                extra,
                seed=self.seed,
                expected_keys=expected,
                case_guards={"not_a_benchmark_condition": good},
                claim_graph_identity_sha256=MOCK_GRAPH_IDENTITY,
            )

        mismatched = copy.deepcopy(self.seed)
        mismatched_key = runner.condition_key(self.missing_conditions[0])
        mismatched[mismatched_key] = {"L": 1.0, "T": 2.0}
        mismatched_guards = {
            mismatched_key: {
                **_case_guards(),
                "claim_graph_identity_sha256": "f" * 64,
            }
        }
        discarded = runner._sanitize_resume_checkpoint(
            mismatched,
            seed=self.seed,
            expected_keys=expected,
            case_guards=mismatched_guards,
            claim_graph_identity_sha256=MOCK_GRAPH_IDENTITY,
        )
        self.assertEqual(discarded, [mismatched_key])
        self.assertNotIn(mismatched_key, mismatched)
        self.assertNotIn(mismatched_key, mismatched_guards)

    def test_force_ledger_guards_are_mandatory_and_numerically_valid(self):
        self.assertTrue(runner._valid_case_guard(_case_guards()))
        self.assertFalse(runner._valid_case_guard({}))
        self.assertFalse(
            runner._valid_force_guard(
                _force_guard(error=2.0e-9, tolerance=1.0e-9)
            )
        )
        self.assertFalse(
            runner._valid_force_guard(_force_guard(passed=False))
        )
        self.assertFalse(
            runner._valid_force_guard(_force_guard(error=-1.0e-12))
        )
        self.assertFalse(
            runner._valid_force_guard(
                {
                    "passed": True,
                    "max_abs_error_N": False,
                    "tolerance_N": 1.0e-9,
                }
            )
        )

        def missing_guard_gpu(**_kwargs):
            return _gpu_result(guards={"force_ledger": _force_guard()})

        with self.assertRaisesRegex(RuntimeError, "force-ledger guards"):
            runner._run_condition(
                missing_guard_gpu, self.missing_conditions[0]
            )

        def failed_guard_gpu(**_kwargs):
            return _gpu_result(
                guards={
                    "force_ledger": _force_guard(
                        error=2.0e-9, tolerance=1.0e-9
                    ),
                    "unclassified_force": _force_guard(),
                    "unclassified_physical_force": _force_guard(),
                    "cycle_reduction": _force_guard(tolerance=1.0e-12),
                    "aero_output_invariance": _force_guard(tolerance=0.0),
                }
            )

        with self.assertRaisesRegex(RuntimeError, "force-ledger guards"):
            runner._run_condition(
                failed_guard_gpu, self.missing_conditions[0]
            )

        def failed_cycle_guard_gpu(**_kwargs):
            return _gpu_result(
                guards={
                    "force_ledger": _force_guard(),
                    "unclassified_force": _force_guard(),
                    "unclassified_physical_force": _force_guard(),
                    "cycle_reduction": _force_guard(
                        error=2.0e-12, tolerance=1.0e-12
                    ),
                    "aero_output_invariance": _force_guard(tolerance=0.0),
                }
            )

        with self.assertRaisesRegex(RuntimeError, "force-ledger guards"):
            runner._run_condition(
                failed_cycle_guard_gpu, self.missing_conditions[0]
            )

        old_two_guard_checkpoint = {
            "force_ledger": _force_guard(),
            "unclassified_force": _force_guard(),
            "claim_manifest_sha256": "a" * 64,
            "claim_graph_identity_sha256": "b" * 64,
        }
        self.assertFalse(
            runner._valid_case_guard(old_two_guard_checkpoint)
        )

    def test_claim_graph_identity_is_required_stable_and_guard_independent(self):
        different_guard = copy.deepcopy(MOCK_CLAIM_MANIFEST)
        different_guard["guards"] = {
            "force_ledger": _force_guard(error=5.0e-13)
        }
        self.assertEqual(
            runner._claim_graph_identity_sha256(different_guard),
            MOCK_GRAPH_IDENTITY,
        )

        different_graph = copy.deepcopy(MOCK_CLAIM_MANIFEST)
        different_graph["topology"] = ["N2"]
        different_graph["nodes"][0]["id"] = "N2"
        different_graph_identity = runner._claim_graph_identity_sha256(
            different_graph
        )
        self.assertNotEqual(different_graph_identity, MOCK_GRAPH_IDENTITY)
        self.assertFalse(
            runner._valid_case_guard(
                {
                    **_case_guards(),
                    "claim_graph_identity_sha256": different_graph_identity,
                },
                MOCK_GRAPH_IDENTITY,
            )
        )

        def missing_manifest_gpu(**_kwargs):
            result = _gpu_result()
            result.pop("claim_manifest")
            return result

        with self.assertRaisesRegex(RuntimeError, "no claim graph manifest"):
            runner._run_condition(
                missing_manifest_gpu, self.missing_conditions[0]
            )

    def test_case_with_different_claim_graph_is_not_checkpointed(self):
        with self.isolated_layout() as layout:
            calls: list[dict[str, object]] = []
            different_graph = copy.deepcopy(MOCK_CLAIM_MANIFEST)
            different_graph["topology"] = ["N2"]
            different_graph["nodes"][0]["id"] = "N2"

            def fake_gpu_run_twist(**kwargs):
                calls.append(kwargs)
                anchor = self.seed[runner.condition_key(runner.ANCHOR)]
                result = _gpu_result(
                    lift=float(anchor["L"]), thrust=float(anchor["T"])
                )
                if len(calls) == 3:
                    result["claim_manifest"] = different_graph
                return result

            with self._fake_modules(fake_gpu_run_twist):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        RuntimeError, "stopped to require a fresh-process resume"
                    ):
                        runner._run_locked(
                            result_path=layout.result,
                            manifest_path=layout.manifest,
                            scorecard_path=layout.scorecard,
                            resume=False,
                        )

            self.assertEqual(len(calls), 3)
            checkpoint = json.loads(
                layout.result.read_text(encoding="utf-8")
            )
            manifest = json.loads(
                layout.manifest.read_text(encoding="utf-8")
            )
            failed_key = runner.condition_key(self.missing_conditions[0])
            self.assertEqual(checkpoint, self.seed)
            self.assertEqual(manifest["status"], "failed_case")
            self.assertIn(
                "claim graph identity differs",
                manifest["failures"][failed_key],
            )

    def test_runtime_identity_binds_actual_solver_dtype_device_and_threads(self):
        np = __import__("numpy")

        class FakeDevice:
            alias = "cuda:1"
            name = "Mock GPU"
            arch = 89

            def __str__(self):
                return "cuda:1"

        fake_warp = types.ModuleType("warp")
        fake_warp.__version__ = "mock-warp"
        fake_warp.get_device = mock.Mock(return_value=FakeDevice())

        fake_cfg = types.ModuleType("fluxvortex.warp_fsi.config")
        fake_cfg.DEVICE = "cuda:1"
        fake_cfg.DTYPE = "wp.float32"
        fake_cfg.NP_DTYPE = np.float32
        fake_cfg.CR_TOL = 1.0e-5
        fake_cfg.NEWTON_TOL = 1.0e-5
        fake_cfg.GEOM_ATOL = 1.0e-4
        fake_cfg.PORT_ATOL = 1.0e-4
        fake_cfg.dtype_name = lambda: "float32"

        fake_fluxvortex = types.ModuleType("fluxvortex")
        fake_warp_fsi = types.ModuleType("fluxvortex.warp_fsi")
        fake_warp_fsi.config = fake_cfg
        fake_fluxvortex.warp_fsi = fake_warp_fsi
        modules = {
            "warp": fake_warp,
            "fluxvortex": fake_fluxvortex,
            "fluxvortex.warp_fsi": fake_warp_fsi,
            "fluxvortex.warp_fsi.config": fake_cfg,
        }
        environment = {
            "FLUXV_DTYPE": "float32",
            "FLUXV_DEVICE": "cuda:1",
            "MKL_NUM_THREADS": "3",
            "NUMEXPR_NUM_THREADS": "4",
        }
        with mock.patch.dict(sys.modules, modules):
            with mock.patch.dict(
                runner.os.environ, environment, clear=False
            ):
                identity = runner._runtime_identity()

        self.assertEqual(identity["solver_config"]["dtype_name"], "float32")
        self.assertEqual(identity["solver_config"]["device"], "cuda:1")
        self.assertEqual(identity["solver_config"]["numpy_dtype"], "float32")
        self.assertEqual(identity["warp_device"]["alias"], "cuda:1")
        fake_warp.get_device.assert_called_once_with("cuda:1")
        self.assertEqual(identity["environment"]["FLUXV_DTYPE"], "float32")
        self.assertEqual(identity["environment"]["FLUXV_DEVICE"], "cuda:1")
        self.assertEqual(identity["environment"]["MKL_NUM_THREADS"], "3")
        self.assertEqual(identity["environment"]["NUMEXPR_NUM_THREADS"], "4")

    def test_campaign_flock_rejects_competitor_and_releases(self):
        with tempfile.TemporaryDirectory(
            prefix="fluxv-sweep184-lock-"
        ) as raw:
            lock = Path(raw) / "campaign.lock"
            with runner._exclusive_campaign_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "another .* holds"):
                    with runner._exclusive_campaign_lock(lock):
                        self.fail("a competing lock unexpectedly succeeded")
            with runner._exclusive_campaign_lock(lock):
                pass

    def test_atomic_write_failure_preserves_target_and_cleans_unique_partial(self):
        with tempfile.TemporaryDirectory(
            prefix="fluxv-sweep184-atomic-"
        ) as raw:
            directory = Path(raw)
            target = directory / "checkpoint.json"
            target.write_text('{"generation": 1}\n', encoding="utf-8")
            original = target.read_bytes()
            with mock.patch.object(
                runner.os, "replace", side_effect=OSError("injected replace")
            ):
                with self.assertRaisesRegex(OSError, "injected replace"):
                    runner._write_json_atomic(target, {"generation": 2})
            self.assertEqual(target.read_bytes(), original)
            leftovers = list(directory.glob("*.partial")) + list(
                directory.glob(".*.partial")
            )
            self.assertEqual(leftovers, [])
            self.assertFalse((directory / "checkpoint.json.partial").exists())

    def test_resume_rejects_source_target_call_and_runtime_drift(self):
        def fake_gpu_run_twist(**_kwargs):
            return _gpu_result()

        fake_modules = self._fake_modules(fake_gpu_run_twist)
        with self.isolated_layout() as layout, fake_modules:
            def write_attempt(**overrides):
                manifest = self._resume_manifest(
                    layout, fake_gpu_run_twist
                )
                manifest.update(overrides)
                layout.result.write_text(
                    json.dumps(self.seed, allow_nan=False), encoding="utf-8"
                )
                layout.manifest.write_text(
                    json.dumps(manifest, allow_nan=False), encoding="utf-8"
                )

            write_attempt(authoritative_source_hashes=None)
            with self.assertRaisesRegex(
                RuntimeError, "authoritative pre-import source snapshot"
            ):
                runner._run_locked(
                    result_path=layout.result,
                    manifest_path=layout.manifest,
                    scorecard_path=layout.scorecard,
                    resume=True,
                )

            write_attempt(claim_graph_identity_sha256=None)
            with self.assertRaisesRegex(
                RuntimeError, "frozen claim graph identity"
            ):
                runner._run_locked(
                    result_path=layout.result,
                    manifest_path=layout.manifest,
                    scorecard_path=layout.scorecard,
                    resume=True,
                )

            write_attempt(source_hashes={"different": "source"})
            with self.assertRaisesRegex(RuntimeError, "source identity"):
                runner._run_locked(
                    result_path=layout.result,
                    manifest_path=layout.manifest,
                    scorecard_path=layout.scorecard,
                    resume=True,
                )

            write_attempt(target_contract_sha256="different-target")
            with self.assertRaisesRegex(RuntimeError, "target184"):
                runner._run_locked(
                    result_path=layout.result,
                    manifest_path=layout.manifest,
                    scorecard_path=layout.scorecard,
                    resume=True,
                )

            write_attempt(call_contract_sha256="different-call")
            with self.assertRaisesRegex(RuntimeError, "call contract"):
                runner._run_locked(
                    result_path=layout.result,
                    manifest_path=layout.manifest,
                    scorecard_path=layout.scorecard,
                    resume=True,
                )

            write_attempt(runtime_identity={"runtime": "different"})
            with self.assertRaisesRegex(RuntimeError, "runtime/device"):
                runner._run_locked(
                    result_path=layout.result,
                    manifest_path=layout.manifest,
                    scorecard_path=layout.scorecard,
                    resume=True,
                )

            tracked = layout.root / "tracked.py"
            tracked.write_text("generation = 1\n", encoding="utf-8")
            digest = runner._sha256_file(tracked)
            tracked.write_text("generation = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source identity drift"):
                REAL_VALIDATE_SOURCE_SNAPSHOT({"tracked.py": digest})

    def test_governed_source_member_drift_rejects_addition_and_removal(self):
        with tempfile.TemporaryDirectory(
            prefix="fluxv-sweep184-members-"
        ) as raw:
            root = Path(raw)
            nodes = root / "platform" / "claim_nodes"
            runtime = root / "platform" / "claim_runtime"
            nodes.mkdir(parents=True)
            runtime.mkdir()
            original_yaml = nodes / "N1.yaml"
            original_python = runtime / "core.py"
            original_yaml.write_text("id: N1\n", encoding="utf-8")
            original_python.write_text("VERSION = 1\n", encoding="utf-8")

            with mock.patch.object(runner, "ROOT", root):
                expected = runner._snapshot_governed_source_members()
                runner._validate_governed_source_members(expected)

                added = nodes / "N7.yaml"
                added.write_text("id: N7\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "directory membership drift"
                ):
                    runner._validate_governed_source_members(expected)

                added.unlink()
                original_python.unlink()
                with self.assertRaisesRegex(
                    RuntimeError, "directory membership drift"
                ):
                    runner._validate_governed_source_members(expected)

    def test_authoritative_snapshot_covers_yml_recursive_runtime_and_exact_set(self):
        with tempfile.TemporaryDirectory(
            prefix="fluxv-sweep184-authority-"
        ) as raw:
            root = Path(raw)
            files = {
                "platform/runner.py": "RUNNER = 1\n",
                "platform/claim_nodes/N1.yml": "id: N1\n",
                "platform/claim_runtime/nested/plugin.py": "VERSION = 1\n",
                "src/fluxvortex/solver.py": "SOLVER = 1\n",
            }
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            with mock.patch.object(runner, "ROOT", root):
                with mock.patch.object(
                    runner, "FROZEN_EXPECTED_HASHES", {}
                ):
                    expected = runner._snapshot_authoritative_sources()
                    self.assertIn(
                        "platform/claim_nodes/N1.yml", expected
                    )
                    self.assertIn(
                        "platform/claim_runtime/nested/plugin.py", expected
                    )
                    runner._validate_authoritative_sources(expected)
                    with self.assertRaisesRegex(
                        RuntimeError, "unregistered project sources"
                    ):
                        runner._validate_loaded_sources_registered(
                            {
                                **expected,
                                "outside/late_import.py": "c" * 64,
                            },
                            expected,
                        )

                    changed = root / "src" / "fluxvortex" / "solver.py"
                    changed.write_text("SOLVER = 2\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        RuntimeError, "source identity drift"
                    ):
                        runner._validate_authoritative_sources(expected)
                    changed.write_text("SOLVER = 1\n", encoding="utf-8")

                    added = root / "platform" / "claim_nodes" / "N2.yaml"
                    added.write_text("id: N2\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        RuntimeError, "path-set drift"
                    ):
                        runner._validate_authoritative_sources(expected)
                    added.unlink()

                    removed = (
                        root
                        / "platform"
                        / "claim_runtime"
                        / "nested"
                        / "plugin.py"
                    )
                    removed.unlink()
                    with self.assertRaisesRegex(
                        RuntimeError, "path-set drift"
                    ):
                        runner._validate_authoritative_sources(expected)

    def test_fresh_import_window_source_drift_stops_before_gpu_call(self):
        with self.isolated_layout() as layout:
            calls: list[dict[str, object]] = []

            def fake_gpu_run_twist(**kwargs):
                calls.append(kwargs)
                return _gpu_result()

            validations = {"count": 0}

            def fail_after_pre_import(_expected):
                validations["count"] += 1
                if validations["count"] == 2:
                    raise RuntimeError("injected pre-import authority drift")

            with self._fake_modules(fake_gpu_run_twist):
                with mock.patch.object(
                    runner,
                    "_validate_authoritative_sources",
                    side_effect=fail_after_pre_import,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "pre-import authority drift"
                    ):
                        runner._run_locked(
                            result_path=layout.result,
                            manifest_path=layout.manifest,
                            scorecard_path=layout.scorecard,
                            resume=False,
                        )

            self.assertEqual(calls, [])

    def test_case_stops_if_claim_yaml_is_added_during_gpu_call(self):
        with self.isolated_layout() as layout:
            calls: list[dict[str, object]] = []

            def fake_gpu_run_twist(**kwargs):
                calls.append(kwargs)
                anchor = self.seed[runner.condition_key(runner.ANCHOR)]
                if len(calls) == 3:
                    nodes = layout.root / "platform" / "claim_nodes"
                    nodes.mkdir(parents=True, exist_ok=True)
                    (nodes / "N7.yaml").write_text(
                        "id: N7\n", encoding="utf-8"
                    )
                return _gpu_result(
                    lift=float(anchor["L"]), thrust=float(anchor["T"])
                )

            with self._fake_modules(fake_gpu_run_twist):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        RuntimeError, "stopped to require a fresh-process resume"
                    ):
                        runner._run_locked(
                            result_path=layout.result,
                            manifest_path=layout.manifest,
                            scorecard_path=layout.scorecard,
                            resume=False,
                        )

            self.assertEqual(len(calls), 3)
            manifest = json.loads(
                layout.manifest.read_text(encoding="utf-8")
            )
            failed_key = runner.condition_key(self.missing_conditions[0])
            self.assertEqual(manifest["status"], "failed_case")
            self.assertIn("directory membership drift", manifest["failures"][failed_key])

    def test_scorecard_window_drift_prevents_scorecard_write(self):
        with self.isolated_layout() as layout:
            checkpoint = copy.deepcopy(self.seed)
            case_guards: dict[str, object] = {}
            for index, condition in enumerate(self.missing_conditions):
                key = runner.condition_key(condition)
                checkpoint[key] = {"L": float(index), "T": -float(index)}
                case_guards[key] = _case_guards()

            def fake_gpu_run_twist(**_kwargs):
                anchor = self.seed[runner.condition_key(runner.ANCHOR)]
                return _gpu_result(
                    lift=float(anchor["L"]), thrust=float(anchor["T"])
                )

            manifest = self._resume_manifest(
                layout,
                fake_gpu_run_twist,
                checkpoint=checkpoint,
                case_guards=case_guards,
            )
            layout.result.write_text(
                json.dumps(checkpoint, allow_nan=False), encoding="utf-8"
            )
            layout.manifest.write_text(
                json.dumps(manifest, allow_nan=False), encoding="utf-8"
            )
            scoring = {"active": False}

            def fake_scorecard(*_args, **_kwargs):
                scoring["active"] = True
                return self._conditional_scorecard(checkpoint, {})

            def reject_after_scoring(_expected):
                if scoring["active"]:
                    raise RuntimeError("injected scorecard-window drift")

            with self._fake_modules(fake_gpu_run_twist):
                with mock.patch.object(
                    runner, "scorecard", side_effect=fake_scorecard
                ):
                    with mock.patch.object(
                        runner,
                        "_validate_authoritative_sources",
                        side_effect=reject_after_scoring,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "scorecard-window drift"
                        ):
                            runner._run_locked(
                                result_path=layout.result,
                                manifest_path=layout.manifest,
                                scorecard_path=layout.scorecard,
                                resume=True,
                            )

            self.assertFalse(layout.scorecard.exists())
            self.assertFalse(layout.fixed_scorecard.exists())

    def test_case_exception_is_checkpointed_and_fails_fast(self):
        with self.isolated_layout() as layout:
            calls: list[dict[str, object]] = []

            def fake_gpu_run_twist(**kwargs):
                calls.append(kwargs)
                if len(calls) == 3:
                    raise ValueError("injected case failure")
                anchor = self.seed[runner.condition_key(runner.ANCHOR)]
                return _gpu_result(
                    lift=float(anchor["L"]), thrust=float(anchor["T"])
                )

            with self._fake_modules(fake_gpu_run_twist):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        RuntimeError, "stopped to require a fresh-process resume"
                    ):
                        runner._run_locked(
                            result_path=layout.result,
                            manifest_path=layout.manifest,
                            scorecard_path=layout.scorecard,
                            resume=False,
                        )

            self.assertEqual(len(calls), 3)
            failed_key = runner.condition_key(self.missing_conditions[0])
            checkpoint = json.loads(
                layout.result.read_text(encoding="utf-8")
            )
            manifest = json.loads(
                layout.manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint, self.seed)
            self.assertEqual(manifest["status"], "failed_case")
            self.assertIn(failed_key, manifest["failures"])
            self.assertIn("injected case failure", manifest["failures"][failed_key])
            self.assertEqual(manifest["completed_new_conditions"], 0)

    def test_publication_is_manifest_last_and_pending_is_idempotent(self):
        with self.isolated_layout() as layout:
            layout.result.write_text('{"result": 1}\n', encoding="utf-8")
            layout.scorecard.write_text(
                '{"scorecard": 1}\n', encoding="utf-8"
            )
            manifest = {
                "status": "running",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
            layout.manifest.write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            events: list[tuple[str, Path, str | None]] = []
            real_write = runner._write_json_atomic
            real_copy = runner._copy_atomic

            def recording_write(path, value):
                events.append(("write", path, value.get("status")))
                return real_write(path, value)

            def recording_copy(source, destination):
                events.append(("copy", destination, None))
                return real_copy(source, destination)

            with mock.patch.object(
                runner, "_write_json_atomic", side_effect=recording_write
            ):
                with mock.patch.object(
                    runner, "_copy_atomic", side_effect=recording_copy
                ):
                    runner._publish_complete(
                        result_path=layout.result,
                        manifest_path=layout.manifest,
                        scorecard_path=layout.scorecard,
                        manifest=manifest,
                    )

            fixed_result_event = next(
                index
                for index, event in enumerate(events)
                if event[:2] == ("copy", layout.fixed_result)
            )
            fixed_score_event = next(
                index
                for index, event in enumerate(events)
                if event[:2] == ("copy", layout.fixed_scorecard)
            )
            fixed_manifest_event = next(
                index
                for index, event in enumerate(events)
                if event[:2] == ("write", layout.fixed_manifest)
            )
            self.assertLess(fixed_result_event, fixed_manifest_event)
            self.assertLess(fixed_score_event, fixed_manifest_event)
            self.assertEqual(
                json.loads(
                    layout.fixed_manifest.read_text(encoding="utf-8")
                )["status"],
                "completed",
            )

        with self.isolated_layout() as layout:
            layout.result.write_text('{"result": 2}\n', encoding="utf-8")
            layout.scorecard.write_text(
                '{"scorecard": 2}\n', encoding="utf-8"
            )
            manifest = {
                "status": "running",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
            layout.manifest.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            real_write = runner._write_json_atomic
            injected = {"done": False}

            def fail_fixed_manifest_once(path, value):
                if path == layout.fixed_manifest and not injected["done"]:
                    injected["done"] = True
                    raise OSError("injected fixed-manifest failure")
                return real_write(path, value)

            with mock.patch.object(
                runner,
                "_write_json_atomic",
                side_effect=fail_fixed_manifest_once,
            ):
                with self.assertRaisesRegex(
                    OSError, "injected fixed-manifest failure"
                ):
                    runner._publish_complete(
                        result_path=layout.result,
                        manifest_path=layout.manifest,
                        scorecard_path=layout.scorecard,
                        manifest=manifest,
                    )
            self.assertEqual(
                json.loads(layout.manifest.read_text(encoding="utf-8"))[
                    "status"
                ],
                "publication_pending",
            )
            self.assertTrue(layout.fixed_result.exists())
            self.assertTrue(layout.fixed_scorecard.exists())
            self.assertFalse(layout.fixed_manifest.exists())

            runner._publish_complete(
                result_path=layout.result,
                manifest_path=layout.manifest,
                scorecard_path=layout.scorecard,
                manifest=manifest,
            )
            self.assertEqual(
                json.loads(layout.manifest.read_text(encoding="utf-8"))[
                    "status"
                ],
                "completed",
            )
            self.assertEqual(
                json.loads(
                    layout.fixed_manifest.read_text(encoding="utf-8")
                )["status"],
                "completed",
            )

    def test_older_completed_run_does_not_replace_newer_fixed_artifacts(self):
        with self.isolated_layout() as layout:
            layout.result.write_text('{"old": true}\n', encoding="utf-8")
            layout.scorecard.write_text(
                '{"old_score": true}\n', encoding="utf-8"
            )
            old_manifest = {
                "status": "publication_pending",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
            layout.manifest.write_text(
                json.dumps(old_manifest), encoding="utf-8"
            )

            layout.fixed_result.write_text(
                '{"newer": true}\n', encoding="utf-8"
            )
            layout.fixed_scorecard.write_text(
                '{"newer_score": true}\n', encoding="utf-8"
            )
            newer_manifest = {
                "status": "completed",
                "started_at": "2026-02-01T00:00:00+00:00",
            }
            layout.fixed_manifest.write_text(
                json.dumps(newer_manifest), encoding="utf-8"
            )
            fixed_bytes = {
                path: path.read_bytes()
                for path in (
                    layout.fixed_result,
                    layout.fixed_scorecard,
                    layout.fixed_manifest,
                )
            }

            runner._publish_complete(
                result_path=layout.result,
                manifest_path=layout.manifest,
                scorecard_path=layout.scorecard,
                manifest=old_manifest,
            )
            for path, expected in fixed_bytes.items():
                self.assertEqual(path.read_bytes(), expected)
            stored_old = json.loads(
                layout.manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(stored_old["status"], "completed")
            self.assertFalse(
                stored_old["publication"]["published_latest"]
            )

    def test_frozen_118_and_exact_66_difference(self):
        seed_coverage = runner._validate_seed(self.seed)
        self.assertEqual(seed_coverage["valid_unique_conditions"], 118)
        self.assertEqual(seed_coverage["missing_unique_conditions"], 66)
        self.assertEqual(len(self.expected_keys), 184)
        self.assertEqual(len(self.missing_conditions), 66)

        groups: dict[tuple[float, float], set[float]] = {}
        for U, freq, twist, aoa in self.missing_conditions:
            self.assertIn(U, (6.0, 10.0))
            self.assertIn(freq, (2.0, 2.3, 2.6))
            self.assertEqual(aoa, 5.0)
            self.assertNotEqual(twist, 22.5)
            groups.setdefault((U, freq), set()).add(twist)
        self.assertEqual(set(groups), {
            (U, freq)
            for U in (6.0, 10.0)
            for freq in (2.0, 2.3, 2.6)
        })
        expected_twists = {
            0.0,
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
            27.5,
            30.0,
            35.0,
            40.0,
            45.0,
        }
        self.assertTrue(
            all(twists == expected_twists for twists in groups.values())
        )


if __name__ == "__main__":
    unittest.main()
