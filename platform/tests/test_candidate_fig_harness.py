from __future__ import annotations

import json
import math
import subprocess
import types
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_benchmark as benchmark  # noqa: E402
import lb_sweep_candidate as runner  # noqa: E402
import plot_candidate_overlay as overlay  # noqa: E402


def strict_v1_record(L=1.0, T=2.0):
    return {
        "L": float(L),
        "T": float(T),
        "L_wind_v41_counterfactual": float(L) - 0.1,
        "T_wind_v41_counterfactual": float(T) + 0.1,
        "claim_guards": {
            name: {"passed": True}
            for name in runner._N3_ONLY_REQUIRED_GUARDS
        },
        "claim_manifest": {
            "closure": runner.N3_ONLY_CLOSURE,
            "internal_stages": [
                {
                    "id": runner.N3_ONLY_CLAIM,
                    "runtime_owner": "N3",
                    "runtime_binding": "internal_stage",
                }
            ],
        },
        "n3_spatial_n3only": {
            "closure": runner.N3_ONLY_CLOSURE,
            "claim_node": runner.N3_ONLY_CLAIM,
        },
    }


class CandidateScopeTests(unittest.TestCase):
    def test_scope_counts_and_nesting_are_fixed(self):
        self.assertEqual(
            {name: len(value) for name, value in runner.SCOPE_CONDITIONS.items()},
            {
                "smoke3": 3,
                "representative32": 32,
                "confirmed151": 151,
                "conditional184": 184,
            },
        )
        self.assertLessEqual(
            set(runner.SCOPE_CONDITIONS["smoke3"]),
            set(runner.SCOPE_CONDITIONS["representative32"]),
        )
        self.assertLessEqual(
            set(runner.SCOPE_CONDITIONS["representative32"]),
            set(runner.SCOPE_CONDITIONS["confirmed151"]),
        )
        self.assertEqual(
            set(runner.SCOPE_CONDITIONS["conditional184"]),
            set(benchmark.CONDITIONS),
        )

    def test_representative32_has_preregistered_12_plus_10_plus_10_shape(self):
        conditions = set(runner.REPRESENTATIVE32)
        fig17 = {
            (
                8.0,
                benchmark.FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ,
                twist,
                5.0,
            )
            for twist in benchmark.TWS
        }
        fig18 = {
            (U, freq, 22.5, 5.0)
            for U in (6.0, 10.0)
            for freq in benchmark.FS
        }
        fig19 = {
            (8.0, freq, 22.5, aoa)
            for aoa in (0.0, 15.0)
            for freq in benchmark.FS
        }
        self.assertEqual(conditions, fig17 | fig18 | fig19)
        self.assertEqual((len(fig17), len(fig18), len(fig19)), (12, 10, 10))

    def test_production_and_quick_step_contracts(self):
        condition = (8.0, 2.0, 22.5, 5.0)
        self.assertEqual(
            runner._resolved_step_grid(runner.PRODUCTION_GRID, condition),
            (720, 720),
        )
        self.assertEqual(
            runner._resolved_step_grid(runner.QUICK_GRID, condition),
            (60, 60),
        )

    def test_candidate_and_resume_paths_cannot_escape_candidate_root(self):
        with self.assertRaises(ValueError):
            runner._candidate_dir("../escape", "20260729_190000")
        with self.assertRaises(ValueError):
            runner._candidate_dir("valid", "../escape")
        with self.assertRaises(ValueError):
            runner._resume_dir("/tmp/outside", "valid")
        self.assertEqual(
            runner._candidate_dir("valid", "20260729_190000").parts[-3:],
            ("valid", "runs", "20260729_190000"),
        )

    def test_debug_prefix_is_never_reported_as_complete(self):
        self.assertEqual(
            runner._final_status(
                valid_count=3,
                selected_count=3,
                max_conditions=3,
            ),
            "incomplete_debug_prefix",
        )
        self.assertEqual(
            runner._final_status(
                valid_count=151,
                selected_count=151,
                max_conditions=None,
            ),
            "complete",
        )
        self.assertEqual(
            runner._final_status(
                valid_count=2,
                selected_count=3,
                max_conditions=None,
            ),
            "completed_with_failures",
        )

    def test_nonfinite_solver_force_is_rejected(self):
        with self.assertRaises(FloatingPointError):
            runner._finite_force_values({"L_wind": math.nan, "T_wind": 1.0})
        with self.assertRaises(FloatingPointError):
            runner._finite_force_values({"L_wind": 1.0, "T_wind": math.inf})
        self.assertEqual(
            runner._finite_force_values({"L_wind": 1, "T_wind": -2}),
            (1.0, -2.0),
        )
        self.assertFalse(
            runner._valid_result(
                {
                    "L": 1.0,
                    "T": 2.0,
                    "claim_guards": {"ledger": {"passed": False}},
                }
            )
        )
        self.assertFalse(runner._valid_result({"L": 1.0, "T": 2.0}))
        valid = strict_v1_record()
        self.assertTrue(runner._valid_result(valid))
        missing_counterfactual = dict(valid)
        missing_counterfactual.pop("T_wind_v41_counterfactual")
        self.assertFalse(runner._valid_result(missing_counterfactual))
        wrong_claim = dict(valid)
        wrong_claim["n3_spatial_n3only"] = dict(
            valid["n3_spatial_n3only"],
            claim_node="N3.1j0",
        )
        self.assertFalse(runner._valid_result(wrong_claim))
        missing_manifest = dict(valid)
        missing_manifest.pop("claim_manifest")
        self.assertFalse(runner._valid_result(missing_manifest))
        wrong_owner = dict(valid)
        wrong_owner["claim_manifest"] = {
            "closure": runner.N3_ONLY_CLOSURE,
            "internal_stages": [
                {
                    "id": runner.N3_ONLY_CLAIM,
                    "runtime_owner": "N2",
                    "runtime_binding": "internal_stage",
                }
            ],
        }
        self.assertFalse(runner._valid_result(wrong_owner))
        extra_stage = dict(valid)
        extra_stage["claim_manifest"] = {
            "closure": runner.N3_ONLY_CLOSURE,
            "internal_stages": [
                dict(valid["claim_manifest"]["internal_stages"][0]),
                {
                    "id": "N3.1j0",
                    "runtime_owner": "N3",
                    "runtime_binding": "internal_stage",
                },
            ],
        }
        self.assertFalse(runner._valid_result(extra_stage))
        for required_guard in runner._N3_ONLY_REQUIRED_GUARDS:
            missing_guard = dict(valid)
            missing_guard["claim_guards"] = dict(valid["claim_guards"])
            missing_guard["claim_guards"].pop(required_guard)
            with self.subTest(required_guard=required_guard):
                self.assertFalse(runner._valid_result(missing_guard))

    def test_v1_shadow_has_no_model_overrides_and_is_fingerprinted(self):
        closure = "n3_spatial_edge_pressure_v1_shadow"
        self.assertIn(closure, runner._CLOSURE_MODEL_OVERRIDES)
        self.assertEqual(runner._CLOSURE_MODEL_OVERRIDES[closure], {})
        fingerprints = runner.implementation_fingerprints(closure)
        expected = {
            "platform/lb_sweep_candidate.py",
            (
                "platform/docs/candidates/"
                "n3_spatial_edge_pressure_v1_shadow/PLAN.md"
            ),
            (
                "platform/docs/candidates/"
                "n3_spatial_edge_pressure_v1_shadow/EXECUTION.md"
            ),
            (
                "platform/docs/candidates/"
                "n3_spatial_edge_pressure_v1_shadow/"
                "DATA_EXPOSURE_ADDENDUM.md"
            ),
            "platform/score_n3_shadow_gates.py",
            "platform/plot_candidate_overlay.py",
            "platform/claim_runtime/core.py",
            "platform/claim_runtime/components.py",
            "platform/claim_runtime/p2_spatial_n3only_shadow.py",
            "platform/diff_uvlm_unsteady_gpu.py",
            "platform/flap_flight_validate.py",
            "platform/_v2_robogeom.py",
            "platform/airfoil_geometry.py",
            "src/fluxvortex/warp_fsi/config.py",
            "src/fluxvortex/warp_fsi/batched_solver.py",
        }
        self.assertLessEqual(expected, set(fingerprints))
        self.assertTrue(
            all(
                isinstance(value, str) and len(value) == 64
                for value in fingerprints.values()
            )
        )

    def test_fingerprint_covers_fresh_local_solver_import_closure(self):
        probe = """
import json
import sys
from pathlib import Path
root = Path.cwd().resolve()
sys.path.insert(0, str(root / "platform"))
sys.path.insert(0, str(root))
import _v2_robo
paths = set()
for module in tuple(sys.modules.values()):
    raw = getattr(module, "__file__", None)
    if not raw:
        continue
    try:
        path = Path(raw).resolve()
        path.relative_to(root)
    except (OSError, ValueError):
        continue
    if path.suffix == ".py":
        paths.add(path.relative_to(root).as_posix())
print("LOCAL_IMPORT_CLOSURE=" + json.dumps(sorted(paths)))
"""
        output = subprocess.check_output(
            [sys.executable, "-c", probe],
            cwd=runner.ROOT,
            text=True,
        )
        marker = next(
            line
            for line in output.splitlines()
            if line.startswith("LOCAL_IMPORT_CLOSURE=")
        )
        imported = set(json.loads(marker.split("=", 1)[1]))
        fingerprinted = set(
            runner.implementation_fingerprints(runner.N3_ONLY_CLOSURE)
        )
        self.assertFalse(
            imported - fingerprinted,
            f"unfingerprinted local imports: {sorted(imported - fingerprinted)}",
        )

    def test_candidate_record_preserves_same_call_counterfactual_and_n3_diag(self):
        n3_diag = {
            "closure": runner.N3_ONLY_CLOSURE,
            "claim_node": runner.N3_ONLY_CLAIM,
            "substitution_identity_residual_N": 0.0,
            "production_channels_unchanged": True,
        }
        strict = strict_v1_record()
        claim_manifest = strict["claim_manifest"]
        record = runner._candidate_record(
            {
                "L_wind": 2.0,
                "T_wind": -3.0,
                "L_wind_v41_counterfactual": 1.5,
                "T_wind_v41_counterfactual": -2.5,
                "n3_spatial_n3only": n3_diag,
                "claim_guards": strict["claim_guards"],
                "claim_manifest": claim_manifest,
            },
            wall_seconds=4.0,
            steps_per_cycle=60,
            wake_rows=60,
            closure=runner.N3_ONLY_CLOSURE,
        )
        self.assertEqual(record["L"], 2.0)
        self.assertEqual(record["T"], -3.0)
        self.assertEqual(record["L_wind_v41_counterfactual"], 1.5)
        self.assertEqual(record["T_wind_v41_counterfactual"], -2.5)
        self.assertIs(record["n3_spatial_n3only"], n3_diag)
        self.assertIs(record["claim_manifest"], claim_manifest)

    def test_same_call_counterfactual_is_atomic_and_finite(self):
        with self.assertRaisesRegex(KeyError, "incomplete same-call"):
            runner._candidate_record(
                {
                    "L_wind": 1.0,
                    "T_wind": 2.0,
                    "L_wind_v41_counterfactual": 3.0,
                },
                wall_seconds=0.1,
                steps_per_cycle=60,
                wake_rows=60,
            )
        with self.assertRaisesRegex(FloatingPointError, "counterfactual"):
            runner._candidate_record(
                {
                    "L_wind": 1.0,
                    "T_wind": 2.0,
                    "L_wind_v41_counterfactual": math.inf,
                    "T_wind_v41_counterfactual": 3.0,
                },
                wall_seconds=0.1,
                steps_per_cycle=60,
                wake_rows=60,
            )

    def test_v1_model_argument_allowlist_blocks_physics_changes(self):
        self.assertEqual(
            runner._model_args(
                [("spatial_p2_quadrature", 24)],
                closure=runner.N3_ONLY_CLOSURE,
            ),
            {"spatial_p2_quadrature": 24},
        )
        with self.assertRaisesRegex(ValueError, "forbids model arguments"):
            runner._model_args(
                [("a0_crit", 0.23)],
                closure=runner.N3_ONLY_CLOSURE,
            )
        with self.assertRaisesRegex(ValueError, "16 or 24"):
            runner._model_args(
                [("spatial_p2_quadrature", 12)],
                closure=runner.N3_ONLY_CLOSURE,
            )

    def test_runtime_environment_records_resolved_numerical_identity(self):
        identity = runner.runtime_environment_identity()
        self.assertIn(identity["fluxv_dtype_resolved"], {"float32", "float64"})
        self.assertTrue(identity["fluxv_device_resolved"])
        self.assertTrue(identity["warp_version"])
        self.assertTrue(identity["python_executable"])
        self.assertEqual(
            identity["device_is_cuda"],
            identity["fluxv_device_resolved"].startswith("cuda"),
        )

    def test_run_directory_lock_is_exclusive_and_recoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            with runner._RunDirectoryLock(run_dir):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with runner._RunDirectoryLock(run_dir):
                        pass
            with runner._RunDirectoryLock(run_dir):
                self.assertTrue((run_dir / ".campaign.lock").is_file())

    def test_atomic_json_refuses_nonfinite_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result.json"
            with self.assertRaises(ValueError):
                runner._write_json_atomic(target, {"bad": math.nan})
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_persisted_bare_force_pair_is_not_a_resumable_result(self):
        condition = runner.SMOKE3[0]
        key = runner.condition_key(condition)
        with self.assertRaisesRegex(
            ValueError,
            "invalid persisted record",
        ):
            runner._validate_checkpoint_results(
                {key: {"L": 1.0, "T": 2.0}},
                conditions=(condition,),
                closure=runner.N3_ONLY_CLOSURE,
            )
        failed = {"fail": "RuntimeError: expected test failure"}
        self.assertEqual(
            runner._validate_checkpoint_results(
                {key: failed},
                conditions=(condition,),
                closure=runner.N3_ONLY_CLOSURE,
            ),
            {key: failed},
        )

    def test_seed_run_reuses_only_valid_overlap_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous_root = runner.CANDIDATE_ROOT
            runner.CANDIDATE_ROOT = Path(temporary)
            try:
                seed_dir = (
                    Path(temporary) / "valid" / "runs" / "20260729_190000"
                )
                seed_dir.mkdir(parents=True)
                overlap = runner.condition_key((8.0, 2.6, 0.0, 5.0))
                nonoverlap = runner.condition_key((8.0, 2.6, 5.0, 5.0))
                third = runner.condition_key((8.0, 2.6, 10.0, 5.0))
                source_identity = {
                    "candidate_id": "valid",
                    "closure": runner.N3_ONLY_CLOSURE,
                    "scope": "smoke3",
                    "condition_count": 3,
                    "condition_keys": [overlap, nonoverlap, third],
                    "grid": {"mode": "quick"},
                    "runtime_environment": {"identity": "same"},
                }
                (seed_dir / "config.json").write_text(
                    json.dumps({"run_identity": source_identity}),
                    encoding="utf-8",
                )
                (seed_dir / "candidate_results.json").write_text(
                    json.dumps(
                        {
                            overlap: strict_v1_record(1.0, 2.0),
                            nonoverlap: strict_v1_record(3.0, 4.0),
                            third: strict_v1_record(5.0, 6.0),
                        }
                    ),
                    encoding="utf-8",
                )
                (seed_dir / "status.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "completed_valid": 3,
                            "failed": 0,
                        }
                    ),
                    encoding="utf-8",
                )
                target_identity = dict(
                    source_identity,
                    scope="representative32",
                    condition_count=32,
                    condition_keys=["different"],
                )
                copied, provenance = runner._load_seed_run(
                    seed_dir,
                    candidate_id="valid",
                    target_identity=target_identity,
                    target_conditions=((8.0, 2.6, 0.0, 5.0),),
                )
                self.assertEqual(set(copied), {overlap})
                self.assertEqual(provenance["copied_valid_condition_count"], 1)
                self.assertEqual(provenance["source_status"], "complete")
                with self.assertRaises(ValueError):
                    runner._load_seed_run(
                        seed_dir,
                        candidate_id="valid",
                        target_identity=dict(target_identity, grid={"mode": "prod"}),
                        target_conditions=((8.0, 2.6, 0.0, 5.0),),
                    )

                (seed_dir / "candidate_results.json").write_text(
                    json.dumps({overlap: {"L": 1.0, "T": 2.0}}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid formal-scope records",
                ):
                    runner._load_seed_run(
                        seed_dir,
                        candidate_id="valid",
                        target_identity=target_identity,
                        target_conditions=((8.0, 2.6, 0.0, 5.0),),
                    )
            finally:
                runner.CANDIDATE_ROOT = previous_root

    def test_campaign_failure_is_fail_fast_and_durable(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous_root = runner.CANDIDATE_ROOT
            runner.CANDIDATE_ROOT = Path(temporary)
            calls = []
            fake_warp = types.ModuleType("warp")
            fake_warp.init = lambda: None
            fake_robo = types.ModuleType("_v2_robo")

            def fail_solver(**kwargs):
                calls.append(kwargs)
                raise RuntimeError("solver-boom")

            fake_robo.gpu_run_twist = fail_solver
            runtime_identity = {"numerical_runtime": "fixed-test-runtime"}
            fingerprints = {"platform/source.py": "a" * 64}
            identity = {
                "candidate_id": runner.N3_ONLY_CLOSURE,
                "closure": runner.N3_ONLY_CLOSURE,
                "scope": "smoke3",
                "condition_count": 3,
                "condition_keys": [
                    runner.condition_key(condition)
                    for condition in runner.SMOKE3
                ],
                "grid": {"mode": "quick"},
                "implementation_sha256": fingerprints,
                "runtime_environment": runtime_identity,
            }
            args = runner.build_parser().parse_args(
                [
                    "--candidate-id",
                    runner.N3_ONLY_CLOSURE,
                    "--closure",
                    runner.N3_ONLY_CLOSURE,
                    "--scope",
                    "smoke3",
                    "--quick",
                    "--timestamp",
                    "20260729_235959",
                ]
            )
            try:
                with (
                    mock.patch.object(
                        runner,
                        "_base_model_config",
                        return_value={},
                    ),
                    mock.patch.object(
                        runner,
                        "_run_identity",
                        return_value=identity,
                    ),
                    mock.patch.object(
                        runner,
                        "runtime_environment_identity",
                        return_value=runtime_identity,
                    ),
                    mock.patch.object(
                        runner,
                        "implementation_fingerprints",
                        return_value=fingerprints,
                    ),
                    mock.patch.object(
                        runner,
                        "git_identity",
                        return_value={"commit": "test"},
                    ),
                    mock.patch.dict(
                        sys.modules,
                        {"warp": fake_warp, "_v2_robo": fake_robo},
                    ),
                ):
                    first_key = runner.condition_key(runner.SMOKE3[0])
                    with self.assertRaisesRegex(
                        RuntimeError,
                        first_key,
                    ):
                        runner.run(args)
                self.assertEqual(len(calls), 1)
                run_dir = (
                    Path(temporary)
                    / runner.N3_ONLY_CLOSURE
                    / "runs"
                    / "20260729_235959"
                )
                status = json.loads(
                    (run_dir / "status.json").read_text(encoding="utf-8")
                )
                results = json.loads(
                    (run_dir / "candidate_results.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(status["status"], "failed_fast")
                self.assertEqual(status["completed_valid"], 0)
                self.assertEqual(status["failed"], 1)
                self.assertEqual(status["remaining"], 2)
                self.assertEqual(set(results), {first_key})
                self.assertIn("solver-boom", results[first_key]["fail"])
            finally:
                runner.CANDIDATE_ROOT = previous_root

    def test_post_call_source_drift_rejects_solver_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous_root = runner.CANDIDATE_ROOT
            runner.CANDIDATE_ROOT = Path(temporary)
            fake_warp = types.ModuleType("warp")
            fake_warp.init = lambda: None
            fake_robo = types.ModuleType("_v2_robo")
            strict = strict_v1_record(1.0, 2.0)
            output = {
                "L_wind": strict["L"],
                "T_wind": strict["T"],
                **{
                    key: value
                    for key, value in strict.items()
                    if key not in {"L", "T"}
                },
            }
            fake_robo.gpu_run_twist = lambda **_kwargs: output
            runtime_identity = {"numerical_runtime": "fixed-test-runtime"}
            fingerprints = {"platform/source.py": "a" * 64}
            changed_fingerprints = {"platform/source.py": "b" * 64}
            identity = {
                "candidate_id": runner.N3_ONLY_CLOSURE,
                "closure": runner.N3_ONLY_CLOSURE,
                "scope": "smoke3",
                "condition_count": 3,
                "condition_keys": [
                    runner.condition_key(condition)
                    for condition in runner.SMOKE3
                ],
                "grid": {"mode": "quick"},
                "implementation_sha256": fingerprints,
                "runtime_environment": runtime_identity,
            }
            args = runner.build_parser().parse_args(
                [
                    "--candidate-id",
                    runner.N3_ONLY_CLOSURE,
                    "--closure",
                    runner.N3_ONLY_CLOSURE,
                    "--scope",
                    "smoke3",
                    "--quick",
                    "--timestamp",
                    "20260729_235958",
                ]
            )
            try:
                with (
                    mock.patch.object(
                        runner,
                        "_base_model_config",
                        return_value={},
                    ),
                    mock.patch.object(
                        runner,
                        "_run_identity",
                        return_value=identity,
                    ),
                    mock.patch.object(
                        runner,
                        "runtime_environment_identity",
                        return_value=runtime_identity,
                    ),
                    mock.patch.object(
                        runner,
                        "implementation_fingerprints",
                        side_effect=[
                            fingerprints,
                            changed_fingerprints,
                        ],
                    ),
                    mock.patch.object(
                        runner,
                        "git_identity",
                        return_value={"commit": "test"},
                    ),
                    mock.patch.dict(
                        sys.modules,
                        {"warp": fake_warp, "_v2_robo": fake_robo},
                    ),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "output was not accepted",
                    ):
                        runner.run(args)
                run_dir = (
                    Path(temporary)
                    / runner.N3_ONLY_CLOSURE
                    / "runs"
                    / "20260729_235958"
                )
                status = json.loads(
                    (run_dir / "status.json").read_text(encoding="utf-8")
                )
                results = json.loads(
                    (run_dir / "candidate_results.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(status["status"], "aborted_source_drift")
                self.assertEqual(results, {})
            finally:
                runner.CANDIDATE_ROOT = previous_root


class CandidateOverlayTests(unittest.TestCase):
    def test_same_call_counterfactual_is_extracted_atomically(self):
        extracted = overlay._same_call_counterfactual_results(
            {
                "condition": {
                    "L": 1.0,
                    "T": 2.0,
                    "L_wind_v41_counterfactual": 3.0,
                    "T_wind_v41_counterfactual": 4.0,
                }
            }
        )
        self.assertEqual(extracted, {"condition": {"L": 3.0, "T": 4.0}})
        with self.assertRaisesRegex(ValueError, "complete same-call"):
            overlay._same_call_counterfactual_results(
                {
                    "condition": {
                        "L": 1.0,
                        "T": 2.0,
                        "L_wind_v41_counterfactual": 3.0,
                    }
                }
            )

    def test_path_allocator_never_returns_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / "fig17_candidate_overlay.png"
            original.write_bytes(b"keep-me")
            selected = overlay._non_overwriting_path(
                directory, "fig17_candidate_overlay", ".png"
            )
            self.assertEqual(selected.name, "fig17_candidate_overlay_01.png")
            self.assertEqual(original.read_bytes(), b"keep-me")

    def test_overlay_generation_does_not_replace_existing_figure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            results_path = directory / "candidate_results.json"
            results_path.write_text(
                json.dumps(
                    {
                        benchmark.condition_key((8.0, 2.0, 0.0, 5.0)): {
                            "L": 1.0,
                            "T": -1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            original = directory / "fig17_candidate_overlay.png"
            original.write_bytes(b"sentinel")
            outputs = overlay.generate_overlays(
                results_path,
                candidate_label="test",
                baseline_json=results_path,
                dpi=40,
            )
            self.assertEqual(original.read_bytes(), b"sentinel")
            self.assertEqual(len(outputs), 3)
            self.assertEqual(outputs[0].name, "fig17_candidate_overlay_01.png")
            self.assertTrue(all(path.is_file() for path in outputs))
            manifests = list(directory.glob("candidate_overlay_manifest*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(
                Path(manifest["baseline_json"]),
                results_path.resolve(),
            )

    def test_overlay_can_use_embedded_same_call_counterfactual(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            results_path = directory / "candidate_results.json"
            results_path.write_text(
                json.dumps(
                    {
                        benchmark.condition_key((8.0, 2.0, 0.0, 5.0)): {
                            "L": 1.0,
                            "T": -1.0,
                            "L_wind_v41_counterfactual": 2.0,
                            "T_wind_v41_counterfactual": -2.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            outputs = overlay.generate_overlays(
                results_path,
                candidate_label="test",
                same_call_counterfactual=True,
                dpi=40,
            )
            self.assertEqual(len(outputs), 3)
            manifest_path = next(
                directory.glob("candidate_overlay_manifest*.json")
            )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["baseline_source_role"],
                "same_call_v41_counterfactual",
            )
            self.assertEqual(
                manifest["baseline_label"],
                "V4.1 same-call counterfactual",
            )

    def test_label_is_recovered_from_run_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "candidate" / "runs" / "stamp"
            run_dir.mkdir(parents=True)
            results_path = run_dir / "candidate_results.json"
            results_path.write_text("{}", encoding="utf-8")
            (run_dir / "config.json").write_text(
                json.dumps(
                    {"run_identity": {"candidate_id": "spatial-candidate"}}
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                overlay._inferred_candidate_label(results_path),
                "spatial-candidate",
            )


if __name__ == "__main__":
    unittest.main()
