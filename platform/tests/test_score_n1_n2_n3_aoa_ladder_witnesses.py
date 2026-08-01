from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import run_n1_n2_n3_aoa_ladder_witnesses as witness  # noqa: E402
import score_n1_n2_n3_aoa_ladder_witnesses as scorer  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blank_cases() -> dict[str, dict[str, np.ndarray]]:
    return {
        case_id: {
            "model_robust": np.zeros(2),
            "model_raw": np.zeros(2),
            "q1_raw": np.zeros(2),
            "q1_robust": np.zeros(2),
            "q2_raw": np.zeros(2),
            "q2_robust": np.zeros(2),
            "q3_raw": np.zeros(2),
            "q3_robust": np.zeros(2),
        }
        for case_id in scorer.EXPECTED_CASES
    }


def _ladder_from_deltas(
    deltas_by_context: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for frequency, context in ((1.4, "f1p4"), (2.6, "f2p6")):
        cumulative = np.zeros(2)
        output[f"aoa_f{str(frequency).replace('.', 'p')}_A0"] = (
            cumulative.copy()
        )
        for aoa, delta in zip(
            (5, 10, 15), deltas_by_context[context]
        ):
            cumulative = cumulative + np.asarray(delta, dtype=float)
            output[
                f"aoa_f{str(frequency).replace('.', 'p')}_A{aoa}"
            ] = cumulative.copy()
    return output


def _classification_fixture(
    selected: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
]:
    residual_deltas = {
        "f1p4": [
            np.asarray([1.0, 0.8]),
            np.asarray([0.9, 1.1]),
            np.asarray([1.2, 0.7]),
        ],
        "f2p6": [
            np.asarray([0.8, 1.2]),
            np.asarray([1.1, 0.9]),
            np.asarray([0.7, 1.0]),
        ],
    }
    opposition_1 = {
        "f1p4": [
            np.asarray([-1.0, 0.2]),
            np.asarray([0.1, -0.8]),
            np.asarray([-0.6, -0.4]),
        ],
        "f2p6": [
            np.asarray([-0.7, 0.3]),
            np.asarray([0.2, -1.1]),
            np.asarray([-1.2, -0.1]),
        ],
    }
    opposition_2 = {
        "f1p4": [
            np.asarray([0.3, -1.1]),
            np.asarray([-0.9, 0.4]),
            np.asarray([-0.2, -0.7]),
        ],
        "f2p6": [
            np.asarray([0.4, -0.8]),
            np.asarray([-1.3, 0.2]),
            np.asarray([-0.5, -1.0]),
        ],
    }
    experiment = _ladder_from_deltas(residual_deltas)
    cases = _blank_cases()
    competitors = [name for name in scorer.TEMPLATE_KEYS if name != selected]
    ladders = {
        selected: _ladder_from_deltas(residual_deltas),
        competitors[0]: _ladder_from_deltas(opposition_1),
        competitors[1]: _ladder_from_deltas(opposition_2),
    }
    for template, values in ladders.items():
        for case_id, value in values.items():
            cases[case_id][f"{template}_raw"] = value
            cases[case_id][f"{template}_robust"] = value
    return experiment, cases


def _valid_raw_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    bool_fields = {
        "n3.event_active",
        "n3.event_onset",
        "n3.tau_v_reset",
    }
    integer_fields = {
        "step",
        "last_cycle_step",
        "cycle_index",
        "nc",
        "ns",
    }
    for name, shape in scorer.RAW_EXPECTED_SHAPES.items():
        if name in bool_fields:
            arrays[name] = np.zeros(shape, dtype=bool)
        elif name in integer_fields:
            arrays[name] = np.zeros(shape, dtype=np.int64)
        elif name == "snapshot_phase":
            arrays[name] = np.full(shape, "post_force_pre_shed")
        elif name == "diagnostic.alignment_source_code":
            arrays[name] = np.full(
                shape, "unresolved_external_kinematics"
            )
        else:
            arrays[name] = np.zeros(shape, dtype=np.float64)
    arrays["last_cycle_step"] = np.arange(scorer.RAW_STEPS)
    arrays["step"] = np.arange(scorer.RAW_STEPS, 2 * scorer.RAW_STEPS)
    arrays["cycle_index"] = np.ones(
        scorer.RAW_STEPS, dtype=np.int64
    )
    arrays["nc"] = np.full(scorer.RAW_STEPS, scorer.RAW_NC)
    arrays["ns"] = np.full(scorer.RAW_STEPS, scorer.RAW_NS)
    arrays["dt_s"] = np.full(scorer.RAW_STEPS, 0.001)
    arrays["time_s"] = arrays["step"].astype(float) * 0.001
    return arrays


def _raw_schema(
    arrays: dict[str, np.ndarray],
    case: object,
) -> dict[str, object]:
    return {
        "schema": witness.RAW_SCHEMA_VERSION,
        "case_id": case.case_id,
        "stage": witness.RAW_STAGE,
        "snapshot_phase": "post_force_pre_shed",
        "time_window": "last_cycle",
        "processing": "none",
        "figure16_alignment_status": "unresolved_external_kinematics",
        "fields": scorer._array_schema_fields(arrays),
        "array_bundle_sha256": witness.base._array_bundle_hash(arrays),
    }


def _guards() -> dict[str, dict[str, bool]]:
    return {
        name: {"passed": True}
        for name in witness.base.REQUIRED_CLAIM_GUARDS
    }


class FixedContractTests(unittest.TestCase):
    def test_exact_eight_cases_and_raw_shape_contract(self) -> None:
        cases = scorer._case_contracts()
        self.assertEqual(len(cases), 8)
        self.assertEqual(
            {case.case_id for case in cases},
            set(scorer.EXPECTED_CASES),
        )
        self.assertEqual(len(scorer.RAW_EXPECTED_SHAPES), 92)
        self.assertEqual(scorer.RAW_STEPS, 240)

    def test_only_frozen_raw_frequency_endpoints_are_used(self) -> None:
        measurements = scorer.benchmark.load_measurements()
        for aoa in (0, 5, 10, 15):
            for panel in ("a", "b"):
                curve = measurements[f"19|{panel}|{aoa}"]
                for frequency in (1.4, 2.6):
                    value, provenance = scorer._endpoint_value(
                        curve, frequency
                    )
                    self.assertTrue(np.isfinite(value))
                    self.assertFalse(provenance["force_interpolated"])
                    self.assertLessEqual(
                        abs(provenance["endpoint_delta"]),
                        provenance["endpoint_tolerance"],
                    )
                with self.assertRaisesRegex(
                    scorer.ScoreContractError,
                    "not an authorized raw endpoint",
                ):
                    scorer._endpoint_value(curve, 2.0)

    def test_ground_truth_is_parsed_from_the_verified_byte_snapshot(
        self,
    ) -> None:
        frozen_bytes = scorer.DATA_SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            replaced_path = Path(temporary) / "data.md"
            replaced_path.write_bytes(b"not the frozen measurement source\n")

            measurements = scorer.benchmark.load_measurements(
                replaced_path,
                source_bytes=frozen_bytes,
            )
            validation = scorer.benchmark.validate_measurement_contract(
                measurements,
                source_path=replaced_path,
                source_bytes=frozen_bytes,
            )

        self.assertTrue(validation["passed"], msg=validation)
        self.assertTrue(
            validation["source"]["parsed_from_verified_bytes"]
        )
        self.assertEqual(
            validation["source"]["sha256"],
            scorer.DATA_SOURCE_SHA256,
        )
        self.assertEqual(len(measurements), 50)
        self.assertAlmostEqual(
            measurements["19|b|15"].values_g[-1],
            scorer.benchmark.load_measurements()[
                "19|b|15"
            ].values_g[-1],
        )


class DecisionRuleTests(unittest.TestCase):
    def test_strict_material_threshold(self) -> None:
        cases = _blank_cases()
        experiment = {
            name: np.zeros(2) for name in scorer.EXPECTED_CASES
        }
        experiment["aoa_f1p4_A5"] = np.asarray(
            [scorer.TAU_CONTRAST_N, -scorer.TAU_CONTRAST_N]
        )
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_raw",
        )
        self.assertEqual(report["status"], "NO_DECISION_OFFSET_ONLY")

    def test_unique_n1_requires_ledger_audit_but_not_mutation(self) -> None:
        experiment, cases = _classification_fixture("q1")
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_robust",
        )
        self.assertEqual(report["status"], "N1_LEDGER_AUDIT_REQUIRED")
        self.assertEqual(report["selected_template"], "q1")

    def test_unique_n2_authorizes_only_pressure_hypothesis(self) -> None:
        experiment, cases = _classification_fixture("q2")
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_robust",
        )
        self.assertEqual(
            report["status"], "ACTIVE_N2_6_SHADOW_PREREG_ALLOWED"
        )
        self.assertEqual(report["selected_template"], "q2")

    def test_unique_n3_authorizes_only_new_state_hypothesis(self) -> None:
        experiment, cases = _classification_fixture("q3")
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_robust",
        )
        self.assertEqual(
            report["status"],
            "ACTIVE_N3_SPATIAL_STATE_AUDIT_REQUIRED",
        )
        self.assertEqual(report["selected_template"], "q3")

    def test_one_context_is_insufficient(self) -> None:
        experiment, cases = _classification_fixture("q2")
        for case_id in tuple(experiment):
            if "f2p6" in case_id:
                experiment[case_id] = np.zeros(2)
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_raw",
        )
        self.assertEqual(
            report["status"], "NO_DECISION_OFFSET_ONLY"
        )

    def test_any_row_with_multiple_supports_is_no_decision(self) -> None:
        experiment, cases = _classification_fixture("q2")
        q1 = _ladder_from_deltas(
            {
                "f1p4": [
                    np.asarray([0.8, 1.0]),
                    np.asarray([-0.7, 0.2]),
                    np.asarray([0.1, -1.0]),
                ],
                "f2p6": [
                    np.asarray([-0.5, 0.2]),
                    np.asarray([0.6, 1.4]),
                    np.asarray([0.2, -0.9]),
                ],
            }
        )
        for case_id, value in q1.items():
            cases[case_id]["q1_raw"] = value
            cases[case_id]["q1_robust"] = value
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_raw",
        )
        self.assertEqual(
            report["status"], "NO_DECISION_MULTIPLE_EXPLANATIONS"
        )

    def test_multi_support_precedes_a_collinear_template_gate(self) -> None:
        residual_deltas = {
            context: [np.asarray([1.0, 1.0])] * 3
            for context in ("f1p4", "f2p6")
        }
        experiment = _ladder_from_deltas(residual_deltas)
        cases = _blank_cases()
        shared = _ladder_from_deltas(residual_deltas)
        reverse = _ladder_from_deltas(
            {
                context: [np.asarray([-1.0, -1.0])] * 3
                for context in ("f1p4", "f2p6")
            }
        )
        for case_id in cases:
            for view in ("raw", "robust"):
                cases[case_id][f"q1_{view}"] = shared[case_id]
                cases[case_id][f"q2_{view}"] = shared[case_id]
                cases[case_id][f"q3_{view}"] = reverse[case_id]
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_raw",
        )
        self.assertEqual(
            report["status"], "NO_DECISION_MULTIPLE_EXPLANATIONS"
        )
        self.assertFalse(report["identifiability"]["passed"])

    def test_inactive_template_fails_closed(self) -> None:
        experiment, cases = _classification_fixture("q2")
        for case in cases.values():
            case["q3_raw"] = np.zeros(2)
            case["q3_robust"] = np.zeros(2)
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_raw",
        )
        self.assertEqual(
            report["status"], "NO_DECISION_COLLINEAR"
        )
        self.assertIn(
            "inactive_template_column",
            report["identifiability"]["reason"],
        )

    def test_collinear_templates_fail_closed(self) -> None:
        experiment, cases = _classification_fixture("q3")
        for case in cases.values():
            for view in ("raw", "robust"):
                case[f"q1_{view}"] = -case[f"q3_{view}"]
                case[f"q2_{view}"] = -2.0 * case[f"q3_{view}"]
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_raw",
        )
        self.assertEqual(
            report["status"], "NO_DECISION_COLLINEAR"
        )
        self.assertGreater(
            report["identifiability"]["condition_number_2"],
            scorer.COLLINEAR_CONDITION_LIMIT,
        )

    def test_frequency_mixed_requires_each_context_unique_gate(self) -> None:
        residual_deltas = {
            context: [np.asarray([1.0, 1.0])] * 3
            for context in ("f1p4", "f2p6")
        }
        experiment = _ladder_from_deltas(residual_deltas)
        cases = _blank_cases()
        template_deltas = {
            "q1": {
                "f1p4": [
                    np.asarray([1.0, 1.0]),
                    np.asarray([-1.0, -1.0]),
                    np.asarray([-1.0, -1.0]),
                ],
                "f2p6": [np.asarray([-1.0, 0.0])] * 3,
            },
            "q2": {
                "f1p4": [np.asarray([-1.0, 0.0])] * 3,
                "f2p6": [
                    np.asarray([1.0, 1.0]),
                    np.asarray([-1.0, -1.0]),
                    np.asarray([-1.0, -1.0]),
                ],
            },
            "q3": {
                "f1p4": [np.asarray([0.0, -1.0])] * 3,
                "f2p6": [np.asarray([0.0, -1.0])] * 3,
            },
        }
        for template, deltas in template_deltas.items():
            ladder = _ladder_from_deltas(deltas)
            for case_id, value in ladder.items():
                cases[case_id][f"{template}_raw"] = value
                cases[case_id][f"{template}_robust"] = value
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_raw",
        )
        self.assertEqual(
            report["status"], "NO_DECISION_INSUFFICIENT_UNIQUENESS"
        )
        self.assertFalse(
            any(
                audit["passed"]
                for audit in report["candidate_audit"].values()
            )
        )

    def test_raw_robust_status_must_match_exactly(self) -> None:
        robust = {"status": "ACTIVE_N2_6_SHADOW_PREREG_ALLOWED"}
        raw = {"status": "ACTIVE_N3_SPATIAL_STATE_AUDIT_REQUIRED"}
        self.assertEqual(
            scorer._final_status(robust, raw),
            "NO_DECISION_PROCESSING_SENSITIVE",
        )


class RawEvidenceTests(unittest.TestCase):
    def test_raw_q3_is_negative_two_wing_booked_direct_mean(self) -> None:
        case = next(
            item
            for item in scorer._case_contracts()
            if item.case_id == "aoa_f1p4_A0"
        )
        arrays = _valid_raw_arrays()
        arrays["n3.ds_booked_solver_accumulator_N"][:, 0] = 1.0
        arrays["n3.ds_booked_solver_accumulator_N"][:, 2] = 2.0
        metrics = scorer._raw_case_metrics(arrays, case)
        np.testing.assert_allclose(metrics["q3_raw"], [2.0, -4.0])
        np.testing.assert_allclose(
            metrics["q3_robust"], [2.0, -4.0]
        )

    def test_exact_raw_schema_rejects_extra_field(self) -> None:
        case = scorer._case_contracts()[0]
        arrays = _valid_raw_arrays()
        arrays["forged.summary_vote"] = np.zeros(scorer.RAW_STEPS)
        schema = _raw_schema(arrays, case)
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "raw field identity mismatch"
        ):
            scorer._validate_raw_schema(
                arrays=arrays, schema=schema, case=case
            )

    def test_one_sample_bundle_is_rejected(self) -> None:
        case = scorer._case_contracts()[0]
        arrays = {
            name: value[:1].copy()
            for name, value in _valid_raw_arrays().items()
        }
        schema = _raw_schema(arrays, case)
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "raw shape mismatch"
        ):
            scorer._validate_raw_schema(
                arrays=arrays, schema=schema, case=case
            )

    def test_nonfinite_raw_value_is_rejected(self) -> None:
        case = scorer._case_contracts()[0]
        arrays = _valid_raw_arrays()
        arrays["reported_pair_wind_lift_N"][7] = np.nan
        schema = _raw_schema(arrays, case)
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "non-finite raw fields"
        ):
            scorer._validate_raw_schema(
                arrays=arrays, schema=schema, case=case
            )

    def test_n3_channel_ledger_is_independently_checked(self) -> None:
        case = scorer._case_contracts()[0]
        arrays = _valid_raw_arrays()
        arrays["n3.booked_solver_accumulator_total_N"][0, 0] = 1.0
        arrays["total_solver_accumulator_body_force_N"][0, 0] = 1.0
        arrays["reported_pair_body_force_N"][0, 0] = 2.0
        arrays["reported_pair_wind_thrust_N"][0] = -2.0
        with self.assertRaisesRegex(
            scorer.ScoreContractError,
            "N3 direct booked-channel identity failed",
        ):
            scorer._validate_raw_identities(arrays, case)


class ArtifactAttackTests(unittest.TestCase):
    def test_traversal_absolute_and_symlink_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            run = parent / "run"
            run.mkdir()
            outside = parent / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            digest = _sha256(outside)
            for identity in (
                {"path": "../outside.json", "sha256": digest},
                {"path": str(outside.resolve()), "sha256": digest},
            ):
                with self.subTest(identity=identity):
                    with self.assertRaises(scorer.ScoreContractError):
                        scorer._verify_artifact(run, identity)
            (run / "escape.json").symlink_to(outside)
            with self.assertRaisesRegex(
                scorer.ScoreContractError, "escaped run directory"
            ):
                scorer._verify_artifact(
                    run,
                    {"path": "escape.json", "sha256": digest},
                )


class CampaignGateTests(unittest.TestCase):
    @staticmethod
    def _runtime() -> dict[str, object]:
        numpy_identity: dict[str, object] = {
            "version": "2.test",
            "module": {
                "module": "numpy",
                "path": "/test/numpy/__init__.py",
                "sha256": "c" * 64,
            },
            "core_extension": {
                "module": "numpy._core._multiarray_umath",
                "path": "/test/numpy/_multiarray_umath.so",
                "sha256": "d" * 64,
            },
            "build_config": {"test": True},
        }
        numpy_identity["build_sha256"] = witness._canonical_hash(
            numpy_identity
        )
        source_root = scorer.ROOT / "src"
        runtime: dict[str, object] = {
            "python": {
                "version": "3.test",
                "executable": "/test/python",
                "implementation": "CPython",
            },
            "platform": "test-platform",
            "environment": {
                "FLUXV_DTYPE": "float64",
                "FLUXV_DEVICE": "cuda:0",
                "PYTHONHASHSEED": "0",
                "CUDA_VISIBLE_DEVICES": "0",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "numpy": numpy_identity,
            "warp_runtime": {
                "version": "1.test",
                "module": {
                    "module": "warp",
                    "path": "/test/warp/__init__.py",
                    "sha256": "e" * 64,
                },
                "native_version": "1.test",
                "clang_version": "test",
                "llvm_version": "test",
                "host_compiler_version": "test",
                "cuda_available": True,
                "cuda_driver_version": 13000,
                "cuda_toolkit_version": 12090,
                "nvrtc_version": 12090,
                "cuda_supported_archs": [89],
                "config": {
                    "version": "1.test",
                    "_git_commit_hash": "test",
                    "cuda_arch_suffix": False,
                    "llvm_cuda": False,
                    "verify_cuda": False,
                    "fast_math": False,
                    "mode": "release",
                },
                "device": {
                    "text": "cuda:0",
                    "alias": "cuda:0",
                    "name": "test GPU",
                    "ordinal": 0,
                    "is_cuda": True,
                    "arch": 89,
                    "compute_arch": "sm_89",
                    "uuid": "GPU-test",
                    "pci_bus_id": "0000:01:00.0",
                },
            },
            "solver_config": {
                "dtype_name": "float64",
                "dtype": "<class 'numpy.float64'>",
                "numpy_dtype": "float64",
                "device": "cuda:0",
            },
            "fluxvortex_module": {
                "module": "fluxvortex",
                "path": str(
                    source_root / "fluxvortex" / "__init__.py"
                ),
                "sha256": "f" * 64,
                "required_root": str(source_root),
                "relative_path": "fluxvortex/__init__.py",
            },
            "authoritative_import_roots": [
                str(source_root.resolve()),
                str(scorer.PLATFORM),
            ],
        }
        runtime["fingerprint_sha256"] = witness._canonical_hash(runtime)
        return runtime

    @classmethod
    def _campaign(cls) -> dict[str, object]:
        cases = scorer._case_contracts()
        graph = "b" * 64
        source = "a" * 64
        runtime = cls._runtime()
        members = {
            "platform/_v2_robo.py": "1" * 64,
            "platform/lb_sweep118.py": "2" * 64,
        }
        root = scorer.ROOT.resolve()

        def module_identity(
            module_name: str,
            relative: str,
        ) -> dict[str, object]:
            return {
                "module": module_name,
                "path": str((root / relative).resolve()),
                "sha256": members[relative],
                "required_root": str(root),
                "relative_path": relative,
            }

        resolved_preconditioner = {
            "closure": "v41",
            "nc": 4,
            "ns": 8,
            "n_cycle": 2,
            "steps_per_cycle": 240,
            "wake_rows": 240,
            "U": 8.0,
            "freq": 2.6,
            "twist_amp_deg": 0.0,
            "aoa_deg": 5.0,
            "twist_phase_deg": -90.0,
        }
        loaded = {
            "_v2_robo": module_identity(
                "_v2_robo", "platform/_v2_robo.py"
            ),
            "lb_sweep118": module_identity(
                "lb_sweep118", "platform/lb_sweep118.py"
            ),
        }
        base_config = {"test_fixture": True}
        binding: dict[str, object] = {
            "schema": "g0c-execution-binding-v1",
            "source_closure_sha256": source,
            "entry_modules": dict(loaded),
            "loaded_governed_modules": loaded,
            "solver_callable": {
                "module": "_v2_robo",
                "qualname": "gpu_run_twist",
            },
            "base_config": base_config,
            "base_config_sha256": witness._canonical_hash(base_config),
            "resolved_calls": {
                witness.PRECONDITIONER_CASE.case_id: (
                    resolved_preconditioner
                ),
                **{
                    case.case_id: {"case_id": case.case_id}
                    for case in cases
                },
            },
        }
        binding["binding_sha256"] = witness._canonical_hash(binding)
        preconditioner = {
            "excluded_from_scientific_metrics": True,
            "purpose": "excluded test preconditioner",
            "case_contract": witness._jsonable(
                witness.asdict(witness.PRECONDITIONER_CASE)
            ),
            "claim_graph_identity_sha256": graph,
            "L_wind_N": 0.0,
            "T_wind_N": 0.0,
            "wall_s": 0.0,
            "claim_guards": _guards(),
            "resolved_call": resolved_preconditioner,
            "execution_binding_sha256": binding["binding_sha256"],
        }
        return {
            "cases": {case.case_id: {} for case in cases},
            "completed_case_count": 8,
            "common_claim_graph_identity_sha256": graph,
            "source_closure_sha256": source,
            "source_closure": {
                "members_sha256": source,
                "members": members,
            },
            "execution_binding": binding,
            "execution_binding_sha256": binding["binding_sha256"],
            "numeric_runtime": runtime,
            "sessions": [
                {
                    "numeric_runtime": runtime,
                    "source_closure_sha256": source,
                    "execution_binding": binding,
                    "execution_binding_sha256": binding[
                        "binding_sha256"
                    ],
                    "completed_case_ids": [
                        case.case_id for case in cases
                    ],
                    "preconditioner": preconditioner,
                }
            ],
        }

    def test_extra_case_is_rejected(self) -> None:
        campaign = self._campaign()
        campaign["cases"]["EXTRA"] = {}
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "case identity set mismatch"
        ):
            scorer._validate_campaign_structure(
                campaign, scorer._case_contracts()
            )

    def test_cross_session_preconditioner_drift_is_rejected(self) -> None:
        campaign = self._campaign()
        resumed = copy.deepcopy(campaign["sessions"][0])
        resumed["completed_case_ids"] = []
        resumed["preconditioner"]["L_wind_N"] = (
            scorer.TAU_F_N + 0.001
        )
        campaign["sessions"].append(resumed)
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "preconditioner force drift"
        ):
            scorer._validate_campaign_structure(
                campaign, scorer._case_contracts()
            )

    def test_runtime_dtype_drift_is_rejected(self) -> None:
        campaign = self._campaign()
        campaign["numeric_runtime"]["solver_config"][
            "numpy_dtype"
        ] = "float32"
        campaign["sessions"][0]["numeric_runtime"] = campaign[
            "numeric_runtime"
        ]
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "solver numeric config"
        ):
            scorer._validate_campaign_structure(
                campaign, scorer._case_contracts()
            )

    def test_execution_module_hash_cannot_diverge_from_closure(self) -> None:
        campaign = self._campaign()
        binding = copy.deepcopy(campaign["execution_binding"])
        binding["entry_modules"]["_v2_robo"]["sha256"] = "9" * 64
        binding["loaded_governed_modules"]["_v2_robo"]["sha256"] = (
            "9" * 64
        )
        binding.pop("binding_sha256")
        binding["binding_sha256"] = witness._canonical_hash(binding)
        campaign["execution_binding"] = binding
        campaign["execution_binding_sha256"] = binding["binding_sha256"]
        campaign["sessions"][0]["execution_binding"] = binding
        campaign["sessions"][0]["execution_binding_sha256"] = binding[
            "binding_sha256"
        ]
        campaign["sessions"][0]["preconditioner"][
            "execution_binding_sha256"
        ] = binding["binding_sha256"]
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "execution module identity"
        ):
            scorer._validate_campaign_structure(
                campaign, scorer._case_contracts()
            )


class FailureSemanticsTests(unittest.TestCase):
    def test_invalid_evidence_exits_two_and_authorizes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "score.json"
            with (
                mock.patch.object(
                    scorer,
                    "score",
                    side_effect=scorer.ScoreContractError("injected"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = scorer.main(
                    [
                        "--run",
                        str(root / "missing"),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "INVALID_EVIDENCE")
            for name in (
                "claim_state_modified",
                "candidate_implementation_authorized",
                "n2p5_candidate_implementation_authorized",
                "falsified_candidate_reactivation_authorized",
                "next_n2p6_shadow_preregistration_authorized",
                "next_n3_aoa_state_shadow_preregistration_authorized",
                "n1_ledger_audit_required",
            ):
                self.assertFalse(report[name], name)


if __name__ == "__main__":
    unittest.main()
