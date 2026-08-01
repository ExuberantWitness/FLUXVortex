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

import fig171819_benchmark as benchmark  # noqa: E402
import run_n1_n2_ledger_phase_witnesses as witness  # noqa: E402
import score_n1_n2_ledger_phase_witnesses as scorer  # noqa: E402


def _case(
    model: tuple[float, float] = (0.0, 0.0),
    q1: tuple[float, float] = (0.0, 0.0),
    q2: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    return {
        "model_robust": np.asarray(model, dtype=float),
        "model_raw": np.asarray(model, dtype=float),
        "q1": np.asarray(q1, dtype=float),
        "q2": np.asarray(q2, dtype=float),
    }


def _blank_inputs() -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
]:
    names = {
        endpoint
        for _, _, high, low in scorer.CONTRASTS
        for endpoint in (high, low)
    }
    experiment = {name: np.zeros(2) for name in names}
    cases = {name: _case() for name in names}
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
    arrays["cycle_index"] = np.full(
        scorer.RAW_STEPS, witness.G0_N_CYCLE - 1
    )
    arrays["nc"] = np.full(scorer.RAW_STEPS, witness.G0_NC)
    arrays["ns"] = np.full(scorer.RAW_STEPS, witness.G0_NS)
    arrays["dt_s"] = np.full(scorer.RAW_STEPS, 0.001)
    arrays["time_s"] = arrays["step"].astype(float) * 0.001
    return arrays


def _raw_schema(
    arrays: dict[str, np.ndarray],
    case: witness.CaseContract,
) -> dict[str, object]:
    return {
        "schema": witness.RAW_SCHEMA_VERSION,
        "case_id": case.case_id,
        "stage": "G0_exploratory_quick_identity",
        "snapshot_phase": "post_force_pre_shed",
        "time_window": "last_cycle",
        "processing": "none",
        "figure16_alignment_status": "unresolved_external_kinematics",
        "fields": scorer._array_schema_fields(arrays),
        "array_bundle_sha256": witness._array_bundle_hash(arrays),
    }


def _guards() -> dict[str, dict[str, bool]]:
    return {
        name: {"passed": True}
        for name in witness.REQUIRED_CLAIM_GUARDS
    }


def _claim_gates() -> dict[str, dict[str, object]]:
    return {
        "N1": {
            "state": "validated",
            "freeze": True,
            "implementation": "claim_runtime.components:UVLMComponent",
            "runtime_role": "physics",
        },
        "N2": {
            "state": "partial",
            "freeze": False,
            "implementation": (
                "claim_runtime.components:KirchhoffLBComponent"
            ),
            "runtime_role": "physics",
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case_fixture(
    root: Path,
    case: witness.CaseContract,
    *,
    forged_summary: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    arrays = _valid_raw_arrays()
    schema = _raw_schema(arrays, case)
    base = root / "cases" / case.case_id
    base.parent.mkdir(parents=True)
    raw_path = base.with_suffix(".raw.npz")
    schema_path = base.with_suffix(".schema.json")
    evidence_path = base.with_suffix(".evidence.json")
    with raw_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    schema_path.write_text(
        json.dumps(schema, sort_keys=True), encoding="utf-8"
    )

    guards = _guards()
    gates = _claim_gates()
    nodes = [
        {
            "id": claim_id,
            "state": gate["state"],
            "freeze": gate["freeze"],
            "runtime_role": gate["runtime_role"],
            "implementation": gate["implementation"],
            "implementation_version": "1",
            "implementation_hash": "",
        }
        for claim_id, gate in gates.items()
    ]
    manifest = {
        "closure": "v41",
        "topology": ["N1", "N2"],
        "nodes": nodes,
        "parameter_sources": {},
        "guards": guards,
    }
    graph_hash = witness._claim_graph_identity_sha256(manifest)
    raw_hash = witness._array_bundle_hash(arrays)
    raw_config = {
        "closure": "v41",
        "nc": witness.G0_NC,
        "ns": witness.G0_NS,
        "n_cycle": witness.G0_N_CYCLE,
        "steps_per_cycle": witness.G0_STEPS_PER_CYCLE,
        "wake_rows": witness.G0_STEPS_PER_CYCLE,
        "U_m_s": case.U_m_s,
        "aoa_deg": case.aoa_deg,
        "freq_hz": case.frequency_Hz,
        "twist_amp_deg": case.solver_twist_amplitude_deg,
        "twist_phase_deg": case.twist_phase_deg,
    }
    evidence = {
        "schema": witness.SCHEMA_VERSION,
        "case_contract": witness.asdict(case),
        "stage": "G0_exploratory_quick_identity",
        "production_grid_claim_allowed": False,
        "source_closure_sha256": "a" * 64,
        "raw_guard": {"passed": True},
        "claim_guards": guards,
        "claim_manifest": manifest,
        "claim_manifest_sha256": witness._canonical_hash(manifest),
        "claim_graph_identity_sha256": graph_hash,
        "raw_array_bundle_sha256": raw_hash,
        "claim_raw_config": raw_config,
        "observer_role": "read_only",
        "aerodynamic_formula_modified": False,
        "force_added_by_runner": False,
        "diagnostic_summary": forged_summary or {},
    }
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True), encoding="utf-8"
    )
    artifacts = {
        "raw_npz": {
            "path": str(raw_path.relative_to(root)),
            "sha256": _sha256(raw_path),
        },
        "schema_json": {
            "path": str(schema_path.relative_to(root)),
            "sha256": _sha256(schema_path),
        },
        "evidence_json": {
            "path": str(evidence_path.relative_to(root)),
            "sha256": _sha256(evidence_path),
        },
    }
    campaign: dict[str, object] = {
        "source_closure_sha256": "a" * 64,
        "common_claim_graph_identity_sha256": graph_hash,
        "cases": {
            case.case_id: {
                "artifacts": artifacts,
                "raw_array_bundle_sha256": raw_hash,
                "claim_graph_identity_sha256": graph_hash,
            }
        },
    }
    return campaign, gates


class EndpointContractTests(unittest.TestCase):
    def test_only_raw_endpoint_is_selected_without_force_interpolation(
        self,
    ) -> None:
        measurements = benchmark.load_measurements()
        value, provenance = scorer._endpoint_value(
            measurements["19|b|15"],
            2.6,
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
            scorer._endpoint_value(measurements["19|b|15"], 2.3)

    def test_frozen_closure_contains_scorer_and_measurements(self) -> None:
        members = witness._source_closure()["members"]
        self.assertIn(
            "platform/score_n1_n2_ledger_phase_witnesses.py",
            members,
        )
        self.assertIn("platform/docs/data.md", members)
        self.assertIn("platform/fig171819_benchmark.py", members)


class DecisionMatrixTests(unittest.TestCase):
    def test_no_material_residual_is_offset_only(self) -> None:
        experiment, cases = _blank_inputs()
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_robust",
        )
        self.assertEqual(report["status"], "NO_DECISION_OFFSET_ONLY")

    def test_two_independent_families_only_support_n2(self) -> None:
        experiment, cases = _blank_inputs()
        experiment["W2"] = np.asarray([1.0, 0.0])
        experiment["W3"] = np.asarray([1.0, 0.0])
        experiment["W6"] = np.asarray([1.0, 0.0])
        cases["W2"] = _case(q1=(-1.0, 0.4), q2=(1.0, 0.0))
        cases["W3"] = _case(q1=(-1.0, 0.4), q2=(1.0, 0.0))
        cases["W6"] = _case(q1=(-1.0, 0.0), q2=(1.0, 0.4))
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_robust",
        )
        self.assertEqual(
            report["status"],
            "ACTIVE_N2_MISSING_PRESSURE_HYPOTHESIS",
        )
        self.assertEqual(report["unique_q2_families"], ["AoA", "U"])

    def test_same_material_contrast_supported_by_both_is_no_decision(
        self,
    ) -> None:
        experiment, cases = _blank_inputs()
        experiment["W2"] = np.asarray([1.0, 0.0])
        cases["W2"] = _case(q1=(1.0, 0.2), q2=(1.0, -0.2))
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_robust",
        )
        self.assertIn(
            report["status"],
            {
                "NO_DECISION_MULTIPLE_EXPLANATIONS",
                "NO_DECISION_COLLINEAR_OR_PROCESSING_SENSITIVE",
            },
        )

    def test_collinear_templates_fail_closed(self) -> None:
        experiment, cases = _blank_inputs()
        experiment["W2"] = np.asarray([1.0, 0.0])
        cases["W2"] = _case(q1=(1.0, 0.0), q2=(2.0, 0.0))
        report = scorer._classify(
            experiment=experiment,
            cases=cases,
            model_key="model_robust",
        )
        self.assertEqual(
            report["status"],
            "NO_DECISION_COLLINEAR_OR_PROCESSING_SENSITIVE",
        )


class ArtifactIntegrityTests(unittest.TestCase):
    def test_artifact_rejects_traversal_absolute_and_symlink_escape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "run"
            root.mkdir()
            outside = parent / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            digest = _sha256(outside)
            identities = (
                {"path": "../outside.json", "sha256": digest},
                {"path": str(outside.resolve()), "sha256": digest},
            )
            for identity in identities:
                with self.subTest(path=identity["path"]):
                    with self.assertRaises(scorer.ScoreContractError):
                        scorer._verify_artifact(root, identity)
            (root / "escape.json").symlink_to(outside)
            with self.assertRaisesRegex(
                scorer.ScoreContractError, "escaped run directory"
            ):
                scorer._verify_artifact(
                    root,
                    {"path": "escape.json", "sha256": digest},
                )

    def test_raw_schema_rejects_one_sample_even_if_self_consistent(
        self,
    ) -> None:
        case = witness._case_contracts()[0]
        arrays = {
            name: value[:1].copy()
            for name, value in _valid_raw_arrays().items()
        }
        schema = _raw_schema(arrays, case)
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "raw shape mismatch"
        ):
            scorer._validate_raw_schema(
                arrays=arrays,
                schema=schema,
                case=case,
            )

    def test_forged_diagnostic_summary_cannot_change_raw_metrics(
        self,
    ) -> None:
        case = witness._case_contracts()[0]
        forged = {
            "reported_robust_cycle_force": {
                "T_N": 1.0e12,
                "L_N": -1.0e12,
            },
            "raw_cycle_mean_total": {
                "T_N": 1.0e12,
                "L_N": -1.0e12,
            },
            "raw_cycle_mean_channels": {
                "n1_leading_edge_suction": {
                    "T_N": 1.0e12,
                    "L_N": -1.0e12,
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign, gates = _write_case_fixture(
                root, case, forged_summary=forged
            )
            loaded = scorer._load_case(root, campaign, case, gates)
        np.testing.assert_array_equal(loaded["model_robust"], np.zeros(2))
        np.testing.assert_array_equal(loaded["model_raw"], np.zeros(2))
        np.testing.assert_array_equal(loaded["q1"], np.zeros(2))


class Figure16IntegrityTests(unittest.TestCase):
    def test_self_consistent_malformed_figure16_trace_is_rejected(
        self,
    ) -> None:
        source = witness.FIG16_SOURCE.read_bytes()
        expected = scorer._parse_frozen_fig16_source(source)
        arrays = {name: value.copy() for name, value in expected.items()}
        arrays["T_tw0_t_over_T"][1] = arrays["T_tw0_t_over_T"][0]
        bundle_hash = witness._array_bundle_hash(arrays)
        schema = {
            "schema": witness.RAW_SCHEMA_VERSION,
            "fields": scorer._array_schema_fields(arrays),
            "data_role": "published_filtered_gt",
            "source": str(witness.FIG16_SOURCE.relative_to(scorer.ROOT)),
            "source_sha256": scorer.FIG16_SOURCE_SHA256,
            "force_conversion": (
                "published grams-force * 9.81 / 1000 -> N"
            ),
            "published_processing": (
                "5th-order Butterworth, 8 Hz; instrument raw unavailable"
            ),
            "runner_processing": "none",
            "refilter_digitization": False,
            "alignment_status": "unresolved_external_kinematics",
            "model_force_cross_correlation_allowed": False,
            "array_bundle_sha256": bundle_hash,
        }
        identity = {
            "array_bundle_sha256": bundle_hash,
            "data_role": "published_filtered_gt",
            "alignment_status": "unresolved_external_kinematics",
        }
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "malformed trace"
        ):
            scorer._validate_fig16_arrays(
                arrays=arrays,
                schema=schema,
                identity=identity,
                expected=expected,
            )


class CampaignGateTests(unittest.TestCase):
    @staticmethod
    def _valid_campaign() -> dict[str, object]:
        cases = witness._case_contracts()
        graph = "b" * 64
        source = "a" * 64
        numpy_identity = {
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
        flux_path = source_root / "fluxvortex" / "__init__.py"
        runtime = {
            "python": {
                "version": "3.test",
                "executable": "/test/python",
                "implementation": "CPython",
            },
            "platform": "test-platform",
            "environment": {
                "FLUXV_DTYPE": "float64",
                "FLUXV_DEVICE": "cuda:0",
                "PYTHONHASHSEED": None,
                "CUDA_VISIBLE_DEVICES": None,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": None,
                "MKL_NUM_THREADS": None,
                "NUMEXPR_NUM_THREADS": None,
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
                "cuda_driver_version": 12000,
                "cuda_toolkit_version": 12000,
                "nvrtc_version": 12000,
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
                    "uuid": None,
                    "pci_bus_id": None,
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
                "path": str(flux_path),
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
        resolved = {
            "closure": "v41",
            "nc": witness.G0_NC,
            "ns": witness.G0_NS,
            "n_cycle": witness.G0_N_CYCLE,
            "steps_per_cycle": witness.G0_STEPS_PER_CYCLE,
            "wake_rows": witness.G0_STEPS_PER_CYCLE,
            "U": 8.0,
            "freq": 2.6,
            "twist_amp_deg": 0.0,
            "aoa_deg": 5.0,
            "twist_phase_deg": witness.PRODUCTION_PHASE_DEG,
        }
        return {
            "cases": {case.case_id: {} for case in cases},
            "completed_case_count": len(cases),
            "common_claim_graph_identity_sha256": graph,
            "source_closure_sha256": source,
            "source_closure": {"members_sha256": source},
            "numeric_runtime": runtime,
            "sessions": [
                {
                    "numeric_runtime": runtime,
                    "source_closure_sha256": source,
                    "completed_case_ids": [
                        case.case_id for case in cases
                    ],
                    "preconditioner": {
                        "excluded_from_scientific_metrics": True,
                        "purpose": "excluded test preconditioner",
                        "case_contract": (
                            scorer._preconditioner_case_contract()
                        ),
                        "claim_graph_identity_sha256": graph,
                        "L_wind_N": 0.0,
                        "T_wind_N": 0.0,
                        "wall_s": 0.0,
                        "claim_guards": _guards(),
                        "resolved_call": resolved,
                    },
                }
            ],
        }

    def test_extra_case_is_rejected(self) -> None:
        campaign = self._valid_campaign()
        campaign["cases"]["EXTRA"] = {}
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "case identity set mismatch"
        ):
            scorer._validate_campaign_structure(
                campaign, witness._case_contracts()
            )

    def test_extra_kinematic_gate_field_is_rejected(self) -> None:
        cases = witness._case_contracts()
        source_closure, identity_gate = witness._campaign_inputs(cases)
        campaign = {
            "source_closure": source_closure,
            "kinematic_identity_gate": identity_gate,
        }
        scorer._validate_kinematic_identity_gate(campaign, cases)
        campaign["kinematic_identity_gate"]["forged_gate"] = True
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "kinematic identity gate mismatch"
        ):
            scorer._validate_kinematic_identity_gate(campaign, cases)

    def test_cross_session_preconditioner_drift_is_rejected(self) -> None:
        campaign = self._valid_campaign()
        resumed = copy.deepcopy(campaign["sessions"][0])
        resumed["completed_case_ids"] = []
        resumed["preconditioner"]["L_wind_N"] = scorer.TAU_F_N + 0.001
        campaign["sessions"].append(resumed)
        with self.assertRaisesRegex(
            scorer.ScoreContractError, "preconditioner force drift"
        ):
            scorer._validate_campaign_structure(
                campaign, witness._case_contracts()
            )

    def test_semantic_claim_gates_preserve_open_n2p5_no_implementation(
        self,
    ) -> None:
        members = {}
        for path in (scorer.N1_YAML, scorer.N2_YAML, witness.PREREG):
            members[str(path.relative_to(scorer.ROOT))] = _sha256(path)
        campaign = {
            "source_closure": {"members": members},
            "preregistration": {
                "path": str(witness.PREREG.relative_to(scorer.ROOT)),
                "sha256": _sha256(witness.PREREG),
            },
        }
        gates = scorer._semantic_claim_gates(campaign)
        self.assertEqual(gates["N1"]["state"], "validated")
        self.assertTrue(gates["N1"]["freeze"])
        self.assertEqual(gates["N2.2"]["state"], "falsified")
        self.assertEqual(gates["N2.5"]["state"], "open")
        self.assertFalse(
            gates["N2.5"]["candidate_implementation_authorized"]
        )
        self.assertIn(gates["N2.6"]["state"], {"partial", "open"})
        self.assertTrue(gates["N2.6"]["movable"])


class FailureSemanticsTests(unittest.TestCase):
    def test_cli_invalid_evidence_never_authorizes_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "score.json"
            stream = io.StringIO()
            with (
                mock.patch.object(
                    scorer,
                    "score",
                    side_effect=scorer.ScoreContractError("injected"),
                ),
                contextlib.redirect_stdout(stream),
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
            self.assertFalse(
                report["candidate_implementation_authorized"]
            )
            self.assertFalse(
                report["n2p5_candidate_implementation_authorized"]
            )
            self.assertFalse(
                report[
                    "next_n2p6_shadow_preregistration_authorized"
                ]
            )


if __name__ == "__main__":
    unittest.main()
