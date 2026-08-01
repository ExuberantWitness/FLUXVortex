from __future__ import annotations

import builtins
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import run_n1_n2_n3_aoa_ladder_witnesses as runner  # noqa: E402


class AoALadderRunnerContractTests(unittest.TestCase):
    def test_exact_eight_case_science_contract(self):
        cases = runner._case_contracts()
        self.assertEqual(
            [case.case_id for case in cases],
            [
                "aoa_f1p4_A0",
                "aoa_f1p4_A5",
                "aoa_f1p4_A10",
                "aoa_f1p4_A15",
                "aoa_f2p6_A0",
                "aoa_f2p6_A5",
                "aoa_f2p6_A10",
                "aoa_f2p6_A15",
            ],
        )
        self.assertTrue(
            all(
                case.U_m_s == 8.0
                and case.nominal_twist_deg == 22.5
                and case.solver_twist_amplitude_deg == 11.25
                and case.twist_phase_deg == -90.0
                for case in cases
            )
        )
        self.assertEqual(
            {
                (case.frequency_Hz, case.aoa_deg)
                for case in cases
            },
            {
                (frequency, aoa)
                for frequency in (1.4, 2.6)
                for aoa in (0.0, 5.0, 10.0, 15.0)
            },
        )

    def test_campaign_contract_freezes_grid_templates_and_no_interpolation(self):
        contract = runner._campaign_contract(runner._case_contracts())
        self.assertEqual(contract["expected_unique_science_solver_calls"], 8)
        self.assertEqual(
            contract["expected_fresh_session_total_solver_invocations"],
            9,
        )
        self.assertEqual(
            contract["grid"],
            {
                "nc": 4,
                "ns": 8,
                "n_cycle": 2,
                "steps_per_cycle": 240,
                "wake_rows": 240,
            },
        )
        self.assertEqual(contract["raw_expected_field_count"], 92)
        self.assertEqual(contract["raw_stage"], runner.RAW_STAGE)
        self.assertEqual(set(contract["templates"]), {"component_order", "Q1", "Q2", "Q3"})
        self.assertFalse(
            contract["figure19_experiment"]["force_interpolation_allowed"]
        )
        self.assertFalse(contract["candidate_implementation_authorized"])

    def test_preconditioner_is_one_excluded_zero_twist_call(self):
        case = runner.PRECONDITIONER_CASE
        self.assertEqual(case.case_id, "excluded_current_source_preconditioner")
        self.assertEqual(case.nominal_twist_deg, 0.0)
        self.assertEqual(case.twist_phase_deg, -90.0)
        self.assertIn("excluded_numeric_runtime_preconditioner", case.roles)
        contract = runner._campaign_contract(runner._case_contracts())
        self.assertEqual(contract["excluded_preconditioner_calls_per_session"], 1)

    def test_full_v41_raw_configuration_is_frozen(self):
        case = runner._case_contracts()[0]
        config = runner._expected_claim_raw_config(case)
        self.assertEqual(
            set(config),
            {
                "closure",
                "nc",
                "ns",
                "n_cycle",
                "steps_per_cycle",
                "wake_rows",
                "U_m_s",
                "aoa_deg",
                "freq_hz",
                "flap_amp_deg",
                "twist_amp_deg",
                "twist_phase_deg",
                "real_geom",
                "sym",
                "lb_closure",
                "lb_hybrid",
                "lb_cds",
                "lb_cds_mem",
                "lb_cds_f2gate",
                "lb_cds_zonly",
                "lb_cds_signed",
                "lb_chop_zonly",
                "lb_ct",
                "lb_cla3d",
                "lb_lesp_crit",
                "d_para_at_U8_N",
                "attached_drag",
            },
        )
        self.assertEqual(config["flap_amp_deg"], 22.5)
        self.assertEqual(config["lb_hybrid"], 0.0)
        self.assertEqual(config["lb_cds"], 2.5)
        self.assertEqual(config["d_para_at_U8_N"], 0.5)
        self.assertEqual(config["attached_drag"], "uiuc")

    def test_plan_path_does_not_import_gpu_modules(self):
        imported = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"warp", "_v2_robo"}:
                raise AssertionError(f"GPU import on plan path: {name}")
            return imported(name, *args, **kwargs)

        fake_closure = {
            "members_sha256": "a" * 64,
            "members": {},
        }
        fake_gate = {
            "passed": True,
            "source_snapshot_bound": True,
        }
        stream = io.StringIO()
        with (
            mock.patch(
                "builtins.__import__",
                side_effect=guarded_import,
            ),
            mock.patch.object(
                runner,
                "_campaign_inputs",
                return_value=(fake_closure, fake_gate),
            ),
            contextlib.redirect_stdout(stream),
        ):
            self.assertEqual(runner.main(["--print-plan"]), 0)
        payload = json.loads(stream.getvalue())
        self.assertFalse(payload["gpu_initialized"])
        self.assertEqual(payload["campaign_stage"], runner.CAMPAIGN_STAGE)

    def test_n5_gate_uses_same_snapshot_bytes_and_requires_minus90(self):
        cases = runner._case_contracts()
        n5_bytes = runner.N5_YAML.read_bytes()
        digest = runner._sha256_bytes(n5_bytes)
        gate = runner._kinematic_identity_gate(
            cases,
            n5_yaml_bytes=n5_bytes,
            expected_n5_yaml_sha256=digest,
            source_closure_sha256="b" * 64,
        )
        self.assertTrue(gate["source_snapshot_bound"])
        self.assertTrue(
            gate["all_science_cases_use_experiment_identity_minus90"]
        )
        self.assertEqual(gate["claim_yaml_sha256"], digest)

    def test_source_closure_names_new_and_old_governed_files(self):
        required = {
            "platform/run_n1_n2_n3_aoa_ladder_witnesses.py",
            "platform/score_n1_n2_n3_aoa_ladder_witnesses.py",
            "platform/tests/test_run_n1_n2_n3_aoa_ladder_witnesses.py",
            "platform/tests/test_score_n1_n2_n3_aoa_ladder_witnesses.py",
            "platform/docs/diag/n1_n2_n3_aoa_ladder_prereg_20260730.md",
            "platform/run_n1_n2_ledger_phase_witnesses.py",
            "platform/score_n1_n2_ledger_phase_witnesses.py",
            "platform/docs/data.md",
            "platform/fig171819_benchmark.py",
        }
        self.assertTrue(required <= set(runner.DIRECT_SOURCE_FILES))
        with (
            mock.patch.object(
                runner,
                "DIRECT_SOURCE_FILES",
                ("platform/definitely_missing_g0c_source.py",),
            ),
            self.assertRaisesRegex(
                runner.WitnessContractError,
                "mandatory G0c",
            ),
        ):
            runner._source_paths()

    def test_preloaded_governed_entry_modules_are_rejected_before_gpu(self):
        cases = runner._case_contracts()
        closure, _ = runner._campaign_inputs(cases)
        for module_name in ("_v2_robo", "lb_sweep118"):
            with (
                self.subTest(module_name=module_name),
                mock.patch.dict(
                    sys.modules,
                    {module_name: types.ModuleType(module_name)},
                ),
                self.assertRaisesRegex(
                    runner.WitnessContractError,
                    "preloaded before source binding",
                ),
            ):
                runner._load_bound_solver(closure, cases)

    def test_science_wrapper_enforces_92_fields_and_raw_stage(self):
        case = runner._case_contracts()[0]
        good_bundle = {
            f"field_{index}": np.zeros(1)
            for index in range(runner.EXPECTED_RAW_FIELD_COUNT)
        }
        expected_resolved = {"closure": "v41"}
        evidence = {
            "stage": runner.RAW_STAGE,
            "resolved_call": expected_resolved,
            "claim_raw_config": runner._expected_claim_raw_config(case),
        }
        execution_binding = {"binding_sha256": "e" * 64}
        with (
            mock.patch.object(
                runner,
                "_build_solver_call",
                return_value={},
            ),
            mock.patch.object(
                runner,
                "_assert_bound_resolved_call",
                return_value=expected_resolved,
            ),
            mock.patch.object(
                runner.base,
                "_execute_case",
                return_value=(good_bundle, evidence),
            ),
        ):
            bundle, augmented = runner._execute_science_case(
                mock.Mock(),
                case,
                base_config={},
                execution_binding=execution_binding,
                case_contract_sha256="c" * 64,
            )
        self.assertEqual(len(bundle), 92)
        self.assertEqual(
            augmented["campaign_stage"],
            runner.CAMPAIGN_STAGE,
        )
        self.assertEqual(augmented["schema"], runner.SCHEMA_VERSION)
        self.assertEqual(
            augmented["campaign_schema"],
            runner.SCHEMA_VERSION,
        )
        self.assertFalse(augmented["candidate_implementation_authorized"])

        with (
            mock.patch.object(
                runner,
                "_build_solver_call",
                return_value={},
            ),
            mock.patch.object(
                runner,
                "_assert_bound_resolved_call",
                return_value=expected_resolved,
            ),
            mock.patch.object(
                runner.base,
                "_execute_case",
                return_value=(
                    dict(list(good_bundle.items())[:-1]),
                    evidence,
                ),
            ),
            self.assertRaisesRegex(
                runner.WitnessContractError,
                "exactly 92",
            ),
        ):
            runner._execute_science_case(
                mock.Mock(),
                case,
                base_config={},
                execution_binding=execution_binding,
                case_contract_sha256="c" * 64,
            )

    def test_resume_rejects_unregistered_science_case(self):
        cases = runner._case_contracts()
        closure = {
            "members_sha256": "d" * 64,
            "members": {},
        }
        gate = {"passed": True}
        execution_binding = {"binding_sha256": "e" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"
            output.mkdir()
            campaign = runner._new_campaign(
                source_closure=closure,
                cases=cases,
                identity_gate=gate,
                execution_binding=execution_binding,
            )
            campaign["cases"]["forged_extra_case"] = {}
            runner.base._write_json_atomic(
                output / "run_manifest.json",
                campaign,
            )
            with self.assertRaisesRegex(
                runner.WitnessContractError,
                "unexpected G0c",
            ):
                runner._open_campaign(
                    output,
                    resume=True,
                    source_closure=closure,
                    cases=cases,
                    identity_gate=gate,
                    execution_binding=execution_binding,
                )


if __name__ == "__main__":
    unittest.main()
