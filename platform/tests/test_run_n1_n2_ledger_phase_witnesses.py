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

import run_n1_n2_ledger_phase_witnesses as runner  # noqa: E402


class N1N2LedgerPhaseWitnessContractTests(unittest.TestCase):
    def test_plan_path_never_imports_warp_or_solver(self):
        imported = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "warp" or name == "_v2_robo":
                raise AssertionError(f"GPU import on plan path: {name}")
            return imported(name, *args, **kwargs)

        stream = io.StringIO()
        with (
            mock.patch("builtins.__import__", side_effect=guarded_import),
            contextlib.redirect_stdout(stream),
        ):
            self.assertEqual(runner.main(["--print-plan"]), 0)
        payload = json.loads(stream.getvalue())
        self.assertFalse(payload["gpu_initialized"])
        self.assertEqual(
            payload["contract"]["grid"],
            {
                "n_cycle": 2,
                "nc": 4,
                "ns": 8,
                "steps_per_cycle": 240,
                "wake_rows": 240,
            },
        )
        self.assertFalse(
            payload["contract"]["production_grid_claim_allowed"]
        )

    def test_case_contract_covers_all_figures_and_separates_phase_identity(self):
        cases = runner._case_contracts()
        self.assertEqual(len(cases), 15)
        coverage = {case.coverage for case in cases}
        self.assertIn("fig17_18_shared_turn_neighbour", coverage)
        self.assertIn("fig18_U_boundary_low", coverage)
        self.assertIn("fig18_U_boundary_high", coverage)
        self.assertIn("fig18_frequency_boundary_low", coverage)
        self.assertIn("fig19_aoa_boundary_low", coverage)
        self.assertIn("fig19_aoa_boundary_high", coverage)

        figure16 = [
            case for case in cases if case.family == "figure16_phase"
        ]
        self.assertEqual(len(figure16), 3)
        self.assertTrue(
            all(
                case.twist_phase_deg == -90.0
                for case in figure16
            )
        )
        for physical_id in ("W1", "W2", "W3", "W4", "W5", "W6"):
            branches = [
                case
                for case in cases
                if case.family == "mean"
                and case.physical_id == physical_id
            ]
            self.assertEqual(
                {case.twist_phase_deg for case in branches},
                {-90.0, 90.0},
            )

    def test_n5p1c_is_a_hard_gate(self):
        cases = runner._case_contracts()
        report = runner._kinematic_identity_gate(cases)
        self.assertTrue(report["passed"])
        self.assertEqual(report["claim_id"], "N5.1c")
        self.assertEqual(
            report["paper_identity_twist_phase_deg"],
            -90.0,
        )

        invalid = {
            "id": "N5",
            "children": [
                {
                    "id": "N5.1c",
                    "state": "partial",
                    "freeze": True,
                }
            ],
        }
        with (
            mock.patch.object(runner, "_load_yaml", return_value=invalid),
            self.assertRaisesRegex(
                runner.WitnessContractError,
                "validated/frozen",
            ),
        ):
            runner._kinematic_identity_gate(cases)

    def test_n5_gate_uses_bytes_and_hash_from_same_source_snapshot(self):
        cases = runner._case_contracts()
        closure, n5_bytes = runner._source_closure_snapshot()
        relative = str(
            runner.N5_YAML.resolve().relative_to(runner.ROOT.resolve())
        )
        expected_hash = closure["members"][relative]
        with mock.patch.object(
            runner,
            "_load_yaml",
            wraps=runner._load_yaml,
        ) as loader:
            gate = runner._kinematic_identity_gate(
                cases,
                n5_yaml_bytes=n5_bytes,
                expected_n5_yaml_sha256=expected_hash,
                source_closure_sha256=closure["members_sha256"],
            )
        self.assertTrue(gate["source_snapshot_bound"])
        self.assertEqual(gate["claim_yaml_sha256"], expected_hash)
        self.assertEqual(
            gate["source_closure_sha256"],
            closure["members_sha256"],
        )
        self.assertEqual(loader.call_args.kwargs["content"], n5_bytes)
        with self.assertRaisesRegex(
            runner.WitnessContractError,
            "do not match",
        ):
            runner._kinematic_identity_gate(
                cases,
                n5_yaml_bytes=n5_bytes,
                expected_n5_yaml_sha256="0" * 64,
                source_closure_sha256=closure["members_sha256"],
            )

    def test_source_closure_includes_governance_and_evidence_inputs(self):
        closure, _ = runner._source_closure_snapshot()
        members = closure["members"]
        required = {
            "platform/claim_dag.py",
            "platform/docs/diag/"
            "research_n2_chordwise_pressure_primary_literature_20260729.md",
            "platform/docs/diag/research_n3_spatial_loads_20260727.md",
            "researchpaper/"
            "Meng2025_Drones_FlappingTwist_RoboEagle_SOURCE.pdf",
        }
        self.assertTrue(required <= set(members))
        for relative in runner.CONDITIONAL_SOURCE_FILES:
            path = runner.ROOT / relative
            self.assertEqual(relative in members, path.is_file())
        self.assertEqual(
            closure["conditional_files"],
            list(runner.CONDITIONAL_SOURCE_FILES),
        )

    def test_solver_import_roots_and_local_fluxvortex_are_authoritative(self):
        original = list(sys.path)
        try:
            sys.path.insert(0, "/untrusted/site-packages")
            roots = runner._prepend_solver_import_paths()
            self.assertEqual(
                sys.path[:2],
                [str(runner.SRC.resolve()), str(runner.PLATFORM.resolve())],
            )
            self.assertEqual(tuple(sys.path[:2]), roots)
        finally:
            sys.path[:] = original

        local_module = types.SimpleNamespace(
            __file__=str(runner.SRC / "fluxvortex" / "__init__.py")
        )
        identity = runner._module_file_identity(
            local_module,
            module_name="fluxvortex",
            required_root=runner.SRC,
        )
        self.assertEqual(identity["relative_path"], "fluxvortex/__init__.py")
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / "fluxvortex.py"
            external.write_text("# external\n", encoding="utf-8")
            with self.assertRaisesRegex(
                runner.WitnessContractError,
                "resolved outside",
            ):
                runner._module_file_identity(
                    types.SimpleNamespace(__file__=str(external)),
                    module_name="fluxvortex",
                    required_root=runner.SRC,
                )

    def test_solver_loader_uses_local_module_and_resolved_solver_device(self):
        solver = mock.Mock(name="gpu_run_twist")
        device = types.SimpleNamespace(name="Mock GPU")
        fake_warp = types.SimpleNamespace(
            init=mock.Mock(),
            get_device=mock.Mock(return_value=device),
        )
        fake_fluxvortex = types.SimpleNamespace(
            __file__=str(runner.SRC / "fluxvortex" / "__init__.py")
        )
        fake_solver_module = types.SimpleNamespace(gpu_run_twist=solver)
        fake_config = types.SimpleNamespace(DEVICE="cuda:1")
        modules = {
            "warp": fake_warp,
            "fluxvortex": fake_fluxvortex,
            "_v2_robo": fake_solver_module,
            "fluxvortex.warp_fsi.config": fake_config,
        }
        with (
            mock.patch.object(
                runner.importlib,
                "import_module",
                side_effect=modules.__getitem__,
            ),
            mock.patch.object(
                runner,
                "_numeric_runtime_fingerprint",
                return_value={"fingerprint_sha256": "f" * 64},
            ) as fingerprint,
        ):
            loaded, runtime = runner._load_solver()
        self.assertIs(loaded, solver)
        self.assertEqual(runtime["fingerprint_sha256"], "f" * 64)
        fake_warp.init.assert_called_once_with()
        fake_warp.get_device.assert_called_once_with("cuda:1")
        self.assertIs(
            fingerprint.call_args.kwargs["fluxvortex_module"],
            fake_fluxvortex,
        )

    def test_runtime_fingerprint_binds_dtype_cuda_and_build_identities(self):
        class FakeDevice:
            alias = "cuda:1"
            name = "Mock GPU"
            ordinal = 1
            is_cuda = True
            arch = 89
            uuid = "GPU-test-uuid"
            pci_bus_id = "00000000:01:00"

            def __str__(self):
                return "cuda:1"

        fake_warp = types.SimpleNamespace(
            __file__=runner.__file__,
            __version__="1.test",
            config=types.SimpleNamespace(
                version="1.test",
                _git_commit_hash="abc",
                cuda_arch_suffix=None,
                llvm_cuda=False,
                verify_cuda=False,
                fast_math=False,
                mode="release",
            ),
            get_warp_version=lambda: "native-test",
            get_warp_clang_version=lambda: "clang-test",
            get_llvm_version=lambda: "llvm-test",
            get_host_compiler_version=lambda: "gcc-test",
            is_cuda_available=lambda: True,
            get_cuda_driver_version=lambda: (12, 9),
            get_cuda_toolkit_version=lambda: (12, 8),
            get_nvrtc_version=lambda: (12, 8),
            get_cuda_supported_archs=lambda: [80, 89],
        )
        fake_config = types.SimpleNamespace(
            DTYPE="wp.float32",
            NP_DTYPE=np.float32,
            DEVICE="cuda:1",
            dtype_name=lambda: "float32",
        )
        local_fluxvortex = types.SimpleNamespace(
            __file__=str(runner.SRC / "fluxvortex" / "__init__.py")
        )
        environment = {
            "FLUXV_DTYPE": "float32",
            "FLUXV_DEVICE": "cuda:1",
            "PYTHONHASHSEED": "17",
        }
        with (
            mock.patch.dict(runner.os.environ, environment, clear=False),
            mock.patch.object(
                runner,
                "_numpy_build_identity",
                return_value={
                    "version": np.__version__,
                    "build_sha256": "numpy-build",
                },
            ),
        ):
            runtime = runner._numeric_runtime_fingerprint(
                wp=fake_warp,
                device=FakeDevice(),
                fluxvortex_module=local_fluxvortex,
                solver_config=fake_config,
            )
        self.assertEqual(
            runtime["environment"]["FLUXV_DTYPE"],
            "float32",
        )
        self.assertEqual(runtime["environment"]["FLUXV_DEVICE"], "cuda:1")
        self.assertEqual(runtime["environment"]["PYTHONHASHSEED"], "17")
        self.assertEqual(runtime["solver_config"]["dtype_name"], "float32")
        self.assertEqual(runtime["solver_config"]["numpy_dtype"], "float32")
        warp = runtime["warp_runtime"]
        self.assertEqual(warp["cuda_driver_version"], [12, 9])
        self.assertEqual(warp["cuda_toolkit_version"], [12, 8])
        self.assertEqual(warp["device"]["uuid"], "GPU-test-uuid")
        self.assertEqual(warp["device"]["compute_arch"], "sm_89")
        self.assertEqual(
            runtime["fluxvortex_module"]["relative_path"],
            "fluxvortex/__init__.py",
        )
        self.assertEqual(len(runtime["fingerprint_sha256"]), 64)

    def test_figure16_digitization_is_rawly_archived_as_published_filtered(self):
        arrays = runner._parse_fig16_digitization()
        self.assertEqual(len(arrays), 12)
        for kind in ("T", "L"):
            for twist in ("0", "22p5", "45"):
                time_values = arrays[f"{kind}_tw{twist}_t_over_T"]
                force_values = arrays[f"{kind}_tw{twist}_force_N"]
                self.assertEqual(time_values.shape, force_values.shape)
                self.assertTrue(np.all(np.diff(time_values) > 0.0))
                self.assertGreater(time_values.size, 20)

    def test_solver_half_wing_channels_are_scaled_once_to_reported_pair(self):
        case = runner.CaseContract(
            case_id="scale",
            family="test",
            physical_id="scale",
            U_m_s=8.0,
            frequency_Hz=2.0,
            nominal_twist_deg=22.5,
            aoa_deg=0.0,
            twist_phase_deg=-90.0,
            roles=("test",),
            coverage="test",
        )
        flat = {
            "n2.separation_booked_solver_accumulator_N": np.array(
                [[1.0, 0.0, 3.0], [2.0, 0.0, 4.0]]
            ),
            "n2.separation_panel_candidate_force_body_N": np.array(
                [
                    [[5.0, 0.0, 7.0]],
                    [[6.0, 0.0, 8.0]],
                ]
            ),
            "phase_solver_rad": np.array([0.0, np.pi]),
            "phase_paper_rad": np.array([-0.5 * np.pi, 0.5 * np.pi]),
        }
        result = {
            "Xh_les": np.array([9.0, 10.0]),
            "Lh_les": np.array([11.0, 12.0]),
        }
        bundle = runner._augment_raw_bundle(flat, result, case, 2)
        np.testing.assert_array_equal(
            bundle[
                "diagnostic.n1.leading_edge_suction_"
                "solver_accumulator_body_force_N"
            ][:, 0],
            [9.0, 10.0],
        )
        # At AoA=0, reported thrust is -Fx.  The one-mesh raw channel is
        # mirrored exactly once, hence -2*[9, 10], not -1x or -4x.
        np.testing.assert_array_equal(
            bundle[
                "diagnostic.wind.n1_leading_edge_suction_T_N"
            ],
            [-18.0, -20.0],
        )
        np.testing.assert_array_equal(
            bundle["diagnostic.wind.n2_separation_booked_T_N"],
            [-2.0, -4.0],
        )


class AtomicAndResumeTests(unittest.TestCase):
    def test_atomic_json_preserves_previous_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            runner._write_json_atomic(path, {"version": 1})
            with (
                mock.patch.object(
                    runner.os,
                    "replace",
                    side_effect=OSError("injected"),
                ),
                self.assertRaisesRegex(OSError, "injected"),
            ):
                runner._write_json_atomic(path, {"version": 2})
            self.assertEqual(runner._load_json(path), {"version": 1})
            self.assertFalse(
                path.with_name(f".{path.name}.partial").exists()
            )

    def test_resume_accepts_only_hash_intact_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            cases = runner._case_contracts()
            source = {
                "schema": "content-addressed-source-closure-v1",
                "members": {"solver.py": "a" * 64},
                "members_sha256": "b" * 64,
            }
            identity = {"passed": True, "claim_id": "N5.1c"}
            campaign = runner._open_campaign(
                output,
                resume=False,
                source_closure=source,
                cases=cases,
                identity_gate=identity,
            )
            case = cases[0]
            paths = runner._case_artifact_paths(output, case.case_id)
            runner._write_npz_atomic(
                paths["raw_npz"],
                {"value": np.arange(3)},
            )
            runner._write_json_atomic(
                paths["schema_json"],
                {"schema": "test"},
            )
            runner._write_json_atomic(
                paths["evidence_json"],
                {"evidence": True},
            )
            campaign["cases"][case.case_id] = {
                "artifacts": {
                    name: runner._artifact_identity(path, output)
                    for name, path in paths.items()
                }
            }
            runner._write_json_atomic(
                output / "run_manifest.json",
                campaign,
            )

            resumed = runner._open_campaign(
                output,
                resume=True,
                source_closure=source,
                cases=cases,
                identity_gate=identity,
            )
            self.assertIn(case.case_id, resumed["cases"])

            paths["evidence_json"].write_text(
                '{"tampered": true}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                runner.WitnessContractError,
                "hash drift",
            ):
                runner._open_campaign(
                    output,
                    resume=True,
                    source_closure=source,
                    cases=cases,
                    identity_gate=identity,
                )

    def test_resume_rejects_extra_case_before_artifact_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            cases = runner._case_contracts()
            source = {
                "schema": "content-addressed-source-closure-v1",
                "members": {"solver.py": "a" * 64},
                "members_sha256": "b" * 64,
            }
            identity = {"passed": True, "claim_id": "N5.1c"}
            campaign = runner._open_campaign(
                output,
                resume=False,
                source_closure=source,
                cases=cases,
                identity_gate=identity,
            )
            campaign["cases"]["not_preregistered"] = {"artifacts": {}}
            runner._write_json_atomic(
                output / "run_manifest.json",
                campaign,
            )
            with self.assertRaisesRegex(
                runner.WitnessContractError,
                "unexpected scientific cases",
            ):
                runner._open_campaign(
                    output,
                    resume=True,
                    source_closure=source,
                    cases=cases,
                    identity_gate=identity,
                )

    def test_source_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "solver.py"
            source.write_text("version = 1\n", encoding="utf-8")
            expected = runner._source_closure_from_paths(
                [source],
                root=root,
            )
            expected["governed_globs"] = []
            source.write_text("version = 2\n", encoding="utf-8")
            current = runner._source_closure_from_paths(
                [source],
                root=root,
            )
            current["governed_globs"] = []
            with (
                mock.patch.object(
                    runner,
                    "_source_closure",
                    return_value=current,
                ),
                self.assertRaisesRegex(
                    runner.WitnessContractError,
                    "source closure drifted",
                ),
            ):
                runner._assert_source_closure(expected)

    def test_resume_numeric_runtime_must_match_exactly(self):
        campaign = {"numeric_runtime": None}
        runtime = {
            "python": "3.x",
            "numpy": "2.x",
            "warp": "1.x",
            "device": "cuda:0",
            "device_name": "GPU",
            "numeric_thread_environment": {"OMP_NUM_THREADS": "1"},
        }
        normalized = runner._register_numeric_runtime(campaign, runtime)
        self.assertEqual(campaign["numeric_runtime"], normalized)
        self.assertEqual(
            runner._register_numeric_runtime(campaign, dict(runtime)),
            normalized,
        )

        changed = dict(runtime)
        changed["device_name"] = "different GPU"
        with self.assertRaisesRegex(
            runner.WitnessContractError,
            "numeric runtime mismatch",
        ):
            runner._register_numeric_runtime(campaign, changed)

    def test_resume_preconditioner_matches_first_force_and_graph_anchor(self):
        def preconditioner(lift, thrust, graph="g" * 64):
            return {
                "L_wind_N": lift,
                "T_wind_N": thrust,
                "claim_graph_identity_sha256": graph,
            }

        first = {
            "preconditioner": preconditioner(10.0, -2.0),
            "completed_case_ids": [],
        }
        current = {
            "preconditioner": preconditioner(10.149, -2.149),
            "completed_case_ids": [],
        }
        campaign = {"sessions": [first, current]}
        report = runner._preconditioner_resume_gate(campaign, current)
        self.assertTrue(report["passed"])
        self.assertFalse(report["reference_is_current_session"])
        self.assertEqual(report["tau_F_N"], 0.15)

        current["preconditioner"] = preconditioner(10.151, -2.0)
        with self.assertRaisesRegex(
            runner.WitnessContractError,
            "force drift",
        ):
            runner._preconditioner_resume_gate(campaign, current)

        current["preconditioner"] = preconditioner(
            10.0,
            -2.0,
            graph="h" * 64,
        )
        with self.assertRaisesRegex(
            runner.WitnessContractError,
            "differs from first session",
        ):
            runner._preconditioner_resume_gate(campaign, current)


if __name__ == "__main__":
    unittest.main()
