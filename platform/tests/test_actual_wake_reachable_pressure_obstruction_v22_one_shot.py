"""Definition tests for the S3ai-v2.2 one-shot execution wrapper.

Every execution-path test replaces the frozen 31-history collector and
aggregator with fakes.  Nothing in this module is permitted to construct a
mesh or execute one physical history.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
import os
from contextlib import ExitStack
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import actual_wake_reachable_pressure_obstruction_v2_guard as frozen  # noqa: E402
import actual_wake_reachable_pressure_obstruction_v22_one_shot as one_shot  # noqa: E402


EXTENDED_VALUE_HASH_KEYS = {
    "mass_active",
    "stage_times",
    "measurement_times",
    "stored_window_residual",
    "direct_window_residual",
    "stored_step_residuals",
    "direct_step_residuals",
    "canonical_material_current_trace",
    "body_cut_trace",
    "canonical_material_release",
    "representation_inventory",
    "stored_weak_pressure",
    "direct_weak_pressure",
    "stage_roles_canonical_JSON",
}


def _fake_observation() -> SimpleNamespace:
    stage_roles = (
        "entrance_prestep_full",
        "measured_midpoint",
        "measured_full",
    )
    return SimpleNamespace(
        mass_active=np.eye(7, dtype=np.float64),
        stored_window_residual=np.arange(7, dtype=np.float64) / 8.0,
        direct_window_residual=np.arange(7, dtype=np.float64) / 9.0,
        stored_step_residuals=np.arange(7, dtype=np.float64)[None, :] / 8.0,
        direct_step_residuals=np.arange(7, dtype=np.float64)[None, :] / 9.0,
        stage_arrays=SimpleNamespace(
            stage_times=np.array((0.0, 0.5, 1.0), dtype=np.float64),
            stage_roles=stage_roles,
            measurement_times=np.array(((0.0, 0.5, 1.0),), dtype=np.float64),
            canonical_material_current_trace=np.arange(
                27, dtype=np.float64
            ).reshape(3, 9),
            body_cut_trace=np.arange(27, dtype=np.float64).reshape(3, 9)
            + 1.0,
            canonical_material_release=np.arange(
                27, dtype=np.float64
            ).reshape(3, 9)
            + 2.0,
            representation_inventory=np.arange(
                27, dtype=np.float64
            ).reshape(3, 9)
            + 3.0,
            stored_weak_pressure=np.arange(
                21, dtype=np.float64
            ).reshape(3, 7)
            + 4.0,
            direct_weak_pressure=np.arange(
                21, dtype=np.float64
            ).reshape(3, 7)
            + 5.0,
        ),
    )


def _canonical_role_hash(roles: tuple[str, ...]) -> str:
    encoded = json.dumps(
        list(roles),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_31_case_fixture():
    """Reuse the frozen runner's existing pure-synthetic 31-case definition."""

    test_path = (
        PLATFORM
        / "tests"
        / "test_actual_wake_reachable_pressure_obstruction_v2_definition.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_one_shot_v2_synthetic_fixture",
        test_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the frozen synthetic definition")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    test_class = (
        module.ActualWakeReachablePressureObstructionV2DefinitionTests
    )
    test_class.setUpClass()
    synthetic_test = test_class(
        methodName="test_synthetic_31_case_aggregate_is_complete_json_dict"
    )

    captured = []
    aggregate = frozen.aggregate_frozen_histories

    def capture(observations, contract, **kwargs):
        captured.append((dict(observations), contract))
        return aggregate(observations, contract, **kwargs)

    forbidden = AssertionError("collector, mesh, or march was invoked")
    with (
        patch.object(
            frozen,
            "aggregate_frozen_histories",
            side_effect=capture,
        ),
        patch.object(
            frozen,
            "_collect_frozen_histories",
            side_effect=forbidden,
        ) as collector,
        patch.object(
            frozen,
            "build_canonical_diamond_wing",
            side_effect=forbidden,
        ) as mesh_builder,
        patch.object(
            frozen,
            "march_actual_boundary_material_wake_explicit_midpoint",
            side_effect=forbidden,
        ) as marcher,
    ):
        synthetic_test.test_synthetic_31_case_aggregate_is_complete_json_dict()

    if len(captured) != 2:
        raise AssertionError("synthetic definition did not emit good/bad inputs")

    def wrapper_ready(observations):
        return {
            name: replace(
                observation,
                diagnostics={
                    **observation.diagnostics,
                    "input_mutation_abs_max": 0.0,
                    "old_state_mutation_abs_max": 0.0,
                },
            )
            for name, observation in observations.items()
        }

    return {
        "contract": captured[0][1],
        "ordered_case_names": tuple(captured[0][0]),
        "good_observations": wrapper_ready(captured[0][0]),
        "no_go_observations": wrapper_ready(captured[1][0]),
        "forbidden_call_counts": {
            "collector": collector.call_count,
            "mesh": mesh_builder.call_count,
            "march": marcher.call_count,
        },
    }


def _synthetic_source_fingerprints() -> dict[str, str]:
    paths = [
        one_shot._WRAPPER_RELATIVE,
        *[
            f"platform/synthetic_source_{index:02d}.py"
            for index in range(62)
        ],
    ]
    return {
        path: hashlib.sha256(path.encode("utf-8")).hexdigest()
        for path in paths
    }


def _synthetic_serialized_envelope(observations, fixture):
    fingerprints = _synthetic_source_fingerprints()
    result = frozen.aggregate_frozen_histories(
        observations,
        fixture["contract"],
        execution_code_fingerprints=fingerprints,
    )
    one_shot._validate_frozen_result(
        result,
        observations,
        contract=fixture["contract"],
        ordered_case_names=fixture["ordered_case_names"],
        source_fingerprints=fingerprints,
    )
    second_audit = {
        "request_path": "synthetic/request.json",
        "request_sha256": "1" * 64,
        "response_path": "synthetic/response.json",
        "response_sha256": "2" * 64,
        "clearance_canonical_sha256": "3" * 64,
    }
    verified = SimpleNamespace(
        payload={
            "authorization": {"id": "synthetic-read-only-audit"},
            "second_bounded_audit": second_audit,
        },
        raw_sha256="4" * 64,
        canonical_sha256="5" * 64,
        token_sha256="6" * 64,
        definition_sha256="7" * 64,
        clearance={
            "verdict": "ACCEPT_EXACT_ONE_SHOT_31_HISTORY_EXECUTION",
            "actual_independence": "genuine-cross-family",
        },
    )
    registry_manifest = hashlib.sha256(
        one_shot._canonical_json_bytes(
            [
                {
                    "name": case.name,
                    "case_identity_sha256": (
                        frozen._case_identity_payload(case)["sha256"]
                    ),
                }
                for case in frozen.frozen_history_cases()
            ]
        )
    ).hexdigest()
    one_shot._augment_result(
        result,
        observations,
        ordered_case_names=fixture["ordered_case_names"],
        verified=verified,
        registry_manifest_sha256=registry_manifest,
        source_fingerprints_start=fingerprints,
        source_fingerprints_end=fingerprints,
        runtime_start={"synthetic": True},
        runtime_end={"synthetic": True},
        execution_input_sha256_start="9" * 64,
        execution_input_sha256_end="9" * 64,
        marker_receipt_sha256="a" * 64,
    )
    serialized = json.loads(
        one_shot._pretty_json_bytes(result).decode("utf-8")
    )
    one_shot._validate_serialized_result(
        serialized,
        ordered_case_names=fixture["ordered_case_names"],
    )
    return result, serialized


def _refresh_scientific_payload_sha256(payload) -> None:
    scientific = dict(payload)
    scientific.pop("scientific_payload_sha256", None)
    scientific.pop("generated_at_utc", None)
    scientific.pop("code_fingerprints", None)
    scientific.pop("one_shot_provenance", None)
    payload["scientific_payload_sha256"] = hashlib.sha256(
        one_shot._canonical_json_bytes(scientific)
    ).hexdigest()


class OneShotDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.authorization = self.root / "authorization.yaml"
        self.result = self.root / "result.json"
        self.marker = self.root / "result.json.lock"

    def _path_patches(self):
        return (
            patch.object(one_shot, "AUTHORIZATION_PATH", self.authorization),
            patch.object(one_shot, "RESULT_PATH", self.result),
            patch.object(one_shot, "ATTEMPT_MARKER_PATH", self.marker),
        )

    def _assert_early_authorization_failure(self) -> None:
        with ExitStack() as stack:
            for path_patch in self._path_patches():
                stack.enter_context(path_patch)
            loader = stack.enter_context(
                patch.object(
                    frozen,
                    "_load_frozen_contract",
                    side_effect=AssertionError("frozen loader ran too early"),
                )
            )
            collector = stack.enter_context(
                patch.object(
                    frozen,
                    "_collect_frozen_histories",
                    side_effect=AssertionError("formal collector was called"),
                )
            )
            stack.enter_context(
                self.assertRaises(one_shot.OneShotAuthorizationError)
            )
            one_shot.run_authorized_once(
                expected_authorization_sha256="0" * 64,
                second_audit_token=b"not-an-authorized-token",
            )
        loader.assert_not_called()
        collector.assert_not_called()

    def test_public_api_and_receipt_shape_are_frozen(self) -> None:
        self.assertTrue(
            issubclass(one_shot.OneShotAuthorizationError, RuntimeError)
        )
        fields = tuple(one_shot.ExecutionReceipt.__dataclass_fields__)
        self.assertEqual(
            fields,
            (
                "result_path",
                "result_sha256",
                "stage_decision",
                "authorization_sha256",
                "attempt_token_sha256",
            ),
        )

    def test_import_has_no_loader_collector_or_result_side_effect(self) -> None:
        result = one_shot.RESULT_PATH
        marker = one_shot.ATTEMPT_MARKER_PATH
        before = (result.exists(), marker.exists())
        with (
            patch.object(
                frozen,
                "_load_frozen_contract",
                side_effect=AssertionError("loader called during import"),
            ) as loader,
            patch.object(
                frozen,
                "_collect_frozen_histories",
                side_effect=AssertionError("collector called during import"),
            ) as collector,
        ):
            importlib.reload(one_shot)
        self.assertEqual((result.exists(), marker.exists()), before)
        loader.assert_not_called()
        collector.assert_not_called()

    def test_missing_ticket_fails_before_frozen_loader_or_collector(self) -> None:
        self._assert_early_authorization_failure()
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.result.exists())

    def test_existing_result_fails_before_frozen_loader_or_collector(self) -> None:
        self.result.write_bytes(b"immutable-existing-result\n")
        self._assert_early_authorization_failure()
        self.assertEqual(
            self.result.read_bytes(), b"immutable-existing-result\n"
        )
        self.assertFalse(self.marker.exists())

    def test_existing_marker_fails_before_frozen_loader_or_collector(self) -> None:
        self.marker.write_bytes(b"consumed-attempt\n")
        self._assert_early_authorization_failure()
        self.assertEqual(self.marker.read_bytes(), b"consumed-attempt\n")
        self.assertFalse(self.result.exists())

    def test_wrong_external_sha_fails_before_marker_and_loader(self) -> None:
        self.authorization.write_bytes(b"authorization: {}\n")
        wrong = hashlib.sha256(b"different bytes").hexdigest()
        self._assert_early_authorization_failure()
        self.assertNotEqual(
            hashlib.sha256(self.authorization.read_bytes()).hexdigest(),
            wrong,
        )
        self.assertFalse(self.marker.exists())

    def test_wrong_second_audit_token_is_fail_closed(self) -> None:
        token = b"c" * 32
        self.assertEqual(
            one_shot._verify_second_audit_token(
                hashlib.sha256(token).hexdigest(), token
            ),
            hashlib.sha256(token).hexdigest(),
        )
        self.assertNotEqual(
            hashlib.sha256(token).digest(),
            hashlib.sha256(b"w" * 32).digest(),
        )
        verifier = getattr(one_shot, "_verify_second_audit_token")
        with self.assertRaises(one_shot.OneShotAuthorizationError):
            verifier(
                hashlib.sha256(token).hexdigest(),
                b"w" * 32,
            )

    def test_second_audit_clearance_is_parsed_from_exact_json(self) -> None:
        trace_root = self.root / ".aris" / "traces" / "bounded"
        trace_root.mkdir(parents=True)
        request = trace_root / "request.json"
        response = trace_root / "response.md"
        request.write_text('{"audit":"request"}\n', encoding="utf-8")
        clearance = {
            "artifact": "S3ai-v2.2-one-shot-execution-clearance",
            "version": 1,
        }
        response.write_text(
            json.dumps(clearance, sort_keys=True), encoding="utf-8"
        )
        metadata = {
            "request_path": str(request.relative_to(self.root)),
            "request_sha256": hashlib.sha256(
                request.read_bytes()
            ).hexdigest(),
            "response_path": str(response.relative_to(self.root)),
            "response_sha256": hashlib.sha256(
                response.read_bytes()
            ).hexdigest(),
            "clearance_canonical_sha256": hashlib.sha256(
                json.dumps(
                    clearance,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        }
        with patch.object(one_shot, "REPO_ROOT", self.root):
            traces, parsed = one_shot._review_trace_hashes(metadata)
        self.assertEqual(parsed, clearance)
        self.assertEqual(len(traces), 2)

        response.write_text(
            f"```json\n{json.dumps(clearance)}\n```\n",
            encoding="utf-8",
        )
        metadata["response_sha256"] = hashlib.sha256(
            response.read_bytes()
        ).hexdigest()
        with (
            patch.object(one_shot, "REPO_ROOT", self.root),
            self.assertRaises(one_shot.OneShotAuthorizationError),
        ):
            one_shot._review_trace_hashes(metadata)

    def test_marker_is_exclusive_and_survives_simulated_crash(self) -> None:
        claim = getattr(one_shot, "_claim_attempt_once")
        receipt = {
            "authorization_sha256": "a" * 64,
            "attempt_token_sha256": "b" * 64,
        }
        with patch.object(one_shot, "REPO_ROOT", self.root):
            claim(
                marker_path=self.marker,
                result_path=self.result,
                receipt=receipt,
            )
        original = self.marker.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "simulated collector crash"):
            raise RuntimeError("simulated collector crash")
        self.assertTrue(self.marker.exists())
        self.assertEqual(self.marker.read_bytes(), original)
        with (
            patch.object(one_shot, "REPO_ROOT", self.root),
            self.assertRaises(one_shot.OneShotAuthorizationError),
        ):
            claim(
                marker_path=self.marker,
                result_path=self.result,
                receipt=receipt,
            )
        self.assertEqual(self.marker.read_bytes(), original)
        self.assertFalse(self.result.exists())

        self.marker.chmod(0o644)
        self.marker.write_bytes(b"tampered attempt receipt\n")
        with (
            patch.object(one_shot, "REPO_ROOT", self.root),
            self.assertRaises(one_shot.OneShotAuthorizationError),
        ):
            one_shot._verify_permanent_attempt_marker(
                self.marker, hashlib.sha256(original).hexdigest()
            )

    def test_one_directory_lease_rejects_parent_swap(self) -> None:
        output = self.root / "diag"
        output.mkdir()
        marker = output / "result.json.lock"
        result = output / "result.json"
        retired = self.root / "diag.retired"
        replacement = self.root / "diag"
        with patch.object(one_shot, "REPO_ROOT", self.root):
            lease = one_shot._acquire_output_directory_lease(
                marker, result
            )
            try:
                one_shot._claim_attempt_once(
                    marker_path=marker,
                    result_path=result,
                    receipt={"attempt": "synthetic"},
                    output_directory=lease,
                )
                output.rename(retired)
                replacement.mkdir()
                with self.assertRaisesRegex(
                    one_shot.OneShotAuthorizationError,
                    "output directory was replaced",
                ):
                    one_shot._atomic_create_json_no_replace(
                        result,
                        {"stage_decision": "PROTOCOL-NO-GO"},
                        output_directory=lease,
                        marker_path=marker,
                    )
            finally:
                os.close(lease.fd)
        self.assertTrue((retired / marker.name).is_file())
        self.assertFalse((retired / result.name).exists())
        self.assertFalse((replacement / result.name).exists())

    def test_leased_publish_binds_marker_inode_and_result_bytes(self) -> None:
        output = self.root / "diag"
        output.mkdir()
        marker = output / "result.json.lock"
        result = output / "result.json"
        payload = {"stage_decision": "PROTOCOL-NO-GO"}
        with patch.object(one_shot, "REPO_ROOT", self.root):
            lease = one_shot._acquire_output_directory_lease(
                marker, result
            )
            try:
                claim = one_shot._claim_attempt_once(
                    marker_path=marker,
                    result_path=result,
                    receipt={"attempt": "synthetic"},
                    output_directory=lease,
                )
                digest = one_shot._atomic_create_json_no_replace(
                    result,
                    payload,
                    output_directory=lease,
                    marker_path=marker,
                    attempt_claim=claim,
                )
            finally:
                os.close(lease.fd)
        self.assertEqual(
            digest, hashlib.sha256(result.read_bytes()).hexdigest()
        )
        self.assertEqual(json.loads(result.read_text()), payload)

    def test_leased_publish_rejects_same_bytes_new_marker_inode(self) -> None:
        output = self.root / "diag"
        output.mkdir()
        marker = output / "result.json.lock"
        result = output / "result.json"
        with patch.object(one_shot, "REPO_ROOT", self.root):
            lease = one_shot._acquire_output_directory_lease(
                marker, result
            )
            try:
                claim = one_shot._claim_attempt_once(
                    marker_path=marker,
                    result_path=result,
                    receipt={"attempt": "synthetic"},
                    output_directory=lease,
                )
                original = marker.read_bytes()
                marker.rename(output / "result.json.lock.retired")
                marker.write_bytes(original)
                with self.assertRaisesRegex(
                    one_shot.OneShotAuthorizationError,
                    "marker inode identity changed",
                ):
                    one_shot._atomic_create_json_no_replace(
                        result,
                        {"stage_decision": "PROTOCOL-NO-GO"},
                        output_directory=lease,
                        marker_path=marker,
                        attempt_claim=claim,
                    )
            finally:
                os.close(lease.fd)
        self.assertFalse(result.exists())

    def test_protocol_no_go_is_persisted_without_promotion(self) -> None:
        publish = getattr(one_shot, "_atomic_create_json_no_replace")
        payload = {
            "stage_decision": "PROTOCOL-NO-GO",
            "production_activation_allowed": False,
            "fixed_space_only": True,
        }
        with patch.object(one_shot, "REPO_ROOT", self.root):
            digest = publish(self.result, payload)
        decoded = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(decoded["stage_decision"], "PROTOCOL-NO-GO")
        self.assertIs(decoded["production_activation_allowed"], False)
        self.assertIs(decoded["fixed_space_only"], True)
        self.assertEqual(
            digest, hashlib.sha256(self.result.read_bytes()).hexdigest()
        )

    def test_atomic_publication_never_replaces_existing_result(self) -> None:
        publish = getattr(one_shot, "_atomic_create_json_no_replace")
        original = b'{"owner":"competing process"}\n'
        self.result.write_bytes(original)
        with (
            patch.object(one_shot, "REPO_ROOT", self.root),
            self.assertRaises(one_shot.OneShotAuthorizationError),
        ):
            publish(self.result, {"owner": "one-shot wrapper"})
        self.assertEqual(self.result.read_bytes(), original)

    def test_nan_and_infinity_are_rejected_without_result(self) -> None:
        publish = getattr(one_shot, "_atomic_create_json_no_replace")
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with (
                    patch.object(one_shot, "REPO_ROOT", self.root),
                    self.assertRaises(
                        (one_shot.OneShotAuthorizationError, ValueError)
                    ),
                ):
                    publish(self.result, {"nonfinite": value})
                self.assertFalse(self.result.exists())

    def test_extended_value_hashes_survive_json_roundtrip(self) -> None:
        observation = _fake_observation()
        compute = getattr(one_shot, "_extended_observation_hashes")
        hashes = compute(observation)
        self.assertEqual(set(hashes), EXTENDED_VALUE_HASH_KEYS)

        arrays = {
            "mass_active": observation.mass_active,
            "stage_times": observation.stage_arrays.stage_times,
            "measurement_times": observation.stage_arrays.measurement_times,
            "stored_window_residual": observation.stored_window_residual,
            "direct_window_residual": observation.direct_window_residual,
            "stored_step_residuals": observation.stored_step_residuals,
            "direct_step_residuals": observation.direct_step_residuals,
            "canonical_material_current_trace": (
                observation.stage_arrays.canonical_material_current_trace
            ),
            "body_cut_trace": observation.stage_arrays.body_cut_trace,
            "canonical_material_release": (
                observation.stage_arrays.canonical_material_release
            ),
            "representation_inventory": (
                observation.stage_arrays.representation_inventory
            ),
            "stored_weak_pressure": (
                observation.stage_arrays.stored_weak_pressure
            ),
            "direct_weak_pressure": (
                observation.stage_arrays.direct_weak_pressure
            ),
        }
        for name, array in arrays.items():
            self.assertEqual(hashes[name], frozen._array_sha256(array))
        self.assertEqual(
            hashes["stage_roles_canonical_JSON"],
            _canonical_role_hash(observation.stage_arrays.stage_roles),
        )

        serializable = {
            "arrays": {
                name: np.asarray(array).tolist()
                for name, array in arrays.items()
            },
            "stage_roles": list(observation.stage_arrays.stage_roles),
            "hashes": hashes,
        }
        roundtrip = json.loads(
            json.dumps(
                serializable,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for name, value in roundtrip["arrays"].items():
            self.assertEqual(
                roundtrip["hashes"][name],
                frozen._array_sha256(np.asarray(value, dtype=np.float64)),
            )
        self.assertEqual(
            roundtrip["hashes"]["stage_roles_canonical_JSON"],
            _canonical_role_hash(tuple(roundtrip["stage_roles"])),
        )

    def test_synthetic_31_case_good_and_no_go_envelopes_roundtrip(
        self,
    ) -> None:
        fixture = _synthetic_31_case_fixture()
        self.assertEqual(
            fixture["forbidden_call_counts"],
            {"collector": 0, "mesh": 0, "march": 0},
        )
        expected_decisions = {
            "good_observations": (
                "FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS"
            ),
            "no_go_observations": "PROTOCOL-NO-GO",
        }
        no_go_serialized = None
        for observation_key, expected_decision in expected_decisions.items():
            with self.subTest(outcome=observation_key):
                _, serialized = _synthetic_serialized_envelope(
                    fixture[observation_key],
                    fixture,
                )
                matched = serialized["v22_span_parity"][
                    "matched_stage_q_families"
                ]
                self.assertEqual(matched["entry_count"], 393)
                self.assertEqual(len(matched["entries"]), 393)
                self.assertEqual(len(serialized["cases"]), 31)
                self.assertEqual(
                    serialized["ordered_case_names"],
                    list(fixture["ordered_case_names"]),
                )
                self.assertEqual(
                    serialized["stage_decision"],
                    expected_decision,
                )
                self.assertIs(
                    serialized["production_activation_allowed"],
                    False,
                )
                self.assertIs(serialized["fixed_space_only"], True)

                all_payloads = {}
                for name in fixture["ordered_case_names"]:
                    case = serialized["cases"][name]
                    raw_payload = one_shot._case_payload_from_serialized(
                        case
                    )
                    hashes, payload_sha = (
                        one_shot._hash_serialized_case_payload(raw_payload)
                    )
                    self.assertEqual(
                        set(case["extended_value_hashes"]),
                        EXTENDED_VALUE_HASH_KEYS,
                    )
                    self.assertEqual(case["extended_value_hashes"], hashes)
                    self.assertEqual(
                        case["observation_payload_sha256"],
                        payload_sha,
                    )
                    mass = np.asarray(
                        case["mass_active"],
                        dtype=np.float64,
                    )
                    self.assertEqual(mass.shape, (7, 7))
                    self.assertEqual(
                        hashes["mass_active"],
                        frozen._array_sha256(mass),
                    )
                    all_payloads[name] = raw_payload
                self.assertEqual(
                    serialized["observation_payload_sha256"],
                    hashlib.sha256(
                        one_shot._canonical_json_bytes(all_payloads)
                    ).hexdigest(),
                )
                scientific = dict(serialized)
                declared_scientific = scientific.pop(
                    "scientific_payload_sha256"
                )
                scientific.pop("generated_at_utc")
                scientific.pop("code_fingerprints")
                scientific.pop("one_shot_provenance")
                self.assertEqual(
                    declared_scientific,
                    hashlib.sha256(
                        one_shot._canonical_json_bytes(scientific)
                    ).hexdigest(),
                )
                if expected_decision == "PROTOCOL-NO-GO":
                    no_go_serialized = serialized

        self.assertIsNotNone(no_go_serialized)
        with patch.object(one_shot, "REPO_ROOT", self.root):
            result_sha = one_shot._atomic_create_json_no_replace(
                self.result,
                no_go_serialized,
            )
        persisted = json.loads(self.result.read_text(encoding="utf-8"))
        one_shot._validate_serialized_result(
            persisted,
            ordered_case_names=fixture["ordered_case_names"],
        )
        self.assertEqual(persisted["stage_decision"], "PROTOCOL-NO-GO")
        self.assertIs(persisted["production_activation_allowed"], False)
        self.assertIs(persisted["fixed_space_only"], True)
        self.assertEqual(len(persisted["cases"]), 31)
        self.assertEqual(
            result_sha,
            hashlib.sha256(self.result.read_bytes()).hexdigest(),
        )

    def test_frozen_validator_rejects_declared_393_with_392_entries(
        self,
    ) -> None:
        fixture = _synthetic_31_case_fixture()
        self.assertEqual(
            fixture["forbidden_call_counts"],
            {"collector": 0, "mesh": 0, "march": 0},
        )
        fingerprints = _synthetic_source_fingerprints()
        result = frozen.aggregate_frozen_histories(
            fixture["good_observations"],
            fixture["contract"],
            execution_code_fingerprints=fingerprints,
        )
        matched = result["v22_span_parity"]["matched_stage_q_families"]
        self.assertEqual(matched["entry_count"], 393)
        self.assertEqual(len(matched["entries"]), 393)
        matched["entries"].pop()
        self.assertEqual(matched["entry_count"], 393)
        self.assertEqual(len(matched["entries"]), 392)
        with self.assertRaisesRegex(
            one_shot.OneShotAuthorizationError,
            "393 entries",
        ):
            one_shot._validate_frozen_result(
                result,
                fixture["good_observations"],
                contract=fixture["contract"],
                ordered_case_names=fixture["ordered_case_names"],
                source_fingerprints=fingerprints,
            )

    def test_matched_stage_identity_and_failed_ledger_are_exact(self) -> None:
        fixture = _synthetic_31_case_fixture()
        fingerprints = _synthetic_source_fingerprints()
        result = frozen.aggregate_frozen_histories(
            fixture["good_observations"],
            fixture["contract"],
            execution_code_fingerprints=fingerprints,
        )
        matched = result["v22_span_parity"]["matched_stage_q_families"]
        matched["entries"][-1] = copy.deepcopy(matched["entries"][0])
        with self.assertRaisesRegex(
            one_shot.OneShotAuthorizationError,
            "duplicated",
        ):
            one_shot._validate_frozen_result(
                result,
                fixture["good_observations"],
                contract=fixture["contract"],
                ordered_case_names=fixture["ordered_case_names"],
                source_fingerprints=fingerprints,
            )

        _, serialized = _synthetic_serialized_envelope(
            fixture["good_observations"], fixture
        )
        for mutation in ("missing_group", "flipped_passed"):
            with self.subTest(mutation=mutation):
                tampered = copy.deepcopy(serialized)
                ledger = tampered["v22_span_parity"][
                    "matched_stage_q_families"
                ]
                if mutation == "missing_group":
                    ledger["entries"][0].pop("group")
                else:
                    ledger["passed"] = not ledger["passed"]
                _refresh_scientific_payload_sha256(tampered)
                with self.assertRaises(one_shot.OneShotAuthorizationError):
                    one_shot._validate_serialized_result(
                        tampered,
                        ordered_case_names=fixture[
                            "ordered_case_names"
                        ],
                    )

    def test_case_configuration_identity_cannot_drift(self) -> None:
        fixture = _synthetic_31_case_fixture()
        fingerprints = _synthetic_source_fingerprints()
        result = frozen.aggregate_frozen_histories(
            fixture["good_observations"],
            fixture["contract"],
            execution_code_fingerprints=fingerprints,
        )
        result["cases"]["A_plus"]["epsilon_signed"] *= 2.0
        with self.assertRaisesRegex(
            one_shot.OneShotAuthorizationError,
            "configuration identity",
        ):
            one_shot._validate_frozen_result(
                result,
                fixture["good_observations"],
                contract=fixture["contract"],
                ordered_case_names=fixture["ordered_case_names"],
                source_fingerprints=fingerprints,
            )

        _, serialized = _synthetic_serialized_envelope(
            fixture["good_observations"], fixture
        )
        serialized["cases"]["A_plus"]["epsilon_signed"] *= 2.0
        _refresh_scientific_payload_sha256(serialized)
        with self.assertRaisesRegex(
            one_shot.OneShotAuthorizationError,
            "configuration drifted",
        ):
            one_shot._validate_serialized_result(
                serialized,
                ordered_case_names=fixture["ordered_case_names"],
            )

    def test_serialized_provenance_cross_bindings_cannot_drift(self) -> None:
        fixture = _synthetic_31_case_fixture()
        _, serialized = _synthetic_serialized_envelope(
            fixture["good_observations"], fixture
        )
        mutations = (
            lambda payload: payload["one_shot_provenance"][
                "source_fingerprints_end"
            ].pop(next(iter(payload["code_fingerprints"]))),
            lambda payload: payload["one_shot_provenance"].update(
                {"execution_input_sha256_end": "f" * 64}
            ),
            lambda payload: payload["one_shot_provenance"][
                "runtime_identity_end"
            ].update({"synthetic": False}),
            lambda payload: payload["one_shot_provenance"].update(
                {"ordered_registry_manifest_sha256": "e" * 64}
            ),
        )
        for mutate in mutations:
            tampered = copy.deepcopy(serialized)
            mutate(tampered)
            with self.assertRaisesRegex(
                one_shot.OneShotAuthorizationError,
                "provenance",
            ):
                one_shot._validate_serialized_result(
                    tampered,
                    ordered_case_names=fixture["ordered_case_names"],
                )

    def test_serialized_validator_rejects_rehashed_missing_mass_active(
        self,
    ) -> None:
        fixture = _synthetic_31_case_fixture()
        self.assertEqual(
            fixture["forbidden_call_counts"],
            {"collector": 0, "mesh": 0, "march": 0},
        )
        _, serialized = _synthetic_serialized_envelope(
            fixture["good_observations"],
            fixture,
        )
        tampered = copy.deepcopy(serialized)
        first_name = fixture["ordered_case_names"][0]
        first_case = tampered["cases"][first_name]
        first_case["mass_active"] = None
        raw_payload = one_shot._case_payload_from_serialized(first_case)
        hashes, payload_sha = one_shot._hash_serialized_case_payload(
            raw_payload
        )
        first_case["extended_value_hashes"] = hashes
        first_case["value_hashes"]["mass_active"] = hashes["mass_active"]
        first_case["observation_payload_sha256"] = payload_sha

        all_payloads = {
            name: one_shot._case_payload_from_serialized(
                tampered["cases"][name]
            )
            for name in fixture["ordered_case_names"]
        }
        tampered["observation_payload_sha256"] = hashlib.sha256(
            one_shot._canonical_json_bytes(all_payloads)
        ).hexdigest()
        _refresh_scientific_payload_sha256(tampered)

        with self.assertRaisesRegex(
            one_shot.OneShotAuthorizationError,
            "invalid shape|nonnumeric array",
        ):
            one_shot._validate_serialized_result(
                tampered,
                ordered_case_names=fixture["ordered_case_names"],
            )

    def test_local_source_closure_contains_62_frozen_plus_wrapper(self) -> None:
        definition, _ = one_shot._load_wrapper_definition()
        expected = definition["wrapper_contract"]["source_closure"][
            "exact_preimplementation_closure"
        ]
        fingerprints = {
            path: hashlib.sha256(
                (one_shot.REPO_ROOT / path).read_bytes()
            ).hexdigest()
            for path in (
                *expected,
                "platform/actual_wake_reachable_pressure_obstruction_v22_one_shot.py",
            )
        }
        self.assertEqual(len(fingerprints), 63)
        self.assertIn(
            "platform/actual_wake_reachable_pressure_obstruction_v22_one_shot.py",
            fingerprints,
        )
        self.assertTrue(set(frozen._code_fingerprints()).issubset(fingerprints))

    def test_clean_process_observes_exact_audited_63_source_set(self) -> None:
        command = (
            "import json,sys;"
            "sys.path.insert(0,'platform');"
            "import actual_wake_reachable_pressure_obstruction_v22_one_shot"
            " as w;"
            "d,_=w._load_wrapper_definition();"
            "f=w._source_fingerprints(d);"
            "print(json.dumps({"
            "'clean':w._IMPORT_WAS_CLEAN,"
            "'pre':list(w._PRE_GUARD_LOCAL_SOURCES),"
            "'post_count':len(w._POST_GUARD_LOCAL_SOURCES),"
            "'fingerprint_count':len(f)"
            "},sort_keys=True))"
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", command],
            cwd=PLATFORM.parent,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        observed = json.loads(completed.stdout)
        self.assertIs(observed["clean"], True)
        self.assertEqual(
            observed["pre"],
            [
                "platform/actual_wake_reachable_pressure_obstruction_v22_one_shot.py"
            ],
        )
        self.assertEqual(observed["post_count"], 63)
        self.assertEqual(observed["fingerprint_count"], 63)

    def test_formal_source_mode_forces_absent_alternate_pyc_cache(
        self,
    ) -> None:
        cache = PLATFORM.parent / ".one_shot_no_bytecode_cache"
        wrapper = (
            PLATFORM
            / "actual_wake_reachable_pressure_obstruction_v22_one_shot.py"
        )
        self.assertFalse(cache.exists())
        command = f"""
import hashlib
import json
import os
import sys
import types

path = {str(wrapper)!r}
descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
try:
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
finally:
    os.close(descriptor)
raw = b"".join(chunks)
module = types.ModuleType("_one_shot_bootstrap_test")
module.__file__ = path
module.__package__ = None
module._BOOTSTRAP_WRAPPER_SHA256 = hashlib.sha256(raw).hexdigest()
sys.modules[module.__name__] = module
exec(compile(raw, path, "exec"), module.__dict__, module.__dict__)
module._require_source_execution_mode()
definition, _ = module._load_wrapper_definition()
module._verify_loaded_local_modules_use_source(definition)
print(json.dumps({{
    "guard_import_mode": module._SOURCE_IMPORT_MODE_AT_GUARD_IMPORT,
    "dont_write": sys.dont_write_bytecode,
    "prefix": sys.pycache_prefix,
    "bootstrap_sha": module._BOOTSTRAP_WRAPPER_SHA256,
    "disk_sha": module._WRAPPER_BYTES_BEFORE_GUARD,
}}, sort_keys=True))
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-X",
                f"pycache_prefix={cache}",
                "-c",
                command,
            ],
            cwd=PLATFORM.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        observed = json.loads(completed.stdout)
        self.assertIs(observed["guard_import_mode"], True)
        self.assertIs(observed["dont_write"], True)
        self.assertEqual(observed["prefix"], str(cache))
        self.assertEqual(observed["bootstrap_sha"], observed["disk_sha"])
        self.assertFalse(cache.exists())

    def test_formal_source_mode_rejects_direct_import_without_bootstrap(
        self,
    ) -> None:
        cache = PLATFORM.parent / ".one_shot_no_bytecode_cache"
        self.assertFalse(cache.exists())
        command = (
            "import json,sys;"
            "sys.path.insert(0,'platform');"
            "import actual_wake_reachable_pressure_obstruction_v22_one_shot"
            " as w;"
            "ok=False;"
            "\ntry:\n w._require_source_execution_mode()\n"
            "except w.OneShotAuthorizationError:\n ok=True\n"
            "print(json.dumps({'rejected':ok,"
            "'bootstrap':w._BOOTSTRAP_WRAPPER_SHA256}))"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-X",
                f"pycache_prefix={cache}",
                "-c",
                command,
            ],
            cwd=PLATFORM.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        observed = json.loads(completed.stdout)
        self.assertIs(observed["rejected"], True)
        self.assertIsNone(observed["bootstrap"])
        self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
