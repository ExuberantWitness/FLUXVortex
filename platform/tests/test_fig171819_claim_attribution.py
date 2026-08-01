from __future__ import annotations

import copy
import inspect
import json
import math
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np


TESTS = Path(__file__).resolve().parent
PLATFORM = TESTS.parent
for path in (PLATFORM, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fig171819_benchmark as benchmark  # noqa: E402
import fig171819_claim_attribution as attribution  # noqa: E402
import fig171819_confirmed_compare as compare  # noqa: E402
import fig171819_residual_fingerprint as residual  # noqa: E402
import test_fig171819_confirmed_compare as compare_test  # noqa: E402


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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": compare._display_path(path),
        "sha256": attribution._sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _execution_envelope(
    stage: str,
    *,
    payload: dict[str, object] | None = None,
    identity_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    default_status = {
        "select-disease": "ACTIVE_DISEASE_FROZEN",
        "prepare": "PREPARED",
        "evaluate": "ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS",
    }[stage]
    scientific_payload = (
        copy.deepcopy(payload) if payload is not None else {"status": default_status}
    )
    bindings = {
        "evidence_commit_sha": "a" * 40,
        "attestation_commit_sha": "b" * 40,
        "attestation_payload_sha256": "c" * 64,
        "authorization_blob_sha256": "d" * 64,
        "launcher_blob_sha256": "e" * 64,
    }
    closure_sha256 = "f" * 64
    h0: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fig171819_evidence_receipt_envelope",
        "stage": "outer_preflight",
        "upstream_receipts": {},
        "body": {
            "status": "PASS",
            "dry_run": False,
            "bindings": copy.deepcopy(bindings),
            "runtime_source_closure_sha256": closure_sha256,
            "git_no_replace_objects": True,
            "git_config_nosystem": True,
            "git_config_global": "/dev/null",
            "git_hooks_disabled": True,
            "checkout_used": False,
            "raw_blob_materialization": True,
            "command_envelope": {"command": stage},
        },
    }
    h0_sha256 = attribution._receipt_object_sha256(h0)
    h1: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fig171819_evidence_receipt_envelope",
        "stage": "inner_launcher",
        "upstream_receipts": {"H0": h0_sha256},
        "body": {
            "status": "PASS",
            "python_executable_realpath": (
                "/home/exuber/anaconda3/envs/fluxvortex/bin/python"
            ),
            "python_isolated_flag": True,
            "python_no_site_flag": True,
            "python_no_bytecode_flag": True,
            "python_startup_contamination_check": "PASS",
            "runtime_environment_manifest_verified": True,
            "runtime_source_closure_verified": True,
            "runtime_source_closure_sha256": closure_sha256,
        },
    }
    h1_sha256 = attribution._receipt_object_sha256(h1)
    scientific_status = scientific_payload["status"]
    h2: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fig171819_evidence_receipt_envelope",
        "stage": "outer_completion",
        "upstream_receipts": {"H0": h0_sha256, "H1": h1_sha256},
        "body": {
            "status": "PASS",
            "transport_status": "PASS",
            "scientific_status": scientific_status,
            "inner_exit_code": (2 if scientific_status == "INVALID_EVIDENCE" else 0),
            "post_run_source_closure": "PASS",
            "post_run_output_inventory": "PASS",
            "post_run_errors": [],
            "intermediate_payload_exact_sha256": (
                attribution._canonical_hash(
                    {"exact_payload_fixture": scientific_payload}
                )
            ),
            "intermediate_payload_canonical_sha256": (
                attribution._canonical_hash(scientific_payload)
            ),
            "intermediate_payload_cleanup_status": "PASS",
            "cleanup_status": "PASS",
            "cleanup_scope": "INTERMEDIATE_PAYLOAD_ONLY",
            "cleanup_target_relative_paths": [
                attribution.ATTRIBUTION_INTERMEDIATE_OUTPUTS[stage]
            ],
            "execution_layout_cleanup_status": "NOT_RUN_RETAINED_FOR_AUDIT",
        },
    }
    h2_sha256 = attribution._receipt_object_sha256(h2)
    envelope: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fig171819_verified_execution_receipt_envelope",
        "stage": stage,
        **bindings,
        "git_no_replace_objects": True,
        "git_config_nosystem": True,
        "git_config_global": "/dev/null",
        "git_hooks_disabled": True,
        "checkout_used": False,
        "raw_blob_materialization": True,
        "runtime_source_closure_verified": True,
        "runtime_source_closure_sha256": closure_sha256,
        "python_executable_realpath": (
            "/home/exuber/anaconda3/envs/fluxvortex/bin/python"
        ),
        "python_isolated_flag": True,
        "python_no_site_flag": True,
        "python_no_bytecode_flag": True,
        "python_startup_contamination_check": "PASS",
        "outer_preflight_receipt_sha256": h0_sha256,
        "inner_launcher_receipt_sha256": h1_sha256,
        "outer_completion_receipt_sha256": h2_sha256,
        "outer_preflight_receipt": h0,
        "inner_launcher_receipt": h1,
        "outer_completion_receipt": h2,
        "post_run_source_closure_verified": True,
        "transport_status": "PASS",
        "scientific_status": scientific_status,
        "cleanup_status": "PASS",
        "cleanup_scope": "INTERMEDIATE_PAYLOAD_ONLY",
        "cleanup_target_relative_paths": [
            attribution.ATTRIBUTION_INTERMEDIATE_OUTPUTS[stage]
        ],
        "execution_layout_cleanup_status": "NOT_RUN_RETAINED_FOR_AUDIT",
    }
    if identity_overrides:
        envelope.update(identity_overrides)
    return envelope


def _scientific_envelope(
    payload: dict,
    stage: str,
    *,
    identity_overrides: dict[str, object] | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "fig171819_scientific_payload_envelope",
        "stage": stage,
        "payload_sha256": attribution._canonical_hash(payload),
        "execution_envelope": _execution_envelope(
            stage,
            payload=payload,
            identity_overrides=identity_overrides,
        ),
        "payload": copy.deepcopy(payload),
        "production_execution_authorized": True,
    }


def _body_from_wind(lift: float, thrust: float, aoa_deg: float) -> list[float]:
    angle = math.radians(aoa_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        -thrust * cosine - lift * sine,
        0.0,
        lift * cosine - thrust * sine,
    ]


def _fingerprint_and_scorecard(
    results: dict[str, dict[str, float]],
) -> tuple[dict, dict]:
    measurements = benchmark.load_measurements(benchmark.DEFAULT_DATA_MD)
    curve_to_family, families, aliases = residual._physical_family_contract()
    samples: list[dict] = []
    official: list[dict] = []
    rows: list[dict] = []
    for curve in benchmark.CURVES_BY_EVIDENCE_SCOPE[benchmark.EVIDENCE_CONFIRMED]:
        measurement = measurements[curve.key]
        nominal_x = (
            benchmark.RAW_FS if curve.abscissa == "frequency_Hz" else benchmark.TWS
        )
        measured_values: list[float] = []
        model_values: list[float] = []
        for index, (nominal, raw_x, measured) in enumerate(
            zip(
                nominal_x,
                measurement.x,
                measurement.values_N,
            )
        ):
            evaluation_x = float(np.clip(raw_x, curve.x[0], curve.x[-1]))
            bracket = residual._condition_bracket(
                np.asarray(curve.x, dtype=float),
                curve.conditions,
                evaluation_x,
            )
            model = float(bracket["left_weight"]) * float(
                results[str(bracket["left_condition_key"])][curve.channel]
            ) + float(bracket["right_weight"]) * float(
                results[str(bracket["right_condition_key"])][curve.channel]
            )
            error = model - float(measured)
            samples.append(
                {
                    "curve": curve.key,
                    "physical_family_id": curve_to_family[curve.key],
                    "figure": curve.figure,
                    "panel": curve.panel,
                    "channel": curve.channel,
                    "abscissa": curve.abscissa,
                    "measurement_index": index,
                    "canonical_nominal_x": float(nominal),
                    "raw_x": float(raw_x),
                    "evaluation_x": evaluation_x,
                    "measurement_N": float(measured),
                    "model_N": model,
                    "error_N": error,
                    "absolute_error_N": abs(error),
                    "squared_error_N2": error * error,
                    **bracket,
                }
            )
            measured_values.append(float(measured))
            model_values.append(model)
        official.append(
            {
                "curve": curve.key,
                "physical_family_id": curve_to_family[curve.key],
                "figure": curve.figure,
                "panel": curve.panel,
                "channel": curve.channel,
                "abscissa": curve.abscissa,
            }
        )
        rows.append(
            {
                "curve": curve.key,
                "figure": curve.figure,
                "panel": curve.panel,
                "channel": curve.channel,
                "abscissa": curve.abscissa,
                "evidence_scope": benchmark.EVIDENCE_CONFIRMED,
                "complete": True,
                "measurement_N": measured_values,
                "model_at_measurement_x_N": model_values,
            }
        )
    if len(samples) != 434 or len(official) != 42:
        raise AssertionError("confirmed synthetic fixture count drift")
    pf_hash = attribution._canonical_hash({"families": families, "aliases": aliases})
    fingerprint = {
        "schema_version": 1,
        "status": "DESCRIPTIVE_FINGERPRINT_COMPLETE",
        "validity_gates": {
            "synthetic_complete": True,
            "fresh_triplet_complete_and_bound": True,
            "force_ledgers_close_within_1e-9_N": True,
            "pf_strata_count_contract": True,
            "fig19_cd_zero_residual_leakage": True,
        },
        "contract": {
            "confirmed_curves": 42,
            "raw_measurement_samples": 434,
            "solver_conditions": 151,
            "physical_curve_families": 34,
            "duplicate_alias_groups": 8,
            "excluded_conditional_curves": sorted(
                attribution.EXPECTED_CONDITIONAL_CURVES
            ),
        },
        "samples": samples,
        "official_curves": official,
        "physical_curve_families": copy.deepcopy(families),
        "duplicate_aliases": [
            {
                "physical_family_id": item["physical_family_id"],
                "official_curve_keys": item["official_curve_keys"],
                "comparisons": [],
            }
            for item in aliases
        ],
        "physical_family_contract_sha256": pf_hash,
    }
    coverage = benchmark.coverage(
        results,
        evidence_scope=benchmark.EVIDENCE_CONFIRMED,
    )
    scorecard = {
        "schema_version": 3,
        "artifact_type": "fig171819_confirmed_scope_scorecard",
        "primary_evidence_scope": benchmark.EVIDENCE_CONFIRMED,
        "coverage": coverage,
        "rows": rows,
        "evidence_scopes": {
            benchmark.EVIDENCE_CONFIRMED: {
                "coverage": coverage,
                "rows": rows,
            }
        },
    }
    return fingerprint, scorecard


def _pf_strata_contract() -> dict:
    curve_to_family, _, _ = residual._physical_family_contract()
    output: dict[str, dict[str, dict]] = {}
    for figure in ("ALL", "17", "18", "19"):
        output[figure] = {}
        for channel in ("ALL", "T", "L"):
            selected = [
                curve
                for curve in benchmark.CURVES_BY_EVIDENCE_SCOPE[
                    benchmark.EVIDENCE_CONFIRMED
                ]
                if (figure == "ALL" or curve.figure == figure)
                and (channel == "ALL" or curve.channel == channel)
            ]
            grouped: dict[str, list[str]] = {}
            for curve in selected:
                grouped.setdefault(curve_to_family[curve.key], []).append(curve.key)
            families = [
                {
                    "physical_family_id": family_id,
                    "official_curve_keys": sorted(keys),
                    "n_official_curves": len(keys),
                }
                for family_id, keys in sorted(grouped.items())
            ]
            output[figure][channel] = {
                "n_physical_families": len(families),
                "n_official_curves": len(selected),
                "families": families,
            }
    return output


class SyntheticCampaign:
    def __init__(self, root: Path) -> None:
        self.root = root
        fresh_root = root / "fresh"
        fresh_root.mkdir(parents=True, exist_ok=True)
        base = compare_test._build_complete_bundle(fresh_root)
        self.paths = {
            **base,
            "scorecard": root / "scorecard.json",
            "fingerprint": root / "fingerprint.json",
            "disease": root / "disease.json",
            "attribution_prereg": root / "attribution_prereg.json",
            "baseline_receipt": root / "baseline_receipt.json",
        }
        self.results = _load_json(self.paths["result"])
        self.manifest = _load_json(self.paths["manifest"])
        self.contributions = _load_json(self.paths["contributions"])
        self._install_parent_force_signal()
        self._write_triplet_and_authorization()
        self.fingerprint, self.scorecard = _fingerprint_and_scorecard(self.results)
        self._publish_receipt()
        with self.trust():
            self.disease = attribution.select_active_disease(
                self.fingerprint,
                self.baseline_receipt,
                fingerprint_path=self.paths["fingerprint"],
                baseline_receipt_path=self.paths["baseline_receipt"],
            )
        if self.disease.get("status") != "ACTIVE_DISEASE_FROZEN":
            raise AssertionError(f"synthetic disease did not freeze: {self.disease}")
        self.disease_artifact = _scientific_envelope(
            self.disease,
            "select-disease",
        )
        _write_json(self.paths["disease"], self.disease_artifact)
        with self.trust():
            self.prereg_payload = attribution.prepare_contract(
                self.fingerprint,
                self.disease_artifact,
                fingerprint_path=self.paths["fingerprint"],
                disease_spec_path=self.paths["disease"],
            )
        if self.prereg_payload.get("status") != "PREPARED":
            raise AssertionError(
                f"synthetic preregistration did not prepare: {self.prereg_payload}"
            )
        self.prereg = _scientific_envelope(
            self.prereg_payload,
            "prepare",
        )
        _write_json(self.paths["attribution_prereg"], self.prereg)

    @contextmanager
    def trust(self):
        authorization_sha256 = attribution._sha256_file(self.paths["authorization"])
        with (
            mock.patch.object(
                attribution,
                "AUTHORIZED_POSTPROCESS_AUTHORIZATION",
                self.paths["authorization"],
            ),
            mock.patch.object(
                attribution,
                "AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256",
                authorization_sha256,
            ),
            mock.patch.object(
                compare,
                "AUTHORIZED_POSTPROCESS_AUTHORIZATION",
                self.paths["authorization"],
            ),
            mock.patch.object(
                compare,
                "AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256",
                authorization_sha256,
            ),
        ):
            yield

    def _install_parent_force_signal(self) -> None:
        for key, case in self.contributions["cases"].items():
            condition = compare.CONDITION_BY_KEY[key]
            twist = float(condition[2])
            shape = math.sin(math.pi * twist / 45.0)
            baseline_thrust = -5.0 * shape
            restored_thrust = 1.0 * shape
            n1_body = _body_from_wind(
                0.0,
                restored_thrust,
                condition[3],
            )
            n3_body = _body_from_wind(
                0.0,
                baseline_thrust - restored_thrust,
                condition[3],
            )
            raw = case["claim_contributions"]
            for items in raw.values():
                for item in items:
                    item["body_force"] = [0.0, 0.0, 0.0]
            raw["N1"][0]["body_force"] = n1_body
            raw["N3"][0]["body_force"] = n3_body
            summary, ledger = compare.witness._contribution_summary(
                raw,
                aoa_deg=condition[3],
            )
            wind = compare.witness._wind_force(ledger, condition[3])
            case["contribution_summary"] = summary
            case["recomputed_ledger"] = {
                "total_body_force_N": ledger.tolist(),
                "total_wind_force": wind,
                "max_body_error_N": 0.0,
                "max_wind_error_N": 0.0,
            }
            self.results[key] = {"L": wind["L_N"], "T": wind["T_N"]}
            case["old_baseline"] = {
                "L_N": wind["L_N"],
                "T_N": wind["T_N"],
            }
            case["signed_old_baseline_delta_N"] = {
                "L_N": 0.0,
                "T_N": 0.0,
            }

    def _write_triplet_and_authorization(self) -> None:
        _write_json(self.paths["result"], self.results)
        _write_json(self.paths["contributions"], self.contributions)
        self.manifest["result_sha256"] = attribution._sha256_file(self.paths["result"])
        self.manifest["contributions_sha256"] = attribution._sha256_file(
            self.paths["contributions"]
        )
        anchor_key = self.manifest["formal_anchor"]["condition_key"]
        anchor_value = copy.deepcopy(self.results[anchor_key])
        anchor_delta = copy.deepcopy(
            self.contributions["cases"][anchor_key]["signed_old_baseline_delta_N"]
        )
        self.manifest["cold_preconditioner"]["value"] = copy.deepcopy(anchor_value)
        self.manifest["formal_anchor"]["value"] = copy.deepcopy(anchor_value)
        self.manifest["formal_anchor"]["old_baseline_delta_N"] = copy.deepcopy(
            anchor_delta
        )
        for session in self.manifest["runtime_sessions"]:
            session["cold_preconditioner"]["value"] = copy.deepcopy(anchor_value)
            session["warm_anchor"]["value"] = copy.deepcopy(anchor_value)
            session["warm_anchor"]["old_baseline_delta_N"] = copy.deepcopy(anchor_delta)
        _write_json(self.paths["manifest"], self.manifest)
        compare_test._write_authorization(
            self.paths["authorization"],
            manifest=self.manifest,
            result_path=self.paths["result"],
            manifest_path=self.paths["manifest"],
            contributions_path=self.paths["contributions"],
        )

    def _publish_receipt(self) -> None:
        bundle = compare.validate_fresh151_bundle(
            result_path=self.paths["result"],
            manifest_path=self.paths["manifest"],
            contributions_path=self.paths["contributions"],
        )
        pf_hash = attribution.AUTHORIZED_PHYSICAL_FAMILY_CONTRACT_SHA256
        payload = compare._bundle_id_payload(
            bundle=bundle,
            data_path=benchmark.DEFAULT_DATA_MD,
            scoring_prereg_path=attribution.AUTHORIZED_POSTPROCESS_PREREG,
            authorization_path=self.paths["authorization"],
            pf_contract_sha256=pf_hash,
        )
        bundle_id = attribution._canonical_hash(payload)
        self.fingerprint["baseline_bundle_id"] = bundle_id
        self.fingerprint["generator"] = _identity(attribution.CONFIRMED_COMPARE_SOURCE)
        _write_json(self.paths["fingerprint"], self.fingerprint)
        _write_json(self.paths["scorecard"], self.scorecard)

        output_paths = {
            "scorecard": self.paths["scorecard"],
            "artifact": self.root / "artifact.json",
            "fingerprint": self.paths["fingerprint"],
            "fig17": self.root / "fig17.png",
            "fig17_sidecar": self.root / "fig17_sidecar.json",
            "fig18": self.root / "fig18.png",
            "fig18_sidecar": self.root / "fig18_sidecar.json",
            "fig19": self.root / "fig19.png",
            "fig19_sidecar": self.root / "fig19_sidecar.json",
        }
        for name, path in output_paths.items():
            if path.exists():
                continue
            if name in {"fig17", "fig18", "fig19"}:
                path.write_bytes(b"synthetic image evidence\n")
            else:
                _write_json(path, {"synthetic": True})
        self.baseline_receipt = {
            "schema_version": 1,
            "artifact_type": "v41_fresh_confirmed42_baseline_bundle_receipt",
            "status": "READY_FOR_CONFIRMED_BASELINE_DIAGNOSIS",
            "baseline_bundle_id": bundle_id,
            "bundle_id_payload": payload,
            "run_id": self.manifest["run_id"],
            "evidence_scope": benchmark.EVIDENCE_CONFIRMED,
            "contract": {
                "official_curves": 42,
                "raw_measurement_samples": 434,
                "solver_conditions": 151,
                "physical_families": 34,
                "duplicate_alias_groups": 8,
                "figure_curve_counts": {"17": 10, "18": 24, "19": 8},
                "conditional_fig19_cd_curves_excluded": sorted(
                    attribution.EXPECTED_CONDITIONAL_CURVES
                ),
            },
            "input_artifacts": {
                "result": _identity(self.paths["result"]),
                "manifest": _identity(self.paths["manifest"]),
                "claim_contributions": _identity(self.paths["contributions"]),
                "measurement_data": _identity(benchmark.DEFAULT_DATA_MD),
                "scoring_preregistration": _identity(
                    attribution.AUTHORIZED_POSTPROCESS_PREREG
                ),
                "postprocess_authorization": _identity(self.paths["authorization"]),
            },
            "validation": {
                **copy.deepcopy(bundle.validation),
                "same_timestamp_complete_resume_revalidation": True,
                "complete_resume_triplet_unchanged": True,
                "postprocess_authorization_validated": True,
            },
            "physical_family_contract": {
                "sha256": pf_hash,
                "family_count": 34,
                "alias_group_count": 8,
            },
            "primary_metric": {
                "name": "physical_family_equal_mean_absolute_error",
                "unit": "N",
                "value": 0.0,
                "physical_family_count": 34,
                "alias_policy": "synthetic",
            },
            "pf_equal_by_figure_channel": _pf_strata_contract(),
            "outputs": {name: _identity(path) for name, path in output_paths.items()},
            "global_promotion_eligible": False,
            "global_promotion_blockers": [
                "Fig19(c,d) authoritative fixed-frequency identity unresolved",
                (
                    "authoritative global condition union unresolved; 184 only if "
                    "shared frequency, 217 possible if channels differ"
                ),
            ],
            "allowed_use": (
                "V4.1 confirmed42 residual diagnosis, unique-claim attribution, "
                "and preregistration of one mechanism candidate"
            ),
            "forbidden_use": (
                "final 50-curve promotion or any claim based on Fig19(c,d)"
            ),
        }
        _write_json(self.paths["baseline_receipt"], self.baseline_receipt)

    def inputs(self) -> attribution.EvaluationInputs:
        return attribution.EvaluationInputs(
            result_path=self.paths["result"],
            manifest_path=self.paths["manifest"],
            contributions_path=self.paths["contributions"],
            scorecard_path=self.paths["scorecard"],
            fingerprint_path=self.paths["fingerprint"],
            prereg_path=self.paths["attribution_prereg"],
            baseline_receipt_path=self.paths["baseline_receipt"],
        )

    def evaluate(self) -> dict:
        with self.trust():
            return attribution.evaluate_attribution(
                self.prereg,
                self.baseline_receipt,
                self.results,
                self.manifest,
                self.contributions,
                self.scorecard,
                self.fingerprint,
                inputs=self.inputs(),
            )

    def preauthorize(self) -> str:
        with self.trust():
            return attribution._preauthorize_evaluation_dereference(
                self.prereg,
                self.baseline_receipt,
                self.manifest,
                self.fingerprint,
                inputs=self.inputs(),
            )


class Fig171819ClaimAttributionTests(unittest.TestCase):
    def campaign(self) -> tuple[tempfile.TemporaryDirectory, SyntheticCampaign]:
        temporary = tempfile.TemporaryDirectory()
        campaign = SyntheticCampaign(Path(temporary.name))
        return temporary, campaign

    @staticmethod
    def _samples(
        curve: str,
        family: str,
        keys: list[str],
        measured: list[float],
        model: list[float],
    ) -> list[dict]:
        return [
            {
                "curve": curve,
                "physical_family_id": family,
                "channel": "T",
                "abscissa": "twist_deg",
                "measurement_index": index,
                "canonical_nominal_x": float(index),
                "raw_x": float(index),
                "evaluation_x": float(index),
                "measurement_N": float(measured[index]),
                "model_N": float(model[index]),
                "left_condition_key": key,
                "right_condition_key": key,
                "left_weight": 1.0,
                "right_weight": 0.0,
            }
            for index, key in enumerate(keys)
        ]

    def _parent_kernel(
        self,
        *,
        family_count: int = 2,
        fail_n3_family: int | None = None,
    ) -> tuple[dict, dict[str, list[dict]], dict]:
        all_keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        families: list[dict] = []
        samples_by_curve: dict[str, list[dict]] = {}
        node_by_condition: dict = {}
        measured = [0.0, 2.0, 0.0]
        model = [0.0, 1.0, -1.0]
        for family_index in range(family_count):
            family_id = f"PF_SYN_{family_index}"
            curve = f"curve_{family_index}"
            keys = all_keys[3 * family_index : 3 * family_index + 3]
            samples = self._samples(curve, family_id, keys, measured, model)
            classified = attribution._classify_curve(
                curve,
                samples,
                channel="T",
                abscissa="twist_deg",
            )
            classified["guard_contrasts"] = attribution._pairwise_guards(
                curve,
                samples,
            )
            support = sorted(
                {
                    key
                    for component in classified["components"]
                    for key in component["effective_support_condition_keys"]
                },
                key=attribution._condition_sort_key,
            )
            families.append(
                {
                    "physical_family_id": family_id,
                    "channel": "T",
                    "official_curve_keys": [curve],
                    "n_official_curves": 1,
                    "curves": [classified],
                    "positive_support_condition_keys": support,
                    "pairwise_guards": copy.deepcopy(classified["guard_contrasts"]),
                }
            )
            samples_by_curve[curve] = samples
            parent_values = [
                model_value - measured_value
                for model_value, measured_value in zip(model, measured)
            ]
            if fail_n3_family == family_index:
                parent_values = [0.0, 0.0, 0.0]
            for key, n3_value in zip(keys, parent_values):
                node_by_condition[key] = {
                    "N2": {"L": 0.0, "T": 0.0},
                    "N3": {"L": 0.0, "T": n3_value},
                }
        return {"families": families}, samples_by_curve, node_by_condition

    @staticmethod
    def _candidate(
        disease_id: str,
        severities: list[float],
        supports: list[list[str]],
    ) -> dict:
        families = [
            {
                "physical_family_id": f"{disease_id}_PF{index}",
                "official_curve_keys": [f"{disease_id}_C{index}"],
                "pf_severity_N": severity,
                "positive_support_condition_keys": support,
            }
            for index, (severity, support) in enumerate(zip(severities, supports))
        ]
        return {
            "global_disease_id": disease_id,
            "global_disease_identity": {"id": disease_id},
            "physical_families": families,
            "support_conflict_graph": attribution._maximum_independent_sets(
                {
                    item["physical_family_id"]: item["positive_support_condition_keys"]
                    for item in families
                }
            ),
        }

    def test_frozen_protocol_stack_is_exact(self) -> None:
        expected = {
            **{
                path: attribution.AUTHORIZED_ACTIVE_PROTOCOL_SHA256_BY_VERSION[version]
                for version, path in zip(
                    range(3, 9),
                    attribution.ACTIVE_DISEASE_PROTOCOLS,
                )
            },
            **{
                path: attribution.AUTHORIZED_PARENT_PROTOCOL_SHA256_BY_VERSION[version]
                for version, path in zip(
                    range(2, 6),
                    attribution.PARENT_PROTOCOLS,
                )
            },
            attribution.EVIDENCE_SCOPE_PREREG: (
                "c7cbf19f5cd388090ae85fb3e350b70a32beb043fbda04dc958ee055543a8a69"
            ),
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                self.assertEqual(attribution._sha256_file(path), digest)
        attribution._validate_frozen_roots()

    def test_evidence_scope_frozen_root_drift_fails_without_file_mutation(
        self,
    ) -> None:
        with mock.patch.object(
            attribution,
            "AUTHORIZED_EVIDENCE_SCOPE_PREREG_SHA256",
            "0" * 64,
        ):
            with self.assertRaisesRegex(
                attribution.InvalidEvidenceError,
                "frozen attribution root SHA-256 drift.*evidence_scope_prereg",
            ):
                attribution._validate_frozen_roots()

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            attribution._loads_json_strict('{"a":1,"a":2}')

    def test_execution_envelope_binds_h0_h1_h2(self) -> None:
        def rehash_receipt_dag(candidate: dict[str, object]) -> None:
            h0 = candidate["outer_preflight_receipt"]
            h0_sha256 = attribution._receipt_object_sha256(h0)
            candidate["outer_preflight_receipt_sha256"] = h0_sha256
            h1 = candidate["inner_launcher_receipt"]
            h1["upstream_receipts"]["H0"] = h0_sha256
            h1_sha256 = attribution._receipt_object_sha256(h1)
            candidate["inner_launcher_receipt_sha256"] = h1_sha256
            h2 = candidate["outer_completion_receipt"]
            h2["upstream_receipts"] = {
                "H0": h0_sha256,
                "H1": h1_sha256,
            }
            candidate["outer_completion_receipt_sha256"] = (
                attribution._receipt_object_sha256(h2)
            )

        payload_source = {"status": "PREPARED"}
        envelope = _execution_envelope("prepare")
        wrapped = _scientific_envelope(
            payload_source,
            "prepare",
        )
        payload, restored = attribution._unwrap_scientific_payload(
            wrapped,
            stage="prepare",
        )
        self.assertEqual(payload, {"status": "PREPARED"})
        self.assertEqual(restored, envelope)
        forged = copy.deepcopy(envelope)
        forged["checkout_used"] = True
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "checkout",
        ):
            attribution._validate_execution_envelope(
                forged,
                stage="prepare",
            )
        fake_digest = copy.deepcopy(envelope)
        fake_digest["outer_preflight_receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "canonical hash",
        ):
            attribution._validate_execution_envelope(
                fake_digest,
                stage="prepare",
            )

        broken_dag = copy.deepcopy(envelope)
        broken_dag["inner_launcher_receipt"]["upstream_receipts"]["H0"] = "0" * 64
        broken_dag["inner_launcher_receipt_sha256"] = (
            attribution._receipt_object_sha256(broken_dag["inner_launcher_receipt"])
        )
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "does not bind H0",
        ):
            attribution._validate_execution_envelope(
                broken_dag,
                stage="prepare",
            )

        semantic_forgery = copy.deepcopy(envelope)
        semantic_forgery["inner_launcher_receipt"]["body"][
            "runtime_source_closure_sha256"
        ] = "0" * 64
        rehash_receipt_dag(semantic_forgery)
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "H1 closure",
        ):
            attribution._validate_execution_envelope(
                semantic_forgery,
                stage="prepare",
            )

        h0_control_forgery = copy.deepcopy(envelope)
        h0_control_forgery["outer_preflight_receipt"]["body"]["git_hooks_disabled"] = (
            False
        )
        rehash_receipt_dag(h0_control_forgery)
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "H0 command/binding",
        ):
            attribution._validate_execution_envelope(
                h0_control_forgery,
                stage="prepare",
            )

        h1_python_forgery = copy.deepcopy(envelope)
        h1_python_forgery["inner_launcher_receipt"]["body"]["python_no_site_flag"] = (
            False
        )
        rehash_receipt_dag(h1_python_forgery)
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "H1 closure",
        ):
            attribution._validate_execution_envelope(
                h1_python_forgery,
                stage="prepare",
            )

        cleanup_target_forgery = copy.deepcopy(envelope)
        cleanup_target_forgery["outer_completion_receipt"]["body"][
            "cleanup_target_relative_paths"
        ] = [".raw-select-disease-payload.json"]
        rehash_receipt_dag(cleanup_target_forgery)
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "H2 transport/cleanup",
        ):
            attribution._validate_execution_envelope(
                cleanup_target_forgery,
                stage="prepare",
            )

        payload_hash_forgery = copy.deepcopy(wrapped)
        payload_hash_forgery["payload"]["synthetic_extra"] = True
        payload_hash_forgery["payload_sha256"] = attribution._canonical_hash(
            payload_hash_forgery["payload"]
        )
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "differs from embedded H2",
        ):
            attribution._unwrap_scientific_payload(
                payload_hash_forgery,
                stage="prepare",
            )
        self.assertFalse(
            hasattr(attribution, "attach_execution_envelope"),
        )

    def test_prepare_api_has_no_baseline_or_contribution_input(self) -> None:
        parameters = inspect.signature(attribution.prepare_contract).parameters
        self.assertNotIn("contributions", parameters)
        self.assertNotIn("baseline_receipt", parameters)
        self.assertNotIn("baseline_receipt_path", parameters)

    def test_prepare_requires_outer_wrapped_selector(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        _write_json(campaign.paths["disease"], campaign.disease)
        with campaign.trust():
            prepared = attribution.prepare_contract(
                campaign.fingerprint,
                campaign.disease,
                fingerprint_path=campaign.paths["fingerprint"],
                disease_spec_path=campaign.paths["disease"],
            )
        self.assertEqual(prepared["status"], "INVALID_EVIDENCE")
        self.assertRegex(
            prepared["failed_gates"][0],
            "outer scientific envelope is required",
        )

    def test_selector_builds_final_ledgers_and_two_views(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        disease = campaign.disease
        self.assertEqual(disease["status"], "ACTIVE_DISEASE_FROZEN")
        self.assertEqual(len(disease["classification_ledger_42"]), 42)
        self.assertEqual(
            set(disease["rankings"]["views"]),
            {"PF_EQUAL_MEAN", "PF_REPLICATION_FLOOR"},
        )
        encoded = json.dumps(disease["rankings"], sort_keys=True)
        self.assertNotIn("point_weighted", encoded)
        self.assertNotIn("official_curve_equal", disease["rankings"]["views"])
        self.assertGreaterEqual(
            disease["max_pairwise_disjoint_pf_count"],
            2,
        )
        self.assertTrue(
            all(
                item["shadow_local_status"] == "SHADOW_PASS"
                for item in disease["shadow_lopo"]["records"]
            )
        )
        for record in disease["classification_ledger_42"]:
            self.assertEqual(
                len(record["measurement_points"]),
                record["shape_diagnostics"]["n_measurements"],
            )
            self.assertIn(
                "peak_gate_values",
                record["shape_diagnostics"],
            )
            self.assertIn(
                "trough_gate_values",
                record["shape_diagnostics"],
            )
        for shadow in disease["shadow_lopo"]["records"]:
            self.assertIn("classified", shadow["ledger_stages"])
            self.assertIn("alias_pair_ledger", shadow)
            self.assertIn("exclusion_ledger", shadow)
        projection = disease["contribution_free_receipt_projection"]
        self.assertNotIn("scorecard_identity", projection)
        self.assertNotIn("baseline_receipt_identity", projection)

    def test_prepare_never_touches_baseline_or_contribution_targets(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        forbidden = {
            campaign.paths["contributions"].resolve(),
            campaign.paths["baseline_receipt"].resolve(),
        }
        original_sha256 = attribution._sha256_file
        original_stat = Path.stat
        original_is_file = Path.is_file
        original_open = Path.open
        touches: list[str] = []

        def blocked(path: Path, operation: str) -> None:
            if Path(path) in forbidden:
                touches.append(operation)
                raise AssertionError(f"Prepare attempted {operation}")

        def guarded_sha256(path: Path) -> str:
            blocked(path, "hash")
            return original_sha256(path)

        def guarded_stat(path: Path, *args, **kwargs):
            blocked(path, "stat")
            return original_stat(path, *args, **kwargs)

        def guarded_is_file(path: Path) -> bool:
            blocked(path, "is_file")
            return original_is_file(path)

        def guarded_open(path: Path, *args, **kwargs):
            blocked(path, "open")
            return original_open(path, *args, **kwargs)

        with (
            campaign.trust(),
            mock.patch.object(
                attribution,
                "_sha256_file",
                side_effect=guarded_sha256,
            ),
            mock.patch.object(Path, "stat", new=guarded_stat),
            mock.patch.object(Path, "is_file", new=guarded_is_file),
            mock.patch.object(Path, "open", new=guarded_open),
        ):
            prepared = attribution.prepare_contract(
                campaign.fingerprint,
                campaign.disease_artifact,
                fingerprint_path=campaign.paths["fingerprint"],
                disease_spec_path=campaign.paths["disease"],
            )
        self.assertEqual(prepared["status"], "PREPARED")
        self.assertEqual(touches, [])

    def test_invalid_receipt_short_circuits_numeric_science(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        receipt = copy.deepcopy(campaign.baseline_receipt)
        del receipt["bundle_id_payload"]["postprocess_authorization_sha256"]
        _write_json(campaign.paths["baseline_receipt"], receipt)
        with campaign.trust():
            output = attribution.select_active_disease(
                campaign.fingerprint,
                receipt,
                fingerprint_path=campaign.paths["fingerprint"],
                baseline_receipt_path=campaign.paths["baseline_receipt"],
            )
        self.assertEqual(output["status"], "INVALID_EVIDENCE")
        self.assertEqual(
            output["rankings"],
            {"evaluation_status": "NOT_EVALUATED_UPSTREAM_FAILURE"},
        )
        self.assertEqual(
            output["shadow_lopo"],
            {"evaluation_status": "NOT_RUN_UPSTREAM_FAILURE"},
        )
        self.assertNotIn("ledger_stages", output)
        self.assertNotIn("disease_id", output)

    def test_prepare_rebuilds_and_rejects_disease_ranking_tamper(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        tampered = copy.deepcopy(campaign.disease)
        first_score = tampered["rankings"]["views"]["PF_EQUAL_MEAN"]["scores"][0]
        first_score["score_N"] += 1.0
        tampered_artifact = _scientific_envelope(
            tampered,
            "select-disease",
        )
        _write_json(campaign.paths["disease"], tampered_artifact)
        with campaign.trust():
            prepared = attribution.prepare_contract(
                campaign.fingerprint,
                tampered_artifact,
                fingerprint_path=campaign.paths["fingerprint"],
                disease_spec_path=campaign.paths["disease"],
            )
        self.assertEqual(prepared["status"], "INVALID_EVIDENCE")
        self.assertRegex(
            prepared["failed_gates"][0],
            "independent reconstruction",
        )
        self.assertNotIn("families", prepared)

    def test_prepare_projection_rejects_contribution_identity_leak(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        tampered = copy.deepcopy(campaign.disease)
        projection = tampered["contribution_free_receipt_projection"]
        projection["contributions_sha256"] = "f" * 64
        tampered_artifact = _scientific_envelope(
            tampered,
            "select-disease",
        )
        _write_json(campaign.paths["disease"], tampered_artifact)
        with campaign.trust():
            prepared = attribution.prepare_contract(
                campaign.fingerprint,
                tampered_artifact,
                fingerprint_path=campaign.paths["fingerprint"],
                disease_spec_path=campaign.paths["disease"],
            )
        self.assertEqual(prepared["status"], "INVALID_EVIDENCE")
        self.assertRegex(
            prepared["failed_gates"][0],
            "projection schema|leaks contribution|forbidden claim/contribution",
        )

    def test_prepare_rejects_contribution_metadata_in_selector_provenance(
        self,
    ) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        tampered = copy.deepcopy(campaign.disease)
        tampered["provenance"]["claim_contributions"] = {
            "path": str(campaign.paths["contributions"]),
            "sha256": attribution._sha256_file(campaign.paths["contributions"]),
        }
        tampered_artifact = _scientific_envelope(
            tampered,
            "select-disease",
        )
        _write_json(campaign.paths["disease"], tampered_artifact)
        with campaign.trust():
            prepared = attribution.prepare_contract(
                campaign.fingerprint,
                tampered_artifact,
                fingerprint_path=campaign.paths["fingerprint"],
                disease_spec_path=campaign.paths["disease"],
            )
        self.assertEqual(prepared["status"], "INVALID_EVIDENCE")
        self.assertRegex(
            prepared["failed_gates"][0],
            "forbidden claim/contribution|provenance leaks|provenance identity",
        )

    def test_complete_resume_proof_is_mandatory(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        receipt = copy.deepcopy(campaign.baseline_receipt)
        receipt["validation"]["complete_resume_triplet_unchanged"] = False
        _write_json(campaign.paths["baseline_receipt"], receipt)
        with campaign.trust():
            output = attribution.select_active_disease(
                campaign.fingerprint,
                receipt,
                fingerprint_path=campaign.paths["fingerprint"],
                baseline_receipt_path=campaign.paths["baseline_receipt"],
            )
        self.assertEqual(output["status"], "INVALID_EVIDENCE")
        self.assertRegex(output["failed_gates"][0], "triplet validation")

    def test_guard_99_of_100_attack_is_rejected(self) -> None:
        guards = compare_test._guards()
        guards["force_ledger"]["max_abs_error_N"] = 99.0e-9
        guards["force_ledger"]["tolerance_N"] = 100.0e-9
        self.assertFalse(attribution._strict_guards(guards))

    def test_authorization_rejects_n1_hash_drift(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        common = campaign.manifest["common_claim_manifest"]
        common["nodes"][0]["implementation_hash"] = f"sha256:{'f' * 64}"
        campaign.manifest["claim_graph_identity_sha256"] = (
            compare.witness._claim_graph_identity_sha256(common)
        )
        _write_json(campaign.paths["manifest"], campaign.manifest)
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "authorized V4.1|authorization",
        ):
            campaign.preauthorize()

    def test_authorization_rejects_call_and_source_drift(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        calls = {}
        for key, case in campaign.contributions["cases"].items():
            resolved = case["resolved_call"]
            resolved["lb_hybrid"] = 1.0
            resolved["steps_per_cycle"] = 1
            resolved["wake_rows"] = 1
            calls[key] = resolved
        campaign.manifest["resolved_call_contract_sha256"] = (
            attribution._canonical_hash(calls)
        )
        source = next(iter(campaign.manifest["solver_source_hashes"]))
        campaign.manifest["solver_source_hashes"][source] = "f" * 64
        _write_json(campaign.paths["manifest"], campaign.manifest)
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "authorization|resolved|identity",
        ):
            campaign.preauthorize()

    def test_authoritative_42_434_151_34_8_contract_is_rebuilt(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        samples, curves, families = attribution._fingerprint_maps(campaign.fingerprint)
        self.assertEqual(len(curves), 42)
        self.assertEqual(sum(map(len, samples.values())), 434)
        self.assertEqual(len(attribution.EXPECTED_CONDITION_KEYS), 151)
        self.assertEqual(len(families), 34)
        self.assertEqual(
            sum(len(item["official_curve_keys"]) > 1 for item in families.values()),
            8,
        )

    def test_coordinated_uppercase_fake_19c_is_rejected(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        forged = copy.deepcopy(campaign.fingerprint)
        old = forged["official_curves"][0]["curve"]
        fake = "19|C|0"
        forged["official_curves"][0]["curve"] = fake
        for sample in forged["samples"]:
            if sample["curve"] == old:
                sample["curve"] = fake
        family_id = forged["official_curves"][0]["physical_family_id"]
        family = next(
            item
            for item in forged["physical_curve_families"]
            if item["physical_family_id"] == family_id
        )
        family["official_curve_keys"] = [
            fake if item == old else item for item in family["official_curve_keys"]
        ]
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "identity|contract",
        ):
            attribution._fingerprint_maps(forged)

    def test_shape_uses_cumulative_reversal_and_strict_boundary(self) -> None:
        cumulative = attribution._shape_components([0.0, 1.0, 0.6, 1.2])
        self.assertEqual(
            cumulative["status"],
            "INELIGIBLE_NO_ROBUST_ZERO_SUM_SHAPE",
        )
        boundary = attribution._shape_components([0.0, 0.3])
        self.assertEqual(
            boundary["status"],
            "INELIGIBLE_NO_ROBUST_ZERO_SUM_SHAPE",
        )

    def test_shape_peak_plateau_and_monotone_end_rules(self) -> None:
        self.assertEqual(
            attribution._shape_components([0.0, 1.0, 0.0])["shape_class"],
            "PEAK",
        )
        plateau_peak = attribution._shape_components([0.0, 1.0, 1.0, 0.0])
        self.assertEqual(
            plateau_peak["status"],
            "INELIGIBLE_NO_ROBUST_ZERO_SUM_SHAPE",
        )
        monotone_plateau = attribution._shape_components([0.0, 0.5, 0.5, 1.0])
        self.assertEqual(
            monotone_plateau["shape_class"],
            "END_INCREASING",
        )

    def test_component_state_covers_pass_reversed_under_over(self) -> None:
        E = 2.0
        self.assertEqual(attribution._baseline_component_state(E, 1.9), "PASS")
        self.assertEqual(
            attribution._baseline_component_state(E, -1.0),
            "REVERSED",
        )
        self.assertEqual(attribution._baseline_component_state(E, 0.0), "UNDER")
        self.assertEqual(attribution._baseline_component_state(E, 3.0), "OVER")

    def test_canonical_alpha_cancels_shared_solver_support(self) -> None:
        key = min(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        samples = self._samples(
            "curve",
            "PF",
            [key, key],
            [0.0, 1.0],
            [0.0, 1.0],
        )
        alpha = attribution._canonical_alpha([-1.0, 1.0], samples)
        self.assertEqual(alpha, [])

    def test_exact_mis_rejects_overlapping_nonidentical_supports(self) -> None:
        keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )[:4]
        report = attribution._maximum_independent_sets(
            {
                "A": [keys[0], keys[1]],
                "B": [keys[0], keys[2]],
                "C": [keys[3]],
            }
        )
        self.assertEqual(report["max_pairwise_disjoint_pf_count"], 2)
        self.assertEqual(
            report["all_maximum_disjoint_pf_sets"],
            [["A", "C"], ["B", "C"]],
        )

    def test_global_disease_allows_cross_pf_geometry_but_withdraws_alias(self) -> None:
        identity = {
            "channel": "T",
            "abscissa": "twist_deg",
            "shape_class": "PEAK",
            "components": [
                {
                    "component_name": "RISE",
                    "sign_E": 1,
                    "baseline_state": "UNDER",
                }
            ],
        }

        def record(curve: str, pf: str, geometry_index: int) -> dict:
            return {
                "curve": curve,
                "physical_family_id": pf,
                "candidate_eligible": True,
                "exclusion_reason": None,
                "global_disease_id": "GD-X",
                "global_disease_identity": identity,
                "alias_geometry_signature": {
                    "global_disease_identity": identity,
                    "components": [
                        {
                            "component_name": "RISE",
                            "canonical_nominal_indices": [0, geometry_index],
                            "measurement_coefficient_vector": [-1.0, 1.0],
                        }
                    ],
                },
                "curve_severity_N": 1.0,
                "components": [
                    {
                        "effective_support_condition_keys": [
                            sorted(
                                attribution.EXPECTED_CONDITION_KEYS,
                                key=attribution._condition_sort_key,
                            )[geometry_index]
                        ]
                    }
                ],
            }

        classified = [
            record("A", "PF_A", 1),
            record("B", "PF_B", 2),
            record("C1", "PF_C", 1),
            record("C2", "PF_C", 2),
        ]
        families = {
            "PF_A": {"official_curve_keys": ["A"]},
            "PF_B": {"official_curve_keys": ["B"]},
            "PF_C": {"official_curve_keys": ["C1", "C2"]},
        }
        ledgers = attribution._build_disease_ledgers(classified, families)
        consensus = ledgers["consensus"][0]["physical_families"]
        self.assertEqual(
            [item["physical_family_id"] for item in consensus],
            ["PF_A", "PF_B"],
        )
        self.assertIn(
            "ALIAS_WITHDRAWN",
            [item["reason"] for item in ledgers["exclusion_ledger"]],
        )

    def test_fixed_mean_floor_views_detect_disagreement(self) -> None:
        keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        ledgers = {
            "pre_alias": [{"global_disease_id": "A"}],
            "consensus": [
                self._candidate("A", [10.0, 0.1], [[keys[0]], [keys[1]]]),
                self._candidate("B", [4.0, 4.0], [[keys[2]], [keys[3]]]),
            ],
        }
        gate = attribution._main_selection_gate(
            ledgers,
            minimum_independent_families=2,
        )
        self.assertEqual(
            gate["gate_status"],
            "NO_DECISION_VIEW_DISAGREEMENT",
        )

    def test_main_reason_short_circuit_states_are_exact(self) -> None:
        no_pre = attribution._main_selection_gate(
            {"pre_alias": [], "consensus": []},
            minimum_independent_families=2,
        )
        self.assertEqual(
            no_pre["gate_status"],
            "NO_DECISION_NO_PRE_REPLICATION_CANDIDATE",
        )
        key = min(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        one_pf = self._candidate("A", [1.0], [[key]])
        no_replication = attribution._main_selection_gate(
            {
                "pre_alias": [{"global_disease_id": "A"}],
                "consensus": [one_pf],
            },
            minimum_independent_families=2,
        )
        self.assertEqual(
            no_replication["gate_status"],
            "NO_DECISION_NO_INDEPENDENT_REPLICATION",
        )

    def test_argmax_numeric_tie_is_not_broken_by_disease_id(self) -> None:
        keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        gate = attribution._main_selection_gate(
            {
                "pre_alias": [{"global_disease_id": "A"}],
                "consensus": [
                    self._candidate("A", [1.0, 1.0], [[keys[0]], [keys[1]]]),
                    self._candidate("B", [1.0, 1.0], [[keys[2]], [keys[3]]]),
                ],
            },
            minimum_independent_families=2,
        )
        self.assertEqual(gate["gate_status"], "NO_DECISION_DISEASE_TIE")
        for view in gate["rankings"]["views"].values():
            self.assertEqual(
                view["argmax_global_disease_ids"],
                ["A", "B"],
            )

    def test_rank_margin_is_strictly_greater_than_point15(self) -> None:
        keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        ledgers = {
            "pre_alias": [{"global_disease_id": "A"}],
            "consensus": [
                self._candidate("A", [0.3, 0.3], [[keys[0]], [keys[1]]]),
                self._candidate("B", [0.15, 0.15], [[keys[2]], [keys[3]]]),
            ],
        }
        gate = attribution._main_selection_gate(
            ledgers,
            minimum_independent_families=2,
        )
        self.assertEqual(
            gate["gate_status"],
            "NO_DECISION_INSUFFICIENT_RANK_MARGIN",
        )
        ledgers["consensus"][1] = self._candidate(
            "B",
            [0.14, 0.14],
            [[keys[2]], [keys[3]]],
        )
        gate = attribution._main_selection_gate(
            ledgers,
            minimum_independent_families=2,
        )
        self.assertEqual(gate["gate_status"], "GATE_QUALIFIED")

    def test_rank_margin_has_no_unregistered_one_e_minus_12_dead_band(
        self,
    ) -> None:
        keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        excess = 5.0e-13
        winner = 0.3
        runner = winner - (attribution.RANK_MARGIN_N + excess)
        gate = attribution._main_selection_gate(
            {
                "pre_alias": [{"global_disease_id": "A"}],
                "consensus": [
                    self._candidate(
                        "A",
                        [winner, winner],
                        [[keys[0]], [keys[1]]],
                    ),
                    self._candidate(
                        "B",
                        [runner, runner],
                        [[keys[2]], [keys[3]]],
                    ),
                ],
            },
            minimum_independent_families=2,
        )
        self.assertGreater(
            winner - runner,
            attribution.RANK_MARGIN_N,
        )
        self.assertEqual(gate["gate_status"], "GATE_QUALIFIED")

    def test_shadow_lopo_uses_nonrecursive_local_status(self) -> None:
        keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        main = attribution._main_selection_gate(
            {
                "pre_alias": [{"global_disease_id": "A"}],
                "consensus": [
                    self._candidate("A", [10.0, 5.0], [[keys[0]], [keys[1]]]),
                    self._candidate("B", [6.0, 4.0], [[keys[2]], [keys[3]]]),
                ],
            },
            minimum_independent_families=2,
        )
        self.assertEqual(main["gate_status"], "GATE_QUALIFIED")
        shadow = attribution._main_selection_gate(
            {
                "pre_alias": [{"global_disease_id": "A"}],
                "consensus": [
                    self._candidate("A", [5.0], [[keys[1]]]),
                    self._candidate("B", [6.0, 4.0], [[keys[2]], [keys[3]]]),
                ],
            },
            minimum_independent_families=1,
        )
        self.assertEqual(
            attribution._shadow_status(shadow, "A"),
            "SHADOW_DISEASE_TIE",
        )

    def test_all_pair_guards_are_not_deduplicated_by_alpha(self) -> None:
        key = min(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        samples = self._samples(
            "curve",
            "PF",
            [key, key, key],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        )
        guards = attribution._pairwise_guards("curve", samples)
        self.assertEqual(len(guards), 3)
        self.assertTrue(all(not guard["canonical_alpha"] for guard in guards))
        self.assertEqual(
            {tuple(guard["measurement_pair"]) for guard in guards},
            {(0, 1), (0, 2), (1, 2)},
        )

    def test_parent_restores_under_side_and_preserves_pass_side(self) -> None:
        prereg, samples, nodes = self._parent_kernel()
        result = attribution._evaluate_node("N3", prereg, samples, nodes)
        self.assertEqual(result["status"], "PARENT_PASS")
        self.assertEqual(result["local_reasons"], [])
        self.assertTrue(result["pass_component_results"])
        self.assertTrue(
            all(
                item["component_restoration_boolean"]
                for item in result["pass_component_results"]
            )
        )

    def test_parent_component_improvement_boundary_is_strict(self) -> None:
        keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )[:3]
        first_samples = self._samples(
            "boundary_curve",
            "PF_BOUNDARY",
            keys,
            [0.0, 0.6, 0.0],
            [0.0, 0.0, -0.6],
        )
        prepared = attribution._classify_curve(
            "boundary_curve",
            first_samples,
            channel="T",
            abscissa="twist_deg",
        )
        prepared["guard_contrasts"] = attribution._pairwise_guards(
            "boundary_curve",
            first_samples,
        )
        nodes = {
            key: {
                "N2": {"L": 0.0, "T": 0.0},
                "N3": {"L": 0.0, "T": parent_value},
            }
            for key, parent_value in zip(keys, [0.0, -0.3, -0.3])
        }
        result = attribution._evaluate_curve(
            prepared,
            first_samples,
            node="N3",
            channel="T",
            node_by_condition=nodes,
        )
        under = next(
            item
            for item in result["component_results"]
            if item["baseline_state"] == "UNDER"
        )
        self.assertAlmostEqual(
            under["absolute_error_improvement_N"],
            0.3,
        )
        self.assertFalse(under["component_restoration_boolean"])

    def test_parent_improvement_has_no_unregistered_one_e_minus_12_dead_band(
        self,
    ) -> None:
        keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )[:3]
        samples = self._samples(
            "near_boundary_curve",
            "PF_NEAR_BOUNDARY",
            keys,
            [0.0, 0.6, 0.0],
            [0.0, 0.0, -0.6],
        )
        prepared = attribution._classify_curve(
            "near_boundary_curve",
            samples,
            channel="T",
            abscissa="twist_deg",
        )
        prepared["guard_contrasts"] = attribution._pairwise_guards(
            "near_boundary_curve",
            samples,
        )
        improvement = 0.3 + 5.0e-13
        nodes = {
            key: {
                "N2": {"L": 0.0, "T": 0.0},
                "N3": {"L": 0.0, "T": parent_value},
            }
            for key, parent_value in zip(
                keys,
                [0.0, -improvement, -improvement],
            )
        }
        result = attribution._evaluate_curve(
            prepared,
            samples,
            node="N3",
            channel="T",
            node_by_condition=nodes,
        )
        under = next(
            item
            for item in result["component_results"]
            if item["baseline_state"] == "UNDER"
        )
        self.assertGreater(
            under["absolute_error_improvement_N"],
            attribution.CONTRAST_TOLERANCE_N,
        )
        self.assertTrue(under["component_restoration_boolean"])

    def test_parent_requires_full_pf_coverage_not_favorable_mis_subset(self) -> None:
        prereg, samples, nodes = self._parent_kernel(
            family_count=3,
            fail_n3_family=2,
        )
        result = attribution._evaluate_node("N3", prereg, samples, nodes)
        self.assertEqual(
            result["positive_support_conflict_graph"]["max_pairwise_disjoint_pf_count"],
            3,
        )
        self.assertEqual(result["status"], "PARENT_FAIL")
        self.assertEqual(len(result["restored_pf_ids"]), 2)
        self.assertIn(
            "PARENT_FAIL_PF_NOT_FULLY_RESTORED",
            result["local_reasons"],
        )

    def test_parent_positive_support_excludes_guard_support(self) -> None:
        prereg, samples, nodes = self._parent_kernel()
        shared = min(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        for family in prereg["families"]:
            for curve in family["curves"]:
                for guard in curve["guard_contrasts"]:
                    guard["canonical_alpha"] = [
                        {
                            "condition_key": shared,
                            "condition": list(attribution._condition_sort_key(shared)),
                            "coefficient": 1.0,
                        }
                    ]
                    nodes.setdefault(
                        shared,
                        {
                            "N2": {"L": 0.0, "T": 0.0},
                            "N3": {"L": 0.0, "T": 0.0},
                        },
                    )
        result = attribution._evaluate_node("N3", prereg, samples, nodes)
        self.assertEqual(
            result["positive_support_conflict_graph"]["edges"],
            [],
        )

    def test_parent_alias_signature_detects_nonuniform_outcome(self) -> None:
        prereg, samples, nodes = self._parent_kernel()
        first = prereg["families"][0]
        all_keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        clone_samples = self._samples(
            "curve_alias",
            first["physical_family_id"],
            all_keys[9:12],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, -1.0],
        )
        clone = attribution._classify_curve(
            "curve_alias",
            clone_samples,
            channel="T",
            abscissa="twist_deg",
        )
        clone["guard_contrasts"] = attribution._pairwise_guards(
            "curve_alias",
            clone_samples,
        )
        samples["curve_alias"] = clone_samples
        first["curves"].append(clone)
        first["official_curve_keys"].append("curve_alias")
        for item in clone_samples:
            key = item["left_condition_key"]
            nodes[key] = {
                "N2": {"L": 0.0, "T": 0.0},
                "N3": {"L": 0.0, "T": 0.0},
            }
        result = attribution._evaluate_node("N3", prereg, samples, nodes)
        self.assertEqual(result["status"], "PARENT_FAIL")
        self.assertIn(
            "PARENT_FAIL_ALIAS_NONUNIFORM",
            result["local_reasons"],
        )

    def test_parent_aliases_with_same_failure_are_not_nonuniform(self) -> None:
        prereg, samples, nodes = self._parent_kernel()
        first = prereg["families"][0]
        original_curve = first["curves"][0]["curve"]
        original_samples = samples[original_curve]
        all_keys = sorted(
            attribution.EXPECTED_CONDITION_KEYS,
            key=attribution._condition_sort_key,
        )
        clone_samples = self._samples(
            "curve_alias_same_failure",
            first["physical_family_id"],
            all_keys[9:12],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, -1.0],
        )
        clone = attribution._classify_curve(
            "curve_alias_same_failure",
            clone_samples,
            channel="T",
            abscissa="twist_deg",
        )
        clone["guard_contrasts"] = attribution._pairwise_guards(
            "curve_alias_same_failure",
            clone_samples,
        )
        samples["curve_alias_same_failure"] = clone_samples
        first["curves"].append(clone)
        first["official_curve_keys"].append("curve_alias_same_failure")
        for item in [*original_samples, *clone_samples]:
            key = item["left_condition_key"]
            nodes[key] = {
                "N2": {"L": 0.0, "T": 0.0},
                "N3": {"L": 0.0, "T": 0.0},
            }
        result = attribution._evaluate_node("N3", prereg, samples, nodes)
        self.assertEqual(result["status"], "PARENT_FAIL")
        self.assertNotIn(
            "PARENT_FAIL_ALIAS_NONUNIFORM",
            result["local_reasons"],
        )
        self.assertIn(
            "PARENT_FAIL_COMPONENT_UNDER_NOT_IMPROVED",
            result["local_reasons"],
        )

    def test_parent_truth_table_is_exact(self) -> None:
        passed = {"status": "PARENT_PASS"}
        failed = {"status": "PARENT_FAIL"}
        self.assertEqual(
            attribution._decision(passed, failed)[0],
            "ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS",
        )
        self.assertEqual(
            attribution._decision(failed, passed)[0],
            "ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS",
        )
        self.assertEqual(
            attribution._decision(passed, passed)[0],
            "NO_DECISION_MULTIPLE_PARENTS",
        )
        both_failed = attribution._decision(failed, failed)
        self.assertEqual(
            both_failed[0],
            "NO_DECISION_NO_PARENT_FULL_COVERAGE",
        )
        self.assertEqual(
            both_failed[2],
            ["NO_DECISION_MISSING_OR_STATE_MEDIATED"],
        )

    def test_global_leaf_inventory_rejects_duplicate_or_extra_leaf(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        key = sorted(campaign.contributions["cases"])[0]
        case = copy.deepcopy(campaign.contributions["cases"][key])
        case["claim_contributions"]["N2"].append(
            copy.deepcopy(case["claim_contributions"]["N2"][0])
        )
        condition = compare.CONDITION_BY_KEY[key]
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "inventory length",
        ):
            attribution._contribution_case(
                case,
                key=key,
                condition=condition,
                result=campaign.results[key],
                manifest_guard=campaign.manifest["case_guards"][key],
                common_manifest=campaign.manifest["common_claim_manifest"],
                graph_identity=campaign.manifest["claim_graph_identity_sha256"],
            )

    def test_full_evaluator_uses_four_state_truth_table_and_never_n6(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        report = campaign.evaluate()
        self.assertIn(
            report["status"],
            {
                "ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS",
                "ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS",
                "NO_DECISION_MULTIPLE_PARENTS",
                "NO_DECISION_NO_PARENT_FULL_COVERAGE",
            },
        )
        self.assertEqual(
            report["parent_evaluation"]["evaluation_status"],
            "EVALUATED",
        )
        self.assertEqual(
            report["N6_negative_control"],
            "NOT_EVALUATED_FOR_PARENT_SELECTION",
        )
        self.assertNotIn("forbidden_control", report)
        self.assertTrue(
            report["validity_gates"]["global_11_leaf_inventory_and_roles_match"]
        )

    def test_running_manifest_is_invalid_before_contribution_dereference(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        campaign.manifest["status"] = "running"
        _write_json(campaign.paths["manifest"], campaign.manifest)
        with self.assertRaisesRegex(
            attribution.InvalidEvidenceError,
            "complete|authorization",
        ):
            campaign.preauthorize()

    def test_public_evaluator_separates_upstream_invalid_state(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        campaign.manifest["status"] = "running"
        _write_json(campaign.paths["manifest"], campaign.manifest)
        report = campaign.evaluate()
        self.assertEqual(report["status"], "INVALID_EVIDENCE")
        self.assertEqual(
            report["parent_evaluation"]["evaluation_status"],
            "NOT_EVALUATED_UPSTREAM_FAILURE",
        )
        self.assertNotIn("N2", report)
        self.assertNotIn("family_equal_mae", report)

    def test_evaluate_requires_outer_wrapped_prepare(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        _write_json(
            campaign.paths["attribution_prereg"],
            campaign.prereg_payload,
        )
        with campaign.trust():
            report = attribution.evaluate_attribution(
                campaign.prereg_payload,
                campaign.baseline_receipt,
                campaign.results,
                campaign.manifest,
                campaign.contributions,
                campaign.scorecard,
                campaign.fingerprint,
                inputs=campaign.inputs(),
            )
        self.assertEqual(report["status"], "INVALID_EVIDENCE")
        self.assertRegex(
            report["failed_gates"][0],
            "outer scientific envelope is required",
        )

    def test_evaluate_rejects_cross_stage_execution_identity_drift(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        execution = campaign.prereg["execution_envelope"]
        execution["runtime_source_closure_sha256"] = "9" * 64
        h0 = execution["outer_preflight_receipt"]
        h0["body"]["runtime_source_closure_sha256"] = "9" * 64
        h0_sha256 = attribution._receipt_object_sha256(h0)
        execution["outer_preflight_receipt_sha256"] = h0_sha256
        h1 = execution["inner_launcher_receipt"]
        h1["upstream_receipts"]["H0"] = h0_sha256
        h1["body"]["runtime_source_closure_sha256"] = "9" * 64
        h1_sha256 = attribution._receipt_object_sha256(h1)
        execution["inner_launcher_receipt_sha256"] = h1_sha256
        h2 = execution["outer_completion_receipt"]
        h2["upstream_receipts"] = {
            "H0": h0_sha256,
            "H1": h1_sha256,
        }
        execution["outer_completion_receipt_sha256"] = (
            attribution._receipt_object_sha256(h2)
        )
        _write_json(campaign.paths["attribution_prereg"], campaign.prereg)
        report = campaign.evaluate()
        self.assertEqual(report["status"], "INVALID_EVIDENCE")
        self.assertRegex(
            report["failed_gates"][0],
            "cross-stage execution identity drift",
        )
        self.assertNotIn("N2", report)

    def test_public_evaluator_separates_invalid_derived_invariant(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        fake_results = [
            {
                "node": "N2",
                "status": "PARENT_PASS",
                "local_reasons": [],
                "family_equal_mae_improvement_N": 0.15,
            },
            {
                "node": "N3",
                "status": "PARENT_FAIL",
                "local_reasons": ["PARENT_FAIL_CURVE_MAE"],
                "family_equal_mae_improvement_N": 0.0,
            },
        ]
        with mock.patch.object(
            attribution,
            "_evaluate_node",
            side_effect=fake_results,
        ):
            report = campaign.evaluate()
        self.assertEqual(report["status"], "INVALID_EVIDENCE")
        self.assertEqual(
            report["parent_evaluation"]["evaluation_status"],
            "EVALUATED_INVALID_INVARIANT",
        )
        self.assertEqual(
            report["family_equal_mae"]["failed_implications"],
            ["N2_PASS_IMPLIES_THRESHOLD"],
        )
        self.assertNotIn("N2", report)

    def test_output_is_deterministic_and_never_authorizes_yaml(self) -> None:
        temporary, campaign = self.campaign()
        self.addCleanup(temporary.cleanup)
        first = campaign.evaluate()
        second = campaign.evaluate()
        self.assertEqual(
            attribution._canonical_hash(first),
            attribution._canonical_hash(second),
        )
        encoded = json.dumps(first, ensure_ascii=False)
        self.assertNotIn("write_yaml", encoded)
        self.assertFalse(first["claim_writeback_allowed"])


if __name__ == "__main__":
    unittest.main()
