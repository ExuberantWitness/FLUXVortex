"""Fail-closed scorer for the preregistered G0c AoA-ladder witnesses.

The scorer never calls the aerodynamic solver and never consumes the runner's
diagnostic summary.  Every quantity allowed to vote is reconstructed from the
content-addressed, last-cycle raw recorder arrays.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_benchmark as benchmark  # noqa: E402
import run_n1_n2_n3_aoa_ladder_witnesses as witness  # noqa: E402
import score_n1_n2_ledger_phase_witnesses as base  # noqa: E402


SCHEMA = "n1-n2-n3-aoa-ladder-score-v1"
TAU_F_N = 0.15
TAU_CONTRAST_N = 0.30
NUMERIC_ZERO = 1.0e-12
COLLINEAR_CONDITION_LIMIT = 20.0
DATA_SOURCE_SHA256 = (
    "ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1"
)
# Updated only when the preregistration itself is intentionally rewritten
# before a campaign.  A source-closure match alone is not allowed to redefine
# the frozen decision rule after seeing results.
PREREG_SHA256 = (
    "8aefb909db41b13f2bf256884e940c57d8c28da6288648ce672ecc56d2909bda"
)

N1_YAML = PLATFORM / "claim_nodes" / "n1_uvlm.yaml"
N2_YAML = PLATFORM / "claim_nodes" / "n2_kirchhoff.yaml"
N3_YAML = PLATFORM / "claim_nodes" / "n3_ds_vortex.yaml"
DATA_SOURCE = PLATFORM / "docs" / "data.md"

ScoreContractError = base.ScoreContractError
ArtifactBlob = base.ArtifactBlob
_array_schema_fields = base._array_schema_fields
_read_once = base._read_once
_read_confined = base._read_confined
_json_from_blob = base._json_from_blob
_npz_from_blob = base._npz_from_blob
_verify_artifact = base._verify_artifact
_maximum_abs = base._maximum_abs
_wind_vectors = base._wind_vectors
_robust_mean = base._robust_mean
_is_sha256 = base._is_sha256
_write_json_atomic = witness.base._write_json_atomic

RAW_STEPS = 240
RAW_NC = 4
RAW_NS = 8
RAW_NPAN = RAW_NC * RAW_NS
RAW_EXPECTED_SHAPES = dict(base.RAW_EXPECTED_SHAPES)
if (
    len(RAW_EXPECTED_SHAPES) != 92
    or RAW_STEPS != base.RAW_STEPS
    or RAW_NS != base.RAW_NS
    or RAW_NPAN != base.RAW_NPAN
):  # pragma: no cover - import-time source-contract obstruction
    raise RuntimeError("the inherited 92-field raw recorder contract drifted")

EXPECTED_CASES: dict[str, tuple[float, float]] = {
    f"aoa_f{str(frequency).replace('.', 'p')}_A{aoa}": (
        frequency,
        float(aoa),
    )
    for frequency in (1.4, 2.6)
    for aoa in (0, 5, 10, 15)
}

CONTRASTS = tuple(
    (
        f"C_f{str(frequency).replace('.', 'p')}_A{high}_minus_A{low}",
        f"f{str(frequency).replace('.', 'p')}",
        f"aoa_f{str(frequency).replace('.', 'p')}_A{high}",
        f"aoa_f{str(frequency).replace('.', 'p')}_A{low}",
    )
    for frequency in (1.4, 2.6)
    for low, high in ((0, 5), (5, 10), (10, 15))
)

TEMPLATE_KEYS = ("q1", "q2", "q3")
TEMPLATE_LABELS = {
    "q1": "N1_LE_suction_withdrawal",
    "q2": "N2_old_panel_candidate_direction",
    "q3": "N3_booked_direct_withdrawal",
}
ACTIVE_STATUS = {
    "q1": "N1_LEDGER_AUDIT_REQUIRED",
    "q2": "ACTIVE_N2_6_SHADOW_PREREG_ALLOWED",
    "q3": "ACTIVE_N3_SPATIAL_STATE_AUDIT_REQUIRED",
}


@dataclass(frozen=True)
class CaseMetrics:
    """Raw-derived metrics for one frozen science case."""

    model_robust: np.ndarray
    model_raw: np.ndarray
    q1_raw: np.ndarray
    q1_robust: np.ndarray
    q2_raw: np.ndarray
    q2_robust: np.ndarray
    q3_raw: np.ndarray
    q3_robust: np.ndarray
    q2_active_step_fraction: float

    def as_mapping(self) -> dict[str, Any]:
        return {
            "model_robust": self.model_robust,
            "model_raw": self.model_raw,
            "q1_raw": self.q1_raw,
            "q1_robust": self.q1_robust,
            "q2_raw": self.q2_raw,
            "q2_robust": self.q2_robust,
            "q3_raw": self.q3_raw,
            "q3_robust": self.q3_robust,
            "q2_active_step_fraction": self.q2_active_step_fraction,
        }


def _case_contracts() -> tuple[Any, ...]:
    cases = tuple(witness._case_contracts())
    actual_ids = {case.case_id for case in cases}
    if len(cases) != 8 or actual_ids != set(EXPECTED_CASES):
        raise ScoreContractError("AoA-ladder case identity set drifted")
    for case in cases:
        frequency, aoa = EXPECTED_CASES[case.case_id]
        expected = {
            "family": "fig19_aoa_ladder",
            "U_m_s": 8.0,
            "frequency_Hz": frequency,
            "nominal_twist_deg": 22.5,
            "solver_twist_amplitude_deg": 11.25,
            "aoa_deg": aoa,
            "twist_phase_deg": -90.0,
        }
        mismatch = {
            name: {"expected": value, "actual": getattr(case, name, None)}
            for name, value in expected.items()
            if getattr(case, name, None) != value
        }
        if mismatch:
            raise ScoreContractError(
                f"{case.case_id}: fixed science contract drifted: {mismatch}"
            )
    return cases


def _validate_raw_schema(
    *,
    arrays: Mapping[str, np.ndarray],
    schema: Mapping[str, Any],
    case: Any,
) -> None:
    actual_names = set(arrays)
    expected_names = set(RAW_EXPECTED_SHAPES)
    if actual_names != expected_names:
        raise ScoreContractError(
            f"{case.case_id}: raw field identity mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    malformed_shapes = {
        name: {"expected": expected, "actual": arrays[name].shape}
        for name, expected in RAW_EXPECTED_SHAPES.items()
        if arrays[name].shape != expected
    }
    if malformed_shapes:
        raise ScoreContractError(
            f"{case.case_id}: raw shape mismatch: {malformed_shapes}"
        )
    non_finite = [
        name
        for name, value in arrays.items()
        if value.dtype.kind in "biufc" and not np.isfinite(value).all()
    ]
    if non_finite:
        raise ScoreContractError(
            f"{case.case_id}: non-finite raw fields: {non_finite}"
        )
    if schema.get("fields") != _array_schema_fields(arrays):
        raise ScoreContractError(f"{case.case_id}: raw schema fields mismatch")
    expected_identity = {
        "schema": witness.RAW_SCHEMA_VERSION,
        "case_id": case.case_id,
        "stage": witness.RAW_STAGE,
        "snapshot_phase": "post_force_pre_shed",
        "time_window": "last_cycle",
        "processing": "none",
        "figure16_alignment_status": "unresolved_external_kinematics",
    }
    mismatches = {
        name: {"expected": expected, "actual": schema.get(name)}
        for name, expected in expected_identity.items()
        if schema.get(name) != expected
    }
    if mismatches:
        raise ScoreContractError(
            f"{case.case_id}: raw schema identity mismatch: {mismatches}"
        )


def _validate_raw_identities(
    arrays: Mapping[str, np.ndarray],
    case: Any,
) -> None:
    # The G0c recorder deliberately reuses the exact 92-field G0 recorder.
    base._validate_raw_identities(arrays, case)
    n3_total = np.asarray(
        arrays["n3.booked_solver_accumulator_total_N"],
        dtype=np.float64,
    )
    n3_parts = np.asarray(
        arrays["n3.ds_booked_solver_accumulator_N"],
        dtype=np.float64,
    ) + np.asarray(
        arrays["n3.vortex_normal_booked_solver_accumulator_N"],
        dtype=np.float64,
    )
    if (
        _maximum_abs(n3_total - n3_parts)
        > witness.base.LEDGER_TOLERANCE_N
    ):
        raise ScoreContractError(
            f"{case.case_id}: N3 direct booked-channel identity failed"
        )


def _raw_case_metrics(
    arrays: Mapping[str, np.ndarray],
    case: Any,
) -> dict[str, Any]:
    inherited = base._raw_case_metrics(arrays, case)
    n3_half_body = np.asarray(
        arrays["n3.ds_booked_solver_accumulator_N"],
        dtype=np.float64,
    )
    q3_t, q3_l = _wind_vectors(2.0 * n3_half_body, case.aoa_deg)

    n1_half = np.asarray(
        arrays[
            "diagnostic.n1.leading_edge_suction_"
            "solver_accumulator_body_force_N"
        ],
        dtype=np.float64,
    )
    q1_t, q1_l = _wind_vectors(2.0 * n1_half, case.aoa_deg)
    q2_half = np.sum(
        np.asarray(
            arrays["n2.separation_panel_candidate_force_body_N"],
            dtype=np.float64,
        ),
        axis=1,
    )
    q2_t, q2_l = _wind_vectors(2.0 * q2_half, case.aoa_deg)
    q2_steps = np.column_stack((q2_t, q2_l))
    q2_norms = np.linalg.norm(q2_steps, axis=1)
    q2_unit = np.zeros_like(q2_steps)
    q2_active = q2_norms > NUMERIC_ZERO
    q2_unit[q2_active] = (
        q2_steps[q2_active] / q2_norms[q2_active, None]
    )

    def raw_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.asarray(
            [float(np.mean(first)), float(np.mean(second))],
            dtype=np.float64,
        )

    def robust_pair(
        first: np.ndarray, second: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            [_robust_mean(first), _robust_mean(second)],
            dtype=np.float64,
        )

    metrics = CaseMetrics(
        model_robust=np.asarray(
            inherited["model_robust"], dtype=np.float64
        ),
        model_raw=np.asarray(inherited["model_raw"], dtype=np.float64),
        q1_raw=-raw_pair(q1_t, q1_l),
        q1_robust=-robust_pair(q1_t, q1_l),
        q2_raw=raw_pair(q2_unit[:, 0], q2_unit[:, 1]),
        q2_robust=robust_pair(q2_unit[:, 0], q2_unit[:, 1]),
        # Withdrawal is the preregistered counterfactual direction.  N3 is
        # booked as a half-wing solver accumulator, hence the explicit 2x.
        q3_raw=-raw_pair(q3_t, q3_l),
        q3_robust=-robust_pair(q3_t, q3_l),
        q2_active_step_fraction=float(np.mean(q2_active)),
    )
    for name, value in metrics.as_mapping().items():
        if name != "q2_active_step_fraction" and not np.isfinite(value).all():
            raise ScoreContractError(
                f"{case.case_id}: non-finite raw-derived metric {name}"
            )
    return metrics.as_mapping()


def _runtime_claim_gates(
    manifest: Mapping[str, Any],
    claim_gates: Mapping[str, Any],
    case_id: str,
) -> None:
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list):
        raise ScoreContractError(f"{case_id}: manifest nodes missing")
    by_id = {
        item.get("id"): item for item in nodes if isinstance(item, Mapping)
    }
    for claim_id in ("N1", "N2", "N3"):
        expected = claim_gates[claim_id]
        node = by_id.get(claim_id)
        if (
            not isinstance(node, Mapping)
            or node.get("state") != expected["state"]
            or node.get("freeze") is not expected["freeze"]
            or node.get("implementation") != expected["implementation"]
            or node.get("runtime_role") != expected["runtime_role"]
        ):
            raise ScoreContractError(
                f"{case_id}: {claim_id} runtime semantic mismatch"
            )


def _load_case(
    run_dir: Path,
    campaign: Mapping[str, Any],
    case: Any,
    claim_gates: Mapping[str, Any],
) -> dict[str, Any]:
    record = campaign["cases"].get(case.case_id)
    if not isinstance(record, Mapping):
        raise ScoreContractError(f"missing case {case.case_id}")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "raw_npz",
        "schema_json",
        "evidence_json",
    }:
        raise ScoreContractError(f"{case.case_id}: artifact identity mismatch")

    # Each parser consumes the same immutable bytes whose digest was checked.
    raw_blob = _verify_artifact(run_dir, artifacts["raw_npz"])
    schema_blob = _verify_artifact(run_dir, artifacts["schema_json"])
    evidence_blob = _verify_artifact(run_dir, artifacts["evidence_json"])
    arrays = _npz_from_blob(raw_blob)
    schema = _json_from_blob(schema_blob)
    evidence = _json_from_blob(evidence_blob)
    _validate_raw_schema(arrays=arrays, schema=schema, case=case)
    _validate_raw_identities(arrays, case)

    expected_evidence = {
        "schema": witness.SCHEMA_VERSION,
        "campaign_schema": witness.SCHEMA_VERSION,
        "campaign_stage": witness.CAMPAIGN_STAGE,
        "stage": witness.RAW_STAGE,
        "campaign_scope": witness.CAMPAIGN_SCOPE,
        "witness_role": "read_only_g0c_supplementary_aoa_attribution",
        "production_grid_claim_allowed": False,
        "observer_role": "read_only",
        "aerodynamic_formula_modified": False,
        "force_added_by_runner": False,
    }
    mismatches = {
        name: {"expected": expected, "actual": evidence.get(name)}
        for name, expected in expected_evidence.items()
        if evidence.get(name) != expected
    }
    if mismatches:
        raise ScoreContractError(
            f"{case.case_id}: evidence role mismatch: {mismatches}"
        )
    if evidence.get("case_contract") != witness._jsonable(
        witness.asdict(case)
    ):
        raise ScoreContractError(f"{case.case_id}: case contract mismatch")
    contract = campaign.get("contract")
    if not isinstance(contract, Mapping):
        raise ScoreContractError(
            f"{case.case_id}: campaign contract missing"
        )
    case_hash = contract.get("case_contract_sha256")
    if (
        not _is_sha256(case_hash)
        or evidence.get("campaign_case_contract_sha256") != case_hash
    ):
        raise ScoreContractError(
            f"{case.case_id}: case contract hash mismatch"
        )
    if evidence.get("source_closure_sha256") != campaign.get(
        "source_closure_sha256"
    ):
        raise ScoreContractError(f"{case.case_id}: source identity mismatch")
    raw_guard = evidence.get("raw_guard")
    if not isinstance(raw_guard, Mapping) or raw_guard.get("passed") is not True:
        raise ScoreContractError(f"{case.case_id}: raw guard failed")
    guards = evidence.get("claim_guards")
    witness.base._validate_claim_guards(guards)

    array_hash = witness.base._array_bundle_hash(arrays)
    expected_array_hash = evidence.get("raw_array_bundle_sha256")
    if (
        not _is_sha256(expected_array_hash)
        or array_hash != expected_array_hash
        or schema.get("array_bundle_sha256") != expected_array_hash
        or record.get("raw_array_bundle_sha256") != expected_array_hash
    ):
        raise ScoreContractError(f"{case.case_id}: raw array hash mismatch")

    manifest = evidence.get("claim_manifest")
    if not isinstance(manifest, Mapping) or manifest.get("guards") != guards:
        raise ScoreContractError(
            f"{case.case_id}: claim manifest/guard mismatch"
        )
    manifest_sha = evidence.get("claim_manifest_sha256")
    if (
        not _is_sha256(manifest_sha)
        or witness._canonical_hash(manifest) != manifest_sha
    ):
        raise ScoreContractError(
            f"{case.case_id}: claim manifest hash mismatch"
        )
    graph_identity = evidence.get("claim_graph_identity_sha256")
    if (
        not _is_sha256(graph_identity)
        or witness.base._claim_graph_identity_sha256(manifest)
        != graph_identity
        or record.get("claim_graph_identity_sha256") != graph_identity
        or campaign.get("common_claim_graph_identity_sha256")
        != graph_identity
    ):
        raise ScoreContractError(
            f"{case.case_id}: claim graph identity mismatch"
        )
    _runtime_claim_gates(manifest, claim_gates, case.case_id)

    execution_binding = campaign.get("execution_binding")
    if not isinstance(execution_binding, Mapping):
        raise ScoreContractError(
            f"{case.case_id}: execution binding is missing"
        )
    if (
        evidence.get("execution_binding_sha256")
        != execution_binding.get("binding_sha256")
        or evidence.get("resolved_call")
        != execution_binding.get("resolved_calls", {}).get(case.case_id)
    ):
        raise ScoreContractError(
            f"{case.case_id}: bound solver call mismatch"
        )
    raw_config = evidence.get("claim_raw_config")
    expected_config = witness._expected_claim_raw_config(case)
    if raw_config != expected_config:
        raise ScoreContractError(f"{case.case_id}: raw config mismatch")
    return _raw_case_metrics(arrays, case)


def _endpoint_value(
    curve: benchmark.MeasurementCurve,
    nominal_frequency: float,
) -> tuple[float, dict[str, Any]]:
    return base._endpoint_value(curve, nominal_frequency)


def _experimental_vectors(
    campaign: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_blob = _read_once(DATA_SOURCE.resolve(strict=True))
    relative = str(DATA_SOURCE.relative_to(ROOT))
    members = campaign.get("source_closure", {}).get("members", {})
    if (
        data_blob.sha256 != DATA_SOURCE_SHA256
        or not isinstance(members, Mapping)
        or members.get(relative) != data_blob.sha256
    ):
        raise ScoreContractError("frozen Fig19 data.md identity mismatch")
    measurements = benchmark.load_measurements(
        DATA_SOURCE,
        source_bytes=data_blob.data,
    )
    validation = benchmark.validate_measurement_contract(
        measurements,
        source_path=DATA_SOURCE,
        source_bytes=data_blob.data,
    )
    if (
        validation.get("passed") is not True
        or validation.get("source", {}).get(
            "parsed_from_verified_bytes"
        )
        is not True
    ):
        raise ScoreContractError(
            f"invalid Fig17/18/19 measurement contract: {validation}"
        )

    values: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for case_id, (frequency, aoa) in EXPECTED_CASES.items():
        thrust, thrust_meta = _endpoint_value(
            measurements[f"19|a|{aoa:g}"], frequency
        )
        lift, lift_meta = _endpoint_value(
            measurements[f"19|b|{aoa:g}"], frequency
        )
        values[case_id] = np.asarray([thrust, lift], dtype=np.float64)
        provenance[case_id] = {
            "T": thrust_meta,
            "L": lift_meta,
            "force_interpolated": False,
        }
    return values, {
        "measurement_validation": validation,
        "source": relative,
        "source_sha256": data_blob.sha256,
        "endpoints": provenance,
        "force_interpolation_used": False,
    }


def _template_relation(
    residual: np.ndarray,
    template: np.ndarray,
) -> str:
    return base._template_relation(residual, template)


def _template_identifiability(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    columns: list[np.ndarray] = []
    norms: dict[str, float] = {}
    for template in TEMPLATE_KEYS:
        values: list[float] = []
        field = f"{template}_TL"
        for row in rows:
            residual = np.asarray(row["residual_TL_N"], dtype=np.float64)
            direction = np.asarray(row[field], dtype=np.float64)
            material = np.abs(residual) > TAU_CONTRAST_N
            values.extend(direction[material].tolist())
        column = np.asarray(values, dtype=np.float64)
        norm = float(np.linalg.norm(column))
        norms[template] = norm
        if norm <= NUMERIC_ZERO:
            return {
                "passed": False,
                "reason": f"inactive_template_column:{template}",
                "column_norms": norms,
                "condition_number_2": math.inf,
                "limit": COLLINEAR_CONDITION_LIMIT,
            }
        columns.append(column / norm)
    matrix = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = (
        float(np.linalg.cond(matrix))
        if matrix.shape[0] >= len(TEMPLATE_KEYS)
        and rank == len(TEMPLATE_KEYS)
        else math.inf
    )
    passed = bool(
        np.isfinite(condition)
        and condition <= COLLINEAR_CONDITION_LIMIT
    )
    return {
        "passed": passed,
        "reason": (
            "normalized_material_axis_template_matrix"
            if passed
            else "condition_number_exceeded"
        ),
        "column_norms": norms,
        "condition_number_2": condition,
        "limit": COLLINEAR_CONDITION_LIMIT,
        "material_axis_count": int(matrix.shape[0]),
        "template_count": int(matrix.shape[1]),
        "matrix_rank": rank,
    }


def _classify(
    *,
    experiment: Mapping[str, np.ndarray],
    cases: Mapping[str, Mapping[str, np.ndarray]],
    model_key: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for contrast_id, context, high, low in CONTRASTS:
        residual = (
            experiment[high]
            - cases[high][model_key]
            - experiment[low]
            + cases[low][model_key]
        )
        view = model_key.removeprefix("model_")
        if view not in {"raw", "robust"}:
            raise ScoreContractError(f"unknown processing view {model_key}")
        templates = {
            template: (
                cases[high][f"{template}_{view}"]
                - cases[low][f"{template}_{view}"]
            )
            for template in TEMPLATE_KEYS
        }
        relations = {
            template: _template_relation(residual, value)
            for template, value in templates.items()
        }
        rows.append(
            {
                "id": contrast_id,
                "context": context,
                "high": high,
                "low": low,
                "residual_TL_N": np.asarray(
                    residual, dtype=np.float64
                ).tolist(),
                "material_axes": [
                    axis
                    for axis, value in zip(("T", "L"), residual)
                    if abs(value) > TAU_CONTRAST_N
                ],
                **{
                    f"{template}_TL": np.asarray(
                        value, dtype=np.float64
                    ).tolist()
                    for template, value in templates.items()
                },
                **{
                    f"{template}_relation": relation
                    for template, relation in relations.items()
                },
                "supporting_templates": [
                    template
                    for template, relation in relations.items()
                    if relation == "supports"
                ],
            }
        )

    material_rows = [row for row in rows if row["material_axes"]]
    material_contexts = {
        row["context"] for row in material_rows
    }
    required_contexts = {context for _, context, _, _ in CONTRASTS}
    if material_contexts != required_contexts:
        return {
            "status": "NO_DECISION_OFFSET_ONLY",
            "contrasts": rows,
            "identifiability": None,
            "selected_template": None,
        }

    if any(len(row["supporting_templates"]) > 1 for row in material_rows):
        return {
            "status": "NO_DECISION_MULTIPLE_EXPLANATIONS",
            "contrasts": rows,
            "identifiability": _template_identifiability(material_rows),
            "selected_template": None,
        }

    identifiability = _template_identifiability(material_rows)
    if not identifiability["passed"]:
        return {
            "status": "NO_DECISION_COLLINEAR",
            "contrasts": rows,
            "identifiability": identifiability,
            "selected_template": None,
        }

    candidates: list[str] = []
    candidate_audit: dict[str, Any] = {}
    contexts = required_contexts
    for template in TEMPLATE_KEYS:
        competitors = [item for item in TEMPLATE_KEYS if item != template]
        context_audit: dict[str, Any] = {}
        for context in sorted(contexts):
            context_rows = [
                row for row in material_rows
                if row["context"] == context
            ]
            selected_support = any(
                row["supporting_templates"] == [template]
                for row in context_rows
            )
            selected_never_opposes = all(
                row[f"{template}_relation"] != "opposes"
                for row in context_rows
            )
            competitor_support_absent = {
                competitor: all(
                    row[f"{competitor}_relation"] != "supports"
                    for row in context_rows
                )
                for competitor in competitors
            }
            competitor_reverse = {
                competitor: any(
                    row[f"{competitor}_relation"] == "opposes"
                    for row in context_rows
                )
                for competitor in competitors
            }
            context_audit[context] = {
                "selected_unique_support": selected_support,
                "selected_never_opposes": selected_never_opposes,
                "competitor_support_absent": competitor_support_absent,
                "competitor_reverse_evidence": competitor_reverse,
                "passed": (
                    selected_support
                    and selected_never_opposes
                    and all(competitor_support_absent.values())
                    and all(competitor_reverse.values())
                ),
            }
        passed = all(
            audit["passed"] for audit in context_audit.values()
        )
        candidate_audit[template] = {
            "contexts": context_audit,
            "passed": passed,
        }
        if passed:
            candidates.append(template)

    if len(candidates) == 1:
        selected = candidates[0]
        status = ACTIVE_STATUS[selected]
    elif len(candidates) > 1:
        selected = None
        status = "NO_DECISION_MULTIPLE_EXPLANATIONS"
    else:
        selected = None
        unique_by_context = {
            context: [
                template
                for template in TEMPLATE_KEYS
                if candidate_audit[template]["contexts"][context][
                    "passed"
                ]
            ]
            for context in sorted(contexts)
        }
        if (
            all(len(values) == 1 for values in unique_by_context.values())
            and len({values[0] for values in unique_by_context.values()})
            > 1
        ):
            status = "NO_DECISION_FREQUENCY_DEPENDENT_MIXED_SOURCE"
        else:
            status = "NO_DECISION_INSUFFICIENT_UNIQUENESS"
    return {
        "status": status,
        "contrasts": rows,
        "identifiability": identifiability,
        "selected_template": selected,
        "candidate_audit": candidate_audit,
    }


def _claim_yaml(
    path: Path,
    campaign: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    return base._claim_yaml(path, campaign)


def _require_claim(
    root: Mapping[str, Any],
    claim_id: str,
    *,
    state: str | tuple[str, ...],
    freeze: bool,
) -> Mapping[str, Any]:
    return base._require_claim(
        root, claim_id, state=state, freeze=freeze
    )


def _semantic_claim_gates(
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    n1_root, n1_sha = _claim_yaml(N1_YAML, campaign)
    n2_root, n2_sha = _claim_yaml(N2_YAML, campaign)
    n3_root, n3_sha = _claim_yaml(N3_YAML, campaign)
    n1 = _require_claim(n1_root, "N1", state="validated", freeze=True)
    n2 = _require_claim(n2_root, "N2", state="partial", freeze=False)
    n22 = _require_claim(
        n2_root, "N2.2", state="falsified", freeze=False
    )
    n25 = _require_claim(n2_root, "N2.5", state="open", freeze=False)
    n26 = _require_claim(
        n2_root, "N2.6", state=("partial", "open"), freeze=False
    )
    n3 = _require_claim(n3_root, "N3", state="partial", freeze=False)
    n310 = _require_claim(
        n3_root, "N3.1.0", state="falsified", freeze=False
    )
    n31h = _require_claim(
        n3_root, "N3.1h", state="partial", freeze=False
    )
    n31i = _require_claim(
        n3_root, "N3.1i", state="open", freeze=False
    )

    runtime_expected = {
        "N1": (
            n1,
            n1_sha,
            "claim_runtime.components:UVLMComponent",
        ),
        "N2": (
            n2,
            n2_sha,
            "claim_runtime.components:KirchhoffLBComponent",
        ),
        "N3": (
            n3,
            n3_sha,
            "claim_runtime.components:DSVortexComponent",
        ),
    }
    runtime_gates: dict[str, Any] = {}
    for claim_id, (claim, yaml_sha, implementation) in (
        runtime_expected.items()
    ):
        if (
            claim.get("implementation") != implementation
            or claim.get("runtime_role") != "physics"
            or "v41" not in claim.get("enabled_in", [])
        ):
            raise ScoreContractError(
                f"{claim_id} runtime semantic gate failed"
            )
        runtime_gates[claim_id] = {
            "id": claim_id,
            "state": claim["state"],
            "freeze": claim["freeze"],
            "implementation": implementation,
            "runtime_role": "physics",
            "yaml_sha256": yaml_sha,
            "passed": True,
        }
    if "implementation" in n25:
        raise ScoreContractError(
            "N2.5 unexpectedly acquired an executable implementation"
        )

    prereg_blob = _read_once(witness.PREREG.resolve(strict=True))
    prereg_relative = str(witness.PREREG.relative_to(ROOT))
    prereg_identity = campaign.get("preregistration")
    members = campaign.get("source_closure", {}).get("members", {})
    if (
        prereg_blob.sha256 != PREREG_SHA256
        or not isinstance(prereg_identity, Mapping)
        or prereg_identity.get("path") != prereg_relative
        or prereg_identity.get("sha256") != prereg_blob.sha256
        or not isinstance(members, Mapping)
        or members.get(prereg_relative) != prereg_blob.sha256
    ):
        raise ScoreContractError("frozen preregistration identity mismatch")

    return {
        **runtime_gates,
        "N2.2": {
            "state": n22["state"],
            "freeze": n22["freeze"],
            "passed": True,
        },
        "N2.5": {
            "state": n25["state"],
            "freeze": n25["freeze"],
            "candidate_implementation_authorized": False,
            "passed": True,
        },
        "N2.6": {
            "state": n26["state"],
            "freeze": n26["freeze"],
            "movable": True,
            "passed": True,
        },
        "N3.1.0": {
            "state": n310["state"],
            "freeze": n310["freeze"],
            "reactivation_authorized": False,
            "passed": True,
        },
        "N3.1h": {
            "state": n31h["state"],
            "freeze": n31h["freeze"],
            "passed": True,
        },
        "N3.1i": {
            "state": n31i["state"],
            "freeze": n31i["freeze"],
            "passed": True,
        },
        "preregistration_sha256": prereg_blob.sha256,
    }


def _validate_preconditioner(
    preconditioner: Any,
    *,
    graph_identity: str,
) -> None:
    if not isinstance(preconditioner, Mapping):
        raise ScoreContractError("session preconditioner is missing")
    expected_case = witness._jsonable(witness.asdict(witness.PRECONDITIONER_CASE))
    if (
        preconditioner.get("excluded_from_scientific_metrics") is not True
        or preconditioner.get("case_contract") != expected_case
        or preconditioner.get("claim_graph_identity_sha256")
        != graph_identity
        or not isinstance(preconditioner.get("purpose"), str)
    ):
        raise ScoreContractError("preconditioner identity mismatch")
    for name in ("L_wind_N", "T_wind_N", "wall_s"):
        if not base._finite_number(
            preconditioner.get(name), nonnegative=(name == "wall_s")
        ):
            raise ScoreContractError(
                f"invalid preconditioner numeric field: {name}"
            )
    witness.base._validate_claim_guards(preconditioner.get("claim_guards"))
    resolved = preconditioner.get("resolved_call")
    expected_call = {
        "closure": "v41",
        "nc": RAW_NC,
        "ns": RAW_NS,
        "n_cycle": 2,
        "steps_per_cycle": RAW_STEPS,
        "wake_rows": RAW_STEPS,
        "U": 8.0,
        "freq": 2.6,
        "twist_amp_deg": 0.0,
        "aoa_deg": 5.0,
        "twist_phase_deg": -90.0,
    }
    if not isinstance(resolved, Mapping) or any(
        resolved.get(name) != expected
        for name, expected in expected_call.items()
    ):
        raise ScoreContractError("preconditioner solver call mismatch")


def _validate_execution_binding(
    campaign: Mapping[str, Any],
    cases: Sequence[Any],
) -> Mapping[str, Any]:
    binding = campaign.get("execution_binding")
    required = {
        "schema",
        "source_closure_sha256",
        "entry_modules",
        "loaded_governed_modules",
        "solver_callable",
        "base_config",
        "base_config_sha256",
        "resolved_calls",
        "binding_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != required:
        raise ScoreContractError("execution binding field identity mismatch")
    payload = dict(binding)
    binding_sha256 = payload.pop("binding_sha256")
    if (
        binding.get("schema") != "g0c-execution-binding-v1"
        or not _is_sha256(binding_sha256)
        or witness._canonical_hash(payload) != binding_sha256
        or campaign.get("execution_binding_sha256") != binding_sha256
        or binding.get("source_closure_sha256")
        != campaign.get("source_closure_sha256")
    ):
        raise ScoreContractError("execution binding hash mismatch")

    closure = campaign.get("source_closure")
    members = (
        closure.get("members")
        if isinstance(closure, Mapping)
        else None
    )
    if not isinstance(members, Mapping):
        raise ScoreContractError("execution source members are missing")

    loaded = binding.get("loaded_governed_modules")
    entries = binding.get("entry_modules")
    if not isinstance(loaded, Mapping) or not isinstance(entries, Mapping):
        raise ScoreContractError("execution module binding is malformed")
    if set(entries) != {"_v2_robo", "lb_sweep118"}:
        raise ScoreContractError("execution entry-module identity mismatch")
    for module_name, identity in loaded.items():
        required_identity = {
            "module",
            "path",
            "sha256",
            "required_root",
            "relative_path",
        }
        if (
            not isinstance(module_name, str)
            or not isinstance(identity, Mapping)
            or set(identity) != required_identity
            or identity.get("module") != module_name
            or identity.get("required_root") != str(ROOT.resolve())
            or not isinstance(identity.get("relative_path"), str)
            or Path(identity.get("path", ""))
            != (ROOT / identity["relative_path"]).resolve()
            or identity.get("sha256")
            != members.get(identity["relative_path"])
        ):
            raise ScoreContractError(
                f"execution module identity mismatch: {module_name}"
            )
    if any(entries[name] != loaded.get(name) for name in entries):
        raise ScoreContractError("execution entry module is not governed")

    if binding.get("solver_callable") != {
        "module": "_v2_robo",
        "qualname": "gpu_run_twist",
    }:
        raise ScoreContractError("solver callable identity mismatch")
    base_config = binding.get("base_config")
    if (
        not isinstance(base_config, Mapping)
        or not _is_sha256(binding.get("base_config_sha256"))
        or witness._canonical_hash(base_config)
        != binding["base_config_sha256"]
    ):
        raise ScoreContractError("bound BASE configuration mismatch")

    resolved_calls = binding.get("resolved_calls")
    expected_ids = {
        witness.PRECONDITIONER_CASE.case_id,
        *(case.case_id for case in cases),
    }
    if (
        not isinstance(resolved_calls, Mapping)
        or set(resolved_calls) != expected_ids
        or any(
            not isinstance(value, Mapping)
            for value in resolved_calls.values()
        )
    ):
        raise ScoreContractError("bound resolved-call identity mismatch")
    return binding


def _validate_campaign_structure(
    campaign: Mapping[str, Any],
    cases: Sequence[Any],
) -> None:
    expected_ids = {case.case_id for case in cases}
    records = campaign.get("cases")
    if not isinstance(records, Mapping) or set(records) != expected_ids:
        raise ScoreContractError("campaign case identity set mismatch")
    if campaign.get("completed_case_count") != 8:
        raise ScoreContractError("campaign completed-case count mismatch")
    graph_identity = campaign.get("common_claim_graph_identity_sha256")
    if not _is_sha256(graph_identity):
        raise ScoreContractError("common claim graph identity is malformed")
    source_identity = campaign.get("source_closure_sha256")
    source_closure = campaign.get("source_closure")
    if (
        not _is_sha256(source_identity)
        or not isinstance(source_closure, Mapping)
        or source_closure.get("members_sha256") != source_identity
    ):
        raise ScoreContractError("source closure identity is malformed")
    execution_binding = _validate_execution_binding(campaign, cases)
    base._validate_numeric_runtime(campaign.get("numeric_runtime"))
    runtime = campaign["numeric_runtime"]

    sessions = campaign.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ScoreContractError("campaign sessions are missing")
    completed: list[str] = []
    reference: Mapping[str, Any] | None = None
    for session in sessions:
        if (
            not isinstance(session, Mapping)
            or session.get("numeric_runtime") != runtime
            or session.get("source_closure_sha256") != source_identity
            or session.get("execution_binding") != execution_binding
            or session.get("execution_binding_sha256")
            != execution_binding["binding_sha256"]
        ):
            raise ScoreContractError("session runtime identity mismatch")
        preconditioner = session.get("preconditioner")
        _validate_preconditioner(
            preconditioner, graph_identity=graph_identity
        )
        assert isinstance(preconditioner, Mapping)
        if (
            preconditioner.get("execution_binding_sha256")
            != execution_binding["binding_sha256"]
            or preconditioner.get("resolved_call")
            != execution_binding["resolved_calls"][
                witness.PRECONDITIONER_CASE.case_id
            ]
        ):
            raise ScoreContractError(
                "preconditioner execution binding mismatch"
            )
        if reference is None:
            reference = preconditioner
        else:
            deltas = {
                name: abs(float(preconditioner[name]) - float(reference[name]))
                for name in ("L_wind_N", "T_wind_N")
            }
            if any(delta > TAU_F_N for delta in deltas.values()):
                raise ScoreContractError(
                    "cross-session preconditioner force drift: "
                    f"{deltas}, tolerance={TAU_F_N}"
                )
        case_ids = session.get("completed_case_ids")
        if (
            not isinstance(case_ids, list)
            or any(not isinstance(item, str) for item in case_ids)
        ):
            raise ScoreContractError("session completed cases malformed")
        completed.extend(case_ids)
    if len(completed) != len(set(completed)) or set(completed) != expected_ids:
        raise ScoreContractError("session completed-case identity mismatch")


def _validate_kinematic_identity_gate(
    campaign: Mapping[str, Any],
    cases: Sequence[Any],
) -> None:
    source_closure, n5_bytes = witness._source_closure_snapshot()
    if campaign.get("source_closure") != source_closure:
        raise ScoreContractError(
            "current scorer/source closure drift during N5.1c validation"
        )
    n5_relative = str(
        witness.N5_YAML.resolve().relative_to(witness.ROOT.resolve())
    )
    n5_sha = source_closure["members"].get(n5_relative)
    expected = witness._kinematic_identity_gate(
        cases,
        n5_yaml_bytes=n5_bytes,
        expected_n5_yaml_sha256=n5_sha,
        source_closure_sha256=source_closure["members_sha256"],
    )
    if campaign.get("kinematic_identity_gate") != expected:
        raise ScoreContractError("N5.1c kinematic identity gate mismatch")


def _final_status(
    robust: Mapping[str, Any],
    raw: Mapping[str, Any],
    agreement: Mapping[str, Any] | None = None,
) -> str:
    if (
        raw.get("status") != robust.get("status")
        or (agreement is not None and agreement.get("passed") is not True)
    ):
        return "NO_DECISION_PROCESSING_SENSITIVE"
    return str(robust["status"])


def _processing_agreement(
    *,
    cases: Mapping[str, Mapping[str, Any]],
    robust: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    case_deltas: dict[str, dict[str, float]] = {}
    for case_id, metrics in cases.items():
        case_deltas[case_id] = {
            name: _maximum_abs(
                np.asarray(metrics[f"{name}_robust"], dtype=np.float64)
                - np.asarray(metrics[f"{name}_raw"], dtype=np.float64)
            )
            for name in ("model", *TEMPLATE_KEYS)
        }
    case_force_gate_passed = all(
        delta <= TAU_F_N
        for values in case_deltas.values()
        for delta in values.values()
    )

    robust_rows = {
        row["id"]: row for row in robust.get("contrasts", [])
    }
    raw_rows = {row["id"]: row for row in raw.get("contrasts", [])}
    contrast_identity = set(robust_rows) == set(raw_rows)
    contrast_deltas: dict[str, dict[str, float]] = {}
    discrete_agreement = contrast_identity
    if contrast_identity:
        for contrast_id in sorted(robust_rows):
            robust_row = robust_rows[contrast_id]
            raw_row = raw_rows[contrast_id]
            contrast_deltas[contrast_id] = {
                name: _maximum_abs(
                    np.asarray(
                        robust_row[
                            "residual_TL_N"
                            if name == "residual"
                            else f"{name}_TL"
                        ],
                        dtype=np.float64,
                    )
                    - np.asarray(
                        raw_row[
                            "residual_TL_N"
                            if name == "residual"
                            else f"{name}_TL"
                        ],
                        dtype=np.float64,
                    )
                )
                for name in ("residual", *TEMPLATE_KEYS)
            }
            robust_discrete = {
                "material_axes": robust_row["material_axes"],
                "supporting_templates": robust_row[
                    "supporting_templates"
                ],
                **{
                    f"{template}_relation": robust_row[
                        f"{template}_relation"
                    ]
                    for template in TEMPLATE_KEYS
                },
            }
            raw_discrete = {
                "material_axes": raw_row["material_axes"],
                "supporting_templates": raw_row[
                    "supporting_templates"
                ],
                **{
                    f"{template}_relation": raw_row[
                        f"{template}_relation"
                    ]
                    for template in TEMPLATE_KEYS
                },
            }
            discrete_agreement = (
                discrete_agreement
                and robust_discrete == raw_discrete
            )
    contrast_force_gate_passed = all(
        delta <= TAU_F_N
        for values in contrast_deltas.values()
        for delta in values.values()
    )
    decision_agreement = (
        robust.get("status") == raw.get("status")
        and robust.get("selected_template")
        == raw.get("selected_template")
        and bool(
            robust.get("identifiability", {}).get("passed")
            if robust.get("identifiability") is not None
            else True
        )
        == bool(
            raw.get("identifiability", {}).get("passed")
            if raw.get("identifiability") is not None
            else True
        )
    )
    passed = (
        case_force_gate_passed
        and contrast_force_gate_passed
        and discrete_agreement
        and decision_agreement
    )
    return {
        "passed": passed,
        "tau_F": TAU_F_N,
        "case_view_max_abs_deltas": case_deltas,
        "case_force_gate_passed": case_force_gate_passed,
        "contrast_view_max_abs_deltas": contrast_deltas,
        "contrast_force_gate_passed": contrast_force_gate_passed,
        "material_relation_and_support_agreement": discrete_agreement,
        "decision_agreement": decision_agreement,
    }


def score(run_dir: Path) -> dict[str, Any]:
    cases_contract = _case_contracts()
    manifest_blob = _read_confined(run_dir, "run_manifest.json")
    campaign = _json_from_blob(manifest_blob)
    if (
        campaign.get("schema") != witness.SCHEMA_VERSION
        or campaign.get("campaign_stage") != witness.CAMPAIGN_STAGE
        or campaign.get("status") != "complete"
        or campaign.get("contract")
        != witness._campaign_contract(cases_contract)
    ):
        raise ScoreContractError("campaign is not a complete frozen G0c run")
    if campaign.get("source_closure") != witness._source_closure():
        raise ScoreContractError("current scorer/source closure drift")
    _validate_campaign_structure(campaign, cases_contract)
    _validate_kinematic_identity_gate(campaign, cases_contract)
    claim_gates = _semantic_claim_gates(campaign)

    cases = {
        case.case_id: _load_case(
            run_dir, campaign, case, claim_gates
        )
        for case in cases_contract
    }
    if set(cases) != set(EXPECTED_CASES):
        raise ScoreContractError("loaded case identity set mismatch")
    experiment, experiment_provenance = _experimental_vectors(campaign)
    robust = _classify(
        experiment=experiment,
        cases=cases,
        model_key="model_robust",
    )
    raw = _classify(
        experiment=experiment,
        cases=cases,
        model_key="model_raw",
    )
    processing_agreement = _processing_agreement(
        cases=cases,
        robust=robust,
        raw=raw,
    )
    status = _final_status(robust, raw, processing_agreement)

    return {
        "schema": SCHEMA,
        "status": status,
        "claim_state_modified": False,
        "candidate_implementation_authorized": False,
        "n2p5_candidate_implementation_authorized": False,
        "falsified_candidate_reactivation_authorized": False,
        "next_n2p6_shadow_preregistration_authorized": (
            status == "ACTIVE_N2_6_SHADOW_PREREG_ALLOWED"
        ),
        "next_n3_aoa_state_shadow_preregistration_authorized": (
            status == "ACTIVE_N3_SPATIAL_STATE_AUDIT_REQUIRED"
        ),
        "n1_ledger_audit_required": (
            status == "N1_LEDGER_AUDIT_REQUIRED"
        ),
        "n1_remains_validated_frozen": True,
        "claim_semantic_gates": claim_gates,
        "thresholds": {
            "tau_F_N": TAU_F_N,
            "tau_contrast_N_strict": TAU_CONTRAST_N,
            "numeric_zero": NUMERIC_ZERO,
            "condition_number_2_limit": COLLINEAR_CONDITION_LIMIT,
        },
        "robust": robust,
        "raw": raw,
        "processing_agreement": processing_agreement,
        "experiment": experiment_provenance,
        "templates": TEMPLATE_LABELS,
        "provenance": {
            "run_manifest": {
                "path": str(manifest_blob.path),
                "sha256": manifest_blob.sha256,
            },
            "preregistration": {
                "path": str(witness.PREREG.relative_to(ROOT)),
                "sha256": claim_gates["preregistration_sha256"],
            },
            "source_closure_sha256": campaign[
                "source_closure_sha256"
            ],
            "phase_identity_used_for_claim_attribution_deg": -90.0,
            "force_interpolation_used": False,
            "runner_diagnostic_summary_used": False,
        },
    }


def _invalid_report(exc: Exception) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "INVALID_EVIDENCE",
        "error": f"{type(exc).__name__}: {exc}",
        "claim_state_modified": False,
        "candidate_implementation_authorized": False,
        "n2p5_candidate_implementation_authorized": False,
        "falsified_candidate_reactivation_authorized": False,
        "next_n2p6_shadow_preregistration_authorized": False,
        "next_n3_aoa_state_shadow_preregistration_authorized": False,
        "n1_ledger_audit_required": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = score(args.run)
        exit_code = 0
    except Exception as exc:
        report = _invalid_report(exc)
        exit_code = 2
    _write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
