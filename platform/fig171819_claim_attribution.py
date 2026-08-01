"""Pre-registered disease selection and parent attribution for Fig. 17/18/19.

The module has three deliberately separated stages:

``select-disease``
    Uses only the frozen confirmed42 residual fingerprint.
``prepare``
    Independently reconstructs and freezes the selected disease, including
    every positive-support contrast and every all-pairs guard.  It has no
    contribution input and never dereferences the contribution artifact.
``evaluate``
    Validates the complete fresh151 force ledger, then performs a frozen-state
    leave-one-parent-force-out attribution for N2 and N3.

All scientific outputs remain hypotheses.  Nothing in this module authorizes
claim-YAML or aerodynamic-model modification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fig171819_benchmark as benchmark
import fig171819_confirmed_compare as confirmed_compare
import fig171819_residual_fingerprint as residual_fingerprint
from fig171819_benchmark import (
    CONDITIONS_BY_EVIDENCE_SCOPE,
    EVIDENCE_CONFIRMED,
    condition_key,
)


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
PARENT_PROTOCOLS = (
    PLATFORM / "docs" / "diag" / "fig171819_parent_attribution_protocol_v2_20260729.md",
    PLATFORM / "docs" / "diag" / "fig171819_parent_attribution_protocol_v3_20260729.md",
    PLATFORM / "docs" / "diag" / "fig171819_parent_attribution_protocol_v4_20260729.md",
    PLATFORM / "docs" / "diag" / "fig171819_parent_attribution_protocol_v5_20260729.md",
)
ACTIVE_DISEASE_PROTOCOLS = tuple(
    PLATFORM
    / "docs"
    / "diag"
    / f"fig171819_active_disease_prereg_v{version}_20260729.md"
    for version in range(3, 9)
)
PROTOCOL = PARENT_PROTOCOLS[-1]
ACTIVE_DISEASE_PROTOCOL = ACTIVE_DISEASE_PROTOCOLS[-1]
CONFIRMED_COMPARE_SOURCE = PLATFORM / "fig171819_confirmed_compare.py"
AUTHORIZED_POSTPROCESS_AUTHORIZATION = (
    PLATFORM / "docs" / "diag" / "v41_fresh151_postprocess_authorization_20260729.json"
)
AUTHORIZED_POSTPROCESS_PREREG = (
    PLATFORM / "docs" / "diag" / "v41_fresh151_postprocess_prereg_20260729.md"
)
EVIDENCE_SCOPE_PREREG = (
    PLATFORM / "docs" / "diag" / "fig171819_evidence_scope_prereg_20260729.md"
)

# 2026-08-01 双 scope 契约终裁同步: confirmed_compare.py 因 benchmark 哈希
# 更新而修改,此为终裁后版本。
AUTHORIZED_CONFIRMED_COMPARE_SHA256 = (
    "d2752adfbf92f39904a074afbdebcea2d3dae181841e7511beac3da03b1a0ec6"
)
AUTHORIZED_PARENT_PROTOCOL_SHA256 = (
    "c6a5ddd76e15a8d275dd3bee4dc94b23968e871c2dbff77c03842270b7d429ab"
)
AUTHORIZED_ACTIVE_DISEASE_PROTOCOL_SHA256 = (
    "a9c306c50910fdc7fa1a6e36cb1132921ce601c02d2a52aecab2ccb26734f3ab"
)
AUTHORIZED_PARENT_PROTOCOL_SHA256_BY_VERSION = {
    2: "ca61e3e58dbc13c5f09a2cc55ce662e4b0c733ddeb9dd9c795e97e9f7619f734",
    3: "f1e1cd98105d21c88da580136adf3ae088de75f53319e883f2c47cb33f1ca1a1",
    4: "71cbd1c1b129d35656d02e107ca39550c7a022b18dc96f33cd25576e1109526a",
    5: "c6a5ddd76e15a8d275dd3bee4dc94b23968e871c2dbff77c03842270b7d429ab",
}
AUTHORIZED_ACTIVE_PROTOCOL_SHA256_BY_VERSION = {
    3: "8faec5de90cf563146b9950c1ca2a5200597623087595ae481f563d2c32f7f0f",
    4: "65ab9595db6b71d0e2fb6eee085dbf87c0d9c7eca0aaf0a4c6f6fae9d623c6bc",
    5: "174d0f18d33035278a46d27c04c68244183762c9a65058702fb2685f01b69940",
    6: "b2ea263cc786791a46de194ee538d0354d294187c2d1262848bf98f2105f78c7",
    7: "9cf9d345f6d12bdbbb45baa0d927f085718b002f769cff6c4220cce82f1c8815",
    8: "a9c306c50910fdc7fa1a6e36cb1132921ce601c02d2a52aecab2ccb26734f3ab",
}
AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256 = (
    "22ca928d5240ed6195fefe9da1f48c121102712f1f39681d813f90646b8d3cab"
)
AUTHORIZED_POSTPROCESS_PREREG_SHA256 = (
    "7e4ae6135ad624f19d2d8aa27b4ff692fba0cb592feb9581b7a7f480f9433d60"
)
AUTHORIZED_EVIDENCE_SCOPE_PREREG_SHA256 = (
    "c7cbf19f5cd388090ae85fb3e350b70a32beb043fbda04dc958ee055543a8a69"
)
AUTHORIZED_MEASUREMENT_DATA_SHA256 = (
    "ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1"
)
# 2026-08-01 双 scope 契约终裁(图19(c,d) 频率身份一手资产穷尽 UNRESOLVED →
# confirmed 晋升域 / conditional_fig19_cd 诊断域): benchmark 源因
# FIG19_CD_FREQUENCY_STATUS='conditional_scope' 修改而漂移,此哈希为终裁后版本。
AUTHORIZED_BENCHMARK_SOURCE_SHA256 = (
    "45b93584550eea4b16969381c59fe1038f68b21cf7cd1371052a19315c2444da"
)
AUTHORIZED_FINGERPRINT_SOURCE_SHA256 = (
    "127db39b6028f1be676a10f95dc932f35e29402fd590dc979d97e269f4bc14e8"
)
AUTHORIZED_PHYSICAL_FAMILY_CONTRACT_SHA256 = (
    "b667d78b305e590c0382ef5078e6bff58ba3466315d0d6d7de8c49532e6d1e55"
)
AUTHORIZED_GRAPH_IDENTITY_SHA256 = (
    "8b58815e1b50d0adb317efaae8c0db3d8506b199c5a3a6e781818a928f50c7cc"
)

EXPECTED_CURVES = 42
EXPECTED_SAMPLES = 434
EXPECTED_CONDITIONS = 151
EXPECTED_PHYSICAL_FAMILIES = 34
EXPECTED_ALIAS_GROUPS = 8
EXPECTED_CONDITIONAL_CURVES = frozenset(
    f"19|{panel}|{aoa:g}" for panel in ("c", "d") for aoa in (0.0, 5.0, 10.0, 15.0)
)
EXPECTED_CONTRIBUTION_NODES = frozenset(("N1", "N2", "N3", "N4", "N6", "R0"))
EXPECTED_GRAPH_NODES = ("N1", "N2", "N3", "N4", "N5", "N6", "R0")
ELIGIBLE_PARENTS = ("N2", "N3")
FORCE_TOLERANCE_N = 0.15
CONTRAST_TOLERANCE_N = 0.30
RANK_MARGIN_N = 0.15
LEDGER_TOLERANCE_N = 1.0e-9
CANONICAL_ZERO_TOLERANCE = 1.0e-12
MIN_INDEPENDENT_FAMILIES = 2
SCHEMA_VERSION = 3

CONTRIBUTION_INVENTORY = {
    "N1": (
        ("uvlm", "physics"),
        ("vortex_impulse", "physics"),
        ("leading_edge_suction", "physics"),
        ("uvlm_remainder", "physics"),
    ),
    "N2": (
        ("separation", "physics"),
        ("profile_drag", "physics"),
    ),
    "N3": (
        ("ds_vortex", "physics"),
        ("vortex_normal", "physics"),
    ),
    "N4": (("ct_consistency", "diagnostic"),),
    "N6": (("rig_drag", "necessary_physics"),),
    "R0": (("numerical_cycle_reduction", "diagnostic"),),
}

PARENT_LOCAL_REASON_ORDER = (
    "PARENT_FAIL_ALIAS_NONUNIFORM",
    "PARENT_FAIL_COMPONENT_REVERSED_NOT_RESTORED",
    "PARENT_FAIL_COMPONENT_UNDER_NOT_IMPROVED",
    "PARENT_FAIL_COMPONENT_OVER_NOT_IMPROVED",
    "PARENT_FAIL_PASS_COMPONENT_DAMAGED",
    "PARENT_FAIL_PAIRWISE_GUARD_DAMAGED",
    "PARENT_FAIL_CURVE_MAE",
    "PARENT_FAIL_PF_NOT_FULLY_RESTORED",
    "PARENT_FAIL_NO_PAIRWISE_DISJOINT_REPLICATION",
)

GRAPH_NODE_CONTRACT = {
    item["id"]: {
        key: copy.deepcopy(value) for key, value in item.items() if key != "id"
    }
    for item in confirmed_compare.AUTHORIZED_V41_GRAPH_CONTRACT["nodes"]
}

CONFIRMED_CONDITIONS = CONDITIONS_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]
CONFIRMED_BY_KEY = {
    condition_key(condition): condition for condition in CONFIRMED_CONDITIONS
}
EXPECTED_CONDITION_KEYS = frozenset(CONFIRMED_BY_KEY)
if len(EXPECTED_CONDITION_KEYS) != EXPECTED_CONDITIONS:
    raise AssertionError("confirmed condition contract drift")


class InvalidEvidenceError(ValueError):
    """The evidence receipt is invalid and must not produce a decision."""


class InvalidPreregistrationError(ValueError):
    """The prepare artifact or disease specification is invalid."""


@dataclass(frozen=True)
class EvaluationInputs:
    result_path: Path
    manifest_path: Path
    contributions_path: Path
    scorecard_path: Path
    fingerprint_path: Path
    prereg_path: Path
    baseline_receipt_path: Path


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_frozen_roots() -> None:
    expected_files = {
        CONFIRMED_COMPARE_SOURCE: AUTHORIZED_CONFIRMED_COMPARE_SHA256,
        AUTHORIZED_POSTPROCESS_AUTHORIZATION: (
            AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256
        ),
        AUTHORIZED_POSTPROCESS_PREREG: AUTHORIZED_POSTPROCESS_PREREG_SHA256,
        EVIDENCE_SCOPE_PREREG: AUTHORIZED_EVIDENCE_SCOPE_PREREG_SHA256,
        Path(benchmark.DEFAULT_DATA_MD): AUTHORIZED_MEASUREMENT_DATA_SHA256,
        Path(benchmark.__file__).resolve(): AUTHORIZED_BENCHMARK_SOURCE_SHA256,
        Path(residual_fingerprint.__file__).resolve(): (
            AUTHORIZED_FINGERPRINT_SOURCE_SHA256
        ),
    }
    expected_files.update(
        {
            path: AUTHORIZED_PARENT_PROTOCOL_SHA256_BY_VERSION[version]
            for version, path in zip(range(2, 6), PARENT_PROTOCOLS)
        }
    )
    expected_files.update(
        {
            path: AUTHORIZED_ACTIVE_PROTOCOL_SHA256_BY_VERSION[version]
            for version, path in zip(range(3, 9), ACTIVE_DISEASE_PROTOCOLS)
        }
    )
    for path, expected_sha256 in expected_files.items():
        try:
            actual = _sha256_file(path)
        except OSError as exc:
            raise InvalidEvidenceError(
                f"missing frozen attribution root {path}: {exc}"
            ) from exc
        if actual != expected_sha256:
            raise InvalidEvidenceError(
                f"frozen attribution root SHA-256 drift: {_display_path(path)}"
            )
    scorer_literals = {
        "authorization": (
            confirmed_compare.AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256,
            AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256,
        ),
        "scoring preregistration": (
            confirmed_compare.AUTHORIZED_POSTPROCESS_PREREG_SHA256,
            AUTHORIZED_POSTPROCESS_PREREG_SHA256,
        ),
        "measurement": (
            confirmed_compare.AUTHORIZED_MEASUREMENT_DATA_SHA256,
            AUTHORIZED_MEASUREMENT_DATA_SHA256,
        ),
        "benchmark": (
            confirmed_compare.AUTHORIZED_BENCHMARK_SOURCE_SHA256,
            AUTHORIZED_BENCHMARK_SOURCE_SHA256,
        ),
        "fingerprint": (
            confirmed_compare.AUTHORIZED_FINGERPRINT_SOURCE_SHA256,
            AUTHORIZED_FINGERPRINT_SOURCE_SHA256,
        ),
        "graph": (
            confirmed_compare.AUTHORIZED_V41_GRAPH_IDENTITY_SHA256,
            AUTHORIZED_GRAPH_IDENTITY_SHA256,
        ),
    }
    for label, (actual, expected) in scorer_literals.items():
        if actual != expected:
            raise InvalidEvidenceError(
                f"confirmed scorer {label} authorization literal drift"
            )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _loads_json_strict(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = _loads_json_strict(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise InvalidEvidenceError(f"{path}: cannot load JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidEvidenceError(f"{path}: expected a JSON object")
    return value


def _require_payload_file_match(
    value: Mapping[str, Any],
    path: Path,
    *,
    label: str,
    error_type: type[ValueError] = InvalidEvidenceError,
) -> None:
    try:
        on_disk = _loads_json_strict(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise error_type(f"{label}: cannot read bound JSON file: {exc}") from exc
    if not isinstance(on_disk, dict) or _canonical_hash(value) != _canonical_hash(
        on_disk
    ):
        raise error_type(f"{label}: in-memory payload does not match bound file")


def _bound_path(record: Mapping[str, Any], *, label: str) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise InvalidEvidenceError(f"{label}: identity path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _validate_identity_metadata(
    record: Any,
    *,
    label: str,
    expected_path_text: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an identity record without dereferencing its target."""

    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256", "size_bytes"}
        or not isinstance(record.get("path"), str)
        or not record["path"]
        or not _is_sha256(record.get("sha256"))
        or not isinstance(record.get("size_bytes"), int)
        or isinstance(record.get("size_bytes"), bool)
        or record["size_bytes"] < 0
    ):
        raise InvalidEvidenceError(f"{label}: malformed file identity metadata")
    if expected_path_text is not None and record["path"] != expected_path_text:
        raise InvalidEvidenceError(f"{label}: identity path drift")
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise InvalidEvidenceError(f"{label}: identity SHA-256 drift")
    return dict(record)


def _validate_file_identity(
    record: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    metadata = _validate_identity_metadata(record, label=label)
    path = _bound_path(metadata, label=label)
    if expected_path is not None and path != expected_path.resolve():
        raise InvalidEvidenceError(f"{label}: identity binds the wrong path")
    if not path.is_file():
        raise InvalidEvidenceError(f"{label}: identity target is not a file")
    if (
        _sha256_file(path) != metadata["sha256"]
        or path.stat().st_size != metadata["size_bytes"]
    ):
        raise InvalidEvidenceError(f"{label}: file identity/hash mismatch")
    return path


def _validate_receipt_pf_strata(receipt: Mapping[str, Any]) -> None:
    curve_to_family, _, _ = residual_fingerprint._physical_family_contract()
    strata = receipt.get("pf_equal_by_figure_channel")
    if not isinstance(strata, Mapping) or set(strata) != {
        "ALL",
        "17",
        "18",
        "19",
    }:
        raise InvalidEvidenceError("baseline PF strata figure contract drift")
    for figure in ("ALL", "17", "18", "19"):
        channels = strata[figure]
        if not isinstance(channels, Mapping) or set(channels) != {"ALL", "T", "L"}:
            raise InvalidEvidenceError(
                f"baseline PF strata {figure} channel contract drift"
            )
        for channel in ("ALL", "T", "L"):
            selected = [
                curve
                for curve in benchmark.CURVES_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]
                if (figure == "ALL" or curve.figure == figure)
                and (channel == "ALL" or curve.channel == channel)
            ]
            grouped: dict[str, list[str]] = {}
            for curve in selected:
                grouped.setdefault(curve_to_family[curve.key], []).append(curve.key)
            expected_families = [
                {
                    "physical_family_id": family_id,
                    "official_curve_keys": sorted(keys),
                    "n_official_curves": len(keys),
                }
                for family_id, keys in sorted(grouped.items())
            ]
            record = channels[channel]
            if (
                not isinstance(record, Mapping)
                or record.get("n_physical_families") != len(expected_families)
                or record.get("n_official_curves") != len(selected)
            ):
                raise InvalidEvidenceError(
                    f"baseline PF strata {figure}/{channel} count drift"
                )
            families = record.get("families")
            if not isinstance(families, list):
                raise InvalidEvidenceError(
                    f"baseline PF strata {figure}/{channel} families missing"
                )
            projection = [
                {
                    "physical_family_id": item.get("physical_family_id"),
                    "official_curve_keys": item.get("official_curve_keys"),
                    "n_official_curves": item.get("n_official_curves"),
                }
                for item in families
                if isinstance(item, Mapping)
            ]
            if _canonical_hash(projection) != _canonical_hash(expected_families):
                raise InvalidEvidenceError(
                    f"baseline PF strata {figure}/{channel} mapping drift"
                )
    primary = receipt.get("primary_metric")
    if (
        not isinstance(primary, Mapping)
        or primary.get("name") != "physical_family_equal_mean_absolute_error"
        or primary.get("unit") != "N"
        or primary.get("physical_family_count") != EXPECTED_PHYSICAL_FAMILIES
        or not isinstance(primary.get("alias_policy"), str)
    ):
        raise InvalidEvidenceError("baseline primary PF metric contract drift")
    _finite(primary.get("value"), label="baseline primary PF metric")


def _validate_baseline_receipt_metadata(
    receipt: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    receipt_path: Path,
    fingerprint_path: Path,
) -> dict[str, Any]:
    """Validate the published contract without touching the fresh triplet.

    In particular this function must never resolve, stat, hash, or open the
    contribution target.  The contribution identity is inert metadata until
    evaluate performs the full authorization-aware dereference.
    """

    _validate_frozen_roots()
    _require_payload_file_match(
        receipt,
        receipt_path,
        label="baseline receipt",
    )
    expected_top_level = {
        "schema_version",
        "artifact_type",
        "status",
        "baseline_bundle_id",
        "bundle_id_payload",
        "run_id",
        "evidence_scope",
        "contract",
        "input_artifacts",
        "validation",
        "physical_family_contract",
        "primary_metric",
        "pf_equal_by_figure_channel",
        "outputs",
        "global_promotion_eligible",
        "global_promotion_blockers",
        "allowed_use",
        "forbidden_use",
    }
    if (
        set(receipt) != expected_top_level
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_type")
        != "v41_fresh_confirmed42_baseline_bundle_receipt"
        or receipt.get("status") != "READY_FOR_CONFIRMED_BASELINE_DIAGNOSIS"
        or receipt.get("evidence_scope") != EVIDENCE_CONFIRMED
    ):
        raise InvalidEvidenceError("baseline receipt identity/status drift")
    contract = receipt.get("contract")
    expected_contract = {
        "official_curves": EXPECTED_CURVES,
        "raw_measurement_samples": EXPECTED_SAMPLES,
        "solver_conditions": EXPECTED_CONDITIONS,
        "physical_families": EXPECTED_PHYSICAL_FAMILIES,
        "duplicate_alias_groups": EXPECTED_ALIAS_GROUPS,
        "figure_curve_counts": {"17": 10, "18": 24, "19": 8},
        "conditional_fig19_cd_curves_excluded": sorted(EXPECTED_CONDITIONAL_CURVES),
    }
    if _canonical_hash(contract) != _canonical_hash(expected_contract):
        raise InvalidEvidenceError("baseline receipt confirmed contract drift")

    bundle_payload = receipt.get("bundle_id_payload")
    bundle_id = receipt.get("baseline_bundle_id")
    expected_payload_keys = {
        "result_sha256",
        "manifest_sha256",
        "contributions_sha256",
        "measurement_data_sha256",
        "benchmark_source_sha256",
        "fingerprint_source_sha256",
        "postprocessor_source_sha256",
        "physical_family_contract_sha256",
        "scoring_preregistration_sha256",
        "postprocess_authorization_sha256",
    }
    if (
        not isinstance(bundle_payload, Mapping)
        or set(bundle_payload) != expected_payload_keys
        or not isinstance(bundle_id, str)
        or bundle_id != _canonical_hash(bundle_payload)
    ):
        raise InvalidEvidenceError("baseline bundle ID/payload mismatch")
    frozen_payload = {
        "measurement_data_sha256": AUTHORIZED_MEASUREMENT_DATA_SHA256,
        "benchmark_source_sha256": AUTHORIZED_BENCHMARK_SOURCE_SHA256,
        "fingerprint_source_sha256": AUTHORIZED_FINGERPRINT_SOURCE_SHA256,
        "postprocessor_source_sha256": AUTHORIZED_CONFIRMED_COMPARE_SHA256,
        "physical_family_contract_sha256": (AUTHORIZED_PHYSICAL_FAMILY_CONTRACT_SHA256),
        "scoring_preregistration_sha256": (AUTHORIZED_POSTPROCESS_PREREG_SHA256),
        "postprocess_authorization_sha256": (
            AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256
        ),
    }
    if any(
        bundle_payload.get(name) != digest for name, digest in frozen_payload.items()
    ):
        raise InvalidEvidenceError("baseline bundle frozen-input SHA-256 drift")
    if fingerprint.get("baseline_bundle_id") != bundle_id:
        raise InvalidEvidenceError("fingerprint baseline bundle ID mismatch")
    pf_sha = fingerprint.get("physical_family_contract_sha256")
    if (
        pf_sha != AUTHORIZED_PHYSICAL_FAMILY_CONTRACT_SHA256
        or bundle_payload.get("physical_family_contract_sha256") != pf_sha
    ):
        raise InvalidEvidenceError("physical-family contract hash mismatch")

    required_fingerprint_gates = (
        "fresh_triplet_complete_and_bound",
        "force_ledgers_close_within_1e-9_N",
        "pf_strata_count_contract",
        "fig19_cd_zero_residual_leakage",
    )
    fingerprint_gates = fingerprint.get("validity_gates")
    if not isinstance(fingerprint_gates, Mapping) or any(
        fingerprint_gates.get(name) is not True for name in required_fingerprint_gates
    ):
        raise InvalidEvidenceError("fresh fingerprint receipt gates are missing")
    generator = fingerprint.get("generator")
    _validate_file_identity(
        generator,
        label="fresh fingerprint generator",
        expected_path=CONFIRMED_COMPARE_SOURCE,
    )
    if generator.get("sha256") != bundle_payload.get("postprocessor_source_sha256"):
        raise InvalidEvidenceError("fingerprint generator/bundle hash mismatch")
    inputs = receipt.get("input_artifacts")
    outputs = receipt.get("outputs")
    if (
        not isinstance(inputs, Mapping)
        or set(inputs)
        != {
            "result",
            "manifest",
            "claim_contributions",
            "measurement_data",
            "scoring_preregistration",
            "postprocess_authorization",
        }
        or not isinstance(outputs, Mapping)
        or set(outputs)
        != {
            "scorecard",
            "artifact",
            "fingerprint",
            "fig17",
            "fig17_sidecar",
            "fig18",
            "fig18_sidecar",
            "fig19",
            "fig19_sidecar",
        }
    ):
        raise InvalidEvidenceError("baseline receipt input/output identities missing")
    identity_metadata = {
        name: _validate_identity_metadata(record, label=f"baseline {name}")
        for name, record in inputs.items()
    }
    output_metadata = {
        name: _validate_identity_metadata(record, label=f"baseline output {name}")
        for name, record in outputs.items()
    }
    _validate_identity_metadata(
        inputs["measurement_data"],
        label="measurement data",
        expected_path_text=_display_path(Path(benchmark.DEFAULT_DATA_MD)),
        expected_sha256=AUTHORIZED_MEASUREMENT_DATA_SHA256,
    )
    _validate_identity_metadata(
        inputs["scoring_preregistration"],
        label="baseline scoring preregistration",
        expected_path_text=_display_path(AUTHORIZED_POSTPROCESS_PREREG),
        expected_sha256=AUTHORIZED_POSTPROCESS_PREREG_SHA256,
    )
    _validate_identity_metadata(
        inputs["postprocess_authorization"],
        label="baseline postprocess authorization",
        expected_path_text=_display_path(AUTHORIZED_POSTPROCESS_AUTHORIZATION),
        expected_sha256=AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256,
    )
    _validate_identity_metadata(
        outputs.get("fingerprint"),
        label="baseline output fingerprint",
        expected_path_text=_display_path(fingerprint_path),
    )
    digest_bindings = {
        "result_sha256": identity_metadata["result"]["sha256"],
        "manifest_sha256": identity_metadata["manifest"]["sha256"],
        "contributions_sha256": identity_metadata["claim_contributions"]["sha256"],
        "measurement_data_sha256": identity_metadata["measurement_data"]["sha256"],
        "scoring_preregistration_sha256": identity_metadata["scoring_preregistration"][
            "sha256"
        ],
        "postprocess_authorization_sha256": identity_metadata[
            "postprocess_authorization"
        ]["sha256"],
    }
    for name, digest in digest_bindings.items():
        if bundle_payload.get(name) != digest:
            raise InvalidEvidenceError(f"baseline bundle {name} drift")

    validation = receipt.get("validation")
    if not isinstance(validation, Mapping):
        raise InvalidEvidenceError("baseline receipt triplet validation missing")
    body_error = _finite(
        validation.get("maximum_recomputed_body_ledger_error_N"),
        label="baseline receipt body ledger error",
    )
    wind_error = _finite(
        validation.get("maximum_recomputed_wind_ledger_error_N"),
        label="baseline receipt wind ledger error",
    )
    guard_error = _finite(
        validation.get("maximum_reported_guard_error_N"),
        label="baseline receipt guard error",
    )
    if (
        validation.get("passed") is not True
        or validation.get("schema_version")
        != confirmed_compare.CHECKPOINT_SCHEMA_VERSION
        or validation.get("run_id") != receipt.get("run_id")
        or validation.get("condition_count") != EXPECTED_CONDITIONS
        or validation.get("evidence_case_count") != EXPECTED_CONDITIONS
        or validation.get("case_guard_count") != EXPECTED_CONDITIONS
        or validation.get("solver_source_count")
        != confirmed_compare.EXPECTED_SOLVER_SOURCES
        or validation.get("control_source_count")
        != confirmed_compare.EXPECTED_CONTROL_SOURCES
        or validation.get("claim_graph_identity_sha256")
        != AUTHORIZED_GRAPH_IDENTITY_SHA256
        or validation.get("ledger_tolerance_N") != LEDGER_TOLERANCE_N
        or body_error > LEDGER_TOLERANCE_N
        or wind_error > LEDGER_TOLERANCE_N
        or guard_error > LEDGER_TOLERANCE_N
        or validation.get("same_timestamp_complete_resume_revalidation") is not True
        or validation.get("complete_resume_triplet_unchanged") is not True
        or validation.get("postprocess_authorization_validated") is not True
    ):
        raise InvalidEvidenceError("baseline receipt triplet validation drift")
    input_receipts = validation.get("input_receipts")
    if not isinstance(input_receipts, Mapping) or any(
        input_receipts.get(name) != digest_bindings[name]
        for name in ("result_sha256", "manifest_sha256", "contributions_sha256")
    ):
        raise InvalidEvidenceError("baseline validation input receipts drift")
    pf_contract = receipt.get("physical_family_contract")
    if (
        not isinstance(pf_contract, Mapping)
        or pf_contract.get("sha256") != pf_sha
        or pf_contract.get("family_count") != EXPECTED_PHYSICAL_FAMILIES
        or pf_contract.get("alias_group_count") != EXPECTED_ALIAS_GROUPS
    ):
        raise InvalidEvidenceError("baseline physical-family receipt drift")
    _validate_receipt_pf_strata(receipt)
    if receipt.get("global_promotion_eligible") is not False:
        raise InvalidEvidenceError(
            "confirmed-only receipt cannot be globally promotable"
        )
    authorization = _load_json(AUTHORIZED_POSTPROCESS_AUTHORIZATION)
    if (
        receipt.get("run_id") != authorization.get("run_id")
        or authorization.get("scope", {}).get("result_path")
        != identity_metadata["result"]["path"]
        or authorization.get("scope", {}).get("manifest_path")
        != identity_metadata["manifest"]["path"]
        or authorization.get("scope", {}).get("contributions_path")
        != identity_metadata["claim_contributions"]["path"]
    ):
        raise InvalidEvidenceError("baseline receipt/authorization scope drift")
    return {
        "baseline_bundle_id": bundle_id,
        "input_identity_metadata": identity_metadata,
        "output_identity_metadata": output_metadata,
        "receipt_path": receipt_path.resolve(),
    }


def _contribution_free_receipt_projection(
    receipt: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    receipt_path: Path,
) -> dict[str, Any]:
    outputs = metadata["output_identity_metadata"]
    return {
        "schema_version": 1,
        "artifact_type": "fig171819_contribution_free_baseline_projection",
        "status": "BASELINE_METADATA_VALIDATED_BEFORE_PREPARE",
        "baseline_bundle_id": metadata["baseline_bundle_id"],
        "run_id": receipt["run_id"],
        "evidence_scope": EVIDENCE_CONFIRMED,
        "contract": copy.deepcopy(receipt["contract"]),
        "scorecard_sha256": outputs["scorecard"]["sha256"],
        "fingerprint_identity": copy.deepcopy(outputs["fingerprint"]),
        "baseline_receipt_sha256": _sha256_file(receipt_path),
        "validation_sha256": _canonical_hash(receipt["validation"]),
        "bundle_id_payload_sha256": _canonical_hash(receipt["bundle_id_payload"]),
        "postprocess_authorization_sha256": (
            AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256
        ),
        "claim_graph_identity_sha256": AUTHORIZED_GRAPH_IDENTITY_SHA256,
    }


def _validate_contribution_free_projection(
    projection: Any,
    fingerprint: Mapping[str, Any],
    *,
    fingerprint_path: Path,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "baseline_bundle_id",
        "run_id",
        "evidence_scope",
        "contract",
        "scorecard_sha256",
        "fingerprint_identity",
        "baseline_receipt_sha256",
        "validation_sha256",
        "bundle_id_payload_sha256",
        "postprocess_authorization_sha256",
        "claim_graph_identity_sha256",
    }
    if not isinstance(projection, Mapping) or set(projection) != expected_keys:
        raise InvalidPreregistrationError(
            "contribution-free baseline projection schema drift"
        )
    forbidden_text = json.dumps(projection, sort_keys=True).lower()
    if any(
        token in forbidden_text
        for token in (
            "claim_contributions",
            "contributions_path",
            "contributions_sha256",
        )
    ):
        raise InvalidPreregistrationError(
            "Prepare projection leaks contribution identity"
        )
    if (
        projection.get("schema_version") != 1
        or projection.get("artifact_type")
        != "fig171819_contribution_free_baseline_projection"
        or projection.get("status") != "BASELINE_METADATA_VALIDATED_BEFORE_PREPARE"
        or projection.get("evidence_scope") != EVIDENCE_CONFIRMED
        or projection.get("claim_graph_identity_sha256")
        != AUTHORIZED_GRAPH_IDENTITY_SHA256
        or projection.get("postprocess_authorization_sha256")
        != AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256
        or not _is_sha256(projection.get("scorecard_sha256"))
        or not _is_sha256(projection.get("baseline_receipt_sha256"))
        or not _is_sha256(projection.get("validation_sha256"))
        or not _is_sha256(projection.get("bundle_id_payload_sha256"))
    ):
        raise InvalidPreregistrationError(
            "contribution-free baseline projection identity drift"
        )
    try:
        fingerprint_identity = _validate_identity_metadata(
            projection["fingerprint_identity"],
            label="Prepare fingerprint identity",
            expected_path_text=_display_path(fingerprint_path),
        )
    except InvalidEvidenceError as exc:
        raise InvalidPreregistrationError(str(exc)) from exc
    if fingerprint_identity["sha256"] != _sha256_file(fingerprint_path):
        raise InvalidPreregistrationError("Prepare fingerprint hash drift")
    if fingerprint.get("baseline_bundle_id") != projection["baseline_bundle_id"]:
        raise InvalidPreregistrationError("Prepare projection/fingerprint bundle drift")
    return copy.deepcopy(dict(projection))


def _projection_as_receipt(projection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_bundle_id": projection["baseline_bundle_id"],
        "outputs": {
            "scorecard": {"sha256": projection["scorecard_sha256"]},
            "fingerprint": copy.deepcopy(projection["fingerprint_identity"]),
        },
    }


def _validate_baseline_receipt_full(
    receipt: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    receipt_path: Path,
    fingerprint_path: Path,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
    scorecard_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _validate_baseline_receipt_metadata(
        receipt,
        fingerprint,
        receipt_path=receipt_path,
        fingerprint_path=fingerprint_path,
    )
    inputs = receipt["input_artifacts"]
    expected_inputs = {
        "result": result_path,
        "manifest": manifest_path,
        "claim_contributions": contributions_path,
        "measurement_data": Path(benchmark.DEFAULT_DATA_MD),
        "scoring_preregistration": AUTHORIZED_POSTPROCESS_PREREG,
        "postprocess_authorization": AUTHORIZED_POSTPROCESS_AUTHORIZATION,
    }
    for name, path in expected_inputs.items():
        _validate_file_identity(
            inputs[name],
            label=f"baseline {name}",
            expected_path=path,
        )
    for name, record in receipt["outputs"].items():
        expected = (
            scorecard_path
            if name == "scorecard"
            else fingerprint_path
            if name == "fingerprint"
            else None
        )
        _validate_file_identity(
            record,
            label=f"baseline output {name}",
            expected_path=expected,
        )

    try:
        confirmed_compare._validate_postprocess_authorization(
            authorization_path=AUTHORIZED_POSTPROCESS_AUTHORIZATION,
            expected_authorization_path=AUTHORIZED_POSTPROCESS_AUTHORIZATION,
            expected_authorization_sha256=(AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256),
            scoring_prereg_path=AUTHORIZED_POSTPROCESS_PREREG,
            expected_prereg_path=AUTHORIZED_POSTPROCESS_PREREG,
            expected_prereg_sha256=AUTHORIZED_POSTPROCESS_PREREG_SHA256,
            result_path=result_path,
            manifest_path=manifest_path,
            contributions_path=contributions_path,
            data_path=Path(benchmark.DEFAULT_DATA_MD),
            manifest=manifest,
        )
        bundle = confirmed_compare.validate_fresh151_bundle(
            result_path=result_path,
            manifest_path=manifest_path,
            contributions_path=contributions_path,
        )
    except confirmed_compare.BaselineContractError as exc:
        raise InvalidEvidenceError(
            f"authorization-aware fresh151 validation failed: {exc}"
        ) from exc

    expected_validation = {
        **copy.deepcopy(bundle.validation),
        "same_timestamp_complete_resume_revalidation": True,
        "complete_resume_triplet_unchanged": True,
        "postprocess_authorization_validated": True,
    }
    if _canonical_hash(receipt.get("validation")) != _canonical_hash(
        expected_validation
    ):
        raise InvalidEvidenceError(
            "baseline receipt validation differs from full recomputation"
        )
    _, families, aliases = residual_fingerprint._physical_family_contract()
    expected_pf_hash = _canonical_hash({"families": families, "aliases": aliases})
    if expected_pf_hash != AUTHORIZED_PHYSICAL_FAMILY_CONTRACT_SHA256:
        raise InvalidEvidenceError("independent physical-family root drift")
    expected_payload = confirmed_compare._bundle_id_payload(
        bundle=bundle,
        data_path=Path(benchmark.DEFAULT_DATA_MD),
        scoring_prereg_path=AUTHORIZED_POSTPROCESS_PREREG,
        authorization_path=AUTHORIZED_POSTPROCESS_AUTHORIZATION,
        pf_contract_sha256=expected_pf_hash,
    )
    if _canonical_hash(receipt.get("bundle_id_payload")) != _canonical_hash(
        expected_payload
    ) or receipt.get("baseline_bundle_id") != _canonical_hash(expected_payload):
        raise InvalidEvidenceError(
            "baseline receipt payload differs from authorized recomputation"
        )
    return {
        **metadata,
        "bundle": bundle,
        "result_path": result_path.resolve(),
        "manifest_path": manifest_path.resolve(),
        "contributions_path": contributions_path.resolve(),
        "scorecard_path": scorecard_path.resolve(),
        "fingerprint_path": fingerprint_path.resolve(),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    partial = Path(partial_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if partial.exists():
            partial.unlink()


def _protocol_stack() -> dict[str, list[dict[str, Any]]]:
    return {
        "active_disease": [
            {
                "version": version,
                "path": _display_path(path),
                "sha256": AUTHORIZED_ACTIVE_PROTOCOL_SHA256_BY_VERSION[version],
            }
            for version, path in zip(range(3, 9), ACTIVE_DISEASE_PROTOCOLS)
        ],
        "parent_attribution": [
            {
                "version": version,
                "path": _display_path(path),
                "sha256": AUTHORIZED_PARENT_PROTOCOL_SHA256_BY_VERSION[version],
            }
            for version, path in zip(range(2, 6), PARENT_PROTOCOLS)
        ],
    }


EXECUTION_ENVELOPE_FIELDS = {
    "schema_version",
    "artifact_type",
    "stage",
    "evidence_commit_sha",
    "attestation_commit_sha",
    "attestation_payload_sha256",
    "authorization_blob_sha256",
    "launcher_blob_sha256",
    "git_no_replace_objects",
    "git_config_nosystem",
    "git_config_global",
    "git_hooks_disabled",
    "checkout_used",
    "raw_blob_materialization",
    "runtime_source_closure_verified",
    "runtime_source_closure_sha256",
    "python_executable_realpath",
    "python_isolated_flag",
    "python_no_site_flag",
    "python_no_bytecode_flag",
    "python_startup_contamination_check",
    "outer_preflight_receipt_sha256",
    "inner_launcher_receipt_sha256",
    "outer_completion_receipt_sha256",
    "outer_preflight_receipt",
    "inner_launcher_receipt",
    "outer_completion_receipt",
    "post_run_source_closure_verified",
    "transport_status",
    "scientific_status",
    "cleanup_status",
    "cleanup_scope",
    "cleanup_target_relative_paths",
    "execution_layout_cleanup_status",
}

EXECUTION_RECEIPT_FIELDS = {
    "schema_version",
    "artifact_type",
    "stage",
    "upstream_receipts",
    "body",
}

ATTRIBUTION_SCIENTIFIC_STATUSES = {
    "select-disease": frozenset(
        {
            "ACTIVE_DISEASE_FROZEN",
            "NO_DECISION_NO_PRE_REPLICATION_CANDIDATE",
            "NO_DECISION_NO_INDEPENDENT_REPLICATION",
            "NO_DECISION_DISEASE_TIE",
            "NO_DECISION_VIEW_DISAGREEMENT",
            "NO_DECISION_INSUFFICIENT_RANK_MARGIN",
            "NO_DECISION_LEAVE_ONE_PF_SENSITIVE",
            "INVALID_EVIDENCE",
        }
    ),
    "prepare": frozenset({"PREPARED", "INVALID_EVIDENCE"}),
    "evaluate": frozenset(
        {
            "ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS",
            "ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS",
            "NO_DECISION_MULTIPLE_PARENTS",
            "NO_DECISION_NO_PARENT_FULL_COVERAGE",
            "INVALID_EVIDENCE",
        }
    ),
}

ATTRIBUTION_INTERMEDIATE_OUTPUTS = {
    "select-disease": ".raw-select-disease-payload.json",
    "prepare": ".raw-prepare-payload.json",
    "evaluate": ".raw-evaluate-payload.json",
}


def _receipt_object_sha256(value: Mapping[str, Any]) -> str:
    try:
        raw = (
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise InvalidEvidenceError(
            "embedded execution receipt is not canonical JSON"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _validate_embedded_execution_receipt(
    value: Any,
    *,
    receipt_stage: str,
    expected_sha256: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != EXECUTION_RECEIPT_FIELDS:
        raise InvalidEvidenceError("embedded execution receipt schema drift")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("artifact_type") != "fig171819_evidence_receipt_envelope"
        or value.get("stage") != receipt_stage
    ):
        raise InvalidEvidenceError("embedded execution receipt identity drift")
    if _receipt_object_sha256(value) != expected_sha256:
        raise InvalidEvidenceError("embedded execution receipt canonical hash drift")
    upstream = value.get("upstream_receipts")
    body = value.get("body")
    if not isinstance(upstream, Mapping) or not isinstance(body, Mapping):
        raise InvalidEvidenceError("embedded execution receipt is malformed")
    return copy.deepcopy(dict(value))


def _validate_execution_envelope(
    envelope: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    if stage not in ATTRIBUTION_SCIENTIFIC_STATUSES:
        raise InvalidEvidenceError("outer execution envelope stage is unknown")
    if set(envelope) != EXECUTION_ENVELOPE_FIELDS:
        raise InvalidEvidenceError("outer execution envelope schema drift")
    if (
        type(envelope.get("schema_version")) is not int
        or envelope.get("schema_version") != 1
        or envelope.get("artifact_type")
        != "fig171819_verified_execution_receipt_envelope"
        or envelope.get("stage") != stage
    ):
        raise InvalidEvidenceError("outer execution envelope identity drift")
    for name in (
        "attestation_payload_sha256",
        "authorization_blob_sha256",
        "launcher_blob_sha256",
        "runtime_source_closure_sha256",
        "outer_preflight_receipt_sha256",
        "inner_launcher_receipt_sha256",
        "outer_completion_receipt_sha256",
    ):
        if not _is_sha256(envelope.get(name)):
            raise InvalidEvidenceError(f"outer execution envelope {name} malformed")
    for name in ("evidence_commit_sha", "attestation_commit_sha"):
        value = envelope.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise InvalidEvidenceError(f"outer execution envelope {name} malformed")
    required_true = (
        "git_no_replace_objects",
        "git_config_nosystem",
        "git_hooks_disabled",
        "raw_blob_materialization",
        "runtime_source_closure_verified",
        "python_isolated_flag",
        "python_no_site_flag",
        "python_no_bytecode_flag",
        "post_run_source_closure_verified",
    )
    if any(envelope.get(name) is not True for name in required_true):
        raise InvalidEvidenceError("outer execution envelope security gate failed")
    if envelope.get("checkout_used") is not False:
        raise InvalidEvidenceError("outer execution envelope used checkout")
    if (
        envelope.get("git_config_global") != "/dev/null"
        or envelope.get("python_executable_realpath")
        != "/home/exuber/anaconda3/envs/fluxvortex/bin/python"
        or envelope.get("python_startup_contamination_check") != "PASS"
        or envelope.get("cleanup_status") != "PASS"
        or envelope.get("transport_status") != "PASS"
        or envelope.get("cleanup_scope") != "INTERMEDIATE_PAYLOAD_ONLY"
        or envelope.get("cleanup_target_relative_paths")
        != [ATTRIBUTION_INTERMEDIATE_OUTPUTS.get(stage)]
        or envelope.get("execution_layout_cleanup_status")
        != "NOT_RUN_RETAINED_FOR_AUDIT"
    ):
        raise InvalidEvidenceError("outer execution envelope runtime contract drift")
    h0 = _validate_embedded_execution_receipt(
        envelope.get("outer_preflight_receipt"),
        receipt_stage="outer_preflight",
        expected_sha256=envelope.get("outer_preflight_receipt_sha256"),
    )
    h1 = _validate_embedded_execution_receipt(
        envelope.get("inner_launcher_receipt"),
        receipt_stage="inner_launcher",
        expected_sha256=envelope.get("inner_launcher_receipt_sha256"),
    )
    h2 = _validate_embedded_execution_receipt(
        envelope.get("outer_completion_receipt"),
        receipt_stage="outer_completion",
        expected_sha256=envelope.get("outer_completion_receipt_sha256"),
    )
    if h0["upstream_receipts"] != {}:
        raise InvalidEvidenceError("embedded H0 has an upstream receipt")
    if h1["upstream_receipts"] != {"H0": envelope["outer_preflight_receipt_sha256"]}:
        raise InvalidEvidenceError("embedded H1 does not bind H0")
    if h2["upstream_receipts"] != {
        "H0": envelope["outer_preflight_receipt_sha256"],
        "H1": envelope["inner_launcher_receipt_sha256"],
    }:
        raise InvalidEvidenceError("embedded H2 does not bind H0 and H1")
    h0_body = h0["body"]
    h1_body = h1["body"]
    h2_body = h2["body"]
    command = h0_body.get("command_envelope")
    binding_names = (
        "evidence_commit_sha",
        "attestation_commit_sha",
        "attestation_payload_sha256",
        "authorization_blob_sha256",
        "launcher_blob_sha256",
    )
    if (
        h0_body.get("status") != "PASS"
        or h0_body.get("dry_run") is not False
        or h0_body.get("bindings") != {name: envelope[name] for name in binding_names}
        or h0_body.get("runtime_source_closure_sha256")
        != envelope["runtime_source_closure_sha256"]
        or h0_body.get("git_no_replace_objects") != envelope["git_no_replace_objects"]
        or h0_body.get("git_config_nosystem") != envelope["git_config_nosystem"]
        or h0_body.get("git_config_global") != envelope["git_config_global"]
        or h0_body.get("git_hooks_disabled") != envelope["git_hooks_disabled"]
        or h0_body.get("checkout_used") != envelope["checkout_used"]
        or h0_body.get("raw_blob_materialization")
        != envelope["raw_blob_materialization"]
        or not isinstance(command, Mapping)
        or command.get("command") != stage
    ):
        raise InvalidEvidenceError("embedded H0 command/binding body drift")
    if (
        h1_body.get("status") != "PASS"
        or h1_body.get("runtime_source_closure_verified") is not True
        or h1_body.get("runtime_source_closure_sha256")
        != envelope["runtime_source_closure_sha256"]
        or h1_body.get("python_executable_realpath")
        != envelope["python_executable_realpath"]
        or h1_body.get("python_isolated_flag") is not True
        or h1_body.get("python_no_site_flag") is not True
        or h1_body.get("python_no_bytecode_flag") is not True
        or h1_body.get("python_startup_contamination_check") != "PASS"
        or h1_body.get("runtime_environment_manifest_verified") is not True
    ):
        raise InvalidEvidenceError("embedded H1 closure/startup body drift")
    scientific_status = h2_body.get("scientific_status")
    if (
        h2_body.get("status") != "PASS"
        or h2_body.get("transport_status") != "PASS"
        or h2_body.get("post_run_source_closure") != "PASS"
        or h2_body.get("post_run_output_inventory") != "PASS"
        or h2_body.get("post_run_errors") != []
        or h2_body.get("intermediate_payload_cleanup_status") != "PASS"
        or h2_body.get("cleanup_status") != "PASS"
        or h2_body.get("cleanup_scope") != "INTERMEDIATE_PAYLOAD_ONLY"
        or h2_body.get("cleanup_target_relative_paths")
        != [ATTRIBUTION_INTERMEDIATE_OUTPUTS.get(stage)]
        or h2_body.get("execution_layout_cleanup_status")
        != "NOT_RUN_RETAINED_FOR_AUDIT"
    ):
        raise InvalidEvidenceError("embedded H2 transport/cleanup body drift")
    if (
        scientific_status not in ATTRIBUTION_SCIENTIFIC_STATUSES.get(stage, ())
        or envelope.get("scientific_status") != scientific_status
        or envelope.get("transport_status") != h2_body.get("transport_status")
        or envelope.get("cleanup_status") != h2_body.get("cleanup_status")
        or envelope.get("cleanup_scope") != h2_body.get("cleanup_scope")
        or envelope.get("cleanup_target_relative_paths")
        != h2_body.get("cleanup_target_relative_paths")
        or envelope.get("execution_layout_cleanup_status")
        != h2_body.get("execution_layout_cleanup_status")
    ):
        raise InvalidEvidenceError("outer execution scientific/cleanup binding drift")
    expected_exit_code = 2 if scientific_status == "INVALID_EVIDENCE" else 0
    if h2_body.get("inner_exit_code") != expected_exit_code:
        raise InvalidEvidenceError(
            "embedded H2 scientific status/inner exit code drift"
        )
    if not _is_sha256(h2_body.get("intermediate_payload_exact_sha256")):
        raise InvalidEvidenceError("embedded H2 exact payload hash malformed")
    if not _is_sha256(h2_body.get("intermediate_payload_canonical_sha256")):
        raise InvalidEvidenceError("embedded H2 canonical payload hash malformed")
    return copy.deepcopy(dict(envelope))


def _unwrap_scientific_payload(
    value: Mapping[str, Any],
    *,
    stage: str,
    required: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if value.get("artifact_type") != "fig171819_scientific_payload_envelope":
        if required:
            raise InvalidEvidenceError(
                f"{stage}: verified outer scientific envelope is required"
            )
        return copy.deepcopy(dict(value)), None
    if set(value) != {
        "schema_version",
        "artifact_type",
        "stage",
        "payload_sha256",
        "execution_envelope",
        "payload",
        "production_execution_authorized",
    }:
        raise InvalidEvidenceError("scientific payload envelope schema drift")
    if (
        value.get("schema_version") != 1
        or value.get("stage") != stage
        or value.get("production_execution_authorized") is not True
    ):
        raise InvalidEvidenceError("scientific payload envelope identity drift")
    payload = value.get("payload")
    envelope = value.get("execution_envelope")
    if not isinstance(payload, Mapping) or not isinstance(envelope, Mapping):
        raise InvalidEvidenceError("scientific payload envelope malformed")
    if value.get("payload_sha256") != _canonical_hash(payload):
        raise InvalidEvidenceError("scientific payload envelope hash drift")
    validated_envelope = _validate_execution_envelope(envelope, stage=stage)
    h2 = validated_envelope["outer_completion_receipt"]
    h2_body = h2["body"]
    if payload.get("status") != validated_envelope["scientific_status"] or value.get(
        "payload_sha256"
    ) != h2_body.get("intermediate_payload_canonical_sha256"):
        raise InvalidEvidenceError(
            "scientific payload status/hash differs from embedded H2"
        )
    return copy.deepcopy(dict(payload)), validated_envelope


EXECUTION_CHAIN_IDENTITY_FIELDS = (
    "evidence_commit_sha",
    "attestation_commit_sha",
    "attestation_payload_sha256",
    "authorization_blob_sha256",
    "launcher_blob_sha256",
    "runtime_source_closure_sha256",
)


def _validate_execution_chain_identity(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    previous_stage: str,
    current_stage: str,
) -> None:
    normalized_previous = _validate_execution_envelope(
        previous,
        stage=previous_stage,
    )
    normalized_current = _validate_execution_envelope(
        current,
        stage=current_stage,
    )
    drift = [
        name
        for name in EXECUTION_CHAIN_IDENTITY_FIELDS
        if normalized_previous[name] != normalized_current[name]
    ]
    if drift:
        raise InvalidEvidenceError(
            "cross-stage execution identity drift: " + ", ".join(drift)
        )


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, str)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise InvalidEvidenceError(f"{label}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidEvidenceError(f"{label}: expected a finite number")
    return result


def _same_number(left: Any, right: Any, tolerance: float = 1.0e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _strict_guards(value: Any) -> bool:
    try:
        confirmed_compare._validate_guard_set(value, label="attribution guard")
    except confirmed_compare.BaselineContractError:
        return False
    return True


def _wind_force(body_force: Sequence[float], aoa_deg: float) -> dict[str, float]:
    body = np.asarray(body_force, dtype=float)
    if body.shape != (3,) or not np.isfinite(body).all():
        raise InvalidEvidenceError("contribution body force is not a finite 3-vector")
    angle = math.radians(aoa_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return {
        "L": float(body[2] * cosine - body[0] * sine),
        "T": float(-(body[0] * cosine + body[2] * sine)),
    }


def _claim_graph_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    closure = manifest.get("closure")
    topology = manifest.get("topology")
    nodes = manifest.get("nodes")
    parameter_sources = manifest.get("parameter_sources")
    if closure != "v41":
        raise InvalidEvidenceError("claim manifest closure is not v41")
    if topology != list(EXPECTED_GRAPH_NODES):
        raise InvalidEvidenceError("claim graph topology drift")
    if not isinstance(nodes, list) or len(nodes) != len(EXPECTED_GRAPH_NODES):
        raise InvalidEvidenceError("claim graph node inventory drift")
    normalized_nodes: list[dict[str, Any]] = []
    for expected_id, node in zip(EXPECTED_GRAPH_NODES, nodes):
        if not isinstance(node, Mapping) or node.get("id") != expected_id:
            raise InvalidEvidenceError("claim graph node order/identity drift")
        expected = GRAPH_NODE_CONTRACT[expected_id]
        for field, expected_value in expected.items():
            if node.get(field) != expected_value:
                raise InvalidEvidenceError(f"{expected_id}: runtime {field} drift")
        normalized_nodes.append(
            {
                field: node[field]
                for field in (
                    "id",
                    "state",
                    "freeze",
                    "runtime_role",
                    "implementation",
                    "implementation_version",
                    "implementation_hash",
                )
            }
        )
    if not isinstance(parameter_sources, Mapping):
        raise InvalidEvidenceError("claim graph parameter sources malformed")
    payload = {
        "closure": closure,
        "topology": list(topology),
        "nodes": normalized_nodes,
        "parameter_sources": dict(sorted(parameter_sources.items())),
    }
    if _canonical_hash(payload) != _canonical_hash(
        confirmed_compare.AUTHORIZED_V41_GRAPH_CONTRACT
    ):
        raise InvalidEvidenceError("claim graph static authorization drift")
    return payload


def _claim_graph_identity(manifest: Mapping[str, Any]) -> str:
    identity = _canonical_hash(_claim_graph_identity_payload(manifest))
    if identity != AUTHORIZED_GRAPH_IDENTITY_SHA256:
        raise InvalidEvidenceError("claim graph identity is not authorized V4.1")
    return identity


def _fingerprint_maps(
    fingerprint: Mapping[str, Any],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if fingerprint.get("status") != "DESCRIPTIVE_FINGERPRINT_COMPLETE":
        raise InvalidEvidenceError("fingerprint is not complete")
    validity = fingerprint.get("validity_gates")
    if (
        not isinstance(validity, Mapping)
        or not validity
        or not all(value is True for value in validity.values())
    ):
        raise InvalidEvidenceError("fingerprint validity gates are not all true")
    contract = fingerprint.get("contract")
    expected_contract = {
        "confirmed_curves": EXPECTED_CURVES,
        "raw_measurement_samples": EXPECTED_SAMPLES,
        "solver_conditions": EXPECTED_CONDITIONS,
        "physical_curve_families": EXPECTED_PHYSICAL_FAMILIES,
        "duplicate_alias_groups": EXPECTED_ALIAS_GROUPS,
        "excluded_conditional_curves": sorted(EXPECTED_CONDITIONAL_CURVES),
    }
    if _canonical_hash(contract) != _canonical_hash(expected_contract):
        raise InvalidEvidenceError("fingerprint confirmed contract drift")

    samples = fingerprint.get("samples")
    curves = fingerprint.get("official_curves")
    families = fingerprint.get("physical_curve_families")
    aliases = fingerprint.get("duplicate_aliases")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLES:
        raise InvalidEvidenceError("fingerprint sample count drift")
    if not isinstance(curves, list) or len(curves) != EXPECTED_CURVES:
        raise InvalidEvidenceError("fingerprint curve count drift")
    if not isinstance(families, list) or len(families) != EXPECTED_PHYSICAL_FAMILIES:
        raise InvalidEvidenceError("fingerprint physical-family count drift")
    if not isinstance(aliases, list) or len(aliases) != EXPECTED_ALIAS_GROUPS:
        raise InvalidEvidenceError("fingerprint alias count drift")

    curve_specs = {
        curve.key: curve
        for curve in benchmark.CURVES_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]
    }
    expected_curve_to_family, expected_families, expected_aliases = (
        residual_fingerprint._physical_family_contract()
    )
    expected_pf_hash = _canonical_hash(
        {"families": expected_families, "aliases": expected_aliases}
    )
    if (
        expected_pf_hash != AUTHORIZED_PHYSICAL_FAMILY_CONTRACT_SHA256
        or fingerprint.get("physical_family_contract_sha256") != expected_pf_hash
    ):
        raise InvalidEvidenceError("independent physical-family contract drift")
    expected_family_by_id = {
        item["physical_family_id"]: item for item in expected_families
    }
    try:
        measurements = benchmark.load_measurements(benchmark.DEFAULT_DATA_MD)
        measurement_validation = benchmark.validate_measurement_contract(
            measurements,
            source_path=benchmark.DEFAULT_DATA_MD,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise InvalidEvidenceError(
            f"cannot rebuild frozen measurement contract: {exc}"
        ) from exc
    if measurement_validation.get("passed") is not True:
        raise InvalidEvidenceError("frozen measurement contract failed")

    samples_by_curve: dict[str, list[dict[str, Any]]] = {}
    seen_samples: set[tuple[str, int]] = set()
    for raw in samples:
        if not isinstance(raw, Mapping):
            raise InvalidEvidenceError("fingerprint sample is malformed")
        sample = dict(raw)
        curve = sample.get("curve")
        index = sample.get("measurement_index")
        if (
            not isinstance(curve, str)
            or curve not in curve_specs
            or not isinstance(index, int)
            or isinstance(index, bool)
        ):
            raise InvalidEvidenceError("fingerprint sample identity malformed")
        identity = (curve, index)
        if identity in seen_samples:
            raise InvalidEvidenceError("duplicate fingerprint sample identity")
        seen_samples.add(identity)
        for field in (
            "canonical_nominal_x",
            "raw_x",
            "evaluation_x",
            "measurement_N",
            "model_N",
            "left_weight",
            "right_weight",
        ):
            _finite(sample.get(field), label=f"{curve}[{index}] {field}")
        spec = curve_specs[curve]
        measurement = measurements[curve]
        nominal_x = (
            benchmark.RAW_FS if spec.abscissa == "frequency_Hz" else benchmark.TWS
        )
        if not 0 <= index < len(measurement.x):
            raise InvalidEvidenceError(f"{curve}: measurement index out of range")
        raw_x = float(measurement.x[index])
        evaluation_x = float(np.clip(raw_x, spec.x[0], spec.x[-1]))
        expected_bracket = residual_fingerprint._condition_bracket(
            np.asarray(spec.x, dtype=float),
            spec.conditions,
            evaluation_x,
        )
        expected_identity = {
            "physical_family_id": expected_curve_to_family[curve],
            "figure": spec.figure,
            "panel": spec.panel,
            "channel": spec.channel,
            "abscissa": spec.abscissa,
        }
        if any(sample.get(name) != value for name, value in expected_identity.items()):
            raise InvalidEvidenceError(f"{curve}[{index}]: curve/PF identity drift")
        expected_numbers = {
            "canonical_nominal_x": nominal_x[index],
            "raw_x": raw_x,
            "evaluation_x": evaluation_x,
            "measurement_N": measurement.values_N[index],
        }
        if any(
            not _same_number(sample.get(name), value)
            for name, value in expected_numbers.items()
        ):
            raise InvalidEvidenceError(
                f"{curve}[{index}]: measurement/abscissa contract drift"
            )
        if any(
            sample.get(name) != value
            if name.endswith("_key")
            else not _same_number(sample.get(name), value)
            for name, value in expected_bracket.items()
        ):
            raise InvalidEvidenceError(
                f"{curve}[{index}]: condition bracket contract drift"
            )
        samples_by_curve.setdefault(curve, []).append(sample)
    if set(samples_by_curve) != set(curve_specs):
        raise InvalidEvidenceError("fingerprint sample curve contract drift")
    for curve, records in samples_by_curve.items():
        records.sort(key=lambda item: item["measurement_index"])
        expected_count = len(measurements[curve].x)
        if [item["measurement_index"] for item in records] != list(
            range(expected_count)
        ):
            raise InvalidEvidenceError(f"{curve}: non-contiguous measurement indices")
        for field in ("canonical_nominal_x", "raw_x", "evaluation_x"):
            coordinates = [float(item[field]) for item in records]
            if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
                raise InvalidEvidenceError(
                    f"{curve}: {field} is not strictly increasing"
                )

    curve_by_key: dict[str, dict[str, Any]] = {}
    for raw in curves:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("curve"), str):
            raise InvalidEvidenceError("official curve record malformed")
        key = str(raw["curve"])
        if key in curve_by_key or key not in curve_specs:
            raise InvalidEvidenceError("official curve identity drift")
        spec = curve_specs[key]
        expected_identity = {
            "physical_family_id": expected_curve_to_family[key],
            "figure": spec.figure,
            "panel": spec.panel,
            "channel": spec.channel,
            "abscissa": spec.abscissa,
        }
        if any(raw.get(name) != value for name, value in expected_identity.items()):
            raise InvalidEvidenceError(f"{key}: official curve/PF identity drift")
        curve_by_key[key] = dict(raw)
    if set(curve_by_key) != set(curve_specs):
        raise InvalidEvidenceError("sample/official curve sets differ")

    family_by_id: dict[str, dict[str, Any]] = {}
    for raw in families:
        if not isinstance(raw, Mapping):
            raise InvalidEvidenceError("physical-family record malformed")
        family = dict(raw)
        family_id = family.get("physical_family_id")
        if not isinstance(family_id, str) or family_id in family_by_id:
            raise InvalidEvidenceError("physical-family identity/schema drift")
        expected = expected_family_by_id.get(family_id)
        if expected is None or any(
            family.get(name) != expected[name]
            for name in (
                "physical_family_id",
                "channel",
                "condition_keys",
                "official_curve_keys",
                "n_official_curves",
            )
        ):
            raise InvalidEvidenceError(f"{family_id}: authoritative PF mapping drift")
        family_by_id[family_id] = family
    if set(family_by_id) != set(expected_family_by_id):
        raise InvalidEvidenceError("physical-family identity set drift")
    alias_projection = [
        {
            "physical_family_id": item.get("physical_family_id"),
            "official_curve_keys": item.get("official_curve_keys"),
        }
        for item in aliases
        if isinstance(item, Mapping)
    ]
    expected_alias_projection = [
        {
            "physical_family_id": item["physical_family_id"],
            "official_curve_keys": item["official_curve_keys"],
        }
        for item in expected_aliases
    ]
    if _canonical_hash(alias_projection) != _canonical_hash(expected_alias_projection):
        raise InvalidEvidenceError("authoritative alias contract drift")
    return samples_by_curve, curve_by_key, family_by_id


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise InvalidEvidenceError("cannot reduce an empty sequence")
    return math.fsum(float(value) for value in values) / len(values)


def _condition_sort_key(key: str) -> tuple[float, float, float, float]:
    condition = CONFIRMED_BY_KEY.get(key)
    if condition is None:
        raise InvalidEvidenceError(f"unknown confirmed condition key {key!r}")
    return tuple(float(value) for value in condition)


def _canonical_alpha(
    coefficients: Sequence[float],
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(coefficients) != len(samples):
        raise InvalidPreregistrationError("coefficient/sample length drift")
    terms: dict[str, list[float]] = {}
    for coefficient, sample in zip(coefficients, samples):
        coefficient = float(coefficient)
        if coefficient == 0.0:
            continue
        for key_field, weight_field in (
            ("left_condition_key", "left_weight"),
            ("right_condition_key", "right_weight"),
        ):
            key = str(sample[key_field])
            _condition_sort_key(key)
            terms.setdefault(key, []).append(coefficient * float(sample[weight_field]))
    output: list[dict[str, Any]] = []
    for key in sorted(terms, key=_condition_sort_key):
        value = math.fsum(terms[key])
        if abs(value) <= CANONICAL_ZERO_TOLERANCE:
            continue
        output.append(
            {
                "condition_key": key,
                "condition": list(_condition_sort_key(key)),
                "coefficient": value,
            }
        )
    return output


def _baseline_component_state(experimental: float, baseline: float) -> str:
    error = abs(baseline - experimental)
    if error <= CONTRAST_TOLERANCE_N:
        return "PASS"
    if baseline * experimental < 0.0:
        return "REVERSED"
    if abs(baseline) < abs(experimental):
        return "UNDER"
    return "OVER"


def _contrast_record(
    contrast_id: str,
    coefficients: Sequence[float],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(coefficients) != len(samples):
        raise InvalidPreregistrationError(f"{contrast_id}: coefficient length drift")
    coefficient_sum = math.fsum(float(value) for value in coefficients)
    if abs(coefficient_sum) > 1.0e-15:
        raise InvalidPreregistrationError(f"{contrast_id}: contrast is not zero-sum")
    coefficient_l1 = math.fsum(abs(float(value)) for value in coefficients)
    measured = [float(item["measurement_N"]) for item in samples]
    model = [float(item["model_N"]) for item in samples]
    experimental = math.fsum(
        float(coefficient) * value for coefficient, value in zip(coefficients, measured)
    )
    baseline = math.fsum(
        float(coefficient) * value for coefficient, value in zip(coefficients, model)
    )
    tolerance = FORCE_TOLERANCE_N * coefficient_l1
    if not _same_number(tolerance, CONTRAST_TOLERANCE_N):
        raise InvalidPreregistrationError(
            f"{contrast_id}: expected a two-point contrast"
        )
    alpha = _canonical_alpha(coefficients, samples)
    return {
        "contrast_id": contrast_id,
        "coefficients": [
            {"measurement_index": index, "coefficient": float(value)}
            for index, value in enumerate(coefficients)
            if float(value) != 0.0
        ],
        "coefficient_l1": coefficient_l1,
        "experimental_contrast_N": experimental,
        "baseline_contrast_N": baseline,
        "contrast_tolerance_N": tolerance,
        "sign_E": 1 if experimental > 0.0 else -1,
        "baseline_state": _baseline_component_state(experimental, baseline),
        "absolute_error_N": abs(baseline - experimental),
        "excess_error_N": max(
            0.0,
            abs(baseline - experimental) - tolerance,
        ),
        "canonical_alpha": alpha,
        "effective_support_condition_keys": [item["condition_key"] for item in alpha],
    }


def _interval_max_change(
    values: Sequence[float],
    start: int,
    stop: int,
    *,
    rebound: bool,
) -> float:
    changes = [
        (float(values[right]) - float(values[left]))
        if rebound
        else (float(values[left]) - float(values[right]))
        for left in range(start, stop + 1)
        for right in range(left + 1, stop + 1)
    ]
    return max(changes, default=0.0)


def _shape_components(values: Sequence[float]) -> dict[str, Any]:
    count = len(values)
    if count < 2:
        raise InvalidEvidenceError("official curve has fewer than two measurements")
    finite = [float(value) for value in values]
    if not all(math.isfinite(value) for value in finite):
        raise InvalidEvidenceError("official curve contains non-finite measurements")
    maximum = max(finite)
    minimum = min(finite)
    maximum_indices = [
        index
        for index, value in enumerate(finite)
        if abs(value - maximum) <= CANONICAL_ZERO_TOLERANCE
    ]
    minimum_indices = [
        index
        for index, value in enumerate(finite)
        if abs(value - minimum) <= CANONICAL_ZERO_TOLERANCE
    ]
    peak_index = maximum_indices[0] if len(maximum_indices) == 1 else None
    trough_index = minimum_indices[0] if len(minimum_indices) == 1 else None
    peak_rise = finite[peak_index] - finite[0] if peak_index is not None else None
    peak_rolloff = finite[peak_index] - finite[-1] if peak_index is not None else None
    peak_pre_drawdown = (
        _interval_max_change(finite, 0, peak_index, rebound=False)
        if peak_index is not None
        else None
    )
    peak_post_rebound = (
        _interval_max_change(finite, peak_index, count - 1, rebound=True)
        if peak_index is not None
        else None
    )
    trough_fall = finite[0] - finite[trough_index] if trough_index is not None else None
    trough_recovery = (
        finite[-1] - finite[trough_index] if trough_index is not None else None
    )
    trough_pre_rebound = (
        _interval_max_change(finite, 0, trough_index, rebound=True)
        if trough_index is not None
        else None
    )
    trough_post_drawdown = (
        _interval_max_change(finite, trough_index, count - 1, rebound=False)
        if trough_index is not None
        else None
    )
    peak_eligible = bool(
        count >= 3
        and peak_index is not None
        and 0 < peak_index < count - 1
        and peak_rise is not None
        and peak_rise > CONTRAST_TOLERANCE_N
        and peak_rolloff is not None
        and peak_rolloff > CONTRAST_TOLERANCE_N
        and peak_pre_drawdown is not None
        and peak_pre_drawdown <= CONTRAST_TOLERANCE_N
        and peak_post_rebound is not None
        and peak_post_rebound <= CONTRAST_TOLERANCE_N
    )
    trough_eligible = bool(
        count >= 3
        and trough_index is not None
        and 0 < trough_index < count - 1
        and trough_fall is not None
        and trough_fall > CONTRAST_TOLERANCE_N
        and trough_recovery is not None
        and trough_recovery > CONTRAST_TOLERANCE_N
        and trough_pre_rebound is not None
        and trough_pre_rebound <= CONTRAST_TOLERANCE_N
        and trough_post_drawdown is not None
        and trough_post_drawdown <= CONTRAST_TOLERANCE_N
    )
    endpoint = finite[-1] - finite[0]
    global_drawdown = _interval_max_change(
        finite,
        0,
        count - 1,
        rebound=False,
    )
    global_rebound = _interval_max_change(
        finite,
        0,
        count - 1,
        rebound=True,
    )
    diagnostics = {
        "n_measurements": count,
        "measurement_values_N": finite,
        "near_maximum_indices": maximum_indices,
        "near_minimum_indices": minimum_indices,
        "peak_eligible": peak_eligible,
        "trough_eligible": trough_eligible,
        "peak_gate_values": {
            "unique_extremum_index": peak_index,
            "interior": (peak_index is not None and 0 < peak_index < count - 1),
            "rise_N": peak_rise,
            "rolloff_N": peak_rolloff,
            "pre_peak_max_drawdown_N": peak_pre_drawdown,
            "post_peak_max_rebound_N": peak_post_rebound,
        },
        "trough_gate_values": {
            "unique_extremum_index": trough_index,
            "interior": (trough_index is not None and 0 < trough_index < count - 1),
            "fall_N": trough_fall,
            "recovery_N": trough_recovery,
            "pre_trough_max_rebound_N": trough_pre_rebound,
            "post_trough_max_drawdown_N": trough_post_drawdown,
        },
        "endpoint_change_N": endpoint,
        "global_max_drawdown_N": global_drawdown,
        "global_max_rebound_N": global_rebound,
        "end_increasing_gate": (
            endpoint > CONTRAST_TOLERANCE_N and global_drawdown <= CONTRAST_TOLERANCE_N
        ),
        "end_decreasing_gate": (
            endpoint < -CONTRAST_TOLERANCE_N and global_rebound <= CONTRAST_TOLERANCE_N
        ),
    }
    if peak_eligible and trough_eligible:
        return {
            "status": "INELIGIBLE_COMPLEX_SHAPE",
            "shape_class": None,
            "component_definitions": [],
            "diagnostics": diagnostics,
        }
    if peak_eligible:
        assert peak_index is not None
        rise = [0.0] * count
        rolloff = [0.0] * count
        rise[0], rise[peak_index] = -1.0, 1.0
        rolloff[peak_index], rolloff[-1] = 1.0, -1.0
        return {
            "status": "CLASSIFIED",
            "shape_class": "PEAK",
            "component_definitions": [
                ("RISE", rise),
                ("ROLLOFF", rolloff),
            ],
            "extremum_nominal_index": peak_index,
            "diagnostics": diagnostics,
        }
    if trough_eligible:
        assert trough_index is not None
        fall = [0.0] * count
        recovery = [0.0] * count
        fall[0], fall[trough_index] = 1.0, -1.0
        recovery[trough_index], recovery[-1] = -1.0, 1.0
        return {
            "status": "CLASSIFIED",
            "shape_class": "TROUGH",
            "component_definitions": [
                ("FALL", fall),
                ("RECOVERY", recovery),
            ],
            "extremum_nominal_index": trough_index,
            "diagnostics": diagnostics,
        }
    coefficients = [0.0] * count
    coefficients[0], coefficients[-1] = -1.0, 1.0
    if (
        endpoint > CONTRAST_TOLERANCE_N
        and diagnostics["global_max_drawdown_N"] <= CONTRAST_TOLERANCE_N
    ):
        return {
            "status": "CLASSIFIED",
            "shape_class": "END_INCREASING",
            "component_definitions": [("END", coefficients)],
            "extremum_nominal_index": None,
            "diagnostics": diagnostics,
        }
    if (
        endpoint < -CONTRAST_TOLERANCE_N
        and diagnostics["global_max_rebound_N"] <= CONTRAST_TOLERANCE_N
    ):
        return {
            "status": "CLASSIFIED",
            "shape_class": "END_DECREASING",
            "component_definitions": [("END", coefficients)],
            "extremum_nominal_index": None,
            "diagnostics": diagnostics,
        }
    return {
        "status": "INELIGIBLE_NO_ROBUST_ZERO_SUM_SHAPE",
        "shape_class": None,
        "component_definitions": [],
        "extremum_nominal_index": None,
        "diagnostics": diagnostics,
    }


def _classify_curve(
    curve: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    channel: str,
    abscissa: str,
) -> dict[str, Any]:
    ordered_samples = sorted(
        samples,
        key=lambda item: int(item["measurement_index"]),
    )
    measured = [float(item["measurement_N"]) for item in ordered_samples]
    shape = _shape_components(measured)
    base = {
        "curve": curve,
        "physical_family_id": str(ordered_samples[0]["physical_family_id"]),
        "channel": channel,
        "abscissa": abscissa,
        "classification_status": shape["status"],
        "shape_class": shape["shape_class"],
        "extremum_nominal_index": shape.get("extremum_nominal_index"),
        "shape_diagnostics": shape["diagnostics"],
        "measurement_points": [
            {
                "measurement_index": int(item["measurement_index"]),
                "canonical_nominal_x": float(item["canonical_nominal_x"]),
                "raw_x": float(item["raw_x"]),
                "evaluation_x": float(item["evaluation_x"]),
                "measurement_N": float(item["measurement_N"]),
                "model_N": float(item["model_N"]),
                "left_condition_key": str(item["left_condition_key"]),
                "left_condition": list(
                    _condition_sort_key(str(item["left_condition_key"]))
                ),
                "left_weight": float(item["left_weight"]),
                "right_condition_key": str(item["right_condition_key"]),
                "right_condition": list(
                    _condition_sort_key(str(item["right_condition_key"]))
                ),
                "right_weight": float(item["right_weight"]),
            }
            for item in ordered_samples
        ],
    }
    if shape["status"] != "CLASSIFIED":
        return {
            **base,
            "components": [],
            "curve_severity_N": 0.0,
            "candidate_eligible": False,
            "exclusion_reason": shape["status"],
            "global_disease_id": None,
            "global_disease_identity": None,
            "alias_geometry_signature": None,
        }
    components = [
        _contrast_record(name, coefficients, ordered_samples)
        for name, coefficients in shape["component_definitions"]
    ]
    if any(
        abs(float(component["experimental_contrast_N"]))
        <= float(component["contrast_tolerance_N"])
        for component in components
    ):
        raise InvalidEvidenceError(
            f"{curve}: classified component is not strictly identifiable"
        )
    identity = {
        "channel": channel,
        "abscissa": abscissa,
        "shape_class": shape["shape_class"],
        "components": [
            {
                "component_name": component["contrast_id"],
                "sign_E": component["sign_E"],
                "baseline_state": component["baseline_state"],
            }
            for component in components
        ],
    }
    geometry = {
        "global_disease_identity": identity,
        "components": [
            {
                "component_name": component["contrast_id"],
                "canonical_nominal_indices": [
                    item["measurement_index"] for item in component["coefficients"]
                ],
                "measurement_coefficient_vector": [
                    item["coefficient"] for item in component["coefficients"]
                ],
            }
            for component in components
        ],
    }
    candidate = any(component["baseline_state"] != "PASS" for component in components)
    return {
        **base,
        "components": components,
        "curve_severity_N": _mean(
            [float(component["excess_error_N"]) for component in components]
        ),
        "candidate_eligible": candidate,
        "exclusion_reason": None if candidate else "BASELINE_COMPONENTS_ALL_PASS",
        "global_disease_id": f"GD-{_canonical_hash(identity)}",
        "global_disease_identity": identity,
        "alias_geometry_signature": geometry,
    }


def _prepare_curve_bundle(
    curve: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility helper exposing the final v3-v8 curve classifier."""

    record = _classify_curve(
        curve,
        samples,
        channel=str(samples[0].get("channel", "T")),
        abscissa=str(samples[0].get("abscissa", "synthetic")),
    )
    return {
        **record,
        "status": (
            "PREPARED"
            if record["classification_status"] == "CLASSIFIED"
            else record["classification_status"]
        ),
        "bundle_type": record["shape_class"],
        "contrasts": record["components"],
        "condition_support": sorted(
            {
                key
                for component in record["components"]
                for key in component["effective_support_condition_keys"]
            },
            key=_condition_sort_key,
        ),
    }


def _pairwise_guards(
    curve: str,
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    guards: list[dict[str, Any]] = []
    count = len(samples)
    for left in range(count):
        for right in range(left + 1, count):
            coefficients = [0.0] * count
            coefficients[left], coefficients[right] = -1.0, 1.0
            guard = _contrast_record(
                f"GUARD_{left}_{right}",
                coefficients,
                samples,
            )
            experimental = float(guard["experimental_contrast_N"])
            baseline = float(guard["baseline_contrast_N"])
            if (
                abs(experimental) > CONTRAST_TOLERANCE_N
                and abs(baseline - experimental) <= CONTRAST_TOLERANCE_N
            ):
                guards.append(
                    {
                        **guard,
                        "curve": curve,
                        "measurement_pair": [left, right],
                        "guard_identity": {
                            "official_curve_key": curve,
                            "coefficients": copy.deepcopy(guard["coefficients"]),
                        },
                    }
                )
    return guards


def _maximum_independent_sets(
    support_by_pf: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    nodes = sorted(support_by_pf)
    support = {node: set(support_by_pf[node]) for node in nodes}
    conflict = {
        node: {
            other
            for other in nodes
            if other != node and bool(support[node].intersection(support[other]))
        }
        for node in nodes
    }
    compatible = {node: set(nodes).difference({node}, conflict[node]) for node in nodes}
    maximum: list[tuple[str, ...]] = []
    maximum_size = 0

    def visit(
        clique: set[str],
        candidates: set[str],
        excluded: set[str],
    ) -> None:
        nonlocal maximum_size
        if not candidates and not excluded:
            ordered = tuple(sorted(clique))
            size = len(ordered)
            if size > maximum_size:
                maximum_size = size
                maximum.clear()
            if size == maximum_size:
                maximum.append(ordered)
            return
        pivot_pool = candidates | excluded
        pivot = (
            max(
                sorted(pivot_pool),
                key=lambda node: len(candidates & compatible[node]),
            )
            if pivot_pool
            else None
        )
        branch_nodes = sorted(
            candidates if pivot is None else candidates.difference(compatible[pivot])
        )
        for node in branch_nodes:
            visit(
                clique | {node},
                candidates & compatible[node],
                excluded & compatible[node],
            )
            candidates.remove(node)
            excluded.add(node)

    visit(set(), set(nodes), set())
    unique_maximum = sorted({item for item in maximum})
    edges = [
        [left, right]
        for index, left in enumerate(nodes)
        for right in nodes[index + 1 :]
        if right in conflict[left]
    ]
    return {
        "evaluation_status": "EVALUATED",
        "nodes": nodes,
        "edges": edges,
        "support_by_physical_family": {
            node: sorted(support[node], key=_condition_sort_key) for node in nodes
        },
        "max_pairwise_disjoint_pf_count": maximum_size,
        "all_maximum_disjoint_pf_sets": [list(item) for item in unique_maximum],
    }


def _build_disease_ledgers(
    classified: Sequence[Mapping[str, Any]],
    family_by_id: Mapping[str, Mapping[str, Any]],
    *,
    excluded_pf: str | None = None,
) -> dict[str, Any]:
    by_curve = {str(item["curve"]): item for item in classified}
    pre: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    for record in sorted(classified, key=lambda item: str(item["curve"])):
        if record["physical_family_id"] == excluded_pf:
            continue
        if not record["candidate_eligible"]:
            exclusions.append(
                {
                    "scope": "official_curve",
                    "curve": record["curve"],
                    "physical_family_id": record["physical_family_id"],
                    "reason": record["exclusion_reason"],
                }
            )
            continue
        disease_id = str(record["global_disease_id"])
        group = pre.setdefault(
            disease_id,
            {
                "global_disease_id": disease_id,
                "global_disease_identity": copy.deepcopy(
                    record["global_disease_identity"]
                ),
                "physical_family_occurrences": {},
            },
        )
        group["physical_family_occurrences"].setdefault(
            record["physical_family_id"], []
        ).append(str(record["curve"]))

    alias_pair_ledger: list[dict[str, Any]] = []
    consensus: dict[str, dict[str, Any]] = {}
    for family_id in sorted(family_by_id):
        if family_id == excluded_pf:
            continue
        curve_keys = list(family_by_id[family_id]["official_curve_keys"])
        records = [by_curve[key] for key in curve_keys]
        pair_results: list[dict[str, Any]] = []
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                same = bool(
                    left["candidate_eligible"]
                    and right["candidate_eligible"]
                    and _canonical_hash(left["alias_geometry_signature"])
                    == _canonical_hash(right["alias_geometry_signature"])
                )
                pair = {
                    "physical_family_id": family_id,
                    "left_curve": left["curve"],
                    "right_curve": right["curve"],
                    "consistent": same,
                    "left_global_disease_id": left["global_disease_id"],
                    "right_global_disease_id": right["global_disease_id"],
                }
                pair_results.append(pair)
                alias_pair_ledger.append(pair)
        family_consensus = bool(
            records
            and all(record["candidate_eligible"] for record in records)
            and (not pair_results or all(item["consistent"] for item in pair_results))
            and len({record["global_disease_id"] for record in records}) == 1
        )
        if not family_consensus:
            if any(record["candidate_eligible"] for record in records):
                exclusions.append(
                    {
                        "scope": "physical_family",
                        "physical_family_id": family_id,
                        "official_curve_keys": curve_keys,
                        "reason": "ALIAS_WITHDRAWN",
                    }
                )
            continue
        disease_id = str(records[0]["global_disease_id"])
        positive_support = sorted(
            {
                key
                for record in records
                for component in record["components"]
                for key in component["effective_support_condition_keys"]
            },
            key=_condition_sort_key,
        )
        pf_record = {
            "physical_family_id": family_id,
            "official_curve_keys": curve_keys,
            "curve_severity_N": {
                record["curve"]: float(record["curve_severity_N"])
                for record in sorted(records, key=lambda item: str(item["curve"]))
            },
            "pf_severity_N": _mean(
                [
                    float(record["curve_severity_N"])
                    for record in sorted(records, key=lambda item: str(item["curve"]))
                ]
            ),
            "positive_support_condition_keys": positive_support,
            "alias_consensus": True,
        }
        group = consensus.setdefault(
            disease_id,
            {
                "global_disease_id": disease_id,
                "global_disease_identity": copy.deepcopy(
                    records[0]["global_disease_identity"]
                ),
                "physical_families": [],
            },
        )
        group["physical_families"].append(pf_record)

    pre_ledger = [
        {
            **{
                key: value
                for key, value in group.items()
                if key != "physical_family_occurrences"
            },
            "physical_family_occurrences": [
                {
                    "physical_family_id": family_id,
                    "official_curve_keys": sorted(curves),
                }
                for family_id, curves in sorted(
                    group["physical_family_occurrences"].items()
                )
            ],
        }
        for _, group in sorted(pre.items())
    ]
    consensus_ledger: list[dict[str, Any]] = []
    for disease_id, group in sorted(consensus.items()):
        families = sorted(
            group["physical_families"],
            key=lambda item: item["physical_family_id"],
        )
        support_analysis = _maximum_independent_sets(
            {
                item["physical_family_id"]: item["positive_support_condition_keys"]
                for item in families
            }
        )
        consensus_ledger.append(
            {
                **{
                    key: value
                    for key, value in group.items()
                    if key != "physical_families"
                },
                "physical_families": families,
                "support_conflict_graph": support_analysis,
            }
        )
    return {
        "pre_alias": pre_ledger,
        "consensus": consensus_ledger,
        "alias_pair_ledger": alias_pair_ledger,
        "exclusion_ledger": exclusions,
    }


def _connected_components(graph: Mapping[str, Any]) -> list[list[str]]:
    nodes = [str(item) for item in graph["nodes"]]
    adjacency = {node: set() for node in nodes}
    for left, right in graph["edges"]:
        adjacency[str(left)].add(str(right))
        adjacency[str(right)].add(str(left))
    remaining = set(nodes)
    output: list[list[str]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node].difference(component), reverse=True))
        remaining.difference_update(component)
        output.append(sorted(component))
    return output


def _selection_audit_diagnostics(
    consensus: Sequence[Mapping[str, Any]],
    classified: Sequence[Mapping[str, Any]],
    samples_by_curve: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    classified_by_curve = {str(item["curve"]): item for item in classified}
    diagnostics: list[dict[str, Any]] = []
    for candidate in sorted(
        consensus,
        key=lambda item: str(item["global_disease_id"]),
    ):
        families = sorted(
            candidate["physical_families"],
            key=lambda item: str(item["physical_family_id"]),
        )
        severity_by_pf = {
            str(item["physical_family_id"]): float(item["pf_severity_N"])
            for item in families
        }
        components = _connected_components(candidate["support_conflict_graph"])
        cluster_means = [
            _mean([severity_by_pf[family_id] for family_id in component])
            for component in components
        ]
        curve_keys = sorted(
            {curve for family in families for curve in family["official_curve_keys"]}
        )
        curve_severities = [
            float(classified_by_curve[curve]["curve_severity_N"])
            for curve in curve_keys
        ]
        centered_absolute_residuals: list[float] = []
        point_count = 0
        for curve in curve_keys:
            records = sorted(
                samples_by_curve[curve],
                key=lambda item: int(item["measurement_index"]),
            )
            residuals = [
                float(item["model_N"]) - float(item["measurement_N"])
                for item in records
            ]
            center = _mean(residuals)
            centered_absolute_residuals.extend(
                abs(value - center) for value in residuals
            )
            point_count += len(records)
        severities = sorted(severity_by_pf.values())
        median = (
            severities[len(severities) // 2]
            if len(severities) % 2
            else _mean(severities[len(severities) // 2 - 1 : len(severities) // 2 + 1])
        )
        diagnostics.append(
            {
                "global_disease_id": candidate["global_disease_id"],
                "support_cluster_equal_mean_N": _mean(cluster_means),
                "support_cluster_status": (
                    "DEGENERATE_WITH_PF_EQUAL_MEAN"
                    if len({len(component) for component in components}) <= 1
                    else "AUDIT_ONLY"
                ),
                "pf_median_N": median,
                "pf_median_status": (
                    "DEGENERATE_FOR_N_LT_3" if len(families) < 3 else "AUDIT_ONLY"
                ),
                "official_curve_equal_contrast_excess_N": _mean(curve_severities),
                "centered_point_weighted_absolute_residual_N": _mean(
                    centered_absolute_residuals
                ),
                "physical_family_count": len(families),
                "official_curve_count": len(curve_keys),
                "raw_point_count": point_count,
                "support_overlap_edges": copy.deepcopy(
                    candidate["support_conflict_graph"]["edges"]
                ),
                "all_maximum_disjoint_pf_sets": copy.deepcopy(
                    candidate["support_conflict_graph"]["all_maximum_disjoint_pf_sets"]
                ),
                "role": "AUDIT_ONLY_NOT_USED_FOR_SELECTION",
            }
        )
    return diagnostics


def _ranking_view(
    eligible: Sequence[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    for candidate in sorted(
        eligible,
        key=lambda item: str(item["global_disease_id"]),
    ):
        severities = [
            float(item["pf_severity_N"]) for item in candidate["physical_families"]
        ]
        score = _mean(severities) if view == "PF_EQUAL_MEAN" else min(severities)
        scores.append(
            {
                "global_disease_id": candidate["global_disease_id"],
                "score_N": score,
            }
        )
    maximum = max(item["score_N"] for item in scores)
    argmax = [
        item["global_disease_id"]
        for item in scores
        if abs(float(item["score_N"]) - maximum) <= CANONICAL_ZERO_TOLERANCE
    ]
    return {
        "scores": scores,
        "maximum_score_N": maximum,
        "argmax_global_disease_ids": argmax,
        "winner_global_disease_id": argmax[0] if len(argmax) == 1 else None,
        "runner_up_score_N": None,
        "winner_margin_N": None,
        "runner_up_status": (
            "PENDING"
            if len(scores) > 1 and len(argmax) == 1
            else "VACUOUS_SINGLE_ELIGIBLE_DISEASE"
            if len(scores) == 1
            else "NOT_DEFINED_NONUNIQUE_ARGMAX"
        ),
    }


def _main_selection_gate(
    ledgers: Mapping[str, Any],
    *,
    minimum_independent_families: int,
) -> dict[str, Any]:
    pre = list(ledgers["pre_alias"])
    consensus = list(ledgers["consensus"])
    if not pre:
        return {
            "gate_status": "NO_DECISION_NO_PRE_REPLICATION_CANDIDATE",
            "winner_global_disease_id": None,
            "rankings": {"evaluation_status": "NOT_EVALUATED_EMPTY_ELIGIBLE_SET"},
        }
    eligible = [
        item
        for item in consensus
        if item["support_conflict_graph"]["max_pairwise_disjoint_pf_count"]
        >= minimum_independent_families
    ]
    if not eligible:
        return {
            "gate_status": "NO_DECISION_NO_INDEPENDENT_REPLICATION",
            "winner_global_disease_id": None,
            "rankings": {"evaluation_status": "NOT_EVALUATED_EMPTY_ELIGIBLE_SET"},
        }
    views = {
        name: _ranking_view(eligible, view=name)
        for name in ("PF_EQUAL_MEAN", "PF_REPLICATION_FLOOR")
    }
    rankings: dict[str, Any] = {
        "evaluation_status": "EVALUATED",
        "eligible_global_disease_ids": sorted(
            str(item["global_disease_id"]) for item in eligible
        ),
        "views": views,
    }
    if any(len(view["argmax_global_disease_ids"]) != 1 for view in views.values()):
        return {
            "gate_status": "NO_DECISION_DISEASE_TIE",
            "winner_global_disease_id": None,
            "rankings": rankings,
        }
    winners = {str(view["winner_global_disease_id"]) for view in views.values()}
    if len(winners) != 1:
        return {
            "gate_status": "NO_DECISION_VIEW_DISAGREEMENT",
            "winner_global_disease_id": None,
            "rankings": rankings,
        }
    winner = next(iter(winners))
    insufficient_margin = False
    if len(eligible) > 1:
        for view in views.values():
            runner = max(
                float(item["score_N"])
                for item in view["scores"]
                if item["global_disease_id"] != winner
            )
            margin = float(view["maximum_score_N"]) - runner
            view["runner_up_score_N"] = runner
            view["winner_margin_N"] = margin
            view["runner_up_status"] = "EVALUATED"
            if margin <= RANK_MARGIN_N:
                insufficient_margin = True
    if insufficient_margin:
        return {
            "gate_status": "NO_DECISION_INSUFFICIENT_RANK_MARGIN",
            "winner_global_disease_id": winner,
            "rankings": rankings,
        }
    return {
        "gate_status": "GATE_QUALIFIED",
        "winner_global_disease_id": winner,
        "rankings": rankings,
    }


def _shadow_status(gate: Mapping[str, Any], expected_winner: str) -> str:
    status = str(gate["gate_status"])
    mapping = {
        "NO_DECISION_NO_PRE_REPLICATION_CANDIDATE": (
            "SHADOW_NO_PRE_REPLICATION_CANDIDATE"
        ),
        "NO_DECISION_NO_INDEPENDENT_REPLICATION": ("SHADOW_NO_INDEPENDENT_REPLICATION"),
        "NO_DECISION_DISEASE_TIE": "SHADOW_DISEASE_TIE",
        "NO_DECISION_VIEW_DISAGREEMENT": "SHADOW_VIEW_DISAGREEMENT",
        "NO_DECISION_INSUFFICIENT_RANK_MARGIN": ("SHADOW_INSUFFICIENT_RANK_MARGIN"),
    }
    if status != "GATE_QUALIFIED":
        return mapping[status]
    if gate["winner_global_disease_id"] != expected_winner:
        return "SHADOW_WINNER_CHANGED"
    return "SHADOW_PASS"


def _derive_active_disease_spec(
    fingerprint: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    samples_by_curve, curve_by_key, family_by_id = _fingerprint_maps(fingerprint)
    classified = [
        _classify_curve(
            curve,
            samples_by_curve[curve],
            channel=str(curve_by_key[curve]["channel"]),
            abscissa=str(curve_by_key[curve]["abscissa"]),
        )
        for curve in sorted(curve_by_key)
    ]
    ledgers = _build_disease_ledgers(classified, family_by_id)
    main = _main_selection_gate(
        ledgers,
        minimum_independent_families=MIN_INDEPENDENT_FAMILIES,
    )
    base = {
        "input_bundle_id": receipt.get("baseline_bundle_id"),
        "scorecard_sha256": receipt.get("outputs", {})
        .get("scorecard", {})
        .get("sha256"),
        "fingerprint_sha256": receipt.get("outputs", {})
        .get("fingerprint", {})
        .get("sha256"),
        "thresholds": {
            "force_tolerance_N": FORCE_TOLERANCE_N,
            "contrast_tolerance_N": CONTRAST_TOLERANCE_N,
            "rank_margin_N": RANK_MARGIN_N,
            "comparison_rule": "strictly_greater_than_for_identifiability_and_margin",
        },
        "ledger_stages": {
            "classified": classified,
            "pre_alias": ledgers["pre_alias"],
            "consensus": ledgers["consensus"],
        },
        "classification_ledger_42": classified,
        "candidate_ledger": ledgers["consensus"],
        "exclusion_ledger": ledgers["exclusion_ledger"],
        "alias_pair_ledger": ledgers["alias_pair_ledger"],
        "rankings": main["rankings"],
        "audit_diagnostics": _selection_audit_diagnostics(
            ledgers["consensus"],
            classified,
            samples_by_curve,
        ),
        "claim_decision": "NO_DECISION",
    }
    if main["gate_status"] != "GATE_QUALIFIED":
        return {
            "status": main["gate_status"],
            **base,
            "all_triggered_reasons": [main["gate_status"]],
            "shadow_lopo": {
                "evaluation_status": (
                    "NOT_RUN_NO_ELIGIBLE_MAIN_DISEASE"
                    if main["gate_status"]
                    in {
                        "NO_DECISION_NO_PRE_REPLICATION_CANDIDATE",
                        "NO_DECISION_NO_INDEPENDENT_REPLICATION",
                    }
                    else "NOT_RUN_MAIN_DECISION_GATE_FAILED"
                )
            },
            "reason": main["gate_status"],
        }

    winner_id = str(main["winner_global_disease_id"])
    winner = next(
        item for item in ledgers["consensus"] if item["global_disease_id"] == winner_id
    )
    winner_pf_ids = [item["physical_family_id"] for item in winner["physical_families"]]
    shadow_records: list[dict[str, Any]] = []
    for removed_pf in winner_pf_ids:
        shadow_classified = [
            copy.deepcopy(item)
            for item in classified
            if item["physical_family_id"] != removed_pf
        ]
        shadow_ledgers = _build_disease_ledgers(
            classified,
            family_by_id,
            excluded_pf=removed_pf,
        )
        shadow_gate = _main_selection_gate(
            shadow_ledgers,
            minimum_independent_families=1,
        )
        shadow_records.append(
            {
                "removed_physical_family_id": removed_pf,
                "shadow_local_status": _shadow_status(
                    shadow_gate,
                    winner_id,
                ),
                "winner_global_disease_id": shadow_gate["winner_global_disease_id"],
                "rankings": shadow_gate["rankings"],
                "ledger_stages": {
                    "classified": shadow_classified,
                    "pre_alias": shadow_ledgers["pre_alias"],
                    "consensus": shadow_ledgers["consensus"],
                },
                "alias_pair_ledger": shadow_ledgers["alias_pair_ledger"],
                "exclusion_ledger": shadow_ledgers["exclusion_ledger"],
            }
        )
    shadow_failed = any(
        item["shadow_local_status"] != "SHADOW_PASS" for item in shadow_records
    )
    if shadow_failed:
        return {
            "status": "NO_DECISION_LEAVE_ONE_PF_SENSITIVE",
            **base,
            "all_triggered_reasons": ["NO_DECISION_LEAVE_ONE_PF_SENSITIVE"],
            "shadow_lopo": {
                "evaluation_status": "EVALUATED",
                "records": shadow_records,
            },
            "reason": "NO_DECISION_LEAVE_ONE_PF_SENSITIVE",
        }

    selected_curve_keys = sorted(
        {
            curve
            for family in winner["physical_families"]
            for curve in family["official_curve_keys"]
        }
    )
    selected_curves: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    guards: list[dict[str, Any]] = []
    classified_by_curve = {str(item["curve"]): item for item in classified}
    for curve in selected_curve_keys:
        selected = copy.deepcopy(classified_by_curve[curve])
        curve_guards = _pairwise_guards(curve, samples_by_curve[curve])
        selected["guard_contrasts"] = curve_guards
        selected_curves.append(selected)
        contrasts.extend(
            {
                "curve": curve,
                **copy.deepcopy(component),
            }
            for component in selected["components"]
        )
        guards.extend(copy.deepcopy(curve_guards))
    support_keys = sorted(
        {
            key
            for family in winner["physical_families"]
            for key in family["positive_support_condition_keys"]
        },
        key=_condition_sort_key,
    )
    return {
        "status": "ACTIVE_DISEASE_FROZEN",
        **base,
        "disease_id": winner_id,
        "global_disease_id": winner_id,
        "global_disease_identity": copy.deepcopy(winner["global_disease_identity"]),
        "channel": winner["global_disease_identity"]["channel"],
        "abscissa": winner["global_disease_identity"]["abscissa"],
        "shape_class": winner["global_disease_identity"]["shape_class"],
        "component_signature": copy.deepcopy(
            winner["global_disease_identity"]["components"]
        ),
        "support_physical_family_ids": winner_pf_ids,
        "support_official_curve_keys": selected_curve_keys,
        "selected_curves": selected_curves,
        "contrasts": contrasts,
        "guard_contrasts": guards,
        "positive_support_condition_keys": support_keys,
        "support_conflict_graph": copy.deepcopy(winner["support_conflict_graph"]),
        "max_pairwise_disjoint_pf_count": winner["support_conflict_graph"][
            "max_pairwise_disjoint_pf_count"
        ],
        "all_maximum_disjoint_pf_sets": copy.deepcopy(
            winner["support_conflict_graph"]["all_maximum_disjoint_pf_sets"]
        ),
        "shadow_lopo": {
            "evaluation_status": "EVALUATED",
            "records": shadow_records,
        },
        "all_triggered_reasons": [],
        "reason": "NODE_ATTRIBUTION_REQUIRED",
    }


def _contains_forbidden_disease_content(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(
                token in normalized
                for token in (
                    "contribution",
                    "candidate_formula",
                    "literature",
                    "parameter",
                )
            ):
                return True
            if _contains_forbidden_disease_content(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_forbidden_disease_content(item) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized.upper() in {"N2", "N3"} or any(
        token in normalized
        for token in (
            "claim_contributions",
            "contributions_path",
            "contributions_sha256",
        )
    )


def _validate_selector_payload_metadata(
    disease: Mapping[str, Any],
    projection: Mapping[str, Any],
    *,
    fingerprint_path: Path,
) -> None:
    if (
        disease.get("schema_version") != SCHEMA_VERSION
        or disease.get("artifact_type") != "fig171819_active_disease_selection"
        or disease.get("production_execution_authorized") is not False
        or disease.get("receipt_envelope_required") is not True
        or _canonical_hash(disease.get("protocol_stack"))
        != _canonical_hash(_protocol_stack())
    ):
        raise InvalidPreregistrationError("selector payload metadata drift")
    provenance = disease.get("provenance")
    if not isinstance(provenance, Mapping) or _contains_forbidden_disease_content(
        provenance
    ):
        raise InvalidPreregistrationError(
            "selector provenance leaks forbidden claim/contribution metadata"
        )
    expected_provenance = {
        "fingerprint": {
            "path": _display_path(fingerprint_path),
            "sha256": _sha256_file(fingerprint_path),
        },
        "baseline_receipt": {
            "sha256": projection["baseline_receipt_sha256"],
        },
        "active_disease_protocol": {
            "path": _display_path(ACTIVE_DISEASE_PROTOCOL),
            "sha256": _sha256_file(ACTIVE_DISEASE_PROTOCOL),
        },
        "active_disease_protocol_stack": copy.deepcopy(
            _protocol_stack()["active_disease"]
        ),
        "generator": {
            "path": _display_path(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
    }
    if _canonical_hash(provenance) != _canonical_hash(expected_provenance):
        raise InvalidPreregistrationError("selector provenance identity drift")


def _validate_active_disease_spec(
    disease: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _derive_active_disease_spec(fingerprint, receipt)
    if expected.get("status") != "ACTIVE_DISEASE_FROZEN":
        raise InvalidPreregistrationError(
            f"active disease is not uniquely frozen: {expected.get('status')}"
        )
    if _contains_forbidden_disease_content(
        {name: disease.get(name) for name in expected}
    ):
        raise InvalidPreregistrationError(
            "disease spec contains forbidden claim/candidate content"
        )
    allowed_extras = {
        "schema_version",
        "artifact_type",
        "provenance",
        "protocol_stack",
        "contribution_free_receipt_projection",
        "production_execution_authorized",
        "receipt_envelope_required",
    }
    unknown = set(disease).difference(expected, allowed_extras)
    if unknown:
        raise InvalidPreregistrationError(
            f"disease spec contains unknown fields: {sorted(unknown)}"
        )
    for name, expected_value in expected.items():
        if _canonical_hash(disease.get(name)) != _canonical_hash(expected_value):
            raise InvalidPreregistrationError(
                f"disease spec {name} differs from independent reconstruction"
            )
    if len(expected["classification_ledger_42"]) != EXPECTED_CURVES:
        raise InvalidPreregistrationError("disease spec classification ledger drift")
    for item in expected["contrasts"]:
        if (
            abs(
                math.fsum(
                    float(coefficient["coefficient"])
                    for coefficient in item["coefficients"]
                )
            )
            > 1.0e-15
        ):
            raise InvalidPreregistrationError("disease spec contrast is not zero-sum")
    return expected


def _prepare_contract_strict(
    fingerprint: Mapping[str, Any],
    disease_spec: Mapping[str, Any],
    *,
    fingerprint_path: Path,
    disease_spec_path: Path,
) -> dict[str, Any]:
    """Freeze the G1 winner without receiving contribution identity metadata."""

    _validate_frozen_roots()
    _require_payload_file_match(
        fingerprint,
        fingerprint_path,
        label="fingerprint",
        error_type=InvalidPreregistrationError,
    )
    _require_payload_file_match(
        disease_spec,
        disease_spec_path,
        label="disease spec",
        error_type=InvalidPreregistrationError,
    )
    try:
        disease_payload, selector_envelope = _unwrap_scientific_payload(
            disease_spec,
            stage="select-disease",
            required=True,
        )
    except InvalidEvidenceError as exc:
        raise InvalidPreregistrationError(str(exc)) from exc
    contribution_free_projection = disease_payload.get(
        "contribution_free_receipt_projection"
    )
    if _contains_forbidden_disease_content(
        {
            key: value
            for key, value in disease_payload.items()
            if key != "contribution_free_receipt_projection"
        }
    ):
        raise InvalidPreregistrationError(
            "selector input contains forbidden claim/contribution metadata"
        )
    projection = _validate_contribution_free_projection(
        contribution_free_projection,
        fingerprint,
        fingerprint_path=fingerprint_path,
    )
    _validate_selector_payload_metadata(
        disease_payload,
        projection,
        fingerprint_path=fingerprint_path,
    )
    receipt_projection = _projection_as_receipt(projection)
    samples_by_curve, _, family_by_id = _fingerprint_maps(fingerprint)
    disease = _validate_active_disease_spec(
        disease_payload,
        fingerprint,
        receipt_projection,
    )
    selected_by_curve = {
        str(item["curve"]): copy.deepcopy(item) for item in disease["selected_curves"]
    }
    family_records: list[dict[str, Any]] = []
    all_positive_support: set[str] = set()
    for family_id in disease["support_physical_family_ids"]:
        source = family_by_id[family_id]
        curves = [selected_by_curve[curve] for curve in source["official_curve_keys"]]
        positive_support = sorted(
            {
                key
                for curve in curves
                for component in curve["components"]
                for key in component["effective_support_condition_keys"]
            },
            key=_condition_sort_key,
        )
        all_positive_support.update(positive_support)
        guards = [
            copy.deepcopy(guard)
            for curve in curves
            for guard in curve["guard_contrasts"]
        ]
        independently_recomputed_guards = [
            guard
            for curve in source["official_curve_keys"]
            for guard in _pairwise_guards(curve, samples_by_curve[curve])
        ]
        if _canonical_hash(guards) != _canonical_hash(independently_recomputed_guards):
            raise InvalidPreregistrationError(
                f"{family_id}: selected all-pairs guard drift"
            )
        family_records.append(
            {
                "physical_family_id": family_id,
                "channel": source["channel"],
                "official_curve_keys": list(source["official_curve_keys"]),
                "n_official_curves": len(source["official_curve_keys"]),
                "curves": curves,
                "positive_support_condition_keys": positive_support,
                "pairwise_guards": guards,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fig171819_parent_attribution_preregistration",
        "status": "PREPARED",
        "evidence_scope": EVIDENCE_CONFIRMED,
        "causal_status": "HYPOTHESIS_ONLY",
        "claim_writeback_allowed": False,
        "confirmed_contract": {
            "curves": EXPECTED_CURVES,
            "measurement_samples": EXPECTED_SAMPLES,
            "solver_conditions": EXPECTED_CONDITIONS,
            "physical_families": EXPECTED_PHYSICAL_FAMILIES,
            "alias_groups": EXPECTED_ALIAS_GROUPS,
            "conditional_fig19_cd_allowed": False,
        },
        "thresholds": {
            "force_tolerance_N": FORCE_TOLERANCE_N,
            "contrast_tolerance_N": CONTRAST_TOLERANCE_N,
            "rank_margin_N": RANK_MARGIN_N,
            "minimum_independent_physical_families": (MIN_INDEPENDENT_FAMILIES),
        },
        "eligible_parent_nodes": list(ELIGIBLE_PARENTS),
        "parent_channel_inventory": {
            "N2": ["separation", "profile_drag"],
            "N3": ["ds_vortex", "vortex_normal"],
        },
        "active_disease": {
            "active_disease_id": disease["global_disease_id"],
            "global_disease_identity": copy.deepcopy(
                disease["global_disease_identity"]
            ),
            "channel": disease["channel"],
            "abscissa": disease["abscissa"],
            "shape_class": disease["shape_class"],
            "component_signature": copy.deepcopy(disease["component_signature"]),
            "physical_family_ids": disease["support_physical_family_ids"],
            "official_curve_keys": disease["support_official_curve_keys"],
        },
        "active_disease_spec": copy.deepcopy(disease),
        "active_disease_sha256": _canonical_hash(disease),
        "baseline_bundle_id": projection["baseline_bundle_id"],
        "contribution_free_baseline_projection": copy.deepcopy(projection),
        "families": family_records,
        "positive_support_condition_keys": sorted(
            all_positive_support,
            key=_condition_sort_key,
        ),
        "positive_support_conflict_graph": copy.deepcopy(
            disease["support_conflict_graph"]
        ),
        "all_maximum_disjoint_pf_sets": copy.deepcopy(
            disease["all_maximum_disjoint_pf_sets"]
        ),
        "alias_rule": "all unordered aliases must share the frozen geometry",
        "guard_rule": (
            "all measurement-pair guards remain separate obligations; "
            "guard support never augments positive restoration support"
        ),
        "protocol_stack": _protocol_stack(),
        "selector_execution_envelope": selector_envelope,
        "production_execution_authorized": False,
        "receipt_envelope_required": True,
        "provenance": {
            "fingerprint": {
                "path": _display_path(fingerprint_path),
                "sha256": _sha256_file(fingerprint_path),
            },
            "disease_spec": {
                "path": _display_path(disease_spec_path),
                "sha256": _sha256_file(disease_spec_path),
                "size_bytes": disease_spec_path.stat().st_size,
            },
            "baseline_receipt_projection_source": {
                "sha256": projection["baseline_receipt_sha256"],
            },
            "protocol_stack": _protocol_stack(),
            "confirmed_scorer": {
                "path": _display_path(CONFIRMED_COMPARE_SOURCE),
                "sha256": _sha256_file(CONFIRMED_COMPARE_SOURCE),
            },
            "postprocess_authorization": {
                "path": _display_path(AUTHORIZED_POSTPROCESS_AUTHORIZATION),
                "sha256": _sha256_file(AUTHORIZED_POSTPROCESS_AUTHORIZATION),
            },
            "generator": {
                "path": _display_path(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
    }


def prepare_contract(
    fingerprint: Mapping[str, Any],
    disease_spec: Mapping[str, Any],
    *,
    fingerprint_path: Path,
    disease_spec_path: Path,
) -> dict[str, Any]:
    """Fail-closed public Prepare stage."""

    try:
        return _prepare_contract_strict(
            fingerprint,
            disease_spec,
            fingerprint_path=fingerprint_path,
            disease_spec_path=disease_spec_path,
        )
    except (InvalidEvidenceError, InvalidPreregistrationError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "fig171819_parent_attribution_preregistration",
            "status": "INVALID_EVIDENCE",
            "failed_gates": [str(exc)],
            "causal_status": "HYPOTHESIS_ONLY",
            "claim_decision": "NO_DECISION",
            "claim_writeback_allowed": False,
            "protocol_stack": _protocol_stack(),
            "production_execution_authorized": False,
            "receipt_envelope_required": True,
        }


def _select_active_disease_strict(
    fingerprint: Mapping[str, Any],
    baseline_receipt: Mapping[str, Any],
    *,
    fingerprint_path: Path,
    baseline_receipt_path: Path,
) -> dict[str, Any]:
    """Deterministically select the G1 disease without contribution access."""

    _require_payload_file_match(
        fingerprint,
        fingerprint_path,
        label="fingerprint",
        error_type=InvalidPreregistrationError,
    )
    metadata = _validate_baseline_receipt_metadata(
        baseline_receipt,
        fingerprint,
        receipt_path=baseline_receipt_path,
        fingerprint_path=fingerprint_path,
    )
    selected = _derive_active_disease_spec(fingerprint, baseline_receipt)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fig171819_active_disease_selection",
        **selected,
        "protocol_stack": _protocol_stack(),
        "contribution_free_receipt_projection": (
            _contribution_free_receipt_projection(
                baseline_receipt,
                metadata,
                receipt_path=baseline_receipt_path,
            )
        ),
        "production_execution_authorized": False,
        "receipt_envelope_required": True,
        "provenance": {
            "fingerprint": {
                "path": _display_path(fingerprint_path),
                "sha256": _sha256_file(fingerprint_path),
            },
            "baseline_receipt": {
                "sha256": _sha256_file(baseline_receipt_path),
            },
            "active_disease_protocol": {
                "path": _display_path(ACTIVE_DISEASE_PROTOCOL),
                "sha256": _sha256_file(ACTIVE_DISEASE_PROTOCOL),
            },
            "active_disease_protocol_stack": copy.deepcopy(
                _protocol_stack()["active_disease"]
            ),
            "generator": {
                "path": _display_path(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
    }


def select_active_disease(
    fingerprint: Mapping[str, Any],
    baseline_receipt: Mapping[str, Any],
    *,
    fingerprint_path: Path,
    baseline_receipt_path: Path,
) -> dict[str, Any]:
    """Fail-closed public G1 selector."""

    try:
        return _select_active_disease_strict(
            fingerprint,
            baseline_receipt,
            fingerprint_path=fingerprint_path,
            baseline_receipt_path=baseline_receipt_path,
        )
    except (InvalidEvidenceError, InvalidPreregistrationError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "fig171819_active_disease_selection",
            "status": "INVALID_EVIDENCE",
            "failed_gates": [str(exc)],
            "claim_decision": "NO_DECISION",
            "rankings": {
                "evaluation_status": "NOT_EVALUATED_UPSTREAM_FAILURE",
            },
            "shadow_lopo": {
                "evaluation_status": "NOT_RUN_UPSTREAM_FAILURE",
            },
            "protocol_stack": _protocol_stack(),
            "production_execution_authorized": False,
            "receipt_envelope_required": True,
        }


def _validate_prereg(
    prereg: Mapping[str, Any],
    baseline_receipt: Mapping[str, Any],
    *,
    prereg_path: Path,
    fingerprint_path: Path,
    baseline_receipt_path: Path,
) -> None:
    _validate_frozen_roots()
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "evidence_scope",
        "causal_status",
        "claim_writeback_allowed",
        "confirmed_contract",
        "thresholds",
        "eligible_parent_nodes",
        "parent_channel_inventory",
        "active_disease",
        "active_disease_spec",
        "active_disease_sha256",
        "baseline_bundle_id",
        "contribution_free_baseline_projection",
        "families",
        "positive_support_condition_keys",
        "positive_support_conflict_graph",
        "all_maximum_disjoint_pf_sets",
        "alias_rule",
        "guard_rule",
        "protocol_stack",
        "selector_execution_envelope",
        "production_execution_authorized",
        "receipt_envelope_required",
        "provenance",
    }
    if (
        set(prereg) != expected_fields
        or prereg.get("schema_version") != SCHEMA_VERSION
        or prereg.get("artifact_type") != "fig171819_parent_attribution_preregistration"
        or prereg.get("status") != "PREPARED"
        or prereg.get("evidence_scope") != EVIDENCE_CONFIRMED
        or prereg.get("causal_status") != "HYPOTHESIS_ONLY"
        or prereg.get("claim_writeback_allowed") is not False
        or prereg.get("production_execution_authorized") is not False
        or prereg.get("receipt_envelope_required") is not True
    ):
        raise InvalidPreregistrationError("preregistration identity/status drift")
    if prereg.get("eligible_parent_nodes") != list(ELIGIBLE_PARENTS):
        raise InvalidPreregistrationError("eligible parent set drift")
    if _canonical_hash(prereg.get("protocol_stack")) != _canonical_hash(
        _protocol_stack()
    ):
        raise InvalidPreregistrationError("preregistered protocol stack drift")
    selector_envelope = prereg.get("selector_execution_envelope")
    if not isinstance(selector_envelope, Mapping):
        raise InvalidPreregistrationError(
            "preregistration lacks verified selector execution envelope"
        )
    try:
        _validate_execution_envelope(
            selector_envelope,
            stage="select-disease",
        )
    except InvalidEvidenceError as exc:
        raise InvalidPreregistrationError(str(exc)) from exc
    if prereg.get("confirmed_contract") != {
        "curves": EXPECTED_CURVES,
        "measurement_samples": EXPECTED_SAMPLES,
        "solver_conditions": EXPECTED_CONDITIONS,
        "physical_families": EXPECTED_PHYSICAL_FAMILIES,
        "alias_groups": EXPECTED_ALIAS_GROUPS,
        "conditional_fig19_cd_allowed": False,
    }:
        raise InvalidPreregistrationError("preregistered confirmed contract drift")
    if prereg.get("parent_channel_inventory") != {
        "N2": ["separation", "profile_drag"],
        "N3": ["ds_vortex", "vortex_normal"],
    }:
        raise InvalidPreregistrationError("parent channel inventory drift")
    thresholds = prereg.get("thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or thresholds.get("force_tolerance_N") != FORCE_TOLERANCE_N
        or thresholds.get("contrast_tolerance_N") != CONTRAST_TOLERANCE_N
        or thresholds.get("rank_margin_N") != RANK_MARGIN_N
        or thresholds.get("minimum_independent_physical_families")
        != MIN_INDEPENDENT_FAMILIES
    ):
        raise InvalidPreregistrationError("preregistered thresholds drift")
    projection = _validate_contribution_free_projection(
        prereg.get("contribution_free_baseline_projection"),
        {"baseline_bundle_id": prereg.get("baseline_bundle_id")},
        fingerprint_path=fingerprint_path,
    )
    if projection["baseline_receipt_sha256"] != _sha256_file(
        baseline_receipt_path
    ) or prereg.get("baseline_bundle_id") != baseline_receipt.get("baseline_bundle_id"):
        raise InvalidPreregistrationError(
            "preregistered contribution-free baseline binding drift"
        )
    provenance = prereg.get("provenance")
    expected_provenance_fields = {
        "fingerprint",
        "disease_spec",
        "baseline_receipt_projection_source",
        "protocol_stack",
        "confirmed_scorer",
        "postprocess_authorization",
        "generator",
    }
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != expected_provenance_fields
    ):
        raise InvalidPreregistrationError("preregistration provenance missing")
    expected_current = {
        "fingerprint": fingerprint_path,
        "confirmed_scorer": CONFIRMED_COMPARE_SOURCE,
        "postprocess_authorization": AUTHORIZED_POSTPROCESS_AUTHORIZATION,
        "generator": Path(__file__).resolve(),
    }
    for name, path in expected_current.items():
        record = provenance.get(name)
        expected_record = {
            "path": _display_path(path),
            "sha256": _sha256_file(path),
        }
        if _canonical_hash(record) != _canonical_hash(expected_record):
            raise InvalidPreregistrationError(
                f"preregistered {name} identity does not match current file"
            )
    if provenance.get("baseline_receipt_projection_source") != {
        "sha256": projection["baseline_receipt_sha256"],
    }:
        raise InvalidPreregistrationError(
            "preregistered baseline projection provenance drift"
        )
    if _canonical_hash(provenance.get("protocol_stack")) != _canonical_hash(
        _protocol_stack()
    ):
        raise InvalidPreregistrationError(
            "preregistered provenance protocol stack drift"
        )
    disease_identity = provenance.get("disease_spec")
    try:
        _validate_file_identity(
            disease_identity,
            label="preregistered disease specification",
        )
    except InvalidEvidenceError as exc:
        raise InvalidPreregistrationError(str(exc)) from exc
    if not prereg_path.is_file():
        raise InvalidPreregistrationError("preregistration was not atomically saved")


def _validate_prereg_fingerprint_contract(
    prereg: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    fingerprint_path: Path,
) -> None:
    samples_by_curve, _, family_by_id = _fingerprint_maps(fingerprint)
    active = prereg.get("active_disease")
    frozen_spec = prereg.get("active_disease_spec")
    if not isinstance(active, Mapping) or not isinstance(frozen_spec, Mapping):
        raise InvalidPreregistrationError("preregistered active disease missing")
    projection = _validate_contribution_free_projection(
        prereg.get("contribution_free_baseline_projection"),
        fingerprint,
        fingerprint_path=fingerprint_path,
    )
    normalized_spec = _validate_active_disease_spec(
        frozen_spec,
        fingerprint,
        _projection_as_receipt(projection),
    )
    normalized = {
        "active_disease_id": normalized_spec["global_disease_id"],
        "global_disease_identity": normalized_spec["global_disease_identity"],
        "channel": normalized_spec["channel"],
        "abscissa": normalized_spec["abscissa"],
        "shape_class": normalized_spec["shape_class"],
        "component_signature": normalized_spec["component_signature"],
        "physical_family_ids": normalized_spec["support_physical_family_ids"],
        "official_curve_keys": normalized_spec["support_official_curve_keys"],
    }
    if _canonical_hash(active) != _canonical_hash(normalized):
        raise InvalidPreregistrationError("preregistered disease payload drift")
    raw_families = prereg.get("families")
    if not isinstance(raw_families, list):
        raise InvalidPreregistrationError("preregistered families missing")
    by_id = {
        item.get("physical_family_id"): item
        for item in raw_families
        if isinstance(item, Mapping)
    }
    if (
        len(by_id) != len(raw_families)
        or list(by_id) != normalized["physical_family_ids"]
    ):
        raise InvalidPreregistrationError("preregistered family order/identity drift")
    selected_by_curve = {
        str(item["curve"]): item for item in normalized_spec["selected_curves"]
    }
    all_positive_support: set[str] = set()
    for family_id in normalized["physical_family_ids"]:
        source = family_by_id[family_id]
        actual = by_id[family_id]
        expected_curves = [
            selected_by_curve[curve] for curve in source["official_curve_keys"]
        ]
        support = sorted(
            {
                key
                for curve in expected_curves
                for component in curve["components"]
                for key in component["effective_support_condition_keys"]
            },
            key=_condition_sort_key,
        )
        all_positive_support.update(support)
        guards = [
            guard
            for curve in source["official_curve_keys"]
            for guard in _pairwise_guards(curve, samples_by_curve[curve])
        ]
        expected = {
            "physical_family_id": family_id,
            "channel": source["channel"],
            "official_curve_keys": list(source["official_curve_keys"]),
            "n_official_curves": len(source["official_curve_keys"]),
            "curves": expected_curves,
            "positive_support_condition_keys": support,
            "pairwise_guards": guards,
        }
        if _canonical_hash(actual) != _canonical_hash(expected):
            raise InvalidPreregistrationError(
                f"{family_id}: preregistered contrast/support drift"
            )
    if prereg.get("positive_support_condition_keys") != sorted(
        all_positive_support,
        key=_condition_sort_key,
    ):
        raise InvalidPreregistrationError(
            "preregistered positive-support condition drift"
        )
    if _canonical_hash(prereg.get("positive_support_conflict_graph")) != (
        _canonical_hash(normalized_spec["support_conflict_graph"])
    ):
        raise InvalidPreregistrationError("preregistered support-conflict graph drift")


def _manifest_graph(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    common = manifest.get("common_claim_manifest")
    if not isinstance(common, Mapping):
        raise InvalidEvidenceError("fresh manifest lacks common claim manifest")
    identity = _claim_graph_identity(common)
    if manifest.get("claim_graph_identity_sha256") != identity:
        raise InvalidEvidenceError("fresh manifest claim graph identity mismatch")
    return common, identity


def _condition_record(condition: Sequence[float]) -> dict[str, float]:
    U, frequency, twist, aoa = (float(item) for item in condition)
    return {
        "U_m_s": U,
        "frequency_Hz": frequency,
        "nominal_twist_deg": twist,
        "solver_twist_amplitude_deg": twist / 2.0,
        "aoa_deg": aoa,
    }


def _validate_resolved_call(
    resolved: Any,
    condition: Sequence[float],
    *,
    key: str,
) -> None:
    try:
        confirmed_compare._validate_resolved_call(
            resolved,
            condition=tuple(float(item) for item in condition),
            label=key,
        )
    except confirmed_compare.BaselineContractError as exc:
        raise InvalidEvidenceError(str(exc)) from exc


def _contribution_case(
    evidence: Mapping[str, Any],
    *,
    key: str,
    condition: Sequence[float],
    result: Mapping[str, Any],
    manifest_guard: Mapping[str, Any],
    common_manifest: Mapping[str, Any],
    graph_identity: str,
) -> dict[str, dict[str, float]]:
    if evidence.get("condition_key") != key:
        raise InvalidEvidenceError(f"{key}: evidence condition key drift")
    if _canonical_hash(evidence.get("condition")) != _canonical_hash(
        _condition_record(condition)
    ):
        raise InvalidEvidenceError(f"{key}: evidence condition record drift")
    _validate_resolved_call(evidence.get("resolved_call"), condition, key=key)
    guards = evidence.get("claim_guards")
    if not _strict_guards(guards) or _canonical_hash(guards) != _canonical_hash(
        manifest_guard
    ):
        raise InvalidEvidenceError(f"{key}: guard evidence missing or failed")
    if evidence.get("claim_graph_identity_sha256") != graph_identity:
        raise InvalidEvidenceError(f"{key}: graph identity drift")
    case_manifest = dict(common_manifest)
    case_manifest["guards"] = dict(guards)
    if evidence.get("claim_manifest_sha256") != _canonical_hash(case_manifest):
        raise InvalidEvidenceError(f"{key}: case claim manifest hash drift")

    raw = evidence.get("claim_contributions")
    if not isinstance(raw, Mapping) or set(raw) != EXPECTED_CONTRIBUTION_NODES:
        raise InvalidEvidenceError(f"{key}: contribution node inventory drift")
    aoa = float(condition[3])
    node_wind: dict[str, dict[str, float]] = {}
    ledger = np.zeros(3, dtype=float)
    recomputed_summary: dict[str, Any] = {}
    for node_id, expected_inventory in CONTRIBUTION_INVENTORY.items():
        items = raw.get(node_id)
        if not isinstance(items, list) or len(items) != len(expected_inventory):
            raise InvalidEvidenceError(
                f"{key}/{node_id}: contribution inventory length drift"
            )
        node_total = np.zeros(3, dtype=float)
        summary_items: list[dict[str, Any]] = []
        for item, (expected_channel, expected_role) in zip(
            items,
            expected_inventory,
        ):
            if (
                not isinstance(item, Mapping)
                or set(item) != {"body_force", "channel", "metadata", "role"}
                or item.get("channel") != expected_channel
                or item.get("role") != expected_role
                or not isinstance(item.get("metadata"), Mapping)
            ):
                raise InvalidEvidenceError(
                    f"{key}/{node_id}: contribution channel/role drift"
                )
            body = np.asarray(item["body_force"], dtype=float)
            if body.shape != (3,) or not np.isfinite(body).all():
                raise InvalidEvidenceError(
                    f"{key}/{node_id}: invalid contribution body force"
                )
            node_total += body
            summary_items.append(
                {
                    "channel": expected_channel,
                    "role": expected_role,
                    "body_force_N": body.tolist(),
                    "wind_force": {
                        f"{name}_N": value
                        for name, value in _wind_force(body, aoa).items()
                    },
                    "metadata": dict(item["metadata"]),
                }
            )
        ledger += node_total
        wind = _wind_force(node_total, aoa)
        node_wind[node_id] = wind
        recomputed_summary[node_id] = {
            "items": summary_items,
            "total_body_force_N": node_total.tolist(),
            "total_wind_force": {f"{name}_N": value for name, value in wind.items()},
        }

    if _canonical_hash(evidence.get("contribution_summary")) != _canonical_hash(
        recomputed_summary
    ):
        raise InvalidEvidenceError(f"{key}: stored contribution summary drift")
    total_wind = _wind_force(ledger, aoa)
    lift = _finite(result.get("L"), label=f"{key} result L")
    thrust = _finite(result.get("T"), label=f"{key} result T")
    error = max(abs(total_wind["L"] - lift), abs(total_wind["T"] - thrust))
    if error > LEDGER_TOLERANCE_N:
        raise InvalidEvidenceError(f"{key}: contribution ledger does not close")
    stored = evidence.get("recomputed_ledger")
    if (
        not isinstance(stored, Mapping)
        or _canonical_hash(stored.get("total_body_force_N"))
        != _canonical_hash(ledger.tolist())
        or max(
            _finite(stored.get("max_body_error_N"), label=f"{key} body error"),
            _finite(stored.get("max_wind_error_N"), label=f"{key} wind error"),
        )
        > LEDGER_TOLERANCE_N
    ):
        raise InvalidEvidenceError(f"{key}: stored ledger evidence drift")
    stored_wind = stored.get("total_wind_force")
    if not isinstance(stored_wind, Mapping) or not (
        _same_number(stored_wind.get("L_N"), total_wind["L"])
        and _same_number(stored_wind.get("T_N"), total_wind["T"])
    ):
        raise InvalidEvidenceError(f"{key}: stored wind ledger drift")
    return node_wind


def _scorecard_rows(scorecard: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if (
        scorecard.get("schema_version") != 3
        or scorecard.get("primary_evidence_scope") != EVIDENCE_CONFIRMED
    ):
        raise InvalidEvidenceError("scorecard schema/evidence scope drift")
    scopes = scorecard.get("evidence_scopes")
    confirmed = scopes.get(EVIDENCE_CONFIRMED) if isinstance(scopes, Mapping) else None
    if not isinstance(confirmed, Mapping):
        raise InvalidEvidenceError("scorecard confirmed scope missing")
    coverage = confirmed.get("coverage")
    rows = confirmed.get("rows")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("complete") is not True
        or coverage.get("valid_unique_conditions") != EXPECTED_CONDITIONS
        or coverage.get("complete_curves") != EXPECTED_CURVES
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_CURVES
    ):
        raise InvalidEvidenceError("scorecard confirmed coverage drift")
    output: dict[str, Mapping[str, Any]] = {}
    count = 0
    curve_specs = {
        curve.key: curve
        for curve in benchmark.CURVES_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]
    }
    measurements = benchmark.load_measurements(benchmark.DEFAULT_DATA_MD)
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("curve"), str):
            raise InvalidEvidenceError("scorecard confirmed row malformed")
        curve = str(row["curve"])
        if curve not in curve_specs or curve in output:
            raise InvalidEvidenceError("conditional/duplicate scorecard row")
        spec = curve_specs[curve]
        expected_identity = {
            "figure": spec.figure,
            "panel": spec.panel,
            "channel": spec.channel,
            "abscissa": spec.abscissa,
            "evidence_scope": EVIDENCE_CONFIRMED,
        }
        if any(row.get(name) != value for name, value in expected_identity.items()):
            raise InvalidEvidenceError(f"{curve}: scorecard curve identity drift")
        measured = row.get("measurement_N")
        model = row.get("model_at_measurement_x_N")
        if not isinstance(measured, list) or not isinstance(model, list):
            raise InvalidEvidenceError(f"{curve}: scorecard arrays missing")
        expected_measurement = measurements[curve]
        if (
            len(measured) != len(model)
            or len(measured) != len(expected_measurement.values_N)
            or any(
                not _same_number(actual, expected)
                for actual, expected in zip(
                    measured,
                    expected_measurement.values_N,
                )
            )
        ):
            raise InvalidEvidenceError(f"{curve}: scorecard array length drift")
        count += len(measured)
        output[curve] = row
    if count != EXPECTED_SAMPLES or set(output) != set(curve_specs):
        raise InvalidEvidenceError("scorecard sample count drift")
    return output


def _validate_scorecard_fingerprint(
    scorecard_rows: Mapping[str, Mapping[str, Any]],
    samples_by_curve: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    if set(scorecard_rows) != set(samples_by_curve):
        raise InvalidEvidenceError("scorecard/fingerprint curve sets differ")
    for curve, samples in samples_by_curve.items():
        row = scorecard_rows[curve]
        measured = row["measurement_N"]
        model = row["model_at_measurement_x_N"]
        if len(samples) != len(measured):
            raise InvalidEvidenceError(f"{curve}: scorecard/fingerprint length drift")
        for index, sample in enumerate(samples):
            if not (
                _same_number(measured[index], sample["measurement_N"])
                and _same_number(model[index], sample["model_N"])
            ):
                raise InvalidEvidenceError(
                    f"{curve}[{index}]: scorecard/fingerprint value drift"
                )


def _validate_complete_inputs(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contribution_file: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    inputs: EvaluationInputs,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, float]],
]:
    samples_by_curve, _, family_by_id = _fingerprint_maps(fingerprint)
    scorecard_rows = _scorecard_rows(scorecard)
    _validate_scorecard_fingerprint(scorecard_rows, samples_by_curve)
    if (
        manifest.get("schema_version") != 2
        or contribution_file.get("schema_version") != 2
    ):
        raise InvalidEvidenceError("fresh manifest/contribution schema drift")
    if manifest.get("status") != "complete":
        raise InvalidEvidenceError("fresh manifest is not complete")
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or contribution_file.get("run_id") != run_id
        or manifest.get("failures") != {}
    ):
        raise InvalidEvidenceError("fresh run identity/failure receipt drift")
    expected_bound_paths = {
        "result_path": _display_path(inputs.result_path),
        "manifest_path": _display_path(inputs.manifest_path),
        "contributions_path": _display_path(inputs.contributions_path),
    }
    for field, expected_path in expected_bound_paths.items():
        if (
            manifest.get(field) != expected_path
            or contribution_file.get(field) != expected_path
        ):
            raise InvalidEvidenceError(f"fresh {field} binding drift")
    if manifest.get("expected_condition_count") != EXPECTED_CONDITIONS:
        raise InvalidEvidenceError("fresh manifest condition count drift")
    expected_keys = manifest.get("expected_condition_keys")
    cases = contribution_file.get("cases")
    guards = manifest.get("case_guards")
    if (
        not isinstance(expected_keys, list)
        or set(expected_keys) != EXPECTED_CONDITION_KEYS
        or set(result) != EXPECTED_CONDITION_KEYS
        or not isinstance(cases, Mapping)
        or set(cases) != EXPECTED_CONDITION_KEYS
        or not isinstance(guards, Mapping)
        or set(guards) != EXPECTED_CONDITION_KEYS
        or manifest.get("completed_condition_count") != EXPECTED_CONDITIONS
    ):
        raise InvalidEvidenceError("fresh result/evidence key sets are not 151/151")
    coverage = manifest.get("final_confirmed_coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("complete") is not True
        or coverage.get("valid_unique_conditions") != EXPECTED_CONDITIONS
        or coverage.get("complete_curves") != EXPECTED_CURVES
    ):
        raise InvalidEvidenceError("fresh final confirmed coverage drift")
    if manifest.get("result_sha256") != _sha256_file(inputs.result_path):
        raise InvalidEvidenceError("fresh result hash mismatch")
    if manifest.get("contributions_sha256") != _sha256_file(inputs.contributions_path):
        raise InvalidEvidenceError("fresh contributions hash mismatch")
    common, graph_identity = _manifest_graph(manifest)
    runtime_identity = manifest.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        raise InvalidEvidenceError("fresh runtime identity missing")

    node_by_condition: dict[str, dict[str, dict[str, float]]] = {}
    resolved_calls: dict[str, Any] = {}
    for key in sorted(EXPECTED_CONDITION_KEYS):
        value = result[key]
        evidence = cases[key]
        if not isinstance(value, Mapping) or not isinstance(evidence, Mapping):
            raise InvalidEvidenceError(f"{key}: malformed result/evidence")
        if set(value) != {"L", "T"}:
            raise InvalidEvidenceError(f"{key}: result schema must be exact L/T")
        condition = CONFIRMED_BY_KEY[key]
        node_by_condition[key] = _contribution_case(
            evidence,
            key=key,
            condition=condition,
            result=value,
            manifest_guard=guards[key],
            common_manifest=common,
            graph_identity=graph_identity,
        )
        resolved_calls[key] = evidence["resolved_call"]
    if manifest.get("resolved_call_contract_sha256") != _canonical_hash(resolved_calls):
        raise InvalidEvidenceError("fresh resolved-call contract hash mismatch")

    for curve, samples in samples_by_curve.items():
        channel = str(family_by_id[samples[0]["physical_family_id"]]["channel"])
        for sample in samples:
            left = str(sample["left_condition_key"])
            right = str(sample["right_condition_key"])
            left_weight = float(sample["left_weight"])
            right_weight = float(sample["right_weight"])
            interpolated = left_weight * float(
                result[left][channel]
            ) + right_weight * float(result[right][channel])
            if abs(interpolated - float(sample["model_N"])) > LEDGER_TOLERANCE_N:
                raise InvalidEvidenceError(
                    f"{curve}: fingerprint model is not the fresh result interpolation"
                )
    return samples_by_curve, family_by_id, node_by_condition


def _dense_coefficients(
    contrast: Mapping[str, Any],
    count: int,
) -> np.ndarray:
    coefficients = np.zeros(count, dtype=float)
    raw = contrast.get("coefficients")
    if not isinstance(raw, list):
        raise InvalidPreregistrationError("contrast coefficients missing")
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise InvalidPreregistrationError("contrast coefficient malformed")
        index = item.get("measurement_index")
        value = item.get("coefficient")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not (0 <= index < count)
            or index in seen
        ):
            raise InvalidPreregistrationError("contrast index out of range")
        seen.add(index)
        coefficients[index] = _finite(
            value,
            label="contrast coefficient",
        )
    if abs(math.fsum(float(item) for item in coefficients)) > 1.0e-15 or not (
        _same_number(
            math.fsum(abs(float(item)) for item in coefficients),
            contrast.get("coefficient_l1"),
        )
    ):
        raise InvalidPreregistrationError("contrast zero-sum/L1 contract drift")
    return coefficients


def _node_curve(
    samples: Sequence[Mapping[str, Any]],
    node: str,
    channel: str,
    node_by_condition: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> np.ndarray:
    values = []
    for sample in samples:
        left = str(sample["left_condition_key"])
        right = str(sample["right_condition_key"])
        values.append(
            float(sample["left_weight"]) * node_by_condition[left][node][channel]
            + float(sample["right_weight"]) * node_by_condition[right][node][channel]
        )
    return np.asarray(values, dtype=float)


def _parent_contrast_from_alpha(
    contrast: Mapping[str, Any],
    *,
    node: str,
    channel: str,
    node_by_condition: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> float:
    alpha = contrast.get("canonical_alpha")
    if not isinstance(alpha, list):
        raise InvalidPreregistrationError("frozen canonical alpha missing")
    terms: list[float] = []
    seen: set[str] = set()
    previous: tuple[float, float, float, float] | None = None
    for item in alpha:
        if not isinstance(item, Mapping):
            raise InvalidPreregistrationError("canonical alpha item malformed")
        key = str(item.get("condition_key"))
        sort_key = _condition_sort_key(key)
        if key in seen or (previous is not None and sort_key <= previous):
            raise InvalidPreregistrationError(
                "canonical alpha key order/uniqueness drift"
            )
        seen.add(key)
        previous = sort_key
        coefficient = _finite(
            item.get("coefficient"),
            label="canonical alpha coefficient",
        )
        if abs(coefficient) <= CANONICAL_ZERO_TOLERANCE:
            raise InvalidPreregistrationError(
                "canonical alpha contains a removable zero"
            )
        terms.append(coefficient * node_by_condition[key][node][channel])
    return math.fsum(terms)


def _evaluate_curve(
    prepared_curve: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    node: str,
    channel: str,
    node_by_condition: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    if (
        prepared_curve.get("classification_status") != "CLASSIFIED"
        or prepared_curve.get("candidate_eligible") is not True
    ):
        raise InvalidPreregistrationError(
            f"{prepared_curve.get('curve')}: selected curve is not disease-eligible"
        )
    measured = np.asarray(
        [float(item["measurement_N"]) for item in samples],
        dtype=float,
    )
    model = np.asarray(
        [float(item["model_N"]) for item in samples],
        dtype=float,
    )
    node_values = _node_curve(samples, node, channel, node_by_condition)
    deleted = model - node_values
    baseline_mae = _mean(
        [abs(float(left - right)) for left, right in zip(model, measured)]
    )
    deleted_mae = _mean(
        [abs(float(left - right)) for left, right in zip(deleted, measured)]
    )
    mae_improvement = baseline_mae - deleted_mae
    component_results: list[dict[str, Any]] = []
    curve_reasons: set[str] = set()
    for contrast in prepared_curve.get("components", []):
        coefficients = _dense_coefficients(contrast, measured.size)
        experimental = math.fsum(
            float(coefficient) * float(value)
            for coefficient, value in zip(coefficients, measured)
        )
        baseline = math.fsum(
            float(coefficient) * float(value)
            for coefficient, value in zip(coefficients, model)
        )
        node_contribution = _parent_contrast_from_alpha(
            contrast,
            node=node,
            channel=channel,
            node_by_condition=node_by_condition,
        )
        direct_contribution = math.fsum(
            float(coefficient) * float(value)
            for coefficient, value in zip(coefficients, node_values)
        )
        if abs(node_contribution - direct_contribution) > LEDGER_TOLERANCE_N:
            raise InvalidEvidenceError(
                f"{prepared_curve['curve']}/{node}: alpha interpolation drift"
            )
        if not (
            _same_number(
                contrast.get("experimental_contrast_N"),
                experimental,
            )
            and _same_number(
                contrast.get("baseline_contrast_N"),
                baseline,
            )
        ):
            raise InvalidPreregistrationError(
                f"{prepared_curve['curve']}: frozen component value drift"
            )
        after_deletion = baseline - node_contribution
        tolerance = float(contrast["contrast_tolerance_N"])
        if abs(experimental) <= tolerance:
            raise InvalidPreregistrationError(
                f"{prepared_curve['curve']}: component is not identifiable"
            )
        state = _baseline_component_state(experimental, baseline)
        if state != contrast.get("baseline_state"):
            raise InvalidPreregistrationError(
                f"{prepared_curve['curve']}: component state drift"
            )
        error_improvement = abs(baseline - experimental) - abs(
            after_deletion - experimental
        )
        harmful_direction = (baseline - experimental) * node_contribution > 0.0
        if state == "PASS":
            passed = bool(
                abs(after_deletion - experimental) <= tolerance
                and after_deletion * experimental > 0.0
            )
            if not passed:
                curve_reasons.add("PARENT_FAIL_PASS_COMPONENT_DAMAGED")
        else:
            passed = bool(
                after_deletion * experimental > 0.0
                and error_improvement > tolerance
                and harmful_direction
            )
            if not passed:
                curve_reasons.add(
                    {
                        "REVERSED": ("PARENT_FAIL_COMPONENT_REVERSED_NOT_RESTORED"),
                        "UNDER": "PARENT_FAIL_COMPONENT_UNDER_NOT_IMPROVED",
                        "OVER": "PARENT_FAIL_COMPONENT_OVER_NOT_IMPROVED",
                    }[state]
                )
        component_results.append(
            {
                "component_name": contrast["contrast_id"],
                "baseline_state": state,
                "experimental_N": experimental,
                "baseline_N": baseline,
                "parent_contribution_N": node_contribution,
                "after_parent_force_erasure_N": after_deletion,
                "contrast_tolerance_N": tolerance,
                "absolute_error_improvement_N": error_improvement,
                "harmful_direction": harmful_direction,
                "component_restoration_boolean": passed,
            }
        )
    guard_results: list[dict[str, Any]] = []
    for guard in prepared_curve.get("guard_contrasts", []):
        coefficients = _dense_coefficients(guard, measured.size)
        experimental = math.fsum(
            float(coefficient) * float(value)
            for coefficient, value in zip(coefficients, measured)
        )
        baseline = math.fsum(
            float(coefficient) * float(value)
            for coefficient, value in zip(coefficients, model)
        )
        parent_contribution = _parent_contrast_from_alpha(
            guard,
            node=node,
            channel=channel,
            node_by_condition=node_by_condition,
        )
        after_deletion = baseline - parent_contribution
        tolerance = float(guard["contrast_tolerance_N"])
        if not (
            abs(experimental) > tolerance
            and abs(baseline - experimental) <= tolerance
            and _same_number(
                guard.get("experimental_contrast_N"),
                experimental,
            )
            and _same_number(guard.get("baseline_contrast_N"), baseline)
        ):
            raise InvalidPreregistrationError(
                f"{prepared_curve['curve']}: frozen guard drift"
            )
        holds = bool(
            abs(after_deletion - experimental) <= tolerance
            and after_deletion * experimental > 0.0
        )
        if not holds:
            curve_reasons.add("PARENT_FAIL_PAIRWISE_GUARD_DAMAGED")
        guard_results.append(
            {
                "guard_identity": copy.deepcopy(guard["guard_identity"]),
                "measurement_pair": copy.deepcopy(guard["measurement_pair"]),
                "experimental_N": experimental,
                "baseline_N": baseline,
                "parent_contribution_N": parent_contribution,
                "after_parent_force_erasure_N": after_deletion,
                "contrast_tolerance_N": tolerance,
                "holds": holds,
            }
        )
    curve_mae_pass = mae_improvement > FORCE_TOLERANCE_N
    if not curve_mae_pass:
        curve_reasons.add("PARENT_FAIL_CURVE_MAE")
    all_pass_components_hold = all(
        item["component_restoration_boolean"]
        for item in component_results
        if item["baseline_state"] == "PASS"
    )
    all_pairwise_guards_hold = all(item["holds"] for item in guard_results)
    final_restored = bool(
        all(item["component_restoration_boolean"] for item in component_results)
        and all_pairwise_guards_hold
        and curve_mae_pass
    )
    alias_signature = {
        "components": [
            {
                "component_name": item["component_name"],
                "component_restoration_boolean": item["component_restoration_boolean"],
            }
            for item in component_results
        ],
        "all_pass_components_hold": all_pass_components_hold,
        "all_pairwise_guards_hold": all_pairwise_guards_hold,
        "curve_mae_pass": curve_mae_pass,
        "final_curve_restored": final_restored,
    }
    return {
        "curve": prepared_curve["curve"],
        "status": "EVALUATED",
        "shape_class": prepared_curve["shape_class"],
        "baseline_mae_N": baseline_mae,
        "after_parent_force_erasure_mae_N": deleted_mae,
        "mae_improvement_N": mae_improvement,
        "curve_mae_pass": curve_mae_pass,
        "component_results": component_results,
        "guard_results": guard_results,
        "alias_parent_signature": alias_signature,
        "final_curve_restored": final_restored,
        "local_reasons": [
            reason for reason in PARENT_LOCAL_REASON_ORDER if reason in curve_reasons
        ],
    }


def _evaluate_node(
    node: str,
    prereg: Mapping[str, Any],
    samples_by_curve: Mapping[str, Sequence[Mapping[str, Any]]],
    node_by_condition: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    family_results: list[dict[str, Any]] = []
    all_support_pf_ids = [
        str(item["physical_family_id"]) for item in prereg["families"]
    ]
    parent_reasons: set[str] = set()
    pf_improvements: list[float] = []
    for family in prereg["families"]:
        channel = family["channel"]
        curves = [
            _evaluate_curve(
                prepared_curve,
                samples_by_curve[prepared_curve["curve"]],
                node=node,
                channel=channel,
                node_by_condition=node_by_condition,
            )
            for prepared_curve in family["curves"]
        ]
        pair_ledger: list[dict[str, Any]] = []
        signatures = {item["curve"]: item["alias_parent_signature"] for item in curves}
        for left_index, left in enumerate(curves):
            for right in curves[left_index + 1 :]:
                consistent = _canonical_hash(
                    left["alias_parent_signature"]
                ) == _canonical_hash(right["alias_parent_signature"])
                pair_ledger.append(
                    {
                        "left_curve": left["curve"],
                        "right_curve": right["curve"],
                        "consistent": consistent,
                    }
                )
        alias_uniform = all(item["consistent"] for item in pair_ledger)
        if not alias_uniform:
            parent_reasons.add("PARENT_FAIL_ALIAS_NONUNIFORM")
        for curve in curves:
            parent_reasons.update(curve["local_reasons"])
        pf_restored = bool(
            curves
            and alias_uniform
            and all(item["final_curve_restored"] for item in curves)
        )
        if not pf_restored:
            parent_reasons.add("PARENT_FAIL_PF_NOT_FULLY_RESTORED")
        pf_improvement = _mean(
            [
                float(item["mae_improvement_N"])
                for item in sorted(curves, key=lambda value: value["curve"])
            ]
        )
        pf_improvements.append(pf_improvement)
        family_results.append(
            {
                "physical_family_id": family["physical_family_id"],
                "official_curve_keys": family["official_curve_keys"],
                "alias_uniform": alias_uniform,
                "alias_parent_signatures": signatures,
                "alias_pair_ledger": pair_ledger,
                "positive_support_condition_keys": family[
                    "positive_support_condition_keys"
                ],
                "pairwise_guard_condition_keys_excluded_from_positive_support": True,
                "pf_restored": pf_restored,
                "pf_mae_improvement_N": pf_improvement,
                "curves": curves,
            }
        )
    support_analysis = _maximum_independent_sets(
        {
            str(item["physical_family_id"]): item["positive_support_condition_keys"]
            for item in prereg["families"]
        }
    )
    if support_analysis["max_pairwise_disjoint_pf_count"] < MIN_INDEPENDENT_FAMILIES:
        parent_reasons.add("PARENT_FAIL_NO_PAIRWISE_DISJOINT_REPLICATION")
    restored_pf_ids = [
        item["physical_family_id"] for item in family_results if item["pf_restored"]
    ]
    full_coverage = restored_pf_ids == all_support_pf_ids
    passes = bool(
        full_coverage
        and support_analysis["max_pairwise_disjoint_pf_count"]
        >= MIN_INDEPENDENT_FAMILIES
    )
    local_reasons = [
        reason for reason in PARENT_LOCAL_REASON_ORDER if reason in parent_reasons
    ]
    if passes and local_reasons:
        raise InvalidEvidenceError(
            f"{node}: PARENT_PASS cannot retain local failure reasons"
        )
    family_equal_improvement = _mean(pf_improvements)
    return {
        "node": node,
        "status": "PARENT_PASS" if passes else "PARENT_FAIL",
        "local_reasons": local_reasons,
        "deletion_semantics": (
            "frozen-state direct additive force erasure; downstream states "
            "are not recomputed"
        ),
        "family_equal_mae_improvement_N": family_equal_improvement,
        "restored_pf_ids": restored_pf_ids,
        "all_support_pf_ids": all_support_pf_ids,
        "positive_support_conflict_graph": support_analysis,
        "all_maximum_disjoint_pf_sets": support_analysis[
            "all_maximum_disjoint_pf_sets"
        ],
        "component_results": [
            {
                "physical_family_id": family["physical_family_id"],
                "curve": curve["curve"],
                **copy.deepcopy(component),
            }
            for family in family_results
            for curve in family["curves"]
            for component in curve["component_results"]
            if component["baseline_state"] != "PASS"
        ],
        "pass_component_results": [
            {
                "physical_family_id": family["physical_family_id"],
                "curve": curve["curve"],
                **copy.deepcopy(component),
            }
            for family in family_results
            for curve in family["curves"]
            for component in curve["component_results"]
            if component["baseline_state"] == "PASS"
        ],
        "guard_results": [
            {
                "physical_family_id": family["physical_family_id"],
                "curve": curve["curve"],
                **copy.deepcopy(guard),
            }
            for family in family_results
            for curve in family["curves"]
            for guard in curve["guard_results"]
        ],
        "curve_mae_results": [
            {
                "physical_family_id": family["physical_family_id"],
                "curve": curve["curve"],
                "baseline_mae_N": curve["baseline_mae_N"],
                "after_parent_force_erasure_mae_N": curve[
                    "after_parent_force_erasure_mae_N"
                ],
                "mae_improvement_N": curve["mae_improvement_N"],
                "curve_mae_pass": curve["curve_mae_pass"],
            }
            for family in family_results
            for curve in family["curves"]
        ],
        "passes_parent_gate": passes,
        "families": family_results,
    }


def _decision(
    n2: Mapping[str, Any],
    n3: Mapping[str, Any],
) -> tuple[str, str | None, list[str]]:
    n2_pass = n2["status"] == "PARENT_PASS"
    n3_pass = n3["status"] == "PARENT_PASS"
    if n2_pass and n3_pass:
        return (
            "NO_DECISION_MULTIPLE_PARENTS",
            None,
            [],
        )
    if n2_pass:
        return (
            "ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS",
            "N2",
            [],
        )
    if n3_pass:
        return (
            "ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS",
            "N3",
            [],
        )
    return (
        "NO_DECISION_NO_PARENT_FULL_COVERAGE",
        None,
        ["NO_DECISION_MISSING_OR_STATE_MEDIATED"],
    )


def _preauthorize_evaluation_dereference(
    prereg: Mapping[str, Any],
    baseline_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    inputs: EvaluationInputs,
) -> str:
    """Authorize the contribution identity before its target is opened."""

    for label, value, path in (
        ("preregistration", prereg, inputs.prereg_path),
        ("baseline receipt", baseline_receipt, inputs.baseline_receipt_path),
        ("manifest", manifest, inputs.manifest_path),
        ("fingerprint", fingerprint, inputs.fingerprint_path),
    ):
        _require_payload_file_match(value, path, label=label)
    prereg_payload, prepare_envelope = _unwrap_scientific_payload(
        prereg,
        stage="prepare",
        required=True,
    )
    assert prepare_envelope is not None
    _validate_prereg(
        prereg_payload,
        baseline_receipt,
        prereg_path=inputs.prereg_path,
        fingerprint_path=inputs.fingerprint_path,
        baseline_receipt_path=inputs.baseline_receipt_path,
    )
    try:
        _validate_execution_chain_identity(
            prereg_payload["selector_execution_envelope"],
            prepare_envelope,
            previous_stage="select-disease",
            current_stage="prepare",
        )
    except InvalidEvidenceError as exc:
        raise InvalidPreregistrationError(str(exc)) from exc
    _validate_prereg_fingerprint_contract(
        prereg_payload,
        fingerprint,
        fingerprint_path=inputs.fingerprint_path,
    )
    metadata = _validate_baseline_receipt_metadata(
        baseline_receipt,
        fingerprint,
        receipt_path=inputs.baseline_receipt_path,
        fingerprint_path=inputs.fingerprint_path,
    )
    contribution_identity = metadata["input_identity_metadata"]["claim_contributions"]
    if (
        manifest.get("schema_version") != confirmed_compare.CHECKPOINT_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("expected_condition_count") != EXPECTED_CONDITIONS
        or manifest.get("completed_condition_count") != EXPECTED_CONDITIONS
        or manifest.get("failures") != {}
    ):
        raise InvalidEvidenceError(
            "contribution dereference requires a complete fresh151 manifest"
        )
    if contribution_identity["path"] != _display_path(inputs.contributions_path):
        raise InvalidEvidenceError(
            "preauthorization contribution path differs from receipt"
        )
    try:
        confirmed_compare._validate_postprocess_authorization(
            authorization_path=AUTHORIZED_POSTPROCESS_AUTHORIZATION,
            expected_authorization_path=AUTHORIZED_POSTPROCESS_AUTHORIZATION,
            expected_authorization_sha256=(AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256),
            scoring_prereg_path=AUTHORIZED_POSTPROCESS_PREREG,
            expected_prereg_path=AUTHORIZED_POSTPROCESS_PREREG,
            expected_prereg_sha256=AUTHORIZED_POSTPROCESS_PREREG_SHA256,
            result_path=inputs.result_path,
            manifest_path=inputs.manifest_path,
            contributions_path=inputs.contributions_path,
            data_path=Path(benchmark.DEFAULT_DATA_MD),
            manifest=manifest,
        )
    except confirmed_compare.BaselineContractError as exc:
        raise InvalidEvidenceError(
            f"contribution dereference was not authorized: {exc}"
        ) from exc
    return str(contribution_identity["sha256"])


def _invalid_parent_receipt(
    message: str,
    *,
    evaluation_status: str,
    family_equal_mae: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fig171819_parent_claim_attribution",
        "status": "INVALID_EVIDENCE",
        "parent_evaluation": {
            "evaluation_status": evaluation_status,
        },
        "failed_gates": [message],
        "causal_status": "HYPOTHESIS_ONLY",
        "claim_decision": "NO_DECISION",
        "claim_writeback_allowed": False,
        "protocol_stack": _protocol_stack(),
        "production_execution_authorized": False,
        "receipt_envelope_required": True,
    }
    if family_equal_mae is not None:
        output["family_equal_mae"] = copy.deepcopy(dict(family_equal_mae))
    return output


def _evaluate_attribution_strict(
    prereg: Mapping[str, Any],
    baseline_receipt: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contributions: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    inputs: EvaluationInputs,
) -> dict[str, Any]:
    """Validate a complete receipt and evaluate N2/N3 parent hypotheses."""

    for label, value, path in (
        ("preregistration", prereg, inputs.prereg_path),
        (
            "baseline receipt",
            baseline_receipt,
            inputs.baseline_receipt_path,
        ),
        ("result", result, inputs.result_path),
        ("manifest", manifest, inputs.manifest_path),
        ("contributions", contributions, inputs.contributions_path),
        ("scorecard", scorecard, inputs.scorecard_path),
        ("fingerprint", fingerprint, inputs.fingerprint_path),
    ):
        _require_payload_file_match(value, path, label=label)
    prereg_payload, prepare_envelope = _unwrap_scientific_payload(
        prereg,
        stage="prepare",
        required=True,
    )
    assert prepare_envelope is not None
    _validate_prereg(
        prereg_payload,
        baseline_receipt,
        prereg_path=inputs.prereg_path,
        fingerprint_path=inputs.fingerprint_path,
        baseline_receipt_path=inputs.baseline_receipt_path,
    )
    try:
        _validate_execution_chain_identity(
            prereg_payload["selector_execution_envelope"],
            prepare_envelope,
            previous_stage="select-disease",
            current_stage="prepare",
        )
    except InvalidEvidenceError as exc:
        raise InvalidPreregistrationError(str(exc)) from exc
    _validate_prereg_fingerprint_contract(
        prereg_payload,
        fingerprint,
        fingerprint_path=inputs.fingerprint_path,
    )
    receipt_binding = _validate_baseline_receipt_full(
        baseline_receipt,
        fingerprint,
        receipt_path=inputs.baseline_receipt_path,
        fingerprint_path=inputs.fingerprint_path,
        result_path=inputs.result_path,
        manifest_path=inputs.manifest_path,
        contributions_path=inputs.contributions_path,
        scorecard_path=inputs.scorecard_path,
        manifest=manifest,
    )
    samples_by_curve, family_by_id, node_by_condition = _validate_complete_inputs(
        result,
        manifest,
        contributions,
        scorecard,
        fingerprint,
        inputs,
    )
    prereg_families = prereg_payload.get("families")
    if not isinstance(prereg_families, list):
        raise InvalidPreregistrationError("preregistered families missing")
    current_family_ids = set(family_by_id)
    selected_family_ids = [
        item.get("physical_family_id")
        for item in prereg_families
        if isinstance(item, Mapping)
    ]
    if (
        len(selected_family_ids) != len(prereg_families)
        or len(set(selected_family_ids)) != len(selected_family_ids)
        or not set(selected_family_ids).issubset(current_family_ids)
        or selected_family_ids
        != prereg_payload["active_disease"]["physical_family_ids"]
    ):
        raise InvalidPreregistrationError("preregistered family identity drift")

    candidate_results = {
        node: _evaluate_node(
            node,
            prereg_payload,
            samples_by_curve,
            node_by_condition,
        )
        for node in ELIGIBLE_PARENTS
    }
    family_equal_mae = {
        "role": "DERIVED_INVARIANT_AUDIT",
        "units": "N",
        "N2_improvement_N": candidate_results["N2"]["family_equal_mae_improvement_N"],
        "N3_improvement_N": candidate_results["N3"]["family_equal_mae_improvement_N"],
    }
    failed_implications = [
        f"{node}_PASS_IMPLIES_THRESHOLD"
        for node in ELIGIBLE_PARENTS
        if candidate_results[node]["status"] == "PARENT_PASS"
        and float(candidate_results[node]["family_equal_mae_improvement_N"])
        <= FORCE_TOLERANCE_N
    ]
    if failed_implications:
        return _invalid_parent_receipt(
            "family-equal MAE derived invariant failed",
            evaluation_status="EVALUATED_INVALID_INVARIANT",
            family_equal_mae={
                **family_equal_mae,
                "failed_implications": failed_implications,
            },
        )
    family_equal_mae["invariant_check"] = "PASS"
    decision, active_parent, secondary = _decision(
        candidate_results["N2"],
        candidate_results["N3"],
    )
    all_reasons = [
        decision,
        *candidate_results["N2"]["local_reasons"],
        *candidate_results["N3"]["local_reasons"],
        *secondary,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fig171819_parent_claim_attribution",
        "status": decision,
        "parent_evaluation": {"evaluation_status": "EVALUATED"},
        "evidence_scope": EVIDENCE_CONFIRMED,
        "confirmed_contract": dict(prereg_payload["confirmed_contract"]),
        "validity_gates": {
            "fresh_baseline_receipt_bundle_bound": True,
            "fresh_manifest_complete": True,
            "confirmed_result_151_of_151": True,
            "confirmed_curves_42": True,
            "measurement_samples_434": True,
            "physical_families_34": True,
            "alias_groups_8": True,
            "conditional_fig19_cd_excluded": True,
            "artifact_hashes_match": True,
            "claim_graph_identity_match": True,
            "runtime_and_call_contract_match": True,
            "all_case_guards_pass": True,
            "all_force_ledgers_close": True,
            "global_11_leaf_inventory_and_roles_match": True,
            "eligible_parent_leaf_aggregates_close": True,
        },
        "active_disease": dict(prereg_payload["active_disease"]),
        "active_disease_sha256": prereg_payload["active_disease_sha256"],
        "prepare_sha256": _canonical_hash(prereg_payload),
        "baseline_bundle_id": receipt_binding["baseline_bundle_id"],
        "preregistered_families": prereg_families,
        "N2": candidate_results["N2"],
        "N3": candidate_results["N3"],
        "family_equal_mae": family_equal_mae,
        "N6_negative_control": "NOT_EVALUATED_FOR_PARENT_SELECTION",
        "all_triggered_reasons": all_reasons,
        "secondary_diagnostics": secondary,
        "reason": (
            "LITERATURE_MECHANISM_ADJUDICATION_REQUIRED"
            if active_parent is not None
            else decision
        ),
        "claim_decision": (
            "ACTIVE_PARENT_HYPOTHESIS" if active_parent is not None else "NO_DECISION"
        ),
        "decision": {
            "code": decision,
            "active_parent": active_parent,
            "reason": (
                "LITERATURE_MECHANISM_ADJUDICATION_REQUIRED"
                if active_parent is not None
                else decision
            ),
            "allowed_next_action": (
                "targeted_primary_literature_for_active_parent"
                if active_parent is not None
                else "more_discriminating_preregistered_evidence"
            ),
        },
        "causal_status": "HYPOTHESIS_ONLY",
        "causal_limit": (
            "The operation erases a reported additive parent force while "
            "holding all state trajectories fixed.  It is not a runtime "
            "intervention; in particular N2 erasure does not recompute N3."
        ),
        "claim_writeback_allowed": False,
        "spatial_panel_load_required": True,
        "force_and_moment_ledger_required": True,
        "posthoc_total_force_redistribution_forbidden": True,
        "protocol_stack": _protocol_stack(),
        "prepare_execution_envelope": prepare_envelope,
        "production_execution_authorized": False,
        "receipt_envelope_required": True,
        "provenance": {
            "preregistration": {
                "path": _display_path(inputs.prereg_path),
                "sha256": _sha256_file(inputs.prereg_path),
            },
            "baseline_receipt": {
                "path": _display_path(inputs.baseline_receipt_path),
                "sha256": _sha256_file(inputs.baseline_receipt_path),
            },
            "result": {
                "path": _display_path(inputs.result_path),
                "sha256": _sha256_file(inputs.result_path),
            },
            "manifest": {
                "path": _display_path(inputs.manifest_path),
                "sha256": _sha256_file(inputs.manifest_path),
            },
            "contributions": {
                "path": _display_path(inputs.contributions_path),
                "sha256": _sha256_file(inputs.contributions_path),
            },
            "scorecard": {
                "path": _display_path(inputs.scorecard_path),
                "sha256": _sha256_file(inputs.scorecard_path),
            },
            "fingerprint": {
                "path": _display_path(inputs.fingerprint_path),
                "sha256": _sha256_file(inputs.fingerprint_path),
            },
            "protocol_stack": _protocol_stack(),
            "confirmed_scorer": {
                "path": _display_path(CONFIRMED_COMPARE_SOURCE),
                "sha256": _sha256_file(CONFIRMED_COMPARE_SOURCE),
            },
            "postprocess_authorization": {
                "path": _display_path(AUTHORIZED_POSTPROCESS_AUTHORIZATION),
                "sha256": _sha256_file(AUTHORIZED_POSTPROCESS_AUTHORIZATION),
            },
            "generator": {
                "path": _display_path(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
    }


def evaluate_attribution(
    prereg: Mapping[str, Any],
    baseline_receipt: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contributions: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    inputs: EvaluationInputs,
) -> dict[str, Any]:
    """Fail-closed public evaluator with validation/science state separation."""

    try:
        return _evaluate_attribution_strict(
            prereg,
            baseline_receipt,
            result,
            manifest,
            contributions,
            scorecard,
            fingerprint,
            inputs=inputs,
        )
    except (InvalidEvidenceError, InvalidPreregistrationError) as exc:
        return _invalid_parent_receipt(
            str(exc),
            evaluation_status="NOT_EVALUATED_UPSTREAM_FAILURE",
        )


def _select_disease_cli(args: argparse.Namespace) -> int:
    try:
        fingerprint = _load_json(args.fingerprint)
        baseline_receipt = _load_json(args.baseline_receipt)
        disease = select_active_disease(
            fingerprint,
            baseline_receipt,
            fingerprint_path=args.fingerprint,
            baseline_receipt_path=args.baseline_receipt,
        )
    except (InvalidEvidenceError, InvalidPreregistrationError) as exc:
        disease = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "fig171819_active_disease_selection",
            "status": "INVALID_EVIDENCE",
            "failed_gates": [str(exc)],
            "claim_decision": "NO_DECISION",
            "rankings": {
                "evaluation_status": "NOT_EVALUATED_UPSTREAM_FAILURE",
            },
            "shadow_lopo": {
                "evaluation_status": "NOT_RUN_UPSTREAM_FAILURE",
            },
            "protocol_stack": _protocol_stack(),
            "production_execution_authorized": False,
            "receipt_envelope_required": True,
        }
    _write_json_atomic(args.output, disease)
    print(
        "active disease selection: "
        f"{disease['status']} ({disease.get('disease_id', 'none')})"
    )
    print(f"saved {args.output}")
    return 2 if disease["status"] == "INVALID_EVIDENCE" else 0


def _prepare_cli(args: argparse.Namespace) -> int:
    try:
        fingerprint = _load_json(args.fingerprint)
        disease = _load_json(args.disease_spec)
        prereg = prepare_contract(
            fingerprint,
            disease,
            fingerprint_path=args.fingerprint,
            disease_spec_path=args.disease_spec,
        )
    except (InvalidEvidenceError, InvalidPreregistrationError) as exc:
        prereg = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "fig171819_parent_attribution_preregistration",
            "status": "INVALID_EVIDENCE",
            "failed_gates": [str(exc)],
            "causal_status": "HYPOTHESIS_ONLY",
            "claim_decision": "NO_DECISION",
            "claim_writeback_allowed": False,
            "protocol_stack": _protocol_stack(),
            "production_execution_authorized": False,
            "receipt_envelope_required": True,
        }
    _write_json_atomic(args.output, prereg)
    if prereg["status"] == "PREPARED":
        print(
            "prepared parent attribution: "
            f"{len(prereg['families'])} physical families, "
            f"{len(prereg['positive_support_condition_keys'])} positive-support "
            "conditions"
        )
    else:
        print("prepared parent attribution: INVALID_EVIDENCE")
    print(f"saved {args.output}")
    return 2 if prereg["status"] == "INVALID_EVIDENCE" else 0


def _evaluate_cli(args: argparse.Namespace) -> int:
    inputs = EvaluationInputs(
        result_path=args.result,
        manifest_path=args.manifest,
        contributions_path=args.contributions,
        scorecard_path=args.scorecard,
        fingerprint_path=args.fingerprint,
        prereg_path=args.prereg,
        baseline_receipt_path=args.baseline_receipt,
    )
    try:
        prereg = _load_json(args.prereg)
        baseline_receipt = _load_json(args.baseline_receipt)
        manifest = _load_json(args.manifest)
        fingerprint = _load_json(args.fingerprint)
        authorized_contribution_sha256 = _preauthorize_evaluation_dereference(
            prereg,
            baseline_receipt,
            manifest,
            fingerprint,
            inputs=inputs,
        )
        try:
            contribution_sha256 = _sha256_file(args.contributions)
        except OSError as exc:
            raise InvalidEvidenceError(
                f"cannot hash preauthorized contribution target: {exc}"
            ) from exc
        if contribution_sha256 != authorized_contribution_sha256:
            raise InvalidEvidenceError(
                "contribution target differs from preauthorized receipt"
            )
        report = evaluate_attribution(
            prereg,
            baseline_receipt,
            _load_json(args.result),
            manifest,
            _load_json(args.contributions),
            _load_json(args.scorecard),
            fingerprint,
            inputs=inputs,
        )
    except (InvalidEvidenceError, InvalidPreregistrationError) as exc:
        report = _invalid_parent_receipt(
            str(exc),
            evaluation_status="NOT_EVALUATED_UPSTREAM_FAILURE",
        )
    _write_json_atomic(args.output, report)
    print(
        f"parent attribution: {report['status']} "
        f"(causal_status={report['causal_status']})"
    )
    print(f"saved {args.output}")
    return 2 if report["status"] == "INVALID_EVIDENCE" else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_disease = subparsers.add_parser("select-disease")
    select_disease.add_argument("--fingerprint", type=Path, required=True)
    select_disease.add_argument("--baseline-receipt", type=Path, required=True)
    select_disease.add_argument("--output", type=Path, required=True)
    select_disease.set_defaults(handler=_select_disease_cli)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--fingerprint", type=Path, required=True)
    prepare.add_argument("--disease-spec", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(handler=_prepare_cli)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prereg", type=Path, required=True)
    evaluate.add_argument("--baseline-receipt", type=Path, required=True)
    evaluate.add_argument("--result", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--contributions", type=Path, required=True)
    evaluate.add_argument("--scorecard", type=Path, required=True)
    evaluate.add_argument("--fingerprint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate_cli)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (InvalidEvidenceError, InvalidPreregistrationError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
