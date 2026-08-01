"""Strict confirmed-scope post-processing for the FLUXV Fig. 17/18/19 campaign.

The baseline command is intentionally fail-closed.  It will not compute a
score until the fresh V4.1 runner has produced a complete, mutually bound
result/manifest/contribution triplet for the 151 source-confirmed conditions.

Only the 42 confirmed curves (434 raw digitized samples) are scored.  The
eight Fig. 19(c,d) curves are contract metadata only: they never enter score
rows, residual samples, physical-family metrics, or plots.

This module is post-processing only.  It does not import an aerodynamic
solver, mutate claim state, or modify the frozen campaign artifacts.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
DOCS = PLATFORM / "docs"

import fig171819_benchmark as benchmark  # noqa: E402
import fig171819_residual_fingerprint as residual_fingerprint  # noqa: E402
import run_claim_witnesses as witness  # noqa: E402


SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 2
LEDGER_TOLERANCE_N = 1.0e-9
EXPECTED_CURVES = 42
EXPECTED_SAMPLES = 434
EXPECTED_CONDITIONS = 151
EXPECTED_PHYSICAL_FAMILIES = 34
EXPECTED_ALIAS_GROUPS = 8
EXPECTED_SOLVER_SOURCES = 140
EXPECTED_CONTROL_SOURCES = 11
# 2026-08-01 双 scope 契约终裁: benchmark 因 FIG19_CD_FREQUENCY_STATUS
# 'conditional_scope' 修改,此为终裁后版本(与 fig171819_claim_attribution 同步)。
AUTHORIZED_BENCHMARK_SOURCE_SHA256 = (
    "45b93584550eea4b16969381c59fe1038f68b21cf7cd1371052a19315c2444da"
)
AUTHORIZED_FINGERPRINT_SOURCE_SHA256 = (
    "127db39b6028f1be676a10f95dc932f35e29402fd590dc979d97e269f4bc14e8"
)
AUTHORIZED_MEASUREMENT_DATA_SHA256 = (
    "ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1"
)
AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256 = (
    "22ca928d5240ed6195fefe9da1f48c121102712f1f39681d813f90646b8d3cab"
)
AUTHORIZED_POSTPROCESS_PREREG_SHA256 = (
    "7e4ae6135ad624f19d2d8aa27b4ff692fba0cb592feb9581b7a7f480f9433d60"
)
AUTHORIZED_POSTPROCESS_AUTHORIZATION = (
    ROOT
    / "platform"
    / "docs"
    / "diag"
    / "v41_fresh151_postprocess_authorization_20260729.json"
)
AUTHORIZED_POSTPROCESS_PREREG = (
    ROOT / "platform" / "docs" / "diag" / "v41_fresh151_postprocess_prereg_20260729.md"
)
AUTHORIZED_V41_GRAPH_IDENTITY_SHA256 = (
    "8b58815e1b50d0adb317efaae8c0db3d8506b199c5a3a6e781818a928f50c7cc"
)
AUTHORIZED_V41_GRAPH_CONTRACT = {
    "closure": "v41",
    "topology": ["N1", "N2", "N3", "N4", "N5", "N6", "R0"],
    "nodes": [
        {
            "id": "N1",
            "state": "validated",
            "freeze": True,
            "runtime_role": "physics",
            "implementation": "claim_runtime.components:UVLMComponent",
            "implementation_version": "1",
            "implementation_hash": (
                "sha256:"
                "f4c5d11c28ba5f4d71132c9c601544ab6b9b2728e404df27bac781fd8304dc2c"
            ),
        },
        {
            "id": "N2",
            "state": "partial",
            "freeze": False,
            "runtime_role": "physics",
            "implementation": "claim_runtime.components:KirchhoffLBComponent",
            "implementation_version": "1",
            "implementation_hash": (
                "sha256:"
                "909d6107e491c201426c40f2571dc34cc5ad46a96b513e9d1232485e3a634626"
            ),
        },
        {
            "id": "N3",
            "state": "partial",
            "freeze": False,
            "runtime_role": "physics",
            "implementation": "claim_runtime.components:DSVortexComponent",
            "implementation_version": "1",
            "implementation_hash": (
                "sha256:"
                "b1bd9babc348114ee84c5fd39deddf515edc2553332385532c4532dda79de4a8"
            ),
        },
        {
            "id": "N4",
            "state": "validated",
            "freeze": True,
            "runtime_role": "diagnostic",
            "implementation": "claim_runtime.components:CTConsistencyComponent",
            "implementation_version": "1",
            "implementation_hash": (
                "sha256:"
                "1b35c6edc52bf23b04c8e8fc3b9bb5cbac39baa958c4e071b9d5abca61a4cc4f"
            ),
        },
        {
            "id": "N5",
            "state": "falsified",
            "freeze": False,
            "runtime_role": "diagnostic",
            "implementation": "claim_runtime.components:TwistResponseObserver",
            "implementation_version": "1",
            "implementation_hash": (
                "sha256:"
                "aa493021506ccc5f603f8f5b5533dbc5628c0c755f449441f7525a81192bc823"
            ),
        },
        {
            "id": "N6",
            "state": "dead_end",
            "freeze": True,
            "runtime_role": "necessary_physics",
            "implementation": "claim_runtime.components:RigDragComponent",
            "implementation_version": "1",
            "implementation_hash": (
                "sha256:"
                "0a882389dd5cdac7d8fe811087ece894da90c215c0d9b4d049ef57bc9f1375db"
            ),
        },
        {
            "id": "R0",
            "state": "validated",
            "freeze": True,
            "runtime_role": "diagnostic",
            "implementation": "claim_runtime.components:CycleReductionComponent",
            "implementation_version": "2-validated",
            "implementation_hash": (
                "sha256:"
                "a0b0f13578b08db42db094919b0e38ce9420480fae0e0c4bcf39d6f956d52f1b"
            ),
        },
    ],
    "parameter_sources": {
        "closure": "resolved gpu_run_twist configuration",
        "lb_hybrid": "resolved gpu_run_twist configuration",
        "lb_cds": "resolved gpu_run_twist configuration",
        "lb_cla3d": "resolved gpu_run_twist configuration",
        "d_para": "resolved gpu_run_twist configuration",
    },
}
AUTHORIZED_AERO_OUTPUT_FIELDS = (
    "L",
    "Fx",
    "T",
    "P",
    "Fx_body",
    "Fz_body",
    "L_body",
    "T_body_f",
    "L_wind",
    "T_wind",
)
AUTHORIZED_GUARD_CONTRACT = {
    "force_ledger": {
        "tolerance_N": 1.0e-9,
        "body_force_required": False,
    },
    "unclassified_force": {
        "tolerance_N": 1.0e-9,
        "body_force_required": True,
    },
    "unclassified_physical_force": {
        "tolerance_N": 1.0e-9,
        "body_force_required": True,
    },
    "cycle_reduction": {
        "tolerance_N": 1.0e-12,
        "body_force_required": True,
    },
    "aero_output_invariance": {
        "tolerance_N": 0.0,
        "body_force_required": False,
        "checked_fields": list(AUTHORIZED_AERO_OUTPUT_FIELDS),
        "changed_fields": [],
    },
}
AUTHORIZED_INTERPRETATION = {
    "contains_partial_force_or_accuracy_results": False,
    "authorizes_only_completed_confirmed151_postprocessing": True,
    "fig19_cd_frequency_identity": "unresolved",
    "global_condition_union": (
        "unresolved: 184 only if Fig19(c,d) share one frequency; "
        "217 if they use different frequencies"
    ),
}
EXPECTED_CONTRIBUTION_NODES = frozenset(("N1", "N2", "N3", "N4", "N6", "R0"))
EXPECTED_CONTRIBUTION_INVENTORY = {
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
GUARD_NAMES = frozenset(
    (
        "force_ledger",
        "unclassified_force",
        "unclassified_physical_force",
        "cycle_reduction",
        "aero_output_invariance",
    )
)
RUNTIME_IDENTITY_KEYS = frozenset(
    (
        "python_executable",
        "python_version",
        "python_implementation",
        "platform",
        "numpy_version",
        "warp_version",
        "solver_config",
        "warp_device",
        "environment",
    )
)
RUNTIME_SOLVER_CONFIG_KEYS = frozenset(
    (
        "dtype_name",
        "dtype",
        "numpy_dtype",
        "device",
        "CR_TOL",
        "NEWTON_TOL",
        "GEOM_ATOL",
        "PORT_ATOL",
    )
)
RUNTIME_WARP_DEVICE_KEYS = frozenset(("text", "alias", "name", "arch"))
RUNTIME_ENVIRONMENT_KEYS = frozenset(
    (
        "FLUXV_DTYPE",
        "FLUXV_DEVICE",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
)
EXPECTED_FIGURE_CURVES = {"17": 10, "18": 24, "19": 8}
EXPECTED_FIGURE_PF = {"17": 10, "18": 24, "19": 8}
EXPECTED_PF_CELL_COUNTS = {
    "ALL": {
        "ALL": (34, 42),
        "T": (17, 21),
        "L": (17, 21),
    },
    "17": {
        "ALL": (10, 10),
        "T": (5, 5),
        "L": (5, 5),
    },
    "18": {
        "ALL": (24, 24),
        "T": (12, 12),
        "L": (12, 12),
    },
    "19": {
        "ALL": (8, 8),
        "T": (4, 4),
        "L": (4, 4),
    },
}
EXPECTED_CONDITIONAL_CURVES = frozenset(
    f"19|{panel}|{aoa:g}" for panel in ("c", "d") for aoa in benchmark.AOAS
)
CONFIRMED_CONDITIONS = tuple(
    benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[benchmark.EVIDENCE_CONFIRMED]
)
CONDITION_BY_KEY = {
    benchmark.condition_key(condition): condition for condition in CONFIRMED_CONDITIONS
}
EXPECTED_KEYS = frozenset(CONDITION_BY_KEY)
if len(EXPECTED_KEYS) != EXPECTED_CONDITIONS:
    raise AssertionError("confirmed151 condition contract drift")


class BaselineContractError(RuntimeError):
    """A fresh-baseline artifact failed a non-negotiable integrity gate."""


@dataclass(frozen=True)
class ValidatedBundle:
    result_path: Path
    manifest_path: Path
    contributions_path: Path
    results: dict[str, Any]
    manifest: dict[str, Any]
    contribution_file: dict[str, Any]
    validation: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_hash(left) == _canonical_hash(right)
    except (TypeError, ValueError):
        return False


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineContractError(f"{path}: cannot load JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineContractError(f"{path}: expected a JSON object")
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise BaselineContractError(f"missing input file: {resolved}")
    return {
        "path": _display_path(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Any) -> None:
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
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    finally:
        if partial.exists():
            partial.unlink()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, (bool, str)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineContractError(message)


def _validate_hash_mapping(
    value: Any,
    *,
    expected_count: int,
    label: str,
) -> None:
    _require(isinstance(value, Mapping), f"{label} is not a mapping")
    _require(
        len(value) == expected_count,
        f"{label} count {len(value)} != {expected_count}",
    )
    _require(
        all(
            isinstance(path, str) and bool(path) and _is_sha256(digest)
            for path, digest in value.items()
        ),
        f"{label} contains malformed path/hash records",
    )


def _validate_identity_record(value: Any, *, label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} identity is not a mapping")
    _require(
        isinstance(value.get("path"), str) and bool(value["path"]),
        f"{label} identity lacks path",
    )
    _require(_is_sha256(value.get("sha256")), f"{label} identity lacks SHA-256")


def _validate_postprocess_authorization(
    *,
    authorization_path: Path,
    expected_authorization_path: Path,
    expected_authorization_sha256: str,
    scoring_prereg_path: Path,
    expected_prereg_path: Path,
    expected_prereg_sha256: str,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
    data_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a completed manifest to the independently frozen root of trust."""

    authorization_path = authorization_path.resolve()
    scoring_prereg_path = scoring_prereg_path.resolve()
    _require(
        authorization_path.is_file(),
        f"missing postprocess authorization: {authorization_path}",
    )
    _require(
        scoring_prereg_path.is_file(),
        f"missing postprocess preregistration: {scoring_prereg_path}",
    )
    _require(
        data_path.resolve().is_file(),
        f"missing measurement data: {data_path.resolve()}",
    )
    _require(
        authorization_path == expected_authorization_path.resolve(),
        "postprocess authorization path is not the authorized root of trust",
    )
    _require(
        scoring_prereg_path == expected_prereg_path.resolve(),
        "postprocess preregistration path is not authorized",
    )
    _require(
        _sha256_file(authorization_path) == expected_authorization_sha256,
        "postprocess authorization SHA-256 mismatch",
    )
    _require(
        _sha256_file(scoring_prereg_path) == expected_prereg_sha256,
        "postprocess preregistration SHA-256 mismatch",
    )
    _require(
        _sha256_file(data_path.resolve()) == AUTHORIZED_MEASUREMENT_DATA_SHA256,
        "measurement-data SHA-256 differs from the frozen scoring input",
    )
    benchmark_path = Path(benchmark.__file__).resolve()
    fingerprint_path = Path(residual_fingerprint.__file__).resolve()
    _require(
        _sha256_file(benchmark_path) == AUTHORIZED_BENCHMARK_SOURCE_SHA256,
        "benchmark scorer SHA-256 differs from the frozen implementation",
    )
    _require(
        _sha256_file(fingerprint_path) == AUTHORIZED_FINGERPRINT_SOURCE_SHA256,
        "residual-fingerprint SHA-256 differs from the frozen implementation",
    )

    authorization = _load_json(authorization_path)
    _require(
        set(authorization)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "run_id",
            "scope",
            "frozen_manifest_identity",
            "claim_graph_contract",
            "guard_contract",
            "frozen_postprocess_inputs",
            "interpretation",
        },
        "postprocess authorization schema mismatch",
    )
    _require(
        authorization.get("schema_version") == 1
        and authorization.get("artifact_type")
        == "v41_fresh151_postprocess_authorization"
        and authorization.get("status")
        == "AUTHORIZED_IDENTITY_ONLY_BEFORE_CAMPAIGN_COMPLETION",
        "postprocess authorization identity/status mismatch",
    )
    run_id = manifest.get("run_id")
    _require(
        authorization.get("run_id") == run_id,
        "authorization/manifest run_id mismatch",
    )

    expected_scope = {
        "result_path": _display_path(result_path.resolve()),
        "manifest_path": _display_path(manifest_path.resolve()),
        "contributions_path": _display_path(contributions_path.resolve()),
        "expected_condition_count": EXPECTED_CONDITIONS,
        "confirmed_curve_count": EXPECTED_CURVES,
        "confirmed_measurement_count": EXPECTED_SAMPLES,
        "physical_family_count": EXPECTED_PHYSICAL_FAMILIES,
        "alias_group_count": EXPECTED_ALIAS_GROUPS,
    }
    _require(
        _canonical_equal(authorization.get("scope"), expected_scope),
        "postprocess authorization scope differs from supplied confirmed151 triplet",
    )

    solver_hashes = manifest.get("solver_source_hashes")
    control_hashes = manifest.get("control_source_hashes")
    _validate_hash_mapping(
        solver_hashes,
        expected_count=EXPECTED_SOLVER_SOURCES,
        label="authorized solver source closure",
    )
    _validate_hash_mapping(
        control_hashes,
        expected_count=EXPECTED_CONTROL_SOURCES,
        label="authorized control source closure",
    )
    for field in (
        "runner",
        "preregistration",
        "integrity_addendum",
        "launch_authorization",
    ):
        _validate_identity_record(manifest.get(field), label=field)
    runtime_identity = manifest.get("runtime_identity")
    _validate_runtime_identity_schema(runtime_identity)
    graph_identity = manifest.get("claim_graph_identity_sha256")
    common_claim_manifest = manifest.get("common_claim_manifest")
    _validate_authorized_graph(common_claim_manifest, graph_identity)
    expected_keys = manifest.get("expected_condition_keys")
    resolved_call_contract = manifest.get("resolved_call_contract_sha256")
    _require(
        isinstance(expected_keys, list)
        and expected_keys == sorted(EXPECTED_KEYS)
        and _is_sha256(resolved_call_contract),
        "manifest condition/call contract is malformed",
    )

    computed_manifest_identity = {
        "solver_source_count": len(solver_hashes),
        "solver_source_hashes_canonical_sha256": _canonical_hash(solver_hashes),
        "control_source_count": len(control_hashes),
        "control_source_hashes_canonical_sha256": _canonical_hash(control_hashes),
        "runner_identity_canonical_sha256": _canonical_hash(manifest["runner"]),
        "preregistration_identity_canonical_sha256": _canonical_hash(
            manifest["preregistration"]
        ),
        "integrity_addendum_identity_canonical_sha256": _canonical_hash(
            manifest["integrity_addendum"]
        ),
        "launch_authorization_identity_canonical_sha256": _canonical_hash(
            manifest["launch_authorization"]
        ),
        "expected_condition_keys_canonical_sha256": _canonical_hash(expected_keys),
        "resolved_call_contract_sha256": resolved_call_contract,
        "runtime_identity_canonical_sha256": _canonical_hash(runtime_identity),
        "claim_graph_identity_sha256": graph_identity,
        "common_claim_manifest_static_canonical_sha256": _canonical_hash(
            _claim_graph_static_payload(common_claim_manifest)
        ),
    }
    _require(
        _canonical_equal(
            authorization.get("frozen_manifest_identity"),
            computed_manifest_identity,
        ),
        "completed manifest identity differs from postprocess authorization",
    )
    _require(
        _canonical_equal(
            authorization.get("claim_graph_contract"),
            AUTHORIZED_V41_GRAPH_CONTRACT,
        )
        and _canonical_equal(
            authorization.get("guard_contract"),
            AUTHORIZED_GUARD_CONTRACT,
        ),
        "authorization graph/guard contract differs from authorized V4.1",
    )
    expected_postprocess_inputs = {
        "measurement_data_sha256": AUTHORIZED_MEASUREMENT_DATA_SHA256,
        "benchmark_source_sha256": AUTHORIZED_BENCHMARK_SOURCE_SHA256,
        "fingerprint_source_sha256": AUTHORIZED_FINGERPRINT_SOURCE_SHA256,
    }
    _require(
        _canonical_equal(
            authorization.get("frozen_postprocess_inputs"),
            expected_postprocess_inputs,
        ),
        "authorization frozen postprocess inputs mismatch",
    )
    _require(
        _canonical_equal(
            authorization.get("interpretation"),
            AUTHORIZED_INTERPRETATION,
        ),
        "authorization interpretation contract mismatch",
    )
    return authorization


def _preflight_complete_manifest(
    *,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    """Check completion before invoking the runner's zero-GPU resume gate."""

    for path in (result_path, manifest_path, contributions_path):
        _require(path.resolve().is_file(), f"missing fresh151 input: {path.resolve()}")
    manifest = _load_json(manifest_path.resolve())
    _require(
        manifest.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "fresh151 manifest schema mismatch",
    )
    _require(
        manifest.get("status") == "complete",
        "fresh151 manifest is not complete; scoring partial/running data is forbidden",
    )
    run_id = manifest.get("run_id")
    _require(
        isinstance(run_id, str) and bool(run_id) and run_id.replace("_", "").isdigit(),
        "fresh151 run_id is not a runner timestamp",
    )
    expected_paths = {
        "result_path": _display_path(result_path.resolve()),
        "manifest_path": _display_path(manifest_path.resolve()),
        "contributions_path": _display_path(contributions_path.resolve()),
    }
    for field, expected in expected_paths.items():
        _require(
            manifest.get(field) == expected,
            f"manifest {field} does not bind supplied file",
        )
    _require(
        manifest.get("expected_condition_count") == EXPECTED_CONDITIONS
        and manifest.get("completed_condition_count") == EXPECTED_CONDITIONS,
        "fresh151 manifest is not a completed 151-condition campaign",
    )
    triplet_receipts = {
        "result": _sha256_file(result_path.resolve()),
        "manifest": _sha256_file(manifest_path.resolve()),
        "contributions": _sha256_file(contributions_path.resolve()),
    }
    return manifest, run_id, triplet_receipts


def _production_completion_validator(
    *,
    run_id: str,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
) -> None:
    """Re-enter the frozen runner in complete/resume mode (zero new cases)."""

    import lb_sweep151_fresh as runner

    expected = tuple(path.resolve() for path in runner._paths(run_id))
    supplied = tuple(
        path.resolve() for path in (result_path, manifest_path, contributions_path)
    )
    _require(
        supplied == expected,
        "supplied triplet is not the frozen runner path set for this timestamp",
    )
    return_code = runner.run(timestamp=run_id, resume=True)
    _require(
        return_code == 0,
        "fresh151 complete-resume revalidation did not return success",
    )


def _claim_graph_static_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "closure": manifest.get("closure"),
        "topology": copy.deepcopy(manifest.get("topology")),
        "nodes": copy.deepcopy(manifest.get("nodes")),
        "parameter_sources": copy.deepcopy(manifest.get("parameter_sources")),
    }


def _validate_authorized_graph(
    common_claim_manifest: Any,
    graph_identity: Any,
) -> None:
    _require(
        isinstance(common_claim_manifest, Mapping),
        "common claim manifest is missing",
    )
    _require(
        graph_identity == AUTHORIZED_V41_GRAPH_IDENTITY_SHA256,
        "claim graph identity is not the authorized V4.1 graph",
    )
    static_payload = _claim_graph_static_payload(common_claim_manifest)
    _require(
        _canonical_equal(static_payload, AUTHORIZED_V41_GRAPH_CONTRACT),
        "claim graph static contract is not authorized V4.1",
    )
    try:
        recomputed = witness._claim_graph_identity_sha256(common_claim_manifest)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise BaselineContractError(
            f"common claim manifest is malformed: {exc}"
        ) from exc
    _require(
        recomputed == AUTHORIZED_V41_GRAPH_IDENTITY_SHA256,
        "common claim manifest does not reproduce authorized V4.1 identity",
    )
    _validate_guard_set(
        common_claim_manifest.get("guards"),
        label="common claim manifest",
    )


def _validate_runtime_identity_schema(value: Any) -> None:
    _require(
        isinstance(value, Mapping) and set(value) == RUNTIME_IDENTITY_KEYS,
        "runtime identity schema mismatch",
    )
    for name in (
        "python_executable",
        "python_version",
        "python_implementation",
        "platform",
        "numpy_version",
        "warp_version",
    ):
        _require(
            isinstance(value.get(name), str) and bool(value[name]),
            f"runtime identity {name} is invalid",
        )
    solver_config = value.get("solver_config")
    _require(
        isinstance(solver_config, Mapping)
        and set(solver_config) == RUNTIME_SOLVER_CONFIG_KEYS,
        "runtime solver-config schema mismatch",
    )
    for name in ("dtype_name", "dtype", "numpy_dtype", "device"):
        _require(
            isinstance(solver_config.get(name), str) and bool(solver_config[name]),
            f"runtime solver-config {name} is invalid",
        )
    for name in ("CR_TOL", "NEWTON_TOL", "GEOM_ATOL", "PORT_ATOL"):
        tolerance = _finite_float(solver_config.get(name))
        _require(
            tolerance is not None and tolerance > 0.0,
            f"runtime solver-config {name} is invalid",
        )
    warp_device = value.get("warp_device")
    _require(
        isinstance(warp_device, Mapping)
        and set(warp_device) == RUNTIME_WARP_DEVICE_KEYS,
        "runtime Warp-device schema mismatch",
    )
    for name in ("text", "alias", "name"):
        _require(
            isinstance(warp_device.get(name), str) and bool(warp_device[name]),
            f"runtime Warp-device {name} is invalid",
        )
    arch = warp_device.get("arch")
    _require(
        isinstance(arch, int) and not isinstance(arch, bool) and arch > 0,
        "runtime Warp-device arch is invalid",
    )
    environment = value.get("environment")
    _require(
        isinstance(environment, Mapping)
        and set(environment) == RUNTIME_ENVIRONMENT_KEYS,
        "runtime environment schema mismatch",
    )
    _require(
        all(item is None or isinstance(item, str) for item in environment.values()),
        "runtime environment contains a non-string value",
    )


def _validate_force_value(value: Any, *, label: str) -> None:
    _require(
        isinstance(value, Mapping) and set(value) == {"L", "T"},
        f"{label}: value must contain exactly L/T",
    )
    _require(
        _finite_float(value.get("L")) is not None
        and _finite_float(value.get("T")) is not None,
        f"{label}: value contains non-finite L/T",
    )


def _validate_force_delta(value: Any, *, label: str) -> None:
    _require(
        isinstance(value, Mapping) and set(value) == {"L_N", "T_N"},
        f"{label}: delta must contain exactly L_N/T_N",
    )
    _require(
        _finite_float(value.get("L_N")) is not None
        and _finite_float(value.get("T_N")) is not None,
        f"{label}: delta contains non-finite values",
    )


def _validate_runtime_sessions(
    manifest: Mapping[str, Any],
    *,
    results: Mapping[str, Any],
    cases: Mapping[str, Any],
    case_guards: Mapping[str, Any],
    graph_identity: str,
) -> None:
    cold = manifest.get("cold_preconditioner")
    formal = manifest.get("formal_anchor")
    _require(
        isinstance(cold, Mapping)
        and set(cold)
        == {
            "value",
            "claim_graph_identity_sha256",
            "guards",
            "discarded_from_results",
        },
        "cold-preconditioner schema mismatch",
    )
    _validate_force_value(cold.get("value"), label="cold preconditioner")
    _require(
        cold.get("claim_graph_identity_sha256") == graph_identity
        and cold.get("discarded_from_results") is True,
        "cold-preconditioner graph/discard contract mismatch",
    )
    _validate_guard_set(cold.get("guards"), label="cold preconditioner")

    _require(
        isinstance(formal, Mapping)
        and set(formal) == {"condition_key", "value", "old_baseline_delta_N", "guards"},
        "formal-anchor schema mismatch",
    )
    anchor_key = formal.get("condition_key")
    _require(
        isinstance(anchor_key, str)
        and anchor_key in EXPECTED_KEYS
        and anchor_key in results
        and anchor_key in cases
        and anchor_key in case_guards,
        "formal-anchor condition is not in confirmed151",
    )
    _validate_force_value(formal.get("value"), label="formal anchor")
    _validate_force_delta(
        formal.get("old_baseline_delta_N"),
        label="formal anchor",
    )
    _validate_guard_set(formal.get("guards"), label="formal anchor")
    _require(
        _canonical_equal(formal["value"], results[anchor_key])
        and _canonical_equal(formal["guards"], case_guards[anchor_key])
        and _canonical_equal(
            formal["old_baseline_delta_N"],
            cases[anchor_key].get("signed_old_baseline_delta_N"),
        ),
        "formal-anchor evidence differs from its saved case",
    )

    sessions = manifest.get("runtime_sessions")
    _require(
        isinstance(sessions, list) and bool(sessions),
        "runtime session evidence is missing",
    )
    for index, session in enumerate(sessions):
        label = f"runtime session {index}"
        _require(
            isinstance(session, Mapping)
            and set(session) == {"started_at", "cold_preconditioner", "warm_anchor"},
            f"{label}: schema mismatch",
        )
        _require(
            isinstance(session.get("started_at"), str) and bool(session["started_at"]),
            f"{label}: start time missing",
        )
        session_cold = session.get("cold_preconditioner")
        session_warm = session.get("warm_anchor")
        _require(
            isinstance(session_cold, Mapping)
            and set(session_cold) == {"value", "graph_identity"},
            f"{label}: cold-anchor schema mismatch",
        )
        _require(
            isinstance(session_warm, Mapping)
            and set(session_warm)
            == {"value", "old_baseline_delta_N", "graph_identity"},
            f"{label}: warm-anchor schema mismatch",
        )
        _validate_force_value(
            session_cold.get("value"),
            label=f"{label} cold anchor",
        )
        _validate_force_value(
            session_warm.get("value"),
            label=f"{label} warm anchor",
        )
        _validate_force_delta(
            session_warm.get("old_baseline_delta_N"),
            label=f"{label} warm anchor",
        )
        _require(
            session_cold.get("graph_identity") == graph_identity
            and session_warm.get("graph_identity") == graph_identity,
            f"{label}: graph identity mismatch",
        )
        _require(
            _canonical_equal(session_cold["value"], cold["value"])
            and _canonical_equal(session_warm["value"], formal["value"])
            and _canonical_equal(
                session_warm["old_baseline_delta_N"],
                formal["old_baseline_delta_N"],
            ),
            f"{label}: anchor evidence differs from completed manifest anchors",
        )


def _condition_record(
    condition: tuple[float, float, float, float],
) -> dict[str, float]:
    U, frequency, twist, aoa = condition
    return {
        "U_m_s": U,
        "frequency_Hz": frequency,
        "nominal_twist_deg": twist,
        "solver_twist_amplitude_deg": twist / 2.0,
        "aoa_deg": aoa,
    }


def _validate_guard_set(value: Any, *, label: str) -> float:
    _require(isinstance(value, Mapping), f"{label}: guards are not a mapping")
    _require(set(value) == GUARD_NAMES, f"{label}: guard inventory mismatch")
    maximum_error = 0.0
    for name in sorted(GUARD_NAMES):
        guard = value[name]
        _require(isinstance(guard, Mapping), f"{label}/{name}: malformed guard")
        contract = AUTHORIZED_GUARD_CONTRACT[name]
        expected_fields = {
            "passed",
            "max_abs_error_N",
            "tolerance_N",
        }
        if contract["body_force_required"]:
            expected_fields.add("body_force_N")
        if name == "aero_output_invariance":
            expected_fields.update(("checked_fields", "changed_fields"))
        _require(
            set(guard) == expected_fields,
            f"{label}/{name}: guard schema differs from authorized V4.1",
        )
        _require(guard.get("passed") is True, f"{label}/{name}: guard failed")
        error = _finite_float(guard.get("max_abs_error_N"))
        tolerance = _finite_float(guard.get("tolerance_N"))
        _require(
            error is not None
            and tolerance is not None
            and error >= 0.0
            and tolerance == float(contract["tolerance_N"])
            and error <= tolerance,
            f"{label}/{name}: error/tolerance differs from authorized V4.1",
        )
        maximum_error = max(maximum_error, error)
        if contract["body_force_required"]:
            body = guard["body_force_N"]
            _require(
                isinstance(body, (list, tuple))
                and len(body) == 3
                and all(_finite_float(component) is not None for component in body),
                f"{label}/{name}: invalid body force",
            )
            if name in {
                "unclassified_force",
                "unclassified_physical_force",
            }:
                body_max = max(abs(float(component)) for component in body)
                _require(
                    body_max == error,
                    f"{label}/{name}: body-force maximum differs from reported error",
                )
            # cycle_reduction.body_force_N is the physical reconciliation
            # contribution, while max_abs_error_N is the error after applying
            # it.  They are deliberately not the same physical quantity.
        if name == "aero_output_invariance":
            _require(
                guard.get("checked_fields") == list(AUTHORIZED_AERO_OUTPUT_FIELDS)
                and guard.get("changed_fields") == [],
                f"{label}/{name}: output-field contract drift",
            )
    return maximum_error


def _validate_contribution_inventory(
    contributions: Any,
    *,
    label: str,
) -> None:
    _require(
        isinstance(contributions, Mapping),
        f"{label}: claim contributions are not a mapping",
    )
    _require(
        set(contributions) == EXPECTED_CONTRIBUTION_NODES,
        f"{label}: contribution node inventory mismatch",
    )
    for node_id, expected in EXPECTED_CONTRIBUTION_INVENTORY.items():
        items = contributions[node_id]
        _require(
            isinstance(items, list) and bool(items),
            f"{label}/{node_id}: empty or malformed contribution list",
        )
        actual: list[tuple[Any, Any]] = []
        for item in items:
            _require(
                isinstance(item, Mapping)
                and set(item) == {"body_force", "channel", "metadata", "role"},
                f"{label}/{node_id}: contribution record schema mismatch",
            )
            body = item.get("body_force")
            _require(
                isinstance(body, (list, tuple))
                and len(body) == 3
                and all(_finite_float(component) is not None for component in body),
                f"{label}/{node_id}: non-finite body force",
            )
            _require(
                isinstance(item.get("metadata"), Mapping),
                f"{label}/{node_id}: metadata is not a mapping",
            )
            actual.append((item.get("channel"), item.get("role")))
        _require(
            tuple(actual) == expected,
            f"{label}/{node_id}: channel/role inventory mismatch",
        )


def _validate_result_record(value: Any, *, key: str) -> tuple[float, float]:
    _require(
        isinstance(value, Mapping) and set(value) == {"L", "T"},
        f"{key}: result must contain exactly L/T",
    )
    lift = _finite_float(value.get("L"))
    thrust = _finite_float(value.get("T"))
    _require(lift is not None and thrust is not None, f"{key}: non-finite L/T")
    return lift, thrust


def _validate_resolved_call(
    value: Any,
    *,
    condition: tuple[float, float, float, float],
    label: str,
) -> None:
    _require(isinstance(value, Mapping), f"{label}: resolved call is not a mapping")
    U, frequency, twist, aoa = condition
    expected = {
        "U": U,
        "freq": frequency,
        "twist_amp_deg": twist / 2.0,
        "aoa_deg": aoa,
        "closure": "v41",
    }
    for name, expected_value in expected.items():
        _require(
            value.get(name) == expected_value,
            f"{label}: resolved call {name} mismatch",
        )


def _validate_stored_ledger(
    stored: Any,
    *,
    ledger_total: np.ndarray,
    wind_total: Mapping[str, float],
    label: str,
) -> None:
    _require(isinstance(stored, Mapping), f"{label}: recomputed ledger missing")
    _require(
        _canonical_equal(stored.get("total_body_force_N"), ledger_total.tolist()),
        f"{label}: stored body ledger mismatch",
    )
    _require(
        _canonical_equal(stored.get("total_wind_force"), wind_total),
        f"{label}: stored wind ledger mismatch",
    )
    for name in ("max_body_error_N", "max_wind_error_N"):
        error = _finite_float(stored.get(name))
        _require(
            error is not None and 0.0 <= error <= LEDGER_TOLERANCE_N,
            f"{label}: stored {name} exceeds tolerance",
        )


def _validate_case(
    *,
    key: str,
    condition: tuple[float, float, float, float],
    result: Any,
    evidence: Any,
    manifest_guard: Any,
    graph_identity: str,
    common_claim_manifest: Mapping[str, Any],
) -> tuple[float, float, float]:
    lift, thrust = _validate_result_record(result, key=key)
    label = f"case {key}"
    _require(isinstance(evidence, Mapping), f"{label}: evidence is not a mapping")
    _require(evidence.get("condition_key") == key, f"{label}: key mismatch")
    _require(
        _canonical_equal(evidence.get("condition"), _condition_record(condition)),
        f"{label}: condition identity mismatch",
    )
    _validate_resolved_call(
        evidence.get("resolved_call"),
        condition=condition,
        label=label,
    )
    _require(
        evidence.get("claim_graph_identity_sha256") == graph_identity,
        f"{label}: graph identity mismatch",
    )
    guards = evidence.get("claim_guards")
    guard_error = _validate_guard_set(guards, label=label)
    _require(
        _canonical_equal(guards, manifest_guard),
        f"{label}: manifest/evidence guard mismatch",
    )
    case_manifest = dict(common_claim_manifest)
    case_manifest["guards"] = dict(guards)
    _require(
        evidence.get("claim_manifest_sha256") == _canonical_hash(case_manifest),
        f"{label}: case manifest hash mismatch",
    )

    old_baseline = evidence.get("old_baseline")
    signed_delta = evidence.get("signed_old_baseline_delta_N")
    _require(
        isinstance(old_baseline, Mapping) and isinstance(signed_delta, Mapping),
        f"{label}: old-baseline diagnostic evidence missing",
    )
    old_lift = _finite_float(old_baseline.get("L_N"))
    old_thrust = _finite_float(old_baseline.get("T_N"))
    _require(
        old_lift is not None and old_thrust is not None,
        f"{label}: old-baseline diagnostic is non-finite",
    )
    _require(
        _canonical_equal(
            signed_delta,
            {"L_N": lift - old_lift, "T_N": thrust - old_thrust},
        ),
        f"{label}: old-baseline delta mismatch",
    )

    contributions = evidence.get("claim_contributions")
    _validate_contribution_inventory(contributions, label=label)
    try:
        summary, ledger_total = witness._contribution_summary(
            contributions,
            aoa_deg=condition[3],
        )
        wind_total = witness._wind_force(ledger_total, condition[3])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise BaselineContractError(
            f"{label}: contribution reduction failed: {exc}"
        ) from exc
    _require(
        _canonical_equal(evidence.get("contribution_summary"), summary),
        f"{label}: stored contribution summary mismatch",
    )

    angle = math.radians(condition[3])
    target_body = np.asarray(
        [
            -thrust * math.cos(angle) - lift * math.sin(angle),
            0.0,
            lift * math.cos(angle) - thrust * math.sin(angle),
        ],
        dtype=float,
    )
    body_error = float(np.max(np.abs(ledger_total - target_body)))
    wind_error = max(
        abs(float(wind_total["L_N"]) - lift),
        abs(float(wind_total["T_N"]) - thrust),
    )
    _require(
        math.isfinite(body_error)
        and math.isfinite(wind_error)
        and body_error <= LEDGER_TOLERANCE_N
        and wind_error <= LEDGER_TOLERANCE_N,
        f"{label}: force ledger does not close to L/T "
        f"(body={body_error}, wind={wind_error})",
    )
    _validate_stored_ledger(
        evidence.get("recomputed_ledger"),
        ledger_total=ledger_total,
        wind_total=wind_total,
        label=label,
    )
    wall_s = _finite_float(evidence.get("wall_s"))
    _require(wall_s is not None and wall_s >= 0.0, f"{label}: invalid wall time")
    return body_error, wind_error, guard_error


def _validate_coverage(
    manifest_coverage: Any,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    recomputed = benchmark.coverage(
        results,
        evidence_scope=benchmark.EVIDENCE_CONFIRMED,
    )
    _require(
        isinstance(manifest_coverage, Mapping),
        "manifest lacks final_confirmed_coverage",
    )
    _require(
        _canonical_equal(manifest_coverage, recomputed),
        "manifest final_confirmed_coverage differs from recomputation",
    )
    expected_scalars = {
        "evidence_scope": benchmark.EVIDENCE_CONFIRMED,
        "expected_unique_conditions": EXPECTED_CONDITIONS,
        "valid_unique_conditions": EXPECTED_CONDITIONS,
        "missing_unique_conditions": 0,
        "expected_curves": EXPECTED_CURVES,
        "complete_curves": EXPECTED_CURVES,
        "partial_curves": 0,
        "empty_curves": 0,
        "complete": True,
    }
    for name, expected in expected_scalars.items():
        _require(
            recomputed.get(name) == expected,
            f"confirmed coverage {name} mismatch",
        )
    _require(
        recomputed.get("missing_condition_keys") == []
        and recomputed.get("extra_condition_keys") == [],
        "confirmed coverage contains missing/extra conditions",
    )
    for figure, expected_curves in EXPECTED_FIGURE_CURVES.items():
        record = recomputed.get("figures", {}).get(figure)
        _require(
            record
            == {
                "expected_curves": expected_curves,
                "complete_curves": expected_curves,
                "partial_curves": 0,
                "empty_curves": 0,
            },
            f"Fig{figure} confirmed coverage mismatch",
        )
    return recomputed


def validate_fresh151_bundle(
    *,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
) -> ValidatedBundle:
    """Validate a completed fresh151 triplet without consulting partial scores."""

    result_path = result_path.resolve()
    manifest_path = manifest_path.resolve()
    contributions_path = contributions_path.resolve()
    results = _load_json(result_path)
    manifest = _load_json(manifest_path)
    contribution_file = _load_json(contributions_path)

    _require(
        manifest.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "fresh151 manifest schema mismatch",
    )
    _require(
        contribution_file.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "fresh151 contribution schema mismatch",
    )
    _require(
        manifest.get("status") == "complete",
        "fresh151 manifest is not complete; scoring partial/running data is forbidden",
    )
    run_id = manifest.get("run_id")
    _require(isinstance(run_id, str) and bool(run_id), "fresh151 run_id is invalid")
    _require(
        contribution_file.get("run_id") == run_id,
        "manifest/contribution run_id mismatch",
    )
    expected_paths = {
        "result_path": _display_path(result_path),
        "manifest_path": _display_path(manifest_path),
        "contributions_path": _display_path(contributions_path),
    }
    for field, expected in expected_paths.items():
        _require(
            manifest.get(field) == expected,
            f"manifest {field} does not bind supplied file",
        )
        _require(
            contribution_file.get(field) == expected,
            f"contribution {field} does not bind supplied file",
        )

    _require(
        manifest.get("expected_condition_count") == EXPECTED_CONDITIONS
        and manifest.get("completed_condition_count") == EXPECTED_CONDITIONS,
        "fresh151 expected/completed count mismatch",
    )
    expected_sorted = sorted(EXPECTED_KEYS)
    _require(
        manifest.get("expected_condition_keys") == expected_sorted,
        "fresh151 expected condition key contract mismatch",
    )
    cases = contribution_file.get("cases")
    case_guards = manifest.get("case_guards")
    _require(isinstance(cases, Mapping), "contribution cases are not a mapping")
    _require(isinstance(case_guards, Mapping), "manifest case_guards are not a mapping")
    key_sets = {
        "results": set(results),
        "contribution cases": set(cases),
        "case guards": set(case_guards),
    }
    for label, keys in key_sets.items():
        _require(
            keys == EXPECTED_KEYS,
            f"{label} key set is not exact confirmed151 "
            f"(actual={len(keys)}, expected={EXPECTED_CONDITIONS})",
        )
    _require(manifest.get("failures") == {}, "fresh151 manifest contains failures")

    result_sha256 = manifest.get("result_sha256")
    contribution_sha256 = manifest.get("contributions_sha256")
    _require(_is_sha256(result_sha256), "manifest result receipt is malformed")
    _require(
        _is_sha256(contribution_sha256),
        "manifest contribution receipt is malformed",
    )
    _require(
        result_sha256 == _sha256_file(result_path),
        "fresh151 result SHA-256 mismatch",
    )
    _require(
        contribution_sha256 == _sha256_file(contributions_path),
        "fresh151 contribution SHA-256 mismatch",
    )

    _validate_hash_mapping(
        manifest.get("solver_source_hashes"),
        expected_count=EXPECTED_SOLVER_SOURCES,
        label="solver source closure",
    )
    _validate_hash_mapping(
        manifest.get("control_source_hashes"),
        expected_count=EXPECTED_CONTROL_SOURCES,
        label="control source closure",
    )
    for field in (
        "preregistration",
        "integrity_addendum",
        "launch_authorization",
        "runner",
    ):
        _validate_identity_record(manifest.get(field), label=field)
    _require(
        _is_sha256(manifest.get("resolved_call_contract_sha256")),
        "resolved-call contract receipt is malformed",
    )
    _validate_runtime_identity_schema(manifest.get("runtime_identity"))

    graph_identity = manifest.get("claim_graph_identity_sha256")
    common_claim_manifest = manifest.get("common_claim_manifest")
    _require(_is_sha256(graph_identity), "claim graph identity is malformed")
    _validate_authorized_graph(common_claim_manifest, graph_identity)
    if not isinstance(common_claim_manifest, Mapping):
        raise AssertionError("unreachable malformed common claim manifest")

    maximum_body_error = 0.0
    maximum_wind_error = 0.0
    maximum_guard_error = 0.0
    for key in expected_sorted:
        body_error, wind_error, guard_error = _validate_case(
            key=key,
            condition=CONDITION_BY_KEY[key],
            result=results[key],
            evidence=cases[key],
            manifest_guard=case_guards[key],
            graph_identity=graph_identity,
            common_claim_manifest=common_claim_manifest,
        )
        maximum_body_error = max(maximum_body_error, body_error)
        maximum_wind_error = max(maximum_wind_error, wind_error)
        maximum_guard_error = max(maximum_guard_error, guard_error)

    resolved_calls = {key: cases[key].get("resolved_call") for key in expected_sorted}
    _require(
        _canonical_hash(resolved_calls) == manifest["resolved_call_contract_sha256"],
        "saved cases do not reproduce the full resolved-call contract",
    )
    _validate_runtime_sessions(
        manifest,
        results=results,
        cases=cases,
        case_guards=case_guards,
        graph_identity=graph_identity,
    )

    final_coverage = _validate_coverage(
        manifest.get("final_confirmed_coverage"),
        results,
    )
    validation = {
        "passed": True,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "condition_count": len(results),
        "evidence_case_count": len(cases),
        "case_guard_count": len(case_guards),
        "solver_source_count": len(manifest["solver_source_hashes"]),
        "control_source_count": len(manifest["control_source_hashes"]),
        "claim_graph_identity_sha256": graph_identity,
        "maximum_recomputed_body_ledger_error_N": maximum_body_error,
        "maximum_recomputed_wind_ledger_error_N": maximum_wind_error,
        "maximum_reported_guard_error_N": maximum_guard_error,
        "ledger_tolerance_N": LEDGER_TOLERANCE_N,
        "coverage": final_coverage,
        "input_receipts": {
            "result_sha256": result_sha256,
            "manifest_sha256": _sha256_file(manifest_path),
            "contributions_sha256": contribution_sha256,
        },
    }
    return ValidatedBundle(
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
        results=results,
        manifest=manifest,
        contribution_file=contribution_file,
        validation=validation,
    )


def _confirmed_scorecard(
    report: Mapping[str, Any],
    *,
    bundle: ValidatedBundle,
) -> dict[str, Any]:
    _require(report.get("schema_version") == 3, "benchmark scorecard is not schema-v3")
    scopes = report.get("evidence_scopes")
    _require(
        isinstance(scopes, Mapping) and benchmark.EVIDENCE_CONFIRMED in scopes,
        "benchmark scorecard lacks confirmed scope",
    )
    confirmed = scopes[benchmark.EVIDENCE_CONFIRMED]
    _require(isinstance(confirmed, Mapping), "confirmed scope payload is malformed")
    rows = confirmed.get("rows")
    aggregates = confirmed.get("aggregates")
    coverage = confirmed.get("coverage")
    _require(isinstance(rows, list), "confirmed score rows are malformed")
    _require(len(rows) == EXPECTED_CURVES, "confirmed score row count mismatch")
    _require(
        sum(len(row.get("error_N", ())) for row in rows) == EXPECTED_SAMPLES,
        "confirmed residual sample count mismatch",
    )
    _require(
        all(
            row.get("evidence_scope") == benchmark.EVIDENCE_CONFIRMED
            and not str(row.get("curve", "")).startswith(("19|c|", "19|d|"))
            for row in rows
        ),
        "Fig19(c,d) leaked into confirmed score rows",
    )
    _require(
        _canonical_equal(coverage, bundle.validation["coverage"]),
        "scorecard confirmed coverage differs from validated runner coverage",
    )
    return {
        "schema_version": 3,
        "artifact_type": "fig171819_confirmed_scope_scorecard",
        "benchmark": report.get("benchmark"),
        "sweep": _display_path(bundle.result_path),
        "primary_evidence_scope": benchmark.EVIDENCE_CONFIRMED,
        "contract": {
            "confirmed_curves": EXPECTED_CURVES,
            "raw_measurement_samples": EXPECTED_SAMPLES,
            "solver_conditions": EXPECTED_CONDITIONS,
            "conditional_curves_excluded": len(EXPECTED_CONDITIONAL_CURVES),
            "measurement_values_interpolated": False,
            "scoring_direction": "model_to_raw_measurement_x",
            "measurement_validation": copy.deepcopy(
                report.get("contract", {}).get("measurement_validation")
            ),
            "source_benchmark_contract": copy.deepcopy(report.get("contract")),
        },
        "coverage": copy.deepcopy(coverage),
        "rows": copy.deepcopy(rows),
        "aggregates": copy.deepcopy(aggregates),
        "evidence_scopes": {
            benchmark.EVIDENCE_CONFIRMED: {
                "coverage": copy.deepcopy(coverage),
                "rows": copy.deepcopy(rows),
                "aggregates": copy.deepcopy(aggregates),
                "residual_fingerprint_ready": True,
                "allowed_use": "baseline_diagnosis_and_preregistration",
            }
        },
        "excluded_evidence_scope": {
            "scope": benchmark.EVIDENCE_CONDITIONAL_FIG19_CD,
            "curve_keys": sorted(EXPECTED_CONDITIONAL_CURVES),
            "reason": "Fig19(c,d) fixed frequency is unresolved.",
            "rows_in_this_scorecard": 0,
            "samples_in_this_scorecard": 0,
        },
        "confirmed_baseline_ready": True,
        "promotion_eligible": False,
        "promotion_blockers": ["fig19_cd_fixed_frequency_unresolved"],
    }


def _identity_with_role(path: Path, role: str) -> dict[str, Any]:
    return {**_file_identity(path), "role": role}


def _build_confirmed_artifact(
    *,
    scorecard: Mapping[str, Any],
    bundle: ValidatedBundle,
    scorecard_path: Path,
    data_path: Path,
    scoring_prereg_path: Path,
) -> dict[str, Any]:
    provenance = {
        "sweep_result": _file_identity(bundle.result_path),
        "runner_manifest": _file_identity(bundle.manifest_path),
        # Kept under the schema-v1 field name for compatibility with the
        # frozen residual-fingerprint validator.  This is the fresh,
        # independently generated confirmed-scope schema-v3 scorecard.
        "original_runner_scorecard": _file_identity(scorecard_path),
        "measurement_data": _file_identity(data_path),
        "scorer_source": _file_identity(Path(benchmark.__file__).resolve()),
    }
    artifact = benchmark.build_evidence_scope_artifact(
        scorecard,
        evidence_scope=benchmark.EVIDENCE_CONFIRMED,
        provenance=provenance,
    )
    artifact["bundle_validation"] = {
        "validation": copy.deepcopy(bundle.validation),
        "claim_contributions": _file_identity(bundle.contributions_path),
        "scoring_preregistration": _file_identity(scoring_prereg_path),
        "postprocessor_source": _file_identity(Path(__file__).resolve()),
        "provenance_field_semantics": {
            "original_runner_scorecard": (
                "fresh independently generated confirmed-scope schema-v3 scorecard"
            )
        },
    }
    return artifact


def _physical_family_contract() -> tuple[
    dict[str, str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    curve_to_family, families, aliases = (
        residual_fingerprint._physical_family_contract()
    )
    _require(
        len(families) == EXPECTED_PHYSICAL_FAMILIES,
        "physical-family count mismatch",
    )
    _require(len(aliases) == EXPECTED_ALIAS_GROUPS, "alias-group count mismatch")
    contract = {
        "families": families,
        "aliases": aliases,
    }
    return curve_to_family, families, aliases, _canonical_hash(contract)


def _turn_kinds(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item.get("kind")) for item in value if isinstance(item, Mapping))


def _pf_equal_strata(
    fingerprint: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    features = fingerprint.get("official_curves")
    _require(isinstance(features, list), "fingerprint official curves are malformed")
    _require(len(features) == EXPECTED_CURVES, "fingerprint curve count mismatch")
    strata: dict[str, dict[str, dict[str, Any]]] = {}
    for figure in ("ALL", "17", "18", "19"):
        strata[figure] = {}
        for channel in ("ALL", "T", "L"):
            selected = [
                feature
                for feature in features
                if (figure == "ALL" or feature.get("figure") == figure)
                and (channel == "ALL" or feature.get("channel") == channel)
            ]
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for feature in selected:
                family_id = feature.get("physical_family_id")
                _require(
                    isinstance(family_id, str),
                    "curve feature lacks physical-family identity",
                )
                grouped[family_id].append(feature)
            family_records: list[dict[str, Any]] = []
            for family_id, members in sorted(grouped.items()):
                family_records.append(
                    {
                        "physical_family_id": family_id,
                        "official_curve_keys": sorted(
                            str(member["curve"]) for member in members
                        ),
                        "n_official_curves": len(members),
                        "mae_N": float(
                            np.mean([float(member["mae_N"]) for member in members])
                        ),
                        "rmse_N": float(
                            np.mean([float(member["rmse_N"]) for member in members])
                        ),
                        "bias_N": float(
                            np.mean([float(member["bias_N"]) for member in members])
                        ),
                        "pearson_r": float(
                            np.mean([float(member["pearson_r"]) for member in members])
                        ),
                        "spearman_r": float(
                            np.mean([float(member["spearman_r"]) for member in members])
                        ),
                        "centered_cosine": float(
                            np.mean(
                                [float(member["centered_cosine"]) for member in members]
                            )
                        ),
                        "segment_slope_sign_match_fraction": float(
                            np.mean(
                                [
                                    float(member["segment_slope_sign_match_fraction"])
                                    for member in members
                                ]
                            )
                        ),
                        "trend_capture_fraction": float(
                            np.mean(
                                [bool(member["captured_legacy"]) for member in members]
                            )
                        ),
                        "turn_topology_match_fraction": float(
                            np.mean(
                                [
                                    _turn_kinds(member["experimental_turns"])
                                    == _turn_kinds(member["model_turns"])
                                    for member in members
                                ]
                            )
                        ),
                    }
                )
            expected_pf, expected_curves = EXPECTED_PF_CELL_COUNTS[figure][channel]
            _require(
                len(family_records) == expected_pf and len(selected) == expected_curves,
                f"PF cell {figure}/{channel} count mismatch",
            )
            strata[figure][channel] = {
                "weighting": (
                    "official curves are averaged inside each physical family; "
                    "physical families then receive equal weight"
                ),
                "n_physical_families": len(family_records),
                "n_official_curves": len(selected),
                "mean_family_mae_N": float(
                    np.mean([record["mae_N"] for record in family_records])
                ),
                "mean_family_rmse_N": float(
                    np.mean([record["rmse_N"] for record in family_records])
                ),
                "mean_family_bias_N": float(
                    np.mean([record["bias_N"] for record in family_records])
                ),
                "mean_family_pearson_r": float(
                    np.mean([record["pearson_r"] for record in family_records])
                ),
                "mean_family_spearman_r": float(
                    np.mean([record["spearman_r"] for record in family_records])
                ),
                "mean_family_centered_cosine": float(
                    np.mean([record["centered_cosine"] for record in family_records])
                ),
                "mean_family_segment_slope_sign_match": float(
                    np.mean(
                        [
                            record["segment_slope_sign_match_fraction"]
                            for record in family_records
                        ]
                    )
                ),
                "mean_family_trend_capture": float(
                    np.mean(
                        [record["trend_capture_fraction"] for record in family_records]
                    )
                ),
                "mean_family_turn_topology_match": float(
                    np.mean(
                        [
                            record["turn_topology_match_fraction"]
                            for record in family_records
                        ]
                    )
                ),
                "families": family_records,
            }
    return strata


def _validate_fingerprint_scope(fingerprint: Mapping[str, Any]) -> None:
    _require(
        fingerprint.get("status") == "DESCRIPTIVE_FINGERPRINT_COMPLETE",
        "fresh residual fingerprint did not complete",
    )
    samples = fingerprint.get("samples")
    curves = fingerprint.get("official_curves")
    families = fingerprint.get("physical_curve_families")
    aliases = fingerprint.get("duplicate_aliases")
    _require(
        isinstance(samples, list) and len(samples) == EXPECTED_SAMPLES,
        "fingerprint sample count mismatch",
    )
    _require(
        isinstance(curves, list) and len(curves) == EXPECTED_CURVES,
        "fingerprint curve count mismatch",
    )
    _require(
        isinstance(families, list) and len(families) == EXPECTED_PHYSICAL_FAMILIES,
        "fingerprint physical-family count mismatch",
    )
    _require(
        isinstance(aliases, list) and len(aliases) == EXPECTED_ALIAS_GROUPS,
        "fingerprint alias count mismatch",
    )
    substantive_curve_keys = {
        str(item.get("curve"))
        for item in [*samples, *curves]
        if isinstance(item, Mapping)
    }
    _require(
        not any(key.startswith(("19|c|", "19|d|")) for key in substantive_curve_keys),
        "Fig19(c,d) leaked into fresh residual fingerprint",
    )
    _require(
        set(fingerprint.get("contract", {}).get("excluded_conditional_curves", ()))
        == EXPECTED_CONDITIONAL_CURVES,
        "fingerprint conditional exclusion contract mismatch",
    )


def _output_paths(out_dir: Path, run_id: str) -> dict[str, Path]:
    stem = f"v41_fresh_{run_id}"
    return {
        "scorecard": out_dir / f"scorecard_confirmed42_{stem}.json",
        "artifact": out_dir / f"residual_evidence_confirmed42_{stem}.json",
        "fingerprint": out_dir / f"residual_fingerprint_confirmed42_{stem}.json",
        "fig17": out_dir / f"fig17_confirmed_overlay_{stem}.png",
        "fig17_sidecar": out_dir / f"fig17_confirmed_overlay_{stem}.json",
        "fig18": out_dir / f"fig18_confirmed_overlay_{stem}.png",
        "fig18_sidecar": out_dir / f"fig18_confirmed_overlay_{stem}.json",
        "fig19": out_dir / f"fig19ab_confirmed_overlay_{stem}.png",
        "fig19_sidecar": out_dir / f"fig19ab_confirmed_overlay_{stem}.json",
        "receipt": out_dir / f"baseline_bundle_confirmed42_{stem}.json",
    }


def _ensure_outputs_absent(paths: Mapping[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    _require(not existing, f"refusing to overwrite baseline outputs: {existing}")


@contextmanager
def _publication_transaction(paths: Mapping[str, Path]) -> Iterator[None]:
    """Serialize publication and remove only this transaction's new outputs."""

    parents = {path.resolve().parent for path in paths.values()}
    _require(
        len(parents) == 1,
        "baseline publication outputs must share one directory",
    )
    out_dir = next(iter(parents))
    lock_path = out_dir.parent / f".{out_dir.name}.confirmed151-postprocess.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise BaselineContractError(
                "another confirmed151 postprocess publication holds the lock"
            ) from exc
        preexisting = {name: path.exists() for name, path in paths.items()}
        try:
            _ensure_outputs_absent(paths)
            yield
        except BaseException as exc:
            cleanup_failures: list[str] = []
            for name, path in paths.items():
                if preexisting[name] or not path.exists():
                    continue
                try:
                    path.unlink()
                except OSError as cleanup_exc:
                    cleanup_failures.append(f"{path}: {cleanup_exc}")
            if out_dir.exists():
                try:
                    _fsync_directory(out_dir)
                except OSError as cleanup_exc:
                    cleanup_failures.append(f"{out_dir}: {cleanup_exc}")
            if cleanup_failures:
                raise BaselineContractError(
                    "baseline publication failed and cleanup was incomplete: "
                    + "; ".join(cleanup_failures)
                ) from exc
            raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _save_figure_atomic(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".partial.png",
        dir=path.parent,
    )
    os.close(descriptor)
    partial = Path(partial_name)
    try:
        figure.savefig(
            partial,
            format="png",
            dpi=140,
            metadata={"Software": "FLUXV confirmed-scope scorer"},
        )
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    finally:
        if partial.exists():
            partial.unlink()


def _plot_figure(
    *,
    figure_id: str,
    rows: Sequence[Mapping[str, Any]],
    image_path: Path,
    sidecar_path: Path,
    bundle_id: str,
    scorecard_path: Path,
    curve_to_family: Mapping[str, str],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_contract = {
        "17": (("a", "b"), (1, 2), "Fig. 17 — confirmed scope"),
        "18": (("a", "b", "c", "d"), (2, 2), "Fig. 18 — confirmed scope"),
        "19": (
            ("a", "b"),
            (1, 2),
            "Fig. 19(a,b) — CONFIRMED42 ONLY; (c,d) excluded",
        ),
    }
    panels, shape, title = panel_contract[figure_id]
    selected = [row for row in rows if row.get("figure") == figure_id]
    expected_curve_count = EXPECTED_FIGURE_CURVES[figure_id]
    _require(
        len(selected) == expected_curve_count,
        f"Fig{figure_id} plot curve count mismatch",
    )
    _require(
        not any(
            str(row.get("curve", "")).startswith(("19|c|", "19|d|")) for row in selected
        ),
        f"Fig{figure_id} plot contains conditional curve",
    )

    figure, raw_axes = plt.subplots(
        shape[0],
        shape[1],
        figsize=(8.0 * shape[1], 5.2 * shape[0]),
        squeeze=False,
    )
    axes = list(raw_axes.ravel())
    panel_records: dict[str, Any] = {}
    for axis, panel in zip(axes, panels):
        panel_rows = sorted(
            (row for row in selected if row.get("panel") == panel),
            key=lambda row: str(row["curve"]),
        )
        for index, row in enumerate(panel_rows):
            color = f"C{index % 10}"
            x = np.asarray(row["measurement_x"], dtype=float)
            measured = np.asarray(row["measurement_N"], dtype=float)
            model = np.asarray(row["model_at_measurement_x_N"], dtype=float)
            label = str(row["curve"]).split("|", maxsplit=2)[-1]
            axis.plot(
                x,
                measured,
                "-x",
                color=color,
                linewidth=1.6,
                markersize=4.5,
                label=f"{label} experiment",
            )
            axis.plot(
                x,
                model,
                "--o",
                color=color,
                linewidth=1.4,
                markersize=3.8,
                markerfacecolor="white",
                label=f"{label} V4.1 fresh",
            )
        axis.set_title(f"({panel})")
        axis.set_xlabel(
            "flapping frequency (Hz)"
            if panel_rows and panel_rows[0]["abscissa"] == "frequency_Hz"
            else "nominal twist amplitude (deg)"
        )
        channel = panel_rows[0]["channel"] if panel_rows else ""
        axis.set_ylabel("thrust (N)" if channel == "T" else "lift (N)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6.5, ncol=2, loc="best")
        panel_records[panel] = {
            "channel": channel,
            "curve_count": len(panel_rows),
            "curve_keys": [str(row["curve"]) for row in panel_rows],
            "physical_family_ids": sorted(
                {curve_to_family[str(row["curve"])] for row in panel_rows}
            ),
        }
    for axis in axes[len(panels) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    _save_figure_atomic(figure, image_path)
    plt.close(figure)

    all_curve_keys = sorted(str(row["curve"]) for row in selected)
    sidecar = {
        "schema_version": 1,
        "artifact_type": "confirmed_scope_overlay_sidecar",
        "figure": figure_id,
        "evidence_scope": benchmark.EVIDENCE_CONFIRMED,
        "baseline_bundle_id": bundle_id,
        "scorecard": _file_identity(scorecard_path),
        "image": _file_identity(image_path),
        "curve_count": len(all_curve_keys),
        "physical_family_count": len({curve_to_family[key] for key in all_curve_keys}),
        "curve_keys": all_curve_keys,
        "panels": panel_records,
        "conditional_fig19_cd_rows": 0,
    }
    _write_json_atomic(sidecar_path, sidecar)
    return sidecar


def _bundle_id_payload(
    *,
    bundle: ValidatedBundle,
    data_path: Path,
    scoring_prereg_path: Path,
    authorization_path: Path,
    pf_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "result_sha256": _sha256_file(bundle.result_path),
        "manifest_sha256": _sha256_file(bundle.manifest_path),
        "contributions_sha256": _sha256_file(bundle.contributions_path),
        "measurement_data_sha256": _sha256_file(data_path),
        "benchmark_source_sha256": _sha256_file(Path(benchmark.__file__).resolve()),
        "fingerprint_source_sha256": _sha256_file(
            Path(residual_fingerprint.__file__).resolve()
        ),
        "postprocessor_source_sha256": _sha256_file(Path(__file__).resolve()),
        "physical_family_contract_sha256": pf_contract_sha256,
        "scoring_preregistration_sha256": _sha256_file(scoring_prereg_path),
        "postprocess_authorization_sha256": _sha256_file(authorization_path),
    }


def _publish_baseline(
    *,
    bundle: ValidatedBundle,
    scoring_prereg_path: Path,
    authorization_path: Path,
    out_dir: Path,
    data_path: Path,
) -> dict[str, Any]:
    """Publish an already validated bundle while holding the transaction lock."""

    scoring_prereg_path = scoring_prereg_path.resolve()
    authorization_path = authorization_path.resolve()
    data_path = data_path.resolve()
    measurements = benchmark.load_measurements(data_path)
    measurement_validation = benchmark.validate_measurement_contract(
        measurements,
        source_path=data_path,
    )
    _require(
        measurement_validation.get("passed") is True,
        "raw Fig17/18/19 measurement contract failed",
    )
    _require(
        measurement_validation.get("actual_curve_count") == 50
        and measurement_validation.get("actual_measurement_samples") == 530
        and measurement_validation.get("evidence_scope_sample_counts", {}).get(
            benchmark.EVIDENCE_CONFIRMED
        )
        == EXPECTED_SAMPLES,
        "measurement count contract mismatch",
    )

    curve_to_family, families, aliases, pf_contract_sha256 = _physical_family_contract()
    bundle_id_payload = _bundle_id_payload(
        bundle=bundle,
        data_path=data_path,
        scoring_prereg_path=scoring_prereg_path,
        authorization_path=authorization_path,
        pf_contract_sha256=pf_contract_sha256,
    )
    bundle_id = _canonical_hash(bundle_id_payload)
    run_id = str(bundle.manifest["run_id"])
    paths = _output_paths(out_dir.resolve(), run_id)

    raw_report = benchmark.scorecard(
        bundle.results,
        sweep_name=_display_path(bundle.result_path),
        measurements=measurements,
        measurement_path=data_path,
    )
    scorecard = _confirmed_scorecard(raw_report, bundle=bundle)
    scorecard["baseline_bundle_id"] = bundle_id
    scorecard["runner_bundle_validation"] = copy.deepcopy(bundle.validation)
    _write_json_atomic(paths["scorecard"], scorecard)

    artifact = _build_confirmed_artifact(
        scorecard=scorecard,
        bundle=bundle,
        scorecard_path=paths["scorecard"],
        data_path=data_path,
        scoring_prereg_path=scoring_prereg_path,
    )
    artifact["baseline_bundle_id"] = bundle_id
    _write_json_atomic(paths["artifact"], artifact)
    artifact_sha256 = _sha256_file(paths["artifact"])

    fingerprint = residual_fingerprint.build_fingerprint(
        artifact,
        input_path=paths["artifact"],
        expected_input_sha256=artifact_sha256,
    )
    _validate_fingerprint_scope(fingerprint)
    strata = _pf_equal_strata(fingerprint)
    _require(
        math.isclose(
            strata["ALL"]["ALL"]["mean_family_mae_N"],
            float(
                fingerprint["aggregates"]["physical_family_equal"]["mean_family_mae_N"]
            ),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "independent PF-equal aggregate disagrees with fingerprint",
    )
    fingerprint["baseline_bundle_id"] = bundle_id
    fingerprint["generator"] = _file_identity(Path(__file__).resolve())
    fingerprint["physical_family_contract_sha256"] = pf_contract_sha256
    fingerprint["aggregates"]["physical_family_equal_by_figure_channel"] = strata
    fingerprint["primary_metric"] = {
        "name": "physical_family_equal_mean_absolute_error",
        "unit": "N",
        "value": strata["ALL"]["ALL"]["mean_family_mae_N"],
        "physical_family_count": EXPECTED_PHYSICAL_FAMILIES,
        "alias_policy": "average aliases within family, then equal-weight families",
    }
    fingerprint["validity_gates"].update(
        {
            "fresh_triplet_complete_and_bound": True,
            "force_ledgers_close_within_1e-9_N": True,
            "pf_strata_count_contract": True,
            "fig19_cd_zero_residual_leakage": True,
        }
    )
    _write_json_atomic(paths["fingerprint"], fingerprint)

    sidecars = {}
    rows = scorecard["rows"]
    for figure_id in ("17", "18", "19"):
        sidecars[figure_id] = _plot_figure(
            figure_id=figure_id,
            rows=rows,
            image_path=paths[f"fig{figure_id}"],
            sidecar_path=paths[f"fig{figure_id}_sidecar"],
            bundle_id=bundle_id,
            scorecard_path=paths["scorecard"],
            curve_to_family=curve_to_family,
        )
        _require(
            sidecars[figure_id]["curve_count"] == EXPECTED_FIGURE_CURVES[figure_id]
            and sidecars[figure_id]["physical_family_count"]
            == EXPECTED_FIGURE_PF[figure_id],
            f"Fig{figure_id} overlay sidecar count mismatch",
        )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "v41_fresh_confirmed42_baseline_bundle_receipt",
        "status": "READY_FOR_CONFIRMED_BASELINE_DIAGNOSIS",
        "baseline_bundle_id": bundle_id,
        "bundle_id_payload": bundle_id_payload,
        "run_id": run_id,
        "evidence_scope": benchmark.EVIDENCE_CONFIRMED,
        "contract": {
            "official_curves": EXPECTED_CURVES,
            "raw_measurement_samples": EXPECTED_SAMPLES,
            "solver_conditions": EXPECTED_CONDITIONS,
            "physical_families": EXPECTED_PHYSICAL_FAMILIES,
            "duplicate_alias_groups": EXPECTED_ALIAS_GROUPS,
            "figure_curve_counts": EXPECTED_FIGURE_CURVES,
            "conditional_fig19_cd_curves_excluded": sorted(EXPECTED_CONDITIONAL_CURVES),
        },
        "input_artifacts": {
            "result": _file_identity(bundle.result_path),
            "manifest": _file_identity(bundle.manifest_path),
            "claim_contributions": _file_identity(bundle.contributions_path),
            "measurement_data": _file_identity(data_path),
            "scoring_preregistration": _file_identity(scoring_prereg_path),
            "postprocess_authorization": _file_identity(authorization_path),
        },
        "validation": {
            **copy.deepcopy(bundle.validation),
            "same_timestamp_complete_resume_revalidation": True,
            "complete_resume_triplet_unchanged": True,
            "postprocess_authorization_validated": True,
        },
        "physical_family_contract": {
            "sha256": pf_contract_sha256,
            "family_count": len(families),
            "alias_group_count": len(aliases),
        },
        "primary_metric": copy.deepcopy(fingerprint["primary_metric"]),
        "pf_equal_by_figure_channel": copy.deepcopy(strata),
        "outputs": {
            name: _file_identity(path)
            for name, path in paths.items()
            if name != "receipt"
        },
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
        "forbidden_use": ("final 50-curve promotion or any claim based on Fig19(c,d)"),
    }
    _write_json_atomic(paths["receipt"], receipt)
    return receipt


def build_baseline(
    *,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
    scoring_prereg_path: Path,
    authorization_path: Path,
    out_dir: Path,
    data_path: Path = benchmark.DEFAULT_DATA_MD,
    completion_validator: Callable[..., None] = _production_completion_validator,
    expected_authorization_path: Path = AUTHORIZED_POSTPROCESS_AUTHORIZATION,
    expected_authorization_sha256: str = (AUTHORIZED_POSTPROCESS_AUTHORIZATION_SHA256),
    expected_prereg_path: Path = AUTHORIZED_POSTPROCESS_PREREG,
    expected_prereg_sha256: str = AUTHORIZED_POSTPROCESS_PREREG_SHA256,
) -> dict[str, Any]:
    """Validate, zero-GPU revalidate, and atomically publish a fresh baseline."""

    result_path = result_path.resolve()
    manifest_path = manifest_path.resolve()
    contributions_path = contributions_path.resolve()
    scoring_prereg_path = scoring_prereg_path.resolve()
    authorization_path = authorization_path.resolve()
    data_path = data_path.resolve()
    preflight_manifest, run_id, receipts_before = _preflight_complete_manifest(
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
    )
    _validate_postprocess_authorization(
        authorization_path=authorization_path,
        expected_authorization_path=expected_authorization_path,
        expected_authorization_sha256=expected_authorization_sha256,
        scoring_prereg_path=scoring_prereg_path,
        expected_prereg_path=expected_prereg_path,
        expected_prereg_sha256=expected_prereg_sha256,
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
        data_path=data_path,
        manifest=preflight_manifest,
    )

    completion_validator(
        run_id=run_id,
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
    )
    receipts_after = {
        "result": _sha256_file(result_path),
        "manifest": _sha256_file(manifest_path),
        "contributions": _sha256_file(contributions_path),
    }
    _require(
        receipts_after == receipts_before,
        "complete-resume revalidation changed the immutable fresh151 triplet",
    )

    bundle = validate_fresh151_bundle(
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
    )
    _validate_postprocess_authorization(
        authorization_path=authorization_path,
        expected_authorization_path=expected_authorization_path,
        expected_authorization_sha256=expected_authorization_sha256,
        scoring_prereg_path=scoring_prereg_path,
        expected_prereg_path=expected_prereg_path,
        expected_prereg_sha256=expected_prereg_sha256,
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
        data_path=data_path,
        manifest=bundle.manifest,
    )
    paths = _output_paths(out_dir.resolve(), run_id)
    with _publication_transaction(paths):
        return _publish_baseline(
            bundle=bundle,
            scoring_prereg_path=scoring_prereg_path,
            authorization_path=authorization_path,
            out_dir=out_dir.resolve(),
            data_path=data_path,
        )


def _baseline_command(args: argparse.Namespace) -> int:
    receipt = build_baseline(
        result_path=args.result,
        manifest_path=args.manifest,
        contributions_path=args.contributions,
        scoring_prereg_path=args.prereg,
        authorization_path=args.authorization,
        out_dir=args.out_dir,
        data_path=args.data,
    )
    print(
        "confirmed baseline bundle ready: "
        f"{receipt['contract']['official_curves']} curves, "
        f"{receipt['contract']['raw_measurement_samples']} samples, "
        f"{receipt['contract']['solver_conditions']} conditions, "
        f"{receipt['contract']['physical_families']} physical families"
    )
    print(f"baseline_bundle_id={receipt['baseline_bundle_id']}")
    receipt_path = _output_paths(
        args.out_dir.resolve(),
        str(receipt["run_id"]),
    )["receipt"]
    print(f"saved {_display_path(receipt_path)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strict confirmed-scope Fig17/18/19 post-processing"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline_parser = subparsers.add_parser(
        "baseline",
        help="validate and score a completed fresh V4.1 confirmed151 bundle",
    )
    baseline_parser.add_argument("--result", type=Path, required=True)
    baseline_parser.add_argument("--manifest", type=Path, required=True)
    baseline_parser.add_argument("--contributions", type=Path, required=True)
    baseline_parser.add_argument("--prereg", type=Path, required=True)
    baseline_parser.add_argument("--authorization", type=Path, required=True)
    baseline_parser.add_argument("--out-dir", type=Path, required=True)
    baseline_parser.add_argument(
        "--data",
        type=Path,
        default=benchmark.DEFAULT_DATA_MD,
    )
    baseline_parser.set_defaults(handler=_baseline_command)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
