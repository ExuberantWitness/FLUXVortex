"""Deterministic residual fingerprint for the confirmed Fig. 17/18/19 scope.

This module is deliberately descriptive.  It cannot select an aerodynamic
claim without a separate node-force attribution artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fig171819_benchmark import (
    CURVES_BY_EVIDENCE_SCOPE,
    EVIDENCE_CONFIRMED,
    RAW_FS,
    ROOT,
    TWS,
    condition_key,
)


EXPECTED_CURVES = 42
EXPECTED_SAMPLES = 434
EXPECTED_CONDITIONS = 151
EXPECTED_PHYSICAL_FAMILIES = 34
EXPECTED_DUPLICATE_ALIAS_GROUPS = 8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _provenance_hashes_match(artifact: Mapping[str, Any]) -> bool:
    required = {
        "sweep_result",
        "runner_manifest",
        "original_runner_scorecard",
        "measurement_data",
        "scorer_source",
    }
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != required:
        return False
    for record in provenance.values():
        if not isinstance(record, Mapping):
            return False
        raw_path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            return False
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or _sha256_file(path) != digest:
            return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if (
        left.size < 2
        or float(np.std(left)) <= 1.0e-12
        or float(np.std(right)) <= 1.0e-12
    ):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left - float(np.mean(left))
    b = right - float(np.mean(right))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1.0e-15 else float(np.dot(a, b) / denominator)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if denominator <= 1.0e-15:
        return 0.0
    return float(np.dot(centered, y - float(np.mean(y))) / denominator)


def _turns(values: np.ndarray) -> list[dict[str, Any]]:
    delta = np.diff(values)
    sign = np.sign(delta)
    turns: list[dict[str, Any]] = []
    previous_index: int | None = None
    for index, value in enumerate(sign):
        if value == 0.0:
            continue
        if previous_index is not None and sign[previous_index] != value:
            turns.append(
                {
                    "index": index,
                    "kind": (
                        "local_max"
                        if sign[previous_index] > 0.0 and value < 0.0
                        else "local_min"
                    ),
                }
            )
        previous_index = index
    return turns


def _condition_bracket(
    solver_x: np.ndarray,
    conditions: Sequence[tuple[float, float, float, float]],
    evaluation_x: float,
) -> dict[str, Any]:
    exact = np.flatnonzero(
        np.isclose(solver_x, evaluation_x, rtol=0.0, atol=1.0e-12)
    )
    if exact.size:
        index = int(exact[0])
        condition = conditions[index]
        return {
            "left_condition_key": condition_key(condition),
            "right_condition_key": condition_key(condition),
            "left_weight": 1.0,
            "right_weight": 0.0,
        }
    right = int(np.searchsorted(solver_x, evaluation_x, side="right"))
    if right <= 0 or right >= solver_x.size:
        raise ValueError(
            f"evaluation x {evaluation_x} is outside solver grid {solver_x.tolist()}"
        )
    left = right - 1
    right_weight = float(
        (evaluation_x - solver_x[left])
        / (solver_x[right] - solver_x[left])
    )
    return {
        "left_condition_key": condition_key(conditions[left]),
        "right_condition_key": condition_key(conditions[right]),
        "left_weight": 1.0 - right_weight,
        "right_weight": right_weight,
    }


def _physical_family_contract() -> tuple[
    dict[str, str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    signature_groups: dict[
        tuple[str, tuple[str, ...]], list[str]
    ] = defaultdict(list)
    for curve in CURVES_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]:
        signature = (
            curve.channel,
            tuple(condition_key(condition) for condition in curve.conditions),
        )
        signature_groups[signature].append(curve.key)

    curve_to_family: dict[str, str] = {}
    families: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for index, (signature, keys) in enumerate(
        sorted(signature_groups.items(), key=lambda item: item[1][0]),
        start=1,
    ):
        family_id = f"PF{index:03d}"
        for key in keys:
            curve_to_family[key] = family_id
        record = {
            "physical_family_id": family_id,
            "channel": signature[0],
            "condition_keys": list(signature[1]),
            "official_curve_keys": sorted(keys),
            "n_official_curves": len(keys),
        }
        families.append(record)
        if len(keys) > 1:
            aliases.append(record)
    if len(families) != EXPECTED_PHYSICAL_FAMILIES:
        raise ValueError(
            f"physical family drift: {len(families)} != "
            f"{EXPECTED_PHYSICAL_FAMILIES}"
        )
    if len(aliases) != EXPECTED_DUPLICATE_ALIAS_GROUPS:
        raise ValueError(
            f"duplicate alias drift: {len(aliases)} != "
            f"{EXPECTED_DUPLICATE_ALIAS_GROUPS}"
        )
    return curve_to_family, families, aliases


def _curve_feature(
    row: Mapping[str, Any],
    *,
    physical_family_id: str,
    total_absolute_error: float,
) -> dict[str, Any]:
    x = np.asarray(row["measurement_x"], dtype=float)
    measured = np.asarray(row["measurement_N"], dtype=float)
    model = np.asarray(row["model_at_measurement_x_N"], dtype=float)
    error = np.asarray(row["error_N"], dtype=float)
    absolute_error = np.abs(error)
    span = float(x[-1] - x[0])
    exp_max = int(np.argmax(measured))
    model_max = int(np.argmax(model))
    exp_min = int(np.argmin(measured))
    model_min = int(np.argmin(model))
    exp_delta = np.diff(measured)
    model_delta = np.diff(model)
    sign_match = np.sign(exp_delta) == np.sign(model_delta)
    return {
        "curve": row["curve"],
        "physical_family_id": physical_family_id,
        "figure": row["figure"],
        "panel": row["panel"],
        "channel": row["channel"],
        "abscissa": row["abscissa"],
        "n_points": int(error.size),
        "mae_N": float(np.mean(absolute_error)),
        "rmse_N": float(np.sqrt(np.mean(error * error))),
        "bias_N": float(np.mean(error)),
        "median_absolute_error_N": float(np.median(absolute_error)),
        "q90_absolute_error_N": float(np.quantile(absolute_error, 0.90)),
        "max_absolute_error_N": float(np.max(absolute_error)),
        "sum_absolute_error_N": float(np.sum(absolute_error)),
        "sum_squared_error_N2": float(np.sum(error * error)),
        "absolute_error_share": (
            0.0
            if total_absolute_error <= 0.0
            else float(np.sum(absolute_error) / total_absolute_error)
        ),
        "trapz_signed_error_N": (
            0.0 if span <= 0.0 else float(np.trapezoid(error, x=x) / span)
        ),
        "trapz_absolute_error_N": (
            0.0
            if span <= 0.0
            else float(np.trapezoid(absolute_error, x=x) / span)
        ),
        "experimental_range_N": float(np.ptp(measured)),
        "model_range_N": float(np.ptp(model)),
        "experimental_std_N": float(np.std(measured)),
        "model_std_N": float(np.std(model)),
        "pearson_r": _correlation(model, measured),
        "spearman_r": _correlation(_rankdata(model), _rankdata(measured)),
        "centered_cosine": _centered_cosine(model, measured),
        "experimental_ols_slope": _linear_slope(x, measured),
        "model_ols_slope": _linear_slope(x, model),
        "experimental_endpoint_delta_N": float(measured[-1] - measured[0]),
        "model_endpoint_delta_N": float(model[-1] - model[0]),
        "segment_slope_sign_match_fraction": float(np.mean(sign_match)),
        "experimental_turns": _turns(measured),
        "model_turns": _turns(model),
        "experimental_argmax": {
            "index": exp_max,
            "x": float(x[exp_max]),
            "value_N": float(measured[exp_max]),
            "interior": 0 < exp_max < measured.size - 1,
            "pre_rise_N": float(measured[exp_max] - measured[0]),
            "post_peak_drop_N": float(measured[exp_max] - measured[-1]),
        },
        "model_argmax": {
            "index": model_max,
            "x": float(x[model_max]),
            "value_N": float(model[model_max]),
            "interior": 0 < model_max < model.size - 1,
            "pre_rise_N": float(model[model_max] - model[0]),
            "post_peak_drop_N": float(model[model_max] - model[-1]),
        },
        "experimental_argmin": {
            "index": exp_min,
            "x": float(x[exp_min]),
            "value_N": float(measured[exp_min]),
        },
        "model_argmin": {
            "index": model_min,
            "x": float(x[model_min]),
            "value_N": float(model[model_min]),
        },
        "captured_legacy": bool(row["captured"]),
    }


def _group_summary(
    features: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for feature in features:
        groups[tuple(feature[field] for field in fields)].append(feature)
    result: list[dict[str, Any]] = []
    for values, records in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        result.append(
            {
                **dict(zip(fields, values)),
                "n_curves": len(records),
                "mean_curve_mae_N": float(
                    np.mean([record["mae_N"] for record in records])
                ),
                "mean_curve_bias_N": float(
                    np.mean([record["bias_N"] for record in records])
                ),
                "mean_slope_sign_match": float(
                    np.mean(
                        [
                            record["segment_slope_sign_match_fraction"]
                            for record in records
                        ]
                    )
                ),
                "legacy_capture_fraction": float(
                    np.mean(
                        [record["captured_legacy"] for record in records]
                    )
                ),
            }
        )
    return result


def _physical_family_group_summary(
    features: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize strata without giving cross-figure aliases extra weight.

    A physical family can have more than one official curve.  Within each
    requested stratum those aliases are averaged first; the resulting family
    records are then averaged with one vote per physical family.
    """

    groups: dict[
        tuple[Any, ...],
        dict[str, list[Mapping[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for feature in features:
        group = tuple(feature[field] for field in fields)
        groups[group][str(feature["physical_family_id"])].append(feature)

    result: list[dict[str, Any]] = []
    for values, family_members in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        family_records = []
        for family_id, members in sorted(family_members.items()):
            family_records.append(
                {
                    "physical_family_id": family_id,
                    "official_curve_keys": sorted(
                        str(member["curve"]) for member in members
                    ),
                    "n_official_curves": len(members),
                    "mean_official_curve_mae_N": float(
                        np.mean([member["mae_N"] for member in members])
                    ),
                    "mean_official_curve_bias_N": float(
                        np.mean([member["bias_N"] for member in members])
                    ),
                    "mean_slope_sign_match": float(
                        np.mean(
                            [
                                member["segment_slope_sign_match_fraction"]
                                for member in members
                            ]
                        )
                    ),
                }
            )
        result.append(
            {
                **dict(zip(fields, values)),
                "n_physical_families": len(family_records),
                "n_official_curves": sum(
                    record["n_official_curves"]
                    for record in family_records
                ),
                "mean_family_mae_N": float(
                    np.mean(
                        [
                            record["mean_official_curve_mae_N"]
                            for record in family_records
                        ]
                    )
                ),
                "mean_family_bias_N": float(
                    np.mean(
                        [
                            record["mean_official_curve_bias_N"]
                            for record in family_records
                        ]
                    )
                ),
                "mean_family_slope_sign_match": float(
                    np.mean(
                        [
                            record["mean_slope_sign_match"]
                            for record in family_records
                        ]
                    )
                ),
                "physical_families": family_records,
            }
        )
    return result


def _duplicate_consistency(
    aliases: Sequence[Mapping[str, Any]],
    rows_by_key: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for alias in aliases:
        keys = alias["official_curve_keys"]
        reference = rows_by_key[keys[0]]
        ref_measurement = np.asarray(reference["measurement_N"], dtype=float)
        comparisons: list[dict[str, Any]] = []
        for key in keys[1:]:
            candidate = rows_by_key[key]
            measured = np.asarray(candidate["measurement_N"], dtype=float)
            if measured.shape != ref_measurement.shape:
                raise ValueError(f"duplicate family {keys}: sample shape mismatch")
            comparisons.append(
                {
                    "left_curve": keys[0],
                    "right_curve": key,
                    "mean_absolute_measurement_difference_N": float(
                        np.mean(np.abs(ref_measurement - measured))
                    ),
                    "max_absolute_measurement_difference_N": float(
                        np.max(np.abs(ref_measurement - measured))
                    ),
                    "endpoint_delta_sign_agrees": bool(
                        np.sign(ref_measurement[-1] - ref_measurement[0])
                        == np.sign(measured[-1] - measured[0])
                    ),
                }
            )
        output.append(
            {
                "physical_family_id": alias["physical_family_id"],
                "official_curve_keys": keys,
                "comparisons": comparisons,
            }
        )
    return output


def _witness_plan(
    features: Sequence[Mapping[str, Any]],
    rows_by_key: Mapping[str, Mapping[str, Any]],
    samples_by_curve: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    selected: dict[str, set[str]] = defaultdict(set)
    cell_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for feature in features:
        cell_groups[(feature["figure"], feature["channel"])].append(feature)
    for records in cell_groups.values():
        worst = max(records, key=lambda item: item["absolute_error_share"])
        selected[worst["curve"]].add("largest_error_share_in_figure_channel")
    for feature in features:
        if not feature["captured_legacy"]:
            selected[feature["curve"]].add("legacy_shape_not_captured")
        if np.sign(feature["experimental_ols_slope"]) != np.sign(
            feature["model_ols_slope"]
        ):
            selected[feature["curve"]].add("ols_slope_sign_mismatch")
        exp_turns = [turn["kind"] for turn in feature["experimental_turns"]]
        model_turns = [turn["kind"] for turn in feature["model_turns"]]
        if exp_turns != model_turns:
            selected[feature["curve"]].add("turn_topology_mismatch")

    feature_by_key = {feature["curve"]: feature for feature in features}
    witness_curves: list[dict[str, Any]] = []
    solver_conditions: set[str] = set()
    for curve_key in sorted(selected):
        feature = feature_by_key[curve_key]
        row = rows_by_key[curve_key]
        error = np.abs(np.asarray(row["error_N"], dtype=float))
        indices = {
            0,
            len(error) - 1,
            int(np.argmax(error)),
            int(feature["experimental_argmax"]["index"]),
            int(feature["experimental_argmin"]["index"]),
            int(feature["model_argmax"]["index"]),
            int(feature["model_argmin"]["index"]),
        }
        point_records: list[dict[str, Any]] = []
        curve_samples = samples_by_curve[curve_key]
        for index in sorted(indices):
            sample = curve_samples[index]
            point_records.append(
                {
                    "measurement_index": index,
                    "raw_x": sample["raw_x"],
                    "error_N": sample["error_N"],
                    "left_condition_key": sample["left_condition_key"],
                    "right_condition_key": sample["right_condition_key"],
                }
            )
            solver_conditions.add(sample["left_condition_key"])
            solver_conditions.add(sample["right_condition_key"])
        witness_curves.append(
            {
                "curve": curve_key,
                "physical_family_id": feature["physical_family_id"],
                "reasons": sorted(selected[curve_key]),
                "points": point_records,
            }
        )
    return {
        "selection_contract": (
            "worst curve per figure/channel plus every legacy-capture, "
            "OLS-slope-sign, or turn-topology failure; points are endpoints, "
            "extrema, and max-|error|"
        ),
        "curves": witness_curves,
        "unique_solver_condition_keys": sorted(
            solver_conditions,
            key=lambda key: tuple(float(value) for value in key.split("_")),
        ),
    }


def build_fingerprint(
    artifact: Mapping[str, Any],
    *,
    input_path: Path,
    expected_input_sha256: str | None = None,
) -> dict[str, Any]:
    input_sha256 = _sha256_file(input_path)
    gates: dict[str, bool] = {
        "input_hash_matches_preregistration": (
            expected_input_sha256 is None
            or input_sha256 == expected_input_sha256
        ),
        "artifact_ready": artifact.get("status")
        == "ready_for_baseline_diagnosis",
        "confirmed_scope": artifact.get("evidence_scope")
        == EVIDENCE_CONFIRMED,
        "curve_count": len(artifact.get("curve_rows", ())) == EXPECTED_CURVES,
        "sample_count": artifact.get("residual_point_count")
        == EXPECTED_SAMPLES,
        "condition_count": artifact.get("coverage", {}).get(
            "valid_unique_conditions"
        )
        == EXPECTED_CONDITIONS,
        "solver_scope_complete": bool(
            artifact.get("coverage", {}).get("complete")
        ),
        "global_promotion_remains_blocked": not bool(
            artifact.get("global_promotion_eligible")
        ),
        "provenance_hashes_match_current_files": (
            _provenance_hashes_match(artifact)
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError(f"residual fingerprint validity gates failed: {failed}")

    curve_specs = {
        curve.key: curve
        for curve in CURVES_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]
    }
    rows = artifact["curve_rows"]
    rows_by_key = {row["curve"]: row for row in rows}
    if set(rows_by_key) != set(curve_specs):
        raise ValueError("confirmed curve rows differ from the frozen 42-curve contract")
    if any(
        row["curve"].startswith(("19|c|", "19|d|")) for row in rows
    ):
        raise ValueError("conditional Fig19(c,d) curve leaked into confirmed rows")
    if any(
        row.get("interpolation", {}).get("measurement_values_interpolated")
        is not False
        for row in rows
    ):
        raise ValueError("measurement force interpolation is forbidden")

    curve_to_family, families, aliases = _physical_family_contract()
    total_absolute_error = float(
        sum(sum(abs(float(value)) for value in row["error_N"]) for row in rows)
    )
    features = [
        _curve_feature(
            row,
            physical_family_id=curve_to_family[row["curve"]],
            total_absolute_error=total_absolute_error,
        )
        for row in rows
    ]
    feature_by_key = {feature["curve"]: feature for feature in features}

    samples: list[dict[str, Any]] = []
    samples_by_curve: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        curve = curve_specs[row["curve"]]
        solver_x = np.asarray(curve.x, dtype=float)
        nominal_measurement_x = (
            RAW_FS if curve.abscissa == "frequency_Hz" else TWS
        )
        evaluation_x = row["interpolation"]["evaluation_x"]
        if len(nominal_measurement_x) != len(row["measurement_x"]):
            raise ValueError(f"{curve.key}: nominal/raw measurement length drift")
        for index, (
            nominal_x,
            raw_x,
            eval_x,
            measurement,
            model,
            error,
        ) in enumerate(
            zip(
                nominal_measurement_x,
                row["measurement_x"],
                evaluation_x,
                row["measurement_N"],
                row["model_at_measurement_x_N"],
                row["error_N"],
            )
        ):
            bracket = _condition_bracket(
                solver_x,
                curve.conditions,
                float(eval_x),
            )
            sample = {
                "curve": curve.key,
                "physical_family_id": curve_to_family[curve.key],
                "figure": curve.figure,
                "panel": curve.panel,
                "channel": curve.channel,
                "abscissa": curve.abscissa,
                "measurement_index": index,
                "canonical_nominal_x": float(nominal_x),
                "raw_x": float(raw_x),
                "evaluation_x": float(eval_x),
                "measurement_N": float(measurement),
                "model_N": float(model),
                "error_N": float(error),
                "absolute_error_N": abs(float(error)),
                "squared_error_N2": float(error) ** 2,
                **bracket,
            }
            samples.append(sample)
            samples_by_curve[curve.key].append(sample)
    if len(samples) != EXPECTED_SAMPLES:
        raise ValueError(
            f"expanded sample count {len(samples)} != {EXPECTED_SAMPLES}"
        )

    family_metrics: list[dict[str, Any]] = []
    for family in families:
        member_features = [
            feature_by_key[key] for key in family["official_curve_keys"]
        ]
        family_metrics.append(
            {
                **family,
                "mean_official_curve_mae_N": float(
                    np.mean([item["mae_N"] for item in member_features])
                ),
                "mean_official_curve_bias_N": float(
                    np.mean([item["bias_N"] for item in member_features])
                ),
                "mean_slope_sign_match": float(
                    np.mean(
                        [
                            item["segment_slope_sign_match_fraction"]
                            for item in member_features
                        ]
                    )
                ),
            }
        )

    curve_ranking = sorted(
        (
            {
                "curve": feature["curve"],
                "physical_family_id": feature["physical_family_id"],
                "channel": feature["channel"],
                "mae_N": feature["mae_N"],
                "bias_N": feature["bias_N"],
                "absolute_error_share": feature["absolute_error_share"],
                "captured_legacy": feature["captured_legacy"],
            }
            for feature in features
        ),
        key=lambda item: (-item["absolute_error_share"], item["curve"]),
    )
    family_ranking = sorted(
        family_metrics,
        key=lambda item: (
            -item["mean_official_curve_mae_N"],
            item["physical_family_id"],
        ),
    )

    return {
        "schema_version": 1,
        "status": "DESCRIPTIVE_FINGERPRINT_COMPLETE",
        "input": {
            "path": _display_path(input_path),
            "sha256": input_sha256,
        },
        "validity_gates": {
            **gates,
            "physical_family_count": len(families)
            == EXPECTED_PHYSICAL_FAMILIES,
            "duplicate_alias_group_count": len(aliases)
            == EXPECTED_DUPLICATE_ALIAS_GROUPS,
            "conditional_curve_excluded": True,
            "measurement_force_not_interpolated": True,
        },
        "contract": {
            "confirmed_curves": EXPECTED_CURVES,
            "raw_measurement_samples": EXPECTED_SAMPLES,
            "solver_conditions": EXPECTED_CONDITIONS,
            "physical_curve_families": EXPECTED_PHYSICAL_FAMILIES,
            "duplicate_alias_groups": EXPECTED_DUPLICATE_ALIAS_GROUPS,
            "excluded_conditional_curves": list(
                artifact["excluded_curve_keys"]
            ),
        },
        "aggregates": {
            "point_weighted": artifact["aggregates"],
            "curve_equal": {
                "n_curves": len(features),
                "mean_mae_N": float(
                    np.mean([feature["mae_N"] for feature in features])
                ),
                "mean_bias_N": float(
                    np.mean([feature["bias_N"] for feature in features])
                ),
            },
            "physical_family_equal": {
                "n_families": len(family_metrics),
                "mean_family_mae_N": float(
                    np.mean(
                        [
                            family["mean_official_curve_mae_N"]
                            for family in family_metrics
                        ]
                    )
                ),
                "mean_family_bias_N": float(
                    np.mean(
                        [
                            family["mean_official_curve_bias_N"]
                            for family in family_metrics
                        ]
                    )
                ),
            },
        },
        "samples": samples,
        "official_curves": features,
        "physical_curve_families": family_metrics,
        "duplicate_aliases": _duplicate_consistency(aliases, rows_by_key),
        "strata": {
            # Preserve the schema-v1 curve-equal keys for historical readers.
            "figure_channel": _group_summary(
                features, ("figure", "channel")
            ),
            "abscissa_channel": _group_summary(
                features, ("abscissa", "channel")
            ),
            "official_curve_equal_figure_channel": _group_summary(
                features, ("figure", "channel")
            ),
            "official_curve_equal_abscissa_channel": _group_summary(
                features, ("abscissa", "channel")
            ),
            "physical_family_equal_figure_channel": (
                _physical_family_group_summary(
                    features, ("figure", "channel")
                )
            ),
            "physical_family_equal_abscissa_channel": (
                _physical_family_group_summary(
                    features, ("abscissa", "channel")
                )
            ),
        },
        "residual_rankings": {
            "official_curves_by_absolute_error_share": curve_ranking,
            "physical_families_by_mean_curve_mae": family_ranking,
        },
        "witness_plan": _witness_plan(
            features,
            rows_by_key,
            samples_by_curve,
        ),
        "claim_attribution": {
            "decision": "NO_DECISION",
            "reason": "NODE_ATTRIBUTION_REQUIRED",
            "required_next_evidence": (
                "For each preregistered witness contrast, record wind-axis "
                "L/T contributions from N1/N2/N3/N4/N6, graph numerical "
                "reduction, ledger guards, and claim graph identity."
            ),
        },
        "movable_space": {
            "eligible_parents_after_unique_attribution": ["N2", "N3"],
            "integrity_only": ["N1", "N4"],
            "observer_only": ["N5"],
            "forbidden": ["N6", "falsified_children", "dead_end_children"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)

    artifact = _load_json(args.artifact)
    fingerprint = build_fingerprint(
        artifact,
        input_path=args.artifact,
        expected_input_sha256=args.expected_input_sha256,
    )
    fingerprint["generator"] = {
        "path": _display_path(Path(__file__).resolve()),
        "sha256": _sha256_file(Path(__file__).resolve()),
    }
    _write_json_atomic(args.output, fingerprint)
    print(
        "confirmed42 residual fingerprint: "
        f"{len(fingerprint['official_curves'])} curves, "
        f"{len(fingerprint['samples'])} samples, "
        f"{len(fingerprint['physical_curve_families'])} physical families"
    )
    print(
        "claim decision: "
        f"{fingerprint['claim_attribution']['decision']} "
        f"({fingerprint['claim_attribution']['reason']})"
    )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
