"""Score the preregistered G1/G2 gates for the N3-only P2 shadow.

This post-processor never calls the aerodynamic solver.  It consumes closed
``lb_sweep_candidate.py`` run directories and:

* compares q16/q24 and dt/dt2 members for G1;
* splits each representative32 record into the candidate and its *same-call*
  V4.1 counterfactual for G2;
* delegates raw Fig. 17/18/19 measurement interpolation and legacy trend
  capture to :mod:`fig171819_benchmark`;
* applies only the thresholds frozen in the candidate ``PLAN.md``.

Examples
--------
First score the independently terminal quadrature stage::

    python platform/score_n3_shadow_gates.py g1-quadrature \
      --q16 RUN_Q16 --q24 RUN_Q24

Only if that stage passes, complete the time-family gate (the q16 run is also
the coarse-dt member)::

    python platform/score_n3_shadow_gates.py g1 \
      --q16 RUN_Q16 --q24 RUN_Q24 --dt-half RUN_DT_HALF

Representative accuracy/trend gate::

    python platform/score_n3_shadow_gates.py g2 \
      --representative-run RUN_REPRESENTATIVE32
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_benchmark as benchmark  # noqa: E402
import lb_sweep_candidate as campaign_runner  # noqa: E402


CANDIDATE_ID = "n3_spatial_edge_pressure_v1_shadow"
CLOSURE = CANDIDATE_ID
NUMERICAL_FAMILY_CANDIDATE_IDS = frozenset(
    {
        CANDIDATE_ID,
        f"{CANDIDATE_ID}_q16",
        f"{CANDIDATE_ID}_q24",
        f"{CANDIDATE_ID}_dt2",
        f"{CANDIDATE_ID}_dt_half",
    }
)
Q16 = 16
Q24 = 24
G1_QUADRATURE_LIMIT_PERCENT = 0.5
G1_TIME_LIMIT_PERCENT = 5.0
G2_CHANNEL_MAE_LIMIT_RATIO = 1.05
RELATIVE_DENOMINATOR_FLOOR_N = 1.0e-12

SMOKE3 = tuple(
    (8.0, benchmark.FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ, twist, 5.0)
    for twist in (0.0, 22.5, 45.0)
)
REPRESENTATIVE32 = tuple(
    sorted(
        {
            *(
                (
                    8.0,
                    benchmark.FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ,
                    twist,
                    5.0,
                )
                for twist in benchmark.TWS
            ),
            *((U, freq, 22.5, 5.0) for U in (6.0, 10.0) for freq in benchmark.FS),
            *((8.0, freq, 22.5, aoa) for aoa in (0.0, 15.0) for freq in benchmark.FS),
        },
        key=lambda item: tuple(float(value) for value in item),
    )
)
REPRESENTATIVE_COMPLETE_CURVES = (
    "17|a|2.6",
    "17|b|2.6",
    "18|a|6.0",
    "18|b|6.0",
    "18|a|10.0",
    "18|b|10.0",
    "18|c|(8.0, 2.6)",
    "18|d|(8.0, 2.6)",
    "19|a|0",
    "19|b|0",
    "19|a|15",
    "19|b|15",
)
SLOPE_WITNESS_CURVES = (
    "17|a|2.6",
    "17|b|2.6",
    "18|a|6.0",
    "18|b|6.0",
    "18|a|10.0",
    "18|b|10.0",
)
if len(REPRESENTATIVE32) != 32:
    raise AssertionError(
        f"representative contract drift: {len(REPRESENTATIVE32)} != 32"
    )


@dataclasses.dataclass(frozen=True)
class RunBundle:
    """One closed candidate run and its immutable scoring provenance."""

    run_dir: Path
    config_path: Path
    results_path: Path
    status_path: Path
    config: Mapping[str, Any]
    results: Mapping[str, Any]
    status: Mapping[str, Any]
    artifact_sha256: Mapping[str, str]

    @property
    def identity(self) -> Mapping[str, Any]:
        value = self.config.get("run_identity")
        if not isinstance(value, Mapping):
            raise ValueError(f"{self.config_path}: missing run_identity")
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc


def _decode_json_bytes(path: Path, payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc


def _resolve_run(source: Path | str) -> RunBundle:
    path = Path(source).expanduser().resolve()
    run_dir = path.parent if path.is_file() else path
    config_path = run_dir / "config.json"
    results_path = (
        path
        if path.is_file() and path.name == "candidate_results.json"
        else run_dir / "candidate_results.json"
    )
    status_path = run_dir / "status.json"
    missing = [
        item
        for item in (config_path, results_path, status_path)
        if not item.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "candidate run is incomplete: " + ", ".join(str(item) for item in missing)
        )
    with campaign_runner._RunDirectoryLock(run_dir):
        payloads = {
            "config": config_path.read_bytes(),
            "results": results_path.read_bytes(),
            "status": status_path.read_bytes(),
        }
        artifact_sha256 = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        }
        config = _decode_json_bytes(config_path, payloads["config"])
        results = _decode_json_bytes(results_path, payloads["results"])
        status = _decode_json_bytes(status_path, payloads["status"])
    if not all(isinstance(value, Mapping) for value in (config, results, status)):
        raise ValueError(f"{run_dir}: config/results/status must be JSON objects")
    bundle = RunBundle(
        run_dir=run_dir,
        config_path=config_path,
        results_path=results_path,
        status_path=status_path,
        config=config,
        results=results,
        status=status,
        artifact_sha256=artifact_sha256,
    )
    _validate_bundle_role(bundle)
    return bundle


def _validate_bundle_role(bundle: RunBundle) -> None:
    identity = bundle.identity
    if identity.get("candidate_id") not in NUMERICAL_FAMILY_CANDIDATE_IDS:
        raise ValueError(
            f"{bundle.run_dir}: candidate_id is outside the frozen "
            f"numerical family {sorted(NUMERICAL_FAMILY_CANDIDATE_IDS)!r}"
        )
    if identity.get("closure") != CLOSURE:
        raise ValueError(f"{bundle.run_dir}: closure must be {CLOSURE!r}")
    if bundle.status.get("status") != "complete":
        raise ValueError(
            f"{bundle.run_dir}: status must be 'complete', got "
            f"{bundle.status.get('status')!r}"
        )


def _valid_force_pair(record: Any, *, context: str) -> tuple[float, float]:
    errors = campaign_runner._result_validation_errors(
        record,
        closure=CLOSURE,
    )
    if errors:
        raise ValueError(
            f"{context}: candidate record failed strict campaign schema: "
            + "; ".join(errors)
        )
    assert isinstance(record, Mapping)
    try:
        lift = float(record["L"])
        thrust = float(record["T"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: missing finite candidate L/T") from exc
    if not math.isfinite(lift) or not math.isfinite(thrust):
        raise ValueError(f"{context}: candidate L/T must be finite")
    return lift, thrust


def _same_call_counterfactual_pair(
    record: Mapping[str, Any], *, context: str
) -> tuple[float, float]:
    keys = (
        "L_wind_v41_counterfactual",
        "T_wind_v41_counterfactual",
    )
    present = tuple(key in record for key in keys)
    if not all(present):
        raise ValueError(
            f"{context}: complete same-call V4.1 counterfactual L/T is required"
        )
    try:
        values = tuple(float(record[key]) for key in keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid counterfactual L/T") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{context}: counterfactual L/T must be finite")
    return values  # type: ignore[return-value]


def _required_sweep(
    results: Mapping[str, Any],
    conditions: Sequence[tuple[float, float, float, float]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for condition in conditions:
        key = benchmark.condition_key(condition)
        lift, thrust = _valid_force_pair(
            results.get(key),
            context=key,
        )
        output[key] = {"L": lift, "T": thrust}
    return output


def _matched_representative_sweeps(
    results: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    candidate = _required_sweep(results, REPRESENTATIVE32)
    counterfactual: dict[str, dict[str, float]] = {}
    for key in candidate:
        record = results[key]
        assert isinstance(record, Mapping)
        lift, thrust = _same_call_counterfactual_pair(record, context=key)
        counterfactual[key] = {"L": lift, "T": thrust}
    return candidate, counterfactual


def _assert_scope(bundle: RunBundle, scope: str, conditions: Sequence[Any]) -> None:
    identity = bundle.identity
    if identity.get("scope") != scope:
        raise ValueError(
            f"{bundle.run_dir}: scope must be {scope!r}, got "
            f"{identity.get('scope')!r}"
        )
    expected_keys = [benchmark.condition_key(condition) for condition in conditions]
    if identity.get("condition_keys") != expected_keys:
        raise ValueError(f"{bundle.run_dir}: condition-key contract drift")
    if int(identity.get("condition_count", -1)) != len(expected_keys):
        raise ValueError(f"{bundle.run_dir}: condition-count contract drift")
    _required_sweep(bundle.results, conditions)


def _spatial_quadrature(identity: Mapping[str, Any]) -> int:
    model_args = identity.get("model_args")
    if not isinstance(model_args, Mapping):
        raise ValueError("run identity lacks model_args")
    raw = model_args.get("spatial_p2_quadrature", Q16)
    if not isinstance(raw, (int, float)) or int(raw) != raw:
        raise ValueError(f"invalid spatial_p2_quadrature {raw!r}")
    return int(raw)


def _identity_without(
    identity: Mapping[str, Any],
    *,
    drop_quadrature: bool,
    drop_time_grid: bool,
) -> dict[str, Any]:
    result = json.loads(json.dumps(identity))
    # candidate_id selects an isolated storage directory (base/q24/dt2); the
    # closure and source fingerprints below define the physical executable.
    result["candidate_id"] = CANDIDATE_ID
    if drop_quadrature:
        model_args = result.get("model_args", {})
        if isinstance(model_args, dict):
            model_args.pop("spatial_p2_quadrature", None)
        resolved = result.get("resolved_model_config_before_closure_profile", {})
        if isinstance(resolved, dict):
            resolved.pop("spatial_p2_quadrature", None)
    if drop_time_grid:
        grid = result.get("grid", {})
        if isinstance(grid, dict):
            grid.pop("steps_per_cycle", None)
            grid.pop("wake_rows", None)
    return result


def _force_refinement_report(
    coarse: Mapping[str, Mapping[str, float]],
    fine: Mapping[str, Mapping[str, float]],
    keys: Sequence[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for key in keys:
        if key not in coarse or key not in fine:
            raise ValueError(f"refinement comparison lacks condition {key}")
        for channel in ("L", "T"):
            coarse_value = float(coarse[key][channel])
            fine_value = float(fine[key][channel])
            if not math.isfinite(coarse_value) or not math.isfinite(fine_value):
                raise ValueError(f"{key}/{channel}: non-finite refinement force")
            absolute = abs(coarse_value - fine_value)
            relative = (
                absolute
                / max(abs(fine_value), RELATIVE_DENOMINATOR_FLOOR_N)
                * 100.0
            )
            rows.append(
                {
                    "condition_key": key,
                    "channel": channel,
                    "coarse_N": coarse_value,
                    "fine_N": fine_value,
                    "absolute_difference_N": absolute,
                    "relative_to_fine_percent": relative,
                }
            )
        coarse_lift = float(coarse[key]["L"])
        coarse_thrust = float(coarse[key]["T"])
        fine_lift = float(fine[key]["L"])
        fine_thrust = float(fine[key]["T"])
        difference_norm = math.hypot(
            coarse_lift - fine_lift,
            coarse_thrust - fine_thrust,
        )
        fine_norm = math.hypot(fine_lift, fine_thrust)
        vector_rows.append(
            {
                "condition_key": key,
                "coarse_wind_axis_force_N": {
                    "L": coarse_lift,
                    "T": coarse_thrust,
                },
                "fine_wind_axis_force_N": {
                    "L": fine_lift,
                    "T": fine_thrust,
                },
                "difference_norm_N": difference_norm,
                "fine_norm_N": fine_norm,
                "relative_to_fine_percent": (
                    difference_norm
                    / max(fine_norm, RELATIVE_DENOMINATOR_FLOOR_N)
                    * 100.0
                ),
            }
        )
    maximum = max(rows, key=lambda row: row["relative_to_fine_percent"])
    maximum_vector = max(
        vector_rows,
        key=lambda row: row["relative_to_fine_percent"],
    )
    return {
        "normalization": (
            "|coarse-fine| / max(|fine|, 1e-12 N); fine is q24 or dt/2"
        ),
        "rows": rows,
        "max_relative_percent": maximum["relative_to_fine_percent"],
        "max_relative_witness": {
            "condition_key": maximum["condition_key"],
            "channel": maximum["channel"],
        },
        "wind_axis_vector_norm": {
            "normalization": (
                "||(L,T)_coarse-(L,T)_fine||_2 / "
                "max(||(L,T)_fine||_2, 1e-12 N)"
            ),
            "rows": vector_rows,
            "max_relative_percent": maximum_vector[
                "relative_to_fine_percent"
            ],
            "max_relative_witness": {
                "condition_key": maximum_vector["condition_key"],
            },
        },
    }


def _quadrature_stage(
    q16: RunBundle,
    q24: RunBundle,
) -> tuple[dict[str, Any], bool]:
    """Validate and score the first, independently terminal G1 stage."""

    for bundle in (q16, q24):
        _assert_scope(bundle, "smoke3", SMOKE3)
    if q16.identity.get("candidate_id") not in {
        CANDIDATE_ID,
        f"{CANDIDATE_ID}_q16",
    }:
        raise ValueError("q16 input uses an incompatible candidate storage role")
    if q24.identity.get("candidate_id") not in {
        CANDIDATE_ID,
        f"{CANDIDATE_ID}_q24",
    }:
        raise ValueError("q24 input uses an incompatible candidate storage role")
    if _spatial_quadrature(q16.identity) != Q16:
        raise ValueError("q16 input does not resolve spatial_p2_quadrature=16")
    if _spatial_quadrature(q24.identity) != Q24:
        raise ValueError("q24 input does not resolve spatial_p2_quadrature=24")

    q16_identity = _identity_without(
        q16.identity, drop_quadrature=True, drop_time_grid=False
    )
    q24_identity = _identity_without(
        q24.identity, drop_quadrature=True, drop_time_grid=False
    )
    if q16_identity != q24_identity:
        raise ValueError("q16/q24 identities differ beyond quadrature order")

    q16_sweep = _required_sweep(q16.results, SMOKE3)
    q24_sweep = _required_sweep(q24.results, SMOKE3)
    smoke_keys = [benchmark.condition_key(condition) for condition in SMOKE3]
    quadrature = _force_refinement_report(q16_sweep, q24_sweep, smoke_keys)
    passed = bool(
        quadrature["max_relative_percent"]
        <= G1_QUADRATURE_LIMIT_PERCENT
    )
    return quadrature, passed


def score_g1_quadrature(
    q16: RunBundle,
    q24: RunBundle,
) -> dict[str, Any]:
    """Score q16->q24 before authorizing any dt/2 computation."""

    quadrature, passed = _quadrature_stage(q16, q24)
    terminal_nogo = not passed
    return {
        "schema_version": 1,
        "artifact_type": (
            "n3_spatial_edge_pressure_v1_shadow_G1_quadrature"
        ),
        "candidate_id": CANDIDATE_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "quadrature_q16_to_q24",
        "gates": {
            "quadrature_q16_to_q24_max_relative_le_0p5_percent": passed,
        },
        "passed": passed,
        "g1_complete": terminal_nogo,
        "terminal_nogo": terminal_nogo,
        "decision": (
            "GO_TO_G1_TIME"
            if passed
            else "TERMINAL_NO_GO"
        ),
        "quadrature_family": quadrature,
        "time_family": {
            "status": (
                "authorized_but_not_run"
                if passed
                else "not_run_by_preregistered_early_stop"
            ),
            "reason": (
                None
                if passed
                else (
                    "q16->q24 exceeded 0.5%; PLAN.md G1 forbids "
                    "continuing to dt/2 or G2"
                )
            ),
        },
        "inputs": {
            "q16": _bundle_identity(q16),
            "q24": _bundle_identity(q24),
            "dt_half": None,
        },
    }


def score_g1(
    q16: RunBundle,
    q24: RunBundle,
    dt_half: RunBundle | None = None,
) -> dict[str, Any]:
    """Apply G1 serially, stopping before dt/2 when quadrature fails."""

    quadrature, quadrature_passed = _quadrature_stage(q16, q24)
    if not quadrature_passed:
        report = score_g1_quadrature(q16, q24)
        report["artifact_type"] = "n3_spatial_edge_pressure_v1_shadow_G1"
        return report
    if dt_half is None:
        raise ValueError(
            "q16->q24 passed; --dt-half is now required to complete G1"
        )
    _assert_scope(dt_half, "smoke3", SMOKE3)
    if dt_half.identity.get("candidate_id") not in {
        CANDIDATE_ID,
        f"{CANDIDATE_ID}_dt2",
        f"{CANDIDATE_ID}_dt_half",
    }:
        raise ValueError("dt-half input uses an incompatible candidate storage role")
    if _spatial_quadrature(dt_half.identity) != Q16:
        raise ValueError("dt-half input must retain the q16 candidate")

    coarse_no_time = _identity_without(
        q16.identity, drop_quadrature=False, drop_time_grid=True
    )
    fine_no_time = _identity_without(
        dt_half.identity, drop_quadrature=False, drop_time_grid=True
    )
    if coarse_no_time != fine_no_time:
        raise ValueError("dt/dt2 identities differ beyond the time grid")
    coarse_grid = q16.identity["grid"]
    fine_grid = dt_half.identity["grid"]
    if not isinstance(coarse_grid, Mapping) or not isinstance(fine_grid, Mapping):
        raise ValueError("run identity lacks a grid mapping")
    coarse_steps = int(coarse_grid["steps_per_cycle"])
    fine_steps = int(fine_grid["steps_per_cycle"])
    coarse_wake = int(coarse_grid["wake_rows"])
    fine_wake = int(fine_grid["wake_rows"])
    if fine_steps != 2 * coarse_steps or fine_wake != 2 * coarse_wake:
        raise ValueError(
            "dt-half run must double steps_per_cycle and wake_rows exactly"
        )

    q16_sweep = _required_sweep(q16.results, SMOKE3)
    dt_half_sweep = _required_sweep(dt_half.results, SMOKE3)
    high_twist_key = benchmark.condition_key(SMOKE3[-1])
    time_family = _force_refinement_report(
        q16_sweep,
        dt_half_sweep,
        (high_twist_key,),
    )
    gates = {
        "quadrature_q16_to_q24_max_relative_le_0p5_percent": (
            quadrature["max_relative_percent"]
            <= G1_QUADRATURE_LIMIT_PERCENT
        ),
        "dt_to_dt2_tw45_L_relative_le_5_percent": next(
            row["relative_to_fine_percent"]
            for row in time_family["rows"]
            if row["channel"] == "L"
        )
        <= G1_TIME_LIMIT_PERCENT,
        "dt_to_dt2_tw45_T_relative_le_5_percent": next(
            row["relative_to_fine_percent"]
            for row in time_family["rows"]
            if row["channel"] == "T"
        )
        <= G1_TIME_LIMIT_PERCENT,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "artifact_type": "n3_spatial_edge_pressure_v1_shadow_G1",
        "candidate_id": CANDIDATE_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "complete_quadrature_and_time_family",
        "gates": gates,
        "passed": passed,
        "g1_complete": True,
        "terminal_nogo": not passed,
        "decision": "GO_TO_G2" if passed else "TERMINAL_NO_GO",
        "quadrature_family": quadrature,
        "time_family": time_family,
        "inputs": {
            "q16": _bundle_identity(q16),
            "q24": _bundle_identity(q24),
            "dt_half": _bundle_identity(dt_half),
        },
    }


def _complete_rows(scorecard: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    scope = scorecard["evidence_scopes"][benchmark.EVIDENCE_CONFIRMED]
    rows = scope["rows"]
    result = {
        str(row["curve"]): row
        for row in rows
        if isinstance(row, Mapping) and row.get("complete")
    }
    actual = tuple(sorted(result))
    expected = tuple(sorted(REPRESENTATIVE_COMPLETE_CURVES))
    if actual != expected:
        raise ValueError(
            "representative32 complete-curve drift: "
            f"actual={actual}, expected={expected}"
        )
    return result


def _row_aggregate(
    rows: Mapping[str, Mapping[str, Any]], channel: str | None
) -> dict[str, Any]:
    selected = [
        row for row in rows.values() if channel is None or row["channel"] == channel
    ]
    errors = np.concatenate(
        [np.asarray(row["error_N"], dtype=float) for row in selected]
    )
    if errors.size == 0 or not np.all(np.isfinite(errors)):
        raise ValueError("representative aggregate has no finite errors")
    return {
        "curve_count": len(selected),
        "measurement_point_count": int(errors.size),
        "mae_N": float(np.mean(np.abs(errors))),
        "rmse_N": float(np.sqrt(np.mean(errors * errors))),
        "bias_N": float(np.mean(errors)),
        "trend_capture_count": sum(bool(row["captured"]) for row in selected),
        "trend_capture_fraction": float(
            np.mean([bool(row["captured"]) for row in selected])
        ),
    }


def _linear_slope(x: Any, y: Any) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    centered = x_array - float(np.mean(x_array))
    denominator = float(np.dot(centered, centered))
    if (
        x_array.ndim != 1
        or y_array.shape != x_array.shape
        or x_array.size < 2
        or not np.all(np.isfinite(x_array))
        or not np.all(np.isfinite(y_array))
        or denominator <= 1.0e-15
    ):
        raise ValueError("invalid slope witness arrays")
    return float(
        np.dot(centered, y_array - float(np.mean(y_array))) / denominator
    )


def _slope_witness_report(
    candidate_rows: Mapping[str, Mapping[str, Any]],
    counterfactual_rows: Mapping[str, Mapping[str, Any]],
    *,
    witness_keys: Sequence[str] = SLOPE_WITNESS_CURVES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    new_reversals: list[str] = []
    for key in witness_keys:
        candidate = candidate_rows[key]
        counterfactual = counterfactual_rows[key]
        measured_slope = _linear_slope(
            candidate["measurement_x"], candidate["measurement_N"]
        )
        candidate_slope = _linear_slope(
            candidate["measurement_x"],
            candidate["model_at_measurement_x_N"],
        )
        counterfactual_slope = _linear_slope(
            counterfactual["measurement_x"],
            counterfactual["model_at_measurement_x_N"],
        )
        measured_sign = int(np.sign(measured_slope))
        candidate_sign = int(np.sign(candidate_slope))
        counterfactual_sign = int(np.sign(counterfactual_slope))
        candidate_reversed = candidate_sign != measured_sign
        counterfactual_reversed = counterfactual_sign != measured_sign
        is_new = candidate_reversed and not counterfactual_reversed
        if is_new:
            new_reversals.append(key)
        rows.append(
            {
                "curve": key,
                "measured_ols_slope": measured_slope,
                "candidate_ols_slope": candidate_slope,
                "counterfactual_ols_slope": counterfactual_slope,
                "measured_sign": measured_sign,
                "candidate_sign": candidate_sign,
                "counterfactual_sign": counterfactual_sign,
                "candidate_reversed_vs_measurement": candidate_reversed,
                "counterfactual_reversed_vs_measurement": counterfactual_reversed,
                "new_candidate_reversal": is_new,
            }
        )
    return {
        "definition": (
            "new reversal = candidate OLS slope sign disagrees with raw "
            "measurement while the same-call V4.1 counterfactual agrees"
        ),
        "witness_curve_count": len(rows),
        "new_reversal_count": len(new_reversals),
        "new_reversal_curves": new_reversals,
        "rows": rows,
    }


def score_g2(
    representative: RunBundle,
    *,
    measurement_path: Path | str = benchmark.DEFAULT_DATA_MD,
) -> dict[str, Any]:
    """Apply the frozen representative32 accuracy and trend gates."""

    _assert_scope(representative, "representative32", REPRESENTATIVE32)
    candidate_sweep, counterfactual_sweep = _matched_representative_sweeps(
        representative.results
    )
    candidate_score = benchmark.scorecard(
        candidate_sweep,
        sweep_name=f"{CANDIDATE_ID}:representative32",
        measurement_path=measurement_path,
    )
    counterfactual_score = benchmark.scorecard(
        counterfactual_sweep,
        sweep_name="same-call V4.1 counterfactual:representative32",
        measurement_path=measurement_path,
    )
    candidate_rows = _complete_rows(candidate_score)
    counterfactual_rows = _complete_rows(counterfactual_score)
    candidate_metrics = {
        name: _row_aggregate(candidate_rows, channel)
        for name, channel in (("ALL", None), ("L", "L"), ("T", "T"))
    }
    counterfactual_metrics = {
        name: _row_aggregate(counterfactual_rows, channel)
        for name, channel in (("ALL", None), ("L", "L"), ("T", "T"))
    }
    slopes = _slope_witness_report(candidate_rows, counterfactual_rows)
    gates = {
        "overall_point_weighted_MAE_strictly_lower": (
            candidate_metrics["ALL"]["mae_N"]
            < counterfactual_metrics["ALL"]["mae_N"]
        ),
        "lift_MAE_no_worse_than_plus_5_percent": (
            candidate_metrics["L"]["mae_N"]
            <= G2_CHANNEL_MAE_LIMIT_RATIO
            * counterfactual_metrics["L"]["mae_N"]
        ),
        "thrust_MAE_no_worse_than_plus_5_percent": (
            candidate_metrics["T"]["mae_N"]
            <= G2_CHANNEL_MAE_LIMIT_RATIO
            * counterfactual_metrics["T"]["mae_N"]
        ),
        "total_trend_capture_count_no_lower": (
            candidate_metrics["ALL"]["trend_capture_count"]
            >= counterfactual_metrics["ALL"]["trend_capture_count"]
        ),
        "no_new_preregistered_slope_sign_reversal": (
            slopes["new_reversal_count"] == 0
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "artifact_type": "n3_spatial_edge_pressure_v1_shadow_G2",
        "candidate_id": CANDIDATE_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence_scope": benchmark.EVIDENCE_CONFIRMED,
        "gates": gates,
        "passed": passed,
        "decision": "GO_TO_CONFIRMED151" if passed else "NO_GO_STOP",
        "aggregation_contract": (
            "Select exactly the 12 confirmed curves completed by the frozen "
            "representative32 conditions; concatenate each row's raw-"
            "measurement error_N for point-weighted MAE; count legacy "
            "captured=true curves without curve-size weighting."
        ),
        "complete_curve_keys": list(REPRESENTATIVE_COMPLETE_CURVES),
        "candidate": candidate_metrics,
        "same_call_v41_counterfactual": counterfactual_metrics,
        "candidate_minus_counterfactual": {
            channel: {
                "mae_N": (
                    candidate_metrics[channel]["mae_N"]
                    - counterfactual_metrics[channel]["mae_N"]
                ),
                "mae_relative_percent": (
                    (
                        candidate_metrics[channel]["mae_N"]
                        - counterfactual_metrics[channel]["mae_N"]
                    )
                    / max(
                        counterfactual_metrics[channel]["mae_N"],
                        RELATIVE_DENOMINATOR_FLOOR_N,
                    )
                    * 100.0
                ),
                "trend_capture_count": (
                    candidate_metrics[channel]["trend_capture_count"]
                    - counterfactual_metrics[channel]["trend_capture_count"]
                ),
            }
            for channel in ("ALL", "L", "T")
        },
        "slope_witnesses": slopes,
        "input": _bundle_identity(representative),
        "measurement_source": {
            "path": str(Path(measurement_path).expanduser().resolve()),
            "sha256": _sha256(Path(measurement_path).expanduser().resolve()),
        },
        "scorer_sources": {
            "this_file": _sha256(Path(__file__).resolve()),
            "benchmark": _sha256(Path(benchmark.__file__).resolve()),
        },
    }


def _bundle_identity(bundle: RunBundle) -> dict[str, Any]:
    return {
        "run_directory": str(bundle.run_dir),
        "config_path": str(bundle.config_path),
        "config_sha256": bundle.artifact_sha256["config"],
        "results_path": str(bundle.results_path),
        "results_sha256": bundle.artifact_sha256["results"],
        "status_path": str(bundle.status_path),
        "status_sha256": bundle.artifact_sha256["status"],
        "run_identity": bundle.identity,
    }


def _non_overwriting_path(directory: Path, stem: str) -> Path:
    path = directory / f"{stem}.json"
    index = 1
    while path.exists():
        path = directory / f"{stem}_{index:02d}.json"
        index += 1
    return path


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="gate", required=True)
    g1_quadrature = subparsers.add_parser(
        "g1-quadrature",
        help="score q16/q24 and stop before dt/2 on failure",
    )
    g1_quadrature.add_argument("--q16", required=True)
    g1_quadrature.add_argument("--q24", required=True)
    g1_quadrature.add_argument("--output")
    g1 = subparsers.add_parser(
        "g1",
        help=(
            "score serial quadrature/time refinement; --dt-half is required "
            "only after q16/q24 passes"
        ),
    )
    g1.add_argument("--q16", required=True)
    g1.add_argument("--q24", required=True)
    g1.add_argument("--dt-half")
    g1.add_argument("--output")
    g2 = subparsers.add_parser("g2", help="score representative32 vs same-call V4.1")
    g2.add_argument("--representative-run", required=True)
    g2.add_argument("--data-md", default=str(benchmark.DEFAULT_DATA_MD))
    g2.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.gate in {"g1", "g1-quadrature"}:
            q16 = _resolve_run(args.q16)
            q24 = _resolve_run(args.q24)
            report = (
                score_g1_quadrature(q16, q24)
                if args.gate == "g1-quadrature"
                else score_g1(
                    q16,
                    q24,
                    (
                        _resolve_run(args.dt_half)
                        if args.dt_half is not None
                        else None
                    ),
                )
            )
            output = (
                Path(args.output).expanduser().resolve()
                if args.output
                else _non_overwriting_path(
                    q16.run_dir,
                    (
                        "G1_quadrature_score"
                        if args.gate == "g1-quadrature"
                        else "G1_score"
                    ),
                )
            )
        else:
            representative = _resolve_run(args.representative_run)
            report = score_g2(
                representative,
                measurement_path=args.data_md,
            )
            output = (
                Path(args.output).expanduser().resolve()
                if args.output
                else _non_overwriting_path(representative.run_dir, "G2_score")
            )
        _write_json_exclusive(output, report)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(output)
    print(report["decision"])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
