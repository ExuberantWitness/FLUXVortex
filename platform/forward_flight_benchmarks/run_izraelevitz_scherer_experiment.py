"""Run the Izraelevitz Figure-14 / Scherer 1968 experimental benchmark.

Figure 11 in Izraelevitz et al. (2017) is a numerical UVLM comparison.  The
paper's experimental comparison is Figure 14, which digitizes Scherer's
finite-wing water-tunnel measurements.  This runner keeps those evidence
roles separate and compares periodic mean thrust coefficient without fitting
phase, amplitude, or residuals to the observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform as system_platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .augmented_uvpm import blend_periodic_ullt_state_shape, run_augmented_fluxv
from .cases import IZRAELEVITZ_2017_FIG14_SCHERER as CASE
from .ptera_adapter import build_izraelevitz_scherer_movement
from .ullt_attached import movement_one_state_ullt, smooth_separation_fraction
from .uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    add_constant_profile_drag,
    movement_polar_residual,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_upgrade_20260812/source_data/"
    "izraelevitz2017_fig14_digitized.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_upgrade_20260812/runs/"
    "20260812_scherer_fig14_experiment_full"
)

MODEL_LABELS = {
    "scherer_1968_experiment": "Scherer 1968 experiment",
    "authors_6state_ullt": "Authors' six-state ULLT",
    "authors_1state_ullt": "Authors' one-state ULLT",
    "authors_qs_added_mass": "Authors' quasi-steady + added mass",
    "fluxv_uvpm": "FluxV old (UVLM + source Cd0)",
    "fluxv_periodic_v1": "FluxV v1 (UVLM/polar + source Cd0)",
    "fluxv_periodic_v2": "FluxV v2 (UVLM/ULLT/polar + source Cd0)",
    "one_state_ullt_local": "Local one-state ULLT + source Cd0",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _baseline_from_augmented(history: dict[str, Any]) -> dict[str, Any]:
    """Extract the untouched UVLM channel from an augmented run."""

    return {
        "phase": np.asarray(history["phase"], dtype=float),
        "lift_n": np.asarray(history["baseline_uvlm_lift_n"], dtype=float),
        "drag_n": np.asarray(history["baseline_uvlm_drag_n"], dtype=float),
        "thrust_n": np.asarray(history["baseline_uvlm_thrust_n"], dtype=float),
        "mean_lift_n": float(history["baseline_uvlm_mean_lift_n"]),
        "mean_drag_n": float(history["baseline_uvlm_mean_drag_n"]),
        "mean_thrust_n": float(history["baseline_uvlm_mean_thrust_n"]),
        "model_semantics": (
            "current FluxV load channel: Ptera prescribed-wake UVLM; VPM "
            "particles do not feed back into wing loads"
        ),
    }


def _condition_history(
    theta_max_deg: float,
    phase_offset_deg: float,
    quality: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    movement, movement_metadata = build_izraelevitz_scherer_movement(
        theta_max_deg, phase_offset_deg, quality
    )
    steps_per_cycle = int(round(CASE.period_s / movement.delta_time))
    cycles = int(movement_metadata["cycles"])
    cycle_range = ((cycles - 1) * steps_per_cycle, cycles * steps_per_cycle - 1)
    output_samples = steps_per_cycle

    polar = movement_polar_residual(
        movement,
        source_cycle_step_range=cycle_range,
        period_s=CASE.period_s,
        freestream_m_s=CASE.freestream_m_s,
        rho_kg_m3=CASE.rho_kg_m3,
        aspect_ratio=CASE.aspect_ratio,
        output_samples=output_samples,
    )
    v1_inviscid = run_augmented_fluxv(
        movement,
        period_s=CASE.period_s,
        rho_kg_m3=CASE.rho_kg_m3,
        freestream_m_s=CASE.freestream_m_s,
        area_m2=CASE.area_m2,
        aspect_ratio=CASE.aspect_ratio,
        polar_residual=polar,
        output_samples=output_samples,
    )
    old_inviscid = _baseline_from_augmented(v1_inviscid)
    ullt_inviscid = movement_one_state_ullt(
        movement,
        source_cycle_step_range=cycle_range,
        period_s=CASE.period_s,
        freestream_m_s=CASE.freestream_m_s,
        rho_kg_m3=CASE.rho_kg_m3,
        aspect_ratio=CASE.aspect_ratio,
        area_m2=CASE.area_m2,
        output_samples=output_samples,
    )
    local_separation = smooth_separation_fraction(
        np.asarray(polar["alpha_rad"], dtype=float),
        attached_limit_deg=DEFAULT_POLAR_PARAMETERS.attached_limit_deg,
        fully_separated_deg=DEFAULT_POLAR_PARAMETERS.fully_separated_deg,
    )
    global_separation = np.average(
        local_separation,
        axis=1,
        weights=np.asarray(ullt_inviscid["strip_area_m2"], dtype=float),
    )
    v2_inviscid = blend_periodic_ullt_state_shape(
        ullt_inviscid,
        v1_inviscid,
        global_separation,
        rho_kg_m3=CASE.rho_kg_m3,
        freestream_m_s=CASE.freestream_m_s,
        area_m2=CASE.area_m2,
    )

    common = {
        "kinematic_residual": polar,
        "rho_kg_m3": CASE.rho_kg_m3,
        "freestream_m_s": CASE.freestream_m_s,
        "area_m2": CASE.area_m2,
    }
    main_cd0 = CASE.profile_drag_coefficient
    histories = {
        "fluxv_uvpm": add_constant_profile_drag(
            old_inviscid, coefficient=main_cd0, **common
        ),
        "fluxv_periodic_v1": add_constant_profile_drag(
            v1_inviscid, coefficient=main_cd0, **common
        ),
        "fluxv_periodic_v2": add_constant_profile_drag(
            v2_inviscid, coefficient=main_cd0, **common
        ),
        "one_state_ullt_local": add_constant_profile_drag(
            ullt_inviscid, coefficient=main_cd0, **common
        ),
    }
    sensitivity: dict[str, dict[str, float]] = {}
    for coefficient in (
        0.0,
        CASE.scherer_static_profile_drag_coefficient,
        CASE.profile_drag_coefficient,
    ):
        key = f"{coefficient:.3f}"
        sensitivity[key] = {}
        for model, raw in (
            ("fluxv_uvpm", old_inviscid),
            ("fluxv_periodic_v1", v1_inviscid),
            ("fluxv_periodic_v2", v2_inviscid),
            ("one_state_ullt_local", ullt_inviscid),
        ):
            adjusted = add_constant_profile_drag(raw, coefficient=coefficient, **common)
            sensitivity[key][model] = float(adjusted["mean_CT"])

    audit = {
        "movement": movement_metadata,
        "cycle_step_range": list(cycle_range),
        "output_samples": output_samples,
        "max_abs_quarter_chord_alpha_deg": polar["max_abs_alpha_deg"],
        "mean_global_separation_fraction": float(np.mean(global_separation)),
        "profile_drag_sensitivity_CT": sensitivity,
        "v1_component_identity_max_abs_n": v1_inviscid["component_identity_max_abs_n"],
    }
    return histories, audit


def _prediction_metrics(
    observations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    model: str,
    theta: float | None,
) -> dict[str, Any]:
    model_rows = [row for row in predictions if row["series"] == model]
    obs_rows = observations
    if theta is not None:
        obs_rows = [row for row in obs_rows if row["theta_max_deg"] == theta]
    lookup = {
        (row["theta_max_deg"], row["phase_offset_deg"]): row["CT"] for row in model_rows
    }
    residual, normalized = [], []
    for row in obs_rows:
        key = (row["theta_max_deg"], row["phase_offset_deg"])
        if key not in lookup:
            continue
        error = float(lookup[key] - row["CT"])
        uncertainty = 0.5 * (float(row["CT_error_minus"]) + float(row["CT_error_plus"]))
        residual.append(error)
        normalized.append(error / uncertainty)
    if not residual:
        raise ValueError(f"model {model} has no matched experimental conditions")
    error = np.asarray(residual)
    observed_values = np.asarray([row["CT"] for row in obs_rows], dtype=float)
    data_range = float(np.ptp(observed_values))
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "theta_max_deg": "all" if theta is None else theta,
        "observation_count": len(error),
        "MAE_CT": float(np.mean(np.abs(error))),
        "RMSE_CT": float(np.sqrt(np.mean(error**2))),
        "bias_CT": float(np.mean(error)),
        "max_abs_error_CT": float(np.max(np.abs(error))),
        "range_NRMSE": float(np.sqrt(np.mean(error**2)) / data_range)
        if data_range > 0.0
        else "",
        "RMS_uncertainty_units": float(np.sqrt(np.mean(np.asarray(normalized) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("smoke", "full"), default="full")
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source_data.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_rows = _read_csv(source)
    observations: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows, start=1):
        parsed = {
            "series": row["series"],
            "model_label": MODEL_LABELS[row["series"]],
            "data_role": row["data_role"],
            "theta_max_deg": float(row["theta_max_deg"]),
            "phase_offset_deg": float(row["phase_offset_deg"]),
            "observation_id": (
                f"obs_{index:02d}"
                if row["data_role"] == "experimental_observation"
                else ""
            ),
            "replicate": int(row["replicate"]),
            "CT": float(row["ct"]),
            "CT_error_minus": float(row["ct_error_minus"])
            if row["ct_error_minus"]
            else "",
            "CT_error_plus": float(row["ct_error_plus"])
            if row["ct_error_plus"]
            else "",
            "mean_thrust_n": "",
            "mean_drag_n": "",
            "profile_drag_coefficient": CASE.profile_drag_coefficient
            if row["data_role"] == "numerical_reference"
            else "",
        }
        all_rows.append(parsed)
        if parsed["data_role"] == "experimental_observation":
            observations.append(parsed)
        else:
            prediction_rows.append(parsed)

    audit: dict[str, Any] = {}
    sensitivity_rows: list[dict[str, Any]] = []
    log_lines: list[str] = []
    experimental_conditions = sorted(
        {
            (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
            for row in observations
        }
    )
    for theta, phase_offset in experimental_conditions:
        started = time.perf_counter()
        histories, condition_audit = _condition_history(
            theta, phase_offset, args.quality
        )
        condition_key = f"theta_{theta:g}_psi_{phase_offset:g}"
        audit[condition_key] = condition_audit
        for model, history in histories.items():
            row = {
                "series": model,
                "model_label": MODEL_LABELS[model],
                "data_role": "local_model_prediction",
                "theta_max_deg": theta,
                "phase_offset_deg": phase_offset,
                "observation_id": "",
                "replicate": 1,
                "CT": float(history["mean_CT"]),
                "CT_error_minus": "",
                "CT_error_plus": "",
                "mean_thrust_n": float(history["mean_thrust_n"]),
                "mean_drag_n": float(history["mean_drag_n"]),
                "profile_drag_coefficient": CASE.profile_drag_coefficient,
            }
            all_rows.append(row)
            prediction_rows.append(row)
        for cd0, values in condition_audit["profile_drag_sensitivity_CT"].items():
            for model, ct in values.items():
                sensitivity_rows.append(
                    {
                        "theta_max_deg": theta,
                        "phase_offset_deg": phase_offset,
                        "profile_drag_coefficient": cd0,
                        "model": model,
                        "CT": ct,
                    }
                )
        elapsed = time.perf_counter() - started
        line = f"{condition_key}: complete in {elapsed:.3f} s"
        log_lines.append(line)
        print(line, flush=True)

    metrics: list[dict[str, Any]] = []
    prediction_models = sorted({row["series"] for row in prediction_rows})
    for model in prediction_models:
        for theta in (None, 15.0, 25.0):
            metrics.append(
                _prediction_metrics(observations, prediction_rows, model, theta)
            )

    result_path = output / "mean_thrust_vs_phase.csv"
    metric_path = output / "accuracy_metrics.csv"
    sensitivity_path = output / "profile_drag_sensitivity.csv"
    _write_csv(
        result_path,
        all_rows,
        [
            "series",
            "model_label",
            "data_role",
            "theta_max_deg",
            "phase_offset_deg",
            "observation_id",
            "replicate",
            "CT",
            "CT_error_minus",
            "CT_error_plus",
            "mean_thrust_n",
            "mean_drag_n",
            "profile_drag_coefficient",
        ],
    )
    _write_csv(
        metric_path,
        metrics,
        [
            "model",
            "model_label",
            "theta_max_deg",
            "observation_count",
            "MAE_CT",
            "RMSE_CT",
            "bias_CT",
            "max_abs_error_CT",
            "range_NRMSE",
            "RMS_uncertainty_units",
        ],
    )
    _write_csv(
        sensitivity_path,
        sensitivity_rows,
        [
            "theta_max_deg",
            "phase_offset_deg",
            "profile_drag_coefficient",
            "model",
            "CT",
        ],
    )
    (output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    source_files = [
        source,
        Path(__file__).resolve(),
        Path(__file__).with_name("cases.py"),
        Path(__file__).with_name("ptera_adapter.py"),
        Path(__file__).with_name("augmented_uvpm.py"),
        Path(__file__).with_name("ullt_attached.py"),
        Path(__file__).with_name("uvlm_polar_correction.py"),
    ]
    manifest = {
        "status": "complete",
        "evaluation_type": "real_gt_digitized_experimental_cycle_mean",
        "quality": args.quality,
        "case": CASE.manifest(),
        "evidence_roles": {
            "Figure_11": "numerical UVLM method audit; not experiment",
            "Figure_14": "Scherer 1968 experimental cycle-mean thrust benchmark",
        },
        "profile_drag_policy": {
            "main": CASE.profile_drag_coefficient,
            "main_source": "Izraelevitz 2017 Figure-14 implementation text",
            "fixed_sensitivity": CASE.scherer_static_profile_drag_coefficient,
            "sensitivity_source": "Scherer 1968 static foil tests",
            "selection_by_observation_error": False,
        },
        "condition_count": len(experimental_conditions),
        "experimental_observation_count": len(observations),
        "audit": audit,
        "environment": {
            "python": sys.version,
            "platform": system_platform.platform(),
            "numpy": np.__version__,
        },
        "source_sha256": {_repo_relative(path): _sha256(path) for path in source_files},
        "result_sha256": {
            _repo_relative(path): _sha256(path)
            for path in (result_path, metric_path, sensitivity_path, output / "run.log")
        },
        "limitations": [
            "Figure 14 publishes cycle-mean thrust only, not experimental phase loads.",
            "Experimental markers and error bars are digitized from vector artwork.",
            "Scherer reports slightly rounded tips; the local UVLM uses rectangular tips.",
            "The periodic v1/v2 upgrade remains exploratory/post-hoc and is not an online causal solver.",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "complete",
        "metrics_all_observations": {
            row["model"]: row for row in metrics if row["theta_max_deg"] == "all"
        },
        "result_files": {
            "predictions": _repo_relative(result_path),
            "metrics": _repo_relative(metric_path),
            "profile_drag_sensitivity": _repo_relative(sensitivity_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
