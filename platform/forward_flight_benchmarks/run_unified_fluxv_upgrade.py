"""Run the exploratory shared FluxV augmentation on two reconstructed papers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .augmented_uvpm import run_augmented_fluxv
from .cases import IZRAELEVITZ_2017_FIG11 as IZ_CASE
from .cases import YANG_2025 as YANG_CASE
from .ptera_adapter import (
    build_izraelevitz_fig11_movement,
    build_yang2025_movement,
)
from .uvlm_polar_correction import movement_polar_residual


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812"
)
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260812_periodic_v1_full"


def _manifest_path(path: Path) -> str:
    """Return a portable repository-relative provenance key when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)
YANG_PRIOR = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/plev2025/runs/20260808_multimodel_full"
)
IZ_PRIOR = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/runs/20260807_rigid_firstpass/izraelevitz_fig11_exact"
)
IZ_REFERENCE = DOC_ROOT / "source_data/izraelevitz2017_fig11_digitized.csv"
GF_TO_N = 0.00980665

MODEL_LABELS = {
    "wind_tunnel_test": "Wind-tunnel test",
    "authors_proposed_modified_uvlm": "Authors' modified UVLM",
    "fluxv_uvpm": "FluxV old (UVLM load channel)",
    "fluxv_periodic_v1": "FluxV augmented v1 (exploratory)",
    "fluxv_periodic_v2": "FluxV augmented v2 (ULLT-state, exploratory)",
    "one_state_ullt_local": "Local one-state ULLT reconstruction",
    "ptera_prescribed_wake_uvlm": "Prescribed-wake UVLM control",
    "ptera_free_wake_uvlm": "Ptera free-wake UVLM",
    "robofalcon2_coefficient_transfer": "RoboFalcon2 coefficient transfer",
    "paper_uvlm": "Authors' UVLM reference",
    "6_state": "6-State ULLT",
    "1_state": "1-State ULLT",
    "qs_added_mass": "QS + added mass",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _cycle_range(movement: Any, period_s: float) -> tuple[int, int]:
    count = len(movement.airplanes[0])
    last = count - 1
    phase_cycles = last * movement.delta_time / period_s
    end = last if np.isclose(phase_cycles, round(phase_cycles), atol=1.0e-10) else count
    steps = int(round(period_s / movement.delta_time))
    return end - steps, end - 1


def _curve_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    error = prediction - reference
    reference_range = float(np.ptp(reference))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
        "range_nrmse": float(np.sqrt(np.mean(error**2)) / reference_range)
        if reference_range > 0.0
        else float("nan"),
        "prediction_half_amplitude": float(np.ptp(prediction) / 2.0),
        "reference_half_amplitude": float(np.ptp(reference) / 2.0),
        "half_amplitude_error": float(
            np.ptp(prediction) / 2.0 - np.ptp(reference) / 2.0
        ),
    }


def _cyclic_phase_delta(prediction: int, reference: int, count: int) -> float:
    raw = (prediction - reference) / count
    return float((raw + 0.5) % 1.0 - 0.5)


def _phase_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    metrics = _curve_metrics(prediction, reference)
    count = len(reference)
    metrics["positive_peak_phase_error_cycle"] = _cyclic_phase_delta(
        int(np.argmax(prediction)), int(np.argmax(reference)), count
    )
    metrics["negative_peak_phase_error_cycle"] = _cyclic_phase_delta(
        int(np.argmin(prediction)), int(np.argmin(reference)), count
    )
    return metrics


def _run_yang(output: Path, log: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prior_means = _read_csv(YANG_PRIOR / "mean_characteristics.csv")
    prior_phase = _read_csv(YANG_PRIOR / "phase_histories.csv")
    means: list[dict[str, Any]] = [dict(row) for row in prior_means]
    phases: list[dict[str, Any]] = [dict(row) for row in prior_phase]
    audit: dict[str, Any] = {"cells": {}, "baseline_reproduction": {}}

    for aoa in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        started = time.perf_counter()
        movement, setup = build_yang2025_movement(aoa, "full")
        cycle_range = _cycle_range(movement, YANG_CASE.period_s)
        residual = movement_polar_residual(
            movement,
            source_cycle_step_range=cycle_range,
            period_s=YANG_CASE.period_s,
            freestream_m_s=YANG_CASE.freestream_m_s,
            rho_kg_m3=YANG_CASE.rho_kg_m3,
            aspect_ratio=YANG_CASE.aspect_ratio,
            output_samples=128,
        )
        history = run_augmented_fluxv(
            movement,
            period_s=YANG_CASE.period_s,
            rho_kg_m3=YANG_CASE.rho_kg_m3,
            freestream_m_s=YANG_CASE.freestream_m_s,
            area_m2=YANG_CASE.area_m2,
            aspect_ratio=YANG_CASE.aspect_ratio,
            polar_residual=residual,
            output_samples=128,
        )
        runtime = time.perf_counter() - started
        old_rows = [
            row
            for row in prior_phase
            if row["model"] == "fluxv_uvpm"
            and np.isclose(float(row["aoa_deg"]), aoa)
        ]
        old_lift = np.asarray([float(row["lift_n"]) for row in old_rows])
        old_drag = np.asarray([float(row["drag_n"]) for row in old_rows])
        baseline_difference = float(
            max(
                np.max(np.abs(old_lift - history["baseline_uvlm_lift_n"])),
                np.max(np.abs(old_drag - history["baseline_uvlm_drag_n"])),
            )
        )
        audit["baseline_reproduction"][str(int(aoa))] = baseline_difference
        audit["cells"][str(int(aoa))] = {
            "setup": setup,
            "cycle_range": list(cycle_range),
            "runtime_s": runtime,
            "max_abs_local_alpha_deg": residual["max_abs_alpha_deg"],
            "component_identity_max_abs_n": history[
                "component_identity_max_abs_n"
            ],
            "particle_count": history["particle_count"],
        }
        means.append(
            {
                "aoa_deg": aoa,
                "model": "fluxv_periodic_v1",
                "model_label": MODEL_LABELS["fluxv_periodic_v1"],
                "status": "exploratory_local",
                "mean_lift_n": history["mean_lift_n"],
                "mean_drag_n": history["mean_drag_n"],
                "mean_thrust_n": history["mean_thrust_n"],
                "mean_lift_gf": history["mean_lift_n"] / GF_TO_N,
                "mean_drag_gf": history["mean_drag_n"] / GF_TO_N,
                "mean_thrust_gf": history["mean_thrust_n"] / GF_TO_N,
                "mean_CL": history["mean_CL"],
                "mean_CD": history["mean_CD"],
                "mean_CT": history["mean_CT"],
                "runtime_s": runtime,
                "digitization_uncertainty_gf": "",
                "data_role": "exploratory_local_model",
                "error": "",
            }
        )
        for index, phase in enumerate(history["phase"]):
            phases.append(
                {
                    "aoa_deg": aoa,
                    "model": "fluxv_periodic_v1",
                    "model_label": MODEL_LABELS["fluxv_periodic_v1"],
                    "phase": phase,
                    "lift_n": history["lift_n"][index],
                    "drag_n": history["drag_n"][index],
                    "thrust_n": history["thrust_n"][index],
                    "CL": history["CL"][index],
                    "CD": history["CD"][index],
                    "CT": history["CT"][index],
                }
            )
        log.append(
            f"Yang AoA={aoa:g}: L={history['mean_lift_n']/GF_TO_N:.4f} gf, "
            f"D={history['mean_drag_n']/GF_TO_N:.4f} gf, runtime={runtime:.2f}s"
        )
    return means, phases, audit


def _run_izraelevitz(log: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    movement, setup = build_izraelevitz_fig11_movement("full")
    cycle_range = _cycle_range(movement, IZ_CASE.period_s)
    residual = movement_polar_residual(
        movement,
        source_cycle_step_range=cycle_range,
        period_s=IZ_CASE.period_s,
        freestream_m_s=IZ_CASE.freestream_m_s,
        rho_kg_m3=IZ_CASE.rho_kg_m3,
        aspect_ratio=IZ_CASE.aspect_ratio,
        output_samples=128,
    )
    started = time.perf_counter()
    improved = run_augmented_fluxv(
        movement,
        period_s=IZ_CASE.period_s,
        rho_kg_m3=IZ_CASE.rho_kg_m3,
        freestream_m_s=IZ_CASE.freestream_m_s,
        area_m2=IZ_CASE.area_m2,
        aspect_ratio=IZ_CASE.aspect_ratio,
        polar_residual=residual,
        output_samples=128,
    )
    runtime = time.perf_counter() - started
    scale = np.sin(np.deg2rad(IZ_CASE.downstroke_midpoint_alpha_deg))
    rows: list[dict[str, Any]] = []
    reference = _read_csv(IZ_REFERENCE)
    reference_models = {
        "paper_uvlm": ("paper_uvlm_CLalpha", "paper_uvlm_CDalpha", "author_reference"),
        "6_state": ("6_state_CLalpha", "6_state_CDalpha", "author_model"),
        "1_state": ("1_state_CLalpha", "1_state_CDalpha", "author_model"),
        "qs_added_mass": (
            "qs_added_mass_CLalpha",
            "qs_added_mass_CDalpha",
            "author_model",
        ),
    }
    for model, (lift_key, drag_key, role) in reference_models.items():
        for source in reference:
            cl_alpha = float(source[lift_key])
            cd_alpha = float(source[drag_key])
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "data_role": role,
                    "phase": source["phase"],
                    "CL_alpha": cl_alpha,
                    "CD_alpha": cd_alpha,
                    "CL": cl_alpha * scale,
                    "CD": cd_alpha * scale,
                    "CT": -cd_alpha * scale,
                }
            )

    prior = _read_csv(IZ_PRIOR / "phase_histories.csv")
    for source in prior:
        if source["model"] == "ptera_prescribed_wake_uvlm":
            continue
        model = source["model"]
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "data_role": "local_model",
                "phase": source["phase"],
                "CL_alpha": source["CL_alpha"],
                "CD_alpha": source["CD_alpha"],
                "CL": source["CL"],
                "CD": -float(source["CT"]),
                "CT": source["CT"],
            }
        )
    for index, phase in enumerate(improved["phase"]):
        rows.append(
            {
                "model": "fluxv_periodic_v1",
                "model_label": MODEL_LABELS["fluxv_periodic_v1"],
                "data_role": "exploratory_local_model",
                "phase": phase,
                "CL_alpha": improved["CL"][index] / scale,
                "CD_alpha": improved["CD"][index] / scale,
                "CL": improved["CL"][index],
                "CD": improved["CD"][index],
                "CT": improved["CT"][index],
            }
        )
    old = [row for row in prior if row["model"] == "fluxv_uvpm"]
    baseline_difference = float(
        max(
            np.max(
                np.abs(
                    np.asarray([float(row["CL"]) for row in old])
                    - improved["baseline_uvlm_CL"]
                )
            ),
            np.max(
                np.abs(
                    np.asarray([float(row["CT"]) for row in old])
                    - improved["baseline_uvlm_CT"]
                )
            ),
        )
    )
    audit = {
        "setup": setup,
        "cycle_range": list(cycle_range),
        "runtime_s": runtime,
        "max_abs_local_alpha_deg": residual["max_abs_alpha_deg"],
        "component_identity_max_abs_n": improved["component_identity_max_abs_n"],
        "baseline_reproduction_max_abs_coefficient": baseline_difference,
        "mean_CT_old_rerun": improved["baseline_uvlm_mean_thrust_n"]
        / (0.5 * IZ_CASE.rho_kg_m3 * IZ_CASE.freestream_m_s**2 * IZ_CASE.area_m2),
        "mean_CT_improved": improved["mean_CT"],
        "unsteady_ac_gain": improved["unsteady_ac_gain"],
        "circulatory_ac_gain": improved["circulatory_ac_gain"],
        "polar_mean_delta_lift_n": residual["mean_delta_lift_n"],
        "polar_mean_delta_drag_n": residual["mean_delta_drag_n"],
        "particle_count": improved["particle_count"],
    }
    log.append(
        f"Izraelevitz Fig11: mean CT={improved['mean_CT']:.6f}, "
        f"runtime={runtime:.2f}s"
    )
    return rows, audit


def _compute_metrics(
    yang_means: list[dict[str, Any]],
    iz_phase: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test = sorted(
        (row for row in yang_means if row["model"] == "wind_tunnel_test"),
        key=lambda row: float(row["aoa_deg"]),
    )
    reference_lift = np.asarray([float(row["mean_lift_gf"]) for row in test])
    reference_drag = np.asarray([float(row["mean_drag_gf"]) for row in test])
    for model in (
        "authors_proposed_modified_uvlm",
        "fluxv_uvpm",
        "fluxv_periodic_v1",
        "fluxv_periodic_v2",
        "one_state_ullt_local",
        "ptera_free_wake_uvlm",
        "robofalcon2_coefficient_transfer",
    ):
        prediction = sorted(
            (row for row in yang_means if row["model"] == model),
            key=lambda row: float(row["aoa_deg"]),
        )
        if not prediction:
            continue
        for channel, key, reference in (
            ("lift", "mean_lift_gf", reference_lift),
            ("drag", "mean_drag_gf", reference_drag),
        ):
            metric = _curve_metrics(
                np.asarray([float(row[key]) for row in prediction]), reference
            )
            rows.append(
                {
                    "paper": "yang2025",
                    "reference": "wind_tunnel_test",
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "channel": channel,
                    "units": "gf",
                    **metric,
                }
            )

    reference_rows = [row for row in iz_phase if row["model"] == "paper_uvlm"]
    ref_lift = np.asarray([float(row["CL_alpha"]) for row in reference_rows])
    ref_drag = np.asarray([float(row["CD_alpha"]) for row in reference_rows])
    for model in (
        "6_state",
        "1_state",
        "qs_added_mass",
        "one_state_ullt_local",
        "fluxv_uvpm",
        "fluxv_periodic_v1",
        "fluxv_periodic_v2",
        "ptera_free_wake_uvlm",
    ):
        prediction = [row for row in iz_phase if row["model"] == model]
        if not prediction:
            continue
        for channel, key, reference in (
            ("lift", "CL_alpha", ref_lift),
            ("drag", "CD_alpha", ref_drag),
        ):
            metric = _phase_metrics(
                np.asarray([float(row[key]) for row in prediction]), reference
            )
            rows.append(
                {
                    "paper": "izraelevitz2017_fig11",
                    "reference": "paper_uvlm",
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "channel": channel,
                    "units": "paper_scaled_coefficient",
                    **metric,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("full",), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    yang_means, yang_phase, yang_audit = _run_yang(output, log)
    iz_phase, iz_audit = _run_izraelevitz(log)
    metrics = _compute_metrics(yang_means, iz_phase)

    mean_fields = [
        "aoa_deg",
        "model",
        "model_label",
        "status",
        "mean_lift_n",
        "mean_drag_n",
        "mean_thrust_n",
        "mean_lift_gf",
        "mean_drag_gf",
        "mean_thrust_gf",
        "mean_CL",
        "mean_CD",
        "mean_CT",
        "runtime_s",
        "digitization_uncertainty_gf",
        "data_role",
        "error",
    ]
    _write_csv(output / "yang2025_mean_characteristics.csv", yang_means, mean_fields)
    _write_csv(
        output / "yang2025_phase_histories.csv",
        yang_phase,
        [
            "aoa_deg",
            "model",
            "model_label",
            "phase",
            "lift_n",
            "drag_n",
            "thrust_n",
            "CL",
            "CD",
            "CT",
        ],
    )
    _write_csv(
        output / "izraelevitz2017_fig11_phase_histories.csv",
        iz_phase,
        [
            "model",
            "model_label",
            "data_role",
            "phase",
            "CL_alpha",
            "CD_alpha",
            "CL",
            "CD",
            "CT",
        ],
    )
    metric_fields = [
        "paper",
        "reference",
        "model",
        "model_label",
        "channel",
        "units",
        "mae",
        "rmse",
        "bias",
        "max_abs_error",
        "range_nrmse",
        "prediction_half_amplitude",
        "reference_half_amplitude",
        "half_amplitude_error",
        "positive_peak_phase_error_cycle",
        "negative_peak_phase_error_cycle",
    ]
    _write_csv(output / "accuracy_metrics.csv", metrics, metric_fields)

    metric_lookup = {
        (row["paper"], row["model"], row["channel"]): row for row in metrics
    }
    old_y_l = metric_lookup[("yang2025", "fluxv_uvpm", "lift")]["mae"]
    old_y_d = metric_lookup[("yang2025", "fluxv_uvpm", "drag")]["mae"]
    new_y_l = metric_lookup[("yang2025", "fluxv_periodic_v1", "lift")]["mae"]
    new_y_d = metric_lookup[("yang2025", "fluxv_periodic_v1", "drag")]["mae"]
    old_i_l = metric_lookup[("izraelevitz2017_fig11", "fluxv_uvpm", "lift")]["rmse"]
    old_i_d = metric_lookup[("izraelevitz2017_fig11", "fluxv_uvpm", "drag")]["rmse"]
    new_i_l = metric_lookup[("izraelevitz2017_fig11", "fluxv_periodic_v1", "lift")]["rmse"]
    new_i_d = metric_lookup[("izraelevitz2017_fig11", "fluxv_periodic_v1", "drag")]["rmse"]
    gates = {
        "yang_both_mae_improved": bool(new_y_l < old_y_l and new_y_d < old_y_d),
        "izraelevitz_both_raw_phase_rmse_improved": bool(
            new_i_l < old_i_l and new_i_d < old_i_d
        ),
        "joint_non_degradation_gate": bool(
            new_y_l < old_y_l
            and new_y_d < old_y_d
            and new_i_l < old_i_l
            and new_i_d < old_i_d
        ),
        "strong_cross_paper_accuracy_claim": bool(
            new_y_l <= 3.5
            and new_y_d <= 2.0
            and new_i_l <= 0.5
            and new_i_d <= 0.5
        ),
    }
    summary = {
        "status": "complete",
        "quality": args.quality,
        "yang2025": {
            "old_fluxv_mae_gf": {"lift": old_y_l, "drag": old_y_d},
            "augmented_v1_mae_gf": {"lift": new_y_l, "drag": new_y_d},
            "relative_reduction_percent": {
                "lift": 100.0 * (old_y_l - new_y_l) / old_y_l,
                "drag": 100.0 * (old_y_d - new_y_d) / old_y_d,
            },
        },
        "izraelevitz2017_fig11": {
            "old_fluxv_raw_phase_rmse": {"lift": old_i_l, "drag": old_i_d},
            "augmented_v1_raw_phase_rmse": {"lift": new_i_l, "drag": new_i_d},
            "relative_reduction_percent": {
                "lift": 100.0 * (old_i_l - new_i_l) / old_i_l,
                "drag": 100.0 * (old_i_d - new_i_d) / old_i_d,
            },
        },
        "gates": gates,
        "claim_classification": (
            "exploratory joint improvement, not a strong cross-paper accuracy result"
            if gates["joint_non_degradation_gate"]
            and not gates["strong_cross_paper_accuracy_claim"]
            else "joint gate failed"
        ),
    }
    manifest = {
        "run_id": output.name,
        "status": "complete",
        "quality": args.quality,
        "model": "fluxv_periodic_v1",
        "model_identity": (
            "UVPMHybridSolver prescribed-wake UVLM load ledger plus source-derived "
            "periodic AC gains and a shared separated-incidence polar residual"
        ),
        "causality_scope": (
            "periodic two-pass postprocessing; not yet an online transient production solver"
        ),
        "execution_fast_path": {
            "enabled": True,
            "reason": (
                "current VPM particles are one-way diagnostics and do not enter the "
                "UVLM AIC or load channel"
            ),
            "regression": (
                "platform/tests/test_augmented_uvpm.py::"
                "test_particle_fast_path_is_exactly_load_equivalent_on_smoke_case"
            ),
            "effect": "skip particle shedding/advection while retaining identical loads",
        },
        "force_contract": {
            "ptera_lift": "L=-Fz_W",
            "ptera_thrust": "T=+Fx_W",
            "plotted_drag": "D=-T=-Fx_W",
            "yang_units": "single-wing gf, 1 gf=0.00980665 N",
            "izraelevitz_paper_scaling": "CLalpha=CL/sin(15deg), CDalpha=CD/sin(15deg)",
        },
        "comparability_limits": [
            "Yang local models use nominal four-bar kinematics; the paper's LDS history is unpublished.",
            "Yang Test and Authors' Proposed values are digitized cycle means; no experimental phase trace is public.",
            "Izraelevitz paper UVLM is a numerical reference, not experimental ground truth.",
            "Old FluxV and Ptera prescribed-wake UVLM are the same load channel.",
            "Izraelevitz local Ptera discretization is not demonstrably time/space converged.",
            "The 15--20 degree polar gate is exploratory and was introduced after the v0 diagnostic.",
        ],
        "source_hashes": {
            _manifest_path(YANG_PRIOR / "mean_characteristics.csv"): _sha256(
                YANG_PRIOR / "mean_characteristics.csv"
            ),
            _manifest_path(YANG_PRIOR / "phase_histories.csv"): _sha256(
                YANG_PRIOR / "phase_histories.csv"
            ),
            _manifest_path(IZ_PRIOR / "phase_histories.csv"): _sha256(
                IZ_PRIOR / "phase_histories.csv"
            ),
            _manifest_path(IZ_REFERENCE): _sha256(IZ_REFERENCE),
        },
        "yang_audit": yang_audit,
        "izraelevitz_audit": iz_audit,
        "gates": gates,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
