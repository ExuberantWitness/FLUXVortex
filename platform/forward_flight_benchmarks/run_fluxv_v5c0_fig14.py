"""Rebase the frozen Figure-14 v4b load ledger from 0.25c to 0.75c.

This is the v5c0 correctness stage.  It does not introduce a new aerodynamic
model.  The runner replaces the local-reference polar/profile terms in the
frozen phase cache, replays the unchanged v4b owner equation, and records the
result under a fresh run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from forward_flight_benchmarks.cases import IZRAELEVITZ_2017_FIG14_SCHERER
from forward_flight_benchmarks.fluxv_v5c_ledger import (
    ProfileDragReference,
    rebase_local_velocity_reference,
)
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_scherer_movement,
)
from forward_flight_benchmarks.run_v4_crosspaper import (
    _fig14_phase_diagnostic,
)
from forward_flight_benchmarks.uvlm_polar_correction import (
    FullAnglePolarParameters,
    movement_polar_residual,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = REPO_ROOT / "docs/forward_flight_large_pitch/reproductions"
V3_ROOT = REPRO_ROOT / "unified_fluxv_upgrade_20260812"
V4_ROOT = REPRO_ROOT / "unified_fluxv_v4_ldvm_stevens_20260812"
DOC_ROOT = REPRO_ROOT / "fluxv_v5c_nextgen_20260814"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5c0_reference_full"

PHASE_CACHE = V4_ROOT / "source_data/izraelevitz2017_fig14_local_phase_cache.csv"
PHASE_CACHE_SHA256 = "e64c67eec46f32caefd2712753084963fb867f30b661b18fb71aeae95837b8b8"
V4_MEANS = (
    V4_ROOT
    / "runs/20260812_fluxv_v4b_crosspaper_full"
    / "izraelevitz2017_fig14_v4_mean_thrust.csv"
)
V4_SUMMARY = V4_ROOT / "runs/20260812_fluxv_v4b_crosspaper_full" / "summary.json"
SOURCE = V3_ROOT / "source_data/izraelevitz2017_fig14_digitized.csv"
PROFILE_SENSITIVITY = (
    V3_ROOT
    / "runs/20260812_scherer_fig14_experiment_full"
    / "profile_drag_sensitivity.csv"
)

OWNED_PARAMETERS = FullAnglePolarParameters(
    section_velocity_reference_fraction_chord=0.25
)
TARGET_PARAMETERS = FullAnglePolarParameters(
    section_velocity_reference_fraction_chord=(
        IZRAELEVITZ_2017_FIG14_SCHERER.pivot_fraction_chord
    )
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cycle_range(movement: Any, period_s: float) -> tuple[int, int]:
    count = len(movement.airplanes[0])
    steps = int(round(period_s / movement.delta_time))
    last = count - 1
    cycles_at_last = last * movement.delta_time / period_s
    end = last if np.isclose(cycles_at_last, round(cycles_at_last)) else count
    return end - steps, end - 1


def _periodic_resample(values: np.ndarray, output_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    source = np.arange(values.size, dtype=float) / values.size
    target = np.arange(output_samples, dtype=float) / output_samples
    return np.interp(target, source, values, period=1.0)


def _history(
    rows: list[dict[str, str]],
    model: str,
    condition: tuple[float, float],
) -> dict[str, Any]:
    theta, psi = condition
    selected = [
        row
        for row in rows
        if row["model"] == model
        and np.isclose(float(row["theta_max_deg"]), theta)
        and np.isclose(float(row["phase_offset_deg"]), psi)
    ]
    selected.sort(key=lambda row: float(row["phase"]))
    if len(selected) != 128:
        raise ValueError(f"expected 128 phase rows for {model}, {condition}")
    return {
        "phase": np.asarray([float(row["phase"]) for row in selected]),
        "lift_n": np.asarray([float(row["lift_n"]) for row in selected]),
        "drag_n": np.asarray([float(row["drag_n"]) for row in selected]),
        "profile_drag_reference_fraction_chord": 0.25,
        "profile_drag_coefficient": (
            IZRAELEVITZ_2017_FIG14_SCHERER.profile_drag_coefficient
        ),
    }


def _metric(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs": float(np.max(np.abs(error))),
    }


def _score(
    observations: list[dict[str, str]],
    predictions: dict[tuple[float, float], float],
) -> dict[str, Any]:
    observed = np.asarray([float(row["ct"]) for row in observations])
    predicted = np.asarray(
        [
            predictions[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))]
            for row in observations
        ]
    )
    all_markers = _metric(observed, predicted)
    by_condition: dict[tuple[float, float], list[float]] = {}
    for row in observations:
        key = (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
        by_condition.setdefault(key, []).append(float(row["ct"]))
    unique_observed = np.asarray(
        [np.mean(by_condition[key]) for key in sorted(by_condition)]
    )
    unique_predicted = np.asarray([predictions[key] for key in sorted(by_condition)])
    theta_metrics = {}
    for theta in (15.0, 25.0):
        selected = [
            row
            for row in observations
            if np.isclose(float(row["theta_max_deg"]), theta)
        ]
        theta_metrics[f"theta_{theta:g}"] = _metric(
            np.asarray([float(row["ct"]) for row in selected]),
            np.asarray(
                [
                    predictions[
                        (
                            float(row["theta_max_deg"]),
                            float(row["phase_offset_deg"]),
                        )
                    ]
                    for row in selected
                ]
            ),
        )
    return {
        "all_14_markers": all_markers,
        "unique_12_conditions": _metric(unique_observed, unique_predicted),
        **theta_metrics,
    }


def _sensitivity_profile_ct(
    rows: list[dict[str, str]], condition: tuple[float, float]
) -> float:
    theta, psi = condition
    selected = [
        row
        for row in rows
        if row["model"] == "fluxv_uvpm"
        and np.isclose(float(row["theta_max_deg"]), theta)
        and np.isclose(float(row["phase_offset_deg"]), psi)
        and float(row["profile_drag_coefficient"]) in (0.0, 0.057)
    ]
    lookup = {
        float(row["profile_drag_coefficient"]): float(row["CT"]) for row in selected
    }
    if set(lookup) != {0.0, 0.057}:
        raise ValueError(f"missing frozen profile sensitivity for {condition}")
    return lookup[0.057] - lookup[0.0]


def run(output: Path, *, steps_per_cycle: int = 256) -> dict[str, Any]:
    if _sha256(PHASE_CACHE) != PHASE_CACHE_SHA256:
        raise RuntimeError("frozen Figure-14 phase cache hash mismatch")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    case = IZRAELEVITZ_2017_FIG14_SCHERER
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    ownership = ProfileDragReference(
        coefficient=case.profile_drag_coefficient,
        fraction_chord=0.25,
    )
    cache_rows = _rows(PHASE_CACHE)
    if len(cache_rows) != 4608:
        raise RuntimeError("unexpected frozen Figure-14 cache row count")
    v4_rows = _rows(V4_MEANS)
    frozen_v4 = {
        (float(row["theta_max_deg"]), float(row["phase_offset_deg"])): float(
            row["v4_CT"]
        )
        for row in v4_rows
    }
    observations = [
        row for row in _rows(SOURCE) if row["series"] == "scherer_1968_experiment"
    ]
    conditions = sorted(frozen_v4)
    sensitivity_rows = _rows(PROFILE_SENSITIVITY)

    means: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    legacy_predictions: dict[tuple[float, float], float] = {}
    corrected_predictions: dict[tuple[float, float], float] = {}
    legacy_replay_residuals: list[float] = []
    slope_residuals: list[float] = []
    identity_residuals: list[float] = []

    for theta, psi in conditions:
        condition = (theta, psi)
        movement, movement_meta = build_izraelevitz_scherer_movement(theta, psi, "full")
        cycle = _cycle_range(movement, case.period_s)
        common = {
            "movement": movement,
            "source_cycle_step_range": cycle,
            "period_s": case.period_s,
            "freestream_m_s": case.freestream_m_s,
            "rho_kg_m3": case.rho_kg_m3,
            "aspect_ratio": case.aspect_ratio,
            "output_samples": 128,
        }
        owned_kinematics = movement_polar_residual(
            **common, parameters=OWNED_PARAMETERS
        )
        target_kinematics = movement_polar_residual(
            **common, parameters=TARGET_PARAMETERS
        )
        old_owned = _history(cache_rows, "fluxv_uvpm", condition)
        polar_owned = _history(cache_rows, "fluxv_periodic_v1", condition)
        old_target = rebase_local_velocity_reference(
            old_owned,
            owned_kinematics,
            target_kinematics,
            ownership=ownership,
            target_fraction_chord=case.pivot_fraction_chord,
            replace_polar_residual=False,
            rho_kg_m3=case.rho_kg_m3,
            freestream_m_s=case.freestream_m_s,
            area_m2=case.area_m2,
        )
        polar_target = rebase_local_velocity_reference(
            polar_owned,
            owned_kinematics,
            target_kinematics,
            ownership=ownership,
            target_fraction_chord=case.pivot_fraction_chord,
            replace_polar_residual=True,
            rho_kg_m3=case.rho_kg_m3,
            freestream_m_s=case.freestream_m_s,
            area_m2=case.area_m2,
        )

        # Replacing a reference by itself must be an exact array identity.
        identity = rebase_local_velocity_reference(
            old_owned,
            owned_kinematics,
            owned_kinematics,
            ownership=ownership,
            target_fraction_chord=0.25,
            replace_polar_residual=False,
            rho_kg_m3=case.rho_kg_m3,
            freestream_m_s=case.freestream_m_s,
            area_m2=case.area_m2,
        )
        identity_residuals.append(
            max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(identity["lift_n"])
                            - np.asarray(old_owned["lift_n"])
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(identity["drag_n"])
                            - np.asarray(old_owned["drag_n"])
                        )
                    )
                ),
            )
        )

        diagnostic = _fig14_phase_diagnostic(
            theta, psi, steps_per_cycle=steps_per_cycle
        )
        p = _periodic_resample(np.asarray(diagnostic["persistence"]), 128)
        delta_ct = _periodic_resample(np.asarray(diagnostic["ldvm_delta_CT"]), 128)
        old_owned_ct = -np.asarray(old_owned["drag_n"]) / q_area
        polar_owned_ct = -np.asarray(polar_owned["drag_n"]) / q_area
        legacy_history = (1.0 - p) * (old_owned_ct + delta_ct) + p * polar_owned_ct
        corrected_history = (1.0 - p) * (
            np.asarray(old_target["CT"]) + delta_ct
        ) + p * np.asarray(polar_target["CT"])
        profile_reference_delta_ct = np.asarray(old_target["CT"]) - old_owned_ct
        polar_reference_delta_ct = (
            np.asarray(polar_target["CT"]) - polar_owned_ct - profile_reference_delta_ct
        )
        legacy_mean = float(np.mean(legacy_history))
        corrected_mean = float(np.mean(corrected_history))
        legacy_predictions[condition] = legacy_mean
        corrected_predictions[condition] = corrected_mean
        legacy_replay_residuals.append(abs(legacy_mean - frozen_v4[condition]))

        profile_owned_ct = (
            -case.profile_drag_coefficient
            * float(owned_kinematics["mean_unit_profile_drag_drag_n"])
            / q_area
        )
        frozen_profile_ct = _sensitivity_profile_ct(sensitivity_rows, condition)
        slope_residuals.append(abs(profile_owned_ct - frozen_profile_ct))
        means.append(
            {
                "theta_max_deg": theta,
                "phase_offset_deg": psi,
                "legacy_v4b_CT_025c": legacy_mean,
                "corrected_v5c0_CT_075c": corrected_mean,
                "reference_correction_CT": corrected_mean - legacy_mean,
                "profile_reference_correction_CT": float(
                    np.mean(profile_reference_delta_ct)
                ),
                "polar_reference_correction_CT": float(
                    np.mean(polar_reference_delta_ct)
                ),
                "persistence_fraction": float(np.mean(p)),
                "ldvm_delta_CT": float(np.mean(delta_ct)),
                "profile_drag_coefficient": case.profile_drag_coefficient,
                "owned_reference_fraction_chord": 0.25,
                "target_reference_fraction_chord": case.pivot_fraction_chord,
                "profile_drag_application_count_after": 1,
                "source_cycle_start": cycle[0],
                "source_cycle_stop": cycle[1],
                "movement_steps_per_cycle": movement_meta["steps_per_cycle"],
            }
        )
        for index, phase in enumerate(np.asarray(old_owned["phase"])):
            phases.extend(
                [
                    {
                        "model": "v4b_legacy_025c",
                        "theta_max_deg": theta,
                        "phase_offset_deg": psi,
                        "phase": phase,
                        "CT": legacy_history[index],
                        "persistence": p[index],
                        "ldvm_delta_CT": delta_ct[index],
                        "owned_old_CT": old_owned_ct[index],
                        "owned_polar_CT": polar_owned_ct[index],
                        "target_old_CT": np.asarray(old_target["CT"])[index],
                        "target_polar_CT": np.asarray(polar_target["CT"])[index],
                        "profile_reference_delta_CT": profile_reference_delta_ct[index],
                        "polar_reference_delta_CT": polar_reference_delta_ct[index],
                    },
                    {
                        "model": "v5c0_corrected_v4b_075c",
                        "theta_max_deg": theta,
                        "phase_offset_deg": psi,
                        "phase": phase,
                        "CT": corrected_history[index],
                        "persistence": p[index],
                        "ldvm_delta_CT": delta_ct[index],
                        "owned_old_CT": old_owned_ct[index],
                        "owned_polar_CT": polar_owned_ct[index],
                        "target_old_CT": np.asarray(old_target["CT"])[index],
                        "target_polar_CT": np.asarray(polar_target["CT"])[index],
                        "profile_reference_delta_CT": profile_reference_delta_ct[index],
                        "polar_reference_delta_CT": polar_reference_delta_ct[index],
                    },
                ]
            )

    max_replay = max(legacy_replay_residuals)
    max_slope = max(slope_residuals)
    max_identity = max(identity_residuals)
    if max_replay > 1.0e-12:
        raise RuntimeError(f"legacy v4b replay residual {max_replay:.3e} exceeds gate")
    if max_slope > 1.0e-12:
        raise RuntimeError(f"0.25c profile slope residual {max_slope:.3e} exceeds gate")
    if max_identity > 1.0e-12:
        raise RuntimeError(
            f"reference identity residual {max_identity:.3e} exceeds gate"
        )

    mean_path = output / "fig14_v5c0_mean_thrust.csv"
    phase_path = output / "fig14_v5c0_phase_histories.csv"
    _write_csv(mean_path, means)
    _write_csv(phase_path, phases)
    legacy_metrics = _score(observations, legacy_predictions)
    corrected_metrics = _score(observations, corrected_predictions)
    summary = {
        "run_id": output.name,
        "status": "v5c0_correctness_baseline_complete",
        "evidence_level": "cache_replay_plus_source_kinematics",
        "legacy_v4b_metrics": legacy_metrics,
        "corrected_v5c0_metrics": corrected_metrics,
        "gates": {
            "legacy_replay_max_abs": max_replay,
            "legacy_replay_pass": max_replay <= 1.0e-12,
            "owned_profile_slope_max_abs": max_slope,
            "owned_profile_slope_pass": max_slope <= 1.0e-12,
            "same_reference_identity_max_abs": max_identity,
            "same_reference_identity_pass": max_identity <= 1.0e-12,
            "profile_drag_application_count": 1,
            "target_reference_is_source_pivot": bool(case.pivot_fraction_chord == 0.75),
        },
        "model_semantics": (
            "v4b local polar/profile terms rebased from 0.25c to the source-defined "
            "0.75c reference; UVLM, LDVM and causal owner histories unchanged"
        ),
        "parameter_selection_data": [],
        "scoring_data_used_only_after_prediction": True,
        "limitations": [
            "This is a ledger correction, not a new LEV or dynamic-stall model.",
            "The corrected result remains development evidence on previously inspected data.",
        ],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": output.name,
        "steps_per_cycle_ldvm": steps_per_cycle,
        "profile_drag_coefficient": case.profile_drag_coefficient,
        "owned_reference_fraction_chord": 0.25,
        "target_reference_fraction_chord": case.pivot_fraction_chord,
        "profile_drag_application_count_after": 1,
        "condition_count": len(conditions),
        "phase_rows": len(phases),
        "parameter_selection_data": [],
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("fluxv_v5c_ledger.py").resolve(),
                Path(__file__).with_name("uvlm_polar_correction.py").resolve(),
                Path(__file__).with_name("run_v4_crosspaper.py").resolve(),
                Path(__file__).with_name("ptera_adapter.py").resolve(),
                Path(__file__).with_name("cases.py").resolve(),
                PHASE_CACHE,
                V4_MEANS,
                V4_SUMMARY,
                SOURCE,
                PROFILE_SENSITIVITY,
            )
        },
        "result_hashes": {
            path.name: _sha256(path) for path in (mean_path, phase_path, summary_path)
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps-per-cycle", type=int, default=256)
    args = parser.parse_args()
    summary = run(args.output_dir.resolve(), steps_per_cycle=args.steps_per_cycle)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
