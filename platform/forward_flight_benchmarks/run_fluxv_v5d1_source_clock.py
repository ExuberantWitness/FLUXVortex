"""Evaluate the source-clock causal owner on the frozen 22-condition suite.

This is an isolated correctness experiment.  It keeps the v4b load branches
unchanged and replaces only the incidence owner's clock by the source-defined
semi-chord travel

``Delta t_tilde = 2 |V_rel,0.75c,perp span| dt / c_local``.

The current adapter exports kinematic strip velocity but not same-time-layer
UVLM-induced velocity.  Every result is consequently non-canonical even when
its numerical no-regression gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from forward_flight_benchmarks.baik2012 import (
    BAIK_2012_CASES,
    build_baik_movement,
)
from forward_flight_benchmarks.cases import (
    IZRAELEVITZ_2017_FIG14_SCHERER,
    YANG_2025,
)
from forward_flight_benchmarks.fluxv_v5d_source_owner import (
    DEFAULT_SOURCE_TIME_OWNER_PARAMETERS,
    source_time_causal_persistence,
)
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_scherer_movement,
    build_yang2025_movement,
)
from forward_flight_benchmarks.run_fluxv_v5c0_fig14 import _cycle_range
from forward_flight_benchmarks.run_fluxv_v5c1_proxy import (
    BAIK_GT,
    BAIK_MACRO_THRESHOLDS,
    BAIK_PHASE,
    BAIK_THRESHOLDS,
    FIG14_GT,
    OUTPUT_SAMPLES,
    V5C0_PHASE,
    V5C0_SUMMARY,
    YANG_GT,
    YANG_PHASE,
    YANG_THRESHOLDS,
    YANG_V4,
    _periodic_resample,
    _phase_history,
    _read_csv,
    _score_baik,
    _score_fig14,
    _score_yang,
    _write_csv,
)
from forward_flight_benchmarks.run_v4_crosspaper import (
    _yang_ldvm_phase_and_ownership,
)
from forward_flight_benchmarks.uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    movement_polar_residual,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = REPO_ROOT / "docs/forward_flight_large_pitch/reproductions"
DOC_ROOT = REPRO_ROOT / "fluxv_v5c_nextgen_20260814"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5d1_source_clock_all22"

OWNER_WARMUP_CYCLES = 80
OWNER_REFERENCE_FRACTION_CHORD = 0.75
YANG_LDVM_STEPS = 256
YANG_STRIPS = 12

SOURCE_OWNER_KINEMATICS = replace(
    DEFAULT_POLAR_PARAMETERS,
    section_velocity_reference_fraction_chord=OWNER_REFERENCE_FRACTION_CHORD,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _source_owner(
    movement: Any,
    *,
    period_s: float,
    freestream_m_s: float,
    rho_kg_m3: float,
    aspect_ratio: float,
    chord_m: float,
) -> dict[str, Any]:
    """Return one established-cycle source-clock owner and strip evidence."""

    source_cycle = _cycle_range(movement, period_s)
    kinematics = movement_polar_residual(
        movement,
        source_cycle_step_range=source_cycle,
        period_s=period_s,
        freestream_m_s=freestream_m_s,
        rho_kg_m3=rho_kg_m3,
        aspect_ratio=aspect_ratio,
        output_samples=OUTPUT_SAMPLES,
        parameters=SOURCE_OWNER_KINEMATICS,
    )
    alpha = np.asarray(kinematics["alpha_rad"], dtype=float)
    speed = np.asarray(kinematics["relative_speed_m_s"], dtype=float)
    if alpha.shape != speed.shape or alpha.ndim != 2:
        raise RuntimeError("source kinematic strip histories do not align")
    step_s = period_s / OUTPUT_SAMPLES
    delta_t_tilde = 2.0 * speed * step_s / chord_m
    area = np.asarray(kinematics["mean_strip_area_m2"], dtype=float)
    tiled_alpha = np.tile(alpha, (OWNER_WARMUP_CYCLES, 1))
    tiled_step = np.tile(delta_t_tilde, (OWNER_WARMUP_CYCLES, 1))
    owner = source_time_causal_persistence(
        tiled_alpha,
        delta_t_tilde=tiled_step,
        strip_weights=area,
    )
    disabled = source_time_causal_persistence(
        tiled_alpha,
        delta_t_tilde="disabled path must not evaluate this value",
        strip_weights={"disabled": "not evaluated"},
        enabled=False,
    )
    last = slice(-OUTPUT_SAMPLES, None)
    previous = slice(-2 * OUTPUT_SAMPLES, -OUTPUT_SAMPLES)
    state_keys = (
        "signed_fast_history_rad",
        "signed_slow_history_rad",
        "magnitude_fast_history_rad",
        "magnitude_slow_history_rad",
    )
    cycle_error = max(
        float(
            np.max(
                np.abs(np.asarray(owner[key])[last] - np.asarray(owner[key])[previous])
            )
        )
        for key in state_keys
    )
    source_formula_error = float(
        np.max(np.abs(delta_t_tilde - 2.0 * speed * step_s / chord_m))
    )
    return {
        "persistence": np.asarray(owner["global_persistence"])[last],
        "previous_persistence": np.asarray(owner["global_persistence"])[previous],
        "strip_persistence": np.asarray(owner["strip_persistence"])[last],
        "previous_strip_persistence": np.asarray(owner["strip_persistence"])[previous],
        "alpha_rad": alpha,
        "relative_speed_m_s": speed,
        "delta_t_tilde": delta_t_tilde,
        "strip_area_m2": area,
        "normalized_strip_weights": np.asarray(owner["normalized_strip_weights"]),
        "signed_fast_history_rad": np.asarray(owner["signed_fast_history_rad"])[last],
        "signed_slow_history_rad": np.asarray(owner["signed_slow_history_rad"])[last],
        "magnitude_fast_history_rad": np.asarray(owner["magnitude_fast_history_rad"])[
            last
        ],
        "magnitude_slow_history_rad": np.asarray(owner["magnitude_slow_history_rad"])[
            last
        ],
        "disabled_max_abs": float(
            max(
                np.max(np.abs(np.asarray(disabled["global_persistence"]))),
                np.max(np.abs(np.asarray(disabled["strip_persistence"]))),
            )
        ),
        "periodic_state_max_abs": cycle_error,
        "source_clock_formula_max_abs": source_formula_error,
        "source_cycle_step_range": list(source_cycle),
        "kinematic_strip_count": int(alpha.shape[1]),
        "parameters": owner["parameters"],
    }


def _append_owner_state(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    case_id: str,
    owner: dict[str, Any],
) -> None:
    strip_count = int(owner["kinematic_strip_count"])
    phase = np.arange(OUTPUT_SAMPLES, dtype=float) / OUTPUT_SAMPLES
    for time_index, phase_value in enumerate(phase):
        for strip_index in range(strip_count):
            rows.append(
                {
                    "benchmark": benchmark,
                    "case_id": case_id,
                    "phase": phase_value,
                    "strip_id": strip_index,
                    "alpha_rad": owner["alpha_rad"][time_index, strip_index],
                    "relative_speed_m_s": owner["relative_speed_m_s"][
                        time_index, strip_index
                    ],
                    "delta_t_tilde": owner["delta_t_tilde"][time_index, strip_index],
                    "strip_area_m2": owner["strip_area_m2"][strip_index],
                    "normalized_strip_weight": owner["normalized_strip_weights"][
                        strip_index
                    ],
                    "strip_persistence": owner["strip_persistence"][
                        time_index, strip_index
                    ],
                    "previous_cycle_strip_persistence": owner[
                        "previous_strip_persistence"
                    ][time_index, strip_index],
                    "global_persistence": owner["persistence"][time_index],
                    "previous_cycle_global_persistence": owner["previous_persistence"][
                        time_index
                    ],
                    "signed_fast_rad": owner["signed_fast_history_rad"][
                        time_index, strip_index
                    ],
                    "signed_slow_rad": owner["signed_slow_history_rad"][
                        time_index, strip_index
                    ],
                    "magnitude_fast_rad": owner["magnitude_fast_history_rad"][
                        time_index, strip_index
                    ],
                    "magnitude_slow_rad": owner["magnitude_slow_history_rad"][
                        time_index, strip_index
                    ],
                    "reference_fraction_chord": OWNER_REFERENCE_FRACTION_CHORD,
                    "canonical_eligible": "false",
                }
            )


def _yang_predictions(
    phase_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    histories = _read_csv(YANG_PHASE)
    frozen = {float(row["aoa_deg"]): row for row in _read_csv(YANG_V4)}
    q_area_per_gf = (
        0.5
        * YANG_2025.rho_kg_m3
        * YANG_2025.freestream_m_s**2
        * YANG_2025.area_m2
        / 0.00980665
    )
    predictions: dict[str, dict[str, Any]] = {}
    replay_error = 0.0
    diagnostics = {
        "disabled_max_abs": 0.0,
        "periodic_state_max_abs": 0.0,
        "source_clock_formula_max_abs": 0.0,
    }
    for aoa in sorted(frozen):
        old = _phase_history(histories, "fluxv_uvpm", aoa_deg=aoa)
        polar = _phase_history(histories, "fluxv_periodic_v1", aoa_deg=aoa)
        ldvm = _yang_ldvm_phase_and_ownership(
            aoa, steps_per_cycle=YANG_LDVM_STEPS, strip_count=YANG_STRIPS
        )
        old_p = _periodic_resample(np.asarray(ldvm["persistence"]))
        delta_cl = _periodic_resample(np.asarray(ldvm["ldvm_delta_CL"]))
        delta_cd = _periodic_resample(np.asarray(ldvm["ldvm_delta_CD"]))
        movement, _ = build_yang2025_movement(
            aoa,
            settings=(2, YANG_STRIPS, OUTPUT_SAMPLES, 3, 2),
        )
        source_owner = _source_owner(
            movement,
            period_s=YANG_2025.period_s,
            freestream_m_s=YANG_2025.freestream_m_s,
            rho_kg_m3=YANG_2025.rho_kg_m3,
            aspect_ratio=YANG_2025.aspect_ratio,
            chord_m=YANG_2025.chord_m,
        )
        new_p = np.asarray(source_owner["persistence"])
        old_cl = np.asarray(old["CL"])
        old_cd = np.asarray(old["CD"])
        polar_cl = np.asarray(polar["CL"])
        polar_cd = np.asarray(polar["CD"])
        transient_cl = old_cl + delta_cl
        transient_cd = old_cd + delta_cd
        baseline_cl = (1.0 - old_p) * transient_cl + old_p * polar_cl
        baseline_cd = (1.0 - old_p) * transient_cd + old_p * polar_cd
        candidate_cl = (1.0 - new_p) * transient_cl + new_p * polar_cl
        candidate_cd = (1.0 - new_p) * transient_cd + new_p * polar_cd
        reference = frozen[aoa]
        baseline_lift = float(np.mean(baseline_cl) * q_area_per_gf)
        baseline_drag = float(np.mean(baseline_cd) * q_area_per_gf)
        replay_error = max(
            replay_error,
            abs(baseline_lift - float(reference["v4_lift_gf"])),
            abs(baseline_drag - float(reference["v4_drag_gf"])),
        )
        case_id = f"aoa_{aoa:g}"
        predictions[case_id] = {
            "aoa_deg": aoa,
            "baseline_lift_gf": float(reference["v4_lift_gf"]),
            "baseline_drag_gf": float(reference["v4_drag_gf"]),
            "proxy_lift_gf": float(np.mean(candidate_cl) * q_area_per_gf),
            "proxy_drag_gf": float(np.mean(candidate_cd) * q_area_per_gf),
        }
        _append_owner_state(
            state_rows,
            benchmark="yang2025",
            case_id=case_id,
            owner=source_owner,
        )
        diagnostics["disabled_max_abs"] = max(
            diagnostics["disabled_max_abs"], source_owner["disabled_max_abs"]
        )
        diagnostics["periodic_state_max_abs"] = max(
            diagnostics["periodic_state_max_abs"],
            source_owner["periodic_state_max_abs"],
        )
        diagnostics["source_clock_formula_max_abs"] = max(
            diagnostics["source_clock_formula_max_abs"],
            source_owner["source_clock_formula_max_abs"],
        )
        for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
            phase_rows.append(
                {
                    "benchmark": "yang2025",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CL": baseline_cl[index],
                    "baseline_CD": baseline_cd[index],
                    "source_clock_CL": candidate_cl[index],
                    "source_clock_CD": candidate_cd[index],
                    "old_persistence": old_p[index],
                    "source_clock_persistence": new_p[index],
                    "ldvm_delta_CL": delta_cl[index],
                    "ldvm_delta_CD": delta_cd[index],
                    "canonical_eligible": "false",
                }
            )
    diagnostics["baseline_replay_max_abs_gf"] = replay_error
    return predictions, diagnostics


def _fig14_predictions(
    phase_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    source = _read_csv(V5C0_PHASE)
    conditions = sorted(
        {
            (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
            for row in source
            if row["model"] == "v5c0_corrected_v4b_075c"
        }
    )
    predictions: dict[str, dict[str, Any]] = {}
    diagnostics = {
        "disabled_max_abs": 0.0,
        "periodic_state_max_abs": 0.0,
        "source_clock_formula_max_abs": 0.0,
    }
    for theta, psi in conditions:
        selected = [
            row
            for row in source
            if row["model"] == "v5c0_corrected_v4b_075c"
            and np.isclose(float(row["theta_max_deg"]), theta)
            and np.isclose(float(row["phase_offset_deg"]), psi)
        ]
        selected.sort(key=lambda row: float(row["phase"]))
        if len(selected) != OUTPUT_SAMPLES:
            raise RuntimeError("Figure-14 corrected phase cache is incomplete")
        baseline_ct = np.asarray([float(row["CT"]) for row in selected])
        old_ct = np.asarray([float(row["target_old_CT"]) for row in selected])
        polar_ct = np.asarray([float(row["target_polar_CT"]) for row in selected])
        delta_ct = np.asarray([float(row["ldvm_delta_CT"]) for row in selected])
        old_p = np.asarray([float(row["persistence"]) for row in selected])
        movement, _ = build_izraelevitz_scherer_movement(
            theta,
            psi,
            settings=(2, 12, OUTPUT_SAMPLES, 4),
        )
        owner = _source_owner(
            movement,
            period_s=IZRAELEVITZ_2017_FIG14_SCHERER.period_s,
            freestream_m_s=IZRAELEVITZ_2017_FIG14_SCHERER.freestream_m_s,
            rho_kg_m3=IZRAELEVITZ_2017_FIG14_SCHERER.rho_kg_m3,
            aspect_ratio=IZRAELEVITZ_2017_FIG14_SCHERER.aspect_ratio,
            chord_m=IZRAELEVITZ_2017_FIG14_SCHERER.chord_m,
        )
        new_p = np.asarray(owner["persistence"])
        candidate_ct = (1.0 - new_p) * (old_ct + delta_ct) + new_p * polar_ct
        case_id = f"theta_{theta:g}_psi_{psi:g}"
        predictions[case_id] = {
            "theta_max_deg": theta,
            "phase_offset_deg": psi,
            "baseline_CT": float(np.mean(baseline_ct)),
            "proxy_CT": float(np.mean(candidate_ct)),
        }
        _append_owner_state(
            state_rows,
            benchmark="izraelevitz2017_fig14",
            case_id=case_id,
            owner=owner,
        )
        for key in diagnostics:
            diagnostics[key] = max(diagnostics[key], owner[key])
        for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
            phase_rows.append(
                {
                    "benchmark": "izraelevitz2017_fig14",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CT": baseline_ct[index],
                    "source_clock_CT": candidate_ct[index],
                    "old_persistence": old_p[index],
                    "source_clock_persistence": new_p[index],
                    "ldvm_delta_CT": delta_ct[index],
                    "canonical_eligible": "false",
                }
            )
    return predictions, diagnostics


def _model_history(
    rows: list[dict[str, str]], case_id: str, model: str
) -> dict[str, np.ndarray]:
    selected = [
        row for row in rows if row["case_id"] == case_id and row["model"] == model
    ]
    selected.sort(key=lambda row: float(row["phase"]))
    if len(selected) != OUTPUT_SAMPLES:
        raise RuntimeError(f"Baik phase cache is incomplete for {case_id}/{model}")
    output = {
        "phase": np.asarray([float(row["phase"]) for row in selected]),
        "CL": np.asarray([float(row["CL"]) for row in selected]),
        "CD": np.asarray([float(row["CD"]) for row in selected]),
    }
    for field in ("persistence", "ldvm_delta_CL", "ldvm_delta_CD"):
        if selected[0][field] != "":
            output[field] = np.asarray([float(row[field]) for row in selected])
    return output


def _baik_predictions(
    phase_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    rows = _read_csv(BAIK_PHASE)
    predictions: dict[str, dict[str, Any]] = {}
    diagnostics = {
        "disabled_max_abs": 0.0,
        "periodic_state_max_abs": 0.0,
        "source_clock_formula_max_abs": 0.0,
    }
    for case_id, case in BAIK_2012_CASES.items():
        old = _model_history(rows, case_id, "fluxv_old")
        baseline = _model_history(rows, case_id, "fluxv_v4b")
        movement, _ = build_baik_movement(
            case,
            settings=(2, 8, OUTPUT_SAMPLES, 3),
        )
        source_cycle = _cycle_range(movement, case.period_s)
        polar = movement_polar_residual(
            movement,
            source_cycle_step_range=source_cycle,
            period_s=case.period_s,
            freestream_m_s=case.freestream_m_s,
            rho_kg_m3=case.rho_kg_m3,
            aspect_ratio=case.geometric_aspect_ratio,
            output_samples=OUTPUT_SAMPLES,
            parameters=DEFAULT_POLAR_PARAMETERS,
        )
        owner = _source_owner(
            movement,
            period_s=case.period_s,
            freestream_m_s=case.freestream_m_s,
            rho_kg_m3=case.rho_kg_m3,
            aspect_ratio=case.geometric_aspect_ratio,
            chord_m=case.chord_m,
        )
        polar_cl = np.asarray(old["CL"]) + np.asarray(polar["delta_lift_n"]) / (
            0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
        )
        polar_cd = np.asarray(old["CD"]) + np.asarray(polar["delta_drag_n"]) / (
            0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
        )
        delta_cl = np.asarray(baseline["ldvm_delta_CL"])
        delta_cd = np.asarray(baseline["ldvm_delta_CD"])
        old_p = np.asarray(baseline["persistence"])
        new_p = np.asarray(owner["persistence"])
        transient_cl = np.asarray(old["CL"]) + delta_cl
        transient_cd = np.asarray(old["CD"]) + delta_cd
        candidate_cl = (1.0 - new_p) * transient_cl + new_p * polar_cl
        candidate_cd = (1.0 - new_p) * transient_cd + new_p * polar_cd
        predictions[case_id] = {
            "phase": np.asarray(old["phase"]),
            "baseline_CL": np.asarray(baseline["CL"]),
            "baseline_CD": np.asarray(baseline["CD"]),
            "proxy_CL": candidate_cl,
            "proxy_CD": candidate_cd,
        }
        _append_owner_state(
            state_rows,
            benchmark="baik2012",
            case_id=case_id,
            owner=owner,
        )
        for key in diagnostics:
            diagnostics[key] = max(diagnostics[key], owner[key])
        for index, phase in enumerate(np.asarray(old["phase"])):
            phase_rows.append(
                {
                    "benchmark": "baik2012",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CL": baseline["CL"][index],
                    "baseline_CD": baseline["CD"][index],
                    "source_clock_CL": candidate_cl[index],
                    "source_clock_CD": candidate_cd[index],
                    "old_persistence": old_p[index],
                    "source_clock_persistence": new_p[index],
                    "ldvm_delta_CL": delta_cl[index],
                    "ldvm_delta_CD": delta_cd[index],
                    "canonical_eligible": "false",
                }
            )
    return predictions, diagnostics


def _gate(
    gate_id: str, measured: float | bool, relation: str, threshold: float | bool
) -> dict[str, Any]:
    if relation == "<=":
        passed = float(measured) <= float(threshold)
    elif relation == ">=":
        passed = float(measured) >= float(threshold)
    elif relation == "==":
        passed = measured == threshold
    else:
        raise ValueError(f"unsupported relation {relation}")
    return {
        "gate_id": gate_id,
        "measured": measured,
        "relation": relation,
        "threshold": threshold,
        "numeric_pass": bool(passed),
        "canonical_eligible": "false",
        "promotion_pass": "false",
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    phase_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    # Predictions are completed before any scorer reads target observations.
    yang_predictions, yang_diag = _yang_predictions(phase_rows, state_rows)
    fig_predictions, fig_diag = _fig14_predictions(phase_rows, state_rows)
    baik_predictions, baik_diag = _baik_predictions(phase_rows, state_rows)
    condition_count = (
        len(yang_predictions) + len(fig_predictions) + len(baik_predictions)
    )
    if condition_count != 22 or len(phase_rows) != 22 * OUTPUT_SAMPLES:
        raise RuntimeError("v5d1 all-22 prediction coverage failed")

    metric_rows: list[dict[str, Any]] = []
    yang = _score_yang(yang_predictions, metric_rows)
    fig14 = _score_fig14(fig_predictions, metric_rows)
    baik = _score_baik(baik_predictions, metric_rows)
    for row in metric_rows:
        row["model"] = "fluxv_v5d1_source_clock"
    v5c0 = json.loads(V5C0_SUMMARY.read_text(encoding="utf-8"))[
        "corrected_v5c0_metrics"
    ]

    diagnostics = (yang_diag, fig_diag, baik_diag)
    mechanical = {
        "disabled_max_abs": max(row["disabled_max_abs"] for row in diagnostics),
        "periodic_state_max_abs": max(
            row["periodic_state_max_abs"] for row in diagnostics
        ),
        "source_clock_formula_max_abs": max(
            row["source_clock_formula_max_abs"] for row in diagnostics
        ),
        "yang_baseline_replay_max_abs_gf": yang_diag["baseline_replay_max_abs_gf"],
        "owner_min": min(float(row["source_clock_persistence"]) for row in phase_rows),
        "owner_max": max(float(row["source_clock_persistence"]) for row in phase_rows),
    }
    gates = [
        _gate("coverage_22_conditions", condition_count, "==", 22),
        _gate("phase_rows_2816", len(phase_rows), "==", 2816),
        _gate("disabled_identity", mechanical["disabled_max_abs"], "<=", 0.0),
        _gate(
            "source_clock_formula",
            mechanical["source_clock_formula_max_abs"],
            "<=",
            1.0e-15,
        ),
        _gate("periodic_state", mechanical["periodic_state_max_abs"], "<=", 1.0e-8),
        _gate("owner_lower_bound", mechanical["owner_min"], ">=", 0.0),
        _gate("owner_upper_bound", mechanical["owner_max"], "<=", 1.0),
        _gate(
            "yang_baseline_replay",
            mechanical["yang_baseline_replay_max_abs_gf"],
            "<=",
            1.0e-10,
        ),
        _gate("yang_lift_mae_gf", yang["lift_mae"], "<=", YANG_THRESHOLDS["lift"]),
        _gate("yang_drag_mae_gf", yang["drag_mae"], "<=", YANG_THRESHOLDS["drag"]),
        _gate(
            "fig14_all14_rmse",
            fig14["all14_rmse"],
            "<=",
            v5c0["all_14_markers"]["rmse"],
        ),
        _gate(
            "fig14_theta15_rmse", fig14["theta_15_rmse"], "<=", v5c0["theta_15"]["rmse"]
        ),
        _gate(
            "fig14_theta25_rmse", fig14["theta_25_rmse"], "<=", v5c0["theta_25"]["rmse"]
        ),
        _gate(
            "fig14_unique12_rmse",
            fig14["unique12_rmse"],
            "<=",
            v5c0["unique_12_conditions"]["rmse"],
        ),
        _gate(
            "baik_macro_CL_rmse", baik["macro"]["CL"], "<=", BAIK_MACRO_THRESHOLDS["CL"]
        ),
        _gate(
            "baik_macro_CD_rmse", baik["macro"]["CD"], "<=", BAIK_MACRO_THRESHOLDS["CD"]
        ),
    ]
    for (case_id, quantity), threshold in BAIK_THRESHOLDS.items():
        gates.append(
            _gate(
                f"baik_{case_id}_{quantity}_rmse",
                baik["rmse"][(case_id, quantity)],
                "<=",
                threshold,
            )
        )
    gates.append(_gate("canonical_strip_inputs_available", False, "==", True))

    mechanical_ids = {
        "coverage_22_conditions",
        "phase_rows_2816",
        "disabled_identity",
        "source_clock_formula",
        "periodic_state",
        "owner_lower_bound",
        "owner_upper_bound",
        "yang_baseline_replay",
    }
    if not all(
        row["numeric_pass"] for row in gates if row["gate_id"] in mechanical_ids
    ):
        failed = [
            row["gate_id"]
            for row in gates
            if row["gate_id"] in mechanical_ids and not row["numeric_pass"]
        ]
        raise RuntimeError(f"v5d1 mechanical gate failed: {failed}")
    paper_pass = {
        "yang2025": all(
            row["numeric_pass"] for row in gates if row["gate_id"].startswith("yang_")
        ),
        "izraelevitz2017_fig14": all(
            row["numeric_pass"] for row in gates if row["gate_id"].startswith("fig14_")
        ),
        "baik2012": all(
            row["numeric_pass"] for row in gates if row["gate_id"].startswith("baik_")
        ),
    }
    status = (
        "source_clock_numeric_pass_canonical_blocked"
        if all(paper_pass.values())
        else "stopped_source_clock_crosspaper_gate_failure"
    )

    condition_rows: list[dict[str, Any]] = []
    condition_rows.extend(
        {"benchmark": "yang2025", "case_id": key, **value}
        for key, value in yang_predictions.items()
    )
    condition_rows.extend(
        {"benchmark": "izraelevitz2017_fig14", "case_id": key, **value}
        for key, value in fig_predictions.items()
    )
    condition_rows.extend(
        {
            "benchmark": "baik2012",
            "case_id": case_id,
            "baseline_filtered_CL_rmse": baik["baseline_rmse"][(case_id, "CL")],
            "source_clock_filtered_CL_rmse": baik["rmse"][(case_id, "CL")],
            "baseline_filtered_CD_rmse": baik["baseline_rmse"][(case_id, "CD")],
            "source_clock_filtered_CD_rmse": baik["rmse"][(case_id, "CD")],
        }
        for case_id in BAIK_2012_CASES
    )
    for row in condition_rows:
        row["canonical_eligible"] = "false"

    phase_path = output / "phase_predictions.csv"
    state_path = output / "owner_state_histories.csv"
    condition_path = output / "condition_predictions.csv"
    metric_path = output / "case_metrics.csv"
    gate_path = output / "gate_results.csv"
    _write_csv(phase_path, phase_rows)
    _write_csv(state_path, state_rows)
    _write_csv(condition_path, condition_rows)
    _write_csv(metric_path, metric_rows)
    _write_csv(gate_path, gates)

    summary = {
        "run_id": output.name,
        "status": status,
        "promotion_status": "blocked_noncanonical_kinematic_strip_adapter",
        "canonical_eligible": False,
        "condition_count": condition_count,
        "phase_row_count": len(phase_rows),
        "owner_state_row_count": len(state_rows),
        "prediction_completed_before_scoring": True,
        "parameter_selection_data": [],
        "owner_warmup_total_cycles": OWNER_WARMUP_CYCLES,
        "owner_warmup_discarded_cycles": OWNER_WARMUP_CYCLES - 1,
        "owner_scored_cycles": 1,
        "owner_reference_fraction_chord": OWNER_REFERENCE_FRACTION_CHORD,
        "owner_parameters": DEFAULT_SOURCE_TIME_OWNER_PARAMETERS.manifest(),
        "mechanical": mechanical,
        "paper_pass": paper_pass,
        "yang": yang,
        "fig14": fig14,
        "baik": {
            "macro": baik["macro"],
            "rmse": {
                f"{key[0]}_{key[1]}": value for key, value in baik["rmse"].items()
            },
        },
        "limitations": [
            "The local clock uses kinematic 0.75c velocity and omits same-time-layer UVLM induction.",
            "This run changes only the causal owner clock; all force branches remain frozen v4b/v5c0 ledgers.",
            "All 22 cases are previously inspected development evidence, not held-out generalization.",
        ],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("fluxv_v5d_source_owner.py").resolve(),
        Path(__file__).with_name("uvlm_polar_correction.py").resolve(),
        Path(__file__).with_name("run_fluxv_v5c0_fig14.py").resolve(),
        Path(__file__).with_name("run_fluxv_v5c1_proxy.py").resolve(),
        Path(__file__).with_name("run_v4_crosspaper.py").resolve(),
        Path(__file__).with_name("causal_incidence_owner.py").resolve(),
        Path(__file__).with_name("ldvm_uvlm_correction.py").resolve(),
        REPO_ROOT / "platform/ldvm_fourier.py",
        REPO_ROOT / "platform/flap_ldvm.py",
        Path(__file__).with_name("baik2012.py").resolve(),
        Path(__file__).with_name("ptera_adapter.py").resolve(),
        Path(__file__).with_name("cases.py").resolve(),
        REPO_ROOT / "src/fluxvortex/solver.py",
        REPO_ROOT / "src/fluxvortex/particles.py",
        REPO_ROOT / "pyproject.toml",
        YANG_PHASE,
        YANG_V4,
        YANG_GT,
        V5C0_PHASE,
        V5C0_SUMMARY,
        FIG14_GT,
        BAIK_PHASE,
        BAIK_GT,
    )
    manifest = {
        "run_id": output.name,
        "argv": sys.argv,
        "condition_count": condition_count,
        "canonical_eligible": False,
        "parameter_selection_data": [],
        "source_clock": DEFAULT_SOURCE_TIME_OWNER_PARAMETERS.manifest(),
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "pterasoftware": _package_version("pterasoftware"),
            "fluxvortex": _package_version("fluxvortex"),
        },
        "result_hashes": {
            path.name: _sha256(path)
            for path in (
                phase_path,
                state_path,
                condition_path,
                metric_path,
                gate_path,
                summary_path,
            )
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
    args = parser.parse_args()
    result = run(args.output_dir.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
