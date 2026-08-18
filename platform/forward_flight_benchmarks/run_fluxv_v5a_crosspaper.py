"""Run the FluxV v5a cache-reuse development ledger across three papers.

This runner deliberately distinguishes a cache-reuse mechanism test from a
canonical v5a result.  The frozen Yang, Figure-14, and Baik phase histories do
not contain UVLM-induced strip velocities, so the currently executable path is
labelled ``kinematic_proxy`` and ``projected_integrated_proxy``.  Numeric gates
are still useful during development, but canonical promotion fails closed.

The retained UVLM histories, experimental data, filters, signs, and v4b
comparators are frozen.  No observation is passed to the v5a force ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .baik2012 import BAIK_2012_CASES, baik_kinematics, sharp_fourier_lowpass
from .cases import IZRAELEVITZ_2017_FIG14_SCHERER, YANG_2025
from .ptera_adapter import build_izraelevitz_scherer_movement
from .run_v4_crosspaper import (
    _fig14_phase_diagnostic,
    _yang_ldvm_phase_and_ownership,
)
from .uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    movement_polar_residual,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = REPO_ROOT / "docs/forward_flight_large_pitch/reproductions"
V3_ROOT = REPRO_ROOT / "unified_fluxv_upgrade_20260812"
V4_ROOT = REPRO_ROOT / "unified_fluxv_v4_ldvm_stevens_20260812"
BAIK_ROOT = REPRO_ROOT / "baik2012_w1_w4"
DOC_ROOT = REPRO_ROOT / "fluxv_v5_nextgen_20260814"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5a_cache_smoke"

YANG_MEANS = (
    V3_ROOT / "runs/20260812_periodic_v2_ullt_full/yang2025_mean_characteristics.csv"
)
YANG_PHASE = (
    V3_ROOT / "runs/20260812_periodic_v2_ullt_full/yang2025_phase_histories.csv"
)
FIG14_PHASE = V4_ROOT / "source_data/izraelevitz2017_fig14_local_phase_cache.csv"
FIG14_GT = V3_ROOT / "source_data/izraelevitz2017_fig14_digitized.csv"
V4_YANG = (
    V4_ROOT
    / "runs/20260812_fluxv_v4b_crosspaper_full/yang2025_v4_mean_characteristics.csv"
)
V4_FIG14 = (
    V4_ROOT
    / "runs/20260812_fluxv_v4b_crosspaper_full/izraelevitz2017_fig14_v4_mean_thrust.csv"
)
BAIK_PHASE = (
    BAIK_ROOT
    / "runs/20260813_baik2012_w1_w4_full_reproducible/model_phase_histories.csv"
)
BAIK_GT = BAIK_ROOT / "source_data/baik2012_w1_w4_corrected_total_cl_cd.csv"

GF_TO_N = 0.00980665
OUTPUT_SAMPLES = 128


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    truth = np.asarray(tuple(observed), dtype=float)
    estimate = np.asarray(tuple(predicted), dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("metric vectors must be aligned and non-empty")
    error = estimate - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _periodic_resample(
    values: np.ndarray, output_samples: int = OUTPUT_SAMPLES
) -> np.ndarray:
    history = np.asarray(values, dtype=float)
    if history.ndim != 1 or history.size < 2:
        raise ValueError("periodic history must be one dimensional")
    source = np.arange(history.size, dtype=float) / history.size
    target = np.arange(output_samples, dtype=float) / output_samples
    return np.interp(target, source, history, period=1.0)


def _phase_history(
    rows: list[dict[str, str]],
    model: str,
    *,
    selectors: dict[str, float | str],
) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["model"] == model]
    for field, value in selectors.items():
        if isinstance(value, str):
            selected = [row for row in selected if row[field] == value]
        else:
            selected = [
                row for row in selected if np.isclose(float(row[field]), float(value))
            ]
    selected.sort(key=lambda row: float(row["phase"]))
    if len(selected) != OUTPUT_SAMPLES:
        raise ValueError(
            f"expected {OUTPUT_SAMPLES} samples for {model}/{selectors}, "
            f"found {len(selected)}"
        )
    phase = np.asarray([float(row["phase"]) for row in selected])
    if not np.allclose(phase, np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
        raise ValueError(f"noncanonical phase grid for {model}/{selectors}")
    output = {"phase": phase}
    for field in ("CL", "CD", "CT", "lift_n", "drag_n"):
        if field in selected[0] and selected[0][field] != "":
            output[field] = np.asarray([float(row[field]) for row in selected])
    return output


def _core_array(result: dict[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in result:
            return np.asarray(result[name], dtype=float)
    raise KeyError(f"v5a core result has none of {names}")


def _apply_periodic_ledger(
    baseline: dict[str, np.ndarray],
    equilibrium: dict[str, np.ndarray],
    ldvm_delta: dict[str, np.ndarray],
    delta_chi: np.ndarray | float,
    *,
    lambda_tau: float,
    mode: str,
    warmup_cycles: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Apply the core compatibility adapter after periodic state warm-up."""

    from .fluxv_v5a import apply_fluxv_v5a_ledger

    if warmup_cycles < 2:
        raise ValueError("at least two warm-up cycles are required")
    n = np.asarray(baseline["CL"]).size
    if n != OUTPUT_SAMPLES:
        raise ValueError("the cache-reuse adapter expects 128 output samples")

    tiled_base = {
        key: np.tile(np.asarray(value, dtype=float), warmup_cycles)
        for key, value in baseline.items()
    }
    tiled_eq = {
        key: np.tile(np.asarray(value, dtype=float), warmup_cycles)
        for key, value in equilibrium.items()
    }
    tiled_ldvm = {
        key: np.tile(np.asarray(value, dtype=float), warmup_cycles)
        for key, value in ldvm_delta.items()
    }
    dchi = np.asarray(delta_chi, dtype=float)
    tiled_dchi: np.ndarray | float
    if dchi.ndim == 0:
        tiled_dchi = float(dchi)
    else:
        if dchi.shape != (n,):
            raise ValueError("delta_chi history must align with a single cycle")
        tiled_dchi = np.tile(dchi, warmup_cycles)
    raw = apply_fluxv_v5a_ledger(
        tiled_base,
        tiled_eq,
        tiled_ldvm,
        delta_chi=tiled_dchi,
        lambda_tau=lambda_tau,
        mode=mode,
    )
    selected = slice((warmup_cycles - 1) * n, warmup_cycles * n)
    previous = slice((warmup_cycles - 2) * n, (warmup_cycles - 1) * n)
    aliases = {
        "CL": ("CL", "corrected_CL"),
        "CD": ("CD", "corrected_CD"),
        "equilibrium_CL": ("equilibrium_CL", "eq_CL"),
        "equilibrium_CD": ("equilibrium_CD", "eq_CD"),
        "raw_ldvm_CL": ("raw_ldvm_CL", "ldvm_delta_CL"),
        "raw_ldvm_CD": ("raw_ldvm_CD", "ldvm_delta_CD"),
        "state_CL": ("state_CL", "lowpass_state_CL", "m_CL"),
        "state_CD": ("state_CD", "lowpass_state_CD", "m_CD"),
        "transient_CL": ("transient_CL", "highpass_CL"),
        "transient_CD": ("transient_CD", "highpass_CD"),
        "ledger_residual_CL": ("ledger_residual_CL",),
        "ledger_residual_CD": ("ledger_residual_CD",),
    }
    output: dict[str, np.ndarray] = {}
    full: dict[str, np.ndarray] = {}
    for canonical, candidates in aliases.items():
        try:
            value = _core_array(raw, *candidates)
        except KeyError:
            if canonical.startswith("ledger_residual"):
                value = np.zeros(warmup_cycles * n)
            else:
                raise
        if value.shape != (warmup_cycles * n,):
            raise ValueError(
                f"unexpected v5a core shape for {canonical}: {value.shape}"
            )
        full[canonical] = value
        output[canonical] = value[selected]
    state_cycle_error = max(
        float(np.max(np.abs(full["state_CL"][selected] - full["state_CL"][previous]))),
        float(np.max(np.abs(full["state_CD"][selected] - full["state_CD"][previous]))),
    )
    ledger_error = max(
        float(np.max(np.abs(output["ledger_residual_CL"]))),
        float(np.max(np.abs(output["ledger_residual_CD"]))),
    )
    return output, {
        "state_cycle_max_abs": state_cycle_error,
        "ledger_max_abs": ledger_error,
    }


def _add_phase_rows(
    destination: list[dict[str, Any]],
    *,
    benchmark: str,
    case_id: str,
    result: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    provider: str,
    incidence_source: str,
) -> None:
    for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
        destination.append(
            {
                "benchmark": benchmark,
                "case_id": case_id,
                "model": "fluxv_v5a_cache_adapter",
                "phase": float(phase),
                "baseline_CL": float(baseline["CL"][index]),
                "baseline_CD": float(baseline["CD"][index]),
                "equilibrium_CL": float(result["equilibrium_CL"][index]),
                "equilibrium_CD": float(result["equilibrium_CD"][index]),
                "raw_ldvm_CL": float(result["raw_ldvm_CL"][index]),
                "raw_ldvm_CD": float(result["raw_ldvm_CD"][index]),
                "state_CL": float(result["state_CL"][index]),
                "state_CD": float(result["state_CD"][index]),
                "transient_CL": float(result["transient_CL"][index]),
                "transient_CD": float(result["transient_CD"][index]),
                "prediction_CL": float(result["CL"][index]),
                "prediction_CD": float(result["CD"][index]),
                "ledger_residual_CL": float(result["ledger_residual_CL"][index]),
                "ledger_residual_CD": float(result["ledger_residual_CD"][index]),
                "incidence_source": incidence_source,
                "aggregation_scope": "projected_integrated_proxy",
                "equilibrium_provider": provider,
            }
        )


def _gate(
    gate_id: str,
    value: float,
    threshold: float,
    relation: str,
    *,
    canonical_eligible: bool,
    note: str = "",
) -> dict[str, Any]:
    if relation == "<=":
        passed = value <= threshold
    elif relation == "<":
        passed = value < threshold
    elif relation == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"unsupported gate relation {relation}")
    return {
        "gate_id": gate_id,
        "measured": float(value),
        "relation": relation,
        "threshold": float(threshold),
        "numeric_pass": bool(passed),
        "canonical_eligible": canonical_eligible,
        "promotion_pass": bool(passed and canonical_eligible),
        "note": note,
    }


def _run_yang(
    *,
    steps_per_cycle: int,
    strip_count: int,
    lambda_tau: float,
    mode: str,
    warmup_cycles: int,
    incidence_source: str,
    phase_rows_out: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    phase_rows = _read_csv(YANG_PHASE)
    mean_rows = _read_csv(YANG_MEANS)
    v4_rows = {float(row["aoa_deg"]): row for row in _read_csv(V4_YANG)}
    truth = {
        float(row["aoa_deg"]): row
        for row in mean_rows
        if row["model"] == "wind_tunnel_test"
    }
    q_area_per_gf = (
        0.5
        * YANG_2025.rho_kg_m3
        * YANG_2025.freestream_m_s**2
        * YANG_2025.area_m2
        / GF_TO_N
    )
    condition_rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for aoa in sorted(truth):
        old = _phase_history(phase_rows, "fluxv_uvpm", selectors={"aoa_deg": aoa})
        polar = _phase_history(
            phase_rows, "fluxv_periodic_v1", selectors={"aoa_deg": aoa}
        )
        diagnostic = _yang_ldvm_phase_and_ownership(
            aoa,
            steps_per_cycle=steps_per_cycle,
            strip_count=strip_count,
        )
        baseline = {"CL": old["CL"], "CD": old["CD"]}
        equilibrium = {
            "CL": polar["CL"] - old["CL"],
            "CD": polar["CD"] - old["CD"],
        }
        ldvm = {
            "CL": _periodic_resample(np.asarray(diagnostic["ldvm_delta_CL"])),
            "CD": _periodic_resample(np.asarray(diagnostic["ldvm_delta_CD"])),
        }
        delta_chi = float(
            YANG_2025.freestream_m_s
            * YANG_2025.period_s
            / YANG_2025.chord_m
            / OUTPUT_SAMPLES
        )
        result, diagnostics = _apply_periodic_ledger(
            baseline,
            equilibrium,
            ldvm,
            delta_chi,
            lambda_tau=lambda_tau,
            mode=mode,
            warmup_cycles=warmup_cycles,
        )
        _add_phase_rows(
            phase_rows_out,
            benchmark="yang2025",
            case_id=f"aoa_{aoa:g}",
            result=result,
            baseline=baseline,
            provider="frozen_fluxv_periodic_v1_minus_uvlm_kinematic_quarter_chord",
            incidence_source=incidence_source,
        )
        prediction_l = float(np.mean(result["CL"]) * q_area_per_gf)
        prediction_d = float(np.mean(result["CD"]) * q_area_per_gf)
        truth_l = float(truth[aoa]["mean_lift_gf"])
        truth_d = float(truth[aoa]["mean_drag_gf"])
        condition_rows.append(
            {
                "benchmark": "yang2025",
                "case_id": f"aoa_{aoa:g}",
                "aoa_deg": aoa,
                "truth_lift_gf": truth_l,
                "truth_drag_gf": truth_d,
                "v4b_lift_gf": float(v4_rows[aoa]["v4_lift_gf"]),
                "v4b_drag_gf": float(v4_rows[aoa]["v4_drag_gf"]),
                "v5a_lift_gf": prediction_l,
                "v5a_drag_gf": prediction_d,
            }
        )
        convergence.append({"case_id": f"aoa_{aoa:g}", **diagnostics})
    observed_l = np.asarray([row["truth_lift_gf"] for row in condition_rows])
    observed_d = np.asarray([row["truth_drag_gf"] for row in condition_rows])
    v5_l = np.asarray([row["v5a_lift_gf"] for row in condition_rows])
    v5_d = np.asarray([row["v5a_drag_gf"] for row in condition_rows])
    v4_l = np.asarray([row["v4b_lift_gf"] for row in condition_rows])
    v4_d = np.asarray([row["v4b_drag_gf"] for row in condition_rows])
    metric_rows = [
        {
            "benchmark": "yang2025",
            "case_id": "six_aoa",
            "model": "fluxv_v5a",
            "quantity": "lift_gf",
            "view": "cycle_mean",
            "observation_count": 6,
            **_metrics(observed_l, v5_l),
        },
        {
            "benchmark": "yang2025",
            "case_id": "six_aoa",
            "model": "fluxv_v5a",
            "quantity": "drag_gf",
            "view": "cycle_mean",
            "observation_count": 6,
            **_metrics(observed_d, v5_d),
        },
        {
            "benchmark": "yang2025",
            "case_id": "six_aoa",
            "model": "fluxv_v4b",
            "quantity": "lift_gf",
            "view": "cycle_mean",
            "observation_count": 6,
            **_metrics(observed_l, v4_l),
        },
        {
            "benchmark": "yang2025",
            "case_id": "six_aoa",
            "model": "fluxv_v4b",
            "quantity": "drag_gf",
            "view": "cycle_mean",
            "observation_count": 6,
            **_metrics(observed_d, v4_d),
        },
    ]
    return (
        condition_rows,
        metric_rows,
        {
            "lift_mae": _metrics(observed_l, v5_l)["mae"],
            "drag_mae": _metrics(observed_d, v5_d)["mae"],
            "max_point_regression": float(
                max(
                    np.max(np.abs(v5_l - observed_l) - np.abs(v4_l - observed_l)),
                    np.max(np.abs(v5_d - observed_d) - np.abs(v4_d - observed_d)),
                )
            ),
            "target_improvement_count": float(
                sum(
                    bool(value)
                    for value in (
                        abs(v5_l[1] - observed_l[1]) < abs(v4_l[1] - observed_l[1]),
                        abs(v5_l[2] - observed_l[2]) < abs(v4_l[2] - observed_l[2]),
                        abs(v5_d[3] - observed_d[3]) < abs(v4_d[3] - observed_d[3]),
                    )
                )
            ),
            "state_cycle_max_abs": max(
                row["state_cycle_max_abs"] for row in convergence
            ),
            "ledger_max_abs": max(row["ledger_max_abs"] for row in convergence),
        },
    )


def _fig14_three_quarter_equilibrium(
    theta: float,
    psi: float,
    *,
    quality: str,
) -> dict[str, np.ndarray]:
    movement, metadata = build_izraelevitz_scherer_movement(theta, psi, quality)
    steps = int(round(IZRAELEVITZ_2017_FIG14_SCHERER.period_s / movement.delta_time))
    cycles = int(metadata["cycles"])
    cycle_range = ((cycles - 1) * steps, cycles * steps - 1)
    parameters = replace(
        DEFAULT_POLAR_PARAMETERS,
        section_velocity_reference_fraction_chord=0.75,
    )
    polar = movement_polar_residual(
        movement,
        source_cycle_step_range=cycle_range,
        period_s=IZRAELEVITZ_2017_FIG14_SCHERER.period_s,
        freestream_m_s=IZRAELEVITZ_2017_FIG14_SCHERER.freestream_m_s,
        rho_kg_m3=IZRAELEVITZ_2017_FIG14_SCHERER.rho_kg_m3,
        aspect_ratio=IZRAELEVITZ_2017_FIG14_SCHERER.aspect_ratio,
        output_samples=OUTPUT_SAMPLES,
        parameters=parameters,
    )
    q_area = (
        0.5
        * IZRAELEVITZ_2017_FIG14_SCHERER.rho_kg_m3
        * IZRAELEVITZ_2017_FIG14_SCHERER.freestream_m_s**2
        * IZRAELEVITZ_2017_FIG14_SCHERER.area_m2
    )
    return {
        "CL": np.asarray(polar["delta_lift_n"], dtype=float) / q_area,
        "CD": np.asarray(polar["delta_drag_n"], dtype=float) / q_area,
    }


def _run_fig14(
    *,
    quality: str,
    steps_per_cycle: int,
    lambda_tau: float,
    mode: str,
    warmup_cycles: int,
    incidence_source: str,
    fig14_equilibrium_source: str,
    phase_rows_out: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    phase_rows = _read_csv(FIG14_PHASE)
    v4_lookup = {
        (float(row["theta_max_deg"]), float(row["phase_offset_deg"])): float(
            row["v4_CT"]
        )
        for row in _read_csv(V4_FIG14)
    }
    conditions = sorted(v4_lookup)
    predictions: dict[tuple[float, float], float] = {}
    condition_rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for theta, psi in conditions:
        selectors = {"theta_max_deg": theta, "phase_offset_deg": psi}
        old = _phase_history(phase_rows, "fluxv_uvpm", selectors=selectors)
        baseline = {"CL": old["CL"], "CD": old["CD"]}
        if fig14_equilibrium_source == "three-quarter-recomputed":
            equilibrium = _fig14_three_quarter_equilibrium(theta, psi, quality=quality)
            provider = "recomputed_kinematic_polar_residual_at_0.75c_cd0_excluded"
        else:
            polar = _phase_history(phase_rows, "fluxv_periodic_v1", selectors=selectors)
            equilibrium = {
                "CL": polar["CL"] - old["CL"],
                "CD": polar["CD"] - old["CD"],
            }
            provider = "legacy_quarter_chord_cache_cd0_cancels_noncanonical"
        diagnostic = _fig14_phase_diagnostic(
            theta, psi, steps_per_cycle=steps_per_cycle
        )
        ldvm = {
            # Figure 14 supplies no lift observations.  The old diagnostic only
            # froze the projected axial discrepancy; do not invent a lift term.
            "CL": np.zeros(OUTPUT_SAMPLES),
            "CD": -_periodic_resample(np.asarray(diagnostic["ldvm_delta_CT"])),
        }
        delta_chi = float(
            IZRAELEVITZ_2017_FIG14_SCHERER.freestream_m_s
            * IZRAELEVITZ_2017_FIG14_SCHERER.period_s
            / IZRAELEVITZ_2017_FIG14_SCHERER.chord_m
            / OUTPUT_SAMPLES
        )
        result, diagnostics = _apply_periodic_ledger(
            baseline,
            equilibrium,
            ldvm,
            delta_chi,
            lambda_tau=lambda_tau,
            mode=mode,
            warmup_cycles=warmup_cycles,
        )
        _add_phase_rows(
            phase_rows_out,
            benchmark="izraelevitz2017_fig14",
            case_id=f"theta_{theta:g}_psi_{psi:g}",
            result=result,
            baseline=baseline,
            provider=provider,
            incidence_source=incidence_source,
        )
        prediction = -float(np.mean(result["CD"]))
        predictions[(theta, psi)] = prediction
        condition_rows.append(
            {
                "benchmark": "izraelevitz2017_fig14",
                "case_id": f"theta_{theta:g}_psi_{psi:g}",
                "theta_max_deg": theta,
                "phase_offset_deg": psi,
                "v4b_CT": v4_lookup[(theta, psi)],
                "v5a_CT": prediction,
                "equilibrium_provider": provider,
            }
        )
        convergence.append({"case_id": f"theta_{theta:g}_psi_{psi:g}", **diagnostics})
    observations = [
        row for row in _read_csv(FIG14_GT) if row["series"] == "scherer_1968_experiment"
    ]
    metric_rows: list[dict[str, Any]] = []
    summary: dict[str, float] = {}
    for theta_group in (None, 15.0, 25.0):
        selected = (
            observations
            if theta_group is None
            else [
                row
                for row in observations
                if np.isclose(float(row["theta_max_deg"]), theta_group)
            ]
        )
        observed = np.asarray([float(row["ct"]) for row in selected])
        v5 = np.asarray(
            [
                predictions[
                    (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
                ]
                for row in selected
            ]
        )
        v4 = np.asarray(
            [
                v4_lookup[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))]
                for row in selected
            ]
        )
        group = "all14" if theta_group is None else f"theta_{theta_group:g}"
        metric_rows.extend(
            [
                {
                    "benchmark": "izraelevitz2017_fig14",
                    "case_id": group,
                    "model": "fluxv_v5a",
                    "quantity": "CT",
                    "view": "cycle_mean",
                    "observation_count": len(selected),
                    **_metrics(observed, v5),
                },
                {
                    "benchmark": "izraelevitz2017_fig14",
                    "case_id": group,
                    "model": "fluxv_v4b",
                    "quantity": "CT",
                    "view": "cycle_mean",
                    "observation_count": len(selected),
                    **_metrics(observed, v4),
                },
            ]
        )
        summary[f"rmse_{group}"] = _metrics(observed, v5)["rmse"]
        if theta_group is None:
            summary["max_abs_error"] = _metrics(observed, v5)["max_abs_error"]
            summary["max_point_regression"] = float(
                np.max(np.abs(v5 - observed) - np.abs(v4 - observed))
            )
    unique_observed: list[float] = []
    unique_predicted: list[float] = []
    for condition in conditions:
        same = [
            float(row["ct"])
            for row in observations
            if np.isclose(float(row["theta_max_deg"]), condition[0])
            and np.isclose(float(row["phase_offset_deg"]), condition[1])
        ]
        unique_observed.append(float(np.mean(same)))
        unique_predicted.append(predictions[condition])
    unique_metrics = _metrics(unique_observed, unique_predicted)
    summary["rmse_unique12"] = unique_metrics["rmse"]
    metric_rows.append(
        {
            "benchmark": "izraelevitz2017_fig14",
            "case_id": "unique12",
            "model": "fluxv_v5a",
            "quantity": "CT",
            "view": "condition_mean",
            "observation_count": 12,
            **unique_metrics,
        }
    )
    summary["state_cycle_max_abs"] = max(
        row["state_cycle_max_abs"] for row in convergence
    )
    summary["ledger_max_abs"] = max(row["ledger_max_abs"] for row in convergence)
    return condition_rows, metric_rows, summary


def _baik_experiment(case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in _read_csv(BAIK_GT) if row["case"] == case_id]
    phase = np.asarray([float(row["phase_t_over_T"]) for row in selected])
    cl = np.asarray([float(row["cl"]) for row in selected])
    cd = np.asarray([float(row["cd"]) for row in selected])
    if phase.size == 401 and np.isclose(phase[-1], 1.0):
        phase, cl, cd = phase[:-1], cl[:-1], cd[:-1]
    if phase.size != 400:
        raise ValueError(f"unexpected Baik GT count for {case_id}")
    return phase, cl, cd


def _run_baik(
    *,
    lambda_tau: float,
    mode: str,
    warmup_cycles: int,
    incidence_source: str,
    phase_rows_out: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    phase_rows = _read_csv(BAIK_PHASE)
    condition_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    quadrant_regressions: list[float] = []
    target_improvements: list[bool] = []
    convergence: list[dict[str, Any]] = []
    v5_rmse: dict[tuple[str, str], float] = {}
    v4_rmse: dict[tuple[str, str], float] = {}
    for case_id, case in BAIK_2012_CASES.items():
        old = _phase_history(phase_rows, "fluxv_old", selectors={"case_id": case_id})
        v4 = _phase_history(phase_rows, "fluxv_v4b", selectors={"case_id": case_id})
        v4_selected = [
            row
            for row in phase_rows
            if row["model"] == "fluxv_v4b" and row["case_id"] == case_id
        ]
        v4_selected.sort(key=lambda row: float(row["phase"]))
        persistence = np.asarray([float(row["persistence"]) for row in v4_selected])
        raw_cl = np.asarray([float(row["ldvm_delta_CL"]) for row in v4_selected])
        raw_cd = np.asarray([float(row["ldvm_delta_CD"]) for row in v4_selected])
        if np.min(persistence) <= 1.0e-8:
            raise ValueError(f"cannot recover Baik equilibrium provider for {case_id}")
        baseline = {"CL": old["CL"], "CD": old["CD"]}
        polar_cl = (v4["CL"] - (1.0 - persistence) * (old["CL"] + raw_cl)) / persistence
        polar_cd = (v4["CD"] - (1.0 - persistence) * (old["CD"] + raw_cd)) / persistence
        equilibrium = {"CL": polar_cl - old["CL"], "CD": polar_cd - old["CD"]}
        ldvm = {"CL": raw_cl, "CD": raw_cd}
        phase = np.arange(OUTPUT_SAMPLES, dtype=float) / OUTPUT_SAMPLES
        heave_rate = np.asarray(baik_kinematics(phase, case)["heave_rate_over_u"])
        speed_ratio = np.sqrt(1.0 + heave_rate**2)
        delta_chi = (
            case.freestream_m_s * case.period_s / case.chord_m / OUTPUT_SAMPLES
        ) * speed_ratio
        result, diagnostics = _apply_periodic_ledger(
            baseline,
            equilibrium,
            ldvm,
            delta_chi,
            lambda_tau=lambda_tau,
            mode=mode,
            warmup_cycles=warmup_cycles,
        )
        _add_phase_rows(
            phase_rows_out,
            benchmark="baik2012",
            case_id=case_id,
            result=result,
            baseline=baseline,
            provider="recovered_exactly_from_frozen_v4b_positive_persistence_ledger",
            incidence_source=incidence_source,
        )
        experiment_phase, experiment_cl, experiment_cd = _baik_experiment(case_id)
        maximum_harmonic = case.experimental_filter_harmonic
        for view, v5_cl, v5_cd, v4_cl, v4_cd in (
            ("raw", result["CL"], result["CD"], v4["CL"], v4["CD"]),
            (
                "filtered_1hz",
                sharp_fourier_lowpass(result["CL"], maximum_harmonic=maximum_harmonic),
                sharp_fourier_lowpass(result["CD"], maximum_harmonic=maximum_harmonic),
                sharp_fourier_lowpass(v4["CL"], maximum_harmonic=maximum_harmonic),
                sharp_fourier_lowpass(v4["CD"], maximum_harmonic=maximum_harmonic),
            ),
        ):
            for quantity, observed, v5_history, v4_history in (
                ("CL", experiment_cl, v5_cl, v4_cl),
                ("CD", experiment_cd, v5_cd, v4_cd),
            ):
                v5_prediction = np.interp(
                    experiment_phase, phase, v5_history, period=1.0
                )
                v4_prediction = np.interp(
                    experiment_phase, phase, v4_history, period=1.0
                )
                v5_metric = _metrics(observed, v5_prediction)
                v4_metric = _metrics(observed, v4_prediction)
                metric_rows.extend(
                    [
                        {
                            "benchmark": "baik2012",
                            "case_id": case_id,
                            "model": "fluxv_v5a",
                            "quantity": quantity,
                            "view": view,
                            "observation_count": 400,
                            **v5_metric,
                        },
                        {
                            "benchmark": "baik2012",
                            "case_id": case_id,
                            "model": "fluxv_v4b",
                            "quantity": quantity,
                            "view": view,
                            "observation_count": 400,
                            **v4_metric,
                        },
                    ]
                )
                if view == "filtered_1hz":
                    v5_rmse[(case_id, quantity)] = v5_metric["rmse"]
                    v4_rmse[(case_id, quantity)] = v4_metric["rmse"]
                    for quadrant in range(4):
                        selected = slice(100 * quadrant, 100 * (quadrant + 1))
                        v5_q = _metrics(observed[selected], v5_prediction[selected])[
                            "rmse"
                        ]
                        v4_q = _metrics(observed[selected], v4_prediction[selected])[
                            "rmse"
                        ]
                        quadrant_regressions.append((v5_q - v4_q) / max(v4_q, 1.0e-12))
                        if quantity == "CL" and (
                            (case_id == "W2" and quadrant == 0)
                            or (case_id == "W3" and quadrant in (0, 1))
                        ):
                            target_improvements.append(v5_q < v4_q)
        condition_rows.append(
            {
                "benchmark": "baik2012",
                "case_id": case_id,
                "filter_maximum_harmonic": maximum_harmonic,
                "equilibrium_provider": "derived_from_v4b_ledger",
                "minimum_frozen_persistence": float(np.min(persistence)),
            }
        )
        convergence.append({"case_id": case_id, **diagnostics})
    macro_v5_cl = float(np.mean([v5_rmse[(case, "CL")] for case in BAIK_2012_CASES]))
    macro_v5_cd = float(np.mean([v5_rmse[(case, "CD")] for case in BAIK_2012_CASES]))
    macro_v4_cl = float(np.mean([v4_rmse[(case, "CL")] for case in BAIK_2012_CASES]))
    macro_v4_cd = float(np.mean([v4_rmse[(case, "CD")] for case in BAIK_2012_CASES]))
    return (
        condition_rows,
        metric_rows,
        {
            "macro_rmse_CL": macro_v5_cl,
            "macro_rmse_CD": macro_v5_cd,
            "macro_v4b_rmse_CL": macro_v4_cl,
            "macro_v4b_rmse_CD": macro_v4_cd,
            "nonregressive_channels": float(
                sum(v5_rmse[key] <= v4_rmse[key] for key in v5_rmse)
            ),
            "target_quadrant_improvements": float(sum(target_improvements)),
            "max_quadrant_relative_regression": float(max(quadrant_regressions)),
            "state_cycle_max_abs": max(
                row["state_cycle_max_abs"] for row in convergence
            ),
            "ledger_max_abs": max(row["ledger_max_abs"] for row in convergence),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--incidence-source",
        choices=("kinematic-proxy", "uvlm-induced"),
        default="kinematic-proxy",
    )
    parser.add_argument(
        "--fig14-equilibrium-source",
        choices=("three-quarter-recomputed", "legacy-quarter-cache"),
        default="three-quarter-recomputed",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "equilibrium_only", "transient_only"),
        default="full",
    )
    parser.add_argument("--lambda-tau", type=float, default=1.0)
    parser.add_argument("--warmup-cycles", type=int, default=20)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()

    if args.lambda_tau <= 0.0 or not np.isfinite(args.lambda_tau):
        parser.error("--lambda-tau must be finite and positive")
    if args.incidence_source == "uvlm-induced":
        parser.error(
            "the frozen caches do not contain UVLM-induced strip velocities; "
            "export and wire that history before requesting this evidence level"
        )
    canonical_eligible = False
    if args.require_canonical:
        parser.error(
            "canonical v5a is blocked: cache adapter uses kinematic incidence and "
            "an integrated projected ledger"
        )

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.allow_existing_output:
        parser.error("output directory exists and is non-empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    # The material-LEV solve can become singular on the 64-step Yang grid.
    # Keep smoke spatially cheap but retain the established 128-step lower
    # bound for this coupled kinematic/LDVM diagnostic.
    steps_per_cycle, strips = (128, 4) if args.quality == "smoke" else (256, 12)

    phase_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    yang_conditions, yang_metrics, yang = _run_yang(
        steps_per_cycle=steps_per_cycle,
        strip_count=strips,
        lambda_tau=args.lambda_tau,
        mode=args.mode,
        warmup_cycles=args.warmup_cycles,
        incidence_source=args.incidence_source,
        phase_rows_out=phase_rows,
    )
    fig_conditions, fig_metrics, fig14 = _run_fig14(
        quality=args.quality,
        steps_per_cycle=steps_per_cycle,
        lambda_tau=args.lambda_tau,
        mode=args.mode,
        warmup_cycles=args.warmup_cycles,
        incidence_source=args.incidence_source,
        fig14_equilibrium_source=args.fig14_equilibrium_source,
        phase_rows_out=phase_rows,
    )
    baik_conditions, baik_metrics, baik = _run_baik(
        lambda_tau=args.lambda_tau,
        mode=args.mode,
        warmup_cycles=args.warmup_cycles,
        incidence_source=args.incidence_source,
        phase_rows_out=phase_rows,
    )
    condition_rows.extend(yang_conditions + fig_conditions + baik_conditions)
    metric_rows.extend(yang_metrics + fig_metrics + baik_metrics)

    gates = [
        _gate(
            "yang_lift_mae_gf",
            yang["lift_mae"],
            4.327,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "yang_drag_mae_gf",
            yang["drag_mae"],
            2.512,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "yang_max_point_regression_gf",
            yang["max_point_regression"],
            0.4,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "yang_three_target_improvements",
            yang["target_improvement_count"],
            3.0,
            ">=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "fig14_all14_rmse",
            fig14["rmse_all14"],
            0.0230,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "fig14_theta15_rmse",
            fig14["rmse_theta_15"],
            0.02268,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "fig14_theta25_rmse",
            fig14["rmse_theta_25"],
            0.02284,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "fig14_unique12_rmse",
            fig14["rmse_unique12"],
            0.02751,
            "<",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "fig14_max_abs_error",
            fig14["max_abs_error"],
            0.050,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "fig14_max_point_regression",
            fig14["max_point_regression"],
            0.0112,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "baik_filtered_macro_CL_rmse",
            baik["macro_rmse_CL"],
            0.95 * baik["macro_v4b_rmse_CL"],
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "baik_filtered_macro_CD_rmse",
            baik["macro_rmse_CD"],
            0.95 * baik["macro_v4b_rmse_CD"],
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "baik_eight_channels_nonregressive",
            baik["nonregressive_channels"],
            8.0,
            ">=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "baik_three_target_quadrants_improve",
            baik["target_quadrant_improvements"],
            3.0,
            ">=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "baik_max_quadrant_regression",
            baik["max_quadrant_relative_regression"],
            0.05,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "periodic_state_convergence",
            max(
                yang["state_cycle_max_abs"],
                fig14["state_cycle_max_abs"],
                baik["state_cycle_max_abs"],
            ),
            1.0e-4,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
        _gate(
            "ledger_closure",
            max(
                yang["ledger_max_abs"], fig14["ledger_max_abs"], baik["ledger_max_abs"]
            ),
            1.0e-12,
            "<=",
            canonical_eligible=canonical_eligible,
        ),
    ]
    gates.append(
        {
            "gate_id": "canonical_incidence_and_spatial_path",
            "measured": args.incidence_source,
            "relation": "==",
            "threshold": "uvlm-induced + strip-local CN/CS",
            "numeric_pass": False,
            "canonical_eligible": False,
            "promotion_pass": False,
            "note": "cache adapter is kinematic_proxy + projected_integrated_proxy",
        }
    )

    phase_path = output / "phase_histories.csv"
    condition_path = output / "condition_predictions.csv"
    metric_path = output / "case_metrics.csv"
    gate_path = output / "gate_results.csv"
    _write_csv(phase_path, phase_rows)
    _write_csv(condition_path, condition_rows)
    _write_csv(metric_path, metric_rows)
    _write_csv(gate_path, gates)
    sources = (
        Path(__file__).resolve(),
        YANG_MEANS,
        YANG_PHASE,
        FIG14_PHASE,
        FIG14_GT,
        V4_YANG,
        V4_FIG14,
        BAIK_PHASE,
        BAIK_GT,
    )
    results = (phase_path, condition_path, metric_path, gate_path)
    summary = {
        "run_id": output.name,
        "status": "partial_development_only",
        "canonical_eligible": False,
        "quality": args.quality,
        "mode": args.mode,
        "lambda_tau": args.lambda_tau,
        "warmup_cycles": args.warmup_cycles,
        "steps_per_cycle": steps_per_cycle,
        "yang_strip_count": strips,
        "incidence_source": "kinematic_proxy",
        "aggregation_scope": "projected_integrated_proxy",
        "fig14_equilibrium_source": args.fig14_equilibrium_source,
        "profile_drag_ledger": {
            "yang": "none declared",
            "figure14": "Cd0=0.057 already in retained UVLM baseline; not re-added",
            "baik": "zero section baseline in current mean-surface adapter",
        },
        "headline": {"yang": yang, "figure14": fig14, "baik": baik},
        "gate_counts": {
            "numeric_pass": sum(bool(row["numeric_pass"]) for row in gates),
            "total": len(gates),
            "promotion_pass": 0,
        },
        "limitations": [
            "frozen caches do not contain UVLM-induced strip velocities",
            "compatibility adapter filters already projected integrated CL/CD, not strip CN/CS",
            "Figure 14 LDVM cache path supplies axial discrepancy only; lift is unscored",
            "Yang has no experimental phase load and is scored on means only",
            "Baik equilibrium provider is exactly reconstructed from the frozen v4b ledger",
        ],
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sources
        },
        "result_hashes": {path.name: _sha256(path) for path in results},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
