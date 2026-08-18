"""Evaluate the frozen Yang A/L/P region-owner shadow on all 22 cases.

The runner preserves the three existing load vertices and changes only their
instantaneous area-weighted ownership:

``A=UVLM``, ``L=UVLM+paired LDVM delta``, ``P=UVLM+polar delta``.

The region boundaries and ``C_alpha=5`` come from Yang et al. (2025), while
each section's already-declared separation angle is frozen before scoring.
Because integrated load caches are blended by strip-area fractions and the
incidence omits same-layer UVLM induction, this is a non-canonical proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from forward_flight_benchmarks.baik2012 import BAIK_2012_CASES, build_baik_movement
from forward_flight_benchmarks.cases import (
    IZRAELEVITZ_2017_FIG14_SCHERER,
    YANG_2025,
)
from forward_flight_benchmarks.fluxv_v5d_region_owner import (
    DEFAULT_REGION_OWNER_PARAMETERS,
    cross_section_region_owner,
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
from forward_flight_benchmarks.run_fluxv_v5d1_source_clock import (
    REPO_ROOT,
    _model_history,
    _source_owner,
)
from forward_flight_benchmarks.run_v4_crosspaper import (
    _yang_ldvm_phase_and_ownership,
)
from forward_flight_benchmarks.uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    movement_polar_residual,
)


REPRO_ROOT = REPO_ROOT / "docs/forward_flight_large_pitch/reproductions"
DOC_ROOT = REPRO_ROOT / "fluxv_v5c_nextgen_20260814"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5d2_region_owner_all22"

YANG_LDVM_STEPS = 256
YANG_STRIPS = 12
YANG_ALPHA_SEP_RAD = np.deg2rad(YANG_2025.separation_aoa_deg)
FIG14_ALPHA_SEP_RAD = np.deg2rad(0.90 / 0.065)
BAIK_ALPHA_SEP_RAD = float(np.arcsin(0.11))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _region_from_movement(
    movement: Any,
    *,
    period_s: float,
    freestream_m_s: float,
    rho_kg_m3: float,
    aspect_ratio: float,
    chord_m: float,
    alpha_sep_rad: float,
) -> dict[str, Any]:
    # Reuse the audited 0.75c kinematic strip exporter.  Its source-time state
    # is ignored here; v5d2 is an instantaneous region shadow.
    kinematic = _source_owner(
        movement,
        period_s=period_s,
        freestream_m_s=freestream_m_s,
        rho_kg_m3=rho_kg_m3,
        aspect_ratio=aspect_ratio,
        chord_m=chord_m,
    )
    result = cross_section_region_owner(
        kinematic["alpha_rad"],
        alpha_sep_rad=alpha_sep_rad,
        strip_weights=kinematic["strip_area_m2"],
    )
    disabled = cross_section_region_owner(
        np.full_like(np.asarray(kinematic["alpha_rad"]), np.nan),
        alpha_sep_rad="not evaluated",
        strip_weights={"not": "evaluated"},
        enabled=False,
    )
    return {
        "weights": result["weights"],
        "strip_region_masks": result["strip_region_masks"],
        "alpha_rad": kinematic["alpha_rad"],
        "strip_area_m2": kinematic["strip_area_m2"],
        "normalized_strip_weights": result["normalized_strip_weights"],
        "alpha_sep_rad": alpha_sep_rad,
        "disabled_max_abs": float(
            max(
                np.max(np.abs(np.asarray(disabled["weights"]["wA"]) - 1.0)),
                np.max(np.abs(np.asarray(disabled["weights"]["wL"]))),
                np.max(np.abs(np.asarray(disabled["weights"]["wP"]))),
            )
        ),
        "weight_sum_max_abs": float(
            result["diagnostics"]["max_abs_weight_sum_residual"]
        ),
        "parameters": result["parameters"],
    }


def _blend_regions(
    attached: np.ndarray,
    lev: np.ndarray,
    polar: np.ndarray,
    region: dict[str, Any],
) -> np.ndarray:
    attached_history = np.asarray(attached, dtype=float)
    lev_history = np.asarray(lev, dtype=float)
    polar_history = np.asarray(polar, dtype=float)
    if (
        attached_history.shape != lev_history.shape
        or attached_history.shape != polar_history.shape
        or attached_history.shape != (OUTPUT_SAMPLES,)
    ):
        raise ValueError("A/L/P branch histories must share the 128-point grid")
    weights = region["weights"]
    output = (
        np.asarray(weights["wA"]) * attached_history
        + np.asarray(weights["wL"]) * lev_history
        + np.asarray(weights["wP"]) * polar_history
    )
    if np.any(~np.isfinite(output)):
        raise FloatingPointError("region-owned load is not finite")
    return output


def _append_region_state(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    case_id: str,
    region: dict[str, Any],
) -> None:
    masks = region["strip_region_masks"]
    alpha = np.asarray(region["alpha_rad"])
    for time_index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
        for strip_index in range(alpha.shape[1]):
            rows.append(
                {
                    "benchmark": benchmark,
                    "case_id": case_id,
                    "phase": phase,
                    "strip_id": strip_index,
                    "alpha_rad": alpha[time_index, strip_index],
                    "alpha_sep_rad": region["alpha_sep_rad"],
                    "strip_area_m2": region["strip_area_m2"][strip_index],
                    "normalized_strip_weight": region["normalized_strip_weights"][
                        strip_index
                    ],
                    "region_A": int(masks["A"][time_index, strip_index]),
                    "region_L": int(masks["L"][time_index, strip_index]),
                    "region_P": int(masks["P"][time_index, strip_index]),
                    "wA": region["weights"]["wA"][time_index],
                    "wL": region["weights"]["wL"][time_index],
                    "wP": region["weights"]["wP"][time_index],
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
    diagnostics = {"disabled_max_abs": 0.0, "weight_sum_max_abs": 0.0}
    for aoa in sorted(frozen):
        old = _phase_history(histories, "fluxv_uvpm", aoa_deg=aoa)
        polar = _phase_history(histories, "fluxv_periodic_v1", aoa_deg=aoa)
        ldvm = _yang_ldvm_phase_and_ownership(
            aoa, steps_per_cycle=YANG_LDVM_STEPS, strip_count=YANG_STRIPS
        )
        delta_cl = _periodic_resample(np.asarray(ldvm["ldvm_delta_CL"]))
        delta_cd = _periodic_resample(np.asarray(ldvm["ldvm_delta_CD"]))
        movement, _ = build_yang2025_movement(
            aoa, settings=(2, YANG_STRIPS, OUTPUT_SAMPLES, 3, 2)
        )
        region = _region_from_movement(
            movement,
            period_s=YANG_2025.period_s,
            freestream_m_s=YANG_2025.freestream_m_s,
            rho_kg_m3=YANG_2025.rho_kg_m3,
            aspect_ratio=YANG_2025.aspect_ratio,
            chord_m=YANG_2025.chord_m,
            alpha_sep_rad=YANG_ALPHA_SEP_RAD,
        )
        old_cl = np.asarray(old["CL"])
        old_cd = np.asarray(old["CD"])
        lev_cl = old_cl + delta_cl
        lev_cd = old_cd + delta_cd
        candidate_cl = _blend_regions(old_cl, lev_cl, np.asarray(polar["CL"]), region)
        candidate_cd = _blend_regions(old_cd, lev_cd, np.asarray(polar["CD"]), region)
        reference = frozen[aoa]
        case_id = f"aoa_{aoa:g}"
        predictions[case_id] = {
            "aoa_deg": aoa,
            "baseline_lift_gf": float(reference["v4_lift_gf"]),
            "baseline_drag_gf": float(reference["v4_drag_gf"]),
            "proxy_lift_gf": float(np.mean(candidate_cl) * q_area_per_gf),
            "proxy_drag_gf": float(np.mean(candidate_cd) * q_area_per_gf),
        }
        _append_region_state(
            state_rows, benchmark="yang2025", case_id=case_id, region=region
        )
        diagnostics["disabled_max_abs"] = max(
            diagnostics["disabled_max_abs"], region["disabled_max_abs"]
        )
        diagnostics["weight_sum_max_abs"] = max(
            diagnostics["weight_sum_max_abs"], region["weight_sum_max_abs"]
        )
        for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
            phase_rows.append(
                {
                    "benchmark": "yang2025",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CL": "",
                    "baseline_CD": "",
                    "region_owner_CL": candidate_cl[index],
                    "region_owner_CD": candidate_cd[index],
                    "wA": region["weights"]["wA"][index],
                    "wL": region["weights"]["wL"][index],
                    "wP": region["weights"]["wP"][index],
                    "canonical_eligible": "false",
                }
            )
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
    diagnostics = {"disabled_max_abs": 0.0, "weight_sum_max_abs": 0.0}
    for theta, psi in conditions:
        selected = [
            row
            for row in source
            if row["model"] == "v5c0_corrected_v4b_075c"
            and np.isclose(float(row["theta_max_deg"]), theta)
            and np.isclose(float(row["phase_offset_deg"]), psi)
        ]
        selected.sort(key=lambda row: float(row["phase"]))
        baseline = np.asarray([float(row["CT"]) for row in selected])
        attached = np.asarray([float(row["target_old_CT"]) for row in selected])
        lev = attached + np.asarray([float(row["ldvm_delta_CT"]) for row in selected])
        polar = np.asarray([float(row["target_polar_CT"]) for row in selected])
        movement, _ = build_izraelevitz_scherer_movement(
            theta, psi, settings=(2, 12, OUTPUT_SAMPLES, 4)
        )
        region = _region_from_movement(
            movement,
            period_s=IZRAELEVITZ_2017_FIG14_SCHERER.period_s,
            freestream_m_s=IZRAELEVITZ_2017_FIG14_SCHERER.freestream_m_s,
            rho_kg_m3=IZRAELEVITZ_2017_FIG14_SCHERER.rho_kg_m3,
            aspect_ratio=IZRAELEVITZ_2017_FIG14_SCHERER.aspect_ratio,
            chord_m=IZRAELEVITZ_2017_FIG14_SCHERER.chord_m,
            alpha_sep_rad=FIG14_ALPHA_SEP_RAD,
        )
        candidate = _blend_regions(attached, lev, polar, region)
        case_id = f"theta_{theta:g}_psi_{psi:g}"
        predictions[case_id] = {
            "theta_max_deg": theta,
            "phase_offset_deg": psi,
            "baseline_CT": float(np.mean(baseline)),
            "proxy_CT": float(np.mean(candidate)),
        }
        _append_region_state(
            state_rows,
            benchmark="izraelevitz2017_fig14",
            case_id=case_id,
            region=region,
        )
        for key in diagnostics:
            diagnostics[key] = max(diagnostics[key], region[key])
        for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
            phase_rows.append(
                {
                    "benchmark": "izraelevitz2017_fig14",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CT": baseline[index],
                    "region_owner_CT": candidate[index],
                    "wA": region["weights"]["wA"][index],
                    "wL": region["weights"]["wL"][index],
                    "wP": region["weights"]["wP"][index],
                    "canonical_eligible": "false",
                }
            )
    return predictions, diagnostics


def _baik_predictions(
    phase_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    rows = _read_csv(BAIK_PHASE)
    predictions: dict[str, dict[str, Any]] = {}
    diagnostics = {"disabled_max_abs": 0.0, "weight_sum_max_abs": 0.0}
    for case_id, case in BAIK_2012_CASES.items():
        old = _model_history(rows, case_id, "fluxv_old")
        baseline = _model_history(rows, case_id, "fluxv_v4b")
        movement, _ = build_baik_movement(case, settings=(2, 8, OUTPUT_SAMPLES, 3))
        source_cycle = _cycle_range(movement, case.period_s)
        polar_delta = movement_polar_residual(
            movement,
            source_cycle_step_range=source_cycle,
            period_s=case.period_s,
            freestream_m_s=case.freestream_m_s,
            rho_kg_m3=case.rho_kg_m3,
            aspect_ratio=case.geometric_aspect_ratio,
            output_samples=OUTPUT_SAMPLES,
            parameters=DEFAULT_POLAR_PARAMETERS,
        )
        q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
        attached_cl = np.asarray(old["CL"])
        attached_cd = np.asarray(old["CD"])
        lev_cl = attached_cl + np.asarray(baseline["ldvm_delta_CL"])
        lev_cd = attached_cd + np.asarray(baseline["ldvm_delta_CD"])
        polar_cl = attached_cl + np.asarray(polar_delta["delta_lift_n"]) / q_area
        polar_cd = attached_cd + np.asarray(polar_delta["delta_drag_n"]) / q_area
        region = _region_from_movement(
            movement,
            period_s=case.period_s,
            freestream_m_s=case.freestream_m_s,
            rho_kg_m3=case.rho_kg_m3,
            aspect_ratio=case.geometric_aspect_ratio,
            chord_m=case.chord_m,
            alpha_sep_rad=BAIK_ALPHA_SEP_RAD,
        )
        candidate_cl = _blend_regions(attached_cl, lev_cl, polar_cl, region)
        candidate_cd = _blend_regions(attached_cd, lev_cd, polar_cd, region)
        predictions[case_id] = {
            "phase": np.asarray(old["phase"]),
            "baseline_CL": np.asarray(baseline["CL"]),
            "baseline_CD": np.asarray(baseline["CD"]),
            "proxy_CL": candidate_cl,
            "proxy_CD": candidate_cd,
        }
        _append_region_state(
            state_rows, benchmark="baik2012", case_id=case_id, region=region
        )
        for key in diagnostics:
            diagnostics[key] = max(diagnostics[key], region[key])
        for index, phase in enumerate(np.asarray(old["phase"])):
            phase_rows.append(
                {
                    "benchmark": "baik2012",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CL": baseline["CL"][index],
                    "baseline_CD": baseline["CD"][index],
                    "region_owner_CL": candidate_cl[index],
                    "region_owner_CD": candidate_cd[index],
                    "wA": region["weights"]["wA"][index],
                    "wL": region["weights"]["wL"][index],
                    "wP": region["weights"]["wP"][index],
                    "canonical_eligible": "false",
                }
            )
    return predictions, diagnostics


def _gate(
    gate_id: str,
    measured: float | bool,
    threshold: float | bool,
    *,
    exact: bool = False,
) -> dict[str, Any]:
    passed = measured == threshold if exact else float(measured) <= float(threshold)
    return {
        "gate_id": gate_id,
        "measured": measured,
        "relation": "==" if exact else "<=",
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
    # All predictions are complete before any scorer reads observations.
    yang_predictions, yang_diag = _yang_predictions(phase_rows, state_rows)
    fig_predictions, fig_diag = _fig14_predictions(phase_rows, state_rows)
    baik_predictions, baik_diag = _baik_predictions(phase_rows, state_rows)
    condition_count = (
        len(yang_predictions) + len(fig_predictions) + len(baik_predictions)
    )
    if condition_count != 22 or len(phase_rows) != 22 * OUTPUT_SAMPLES:
        raise RuntimeError("v5d2 all-22 prediction coverage failed")

    metrics: list[dict[str, Any]] = []
    yang = _score_yang(yang_predictions, metrics)
    fig14 = _score_fig14(fig_predictions, metrics)
    baik = _score_baik(baik_predictions, metrics)
    for row in metrics:
        row["model"] = "fluxv_v5d2_region_owner"
    v5c0 = json.loads(V5C0_SUMMARY.read_text(encoding="utf-8"))[
        "corrected_v5c0_metrics"
    ]
    diagnostics = (yang_diag, fig_diag, baik_diag)
    mechanical = {
        "disabled_max_abs": max(row["disabled_max_abs"] for row in diagnostics),
        "weight_sum_max_abs": max(row["weight_sum_max_abs"] for row in diagnostics),
        "weight_min": min(
            float(row[key]) for row in phase_rows for key in ("wA", "wL", "wP")
        ),
        "weight_max": max(
            float(row[key]) for row in phase_rows for key in ("wA", "wL", "wP")
        ),
    }
    gates = [
        _gate("coverage_22_conditions", condition_count, 22, exact=True),
        _gate("phase_rows_2816", len(phase_rows), 2816, exact=True),
        _gate("disabled_identity", mechanical["disabled_max_abs"], 0.0),
        _gate("weight_sum_closure", mechanical["weight_sum_max_abs"], 1.0e-12),
        _gate("weight_lower_bound", -mechanical["weight_min"], 0.0),
        _gate("weight_upper_bound", mechanical["weight_max"], 1.0),
        _gate("yang_lift_mae_gf", yang["lift_mae"], YANG_THRESHOLDS["lift"]),
        _gate("yang_drag_mae_gf", yang["drag_mae"], YANG_THRESHOLDS["drag"]),
        _gate("fig14_all14_rmse", fig14["all14_rmse"], v5c0["all_14_markers"]["rmse"]),
        _gate("fig14_theta15_rmse", fig14["theta_15_rmse"], v5c0["theta_15"]["rmse"]),
        _gate("fig14_theta25_rmse", fig14["theta_25_rmse"], v5c0["theta_25"]["rmse"]),
        _gate(
            "fig14_unique12_rmse",
            fig14["unique12_rmse"],
            v5c0["unique_12_conditions"]["rmse"],
        ),
        _gate("baik_macro_CL_rmse", baik["macro"]["CL"], BAIK_MACRO_THRESHOLDS["CL"]),
        _gate("baik_macro_CD_rmse", baik["macro"]["CD"], BAIK_MACRO_THRESHOLDS["CD"]),
    ]
    for key, threshold in BAIK_THRESHOLDS.items():
        gates.append(
            _gate(f"baik_{key[0]}_{key[1]}_rmse", baik["rmse"][key], threshold)
        )
    gates.append(_gate("canonical_strip_inputs_available", False, True, exact=True))
    mechanical_ids = {
        "coverage_22_conditions",
        "phase_rows_2816",
        "disabled_identity",
        "weight_sum_closure",
        "weight_lower_bound",
        "weight_upper_bound",
    }
    if not all(
        row["numeric_pass"] for row in gates if row["gate_id"] in mechanical_ids
    ):
        raise RuntimeError("v5d2 mechanical gate failed")
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
        "region_owner_numeric_pass_canonical_blocked"
        if all(paper_pass.values())
        else "stopped_region_owner_crosspaper_gate_failure"
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
            "region_owner_filtered_CL_rmse": baik["rmse"][(case_id, "CL")],
            "baseline_filtered_CD_rmse": baik["baseline_rmse"][(case_id, "CD")],
            "region_owner_filtered_CD_rmse": baik["rmse"][(case_id, "CD")],
            "canonical_eligible": "false",
        }
        for case_id in BAIK_2012_CASES
    )
    for row in condition_rows:
        row["canonical_eligible"] = "false"

    phase_path = output / "phase_predictions.csv"
    state_path = output / "region_state_histories.csv"
    condition_path = output / "condition_predictions.csv"
    metric_path = output / "case_metrics.csv"
    gate_path = output / "gate_results.csv"
    _write_csv(phase_path, phase_rows)
    _write_csv(state_path, state_rows)
    _write_csv(condition_path, condition_rows)
    _write_csv(metric_path, metrics)
    _write_csv(gate_path, gates)
    summary = {
        "run_id": output.name,
        "status": status,
        "promotion_status": "blocked_noncanonical_integrated_branch_proxy",
        "canonical_eligible": False,
        "condition_count": condition_count,
        "phase_row_count": len(phase_rows),
        "region_state_row_count": len(state_rows),
        "prediction_completed_before_scoring": True,
        "parameter_selection_data": [],
        "region_owner_parameters": DEFAULT_REGION_OWNER_PARAMETERS.manifest(),
        "section_separation_inputs": {
            "yang_rad": YANG_ALPHA_SEP_RAD,
            "fig14_rad": FIG14_ALPHA_SEP_RAD,
            "baik_rad": BAIK_ALPHA_SEP_RAD,
        },
        "mechanical": mechanical,
        "paper_pass": paper_pass,
        "yang": yang,
        "fig14": fig14,
        "baik": {
            "macro": baik["macro"],
            "rmse": {f"{k[0]}_{k[1]}": v for k, v in baik["rmse"].items()},
        },
        "limitations": [
            "C_alpha=5 is a Yang cross-section transfer hypothesis for Fig14 and Baik.",
            "Kinematic 0.75c incidence omits same-time-layer UVLM induction.",
            "Integrated branch loads are blended by strip-area fractions rather than co-located strip forces.",
            "All 22 conditions are previously inspected development evidence.",
        ],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("fluxv_v5d_region_owner.py").resolve(),
        Path(__file__).with_name("run_fluxv_v5d1_source_clock.py").resolve(),
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
        "canonical_eligible": False,
        "condition_count": condition_count,
        "parameter_selection_data": [],
        "region_owner": DEFAULT_REGION_OWNER_PARAMETERS.manifest(),
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
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
