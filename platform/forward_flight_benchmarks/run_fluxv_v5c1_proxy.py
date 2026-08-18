"""Run the non-canonical FluxV v5c1 suction-loss proxy on all 22 cases.

This runner is deliberately a cache adapter.  It reuses the audited v5c0
Figure-14 phase histories and the frozen v4b Yang/Baik ledgers, then recomputes
the already-declared paired-LDVM section histories solely to expose the
separated branch's pre-constraint LESP and axial ``CSf``.  The v5c1 state may
only replace that explicit axial-suction line through the existing ``(1-p)``
transient owner.

The frozen caches do not contain same-time-layer UVLM-induced strip velocity,
so every output is marked ``canonical_eligible=false``.  Observation columns
are not used by a prediction formula, and no parameter is selected from any
observed load.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from forward_flight_benchmarks.baik2012 import (
    BAIK_2012_CASES,
    baik_kinematics,
    sharp_fourier_lowpass,
)
from forward_flight_benchmarks.cases import (
    IZRAELEVITZ_2017_FIG14_SCHERER,
    YANG_2025,
)
from forward_flight_benchmarks.causal_incidence_owner import (
    causal_incidence_persistence,
)
from forward_flight_benchmarks.fluxv_v5c_suction import (
    DEFAULT_SUCTION_PARAMETERS,
    project_axial_suction_loss_to_wind_axes,
    run_rate_sensitive_suction_history,
)
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
    project_ldvm_delta_to_finite_wing,
    run_ldvm_separation_pair,
)
from forward_flight_benchmarks.run_v4_crosspaper import _yang_kinematics


REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = REPO_ROOT / "docs/forward_flight_large_pitch/reproductions"
V3_ROOT = REPRO_ROOT / "unified_fluxv_upgrade_20260812"
V4_ROOT = REPRO_ROOT / "unified_fluxv_v4_ldvm_stevens_20260812"
V5C_ROOT = REPRO_ROOT / "fluxv_v5c_nextgen_20260814"
BAIK_ROOT = REPRO_ROOT / "baik2012_w1_w4"

DEFAULT_OUTPUT = (
    V5C_ROOT / "runs/20260814_fluxv_v5c1_proxy_all22_pole05_rate1_reproducible"
)
V5C0_RUN = V5C_ROOT / "runs/20260814_fluxv_v5c0_reference_audited"
V5C0_PHASE = V5C0_RUN / "fig14_v5c0_phase_histories.csv"
V5C0_SUMMARY = V5C0_RUN / "summary.json"
YANG_PHASE = (
    V3_ROOT / "runs/20260812_periodic_v2_ullt_full/yang2025_phase_histories.csv"
)
YANG_V4 = (
    V4_ROOT
    / "runs/20260812_fluxv_v4b_crosspaper_full/yang2025_v4_mean_characteristics.csv"
)
YANG_GT = REPRO_ROOT / "plev2025/source_data/yang2025_fig11_rigid_digitized.csv"
FIG14_GT = V3_ROOT / "source_data/izraelevitz2017_fig14_digitized.csv"
BAIK_PHASE = (
    BAIK_ROOT
    / "runs/20260813_baik2012_w1_w4_full_reproducible/model_phase_histories.csv"
)
BAIK_GT = BAIK_ROOT / "source_data/baik2012_w1_w4_corrected_total_cl_cd.csv"

OUTPUT_SAMPLES = 128
STATE_TOTAL_CYCLES = 12
STATE_DISCARDED_CYCLES = STATE_TOTAL_CYCLES - 1
STATE_SCORED_CYCLES = 1
YANG_LDVM_STEPS = 256
YANG_STRIPS = 12
FIG14_LDVM_STEPS = 256
BAIK_LDVM_STEPS = 512
BAIK_WAKE_STEPS = 256

YANG_THRESHOLDS = {"lift": 4.554509816656029, "drag": 2.643997473596851}
BAIK_THRESHOLDS = {
    ("W1", "CL"): 0.5156649730771129,
    ("W1", "CD"): 0.16090663539311115,
    ("W2", "CL"): 1.0323056437701223,
    ("W2", "CD"): 0.725677862528139,
    ("W3", "CL"): 0.3743187762452229,
    ("W3", "CD"): 0.2625914511712678,
    ("W4", "CL"): 0.7078780735787529,
    ("W4", "CD"): 0.2314337267583123,
}
BAIK_MACRO_THRESHOLDS = {"CL": 0.6575418666678028, "CD": 0.34515241896270754}


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _periodic_resample(values: np.ndarray, samples: int = OUTPUT_SAMPLES) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("periodic history must be a finite one-dimensional array")
    source = np.arange(array.size, dtype=float) / array.size
    target = np.arange(samples, dtype=float) / samples
    return np.interp(target, source, array, period=1.0)


def _metric(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("observed and predicted histories must align")
    error = predicted - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _phase_history(
    rows: list[dict[str, str]], model: str, **selectors: str | float
) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["model"] == model]
    for key, value in selectors.items():
        if isinstance(value, str):
            selected = [row for row in selected if row[key] == value]
        else:
            selected = [
                row for row in selected if np.isclose(float(row[key]), float(value))
            ]
    selected.sort(key=lambda row: float(row["phase"]))
    if len(selected) != OUTPUT_SAMPLES:
        raise ValueError(f"expected 128 rows for {model}/{selectors}")
    output = {"phase": np.asarray([float(row["phase"]) for row in selected])}
    for field in ("CL", "CD", "CT", "lift_n", "drag_n", "persistence"):
        if field in selected[0] and selected[0][field] != "":
            output[field] = np.asarray([float(row[field]) for row in selected])
    return output


def _periodic_state_proxy(
    *,
    a0_pre: np.ndarray,
    separated_cs: np.ndarray,
    delta_tau: np.ndarray,
    alpha_rad: np.ndarray,
    lesp_critical: float,
    aspect_ratio: float,
) -> dict[str, Any]:
    """Return one periodic axial-only proxy correction and state ledger."""

    a0 = np.asarray(a0_pre, dtype=float)
    base_cs = np.asarray(separated_cs, dtype=float)
    convective_step = np.asarray(delta_tau, dtype=float)
    alpha = np.asarray(alpha_rad, dtype=float)
    if not (a0.shape == base_cs.shape == convective_step.shape == alpha.shape):
        raise ValueError("proxy section histories must have identical shapes")
    if a0.ndim != 1 or a0.size < 8:
        raise ValueError("proxy section history must contain one resolved cycle")
    if np.any(base_cs < -1.0e-14):
        raise ValueError("separated CSf must be nonnegative")
    base_cs = np.maximum(base_cs, 0.0)

    def tiled(value: np.ndarray) -> np.ndarray:
        return np.tile(value, STATE_TOTAL_CYCLES)[:, None]

    common = {
        "a0_pre": tiled(a0),
        "lesp_critical": np.full((STATE_TOTAL_CYCLES * a0.size, 1), lesp_critical),
        "delta_tau": tiled(convective_step),
        "base_suction_coefficient": tiled(base_cs),
    }
    active = run_rate_sensitive_suction_history(**common, enabled=True)
    disabled = run_rate_sensitive_suction_history(**common, enabled=False)
    last = slice(-a0.size, None)
    previous = slice(-2 * a0.size, -a0.size)
    gain = (1.0 / (1.0 + 2.0 / aspect_ratio)) ** 2
    delta_cs = gain * np.asarray(active["delta_suction_coefficient"])[last, 0]
    projection = project_axial_suction_loss_to_wind_axes(delta_cs, alpha)
    cycle_state_error = float(
        np.max(
            np.abs(
                np.asarray(active["chi_after"])[last, 0]
                - np.asarray(active["chi_after"])[previous, 0]
            )
        )
    )
    out = {
        key: np.asarray(active[key])[last, 0]
        for key in (
            "j",
            "j_rate",
            "supercritical_gate",
            "chi_equilibrium",
            "chi_after",
            "loss_fraction",
            "base_suction_coefficient",
            "target_suction_coefficient",
            "delta_suction_coefficient",
            "delta_normal_coefficient",
        )
    }
    previous_chi = np.asarray(active["chi_after"])[previous, 0]
    out["chi_previous_cycle"] = previous_chi
    out["chi_cycle_difference"] = out["chi_after"] - previous_chi
    out.update(
        delta_CL=np.asarray(projection["delta_CL"]),
        delta_CD=np.asarray(projection["delta_CD"]),
        delta_CN=np.asarray(projection["delta_CN"]),
        delta_CS=np.asarray(projection["delta_CS"]),
        cycle_state_error=cycle_state_error,
        disabled_max_abs=float(np.max(np.abs(disabled["delta_suction_coefficient"]))),
        axial_gain=gain,
    )
    return out


def _append_state_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    case_id: str,
    strip_id: int,
    a0_pre: np.ndarray,
    result: dict[str, Any],
    owner: np.ndarray,
) -> None:
    fields = {
        "a0_pre": _periodic_resample(a0_pre),
        "j": _periodic_resample(result["j"]),
        "j_rate": _periodic_resample(result["j_rate"]),
        "supercritical_gate": _periodic_resample(result["supercritical_gate"]),
        "chi_equilibrium": _periodic_resample(result["chi_equilibrium"]),
        "chi": _periodic_resample(result["chi_after"]),
        "chi_previous_cycle": _periodic_resample(result["chi_previous_cycle"]),
        "chi_cycle_difference": _periodic_resample(result["chi_cycle_difference"]),
        "loss_fraction": _periodic_resample(result["loss_fraction"]),
        "separated_CSf": _periodic_resample(result["base_suction_coefficient"]),
        "target_CS": _periodic_resample(result["target_suction_coefficient"]),
        "delta_CS_2d": _periodic_resample(result["delta_suction_coefficient"]),
        "delta_CS_projected": _periodic_resample(result["delta_CS"]),
        "delta_CL_projected": _periodic_resample(result["delta_CL"]),
        "delta_CD_projected": _periodic_resample(result["delta_CD"]),
        "delta_CN": _periodic_resample(result["delta_CN"]),
    }
    for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
        rows.append(
            {
                "benchmark": benchmark,
                "case_id": case_id,
                "phase": phase,
                "strip_id": strip_id,
                **{key: value[index] for key, value in fields.items()},
                "transient_owner": 1.0 - owner[index],
                "incidence_source": "kinematic_proxy_no_uvlm_induction",
                "canonical_eligible": "false",
            }
        )


def _yang_predictions(
    phase_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    source_rows = _read_csv(YANG_PHASE)
    v4_rows = {
        float(row["aoa_deg"]): {
            "v4_lift_gf": float(row["v4_lift_gf"]),
            "v4_drag_gf": float(row["v4_drag_gf"]),
        }
        for row in _read_csv(YANG_V4)
    }
    case = YANG_2025
    threshold = LESPThreshold(
        value=float(np.sin(np.deg2rad(case.separation_aoa_deg))),
        section_family="1 mm balsa flat plate",
        reynolds=case.reynolds,
        source="Yang 2025 alpha_sep=5 deg mapped as Lcrit=sin(alpha_sep)",
        source_role="paper-parameter mapping hypothesis; no force fit",
    )
    predictions: dict[str, dict[str, Any]] = {}
    diagnostics = {
        "disabled_max_abs": 0.0,
        "normal_max_abs": 0.0,
        "state_cycle_max_abs": 0.0,
        "baseline_replay_max_abs": 0.0,
    }
    q_area_per_gf = (
        0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2 / 0.00980665
    )
    for aoa in sorted(v4_rows):
        old = _phase_history(source_rows, "fluxv_uvpm", aoa_deg=aoa)
        polar = _phase_history(source_rows, "fluxv_periodic_v1", aoa_deg=aoa)
        owner_history = _yang_kinematics(
            aoa,
            steps_per_cycle=YANG_LDVM_STEPS,
            cycles=12,
            strip_count=YANG_STRIPS,
        )
        owner = causal_incidence_persistence(
            np.asarray(owner_history["effective_alpha_rad"]),
            delta_time_convective=float(owner_history["delta_time_convective"]),
        )
        p_internal = np.asarray(owner["global_persistence"])[-YANG_LDVM_STEPS:]
        p = _periodic_resample(p_internal)

        ldvm_history = _yang_kinematics(
            aoa,
            steps_per_cycle=YANG_LDVM_STEPS,
            cycles=2,
            strip_count=YANG_STRIPS,
        )
        selected = slice(YANG_LDVM_STEPS, 2 * YANG_LDVM_STEPS)
        baseline_delta_cl = np.zeros(YANG_LDVM_STEPS)
        baseline_delta_cd = np.zeros(YANG_LDVM_STEPS)
        proxy_delta_cl = np.zeros(YANG_LDVM_STEPS)
        proxy_delta_cd = np.zeros(YANG_LDVM_STEPS)
        alpha_install = np.deg2rad(aoa)
        flap = np.asarray(ldvm_history["flap_rad"])[selected]
        dt = float(ldvm_history["delta_time_convective"])
        for strip_id, radius in enumerate(np.asarray(ldvm_history["radii_m"])):
            heave = -(radius / case.chord_m) * np.asarray(
                ldvm_history["flap_rate_per_convective_time"]
            )
            pair = run_ldvm_separation_pair(
                alpha_rad=np.full(2 * YANG_LDVM_STEPS, alpha_install),
                alpha_rate_per_convective_time=np.zeros(2 * YANG_LDVM_STEPS),
                heave_rate_over_u=heave,
                delta_time_convective=dt,
                pivot_fraction_chord=0.25,
                threshold=threshold,
                settings=LDVMSectionSettings(
                    ndiv=32,
                    naterm=14,
                    max_wake_steps=YANG_LDVM_STEPS,
                ),
            )
            projection = project_ldvm_delta_to_finite_wing(
                np.asarray(pair["delta"]["CNc"])[selected],
                np.asarray(pair["delta"]["CNnc"])[selected],
                np.asarray(pair["delta"]["CNnonl"])[selected],
                np.asarray(pair["delta"]["CSf"])[selected],
                np.full(YANG_LDVM_STEPS, alpha_install),
                aspect_ratio=case.aspect_ratio,
            )
            baseline_delta_cl += (
                np.asarray(projection["delta_CL"]) * np.cos(flap) / YANG_STRIPS
            )
            baseline_delta_cd += np.asarray(projection["delta_CD"]) / YANG_STRIPS
            proxy = _periodic_state_proxy(
                a0_pre=np.asarray(pair["separated"]["lesp"])[selected],
                separated_cs=np.asarray(pair["separated"]["CSf"])[selected],
                delta_tau=dt * np.sqrt(1.0 + heave[selected] ** 2),
                alpha_rad=np.full(YANG_LDVM_STEPS, alpha_install),
                lesp_critical=threshold.value,
                aspect_ratio=case.aspect_ratio,
            )
            proxy_delta_cl += proxy["delta_CL"] * np.cos(flap) / YANG_STRIPS
            proxy_delta_cd += proxy["delta_CD"] / YANG_STRIPS
            diagnostics["disabled_max_abs"] = max(
                diagnostics["disabled_max_abs"], proxy["disabled_max_abs"]
            )
            diagnostics["normal_max_abs"] = max(
                diagnostics["normal_max_abs"],
                float(np.max(np.abs(proxy["delta_CN"]))),
            )
            diagnostics["state_cycle_max_abs"] = max(
                diagnostics["state_cycle_max_abs"], proxy["cycle_state_error"]
            )
            _append_state_rows(
                state_rows,
                benchmark="yang2025",
                case_id=f"aoa_{aoa:g}",
                strip_id=strip_id,
                a0_pre=np.asarray(pair["separated"]["lesp"])[selected],
                result=proxy,
                owner=p,
            )

        v4_cl = (1.0 - p) * (
            np.asarray(old["CL"]) + _periodic_resample(baseline_delta_cl)
        ) + p * np.asarray(polar["CL"])
        v4_cd = (1.0 - p) * (
            np.asarray(old["CD"]) + _periodic_resample(baseline_delta_cd)
        ) + p * np.asarray(polar["CD"])
        correction_cl = (1.0 - p) * _periodic_resample(proxy_delta_cl)
        correction_cd = (1.0 - p) * _periodic_resample(proxy_delta_cd)
        proxy_cl = v4_cl + correction_cl
        proxy_cd = v4_cd + correction_cd
        mean_lift = float(np.mean(proxy_cl) * q_area_per_gf)
        mean_drag = float(np.mean(proxy_cd) * q_area_per_gf)
        diagnostics["baseline_replay_max_abs"] = max(
            diagnostics["baseline_replay_max_abs"],
            abs(float(np.mean(v4_cl) * q_area_per_gf) - v4_rows[aoa]["v4_lift_gf"]),
            abs(float(np.mean(v4_cd) * q_area_per_gf) - v4_rows[aoa]["v4_drag_gf"]),
        )
        case_id = f"aoa_{aoa:g}"
        predictions[case_id] = {
            "aoa_deg": aoa,
            "baseline_lift_gf": v4_rows[aoa]["v4_lift_gf"],
            "baseline_drag_gf": v4_rows[aoa]["v4_drag_gf"],
            "proxy_lift_gf": mean_lift,
            "proxy_drag_gf": mean_drag,
        }
        for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
            phase_rows.append(
                {
                    "benchmark": "yang2025",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CL": v4_cl[index],
                    "baseline_CD": v4_cd[index],
                    "baseline_CT": -v4_cd[index],
                    "proxy_CL": proxy_cl[index],
                    "proxy_CD": proxy_cd[index],
                    "proxy_CT": -proxy_cd[index],
                    "correction_CL": correction_cl[index],
                    "correction_CD": correction_cd[index],
                    "correction_CT": -correction_cd[index],
                    "transient_owner": 1.0 - p[index],
                    "baseline_model": "fluxv_v4b",
                    "canonical_eligible": "false",
                }
            )
    return predictions, diagnostics


def _fig14_pair(theta: float, psi: float) -> tuple[dict[str, Any], np.ndarray, float]:
    case = IZRAELEVITZ_2017_FIG14_SCHERER
    omega_star = np.pi * case.strouhal / case.heave_to_chord
    period_star = 2.0 * np.pi / omega_star
    dt = period_star / FIG14_LDVM_STEPS
    phase = np.arange(4 * FIG14_LDVM_STEPS) * 2.0 * np.pi / FIG14_LDVM_STEPS
    alpha = np.deg2rad(theta) * np.cos(phase + np.deg2rad(psi))
    alpha_rate = -np.deg2rad(theta) * omega_star * np.sin(phase + np.deg2rad(psi))
    heave = -case.heave_to_chord * omega_star * np.sin(phase)
    threshold = LESPThreshold(
        value=float(np.sin(np.deg2rad(0.90 / 0.065))),
        section_family=case.section_name,
        reynolds=case.freestream_m_s * case.chord_m / case.nu_m2_s,
        source="Scherer static CLa=0.065/deg and CLmax=0.90",
        source_role="static-polar-derived hypothesis; no Figure-14 force fit",
    )
    pair = run_ldvm_separation_pair(
        alpha_rad=alpha,
        alpha_rate_per_convective_time=alpha_rate,
        heave_rate_over_u=heave,
        delta_time_convective=dt,
        pivot_fraction_chord=case.pivot_fraction_chord,
        threshold=threshold,
        settings=LDVMSectionSettings(
            ndiv=50,
            naterm=24,
            max_wake_steps=4 * FIG14_LDVM_STEPS,
        ),
    )
    selected = slice(3 * FIG14_LDVM_STEPS, 4 * FIG14_LDVM_STEPS)
    return pair, np.sqrt(1.0 + heave[selected] ** 2) * dt, threshold.value


def _fig14_predictions(
    phase_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    source = _read_csv(V5C0_PHASE)
    groups = {
        (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
        for row in source
        if row["model"] == "v5c0_corrected_v4b_075c"
    }
    predictions: dict[str, dict[str, Any]] = {}
    diagnostics = {
        "disabled_max_abs": 0.0,
        "normal_max_abs": 0.0,
        "state_cycle_max_abs": 0.0,
        "owner_ledger_max_abs": 0.0,
    }
    case = IZRAELEVITZ_2017_FIG14_SCHERER
    for theta, psi in sorted(groups):
        base = _phase_history(
            source,
            "v5c0_corrected_v4b_075c",
            theta_max_deg=theta,
            phase_offset_deg=psi,
        )
        pair, delta_tau, critical = _fig14_pair(theta, psi)
        selected = slice(3 * FIG14_LDVM_STEPS, 4 * FIG14_LDVM_STEPS)
        phase_internal = np.arange(FIG14_LDVM_STEPS) * 2.0 * np.pi / FIG14_LDVM_STEPS
        alpha = np.deg2rad(theta) * np.cos(phase_internal + np.deg2rad(psi))
        proxy = _periodic_state_proxy(
            a0_pre=np.asarray(pair["separated"]["lesp"])[selected],
            separated_cs=np.asarray(pair["separated"]["CSf"])[selected],
            delta_tau=delta_tau,
            alpha_rad=alpha,
            lesp_critical=critical,
            aspect_ratio=case.aspect_ratio,
        )
        correction_cd_raw = _periodic_resample(proxy["delta_CD"])
        p = np.asarray(base["persistence"])
        correction_cd = (1.0 - p) * correction_cd_raw
        baseline_ct = np.asarray(base["CT"])
        proxy_ct = baseline_ct - correction_cd
        case_id = f"theta_{theta:g}_psi_{psi:g}"
        predictions[case_id] = {
            "theta_max_deg": theta,
            "phase_offset_deg": psi,
            "baseline_CT": float(np.mean(baseline_ct)),
            "proxy_CT": float(np.mean(proxy_ct)),
        }
        diagnostics["disabled_max_abs"] = max(
            diagnostics["disabled_max_abs"], proxy["disabled_max_abs"]
        )
        diagnostics["normal_max_abs"] = max(
            diagnostics["normal_max_abs"],
            float(np.max(np.abs(proxy["delta_CN"]))),
        )
        diagnostics["state_cycle_max_abs"] = max(
            diagnostics["state_cycle_max_abs"], proxy["cycle_state_error"]
        )
        diagnostics["owner_ledger_max_abs"] = max(
            diagnostics["owner_ledger_max_abs"],
            float(np.max(np.abs(proxy_ct - baseline_ct + correction_cd))),
        )
        _append_state_rows(
            state_rows,
            benchmark="izraelevitz2017_fig14",
            case_id=case_id,
            strip_id=0,
            a0_pre=np.asarray(pair["separated"]["lesp"])[selected],
            result=proxy,
            owner=p,
        )
        for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
            phase_rows.append(
                {
                    "benchmark": "izraelevitz2017_fig14",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CL": "",
                    "baseline_CD": -baseline_ct[index],
                    "baseline_CT": baseline_ct[index],
                    "proxy_CL": "",
                    "proxy_CD": -proxy_ct[index],
                    "proxy_CT": proxy_ct[index],
                    "correction_CL": "",
                    "correction_CD": correction_cd[index],
                    "correction_CT": -correction_cd[index],
                    "transient_owner": 1.0 - p[index],
                    "baseline_model": "v5c0_corrected_v4b_075c",
                    "canonical_eligible": "false",
                }
            )
    return predictions, diagnostics


def _baik_predictions(
    phase_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    source = _read_csv(BAIK_PHASE)
    predictions: dict[str, dict[str, Any]] = {}
    diagnostics = {
        "disabled_max_abs": 0.0,
        "normal_max_abs": 0.0,
        "state_cycle_max_abs": 0.0,
        "owner_ledger_max_abs": 0.0,
    }
    for case_id, case in BAIK_2012_CASES.items():
        base = _phase_history(source, "fluxv_v4b", case_id=case_id)
        phase_internal = np.arange(BAIK_LDVM_STEPS) / BAIK_LDVM_STEPS
        kinematics = baik_kinematics(phase_internal, case)
        alpha = np.deg2rad(kinematics["geometric_alpha_deg"])
        dt = case.freestream_m_s * case.period_s / case.chord_m / BAIK_LDVM_STEPS
        alpha_rate = (np.roll(alpha, -1) - np.roll(alpha, 1)) / (2.0 * dt)
        heave = np.asarray(kinematics["heave_rate_over_u"])
        threshold = LESPThreshold(
            value=0.11,
            section_family="rounded flat plate",
            reynolds=case.reynolds,
            source=(
                "Ramesh 2013 body-text Lcrit=0.11; frozen cross-Re/thickness "
                "transfer hypothesis"
            ),
            source_role="published flat-plate transfer hypothesis; no Baik fit",
        )
        cycles = 3
        pair = run_ldvm_separation_pair(
            alpha_rad=np.tile(alpha, cycles),
            alpha_rate_per_convective_time=np.tile(alpha_rate, cycles),
            heave_rate_over_u=np.tile(heave, cycles),
            delta_time_convective=dt,
            pivot_fraction_chord=case.pivot_fraction_chord,
            threshold=threshold,
            settings=LDVMSectionSettings(
                ndiv=32,
                naterm=14,
                max_wake_steps=BAIK_WAKE_STEPS,
                core_radius_chord=0.02,
            ),
        )
        selected = slice((cycles - 1) * BAIK_LDVM_STEPS, cycles * BAIK_LDVM_STEPS)
        proxy = _periodic_state_proxy(
            a0_pre=np.asarray(pair["separated"]["lesp"])[selected],
            separated_cs=np.asarray(pair["separated"]["CSf"])[selected],
            delta_tau=dt * np.sqrt(1.0 + heave**2),
            alpha_rad=alpha,
            lesp_critical=threshold.value,
            aspect_ratio=case.geometric_aspect_ratio,
        )
        p = np.asarray(base["persistence"])
        correction_cl = (1.0 - p) * _periodic_resample(proxy["delta_CL"])
        correction_cd = (1.0 - p) * _periodic_resample(proxy["delta_CD"])
        baseline_cl = np.asarray(base["CL"])
        baseline_cd = np.asarray(base["CD"])
        proxy_cl = baseline_cl + correction_cl
        proxy_cd = baseline_cd + correction_cd
        predictions[case_id] = {
            "phase": np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES,
            "baseline_CL": baseline_cl,
            "baseline_CD": baseline_cd,
            "proxy_CL": proxy_cl,
            "proxy_CD": proxy_cd,
        }
        diagnostics["disabled_max_abs"] = max(
            diagnostics["disabled_max_abs"], proxy["disabled_max_abs"]
        )
        diagnostics["normal_max_abs"] = max(
            diagnostics["normal_max_abs"],
            float(np.max(np.abs(proxy["delta_CN"]))),
        )
        diagnostics["state_cycle_max_abs"] = max(
            diagnostics["state_cycle_max_abs"], proxy["cycle_state_error"]
        )
        diagnostics["owner_ledger_max_abs"] = max(
            diagnostics["owner_ledger_max_abs"],
            float(np.max(np.abs(proxy_cl - baseline_cl - correction_cl))),
            float(np.max(np.abs(proxy_cd - baseline_cd - correction_cd))),
        )
        _append_state_rows(
            state_rows,
            benchmark="baik2012",
            case_id=case_id,
            strip_id=0,
            a0_pre=np.asarray(pair["separated"]["lesp"])[selected],
            result=proxy,
            owner=p,
        )
        for index, phase in enumerate(np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES):
            phase_rows.append(
                {
                    "benchmark": "baik2012",
                    "case_id": case_id,
                    "phase": phase,
                    "baseline_CL": baseline_cl[index],
                    "baseline_CD": baseline_cd[index],
                    "baseline_CT": -baseline_cd[index],
                    "proxy_CL": proxy_cl[index],
                    "proxy_CD": proxy_cd[index],
                    "proxy_CT": -proxy_cd[index],
                    "correction_CL": correction_cl[index],
                    "correction_CD": correction_cd[index],
                    "correction_CT": -correction_cd[index],
                    "transient_owner": 1.0 - p[index],
                    "baseline_model": "fluxv_v4b",
                    "canonical_eligible": "false",
                }
            )
    return predictions, diagnostics


def _score_yang(
    predictions: dict[str, dict[str, Any]], metrics: list[dict[str, Any]]
) -> dict[str, float]:
    ordered = sorted(predictions.values(), key=lambda row: row["aoa_deg"])
    observation_rows = {float(row["aoa_deg"]): row for row in _read_csv(YANG_V4)}
    out: dict[str, float] = {}
    for quantity in ("lift", "drag"):
        observed = np.asarray(
            [
                float(observation_rows[row["aoa_deg"]][f"test_{quantity}_gf"])
                for row in ordered
            ]
        )
        predicted = np.asarray([row[f"proxy_{quantity}_gf"] for row in ordered])
        baseline = np.asarray([row[f"baseline_{quantity}_gf"] for row in ordered])
        result = _metric(observed, predicted)
        baseline_metric = _metric(observed, baseline)
        metrics.append(
            {
                "benchmark": "yang2025",
                "scope": "all_6_aoa",
                "case_id": "all",
                "quantity": quantity,
                "view": "cycle_mean",
                "model": "fluxv_v5c1_proxy",
                "n": 6,
                **result,
                "baseline_rmse": baseline_metric["rmse"],
                "baseline_mae": baseline_metric["mae"],
            }
        )
        out[f"{quantity}_mae"] = result["mae"]
        out[f"{quantity}_baseline_mae"] = baseline_metric["mae"]
        out[f"{quantity}_max_point_regression"] = float(
            np.max(np.abs(predicted - observed) - np.abs(baseline - observed))
        )
    return out


def _score_fig14(
    predictions: dict[str, dict[str, Any]], metrics: list[dict[str, Any]]
) -> dict[str, float]:
    observations = [
        row for row in _read_csv(FIG14_GT) if row["series"] == "scherer_1968_experiment"
    ]
    lookup = {
        (row["theta_max_deg"], row["phase_offset_deg"]): row
        for row in predictions.values()
    }

    def score(rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float]]:
        observed = np.asarray([float(row["ct"]) for row in rows])
        proxy = np.asarray(
            [
                lookup[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))][
                    "proxy_CT"
                ]
                for row in rows
            ]
        )
        baseline = np.asarray(
            [
                lookup[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))][
                    "baseline_CT"
                ]
                for row in rows
            ]
        )
        return _metric(observed, proxy), _metric(observed, baseline)

    all_result, all_baseline = score(observations)
    out = {
        "all14_rmse": all_result["rmse"],
        "all14_baseline_rmse": all_baseline["rmse"],
    }
    metrics.append(
        {
            "benchmark": "izraelevitz2017_fig14",
            "scope": "all_14_markers",
            "case_id": "all",
            "quantity": "CT",
            "view": "cycle_mean",
            "model": "fluxv_v5c1_proxy",
            "n": 14,
            **all_result,
            "baseline_rmse": all_baseline["rmse"],
            "baseline_mae": all_baseline["mae"],
        }
    )
    for theta in (15.0, 25.0):
        selected = [
            row
            for row in observations
            if np.isclose(float(row["theta_max_deg"]), theta)
        ]
        result, baseline = score(selected)
        out[f"theta_{theta:g}_rmse"] = result["rmse"]
        out[f"theta_{theta:g}_baseline_rmse"] = baseline["rmse"]
        metrics.append(
            {
                "benchmark": "izraelevitz2017_fig14",
                "scope": f"theta_{theta:g}_markers",
                "case_id": f"theta_{theta:g}",
                "quantity": "CT",
                "view": "cycle_mean",
                "model": "fluxv_v5c1_proxy",
                "n": len(selected),
                **result,
                "baseline_rmse": baseline["rmse"],
                "baseline_mae": baseline["mae"],
            }
        )
    grouped: dict[tuple[float, float], list[float]] = {}
    for row in observations:
        key = (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
        grouped.setdefault(key, []).append(float(row["ct"]))
    keys = sorted(grouped)
    observed = np.asarray([np.mean(grouped[key]) for key in keys])
    proxy = np.asarray([lookup[key]["proxy_CT"] for key in keys])
    baseline = np.asarray([lookup[key]["baseline_CT"] for key in keys])
    unique = _metric(observed, proxy)
    unique_baseline = _metric(observed, baseline)
    out["unique12_rmse"] = unique["rmse"]
    out["unique12_baseline_rmse"] = unique_baseline["rmse"]
    out["max_point_regression"] = float(
        np.max(np.abs(proxy - observed) - np.abs(baseline - observed))
    )
    metrics.append(
        {
            "benchmark": "izraelevitz2017_fig14",
            "scope": "unique_12_conditions",
            "case_id": "all",
            "quantity": "CT",
            "view": "cycle_mean",
            "model": "fluxv_v5c1_proxy",
            "n": 12,
            **unique,
            "baseline_rmse": unique_baseline["rmse"],
            "baseline_mae": unique_baseline["mae"],
        }
    )
    return out


def _baik_experiment(case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [row for row in _read_csv(BAIK_GT) if row["case"] == case_id]
    phase = np.asarray([float(row["phase_t_over_T"]) for row in rows])
    cl = np.asarray([float(row["cl"]) for row in rows])
    cd = np.asarray([float(row["cd"]) for row in rows])
    if phase.size == 401 and np.isclose(phase[-1], 1.0):
        return phase[:-1], cl[:-1], cd[:-1]
    raise ValueError(f"unexpected Baik ground-truth grid for {case_id}")


def _score_baik(
    predictions: dict[str, dict[str, Any]], metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    rmse: dict[tuple[str, str], float] = {}
    baseline_rmse: dict[tuple[str, str], float] = {}
    for case_id, case in BAIK_2012_CASES.items():
        observed_phase, observed_cl, observed_cd = _baik_experiment(case_id)
        prediction = predictions[case_id]
        phase = np.asarray(prediction["phase"])
        harmonic = case.experimental_filter_harmonic
        for quantity, observed in (("CL", observed_cl), ("CD", observed_cd)):
            proxy_history = sharp_fourier_lowpass(
                np.asarray(prediction[f"proxy_{quantity}"]), maximum_harmonic=harmonic
            )
            baseline_history = sharp_fourier_lowpass(
                np.asarray(prediction[f"baseline_{quantity}"]),
                maximum_harmonic=harmonic,
            )
            proxy = np.interp(observed_phase, phase, proxy_history, period=1.0)
            baseline = np.interp(observed_phase, phase, baseline_history, period=1.0)
            result = _metric(observed, proxy)
            base_result = _metric(observed, baseline)
            rmse[(case_id, quantity)] = result["rmse"]
            baseline_rmse[(case_id, quantity)] = base_result["rmse"]
            metrics.append(
                {
                    "benchmark": "baik2012",
                    "scope": "case_phase_history",
                    "case_id": case_id,
                    "quantity": quantity,
                    "view": "filtered_1hz",
                    "model": "fluxv_v5c1_proxy",
                    "n": 400,
                    **result,
                    "baseline_rmse": base_result["rmse"],
                    "baseline_mae": base_result["mae"],
                }
            )
    macro: dict[str, float] = {}
    for quantity in ("CL", "CD"):
        macro[quantity] = float(
            np.mean([rmse[(case_id, quantity)] for case_id in BAIK_2012_CASES])
        )
        macro[f"baseline_{quantity}"] = float(
            np.mean([baseline_rmse[(case_id, quantity)] for case_id in BAIK_2012_CASES])
        )
        metrics.append(
            {
                "benchmark": "baik2012",
                "scope": "macro_4_cases",
                "case_id": "all",
                "quantity": quantity,
                "view": "filtered_1hz",
                "model": "fluxv_v5c1_proxy",
                "n": 1600,
                "mae": "",
                "rmse": macro[quantity],
                "bias": "",
                "max_abs_error": "",
                "baseline_rmse": macro[f"baseline_{quantity}"],
                "baseline_mae": "",
            }
        )
    return {"rmse": rmse, "baseline_rmse": baseline_rmse, "macro": macro}


def _gate(
    gate_id: str, measured: float | bool, relation: str, threshold: float | bool
) -> dict[str, Any]:
    if relation == "<=":
        passed = float(measured) <= float(threshold)
    elif relation == "==":
        passed = measured == threshold
    elif relation == ">=":
        passed = float(measured) >= float(threshold)
    else:
        raise ValueError(f"unsupported gate relation: {relation}")
    return {
        "gate_id": gate_id,
        "measured": measured,
        "relation": relation,
        "threshold": threshold,
        "numeric_pass": bool(passed),
        "canonical_eligible": "false",
        "promotion_pass": "false",
    }


def _assert_source_frozen_parameters() -> dict[str, float | str]:
    manifest = DEFAULT_SUCTION_PARAMETERS.manifest()
    if manifest["state_pole_per_convective_time"] != 0.5:
        raise RuntimeError("v5c1 proxy requires the source-frozen pole 2*|b|=0.5")
    if manifest["observation_fit"] != "none":
        raise RuntimeError("v5c1 proxy parameters must be observation-free")
    return manifest


def run(output: Path) -> dict[str, Any]:
    """Build and score one frozen, all-22 non-canonical proxy run."""

    parameter_manifest = _assert_source_frozen_parameters()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    phase_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []

    # Prediction formulas use only frozen baselines/kinematics; scoring is separate.
    yang_predictions, yang_diagnostics = _yang_predictions(phase_rows, state_rows)
    fig_predictions, fig_diagnostics = _fig14_predictions(phase_rows, state_rows)
    baik_predictions, baik_diagnostics = _baik_predictions(phase_rows, state_rows)
    condition_count = (
        len(yang_predictions) + len(fig_predictions) + len(baik_predictions)
    )
    if condition_count != 22 or len(phase_rows) != 22 * OUTPUT_SAMPLES:
        raise RuntimeError("all-22 proxy coverage contract failed")

    metric_rows: list[dict[str, Any]] = []
    yang = _score_yang(yang_predictions, metric_rows)
    fig14 = _score_fig14(fig_predictions, metric_rows)
    baik = _score_baik(baik_predictions, metric_rows)
    v5c0_summary = json.loads(V5C0_SUMMARY.read_text(encoding="utf-8"))
    v5c0_metrics = v5c0_summary["corrected_v5c0_metrics"]

    diagnostics = (yang_diagnostics, fig_diagnostics, baik_diagnostics)
    mechanical = {
        "disabled_max_abs": max(row["disabled_max_abs"] for row in diagnostics),
        "normal_max_abs": max(row["normal_max_abs"] for row in diagnostics),
        "state_cycle_max_abs": max(row["state_cycle_max_abs"] for row in diagnostics),
        "owner_ledger_max_abs": max(
            row.get("owner_ledger_max_abs", 0.0) for row in diagnostics
        ),
        "yang_v4b_replay_max_abs_gf": yang_diagnostics["baseline_replay_max_abs"],
        "axial_suction_increase_max": max(
            float(row["target_CS"]) - float(row["separated_CSf"]) for row in state_rows
        ),
        "positive_delta_suction_max": max(
            float(row["delta_CS_2d"]) for row in state_rows
        ),
    }
    gates: list[dict[str, Any]] = [
        _gate("coverage_22_conditions", condition_count, "==", 22),
        _gate("phase_rows_2816", len(phase_rows), "==", 2816),
        _gate("disabled_identity", mechanical["disabled_max_abs"], "<=", 1.0e-12),
        _gate("normal_owner_unchanged", mechanical["normal_max_abs"], "<=", 1.0e-12),
        _gate(
            "owner_ledger_closure", mechanical["owner_ledger_max_abs"], "<=", 1.0e-12
        ),
        _gate(
            "axial_suction_magnitude_not_increased",
            mechanical["axial_suction_increase_max"],
            "<=",
            1.0e-12,
        ),
        _gate(
            "axial_suction_delta_nonpositive",
            mechanical["positive_delta_suction_max"],
            "<=",
            1.0e-12,
        ),
        _gate("periodic_state", mechanical["state_cycle_max_abs"], "<=", 1.0e-4),
        _gate(
            "yang_v4b_replay",
            mechanical["yang_v4b_replay_max_abs_gf"],
            "<=",
            1.0e-10,
        ),
        _gate("yang_lift_mae_gf", yang["lift_mae"], "<=", YANG_THRESHOLDS["lift"]),
        _gate("yang_drag_mae_gf", yang["drag_mae"], "<=", YANG_THRESHOLDS["drag"]),
        _gate(
            "yang_max_point_regression_gf",
            max(yang["lift_max_point_regression"], yang["drag_max_point_regression"]),
            "<=",
            0.4,
        ),
        _gate(
            "fig14_all14_rmse",
            fig14["all14_rmse"],
            "<=",
            v5c0_metrics["all_14_markers"]["rmse"],
        ),
        _gate(
            "fig14_theta15_rmse",
            fig14["theta_15_rmse"],
            "<=",
            v5c0_metrics["theta_15"]["rmse"],
        ),
        _gate(
            "fig14_theta25_rmse",
            fig14["theta_25_rmse"],
            "<=",
            v5c0_metrics["theta_25"]["rmse"],
        ),
        _gate(
            "fig14_unique12_rmse",
            fig14["unique12_rmse"],
            "<=",
            v5c0_metrics["unique_12_conditions"]["rmse"],
        ),
        _gate(
            "fig14_max_point_regression", fig14["max_point_regression"], "<=", 0.0112
        ),
        _gate(
            "baik_macro_CL_rmse", baik["macro"]["CL"], "<=", BAIK_MACRO_THRESHOLDS["CL"]
        ),
        _gate(
            "baik_macro_CD_rmse", baik["macro"]["CD"], "<=", BAIK_MACRO_THRESHOLDS["CD"]
        ),
    ]
    for key, threshold in BAIK_THRESHOLDS.items():
        gates.append(
            _gate(
                f"baik_{key[0]}_{key[1]}_rmse",
                baik["rmse"][key],
                "<=",
                threshold,
            )
        )
    gates.append(_gate("canonical_strip_inputs_available", False, "==", True))

    mechanical_ids = {
        "coverage_22_conditions",
        "phase_rows_2816",
        "disabled_identity",
        "normal_owner_unchanged",
        "owner_ledger_closure",
        "axial_suction_magnitude_not_increased",
        "axial_suction_delta_nonpositive",
        "periodic_state",
        "yang_v4b_replay",
    }
    mechanical_pass = all(
        row["numeric_pass"] for row in gates if row["gate_id"] in mechanical_ids
    )
    if not mechanical_pass:
        failed = [
            row["gate_id"]
            for row in gates
            if row["gate_id"] in mechanical_ids and not row["numeric_pass"]
        ]
        raise RuntimeError(f"v5c1 proxy mechanical stop: {failed}")
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
    numeric_paper_pass = all(paper_pass.values())
    status = (
        "proxy_numeric_pass_canonical_blocked"
        if numeric_paper_pass
        else "stopped_proxy_crosspaper_gate_failure"
    )

    condition_rows: list[dict[str, Any]] = []
    condition_rows.extend(
        {"benchmark": "yang2025", "case_id": case_id, **row}
        for case_id, row in yang_predictions.items()
    )
    condition_rows.extend(
        {"benchmark": "izraelevitz2017_fig14", "case_id": case_id, **row}
        for case_id, row in fig_predictions.items()
    )
    condition_rows.extend(
        {
            "benchmark": "baik2012",
            "case_id": case_id,
            "baseline_filtered_CL_rmse": baik["baseline_rmse"][(case_id, "CL")],
            "proxy_filtered_CL_rmse": baik["rmse"][(case_id, "CL")],
            "baseline_filtered_CD_rmse": baik["baseline_rmse"][(case_id, "CD")],
            "proxy_filtered_CD_rmse": baik["rmse"][(case_id, "CD")],
        }
        for case_id in BAIK_2012_CASES
    )
    for row in condition_rows:
        row["canonical_eligible"] = "false"

    phase_path = output / "phase_predictions.csv"
    state_path = output / "state_histories.csv"
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
        "promotion_status": "blocked_noncanonical_cache_proxy",
        "canonical_eligible": False,
        "condition_count": condition_count,
        "phase_row_count": len(phase_rows),
        "state_row_count": len(state_rows),
        "paper_gate_pass": paper_pass,
        "mechanical_metrics": mechanical,
        "headline": {"yang2025": yang, "figure14": fig14, "baik2012": baik["macro"]},
        "parameters": parameter_manifest,
        "parameter_selection_data": [],
        "observations_not_used_in_prediction": True,
        "model_semantics": (
            "non-canonical LDVM-state proxy: separated pre-constraint LESP drives "
            "the frozen rate state; separated CSf is the only replaceable force "
            "line; the correction is multiplied by the existing (1-p) owner"
        ),
        "stop_rule": (
            "any mechanical failure raises; any paper accuracy gate failure freezes "
            "this proxy as NO-GO; canonical promotion is impossible without a fresh "
            "same-time-layer UVLM strip export"
        ),
        "limitations": [
            "A0/LESP and velocity are kinematic paired-LDVM proxies without UVLM-induced strip velocity.",
            "Proxy delta_tau uses sqrt(1+(h_dot/U)^2)*U_inf*dt/c from the LDVM inputs; it omits same-layer induced velocity and any unresolved 0.75c pitch-point contribution.",
            "Yang and Baik retain frozen v4b; Figure 14 retains audited v5c0.",
            "The 22 development cases were previously inspected and are not held out.",
        ],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sources = (
        Path(__file__).resolve(),
        Path(__file__).with_name("fluxv_v5c_suction.py").resolve(),
        Path(__file__).with_name("baik2012.py").resolve(),
        Path(__file__).with_name("cases.py").resolve(),
        Path(__file__).with_name("causal_incidence_owner.py").resolve(),
        Path(__file__).with_name("ldvm_uvlm_correction.py").resolve(),
        Path(__file__).with_name("ptera_adapter.py").resolve(),
        Path(__file__).with_name("run_v4_crosspaper.py").resolve(),
        Path(__file__).with_name("uvlm_polar_correction.py").resolve(),
        REPO_ROOT / "platform/ldvm_fourier.py",
        REPO_ROOT / "platform/flap_ldvm.py",
        REPO_ROOT / "platform/tests/test_fluxv_v5c1_proxy.py",
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
    results = (
        phase_path,
        state_path,
        condition_path,
        metric_path,
        gate_path,
        summary_path,
    )
    manifest = {
        "schema_version": 2,
        "run_id": output.name,
        "cli": sys.argv,
        "evidence_level": "cache_proxy_no_uvlm_induced_strip_velocity",
        "canonical_eligible": False,
        "condition_contract": {"yang": 6, "figure14": 12, "baik": 4, "total": 22},
        "numerical_contract": {
            "output_samples": OUTPUT_SAMPLES,
            "state_total_cycles": STATE_TOTAL_CYCLES,
            "state_discarded_cycles": STATE_DISCARDED_CYCLES,
            "state_scored_cycles": STATE_SCORED_CYCLES,
            "proxy_delta_tau_provider": (
                "sqrt(1+(h_dot/U_inf)^2)*(U_inf/c)*dt from paired-LDVM "
                "kinematics; no UVLM-induced or unresolved 0.75c pitch-point velocity"
            ),
            "canonical_delta_tau_required": (
                "|V_rel at 0.75c perpendicular to span|*dt/c_local from the "
                "same UVLM time layer"
            ),
            "delta_tau_positive_fail_closed": True,
            "yang_ldvm_steps": YANG_LDVM_STEPS,
            "yang_strips": YANG_STRIPS,
            "figure14_ldvm_steps": FIG14_LDVM_STEPS,
            "baik_ldvm_steps": BAIK_LDVM_STEPS,
            "baik_wake_steps": BAIK_WAKE_STEPS,
        },
        "parameters": parameter_manifest,
        "parameter_selection_data": [],
        "environment": {
            "python": sys.version,
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "pterasoftware": _package_version("pterasoftware"),
        },
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sources
        },
        "result_hashes": {path.name: _sha256(path) for path in results},
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
