"""Run the exploratory causal LDVM/UVLM v4 across Yang and Figure 14.

The model has two mutually exclusive baseline owners plus one discrepancy:

1. retained UVLM plus a separated-minus-attached Ramesh LDVM increment for
   transient/reversing flow; and
2. retained UVLM/full-angle polar for persistent installed incidence.

One-state ULLT remains the attached-flow reference for Figure 11; it is not
added to the UVLM/LDVM transient force ledger.

The persistent weight is produced by the causal two-pole incidence state in
``causal_incidence_owner``.  This runner evaluates periodic cycle means; it
does not claim that the independent 2-D section wake is already a fully
coupled three-dimensional UVLM LEV row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from forward_flight_benchmarks.cases import (
    IZRAELEVITZ_2017_FIG14_SCHERER,
    IZRAELEVITZ_2017_FIG11,
    YANG_2025,
    fourbar_flap_angle_deg,
    fourbar_zero_phase_rad,
)
from forward_flight_benchmarks.causal_incidence_owner import (
    causal_incidence_persistence,
)
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
    project_ldvm_delta_to_finite_wing,
    run_ldvm_separation_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "unified_fluxv_upgrade_20260812"
)
DOC_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "unified_fluxv_v4_ldvm_stevens_20260812"
)
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260812_fluxv_v4_crosspaper_development"

YANG_BASE = (
    V3_ROOT / "runs/20260812_periodic_v2_ullt_full/yang2025_mean_characteristics.csv"
)
YANG_PHASE_BASE = (
    V3_ROOT / "runs/20260812_periodic_v2_ullt_full/yang2025_phase_histories.csv"
)
FIG14_BASE = (
    V3_ROOT / "runs/20260812_scherer_fig14_experiment_full/mean_thrust_vs_phase.csv"
)
FIG14_SOURCE = V3_ROOT / "source_data/izraelevitz2017_fig14_digitized.csv"
FIG14_PHASE_CACHE = DOC_ROOT / "source_data/izraelevitz2017_fig14_local_phase_cache.csv"
FIG11_METRICS = V3_ROOT / "runs/20260812_periodic_v2_ullt_full/accuracy_metrics.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
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


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs": float(np.max(np.abs(error))),
    }


def _periodic_resample(values: np.ndarray, output_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("periodic history must be a one-dimensional array")
    source_phase = np.arange(values.size, dtype=float) / values.size
    target_phase = np.arange(output_samples, dtype=float) / output_samples
    return np.interp(target_phase, source_phase, values, period=1.0)


def _phase_history(
    rows: list[dict[str, str]],
    model: str,
    *,
    aoa_deg: float | None = None,
    condition: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["model"] == model]
    if aoa_deg is not None:
        selected = [
            row for row in selected if np.isclose(float(row["aoa_deg"]), aoa_deg)
        ]
    if condition is not None:
        theta, psi = condition
        selected = [
            row
            for row in selected
            if np.isclose(float(row["theta_max_deg"]), theta)
            and np.isclose(float(row["phase_offset_deg"]), psi)
        ]
    selected.sort(key=lambda row: float(row["phase"]))
    if not selected:
        raise ValueError(f"no phase history for {model}")
    output = {"phase": np.asarray([float(row["phase"]) for row in selected])}
    for key in ("lift_n", "drag_n", "CL", "CD", "CT"):
        if key in selected[0] and selected[0][key] != "":
            output[key] = np.asarray([float(row[key]) for row in selected])
    return output


def _yang_kinematics(
    aoa_deg: float,
    *,
    steps_per_cycle: int,
    cycles: int,
    strip_count: int,
) -> dict[str, np.ndarray | float]:
    case = YANG_2025
    phase = np.arange(cycles * steps_per_cycle) * 2.0 * np.pi / steps_per_cycle
    flap = np.deg2rad(
        fourbar_flap_angle_deg(phase + fourbar_zero_phase_rad(case), case)
    )
    convective_period = case.freestream_m_s * case.period_s / case.chord_m
    delta_time = convective_period / steps_per_cycle
    flap_rate = np.gradient(np.unwrap(flap), delta_time, edge_order=2)
    radii = (
        case.wing_root_offset_m
        + (np.arange(strip_count, dtype=float) + 0.5) * case.span_m / strip_count
    )

    # Effective quarter-chord section incidence used by the persistent owner.
    alpha_install = np.deg2rad(aoa_deg)
    effective_alpha = np.empty((phase.size, strip_count))
    for time_index, (angle, rate) in enumerate(zip(flap, flap_rate)):
        ca, sa = np.cos(angle), np.sin(angle)
        ci, si = np.cos(alpha_install), np.sin(alpha_install)
        rotation = np.array(
            [
                [ci, 0.0, si],
                [sa * si, ca, -sa * ci],
                [-ca * si, sa, ca * ci],
            ]
        )
        chord_hat = rotation[:, 0]
        span_hat = rotation[:, 1]
        normal_hat = np.cross(chord_hat, span_hat)
        for strip_index, radius in enumerate(radii):
            # ``rate`` is d(phi)/d(t U/c), so use coordinates normalized by
            # chord and a unit nondimensional freestream in this cross product.
            point = rotation @ np.array([0.0, radius / case.chord_m, 0.0])
            surface_velocity = np.cross(np.array([rate, 0.0, 0.0]), point)
            relative = np.array([1.0, 0.0, 0.0]) - surface_velocity
            relative -= np.dot(relative, span_hat) * span_hat
            effective_alpha[time_index, strip_index] = np.arctan2(
                np.dot(relative, normal_hat), np.dot(relative, chord_hat)
            )
    return {
        "phase": phase,
        "flap_rad": flap,
        "flap_rate_per_convective_time": flap_rate,
        "radii_m": radii,
        "effective_alpha_rad": effective_alpha,
        "delta_time_convective": delta_time,
        "convective_period": convective_period,
    }


def _yang_ldvm_phase_and_ownership(
    aoa_deg: float,
    *,
    steps_per_cycle: int,
    strip_count: int,
) -> dict[str, np.ndarray | float]:
    # Twelve previous cycles make the slow -0.045 state effectively periodic;
    # every update remains causal and sees only repeated past kinematics.
    owner_cycles = 12
    history = _yang_kinematics(
        aoa_deg,
        steps_per_cycle=steps_per_cycle,
        cycles=owner_cycles,
        strip_count=strip_count,
    )
    owner = causal_incidence_persistence(
        np.asarray(history["effective_alpha_rad"]),
        delta_time_convective=float(history["delta_time_convective"]),
    )
    selected_owner = np.asarray(owner["global_persistence"])[-steps_per_cycle:]

    # LDVM itself needs only two cycles for the current diagnostic; cap wake
    # history to one cycle to avoid presenting independent strips as a 3-D wake.
    ldvm_history = _yang_kinematics(
        aoa_deg,
        steps_per_cycle=steps_per_cycle,
        cycles=2,
        strip_count=strip_count,
    )
    case = YANG_2025
    threshold = LESPThreshold(
        value=float(np.sin(np.deg2rad(case.separation_aoa_deg))),
        section_family="1 mm balsa flat plate",
        reynolds=case.reynolds,
        source=(
            "Yang 2025 published alpha_sep=5 deg mapped as the explicit "
            "hypothesis Lcrit=sin(alpha_sep); not fitted to Figure 11 loads"
        ),
        source_role="paper-parameter mapping hypothesis",
    )
    alpha_install = np.deg2rad(aoa_deg)
    delta_cl = np.zeros(steps_per_cycle)
    delta_cd = np.zeros(steps_per_cycle)
    shedding = np.zeros(steps_per_cycle)
    selected = slice(steps_per_cycle, 2 * steps_per_cycle)
    for radius in np.asarray(ldvm_history["radii_m"]):
        pair = run_ldvm_separation_pair(
            alpha_rad=np.full(2 * steps_per_cycle, alpha_install),
            alpha_rate_per_convective_time=np.zeros(2 * steps_per_cycle),
            heave_rate_over_u=(
                -(radius / case.chord_m)
                * np.asarray(ldvm_history["flap_rate_per_convective_time"])
            ),
            delta_time_convective=float(ldvm_history["delta_time_convective"]),
            pivot_fraction_chord=0.25,
            threshold=threshold,
            settings=LDVMSectionSettings(
                ndiv=32,
                naterm=14,
                max_wake_steps=steps_per_cycle,
            ),
        )
        projection = project_ldvm_delta_to_finite_wing(
            np.asarray(pair["delta"]["CNc"])[selected],
            np.asarray(pair["delta"]["CNnc"])[selected],
            np.asarray(pair["delta"]["CNnonl"])[selected],
            np.asarray(pair["delta"]["CSf"])[selected],
            np.full(steps_per_cycle, alpha_install),
            aspect_ratio=case.aspect_ratio,
        )
        # The section lift direction rotates with the flapping plane.  Its
        # global vertical projection is cos(phi); streamwise drag is unchanged.
        flap = np.asarray(ldvm_history["flap_rad"])[selected]
        delta_cl += np.asarray(projection["delta_CL"]) * np.cos(flap) / strip_count
        delta_cd += np.asarray(projection["delta_CD"]) / strip_count
        shedding += np.asarray(pair["shed_lev"], dtype=float)[selected] / strip_count
    return {
        "persistence": selected_owner,
        "ldvm_delta_CL": delta_cl,
        "ldvm_delta_CD": delta_cd,
        "shedding": shedding,
        "persistence_fraction": float(np.mean(selected_owner)),
        "mean_ldvm_delta_CL": float(np.mean(delta_cl)),
        "mean_ldvm_delta_CD": float(np.mean(delta_cd)),
        "ldvm_shedding_fraction": float(np.mean(shedding)),
        "lesp_critical": threshold.value,
    }


def _run_yang(
    steps_per_cycle: int, strip_count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _rows(YANG_BASE)
    phase_rows = _rows(YANG_PHASE_BASE)
    by_model = {
        model: {float(row["aoa_deg"]): row for row in rows if row["model"] == model}
        for model in (
            "wind_tunnel_test",
            "fluxv_uvpm",
            "fluxv_periodic_v1",
            "one_state_ullt_local",
        )
    }
    q_area_per_gf = (
        0.5
        * YANG_2025.rho_kg_m3
        * YANG_2025.freestream_m_s**2
        * YANG_2025.area_m2
        / 0.00980665
    )
    output: list[dict[str, Any]] = []
    errors_l: list[float] = []
    errors_d: list[float] = []
    for aoa in sorted(by_model["wind_tunnel_test"]):
        diagnostic = _yang_ldvm_phase_and_ownership(
            aoa, steps_per_cycle=steps_per_cycle, strip_count=strip_count
        )
        output_samples = 128
        old = _phase_history(phase_rows, "fluxv_uvpm", aoa_deg=aoa)
        polar = _phase_history(phase_rows, "fluxv_periodic_v1", aoa_deg=aoa)
        persistence = _periodic_resample(
            np.asarray(diagnostic["persistence"]), output_samples
        )
        shedding = _periodic_resample(
            np.asarray(diagnostic["shedding"]), output_samples
        )
        delta_l = (
            _periodic_resample(np.asarray(diagnostic["ldvm_delta_CL"]), output_samples)
            * q_area_per_gf
        )
        delta_d = (
            _periodic_resample(np.asarray(diagnostic["ldvm_delta_CD"]), output_samples)
            * q_area_per_gf
        )
        old_l_history = np.asarray(old["lift_n"]) / 0.00980665
        old_d_history = np.asarray(old["drag_n"]) / 0.00980665
        polar_l_history = np.asarray(polar["lift_n"]) / 0.00980665
        polar_d_history = np.asarray(polar["drag_n"]) / 0.00980665
        # UVLM remains the sole transient baseline owner.  The paired LDVM
        # separated-minus-attached term is a discrepancy increment and is
        # exactly zero before LESP onset.  ULLT stays an external attached-flow
        # comparator/Figure-11 limit rather than being added as another owner.
        transient_l = old_l_history + delta_l
        transient_d = old_d_history + delta_d
        v4_l_history = (1.0 - persistence) * transient_l + persistence * polar_l_history
        v4_d_history = (1.0 - persistence) * transient_d + persistence * polar_d_history
        v4_l = float(np.mean(v4_l_history))
        v4_d = float(np.mean(v4_d_history))
        old_l = float(np.mean(old_l_history))
        old_d = float(np.mean(old_d_history))
        polar_l = float(np.mean(polar_l_history))
        polar_d = float(np.mean(polar_d_history))
        test_l = float(by_model["wind_tunnel_test"][aoa]["mean_lift_gf"])
        test_d = float(by_model["wind_tunnel_test"][aoa]["mean_drag_gf"])
        errors_l.append(v4_l - test_l)
        errors_d.append(v4_d - test_d)
        output.append(
            {
                "aoa_deg": aoa,
                "test_lift_gf": test_l,
                "test_drag_gf": test_d,
                "old_fluxv_lift_gf": old_l,
                "old_fluxv_drag_gf": old_d,
                "v1_polar_lift_gf": polar_l,
                "v1_polar_drag_gf": polar_d,
                "v4_lift_gf": v4_l,
                "v4_drag_gf": v4_d,
                "persistence_fraction": float(np.mean(persistence)),
                "ldvm_shedding_fraction": float(np.mean(shedding)),
                "ldvm_delta_CL": float(np.mean(delta_l / q_area_per_gf)),
                "ldvm_delta_CD": float(np.mean(delta_d / q_area_per_gf)),
                "lesp_critical": diagnostic["lesp_critical"],
            }
        )
    return output, {
        "lift": _metrics(np.zeros(len(errors_l)), np.asarray(errors_l)),
        "drag": _metrics(np.zeros(len(errors_d)), np.asarray(errors_d)),
    }


def _fig14_effective_alpha(
    theta_deg: float, psi_deg: float, phase: np.ndarray
) -> np.ndarray:
    case = IZRAELEVITZ_2017_FIG14_SCHERER
    theta = np.deg2rad(theta_deg) * np.cos(phase + np.deg2rad(psi_deg))
    heave_velocity_over_u = -np.pi * case.strouhal * np.sin(phase)
    return theta + np.arctan2(-heave_velocity_over_u, 1.0)


def _fig14_phase_diagnostic(
    theta_deg: float,
    psi_deg: float,
    *,
    steps_per_cycle: int,
) -> dict[str, np.ndarray | float]:
    case = IZRAELEVITZ_2017_FIG14_SCHERER
    omega_star = np.pi * case.strouhal / case.heave_to_chord
    period_star = 2.0 * np.pi / omega_star
    delta_time = period_star / steps_per_cycle
    owner_cycles = 20
    phase_owner = (
        np.arange(owner_cycles * steps_per_cycle) * 2.0 * np.pi / steps_per_cycle
    )
    alpha_effective = _fig14_effective_alpha(theta_deg, psi_deg, phase_owner)
    owner = causal_incidence_persistence(
        alpha_effective, delta_time_convective=delta_time
    )
    persistence = np.asarray(owner["global_persistence"])[-steps_per_cycle:]

    phase = np.arange(4 * steps_per_cycle) * 2.0 * np.pi / steps_per_cycle
    alpha = np.deg2rad(theta_deg) * np.cos(phase + np.deg2rad(psi_deg))
    alpha_rate = (
        -np.deg2rad(theta_deg) * omega_star * np.sin(phase + np.deg2rad(psi_deg))
    )
    heave_rate = -case.heave_to_chord * omega_star * np.sin(phase)
    alpha_stall = np.deg2rad(0.90 / 0.065)
    threshold = LESPThreshold(
        value=float(np.sin(alpha_stall)),
        section_family=case.section_name,
        reynolds=case.freestream_m_s * case.chord_m / case.nu_m2_s,
        source=(
            "Scherer static CLa=0.065/deg and CLmax=0.90; "
            "Lcrit hypothesis=sin(CLmax/CLa)"
        ),
        source_role="static-polar-derived hypothesis; no Figure-14 force fit",
    )
    pair = run_ldvm_separation_pair(
        alpha_rad=alpha,
        alpha_rate_per_convective_time=alpha_rate,
        heave_rate_over_u=heave_rate,
        delta_time_convective=delta_time,
        pivot_fraction_chord=case.pivot_fraction_chord,
        threshold=threshold,
        settings=LDVMSectionSettings(
            ndiv=50,
            naterm=24,
            max_wake_steps=4 * steps_per_cycle,
        ),
    )
    selected = slice(3 * steps_per_cycle, 4 * steps_per_cycle)
    projection = project_ldvm_delta_to_finite_wing(
        np.asarray(pair["delta"]["CNc"])[selected],
        np.asarray(pair["delta"]["CNnc"])[selected],
        np.asarray(pair["delta"]["CNnonl"])[selected],
        np.asarray(pair["delta"]["CSf"])[selected],
        alpha[selected],
        aspect_ratio=case.aspect_ratio,
    )
    return {
        "persistence": persistence,
        "ldvm_delta_CT": -np.asarray(projection["delta_CD"]),
        "shedding": np.asarray(pair["shed_lev"], dtype=float)[selected],
        "persistence_fraction": float(np.mean(persistence)),
        "mean_ldvm_delta_CT": -float(np.mean(projection["delta_CD"])),
        "ldvm_shedding_fraction": float(
            np.mean(np.asarray(pair["shed_lev"], dtype=float)[selected])
        ),
        "lesp_critical": threshold.value,
    }


def _run_fig14(steps_per_cycle: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = _rows(FIG14_BASE)
    phase_rows = _rows(FIG14_PHASE_CACHE)
    observations = [
        row for row in _rows(FIG14_SOURCE) if row["series"] == "scherer_1968_experiment"
    ]
    base: dict[str, dict[tuple[float, float], float]] = {}
    for model in ("fluxv_uvpm", "fluxv_periodic_v1", "one_state_ullt_local"):
        base[model] = {
            (float(row["theta_max_deg"]), float(row["phase_offset_deg"])): float(
                row["CT"]
            )
            for row in base_rows
            if row["series"] == model
        }
    output: list[dict[str, Any]] = []
    predictions: dict[tuple[float, float], float] = {}
    for condition in sorted(base["fluxv_uvpm"]):
        theta, psi = condition
        diagnostic = _fig14_phase_diagnostic(
            theta, psi, steps_per_cycle=steps_per_cycle
        )
        output_samples = 128
        old_history = np.asarray(
            _phase_history(phase_rows, "fluxv_uvpm", condition=condition)["CT"]
        )
        persistent_history = np.asarray(
            _phase_history(phase_rows, "fluxv_periodic_v1", condition=condition)["CT"]
        )
        p = _periodic_resample(np.asarray(diagnostic["persistence"]), output_samples)
        shedding = _periodic_resample(
            np.asarray(diagnostic["shedding"]), output_samples
        )
        delta_ct = _periodic_resample(
            np.asarray(diagnostic["ldvm_delta_CT"]), output_samples
        )
        transient = old_history + delta_ct
        prediction_history = (1.0 - p) * transient + p * persistent_history
        prediction = float(np.mean(prediction_history))
        persistent = float(np.mean(persistent_history))
        predictions[condition] = prediction
        output.append(
            {
                "theta_max_deg": theta,
                "phase_offset_deg": psi,
                "old_fluxv_CT": base["fluxv_uvpm"][condition],
                "attached_ullt_CT": base["one_state_ullt_local"][condition],
                "persistent_polar_CT": persistent,
                "v4_CT": prediction,
                "persistence_fraction": float(np.mean(p)),
                "ldvm_delta_CT": float(np.mean(delta_ct)),
                "ldvm_shedding_fraction": float(np.mean(shedding)),
                "lesp_critical": diagnostic["lesp_critical"],
            }
        )
    observed = np.asarray([float(row["ct"]) for row in observations])
    predicted = np.asarray(
        [
            predictions[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))]
            for row in observations
        ]
    )
    return output, _metrics(observed, predicted)


def _fig11_onset_audit() -> dict[str, Any]:
    """Verify that the source NACA0012 threshold leaves Figure 11 attached."""

    case = IZRAELEVITZ_2017_FIG11
    steps_per_cycle = 256
    cycles = 4
    phase = np.arange(cycles * steps_per_cycle) * 2.0 * np.pi / steps_per_cycle
    omega_star = np.pi * case.strouhal / case.heave_to_chord
    period_star = 2.0 * np.pi / omega_star
    angle_amplitude = np.deg2rad(abs(case.pitch_amplitude_deg))
    alpha = -angle_amplitude * np.sin(phase)
    alpha_rate = -angle_amplitude * omega_star * np.cos(phase)
    heave_rate = -case.heave_to_chord * omega_star * np.sin(phase)
    threshold = LESPThreshold(
        value=0.29,
        section_family="NACA 0012",
        reynolds=10_000.0,
        source="Martinez-Carmena et al. SVDVM Case II published Lcrit",
    )
    pair = run_ldvm_separation_pair(
        alpha_rad=alpha,
        alpha_rate_per_convective_time=alpha_rate,
        heave_rate_over_u=heave_rate,
        delta_time_convective=period_star / steps_per_cycle,
        pivot_fraction_chord=0.25,
        threshold=threshold,
        settings=LDVMSectionSettings(
            ndiv=50,
            naterm=24,
            max_wake_steps=2 * steps_per_cycle,
        ),
    )
    selected = slice((cycles - 1) * steps_per_cycle, cycles * steps_per_cycle)
    return {
        "lesp_critical": threshold.value,
        "max_abs_pre_cap_lesp": float(
            np.max(np.abs(np.asarray(pair["separated"]["lesp"])[selected]))
        ),
        "material_lev_count": int(pair["lev_count"][-1]),
        "shedding_fraction": float(
            np.mean(np.asarray(pair["shed_lev"], dtype=float)[selected])
        ),
        "branch": "attached_one_state_ullt",
    }


def _plot(
    yang: list[dict[str, Any]], fig14: list[dict[str, Any]], output: Path
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    aoa = np.asarray([row["aoa_deg"] for row in yang])
    for axis, channel in zip(axes[:2], ("lift", "drag")):
        axis.errorbar(
            aoa,
            [row[f"test_{channel}_gf"] for row in yang],
            yerr=0.4,
            color="black",
            marker="o",
            label="Yang wind-tunnel (digitized)",
        )
        axis.plot(
            aoa,
            [row[f"old_fluxv_{channel}_gf"] for row in yang],
            "--",
            label="FluxV old",
        )
        axis.plot(
            aoa, [row[f"v4_{channel}_gf"] for row in yang], "-s", label="FluxV v4"
        )
        axis.set_xlabel("Installation angle [deg]")
        axis.set_ylabel(f"Mean {channel} [gf]")
        axis.grid(alpha=0.25)
    for theta, marker in ((15.0, "o"), (25.0, "s")):
        selected = [row for row in fig14 if row["theta_max_deg"] == theta]
        axes[2].plot(
            [row["phase_offset_deg"] for row in selected],
            [row["old_fluxv_CT"] for row in selected],
            "--",
            marker=marker,
            label=f"old, theta={theta:g} deg",
        )
        axes[2].plot(
            [row["phase_offset_deg"] for row in selected],
            [row["v4_CT"] for row in selected],
            "-",
            marker=marker,
            label=f"v4, theta={theta:g} deg",
        )
    observations = [
        row for row in _rows(FIG14_SOURCE) if row["series"] == "scherer_1968_experiment"
    ]
    for theta, marker in ((15.0, "o"), (25.0, "s")):
        selected = [row for row in observations if float(row["theta_max_deg"]) == theta]
        axes[2].errorbar(
            [float(row["phase_offset_deg"]) for row in selected],
            [float(row["ct"]) for row in selected],
            yerr=[
                [float(row["ct_error_minus"]) for row in selected],
                [float(row["ct_error_plus"]) for row in selected],
            ],
            color="black",
            linestyle="none",
            marker=marker,
            alpha=0.75,
        )
    axes[2].set_xlabel("Heave-pitch phase offset [deg]")
    axes[2].set_ylabel(r"Mean thrust coefficient $C_T$")
    axes[2].grid(alpha=0.25)
    for axis in axes:
        axis.legend(frameon=False, fontsize=7.4)
    fig.tight_layout()
    png = output / "fluxv_v4_yang_fig14_comparison.png"
    pdf = output / "fluxv_v4_yang_fig14_comparison.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps-per-cycle", type=int, default=128)
    parser.add_argument("--yang-strips", type=int, default=12)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    yang_rows, yang_metrics = _run_yang(args.steps_per_cycle, args.yang_strips)
    fig14_rows, fig14_metrics = _run_fig14(args.steps_per_cycle)
    fig11_onset = _fig11_onset_audit()
    yang_path = output / "yang2025_v4_mean_characteristics.csv"
    fig14_path = output / "izraelevitz2017_fig14_v4_mean_thrust.csv"
    _write_csv(yang_path, yang_rows)
    _write_csv(fig14_path, fig14_rows)
    figures = _plot(yang_rows, fig14_rows, output)
    fig11_rows = [
        row
        for row in _rows(FIG11_METRICS)
        if row["paper"] == "izraelevitz2017_fig11"
        and row["model"] == "fluxv_periodic_v2"
    ]
    summary = {
        "run_id": output.name,
        "status": "development_mechanism_diagnostic",
        "yang_metrics": yang_metrics,
        "fig14_metrics": fig14_metrics,
        "fig11_no_onset_exact_v2_reference": fig11_rows,
        "fig11_ldvm_onset_audit": fig11_onset,
        "steps_per_cycle": args.steps_per_cycle,
        "yang_strip_count": args.yang_strips,
        "model_semantics": (
            "phase-resolved causal two-pole incidence owner blends a retained "
            "UVLM+paired-LDVM discrepancy branch with a retained "
            "UVLM+full-angle-polar persistent branch; one-state ULLT is the "
            "separate Figure-11 attached-flow limit"
        ),
        "limitations": [
            "LDVM wake remains independent 2-D strips, not a coupled 3-D LEV row",
            "paired LDVM is an additive discrepancy, not a conservative common-wake coupling",
            "Yang uses nominal four-bar rather than unpublished LDS motion",
            "LESP mappings are source-derived hypotheses and require new confirmation",
            "current LDVM time step is a development discretization, not convergence",
        ],
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("causal_incidence_owner.py").resolve(),
                Path(__file__).with_name("ldvm_uvlm_correction.py").resolve(),
                REPO_ROOT / "platform/ldvm_fourier.py",
                YANG_BASE,
                YANG_PHASE_BASE,
                FIG14_BASE,
                FIG14_SOURCE,
                FIG14_PHASE_CACHE,
                FIG11_METRICS,
            )
        },
        "result_hashes": {
            path.name: _sha256(path) for path in (yang_path, fig14_path, *figures)
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"yang": yang_metrics, "fig14": fig14_metrics}, indent=2))


if __name__ == "__main__":
    main()
