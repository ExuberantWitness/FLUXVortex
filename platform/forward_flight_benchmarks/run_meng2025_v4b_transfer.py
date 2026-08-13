"""Audit frozen FluxV transfer behavior on Meng et al. (2025) Figure 16."""

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

from .ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
    project_ldvm_delta_to_finite_wing,
    run_ldvm_separation_pair,
)
from .meng2025_case import MENG_2025, STANDARD_GRAVITY_M_S2, build_meng2025_movement
from .ptera_adapter import run_model
from .uvlm_polar_correction import movement_polar_residual


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA = (
    Path(__file__).resolve().parent
    / "source_data/meng2025_fig16_mean_lines_digitized.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "meng2025_fluxv_transfer_20260813/runs/20260813_fig16_smoke"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _observations() -> dict[float, dict[str, float]]:
    with SOURCE_DATA.open(newline="", encoding="utf-8") as stream:
        return {
            float(row["twist_amplitude_peak_to_peak_deg"]): {
                "mean_lift_gf": float(row["mean_lift_gf"]),
                "mean_net_thrust_gf": float(row["mean_net_thrust_gf"]),
                "lift_digitization_uncertainty_gf": float(
                    row["lift_digitization_uncertainty_gf"]
                ),
                "thrust_digitization_uncertainty_gf": float(
                    row["thrust_digitization_uncertainty_gf"]
                ),
            }
            for row in csv.DictReader(stream)
        }


def _periodic_resample(values: np.ndarray, output_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    phase = np.arange(values.size, dtype=float) / values.size
    target = np.arange(output_samples, dtype=float) / output_samples
    return np.interp(target, phase, values, period=1.0)


def _periodic_derivative(values: np.ndarray, step: float) -> np.ndarray:
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * step)


def _strip_quadrature(strip_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strip_count < 2:
        raise ValueError("strip_count must be at least two")
    case = MENG_2025
    edges = np.linspace(0.0, case.half_span_m, strip_count + 1)
    radii = 0.5 * (edges[:-1] + edges[1:])
    chords = case.chord_m(radii)
    full_wing_area = 2.0 * chords * np.diff(edges)
    full_wing_area *= case.area_m2 / np.sum(full_wing_area)
    return radii, chords, full_wing_area


def _unsupported_ldvm_stress(
    twist_pp_deg: float, *, steps_per_cycle: int, strip_count: int
) -> dict[str, Any]:
    """Run a rejected Yang-threshold transfer only as a convergence stress test."""

    case = MENG_2025
    cycles = 2
    phase = np.arange(cycles * steps_per_cycle, dtype=float) / steps_per_cycle
    tau = 2.0 * np.pi * phase
    flap = np.deg2rad(0.5 * case.flap_amplitude_peak_to_peak_deg * np.cos(tau))
    pitch = np.deg2rad(-0.5 * twist_pp_deg * np.sin(tau))
    radii, chords, strip_areas = _strip_quadrature(strip_count)
    selected = slice(steps_per_cycle, 2 * steps_per_cycle)
    delta_lift = np.zeros(steps_per_cycle)
    delta_drag = np.zeros(steps_per_cycle)
    shedding = np.zeros(steps_per_cycle)
    threshold = LESPThreshold(
        value=float(np.sin(np.deg2rad(5.0))),
        section_family="Meng unreported thin rigid membrane; Yang thin-plate transfer",
        reynolds=case.reynolds_mean,
        source=(
            "unchanged Yang-2025 mapping Lcrit=sin(alpha_sep), alpha_sep=5 deg; "
            "Meng publishes no section threshold"
        ),
        source_role="unsupported cross-section transfer hypothesis; no Meng force fit",
    )
    q_inf = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2
    for radius, chord, area in zip(radii, chords, strip_areas, strict=True):
        omega_star = 2.0 * np.pi * case.frequency_hz * chord / case.freestream_m_s
        alpha = np.deg2rad(case.installation_aoa_deg) + pitch
        alpha_rate = -0.5 * np.deg2rad(twist_pp_deg) * omega_star * np.cos(tau)
        flap_rate = (
            -0.5
            * np.deg2rad(case.flap_amplitude_peak_to_peak_deg)
            * omega_star
            * np.sin(tau)
        )
        heave_rate_over_u = -(radius / chord) * flap_rate
        pair = run_ldvm_separation_pair(
            alpha_rad=alpha,
            alpha_rate_per_convective_time=alpha_rate,
            heave_rate_over_u=heave_rate_over_u,
            delta_time_convective=(
                case.period_s * case.freestream_m_s / chord / steps_per_cycle
            ),
            pivot_fraction_chord=case.main_spar_fraction_root_chord,
            threshold=threshold,
            settings=LDVMSectionSettings(
                ndiv=50, naterm=24, max_wake_steps=steps_per_cycle
            ),
        )
        projection = project_ldvm_delta_to_finite_wing(
            np.asarray(pair["delta"]["CNc"])[selected],
            np.asarray(pair["delta"]["CNnc"])[selected],
            np.asarray(pair["delta"]["CNnonl"])[selected],
            np.asarray(pair["delta"]["CSf"])[selected],
            alpha[selected],
            aspect_ratio=case.aspect_ratio,
        )
        delta_lift += (
            q_inf * area * np.asarray(projection["delta_CL"]) * np.cos(flap[selected])
        )
        delta_drag += q_inf * area * np.asarray(projection["delta_CD"])
        shedding += (
            area / case.area_m2 * np.asarray(pair["shed_lev"], dtype=float)[selected]
        )
    return {
        "steps_per_cycle": steps_per_cycle,
        "strip_count": strip_count,
        "mean_delta_lift_n": float(np.mean(delta_lift)),
        "mean_delta_drag_n": float(np.mean(delta_drag)),
        "max_abs_delta_lift_n": float(np.max(np.abs(delta_lift))),
        "mean_shedding_fraction": float(np.mean(shedding)),
        "threshold": threshold.manifest(),
    }


def _run_baseline(
    twist_pp_deg: float, *, quality: str, output_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case = MENG_2025
    movement, movement_manifest = build_meng2025_movement(twist_pp_deg, quality=quality)
    old = run_model(
        movement,
        "fluxv_uvpm",
        period_s=case.period_s,
        rho=case.rho_kg_m3,
        speed=case.freestream_m_s,
        area=case.area_m2,
        output_samples=output_samples,
    )
    polar = movement_polar_residual(
        movement,
        source_cycle_step_range=old["source_cycle_step_range"],
        period_s=case.period_s,
        freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3,
        aspect_ratio=case.aspect_ratio,
        output_samples=output_samples,
    )
    old_lift = np.asarray(old["lift_n"], dtype=float)
    old_thrust = np.asarray(old["thrust_n"], dtype=float)
    polar_lift = old_lift + np.asarray(polar["delta_lift_n"], dtype=float)
    polar_thrust = old_thrust - np.asarray(polar["delta_drag_n"], dtype=float)
    to_gf = 1000.0 / STANDARD_GRAVITY_M_S2
    summary = {
        "twist_amplitude_peak_to_peak_deg": float(twist_pp_deg),
        "old_mean_lift_gf": float(np.mean(old_lift) * to_gf),
        "old_mean_net_thrust_gf": float(np.mean(old_thrust) * to_gf),
        "polar_mean_lift_gf": float(np.mean(polar_lift) * to_gf),
        "polar_mean_net_thrust_gf": float(np.mean(polar_thrust) * to_gf),
        "polar_mean_delta_lift_n": float(np.mean(polar["delta_lift_n"])),
        "polar_mean_delta_drag_n": float(np.mean(polar["delta_drag_n"])),
        "runtime_s": float(old["runtime_s"]),
        "movement_manifest": movement_manifest,
    }
    phases: list[dict[str, Any]] = []
    for index, phase in enumerate(np.asarray(old["phase"], dtype=float)):
        phases.append(
            {
                "twist_amplitude_peak_to_peak_deg": twist_pp_deg,
                "phase": float(phase),
                "old_lift_n": float(old_lift[index]),
                "old_net_thrust_n": float(old_thrust[index]),
                "polar_lift_n": float(polar_lift[index]),
                "polar_net_thrust_n": float(polar_thrust[index]),
            }
        )
    return summary, phases


def _plot_means(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    obs = _observations()
    twist = np.asarray(sorted(obs))
    predicted_twist = np.asarray(
        [row["twist_amplitude_peak_to_peak_deg"] for row in rows]
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), squeeze=False)
    for axis, quantity, ylabel in (
        (axes[0, 0], "lift", "Mean lift (gf)"),
        (axes[0, 1], "net_thrust", "Mean net thrust (gf)"),
    ):
        axis.errorbar(
            twist,
            [obs[value][f"mean_{quantity}_gf"] for value in twist],
            yerr=[
                obs[value][
                    f"{quantity.replace('net_', '')}_digitization_uncertainty_gf"
                ]
                for value in twist
            ],
            fmt="o-",
            color="black",
            capsize=3,
            label="Figure 16 balance mean",
        )
        axis.plot(
            predicted_twist,
            [row[f"old_mean_{quantity}_gf"] for row in rows],
            "s--",
            color="tab:blue",
            label="FluxV old (pure-wing)",
        )
        axis.plot(
            predicted_twist,
            [row[f"polar_mean_{quantity}_gf"] for row in rows],
            "^-",
            color="tab:orange",
            label="FluxV old + frozen polar",
        )
        axis.set_xlabel("Published twist amplitude, peak-to-peak (deg)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Meng 2025 Figure 16: definition-qualified mean-load diagnostic")
    fig.tight_layout()
    png = output / "meng2025_fig16_mean_loads.png"
    pdf = output / "meng2025_fig16_mean_loads.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--twists", type=float, nargs="+", default=(0.0, 22.5, 45.0))
    parser.add_argument("--stress-steps", type=int, nargs="+", default=(48, 96, 128))
    parser.add_argument("--stress-strips", type=int, default=8)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mean_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for twist in args.twists:
        print(f"running Meng Figure 16 twist={twist:g} deg", flush=True)
        result, phases = _run_baseline(twist, quality=args.quality, output_samples=128)
        observation = _observations().get(twist)
        if observation is not None:
            result.update(
                {f"experiment_{key}": value for key, value in observation.items()}
            )
        mean_rows.append(result)
        phase_rows.extend(phases)

    stress_rows: list[dict[str, Any]] = []
    for steps in args.stress_steps:
        print(f"running unsupported LDVM stress at {steps} steps/cycle", flush=True)
        try:
            stress_rows.append(
                {
                    "status": "invalid_unconverged_cross_section_transfer",
                    **_unsupported_ldvm_stress(
                        22.5, steps_per_cycle=steps, strip_count=args.stress_strips
                    ),
                }
            )
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            stress_rows.append(
                {
                    "status": f"failed:{type(error).__name__}",
                    "steps_per_cycle": steps,
                    "strip_count": args.stress_strips,
                    "error": str(error),
                }
            )

    means_path = output / "mean_comparison.csv"
    phases_path = output / "phase_histories.csv"
    stress_path = output / "unsupported_ldvm_stress.csv"
    _write_csv(means_path, mean_rows)
    _write_csv(phases_path, phase_rows)
    _write_csv(stress_path, stress_rows)
    figures = _plot_means(mean_rows, output)
    result_files = (means_path, phases_path, stress_path, *figures)
    summary = {
        "run_id": output.name,
        "quality": args.quality,
        "status": "definition_qualified_diagnostic_full_v4b_rejected",
        "twists_peak_to_peak_deg": list(args.twists),
        "force_contract": (
            "model is pure-wing aerodynamic load; experiment is balance-level net "
            "thrust with unreported support/mechanism and wind-off inertial tare"
        ),
        "conclusion": (
            "old and frozen polar branches are reportable; the forced Yang LESP "
            "threshold is unsupported and non-convergent, so no full-v4b score is made"
        ),
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("meng2025_case.py").resolve(),
                Path(__file__).with_name("ldvm_uvlm_correction.py").resolve(),
                Path(__file__).with_name("uvlm_polar_correction.py").resolve(),
                SOURCE_DATA,
            )
        },
        "result_hashes": {path.name: _sha256(path) for path in result_files},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(mean_rows, indent=2, default=str))


if __name__ == "__main__":
    main()
