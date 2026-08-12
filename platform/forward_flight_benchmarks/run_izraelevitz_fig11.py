"""Verified reproduction run for Izraelevitz 2017 Figure 11."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cases import IZRAELEVITZ_2017_FIG11 as CASE
from .ptera_adapter import (
    MODEL_SEMANTICS,
    build_izraelevitz_fig11_movement,
    run_model,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/runs/20260807_rigid_firstpass/izraelevitz_fig11_exact"
)
MODELS = (
    "fluxv_uvpm",
    "ptera_prescribed_wake_uvlm",
    "ptera_free_wake_uvlm",
)
COLORS = {
    "fluxv_uvpm": "#0057b8",
    "ptera_prescribed_wake_uvlm": "#8f55b5",
    "ptera_free_wake_uvlm": "#e4572e",
}
LABELS = {
    "fluxv_uvpm": "FluxV UVPM (load channel)",
    "ptera_prescribed_wake_uvlm": "Prescribed-wake UVLM",
    "ptera_free_wake_uvlm": "Free-wake UVLM",
}
CONVERGENCE = {
    "g0_2x7_24x4": (2, 7, 24, 4),
    "g1_3x9_24x4": (3, 9, 24, 4),
    "g2_4x12_24x4_production": (4, 12, 24, 4),
    "t1_4x12_30x4": (4, 12, 30, 4),
    "t2_4x12_32x4_last_stable": (4, 12, 32, 4),
}


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _run(output: Path) -> dict[str, Any]:
    results: dict[str, Any] = {"models": {}, "convergence": {}, "failures": {}}
    for model in MODELS:
        print(f"RUN Figure11 {model}", flush=True)
        movement, setup = build_izraelevitz_fig11_movement("full")
        history = run_model(
            movement,
            model,
            period_s=CASE.period_s,
            rho=CASE.rho_kg_m3,
            speed=CASE.freestream_m_s,
            area=CASE.area_m2,
        )
        magnitude = float(max(np.max(np.abs(history["CL"])), np.max(np.abs(history["CT"]))))
        if not np.isfinite(magnitude) or magnitude > 1.0e3:
            results["failures"][model] = f"numerical divergence: max coefficient={magnitude:.6e}"
            continue
        history["setup"] = setup
        results["models"][model] = history
        print(
            f"DONE Figure11 {model}: CL={history['mean_CL']:.6f} "
            f"CT={history['mean_CT']:.6f}",
            flush=True,
        )
    for name, settings in CONVERGENCE.items():
        print(f"RUN convergence {name}", flush=True)
        movement, setup = build_izraelevitz_fig11_movement("full", settings=settings)
        history = run_model(
            movement,
            "ptera_prescribed_wake_uvlm",
            period_s=CASE.period_s,
            rho=CASE.rho_kg_m3,
            speed=CASE.freestream_m_s,
            area=CASE.area_m2,
        )
        history["setup"] = setup
        results["convergence"][name] = history
        print(
            f"DONE convergence {name}: CL={history['mean_CL']:.6f} "
            f"CT={history['mean_CT']:.6f}",
            flush=True,
        )
    reference = results["convergence"]["g2_4x12_24x4_production"]
    for history in results["convergence"].values():
        history["rms_CL_vs_production"] = float(
            np.sqrt(np.mean((np.asarray(history["CL"]) - np.asarray(reference["CL"])) ** 2))
        )
        history["rms_CT_vs_production"] = float(
            np.sqrt(np.mean((np.asarray(history["CT"]) - np.asarray(reference["CT"])) ** 2))
        )
    fluxv = results["models"].get("fluxv_uvpm")
    control = results["models"].get("ptera_prescribed_wake_uvlm")
    results["audit"] = {
        "fluxv_control_max_abs_difference": None
        if fluxv is None or control is None
        else float(
            max(
                np.max(np.abs(np.asarray(fluxv["CL"]) - np.asarray(control["CL"]))),
                np.max(np.abs(np.asarray(fluxv["CT"]) - np.asarray(control["CT"]))),
            )
        ),
        "production_vs_previous_grid_rms_CL": results["convergence"]["g1_3x9_24x4"]["rms_CL_vs_production"],
        "production_vs_previous_grid_rms_CT": results["convergence"]["g1_3x9_24x4"]["rms_CT_vs_production"],
        "production_vs_last_stable_time_rms_CL": results["convergence"]["t2_4x12_32x4_last_stable"]["rms_CL_vs_production"],
        "production_vs_last_stable_time_rms_CT": results["convergence"]["t2_4x12_32x4_last_stable"]["rms_CT_vs_production"],
        "production_mean_CT": reference["mean_CT"],
        "last_stable_time_mean_CT": results["convergence"]["t2_4x12_32x4_last_stable"]["mean_CT"],
        "known_instability_onset": "4x12 grid diverges at >=36 steps/cycle; do not claim temporal convergence",
    }
    return results


def _write_csv(output: Path, results: dict[str, Any]) -> None:
    with (output / "phase_histories.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("model", "phase", "CL", "CT", "CL_alpha", "CD_alpha")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        scale = np.sin(np.deg2rad(CASE.downstroke_midpoint_alpha_deg))
        for model, history in results["models"].items():
            for phase, cl, ct in zip(history["phase"], history["CL"], history["CT"]):
                writer.writerow(
                    {
                        "model": model,
                        "phase": phase,
                        "CL": cl,
                        "CT": ct,
                        "CL_alpha": cl / scale,
                        "CD_alpha": -ct / scale,
                    }
                )
    with (output / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "configuration",
            "chord_panels",
            "semispan_panels",
            "steps_per_cycle",
            "cycles",
            "mean_CL",
            "mean_CT",
            "rms_CL_vs_production",
            "rms_CT_vs_production",
            "runtime_s",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, history in results["convergence"].items():
            setup = history["setup"]
            writer.writerow(
                {
                    "configuration": name,
                    "chord_panels": setup["grid_chord_semispan"][0],
                    "semispan_panels": setup["grid_chord_semispan"][1],
                    "steps_per_cycle": setup["steps_per_cycle"],
                    "cycles": setup["cycles"],
                    "mean_CL": history["mean_CL"],
                    "mean_CT": history["mean_CT"],
                    "rms_CL_vs_production": history["rms_CL_vs_production"],
                    "rms_CT_vs_production": history["rms_CT_vs_production"],
                    "runtime_s": history["runtime_s"],
                }
            )


def _plot(output: Path, results: dict[str, Any]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scale = np.sin(np.deg2rad(CASE.downstroke_midpoint_alpha_deg))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for model, history in results["models"].items():
        phase = history["phase"]
        axes[0, 0].plot(phase, history["CL"], color=COLORS[model], label=LABELS[model])
        axes[1, 0].plot(phase, history["CT"], color=COLORS[model], label=LABELS[model])
        axes[0, 1].plot(phase, np.asarray(history["CL"]) / scale, color=COLORS[model], label=LABELS[model])
        axes[1, 1].plot(phase, -np.asarray(history["CT"]) / scale, color=COLORS[model], label=LABELS[model])
    axes[0, 0].set_ylabel("standard CL")
    axes[1, 0].set_ylabel("standard CT")
    axes[0, 1].set_ylabel(r"paper-scaled $C_{L\alpha}$")
    axes[1, 1].set_ylabel(r"paper-scaled $C_{D\alpha}$")
    for axis in axes[1, :]:
        axis.set_xlabel("cycle phase t/T")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("Izraelevitz 2017 Figure 11 — AR=3 heave-pitch rigid wing")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures / "fig11_existing_models_phase_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, history in results["convergence"].items():
        axes[0].plot(history["phase"], history["CL"], label=name)
        axes[1].plot(history["phase"], history["CT"], label=name)
    axes[0].set_ylabel("CL")
    axes[1].set_ylabel("CT")
    axes[1].set_xlabel("cycle phase t/T")
    axes[0].set_title("Figure 11 prescribed-wake discretization audit")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "fig11_convergence.png", dpi=220)
    plt.close(fig)

    y = np.linspace(-CASE.semispan_m, CASE.semispan_m, 500)
    chord = CASE.chord_m(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].fill_between(y, -0.25 * chord, 0.75 * chord, color="#9ecae1")
    axes[0].axhline(0.0, color="k", ls="--", lw=0.8, label="quarter-chord axis")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("span y [m]")
    axes[0].set_ylabel("x [m]")
    axes[0].set_title("Analytic elliptic AR=3 planform")
    axes[0].legend(fontsize=8)
    phase = np.linspace(0.0, 1.0, 500)
    axes[1].plot(phase, CASE.heave_amplitude_m * np.cos(2 * np.pi * phase) / CASE.midspan_chord_m, label="z/cmid")
    axes[1].plot(phase, CASE.pitch_amplitude_deg * np.sin(2 * np.pi * phase) / 30.0, label="theta/30°")
    axes[1].set_xlabel("cycle phase t/T")
    axes[1].set_ylabel("normalized motion")
    axes[1].set_title("Published heave-pitch phase relation")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fig11_geometry_kinematics.png", dpi=220)
    plt.close(fig)


def _write_report(output: Path, results: dict[str, Any]) -> None:
    audit = results["audit"]
    lines = [
        "# Izraelevitz 2017 Figure 11 verified reconstruction",
        "",
        "This is the primary passing Izraelevitz case. It replaces Figure 13 for accuracy work until the latter's unpublished 3-D rotation convention is closed.",
        "",
        "## Exact paper inputs",
        "",
        f"- Elliptic flat-plate center surface, AR={CASE.aspect_ratio:g}, straight quarter-chord axis.",
        f"- h/cmid={CASE.heave_to_chord:g}, St={CASE.strouhal:g}, kmid={CASE.reduced_frequency_midspan:.6f}.",
        f"- z=h cos(omega t), theta=theta_max sin(omega t), theta_max={CASE.pitch_amplitude_deg:.6f} deg about quarter chord.",
        "- c_mid=0.1 m and U=1 m/s are similarity scales; all paper nondimensional parameters are preserved.",
        "",
        "## Numerical audit",
        "",
        f"- FluxV versus its prescribed-wake load control, maximum full-history coefficient difference: `{audit['fluxv_control_max_abs_difference']:.3e}`.",
        f"- Production grid versus previous grid RMS: CL `{audit['production_vs_previous_grid_rms_CL']:.4f}`, CT `{audit['production_vs_previous_grid_rms_CT']:.4f}`.",
        f"- Production versus last stable time-step RMS: CL `{audit['production_vs_last_stable_time_rms_CL']:.4f}`, CT `{audit['production_vs_last_stable_time_rms_CT']:.4f}`.",
        f"- Mean CT changes from `{audit['production_mean_CT']:.4f}` at 24 steps/cycle to `{audit['last_stable_time_mean_CT']:.4f}` at 32; at 36 and above the Ptera wake solution diverges. The case is numerically finite but not formally time-converged.",
        "",
        "## Model means",
        "",
        "| model | mean CL | mean CT | runtime [s] |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        history = results["models"].get(model)
        if history:
            lines.append(
                f"| {model} | {history['mean_CL']:.6f} | {history['mean_CT']:.6f} | {history['runtime_s']:.2f} |"
            )
    lines.extend(
        [
            "",
            "The FluxV row invokes the actual UVPMHybridSolver. Its particles do not feed back into loads in the current implementation, so equality with the prescribed-wake UVLM control is expected and not independent validation.",
            "",
            "Paper Figure 11 compares its state-space ULLT with the authors' UVLM, not experiment. The current output therefore establishes a reproducible geometry/kinematics and existing-model baseline; it is not yet a paper-model accuracy score.",
        ]
    )
    if results["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{key}`: {value}" for key, value in results["failures"].items())
    (output / "VERIFICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = _run(args.output_dir)
    payload = {
        "case": CASE.manifest(),
        "model_semantics": MODEL_SEMANTICS,
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=_default) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir, results)
    _plot(args.output_dir, results)
    _write_report(args.output_dir, results)
    print(f"OUTPUT {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
