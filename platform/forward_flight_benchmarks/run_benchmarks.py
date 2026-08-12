"""Run and document the Izraelevitz 2017/Yang 2023 rigid-wing benchmarks."""

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

from .cases import (
    IZRAELEVITZ_2017,
    YANG_2023,
    fourbar_extrema_deg,
    fourbar_flap_angle_deg,
    fourbar_zero_phase_rad,
    izraelevitz_tip_alpha_history,
)
from .ptera_adapter import (
    MODEL_SEMANTICS,
    build_izraelevitz_movement,
    build_yang_movement,
    run_model,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/source_data/yang2023_fig7_rigid_digitized.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/runs/20260807_rigid_firstpass"
)
GF_TO_N = 0.00980665
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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)}")


def _read_yang_reference() -> list[dict[str, Any]]:
    with SOURCE_DATA.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["aoa_deg"] = float(row["aoa_deg"])
        row["value_gf"] = float(row["value_gf"])
        row["digitization_uncertainty_gf"] = float(row["digitization_uncertainty_gf"])
    return rows


def _reference_curve(
    rows: list[dict[str, Any]], role: str, observable: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["source_role"] == role and row["observable"] == observable
    ]
    selected.sort(key=lambda row: row["aoa_deg"])
    return (
        np.asarray([row["aoa_deg"] for row in selected]),
        np.asarray([row["value_gf"] for row in selected]),
    )


def _run_matrix(quality: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: dict[str, Any] = {"yang": {}, "izraelevitz": {}, "failures": {}}
    history_rows: list[dict[str, Any]] = []
    for aoa in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        aoa_key = f"{aoa:g}"
        results["yang"][aoa_key] = {}
        for model in MODELS:
            print(f"RUN Yang AoA={aoa:g} model={model}", flush=True)
            try:
                movement, setup = build_yang_movement(aoa, quality)
                history = run_model(
                    movement,
                    model,
                    period_s=YANG_2023.period_s,
                    rho=YANG_2023.rho_kg_m3,
                    speed=YANG_2023.freestream_m_s,
                    area=YANG_2023.area_m2,
                )
                history["setup"] = setup
                results["yang"][aoa_key][model] = history
                for index, phase in enumerate(history["phase"]):
                    history_rows.append(
                        {
                            "case_id": YANG_2023.case_id,
                            "aoa_deg": aoa,
                            "model": model,
                            "phase": phase,
                            "lift_n": history["lift_n"][index],
                            "thrust_n": history["thrust_n"][index],
                            "CL": history["CL"][index],
                            "CT": history["CT"][index],
                        }
                    )
                print(
                    f"DONE Yang AoA={aoa:g} {model}: "
                    f"L={history['mean_lift_n']/GF_TO_N:.3f} gf "
                    f"T={history['mean_thrust_n']/GF_TO_N:.3f} gf",
                    flush=True,
                )
            except Exception as error:  # preserve the failed matrix cell
                key = f"yang_aoa_{aoa:g}_{model}"
                results["failures"][key] = f"{type(error).__name__}: {error}"
                print(f"FAILED {key}: {error}", flush=True)
    for model in MODELS:
        print(f"RUN Izraelevitz model={model}", flush=True)
        try:
            movement, setup = build_izraelevitz_movement(quality)
            history = run_model(
                movement,
                model,
                period_s=IZRAELEVITZ_2017.period_s,
                rho=IZRAELEVITZ_2017.rho_kg_m3,
                speed=IZRAELEVITZ_2017.freestream_m_s,
                area=IZRAELEVITZ_2017.area_m2,
            )
            history["setup"] = setup
            results["izraelevitz"][model] = history
            for index, phase in enumerate(history["phase"]):
                history_rows.append(
                    {
                        "case_id": IZRAELEVITZ_2017.case_id,
                        "aoa_deg": "",
                        "model": model,
                        "phase": phase,
                        "lift_n": history["lift_n"][index],
                        "thrust_n": history["thrust_n"][index],
                        "CL": history["CL"][index],
                        "CT": history["CT"][index],
                    }
                )
            print(
                f"DONE Izraelevitz {model}: CL={history['mean_CL']:.4f} "
                f"CT={history['mean_CT']:.4f}",
                flush=True,
            )
        except Exception as error:
            key = f"izraelevitz_{model}"
            results["failures"][key] = f"{type(error).__name__}: {error}"
            print(f"FAILED {key}: {error}", flush=True)
    return results, history_rows


def _sanitize_results(results: dict[str, Any]) -> dict[str, Any]:
    """Remove numerically divergent cells while preserving a loud diagnostic."""

    failures = results.setdefault("failures", {})
    for aoa_key, cells in list(results.get("yang", {}).items()):
        for model, history in list(cells.items()):
            magnitude = float(
                max(np.max(np.abs(history["CL"])), np.max(np.abs(history["CT"])))
            )
            if not np.isfinite(magnitude) or magnitude > 1.0e3:
                failures[f"yang_aoa_{aoa_key}_{model}"] = (
                    f"numerical divergence: max(|CL|,|CT|)={magnitude:.6e}"
                )
                del cells[model]
    for model, history in list(results.get("izraelevitz", {}).items()):
        magnitude = float(
            max(np.max(np.abs(history["CL"])), np.max(np.abs(history["CT"])))
        )
        if not np.isfinite(magnitude) or magnitude > 1.0e3:
            failures[f"izraelevitz_{model}"] = (
                f"numerical divergence: max(|CL|,|CT|)={magnitude:.6e}"
            )
            del results["izraelevitz"][model]
    return results


def _history_rows_from_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aoa_key, cells in results.get("yang", {}).items():
        for model, history in cells.items():
            for index, phase in enumerate(history["phase"]):
                rows.append(
                    {
                        "case_id": YANG_2023.case_id,
                        "aoa_deg": float(aoa_key),
                        "model": model,
                        "phase": phase,
                        "lift_n": history["lift_n"][index],
                        "thrust_n": history["thrust_n"][index],
                        "CL": history["CL"][index],
                        "CT": history["CT"][index],
                    }
                )
    for model, history in results.get("izraelevitz", {}).items():
        for index, phase in enumerate(history["phase"]):
            rows.append(
                {
                    "case_id": IZRAELEVITZ_2017.case_id,
                    "aoa_deg": "",
                    "model": model,
                    "phase": phase,
                    "lift_n": history["lift_n"][index],
                    "thrust_n": history["thrust_n"][index],
                    "CL": history["CL"][index],
                    "CT": history["CT"][index],
                }
            )
    return rows


def _calculate_audit(results: dict[str, Any], reference: list[dict[str, Any]]) -> dict[str, Any]:
    audit: dict[str, Any] = {"yang_error_vs_rigid_test": {}, "fluxv_load_identity": {}}
    for model in MODELS:
        predicted_lift, predicted_thrust = [], []
        for aoa in (0, 5, 10, 15, 20, 25):
            cell = results["yang"].get(str(aoa), {}).get(model)
            if cell is None:
                continue
            predicted_lift.append(cell["mean_lift_n"] / GF_TO_N)
            predicted_thrust.append(cell["mean_thrust_n"] / GF_TO_N)
        if len(predicted_lift) == 6:
            _, observed_lift = _reference_curve(reference, "rigid_test", "lift")
            _, observed_thrust = _reference_curve(reference, "rigid_test", "thrust")
            lift_error = np.asarray(predicted_lift) - observed_lift
            thrust_error = np.asarray(predicted_thrust) - observed_thrust
            audit["yang_error_vs_rigid_test"][model] = {
                "lift_mae_gf": float(np.mean(np.abs(lift_error))),
                "lift_rmse_gf": float(np.sqrt(np.mean(lift_error**2))),
                "thrust_mae_gf": float(np.mean(np.abs(thrust_error))),
                "thrust_rmse_gf": float(np.sqrt(np.mean(thrust_error**2))),
                "predicted_lift_gf": predicted_lift,
                "predicted_thrust_gf": predicted_thrust,
            }
    # FluxV and prescribed UVLM should be identical because that is FluxV's
    # current load channel. Audit the complete histories, not just their means.
    comparisons = []
    for aoa in (0, 5, 10, 15, 20, 25):
        cells = results["yang"].get(str(aoa), {})
        if all(name in cells for name in ("fluxv_uvpm", "ptera_prescribed_wake_uvlm")):
            fluxv = cells["fluxv_uvpm"]
            control = cells["ptera_prescribed_wake_uvlm"]
            comparisons.extend(np.abs(np.asarray(fluxv["CL"]) - np.asarray(control["CL"])))
            comparisons.extend(np.abs(np.asarray(fluxv["CT"]) - np.asarray(control["CT"])))
    cells = results["izraelevitz"]
    if all(name in cells for name in ("fluxv_uvpm", "ptera_prescribed_wake_uvlm")):
        for coefficient in ("CL", "CT"):
            comparisons.extend(
                np.abs(
                    np.asarray(cells["fluxv_uvpm"][coefficient])
                    - np.asarray(cells["ptera_prescribed_wake_uvlm"][coefficient])
                )
            )
    audit["fluxv_load_identity"] = {
        "max_abs_coefficient_difference": float(max(comparisons)) if comparisons else None,
        "expected": "roundoff-level equality",
    }
    phase, alpha = izraelevitz_tip_alpha_history()
    audit["izraelevitz_rotation_convention"] = {
        "reconstructed_tip_three_quarter_chord_alpha_min_deg": float(alpha.min()),
        "reconstructed_tip_three_quarter_chord_alpha_max_deg": float(alpha.max()),
        "paper_reported_alpha_min_deg": IZRAELEVITZ_2017.paper_alpha_min_deg,
        "paper_reported_alpha_max_deg": IZRAELEVITZ_2017.paper_alpha_max_deg,
        "status": (
            "rotation convention not uniquely closed by paper text; this diagnostic "
            "difference is retained and no kinematic fitting was performed"
        ),
    }
    return audit


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("case_id", "aoa_deg", "model", "phase", "lift_n", "thrust_n", "CL", "CT")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_geometries(output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    y = np.linspace(-IZRAELEVITZ_2017.semispan_m, IZRAELEVITZ_2017.semispan_m, 500)
    chord = IZRAELEVITZ_2017.chord_m(y)
    axes[0].fill_between(y, -0.25 * chord, 0.75 * chord, color="#9ecae1", alpha=0.8)
    axes[0].plot(y, -0.25 * chord, color="#0057b8")
    axes[0].plot(y, 0.75 * chord, color="#0057b8")
    axes[0].axhline(0.0, color="k", lw=0.7, ls="--", label="straight quarter chord")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("span y [m]")
    axes[0].set_ylabel("x from quarter chord [m]")
    axes[0].set_title("Analytic elliptic AR=6 planform")
    axes[0].legend(fontsize=8)

    phase, alpha = izraelevitz_tip_alpha_history()
    axes[1].plot(phase, alpha, color="#0057b8", label="reconstructed convention")
    axes[1].axhline(IZRAELEVITZ_2017.paper_alpha_min_deg, color="k", ls=":")
    axes[1].axhline(IZRAELEVITZ_2017.paper_alpha_max_deg, color="k", ls=":", label="paper range")
    axes[1].set_xlabel("cycle phase t/T")
    axes[1].set_ylabel(r"tip $\alpha_{3/4c}$ [deg]")
    axes[1].set_title("Kinematic convention audit")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "izraelevitz2017_geometry_kinematics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].fill(
        [0.0, YANG_2023.span_m, YANG_2023.span_m, 0.0],
        [0.0, 0.0, YANG_2023.chord_m, YANG_2023.chord_m],
        color="#a1d99b",
        edgecolor="#238b45",
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("single-wing span [m]")
    axes[0].set_ylabel("chord [m]")
    axes[0].set_title("130 × 250 mm rigid rectangle")
    q = np.linspace(0.0, 2.0 * np.pi, 600)
    theta = fourbar_flap_angle_deg(q + fourbar_zero_phase_rad(), YANG_2023)
    axes[1].plot(q / (2.0 * np.pi), theta, color="#238b45")
    lower, upper = fourbar_extrema_deg(YANG_2023)
    axes[1].axhline(-15.0, color="k", ls=":")
    axes[1].axhline(45.0, color="k", ls=":", label="paper targets")
    axes[1].scatter([q[np.argmin(theta)] / (2 * np.pi), q[np.argmax(theta)] / (2 * np.pi)], [lower, upper], color="#d95f0e")
    axes[1].set_xlabel("crank cycle phase")
    axes[1].set_ylabel("wing flap angle [deg]")
    axes[1].set_title("Exact four-bar reconstruction")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "yang2023_rigid_geometry_kinematics.png", dpi=220)
    plt.close(fig)


def _plot_results(output: Path, results: dict[str, Any], reference: list[dict[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    for model in MODELS:
        history = results["izraelevitz"].get(model)
        if history is None:
            continue
        axes[0].plot(history["phase"], history["CL"], color=COLORS[model], label=LABELS[model])
        axes[1].plot(history["phase"], history["CT"], color=COLORS[model], label=LABELS[model])
    axes[0].set_ylabel("CL")
    axes[1].set_ylabel("CT")
    axes[1].set_xlabel("cycle phase t/T")
    axes[0].set_title("Izraelevitz 2017 Figure 13 case — existing-model histories")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "izraelevitz2017_fig13_existing_models.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, observable in zip(axes, ("lift", "thrust")):
        x_test, y_test = _reference_curve(reference, "rigid_test", observable)
        x_sim, y_sim = _reference_curve(reference, "rigid_sim", observable)
        axis.plot(x_test, y_test, "ko--", label="paper rigid test (digitized)")
        axis.plot(x_sim, y_sim, marker="*", color="#777777", label="paper PLEV simulation (digitized)")
        for model in MODELS:
            values = []
            for aoa in (0, 5, 10, 15, 20, 25):
                cell = results["yang"].get(str(aoa), {}).get(model)
                key = "mean_lift_n" if observable == "lift" else "mean_thrust_n"
                values.append(np.nan if cell is None else cell[key] / GF_TO_N)
            axis.plot(x_test, values, marker="s", ms=4, color=COLORS[model], label=LABELS[model])
        axis.set_xlabel("installation angle [deg]")
        axis.set_ylabel(f"cycle-average {observable} [gf]")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("Yang 2023 rigid wing — Figure 7 reconstruction and current models")
    fig.tight_layout()
    fig.savefig(figures / "yang2023_fig7_rigid_existing_models.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(6, 2, figsize=(11, 17), sharex=True)
    for row, aoa in enumerate((0, 5, 10, 15, 20, 25)):
        cells = results["yang"].get(str(aoa), {})
        for model in MODELS:
            history = cells.get(model)
            if history is None:
                continue
            axes[row, 0].plot(history["phase"], np.asarray(history["lift_n"]) / GF_TO_N, color=COLORS[model], label=LABELS[model])
            axes[row, 1].plot(history["phase"], np.asarray(history["thrust_n"]) / GF_TO_N, color=COLORS[model], label=LABELS[model])
        axes[row, 0].set_ylabel(f"AoA {aoa}°\nlift [gf]")
        axes[row, 1].set_ylabel(f"AoA {aoa}°\nthrust [gf]")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("cycle phase t/T")
    axes[-1, 1].set_xlabel("cycle phase t/T")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    fig.suptitle("Yang 2023 rigid wing — complete last-cycle load histories")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(figures / "yang2023_all_conditions_phase_loads.png", dpi=220)
    plt.close(fig)


def _write_report(output: Path, quality: str, payload: dict[str, Any]) -> None:
    audit = payload["audit"]
    yang_metrics = audit["yang_error_vs_rigid_test"]
    lines = [
        "# Rigid-wing reproduction verification report",
        "",
        f"Run quality: `{quality}`.",
        "",
        "## Reconstructed cases",
        "",
        "- Izraelevitz 2017 Figure 13: analytic elliptic AR=6 planform, two independently represented half-wings, St=0.3, 30° flap, -30° linear tip twist and beta=75°. The paper case is nondimensional; c_mid=0.1 m and U=1 m/s are a stated similarity scaling.",
        "- Yang 2023 Figure 7: single 0.130×0.250 m rigid rectangular wing, U=5.5 m/s, f=2.5 Hz, and all six installation angles. The published four-bar dimensions recover -14.991° to 45.010° without fitting.",
        "",
        "## Model interpretation",
        "",
        "`fluxv_uvpm` invokes the actual FluxV solver class. Its present load channel is prescribed-ring UVLM; particles are one-way wake state. Therefore equality to the prescribed-wake control is expected and is an implementation audit, not independent model agreement.",
        "",
        f"Maximum full-history FluxV/control coefficient difference: `{audit['fluxv_load_identity']['max_abs_coefficient_difference']:.3e}`.",
        "",
        "## Yang error against digitized rigid-wing wind-tunnel means",
        "",
        "| model | lift MAE [gf] | thrust MAE [gf] |",
        "|---|---:|---:|",
    ]
    for model in MODELS:
        metric = yang_metrics.get(model)
        if metric:
            lines.append(f"| {model} | {metric['lift_mae_gf']:.3f} | {metric['thrust_mae_gf']:.3f} |")
    convention = audit["izraelevitz_rotation_convention"]
    lines.extend(
        [
            "",
            "## Important limitations",
            "",
            "- Yang's modified UVLM+PLEV cannot be reproduced exactly from this conference paper: the PLEV circulation closure, core coefficient, mesh and time step are not published. Its plotted simulation is included only as digitized context.",
            f"- Izraelevitz's text gives beta and scalar motion laws but no rotation matrix. The explicit tilted-axis convention used here produces tip 3/4-chord alpha `{convention['reconstructed_tip_three_quarter_chord_alpha_min_deg']:.2f}..{convention['reconstructed_tip_three_quarter_chord_alpha_max_deg']:.2f}°`, versus the paper's `{convention['paper_reported_alpha_min_deg']:.1f}..{convention['paper_reported_alpha_max_deg']:.1f}°`. No phase, angle or amplitude was fitted to hide this unresolved convention.",
            "- The Izraelevitz paper validation target is its UVLM calculation, not experiment. These curves are therefore a cross-solver numerical comparison, not an experimental accuracy claim.",
            "- Inviscid UVLM does not reproduce Yang's PLEV/separation physics; the error table is diagnostic, not a calibrated final model result.",
            "",
            "## Files",
            "",
            "- `case_manifest.json`: exact inputs and conventions.",
            "- `metrics.json`: means, errors, runtimes, failures and model semantics.",
            "- `all_phase_histories.csv`: every plotted numerical history.",
            "- `figures/`: geometry, kinematics, phase histories and Figure 7 comparison.",
        ]
    )
    if payload["results"]["failures"]:
        lines.extend(["", "## Explicitly failed matrix cells", ""])
        for key, reason in payload["results"]["failures"].items():
            lines.append(f"- `{key}`: {reason}")
    (output / "VERIFICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="re-audit and regenerate an existing metrics.json without rerunning solvers",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = _read_yang_reference()
    metrics_path = args.output_dir / "metrics.json"
    if args.reuse_existing:
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        results = previous["results"]
    else:
        results, _ = _run_matrix(args.quality)
    results = _sanitize_results(results)
    history_rows = _history_rows_from_results(results)
    audit = _calculate_audit(results, reference)
    payload = {
        "quality": args.quality,
        "model_semantics": MODEL_SEMANTICS,
        "results": results,
        "audit": audit,
        "source_data": str(SOURCE_DATA),
    }
    manifest = {
        "izraelevitz2017": IZRAELEVITZ_2017.manifest(),
        "yang2023": YANG_2023.manifest(),
        "model_semantics": MODEL_SEMANTICS,
    }
    (args.output_dir / "case_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    _write_history_csv(args.output_dir / "all_phase_histories.csv", history_rows)
    _plot_geometries(args.output_dir)
    _plot_results(args.output_dir, results, reference)
    _write_report(args.output_dir, args.quality, payload)
    print(f"OUTPUT {args.output_dir.resolve()}", flush=True)
    if results["failures"]:
        print(f"COMPLETED_WITH_FAILURES {results['failures']}", flush=True)


if __name__ == "__main__":
    main()
