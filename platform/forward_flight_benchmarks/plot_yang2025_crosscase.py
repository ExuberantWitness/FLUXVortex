"""Plot Yang 2025 multi-model mean characteristics and phase histories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GF_TO_N = 0.00980665
MODEL_ORDER = (
    "wind_tunnel_test",
    "authors_proposed_modified_uvlm",
    "fluxv_uvpm",
    "ptera_prescribed_wake_uvlm",
    "ptera_free_wake_uvlm",
    "robofalcon2_coefficient_transfer",
)
STYLES = {
    "wind_tunnel_test": dict(color="#111111", marker="o", ls="--", lw=1.7, ms=5.5),
    "authors_proposed_modified_uvlm": dict(
        color="#6F6F6F", marker="*", ls=":", lw=1.6, ms=8.0
    ),
    "fluxv_uvpm": dict(color="#0072B2", marker="s", ls="-", lw=2.0, ms=4.8),
    "ptera_prescribed_wake_uvlm": dict(
        color="#56B4E9", marker="x", ls="None", lw=1.2, ms=6.2, mew=1.4
    ),
    "ptera_free_wake_uvlm": dict(
        color="#D55E00", marker="^", ls="-.", lw=1.8, ms=5.2
    ),
    "robofalcon2_coefficient_transfer": dict(
        color="#009E73", marker="D", ls="-", lw=1.7, ms=4.5
    ),
}
SHORT_LABELS = {
    "wind_tunnel_test": "Wind-tunnel test",
    "authors_proposed_modified_uvlm": "Authors' proposed",
    "fluxv_uvpm": "FluxV current load channel",
    "ptera_prescribed_wake_uvlm": "Prescribed-UVLM control",
    "ptera_free_wake_uvlm": "Free-wake UVLM",
    "robofalcon2_coefficient_transfer": "RoboFalcon2 coefficient transfer",
}
PHASE_LABELS = {
    **SHORT_LABELS,
    "ptera_prescribed_wake_uvlm": "Prescribed-UVLM shared-channel audit",
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_mean(run_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] in {"ok", "reference"}:
            grouped[row["model"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.45), constrained_layout=False)
    for model in MODEL_ORDER:
        values = sorted(grouped.get(model, []), key=lambda row: float(row["aoa_deg"]))
        if not values:
            continue
        x = np.asarray([float(row["aoa_deg"]) for row in values])
        style = dict(STYLES[model])
        for axis, field in zip(axes, ("mean_lift_gf", "mean_drag_gf")):
            y = np.asarray([float(row[field]) for row in values])
            axis.plot(x, y, label=SHORT_LABELS[model], **style)
            if model == "wind_tunnel_test":
                uncertainty = np.asarray(
                    [float(row["digitization_uncertainty_gf"]) for row in values]
                )
                axis.errorbar(
                    x,
                    y,
                    yerr=uncertainty,
                    fmt="none",
                    ecolor="#111111",
                    elinewidth=0.7,
                    capsize=2.0,
                    zorder=5,
                )
    axes[0].set_ylabel("Cycle-mean lift (gf)")
    axes[1].set_ylabel("Cycle-mean drag, $D=-T$ (gf)")
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.set_xlabel(r"Geometric installation angle, $\alpha_g$ (deg)")
        axis.set_xticks(np.arange(0.0, 26.0, 5.0))
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)
        axis.text(0.025, 0.95, label, transform=axis.transAxes, va="top", fontweight="bold")
    axes[0].axhline(0.0, color="#888888", lw=0.7)
    axes[1].axhline(0.0, color="#888888", lw=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.5,
    )
    fig.text(
        0.5,
        0.015,
        "Local curves use reconstructed nominal four-bar motion; error bars show digitization uncertainty only.",
        ha="center",
        va="bottom",
        fontsize=7.1,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.79, wspace=0.28)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    pdf = figure_dir / "yang2025_multimodel_mean_characteristics.pdf"
    png = figure_dir / "yang2025_multimodel_mean_characteristics.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_phase(run_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    grouped: dict[tuple[float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["aoa_deg"]), row["model"])].append(row)
    angles = sorted({key[0] for key in grouped})
    local_order = MODEL_ORDER[2:]
    fig, axes = plt.subplots(
        len(angles),
        2,
        figsize=(7.25, 1.48 * len(angles) + 0.9),
        sharex=True,
        squeeze=False,
    )
    for row_index, aoa in enumerate(angles):
        for model in local_order:
            values = sorted(grouped.get((aoa, model), []), key=lambda row: float(row["phase"]))
            if not values:
                continue
            phase = np.asarray([float(row["phase"]) for row in values])
            lift_gf = np.asarray([float(row["lift_n"]) / GF_TO_N for row in values])
            drag_gf = np.asarray([float(row["drag_n"]) / GF_TO_N for row in values])
            style = dict(STYLES[model])
            style["marker"] = None
            style.pop("ms", None)
            style.pop("mew", None)
            if model == "ptera_prescribed_wake_uvlm":
                # The control coincides with FluxV's current load channel.
                # A dotted overlay makes that identity visible in phase plots.
                style["ls"] = ":"
            axes[row_index, 0].plot(
                phase, lift_gf, label=PHASE_LABELS[model], **style
            )
            axes[row_index, 1].plot(
                phase, drag_gf, label=PHASE_LABELS[model], **style
            )
        axes[row_index, 0].set_ylabel(f"{aoa:g}°\nLift (gf)")
        axes[row_index, 1].set_ylabel(f"{aoa:g}°\nDrag (gf)")
        for axis in axes[row_index]:
            axis.axhline(0.0, color="#888888", lw=0.55)
            axis.grid(True, color="#DDDDDD", linewidth=0.5, alpha=0.7)
            axis.set_xlim(0.0, 1.0)
    axes[0, 0].text(0.015, 0.90, "(a)", transform=axes[0, 0].transAxes, fontweight="bold")
    axes[0, 1].text(0.015, 0.90, "(b)", transform=axes[0, 1].transAxes, fontweight="bold")
    axes[-1, 0].set_xlabel("Cycle phase")
    axes[-1, 1].set_xlabel("Cycle phase")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.8,
    )
    fig.text(
        0.5,
        0.010,
        "Nominal four-bar motion; $D=-T$. FluxV/prescribed UVLM share one load channel; RoboFalcon2 is a coefficient transfer.\n"
        "No phase-resolved experimental or authors' proposed-model loads are public for these cases.",
        ha="center",
        fontsize=7.1,
    )
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.085, top=0.92, hspace=0.18, wspace=0.25)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    pdf = figure_dir / "yang2025_multimodel_phase_histories.pdf"
    png = figure_dir / "yang2025_multimodel_phase_histories.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_captions(run_dir: Path) -> Path:
    include_dir = run_dir / "figures" / "latex_includes"
    include_dir.mkdir(parents=True, exist_ok=True)
    path = include_dir / "yang2025_multimodel_captions.tex"
    path.write_text(
        "% Generated captions for the Yang 2025 cross-case figures.\n"
        "\\newcommand{\\YangMeanCaption}{Cycle-mean lift and drag characteristics "
        "for the Yang 2025 rigid-wing cases. Wind-tunnel points and the authors' "
        "proposed modified-UVLM results are digitized from Fig.~11. Local models "
        "share the published geometry and reconstructed nominal four-bar motion; "
        "the unpublished laser-measured motion is unavailable. Drag is defined as "
        "$D=-T$. The FluxV and prescribed-wake curves share one load channel, and "
        "the RoboFalcon2 curve is a cross-domain coefficient transfer.}\n"
        "\\newcommand{\\YangPhaseCaption}{Last-cycle lift and drag histories for "
        "the local models on the six Yang 2025 installation-angle cases. All "
        "curves use reconstructed nominal four-bar motion rather than the "
        "unpublished laser-measured history, and drag is defined as $D=-T$. "
        "FluxV and the prescribed-UVLM audit are the same load channel, not "
        "independent agreement; RoboFalcon2 is an out-of-domain coefficient "
        "transfer. No phase-resolved experimental or author-model force "
        "histories are public.}\n",
        encoding="utf-8",
    )
    return path


def write_accuracy_audit(run_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    """Write aggregate errors without unstable near-zero pointwise percentages."""

    valid = [row for row in rows if row["status"] in {"ok", "reference"}]
    grouped: dict[str, dict[float, dict[str, str]]] = defaultdict(dict)
    for row in valid:
        grouped[row["model"]][float(row["aoa_deg"])] = row
    test = grouped["wind_tunnel_test"]
    metric_rows: list[dict[str, float | str]] = []
    for model in MODEL_ORDER[1:]:
        common = sorted(set(test) & set(grouped.get(model, {})))
        if not common:
            continue
        lift_test = np.asarray([float(test[angle]["mean_lift_gf"]) for angle in common])
        drag_test = np.asarray([float(test[angle]["mean_drag_gf"]) for angle in common])
        lift_model = np.asarray(
            [float(grouped[model][angle]["mean_lift_gf"]) for angle in common]
        )
        drag_model = np.asarray(
            [float(grouped[model][angle]["mean_drag_gf"]) for angle in common]
        )
        for observable, truth, prediction in (
            ("lift", lift_test, lift_model),
            ("drag", drag_test, drag_model),
        ):
            error = prediction - truth
            mae = float(np.mean(np.abs(error)))
            peak_truth = float(np.max(np.abs(truth)))
            metric_rows.append(
                {
                    "model": model,
                    "model_label": SHORT_LABELS[model],
                    "observable": observable,
                    "points": len(common),
                    "mae_gf": mae,
                    "rmse_gf": float(np.sqrt(np.mean(error**2))),
                    "max_abs_error_gf": float(np.max(np.abs(error))),
                    "mae_over_peak_test_percent": 100.0 * mae / peak_truth,
                }
            )
    metric_path = run_dir / "accuracy_metrics.csv"
    fields = [
        "model",
        "model_label",
        "observable",
        "points",
        "mae_gf",
        "rmse_gf",
        "max_abs_error_gf",
        "mae_over_peak_test_percent",
    ]
    with metric_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    fluxv = grouped.get("fluxv_uvpm", {})
    prescribed = grouped.get("ptera_prescribed_wake_uvlm", {})
    overlap = sorted(set(fluxv) & set(prescribed))
    if overlap:
        channel_difference = max(
            max(
                abs(float(fluxv[a]["mean_lift_gf"]) - float(prescribed[a]["mean_lift_gf"])),
                abs(float(fluxv[a]["mean_drag_gf"]) - float(prescribed[a]["mean_drag_gf"])),
            )
            for a in overlap
        )
    else:
        channel_difference = float("nan")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    report = run_dir / "VERIFICATION_REPORT.md"
    lines = [
        "# Yang 2025 existing-model cross-case verification",
        "",
        f"- Run: `{manifest['run_id']}`",
        f"- Quality: `{manifest['quality']}`",
        f"- Successful cells: `{manifest.get('successful_cells', 'unknown')}`",
        f"- Failed cells: `{manifest.get('failed_cells', 'unknown')}`",
        "- Trust class: `cross-case diagnostic / partially comparable`",
        "- Force convention: plotted drag is `D=-T`.",
        "",
        "## Aggregate error against wind-tunnel means",
        "",
        "Pointwise relative errors are intentionally omitted near zero. The final column is MAE normalized by the peak absolute test load over the six cases.",
        "",
        "| Model | Observable | MAE (gf) | RMSE (gf) | Max abs. error (gf) | MAE/peak test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['model_label']} | {row['observable']} | "
            f"{row['mae_gf']:.3f} | {row['rmse_gf']:.3f} | "
            f"{row['max_abs_error_gf']:.3f} | "
            f"{row['mae_over_peak_test_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Integrity and comparability notes",
            "",
            f"- Maximum cycle-mean FluxV/prescribed-control difference: `{channel_difference:.12g} gf`. These are one load channel, not independent corroboration.",
            "- All local curves use the nominal four-bar reconstruction; the paper's laser-measured motion history is not public.",
            "- The authors' `Proposed` curve is their complete modified UVLM (PLEV + AWS + wake/core treatment), not the local circulation-only PLEV core.",
            "- RoboFalcon2 is shown as a coefficient transfer outside its native geometry, Reynolds number, and 6--12 m/s calibration-speed range.",
            "- Phase histories have no public experimental or author-model phase-resolved truth for direct validation.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return metric_path, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    _configure_style()
    mean_rows = _load_csv(run_dir / "mean_characteristics.csv")
    mean_files = plot_mean(run_dir, mean_rows)
    phase_files = plot_phase(run_dir, _load_csv(run_dir / "phase_histories.csv"))
    caption = write_captions(run_dir)
    audit_files = write_accuracy_audit(run_dir, mean_rows)
    for path in (*mean_files, *phase_files, caption, *audit_files):
        print(path.resolve())


if __name__ == "__main__":
    main()
