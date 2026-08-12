"""Plot the Figure-14 Scherer experiment and model predictions from CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_upgrade_20260812/runs/"
    "20260812_scherer_fig14_experiment_full"
)
MODEL_LABELS = {
    "scherer_1968_experiment": "Scherer 1968 experiment",
    "authors_6state_ullt": "Authors' six-state ULLT",
    "authors_1state_ullt": "Authors' one-state ULLT",
    "authors_qs_added_mass": "Authors' quasi-steady + added mass",
    "fluxv_uvpm": "FluxV old (UVLM)",
    "fluxv_periodic_v1": "FluxV v1",
    "fluxv_periodic_v2": "FluxV v2",
    "one_state_ullt_local": "Local one-state ULLT",
}


STYLES = {
    "authors_6state_ullt": dict(color="#7B6FD0", ls="--", lw=1.5, marker=""),
    "authors_1state_ullt": dict(color="#1B9E77", ls="-.", lw=1.5, marker=""),
    "authors_qs_added_mass": dict(color="#999999", ls=":", lw=1.4, marker=""),
    "fluxv_uvpm": dict(color="#2B6CB0", ls="--", lw=2.0, marker="o"),
    # v1 and v2 have identical cycle means by construction; draw their common
    # curve once instead of hiding one coincident trace below the other.
    "fluxv_periodic_v2": dict(color="#C53030", ls="-", lw=2.4, marker="D"),
    "one_state_ullt_local": dict(
        color="#8C564B", ls=(0, (5, 2, 1, 2)), lw=1.8, marker="^"
    ),
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir.parent.parent / "figures"
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = _read(run_dir / "mean_thrust_vs_phase.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.25), sharex=True)
    plotted_labels: set[str] = set()
    for axis, theta in zip(axes, (15.0, 25.0)):
        observations = [
            row
            for row in rows
            if row["data_role"] == "experimental_observation"
            and np.isclose(_as_float(row, "theta_max_deg"), theta)
        ]
        x = np.asarray([_as_float(row, "phase_offset_deg") for row in observations])
        y = np.asarray([_as_float(row, "CT") for row in observations])
        yerr = np.asarray(
            [
                [_as_float(row, "CT_error_minus") for row in observations],
                [_as_float(row, "CT_error_plus") for row in observations],
            ]
        )
        label = MODEL_LABELS["scherer_1968_experiment"]
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="s",
            ms=4.6,
            mfc="white",
            mec="black",
            mew=1.0,
            ecolor="black",
            elinewidth=0.85,
            capsize=2.2,
            zorder=10,
            label=label if label not in plotted_labels else "_nolegend_",
        )
        plotted_labels.add(label)

        for series in STYLES:
            selected = [
                row
                for row in rows
                if row["series"] == series
                and np.isclose(_as_float(row, "theta_max_deg"), theta)
            ]
            selected.sort(key=lambda row: _as_float(row, "phase_offset_deg"))
            if not selected:
                continue
            label = (
                "FluxV improved v1/v2 (identical mean)"
                if series == "fluxv_periodic_v2"
                else MODEL_LABELS[series]
            )
            style = dict(STYLES[series])
            marker = style.pop("marker")
            axis.plot(
                [_as_float(row, "phase_offset_deg") for row in selected],
                [_as_float(row, "CT") for row in selected],
                marker=marker,
                markersize=3.2 if marker else 0.0,
                markevery=1,
                label=label if label not in plotted_labels else "_nolegend_",
                **style,
            )
            plotted_labels.add(label)

        axis.axhline(0.0, color="#BBBBBB", lw=0.7, zorder=0)
        axis.set_xlim(10.0, 110.0)
        axis.set_xticks(np.arange(15.0, 106.0, 15.0))
        axis.set_xlabel(r"Pitch--heave phase offset $\psi$ (condition, deg)")
        axis.text(
            0.04,
            0.94,
            rf"$\theta_{{\max}}={theta:.0f}^\circ$",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        axis.grid(True, color="#E5E7EB", lw=0.55, alpha=0.75)
    axes[0].set_ylabel(r"Mean thrust coefficient $\overline{C_T}$")
    for axis in axes:
        axis.set_ylim(-0.62, 0.38)

    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        columnspacing=1.1,
        handlelength=2.5,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.82, wspace=0.20)
    stem = output / "izraelevitz2017_fig14_scherer_experiment"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Izraelevitz et al. (2017) Figure 14 / Scherer (1968) experimental "
        "mean-thrust benchmark. Open squares and error bars are digitized "
        "experimental observations; lines are digitized author references or "
        "local predictions. FluxV v1 and v2 have identical cycle means by "
        "construction and therefore share one plotted curve. All inviscid "
        "predictions use the Figure-14 Cd0=0.057 addition. Each value is a "
        "cycle mean, not an instantaneous load history. Error bars are "
        "digitized from the source, which does not define their statistical "
        "meaning; replicate markers are retained without averaging. Figure 11 "
        "is not experimental.\n"
    )
    stem.with_name(stem.name + "_caption.txt").write_text(caption, encoding="utf-8")


if __name__ == "__main__":
    main()
