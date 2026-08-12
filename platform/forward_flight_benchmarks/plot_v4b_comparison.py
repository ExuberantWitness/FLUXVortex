"""Plot the frozen FluxV v4b cross-paper comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812"
)
V4_ROOT = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_v4_ldvm_stevens_20260812"
)
DEFAULT_RUN = V4_ROOT / "runs/20260812_fluxv_v4b_crosspaper_full"
YANG_BASE = (
    V3_ROOT / "runs/20260812_periodic_v2_ullt_full/yang2025_mean_characteristics.csv"
)
FIG14_SOURCE = V3_ROOT / "source_data/izraelevitz2017_fig14_digitized.csv"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _yang_lookup(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    return sorted(
        (row for row in rows if row["model"] == model),
        key=lambda row: float(row["aoa_deg"]),
    )


def plot(run: Path) -> list[Path]:
    v4 = _read(run / "yang2025_v4_mean_characteristics.csv")
    base = _read(YANG_BASE)
    fig14_v4 = _read(run / "izraelevitz2017_fig14_v4_mean_thrust.csv")
    fig14_source = _read(FIG14_SOURCE)

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    aoa = np.asarray([float(row["aoa_deg"]) for row in v4])
    model_styles = (
        ("authors_proposed_modified_uvlm", "Authors' modified UVLM", "#CC79A7", ":"),
        ("fluxv_uvpm", "FluxV old", "#0072B2", "--"),
        ("fluxv_periodic_v1", "FluxV v1 polar", "#E69F00", "-."),
    )
    for axis, channel in zip(axes[:2], ("lift", "drag")):
        test = _yang_lookup(base, "wind_tunnel_test")
        axis.errorbar(
            aoa,
            [float(row[f"mean_{channel}_gf"]) for row in test],
            yerr=0.4,
            color="black",
            marker="o",
            linewidth=1.2,
            label="Yang wind-tunnel",
        )
        for model, label, color, linestyle in model_styles:
            selected = _yang_lookup(base, model)
            axis.plot(
                aoa,
                [float(row[f"mean_{channel}_gf"]) for row in selected],
                color=color,
                linestyle=linestyle,
                linewidth=1.35,
                label=label,
            )
        axis.plot(
            aoa,
            [float(row[f"v4_{channel}_gf"]) for row in v4],
            color="#009E73",
            marker="s",
            linewidth=1.7,
            label="FluxV v4b (UVLM + causal LDVM)",
        )
        axis.set_xlabel("Installation angle [deg]")
        axis.set_ylabel(f"Cycle-mean {channel} [gf]")
        axis.grid(alpha=0.25)

    panel = axes[2]
    colors = {15.0: "#56B4E9", 25.0: "#D55E00"}
    for theta in (15.0, 25.0):
        source_model = [
            row
            for row in fig14_source
            if row["series"] == "authors_1state_ullt"
            and np.isclose(float(row["theta_max_deg"]), theta)
        ]
        source_model.sort(key=lambda row: float(row["phase_offset_deg"]))
        selected = [
            row for row in fig14_v4 if np.isclose(float(row["theta_max_deg"]), theta)
        ]
        selected.sort(key=lambda row: float(row["phase_offset_deg"]))
        panel.plot(
            [float(row["phase_offset_deg"]) for row in source_model],
            [float(row["ct"]) for row in source_model],
            color=colors[theta],
            linestyle=":",
            linewidth=1.15,
            label=f"Authors' 1-state, {theta:g} deg",
        )
        panel.plot(
            [float(row["phase_offset_deg"]) for row in selected],
            [float(row["old_fluxv_CT"]) for row in selected],
            color=colors[theta],
            linestyle="--",
            linewidth=1.2,
            label=f"FluxV old, {theta:g} deg",
        )
        panel.plot(
            [float(row["phase_offset_deg"]) for row in selected],
            [float(row["v4_CT"]) for row in selected],
            color=colors[theta],
            marker="s",
            linewidth=1.6,
            label=f"FluxV v4b, {theta:g} deg",
        )
        observations = [
            row
            for row in fig14_source
            if row["series"] == "scherer_1968_experiment"
            and np.isclose(float(row["theta_max_deg"]), theta)
        ]
        panel.errorbar(
            [float(row["phase_offset_deg"]) for row in observations],
            [float(row["ct"]) for row in observations],
            yerr=[
                [float(row["ct_error_minus"]) for row in observations],
                [float(row["ct_error_plus"]) for row in observations],
            ],
            color="black",
            marker="o" if theta == 15.0 else "^",
            linestyle="none",
            markersize=4,
            alpha=0.75,
            label=f"Scherer experiment, {theta:g} deg",
        )
    panel.set_xlabel("Heave-pitch phase offset [deg]")
    panel.set_ylabel(r"Cycle-mean thrust coefficient $C_T$")
    panel.grid(alpha=0.25)
    for axis in axes:
        axis.legend(frameon=False, fontsize=6.8)
    figure.tight_layout()
    output = run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / "fluxv_v4b_crosspaper_comparison.png",
        output / "fluxv_v4b_crosspaper_comparison.pdf",
    ]
    figure.savefig(paths[0], dpi=240, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    for path in plot(args.run.resolve()):
        print(path)


if __name__ == "__main__":
    main()
