"""Shared publication style for the FluxV all-condition comparison figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


MODEL_STYLES = {
    "experiment": dict(color="#111111", marker="o", linestyle="none", zorder=8),
    "authors_proposed_modified_uvlm": dict(
        color="#009E73", marker="s", linestyle="-", linewidth=1.6
    ),
    "authors_1state_ullt": dict(
        color="#56B4E9", marker="^", linestyle="--", linewidth=1.35
    ),
    "authors_6state_ullt": dict(
        color="#0072B2", marker="v", linestyle="-.", linewidth=1.35
    ),
    "authors_qs_added_mass": dict(
        color="#CC79A7", marker="x", linestyle=":", linewidth=1.35
    ),
    "one_state_ullt_local": dict(
        color="#56B4E9", marker="D", linestyle="-", linewidth=1.35
    ),
    "ptera_free_wake_uvlm": dict(
        color="#009E73", marker="P", linestyle=":", linewidth=1.35
    ),
    "fluxv_old": dict(color="#777777", marker="o", linestyle="--", linewidth=1.4),
    "fluxv_v1_v2": dict(color="#CC79A7", marker="x", linestyle=":", linewidth=1.35),
    "fluxv_v3": dict(color="#E69F00", marker="^", linestyle="-.", linewidth=1.35),
    "fluxv_v4b": dict(color="#0072B2", marker="s", linestyle="-", linewidth=1.8),
    "fluxv_v5a": dict(color="#D55E00", marker="D", linestyle="--", linewidth=1.8),
    "theodorsen": dict(color="#009E73", marker=None, linestyle=":", linewidth=1.5),
    "v5b": dict(color="#D55E00", marker="D", linestyle="--", linewidth=1.7),
}


def configure_matplotlib() -> None:
    """Install deterministic, compact settings suitable for paper figures."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "lines.markersize": 4.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.55,
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    """Place a consistent panel letter just inside the upper-left corner."""

    ax.text(
        0.015,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=9.0,
        zorder=20,
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    """Save paired deterministic PNG/PDF outputs."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    return png, pdf
