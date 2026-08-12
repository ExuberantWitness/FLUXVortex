"""Grid/time audit for the reconstructed Izraelevitz Figure 13 case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cases import IZRAELEVITZ_2017
from .ptera_adapter import build_izraelevitz_movement, run_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/runs/20260807_rigid_firstpass/convergence"
)
CONFIGS = {
    "c0_2x6_24x2": (2, 6, 24, 2, 2),
    "c1_2x6_48x4": (2, 6, 48, 4, 2),
    "c2_3x8_48x4": (3, 8, 48, 4, 2),
    "c3_4x10_72x4": (4, 10, 72, 4, 2),
    "c4_5x12_96x4": (5, 12, 96, 4, 2),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    rows = []
    for name, settings in CONFIGS.items():
        print(f"RUN {name} settings={settings}", flush=True)
        movement, setup = build_izraelevitz_movement("full", settings=settings)
        history = run_model(
            movement,
            "ptera_prescribed_wake_uvlm",
            period_s=IZRAELEVITZ_2017.period_s,
            rho=IZRAELEVITZ_2017.rho_kg_m3,
            speed=IZRAELEVITZ_2017.freestream_m_s,
            area=IZRAELEVITZ_2017.area_m2,
        )
        history["setup"] = setup
        results[name] = history
        rows.append(
            {
                "configuration": name,
                "chord_panels": settings[0],
                "semispan_panels": settings[1],
                "steps_per_cycle": settings[2],
                "cycles": settings[3],
                "max_wake_cycles": settings[4],
                "mean_CL": history["mean_CL"],
                "mean_CT": history["mean_CT"],
                "runtime_s": history["runtime_s"],
            }
        )
        print(
            f"DONE {name}: CL={history['mean_CL']:.6f} CT={history['mean_CT']:.6f}",
            flush=True,
        )
    finest = results["c4_5x12_96x4"]
    for name, history in results.items():
        history["rms_CL_vs_finest"] = float(
            np.sqrt(np.mean((np.asarray(history["CL"]) - np.asarray(finest["CL"])) ** 2))
        )
        history["rms_CT_vs_finest"] = float(
            np.sqrt(np.mean((np.asarray(history["CT"]) - np.asarray(finest["CT"])) ** 2))
        )
    with (args.output_dir / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "convergence.json").write_text(
        json.dumps(
            results,
            indent=2,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else value.item()
            if isinstance(value, (np.floating, np.integer))
            else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, history in results.items():
        axes[0].plot(history["phase"], history["CL"], label=name)
        axes[1].plot(history["phase"], history["CT"], label=name)
    axes[0].set_ylabel("CL")
    axes[1].set_ylabel("CT")
    axes[1].set_xlabel("cycle phase t/T")
    axes[0].set_title("Izraelevitz 2017 reconstructed-case discretization audit")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "convergence_histories.png", dpi=220)
    plt.close(fig)
    print(f"OUTPUT {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
