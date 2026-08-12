"""Record bounded numerical-sensitivity checks for the Yang 2025 cross-case run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .cases import YANG_2025
from .ptera_adapter import build_yang2025_movement, run_model
from .run_yang2025_crosscase import GF_TO_N


def _read_means(path: Path) -> dict[tuple[float, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (float(row["aoa_deg"]), row["model"]): row
            for row in csv.DictReader(stream)
            if row["status"] == "ok"
        }


def run_sensitivity(smoke_dir: Path, full_dir: Path) -> None:
    output_dir = full_dir / "sensitivity"
    output_dir.mkdir(exist_ok=True)
    smoke = _read_means(smoke_dir / "mean_characteristics.csv")
    full = _read_means(full_dir / "mean_characteristics.csv")
    fields = [
        "aoa_deg",
        "model",
        "observable",
        "smoke_gf",
        "full_gf",
        "full_minus_smoke_gf",
    ]
    changes: list[dict[str, float | str]] = []
    for key in sorted(set(smoke) & set(full)):
        aoa, model = key
        for observable, field in (
            ("lift", "mean_lift_gf"),
            ("drag", "mean_drag_gf"),
        ):
            coarse = float(smoke[key][field])
            production = float(full[key][field])
            changes.append(
                {
                    "aoa_deg": aoa,
                    "model": model,
                    "observable": observable,
                    "smoke_gf": coarse,
                    "full_gf": production,
                    "full_minus_smoke_gf": production - coarse,
                }
            )
    change_csv = output_dir / "coarse_to_full_changes.csv"
    with change_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changes)

    aoa = 15.0
    movement, metadata = build_yang2025_movement(
        aoa, settings=(8, 12, 100, 3, 3)
    )
    wake3 = run_model(
        movement,
        "ptera_free_wake_uvlm",
        period_s=YANG_2025.period_s,
        rho=YANG_2025.rho_kg_m3,
        speed=YANG_2025.freestream_m_s,
        area=YANG_2025.area_m2,
        output_samples=128,
    )
    wake2_row = full[(aoa, "ptera_free_wake_uvlm")]
    comparison = {
        "case": "Yang 2025 rigid AoA 15 deg",
        "model": "ptera_free_wake_uvlm",
        "common_setting": "8x12 panels, 100 steps/cycle, 3 cycles",
        "wake_cycles_2": {
            "mean_lift_gf": float(wake2_row["mean_lift_gf"]),
            "mean_drag_gf": float(wake2_row["mean_drag_gf"]),
        },
        "wake_cycles_3": {
            "mean_lift_gf": float(wake3["mean_lift_n"]) / GF_TO_N,
            "mean_drag_gf": -float(wake3["mean_thrust_n"]) / GF_TO_N,
            "runtime_s": float(wake3["runtime_s"]),
        },
        "wake3_movement_metadata": metadata,
    }
    comparison["difference_wake3_minus_wake2_gf"] = {
        "lift": comparison["wake_cycles_3"]["mean_lift_gf"]
        - comparison["wake_cycles_2"]["mean_lift_gf"],
        "drag": comparison["wake_cycles_3"]["mean_drag_gf"]
        - comparison["wake_cycles_2"]["mean_drag_gf"],
    }
    wake_json = output_dir / "wake_retention_15deg.json"
    wake_json.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    max_change: dict[tuple[str, str], float] = {}
    for row in changes:
        key = (str(row["model"]), str(row["observable"]))
        max_change[key] = max(
            max_change.get(key, 0.0), abs(float(row["full_minus_smoke_gf"]))
        )
    report = output_dir / "NUMERICAL_SENSITIVITY.md"
    lines = [
        "# Bounded numerical-sensitivity audit",
        "",
        "The smoke/full comparison changes grid, time step, cycle count, and (for the RoboFalcon transfer) strip count together. It is a bounded coarse-to-production check, not a formal convergence order estimate.",
        "",
        "## Maximum absolute coarse-to-production change over six angles",
        "",
        "| Model | Observable | Max change (gf) |",
        "|---|---:|---:|",
    ]
    for (model, observable), value in sorted(max_change.items()):
        lines.append(f"| `{model}` | {observable} | {value:.3f} |")
    diff = comparison["difference_wake3_minus_wake2_gf"]
    lines.extend(
        [
            "",
            "## Wake-retention check at 15 degrees",
            "",
            f"At the common 8x12, 100-step, 3-cycle setting, retaining three rather than two wake cycles changed mean lift by `{diff['lift']:.4f} gf` and mean drag by `{diff['drag']:.4f} gf`.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    for path in (change_csv, wake_json, report):
        print(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_dir", type=Path)
    parser.add_argument("full_dir", type=Path)
    args = parser.parse_args()
    run_sensitivity(args.smoke_dir.resolve(), args.full_dir.resolve())


if __name__ == "__main__":
    main()
