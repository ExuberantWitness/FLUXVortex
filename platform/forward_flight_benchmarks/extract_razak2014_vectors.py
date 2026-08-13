#!/usr/bin/env python3
"""Extract Lambert/Razak experimental force histories from vector PDF paths.

The source plots are vector graphics.  The experiment is the long-dashed path;
for Figures 13--15 Katz uses the same dash pattern, so the experimental path is
identified additionally by the one-path-per-sample topology and by requiring its
vertices to coincide with the centres of the adjacent error bars.
"""

from __future__ import annotations

import csv
import hashlib
import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PlotSpec:
    figure: int
    case_id: str
    quantity: str
    page_index: int
    drawing_index: int
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float
    value_min: float
    value_max: float
    nominal_u_mps: float
    nominal_f_hz: float
    theta_min_deg: float
    theta_max_deg: float
    phase_relation: str


SPECS = (
    PlotSpec(
        9,
        "fig09_pitch_leading_u6_f0p79_theta_m4_p8",
        "CL",
        11,
        321,
        111.805191,
        276.304016,
        336.370361,
        466.377625,
        -0.2,
        1.0,
        6.0,
        0.79,
        -4.0,
        8.0,
        "pitch_leading_90deg",
    ),
    PlotSpec(
        9,
        "fig09_pitch_leading_u6_f0p79_theta_m4_p8",
        "CD",
        11,
        566,
        326.552185,
        491.051025,
        336.370361,
        466.377625,
        -0.08,
        0.08,
        6.0,
        0.79,
        -4.0,
        8.0,
        "pitch_leading_90deg",
    ),
    PlotSpec(
        10,
        "fig10_pitch_leading_u6_f1p23_theta_m10_p2",
        "CL",
        12,
        243,
        111.805191,
        276.304016,
        143.848053,
        273.855713,
        -0.8,
        1.0,
        6.0,
        1.23,
        -10.0,
        2.0,
        "pitch_leading_90deg",
    ),
    PlotSpec(
        10,
        "fig10_pitch_leading_u6_f1p23_theta_m10_p2",
        "CD",
        12,
        488,
        326.552185,
        491.051025,
        143.848053,
        273.855713,
        -0.10,
        0.04,
        6.0,
        1.23,
        -10.0,
        2.0,
        "pitch_leading_90deg",
    ),
    PlotSpec(
        11,
        "fig11_pitch_leading_u14p8_f1p23_theta_m8_p4",
        "CL",
        12,
        742,
        111.805191,
        276.304016,
        498.442261,
        628.449890,
        0.0,
        0.4,
        14.8,
        1.23,
        -8.0,
        4.0,
        "pitch_leading_90deg",
    ),
    PlotSpec(
        11,
        "fig11_pitch_leading_u14p8_f1p23_theta_m8_p4",
        "CD",
        12,
        990,
        326.552185,
        491.051025,
        498.442261,
        628.449890,
        -0.02,
        0.04,
        14.8,
        1.23,
        -8.0,
        4.0,
        "pitch_leading_90deg",
    ),
    PlotSpec(
        13,
        "fig13_pitch_lagging_u9p4_f1p23_theta_m5_p7",
        "CL",
        14,
        250,
        111.773499,
        276.125122,
        402.741669,
        532.454163,
        -0.6,
        1.2,
        9.4,
        1.23,
        -5.0,
        7.0,
        "pitch_lagging_90deg",
    ),
    PlotSpec(
        13,
        "fig13_pitch_lagging_u9p4_f1p23_theta_m5_p7",
        "CD",
        14,
        501,
        326.520996,
        490.872192,
        402.741669,
        532.454163,
        -0.08,
        0.04,
        9.4,
        1.23,
        -5.0,
        7.0,
        "pitch_lagging_90deg",
    ),
    PlotSpec(
        14,
        "fig14_pitch_lagging_u6_f1p5_theta_m12_p0",
        "CL",
        15,
        234,
        111.036499,
        270.965607,
        104.291206,
        230.686279,
        -1.0,
        1.5,
        6.0,
        1.5,
        -12.0,
        0.0,
        "pitch_lagging_90deg",
    ),
    PlotSpec(
        14,
        "fig14_pitch_lagging_u6_f1p5_theta_m12_p0",
        "CD",
        15,
        487,
        326.520996,
        490.872192,
        104.659462,
        234.371246,
        -0.25,
        0.15,
        6.0,
        1.5,
        -12.0,
        0.0,
        "pitch_lagging_90deg",
    ),
    PlotSpec(
        15,
        "fig15_pitch_lagging_u6_f1p5_theta_p4_p16",
        "CL",
        15,
        728,
        111.773499,
        276.125122,
        459.252716,
        588.965210,
        -0.5,
        1.5,
        6.0,
        1.5,
        4.0,
        16.0,
        "pitch_lagging_90deg",
    ),
    PlotSpec(
        15,
        "fig15_pitch_lagging_u6_f1p5_theta_p4_p16",
        "CD",
        15,
        972,
        325.783997,
        485.712799,
        458.884521,
        585.279602,
        -0.15,
        0.10,
        6.0,
        1.5,
        4.0,
        16.0,
        "pitch_lagging_90deg",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_points(drawing: dict) -> tuple[np.ndarray, np.ndarray]:
    items = drawing["items"]
    if not items or any(item[0] != "l" for item in items):
        raise RuntimeError("Expected a path consisting solely of line segments")
    points = [items[0][1]] + [item[-1] for item in items]
    x = np.asarray([point.x for point in points], dtype=float)
    y = np.asarray([point.y for point in points], dtype=float)
    if np.any(np.diff(x) <= 0):
        raise RuntimeError("Experimental path is not strictly increasing in x")
    return x, y


def y_to_value(spec: PlotSpec, y: np.ndarray | float) -> np.ndarray | float:
    scale = (spec.value_max - spec.value_min) / (spec.y_bottom - spec.y_top)
    return spec.value_min + (spec.y_bottom - y) * scale


def extract_error_bars(
    page: fitz.Page, spec: PlotSpec, x: np.ndarray, y: np.ndarray
) -> list[dict]:
    plot_height = spec.y_bottom - spec.y_top
    value_scale = (spec.value_max - spec.value_min) / plot_height
    result = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        rect = drawing["rect"]
        if not (
            drawing.get("dashes") == "[] 0"
            and drawing.get("color") == (0.0, 0.0, 0.0)
            and len(drawing["items"]) == 1
            and rect.width < 1e-5
            and 3.0 < rect.height < 0.9 * plot_height
            and spec.x_left - 0.02 <= rect.x0 <= spec.x_right + 0.02
            and spec.y_top - 0.02 <= rect.y0 <= rect.y1 <= spec.y_bottom + 0.02
        ):
            continue
        centre_from_path = float(np.interp(rect.x0, x, y))
        centre_from_bar = 0.5 * (rect.y0 + rect.y1)
        centre_delta = centre_from_bar - centre_from_path
        if abs(centre_delta) >= 0.02:
            continue
        result.append(
            {
                "drawing_index": drawing_index,
                "x_pdf": rect.x0,
                "y_centre_pdf": centre_from_bar,
                "y_low_pdf": rect.y1,
                "y_high_pdf": rect.y0,
                "centre_path_delta_pdf_pt": centre_delta,
                "std": 0.5 * rect.height * value_scale,
            }
        )
    result.sort(key=lambda row: row["x_pdf"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source = args.source_pdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(source)
    source_hash = sha256(source)
    curve_rows = []
    bar_rows = []
    audit = {
        "source": source.name,
        "source_sha256": source_hash,
        "plots": [],
        "source_url": "https://www.mdpi.com/2226-4310/4/2/22",
    }

    for spec in SPECS:
        page = document[spec.page_index]
        drawings = page.get_drawings()
        experiment = drawings[spec.drawing_index]
        x, y = path_points(experiment)
        dash_pattern = experiment.get("dashes")
        if not dash_pattern or dash_pattern == "[] 0":
            raise RuntimeError(
                f"Figure {spec.figure} {spec.quantity}: experiment is not dashed"
            )
        phase = np.clip((x - spec.x_left) / (spec.x_right - spec.x_left), 0.0, 1.0)
        value = y_to_value(spec, y)
        bars = extract_error_bars(page, spec, x, y)
        if len(bars) != len(x) - 1:
            raise RuntimeError(
                f"Figure {spec.figure} {spec.quantity}: expected one error bar for each "
                f"unclipped sample ({len(x) - 1}), found {len(bars)}"
            )

        for point_index, (x_value, y_value, phase_value, coefficient) in enumerate(
            zip(x, y, phase, value, strict=True)
        ):
            curve_rows.append(
                {
                    "figure": spec.figure,
                    "case_id": spec.case_id,
                    "quantity": spec.quantity,
                    "point_index": point_index,
                    "phase": phase_value,
                    "coefficient": coefficient,
                    "x_pdf": x_value,
                    "y_pdf": y_value,
                    "is_plot_boundary_intersection": int(point_index == len(x) - 1),
                }
            )

        for point_index, bar in enumerate(bars):
            phase_value = float(
                np.clip(
                    (bar["x_pdf"] - spec.x_left) / (spec.x_right - spec.x_left),
                    0.0,
                    1.0,
                )
            )
            coefficient = float(y_to_value(spec, bar["y_centre_pdf"]))
            std = bar["std"]
            bar_rows.append(
                {
                    "figure": spec.figure,
                    "case_id": spec.case_id,
                    "quantity": spec.quantity,
                    "sample_index": point_index,
                    "phase": phase_value,
                    "coefficient": coefficient,
                    "std": std,
                    "lower": coefficient - std,
                    "upper": coefficient + std,
                    "x_pdf": bar["x_pdf"],
                    "y_centre_pdf": bar["y_centre_pdf"],
                }
            )

        audit["plots"].append(
            {
                **asdict(spec),
                "dash_pattern": dash_pattern,
                "path_point_count_including_clipped_boundary": len(x),
                "error_bar_count": len(bars),
                "path_phase_min": float(phase.min()),
                "path_phase_max": float(phase.max()),
                "max_abs_error_bar_centre_minus_path_pdf_pt": float(
                    max(abs(bar["centre_path_delta_pdf_pt"]) for bar in bars)
                ),
                "last_segment_x_length_pdf_pt": float(x[-1] - x[-2]),
                "median_full_segment_x_length_pdf_pt": float(
                    np.median(np.diff(x)[:-1])
                ),
            }
        )

    curve_path = output_dir / "razak_lambert_experiment_curves_raw.csv"
    bar_path = output_dir / "razak_lambert_experiment_errorbar_centres.csv"
    audit_path = output_dir / "razak_lambert_vector_audit.json"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)
    with bar_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=bar_rows[0].keys())
        writer.writeheader()
        writer.writerows(bar_rows)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    cases = sorted({row["case_id"] for row in curve_rows})
    fig, axes = plt.subplots(
        len(cases), 2, figsize=(10, 2.45 * len(cases)), sharex=True
    )
    for row_index, case_id in enumerate(cases):
        for column_index, quantity in enumerate(("CL", "CD")):
            axis = axes[row_index, column_index]
            curve = [
                row
                for row in curve_rows
                if row["case_id"] == case_id and row["quantity"] == quantity
            ]
            bars = [
                row
                for row in bar_rows
                if row["case_id"] == case_id and row["quantity"] == quantity
            ]
            axis.plot(
                [row["phase"] for row in curve],
                [row["coefficient"] for row in curve],
                color="tab:blue",
                linewidth=1.2,
                label="vector path",
            )
            axis.errorbar(
                [row["phase"] for row in bars],
                [row["coefficient"] for row in bars],
                yerr=[row["std"] for row in bars],
                fmt="none",
                ecolor="0.35",
                elinewidth=0.55,
                capsize=1.4,
                label="PDF error bars",
            )
            axis.grid(alpha=0.25)
            axis.set_ylabel(quantity)
            if column_index == 0:
                axis.set_title(case_id, fontsize=8, loc="left")
            if row_index == len(cases) - 1:
                axis.set_xlabel("normalized plotted phase")
    fig.tight_layout()
    overlay_path = output_dir / "razak_lambert_extracted_vectors.png"
    fig.savefig(overlay_path, dpi=180)
    plt.close(fig)

    outputs = [curve_path, bar_path, audit_path, overlay_path]
    hashes = {path.name: sha256(path) for path in outputs}
    hash_path = output_dir / "SHA256SUMS.json"
    hash_path.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_sha256": source_hash, "outputs": hashes}, indent=2))


if __name__ == "__main__":
    main()
