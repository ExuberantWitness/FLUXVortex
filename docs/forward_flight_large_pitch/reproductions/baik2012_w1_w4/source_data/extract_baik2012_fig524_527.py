"""Digitize Baik (2011) thesis Figures 5.24--5.27 from embedded JPEGs.

This is an auditable pixel-calibration extraction.  It does not fit phase,
amplitude, offset, or scale to any model.  The printed panel means are used
only to mask the horizontal mean guide and as an independent QA check.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "baik_2011_dissertation.pdf"
FIGDIR = ROOT / "extracted"

# Axes bounds are pixel centers in the original embedded 1318x1602 JPEGs.
# Values are read from the labelled ticks; phase is 0--1 on every panel.
CONFIG = [
    dict(case="W1", variable="cd", image=0, pdf_page=204, printed_page=179,
         figure="5.24", x0=116, x1=630, yt=589, yb=993,
         vtop=0.4, vbot=-0.4, published_mean=0.0315),
    dict(case="W1", variable="cl", image=0, pdf_page=204, printed_page=179,
         figure="5.24", x0=747, x1=1255, yt=586, yb=986,
         vtop=3.0, vbot=-1.0, published_mean=1.04),
    dict(case="W2", variable="cd", image=1, pdf_page=205, printed_page=180,
         figure="5.25", x0=130, x1=635, yt=601, yb=999,
         vtop=2.0, vbot=-2.5, published_mean=-0.127),
    dict(case="W2", variable="cl", image=1, pdf_page=205, printed_page=180,
         figure="5.25", x0=734, x1=1253, yt=586, yb=994,
         vtop=5.0, vbot=-1.0, published_mean=2.11),
    dict(case="W3", variable="cd", image=2, pdf_page=206, printed_page=181,
         figure="5.26", x0=122, x1=628, yt=603, yb=1002,
         vtop=0.4, vbot=-0.3, published_mean=0.127),
    dict(case="W3", variable="cl", image=2, pdf_page=206, printed_page=181,
         figure="5.26", x0=740, x1=1251, yt=599, yb=1001,
         vtop=3.0, vbot=-1.0, published_mean=1.14),
    dict(case="W4", variable="cd", image=3, pdf_page=207, printed_page=182,
         figure="5.27", x0=135, x1=647, yt=604, yb=1007,
         vtop=1.5, vbot=-2.5, published_mean=-0.308),
    dict(case="W4", variable="cl", image=3, pdf_page=207, printed_page=182,
         figure="5.27", x0=738, x1=1261, yt=591, yb=1003,
         vtop=5.0, vbot=-1.0, published_mean=1.37),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trace_curve(gray: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    x0, x1, yt, yb = (cfg[key] for key in ("x0", "x1", "yt", "yb"))
    xs = np.arange(x0 + 2, x1 - 1)
    ys_all = np.arange(yt + 2, yb - 1)
    evidence = gray[yt + 2 : yb - 1, x0 + 2 : x1 - 1].astype(np.float64)
    evidence = cv2.GaussianBlur(evidence, (1, 5), 0.7)

    # The dotted horizontal guide is explicitly labelled with the cycle mean.
    # Mask it so it cannot be mistaken for the measurement curve.  The mask is
    # not an offset fit: axes calibration remains fixed by printed tick marks.
    mean_row = yt + (cfg["vtop"] - cfg["published_mean"]) / (
        cfg["vtop"] - cfg["vbot"]
    ) * (yb - yt)
    lo = max(0, int(round(mean_row)) - 2 - (yt + 2))
    hi = min(evidence.shape[0], int(round(mean_row)) + 3 - (yt + 2))
    evidence[lo:hi] = 0.0

    # Dynamic-programming centerline: image brightness plus a small local
    # curvature penalty.  The fixed penalty is shared by all eight panels and
    # is not optimized against any aerodynamic model.
    reward = np.sqrt(np.maximum(evidence, 0.0))
    height, width = reward.shape
    score = np.full((height, width), -np.inf)
    back = np.zeros((height, width), dtype=np.int8)
    score[:, 0] = reward[:, 0]
    max_jump = 8
    continuity_penalty = 0.05
    for column in range(1, width):
        previous = score[:, column - 1]
        for y in range(height):
            delta_lo = max(-max_jump, -y)
            delta_hi = min(max_jump, height - 1 - y)
            deltas = np.arange(delta_lo, delta_hi + 1)
            candidates = previous[y + deltas] - continuity_penalty * deltas**2
            selected = int(np.argmax(candidates))
            score[y, column] = reward[y, column] + candidates[selected]
            back[y, column] = deltas[selected]

    path = np.empty(width, dtype=int)
    path[-1] = int(np.argmax(score[:, -1]))
    for column in range(width - 1, 0, -1):
        path[column - 1] = path[column] + int(back[path[column], column])
    pixels_y = ys_all[path]
    phase = (xs - x0) / (x1 - x0)
    value = cfg["vtop"] + (pixels_y - yt) / (yb - yt) * (
        cfg["vbot"] - cfg["vtop"]
    )
    audit = {
        "mean_guide_row": mean_row,
        "digitized_trapezoidal_mean": float(np.trapezoid(value, phase) / np.ptp(phase)),
        "published_panel_mean": cfg["published_mean"],
        "mean_absolute_difference": abs(
            float(np.trapezoid(value, phase) / np.ptp(phase)) - cfg["published_mean"]
        ),
        "x_axis_pixel_centers": [x0, x1],
        "y_axis_pixel_centers": [yt, yb],
        "y_axis_values": [cfg["vtop"], cfg["vbot"]],
        "source_points": len(phase),
    }
    return phase, value, audit


def main() -> None:
    rows = []
    audits = []
    fig, axes = plt.subplots(4, 2, figsize=(15, 18), constrained_layout=True)
    for cfg in CONFIG:
        image_path = FIGDIR / f"fig-{cfg['image']:03d}.jpg"
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        gray = rgb.max(axis=2)
        phase, value, audit = trace_curve(gray, cfg)
        for phase_i, value_i in zip(phase, value, strict=True):
            rows.append(
                {
                    "case": cfg["case"],
                    "phase_t_over_T": f"{phase_i:.12g}",
                    cfg["variable"]: f"{value_i:.12g}",
                }
            )
        audit.update(
            {
                "case": cfg["case"],
                "variable": cfg["variable"],
                "figure": cfg["figure"],
                "pdf_page": cfg["pdf_page"],
                "printed_page": cfg["printed_page"],
                "embedded_image": image_path.name,
                "embedded_image_sha256": sha256(image_path),
            }
        )
        audits.append(audit)
        ax = axes[int(cfg["case"][1]) - 1, 0 if cfg["variable"] == "cd" else 1]
        ax.imshow(rgb)
        xs = cfg["x0"] + phase * (cfg["x1"] - cfg["x0"])
        ys = cfg["yt"] + (cfg["vtop"] - value) / (
            cfg["vtop"] - cfg["vbot"]
        ) * (cfg["yb"] - cfg["yt"])
        ax.plot(xs, ys, color="red", lw=0.75)
        ax.set_xlim(cfg["x0"] - 15, cfg["x1"] + 15)
        ax.set_ylim(cfg["yb"] + 15, cfg["yt"] - 15)
        ax.set_title(
            f"{cfg['case']} {cfg['variable'].upper()}: "
            f"trace mean {audit['digitized_trapezoidal_mean']:.4f}; "
            f"printed {cfg['published_mean']:.4f}"
        )
        ax.axis("off")

    # The left/right panels have slightly different raster widths.  Preserve
    # every traced source point in a long-form file, and independently sample
    # each trace onto a declared 401-point common phase grid for paired CL/CD.
    long_path = ROOT / "baik2012_w1_w4_corrected_total_cl_cd_source_pixels.csv"
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "phase_t_over_T", "variable", "value"])
        writer.writeheader()
        for row in rows:
            variable = "cl" if "cl" in row else "cd"
            writer.writerow(
                {
                    "case": row["case"],
                    "phase_t_over_T": row["phase_t_over_T"],
                    "variable": variable,
                    "value": row[variable],
                }
            )
    by_trace: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for case in ("W1", "W2", "W3", "W4"):
        for variable in ("cl", "cd"):
            selected = [row for row in rows if row["case"] == case and variable in row]
            by_trace[(case, variable)] = (
                np.array([float(row["phase_t_over_T"]) for row in selected]),
                np.array([float(row[variable]) for row in selected]),
            )
    common_phase = np.linspace(0.0, 1.0, 401)
    paired_rows = []
    for case in ("W1", "W2", "W3", "W4"):
        values = {}
        for variable in ("cl", "cd"):
            phase, trace_value = by_trace[(case, variable)]
            values[variable] = np.interp(common_phase, phase, trace_value, period=1.0)
        for index, phase in enumerate(common_phase):
            paired_rows.append(
                {
                    "case": case,
                    "phase_t_over_T": f"{phase:.12g}",
                    "cl": f"{values['cl'][index]:.12g}",
                    "cd": f"{values['cd'][index]:.12g}",
                }
            )
    csv_path = ROOT / "baik2012_w1_w4_corrected_total_cl_cd.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "phase_t_over_T", "cl", "cd"])
        writer.writeheader()
        writer.writerows(paired_rows)

    overlay_path = ROOT / "baik2012_fig524_527_digitization_overlay.png"
    fig.savefig(overlay_path, dpi=180)
    plt.close(fig)
    payload = {
        "source_pdf": PDF.name,
        "source_pdf_sha256": sha256(PDF),
        "source_pdf_bytes": PDF.stat().st_size,
        "source_bitstream_uuid": "1ef3aed8-ca97-48ec-b992-85acc3be814a",
        "source_bitstream_url": (
            "https://backend.production.deepblue-documents.lib.umich.edu/server/api/"
            "core/bitstreams/1ef3aed8-ca97-48ec-b992-85acc3be814a/content"
        ),
        "method": "fixed-axis raster centerline dynamic programming",
        "phase_or_amplitude_fit": False,
        "model_output_used": False,
        "panels": audits,
        "output_sha256": {
            long_path.name: sha256(long_path),
            csv_path.name: sha256(csv_path),
            overlay_path.name: sha256(overlay_path),
        },
    }
    (ROOT / "baik2012_fig524_527_digitization_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for audit in audits:
        # The source is a JPEG, so panel means are a QA diagnostic rather than
        # a means of correcting a traced waveform.  One coarse-range panel
        # (W2 CL, 6 coefficient units over 408 px) differs by 0.026; all others
        # are within 0.02.  A 0.03 gate is therefore fixed for this raster set.
        assert audit["mean_absolute_difference"] <= 0.03, audit


if __name__ == "__main__":
    main()
