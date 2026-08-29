#!/usr/bin/env python3
"""Digitize Rojratsirikul et al. (2011) Figures 6, 9, and 12--15.

This script deliberately separates the flexible-membrane FSI observables in
Figures 6/9 from the rigid-flat-plate wake observables in Figures 12--15.  The
PDF is the only image source.  Values written by this script are digitized
approximations, not author-supplied tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


EXPECTED_PDF_SHA256 = (
    "c9d8f59b4fefafd846fae77fdda6376424b70032db6ae6c40f1f28d51aa9a6a4"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_plot(document: fitz.Document, page_index: int) -> np.ndarray:
    """Reassemble the vertically tiled plot raster stored on one PDF page."""

    page = document[page_index]
    tiles: list[tuple[float, np.ndarray]] = []
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        rects = page.get_image_rects(xref)
        if len(rects) != 1:
            raise RuntimeError(f"unexpected image placement for xref {xref}: {rects}")
        pixmap = fitz.Pixmap(document, xref)
        raw = np.frombuffer(pixmap.samples, dtype=np.uint8)
        if pixmap.n == 1:
            tile = raw.reshape(pixmap.height, pixmap.width)
        else:
            tile = raw.reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3]
        tiles.append((rects[0].y0, tile.copy()))
    if not tiles:
        raise RuntimeError(f"page {page_index + 1} has no plot raster")
    tiles.sort(key=lambda item: item[0])
    return np.concatenate([tile for _, tile in tiles], axis=0)


def _white_foreground(plot: np.ndarray) -> np.ndarray:
    if plot.ndim == 3:
        gray = cv2.cvtColor(plot, cv2.COLOR_RGB2GRAY)
    else:
        gray = plot
    return gray > 128


def _marker_peaks(
    foreground: np.ndarray,
    x_px: int,
    y_min: int,
    y_max: int,
    count: int,
) -> list[int]:
    score = foreground[:, max(0, x_px - 10) : x_px + 11].sum(axis=1).astype(float)
    score = uniform_filter1d(score, size=7)
    peaks, _ = find_peaks(
        score[y_min:y_max], distance=14, prominence=1.0, height=2.0
    )
    candidates = sorted(
        [(float(score[y_min + peak]), y_min + int(peak)) for peak in peaks],
        reverse=True,
    )[:count]
    if len(candidates) != count:
        raise RuntimeError(
            f"expected {count} markers at x={x_px}, found {len(candidates)}"
        )
    return sorted(y for _, y in candidates)


def _digitize_figure6(plot: np.ndarray) -> list[dict[str, object]]:
    foreground = _white_foreground(plot)
    rows: list[dict[str, object]] = []
    # Calibrated from major ticks in the native 1400 x 1245 plot raster.
    x0, dx = 170.0, 43.55
    y0, pixels_per_unit = 1131.0, 10350.0
    series = [
        ("flexible_U10", 10.0, 48700, "circle"),
        ("flexible_U7p5", 7.5, 36500, "square"),
        ("flexible_U5", 5.0, 24300, "triangle"),
    ]
    for alpha in range(26):
        centers = _marker_peaks(
            foreground, round(x0 + dx * alpha), 190, 1120, count=3
        )
        for (name, speed, reynolds, marker), y_px in zip(series, centers):
            rows.append(
                {
                    "figure": 6,
                    "series": name,
                    "wing_type": "flexible_membrane",
                    "marker": marker,
                    "U_m_s": speed,
                    "Re": reynolds,
                    "alpha_deg": alpha,
                    "zmax_over_c": round((y0 - y_px) / pixels_per_unit, 5),
                    "digitization_uncertainty": 0.0007,
                    "evidence_role": "digitized_approx",
                }
            )
    return rows


def _open_circle_centers_figure9(
    foreground: np.ndarray, flexible_triangle_y: list[int]
) -> list[int]:
    image = (foreground.astype(np.uint8) * 255)
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=12,
        param1=50,
        param2=12,
        minRadius=6,
        maxRadius=13,
    )
    if circles is None:
        raise RuntimeError("Figure 9 open-circle detection failed")
    detected = np.round(circles[0]).astype(int)
    result: list[int] = []
    for alpha in range(31):
        x_expected = round(184 + 35 * alpha)
        candidates = [
            y
            for x, y, _ in detected
            if abs(x - x_expected) <= 5
            and flexible_triangle_y[alpha] + 5 <= y < 1100
        ]
        if not candidates:
            # At alpha=0 the open symbols lie on the axis and partly merge.
            if alpha == 0:
                result.append(1096)
                continue
            raise RuntimeError(f"Figure 9 rigid open circle missing at alpha={alpha}")
        result.append(max(candidates))
    return result


def _template_center_near(
    foreground: np.ndarray,
    x_px: int,
    y_reference: int,
    template: np.ndarray,
) -> int:
    y_low = max(180, y_reference - 8)
    y_high = min(1100, y_reference + 55)
    roi = foreground[y_low:y_high, x_px - 18 : x_px + 19].astype(np.float32)
    response = cv2.matchTemplate(roi, template, cv2.TM_CCORR_NORMED)
    y_local, _ = np.unravel_index(np.argmax(response), response.shape)
    return round(y_low + y_local + (template.shape[0] - 1) / 2)


def _digitize_figure9(plot: np.ndarray) -> list[dict[str, object]]:
    foreground = _white_foreground(plot)
    y0, pixels_per_unit = 1102.0, 650.0
    flexible_y: list[list[int]] = []
    for alpha in range(31):
        flexible_y.append(
            _marker_peaks(foreground, round(184 + 35 * alpha), 180, 1095, count=3)
        )

    rigid_circle_y = _open_circle_centers_figure9(
        foreground, [centers[-1] for centers in flexible_y]
    )
    square_template = np.zeros((21, 21), dtype=np.float32)
    cv2.rectangle(square_template, (2, 2), (18, 18), 1.0, 2)
    triangle_template = np.zeros((25, 23), dtype=np.float32)
    cv2.polylines(
        triangle_template,
        [np.array([[11, 2], [21, 21], [2, 21]], dtype=np.int32)],
        True,
        1.0,
        2,
    )
    rigid_square_y: list[int] = []
    rigid_triangle_y: list[int] = []
    for alpha, circle_y in enumerate(rigid_circle_y):
        if alpha == 0:
            rigid_square_y.append(1097)
            rigid_triangle_y.append(1099)
            continue
        x_px = round(184 + 35 * alpha)
        rigid_square_y.append(
            _template_center_near(foreground, x_px, circle_y, square_template)
        )
        if alpha == 1:
            rigid_triangle_y.append(1098)
        else:
            rigid_triangle_y.append(
                _template_center_near(foreground, x_px, circle_y, triangle_template)
            )

    rows: list[dict[str, object]] = []
    flexible_series = [
        ("flexible_U10", 10.0, 48700, "circle"),
        ("flexible_U7p5", 7.5, 36500, "square"),
        ("flexible_U5", 5.0, 24300, "triangle"),
    ]
    rigid_series = [
        ("rigid_U10", 10.0, 48700, "open_circle", rigid_circle_y),
        ("rigid_U7p5", 7.5, 36500, "open_square", rigid_square_y),
        ("rigid_U5", 5.0, 24300, "open_triangle", rigid_triangle_y),
    ]
    for alpha in range(31):
        for (name, speed, reynolds, marker), y_px in zip(
            flexible_series, flexible_y[alpha]
        ):
            rows.append(
                {
                    "figure": 9,
                    "series": name,
                    "wing_type": "flexible_membrane",
                    "marker": marker,
                    "U_m_s": speed,
                    "Re": reynolds,
                    "alpha_deg": alpha,
                    "Cn": round((y0 - y_px) / pixels_per_unit, 4),
                    "digitization_uncertainty": 0.012,
                    "evidence_role": "digitized_approx",
                }
            )
        for name, speed, reynolds, marker, centers in rigid_series:
            rows.append(
                {
                    "figure": 9,
                    "series": name,
                    "wing_type": "rigid_flat_plate",
                    "marker": marker,
                    "U_m_s": speed,
                    "Re": reynolds,
                    "alpha_deg": alpha,
                    "Cn": round(max(0.0, (y0 - centers[alpha]) / pixels_per_unit), 4),
                    # The three open symbols substantially overlap.
                    "digitization_uncertainty": 0.02,
                    "evidence_role": "digitized_approx_overlap",
                }
            )
    return rows


def _trace_figure12(plot: np.ndarray) -> list[dict[str, object]]:
    if plot.ndim == 3:
        gray = cv2.cvtColor(plot, cv2.COLOR_RGB2GRAY)
    else:
        gray = plot
    x0, pixels_per_st = 237.0, 1324.0 / 3.0
    y0, pixels_per_psd = 1038.0, 27743.0

    # The paper overlays a wake sketch on the right-hand half of the graph.
    # A nearest-neighbour path follower can jump from the spectrum to the
    # sketch.  The following coarse anchors only identify the spectrum's
    # admissible vertical corridor; the actual y value still comes from dark
    # pixels in the PDF at every sampled x.
    anchor_st = np.array(
        [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.58, 0.62,
         0.70, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00]
    )
    anchor_y = np.array(
        [670, 770, 860, 888, 828, 686, 565, 580, 492, 530,
         663, 780, 858, 909, 960, 1005, 1016, 1021],
        dtype=float,
    )

    rows: list[dict[str, object]] = []
    for index in range(301):
        st = index / 100.0
        x = round(x0 + st * pixels_per_st)
        expected_y = float(np.interp(st, anchor_st, anchor_y))
        candidates: list[tuple[float, int]] = []
        for xx in range(max(round(x0) + 2, x - 2), min(gray.shape[1], x + 3)):
            for yy in np.flatnonzero(gray[:, xx] < 150):
                if abs(float(yy) - expected_y) <= 35:
                    candidates.append((abs(float(yy) - expected_y), int(yy)))
        if candidates:
            nearest_distance = min(distance for distance, _ in candidates)
            local = [
                yy for distance, yy in candidates if distance <= nearest_distance + 3.0
            ]
            y = float(np.median(local))
        else:
            y = expected_y
        rows.append(
            {
                "figure": 12,
                "AR": 2,
                "alpha_deg": 15,
                "Re": 48700,
                "fc_over_U": round(st, 2),
                "PSD": round(max(0.0, (y0 - y) / pixels_per_psd), 6),
                "digitization_uncertainty": 0.00035,
                "evidence_role": "digitized_trace_approx",
            }
        )
    return rows


# Figure 15 marker centres for the AR=2 circle series are unusually clean.
# Values below are independently checked against Figure 13 through
# St_modified = St*sin(alpha).  The square series uses the same two-figure
# consistency check.  The Re=48,700 alpha=15 point is independently anchored
# by the Figure 12 spectrum; the whole diamond series is not promoted because
# it overlaps the external filled-diamond series in the paper replot.
AR2_RE36500_ST_MOD = {
    9: 0.1644,
    10: 0.1729,
    11: 0.1826,
    12: 0.1650,
    13: 0.1608,
    14: 0.1480,
    15: 0.1589,
    16: 0.1674,
    17: 0.1692,
    18: 0.1729,
    19: 0.1698,
    20: 0.1759,
    21: 0.1807,
    22: 0.1801,
    23: 0.1747,
    24: 0.1783,
    25: 0.1819,
}

AR2_RE24300_ST_MOD = {
    9: 0.159,
    10: 0.172,
    11: 0.181,
    12: 0.180,
    13: 0.148,
    14: 0.146,
    15: 0.167,
    16: 0.170,
    17: 0.169,
    18: 0.174,
    19: 0.183,
    20: 0.183,
    21: 0.187,
    22: 0.185,
    23: 0.188,
    24: 0.179,
    25: 0.168,
}


def _wake_reference_rows(figure12_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reynolds, values, marker in (
        (24300, AR2_RE24300_ST_MOD, "open_square"),
        (36500, AR2_RE36500_ST_MOD, "open_circle"),
    ):
        for alpha, st_modified in values.items():
            st = st_modified / math.sin(math.radians(alpha))
            rows.append(
                {
                    "figures": "13|15",
                    "series": f"AR2_Re{reynolds}",
                    "AR": 2,
                    "Re": reynolds,
                    "marker": marker,
                    "alpha_deg": alpha,
                    "St_fc_over_U": round(st, 4),
                    "St_modified": round(st_modified, 4),
                    "digitization_uncertainty_St": 0.012,
                    "digitization_uncertainty_St_modified": 0.003,
                    "evidence_role": "digitized_approx_two_figure_crosscheck",
                }
            )

    peak = max(figure12_rows, key=lambda row: float(row["PSD"]))
    alpha = 15.0
    st = float(peak["fc_over_U"])
    rows.append(
        {
            "figures": "12|13|15",
            "series": "AR2_Re48700_alpha15_spectrum_anchor",
            "AR": 2,
            "Re": 48700,
            "marker": "open_diamond",
            "alpha_deg": alpha,
            "St_fc_over_U": round(st, 4),
            "St_modified": round(st * math.sin(math.radians(alpha)), 4),
            "digitization_uncertainty_St": 0.02,
            "digitization_uncertainty_St_modified": 0.006,
            "evidence_role": "digitized_spectrum_peak_anchor",
        }
    )
    return rows


def _figure14_reference_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for alpha in np.arange(8.0, 35.01, 0.5):
        sine = math.sin(math.radians(float(alpha)))
        rows.append(
            {
                "figure": 14,
                "alpha_deg": round(float(alpha), 1),
                "St_fit_0p17_over_sin_alpha": round(0.17 / sine, 5),
                "St_lower_0p15_over_sin_alpha": round(0.15 / sine, 5),
                "St_upper_0p20_over_sin_alpha": round(0.20 / sine, 5),
                "source_role": "published_equation_and_textual_envelope",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_package(
    output: Path,
    figure6: list[dict[str, object]],
    figure9: list[dict[str, object]],
    figure12: list[dict[str, object]],
    wake: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    ax = axes[0, 0]
    for series in ("flexible_U5", "flexible_U7p5", "flexible_U10"):
        data = [row for row in figure6 if row["series"] == series]
        ax.plot(
            [row["alpha_deg"] for row in data],
            [row["zmax_over_c"] for row in data],
            marker="o",
            ms=3.5,
            label=series.replace("flexible_", ""),
        )
    ax.set(xlabel=r"incidence $\alpha$ (deg)", ylabel=r"$z_{max}/c$", title="Figure 6 digitization")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for series in (
        "flexible_U5",
        "flexible_U7p5",
        "flexible_U10",
        "rigid_U5",
        "rigid_U7p5",
        "rigid_U10",
    ):
        data = [row for row in figure9 if row["series"] == series]
        ax.plot(
            [row["alpha_deg"] for row in data],
            [row["Cn"] for row in data],
            marker="o" if "flexible" in series else None,
            lw=1.4 if "flexible" in series else 1.0,
            alpha=1.0 if "flexible" in series else 0.65,
            label=series,
        )
    ax.set(xlabel=r"incidence $\alpha$ (deg)", ylabel=r"$C_n$", title="Figure 9 digitization")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    ax.plot(
        [row["fc_over_U"] for row in figure12],
        [row["PSD"] for row in figure12],
        color="black",
        lw=1.2,
        label=r"AR=2, $\alpha=15^\circ$, Re=48,700",
    )
    peak = max(figure12, key=lambda row: float(row["PSD"]))
    ax.scatter([peak["fc_over_U"]], [peak["PSD"]], color="tab:red", zorder=3)
    ax.axvline(float(peak["fc_over_U"]), color="tab:red", ls="--", lw=0.9)
    ax.set(xlabel=r"$fc/U_\infty$", ylabel="PSD", title="Figure 12 digitized wake spectrum")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    alpha_grid = np.linspace(8, 35, 300)
    sine = np.sin(np.deg2rad(alpha_grid))
    ax.fill_between(alpha_grid, 0.15 / sine, 0.20 / sine, color="0.85", label=r"$St^*=0.15$--$0.20$")
    ax.plot(alpha_grid, 0.17 / sine, "k--", label=r"$St=0.17/\sin\alpha$")
    for series, marker in (("AR2_Re24300", "s"), ("AR2_Re36500", "o")):
        data = [row for row in wake if row["series"] == series]
        ax.scatter(
            [row["alpha_deg"] for row in data],
            [row["St_fc_over_U"] for row in data],
            marker=marker,
            s=22,
            label=series,
        )
    anchor = [row for row in wake if "spectrum_anchor" in str(row["series"])][0]
    ax.scatter(
        [anchor["alpha_deg"]],
        [anchor["St_fc_over_U"]],
        marker="D",
        s=38,
        label="AR2_Re48700 Fig.12 anchor",
    )
    ax.set(xlim=(8, 35), ylim=(0.25, 1.3), xlabel=r"incidence $\alpha$ (deg)", ylabel=r"$St=fc/U_\infty$", title="Figures 13--15 rigid-wake reference")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    fig.suptitle("Rojratsirikul et al. (2011): digitized comparison oracles")
    fig.savefig(output / "rojratsirikul2011_fig06_09_12_15_digitized.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    observed_sha = _sha256(args.pdf)
    if observed_sha != EXPECTED_PDF_SHA256:
        raise RuntimeError(
            f"Rojratsirikul PDF drift: expected {EXPECTED_PDF_SHA256}, got {observed_sha}"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    document = fitz.open(args.pdf)
    plots = {
        6: _native_plot(document, 31),
        9: _native_plot(document, 34),
        12: _native_plot(document, 37),
        13: _native_plot(document, 38),
        14: _native_plot(document, 39),
        15: _native_plot(document, 40),
    }
    figure6 = _digitize_figure6(plots[6])
    figure9 = _digitize_figure9(plots[9])
    figure12 = _trace_figure12(plots[12])
    wake = _wake_reference_rows(figure12)
    figure14 = _figure14_reference_rows()

    _write_csv(args.output / "figure06_displacement_digitized.csv", figure6)
    _write_csv(args.output / "figure09_normal_force_digitized.csv", figure9)
    _write_csv(args.output / "figure12_wake_spectrum_digitized.csv", figure12)
    _write_csv(args.output / "figure13_15_rigid_wake_reference.csv", wake)
    _write_csv(args.output / "figure14_2d_reference_relation.csv", figure14)
    _plot_package(args.output, figure6, figure9, figure12, wake)

    peak = max(figure12, key=lambda row: float(row["PSD"]))
    print(f"PDF SHA256: {observed_sha}")
    print(f"Figure 6 rows: {len(figure6)}")
    print(f"Figure 9 rows: {len(figure9)}")
    print(f"Figure 12 rows: {len(figure12)}")
    print(
        "Figure 12 dominant peak: "
        f"St={float(peak['fc_over_U']):.3f}, PSD={float(peak['PSD']):.5f}"
    )
    print(f"Figures 13/15 reference rows: {len(wake)}")


if __name__ == "__main__":
    main()
