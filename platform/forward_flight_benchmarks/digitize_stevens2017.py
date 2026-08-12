"""Digitize the experimental lift histories in Stevens & Babinsky (2017).

The paper embeds the Figure 21 curves as vector paths.  Figure 13/14 contain
the same measurements, but the leading-edge-axis start-up peak is clipped by
their ``0 <= C_L <= 6`` plotting range.  Figure 21 republishes the experiment
over ``0 <= s/c <= 5`` without that clipping and is therefore the normative
source used by the Stevens validation runner.

This utility intentionally extracts only the two black dashed *experimental*
paths.  The coloured reduced-order-model component paths are not observations
and are not written to the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")
_TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")
_MATRIX = re.compile(r"matrix\(([^)]+)\)")


def _sample_svg_path(path_data: str, samples_per_cubic: int = 40) -> np.ndarray:
    """Return densely sampled points for the M/L/C commands used by the PDF."""

    tokens = _TOKEN.findall(path_data)
    points: list[np.ndarray] = []
    cursor = np.zeros(2, dtype=float)
    command: str | None = None
    index = 0
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index].upper()
            index += 1
        if command == "M":
            cursor = np.array([float(tokens[index]), float(tokens[index + 1])])
            index += 2
            points.append(cursor.copy())
            command = "L"
        elif command == "L":
            cursor = np.array([float(tokens[index]), float(tokens[index + 1])])
            index += 2
            points.append(cursor.copy())
        elif command == "C":
            start = cursor.copy()
            control_1 = np.array([float(tokens[index]), float(tokens[index + 1])])
            control_2 = np.array([float(tokens[index + 2]), float(tokens[index + 3])])
            end = np.array([float(tokens[index + 4]), float(tokens[index + 5])])
            index += 6
            for fraction in np.linspace(0.0, 1.0, samples_per_cubic + 1)[1:]:
                complement = 1.0 - fraction
                points.append(
                    complement**3 * start
                    + 3.0 * complement**2 * fraction * control_1
                    + 3.0 * complement * fraction**2 * control_2
                    + fraction**3 * end
                )
            cursor = end
        elif command == "Z":
            command = None
        else:  # pragma: no cover - guards against a changed publisher encoding.
            raise ValueError(f"unsupported SVG path command: {command!r}")
    if not points:
        raise ValueError("empty SVG path")
    return np.asarray(points, dtype=float)


def _apply_transform(points: np.ndarray, transform: str) -> np.ndarray:
    match = _MATRIX.search(transform)
    if match is None:
        raise ValueError(f"unsupported SVG transform: {transform!r}")
    values = [float(value) for value in _NUMBER.findall(match.group(1))]
    if len(values) != 6:
        raise ValueError(f"expected six affine coefficients, got {values!r}")
    a, b, c, d, translate_x, translate_y = values
    return np.column_stack(
        (
            a * points[:, 0] + c * points[:, 1] + translate_x,
            b * points[:, 0] + d * points[:, 1] + translate_y,
        )
    )


def _experimental_paths(svg_path: Path) -> list[np.ndarray]:
    root = ET.parse(svg_path).getroot()
    candidates: list[np.ndarray] = []
    for element in root.iter():
        if not element.tag.endswith("path"):
            continue
        style = element.attrib.get("style", "")
        path_data = element.attrib.get("d", "")
        if (
            "stroke:rgb(0%,0%,0%)" not in style
            or "stroke-dasharray" not in style
            or len(path_data) < 500
        ):
            continue
        candidates.append(
            _apply_transform(
                _sample_svg_path(path_data), element.attrib.get("transform", "")
            )
        )
    if len(candidates) != 2:
        raise ValueError(
            "expected exactly two long black dashed Figure-21 paths, "
            f"found {len(candidates)}"
        )
    return candidates


def _map_panel(
    points: np.ndarray,
    *,
    x_origin: float,
    y_origin: float,
    local_width: float,
    local_height: float,
    y_min: float,
    y_max: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Poppler emits the common affine scale shown below on this page.  The
    # panel geometry values come directly from the vector grid paths, rather
    # than from raster pixels.
    scale = 0.998785
    width = scale * local_width
    height = scale * local_height
    s_over_c = 5.0 * (points[:, 0] - x_origin) / width
    lift = y_min + (y_origin - points[:, 1]) / height * (y_max - y_min)
    valid = (
        (s_over_c >= -1.0e-3)
        & (s_over_c <= 5.001)
        & (lift >= y_min - 0.2)
        & (lift <= y_max + 0.2)
    )
    s_over_c = s_over_c[valid]
    lift = lift[valid]
    order = np.argsort(s_over_c, kind="stable")
    s_over_c = s_over_c[order]
    lift = lift[order]
    if s_over_c.size < 50:
        raise ValueError("too few vector samples survived the panel calibration")

    target = np.linspace(0.0, 5.0, samples)
    # The mid-chord experimental polyline begins at s/c=0.  Preserve its first
    # vertex explicitly; sorting Bezier samples with repeated x values can
    # otherwise choose the following vertex for the endpoint.
    return target, np.interp(target, s_over_c, lift, left=lift[0], right=lift[-1])


def digitize(pdf_path: Path, output_csv: Path, samples: int = 501) -> None:
    with tempfile.TemporaryDirectory(prefix="stevens2017-vector-") as directory:
        svg_path = Path(directory) / "page15.svg"
        subprocess.run(
            [
                "pdftocairo",
                "-f",
                "15",
                "-l",
                "15",
                "-svg",
                str(pdf_path),
                str(svg_path),
            ],
            check=True,
        )
        paths = _experimental_paths(svg_path)

    # Assign paths by their global x origin.  The left panel is the leading-
    # edge pitch axis; the right panel is the mid-chord pitch axis.
    paths.sort(key=lambda values: float(np.min(values[:, 0])))
    s_over_c, leading_edge = _map_panel(
        paths[0],
        x_origin=78.103806,
        y_origin=659.922720,
        local_width=211.884052,
        local_height=123.787937,
        y_min=-7.0,
        y_max=15.0,
        samples=samples,
    )
    s_mid, mid_chord = _map_panel(
        paths[1],
        x_origin=323.332762,
        y_origin=659.590117,
        local_width=215.816123,
        local_height=126.083122,
        y_min=-1.0,
        y_max=8.0,
        samples=samples,
    )
    if not np.array_equal(s_over_c, s_mid):
        raise AssertionError("panel grids differ")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "s_over_c",
                "experiment_CL_leading_edge_axis",
                "experiment_CL_mid_chord_axis",
                "source_figure",
            ]
        )
        for distance, le_value, mid_value in zip(s_over_c, leading_edge, mid_chord):
            writer.writerow(
                [f"{distance:.8f}", f"{le_value:.9f}", f"{mid_value:.9f}", 21]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--samples", type=int, default=501)
    arguments = parser.parse_args()
    digitize(arguments.pdf, arguments.output_csv, arguments.samples)


if __name__ == "__main__":
    main()
