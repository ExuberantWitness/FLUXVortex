#!/usr/bin/env python3
"""Deterministically digitize Riziotis & Voutsinas (2008), Figure 12.

The published PDF stores the eight double-wake curves and pressure-tap
diamonds as vector paths.  This utility reads those vectors directly; it does
not perform image thresholding or fit curves to target values.

Accepted inputs are either the original article PDF or the two SVG pages
created by ``pdftocairo -svg``.  Extraction deliberately fails closed if the
expected eight panels, 75--79-point model polylines, or 15 pressure-tap
stations on each surface cannot be identified unambiguously.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


DOI = "10.1002/fld.1525"
SOURCE_URL = "https://doi.org/10.1002/fld.1525"
SOURCE_ID = "riziotis_voutsinas_2008_fig12"
SOURCE_PDF_SHA256 = "cc4970b38b3586affc4805a84e526fcb0049ba2dfa42219c01379e2a8f48fa84"
EXTRACTION_VERSION = "riziotis_fig12_vector_v2"
DATA_STROKE_WIDTH = 0.284
DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "docs" / "diag" / "n26e1_fig12"
)
OFFICIAL_TAP_X = (
    0.00025,
    0.0025,
    0.01,
    0.025,
    0.05,
    0.10,
    0.17,
    0.26,
    0.37,
    0.50,
    0.59,
    0.70,
    0.83,
    0.95,
    0.98,
)
PANEL_METADATA = (
    ("a", "upstroke", 12.0, 17),
    ("b", "upstroke", 15.0, 17),
    ("c", "upstroke", 18.0, 17),
    ("d", "upstroke", 19.0, 17),
    ("e", "downstroke", 12.0, 17),
    ("f", "downstroke", 15.0, 17),
    ("g", "downstroke", 18.0, 18),
    ("h", "downstroke", 19.0, 18),
)
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
TOKEN_RE = re.compile(rf"[A-Za-z]|{NUMBER_PATTERN}")
STYLE_WIDTH_RE = re.compile(rf"(?:^|;)stroke-width:({NUMBER_PATTERN})(?:;|$)")
MATRIX_RE = re.compile(
    rf"^matrix\(\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*,\s*"
    rf"({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*,\s*"
    rf"({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\)$"
)


class DigitizationError(RuntimeError):
    """Raised when source vectors cannot be interpreted unambiguously."""


@dataclass(frozen=True)
class AxisFrame:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    y_zero: float
    svg_units_per_cp: float
    cp_tick_step: float
    horizontal_grid_count: int

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    def x_over_c(self, svg_x: float) -> float:
        return (svg_x - self.x_min) / self.width

    def minus_cp(self, svg_y: float) -> float:
        return (svg_y - self.y_zero) / self.svg_units_per_cp

    def svg_from_data(self, x_over_c: float, minus_cp: float) -> tuple[float, float]:
        return (
            self.x_min + x_over_c * self.width,
            self.y_zero + minus_cp * self.svg_units_per_cp,
        )


@dataclass(frozen=True)
class RawDatum:
    category: str
    surface: str
    point_index: int
    svg_x: float
    svg_y: float
    vector_path_id: str
    tap_x_nominal: float | None = None
    tap_x_residual: float | None = None


@dataclass(frozen=True)
class PanelExtraction:
    panel: str
    phase: str
    alpha_deg: float
    phase_rad: float | None
    source_page: int
    source_sha256: str
    source_name: str
    transform: str
    frame: AxisFrame
    model_path_points: int
    excluded_legend_diamonds: int
    excluded_legend_vector_path_ids: tuple[str, ...]
    data: tuple[RawDatum, ...]


@dataclass(frozen=True)
class SourcePage:
    path: Path
    page_number: int
    sha256: str
    source_name: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stroke_width(style: str) -> float | None:
    match = STYLE_WIDTH_RE.search(style)
    return float(match.group(1)) if match else None


def _axis_bbox_pt(frame: AxisFrame, transform: str) -> str:
    match = MATRIX_RE.match(transform)
    if match is None:
        raise DigitizationError(
            f"unsupported or missing SVG-to-page transform {transform!r}"
        )
    a, b, c, d, e, f = (float(value) for value in match.groups())
    corners = (
        (frame.x_min, frame.y_min),
        (frame.x_min, frame.y_max),
        (frame.x_max, frame.y_min),
        (frame.x_max, frame.y_max),
    )
    transformed = [
        (a * x_value + c * y_value + e, b * x_value + d * y_value + f)
        for x_value, y_value in corners
    ]
    x_values = [point[0] for point in transformed]
    y_values = [point[1] for point in transformed]
    return (
        f"[{_format_float(min(x_values))},"
        f"{_format_float(min(y_values))},"
        f"{_format_float(max(x_values))},"
        f"{_format_float(max(y_values))}]"
    )


def _parse_subpaths(path_data: str) -> list[list[tuple[float, float]]]:
    tokens = TOKEN_RE.findall(path_data)
    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    index = 0
    while index < len(tokens):
        command = tokens[index]
        if command in ("M", "L"):
            if index + 2 >= len(tokens):
                raise DigitizationError("truncated SVG path coordinate pair")
            if command == "M" and current:
                subpaths.append(current)
                current = []
            current.append((float(tokens[index + 1]), float(tokens[index + 2])))
            index += 3
        elif command in ("Z", "z"):
            index += 1
        elif command.isalpha():
            raise DigitizationError(
                f"unsupported SVG path command {command!r}; expected M/L/Z"
            )
        else:
            raise DigitizationError(
                "implicit SVG coordinates are not supported; "
                "regenerate with pdftocairo"
            )
    if current:
        subpaths.append(current)
    return subpaths


def _deduplicate(values: Iterable[float], *, tolerance: float) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
        else:
            result[-1] = 0.5 * (result[-1] + value)
    return result


def _find_raw_frames(
    path_records: Sequence[tuple[float | None, str, list[list[tuple[float, float]]]]],
) -> list[tuple[float, float, float, float]]:
    candidates: list[tuple[float, float, float, float]] = []
    for width, _transform, subpaths in path_records:
        if width is None or abs(width - 0.5) > 1.0e-4:
            continue
        for points in subpaths:
            # Poppler encodes the fourth edge with ``Z``; _parse_subpaths
            # intentionally stores vertices only, so a rectangle has four
            # coordinates rather than a duplicated closing coordinate.
            if len(points) != 4:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            if (
                len(_deduplicate(xs, tolerance=1e-5)) == 2
                and len(_deduplicate(ys, tolerance=1e-5)) == 2
                and 150.0 < x_max - x_min < 175.0
                and 120.0 < y_max - y_min < 132.0
            ):
                candidates.append((x_min, x_max, y_min, y_max))

    frames: list[tuple[float, float, float, float]] = []
    for candidate in candidates:
        if not any(
            max(abs(left - right) for left, right in zip(candidate, known)) < 0.02
            for known in frames
        ):
            frames.append(candidate)

    # Reading order is visual row (SVG path y descends after the page's
    # y-flip), then x.  Nominally co-row frames differ by a few tenths of an
    # SVG unit, so sorting on raw y before x can swap left/right panels.
    rows: list[list[tuple[float, float, float, float]]] = []
    for frame in sorted(frames, key=lambda item: -0.5 * (item[2] + item[3])):
        center_y = 0.5 * (frame[2] + frame[3])
        if not rows:
            rows.append([frame])
            continue
        row_center = sum(0.5 * (item[2] + item[3]) for item in rows[-1]) / len(rows[-1])
        if abs(center_y - row_center) <= 5.0:
            rows[-1].append(frame)
        else:
            rows.append([frame])
    return [frame for row in rows for frame in sorted(row, key=lambda item: item[0])]


def _horizontal_grid_y(
    raw_frame: tuple[float, float, float, float],
    path_records: Sequence[tuple[float | None, str, list[list[tuple[float, float]]]]],
) -> list[float]:
    x_min, x_max, y_min, y_max = raw_frame
    frame_width = x_max - x_min
    values: list[float] = []
    for width, _transform, subpaths in path_records:
        if width is None or not (0.134 <= width <= 0.143):
            continue
        for points in subpaths:
            if len(points) != 2:
                continue
            (x_0, y_0), (x_1, y_1) = points
            if abs(y_1 - y_0) > 0.01:
                continue
            overlap = max(0.0, min(max(x_0, x_1), x_max) - max(min(x_0, x_1), x_min))
            if overlap < 0.50 * frame_width:
                continue
            if y_min - 0.02 <= y_0 <= y_max + 0.02:
                values.append(0.5 * (y_0 + y_1))
    return _deduplicate(values, tolerance=0.05)


def _build_axis_frames(
    raw_frames: Sequence[tuple[float, float, float, float]],
    path_records: Sequence[tuple[float | None, str, list[list[tuple[float, float]]]]],
) -> list[AxisFrame]:
    frames: list[AxisFrame] = []
    for raw_frame in raw_frames:
        grid_y = _horizontal_grid_y(raw_frame, path_records)
        if len(grid_y) == 9:
            tick_step = 1.0
            # These panels label -1, 0, ..., 7.  SVG y increases upward in
            # the untransformed path coordinates, hence zero is level 2.
            y_zero = grid_y[1]
        elif len(grid_y) == 5:
            tick_step = 2.0
            # These panels label 0, 2, ..., 8.
            y_zero = grid_y[0]
        else:
            raise DigitizationError(
                "could not infer Figure 12 pressure axis: expected 9 "
                f"or 5 horizontal grid levels, found {len(grid_y)} "
                f"inside frame {raw_frame}"
            )
        spacings = [right - left for left, right in zip(grid_y[:-1], grid_y[1:])]
        spacing = median(spacings)
        if max(abs(value - spacing) for value in spacings) > 0.08:
            raise DigitizationError(
                f"nonuniform pressure grid in frame {raw_frame}: {grid_y}"
            )
        frames.append(
            AxisFrame(
                x_min=raw_frame[0],
                x_max=raw_frame[1],
                y_min=raw_frame[2],
                y_max=raw_frame[3],
                y_zero=y_zero,
                svg_units_per_cp=spacing / tick_step,
                cp_tick_step=tick_step,
                horizontal_grid_count=len(grid_y),
            )
        )
    return frames


def _inside_frame(
    point: tuple[float, float], frame: AxisFrame, *, margin: float = 2.0
) -> bool:
    return (
        frame.x_min - margin <= point[0] <= frame.x_max + margin
        and frame.y_min - margin <= point[1] <= frame.y_max + margin
    )


def _unique_frame_index(
    points: Sequence[tuple[float, float]], frames: Sequence[AxisFrame]
) -> int:
    matches: list[int] = []
    for index, frame in enumerate(frames):
        fraction_inside = sum(_inside_frame(point, frame) for point in points) / len(
            points
        )
        if fraction_inside >= 0.95:
            matches.append(index)
    if len(matches) != 1:
        raise DigitizationError(
            f"vector path belongs to {len(matches)} panels, expected one"
        )
    return matches[0]


def _split_model_curve(
    points: Sequence[tuple[float, float]],
) -> dict[str, list[tuple[float, float]]]:
    if not 75 <= len(points) <= 79:
        raise DigitizationError(
            f"model curve has {len(points)} points, expected 75--79"
        )
    x_values = [point[0] for point in points]
    x_min = min(x_values)
    leading_edge = [
        index for index, value in enumerate(x_values) if abs(value - x_min) < 1e-6
    ]
    if len(leading_edge) != 2 or leading_edge[1] != leading_edge[0] + 1:
        raise DigitizationError(
            "model curve does not have the expected two-point leading-edge turn"
        )
    first, second = leading_edge
    if any(
        right > left + 1e-5
        for left, right in zip(x_values[:first], x_values[1 : first + 1])
    ):
        raise DigitizationError("first model branch is not TE-to-LE monotone")
    if any(
        right < left - 1e-5
        for left, right in zip(x_values[second:-1], x_values[second + 1 :])
    ):
        raise DigitizationError("second model branch is not LE-to-TE monotone")

    first_branch = list(points[: first + 1])
    second_branch = list(points[second:])
    if median(point[1] for point in first_branch) >= median(
        point[1] for point in second_branch
    ):
        raise DigitizationError(
            "model branch orientation is inconsistent with positive-angle "
            "NACA0015 pressure loading"
        )
    return {
        "lower": list(reversed(first_branch)),
        "upper": second_branch,
    }


def _diamond_center(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if len(points) != 4:
        raise DigitizationError("pressure-tap marker is not a four-vertex diamond")
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    width = max(x_values) - min(x_values)
    height = max(y_values) - min(y_values)
    if not (1.5 < width < 2.2 and 1.5 < height < 2.2):
        raise DigitizationError(
            f"unexpected pressure-tap diamond dimensions {width} x {height}"
        )
    return sum(x_values) / 4.0, sum(y_values) / 4.0


def _phase_radians(phase: str, alpha_deg: float) -> float | None:
    # At alpha=19 degrees the harmonic pitch is exactly at reversal.  The
    # source caption says up/downstroke, but a signed phase is not uniquely
    # defined at zero angular velocity; preserve that ambiguity as null.
    if alpha_deg == 19.0:
        return None
    ratio = (alpha_deg - 11.0) / 8.0
    ascending = math.asin(ratio)
    if phase == "upstroke":
        return ascending
    if phase == "downstroke":
        return math.pi - ascending
    raise DigitizationError(f"unknown phase {phase!r}")


def parse_svg_page(
    source: SourcePage,
    metadata: Sequence[tuple[str, str, float, int]],
) -> list[PanelExtraction]:
    """Parse one vector page and bind its panels to supplied metadata."""

    try:
        root = ET.parse(source.path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise DigitizationError(f"cannot parse SVG {source.path}: {exc}") from exc

    path_records: list[tuple[float | None, str, list[list[tuple[float, float]]]]] = []
    data_subpaths: list[tuple[str, list[tuple[float, float]]]] = []
    path_ordinal = 0
    for element in root.iter():
        if not element.tag.endswith("path"):
            continue
        path_ordinal += 1
        width = _stroke_width(element.attrib.get("style", ""))
        if width is None:
            continue
        subpaths = _parse_subpaths(element.attrib.get("d", ""))
        path_records.append(
            (
                width,
                element.attrib.get("transform", ""),
                subpaths,
            )
        )
        if abs(width - DATA_STROKE_WIDTH) <= 1e-4:
            for subpath_ordinal, points in enumerate(subpaths, start=1):
                data_subpaths.append(
                    (
                        f"svg_path_{path_ordinal:04d}:"
                        f"subpath_{subpath_ordinal:03d}",
                        points,
                    )
                )

    raw_frames = _find_raw_frames(path_records)
    if len(raw_frames) != len(metadata):
        raise DigitizationError(
            f"{source.path.name}: found {len(raw_frames)} plot frames, "
            f"expected {len(metadata)}"
        )
    frames = _build_axis_frames(raw_frames, path_records)

    relevant_transforms = {
        transform
        for width, transform, _subpaths in path_records
        if width is not None
        and (
            abs(width - DATA_STROKE_WIDTH) < 1e-4
            or abs(width - 0.5) < 1e-4
            or 0.134 <= width <= 0.143
        )
    }
    if len(relevant_transforms) != 1:
        raise DigitizationError(
            f"{source.path.name}: target vectors use inconsistent transforms"
        )
    transform = next(iter(relevant_transforms))

    model_by_frame: list[tuple[list[tuple[float, float]], str] | None] = [
        None for _ in frames
    ]
    diamonds_by_frame: list[list[tuple[float, float, str]]] = [[] for _ in frames]
    for vector_path_id, points in data_subpaths:
        if 75 <= len(points) <= 79:
            frame_index = _unique_frame_index(points, frames)
            if model_by_frame[frame_index] is not None:
                raise DigitizationError(
                    f"panel {frame_index}: duplicate 75--79-point model path"
                )
            model_by_frame[frame_index] = (points, vector_path_id)
        elif len(points) == 4:
            center = _diamond_center(points)
            matching_frames = [
                index
                for index, frame in enumerate(frames)
                if _inside_frame(center, frame)
            ]
            if len(matching_frames) == 1:
                diamonds_by_frame[matching_frames[0]].append(
                    (center[0], center[1], vector_path_id)
                )

    result: list[PanelExtraction] = []
    for frame_index, (frame, panel_meta) in enumerate(zip(frames, metadata)):
        panel, phase, alpha_deg, source_page = panel_meta
        if source_page != source.page_number:
            raise DigitizationError(
                f"metadata page mismatch for panel {panel}: "
                f"{source_page} != {source.page_number}"
            )
        model_record = model_by_frame[frame_index]
        if model_record is None:
            raise DigitizationError(f"panel {panel}: model curve was not found")
        model_points, model_vector_path_id = model_record
        model_surfaces = _split_model_curve(model_points)
        panel_data: list[RawDatum] = []
        for surface in ("upper", "lower"):
            for point_index, (svg_x, svg_y) in enumerate(model_surfaces[surface]):
                panel_data.append(
                    RawDatum(
                        category="model",
                        surface=surface,
                        point_index=point_index,
                        svg_x=svg_x,
                        svg_y=svg_y,
                        vector_path_id=model_vector_path_id,
                    )
                )

        tap_groups: dict[float, list[tuple[float, float, float, str]]] = {
            value: [] for value in OFFICIAL_TAP_X
        }
        unmatched: list[tuple[float, float, str]] = []
        tap_tolerance = max(0.001, 1.5 * DATA_STROKE_WIDTH / frame.width)
        for svg_x, svg_y, vector_path_id in diamonds_by_frame[frame_index]:
            x_over_c = frame.x_over_c(svg_x)
            nearest = min(
                OFFICIAL_TAP_X, key=lambda candidate: abs(x_over_c - candidate)
            )
            residual = x_over_c - nearest
            if abs(residual) <= tap_tolerance:
                tap_groups[nearest].append((svg_x, svg_y, residual, vector_path_id))
            else:
                unmatched.append((svg_x, svg_y, vector_path_id))

        bad_counts = {
            tap: len(points) for tap, points in tap_groups.items() if len(points) != 2
        }
        if bad_counts or len(unmatched) != 1:
            raise DigitizationError(
                f"panel {panel}: ambiguous measurement vectors; tap counts "
                f"{bad_counts or 'all two'}, unmatched legends={len(unmatched)}"
            )

        for tap_index, tap_x in enumerate(OFFICIAL_TAP_X):
            pressure_pair = sorted(
                tap_groups[tap_x],
                key=lambda point: frame.minus_cp(point[1]),
                reverse=True,
            )
            if (
                abs(
                    frame.minus_cp(pressure_pair[0][1])
                    - frame.minus_cp(pressure_pair[1][1])
                )
                < 5e-3
            ):
                raise DigitizationError(
                    f"panel {panel}, tap {tap_x}: upper/lower assignment "
                    "is ambiguous"
                )
            for surface, (
                svg_x,
                svg_y,
                residual,
                vector_path_id,
            ) in zip(("upper", "lower"), pressure_pair):
                panel_data.append(
                    RawDatum(
                        category="experiment",
                        surface=surface,
                        point_index=tap_index,
                        svg_x=svg_x,
                        svg_y=svg_y,
                        vector_path_id=vector_path_id,
                        tap_x_nominal=tap_x,
                        tap_x_residual=residual,
                    )
                )

        result.append(
            PanelExtraction(
                panel=panel,
                phase=phase,
                alpha_deg=alpha_deg,
                phase_rad=_phase_radians(phase, alpha_deg),
                source_page=source.page_number,
                source_sha256=source.sha256,
                source_name=source.source_name,
                transform=transform,
                frame=frame,
                model_path_points=len(model_points),
                excluded_legend_diamonds=len(unmatched),
                excluded_legend_vector_path_ids=tuple(item[2] for item in unmatched),
                data=tuple(panel_data),
            )
        )
    return result


def _prepare_sources(
    inputs: Sequence[Path],
    *,
    pdf_pages: tuple[int, int],
    temporary_directory: Path,
) -> tuple[list[SourcePage], dict[str, str]]:
    if len(inputs) == 1 and inputs[0].is_dir():
        svg_candidates = sorted(inputs[0].glob("*.svg"))
        inputs = svg_candidates

    provenance: dict[str, str] = {}
    if len(inputs) == 1 and inputs[0].suffix.lower() == ".pdf":
        pdf_path = inputs[0].resolve()
        if not pdf_path.is_file():
            raise DigitizationError(f"PDF does not exist: {pdf_path}")
        converter = shutil.which("pdftocairo")
        if converter is None:
            raise DigitizationError("pdftocairo is required when the input is a PDF")
        provenance["input_pdf_name"] = pdf_path.name
        provenance["input_pdf_sha256"] = sha256_file(pdf_path)
        sources: list[SourcePage] = []
        for canonical_page, input_page in zip((17, 18), pdf_pages):
            svg_path = temporary_directory / f"page{canonical_page}.svg"
            command = [
                converter,
                "-svg",
                "-f",
                str(input_page),
                "-l",
                str(input_page),
                str(pdf_path),
                str(svg_path),
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            if completed.returncode != 0 or not svg_path.is_file():
                raise DigitizationError(
                    "pdftocairo failed for page "
                    f"{input_page}: {completed.stderr.strip()}"
                )
            sources.append(
                SourcePage(
                    path=svg_path,
                    page_number=canonical_page,
                    sha256=sha256_file(svg_path),
                    source_name=(f"riziotis2008_pdf_page_{canonical_page}.svg"),
                )
            )
        return sources, provenance

    if len(inputs) != 2 or any(path.suffix.lower() != ".svg" for path in inputs):
        raise DigitizationError(
            "provide either one source PDF or exactly two SVG page paths"
        )
    resolved = [path.resolve() for path in inputs]
    if any(not path.is_file() for path in resolved):
        missing = [str(path) for path in resolved if not path.is_file()]
        raise DigitizationError(f"SVG input does not exist: {missing}")

    # The page containing six frames is article page 201 / PDF page 17; the
    # continuation containing two frames is page 202 / PDF page 18.  Determine
    # that identity from content rather than trusting filenames or CLI order.
    classified: dict[int, Path] = {}
    for path in resolved:
        root = ET.parse(path).getroot()
        records = []
        for element in root.iter():
            if not element.tag.endswith("path"):
                continue
            width = _stroke_width(element.attrib.get("style", ""))
            if width is None:
                continue
            records.append(
                (
                    width,
                    element.attrib.get("transform", ""),
                    _parse_subpaths(element.attrib.get("d", "")),
                )
            )
        count = len(_find_raw_frames(records))
        page_number = {6: 17, 2: 18}.get(count)
        if page_number is None or page_number in classified:
            raise DigitizationError(
                f"cannot uniquely classify SVG {path.name}: {count} frames"
            )
        classified[page_number] = path
    if set(classified) != {17, 18}:
        raise DigitizationError("SVG inputs do not contain the 6+2 Figure 12 panels")

    sources = [
        SourcePage(
            path=classified[page],
            page_number=page,
            sha256=sha256_file(classified[page]),
            source_name=f"riziotis2008_pdf_page_{page}.svg",
        )
        for page in (17, 18)
    ]
    return sources, provenance


CSV_FIELDS = (
    "source_id",
    "source_sha256",
    "doi",
    "pdf_page",
    "article_page",
    "figure",
    "panel",
    "series",
    "surface",
    "phase_branch",
    "alpha_label_deg",
    "phase_rad",
    "phase_status",
    "tap_id",
    "x_over_c",
    "minus_cp",
    "cp",
    "vector_path_id",
    "axis_bbox_pt",
    "digitization_sigma_xc",
    "digitization_sigma_cp",
    "extraction_version",
    "exclusion_flag",
    "notes",
    # Additional vector-level audit fields retain the unsnapped coordinate
    # and exact derived-page identity behind the contract-level values.
    "source_vector_sha256",
    "source_vector_name",
    "point_index",
    "vector_x_over_c",
    "tap_x_nominal",
    "tap_x_residual",
    "svg_x",
    "svg_y",
    "vector_half_stroke_xc",
    "vector_half_stroke_cp",
)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _tap_id(surface: str, tap_index: int) -> str:
    if not 0 <= tap_index < len(OFFICIAL_TAP_X):
        raise DigitizationError(f"tap index outside source contract: {tap_index}")
    if surface == "upper":
        return f"X{15 - tap_index}"
    if surface == "lower":
        return f"X{16 + tap_index}"
    raise DigitizationError(f"unknown tap surface {surface!r}")


def _phase_status(panel: PanelExtraction) -> str:
    if panel.phase_rad is None:
        return "reported categorical branch; exact phase undisclosed"
    return "derived from reported sinusoid and categorical branch"


def _csv_rows(panels: Sequence[PanelExtraction]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for panel in panels:
        frame = panel.frame
        vector_half_stroke_x = 0.5 * DATA_STROKE_WIDTH / frame.width
        vector_half_stroke_cp = 0.5 * DATA_STROKE_WIDTH / frame.svg_units_per_cp
        axis_bbox = _axis_bbox_pt(frame, panel.transform)
        for datum in panel.data:
            vector_x_over_c = frame.x_over_c(datum.svg_x)
            x_over_c = (
                datum.tap_x_nominal
                if datum.category == "experiment"
                else vector_x_over_c
            )
            minus_cp = frame.minus_cp(datum.svg_y)
            series = (
                "published_double_wake"
                if datum.category == "model"
                else "experimental_measurements"
            )
            tap_id = (
                _tap_id(datum.surface, datum.point_index)
                if datum.category == "experiment"
                else ""
            )
            notes = (
                "unsmoothed published double-wake vector polyline"
                if datum.category == "model"
                else (
                    "published measurement diamond; x_over_c snapped to "
                    "Report 9221 tap station; vector_x_over_c retained"
                )
            )
            rows.append(
                {
                    "source_id": SOURCE_ID,
                    "source_sha256": SOURCE_PDF_SHA256,
                    "doi": DOI,
                    "pdf_page": str(panel.source_page),
                    "article_page": str({17: 201, 18: 202}[panel.source_page]),
                    "figure": "12",
                    "panel": panel.panel,
                    "series": series,
                    "surface": datum.surface,
                    "phase_branch": panel.phase,
                    "alpha_label_deg": _format_float(panel.alpha_deg),
                    "phase_rad": _format_float(panel.phase_rad),
                    "phase_status": _phase_status(panel),
                    "tap_id": tap_id,
                    "x_over_c": _format_float(x_over_c),
                    "minus_cp": _format_float(minus_cp),
                    "cp": _format_float(-minus_cp),
                    "vector_path_id": datum.vector_path_id,
                    "axis_bbox_pt": axis_bbox,
                    "digitization_sigma_xc": "0.002",
                    "digitization_sigma_cp": ("0.1" if x_over_c < 0.01 else "0.03"),
                    "extraction_version": EXTRACTION_VERSION,
                    "exclusion_flag": "false",
                    "notes": notes,
                    "source_vector_sha256": panel.source_sha256,
                    "source_vector_name": panel.source_name,
                    "point_index": str(datum.point_index),
                    "vector_x_over_c": _format_float(vector_x_over_c),
                    "tap_x_nominal": _format_float(datum.tap_x_nominal),
                    "tap_x_residual": _format_float(datum.tap_x_residual),
                    "svg_x": _format_float(datum.svg_x),
                    "svg_y": _format_float(datum.svg_y),
                    "vector_half_stroke_xc": _format_float(vector_half_stroke_x),
                    "vector_half_stroke_cp": _format_float(vector_half_stroke_cp),
                }
            )
    return rows


def _roundtrip_errors(
    panel: PanelExtraction, rows: Sequence[dict[str, str]]
) -> tuple[float, float]:
    x_error = 0.0
    y_error = 0.0
    for datum, row in zip(panel.data, rows):
        svg_x, svg_y = panel.frame.svg_from_data(
            float(row["vector_x_over_c"]), float(row["minus_cp"])
        )
        x_error = max(x_error, abs(svg_x - datum.svg_x))
        y_error = max(y_error, abs(svg_y - datum.svg_y))
    return x_error, y_error


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    panels: Sequence[PanelExtraction],
    rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    offset = 0
    for panel in panels:
        panel_rows = rows[offset : offset + len(panel.data)]
        offset += len(panel.data)
        model_upper = sum(
            row["series"] == "published_double_wake" and row["surface"] == "upper"
            for row in panel_rows
        )
        model_lower = sum(
            row["series"] == "published_double_wake" and row["surface"] == "lower"
            for row in panel_rows
        )
        exp_upper = sum(
            row["series"] == "experimental_measurements" and row["surface"] == "upper"
            for row in panel_rows
        )
        exp_lower = sum(
            row["series"] == "experimental_measurements" and row["surface"] == "lower"
            for row in panel_rows
        )
        tap_residuals = [
            abs(float(row["tap_x_residual"]))
            for row in panel_rows
            if row["tap_x_residual"]
        ]
        x_roundtrip, y_roundtrip = _roundtrip_errors(panel, panel_rows)
        summaries.append(
            {
                "panel": panel.panel,
                "source_page": str(panel.source_page),
                "phase": panel.phase,
                "alpha_deg": _format_float(panel.alpha_deg),
                "phase_rad": _format_float(panel.phase_rad),
                "model_path_points": str(panel.model_path_points),
                "model_upper_points": str(model_upper),
                "model_lower_points": str(model_lower),
                "experiment_upper_taps": str(exp_upper),
                "experiment_lower_taps": str(exp_lower),
                "excluded_legend_diamonds": str(panel.excluded_legend_diamonds),
                "excluded_legend_vector_path_ids": "|".join(
                    panel.excluded_legend_vector_path_ids
                ),
                "max_tap_x_residual": _format_float(max(tap_residuals)),
                "roundtrip_max_svg_x": _format_float(x_roundtrip),
                "roundtrip_max_svg_y": _format_float(y_roundtrip),
                "frame_x_min": _format_float(panel.frame.x_min),
                "frame_x_max": _format_float(panel.frame.x_max),
                "frame_y_zero": _format_float(panel.frame.y_zero),
                "svg_units_per_cp": _format_float(panel.frame.svg_units_per_cp),
                "source_sha256": SOURCE_PDF_SHA256,
                "source_vector_sha256": panel.source_sha256,
            }
        )
    fields = tuple(summaries[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_overlay(
    path: Path,
    panels: Sequence[PanelExtraction],
    rows: Sequence[dict[str, str]],
) -> None:
    width, height = 1120, 780
    cell_width, cell_height = 280, 390
    plot_left, plot_top, plot_width, plot_height = 44, 42, 218, 310
    chunks = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>"
        ".axis{stroke:#777;stroke-width:1;fill:none}"
        ".model{stroke:#111;stroke-width:1.25;fill:none}"
        ".sample{fill:#e34a33;fill-opacity:.58}"
        ".tap{fill:none;stroke:#2166ac;stroke-width:1.2}"
        ".title{font:14px sans-serif;fill:#111}"
        ".legend{font:11px sans-serif;fill:#444}"
        "</style>",
        (
            '<text x="12" y="18" class="legend">'
            "Black: source model path; red: persisted-data round trip; "
            "blue diamonds: experiment taps</text>"
        ),
    ]
    offset = 0
    for panel_index, panel in enumerate(panels):
        panel_rows = rows[offset : offset + len(panel.data)]
        offset += len(panel.data)
        column = panel_index % 4
        row_index = panel_index // 4
        origin_x = column * cell_width
        origin_y = row_index * cell_height
        values = [float(row["minus_cp"]) for row in panel_rows]
        y_min = math.floor(min(values))
        y_max = math.ceil(max(values))
        if y_max - y_min < 2:
            y_max = y_min + 2

        def plot_xy(x_value: float, y_value: float) -> tuple[float, float]:
            x_plot = origin_x + plot_left + x_value * plot_width
            y_plot = (
                origin_y + plot_top + (y_max - y_value) / (y_max - y_min) * plot_height
            )
            return x_plot, y_plot

        chunks.append(
            f'<rect x="{origin_x + plot_left}" y="{origin_y + plot_top}" '
            f'width="{plot_width}" height="{plot_height}" class="axis"/>'
        )
        title = (
            f"({panel.panel}) {panel.phase} "
            f"{panel.alpha_deg:g}°; -Cp [{y_min}, {y_max}]"
        )
        chunks.append(
            f'<text x="{origin_x + plot_left}" y="{origin_y + 34}" '
            f'class="title">{_xml_escape(title)}</text>'
        )
        for surface in ("upper", "lower"):
            source_points = [
                datum
                for datum in panel.data
                if datum.category == "model" and datum.surface == surface
            ]
            source_polyline = " ".join(
                f"{plot_xy(panel.frame.x_over_c(point.svg_x), panel.frame.minus_cp(point.svg_y))[0]:.3f},"
                f"{plot_xy(panel.frame.x_over_c(point.svg_x), panel.frame.minus_cp(point.svg_y))[1]:.3f}"
                for point in source_points
            )
            chunks.append(f'<polyline points="{source_polyline}" class="model"/>')
        for row in panel_rows:
            x_plot, y_plot = plot_xy(float(row["x_over_c"]), float(row["minus_cp"]))
            if row["series"] == "published_double_wake":
                chunks.append(
                    f'<circle cx="{x_plot:.3f}" cy="{y_plot:.3f}" '
                    'r="1.05" class="sample"/>'
                )
            else:
                radius = 2.8
                points = " ".join(
                    (
                        f"{x_plot:.3f},{y_plot - radius:.3f}",
                        f"{x_plot + radius:.3f},{y_plot:.3f}",
                        f"{x_plot:.3f},{y_plot + radius:.3f}",
                        f"{x_plot - radius:.3f},{y_plot:.3f}",
                    )
                )
                chunks.append(f'<polygon points="{points}" class="tap"/>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _write_readme(
    path: Path,
    panels: Sequence[PanelExtraction],
    summaries: Sequence[dict[str, str]],
    _provenance: dict[str, str],
) -> None:
    source_lines = [
        "- Audited source PDF SHA-256: " f"`{SOURCE_PDF_SHA256}`",
    ]
    for page in (17, 18):
        panel = next(item for item in panels if item.source_page == page)
        source_lines.append(
            f"- Derived/source SVG page {page} SHA-256: " f"`{panel.source_sha256}`"
        )
    max_tap = max(float(item["max_tap_x_residual"]) for item in summaries)
    max_roundtrip_x = max(float(item["roundtrip_max_svg_x"]) for item in summaries)
    max_roundtrip_y = max(float(item["roundtrip_max_svg_y"]) for item in summaries)
    text = f"""# Riziotis & Voutsinas (2008), Figure 12 vector digitization

This directory is a deterministic vector extraction of Figure 12 from:

V. A. Riziotis and S. G. Voutsinas, “Dynamic stall modelling on airfoils
based on strong viscous–inviscid interaction coupling,” *International
Journal for Numerical Methods in Fluids* 56 (2008), 185–208.
[doi:{DOI}]({SOURCE_URL})

The row schema follows the
[frozen N2.6e1 source-response contract](../n26e1_source_response_contract_20260730.md).

No raster tracing, force fitting, or curve smoothing is used. The script
`platform/digitize_riziotis_fig12.py` reads the publication's
`stroke-width:0.284` vector objects, identifies one 75–79-point published
double-wake polyline per panel, maps through the vector axis frames, and
validates two experimental diamonds at each of the 15 published pressure-tap
stations on both surfaces. The one legend diamond in each panel is excluded.
Ambiguous topology or counts terminate extraction.

The Figure 12 caption defines the source case as NACA0015,
`Re = 1.5e6`, `Ma = 0.12`, mean pitch `alpha0 = 11°`, pitch amplitude
`alpha1 = 8°`, and reduced frequency `k = 0.05`. The 15 pressure-tap
stations resolved from the common vector abscissae are:
`{", ".join(f"{value:g}" for value in OFFICIAL_TAP_X)}` x/c.
On the upper surface these are taps `X15` through `X1` from leading to
trailing edge; on the lower surface they are `X16` through `X30`.

## Files

- `fig12_digitized.csv`: model curve and experimental pressure taps. It stores
  both `minus_cp` (the plotted ordinate) and `cp`, plus surface, motion phase,
  series, the complete frozen source-contract provenance, and unsnapped
  source-vector coordinates. Experimental `x_over_c` is the Report 9221
  nominal tap coordinate; `vector_x_over_c` retains the extracted coordinate.
- `panel_summary.csv`: counts, frame calibration, pressure-tap residual, and
  persisted-CSV round-trip errors for each panel.
- `roundtrip_overlay.svg`: all eight panels. Black curves are source vectors,
  red samples are coordinates after CSV formatting and inverse mapping, and
  blue diamonds are the extracted experimental taps.

`phase_rad` follows `alpha = 11° + 8° sin(phase_rad)` away from reversal.
It is intentionally empty for both 19° panels: angular velocity is zero at
the maximum angle, so an upstroke/downstroke signed phase is not uniquely
defined at that instant.

The contract uncertainty columns are the preregistered digitization values:
`digitization_sigma_xc=0.002`, and `digitization_sigma_cp=0.10` for
`x/c<0.01` or `0.03` elsewhere. The additional `vector_half_stroke_*`
columns report the smaller mechanical coordinate-resolution estimate from
half the published vector stroke. Neither quantity is experimental pressure
uncertainty.

## Provenance

{chr(10).join(source_lines)}
- SVG generator used for PDF input: `pdftocairo -svg`, one page per call.
- Maximum absolute tap-location residual from the 15 nominal stations:
  `{max_tap:.6g}` x/c.
- Maximum persisted-value round-trip error:
  `{max_roundtrip_x:.3g}` SVG x units and `{max_roundtrip_y:.3g}` SVG y units.
- Extracted rows: `{sum(len(panel.data) for panel in panels)}` total =
  `{sum(panel.model_path_points for panel in panels)}` model +
  `{sum(sum(d.category == "experiment" for d in panel.data) for panel in panels)}` experiment.

Reproduce from a legally obtained source PDF:

```bash
python platform/digitize_riziotis_fig12.py /path/to/riziotis2008_full.pdf
```

Or from the two vector pages in either order:

```bash
python platform/digitize_riziotis_fig12.py page17.svg page18.svg
```
"""
    path.write_text(text, encoding="utf-8")


def digitize(
    inputs: Sequence[Path],
    *,
    output_directory: Path,
    pdf_pages: tuple[int, int] = (17, 18),
) -> tuple[list[PanelExtraction], list[dict[str, str]]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="riziotis_fig12_") as temp_name:
        sources, provenance = _prepare_sources(
            inputs,
            pdf_pages=pdf_pages,
            temporary_directory=Path(temp_name),
        )
        page17 = next(source for source in sources if source.page_number == 17)
        page18 = next(source for source in sources if source.page_number == 18)
        panels = [
            *parse_svg_page(page17, PANEL_METADATA[:6]),
            *parse_svg_page(page18, PANEL_METADATA[6:]),
        ]

    if [panel.panel for panel in panels] != list("abcdefgh"):
        raise DigitizationError("panel identity/order validation failed")
    rows = _csv_rows(panels)
    if len(rows) != 859:
        raise DigitizationError(f"unexpected total row count {len(rows)}, expected 859")
    if sum(row["series"] == "experimental_measurements" for row in rows) != 240:
        raise DigitizationError("expected exactly 240 experimental tap rows")

    _write_csv(output_directory / "fig12_digitized.csv", rows)
    summaries = _write_summary(output_directory / "panel_summary.csv", panels, rows)
    _write_overlay(output_directory / "roundtrip_overlay.svg", panels, rows)
    _write_readme(
        output_directory / "README.md",
        panels,
        summaries,
        provenance,
    )
    return panels, rows


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="one article PDF, two SVG pages, or a directory containing them",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="persistent output directory",
    )
    parser.add_argument(
        "--pdf-pages",
        nargs=2,
        type=int,
        default=(17, 18),
        metavar=("FIG12_PAGE", "CONTINUATION_PAGE"),
        help="1-based PDF pages containing Figure 12 (default: 17 18)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        panels, rows = digitize(
            args.inputs,
            output_directory=args.output_dir,
            pdf_pages=tuple(args.pdf_pages),
        )
    except DigitizationError as exc:
        raise SystemExit(f"digitization failed closed: {exc}") from exc
    model_count = sum(row["series"] == "published_double_wake" for row in rows)
    experiment_count = len(rows) - model_count
    print(
        f"wrote {len(rows)} rows ({model_count} model, "
        f"{experiment_count} experiment) for "
        f"{len(panels)} panels to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
