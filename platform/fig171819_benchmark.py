"""Canonical Fig. 17/18/19 benchmark contract for the FLUXV campaign.

The historical 118-condition sweep covers all of Fig. 17, Fig. 18(a,b),
the U=8 rows of Fig. 18(c,d), and all of Fig. 19.  It does not cover the
U=6/10 twist families in Fig. 18(c,d).  This module makes the complete
contract explicit and scores only complete curves as promotion evidence.

There are two deliberately separate grids:

* the solver contract contains 184 unique conditions (12 nominal twist
  values or 5 frequencies per curve);
* ``docs/data.md`` contains 530 independently digitized measurement samples.

Scoring interpolates the *model curve* from the solver grid to each original
measurement abscissa.  Measurement values are never interpolated onto the
solver grid.  Tiny endpoint offsets caused by plot digitization are projected
to the published physical axis boundary and reported; material extrapolation
is forbidden.

No aerodynamic model is imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
DOCS = PLATFORM / "docs"
DEFAULT_REPRO = DOCS / "repro_data.json"
DEFAULT_DATA_MD = DOCS / "data.md"
FROZEN_DATA_MD_SHA256 = (
    "ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1"
)

TWS = (0.0, 5.0, 10.0, 15.0, 20.0, 22.5, 25.0, 27.5, 30.0, 35.0, 40.0, 45.0)
FS = (1.4, 1.7, 2.0, 2.3, 2.6)
RAW_FS = (1.4, 1.5, 1.7, 2.0, 2.3, 2.5, 2.6)
WINDS = (6.0, 8.0, 10.0)
AOAS = (0.0, 5.0, 10.0, 15.0)
FIG18_TWIST_FREQS = (2.0, 2.3, 2.6)
GRAM_FORCE_TO_NEWTON = 9.80665e-3
FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ = 2.6
# 2026-08-01 双 scope 契约终裁(fig19_cd_frequency_identity_exhaustion 一手资产穷尽):
# Fig19(c,d) 固定频率在公开资产中 UNRESOLVED 是终态,不可写为 ground truth。
# confirmed 域(42 曲线/151 条件,不含 fig19cd)= 晋升域;conditional_fig19_cd
# (8 曲线/48 条件)= 诊断域,禁止晋升/证伪 claim。本状态表示该契约已生效,
# fig19_cd 不再作为全局 promotion blocker(它只约束 conditional 域)。
FIG19_CD_FREQUENCY_STATUS = "conditional_scope"
EVIDENCE_CONFIRMED = "confirmed"
EVIDENCE_CONDITIONAL_FIG19_CD = "conditional_fig19_cd"
EVIDENCE_SCOPES = (
    EVIDENCE_CONFIRMED,
    EVIDENCE_CONDITIONAL_FIG19_CD,
)

# WebPlotDigitizer coordinates can fall just outside the drawn physical axis.
# These tolerances are far smaller than the acquisition spacing (2.5/5 deg
# and 0.1/0.2 Hz respectively) and only authorize projection to an endpoint,
# never model extrapolation.
DIGITIZATION_ENDPOINT_TOLERANCE = {
    "twist_deg": 0.25,
    "frequency_Hz": 0.01,
}

# The source PDF legend, rather than the original data.md 工况 labels, fixes
# these Fig. 18 thrust identities.  This is the same evidence-backed D1
# correction used by correct_fig18_curve_identity.py.  Values and x positions
# remain byte-for-byte derived from data.md; only their physical curve identity
# changes, and the transform is recorded in every scorecard.
FIG18_THRUST_IDENTITY_REMAP = {
    "18|a|6.0": "18|a|10.0",
    "18|a|10.0": "18|a|6.0",
    "18|c|(6.0, 2.0)": "18|c|(10.0, 2.0)",
    "18|c|(10.0, 2.0)": "18|c|(6.0, 2.0)",
    "18|c|(6.0, 2.3)": "18|c|(10.0, 2.3)",
    "18|c|(10.0, 2.3)": "18|c|(6.0, 2.3)",
    "18|c|(6.0, 2.6)": "18|c|(10.0, 2.6)",
    "18|c|(10.0, 2.6)": "18|c|(6.0, 2.6)",
}

Condition = tuple[float, float, float, float]  # U, f, nominal twist, AoA


def condition_key(condition: Condition) -> str:
    U, freq, twist, aoa = condition
    return f"{U:g}_{freq:g}_{twist:g}_{aoa:g}"


@dataclass(frozen=True)
class CurveSpec:
    figure: str
    panel: str
    key: str
    channel: str
    abscissa: str
    x: tuple[float, ...]
    conditions: tuple[Condition, ...]
    evidence_scope: str = EVIDENCE_CONFIRMED

    def __post_init__(self) -> None:
        if self.channel not in {"L", "T"}:
            raise ValueError(f"{self.key}: invalid channel {self.channel!r}")
        if self.abscissa not in {"twist_deg", "frequency_Hz"}:
            raise ValueError(f"{self.key}: invalid abscissa {self.abscissa!r}")
        if len(self.x) != len(self.conditions):
            raise ValueError(f"{self.key}: x/condition length mismatch")
        if self.evidence_scope not in EVIDENCE_SCOPES:
            raise ValueError(
                f"{self.key}: invalid evidence scope {self.evidence_scope!r}"
            )


@dataclass(frozen=True)
class MeasurementCurve:
    """One raw digitized curve with its corrected physical identity."""

    key: str
    source_key: str
    figure: str
    panel: str
    channel: str
    abscissa: str
    x: tuple[float, ...]
    values_g: tuple[float, ...]
    values_N: tuple[float, ...]
    condition_label: str
    source_line_start: int
    source_line_end: int
    identity_correction: str | None = None

    def __post_init__(self) -> None:
        lengths = {len(self.x), len(self.values_g), len(self.values_N)}
        if len(lengths) != 1:
            raise ValueError(f"{self.key}: measurement array length mismatch")


def curve_specs() -> tuple[CurveSpec, ...]:
    curves: list[CurveSpec] = []

    # Fig. 17(a,b): U=8, AoA=5, twist sweep at five frequencies.
    for freq in FS:
        conditions = tuple((8.0, freq, twist, 5.0) for twist in TWS)
        curves.extend(
            (
                CurveSpec(
                    "17", "a", f"17|a|{freq:.1f}", "T", "twist_deg", TWS, conditions
                ),
                CurveSpec(
                    "17", "b", f"17|b|{freq:.1f}", "L", "twist_deg", TWS, conditions
                ),
            )
        )

    # Fig. 18(a,b): nominal twist=22.5, AoA=5, frequency sweep by wind.
    for U in WINDS:
        conditions = tuple((U, freq, 22.5, 5.0) for freq in FS)
        curves.extend(
            (
                CurveSpec(
                    "18", "a", f"18|a|{U:.1f}", "T", "frequency_Hz", FS, conditions
                ),
                CurveSpec(
                    "18", "b", f"18|b|{U:.1f}", "L", "frequency_Hz", FS, conditions
                ),
            )
        )

    # Fig. 18(c,d): AoA=5, twist sweep by wind and frequency.
    for U in WINDS:
        for freq in FIG18_TWIST_FREQS:
            conditions = tuple((U, freq, twist, 5.0) for twist in TWS)
            parameter = f"({U:.1f}, {freq:.1f})"
            curves.extend(
                (
                    CurveSpec(
                        "18",
                        "c",
                        f"18|c|{parameter}",
                        "T",
                        "twist_deg",
                        TWS,
                        conditions,
                    ),
                    CurveSpec(
                        "18",
                        "d",
                        f"18|d|{parameter}",
                        "L",
                        "twist_deg",
                        TWS,
                        conditions,
                    ),
                )
            )

    # Fig. 19(a,b,c,d): U=8, four AoA families.
    for aoa in AOAS:
        freq_conditions = tuple((8.0, freq, 22.5, aoa) for freq in FS)
        twist_conditions = tuple(
            (
                8.0,
                FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ,
                twist,
                aoa,
            )
            for twist in TWS
        )
        curves.extend(
            (
                CurveSpec(
                    "19", "a", f"19|a|{aoa:g}", "T", "frequency_Hz", FS, freq_conditions
                ),
                CurveSpec(
                    "19", "b", f"19|b|{aoa:g}", "L", "frequency_Hz", FS, freq_conditions
                ),
                CurveSpec(
                    "19",
                    "c",
                    f"19|c|{aoa:g}",
                    "T",
                    "twist_deg",
                    TWS,
                    twist_conditions,
                    evidence_scope=EVIDENCE_CONDITIONAL_FIG19_CD,
                ),
                CurveSpec(
                    "19",
                    "d",
                    f"19|d|{aoa:g}",
                    "L",
                    "twist_deg",
                    TWS,
                    twist_conditions,
                    evidence_scope=EVIDENCE_CONDITIONAL_FIG19_CD,
                ),
            )
        )

    result = tuple(curves)
    if len(result) != 50:
        raise AssertionError(
            f"benchmark contract drift: {len(result)} curves, expected 50"
        )
    return result


CURVES = curve_specs()
CURVES_BY_EVIDENCE_SCOPE = {
    scope: tuple(curve for curve in CURVES if curve.evidence_scope == scope)
    for scope in EVIDENCE_SCOPES
}
CONDITIONS_BY_EVIDENCE_SCOPE = {
    scope: tuple(
        sorted(
            {
                condition
                for curve in CURVES_BY_EVIDENCE_SCOPE[scope]
                for condition in curve.conditions
            },
            key=lambda item: tuple(float(value) for value in item),
        )
    )
    for scope in EVIDENCE_SCOPES
}
CONDITIONS = tuple(
    sorted(
        {condition for curve in CURVES for condition in curve.conditions},
        key=lambda item: tuple(float(value) for value in item),
    )
)
if len(CONDITIONS) != 184:
    raise AssertionError(
        f"benchmark contract drift: {len(CONDITIONS)} conditions, expected 184"
    )
if FIG19_CD_FREQUENCY_STATUS == "unresolved":
    confirmed_conditions = set(
        CONDITIONS_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]
    )
    conditional_conditions = set(
        CONDITIONS_BY_EVIDENCE_SCOPE[EVIDENCE_CONDITIONAL_FIG19_CD]
    )
    evidence_counts = (
        len(CURVES_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]),
        len(
            CURVES_BY_EVIDENCE_SCOPE[
                EVIDENCE_CONDITIONAL_FIG19_CD
            ]
        ),
        len(confirmed_conditions),
        len(conditional_conditions),
        len(confirmed_conditions & conditional_conditions),
        len(conditional_conditions - confirmed_conditions),
    )
    if evidence_counts != (42, 8, 151, 48, 15, 33):
        raise AssertionError(
            "evidence-scope contract drift: "
            f"{evidence_counts!r} != (42, 8, 151, 48, 15, 33)"
        )


_FIGURE_RE = re.compile(r"Figure\s+(\d+)\.?\s*\(([a-d])\)", re.IGNORECASE)
_POINT_RE = re.compile(
    r"^\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+)"
    r"\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+)"
    r"\s*$"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _condition_value(label: str, pattern: str, name: str) -> float:
    match = re.search(pattern, label, re.IGNORECASE)
    if match is None:
        raise ValueError(f"measurement condition {label!r}: missing {name}")
    return float(match.group(1))


def _source_curve_key(figure: str, panel: str, condition_label: str) -> str:
    if figure == "17":
        freq = _condition_value(condition_label, r"([\d.]+)\s*Hz", "frequency")
        return f"17|{panel}|{freq:.1f}"
    if figure == "18" and panel in {"a", "b"}:
        wind = _condition_value(condition_label, r"([\d.]+)\s*m/s", "wind speed")
        return f"18|{panel}|{wind:.1f}"
    if figure == "18" and panel in {"c", "d"}:
        wind = _condition_value(condition_label, r"([\d.]+)\s*m/s", "wind speed")
        freq = _condition_value(condition_label, r"([\d.]+)\s*Hz", "frequency")
        return f"18|{panel}|({wind:.1f}, {freq:.1f})"
    if figure == "19":
        aoa = _condition_value(condition_label, r"([\d.]+)\s*度", "AoA")
        return f"19|{panel}|{aoa:g}"
    raise ValueError(
        f"unsupported measurement identity: Fig{figure}({panel}), "
        f"condition={condition_label!r}"
    )


def _column_identity(line: str) -> tuple[str, str] | None:
    if "/g" not in line.lower():
        return None
    if re.search(r"\b[Tt]h(?:urst|rust)\b", line):
        channel = "T"
    elif re.search(r"[Ll]ift", line):
        channel = "L"
    else:
        raise ValueError(f"unrecognized measurement force column: {line.strip()!r}")
    if re.search(r"[Tt]wist\s+amplitude", line):
        abscissa = "twist_deg"
    elif re.search(r"[Ff]lapping\s+[Ff]requency", line):
        abscissa = "frequency_Hz"
    else:
        raise ValueError(f"unrecognized measurement abscissa column: {line.strip()!r}")
    return channel, abscissa


def load_measurements(
    path: Path | str = DEFAULT_DATA_MD,
    *,
    source_bytes: bytes | None = None,
) -> dict[str, MeasurementCurve]:
    """Parse all raw Fig. 17/18/19 digitized samples from ``data.md``.

    Figure captions are not used to infer force channels because the source
    captions for Fig. 18(c,d) and Fig. 19(c,d) are swapped.  The actual data
    column header is authoritative.  The evidence-backed Fig. 18 thrust wind
    identity correction is applied in-memory and retained as provenance.
    """

    source = Path(path)
    curves: dict[str, MeasurementCurve] = {}
    figure: str | None = None
    panel: str | None = None
    current: dict[str, Any] | None = None

    def flush(end_line: int) -> None:
        nonlocal current
        if current is None:
            return
        if not current["points"]:
            raise ValueError(
                f"{source}:{current['line_start']}: measurement curve has no points"
            )
        if current["channel"] is None or current["abscissa"] is None:
            raise ValueError(
                f"{source}:{current['line_start']}: missing data column header"
            )
        source_key = _source_curve_key(
            current["figure"], current["panel"], current["condition_label"]
        )
        key = FIG18_THRUST_IDENTITY_REMAP.get(source_key, source_key)
        correction = None
        if key != source_key:
            correction = (
                "Fig18 thrust wind identity corrected from source-PDF legend: "
                f"{source_key} -> {key}"
            )
        if key in curves:
            raise ValueError(f"{source}: duplicate measurement curve identity {key!r}")
        x = tuple(point[0] for point in current["points"])
        values_g = tuple(point[1] for point in current["points"])
        curves[key] = MeasurementCurve(
            key=key,
            source_key=source_key,
            figure=current["figure"],
            panel=current["panel"],
            channel=current["channel"],
            abscissa=current["abscissa"],
            x=x,
            values_g=values_g,
            values_N=tuple(value * GRAM_FORCE_TO_NEWTON for value in values_g),
            condition_label=current["condition_label"],
            source_line_start=current["line_start"],
            source_line_end=end_line,
            identity_correction=correction,
        )
        current = None

    if source_bytes is None:
        text = source.read_text(encoding="utf-8")
    else:
        try:
            text = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{source}: measurement source is not UTF-8") from exc
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        figure_match = _FIGURE_RE.search(line)
        if figure_match is not None:
            flush(line_number - 1)
            figure, panel = figure_match.group(1), figure_match.group(2).lower()
            continue
        if "工况" in line:
            flush(line_number - 1)
            if figure is None or panel is None:
                raise ValueError(
                    f"{source}:{line_number}: condition appears before a Figure heading"
                )
            condition_parts = re.split(r"[：:]", line, maxsplit=1)
            condition_label = condition_parts[-1].strip()
            current = {
                "figure": figure,
                "panel": panel,
                "condition_label": condition_label,
                "channel": None,
                "abscissa": None,
                "points": [],
                "line_start": line_number,
            }
            continue
        column = _column_identity(line)
        if column is not None:
            if current is None:
                raise ValueError(
                    f"{source}:{line_number}: data column appears before a condition"
                )
            current["channel"], current["abscissa"] = column
            continue
        point_match = _POINT_RE.match(line)
        if point_match is not None:
            if current is None:
                raise ValueError(
                    f"{source}:{line_number}: data point appears before a condition"
                )
            current["points"].append(
                (float(point_match.group(1)), float(point_match.group(2)))
            )
    flush(len(lines))
    return curves


def validate_measurement_contract(
    measurements: Mapping[str, MeasurementCurve],
    *,
    source_path: Path | str = DEFAULT_DATA_MD,
    source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate curve identity, channel, abscissa, raw x, units, and provenance."""

    source = Path(source_path)
    expected = {curve.key: curve for curve in CURVES}
    actual = set(measurements)
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    malformed: list[str] = []
    endpoint_projection_candidates = 0
    for key, curve in expected.items():
        measurement = measurements.get(key)
        if not isinstance(measurement, MeasurementCurve):
            if key in measurements:
                malformed.append(key)
            continue
        raw_x = np.asarray(measurement.x, dtype=float)
        values_g = np.asarray(measurement.values_g, dtype=float)
        values_N = np.asarray(measurement.values_N, dtype=float)
        nominal_raw_x = np.asarray(
            RAW_FS if curve.abscissa == "frequency_Hz" else TWS,
            dtype=float,
        )
        tolerance = DIGITIZATION_ENDPOINT_TOLERANCE[curve.abscissa]
        model_x = np.asarray(curve.x, dtype=float)
        identity_ok = (
            measurement.key == key
            and measurement.figure == curve.figure
            and measurement.panel == curve.panel
            and measurement.channel == curve.channel
            and measurement.abscissa == curve.abscissa
        )
        arrays_ok = (
            raw_x.ndim == 1
            and values_g.ndim == 1
            and values_N.ndim == 1
            and raw_x.size == values_g.size == values_N.size
            and raw_x.size == nominal_raw_x.size
            and raw_x.size >= 2
            and np.isfinite(raw_x).all()
            and np.isfinite(values_g).all()
            and np.isfinite(values_N).all()
            and np.all(np.diff(raw_x) > 0.0)
            and np.allclose(
                values_N,
                values_g * GRAM_FORCE_TO_NEWTON,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        x_ok = bool(
            arrays_ok
            and np.max(np.abs(raw_x - nominal_raw_x)) <= tolerance
            and raw_x[0] >= model_x[0] - tolerance
            and raw_x[-1] <= model_x[-1] + tolerance
        )
        if arrays_ok:
            endpoint_projection_candidates += int(raw_x[0] < model_x[0])
            endpoint_projection_candidates += int(raw_x[-1] > model_x[-1])
        if not identity_ok or not arrays_ok or not x_ok:
            malformed.append(key)

    if source_bytes is None:
        digest = _sha256_file(source)
        source_size_bytes = source.stat().st_size
    else:
        digest = hashlib.sha256(source_bytes).hexdigest()
        source_size_bytes = len(source_bytes)
    source_identity_passed = digest == FROZEN_DATA_MD_SHA256
    sample_count = sum(
        len(curve.x)
        for curve in measurements.values()
        if isinstance(curve, MeasurementCurve)
    )
    corrections = sorted(
        {
            curve.identity_correction
            for curve in measurements.values()
            if isinstance(curve, MeasurementCurve)
            and curve.identity_correction is not None
        }
    )
    evidence_scope_sample_counts = {
        scope: sum(
            len(measurement.x)
            for curve in CURVES_BY_EVIDENCE_SCOPE[scope]
            if isinstance(
                (measurement := measurements.get(curve.key)),
                MeasurementCurve,
            )
        )
        for scope in EVIDENCE_SCOPES
    }
    return {
        "passed": (
            not missing
            and not unexpected
            and not malformed
            and len(measurements) == 50
            and sample_count == 530
            and source_identity_passed
        ),
        "expected_curve_count": 50,
        "actual_curve_count": len(measurements),
        "expected_measurement_samples": 530,
        "actual_measurement_samples": sample_count,
        "missing_curve_keys": missing,
        "unexpected_curve_keys": unexpected,
        "malformed_curve_keys": sorted(set(malformed)),
        "channel_sample_counts": {
            channel: sum(
                len(curve.x)
                for curve in measurements.values()
                if isinstance(curve, MeasurementCurve) and curve.channel == channel
            )
            for channel in ("T", "L")
        },
        "evidence_scope_sample_counts": evidence_scope_sample_counts,
        "endpoint_projection_candidates": endpoint_projection_candidates,
        "identity_corrections": corrections,
        "source": {
            "path": _source_display_path(source),
            "sha256": digest,
            "frozen_sha256": FROZEN_DATA_MD_SHA256,
            "identity_passed": source_identity_passed,
            "size_bytes": source_size_bytes,
            "parser": "fig171819_benchmark.load_measurements",
            "parsed_from_verified_bytes": source_bytes is not None,
            "units": {
                "source_force": "gram-force",
                "score_force": "N",
                "gram_force_to_newton": GRAM_FORCE_TO_NEWTON,
            },
        },
    }


def _valid_result(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        return math.isfinite(float(value["L"])) and math.isfinite(float(value["T"]))
    except (KeyError, TypeError, ValueError):
        return False


def validate_repro_contract(repro: Mapping[str, Any]) -> dict[str, Any]:
    expected = {curve.key for curve in CURVES}
    actual = {key for key in repro if key.startswith(("17|", "18|", "19|"))}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    malformed: list[str] = []
    for curve in CURVES:
        record = repro.get(curve.key)
        if not isinstance(record, Mapping):
            continue
        try:
            x = np.asarray(record["x"], dtype=float)
            exp = np.asarray(record["exp"], dtype=float)
            kind = str(record["kind"])
        except (KeyError, TypeError, ValueError):
            malformed.append(curve.key)
            continue
        if (
            kind != curve.channel
            or x.ndim != 1
            or exp.ndim != 1
            or x.size != exp.size
            or x.size < 2
            or not np.isfinite(x).all()
            or not np.isfinite(exp).all()
            or np.any(np.diff(x) <= 0.0)
        ):
            malformed.append(curve.key)
    return {
        "passed": not missing and not unexpected and not malformed,
        "expected_curve_count": len(expected),
        "actual_curve_count": len(actual),
        "missing_curve_keys": missing,
        "unexpected_curve_keys": unexpected,
        "malformed_curve_keys": sorted(malformed),
    }


def coverage(
    sweep: Mapping[str, Any],
    *,
    evidence_scope: str | None = None,
) -> dict[str, Any]:
    if evidence_scope is not None and evidence_scope not in EVIDENCE_SCOPES:
        raise ValueError(f"unknown evidence scope {evidence_scope!r}")
    selected_curves = (
        CURVES
        if evidence_scope is None
        else CURVES_BY_EVIDENCE_SCOPE[evidence_scope]
    )
    selected_conditions = (
        CONDITIONS
        if evidence_scope is None
        else CONDITIONS_BY_EVIDENCE_SCOPE[evidence_scope]
    )
    expected_keys = {
        condition_key(condition) for condition in selected_conditions
    }
    global_expected_keys = {
        condition_key(condition) for condition in CONDITIONS
    }
    valid_keys = {key for key in expected_keys if _valid_result(sweep.get(key))}
    missing_keys = sorted(
        expected_keys - valid_keys,
        key=lambda key: tuple(float(value) for value in key.split("_")),
    )
    extra_keys = sorted(set(sweep) - global_expected_keys)
    figures: dict[str, dict[str, int]] = {}
    complete_curve_count = 0
    partial_curve_count = 0
    empty_curve_count = 0
    for figure in ("17", "18", "19"):
        specs = [
            curve for curve in selected_curves if curve.figure == figure
        ]
        n_complete = 0
        n_partial = 0
        n_empty = 0
        for curve in specs:
            n = sum(
                condition_key(condition) in valid_keys for condition in curve.conditions
            )
            if n == len(curve.conditions):
                n_complete += 1
            elif n:
                n_partial += 1
            else:
                n_empty += 1
        figures[figure] = {
            "expected_curves": len(specs),
            "complete_curves": n_complete,
            "partial_curves": n_partial,
            "empty_curves": n_empty,
        }
        complete_curve_count += n_complete
        partial_curve_count += n_partial
        empty_curve_count += n_empty
    return {
        "evidence_scope": evidence_scope or "all",
        "expected_unique_conditions": len(expected_keys),
        "valid_unique_conditions": len(valid_keys),
        "missing_unique_conditions": len(missing_keys),
        "missing_condition_keys": missing_keys,
        "extra_condition_keys": extra_keys,
        "expected_curves": len(selected_curves),
        "complete_curves": complete_curve_count,
        "partial_curves": partial_curve_count,
        "empty_curves": empty_curve_count,
        "figures": figures,
        "complete": len(valid_keys) == len(expected_keys),
    }


def _captured(model: np.ndarray, measured: np.ndarray, correlation: float) -> bool:
    scale = max(float(np.max(np.abs(measured))), 1.0e-9)
    measured_range = float(np.ptp(measured))
    model_range = float(np.ptp(model))
    if measured_range < 0.05 * scale:
        return model_range < max(2.0 * measured_range, 0.10 * scale)
    return correlation >= 0.5


def interpolate_model_to_measurement_x(
    model_x: Sequence[float],
    model_values: Sequence[float],
    measurement_x: Sequence[float],
    *,
    abscissa: str,
    curve_key: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpolate model values to raw measurement x without extrapolation.

    Small out-of-range endpoint offsets that fall inside the declared
    digitization tolerance are projected to the corresponding physical model
    boundary.  The original measurement x values are not mutated and every
    projection is returned as provenance.
    """

    if abscissa not in DIGITIZATION_ENDPOINT_TOLERANCE:
        raise ValueError(f"{curve_key}: unknown abscissa {abscissa!r}")
    mx = np.asarray(model_x, dtype=float)
    my = np.asarray(model_values, dtype=float)
    tx = np.asarray(measurement_x, dtype=float)
    prefix = f"{curve_key}: " if curve_key else ""
    if (
        mx.ndim != 1
        or my.ndim != 1
        or mx.size != my.size
        or mx.size < 2
        or not np.isfinite(mx).all()
        or not np.isfinite(my).all()
        or np.any(np.diff(mx) <= 0.0)
    ):
        raise ValueError(prefix + "invalid model interpolation curve")
    if (
        tx.ndim != 1
        or tx.size < 2
        or not np.isfinite(tx).all()
        or np.any(np.diff(tx) <= 0.0)
    ):
        raise ValueError(prefix + "invalid measurement abscissae")

    tolerance = DIGITIZATION_ENDPOINT_TOLERANCE[abscissa]
    lower_overrun = float(mx[0] - np.min(tx))
    upper_overrun = float(np.max(tx) - mx[-1])
    if lower_overrun > tolerance or upper_overrun > tolerance:
        raise ValueError(
            prefix + "measurement x lies outside model interpolation domain: "
            f"measurement=[{np.min(tx):.15g}, {np.max(tx):.15g}], "
            f"model=[{mx[0]:.15g}, {mx[-1]:.15g}], "
            f"digitization_tolerance={tolerance:g}"
        )

    evaluation_x = np.clip(tx, mx[0], mx[-1])
    projected_indices = np.flatnonzero(evaluation_x != tx)
    projected = [
        {
            "index": int(index),
            "raw_x": float(tx[index]),
            "evaluation_x": float(evaluation_x[index]),
        }
        for index in projected_indices
    ]
    values = np.interp(evaluation_x, mx, my)
    return values, {
        "direction": "model_to_measurement_x",
        "measurement_values_interpolated": False,
        "model_domain": [float(mx[0]), float(mx[-1])],
        "raw_measurement_domain": [float(tx[0]), float(tx[-1])],
        "digitization_endpoint_tolerance": tolerance,
        "boundary_projection_count": len(projected),
        "boundary_projections": projected,
        "evaluation_x": evaluation_x.tolist(),
    }


def _curve_row(
    curve: CurveSpec,
    measurements: Mapping[str, MeasurementCurve],
    sweep: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray | None]:
    available: list[bool] = []
    model_values: list[float] = []
    model_x: list[float] = []
    for x, condition in zip(curve.x, curve.conditions):
        value = sweep.get(condition_key(condition))
        valid = _valid_result(value)
        available.append(valid)
        if valid:
            model_x.append(float(x))
            model_values.append(float(value[curve.channel]))
    row: dict[str, Any] = {
        "figure": curve.figure,
        "panel": curve.panel,
        "curve": curve.key,
        "channel": curve.channel,
        "abscissa": curve.abscissa,
        "evidence_scope": curve.evidence_scope,
        "expected_points": len(curve.conditions),
        "expected_solver_points": len(curve.conditions),
        "expected_measurement_points": len(measurements[curve.key].x),
        "available_points": sum(available),
        "available_solver_points": sum(available),
        "complete": all(available),
    }
    if not all(available):
        return row, None

    measurement = measurements[curve.key]
    raw_x = np.asarray(measurement.x, dtype=float)
    measured = np.asarray(measurement.values_N, dtype=float)
    model, interpolation = interpolate_model_to_measurement_x(
        model_x,
        model_values,
        raw_x,
        abscissa=curve.abscissa,
        curve_key=curve.key,
    )
    error = model - measured
    if np.std(model) > 1.0e-12 and np.std(measured) > 1.0e-12:
        correlation = float(np.corrcoef(model, measured)[0, 1])
    else:
        correlation = 0.0
    row.update(
        {
            "mae_N": float(np.mean(np.abs(error))),
            "rmse_N": float(np.sqrt(np.mean(error * error))),
            "bias_N": float(np.mean(error)),
            "pearson_r": correlation,
            "captured": _captured(model, measured, correlation),
            "measurement_points": len(measured),
            "source_curve_key": measurement.source_key,
            "source_line_range": [
                measurement.source_line_start,
                measurement.source_line_end,
            ],
            "identity_correction": measurement.identity_correction,
            "measurement_x": raw_x.tolist(),
            "measurement_N": measured.tolist(),
            "model_at_measurement_x_N": model.tolist(),
            "error_N": error.tolist(),
            "interpolation": interpolation,
        }
    )
    return row, error


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate only the explicitly supplied evidence rows."""

    errors: dict[tuple[str, str], list[float]] = {}
    captured: dict[tuple[str, str], list[bool]] = {}
    curve_metrics: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        if not row.get("complete") or "error_N" not in row:
            continue
        figure = str(row["figure"])
        channel = str(row["channel"])
        error = np.asarray(row["error_N"], dtype=float)
        for key in (
            (figure, channel),
            (figure, "ALL"),
            ("ALL", channel),
            ("ALL", "ALL"),
        ):
            errors.setdefault(key, []).extend(error.tolist())
            captured.setdefault(key, []).append(bool(row["captured"]))
            metrics = curve_metrics.setdefault(
                key, {"mae_N": [], "rmse_N": [], "bias_N": []}
            )
            for name in metrics:
                metrics[name].append(float(row[name]))

    aggregates: dict[str, Any] = {}
    for figure in ("17", "18", "19", "ALL"):
        aggregates[figure] = {}
        for channel in ("T", "L", "ALL"):
            key = (figure, channel)
            values = np.asarray(errors.get(key, ()), dtype=float)
            capture_values = captured.get(key, ())
            if values.size == 0:
                aggregates[figure][channel] = None
                continue
            aggregates[figure][channel] = {
                "n_points": int(values.size),
                "n_complete_curves": len(capture_values),
                "mae_N": float(np.mean(np.abs(values))),
                "rmse_N": float(np.sqrt(np.mean(values * values))),
                "bias_N": float(np.mean(values)),
                "mean_curve_mae_N": float(
                    np.mean(curve_metrics[key]["mae_N"])
                ),
                "mean_curve_rmse_N": float(
                    np.mean(curve_metrics[key]["rmse_N"])
                ),
                "mean_curve_bias_N": float(
                    np.mean(curve_metrics[key]["bias_N"])
                ),
                "trend_capture": float(np.mean(capture_values)),
            }
    return aggregates


def scorecard(
    sweep: Mapping[str, Any],
    repro: Mapping[str, Any] | None = None,
    *,
    sweep_name: str = "",
    measurements: Mapping[str, MeasurementCurve] | None = None,
    measurement_path: Path | str = DEFAULT_DATA_MD,
) -> dict[str, Any]:
    """Score complete model curves against every raw data.md sample.

    ``repro`` is retained only for compatibility with the existing 184-runner.
    It is never used as ground truth because that JSON contains resampled
    measurements.  Callers may inject a parsed ``measurements`` mapping for
    deterministic tests; production defaults to ``measurement_path``.
    """

    del repro
    parsed_measurements = (
        load_measurements(measurement_path)
        if measurements is None
        else dict(measurements)
    )
    measurement_validation = validate_measurement_contract(
        parsed_measurements,
        source_path=measurement_path,
    )
    if not measurement_validation["passed"]:
        raise ValueError(
            f"invalid Fig. 17/18/19 raw measurement contract: {measurement_validation}"
        )

    all_rows: list[dict[str, Any]] = []
    for curve in CURVES:
        row, _ = _curve_row(curve, parsed_measurements, sweep)
        all_rows.append(row)

    rows_by_scope = {
        scope: [
            row for row in all_rows if row["evidence_scope"] == scope
        ]
        for scope in EVIDENCE_SCOPES
    }
    aggregates_by_scope = {
        scope: _aggregate_rows(rows_by_scope[scope])
        for scope in EVIDENCE_SCOPES
    }

    sweep_coverage = coverage(sweep)
    coverage_by_scope = {
        scope: coverage(sweep, evidence_scope=scope)
        for scope in EVIDENCE_SCOPES
    }
    # 双 scope 契约(2026-08-01): confirmed 域 identity 冻结即 gate0 冻结,
    # fig19_cd 频率 unresolved 只约束 conditional_fig19_cd 诊断域,不阻塞晋升。
    gate0_identity_frozen = bool(
        measurement_validation["source"]["identity_passed"]
    )
    confirmed_ready = bool(
        measurement_validation["passed"]
        and coverage_by_scope[EVIDENCE_CONFIRMED]["complete"]
    )
    confirmed_rows = rows_by_scope[EVIDENCE_CONFIRMED]
    confirmed_aggregates = aggregates_by_scope[EVIDENCE_CONFIRMED]
    return {
        "schema_version": 3,
        "benchmark": "Meng2025_Fig17_Fig18_Fig19",
        "sweep": sweep_name,
        "primary_evidence_scope": EVIDENCE_CONFIRMED,
        "contract": {
            "expected_curves": len(CURVES),
            "expected_unique_conditions": len(CONDITIONS),
            "solver_grid_conditions": len(CONDITIONS),
            "measurement_samples": measurement_validation["actual_measurement_samples"],
            "scoring_direction": "model_to_raw_measurement_x",
            "measurement_values_interpolated": False,
            "evidence_scope_curve_counts": {
                scope: len(CURVES_BY_EVIDENCE_SCOPE[scope])
                for scope in EVIDENCE_SCOPES
            },
            "evidence_scope_condition_counts": {
                scope: len(CONDITIONS_BY_EVIDENCE_SCOPE[scope])
                for scope in EVIDENCE_SCOPES
            },
            "status": (
                "conditional_diagnostic"
                if not gate0_identity_frozen
                else "identity_frozen"
            ),
            "gate0_identity_frozen": gate0_identity_frozen,
            "fig19_cd_frequency": {
                "status": FIG19_CD_FREQUENCY_STATUS,
                "conditional_assumption_Hz": (FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ),
                # 双 scope 契约: 频率 unresolved 只约束 conditional 诊断域,
                # confirmed 晋升域不再被其阻塞。
                "promotion_blocked_until_resolved": (
                    FIG19_CD_FREQUENCY_STATUS not in (
                        "resolved",
                        "conditional_scope",
                    )
                ),
                "scope_contract": (
                    "confirmed=promotion_domain; "
                    "conditional_fig19_cd=diagnostic_only"
                ),
            },
            "figure_curve_counts": {
                figure: sum(curve.figure == figure for curve in CURVES)
                for figure in ("17", "18", "19")
            },
            "measurement_validation": measurement_validation,
        },
        "coverage": sweep_coverage,
        "rows": confirmed_rows,
        "aggregates": confirmed_aggregates,
        "evidence_scopes": {
            EVIDENCE_CONFIRMED: {
                "coverage": coverage_by_scope[EVIDENCE_CONFIRMED],
                "rows": confirmed_rows,
                "aggregates": confirmed_aggregates,
                "residual_fingerprint_ready": confirmed_ready,
                "allowed_use": "baseline_diagnosis_and_preregistration",
            },
            EVIDENCE_CONDITIONAL_FIG19_CD: {
                "coverage": coverage_by_scope[
                    EVIDENCE_CONDITIONAL_FIG19_CD
                ],
                "rows": rows_by_scope[
                    EVIDENCE_CONDITIONAL_FIG19_CD
                ],
                "aggregates": aggregates_by_scope[
                    EVIDENCE_CONDITIONAL_FIG19_CD
                ],
                "residual_fingerprint_ready": False,
                "conditional_assumption_Hz": (
                    FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ
                ),
                "forbidden_use": "claim_decision_or_global_promotion",
            },
        },
        "diagnostic_all_scopes_not_for_claim_decision": {
            "warning": (
                "Mixed-scope statistics are diagnostic only and must not "
                "select or promote an aerodynamic claim."
            ),
            "aggregates": _aggregate_rows(all_rows),
        },
        "promotion_eligible": bool(
            measurement_validation["passed"]
            and confirmed_ready
            and gate0_identity_frozen
        ),
        "promotion_blockers": [
            blocker
            for blocker, active in (
                (
                    "raw_measurement_contract_invalid",
                    not measurement_validation["passed"],
                ),
                ("solver_grid_incomplete", not confirmed_ready),
                (
                    "fig19_cd_fixed_frequency_unresolved",
                    FIG19_CD_FREQUENCY_STATUS not in (
                        "resolved",
                        "conditional_scope",
                    ),
                ),
            )
            if active
        ],
    }


def build_evidence_scope_artifact(
    report: Mapping[str, Any],
    *,
    evidence_scope: str = EVIDENCE_CONFIRMED,
    require_complete: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a self-contained, non-promotional residual evidence artifact."""

    if evidence_scope not in EVIDENCE_SCOPES:
        raise ValueError(f"unknown evidence scope {evidence_scope!r}")
    if report.get("schema_version") != 3:
        raise ValueError("evidence artifact requires scorecard schema version 3")
    scopes = report.get("evidence_scopes")
    if not isinstance(scopes, Mapping) or evidence_scope not in scopes:
        raise ValueError(f"scorecard lacks evidence scope {evidence_scope!r}")
    scope_report = scopes[evidence_scope]
    if not isinstance(scope_report, Mapping):
        raise ValueError(f"invalid evidence scope payload {evidence_scope!r}")
    scope_coverage = scope_report.get("coverage")
    scope_rows = scope_report.get("rows")
    if not isinstance(scope_coverage, Mapping) or not isinstance(
        scope_rows, list
    ):
        raise ValueError(f"incomplete evidence scope payload {evidence_scope!r}")
    if require_complete and not scope_coverage.get("complete"):
        raise ValueError(
            f"evidence scope {evidence_scope!r} is not solver-complete"
        )
    if require_complete and any(not row.get("complete") for row in scope_rows):
        raise ValueError(
            f"evidence scope {evidence_scope!r} has incomplete curve rows"
        )

    residual_points = sum(
        len(row.get("error_N", ())) for row in scope_rows
    )
    excluded_curves = [
        curve.key for curve in CURVES if curve.evidence_scope != evidence_scope
    ]
    return {
        "schema_version": 1,
        "artifact_type": "baseline_residual_evidence_scope",
        "benchmark": report.get("benchmark"),
        "sweep": report.get("sweep"),
        "evidence_scope": evidence_scope,
        "status": (
            "ready_for_baseline_diagnosis"
            if scope_report.get("residual_fingerprint_ready")
            else "not_ready"
        ),
        "coverage": dict(scope_coverage),
        "curve_rows": list(scope_rows),
        "aggregates": scope_report.get("aggregates"),
        "residual_point_count": residual_points,
        "excluded_curve_keys": excluded_curves,
        "exclusion_reason": (
            "Fig19(c,d) fixed frequency is unresolved; its 2.6 Hz "
            "solver mapping is conditional."
        ),
        "provenance": dict(provenance or {}),
        "global_promotion_eligible": bool(
            report.get("promotion_eligible", False)
        ),
        "global_promotion_blockers": list(
            report.get("promotion_blockers", ())
        ),
        "allowed_use": "baseline_diagnosis_and_preregistration",
        "forbidden_use": "final_50_curve_claim_promotion",
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _source_display_path(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_MD,
        help="raw digitized Fig. 17/18/19 measurement source",
    )
    parser.add_argument("--repro", type=Path, default=DEFAULT_REPRO)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--confirmed-output",
        type=Path,
        help="write a confirmed-only residual evidence artifact",
    )
    parser.add_argument(
        "--runner-manifest",
        type=Path,
        help="frozen runner manifest used by --confirmed-output",
    )
    parser.add_argument(
        "--runner-scorecard",
        type=Path,
        help="original runner scorecard used by --confirmed-output",
    )
    args = parser.parse_args(argv)
    if args.confirmed_output is not None and (
        args.runner_manifest is None or args.runner_scorecard is None
    ):
        parser.error(
            "--confirmed-output requires --runner-manifest and "
            "--runner-scorecard"
        )

    sweep = _load_json(args.sweep)
    repro = _load_json(args.repro)
    report = scorecard(
        sweep,
        repro,
        sweep_name=str(args.sweep),
        measurement_path=args.data,
    )
    coverage_report = report["coverage"]
    overall = report["aggregates"]["ALL"]["ALL"]
    print(
        f"Fig17/18/19 coverage: {coverage_report['complete_curves']}/"
        f"{coverage_report['expected_curves']} complete curves; "
        f"{coverage_report['valid_unique_conditions']}/"
        f"{coverage_report['expected_unique_conditions']} unique conditions; "
        f"{report['contract']['measurement_samples']} raw measurement samples"
    )
    if overall is not None:
        print(
            f"confirmed-scope aggregate: MAE={overall['mae_N']:.6f} N "
            f"RMSE={overall['rmse_N']:.6f} N "
            f"bias={overall['bias_N']:+.6f} N "
            f"capture={overall['trend_capture']:.3f}"
        )
    for figure in ("17", "18", "19"):
        item = coverage_report["figures"][figure]
        print(
            f"  Fig{figure}: {item['complete_curves']}/"
            f"{item['expected_curves']} complete"
        )
    if args.output is not None:
        _write_json_atomic(args.output, report)
        print(f"saved {args.output}")
    if args.confirmed_output is not None:
        provenance = {
            "sweep_result": _file_identity(args.sweep),
            "runner_manifest": _file_identity(args.runner_manifest),
            "original_runner_scorecard": _file_identity(
                args.runner_scorecard
            ),
            "measurement_data": _file_identity(args.data),
            "scorer_source": _file_identity(Path(__file__).resolve()),
        }
        artifact = build_evidence_scope_artifact(
            report,
            evidence_scope=EVIDENCE_CONFIRMED,
            provenance=provenance,
        )
        _write_json_atomic(args.confirmed_output, artifact)
        print(f"saved {args.confirmed_output}")
    if report["promotion_blockers"]:
        print("promotion blockers: " + ", ".join(report["promotion_blockers"]))
    return 0 if report["promotion_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
