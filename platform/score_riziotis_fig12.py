#!/usr/bin/env python3
"""Strict source-response scorer for Riziotis & Voutsinas (2008), Fig. 12.

This module implements only the frozen N2.6e1 response gate.  It does not
contain, call, or tune an aerodynamic model.

Candidate CSV rows must identify every published double-wake reference point
either with ``panel,surface,point_index`` or with the derived stable key
``fig12:{panel}:{surface}:{point_index:03d}``.  Every row must also provide
``x_over_c`` and ``cp_candidate``.  The scorer requires the candidate x
coordinate to match the corresponding frozen reference coordinate; it never
interpolates, extrapolates, or performs nearest-neighbour matching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_REFERENCE_CSV = (
    Path(__file__).resolve().parent
    / "docs"
    / "diag"
    / "n26e1_fig12"
    / "fig12_digitized.csv"
)
EXPECTED_REFERENCE_CSV_SHA256 = (
    "f59f8c0ee05ce371d7d89d1e1b2f8081ea700afc43dec2279e63e7380eb8f5c8"
)
EXPECTED_SOURCE_ID = "riziotis_voutsinas_2008_fig12"
EXPECTED_SOURCE_PDF_SHA256 = (
    "cc4970b38b3586affc4805a84e526fcb0049ba2dfa42219c01379e2a8f48fa84"
)
EXPECTED_EXTRACTION_VERSION = "riziotis_fig12_vector_v2"
PANELS = tuple("abcdefgh")
SURFACES = ("upper", "lower")
EXPECTED_REFERENCE_ROWS = 619
NRMSE_LIMIT = 0.05
X_MATCH_ABS_TOL = 1.0e-10
REPORT_SCHEMA_VERSION = "n26e1_fig12_source_response_score_v1"


class ScoringInputError(RuntimeError):
    """Raised when a reference or candidate violates the strict row contract."""


@dataclass(frozen=True, order=True)
class RowKey:
    panel: str
    surface: str
    point_index: int

    @property
    def stable(self) -> str:
        return f"fig12:{self.panel}:{self.surface}:{self.point_index:03d}"

    @classmethod
    def parse_stable(cls, value: str) -> "RowKey":
        parts = value.split(":")
        if len(parts) != 4 or parts[0] != "fig12":
            raise ScoringInputError(
                f"invalid row_key {value!r}; expected "
                "fig12:<panel>:<surface>:<zero-padded-index>"
            )
        try:
            point_index = int(parts[3])
        except ValueError as exc:
            raise ScoringInputError(
                f"invalid point index in row_key {value!r}"
            ) from exc
        key = cls(parts[1], parts[2], point_index)
        if value != key.stable:
            raise ScoringInputError(
                f"row_key is not canonical: {value!r} != {key.stable!r}"
            )
        key.validate()
        return key

    def validate(self) -> None:
        if self.panel not in PANELS:
            raise ScoringInputError(f"unknown Fig. 12 panel {self.panel!r}")
        if self.surface not in SURFACES:
            raise ScoringInputError(f"unknown surface {self.surface!r}")
        if self.point_index < 0:
            raise ScoringInputError(f"negative point_index in {self.stable!r}")


@dataclass(frozen=True)
class ReferencePoint:
    key: RowKey
    x_over_c: float
    cp: float
    source_vector_sha256: str
    vector_path_id: str


@dataclass(frozen=True)
class CandidatePoint:
    key: RowKey
    x_over_c: float
    cp_candidate: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: str, *, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoringInputError(
            f"row {row_number}: {field} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ScoringInputError(
            f"row {row_number}: {field} must be finite, got {value!r}"
        )
    return parsed


def _integer(value: str, *, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ScoringInputError(
            f"row {row_number}: {field} is not an integer: {value!r}"
        ) from exc
    return parsed


def _require_columns(
    fieldnames: Sequence[str] | None,
    required: set[str],
    *,
    role: str,
) -> None:
    available = set(fieldnames or ())
    missing = required - available
    if missing:
        raise ScoringInputError(
            f"{role} CSV is missing required columns: {sorted(missing)}"
        )


def load_frozen_reference(
    path: Path = DEFAULT_REFERENCE_CSV,
) -> dict[RowKey, ReferencePoint]:
    """Load and integrity-check the sole frozen source-response oracle."""

    path = path.resolve()
    if not path.is_file():
        raise ScoringInputError(f"frozen reference CSV does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != EXPECTED_REFERENCE_CSV_SHA256:
        raise ScoringInputError(
            "frozen reference CSV fingerprint drift: "
            f"{actual_sha256} != {EXPECTED_REFERENCE_CSV_SHA256}"
        )

    required = {
        "source_id",
        "source_sha256",
        "series",
        "panel",
        "surface",
        "point_index",
        "x_over_c",
        "cp",
        "source_vector_sha256",
        "vector_path_id",
        "extraction_version",
        "exclusion_flag",
    }
    result: dict[RowKey, ReferencePoint] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader.fieldnames, required, role="reference")
        for row_number, row in enumerate(reader, start=2):
            if row["series"] != "published_double_wake":
                continue
            if row["source_id"] != EXPECTED_SOURCE_ID:
                raise ScoringInputError(
                    f"reference row {row_number}: unexpected source_id"
                )
            if row["source_sha256"] != EXPECTED_SOURCE_PDF_SHA256:
                raise ScoringInputError(
                    f"reference row {row_number}: unexpected source SHA-256"
                )
            if row["extraction_version"] != EXPECTED_EXTRACTION_VERSION:
                raise ScoringInputError(
                    f"reference row {row_number}: unexpected extraction version"
                )
            if row["exclusion_flag"].lower() != "false":
                raise ScoringInputError(
                    f"reference row {row_number}: scored row is excluded"
                )
            key = RowKey(
                panel=row["panel"],
                surface=row["surface"],
                point_index=_integer(
                    row["point_index"],
                    field="point_index",
                    row_number=row_number,
                ),
            )
            key.validate()
            if key in result:
                raise ScoringInputError(f"duplicate frozen reference key {key.stable}")
            vector_path_id = row["vector_path_id"].strip()
            source_vector_sha256 = row["source_vector_sha256"].strip()
            if not vector_path_id or not source_vector_sha256:
                raise ScoringInputError(
                    f"reference row {row_number}: missing vector provenance"
                )
            result[key] = ReferencePoint(
                key=key,
                x_over_c=_finite_float(
                    row["x_over_c"],
                    field="x_over_c",
                    row_number=row_number,
                ),
                cp=_finite_float(row["cp"], field="cp", row_number=row_number),
                source_vector_sha256=source_vector_sha256,
                vector_path_id=vector_path_id,
            )

    if len(result) != EXPECTED_REFERENCE_ROWS:
        raise ScoringInputError(
            f"frozen reference contains {len(result)} model rows, "
            f"expected {EXPECTED_REFERENCE_ROWS}"
        )
    expected_groups = {(panel, surface) for panel in PANELS for surface in SURFACES}
    actual_groups = {(key.panel, key.surface) for key in result}
    if actual_groups != expected_groups:
        raise ScoringInputError(
            "frozen reference does not contain exactly 8 panels x 2 surfaces"
        )
    for panel, surface in sorted(expected_groups):
        points = sorted(
            (
                point
                for key, point in result.items()
                if key.panel == panel and key.surface == surface
            ),
            key=lambda point: point.key.point_index,
        )
        indices = [point.key.point_index for point in points]
        if indices != list(range(len(points))):
            raise ScoringInputError(
                f"reference indices are not contiguous for {panel}/{surface}"
            )
        x_values = [point.x_over_c for point in points]
        if any(right < left for left, right in zip(x_values[:-1], x_values[1:])):
            raise ScoringInputError(
                f"reference x/c is not monotone for {panel}/{surface}"
            )
    return result


def _candidate_key(
    row: Mapping[str, str],
    *,
    row_number: int,
    has_structured_key: bool,
    has_stable_key: bool,
) -> RowKey:
    structured: RowKey | None = None
    stable: RowKey | None = None
    if has_structured_key:
        structured = RowKey(
            panel=row["panel"],
            surface=row["surface"],
            point_index=_integer(
                row["point_index"],
                field="point_index",
                row_number=row_number,
            ),
        )
        structured.validate()
    if has_stable_key:
        stable = RowKey.parse_stable(row["row_key"])
    if structured is not None and stable is not None and structured != stable:
        raise ScoringInputError(
            f"row {row_number}: structured key {structured.stable} "
            f"disagrees with row_key {stable.stable}"
        )
    assert structured is not None or stable is not None
    return structured if structured is not None else stable  # type: ignore[return-value]


def load_candidate(
    path: Path,
    reference: Mapping[RowKey, ReferencePoint],
) -> dict[RowKey, CandidatePoint]:
    """Load a complete, one-to-one candidate without spatial resampling."""

    path = path.resolve()
    if not path.is_file():
        raise ScoringInputError(f"candidate CSV does not exist: {path}")
    result: dict[RowKey, CandidatePoint] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        _require_columns(
            reader.fieldnames,
            {"x_over_c", "cp_candidate"},
            role="candidate",
        )
        structured_fields = {"panel", "surface", "point_index"}
        structured_present = structured_fields & fields
        if structured_present and structured_present != structured_fields:
            raise ScoringInputError(
                "candidate CSV has a partial structured key; require all of "
                "panel,surface,point_index"
            )
        has_structured_key = structured_fields <= fields
        has_stable_key = "row_key" in fields
        if not has_structured_key and not has_stable_key:
            raise ScoringInputError(
                "candidate CSV requires panel,surface,point_index or row_key"
            )

        for row_number, row in enumerate(reader, start=2):
            key = _candidate_key(
                row,
                row_number=row_number,
                has_structured_key=has_structured_key,
                has_stable_key=has_stable_key,
            )
            if key in result:
                raise ScoringInputError(
                    f"candidate row {row_number}: duplicate key {key.stable}"
                )
            if key not in reference:
                raise ScoringInputError(
                    f"candidate row {row_number}: unknown key {key.stable}"
                )
            x_over_c = _finite_float(
                row["x_over_c"],
                field="x_over_c",
                row_number=row_number,
            )
            cp_candidate = _finite_float(
                row["cp_candidate"],
                field="cp_candidate",
                row_number=row_number,
            )
            x_reference = reference[key].x_over_c
            if abs(x_over_c - x_reference) > X_MATCH_ABS_TOL:
                raise ScoringInputError(
                    f"candidate row {row_number}: x/c mismatch for "
                    f"{key.stable}: {x_over_c:.17g} vs "
                    f"{x_reference:.17g}; interpolation/extrapolation is "
                    "forbidden"
                )
            result[key] = CandidatePoint(
                key=key,
                x_over_c=x_over_c,
                cp_candidate=cp_candidate,
            )

    missing = sorted(set(reference) - set(result))
    if missing:
        preview = ", ".join(key.stable for key in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ScoringInputError(
            f"candidate is missing {len(missing)} frozen rows: " f"{preview}{suffix}"
        )
    if len(result) != len(reference):
        raise ScoringInputError(
            f"candidate/reference row-count mismatch: "
            f"{len(result)} != {len(reference)}"
        )
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    """NumPy-compatible default linear percentile (Hyndman-Fan type 7)."""

    if not values:
        raise ScoringInputError("cannot compute a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def score_candidate(path: Path) -> dict[str, object]:
    reference = load_frozen_reference()
    candidate = load_candidate(path, reference)
    group_reports: list[dict[str, object]] = []
    for panel in PANELS:
        for surface in SURFACES:
            keys = sorted(
                key
                for key in reference
                if key.panel == panel and key.surface == surface
            )
            reference_cp = [reference[key].cp for key in keys]
            errors = [candidate[key].cp_candidate - reference[key].cp for key in keys]
            rmse = math.sqrt(math.fsum(error * error for error in errors) / len(errors))
            robust_range = _percentile(reference_cp, 0.95) - _percentile(
                reference_cp, 0.05
            )
            if not math.isfinite(robust_range) or robust_range <= 0.0:
                raise ScoringInputError(
                    f"non-positive P95-P5 reference range for " f"{panel}/{surface}"
                )
            nrmse = rmse / robust_range
            group_reports.append(
                {
                    "panel": panel,
                    "surface": surface,
                    "n_points": len(keys),
                    "rmse_cp": rmse,
                    "reference_p05_cp": _percentile(reference_cp, 0.05),
                    "reference_p95_cp": _percentile(reference_cp, 0.95),
                    "reference_p95_minus_p05_cp": robust_range,
                    "nrmse": nrmse,
                    "limit": NRMSE_LIMIT,
                    "passed": nrmse <= NRMSE_LIMIT,
                    "maximum_absolute_error_cp": max(abs(error) for error in errors),
                }
            )

    passed_count = sum(bool(group["passed"]) for group in group_reports)
    worst = max(group_reports, key=lambda group: float(group["nrmse"]))
    all_passed = passed_count == len(group_reports)
    candidate_path = path.resolve()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS" if all_passed else "FAIL",
        "passed": all_passed,
        "gate": {
            "claim": "N2.6e1",
            "name": "published_model_source_response",
            "formula": (
                "RMSE(cp_candidate-cp_published_double_wake)"
                "/(P95-P5)(cp_published_double_wake)"
            ),
            "per_panel_per_surface_limit": NRMSE_LIMIT,
            "required_independent_groups": 16,
            "aggregation_policy": "all groups must pass; no averaging",
            "percentile_method": (
                "Hyndman-Fan type 7 linear percentile (NumPy default)"
            ),
            "x_match_abs_tolerance": X_MATCH_ABS_TOL,
            "spatial_resampling": "forbidden",
        },
        "reference": {
            "path": str(DEFAULT_REFERENCE_CSV),
            "csv_sha256": EXPECTED_REFERENCE_CSV_SHA256,
            "source_id": EXPECTED_SOURCE_ID,
            "source_pdf_sha256": EXPECTED_SOURCE_PDF_SHA256,
            "row_count": len(reference),
        },
        "candidate": {
            "path": str(candidate_path),
            "csv_sha256": _sha256_file(candidate_path),
            "row_count": len(candidate),
        },
        "summary": {
            "passed_groups": passed_count,
            "required_groups": len(group_reports),
            "failed_groups": len(group_reports) - passed_count,
            "maximum_nrmse": float(worst["nrmse"]),
            "worst_panel": worst["panel"],
            "worst_surface": worst["surface"],
        },
        "groups": group_reports,
    }


def format_summary(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    assert isinstance(summary, Mapping)
    return (
        f"{report['status']}: "
        f"{summary['passed_groups']}/{summary['required_groups']} "
        "Fig.12 panel/surface gates passed; "
        f"max nRMSE={float(summary['maximum_nrmse']):.6g} at "
        f"{summary['worst_panel']}/{summary['worst_surface']}"
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate_csv",
        type=Path,
        help=(
            "candidate rows with panel/surface/point_index (or row_key), "
            "x_over_c, and cp_candidate"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional path receiving the same JSON report emitted on stdout",
    )
    return parser


def _error_report(exc: Exception, candidate_path: Path) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "INPUT_ERROR",
        "passed": False,
        "candidate": {"path": str(candidate_path.resolve())},
        "errors": [str(exc)],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        report = score_candidate(args.candidate_csv)
    except (ScoringInputError, OSError) as exc:
        report = _error_report(exc, args.candidate_csv)
        exit_code = 2
    else:
        exit_code = 0 if bool(report["passed"]) else 1
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(serialized)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    if report["status"] == "INPUT_ERROR":
        print(f"INPUT_ERROR: {report['errors'][0]}", file=sys.stderr)
    else:
        print(format_summary(report), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
