#!/usr/bin/env python3
"""Audit the P0 independent dynamic-pressure observation ledger.

Only small selected ZIP members are fetched with HTTP Range requests.  No
FLUXV prediction is executed and no aerodynamic coefficient is fitted.
"""
from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from audit_high_fidelity_field_candidates import _zip_index, _zip_member


PLATFORM = Path(__file__).resolve().parent
CASES = (
    PLATFORM
    / "docs"
    / "diag"
    / "unified_pressure_observation_cases.yaml"
)
OUTPUT = (
    PLATFORM
    / "docs"
    / "diag"
    / "unified_pressure_observation_results.json"
)


class PressureObservationError(ValueError):
    """The public pressure asset violates its frozen observation contract."""


def _finite(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise PressureObservationError(f"{name} contains non-finite values")
    return array


def integrate_profile_pressure(
    coordinates: Any,
    cp: Any,
    *,
    excluded_segments: Iterable[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate pressure coefficients on an ordered clockwise 2-D profile.

    ``excluded_segments`` contains indices ``i`` for edges ``i -> i+1``.
    The returned coefficients follow the public asset convention:
    ``(Cn, Ct, Cm_LE)``.
    """
    xy = _finite("coordinates", coordinates)
    pressure = _finite("cp", cp)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        raise PressureObservationError("coordinates must have shape (n,2)")
    if pressure.ndim == 1:
        pressure = pressure[None, :]
    if pressure.ndim != 2 or pressure.shape[1] != len(xy):
        raise PressureObservationError(
            "cp must have shape (phase,n_coordinate)"
        )
    use = np.ones(len(xy) - 1, dtype=bool)
    for raw_index in excluded_segments:
        index = int(raw_index)
        if index < 0 or index >= len(use):
            raise PressureObservationError(
                f"excluded segment {index} is out of range"
            )
        use[index] = False

    delta = np.diff(xy, axis=0)
    midpoint = 0.5 * (xy[:-1] + xy[1:])
    cp_midpoint = 0.5 * (pressure[:, :-1] + pressure[:, 1:])
    cn = -np.sum(cp_midpoint[:, use] * delta[None, use, 0], axis=1)
    ct = np.sum(cp_midpoint[:, use] * delta[None, use, 1], axis=1)
    cm_le = np.sum(
        cp_midpoint[:, use]
        * (
            midpoint[None, use, 0] * delta[None, use, 0]
            + midpoint[None, use, 1] * delta[None, use, 1]
        ),
        axis=1,
    )
    return cn, ct, cm_le


def _table(payload: bytes) -> tuple[list[str], np.ndarray]:
    text = payload.decode("utf-8", errors="strict")
    rows = list(csv.reader(StringIO(text), delimiter="\t"))
    if len(rows) < 2:
        raise PressureObservationError("TSV has no data rows")
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        raise PressureObservationError("TSV row width is inconsistent")
    values = np.asarray(
        [
            [
                float(cell) if cell.strip() else np.nan
                for cell in row
            ]
            for row in rows[1:]
        ],
        dtype=float,
    )
    return rows[0], values


def _residual(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    error = np.asarray(observed) - np.asarray(reference)
    return {
        "max_abs": float(np.max(np.abs(error), initial=0.0)),
        "rms": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def _member_table(
    url: str,
    records: dict[str, dict[str, Any]],
    name: str,
) -> tuple[list[str], np.ndarray]:
    if name not in records:
        raise PressureObservationError(f"ZIP member is missing: {name}")
    return _table(_zip_member(url, records[name]))


def _case_stem(configuration: str, frequency: str) -> str:
    return (
        f"Re1m_forced_{configuration}_meanAoA10_amp10_"
        f"freq{frequency}.tsv"
    )


def main() -> int:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    asset = contract["asset"]
    selection = contract["selection"]
    url = asset["archive_url"]
    index = _zip_index(url, int(asset["archive_size_bytes"]))
    records = index["records"]

    coordinate_header, coordinates = _member_table(
        url, records, selection["coordinate_member"]
    )
    coordinate_count = int(contract["data_identity"]["coordinate_count"])
    if coordinate_header != ["x/c", "y/c"]:
        raise PressureObservationError("unexpected coordinate header")
    if coordinates.shape != (coordinate_count, 2):
        raise PressureObservationError("unexpected coordinate shape")
    if not np.all(np.isfinite(coordinates)):
        raise PressureObservationError("coordinates contain non-finite values")

    base_edge = contract["data_identity"][
        "blunt_trailing_edge_base_segment"
    ]
    if base_edge != [121, 122]:
        raise PressureObservationError("unexpected frozen base-edge identity")
    base_segment = int(base_edge[0])
    if not (
        coordinates[base_segment, 0] == coordinates[base_segment + 1, 0]
        == np.max(coordinates[:, 0])
        and coordinates[base_segment, 1] > 0.0
        and coordinates[base_segment + 1, 1] < 0.0
    ):
        raise PressureObservationError(
            "frozen trailing-edge base segment is not present"
        )

    cases: dict[str, Any] = {}
    cp_headers: list[str] | None = None
    phase_alignment_max = 0.0
    cn_roundtrip_max = 0.0
    base_exclusion_improves_every_case = True
    malformed_trailing_rows = 0
    for configuration in selection["configurations"]:
        for frequency in selection["frequencies_hz"]:
            stem = _case_stem(configuration, frequency)
            mean_name = f"{selection['cp_mean_prefix']}/{stem}"
            std_name = f"{selection['cp_std_prefix']}/{stem}"
            force_name = f"{selection['force_prefix']}/{stem}"
            mean_header, mean = _member_table(
                url, records, mean_name
            )
            std_header, std = _member_table(url, records, std_name)
            force_header, force = _member_table(
                url, records, force_name
            )
            if cp_headers is None:
                cp_headers = mean_header
            if mean_header != cp_headers or std_header != cp_headers:
                raise PressureObservationError(
                    f"{stem}: Cp mean/std headers differ"
                )
            if force_header != [
                "Phase",
                "AoA_unc",
                "Cn_unc",
                "Ct_unc",
                "Cm_LE_unc",
            ]:
                raise PressureObservationError(
                    f"{stem}: unexpected force header"
                )
            if mean.shape[1] != coordinate_count + 2:
                raise PressureObservationError(
                    f"{stem}: Cp/coordinate width mismatch"
                )
            if std.shape != mean.shape:
                raise PressureObservationError(
                    f"{stem}: mean/std shape mismatch"
                )
            if force.shape[0] != mean.shape[0]:
                raise PressureObservationError(
                    f"{stem}: Cp/force row-count mismatch"
                )

            valid = np.all(np.isfinite(mean), axis=1)
            std_valid = np.all(np.isfinite(std), axis=1)
            if not np.array_equal(valid, std_valid):
                raise PressureObservationError(
                    f"{stem}: Cp mean/std validity masks differ"
                )
            invalid_indices = np.flatnonzero(~valid)
            if len(invalid_indices):
                if not (
                    len(invalid_indices) == 1
                    and invalid_indices[0] == len(mean) - 1
                    and np.isfinite(mean[-1, 0])
                    and np.all(np.isnan(mean[-1, 1:]))
                ):
                    raise PressureObservationError(
                        f"{stem}: non-trailing malformed Cp rows"
                    )
                malformed_trailing_rows += 1

            mean = mean[valid]
            std = std[valid]
            force = force[valid]
            if not (
                np.all(np.isfinite(mean))
                and np.all(np.isfinite(std))
                and np.all(np.isfinite(force))
            ):
                raise PressureObservationError(
                    f"{stem}: retained rows are non-finite"
                )
            alignment = float(
                np.max(np.abs(mean[:, :2] - force[:, :2]), initial=0.0)
            )
            phase_alignment_max = max(phase_alignment_max, alignment)

            cp = mean[:, 2:]
            cn_profile, ct_profile, cm_profile = integrate_profile_pressure(
                coordinates,
                cp,
                excluded_segments=(base_segment,),
            )
            _, ct_with_base, _ = integrate_profile_pressure(
                coordinates, cp
            )
            cn_residual = _residual(force[:, 2], cn_profile)
            ct_residual = _residual(force[:, 3], ct_profile)
            ct_with_base_residual = _residual(force[:, 3], ct_with_base)
            cm_residual = _residual(force[:, 4], cm_profile)
            cn_roundtrip_max = max(
                cn_roundtrip_max, cn_residual["max_abs"]
            )
            improves = (
                ct_residual["rms"] < ct_with_base_residual["rms"]
            )
            base_exclusion_improves_every_case &= improves
            cases[stem] = {
                "configuration": configuration,
                "frequency_hz": float(frequency),
                "retained_phase_count": int(len(mean)),
                "discarded_trailing_rows": int(np.sum(~valid)),
                "phase_aoa_alignment_max_abs": alignment,
                "cp_cycle_std": {
                    "min": float(np.min(std[:, 2:])),
                    "max": float(np.max(std[:, 2:])),
                    "mean": float(np.mean(std[:, 2:])),
                },
                "Cn_profile_roundtrip": cn_residual,
                "Ct_profile_roundtrip": ct_residual,
                "Ct_if_blunt_base_included": ct_with_base_residual,
                "Cm_LE_profile_roundtrip": cm_residual,
                "base_exclusion_reduces_Ct_rms": bool(improves),
            }

    tolerances = contract["deterministic_identity_gates"]
    checks = {
        "archive_entry_count": (
            index["entry_count"] == asset["archive_entry_count"]
        ),
        "case_count": (
            len(cases) == selection["expected_case_count"]
        ),
        "coordinate_profile_identity": (
            coordinates.shape == (coordinate_count, 2)
            and base_segment == 121
        ),
        "phase_and_aoa_serialization": (
            phase_alignment_max
            <= float(tolerances["phase_and_aoa_serialization_abs"])
        ),
        "normal_force_roundtrip": (
            cn_roundtrip_max
            <= float(tolerances["normal_force_roundtrip_abs"])
        ),
        "blunt_base_exclusion_identity": (
            base_exclusion_improves_every_case
        ),
    }
    result = {
        "version": 1,
        "as_of": contract["as_of"],
        "claim_node": contract["claim_node"],
        "asset": {
            "persistent_id": asset["persistent_id"],
            "archive_size_bytes": asset["archive_size_bytes"],
            "archive_entry_count": index["entry_count"],
            "range_only": True,
        },
        "geometry": {
            "coordinate_count": int(len(coordinates)),
            "signed_polygon_area": float(
                0.5
                * np.sum(
                    coordinates[:-1, 0] * coordinates[1:, 1]
                    - coordinates[1:, 0] * coordinates[:-1, 1]
                )
            ),
            "blunt_trailing_edge_base_segment": base_edge,
            "base_segment_height_over_chord": float(
                abs(
                    coordinates[base_segment + 1, 1]
                    - coordinates[base_segment, 1]
                )
            ),
        },
        "case_count": len(cases),
        "malformed_trailing_rows_discarded": malformed_trailing_rows,
        "max_phase_aoa_alignment_abs": phase_alignment_max,
        "max_Cn_profile_roundtrip_abs": cn_roundtrip_max,
        "cases": cases,
        "checks": checks,
        "decision": {
            "data_ledger_identity": (
                "PASS" if all(checks.values()) else "FAIL"
            ),
            "pressure_observation_gate":
                "GO-FOR-UNIFIED-PRESSURE-FALSIFICATION",
            "model_comparison_executed": False,
            "Ct_Cm_physical_uncertainty_qualified": False,
            "spatial_state_gate": "NO-GO",
            "physical_promotion": False,
            "production_formula_changed": False,
        },
        "passed": all(checks.values()),
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

