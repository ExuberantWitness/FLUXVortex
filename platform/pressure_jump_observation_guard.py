#!/usr/bin/env python3
"""Audit the gauge-free pressure-jump observation and its nonuniqueness.

The public upper/lower pressure is reduced to the thin-sheet observable
``Cp_lower-Cp_upper``.  A fixed common-mode pressure witness demonstrates that
the same jump cannot determine thick-profile tangential force.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from audit_high_fidelity_field_candidates import _zip_index
from unified_pressure_observation_guard import (
    _case_stem,
    _member_table,
    _residual,
    integrate_profile_pressure,
)


PLATFORM = Path(__file__).resolve().parent
CASES = (
    PLATFORM
    / "docs"
    / "diag"
    / "pressure_jump_observation_cases.yaml"
)
OUTPUT = (
    PLATFORM
    / "docs"
    / "diag"
    / "pressure_jump_observation_results.json"
)


class PressureJumpObservationError(ValueError):
    """The side-pairing or pressure-jump contract is violated."""


def pair_pressure_jump(
    coordinates: Any,
    profile_cp: Any,
    *,
    side_point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return paired x, upper Cp, lower Cp, and lower-minus-upper jump."""
    xy = np.asarray(coordinates, dtype=float)
    pressure = np.asarray(profile_cp, dtype=float)
    if (
        xy.shape != (2 * side_point_count, 2)
        or pressure.ndim != 2
        or pressure.shape[1] != 2 * side_point_count
        or not np.all(np.isfinite(xy))
        or not np.all(np.isfinite(pressure))
    ):
        raise PressureJumpObservationError(
            "coordinates/Cp do not satisfy the paired two-side shape"
        )
    x_upper = xy[:side_point_count, 0]
    x_lower = xy[side_point_count:, 0][::-1]
    if not np.allclose(x_upper, x_lower, rtol=0.0, atol=1.0e-14):
        raise PressureJumpObservationError(
            "upper/lower chordwise coordinates are not paired"
        )
    cp_upper = pressure[:, :side_point_count]
    cp_lower = pressure[:, side_point_count:][:, ::-1]
    return x_upper, cp_upper, cp_lower, cp_lower - cp_upper


def integrate_normal_pressure_jump(x: Any, delta_cp: Any) -> np.ndarray:
    """Integrate lower-minus-upper pressure jump along paired chord points."""
    coordinate = np.asarray(x, dtype=float)
    jump = np.asarray(delta_cp, dtype=float)
    if (
        coordinate.ndim != 1
        or jump.ndim != 2
        or jump.shape[1] != len(coordinate)
        or len(coordinate) < 2
        or not np.all(np.isfinite(coordinate))
        or not np.all(np.isfinite(jump))
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise PressureJumpObservationError(
            "x/delta_cp must be finite with strictly increasing paired x"
        )
    return np.sum(
        0.5 * (jump[:, :-1] + jump[:, 1:])
        * np.diff(coordinate)[None, :],
        axis=1,
    )


def paired_common_mode_profile(
    common_mode: Any,
) -> np.ndarray:
    """Pack one paired LE->TE common mode into archive contour order."""
    common = np.asarray(common_mode, dtype=float)
    if common.ndim != 2 or not np.all(np.isfinite(common)):
        raise PressureJumpObservationError(
            "common_mode must have finite shape (phase,side_point)"
        )
    return np.concatenate((common, common[:, ::-1]), axis=1)


def main() -> int:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    asset = contract["asset"]
    selection = contract["selection"]
    gates = contract["deterministic_gates"]
    url = asset["archive_url"]
    index = _zip_index(url, int(asset["archive_size_bytes"]))
    records = index["records"]
    coordinate_header, coordinates = _member_table(
        url, records, selection["coordinate_member"]
    )
    if coordinate_header != ["x/c", "y/c"]:
        raise PressureJumpObservationError(
            "unexpected coordinate header"
        )
    side_count = int(contract["pairing_identity"]["side_point_count"])
    x_upper = coordinates[:side_count, 0]
    x_lower = coordinates[side_count:, 0][::-1]
    paired_x_max = float(np.max(np.abs(x_upper - x_lower), initial=0.0))

    case_results: dict[str, Any] = {}
    published_cn_roundtrip_max = 0.0
    jump_invariance_max = 0.0
    cn_invariance_max = 0.0
    ct_counterexample_max = 0.0
    cm_counterexample_max = 0.0
    for configuration in selection["configurations"]:
        for frequency in selection["frequencies_hz"]:
            stem = _case_stem(configuration, frequency)
            mean_name = f"{selection['cp_mean_prefix']}/{stem}"
            force_name = f"{selection['force_prefix']}/{stem}"
            _, mean = _member_table(url, records, mean_name)
            _, force = _member_table(url, records, force_name)
            valid = np.all(np.isfinite(mean), axis=1)
            mean = mean[valid]
            force = force[valid]
            if not np.all(np.isfinite(force)):
                raise PressureJumpObservationError(
                    f"{stem}: force contains non-finite values"
                )

            x, cp_upper, cp_lower, jump = pair_pressure_jump(
                coordinates,
                mean[:, 2:],
                side_point_count=side_count,
            )
            cn_jump = integrate_normal_pressure_jump(x, jump)
            cn_residual = _residual(force[:, 2], cn_jump)
            published_cn_roundtrip_max = max(
                published_cn_roundtrip_max,
                cn_residual["max_abs"],
            )

            phase = mean[:, 0]
            common = (
                0.2
                * np.sin(phase)[:, None]
                * (x[None, :] - 0.5)
            )
            shifted_profile = (
                mean[:, 2:] + paired_common_mode_profile(common)
            )
            _, _, _, shifted_jump = pair_pressure_jump(
                coordinates,
                shifted_profile,
                side_point_count=side_count,
            )
            shifted_cn = integrate_normal_pressure_jump(
                x, shifted_jump
            )
            jump_change = float(
                np.max(np.abs(shifted_jump - jump), initial=0.0)
            )
            cn_change = float(
                np.max(np.abs(shifted_cn - cn_jump), initial=0.0)
            )
            jump_invariance_max = max(
                jump_invariance_max, jump_change
            )
            cn_invariance_max = max(cn_invariance_max, cn_change)

            _, ct_original, cm_original = integrate_profile_pressure(
                coordinates,
                mean[:, 2:],
                excluded_segments=(side_count - 1,),
            )
            _, ct_shifted, cm_shifted = integrate_profile_pressure(
                coordinates,
                shifted_profile,
                excluded_segments=(side_count - 1,),
            )
            ct_change = float(
                np.max(np.abs(ct_shifted - ct_original), initial=0.0)
            )
            cm_change = float(
                np.max(np.abs(cm_shifted - cm_original), initial=0.0)
            )
            ct_counterexample_max = max(
                ct_counterexample_max, ct_change
            )
            cm_counterexample_max = max(
                cm_counterexample_max, cm_change
            )
            case_results[stem] = {
                "configuration": configuration,
                "frequency_hz": float(frequency),
                "phase_count": int(len(mean)),
                "published_Cn_from_delta_Cp": cn_residual,
                "common_mode_jump_change_max_abs": jump_change,
                "common_mode_Cn_change_max_abs": cn_change,
                "common_mode_Ct_change_max_abs": ct_change,
                "common_mode_Cm_LE_change_max_abs": cm_change,
            }

    checks = {
        "archive_entry_count": (
            index["entry_count"] == asset["archive_entry_count"]
        ),
        "case_count": (
            len(case_results) == selection["expected_case_count"]
        ),
        "paired_x_identity": (
            paired_x_max <= float(gates["paired_x_max_abs"])
        ),
        "published_Cn_from_delta_Cp": (
            published_cn_roundtrip_max
            <= float(gates["published_Cn_roundtrip_abs"])
        ),
        "jump_common_mode_invariance": (
            jump_invariance_max
            <= float(gates["jump_common_mode_invariance_abs"])
        ),
        "Cn_common_mode_invariance": (
            cn_invariance_max
            <= float(gates["Cn_common_mode_invariance_abs"])
        ),
        "Ct_common_mode_nonuniqueness_witness": (
            ct_counterexample_max
            >= float(gates["Ct_counterexample_min_change"])
        ),
    }
    result = {
        "version": 1,
        "as_of": contract["as_of"],
        "claim_nodes": contract["claim_nodes"],
        "asset": {
            "persistent_id": asset["persistent_id"],
            "range_only": True,
        },
        "case_count": len(case_results),
        "paired_x_max_abs": paired_x_max,
        "max_published_Cn_roundtrip_abs": published_cn_roundtrip_max,
        "common_mode_witness": {
            "definition": contract["common_mode_counterexample"][
                "definition"
            ],
            "max_jump_change_abs": jump_invariance_max,
            "max_Cn_change_abs": cn_invariance_max,
            "max_Ct_change_abs": ct_counterexample_max,
            "max_Cm_LE_change_abs": cm_counterexample_max,
        },
        "cases": case_results,
        "checks": checks,
        "decision": {
            "pressure_jump_observation_gate": "GO",
            "absolute_side_pressure_from_jump": "NO-GO",
            "thick_profile_tangential_force_from_jump": "NO-GO",
            "requires_mean_potential_thickness_component": True,
            "model_comparison_executed": False,
            "spatial_state_physical_promotion": False,
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

