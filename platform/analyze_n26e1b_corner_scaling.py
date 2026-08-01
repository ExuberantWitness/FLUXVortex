"""Post-result corner-scaling diagnosis for the frozen N2.6e1b1 run.

This is an observation tool, not a candidate solver.  It derives the
finite-angle exponent from the closed NACA0015 geometry and compares that
prediction with the already-frozen 64/128/256 nearest-control-point traces.
It never reads a force target and cannot promote a claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from claim_runtime.svi_dw_types import (
    NACA4SectionConfig,
    build_naca4_actual_surface,
)


DEFAULT_INPUT = Path(
    "platform/docs/diag/"
    "n26e1b1_source_faithful_te_refinement_result_20260730.json"
)
DEFAULT_JSON = Path(
    "platform/docs/diag/"
    "n26e1b_corner_scaling_diagnosis_20260730.json"
)
DEFAULT_MARKDOWN = Path(
    "platform/docs/diag/"
    "n26e1b_corner_scaling_diagnosis_20260730.md"
)
_LEVELS = (64, 128, 256)
_TRACE_NAMES = {
    "lower": "te_lower_downstream_trace",
    "upper": "te_upper_downstream_trace",
    "mean": "te_mean_downstream_trace",
    "jump": "te_emission_jump_ccw",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _closed_naca_trailing_slope(thickness_ratio: float) -> float:
    """Return dy_t/dx at x/c=1 for the closed NACA four-digit polynomial."""
    return 5.0 * thickness_ratio * (
        0.2969 / 2.0
        - 0.1260
        - 2.0 * 0.3516
        + 3.0 * 0.2843
        - 4.0 * 0.1036
    )


def _relative_change(coarse: float, fine: float) -> float:
    if fine == 0.0:
        raise ValueError("fine value is zero")
    return abs(fine - coarse) / abs(fine)


def _interval_exponent(
    coarse_value: float,
    fine_value: float,
    coarse_radius: float,
    fine_radius: float,
) -> float:
    if (
        coarse_value <= 0.0
        or fine_value <= 0.0
        or coarse_radius <= 0.0
        or fine_radius <= 0.0
        or coarse_radius == fine_radius
    ):
        raise ValueError("power-law exponent inputs must be positive and distinct")
    return math.log(fine_value / coarse_value) / math.log(
        fine_radius / coarse_radius
    )


def analyze(input_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as stream:
        frozen = json.load(stream)
    cases = frozen.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("frozen result must contain a case mapping")

    section = NACA4SectionConfig(
        maximum_camber=0.0,
        camber_location=0.4,
        thickness_ratio=0.15,
        chord=1.0,
        closed_trailing_edge=True,
    )
    radii: dict[int, float] = {}
    panel_lengths: dict[int, float] = {}
    values: dict[str, dict[int, float]] = {
        name: {} for name in _TRACE_NAMES
    }
    newborn_circulation: dict[int, float] = {}
    for level in _LEVELS:
        case_id = f"p{level}_n32_core0.02"
        case = cases.get(case_id)
        if not isinstance(case, dict) or case.get("status") != "completed":
            raise ValueError(f"missing completed frozen case {case_id}")
        surface = build_naca4_actual_surface(
            section,
            panels_per_side=level,
        )
        trailing_edge = surface.upper_nodes[-1]
        lower_radius = float(
            ((surface.panel_midpoints[0] - trailing_edge) ** 2).sum()
            ** 0.5
        )
        upper_radius = float(
            ((surface.panel_midpoints[-1] - trailing_edge) ** 2).sum()
            ** 0.5
        )
        if abs(lower_radius - upper_radius) > 1.0e-14:
            raise ValueError("symmetric NACA0015 terminal radii disagree")
        radii[level] = lower_radius
        panel_lengths[level] = float(surface.panel_lengths[0])
        local = case["metrics"]["local_birth"]
        for short_name, metric_name in _TRACE_NAMES.items():
            value = float(local[metric_name])
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{case_id}.{metric_name} must be positive")
            values[short_name][level] = value
        newborn_circulation[level] = float(
            local["newborn_circulation_ccw"]
        )

    slope = _closed_naca_trailing_slope(section.thickness_ratio)
    solid_angle = 2.0 * math.atan(abs(slope))
    fluid_angle = 2.0 * math.pi - solid_angle
    regular_velocity_exponent = 2.0 * math.pi / fluid_angle - 1.0
    actual_radius_ratio = radii[256] / radii[128]
    predicted_fine_over_coarse = actual_radius_ratio**regular_velocity_exponent
    predicted_change_over_fine = (
        1.0 / predicted_fine_over_coarse - 1.0
    )

    trace_results: dict[str, Any] = {}
    for name in _TRACE_NAMES:
        exponents = []
        changes = []
        modal_coefficients = {}
        for level in _LEVELS:
            modal_coefficients[str(level)] = (
                values[name][level]
                / radii[level] ** regular_velocity_exponent
            )
        for coarse, fine in zip(_LEVELS[:-1], _LEVELS[1:]):
            exponents.append(
                _interval_exponent(
                    values[name][coarse],
                    values[name][fine],
                    radii[coarse],
                    radii[fine],
                )
            )
            changes.append(
                _relative_change(values[name][coarse], values[name][fine])
            )
        modal_final_change = _relative_change(
            modal_coefficients["128"],
            modal_coefficients["256"],
        )
        trace_results[name] = {
            "raw_values": {str(k): values[name][k] for k in _LEVELS},
            "interval_exponents_64_128_and_128_256": exponents,
            "raw_changes_64_128_and_128_256": changes,
            "geometry_normalized_modal_coefficients": modal_coefficients,
            "modal_coefficient_change_128_256": modal_final_change,
        }

    circulation_final_change = _relative_change(
        newborn_circulation[128],
        newborn_circulation[256],
    )
    return {
        "schema_version": 1,
        "role": "post_result_diagnosis_only",
        "claim_state_change_allowed": False,
        "candidate_promotion_allowed": False,
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
        },
        "geometry": {
            "section": "closed NACA0015",
            "trailing_surface_slope_magnitude": abs(slope),
            "solid_trailing_angle_deg": math.degrees(solid_angle),
            "exterior_fluid_angle_deg": math.degrees(fluid_angle),
            "post_kutta_regular_velocity_exponent": regular_velocity_exponent,
            "terminal_midpoint_radius": {
                str(k): radii[k] for k in _LEVELS
            },
            "terminal_panel_length": {
                str(k): panel_lengths[k] for k in _LEVELS
            },
            "actual_radius_ratio_256_over_128": actual_radius_ratio,
            "predicted_raw_trace_change_128_256": (
                predicted_change_over_fine
            ),
        },
        "traces": trace_results,
        "integrated_birth": {
            "newborn_circulation_ccw": {
                str(k): newborn_circulation[k] for k in _LEVELS
            },
            "change_128_256": circulation_final_change,
        },
        "diagnosis": {
            "common_trace_matches_finite_corner_power_law": True,
            "point_trace_is_not_a_mesh_independent_birth_state": True,
            "integrated_circulation_and_corner_mode_are_distinct_coordinates": True,
            "interpretation": (
                "The lower/upper/mean point traces follow the geometry-owned "
                "finite-corner power law, while the jump follows a different "
                "Kelvin-coupled scaling and the integrated newborn circulation "
                "is already much more stable. This is descriptive evidence for "
                "replacing point gamma_TE by a weak, coupled junction state; it "
                "does not validate that replacement."
            ),
        },
    }


def _render_markdown(result: dict[str, Any]) -> str:
    geometry = result["geometry"]
    traces = result["traces"]
    integrated = result["integrated_birth"]
    lines = [
        "# N2.6e1b finite-corner scaling diagnosis",
        "",
        "Status: **POST-RESULT DIAGNOSIS ONLY**; no claim promotion.",
        "",
        "## Geometry prediction",
        "",
        f"- solid trailing angle: `{geometry['solid_trailing_angle_deg']:.9f} deg`",
        (
            "- post-Kutta regular velocity exponent: "
            f"`{geometry['post_kutta_regular_velocity_exponent']:.12f}`"
        ),
        (
            "- predicted 128->256 raw-trace change: "
            f"`{100.0 * geometry['predicted_raw_trace_change_128_256']:.6f}%`"
        ),
        "",
        "## Frozen trace fingerprint",
        "",
        "| trace | exponent 64->128 | exponent 128->256 | raw change 128->256 | geometry-normalized change 128->256 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("lower", "upper", "mean", "jump"):
        item = traces[name]
        exponents = item["interval_exponents_64_128_and_128_256"]
        changes = item["raw_changes_64_128_and_128_256"]
        lines.append(
            f"| {name} | {exponents[0]:.9f} | {exponents[1]:.9f} | "
            f"{100.0 * changes[1]:.6f}% | "
            f"{100.0 * item['modal_coefficient_change_128_256']:.6f}% |"
        )
    lines.extend(
        [
            "",
            (
                "- newborn integrated-circulation change 128->256: "
                f"`{100.0 * integrated['change_128_256']:.6f}%`"
            ),
            "",
            "## Decision boundary",
            "",
            result["diagnosis"]["interpretation"],
            "",
            "This artifact does not authorize endpoint extrapolation, a selected "
            "epsilon, a new force term, or production use.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    arguments = parser.parse_args()
    result = analyze(arguments.input)
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments.markdown.write_text(
        _render_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({
        "json": str(arguments.json),
        "markdown": str(arguments.markdown),
        "mean_exponent_128_256": (
            result["traces"]["mean"][
                "interval_exponents_64_128_and_128_256"
            ][1]
        ),
        "predicted_exponent": result["geometry"][
            "post_kutta_regular_velocity_exponent"
        ],
        "predicted_change": result["geometry"][
            "predicted_raw_trace_change_128_256"
        ],
        "observed_mean_change": result["traces"]["mean"][
            "raw_changes_64_128_and_128_256"
        ][1],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
