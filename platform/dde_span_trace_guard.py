"""Run the preregistered half-wing C1 P2 span-trace gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.continuous_shedding import (
    newborn_halfwing_shedding_band,
    reconstruct_halfwing_p2_trace,
)


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "docs" / "diag" / "dde_span_trace_cases.yaml"
RESULT_PATH = HERE / "docs" / "diag" / "dde_span_trace_results.json"


def evaluate(edges: np.ndarray) -> dict:
    midpoint = 0.5 * (edges[:-1] + edges[1:])
    reconstruction = reconstruct_halfwing_p2_trace(
        1.0 - midpoint**2,
        edges,
    )
    expected = 1.0 - reconstruction.span_p2_coordinates**2
    return {
        "strip_count": len(midpoint),
        "report": asdict(reconstruction.report),
        "manufactured_node_error": float(
            np.max(np.abs(reconstruction.p2_values - expected))
        ),
    }


def run(spec: dict) -> dict:
    cases = []
    for count in spec["grids"]["uniform_strip_counts"]:
        cases.append(
            {
                "grid": "uniform",
                **evaluate(np.linspace(0.0, 1.0, int(count) + 1)),
            }
        )
    exponent = float(spec["grids"]["stretched_exponent"])
    for count in spec["grids"]["stretched_strip_counts"]:
        cases.append(
            {
                "grid": "stretched",
                **evaluate(
                    np.linspace(0.0, 1.0, int(count) + 1) ** exponent
                ),
            }
        )

    # Exercise the exact same three-row adapter that creates a material band.
    edges = np.linspace(0.0, 1.0, 9)
    midpoint = 0.5 * (edges[:-1] + edges[1:])
    geometry = np.column_stack(
        (np.zeros(len(edges)), edges, np.zeros(len(edges)))
    )
    band = newborn_halfwing_shedding_band(
        sheet_id="guard-lev-row",
        vortex_family="LEV_SUCTION",
        previous_edge=geometry,
        current_edge=geometry + [0.05, 0.0, 0.01],
        span_edges=edges,
        time_nodes=[0.0, 0.5, 1.0],
        strip_strength_rows=np.array(
            [
                (1.0 + 0.2 * time) * (1.0 - midpoint**2)
                for time in (0.0, 0.5, 1.0)
            ]
        ),
    )
    guards = spec["guards"]
    limit = {name: float(value) for name, value in guards.items()}
    reports = [case["report"] for case in cases]
    checks = {
        "full_rank": all(
            report["degrees_of_freedom"] - report["rank"]
            <= limit["matrix_rank_defect_max"]
            for report in reports
        ),
        "conditioning": max(
            report["condition_number"] for report in reports
        )
        <= limit["condition_number_max"],
        "midpoint_identity": max(
            report["max_midpoint_residual"] for report in reports
        )
        <= limit["midpoint_residual_max"],
        "root_symmetry": max(
            report["root_derivative_residual"] for report in reports
        )
        <= limit["root_derivative_residual_max"],
        "tip_zero": max(
            report["tip_value_residual"] for report in reports
        )
        <= limit["tip_value_residual_max"],
        "internal_gradient_continuity": max(
            report["max_internal_derivative_jump"]
            for report in reports
        )
        <= limit["internal_derivative_jump_max"],
        "manufactured_quadratic": max(
            case["manufactured_node_error"] for case in cases
        )
        <= limit["manufactured_node_error_max"],
        "newborn_band_continuity": (
            band.band.surface.continuity_report().compatible
            and all(report.passed for report in band.trace_reports)
        ),
    }
    return {
        "spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": spec["role"],
        "cases": cases,
        "newborn_band_continuity": (
            asdict(band.band.surface.continuity_report())
        ),
        "checks": checks,
        "all_pass": all(checks.values()),
        "promotion": spec["promotion_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    payload = run(spec)
    if args.write:
        RESULT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
