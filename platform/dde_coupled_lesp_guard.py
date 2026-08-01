"""Run the preregistered N1-AIC / DDE source coupled LESP gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.continuous_shedding import (
    newborn_halfwing_shedding_band,
)
from claim_runtime.coupled_lesp_dde import (
    build_dde_unit_strip_normal_influence,
    solve_coupled_lesp_dde_stage,
)
from claim_runtime.hirato_equations import lesp_eq6
from claim_runtime.hirato_live_shadow import (
    build_bound_aic,
    build_bound_lattice,
)


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "docs" / "diag" / "dde_coupled_lesp_cases.yaml"
RESULT_PATH = HERE / "docs" / "diag" / "dde_coupled_lesp_results.json"


def stage_payload(stage) -> dict:
    payload = asdict(stage)
    for name in (
        "bound_pre",
        "bound_post",
        "gamma_lev",
        "a0_pre",
        "a0_post",
        "active",
    ):
        payload[name] = payload[name].tolist()
    return payload


def canonical_geometry(spec: dict, *, nc: int, ns: int):
    chord = float(spec["chord"])
    half_span = float(spec["half_span"])
    alpha = np.deg2rad(float(spec["pitch_deg"]))
    chord_coordinate = np.linspace(0.0, chord, nc + 1)
    span_coordinate = np.linspace(0.0, half_span, ns + 1)
    tangent = np.array([np.cos(alpha), 0.0, -np.sin(alpha)])
    normal = np.array([np.sin(alpha), 0.0, np.cos(alpha)])
    corners = np.array(
        [
            chord_value * tangent + np.array([0.0, span_value, 0.0])
            for chord_value in chord_coordinate
            for span_value in span_coordinate
        ]
    ).reshape(nc + 1, ns + 1, 3)
    velocity = np.zeros_like(corners)
    lattice = build_bound_lattice(
        corners,
        velocity,
        nc=nc,
        ns=ns,
    )
    previous_edge = corners[0].copy()
    current_edge = (
        previous_edge
        + float(spec["nascent_chord_displacement"]) * tangent
        + float(spec["nascent_normal_displacement"]) * normal
    )
    return lattice, span_coordinate, previous_edge, current_edge


def run_case(spec: dict, grid: dict) -> dict:
    nc = int(grid["nc"])
    ns = int(grid["ns"])
    geometry = spec["geometry"]
    lattice, span_edges, previous_edge, current_edge = (
        canonical_geometry(geometry, nc=nc, ns=ns)
    )
    aic = build_bound_aic(lattice, mirror_symmetry=True)
    unit = build_dde_unit_strip_normal_influence(
        collocation=lattice.collocation,
        normals=lattice.normals,
        previous_edge=previous_edge,
        current_edge=current_edge,
        span_edges=span_edges,
        time_nodes=spec["time_nodes"],
        quadrature_order=int(spec["line_quadrature_order"]),
        mirror_symmetry=True,
    )
    eta = (
        0.5 * (span_edges[:-1] + span_edges[1:]) / span_edges[-1]
    )
    scale = lesp_eq6(
        np.ones(ns),
        float(spec["u_infinity"]),
        lattice.chord,
        lattice.delta_x_front,
    )
    stages = []
    gamma_rows = []
    for root_a0 in spec["preconstraint_a0_root"]:
        target_a0 = float(root_a0) * (
            1.0
            - (
                1.0 - float(spec["preconstraint_tip_fraction"])
            )
            * eta**2
        )
        target_bound = np.empty(nc * ns)
        front = target_a0 / scale
        for chord_index in range(nc):
            target_bound[
                chord_index * ns : (chord_index + 1) * ns
            ] = front * (0.7**chord_index)
        result = solve_coupled_lesp_dde_stage(
            aic=aic,
            rhs_without_nascent_lev=aic @ target_bound,
            unit_lev_normal_influence=unit.normal_influence,
            u_infinity=float(spec["u_infinity"]),
            chord=lattice.chord,
            delta_x_front=lattice.delta_x_front,
            lesp_crit=float(spec["lesp_crit"]),
        )
        stages.append(result)
        gamma_rows.append(result.gamma_lev)
    birth = newborn_halfwing_shedding_band(
        sheet_id=f"coupled-lev-nc{nc}-ns{ns}",
        vortex_family="LEV_SUCTION",
        previous_edge=previous_edge,
        current_edge=current_edge,
        span_edges=span_edges,
        time_nodes=spec["time_nodes"],
        strip_strength_rows=np.asarray(gamma_rows),
    )
    midpoint_difference = float(
        np.max(
            np.abs(
                gamma_rows[1]
                - 0.5 * (gamma_rows[0] + gamma_rows[2])
            )
        )
    )
    return {
        "grid": {"nc": nc, "ns": ns},
        "unit_influence_finite": unit.finite,
        "stages": [stage_payload(stage) for stage in stages],
        "midpoint_difference_from_endpoint_average": midpoint_difference,
        "newborn_continuity": asdict(
            birth.band.surface.continuity_report()
        ),
        "trace_reports": [
            asdict(report) for report in birth.trace_reports
        ],
    }


def run(spec: dict) -> dict:
    cases = [run_case(spec, grid) for grid in spec["grids"]]
    guards = {
        name: float(value)
        for name, value in spec["guards"].items()
    }
    stages = [
        stage for case in cases for stage in case["stages"]
    ]
    checks = {
        "unit_fields_finite": all(
            case["unit_influence_finite"] for case in cases
        ),
        "all_strips_active": all(
            all(stage["active"]) for stage in stages
        ),
        "bound_equation": max(
            stage["bound_equation_max_abs_residual"]
            for stage in stages
        )
        <= guards["bound_equation_residual_max"],
        "lesp_constraint": max(
            stage["lesp_active_max_abs_residual"]
            for stage in stages
        )
        <= guards["lesp_active_residual_max"],
        "aic_conditioning": max(
            stage["aic_condition_number"] for stage in stages
        )
        <= guards["aic_condition_number_max"],
        "lesp_conditioning": max(
            stage["lesp_condition_number"] for stage in stages
        )
        <= guards["lesp_condition_number_max"],
        "explicit_midpoint_not_endpoint_average": min(
            case["midpoint_difference_from_endpoint_average"]
            for case in cases
        )
        >= guards["midpoint_noninterpolation_min"],
        "newborn_trace_continuity": all(
            case["newborn_continuity"]["compatible"]
            and case["newborn_continuity"]["max_trace_jump"]
            <= guards["newborn_trace_jump_max"]
            for case in cases
        ),
    }
    return {
        "spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": spec["role"],
        "cases": cases,
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
