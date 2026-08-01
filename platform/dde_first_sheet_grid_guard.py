"""Run the preregistered P-R first-sheet span-grid convergence gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.continuous_shedding import (
    newborn_halfwing_shedding_band,
    reconstruct_halfwing_p2_trace,
)
from claim_runtime.coupled_lesp_dde import (
    _mirror_halfwing_surface,
    build_dde_unit_strip_normal_influence,
    solve_coupled_lesp_dde_stage,
)
from claim_runtime.hirato_equations import (
    embed_chord_normal_displacement,
    first_lev_displacement_ramesh_2d,
    lesp_eq6,
)
from claim_runtime.hirato_live_shadow import (
    build_bound_aic,
    build_bound_lattice,
)


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "docs" / "diag" / "dde_first_sheet_grid_cases.yaml"
RESULT_PATH = HERE / "docs" / "diag" / "dde_first_sheet_grid_results.json"


def geometry(spec: dict, *, nc: int, ns: int):
    chord = float(spec["chord"])
    half_span = float(spec["half_span"])
    alpha = np.deg2rad(float(spec["pitch_deg"]))
    tangent = np.array([np.cos(alpha), 0.0, -np.sin(alpha)])
    normal = np.array([np.sin(alpha), 0.0, np.cos(alpha)])
    span = np.linspace(0.0, half_span, ns + 1)
    chord_nodes = np.linspace(0.0, chord, nc + 1)
    corners = np.array(
        [
            x * tangent + np.array([0.0, y, 0.0])
            for x in chord_nodes
            for y in span
        ]
    ).reshape(nc + 1, ns + 1, 3)
    lattice = build_bound_lattice(
        corners,
        np.zeros_like(corners),
        nc=nc,
        ns=ns,
    )
    return lattice, span, corners[0], tangent, normal


def solve_grid(spec: dict, ns: int) -> dict:
    nc = int(spec["grid"]["nc"])
    lattice, span_edges, leading_edge, tangent, normal = geometry(
        spec["geometry"],
        nc=nc,
        ns=ns,
    )
    eta_mid = (
        0.5 * (span_edges[:-1] + span_edges[1:])
        / span_edges[-1]
    )
    a0_pre = float(spec["preconstraint_a0_root"]) * (
        1.0
        - (
            1.0 - float(spec["preconstraint_tip_fraction"])
        )
        * eta_mid**2
    )
    # A0 is an onset/placement observer, not the free-sheet potential jump.
    # The registered analytic profile can therefore be evaluated at vertices
    # directly; applying the DDE zero-tip strength condition to A0 would
    # collapse the tip geometry and conflate two different physical fields.
    eta_vertex = span_edges / span_edges[-1]
    a0_vertex = float(spec["preconstraint_a0_root"]) * (
        1.0
        - (
            1.0 - float(spec["preconstraint_tip_fraction"])
        )
        * eta_vertex**2
    )
    alpha = np.full(ns + 1, np.deg2rad(spec["geometry"]["pitch_deg"]))
    displacement_2d = first_lev_displacement_ramesh_2d(
        float(spec["u_infinity"]),
        a0_vertex,
        alpha,
        float(spec["time_step"]),
    )
    current_edge = leading_edge + embed_chord_normal_displacement(
        displacement_2d,
        np.broadcast_to(tangent, (ns + 1, 3)),
        np.broadcast_to(normal, (ns + 1, 3)),
    )
    suction_offset = (current_edge - leading_edge) @ normal
    aic = build_bound_aic(lattice, mirror_symmetry=True)
    unit = build_dde_unit_strip_normal_influence(
        collocation=lattice.collocation,
        normals=lattice.normals,
        previous_edge=leading_edge,
        current_edge=current_edge,
        span_edges=span_edges,
        time_nodes=[0.0, 0.5, 1.0],
        quadrature_order=int(spec["line_quadrature_order"]),
        mirror_symmetry=True,
    )
    scale = lesp_eq6(
        np.ones(ns),
        float(spec["u_infinity"]),
        lattice.chord,
        lattice.delta_x_front,
    )
    target_bound = np.empty(nc * ns)
    front = a0_pre / scale
    for chord_index in range(nc):
        target_bound[
            chord_index * ns : (chord_index + 1) * ns
        ] = front * (0.7**chord_index)
    stage = solve_coupled_lesp_dde_stage(
        aic=aic,
        rhs_without_nascent_lev=aic @ target_bound,
        unit_lev_normal_influence=unit.normal_influence,
        u_infinity=float(spec["u_infinity"]),
        chord=lattice.chord,
        delta_x_front=lattice.delta_x_front,
        lesp_crit=float(spec["lesp_crit"]),
    )
    trace = reconstruct_halfwing_p2_trace(
        stage.gamma_lev,
        span_edges,
    )
    trace_coordinates = np.linspace(
        span_edges[0],
        span_edges[-1],
        int(spec["trace_samples"]),
    )
    trace_values = trace.evaluate(trace_coordinates)
    birth = newborn_halfwing_shedding_band(
        sheet_id=f"pr-first-ns{ns}",
        vortex_family="LEV_SUCTION",
        previous_edge=leading_edge,
        current_edge=current_edge,
        span_edges=span_edges,
        time_nodes=[0.0, 0.5, 1.0],
        strip_strength_rows=np.tile(stage.gamma_lev, (3, 1)),
    )
    probe_spec = spec["field_probes"]
    probes = np.array(
        [
            float(probe_spec["chord_fraction"]) * tangent
            + np.array([0.0, fraction * span_edges[-1], 0.0])
            + float(probe_spec["normal_offset"]) * normal
            for fraction in probe_spec["span_fractions"]
        ]
    )
    field = birth.band.surface.induced_velocity_line_reduced(
        probes,
        quadrature_order=int(spec["line_quadrature_order"]),
    )
    field += _mirror_halfwing_surface(
        birth.band.surface
    ).induced_velocity_line_reduced(
        probes,
        quadrature_order=int(spec["line_quadrature_order"]),
    )
    return {
        "ns": ns,
        "trace": trace_values,
        "field": field,
        "gamma_lev": stage.gamma_lev,
        "trace_total_variation": float(
            np.sum(np.abs(np.diff(trace_values)))
        ),
        "trace_peak_abs": float(
            np.max(np.abs(trace_values), initial=0.0)
        ),
        "minimum_suction_side_offset": float(np.min(suction_offset)),
        "lesp_residual": stage.lesp_active_max_abs_residual,
        "lesp_condition": stage.lesp_condition_number,
    }


def relative_change(first, second) -> float:
    difference = np.max(np.linalg.norm(second - first, axis=-1))
    scale = max(
        float(np.max(np.linalg.norm(second, axis=-1))),
        np.finfo(float).eps,
    )
    return float(difference / scale)


def run(spec: dict) -> dict:
    states = [
        solve_grid(spec, int(ns))
        for ns in spec["grid"]["ns"]
    ]
    trace_change = [
        relative_change(
            states[index]["trace"][:, None],
            states[index + 1]["trace"][:, None],
        )
        for index in range(2)
    ]
    field_change = [
        relative_change(
            states[index]["field"],
            states[index + 1]["field"],
        )
        for index in range(2)
    ]
    guards = {
        name: float(value)
        for name, value in spec["guards"].items()
    }
    trace_ratio = trace_change[0] / max(
        trace_change[1], np.finfo(float).eps
    )
    field_ratio = field_change[0] / max(
        field_change[1], np.finfo(float).eps
    )
    checks = {
        "lesp_constraint": max(
            state["lesp_residual"] for state in states
        )
        <= guards["lesp_active_residual_max"],
        "lesp_conditioning": max(
            state["lesp_condition"] for state in states
        )
        <= guards["lesp_condition_number_max"],
        "suction_side_geometry": min(
            state["minimum_suction_side_offset"] for state in states
        )
        > guards["minimum_suction_side_offset"],
        "trace_cauchy": (
            trace_ratio >= guards["trace_cauchy_ratio_min"]
            and trace_change[1]
            <= guards["trace_finest_relative_change_max"]
        ),
        "field_cauchy": (
            field_ratio >= guards["field_cauchy_ratio_min"]
            and field_change[1]
            <= guards["field_finest_relative_change_max"]
        ),
    }
    return {
        "spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": spec["role"],
        "grids": [
            {
                key: value
                for key, value in state.items()
                if key not in ("trace", "field", "gamma_lev")
            }
            | {"gamma_lev": state["gamma_lev"].tolist()}
            for state in states
        ],
        "trace_relative_changes": trace_change,
        "trace_cauchy_ratio": trace_ratio,
        "field_relative_changes": field_change,
        "field_cauchy_ratio": field_ratio,
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
