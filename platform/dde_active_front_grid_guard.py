"""Run the preregistered active-front terminated P-R span-grid gate."""
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
from claim_runtime.hirato_live_shadow import build_bound_aic
from dde_first_sheet_grid_guard import geometry, relative_change


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "docs" / "diag" / "dde_active_front_grid_cases.yaml"
RESULT_PATH = HERE / "docs" / "diag" / "dde_active_front_grid_results.json"


def analytic_a0(eta, *, root, critical, front):
    value = np.asarray(eta, dtype=float)
    return critical + (root - critical) * (
        1.0 - (value / front) ** 2
    )


def solve_grid(spec: dict, ns: int) -> dict:
    nc = int(spec["grid"]["nc"])
    lattice, span_edges, leading_edge, tangent, normal = geometry(
        spec["geometry"],
        nc=nc,
        ns=ns,
    )
    half_span = span_edges[-1]
    front_fraction = float(spec["active_front_span_fraction"])
    eta_mid = (
        0.5 * (span_edges[:-1] + span_edges[1:]) / half_span
    )
    a0_pre = analytic_a0(
        eta_mid,
        root=float(spec["preconstraint_a0_root"]),
        critical=float(spec["lesp_crit"]),
        front=front_fraction,
    )
    active = np.abs(a0_pre) > float(spec["lesp_crit"])
    active_indices = np.flatnonzero(active)
    if (
        active_indices.size == 0
        or not np.array_equal(
            active_indices,
            np.arange(active_indices.size),
        )
    ):
        raise RuntimeError(
            "registered A0 profile did not create one root-connected component"
        )
    active_count = int(active_indices.size)
    active_span = span_edges[: active_count + 1]
    front_error = abs(
        float(active_span[-1] / half_span) - front_fraction
    )
    eta_vertex = active_span / half_span
    a0_vertex = analytic_a0(
        eta_vertex,
        root=float(spec["preconstraint_a0_root"]),
        critical=float(spec["lesp_crit"]),
        front=front_fraction,
    )
    alpha = np.full(
        active_count + 1,
        np.deg2rad(spec["geometry"]["pitch_deg"]),
    )
    displacement = embed_chord_normal_displacement(
        first_lev_displacement_ramesh_2d(
            float(spec["u_infinity"]),
            a0_vertex,
            alpha,
            float(spec["time_step"]),
        ),
        np.broadcast_to(tangent, (active_count + 1, 3)),
        np.broadcast_to(normal, (active_count + 1, 3)),
    )
    current_edge = leading_edge[: active_count + 1] + displacement
    unit_active = build_dde_unit_strip_normal_influence(
        collocation=lattice.collocation,
        normals=lattice.normals,
        previous_edge=leading_edge[: active_count + 1],
        current_edge=current_edge,
        span_edges=active_span,
        time_nodes=[0.0, 0.5, 1.0],
        quadrature_order=int(spec["line_quadrature_order"]),
        mirror_symmetry=True,
    )
    influence = np.zeros((nc * ns, ns))
    influence[:, :active_count] = unit_active.normal_influence
    aic = build_bound_aic(lattice, mirror_symmetry=True)
    scale = lesp_eq6(
        np.ones(ns),
        float(spec["u_infinity"]),
        lattice.chord,
        lattice.delta_x_front,
    )
    target_bound = np.empty(nc * ns)
    front_gamma = a0_pre / scale
    for chord_index in range(nc):
        target_bound[
            chord_index * ns : (chord_index + 1) * ns
        ] = front_gamma * (0.7**chord_index)
    stage = solve_coupled_lesp_dde_stage(
        aic=aic,
        rhs_without_nascent_lev=aic @ target_bound,
        unit_lev_normal_influence=influence,
        u_infinity=float(spec["u_infinity"]),
        chord=lattice.chord,
        delta_x_front=lattice.delta_x_front,
        lesp_crit=float(spec["lesp_crit"]),
    )
    active_gamma = stage.gamma_lev[:active_count]
    trace = reconstruct_halfwing_p2_trace(
        active_gamma,
        active_span,
    )
    common_span = np.linspace(
        0.0,
        front_fraction * half_span,
        int(spec["trace_samples"]),
    )
    trace_values = trace.evaluate(common_span)
    birth = newborn_halfwing_shedding_band(
        sheet_id=f"active-front-ns{ns}",
        vortex_family="LEV_SUCTION",
        previous_edge=leading_edge[: active_count + 1],
        current_edge=current_edge,
        span_edges=active_span,
        time_nodes=[0.0, 0.5, 1.0],
        strip_strength_rows=np.tile(active_gamma, (3, 1)),
    )
    probe = spec["field_probes"]
    probes = np.array(
        [
            float(probe["chord_fraction"]) * tangent
            + np.array([0.0, fraction * half_span, 0.0])
            + float(probe["normal_offset"]) * normal
            for fraction in probe["span_fractions"]
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
        "active_count": active_count,
        "active_front_location_error": front_error,
        "inactive_gamma_max": float(
            np.max(
                np.abs(stage.gamma_lev[active_count:]),
                initial=0.0,
            )
        ),
        "lesp_residual": stage.lesp_active_max_abs_residual,
        "lesp_condition": stage.lesp_condition_number,
        "trace": trace_values,
        "field": field,
        "trace_total_variation": float(
            np.sum(np.abs(np.diff(trace_values)))
        ),
        "trace_peak_abs": float(
            np.max(np.abs(trace_values), initial=0.0)
        ),
    }


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
    trace_ratio = trace_change[0] / max(
        trace_change[1], np.finfo(float).eps
    )
    field_ratio = field_change[0] / max(
        field_change[1], np.finfo(float).eps
    )
    guards = {
        name: float(value)
        for name, value in spec["guards"].items()
    }
    checks = {
        "active_front_identity": max(
            state["active_front_location_error"] for state in states
        )
        <= guards["active_front_location_error_max"],
        "inactive_strength_zero": max(
            state["inactive_gamma_max"] for state in states
        )
        <= guards["inactive_gamma_max"],
        "lesp_constraint": max(
            state["lesp_residual"] for state in states
        )
        <= guards["lesp_active_residual_max"],
        "lesp_conditioning": max(
            state["lesp_condition"] for state in states
        )
        <= guards["lesp_condition_number_max"],
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
                if key not in ("trace", "field")
            }
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
