"""Preregistered N3.1j4c3 vertex-star normal-limit dynamic gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.material_sheet_advection import (
    advance_surface_normal_geometry_heun,
    self_induced_vertex_star_normal_velocity,
)
from claim_runtime.sheet_velocity_projection import (
    project_vertex_star_normal_geometry_velocity,
)
from dde_canonical_field_guard import structured_sheet
from dde_cylindrical_advection_guard import (
    cylindrical_sheet,
    mesh_quality,
    rotational_equivariance,
)


HERE = Path(__file__).resolve().parent
SPEC_PATH = (
    HERE / "docs" / "diag" / "dde_vertex_star_advection_cases.yaml"
)
RESULT_PATH = (
    HERE / "docs" / "diag" / "dde_vertex_star_advection_results.json"
)


def integrate(surface, *, final_time, steps, provider):
    state = surface
    reports = []
    dt = final_time / steps
    for _ in range(steps):
        result = advance_surface_normal_geometry_heun(
            state,
            dt=dt,
            velocity_provider=provider,
        )
        state = result.surface
        reports.append(result.report)
    return state, reports


def normal_ode_provider(rate: float, source: float):
    def provider(surface):
        points, _, _ = surface.interior_collocation_points()
        velocity = np.column_stack(
            (
                np.zeros(len(points)),
                np.zeros(len(points)),
                rate * points[:, 2] + source,
            )
        )
        return project_vertex_star_normal_geometry_velocity(
            surface,
            velocity,
        )

    return provider


def run(spec: dict) -> dict:
    guards = spec["guards"]
    ode = spec["normal_ode_case"]
    planar = structured_sheet(2)
    exact_height = (
        ode["source"]
        / ode["rate"]
        * (np.exp(ode["rate"] * ode["final_time"]) - 1.0)
    )
    ode_errors = []
    for steps in ode["steps"]:
        state, _ = integrate(
            planar,
            final_time=ode["final_time"],
            steps=int(steps),
            provider=normal_ode_provider(
                ode["rate"],
                ode["source"],
            ),
        )
        ode_errors.append(
            float(np.max(np.abs(state.vertices[:, 2] - exact_height)))
        )
    ode_ratios = [
        ode_errors[index] / ode_errors[index + 1]
        for index in range(len(ode_errors) - 1)
    ]

    cylinder_spec = spec["cylinder"]
    initial = cylindrical_sheet(cylinder_spec)
    topology = {
        "continuity": asdict(initial.continuity_report()),
        "boundary": asdict(initial.boundary_report()),
    }
    dynamic = spec["self_induced_case"]
    finest_order = int(dynamic["line_quadrature_orders"][-1])
    initial_projection = self_induced_vertex_star_normal_velocity(
        initial,
        quadrature_order=finest_order,
    )
    initial_speed_ratio = float(
        np.max(np.abs(initial_projection.vertex_normal_speed))
        / max(
            initial_projection.report.max_input_norm,
            np.finfo(float).eps,
        )
    )
    states = []
    report_sets = []
    for steps in dynamic["steps"]:
        state, reports = integrate(
            initial,
            final_time=dynamic["final_time"],
            steps=int(steps),
            provider=lambda current, order=finest_order: (
                self_induced_vertex_star_normal_velocity(
                    current,
                    quadrature_order=order,
                )
            ),
        )
        states.append(state)
        report_sets.append(reports)
    coarse_change = float(
        np.max(
            np.linalg.norm(
                states[1].vertices - states[0].vertices,
                axis=1,
            )
        )
    )
    fine_change = float(
        np.max(
            np.linalg.norm(
                states[2].vertices - states[1].vertices,
                axis=1,
            )
        )
    )
    time_ratio = (
        coarse_change / fine_change if fine_change > 0.0 else np.inf
    )
    displacement = float(
        np.max(
            np.linalg.norm(
                states[-1].vertices - initial.vertices,
                axis=1,
            )
        )
    )

    quadrature_states = []
    for order in dynamic["line_quadrature_orders"]:
        state, _ = integrate(
            initial,
            final_time=dynamic["final_time"],
            steps=int(dynamic["steps"][-1]),
            provider=lambda current, order=int(order): (
                self_induced_vertex_star_normal_velocity(
                    current,
                    quadrature_order=order,
                )
            ),
        )
        quadrature_states.append(state)
    quadrature_change = float(
        np.max(
            np.linalg.norm(
                quadrature_states[-1].vertices
                - quadrature_states[-2].vertices,
                axis=1,
            )
        )
    )
    quadrature_fraction = quadrature_change / max(
        displacement,
        np.finfo(float).eps,
    )
    symmetry = rotational_equivariance(
        states[-1],
        circumferential_nodes=int(
            cylinder_spec["circumferential_nodes"]
        ),
        axial_cells=int(cylinder_spec["axial_cells"]),
    )
    minimum_area_ratio, maximum_edge_ratio = mesh_quality(
        initial,
        states[-1],
    )
    maximum_condition = max(
        max(
            report.stage0_projection.condition_number,
            report.stage1_projection.condition_number,
        )
        for reports in report_sets
        for report in reports
    )
    potential_jump_residual = max(
        report.kelvin.max_material_mu_residual
        for reports in report_sets
        for report in reports
    )
    all_steps = all(
        report.passed
        for reports in report_sets
        for report in reports
    )
    checks = {
        "initial_topology": (
            topology["continuity"]["compatible"]
            and topology["boundary"]["compatible"]
        ),
        "normal_ode_second_order": min(ode_ratios)
        >= guards["normal_ode_error_ratio_min"],
        "self_time_cauchy": (
            time_ratio >= guards["self_time_cauchy_ratio_min"]
            and fine_change
            <= guards["self_finest_change_over_radius_max"]
        ),
        "line_quadrature": quadrature_fraction
        <= guards["quadrature_change_over_displacement_max"],
        "rotational_equivariance": symmetry
        <= guards["rotational_equivariance_max_abs"],
        "vertex_star_conditioning": maximum_condition
        <= guards["vertex_star_condition_number_max"],
        "vertex_speed_bounded": initial_speed_ratio
        <= guards["vertex_speed_over_sample_speed_max"],
        "mesh_quality": (
            minimum_area_ratio >= guards["minimum_area_ratio"]
            and maximum_edge_ratio
            <= guards["maximum_edge_length_ratio"]
        ),
        "potential_jump_identity": potential_jump_residual
        <= guards["potential_jump_residual_max"],
        "all_step_guards": all_steps,
    }
    return {
        "spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": spec["role"],
        "normal_ode": {
            "errors": ode_errors,
            "error_ratios": ode_ratios,
        },
        "initial_topology": topology,
        "self_induced": {
            "coarse_to_mid_change": coarse_change,
            "mid_to_fine_change": fine_change,
            "time_cauchy_ratio": time_ratio,
            "finest_displacement": displacement,
            "quadrature_change_over_displacement": quadrature_fraction,
            "rotational_equivariance_max_abs": symmetry,
            "maximum_vertex_star_condition_number": maximum_condition,
            "initial_vertex_speed_over_sample_speed": initial_speed_ratio,
            "minimum_area_ratio": minimum_area_ratio,
            "maximum_edge_length_ratio": maximum_edge_ratio,
            "max_potential_jump_residual": potential_jump_residual,
            "last_step_report": asdict(report_sets[-1][-1]),
        },
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
