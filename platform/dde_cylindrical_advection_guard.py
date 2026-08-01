"""Preregistered cylindrical follow-up to the planar advection NO-GO."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.distributed_doublet import QuadraticDoubletSurface
from claim_runtime.material_sheet_advection import (
    advance_material_surface_heun,
    self_induced_geometry_velocity,
)
from claim_runtime.sheet_velocity_projection import (
    project_sheet_average_velocity,
)


HERE = Path(__file__).resolve().parent
SPEC_PATH = (
    HERE / "docs" / "diag" / "dde_cylindrical_advection_cases.yaml"
)
RESULT_PATH = (
    HERE / "docs" / "diag" / "dde_cylindrical_advection_results.json"
)


def cylindrical_sheet(spec: dict) -> QuadraticDoubletSurface:
    radius = float(spec["radius"])
    length = float(spec["length"])
    circumferential_nodes = int(spec["circumferential_nodes"])
    axial_cells = int(spec["axial_cells"])
    theta_step = 2.0 * np.pi / circumferential_nodes
    phase_increment = (
        float(spec.get("axial_phase_increment_fraction", 0.0))
        * theta_step
    )
    axial = np.linspace(0.0, length, axial_cells + 1)
    vertices = np.array(
        [
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                coordinate,
            ]
            for axial_index, coordinate in enumerate(axial)
            for angle in (
                theta_step * np.arange(circumferential_nodes)
                + axial_index * phase_increment
            )
        ]
    )

    def index(axial_index: int, theta_index: int) -> int:
        return (
            axial_index * circumferential_nodes
            + theta_index % circumferential_nodes
        )

    faces = []
    for axial_index in range(axial_cells):
        for theta_index in range(circumferential_nodes):
            lower_left = index(axial_index, theta_index)
            lower_right = index(axial_index, theta_index + 1)
            upper_left = index(axial_index + 1, theta_index)
            upper_right = index(axial_index + 1, theta_index + 1)
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))
    face_array = np.asarray(faces, dtype=np.int64)
    face_mu = []
    for face in face_array:
        triangle = vertices[face]
        nodes = np.vstack(
            (
                triangle,
                0.5 * (triangle[0] + triangle[1]),
                0.5 * (triangle[1] + triangle[2]),
                0.5 * (triangle[2] + triangle[0]),
            )
        )
        normalized_axial = nodes[:, 2] / length
        face_mu.append(
            normalized_axial * (1.0 - normalized_axial)
        )
    return QuadraticDoubletSurface(
        vertices,
        face_array,
        np.asarray(face_mu),
    )


def rotation_provider(angular_speed: float):
    def provider(surface):
        points, _, _ = surface.interior_collocation_points()
        velocity = np.column_stack(
            (
                -angular_speed * points[:, 1],
                angular_speed * points[:, 0],
                np.zeros(len(points)),
            )
        )
        return project_sheet_average_velocity(surface, velocity)

    return provider


def integrate(surface, *, final_time, steps, provider):
    state = surface
    reports = []
    dt = final_time / steps
    for _ in range(steps):
        result = advance_material_surface_heun(
            state,
            dt=dt,
            velocity_provider=provider,
        )
        state = result.surface
        reports.append(result.report)
    return state, reports


def mesh_quality(initial, final) -> tuple[float, float]:
    area_ratio = [
        final.element(index).area / initial.element(index).area
        for index in range(len(initial))
    ]
    edges = {
        tuple(sorted((int(face[a]), int(face[b]))))
        for face in initial.faces
        for a, b in ((0, 1), (1, 2), (2, 0))
    }
    edge_ratio = []
    for start, end in edges:
        edge_ratio.append(
            np.linalg.norm(final.vertices[end] - final.vertices[start])
            / np.linalg.norm(
                initial.vertices[end] - initial.vertices[start]
            )
        )
    return float(min(area_ratio)), float(max(edge_ratio))


def rotational_equivariance(
    surface,
    *,
    circumferential_nodes: int,
    axial_cells: int,
) -> float:
    vertices = surface.vertices.reshape(
        axial_cells + 1,
        circumferential_nodes,
        3,
    )
    angle = 2.0 * np.pi / circumferential_nodes
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = vertices @ rotation.T
    actual = np.roll(vertices, -1, axis=1)
    return float(np.max(np.abs(actual - expected)))


def run(spec: dict) -> dict:
    cylinder = spec["cylinder"]
    guards = spec["guards"]
    initial = cylindrical_sheet(cylinder)
    topology = {
        "continuity": asdict(initial.continuity_report()),
        "boundary": asdict(initial.boundary_report()),
    }

    rotation = spec["rotation_case"]
    angle = rotation["angular_speed"] * rotation["final_time"]
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    exact = initial.vertices @ matrix.T
    rotation_errors = []
    for steps in rotation["steps"]:
        state, _ = integrate(
            initial,
            final_time=rotation["final_time"],
            steps=int(steps),
            provider=rotation_provider(rotation["angular_speed"]),
        )
        rotation_errors.append(
            float(
                np.max(
                    np.linalg.norm(state.vertices - exact, axis=1)
                )
            )
        )
    rotation_ratios = [
        rotation_errors[index] / rotation_errors[index + 1]
        for index in range(len(rotation_errors) - 1)
    ]

    dynamic = spec["self_induced_case"]
    finest_order = int(dynamic["line_quadrature_orders"][-1])
    states = []
    report_sets = []
    for steps in dynamic["steps"]:
        state, reports = integrate(
            initial,
            final_time=dynamic["final_time"],
            steps=int(steps),
            provider=lambda current, order=finest_order: (
                self_induced_geometry_velocity(
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
                self_induced_geometry_velocity(
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
            cylinder["circumferential_nodes"]
        ),
        axial_cells=int(cylinder["axial_cells"]),
    )
    minimum_area_ratio, maximum_edge_ratio = mesh_quality(
        initial,
        states[-1],
    )
    projection_fraction = max(
        max(
            report.stage0_projection.max_abs_residual_fraction,
            report.stage1_projection.max_abs_residual_fraction,
        )
        for reports in report_sets
        for report in reports
    )
    kelvin_residual = max(
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
        "rotation_second_order": min(rotation_ratios)
        >= guards["rotation_error_ratio_min"],
        "self_time_cauchy": (
            time_ratio >= guards["self_time_cauchy_ratio_min"]
            and fine_change
            <= guards["self_finest_change_over_radius_max"]
        ),
        "line_quadrature": quadrature_fraction
        <= guards["quadrature_change_over_displacement_max"],
        "rotational_equivariance": symmetry
        <= guards["rotational_equivariance_max_abs"],
        "projection_residual": projection_fraction
        <= guards["projection_residual_fraction_max"],
        "mesh_quality": (
            minimum_area_ratio >= guards["minimum_area_ratio"]
            and maximum_edge_ratio
            <= guards["maximum_edge_length_ratio"]
        ),
        "material_kelvin": kelvin_residual
        <= guards["material_mu_residual_max"],
        "all_step_guards": all_steps,
    }
    return {
        "spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": spec["role"],
        "initial_topology": topology,
        "rotation": {
            "errors": rotation_errors,
            "error_ratios": rotation_ratios,
        },
        "self_induced": {
            "coarse_to_mid_change": coarse_change,
            "mid_to_fine_change": fine_change,
            "time_cauchy_ratio": time_ratio,
            "finest_displacement": displacement,
            "quadrature_change_over_displacement": quadrature_fraction,
            "rotational_equivariance_max_abs": symmetry,
            "maximum_projection_residual_fraction": projection_fraction,
            "minimum_area_ratio": minimum_area_ratio,
            "maximum_edge_length_ratio": maximum_edge_ratio,
            "max_material_mu_residual": kelvin_residual,
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
