"""Preregistered no-force dynamic gates for N3.1j4c."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.material_sheet_advection import (
    advance_material_surface_heun,
    self_induced_geometry_velocity,
)
from claim_runtime.sheet_velocity_projection import (
    project_sheet_average_velocity,
)
from dde_canonical_field_guard import structured_sheet


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "docs" / "diag" / "dde_material_advection_cases.yaml"
RESULT_PATH = HERE / "docs" / "diag" / "dde_material_advection_results.json"


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
    area_ratio = []
    for face_index in range(len(initial)):
        area_ratio.append(
            final.element(face_index).area
            / initial.element(face_index).area
        )
    edges = {
        tuple(sorted((int(face[a]), int(face[b]))))
        for face in initial.faces
        for a, b in ((0, 1), (1, 2), (2, 0))
    }
    edge_ratio = []
    for start, end in edges:
        initial_length = np.linalg.norm(
            initial.vertices[end] - initial.vertices[start]
        )
        final_length = np.linalg.norm(
            final.vertices[end] - final.vertices[start]
        )
        edge_ratio.append(final_length / initial_length)
    return float(min(area_ratio)), float(max(edge_ratio))


def mirror_residual(surface, cells: int) -> float:
    vertices = surface.vertices.reshape(cells + 1, cells + 1, 3)
    x_reflected = vertices[:, ::-1].copy()
    x_reflected[..., 0] = 1.0 - x_reflected[..., 0]
    y_reflected = vertices[::-1, :].copy()
    y_reflected[..., 1] = 1.0 - y_reflected[..., 1]
    return float(
        max(
            np.max(np.abs(vertices - x_reflected)),
            np.max(np.abs(vertices - y_reflected)),
        )
    )


def run(spec: dict) -> dict:
    guards = spec["guards"]
    initial = structured_sheet(
        int(spec["self_induced_case"]["mesh_cells"])
    )

    rotation = spec["rotation_case"]
    angle = rotation["angular_speed"] * rotation["final_time"]
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    exact = initial.vertices @ rotation_matrix.T
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

    self_case = spec["self_induced_case"]
    finest_quadrature = int(self_case["line_quadrature_orders"][-1])
    self_states = []
    self_reports = []
    for steps in self_case["steps"]:
        state, reports = integrate(
            initial,
            final_time=self_case["final_time"],
            steps=int(steps),
            provider=lambda current, order=finest_quadrature: (
                self_induced_geometry_velocity(
                    current,
                    quadrature_order=order,
                )
            ),
        )
        self_states.append(state)
        self_reports.append(reports)
    coarse_change = float(
        np.max(
            np.linalg.norm(
                self_states[1].vertices - self_states[0].vertices,
                axis=1,
            )
        )
    )
    fine_change = float(
        np.max(
            np.linalg.norm(
                self_states[2].vertices - self_states[1].vertices,
                axis=1,
            )
        )
    )
    time_ratio = (
        coarse_change / fine_change if fine_change > 0.0 else np.inf
    )
    finest_displacement = float(
        np.max(
            np.linalg.norm(
                self_states[-1].vertices - initial.vertices,
                axis=1,
            )
        )
    )

    quadrature_states = []
    for order in self_case["line_quadrature_orders"]:
        state, _ = integrate(
            initial,
            final_time=self_case["final_time"],
            steps=int(self_case["steps"][-1]),
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
        finest_displacement,
        np.finfo(float).eps,
    )
    minimum_area_ratio, maximum_edge_ratio = mesh_quality(
        initial,
        self_states[-1],
    )
    symmetry = mirror_residual(
        self_states[-1],
        int(self_case["mesh_cells"]),
    )
    kelvin_residual = max(
        report.kelvin.max_material_mu_residual
        for reports in self_reports
        for report in reports
    )
    all_step_guards = all(
        report.passed
        for reports in self_reports
        for report in reports
    )
    checks = {
        "rotation_second_order": min(rotation_ratios)
        >= guards["rotation_error_ratio_min"],
        "self_time_cauchy": (
            time_ratio >= guards["self_time_cauchy_ratio_min"]
            and fine_change
            <= guards["self_finest_change_over_chord_max"]
        ),
        "line_quadrature": quadrature_fraction
        <= guards["quadrature_change_over_displacement_max"],
        "mirror_symmetry": symmetry
        <= guards["mirror_symmetry_max_abs"],
        "mesh_quality": (
            minimum_area_ratio >= guards["minimum_area_ratio"]
            and maximum_edge_ratio
            <= guards["maximum_edge_length_ratio"]
        ),
        "material_kelvin": kelvin_residual
        <= guards["material_mu_residual_max"],
        "all_step_guards": all_step_guards,
    }
    return {
        "spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": spec["role"],
        "rotation": {
            "errors": rotation_errors,
            "error_ratios": rotation_ratios,
        },
        "self_induced": {
            "coarse_to_mid_change": coarse_change,
            "mid_to_fine_change": fine_change,
            "time_cauchy_ratio": time_ratio,
            "finest_displacement": finest_displacement,
            "quadrature_change": quadrature_change,
            "quadrature_change_over_displacement": quadrature_fraction,
            "mirror_symmetry_max_abs": symmetry,
            "minimum_area_ratio": minimum_area_ratio,
            "maximum_edge_length_ratio": maximum_edge_ratio,
            "max_material_mu_residual": kelvin_residual,
            "last_step_report": asdict(self_reports[-1][-1]),
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
