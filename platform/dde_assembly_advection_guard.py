"""Run the preregistered explicit-interface assembly equivalence gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.distributed_doublet import (
    DistributedDoubletError,
    QuadraticDoubletAssembly,
    QuadraticDoubletPatch,
    QuadraticDoubletSurface,
)
from claim_runtime.material_sheet_advection import (
    advance_assembly_normal_geometry_heun,
    advance_surface_normal_geometry_heun,
    self_induced_assembly_vertex_star_normal_velocity,
    self_induced_vertex_star_normal_velocity,
)
from dde_cylindrical_advection_guard import cylindrical_sheet


HERE = Path(__file__).resolve().parent
SPEC_PATH = (
    HERE / "docs" / "diag" / "dde_assembly_advection_cases.yaml"
)
RESULT_PATH = (
    HERE / "docs" / "diag" / "dde_assembly_advection_results.json"
)


def split_axial_bands(
    surface: QuadraticDoubletSurface,
    *,
    circumferential_nodes: int,
    axial_cells: int,
) -> tuple[QuadraticDoubletAssembly, tuple[np.ndarray, ...]]:
    """Split a cylindrical surface without geometric proximity matching."""
    expected_faces = 2 * circumferential_nodes * axial_cells
    if len(surface) != expected_faces:
        raise DistributedDoubletError(
            "surface face ordering does not match the cylindrical generator"
        )
    patches = []
    local_to_global = []
    for axial_index in range(axial_cells):
        start = 2 * circumferential_nodes * axial_index
        stop = start + 2 * circumferential_nodes
        global_faces = surface.faces[start:stop]
        global_vertices = np.unique(global_faces)
        global_to_local = {
            int(global_index): local_index
            for local_index, global_index in enumerate(global_vertices)
        }
        local_faces = np.array(
            [
                [global_to_local[int(index)] for index in face]
                for face in global_faces
            ],
            dtype=np.int64,
        )
        patch_surface = QuadraticDoubletSurface(
            surface.vertices[global_vertices],
            local_faces,
            surface.face_mu[start:stop],
        )
        roles = {}
        for local_edge in patch_surface.boundary_edge_traces():
            global_edge = tuple(
                sorted(
                    int(global_vertices[local_index])
                    for local_index in local_edge
                )
            )
            rings = {
                index // circumferential_nodes
                for index in global_edge
            }
            if len(rings) != 1:
                raise DistributedDoubletError(
                    "unexpected circumferential patch boundary"
                )
            ring = next(iter(rings))
            if ring in (0, axial_cells):
                role = "zero"
            else:
                role = f"interface:ring-{ring}:{global_edge[0]}-{global_edge[1]}"
            roles[local_edge] = role
        patches.append(
            QuadraticDoubletPatch(
                f"axial-band-{axial_index}",
                patch_surface,
                roles,
            )
        )
        local_to_global.append(global_vertices)
    return QuadraticDoubletAssembly(patches), tuple(local_to_global)


def assembly_global_vertices(
    assembly: QuadraticDoubletAssembly,
    local_to_global: tuple[np.ndarray, ...],
    *,
    vertex_count: int,
) -> np.ndarray:
    values: list[list[np.ndarray]] = [[] for _ in range(vertex_count)]
    for patch, mapping in zip(assembly.patches, local_to_global):
        for local_index, global_index in enumerate(mapping):
            values[int(global_index)].append(
                patch.surface.vertices[local_index]
            )
    if any(not group for group in values):
        raise DistributedDoubletError("assembly omitted a global vertex")
    return np.array(
        [np.mean(np.asarray(group), axis=0) for group in values]
    )


def assembly_global_velocity(
    projection,
    local_to_global: tuple[np.ndarray, ...],
    *,
    vertex_count: int,
) -> np.ndarray:
    values: list[list[np.ndarray]] = [[] for _ in range(vertex_count)]
    for patch_index, mapping in enumerate(local_to_global):
        velocity = projection.vertex_velocity(patch_index)
        for local_index, global_index in enumerate(mapping):
            values[int(global_index)].append(velocity[local_index])
    return np.array(
        [np.mean(np.asarray(group), axis=0) for group in values]
    )


def integrate_surface(initial, *, final_time, steps, order):
    state = initial
    reports = []
    for _ in range(steps):
        result = advance_surface_normal_geometry_heun(
            state,
            dt=final_time / steps,
            velocity_provider=lambda current: (
                self_induced_vertex_star_normal_velocity(
                    current,
                    quadrature_order=order,
                )
            ),
        )
        state = result.surface
        reports.append(result.report)
    return state, reports


def integrate_assembly(initial, *, final_time, steps, order):
    state = initial
    reports = []
    for _ in range(steps):
        result = advance_assembly_normal_geometry_heun(
            state,
            dt=final_time / steps,
            velocity_provider=lambda current: (
                self_induced_assembly_vertex_star_normal_velocity(
                    current,
                    quadrature_order=order,
                )
            ),
        )
        state = result.assembly
        reports.append(result.report)
    return state, reports


def run(spec: dict) -> dict:
    cylinder = spec["cylinder"]
    dynamic = spec["self_induced_case"]
    guards = spec["guards"]
    surface = cylindrical_sheet(cylinder)
    assembly, local_to_global = split_axial_bands(
        surface,
        circumferential_nodes=int(cylinder["circumferential_nodes"]),
        axial_cells=int(cylinder["axial_cells"]),
    )
    initial_topology = assembly.topology_report()
    order = int(dynamic["line_quadrature_order"])
    monolithic_projection = self_induced_vertex_star_normal_velocity(
        surface,
        quadrature_order=order,
    )
    assembly_projection = (
        self_induced_assembly_vertex_star_normal_velocity(
            assembly,
            quadrature_order=order,
        )
    )
    projected_global = assembly_global_velocity(
        assembly_projection,
        local_to_global,
        vertex_count=len(surface.vertices),
    )
    initial_velocity_difference = float(
        np.max(
            np.linalg.norm(
                projected_global
                - monolithic_projection.vertex_velocity,
                axis=1,
            )
        )
    )

    assembly_states = []
    monolithic_states = []
    report_sets = []
    representation_differences = []
    for steps_value in dynamic["steps"]:
        steps = int(steps_value)
        assembly_state, assembly_reports = integrate_assembly(
            assembly,
            final_time=float(dynamic["final_time"]),
            steps=steps,
            order=order,
        )
        monolithic_state, _ = integrate_surface(
            surface,
            final_time=float(dynamic["final_time"]),
            steps=steps,
            order=order,
        )
        global_vertices = assembly_global_vertices(
            assembly_state,
            local_to_global,
            vertex_count=len(surface.vertices),
        )
        representation_differences.append(
            float(
                np.max(
                    np.linalg.norm(
                        global_vertices - monolithic_state.vertices,
                        axis=1,
                    )
                )
            )
        )
        assembly_states.append(global_vertices)
        monolithic_states.append(monolithic_state.vertices)
        report_sets.append(assembly_reports)

    coarse_change = float(
        np.max(
            np.linalg.norm(
                assembly_states[1] - assembly_states[0],
                axis=1,
            )
        )
    )
    fine_change = float(
        np.max(
            np.linalg.norm(
                assembly_states[2] - assembly_states[1],
                axis=1,
            )
        )
    )
    time_ratio = (
        coarse_change / fine_change if fine_change > 0.0 else np.inf
    )
    max_seam_gap = max(
        report.topology.max_interface_geometry_gap
        for reports in report_sets
        for report in reports
    )
    max_kelvin = max(
        patch.max_material_mu_residual
        for reports in report_sets
        for report in reports
        for patch in report.patch_kelvin
    )
    all_steps = all(
        report.passed
        for reports in report_sets
        for report in reports
    )
    checks = {
        "initial_topology": initial_topology.compatible,
        "initial_velocity_representation_invariance": (
            initial_velocity_difference
            <= guards["initial_velocity_max_abs_difference"]
        ),
        "dynamic_representation_invariance": (
            max(representation_differences)
            <= guards["final_geometry_max_abs_difference"]
        ),
        "assembly_time_cauchy": (
            time_ratio >= guards["assembly_time_cauchy_ratio_min"]
        ),
        "seam_geometry_identity": (
            max_seam_gap <= guards["seam_geometry_gap_max"]
        ),
        "potential_jump_identity": (
            max_kelvin <= guards["potential_jump_residual_max"]
        ),
        "all_step_guards": all_steps,
    }
    return {
        "spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": spec["role"],
        "initial_topology": asdict(initial_topology),
        "initial_velocity_max_abs_difference": initial_velocity_difference,
        "dynamic": {
            "steps": [int(value) for value in dynamic["steps"]],
            "representation_max_abs_differences": representation_differences,
            "coarse_to_mid_change": coarse_change,
            "mid_to_fine_change": fine_change,
            "time_cauchy_ratio": time_ratio,
            "max_seam_geometry_gap": max_seam_gap,
            "max_potential_jump_residual": max_kelvin,
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
