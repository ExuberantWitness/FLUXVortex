"""Run the preregistered S3j continuous-P2 ALE transport spatial gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.p2_surface_material_transport import (  # noqa: E402
    assemble_p2_surface_material_transport,
)


CASES = (
    HERE / "docs" / "diag"
    / "p2_wake_gauge_transport_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "p2_wake_gauge_transport_results.json"
)


def _mesh(cells: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, cells + 1)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.column_stack(
        (first.ravel(), second.ravel(), np.zeros(first.size))
    )

    def index(i: int, j: int) -> int:
        return i * (cells + 1) + j

    faces = []
    for i in range(cells):
        for j in range(cells):
            v00 = index(i, j)
            v10 = index(i + 1, j)
            v01 = index(i, j + 1)
            v11 = index(i + 1, j + 1)
            if (i + j) % 2 == 0:
                faces.extend(((v00, v10, v11), (v00, v11, v01)))
            else:
                faces.extend(((v00, v10, v01), (v10, v11, v01)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _velocity(points: np.ndarray, first: float, second: float) -> np.ndarray:
    return np.column_stack(
        (
            first * np.sin(np.pi * points[:, 0]),
            second * np.sin(np.pi * points[:, 1]),
            np.zeros(len(points)),
        )
    )


def _flow_inverse(value: np.ndarray, coefficient: float, time: float) -> np.ndarray:
    return (
        2.0
        / np.pi
        * np.arctan(
            np.tan(0.5 * np.pi * value)
            * np.exp(-coefficient * np.pi * time)
        )
    )


def _mu(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        0.15
        + 0.4 * first
        + 0.2 * second
        + 0.3 * first * second
        + 0.25 * first**2
        - 0.1 * second**2
    )


def _exact(points: np.ndarray, *, time: float, first: float, second: float):
    first0 = _flow_inverse(points[:, 0], first, time)
    second0 = _flow_inverse(points[:, 1], second, time)
    return _mu(first0, second0)


def _advance(operator, initial: np.ndarray, time: float) -> np.ndarray:
    generator = -np.linalg.solve(
        operator.mass_matrix,
        operator.advection_matrix,
    )
    return expm(time * generator) @ initial


def _mass_relative_error(operator, value: np.ndarray, exact: np.ndarray) -> float:
    difference = value - exact
    numerator = float(
        difference @ operator.mass_matrix @ difference
    )
    denominator = float(exact @ operator.mass_matrix @ exact)
    return float(np.sqrt(numerator / denominator))


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = axis / np.linalg.norm(axis)
    cross = np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        )
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(direction, direction)
        + np.sin(angle) * cross
    )


def _shared_trace_jump(operator, value: np.ndarray) -> float:
    records: dict[tuple[int, int], list[np.ndarray]] = {}
    local_edges = ((0, 1, 3), (1, 2, 4), (2, 0, 5))
    for face, dofs in zip(
        operator.topology.faces,
        operator.topology.local_to_global,
    ):
        local = value[dofs]
        for first, second, middle in local_edges:
            a = int(face[first])
            b = int(face[second])
            if a < b:
                trace = np.array((local[first], local[middle], local[second]))
                edge = (a, b)
            else:
                trace = np.array((local[second], local[middle], local[first]))
                edge = (b, a)
            records.setdefault(edge, []).append(trace)
    return max(
        (
            float(np.max(np.abs(item[0] - item[1])))
            for item in records.values()
            if len(item) == 2
        ),
        default=0.0,
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    first = float(canonical["material_velocity"]["a"])
    second = float(canonical["material_velocity"]["b"])
    time = float(canonical["time"]["end"])
    order = int(canonical["quadrature"]["assembly_order"])
    families = [
        int(value)
        for value in canonical["surface"]["structured_cell_families"]
    ]
    rows = []
    operators = []
    final_values = []
    for cells in families:
        vertices, faces = _mesh(cells)
        operator = assemble_p2_surface_material_transport(
            vertices,
            faces,
            relative_velocity_provider=lambda points: _velocity(
                points,
                first,
                second,
            ),
            quadrature_order=order,
        )
        coordinates = operator.topology.degree_of_freedom_coordinates
        initial = _mu(coordinates[:, 0], coordinates[:, 1])
        final = _advance(operator, initial, time)
        exact = _exact(
            coordinates,
            time=time,
            first=first,
            second=second,
        )
        error = _mass_relative_error(operator, final, exact)
        rows.append(
            {
                "cells": cells,
                "vertices": len(vertices),
                "faces": len(faces),
                "P2_dofs": operator.topology.degree_of_freedom_count,
                "mass_rank": operator.mass_rank,
                "mass_condition_number": operator.mass_condition_number,
                "constant_rate_abs_max": operator.constant_rate_residual,
                "relative_velocity_normal_abs_max": (
                    operator.maximum_relative_velocity_normal_component
                ),
                "shared_trace_jump_abs_max": _shared_trace_jump(
                    operator,
                    final,
                ),
                "relative_L2_error": error,
            }
        )
        operators.append(operator)
        final_values.append(final)
    errors = [row["relative_L2_error"] for row in rows]
    cauchy_ratios = [
        first_error / second_error
        for first_error, second_error in zip(errors, errors[1:])
    ]
    minimum_cauchy = min(cauchy_ratios)

    finest = operators[-1]
    finest_vertices = finest.topology.vertices
    finest_faces = finest.topology.faces
    rigid = canonical["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    translation = np.asarray(rigid["translation"], dtype=float)
    moved_vertices = finest_vertices @ rotation.T + translation

    def moved_velocity(points: np.ndarray) -> np.ndarray:
        base = (points - translation) @ rotation
        return _velocity(base, first, second) @ rotation.T

    moved = assemble_p2_surface_material_transport(
        moved_vertices,
        finest_faces,
        relative_velocity_provider=moved_velocity,
        quadrature_order=order,
    )
    rigid_mass = float(
        np.max(
            np.abs(moved.mass_matrix - finest.mass_matrix),
            initial=0.0,
        )
    )
    rigid_advection = float(
        np.max(
            np.abs(
                moved.advection_matrix - finest.advection_matrix
            ),
            initial=0.0,
        )
    )
    moved_coordinates = (
        moved.topology.degree_of_freedom_coordinates - translation
    ) @ rotation
    moved_initial = _mu(
        moved_coordinates[:, 0],
        moved_coordinates[:, 1],
    )
    moved_final = _advance(moved, moved_initial, time)
    rigid_final = float(
        np.max(
            np.abs(moved_final - final_values[-1]),
            initial=0.0,
        )
    )

    counter_order = int(
        canonical["quadrature"]["counterfactual_order"]
    )
    quadrature = assemble_p2_surface_material_transport(
        finest_vertices,
        finest_faces,
        relative_velocity_provider=lambda points: _velocity(
            points,
            first,
            second,
        ),
        quadrature_order=counter_order,
    )
    coordinates = quadrature.topology.degree_of_freedom_coordinates
    quadrature_final = _advance(
        quadrature,
        _mu(coordinates[:, 0], coordinates[:, 1]),
        time,
    )
    quadrature_difference = float(
        np.max(
            np.abs(quadrature_final - final_values[-1]),
            initial=0.0,
        )
    )
    rank_deficiency = max(
        row["P2_dofs"] - row["mass_rank"] for row in rows
    )
    constant_residual = max(
        row["constant_rate_abs_max"] for row in rows
    )
    trace_jump = max(
        row["shared_trace_jump_abs_max"] for row in rows
    )
    monotone = all(
        second_error < first_error
        for first_error, second_error in zip(errors, errors[1:])
    )
    checks = {
        "mass_is_full_rank": (
            rank_deficiency
            <= int(thresholds["mass_rank_deficiency_max"])
        ),
        "constant_scalar_is_preserved": (
            constant_residual
            <= float(thresholds["constant_rate_abs_max"])
        ),
        "shared_P2_traces_remain_exact": (
            trace_jump
            <= float(thresholds["shared_trace_jump_abs_max"])
        ),
        "relative_L2_error_is_monotone": monotone,
        "spatial_Cauchy_gate_passes": (
            minimum_cauchy
            >= float(thresholds["l2_cauchy_ratio_min"])
        ),
        "finest_relative_L2_error_passes": (
            errors[-1]
            <= float(thresholds["finest_relative_l2_error_max"])
        ),
        "rigid_frame_matrices_are_objective": (
            max(rigid_mass, rigid_advection)
            <= float(
                thresholds["rigid_matrix_abs_difference_max"]
            )
        ),
        "rigid_frame_final_scalar_is_objective": (
            rigid_final
            <= float(
                thresholds[
                    "rigid_final_scalar_abs_difference_max"
                ]
            )
        ),
        "quadrature_gate_passes": (
            quadrature_difference
            <= float(
                thresholds[
                    "quadrature_final_scalar_abs_difference_max"
                ]
            )
        ),
    }
    result = {
        "artifact": "p2_wake_gauge_transport_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "mesh_family": rows,
        "aggregate_metrics": {
            "mass_rank_deficiency_max": rank_deficiency,
            "constant_rate_abs_max": constant_residual,
            "shared_trace_jump_abs_max": trace_jump,
            "relative_L2_errors": errors,
            "relative_L2_cauchy_ratios": cauchy_ratios,
            "minimum_relative_L2_cauchy_ratio": minimum_cauchy,
            "finest_relative_L2_error": errors[-1],
            "rigid_mass_matrix_abs_difference": rigid_mass,
            "rigid_advection_matrix_abs_difference": rigid_advection,
            "rigid_final_scalar_abs_difference": rigid_final,
            "quadrature_final_scalar_abs_difference": (
                quadrature_difference
            ),
        },
        "forbidden_quantities_absent": [
            "mass_lumping",
            "upwind",
            "artificial_diffusion",
            "limiter",
            "ridge",
            "wake_core",
            "actual_induced_velocity",
            "pressure",
            "force",
            "target_load",
            "structural_dynamics",
        ],
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    payload = run()
    print(
        json.dumps(payload, ensure_ascii=False, indent=2),
        flush=True,
    )
    raise SystemExit(
        0 if payload["stage_decision"] == "GO" else 1
    )
