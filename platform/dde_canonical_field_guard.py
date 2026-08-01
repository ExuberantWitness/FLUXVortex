"""Preregistered no-force canonical gates for the N3.1j4 DDE field.

The runner verifies a static continuous P2 potential-jump sheet.  It neither
moves wake vertices nor computes pressure/force, so passing cannot promote a
roll-up model or alter the V4.1 production chain.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.distributed_doublet import (
    QuadraticDoubletSurface,
)


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "docs" / "diag" / "dde_canonical_field_cases.yaml"
RESULT_PATH = HERE / "docs" / "diag" / "dde_canonical_field_results.json"


def potential_jump(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return x * (1.0 - x) * y * (1.0 - y)


def structured_sheet(cells: int) -> QuadraticDoubletSurface:
    if not isinstance(cells, int) or cells < 1:
        raise ValueError("cells must be a positive integer")
    coordinate = np.linspace(0.0, 1.0, cells + 1)
    vertices = np.array(
        [[x, y, 0.0] for y in coordinate for x in coordinate],
        dtype=float,
    )

    def index(i: int, j: int) -> int:
        return j * (cells + 1) + i

    faces = []
    for j in range(cells):
        for i in range(cells):
            lower_left = index(i, j)
            lower_right = index(i + 1, j)
            upper_right = index(i + 1, j + 1)
            upper_left = index(i, j + 1)
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))
    face_array = np.asarray(faces, dtype=np.int64)
    face_mu = []
    for face in face_array:
        triangle = vertices[face]
        material_nodes = np.vstack(
            (
                triangle,
                0.5 * (triangle[0] + triangle[1]),
                0.5 * (triangle[1] + triangle[2]),
                0.5 * (triangle[2] + triangle[0]),
            )
        )
        face_mu.append(potential_jump(material_nodes))
    return QuadraticDoubletSurface(
        vertices,
        face_array,
        np.asarray(face_mu),
    )


def strict_owner(
    surface: QuadraticDoubletSurface,
    point: np.ndarray,
) -> tuple[int, np.ndarray]:
    owners = []
    for face_index in range(len(surface)):
        element = surface.element(face_index)
        barycentric = element.barycentric_coordinates(
            point[None, :],
            plane_tolerance=1.0e-13,
        )[0]
        if np.all(barycentric > 1.0e-12) and np.all(
            barycentric < 1.0 - 1.0e-12
        ):
            owners.append((face_index, barycentric))
    if len(owners) != 1:
        raise RuntimeError(
            f"canonical point has {len(owners)} strict owners, expected one"
        )
    return owners[0]


def segment_velocity(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    strength: float,
) -> np.ndarray:
    r1 = points - start
    r2 = points - end
    filament = end - start
    cross = np.cross(r1, r2)
    denominator = np.einsum("ij,ij->i", cross, cross)
    direction = (
        r1 / np.linalg.norm(r1, axis=1)[:, None]
        - r2 / np.linalg.norm(r2, axis=1)[:, None]
    )
    coefficient = direction @ filament
    return (
        strength
        * cross
        * coefficient[:, None]
        / (4.0 * np.pi * denominator[:, None])
    )


def constant_ring_gate(limit: float) -> dict:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.1, 0.9, 0.0]]
    )
    strength = 0.37
    surface = QuadraticDoubletSurface(
        vertices,
        [[0, 1, 2]],
        np.full((1, 6), strength),
    )
    points = np.array(
        [[0.23, 0.31, 0.7], [1.4, -0.5, 1.1], [-0.4, 0.2, 0.6]]
    )
    expected = np.zeros_like(points)
    for start, end in ((0, 1), (1, 2), (2, 0)):
        expected += segment_velocity(
            points,
            vertices[start],
            vertices[end],
            strength,
        )
    actual = surface.induced_velocity(points, quadrature_order=48)
    error = np.linalg.norm(actual - expected, axis=1)
    scale = np.maximum(np.linalg.norm(expected, axis=1), 1.0e-15)
    relative_error = float(np.max(error / scale))
    return {
        "max_relative_error": relative_error,
        "limit": limit,
        "passed": relative_error <= limit,
    }


def cauchy_metrics(values: list[np.ndarray]) -> dict:
    coarse_change = float(np.linalg.norm(values[1] - values[0]))
    fine_change = float(np.linalg.norm(values[2] - values[1]))
    fine_scale = max(float(np.linalg.norm(values[2])), 1.0e-15)
    ratio = (
        coarse_change / fine_change if fine_change > 0.0 else np.inf
    )
    return {
        "coarse_to_mid_change": coarse_change,
        "mid_to_fine_change": fine_change,
        "cauchy_ratio": ratio,
        "finest_relative_change": fine_change / fine_scale,
        "monotone": fine_change < coarse_change,
    }


def run(spec: dict) -> dict:
    manufactured = spec["manufactured_sheet"]
    quadrature = spec["quadrature"]
    guards = spec["guards"]
    surfaces = [
        structured_sheet(int(cells))
        for cells in manufactured["mesh_cells"]
    ]
    topology = []
    off_sheet_values = []
    on_sheet_values = []
    off_target = np.asarray(
        manufactured["off_sheet_target"],
        dtype=float,
    )
    on_target = np.asarray(
        manufactured["on_sheet_target"],
        dtype=float,
    )
    for cells, surface in zip(
        manufactured["mesh_cells"],
        surfaces,
    ):
        continuity = surface.continuity_report(
            tolerance=guards["topology_tolerance"]
        )
        boundary = surface.boundary_report(
            tolerance=guards["topology_tolerance"]
        )
        off_velocity, off_report = surface.induced_velocity_converged(
            off_target[None, :],
            orders=tuple(quadrature["off_sheet_orders"]),
            absolute_tolerance=guards["field_absolute_tolerance"],
            relative_tolerance=guards["field_relative_tolerance"],
        )
        owner, barycentric = strict_owner(surface, on_target)
        on_velocity, on_report = (
            surface.induced_velocity_sheet_average_converged(
                [owner],
                barycentric[None, :],
                orders=tuple(quadrature["on_sheet_orders"]),
                absolute_tolerance=guards["field_absolute_tolerance"],
                relative_tolerance=guards["field_relative_tolerance"],
            )
        )
        topology.append(
            {
                "cells": int(cells),
                "continuity": asdict(continuity),
                "boundary": asdict(boundary),
                "off_sheet_quadrature": asdict(off_report),
                "on_sheet_quadrature": asdict(on_report),
            }
        )
        off_sheet_values.append(off_velocity[0])
        on_sheet_values.append(on_velocity[0])

    finest = surfaces[-1]
    symmetry_point = np.asarray(
        manufactured["symmetry_target"],
        dtype=float,
    )
    symmetry_velocity = finest.induced_velocity(
        symmetry_point[None, :],
        quadrature_order=quadrature["off_sheet_orders"][-1],
    )[0]
    symmetry_fraction = float(
        np.linalg.norm(symmetry_velocity[:2])
        / max(np.linalg.norm(symmetry_velocity), 1.0e-15)
    )

    far_points = np.array(
        [
            [0.5, 0.5, float(z)]
            for z in manufactured["far_field_z"]
        ]
    )
    far_velocity = finest.induced_velocity(
        far_points,
        quadrature_order=quadrature["off_sheet_orders"][-1],
    )
    far_norm = np.linalg.norm(far_velocity, axis=1)
    distance_ratio = (
        manufactured["far_field_z"][1]
        / manufactured["far_field_z"][0]
    )
    decay_exponent = float(
        np.log(far_norm[0] / far_norm[1])
        / np.log(distance_ratio)
    )

    transform = np.array(
        [[1.1, 0.2, 0.0], [-0.1, 0.9, 0.15], [0.03, 0.0, 1.05]]
    )
    moved = finest.material_update(
        finest.vertices @ transform.T + np.array([0.2, -0.1, 0.3])
    )
    kelvin = finest.kelvin_report(
        moved,
        tolerance=guards["material_mu_residual_max"],
    )

    off_cauchy = cauchy_metrics(off_sheet_values)
    on_cauchy = cauchy_metrics(on_sheet_values)
    topology_passed = all(
        row["continuity"]["compatible"]
        and row["boundary"]["compatible"]
        and row["off_sheet_quadrature"]["converged"]
        and row["on_sheet_quadrature"]["converged"]
        for row in topology
    )
    symmetry_passed = (
        symmetry_fraction
        <= guards["symmetry_tangential_fraction_max"]
    )
    far_passed = (
        guards["far_field_decay_exponent_min"]
        <= decay_exponent
        <= guards["far_field_decay_exponent_max"]
    )
    off_cauchy_passed = (
        off_cauchy["monotone"]
        and off_cauchy["cauchy_ratio"]
        >= guards["off_sheet_cauchy_ratio_min"]
        and off_cauchy["finest_relative_change"]
        <= guards["off_sheet_finest_relative_change_max"]
    )
    on_cauchy_passed = (
        (
            not guards["on_sheet_monotone_cauchy"]
            or on_cauchy["monotone"]
        )
        and on_cauchy["finest_relative_change"]
        <= guards["on_sheet_finest_relative_change_max"]
    )
    constant_ring = constant_ring_gate(
        guards["constant_ring_relative_error_max"]
    )
    checks = {
        "constant_ring": constant_ring["passed"],
        "topology_and_quadrature": topology_passed,
        "symmetry": symmetry_passed,
        "far_field": far_passed,
        "off_sheet_mesh_cauchy": off_cauchy_passed,
        "on_sheet_mesh_cauchy": on_cauchy_passed,
        "material_kelvin": kelvin.passed,
    }
    return {
        "spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": spec["role"],
        "constant_ring": constant_ring,
        "topology": topology,
        "off_sheet_values": [
            value.tolist() for value in off_sheet_values
        ],
        "on_sheet_values": [
            value.tolist() for value in on_sheet_values
        ],
        "off_sheet_cauchy": off_cauchy,
        "on_sheet_cauchy": on_cauchy,
        "symmetry": {
            "velocity": symmetry_velocity.tolist(),
            "tangential_fraction": symmetry_fraction,
            "passed": symmetry_passed,
        },
        "far_field": {
            "velocity_norm": far_norm.tolist(),
            "decay_exponent": decay_exponent,
            "passed": far_passed,
        },
        "material_kelvin": asdict(kelvin),
        "checks": checks,
        "all_pass": all(checks.values()),
        "promotion": spec["promotion_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write the full record to {RESULT_PATH}",
    )
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
