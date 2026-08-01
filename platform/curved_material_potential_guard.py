#!/usr/bin/env python3
"""Run the preregistered N3.1j3b3 curved-potential/history gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.distributed_doublet import QuadraticDoubletSurface
from claim_runtime.doublet_potential import (
    surface_doublet_potential,
    surface_sheet_average_potential,
)
from claim_runtime.material_potential_history import (
    material_potential_history_rate,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "curved_material_potential_cases.yaml"
RESULT_PATH = (
    ROOT / "docs" / "diag" / "curved_material_potential_results.json"
)


def _octasphere(level: int, shift=None):
    vertices = [
        np.asarray(point, dtype=float)
        for point in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
    ]
    faces = [
        (4, 0, 2),
        (4, 2, 1),
        (4, 1, 3),
        (4, 3, 0),
        (5, 2, 0),
        (5, 1, 2),
        (5, 3, 1),
        (5, 0, 3),
    ]
    roots = list(range(8))
    for _ in range(level):
        midpoint_cache = {}
        refined_faces = []
        refined_roots = []

        def midpoint(first: int, second: int) -> int:
            key = tuple(sorted((first, second)))
            if key not in midpoint_cache:
                point = vertices[first] + vertices[second]
                point /= np.linalg.norm(point)
                midpoint_cache[key] = len(vertices)
                vertices.append(point)
            return midpoint_cache[key]

        for face, root in zip(faces, roots):
            first, second, third = face
            edge01 = midpoint(first, second)
            edge12 = midpoint(second, third)
            edge20 = midpoint(third, first)
            refined_faces.extend(
                (
                    (first, edge01, edge20),
                    (edge01, second, edge12),
                    (edge20, edge12, third),
                    (edge01, edge12, edge20),
                )
            )
            refined_roots.extend((root, root, root, root))
        faces = refined_faces
        roots = refined_roots

    base_vertices = np.asarray(vertices, dtype=float)
    face_array = np.asarray(faces, dtype=np.int64)
    for face_index, face in enumerate(face_array):
        triangle = base_vertices[face]
        area_vector = np.cross(
            triangle[1] - triangle[0],
            triangle[2] - triangle[0],
        )
        if np.dot(area_vector, np.mean(triangle, axis=0)) < 0.0:
            face_array[face_index, [1, 2]] = face_array[
                face_index, [2, 1]
            ]
    face_mu = []
    for face in face_array:
        triangle = base_vertices[face]
        material_nodes = np.vstack(
            (
                triangle,
                0.5 * (triangle[0] + triangle[1]),
                0.5 * (triangle[1] + triangle[2]),
                0.5 * (triangle[2] + triangle[0]),
            )
        )
        face_mu.append(material_nodes[:, 2])
    translation = (
        np.zeros(3, dtype=float)
        if shift is None
        else np.asarray(shift, dtype=float)
    )
    surface = QuadraticDoubletSurface(
        base_vertices + translation,
        face_array,
        np.asarray(face_mu),
    )
    return surface, np.asarray(roots, dtype=np.int64)


def _patches(surface: QuadraticDoubletSurface, roots):
    patches = []
    global_face_groups = []
    for root in sorted(np.unique(roots)):
        global_faces = np.flatnonzero(roots == root)
        face_vertices = surface.faces[global_faces]
        used_vertices = np.unique(face_vertices)
        local_faces = np.searchsorted(used_vertices, face_vertices)
        patches.append(
            QuadraticDoubletSurface(
                surface.vertices[used_vertices],
                local_faces,
                surface.face_mu[global_faces],
            )
        )
        global_face_groups.append(global_faces)
    return patches, global_face_groups


def _strictly_decreasing(values) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(array) < 0.0))


def run() -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text(encoding="utf-8"))
    order = int(prereg["spatial_case"]["quadrature_order"])
    points = np.asarray(
        prereg["spatial_case"]["off_surface_points"],
        dtype=float,
    )
    expected_off_surface = np.asarray(
        [
            (2.0 / 3.0) * 0.6,
            -1.0 / (3.0 * 1.4**2),
            -(2.0 / 3.0) * 0.6,
            1.0 / (3.0 * 1.4**2),
        ]
    )
    off_surface_errors = []
    sheet_pv_errors = []
    spatial_records = []
    surfaces = {}
    roots_by_level = {}
    for level in (0, 1, 2, 3):
        surface, roots = _octasphere(level)
        surfaces[level] = surface
        roots_by_level[level] = roots
        potential = surface_doublet_potential(
            surface,
            points,
            quadrature_order=order,
        )
        off_error = float(
            np.max(np.abs(potential - expected_off_surface))
        )
        owners = np.arange(len(surface.faces), dtype=np.int64)
        barycentric = np.tile(
            np.asarray([1.0 / 3.0] * 3),
            (len(owners), 1),
        )
        sheet_points = np.asarray(
            [
                barycentric[index]
                @ surface.vertices[surface.faces[index]]
                for index in owners
            ]
        )
        sheet_direction = sheet_points / np.linalg.norm(
            sheet_points, axis=1
        )[:, None]
        expected_pv = sheet_direction[:, 2] / 6.0
        principal_value = surface_sheet_average_potential(
            surface,
            owners,
            barycentric,
            quadrature_order=order,
        )
        pv_error = float(
            np.max(np.abs(principal_value - expected_pv))
        )
        off_surface_errors.append(off_error)
        sheet_pv_errors.append(pv_error)
        spatial_records.append(
            {
                "level": level,
                "faces": int(len(surface.faces)),
                "max_off_surface_abs_error": off_error,
                "max_sheet_pv_abs_error": pv_error,
            }
        )

    patch_surface = surfaces[2]
    patch_roots = roots_by_level[2]
    owners = np.arange(len(patch_surface.faces), dtype=np.int64)
    barycentric = np.tile(
        np.asarray([1.0 / 3.0] * 3),
        (len(owners), 1),
    )
    sheet_points = np.asarray(
        [
            barycentric[index]
            @ patch_surface.vertices[patch_surface.faces[index]]
            for index in owners
        ]
    )
    monolithic_pv = surface_sheet_average_potential(
        patch_surface,
        owners,
        barycentric,
        quadrature_order=order,
    )
    assembled_pv = np.zeros_like(monolithic_pv)
    patch_list, face_groups = _patches(patch_surface, patch_roots)
    for patch, global_faces in zip(patch_list, face_groups):
        local_owner = np.arange(len(global_faces), dtype=np.int64)
        assembled_pv[global_faces] += surface_sheet_average_potential(
            patch,
            local_owner,
            barycentric[global_faces],
            quadrature_order=order,
        )
        other_faces = np.setdiff1d(
            owners,
            global_faces,
            assume_unique=True,
        )
        assembled_pv[other_faces] += surface_doublet_potential(
            patch,
            sheet_points[other_faces],
            quadrature_order=order,
        )
    patch_residual = float(
        np.max(np.abs(assembled_pv - monolithic_pv), initial=0.0)
    )

    source_speed = float(prereg["temporal_case"]["source_speed"])
    wall_point = np.asarray(
        prereg["temporal_case"]["fixed_wall_point"],
        dtype=float,
    )[None, :]
    time_errors = []
    time_records = []
    material_mu_residual = 0.0
    for half_window in prereg["temporal_case"]["half_windows"]:
        dt = float(half_window)
        times = np.asarray([-dt, 0.0, dt])
        history_surfaces = [
            _octasphere(
                2,
                shift=np.asarray([source_speed * time, 0.0, 0.0]),
            )[0]
            for time in times
        ]
        wall_history = np.repeat(wall_point[None, :, :], 3, axis=0)
        rate = material_potential_history_rate(
            history_surfaces,
            wall_history,
            times,
            target_index=2,
            quadrature_order=order,
        )
        current_velocity = history_surfaces[2].induced_velocity(
            wall_point,
            quadrature_order=order,
        )
        reference_rate = -source_speed * current_velocity[:, 0]
        error = float(
            np.max(
                np.abs(rate.wall_material_rate - reference_rate),
                initial=0.0,
            )
        )
        time_errors.append(error)
        material_mu_residual = max(
            material_mu_residual,
            rate.max_material_mu_residual,
        )
        time_records.append(
            {
                "half_window": dt,
                "computed_rate": float(rate.wall_material_rate[0]),
                "translation_identity_rate": float(reference_rate[0]),
                "absolute_error": error,
                "derivative_weights": rate.derivative_weights.tolist(),
            }
        )

    off_last_ratio = off_surface_errors[-2] / off_surface_errors[-1]
    pv_last_ratio = sheet_pv_errors[-2] / sheet_pv_errors[-1]
    time_ratios = (
        np.asarray(time_errors[:-1]) / np.asarray(time_errors[1:])
    )
    thresholds = prereg["thresholds"]
    checks = {
        "off_surface_error_monotone": _strictly_decreasing(
            off_surface_errors
        ),
        "off_surface_last_cauchy_ratio": (
            off_last_ratio
            >= float(thresholds["min_off_surface_last_cauchy_ratio"])
        ),
        "off_surface_finest_abs_error": (
            off_surface_errors[-1]
            <= float(thresholds["max_off_surface_finest_abs_error"])
        ),
        "sheet_pv_error_monotone": _strictly_decreasing(
            sheet_pv_errors
        ),
        "sheet_pv_last_cauchy_ratio": (
            pv_last_ratio
            >= float(thresholds["min_sheet_pv_last_cauchy_ratio"])
        ),
        "sheet_pv_finest_abs_error": (
            sheet_pv_errors[-1]
            <= float(thresholds["max_sheet_pv_finest_abs_error"])
        ),
        "patch_representation_invariance": (
            patch_residual
            <= float(thresholds["max_patch_representation_residual"])
        ),
        "time_second_order": (
            float(np.min(time_ratios))
            >= float(thresholds["min_time_error_ratio"])
        ),
        "finest_time_rate_abs_error": (
            time_errors[-1]
            <= float(thresholds["max_finest_time_rate_abs_error"])
        ),
        "material_mu_identity": (
            material_mu_residual
            <= float(thresholds["max_material_mu_residual"])
        ),
    }
    result = {
        "claim": prereg["claim"],
        "spatial_records": spatial_records,
        "off_surface_last_cauchy_ratio": off_last_ratio,
        "sheet_pv_last_cauchy_ratio": pv_last_ratio,
        "patch_representation_residual": patch_residual,
        "time_records": time_records,
        "time_error_ratios": time_ratios.tolist(),
        "max_material_mu_residual": material_mu_residual,
        "checks": checks,
        "all_pass": all(checks.values()),
        "physical_promotion": False,
        "scope_limit": prereg["scope_limit"],
        "claim_effect": {
            "N3.1j3b3": (
                "eligible_for_narrow_validation"
                if all(checks.values())
                else "remains_open"
            ),
            "N3.1j3": "remains_open",
            "N3.1j4b5b": "remains_open",
            "N2.6c": "remains_open",
            "production_force": "blocked",
        },
    }
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if run()["all_pass"] else 1)
