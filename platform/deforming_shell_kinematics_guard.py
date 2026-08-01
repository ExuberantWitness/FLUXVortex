"""Run preregistered N2.6d2a deforming-shell geometry gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.deforming_shell_kinematics import (
    DeformingShellError,
    deforming_dual_surface_kinematics,
    dual_surface_snapshot,
    rigidly_transform_surface_geometry,
    structured_surface_geometry,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "deforming_shell_kinematics_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "deforming_shell_kinematics_results.json"


def _patch():
    xi = np.linspace(0.0, 1.0, 17)
    eta = np.linspace(-0.6, 0.6, 15)
    x, y = np.meshgrid(xi, eta, indexing="ij")
    points = np.stack(
        (x, y, 0.08*x*x+0.03*x*y-0.02*y*y),
        axis=2,
    )
    return xi, eta, points


def run(*, write: bool = False) -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())
    xi, eta, points = _patch()
    geometry = structured_surface_geometry(points, xi=xi, eta=eta)
    thickness = 0.01+0.002*np.outer(xi, np.ones_like(eta))
    shell = dual_surface_snapshot(geometry, half_thickness=thickness)
    mean_error = float(np.max(np.abs(
        0.5*(shell.upper_surface+shell.lower_surface)-points
    )))
    thickness_error = float(np.max(np.abs(
        0.5*np.linalg.norm(
            shell.upper_surface-shell.lower_surface,
            axis=2,
        )-thickness
    )))
    director_unit_error = float(np.max(np.abs(
        np.linalg.norm(shell.director, axis=2)-1.0
    )))

    axis = np.array([0.4, -0.7, 1.1])
    axis /= np.linalg.norm(axis)
    angle = 0.83
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    rotation = (
        np.eye(3)*np.cos(angle)
        +(1.0-np.cos(angle))*np.outer(axis, axis)
        +np.sin(angle)*cross
    )
    moved = rigidly_transform_surface_geometry(
        geometry,
        rotation=rotation,
        translation=[0.3, -0.2, 0.7],
        xi=xi,
        eta=eta,
    )
    metric_error = float(np.max(np.abs(
        moved.first_fundamental_form-geometry.first_fundamental_form
    )))
    area_error = float(np.max(np.abs(
        moved.area_jacobian-geometry.area_jacobian
    )))
    curvature_error = float(max(
        np.max(np.abs(moved.mean_curvature-geometry.mean_curvature)),
        np.max(np.abs(
            moved.gaussian_curvature-geometry.gaussian_curvature
        )),
    ))
    director_rotation_error = float(np.max(np.abs(
        moved.director
        - np.einsum("ij,...j->...i", rotation, geometry.director)
    )))

    dt = 0.125
    translation_velocity = np.array([0.5, -1.0, 0.25])
    translated = deforming_dual_surface_kinematics(
        previous_mean_surface=points-dt*translation_velocity,
        current_mean_surface=points,
        next_mean_surface=points+dt*translation_velocity,
        xi=xi,
        eta=eta,
        half_thickness=np.full(points.shape[:2], 0.012),
        dt=dt,
    )
    expected_velocity = np.broadcast_to(
        translation_velocity,
        points.shape,
    )
    velocity_error = float(max(
        np.max(np.abs(translated.mean_velocity-expected_velocity)),
        np.max(np.abs(translated.upper_velocity-expected_velocity)),
        np.max(np.abs(translated.lower_velocity-expected_velocity)),
    ))
    acceleration_error = float(max(
        np.max(np.abs(translated.mean_acceleration)),
        np.max(np.abs(translated.upper_acceleration)),
        np.max(np.abs(translated.lower_acceleration)),
    ))

    radius = 0.8
    theta_coordinate = np.linspace(-0.55, 0.55, 101)
    span_coordinate = np.linspace(-0.4, 0.4, 17)
    theta, span = np.meshgrid(
        theta_coordinate,
        span_coordinate,
        indexing="ij",
    )
    cylinder = np.stack(
        (
            radius*np.sin(theta),
            span,
            radius*(1.0-np.cos(theta)),
        ),
        axis=2,
    )
    cylinder_geometry = structured_surface_geometry(
        cylinder,
        xi=theta_coordinate,
        eta=span_coordinate,
    )
    interior = np.s_[2:-2, 2:-2]
    cylinder_area_error = float(np.max(np.abs(
        cylinder_geometry.area_jacobian[interior]-radius
    )))
    cylinder_mean_error = float(np.max(np.abs(
        np.abs(cylinder_geometry.mean_curvature[interior])
        - 0.5/radius
    )))
    cylinder_gaussian_error = float(np.max(np.abs(
        cylinder_geometry.gaussian_curvature[interior]
    )))

    collapsed = points.copy()
    collapsed[8, :, :] = collapsed[7, :, :]
    collapsed[9, :, :] = collapsed[7, :, :]
    raised = False
    try:
        structured_surface_geometry(collapsed, xi=xi, eta=eta)
    except DeformingShellError:
        raised = True

    metrics = {
        "dual_side_material_pairing": {
            "mean_reconstruction_error": mean_error,
            "thickness_reconstruction_error": thickness_error,
            "director_unit_error": director_unit_error,
        },
        "proper_rigid_transform": {
            "metric_invariance_error": metric_error,
            "area_invariance_error": area_error,
            "curvature_invariance_error": curvature_error,
            "director_rotation_error": director_rotation_error,
        },
        "constant_translation": {
            "velocity_error": velocity_error,
            "acceleration_error": acceleration_error,
        },
        "cylindrical_bending": {
            "interior_area_jacobian_error": cylinder_area_error,
            "interior_mean_curvature_magnitude_error": cylinder_mean_error,
            "interior_gaussian_curvature_error": cylinder_gaussian_error,
        },
        "degenerate_or_inverted_surface": {
            "must_raise": raised,
        },
    }
    thresholds = {
        case["id"]: case["gates"]
        for case in prereg["cases"]
    }
    passed = {}
    for case_id, gates in thresholds.items():
        passed[case_id] = {}
        for metric, threshold in gates.items():
            value = metrics[case_id][metric]
            if isinstance(threshold, bool):
                passed[case_id][metric] = value is threshold
            else:
                passed[case_id][metric] = value <= float(threshold)
    result = {
        "artifact": prereg["artifact"],
        "claim_node": prereg["claim_node"],
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": passed,
        "all_pass": all(
            value
            for case in passed.values()
            for value in case.values()
        ),
        "scope_limit": prereg["promotion"]["must_not_claim"],
        "interpretation": (
            "The structured material surface supplies objective dual-side "
            "geometry, curvature and finite-difference kinematics. This does "
            "not validate a structural map, shell dynamics or aerodynamic load."
        ),
    }
    if write:
        RESULT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2)+"\n"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run(write=arguments.write), ensure_ascii=False, indent=2))

