#!/usr/bin/env python3
"""Run preregistered N2.6c1b material-surface geometry identity guards."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PLATFORM = Path(__file__).resolve().parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.material_spike_geometry import (  # noqa: E402
    MaterialSpikeGeometryError,
    material_curvature_change,
    material_surface_curvature,
    proper_euclidean_observer_transform,
)
from tests.test_material_spike_geometry import (  # noqa: E402
    cylinder,
    plane,
    rotation_matrix,
)


def _maximum(value) -> float:
    return float(np.max(np.abs(value)))


def main() -> int:
    u_plane, v_plane, flat = plane(nu=41, nv=21)
    flat_geometry = material_surface_curvature(
        flat,
        u=u_plane,
        v=v_plane,
    )
    plane_metric_error = _maximum(
        flat_geometry.first_fundamental_form-np.eye(2)
    )
    plane_curvature_error = _maximum(
        flat_geometry.principal_curvatures
    )

    cylinder_errors: list[float] = []
    for count in (31, 121):
        u_cylinder, v_cylinder, cylinder_points = cylinder(0.8, count)
        cylinder_geometry = material_surface_curvature(
            cylinder_points,
            u=u_cylinder,
            v=v_cylinder,
        )
        cylinder_errors.append(_maximum(
            cylinder_geometry.principal_curvatures
            -np.array([0.0, 1.25])
        ))

    static_change = material_curvature_change(
        cylinder_points,
        cylinder_points.copy(),
        u=u_cylinder,
        v=v_cylinder,
        initial_time=0.0,
        final_time=0.25,
    )
    static_change_error = _maximum(static_change.weingarten_change)

    uu, _ = np.meshgrid(u_plane, v_plane, indexing="ij")
    fold_strength = 2.4
    folded = flat.copy()
    folded[..., 2] = 0.5*fold_strength*uu*uu
    fold_change = material_curvature_change(
        flat,
        folded,
        u=u_plane,
        v=v_plane,
        initial_time=0.0,
        final_time=0.1,
    )
    centre = (len(u_plane)//2, len(v_plane)//2)
    fold_origin_error = _maximum(
        fold_change.principal_curvature_changes[centre]
        -np.array([0.0, fold_strength])
    )

    moved_initial = proper_euclidean_observer_transform(
        flat,
        rotation=rotation_matrix([0.3, -0.2, 0.9], 0.51),
        translation=[0.4, -0.7, 0.2],
    )
    moved_final = proper_euclidean_observer_transform(
        folded,
        rotation=rotation_matrix([-0.4, 0.8, 0.1], -0.73),
        translation=[-0.3, 0.1, 0.9],
    )
    moved_change = material_curvature_change(
        moved_initial,
        moved_final,
        u=u_plane,
        v=v_plane,
        initial_time=0.0,
        final_time=0.1,
    )
    observer_residual = max(
        _maximum(
            moved_change.principal_curvature_changes
            -fold_change.principal_curvature_changes
        ),
        _maximum(
            moved_change.mean_curvature_change
            -fold_change.mean_curvature_change
        ),
        _maximum(
            moved_change.gaussian_curvature_change
            -fold_change.gaussian_curvature_change
        ),
    )

    baseline_cylinder = material_surface_curvature(
        cylinder_points,
        u=u_cylinder,
        v=v_cylinder,
    )
    reparameterized = material_surface_curvature(
        cylinder_points,
        u=3.7*u_cylinder+0.2,
        v=0.6*v_cylinder-0.4,
    )
    reparameterization_residual = _maximum(
        reparameterized.principal_curvatures[2:-2, 2:-2]
        -baseline_cylinder.principal_curvatures[2:-2, 2:-2]
    )

    improper_observer_rejected = False
    try:
        proper_euclidean_observer_transform(
            flat,
            rotation=np.diag([1.0, 1.0, -1.0]),
            translation=[0.0, 0.0, 0.0],
        )
    except MaterialSpikeGeometryError:
        improper_observer_rejected = True

    passed = bool(
        plane_metric_error <= 3.0e-14
        and plane_curvature_error <= 3.0e-14
        and cylinder_errors[1] < 0.08*cylinder_errors[0]
        and cylinder_errors[1] <= 2.0e-4
        and static_change_error == 0.0
        and fold_origin_error <= 3.0e-12
        and observer_residual <= 3.0e-12
        and reparameterization_residual <= 3.0e-11
        and improper_observer_rejected
    )
    result = {
        "version": 1,
        "scope": (
            "N2.6c1b material-surface Weingarten identity; "
            "not physical separation"
        ),
        "preregistered_cases": (
            "platform/docs/diag/material_spike_geometry_cases.yaml"
        ),
        "plane": {
            "max_metric_error": plane_metric_error,
            "max_curvature_error": plane_curvature_error,
        },
        "cylinder": {
            "coarse_max_curvature_error": cylinder_errors[0],
            "fine_max_curvature_error": cylinder_errors[1],
            "fine_to_coarse_ratio": cylinder_errors[1]/cylinder_errors[0],
        },
        "static_surface": {
            "max_weingarten_change": static_change_error,
        },
        "parabolic_fold": {
            "origin_principal_change_error": fold_origin_error,
            "origin_largest_change": float(
                fold_change.largest_principal_curvature_change[centre]
            ),
        },
        "objectivity": {
            "time_dependent_observer_max_residual": observer_residual,
            "improper_observer_rejected": improper_observer_rejected,
        },
        "reparametrization": {
            "interior_principal_curvature_residual": (
                reparameterization_residual
            ),
        },
        "physical_promotion": {
            "eligible": False,
            "reason": (
                "manufactured geometry validates Eq.3.3/4.1 identities only; "
                "no independent velocity field, flow map, or ridge validation"
            ),
        },
        "passed": passed,
    }
    output = (
        PLATFORM/"docs"/"diag"/"material_spike_geometry_results.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
