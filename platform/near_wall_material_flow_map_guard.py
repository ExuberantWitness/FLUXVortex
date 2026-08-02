#!/usr/bin/env python3
"""Run preregistered N2.6c1b2a material flow-map identity guards."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PLATFORM = Path(__file__).resolve().parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.material_spike_geometry import (  # noqa: E402
    material_curvature_change,
)
from claim_runtime.near_wall_material_flow_map import (  # noqa: E402
    NearWallFlowMapError,
    integrate_near_wall_material_flow_map,
)
from tests.test_material_spike_geometry import plane  # noqa: E402
from tests.test_near_wall_material_flow_map import (  # noqa: E402
    rotation_z,
)


def _max_norm(value) -> float:
    return float(np.max(np.linalg.norm(value, axis=-1)))


def main() -> int:
    u, v, flat = plane(nu=31, nv=15)
    translation_velocity = np.array([0.4, -0.2, 0.1])
    translation = integrate_near_wall_material_flow_map(
        flat,
        initial_time=0.2,
        final_time=0.7,
        maximum_step=0.07,
        velocity_field=lambda points, time: np.broadcast_to(
            translation_velocity,
            points.shape,
        ),
    )
    translation_error = _max_norm(
        translation.final_position
        -(flat+0.5*translation_velocity)
    )

    angular_speed = 0.9
    duration = 0.8

    def rotation_velocity(points, time):
        del time
        result = np.empty_like(points)
        result[..., 0] = -angular_speed*points[..., 1]
        result[..., 1] = angular_speed*points[..., 0]
        result[..., 2] = 0.0
        return result

    exact_rotation = flat@rotation_z(angular_speed*duration).T
    rotation_errors = []
    for steps in (2, 4, 8):
        trajectory = integrate_near_wall_material_flow_map(
            flat,
            initial_time=0.0,
            final_time=duration,
            maximum_step=duration/steps,
            velocity_field=rotation_velocity,
        )
        rotation_errors.append(
            _max_norm(trajectory.final_position-exact_rotation)
        )

    observer_rate = 0.8
    observer_final_time = 0.6

    def observer_translation(time):
        return np.array([
            0.1*time*time,
            -0.2*np.sin(time),
            0.05*time,
        ])

    def observer_translation_rate(time):
        return np.array([
            0.2*time,
            -0.2*np.cos(time),
            0.05,
        ])

    observer_spin = np.array([
        [0.0, -observer_rate, 0.0],
        [observer_rate, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    def observer_velocity(points, time):
        offset = observer_translation(time)
        return (
            (points-offset)@observer_spin.T
            +observer_translation_rate(time)
        )

    observer = integrate_near_wall_material_flow_map(
        flat+observer_translation(0.0),
        initial_time=0.0,
        final_time=observer_final_time,
        maximum_step=0.005,
        velocity_field=observer_velocity,
    )
    observer_exact = (
        flat@rotation_z(observer_rate*observer_final_time).T
        +observer_translation(observer_final_time)
    )
    observer_error = _max_norm(
        observer.final_position-observer_exact
    )

    heights = (0.0, 0.01, 0.025)
    initial_layers = np.stack([
        flat+np.array([0.0, 0.0, height])
        for height in heights
    ])
    fold_rate = 1.7
    fold_final_time = 0.4

    def fold_velocity(points, time):
        del time
        result = np.zeros_like(points)
        result[..., 2] = (
            fold_rate*points[..., 0]**2*points[..., 2]
        )
        return result

    fold = integrate_near_wall_material_flow_map(
        initial_layers,
        initial_time=0.0,
        final_time=fold_final_time,
        maximum_step=0.005,
        velocity_field=fold_velocity,
    )
    fold_exact = initial_layers.copy()
    fold_exact[..., 2] = (
        initial_layers[..., 2]
        *np.exp(
            fold_rate
            *initial_layers[..., 0]**2
            *fold_final_time
        )
    )
    fold_position_error = _max_norm(
        fold.final_position-fold_exact
    )
    wall_motion_error = _max_norm(
        fold.final_position[0]-initial_layers[0]
    )
    numerical_change = material_curvature_change(
        initial_layers[2],
        fold.final_position[2],
        u=u,
        v=v,
        initial_time=0.0,
        final_time=fold_final_time,
    )
    exact_change = material_curvature_change(
        initial_layers[2],
        fold_exact[2],
        u=u,
        v=v,
        initial_time=0.0,
        final_time=fold_final_time,
    )
    fold_curvature_error = float(np.max(np.abs(
        numerical_change.largest_principal_curvature_change
        -exact_change.largest_principal_curvature_change
    )))

    malformed_velocity_rejected = False
    try:
        integrate_near_wall_material_flow_map(
            flat,
            initial_time=0.0,
            final_time=0.1,
            maximum_step=0.01,
            velocity_field=lambda points, time: np.zeros((2, 3)),
        )
    except NearWallFlowMapError:
        malformed_velocity_rejected = True

    passed = bool(
        translation_error <= 5.0e-16
        and rotation_errors[0]/rotation_errors[1] >= 14.0
        and rotation_errors[1]/rotation_errors[2] >= 14.0
        and observer_error <= 2.0e-11
        and fold_position_error <= 3.0e-13
        and wall_motion_error == 0.0
        and fold_curvature_error <= 3.0e-11
        and malformed_velocity_rejected
    )
    result = {
        "version": 1,
        "scope": (
            "N2.6c1b2a numerical material flow-map identity; "
            "not physical separation"
        ),
        "preregistered_cases": (
            "platform/docs/diag/near_wall_material_flow_map_cases.yaml"
        ),
        "uniform_translation": {
            "max_position_error": translation_error,
            "material_shape_preserved": (
                translation.initial_shape == flat.shape
            ),
        },
        "rigid_rotation": {
            "errors": rotation_errors,
            "coarse_to_medium_ratio": (
                rotation_errors[0]/rotation_errors[1]
            ),
            "medium_to_fine_ratio": (
                rotation_errors[1]/rotation_errors[2]
            ),
        },
        "time_dependent_observer": {
            "max_position_error": observer_error,
        },
        "no_slip_fold": {
            "max_position_error": fold_position_error,
            "wall_motion_error": wall_motion_error,
            "largest_curvature_change_error": fold_curvature_error,
        },
        "malformed_velocity_rejected": malformed_velocity_rejected,
        "physical_promotion": {
            "eligible": False,
            "reason": (
                "analytic fields validate trajectory integration only; "
                "representative independent near-wall field data and "
                "interpolation remain absent"
            ),
        },
        "passed": passed,
    }
    output = (
        PLATFORM/"docs"/"diag"/"near_wall_material_flow_map_results.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
