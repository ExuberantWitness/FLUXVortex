"""Run preregistered N2.6b4 -> N2.6c2c profile sufficiency gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.near_wall_profile_collapse import (
    NearWallProfile,
    NearWallProfileError,
    project_near_wall_profile,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "ibl_to_ves_profile_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "ibl_to_ves_profile_results.json"


def _profile(coordinate, velocity_x, *, edge):
    coordinate = np.asarray(coordinate, dtype=float)
    velocity = np.zeros((len(coordinate), 3))
    velocity[:, 0] = velocity_x
    return NearWallProfile(
        normal_coordinate=coordinate,
        density=np.ones(len(coordinate)),
        velocity=velocity,
        external_tangential_velocity=[1.0, 0.0, 0.0],
        outer_velocity_plus=[1.0, 0.0, -0.08],
        outer_velocity_minus=[0.0, 0.0, 0.0],
        sheet_normal=[0.0, 0.0, 1.0],
        edge_convention=edge,
    )


def _maximum_defect_difference(first, second) -> float:
    return float(max(
        np.max(np.abs(first.mass_flux_defect-second.mass_flux_defect)),
        np.max(np.abs(
            first.momentum_flux_defect-second.momentum_flux_defect
        )),
    ))


def run(*, write: bool = False) -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())

    coordinate = np.linspace(0.0, 1.0, 5001)
    linear = project_near_wall_profile(
        _profile(coordinate, coordinate, edge="analytic-linear-edge")
    )
    analytic_error = float(max(
        abs(linear.mass_flux_defect[0]-0.5),
        abs(linear.momentum_flux_trace-1.0/6.0),
        abs(linear.surface_mass_density-1.0),
        abs(linear.surface_momentum[0]-0.5),
    ))

    axis = np.array([0.3, -0.8, 0.4])
    axis /= np.linalg.norm(axis)
    angle = 0.73
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
    base_profile = _profile(coordinate, coordinate, edge="rotation-edge")
    moved_profile = NearWallProfile(
        normal_coordinate=coordinate,
        density=base_profile.density,
        velocity=base_profile.velocity@rotation.T,
        external_tangential_velocity=
            base_profile.external_tangential_velocity@rotation.T,
        outer_velocity_plus=base_profile.outer_velocity_plus@rotation.T,
        outer_velocity_minus=base_profile.outer_velocity_minus@rotation.T,
        sheet_normal=base_profile.sheet_normal@rotation.T,
        edge_convention=base_profile.edge_convention,
    )
    base = project_near_wall_profile(base_profile)
    moved = project_near_wall_profile(moved_profile)
    rotation_error = float(max(
        np.max(np.abs(
            moved.mass_flux_defect-base.mass_flux_defect@rotation.T
        )),
        np.max(np.abs(
            moved.momentum_flux_defect
            -rotation@base.momentum_flux_defect@rotation.T
        )),
        np.max(np.abs(
            moved.surface_momentum-base.surface_momentum@rotation.T
        )),
        np.max(np.abs(
            moved.vortex_sheet_strength
            -base.vortex_sheet_strength@rotation.T
        )),
        abs(moved.entrainment_strength-base.entrainment_strength),
    ))

    extension = np.linspace(1.0002, 1.2, 1000)
    extended_coordinate = np.concatenate((coordinate, extension))
    extended_velocity = np.concatenate((
        coordinate,
        np.ones(len(extension)),
    ))
    extended = project_near_wall_profile(_profile(
        extended_coordinate,
        extended_velocity,
        edge="uniform-outer-plateau-included",
    ))
    plateau_defect_error = _maximum_defect_difference(linear, extended)
    plateau_mass_difference = float(
        extended.surface_mass_density-linear.surface_mass_density
    )
    plateau_momentum_difference = float(
        extended.surface_momentum[0]-linear.surface_momentum[0]
    )

    delta_b = 15.0/14.0
    coordinate_b = np.linspace(0.0, delta_b, 24001)
    eta = coordinate_b/delta_b
    velocity_b = (
        eta
        +0.2*eta*(1.0-eta)
        -0.2819747927945442*eta*(1.0-eta)*(1.0-2.0*eta)
    )
    distinct = project_near_wall_profile(
        _profile(coordinate_b, velocity_b, edge="equal-defect-profile-b")
    )
    distinct_defect_error = _maximum_defect_difference(linear, distinct)
    distinct_mass_difference = float(
        distinct.surface_mass_density-linear.surface_mass_density
    )
    distinct_momentum_difference = float(
        distinct.surface_momentum[0]-linear.surface_momentum[0]
    )
    minimum_profile_slope = float(np.min(
        np.diff(velocity_b)/np.diff(coordinate_b)
    ))

    valid = dict(
        normal_coordinate=[0.0, 1.0],
        density=[1.0, 1.0],
        velocity=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        external_tangential_velocity=[1.0, 0.0, 0.0],
        outer_velocity_plus=[1.0, 0.0, 0.0],
        outer_velocity_minus=[0.0, 0.0, 0.0],
        sheet_normal=[0.0, 0.0, 1.0],
        edge_convention="explicit",
    )
    invalid_rejected = 0
    for replacement in (
        {"normal_coordinate": [0.0, 0.0]},
        {"density": [1.0, 0.0]},
        {"sheet_normal": [0.0, 0.0, 2.0]},
        {"external_tangential_velocity": [1.0, 0.0, 0.1]},
        {"edge_convention": ""},
    ):
        bad = dict(valid)
        bad.update(replacement)
        try:
            NearWallProfile(**bad)
        except NearWallProfileError:
            invalid_rejected += 1

    metrics = {
        "analytic_linear_2d": {
            "maximum_absolute_error": analytic_error,
        },
        "rigid_rotation": {
            "maximum_covariance_error": rotation_error,
        },
        "outer_plateau_gauge": {
            "ibl_defect_max_difference": plateau_defect_error,
            "surface_mass_difference": plateau_mass_difference,
            "surface_momentum_difference": plateau_momentum_difference,
        },
        "equal_defect_distinct_state": {
            "ibl_defect_max_difference": distinct_defect_error,
            "surface_mass_difference": distinct_mass_difference,
            "surface_momentum_difference": distinct_momentum_difference,
            "minimum_profile_slope": minimum_profile_slope,
        },
        "invalid_inputs": {
            "rejected_count": invalid_rejected,
            "expected_rejected_count": 5,
        },
    }
    passed = {
        "analytic_linear_2d":
            analytic_error <= 2.0e-7,
        "rigid_rotation":
            rotation_error <= 2.0e-12,
        "outer_plateau_gauge": (
            plateau_defect_error <= 2.0e-7
            and plateau_mass_difference >= 0.19
            and plateau_momentum_difference >= 0.19
        ),
        "equal_defect_distinct_state": (
            distinct_defect_error <= 2.0e-7
            and distinct_mass_difference >= 0.07
            and distinct_momentum_difference >= 0.07
            and minimum_profile_slope >= -1.0e-12
        ),
        "invalid_inputs": invalid_rejected == 5,
    }
    result = {
        "preregistered_case_file": str(CASE_PATH.relative_to(ROOT)),
        "claim": "N2.6b4 -> N2.6c2c profile/edge information sufficiency",
        "metrics": metrics,
        "passed": passed,
        "all_passed": all(passed.values()),
        "interpretation": (
            "Passing proves that finite IBL deficit moments do not uniquely "
            "determine VES actual mass/momentum and that explicit profile/edge "
            "state is a necessary interface. It does not close LEV release."
        ),
        "preregistered": bool(prereg["frozen_before_implementation"]),
    }
    if write:
        RESULT_PATH.write_text(json.dumps(result, indent=2)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run(write=args.write)
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
