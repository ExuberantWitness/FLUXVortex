#!/usr/bin/env python3
"""Run preregistered N2.6c2c1 moving-separation-flux identity gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.moving_separation_flux import (
    moving_separation_ibl_flux,
)
from claim_runtime.surface_ibl_state import (
    SurfaceIBLError,
    SurfaceIBLFields,
    rotate_surface_ibl_fields,
    surface_ibl_physical_flux,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "moving_separation_flux_cases.yaml"
RESULT_PATH = (
    ROOT / "docs" / "diag" / "moving_separation_flux_results.json"
)


def _fields(
    *,
    mass,
    tensor,
    energy_flux,
    external_velocity,
    normal,
) -> SurfaceIBLFields:
    mass = np.asarray(mass, dtype=float)
    count = len(mass)
    return SurfaceIBLFields(
        mass_flux_defect=mass,
        momentum_flux_defect=np.asarray(tensor, dtype=float),
        kinetic_energy_defect_flux=np.asarray(
            energy_flux,
            dtype=float,
        ),
        external_tangential_velocity=np.asarray(
            external_velocity,
            dtype=float,
        ),
        external_velocity_surface_gradient=np.zeros((count, 3, 3)),
        wall_shear_over_density=np.zeros((count, 3)),
        dissipation_integral=np.zeros(count),
        surface_normal=np.asarray(normal, dtype=float),
    )


def _planar_fields() -> SurfaceIBLFields:
    return _fields(
        mass=[[0.4, -0.2, 0.0]],
        tensor=[[
            [0.6, 0.15, 0.0],
            [-0.1, 0.35, 0.0],
            [0.0, 0.0, 0.0],
        ]],
        energy_flux=[[0.8, -0.3, 0.0]],
        external_velocity=[[1.7, -0.4, 0.0]],
        normal=[[0.0, 0.0, 1.0]],
    )


def _sweep_only_planar_fields() -> SurfaceIBLFields:
    return _fields(
        mass=[[0.4, -0.2, 0.0]],
        tensor=[[
            [0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.0, 0.0],
        ]],
        energy_flux=[[0.0, 0.0, 0.0]],
        external_velocity=[[0.0, 0.0, 0.0]],
        normal=[[0.0, 0.0, 1.0]],
    )


def _rotation_matrix() -> np.ndarray:
    axis = np.array([0.3, -0.8, 0.4])
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3)*np.cos(angle)
        +(1.0-np.cos(angle))*np.outer(axis, axis)
        +np.sin(angle)*cross
    )


def _maximum(value) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def run(*, write: bool = False) -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text(encoding="utf-8"))

    planar = _planar_fields()
    conormal = np.array([[1.0, 0.0, 0.0]])
    measure = np.array([1.3])
    stationary = moving_separation_ibl_flux(
        planar,
        outward_surface_conormal=conormal,
        relative_conormal_speed=0.0,
        edge_measure=measure,
    )
    physical = surface_ibl_physical_flux(
        planar,
        outward_surface_conormal=conormal,
        edge_measure=measure,
    )
    stationary_momentum_error = _maximum(
        stationary.relative_momentum_defect_out-physical.momentum_out
    )
    stationary_energy_error = _maximum(
        stationary.relative_energy_defect_out-physical.energy_out
    )

    sweep_fields = _sweep_only_planar_fields()
    line_measure = 1.4
    expanding_speed = 0.3
    expanding = moving_separation_ibl_flux(
        sweep_fields,
        outward_surface_conormal=conormal,
        relative_conormal_speed=expanding_speed,
        edge_measure=line_measure,
    )
    expanding_storage_momentum = (
        expanding_speed
        * line_measure
        * sweep_fields.mass_flux_defect[0]
    )
    expanding_storage_energy = (
        expanding_speed
        * line_measure
        * sweep_fields.momentum_flux_trace[0]
    )
    expanding_residual = max(
        _maximum(
            expanding_storage_momentum
            + expanding.relative_momentum_defect_out[0]
        ),
        abs(
            expanding_storage_energy
            + expanding.relative_energy_defect_out[0]
        ),
    )
    contracting_speed = -0.22
    contracting = moving_separation_ibl_flux(
        sweep_fields,
        outward_surface_conormal=conormal,
        relative_conormal_speed=contracting_speed,
        edge_measure=line_measure,
    )
    contracting_residual = max(
        _maximum(
            contracting_speed
            * line_measure
            * sweep_fields.mass_flux_defect[0]
            + contracting.relative_momentum_defect_out[0]
        ),
        abs(
            contracting_speed
            * line_measure
            * sweep_fields.momentum_flux_trace[0]
            + contracting.relative_energy_defect_out[0]
        ),
    )

    theta = 0.91
    theta_rate = 0.17
    count = 2048
    azimuth = 2.0*np.pi*np.arange(count)/count
    cosine = np.cos(azimuth)
    sine = np.sin(azimuth)
    normal = np.column_stack((
        np.sin(theta)*cosine,
        np.sin(theta)*sine,
        np.full(count, np.cos(theta)),
    ))
    sphere_conormal = np.column_stack((
        np.cos(theta)*cosine,
        np.cos(theta)*sine,
        np.full(count, -np.sin(theta)),
    ))
    azimuth_tangent = np.column_stack((
        -sine,
        cosine,
        np.zeros(count),
    ))
    mass = np.array([0.0, 0.0, 1.0])-np.cos(theta)*normal
    trace = 0.63
    tensor = (
        trace
        * np.einsum("ni,nj->nij", azimuth_tangent, azimuth_tangent)
    )
    sphere_fields = _fields(
        mass=mass,
        tensor=tensor,
        energy_flux=np.zeros((count, 3)),
        external_velocity=np.zeros((count, 3)),
        normal=normal,
    )
    quadrature_measure = np.full(
        count,
        2.0*np.pi*np.sin(theta)/count,
    )
    sphere_flux = moving_separation_ibl_flux(
        sphere_fields,
        outward_surface_conormal=sphere_conormal,
        relative_conormal_speed=theta_rate,
        edge_measure=quadrature_measure,
    )
    analytic_momentum_storage_rate = np.array([
        0.0,
        0.0,
        2.0*np.pi*theta_rate*np.sin(theta)**3,
    ])
    analytic_energy_storage_rate = (
        trace*2.0*np.pi*np.sin(theta)*theta_rate
    )
    sphere_momentum_residual = _maximum(
        analytic_momentum_storage_rate
        + np.sum(sphere_flux.relative_momentum_defect_out, axis=0)
    )
    sphere_energy_residual = abs(
        analytic_energy_storage_rate
        + float(np.sum(sphere_flux.relative_energy_defect_out))
    )

    combined_speed = -0.28
    combined = moving_separation_ibl_flux(
        planar,
        outward_surface_conormal=conormal,
        relative_conormal_speed=combined_speed,
        edge_measure=measure,
    )
    expected_sweep_momentum = (
        -combined_speed
        * planar.mass_flux_defect
        * measure[:, None]
    )
    expected_sweep_energy = (
        -combined_speed
        * planar.momentum_flux_trace
        * measure
    )
    superposition_momentum_error = _maximum(
        combined.relative_momentum_defect_out
        - physical.momentum_out
        - expected_sweep_momentum
    )
    superposition_energy_error = _maximum(
        combined.relative_energy_defect_out
        - physical.energy_out
        - expected_sweep_energy
    )

    rotation = _rotation_matrix()
    rotated_fields = rotate_surface_ibl_fields(planar, rotation)
    rotated_conormal = conormal@rotation.T
    rotated_flux = moving_separation_ibl_flux(
        rotated_fields,
        outward_surface_conormal=rotated_conormal,
        relative_conormal_speed=combined_speed,
        edge_measure=measure,
    )
    momentum_rotation_error = _maximum(
        rotated_flux.relative_momentum_defect_out
        - combined.relative_momentum_defect_out@rotation.T
    )
    energy_invariance_error = _maximum(
        rotated_flux.relative_energy_defect_out
        - combined.relative_energy_defect_out
    )

    reversed_flux = moving_separation_ibl_flux(
        planar,
        outward_surface_conormal=-conormal,
        relative_conormal_speed=-combined_speed,
        edge_measure=measure,
    )
    momentum_antisymmetry_error = _maximum(
        reversed_flux.relative_momentum_defect_out
        + combined.relative_momentum_defect_out
    )
    energy_antisymmetry_error = _maximum(
        reversed_flux.relative_energy_defect_out
        + combined.relative_energy_defect_out
    )

    invalid_rejected = 0
    invalid_cases = (
        dict(
            outward_surface_conormal=[[1.0, 0.0, 0.1]],
            relative_conormal_speed=0.0,
            edge_measure=1.0,
        ),
        dict(
            outward_surface_conormal=conormal,
            relative_conormal_speed=np.nan,
            edge_measure=1.0,
        ),
        dict(
            outward_surface_conormal=conormal,
            relative_conormal_speed=0.0,
            edge_measure=0.0,
        ),
        dict(
            outward_surface_conormal=conormal,
            relative_conormal_speed=[0.0, 0.1],
            edge_measure=1.0,
        ),
    )
    for invalid in invalid_cases:
        try:
            moving_separation_ibl_flux(planar, **invalid)
        except SurfaceIBLError:
            invalid_rejected += 1

    metrics = {
        "stationary_boundary_reduction": {
            "momentum_error": stationary_momentum_error,
            "energy_error": stationary_energy_error,
        },
        "planar_expanding_and_contracting_control_region": {
            "expanding_inventory_residual": expanding_residual,
            "contracting_inventory_residual": contracting_residual,
        },
        "spherical_cap_moving_latitude": {
            "momentum_inventory_residual": sphere_momentum_residual,
            "energy_inventory_residual": sphere_energy_residual,
        },
        "physical_flux_plus_boundary_sweep": {
            "momentum_superposition_error":
                superposition_momentum_error,
            "energy_superposition_error": superposition_energy_error,
        },
        "proper_rotation_objectivity": {
            "momentum_rotation_error": momentum_rotation_error,
            "energy_invariance_error": energy_invariance_error,
        },
        "orientation_reversal": {
            "momentum_antisymmetry_error":
                momentum_antisymmetry_error,
            "energy_antisymmetry_error": energy_antisymmetry_error,
        },
        "invalid_inputs": {
            "rejected_count": invalid_rejected,
        },
    }
    thresholds = {
        case["id"]: {
            key: float(value)
            for key, value in case["gates"].items()
        }
        for case in prereg["cases"]
    }
    passed = {}
    for case_id, case_thresholds in thresholds.items():
        passed[case_id] = {}
        for metric, threshold in case_thresholds.items():
            value = metrics[case_id][metric]
            if metric == "rejected_count":
                passed[case_id][metric] = bool(value == int(threshold))
            else:
                passed[case_id][metric] = bool(value <= threshold)
    all_pass = all(
        value
        for case in passed.values()
        for value in case.values()
    )
    result = {
        "version": 1,
        "claim_node": prereg["claim_node"],
        "preregistered_case_file": str(CASE_PATH.relative_to(ROOT)),
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": passed,
        "all_pass": all_pass,
        "physical_promotion": {
            "eligible": False,
            "reason": (
                "The guard validates only a moving-boundary transport "
                "identity for already-known IBL channels; no separation "
                "manifold, profile closure, or newborn VES state is supplied."
            ),
        },
    }
    if write:
        RESULT_PATH.write_text(
            json.dumps(result, indent=2, sort_keys=True)+"\n",
            encoding="utf-8",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    result = run(write=arguments.write)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
