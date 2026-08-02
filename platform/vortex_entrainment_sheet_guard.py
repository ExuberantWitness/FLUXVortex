"""Run preregistered N2.6c2b vortex-entrainment-sheet equation gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.vortex_entrainment_sheet import (
    equal_density_mass_flux_jump,
    release_junction_report,
    vortex_entrainment_local_balance,
    vortex_entrainment_velocity,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "vortex_entrainment_sheet_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "vortex_entrainment_sheet_results.json"


def _square_quadrature(count=60):
    coordinate = (np.arange(count)+0.5)/count-0.5
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    area = np.full(x.size, 1.0/x.size)
    normal = np.tile([0.0, 0.0, 1.0], (x.size, 1))
    return points, area, normal


def run(*, write: bool = False) -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())
    source, area, normal = _square_quadrature()
    gamma = np.tile([0.0, 0.8, 0.0], (len(source), 1))
    q = np.full(len(source), 0.35)
    target = np.array([[0.0, 0.0, 1.7]])
    combined = vortex_entrainment_velocity(
        target, source, area, gamma, q, normal
    )
    vortex_only = vortex_entrainment_velocity(
        target, source, area, gamma, np.zeros_like(q), normal
    )
    entrainment_only = vortex_entrainment_velocity(
        target, source, area, np.zeros_like(gamma), q, normal
    )
    forbidden_component_error = float(max(
        np.max(np.abs(vortex_only.velocity[0, 1:])),
        np.max(np.abs(entrainment_only.velocity[0, :2])),
    ))
    superposition_error = float(np.max(np.abs(
        combined.velocity
        -vortex_only.velocity
        -entrainment_only.velocity
    )))

    axis = np.array([0.3, -0.8, 0.5])
    axis /= np.linalg.norm(axis)
    angle = 0.91
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
    rotated = vortex_entrainment_velocity(
        target@rotation.T,
        source@rotation.T,
        area,
        gamma@rotation.T,
        q,
        normal@rotation.T,
    )
    rotation_error = float(np.max(np.abs(
        rotated.velocity-combined.velocity@rotation.T
    )))

    far_target = np.array([[0.0, 0.0, 100.0]])
    far = vortex_entrainment_velocity(
        far_target,
        source,
        area,
        np.zeros_like(gamma),
        q,
        normal,
    )
    total_q = float(area@q)
    expected_far = -total_q*far_target[0]/(
        4.0*np.pi*np.linalg.norm(far_target[0])**3
    )
    far_error = float(
        np.linalg.norm(far.velocity[0]-expected_far)
        /np.linalg.norm(expected_far)
    )

    rho_s = 0.42
    divergence = -0.17
    bulk_density = 1.225
    q_local = 0.31
    mass_jump = float(equal_density_mass_flux_jump(
        bulk_density,
        np.array(q_local),
    ))
    drho_s = -(rho_s*divergence+mass_jump)
    acceleration = np.array([0.3, -0.4, 0.2])
    stress_divergence = np.array([0.02, 0.03, -0.01])
    body_force = np.array([0.0, 0.0, -0.2])
    momentum_jump = np.array([0.04, -0.06, 0.08])
    pressure_jump = 0.15
    local_normal = np.array([0.0, 0.0, 1.0])
    shear_jump = (
        rho_s*acceleration
        -(stress_divergence+rho_s*body_force)
        +momentum_jump
        +pressure_jump*local_normal
    )
    balance = vortex_entrainment_local_balance(
        surface_mass_density=rho_s,
        material_surface_mass_rate=drho_s,
        surface_velocity_divergence=divergence,
        outer_mass_flux_jump=mass_jump,
        material_surface_acceleration=acceleration,
        surface_stress_divergence=stress_divergence,
        surface_body_force=body_force,
        outer_momentum_flux_jump=momentum_jump,
        pressure_jump=pressure_jump,
        normal=local_normal,
        shear_stress_jump=shear_jump,
    )
    missing_q = vortex_entrainment_local_balance(
        surface_mass_density=rho_s,
        material_surface_mass_rate=drho_s,
        surface_velocity_divergence=divergence,
        outer_mass_flux_jump=0.0,
        material_surface_acceleration=acceleration,
        surface_stress_divergence=stress_divergence,
        surface_body_force=body_force,
        outer_momentum_flux_jump=momentum_jump,
        pressure_jump=pressure_jump,
        normal=local_normal,
        shear_stress_jump=shear_jump,
    )
    missing_pressure = vortex_entrainment_local_balance(
        surface_mass_density=rho_s,
        material_surface_mass_rate=drho_s,
        surface_velocity_divergence=divergence,
        outer_mass_flux_jump=mass_jump,
        material_surface_acceleration=acceleration,
        surface_stress_divergence=stress_divergence,
        surface_body_force=body_force,
        outer_momentum_flux_jump=momentum_jump,
        pressure_jump=0.0,
        normal=local_normal,
        shear_stress_jump=shear_jump,
    )

    conventional = vortex_entrainment_velocity(
        target,
        source,
        area,
        gamma,
        np.zeros_like(q),
        normal,
    )
    conventional_balance = vortex_entrainment_local_balance(
        surface_mass_density=0.0,
        material_surface_mass_rate=0.0,
        surface_velocity_divergence=0.0,
        outer_mass_flux_jump=0.0,
        material_surface_acceleration=np.zeros(3),
        surface_stress_divergence=np.zeros(3),
        surface_body_force=np.zeros(3),
        outer_momentum_flux_jump=np.zeros(3),
        pressure_jump=0.0,
        normal=local_normal,
        shear_stress_jump=np.zeros(3),
    )
    conventional_error = float(np.max(np.abs(
        conventional.velocity-vortex_only.velocity
    )))

    circulation = np.array([
        [0.1, -0.2, 0.05],
        [-0.03, 0.04, 0.02],
    ])
    mass = np.array([0.03, 0.04])
    momentum = np.array([
        [0.2, -0.1, 0.3],
        [-0.05, 0.07, 0.02],
    ])
    entrainment = np.array([0.012, 0.018])
    junction = release_junction_report(
        incoming_circulation_rate=circulation,
        newborn_circulation_rate=circulation.sum(axis=0),
        incoming_mass_rate=mass,
        newborn_mass_rate=mass.sum(),
        incoming_momentum_rate=momentum,
        newborn_momentum_rate=momentum.sum(axis=0),
        incoming_entrainment_rate=entrainment,
        newborn_entrainment_rate=entrainment.sum(),
    )
    junction_missing = release_junction_report(
        incoming_circulation_rate=circulation,
        newborn_circulation_rate=circulation.sum(axis=0),
        incoming_mass_rate=mass,
        newborn_mass_rate=mass.sum(),
        incoming_momentum_rate=momentum,
        newborn_momentum_rate=momentum.sum(axis=0),
        incoming_entrainment_rate=entrainment,
        newborn_entrainment_rate=0.0,
    )

    metrics = {
        "orthogonal_vortex_entrainment_influence": {
            "forbidden_component_error": forbidden_component_error,
            "superposition_error": superposition_error,
            "nonzero_vortex_signal":
                bool(abs(vortex_only.velocity[0, 0]) > 1.0e-4),
            "nonzero_entrainment_signal":
                bool(abs(entrainment_only.velocity[0, 2]) > 1.0e-4),
        },
        "proper_rotation_covariance": {
            "velocity_rotation_error": rotation_error,
        },
        "source_monopole_far_field": {
            "relative_far_field_error": far_error,
        },
        "local_mass_momentum_balance": {
            "mass_residual": abs(balance.mass_residual),
            "momentum_residual":
                float(np.max(np.abs(balance.momentum_residual))),
            "omitted_entrainment_residual_minimum":
                abs(missing_q.mass_residual),
            "omitted_pressure_residual_minimum":
                float(np.max(np.abs(missing_pressure.momentum_residual))),
        },
        "conventional_vortex_sheet_limit": {
            "velocity_limit_error": conventional_error,
            "mass_residual": abs(conventional_balance.mass_residual),
            "momentum_residual": float(np.max(np.abs(
                conventional_balance.momentum_residual
            ))),
        },
        "named_release_junction": {
            "maximum_conservation_residual":
                junction.maximum_absolute_residual,
            "omitted_channel_residual_minimum":
                junction_missing.maximum_absolute_residual,
        },
    }
    thresholds = {
        case["id"]: case["gates"]
        for case in prereg["cases"]
    }
    passed = {}
    minimum_metrics = {
        "omitted_entrainment_residual_minimum",
        "omitted_pressure_residual_minimum",
        "omitted_channel_residual_minimum",
    }
    for case_id, gates in thresholds.items():
        passed[case_id] = {}
        for metric, threshold in gates.items():
            value = metrics[case_id][metric]
            if isinstance(threshold, bool):
                passed[case_id][metric] = value is threshold
            elif metric in minimum_metrics:
                passed[case_id][metric] = value >= float(threshold)
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
            "The VES companion state has an independent entrainment "
            "influence and explicit mass/momentum/release ledgers. The "
            "smooth-leading-edge release closure and production pressure "
            "remain open."
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

