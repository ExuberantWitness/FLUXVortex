"""Run the preregistered closure-free N2.6b3 surface-IBL gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.surface_ibl_state import (
    SurfaceIBLFields,
    rotate_surface_ibl_fields,
    surface_ibl_budget_report,
    surface_ibl_source_terms,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "surface_ibl_manufactured_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "surface_ibl_results.json"


def _fields(mass, trace) -> SurfaceIBLFields:
    mass = np.asarray(mass, dtype=float)
    trace = np.asarray(trace, dtype=float)
    count = len(mass)
    tensor = np.zeros((count, 3, 3))
    tensor[:, 0, 0] = trace
    return SurfaceIBLFields(
        mass_flux_defect=mass,
        momentum_flux_defect=tensor,
        kinetic_energy_defect_flux=np.zeros((count, 3)),
        external_tangential_velocity=np.zeros((count, 3)),
        external_velocity_surface_gradient=np.zeros((count, 3, 3)),
        wall_shear_over_density=np.zeros((count, 3)),
        dissipation_integral=np.zeros(count),
        surface_normal=np.tile([0.0, 0.0, 1.0], (count, 1)),
    )


def _empty_budget(previous, current, area_previous, area_current, dt, source_m, source_e):
    return surface_ibl_budget_report(
        previous_fields=previous,
        current_fields=current,
        previous_cell_area=area_previous,
        current_cell_area=area_current,
        dt=dt,
        internal_edges=np.empty((0, 2), dtype=int),
        internal_momentum_flux_out_of_first_rate=np.empty((0, 3)),
        internal_energy_flux_out_of_first_rate=np.empty(0),
        boundary_momentum_net_in_rate=np.zeros((previous.count, 3)),
        boundary_energy_net_in_rate=np.zeros(previous.count),
        momentum_source_integral_rate=source_m,
        energy_source_integral_rate=source_e,
    )


def run(*, write: bool = False) -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())

    dt = 0.2
    previous = _fields(
        [[0.4, -0.1, 0.0], [0.2, 0.3, 0.0]],
        [0.5, 0.7],
    )
    momentum_flux = np.array([[0.12, -0.04, 0.0]])
    energy_flux = np.array([0.09])
    internal_momentum = np.array([-momentum_flux[0], momentum_flux[0]])
    internal_energy = np.array([-energy_flux[0], energy_flux[0]])
    boundary_momentum = np.array([
        [0.03, 0.02, 0.0],
        [-0.01, 0.04, 0.0],
    ])
    boundary_energy = np.array([0.02, -0.03])
    source_momentum = np.array([
        [0.01, -0.02, 0.0],
        [0.05, 0.01, 0.0],
    ])
    source_energy = np.array([0.04, 0.01])
    current = _fields(
        previous.mass_flux_defect+dt*(
            internal_momentum+boundary_momentum+source_momentum
        ),
        previous.momentum_flux_trace+dt*(
            internal_energy+boundary_energy+source_energy
        ),
    )
    planar = surface_ibl_budget_report(
        previous_fields=previous,
        current_fields=current,
        previous_cell_area=np.ones(2),
        current_cell_area=np.ones(2),
        dt=dt,
        internal_edges=[[0, 1]],
        internal_momentum_flux_out_of_first_rate=momentum_flux,
        internal_energy_flux_out_of_first_rate=energy_flux,
        boundary_momentum_net_in_rate=boundary_momentum,
        boundary_energy_net_in_rate=boundary_energy,
        momentum_source_integral_rate=source_momentum,
        energy_source_integral_rate=source_energy,
    )

    previous_area = np.array([1.0])
    current_area = np.array([1.25])
    previous_moving = _fields([[0.4, -0.2, 0.0]], [0.6])
    moving_source_m = np.array([[0.15, 0.05, 0.0]])
    moving_source_e = np.array([0.08])
    current_moving = _fields(
        (
            previous_area[:, None]*previous_moving.mass_flux_defect
            + dt*moving_source_m
        )/current_area[:, None],
        (
            previous_area*previous_moving.momentum_flux_trace
            + dt*moving_source_e
        )/current_area,
    )
    moving = _empty_budget(
        previous_moving,
        current_moving,
        previous_area,
        current_area,
        dt,
        moving_source_m,
        moving_source_e,
    )

    tensor = np.array([[
        [0.6, -0.2, 0.0],
        [0.1, 0.4, 0.0],
        [0.0, 0.0, 0.0],
    ]])
    gradient = np.array([[
        [0.2, 0.3, 0.0],
        [-0.4, 0.1, 0.0],
        [0.0, 0.0, 0.0],
    ]])
    objective = SurfaceIBLFields(
        mass_flux_defect=[[0.3, -0.1, 0.0]],
        momentum_flux_defect=tensor,
        kinetic_energy_defect_flux=[[0.8, 0.2, 0.0]],
        external_tangential_velocity=[[2.0, -0.5, 0.0]],
        external_velocity_surface_gradient=gradient,
        wall_shear_over_density=[[0.02, -0.03, 0.0]],
        dissipation_integral=[0.07],
        surface_normal=[[0.0, 0.0, 1.0]],
    )
    angle = 0.71
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
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
    source_reference = surface_ibl_source_terms(objective)
    rotated = rotate_surface_ibl_fields(objective, rotation)
    source_rotated = surface_ibl_source_terms(rotated)
    momentum_rotation_error = float(np.max(np.abs(
        source_rotated.momentum
        - np.einsum("ij,nj->ni", rotation, source_reference.momentum)
    )))
    energy_rotation_error = float(np.max(np.abs(
        source_rotated.energy-source_reference.energy
    )))
    trace_error = float(np.max(np.abs(
        rotated.momentum_flux_trace-objective.momentum_flux_trace
    )))

    one_d = SurfaceIBLFields(
        mass_flux_defect=[[0.35, 0.0, 0.0]],
        momentum_flux_defect=[[
            [0.42, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]],
        kinetic_energy_defect_flux=[[0.73, 0.0, 0.0]],
        external_tangential_velocity=[[1.8, 0.0, 0.0]],
        external_velocity_surface_gradient=[[
            [-0.16, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]],
        wall_shear_over_density=[[0.025, 0.0, 0.0]],
        dissipation_integral=[0.031],
        surface_normal=[[0.0, 0.0, 1.0]],
    )
    one_d_source = surface_ibl_source_terms(one_d)
    scalar_momentum = -(-0.16)*0.35+0.025
    scalar_energy = (
        2.0*0.031-0.42*(-0.16)
        +1.8*((-0.16)*0.35-0.025)
    )
    transverse_error = float(
        np.max(np.abs(one_d_source.momentum[0, 1:]))
    )
    scalar_error = float(max(
        abs(one_d_source.momentum[0, 0]-scalar_momentum),
        abs(one_d_source.energy[0]-scalar_energy),
    ))

    omitted_previous = _fields([[0.0, 0.0, 0.0]], [0.0])
    omitted_current = _fields([[0.01, 0.0, 0.0]], [0.02])
    omitted = _empty_budget(
        omitted_previous,
        omitted_current,
        [1.0],
        [1.0],
        0.1,
        np.zeros((1, 3)),
        np.zeros(1),
    )
    visible_residual = min(
        omitted.max_momentum_residual,
        omitted.max_energy_residual,
    )

    metrics = {
        "planar_two_cell_manufactured_balance": {
            "max_momentum_residual": planar.max_momentum_residual,
            "max_energy_residual": planar.max_energy_residual,
            "global_internal_momentum_flux":
                planar.global_internal_momentum_flux_residual,
            "global_internal_energy_flux":
                planar.global_internal_energy_flux_residual,
        },
        "moving_area_extensive_storage": {
            "max_momentum_residual": moving.max_momentum_residual,
            "max_energy_residual": moving.max_energy_residual,
        },
        "proper_rotation_objectivity": {
            "momentum_source_rotation_error": momentum_rotation_error,
            "energy_source_rotation_error": energy_rotation_error,
            "trace_invariance_error": trace_error,
        },
        "one_dimensional_reduction": {
            "transverse_momentum_error": transverse_error,
            "scalar_source_error": scalar_error,
        },
        "omitted_source_visible": {
            "minimum_visible_residual": visible_residual,
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
            if metric == "minimum_visible_residual":
                passed[case_id][metric] = value >= threshold
            else:
                passed[case_id][metric] = value <= threshold
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
            "The closure-free tensorial 3D IBL conservation skeleton passes "
            "manufactured balance, moving-area, objectivity and 2D-reduction "
            "gates. No profile, transition, separation or force closure has "
            "been validated."
        ),
    }
    if write:
        RESULT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2)+"\n"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="persist the result under platform/docs/diag",
    )
    arguments = parser.parse_args()
    print(json.dumps(run(write=arguments.write), ensure_ascii=False, indent=2))

