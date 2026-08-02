"""Run the preregistered N3.1j3b2 DDE scalar-potential gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.distributed_doublet import QuadraticDoubletSurface
from claim_runtime.doublet_potential import (
    dde_potential_side_limits,
    surface_doublet_potential,
    surface_sheet_average_potential,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "doublet_potential_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "doublet_potential_results.json"


def run() -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.15, 0.95, 0.0]]
    )
    surface = QuadraticDoubletSurface(
        vertices,
        [[0, 1, 2]],
        [[0.2, -0.1, 0.5, 0.35, 0.15, -0.05]],
    )
    points = np.array(
        [[0.25, 0.22, 0.7], [1.4, -0.4, 0.8], [-0.3, 0.5, -0.6]]
    )
    step = 2.0e-6
    gradient = np.zeros_like(points)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        high = surface_doublet_potential(
            surface, points+offset, quadrature_order=48
        )
        low = surface_doublet_potential(
            surface, points-offset, quadrature_order=48
        )
        gradient[:, axis] = (high-low)/(2.0*step)
    velocity = surface.induced_velocity(points, quadrature_order=48)
    difference = np.linalg.norm(gradient-velocity, axis=1)
    scale = np.maximum(np.linalg.norm(velocity, axis=1), 1.0e-30)
    principal_value = surface_sheet_average_potential(
        surface,
        [0, 0],
        [[0.2, 0.3, 0.5], [1.0/3.0, 1.0/3.0, 1.0/3.0]],
    )
    limits = dde_potential_side_limits(
        [0.2, -0.7, 1.1], [0.4, -0.3, 0.9]
    )
    metrics = {
        "max_gradient_velocity_abs_residual": float(
            np.max(difference, initial=0.0)
        ),
        "max_gradient_velocity_rel_residual": float(
            np.max(difference/scale, initial=0.0)
        ),
        "max_planar_principal_value_residual": float(
            np.max(np.abs(principal_value), initial=0.0)
        ),
        "max_plemelj_jump_residual": limits.max_jump_residual,
    }
    thresholds = {
        key: float(value)
        for key, value in prereg["thresholds"].items()
    }
    passed = {
        key: metrics[key] <= thresholds[key]
        for key in thresholds
    }
    result = {
        "claim": prereg["claim"],
        "target": prereg["target"],
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": passed,
        "all_pass": all(passed.values()),
        "scope_limit": prereg["scope_limit"],
        "interpretation": (
            "The scalar double-layer observer is equation-consistent with "
            "the independent DDE velocity and the frozen N1/DDE jump sign. "
            "Curved multi-patch convergence and material time history remain "
            "open."
        ),
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
