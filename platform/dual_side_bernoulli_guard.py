"""Run the preregistered N3.1j3b dual-side Bernoulli algebra gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.dual_side_bernoulli import (
    dual_side_moving_bernoulli,
    paired_thin_sheet_source_report,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "dual_side_bernoulli_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "dual_side_bernoulli_results.json"


def run() -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())
    rng = np.random.default_rng(int(prereg["seed"]))
    count = int(prereg["count"])
    density = 1.225
    mean_rate = rng.normal(size=count)
    jump_rate = rng.normal(size=count)
    mean_velocity = rng.normal(size=(count, 3))
    jump_gradient = rng.normal(size=(count, 3))
    wall_velocity = rng.normal(size=(count, 3))
    gauge = rng.normal(size=count)
    reference = dual_side_moving_bernoulli(
        density=density,
        mean_potential_wall_rate=mean_rate,
        potential_jump_wall_rate=jump_rate,
        mean_velocity=mean_velocity,
        potential_jump_surface_gradient=jump_gradient,
        wall_velocity=wall_velocity,
    )
    shifted = dual_side_moving_bernoulli(
        density=density,
        mean_potential_wall_rate=mean_rate,
        potential_jump_wall_rate=jump_rate,
        mean_velocity=mean_velocity,
        potential_jump_surface_gradient=jump_gradient,
        wall_velocity=wall_velocity,
        bernoulli_gauge=gauge,
    )
    comoving = dual_side_moving_bernoulli(
        density=density,
        mean_potential_wall_rate=mean_rate,
        potential_jump_wall_rate=np.zeros(count),
        mean_velocity=wall_velocity,
        potential_jump_surface_gradient=np.zeros((count, 3)),
        wall_velocity=wall_velocity,
    )
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    acceleration = rng.normal(size=(count, 3))
    gradient_plus = rng.normal(size=(count, 3))
    gradient_minus = rng.normal(size=(count, 3))
    source = paired_thin_sheet_source_report(
        normal_plus=normal,
        wall_acceleration=acceleration,
        specific_pressure_gradient_plus=gradient_plus,
        specific_pressure_gradient_minus=gradient_minus,
        specific_pressure_jump_gradient=gradient_minus-gradient_plus,
    )
    metrics = {
        "max_pressure_jump_identity_residual":
            reference.max_jump_identity_residual,
        "max_gauge_jump_residual": float(np.max(np.abs(
            shifted.pressure_jump-reference.pressure_jump
        ))),
        "max_comoving_jump_residual": float(np.max(np.abs(
            comoving.pressure_jump
        ))),
        "max_paired_source_identity_residual":
            source.max_identity_residual,
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
            "Dual-side pressures are algebraically compatible with the "
            "single unified pressure ledger. This does not yet supply the "
            "DDE mean potential, curved pressure gradients, boundary-layer "
            "inventory, or separation release."
        ),
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
