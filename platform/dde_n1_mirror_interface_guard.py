"""Run the preregistered half-wing N1--DDE image-orientation guard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.coupled_lesp_dde import mirror_halfwing_surface
from claim_runtime.distributed_doublet import QuadraticDoubletSurface
from claim_runtime.hirato_shadow import mirrored_ring_field
from claim_runtime.n1_dde_interface import n1_gamma_to_dde_mu
from diff_uvlm_unsteady import _ring_vel


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "dde_n1_mirror_interface_cases.yaml"
RESULT_PATH = (
    ROOT / "docs" / "diag" / "dde_n1_mirror_interface_results.json"
)


def run() -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())
    case = prereg["case"]
    thresholds = prereg["thresholds"]
    gamma = float(case["gamma_n1"])
    y0, y1 = map(float, case["original_span_interval"])
    qfl = np.array([0.0, y0, 0.0])
    qfr = np.array([0.0, y1, 0.0])
    qbr = np.array([1.0, y1, 0.0])
    qbl = np.array([1.0, y0, 0.0])
    ring = np.array([[qfl, qfr, qbr, qbl]])
    normal = np.array([[0.0, 0.0, 1.0]])
    mu = n1_gamma_to_dde_mu(
        [gamma],
        n1_normal=normal,
        dde_normal=normal,
    )[0]
    surface = QuadraticDoubletSurface(
        np.array([qfl, qbl, qbr, qfr]),
        [[0, 1, 2], [0, 2, 3]],
        np.full((2, 6), mu),
    )
    points = np.asarray(case["probe_points"], dtype=float)
    image_points = points.copy()
    image_points[:, 1] *= -1.0
    original_velocity = np.asarray(
        [gamma * _ring_vel(point, ring[0]) for point in points]
    )
    n1_image_velocity = np.asarray(
        [
            gamma * _ring_vel(point, mirrored_ring_field(ring)[0])
            for point in image_points
        ]
    )
    dde_image_velocity = mirror_halfwing_surface(
        surface
    ).induced_velocity_line_reduced(
        image_points,
        quadrature_order=96,
    )
    expected_image = original_velocity * np.asarray(
        case["expected_vector_reflection"],
        dtype=float,
    )
    symmetry_error = float(
        np.max(np.abs(n1_image_velocity - expected_image))
    )
    interface_error = float(
        np.max(np.abs(dde_image_velocity - n1_image_velocity))
    )
    interface_scale = float(np.max(np.abs(n1_image_velocity)))
    interface_relative = interface_error / max(interface_scale, 1.0e-30)
    passed = bool(
        mu / gamma == float(case["expected_mu_over_gamma"])
        and symmetry_error
        <= float(thresholds["max_n1_vector_symmetry_error"])
        and interface_relative
        <= float(thresholds["max_relative_n1_dde_image_error"])
    )
    result = {
        "claim": prereg["claim"],
        "mu_over_gamma": float(mu / gamma),
        "n1_vector_symmetry_error": symmetry_error,
        "max_abs_n1_dde_image_error": interface_error,
        "relative_n1_dde_image_error": interface_relative,
        "passed": passed,
        "scope_limit": prereg["scope_limit"],
        "interpretation": (
            "GO only for the half-wing image orientation of the frozen "
            "N1/DDE interface. This is not a shedding, pressure, force, "
            "or production validation."
        ),
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
