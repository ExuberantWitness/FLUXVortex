"""Run the preregistered N1--DDE convention and global-algebra guards."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.distributed_doublet import QuadraticDoubletSurface
from claim_runtime.global_conservation import (
    GlobalConservationError,
    GlobalConservationSystem,
)
from claim_runtime.n1_dde_interface import (
    dde_mu_to_n1_gamma,
    n1_gamma_to_dde_mu,
)
from diff_uvlm_unsteady import _ring_vel


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "dde_global_conservation_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "dde_global_conservation_results.json"


def run() -> dict:
    case = yaml.safe_load(CASE_PATH.read_text())
    interface = case["interface_case"]
    gamma = float(interface["gamma_n1"])
    n1_normal = np.asarray([interface["n1_normal"]], dtype=float)
    dde_normal = np.asarray([interface["dde_normal"]], dtype=float)
    mu = n1_gamma_to_dde_mu(
        [gamma],
        n1_normal=n1_normal,
        dde_normal=dde_normal,
    )
    roundtrip = dde_mu_to_n1_gamma(
        mu,
        n1_normal=n1_normal,
        dde_normal=dde_normal,
    )

    qfl = np.array([0.0, 0.0, 0.0])
    qfr = np.array([0.0, 1.0, 0.0])
    qbr = np.array([1.0, 1.0, 0.0])
    qbl = np.array([1.0, 0.0, 0.0])
    n1_ring = np.array([qfl, qfr, qbr, qbl])
    dde_vertices = np.array([qfl, qbl, qbr, qfr])
    dde = QuadraticDoubletSurface(
        dde_vertices,
        [[0, 1, 2], [0, 2, 3]],
        np.full((2, 6), mu[0]),
    )
    points = np.asarray(interface["induced_velocity_points"], dtype=float)
    n1_velocity = np.asarray(
        [gamma * _ring_vel(point, n1_ring) for point in points]
    )
    dde_velocity = dde.induced_velocity_line_reduced(
        points,
        quadrature_order=96,
    )
    induced_error = float(np.max(np.abs(n1_velocity - dde_velocity)))
    induced_scale = float(np.max(np.abs(dde_velocity)))
    induced_relative = induced_error / max(induced_scale, 1.0e-30)
    roundtrip_error = float(np.max(np.abs(roundtrip - gamma)))

    algebra = case["global_algebra_case"]
    truth = np.array([0.18, -0.07, 0.11, -0.04])
    system = GlobalConservationSystem(
        ["Gamma_b0", "Gamma_b1", "mu_new0", "mu_new1"],
        velocity_reference=float(algebra["velocity_reference"]),
        length_reference=float(algebra["length_reference"]),
    )
    blocks = {
        "no_penetration": np.array(
            [[1.2, 0.1, 0.5, -0.2], [0.2, 1.1, -0.1, 0.4]]
        ),
        "trace_continuity": np.array([[0.0, 0.0, 1.0, -1.0]]),
        "vorticity_compatibility": np.array([[0.0, 0.0, 2.0, 1.0]]),
        "material_kelvin": np.array([[1.0, 1.0, 1.0, 1.0]]),
        "kutta_interface": np.array([[1.0, 0.0, -1.0, 0.0]]),
        "mirror_symmetry": np.array([[1.0, -1.0, 0.0, 0.0]]),
        "free_edge": np.array([[0.0, 0.0, 0.0, 1.0]]),
    }
    for name, matrix in blocks.items():
        system.add_block(name, matrix, matrix @ truth)
    solution = system.solve(
        normalized_tolerance=float(
            algebra["normalized_residual_tolerance"]
        )
    )
    state_error = float(np.max(np.abs(solution.values - truth)))

    lesp_rejected = False
    try:
        forbidden = GlobalConservationSystem(
            ["Gamma_b", "mu_new"],
            velocity_reference=1.0,
            length_reference=1.0,
        )
        forbidden.add_block("lesp_amplitude", np.eye(2), np.zeros(2))
    except GlobalConservationError:
        lesp_rejected = True

    interface_pass = bool(
        mu[0] / gamma == float(interface["expected_mu_over_gamma"])
        and induced_relative
        <= float(interface["max_relative_induced_velocity_error"])
        and roundtrip_error <= float(interface["max_roundtrip_error"])
    )
    algebra_pass = bool(
        solution.passed
        and state_error <= float(algebra["state_tolerance"])
        and set(solution.block_reports) == set(algebra["required_blocks"])
        and lesp_rejected
    )
    result = {
        "claim": case["claim"],
        "scope": case["scope"],
        "interface": {
            "mu_over_gamma": float(mu[0] / gamma),
            "max_abs_induced_velocity_error": induced_error,
            "max_relative_induced_velocity_error": induced_relative,
            "roundtrip_error": roundtrip_error,
            "passed": interface_pass,
        },
        "global_algebra": {
            "rank": solution.rank,
            "unknown_count": solution.unknown_count,
            "equation_count": solution.equation_count,
            "condition_number": solution.condition_number,
            "max_abs_normalized_residual":
                solution.max_abs_normalized_residual,
            "state_error": state_error,
            "lesp_amplitude_block_rejected": lesp_rejected,
            "block_reports": {
                name: {
                    "rows": report.rows,
                    "dimension": report.residual_dimension,
                    "physical_scale": report.physical_scale,
                    "max_abs_residual": report.max_abs_residual,
                    "max_abs_normalized_residual":
                        report.max_abs_normalized_residual,
                }
                for name, report in solution.block_reports.items()
            },
            "passed": algebra_pass,
        },
        "all_pass": bool(interface_pass and algebra_pass),
        "interpretation": (
            "GO only for the N1/DDE orientation adapter and named algebraic "
            "ledger. Physical global LEV equations, geometry, pressure, "
            "force, and production remain unvalidated."
        ),
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
