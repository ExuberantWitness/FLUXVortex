"""Run the preregistered local material P2 Kelvin--Helmholtz identity."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from claim_runtime.distributed_doublet import QuadraticDoubletElement
from claim_runtime.material_helmholtz import (
    material_p2_helmholtz_report,
)


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "docs" / "diag" / "dde_material_helmholtz_cases.yaml"
RESULT_PATH = ROOT / "docs" / "diag" / "dde_material_helmholtz_results.json"


def run() -> dict:
    prereg = yaml.safe_load(CASE_PATH.read_text())
    thresholds = prereg["thresholds"]
    cases = []
    for case in prereg["cases"]:
        reference = QuadraticDoubletElement(
            case["reference_vertices"],
            case["material_mu"],
        )
        current = reference.material_update(case["current_vertices"])
        report = material_p2_helmholtz_report(
            reference,
            current,
            case["barycentric"],
            tolerance=max(float(value) for value in thresholds.values()),
        )
        passed = bool(
            report.max_material_mu_residual
            <= float(thresholds["max_material_mu_residual"])
            and report.max_material_derivative_residual
            <= float(thresholds["max_material_derivative_residual"])
            and report.max_cauchy_vector_density_residual
            <= float(
                thresholds["max_cauchy_vector_density_residual"]
            )
        )
        cases.append(
            {
                "name": case["name"],
                "points": report.points,
                "max_material_mu_residual":
                    report.max_material_mu_residual,
                "max_material_derivative_residual":
                    report.max_material_derivative_residual,
                "max_cauchy_vector_density_residual":
                    report.max_cauchy_vector_density_residual,
                "passed": passed,
            }
        )
    result = {
        "claim": prereg["claim"],
        "target": prereg["target"],
        "cases": cases,
        "all_pass": all(case["passed"] for case in cases),
        "scope_limit": prereg["scope_limit"],
        "interpretation": (
            "Material P2 potential-jump preservation supplies the local "
            "piecewise-affine Cauchy stretching identity. Curved seams and "
            "global wake compatibility remain open."
        ),
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

