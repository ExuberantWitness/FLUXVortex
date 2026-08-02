#!/usr/bin/env python3
"""Run preregistered N2.6c1b3a backbone-condition identity guards."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PLATFORM = Path(__file__).resolve().parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from tests.test_material_backbone_conditions import (  # noqa: E402
    diagnostics,
    directions,
    grid,
)


def main() -> int:
    u, v, (uu, vv) = grid()
    ridge = 2.0*np.exp(-(uu/0.35)**2)
    ridge_curvature = np.stack(
        (np.zeros_like(ridge), ridge),
        axis=-1,
    )
    ridge_result = diagnostics(
        ridge_curvature,
        directions(ridge.shape),
        u,
        v,
    )
    centre_u = len(u)//2
    ridge_on_line_count = int(np.count_nonzero(
        ridge_result.two_dimensional_candidate[centre_u]
    ))
    ridge_off_line_count = int(
        np.count_nonzero(ridge_result.two_dimensional_candidate)
        -ridge_on_line_count
    )

    isolated = 1.7*np.exp(
        -(uu/0.33)**2-(vv/0.27)**2
    )
    isolated_result = diagnostics(
        np.stack((isolated, isolated), axis=-1),
        directions(isolated.shape),
        u,
        v,
    )
    centre = (len(u)//2, len(v)//2)
    isolated_count = int(np.count_nonzero(
        isolated_result.one_dimensional_candidate
    ))

    downwelling = -np.exp(-uu**2-vv**2)
    downwelling_result = diagnostics(
        np.stack((downwelling, downwelling), axis=-1),
        directions(downwelling.shape),
        u,
        v,
    )
    minimum = 1.0+uu**2
    minimum_result = diagnostics(
        np.stack((np.zeros_like(minimum), minimum), axis=-1),
        directions(minimum.shape),
        u,
        v,
    )

    derivative_errors = []
    for count in (41, 81, 161):
        u_refined, v_refined, (uu_refined, _) = grid(
            nu=count,
            nv=31,
        )
        sigma = 0.43
        field = np.exp(-(uu_refined/sigma)**2)
        result = diagnostics(
            np.stack((np.zeros_like(field), field), axis=-1),
            directions(field.shape),
            u_refined,
            v_refined,
        )
        exact_first = -2.0*uu_refined/sigma**2*field
        exact_second = (
            (4.0*uu_refined**2/sigma**4-2.0/sigma**2)
            *field
        )
        interior = np.s_[3:-3, 3:-3]
        derivative_errors.append(float(max(
            np.max(np.abs(
                result.directional_derivative[interior]
                -exact_first[interior]
            )),
            np.max(np.abs(
                result.directional_second_derivative[interior]
                -exact_second[interior]
            )),
        )))

    reparameterized = diagnostics(
        ridge_curvature,
        directions(ridge.shape),
        3.2*u+0.4,
        0.7*v-0.2,
    )
    reparameterization_mask_difference = int(np.count_nonzero(
        reparameterized.candidate != ridge_result.candidate
    ))

    downwelling_rejected = not bool(
        downwelling_result.candidate[centre]
    )
    directional_minimum_rejected = not bool(
        minimum_result.candidate[centre]
    )
    passed = bool(
        ridge_on_line_count == len(v)
        and ridge_off_line_count == 0
        and isolated_count == 1
        and bool(isolated_result.one_dimensional_candidate[centre])
        and downwelling_rejected
        and directional_minimum_rejected
        and derivative_errors[0]/derivative_errors[1] >= 3.7
        and derivative_errors[1]/derivative_errors[2] >= 3.7
        and reparameterization_mask_difference == 0
    )
    result = {
        "version": 1,
        "scope": (
            "N2.6c1b3a local material-backbone differential conditions; "
            "not physical ridge extraction"
        ),
        "preregistered_cases": (
            "platform/docs/diag/material_backbone_conditions_cases.yaml"
        ),
        "two_dimensional_gaussian_ridge": {
            "expected_line_nodes": len(v),
            "detected_line_nodes": ridge_on_line_count,
            "off_line_candidates": ridge_off_line_count,
        },
        "one_dimensional_gaussian_maximum": {
            "candidate_count": isolated_count,
            "origin_detected": bool(
                isolated_result.one_dimensional_candidate[centre]
            ),
            "origin_hessian_eigenvalues": (
                isolated_result.hessian_eigenvalues[centre].tolist()
            ),
        },
        "sign_and_curvature_rejection": {
            "downwelling_rejected": downwelling_rejected,
            "directional_minimum_rejected": (
                directional_minimum_rejected
            ),
        },
        "grid_refinement": {
            "maximum_derivative_errors": derivative_errors,
            "coarse_to_medium_ratio": (
                derivative_errors[0]/derivative_errors[1]
            ),
            "medium_to_fine_ratio": (
                derivative_errors[1]/derivative_errors[2]
            ),
        },
        "linear_reparametrization": {
            "candidate_mask_difference": (
                reparameterization_mask_difference
            ),
        },
        "physical_promotion": {
            "eligible": False,
            "reason": (
                "analytic scalar fields validate local differential "
                "conditions only; no connected multilayer ridge, "
                "representative field, or independent separation location"
            ),
        },
        "passed": passed,
    }
    output = (
        PLATFORM/"docs"/"diag"/"material_backbone_conditions_results.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
