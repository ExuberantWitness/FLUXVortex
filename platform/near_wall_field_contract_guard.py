#!/usr/bin/env python3
"""Run the preregistered N2.6b4f near-wall field contract guards."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np

PLATFORM = Path(__file__).resolve().parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.near_wall_field_contract import (  # noqa: E402
    NearWallFieldContractError,
    assert_no_forbidden_targets,
    validate_near_wall_field_dataset,
)
from tests.test_near_wall_field_contract import (  # noqa: E402
    manufactured_dataset,
    rotation_matrix,
    target_coverage,
)


def main() -> int:
    dataset = manufactured_dataset()
    baseline = validate_near_wall_field_dataset(
        dataset,
        requirements=target_coverage(),
    )

    rotation = rotation_matrix()
    rotated = replace(
        dataset,
        velocity=dataset.velocity@rotation.T,
        wall_position=dataset.wall_position@rotation.T,
        wall_velocity=dataset.wall_velocity@rotation.T,
        surface_normal=dataset.surface_normal@rotation.T,
        tangent_basis=dataset.tangent_basis@rotation.T,
        edge_position=dataset.edge_position@rotation.T,
        edge_velocity=dataset.edge_velocity@rotation.T,
    )
    rotation_report = validate_near_wall_field_dataset(
        rotated,
        requirements=target_coverage(),
    )

    residual_names = (
        "max_no_slip_error",
        "max_edge_velocity_error",
        "max_edge_position_error",
        "max_normal_norm_error",
        "max_basis_norm_error",
        "max_basis_orthogonality_error",
        "max_handedness_error",
    )
    rotation_residual_difference = max(
        abs(getattr(baseline, name)-getattr(rotation_report, name))
        for name in residual_names
    )

    bad_velocity = dataset.velocity.copy()
    bad_velocity[1, 2, 0, 0] += 0.02
    no_slip_report = validate_near_wall_field_dataset(
        replace(dataset, velocity=bad_velocity),
        requirements=target_coverage(),
    )

    forbidden_rejected = False
    try:
        assert_no_forbidden_targets({
            "profile": {"velocity": [0.0, 1.0]},
            "lift_target": [7.8],
        })
    except NearWallFieldContractError:
        forbidden_rejected = True

    passed = bool(
        baseline.schema_valid
        and baseline.identity_valid
        and not baseline.production_evidence_eligible
        and rotation_report.identity_valid
        and rotation_residual_difference <= 5.0e-15
        and not no_slip_report.identity_valid
        and no_slip_report.max_no_slip_error > 0.019
        and forbidden_rejected
    )
    result = {
        "version": 1,
        "scope": "N2.6b4f field schema/identity; not physical closure",
        "preregistered_cases": (
            "platform/docs/diag/near_wall_field_contract_cases.yaml"
        ),
        "baseline": asdict(baseline),
        "rotation": {
            "identity_valid": rotation_report.identity_valid,
            "max_residual_difference": rotation_residual_difference,
        },
        "no_slip_violation": {
            "identity_valid": no_slip_report.identity_valid,
            "max_no_slip_error": no_slip_report.max_no_slip_error,
        },
        "forbidden_target_rejected": forbidden_rejected,
        "physical_promotion": {
            "eligible": baseline.production_evidence_eligible,
            "reason": (
                "manufactured data validate only schema and identities; "
                "independent representative field data remain absent"
            ),
        },
        "passed": passed,
    }

    output = PLATFORM/"docs"/"diag"/"near_wall_field_contract_results.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
