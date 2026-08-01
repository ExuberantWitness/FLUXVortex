"""Persist one S3ac implementation smoke without issuing a claim verdict."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_body_wake_velocity_ledger_guard import (  # noqa: E402
    _canonical_state,
)
from actual_wake_owned_stage_velocity_guard import (  # noqa: E402
    _actual_ledger,
)
from claim_runtime.actual_wake_repeated_insertion import (  # noqa: E402
    advance_actual_wake_repeated_insertion_midpoint,
)


S3T_CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_owned_stage_velocity_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_repeated_insertion_smoke_results.json"
)


def main() -> int:
    contract = yaml.safe_load(S3T_CASES.read_text(encoding="utf-8"))
    (
        mesh,
        body_topology,
        _upper,
        _lower,
        _cut_edges,
        _endpoints,
        _pre_solve_history,
        attachment,
        solution,
    ) = _canonical_state()

    def provider(actual_solution, query):
        return _actual_ledger(
            actual_solution,
            query,
            contract,
        ).total

    step = advance_actual_wake_repeated_insertion_midpoint(
        mesh,
        body_topology,
        solution,
        attachment,
        timestep=0.002,
        physical_velocity_provider=provider,
        transport_quadrature_order=7,
        boundary_quadrature_order=10,
        step_index=0,
    )
    result = {
        "artifact": "S3ac_implementation_smoke_only",
        "claim_verdict_allowed": False,
        "report": asdict(step.report),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
