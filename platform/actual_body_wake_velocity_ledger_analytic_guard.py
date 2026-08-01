"""Rerun the full S3n ledger with the S3q-validated analytic operator."""
from __future__ import annotations

import json
from pathlib import Path

from actual_body_wake_velocity_ledger_guard import run


HERE = Path(__file__).resolve().parent
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_body_wake_velocity_ledger_analytic_results.json"
)


if __name__ == "__main__":
    payload = run(
        edge_quadrature="target_sinh_analytic_sheet",
        results_path=RESULTS,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(
        0 if payload["stage_decision"] == "GO" else 1
    )
