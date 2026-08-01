"""Identify the first amplifying channel in the open S3ac composition."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_body_wake_velocity_ledger_guard import _canonical_state  # noqa: E402
from actual_wake_owned_stage_velocity_guard import _actual_ledger  # noqa: E402
from claim_runtime.actual_wake_repeated_insertion import (  # noqa: E402
    advance_actual_wake_repeated_insertion_midpoint,
)


CASES = (
    HERE / "docs" / "diag" / "actual_wake_repeated_insertion_cases.yaml"
)
S3T_CASES = (
    HERE / "docs" / "diag" / "actual_wake_owned_stage_velocity_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_repeated_insertion_stage_probe_results.json"
)


def _maximum_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array), initial=0.0))


def _solution_metrics(solution) -> dict:
    vertices = np.concatenate(
        [band.surface.vertices for band in solution.wake_history.bands],
        axis=0,
    )
    return {
        "band_count": len(solution.wake_history.bands),
        "wake_coordinate_abs_max": _maximum_abs(vertices),
        "wake_potential_jump_abs_max": max(
            _maximum_abs(band.potential_jump_rows)
            for band in solution.wake_history.bands
        ),
        "global_body_potential_abs_max": _maximum_abs(
            solution.global_body_potential
        ),
        "body_face_potential_abs_max": _maximum_abs(
            solution.body_face_potential
        ),
        "actual_matrix_condition_number": float(
            solution.condition_number
        ),
        "actual_relative_weak_residual": float(
            solution.relative_weak_residual
        ),
    }


def _channel_metrics(ledger) -> dict:
    return {
        "external_incident_abs_max": _maximum_abs(
            ledger.external_incident
        ),
        "body_source_abs_max": _maximum_abs(ledger.body_source),
        "body_doublet_abs_max": _maximum_abs(ledger.body_doublet),
        "wake_sheet_average_abs_max": _maximum_abs(
            ledger.wake_sheet_average
        ),
        "total_velocity_abs_max": _maximum_abs(ledger.total),
        "body_doublet_quadrature_order": int(
            ledger.body_doublet_report.quadrature_order
        ),
        "body_doublet_max_abs_change": float(
            ledger.body_doublet_report.max_abs_change
        ),
        "wake_quadrature_order": int(
            ledger.wake_sheet_average_report.quadrature_order
        ),
        "wake_max_abs_change": float(
            ledger.wake_sheet_average_report.max_abs_change
        ),
        "query_reconstruction_error": float(
            ledger.query_reconstruction_error
        ),
        "wake_representation_error": float(
            ledger.wake_representation_error
        ),
    }


def _write(payload: dict) -> None:
    RESULTS.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    s3t_contract = yaml.safe_load(
        S3T_CASES.read_text(encoding="utf-8")
    )
    canonical = contract["canonical"]
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
    step_family = 12
    timestep = (
        float(canonical["time"]["end"])
        - float(canonical["time"]["start"])
    ) / step_family
    payload = {
        "artifact": "actual_wake_repeated_insertion_stage_probe",
        "claim_node": contract["claim_node"],
        "stage": "S3ac_first_amplifying_channel",
        "step_family": step_family,
        "timestep": timestep,
        "status": "running",
        "provider_stages": [],
        "completed_steps": [],
        "production_activation_allowed": False,
    }
    context = {"step_index": -1, "call_in_step": 0}
    _write(payload)

    def provider(actual_solution, query):
        phase = (
            "released_initial"
            if context["call_in_step"] == 0
            else "released_midpoint"
        )
        before = _solution_metrics(actual_solution)
        ledger = _actual_ledger(
            actual_solution,
            query,
            s3t_contract,
        )
        payload["provider_stages"].append(
            {
                "step_index": int(context["step_index"]),
                "phase": phase,
                "query_coordinate_abs_max": _maximum_abs(query.points),
                "solution": before,
                "channels": _channel_metrics(ledger),
            }
        )
        context["call_in_step"] += 1
        _write(payload)
        return ledger.total

    current = solution
    for step_index in range(2):
        context["step_index"] = step_index
        context["call_in_step"] = 0
        print(
            f"S3ac-stage-probe step {step_index + 1}/2",
            flush=True,
        )
        step = advance_actual_wake_repeated_insertion_midpoint(
            mesh,
            body_topology,
            current,
            attachment,
            timestep=timestep,
            physical_velocity_provider=provider,
            transport_quadrature_order=int(
                canonical["quadrature"]["geometry_and_transport_order"]
            ),
            boundary_quadrature_order=int(
                canonical["quadrature"]["actual_boundary_order"]
            ),
            step_index=step_index,
        )
        payload["completed_steps"].append(
            {
                "step_index": step_index,
                "half_stage": _solution_metrics(
                    step.half_stage.solution
                ),
                "endpoint_stage": _solution_metrics(
                    step.endpoint_stage.solution
                ),
                "old_geometry_change_abs_max": float(
                    step.report.old_geometry_change_abs_max
                ),
                "old_scalar_change_abs_max": float(
                    step.report.old_scalar_change_abs_max
                ),
                "old_mass_condition_number_max": float(
                    step.report.old_mass_condition_number_max
                ),
            }
        )
        current = step.endpoint_stage.solution
        _write(payload)
    payload["status"] = "completed"
    _write(payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    run()
