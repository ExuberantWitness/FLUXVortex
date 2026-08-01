"""Localize the preregistered S3ac off-plane failure without changing physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

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
from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
    QuadraticDoubletSurface,
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
    / "actual_wake_repeated_insertion_failure_probe_results.json"
)


def _maximum_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array), initial=0.0))


def _history_geometry_metrics(solution) -> dict:
    vertices = np.concatenate(
        [band.surface.vertices for band in solution.wake_history.bands],
        axis=0,
    )
    return {
        "band_count": len(solution.wake_history.bands),
        "coordinate_abs_max": _maximum_abs(vertices),
        "coordinate_norm_max": float(
            np.max(np.linalg.norm(vertices, axis=1), initial=0.0)
        ),
        "potential_jump_abs_max": max(
            _maximum_abs(band.potential_jump_rows)
            for band in solution.wake_history.bands
        ),
    }


def _projection_failure_metrics(solution, query) -> dict:
    surface = QuadraticDoubletSurface(
        solution.mesh.vertices,
        solution.mesh.faces,
        solution.body_face_potential,
    )
    plane_tolerance = 128.0 * np.finfo(float).eps
    records = []
    for face_index in range(len(surface)):
        element = surface.element(face_index)
        delta = query.points - element.vertices[0]
        signed_distance = delta @ element.normal
        projected = (
            query.points - signed_distance[:, None] * element.normal
        )
        residual = np.abs(
            (projected - element.vertices[0]) @ element.normal
        )
        length_scale = max(np.sqrt(2.0 * element.area), 1.0)
        allowed = 4.0 * plane_tolerance * length_scale
        point_index = int(np.argmax(residual))
        records.append(
            {
                "face_index": face_index,
                "query_point_index": point_index,
                "projection_plane_residual": float(residual[point_index]),
                "allowed_projection_plane_residual": float(allowed),
                "residual_over_allowed": float(
                    residual[point_index]
                    / max(allowed, np.finfo(float).tiny)
                ),
                "signed_distance_abs": float(
                    abs(signed_distance[point_index])
                ),
                "query_point": query.points[point_index].tolist(),
                "projected_point": projected[point_index].tolist(),
                "element_vertices": element.vertices.tolist(),
                "element_normal": element.normal.tolist(),
                "element_area": float(element.area),
            }
        )
    worst = max(records, key=lambda item: item["residual_over_allowed"])
    return {
        "query_point_count": len(query.points),
        "query_coordinate_abs_max": _maximum_abs(query.points),
        "body_coordinate_abs_max": _maximum_abs(solution.mesh.vertices),
        "body_potential_abs_max": _maximum_abs(
            solution.body_face_potential
        ),
        "worst_projection": worst,
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
    step_count = 12
    duration = (
        float(canonical["time"]["end"])
        - float(canonical["time"]["start"])
    )
    timestep = duration / step_count
    progress = []
    provider_call = 0
    context = {"step_index": -1}

    def provider(actual_solution, query):
        nonlocal provider_call
        provider_call += 1
        try:
            return _actual_ledger(
                actual_solution,
                query,
                s3t_contract,
            ).total
        except DistributedDoubletError:
            failure = {
                "step_index": int(context["step_index"]),
                "provider_call": int(provider_call),
                "history": _history_geometry_metrics(actual_solution),
                "projection": _projection_failure_metrics(
                    actual_solution,
                    query,
                ),
            }
            raise RuntimeError(
                "S3ac diagnostic provider failure: "
                + json.dumps(failure, sort_keys=True)
            ) from None

    payload = {
        "artifact": "actual_wake_repeated_insertion_failure_probe",
        "claim_node": contract["claim_node"],
        "stage": "S3ac_failure_localization",
        "step_family": step_count,
        "timestep": timestep,
        "status": "running",
        "completed_steps": progress,
        "production_activation_allowed": False,
    }
    _write(payload)
    current = solution
    try:
        for step_index in range(step_count):
            context["step_index"] = step_index
            print(
                f"S3ac-probe step {step_index + 1}/{step_count}",
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
                    canonical["quadrature"][
                        "geometry_and_transport_order"
                    ]
                ),
                boundary_quadrature_order=int(
                    canonical["quadrature"]["actual_boundary_order"]
                ),
                step_index=step_index,
            )
            current = step.endpoint_stage.solution
            progress.append(
                {
                    "step_index": step_index,
                    "provider_call_count": provider_call,
                    "history": _history_geometry_metrics(current),
                    "report": {
                        "old_transport_normalized_residual": max(
                            step.report
                            .half_old_transport_normalized_residual,
                            step.report
                            .full_old_transport_normalized_residual,
                        ),
                        "actual_boundary_relative_weak_residual": (
                            step.report
                            .actual_boundary_relative_weak_residual_max
                        ),
                        "minimum_newborn_face_area": (
                            step.report.minimum_newborn_face_area
                        ),
                    },
                }
            )
            payload["completed_steps"] = progress
            _write(payload)
    except Exception as error:
        payload.update(
            {
                "status": "localized_failure",
                "failed_step_index": int(context["step_index"]),
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write(payload)
        print(json.dumps(payload, indent=2), flush=True)
        return payload
    payload["status"] = "unexpected_no_failure"
    _write(payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    run()
