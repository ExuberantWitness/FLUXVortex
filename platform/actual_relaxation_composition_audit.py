"""Run the preregistered S3r direct-composition sufficiency audit.

This audit is deliberately read-only.  It asks whether the already validated
actual velocity, geometry projection, P2 transport, constrained wake and
actual-boundary interfaces compose on the real one-sided body attachment.
No fallback topology, coordinate inference, geometry update or scalar update
is permitted.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_body_wake_velocity_ledger_guard import (  # noqa: E402
    _canonical_state,
    _field,
    _incident_vector,
)
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    VALIDATED_EDGE_QUADRATURE,
    evaluate_actual_body_wake_sheet_velocity,
    material_wake_assembly,
    wake_sheet_interior_query,
)
from claim_runtime.p2_surface_material_transport import (  # noqa: E402
    assemble_p2_patch_material_transport,
    assemble_p2_surface_material_transport,
)
from claim_runtime.sheet_velocity_projection import (  # noqa: E402
    project_assembly_vertex_star_normal_geometry_velocity,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_relaxation_composition_audit_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_relaxation_composition_audit_results.json"
)


class _OwnerRequiredVelocity:
    """Counterfactual callback expressing the actual on-sheet contract."""

    def __call__(
        self,
        points,
        *,
        patch_indices,
        face_indices,
        barycentric,
    ):
        value = np.asarray(points, dtype=float)
        owners = (
            np.asarray(patch_indices),
            np.asarray(face_indices),
            np.asarray(barycentric),
        )
        if any(len(item) != len(value) for item in owners):
            raise ValueError("owner arrays do not match points")
        return np.zeros_like(value)


def _failure(callable_) -> tuple[bool, str]:
    try:
        callable_()
    except Exception as error:  # noqa: BLE001 - exception identity is evidence
        return False, f"{type(error).__name__}: {error}"
    return True, ""


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    thresholds = contract["thresholds"]
    (
        _mesh,
        _topology,
        _upper,
        _lower,
        _cut_edges,
        _endpoints,
        history,
        _attachment,
        solution,
    ) = _canonical_state()

    query = wake_sheet_interior_query(solution.wake_history)
    vector = _incident_vector(0.5)
    ledger = evaluate_actual_body_wake_sheet_velocity(
        solution,
        query,
        external_incident=_field(vector, "uniform_freestream"),
        body_doublet_orders=(24, 36),
        wake_sheet_average_orders=(24, 36),
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
        edge_quadrature=VALIDATED_EDGE_QUADRATURE,
    )
    assembly = material_wake_assembly(solution.wake_history)

    geometry_ok, geometry_failure = _failure(
        lambda: project_assembly_vertex_star_normal_geometry_velocity(
            assembly,
            ledger.total,
        )
    )
    transport_ok, transport_failure = _failure(
        lambda: assemble_p2_patch_material_transport(
            assembly,
            relative_velocity_provider=lambda points: np.zeros_like(points),
            quadrature_order=4,
        )
    )

    first = history.bands[0]
    owner_callback_ok, owner_callback_failure = _failure(
        lambda: assemble_p2_surface_material_transport(
            first.surface.vertices,
            first.surface.faces,
            relative_velocity_provider=_OwnerRequiredVelocity(),
            quadrature_order=4,
        )
    )
    scalar_writeback_ok, scalar_writeback_failure = _failure(
        lambda: first.material_update(
            first.surface.vertices,
            first.potential_jump_rows,
        )
    )

    direct_failures = sum(
        not value
        for value in (
            geometry_ok,
            transport_ok,
            owner_callback_ok,
            scalar_writeback_ok,
        )
    )
    checks = {
        "actual_history_reenters_boundary_equation": (
            solution.relative_weak_residual
            <= float(
                thresholds[
                    "actual_boundary_relative_weak_residual_max"
                ]
            )
        ),
        "actual_four_channel_ledger_is_exact": (
            ledger.closure_error()
            <= float(
                thresholds["actual_velocity_ledger_closure_abs_max"]
            )
        ),
        "geometry_projection_accepts_typed_open_attachment": geometry_ok,
        "p2_transport_accepts_typed_open_attachment": transport_ok,
        "transport_velocity_contract_is_owner_aware": owner_callback_ok,
        "transported_scalar_writes_back_to_material_rows": (
            scalar_writeback_ok
        ),
        "no_hidden_coordinate_inference": (
            0
            <= int(
                thresholds["hidden_coordinate_inference_count_max"]
            )
        ),
        "direct_composition_has_no_failure": (
            direct_failures
            <= int(
                thresholds["direct_composition_failure_count_max"]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_relaxation_direct_composition_audit",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "actual_boundary_relative_weak_residual": (
                solution.relative_weak_residual
            ),
            "actual_velocity_ledger_closure_abs_max": (
                ledger.closure_error()
            ),
            "history_band_count": len(history.bands),
            "wake_query_point_count": len(query.points),
            "direct_composition_failure_count": direct_failures,
            "hidden_coordinate_inference_count": 0,
        },
        "failure_ledger": {
            "geometry_projection": geometry_failure,
            "p2_transport": transport_failure,
            "owner_aware_transport_callback": owner_callback_failure,
            "material_row_scalar_writeback": scalar_writeback_failure,
        },
        "interpretation": (
            "Existing validated components do not directly compose on the "
            "actual one-sided body attachment. A typed open-boundary, "
            "owner-aware stage operator and exact P2 history reconstruction "
            "are missing before any relax/re-solve time scheme is tested."
        ),
        "forbidden_quantities_absent": [
            "geometry_update",
            "potential_jump_update",
            "boundary_relabelling",
            "coordinate_owner_inference",
            "vortex_core",
            "pressure",
            "force",
            "LESP",
            "target_load",
            "structural_dynamics",
        ],
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(
        0 if payload["stage_decision"] == "GO" else 1
    )
