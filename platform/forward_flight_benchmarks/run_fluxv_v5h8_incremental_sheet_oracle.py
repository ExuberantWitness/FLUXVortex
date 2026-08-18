"""Run the observation-free FluxV v5h8 incremental-sheet oracle.

The runner is intentionally executable as an isolated sibling script.  It
does not import the ``forward_flight_benchmarks`` package initializer and has
no Ptera, load, feedback, target-case, or observation path.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Final

import numpy as np

import fluxv_v5h8_incremental_sheet as sheet
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian


RUN_ID: Final = "20260815_fluxv_v5h8_incremental_sheet_oracle"
SCHEMA_ID: Final = "fluxv-v5h8-incremental-sheet-oracle-v1"
EXPECTED_SHEET_MODULE_SHA256: Final = (
    "f30b5fbf6d2f1718bbbecec669a47dfb1c3c001942d2212ce814098966a610e2"
)

SPAN_M: Final = 0.60
PANEL_STEP_M: Final = 0.04
DELTA_GAMMA_M2_PER_S: Final = 0.016
SMOOTHING_RADIUS_M: Final = 0.085
TARGET_SPACING_M: Final = 0.02
RELEASE_COUNTS: Final = (1, 2, 3, 4)
PARTICLE_CAP: Final = 1_000

FIELD_IMPULSE_TOLERANCE: Final = 2.0e-14
JACOBIAN_TOLERANCE: Final = 2.0e-14
CLONE_TOLERANCE: Final = 1.0e-14
FRESH_REDEPOSITION_MIN_RELATIVE_DIFFERENCE: Final = 1.0e-8

FIXED_PROBES_GP1_M: Final = np.asarray(
    ((0.05, 0.30, 0.30), (0.10, 0.15, 0.40), (0.20, 0.45, 0.50)),
    dtype=np.float64,
)
FIXED_PROBES_GP1_M.setflags(write=False)

AFFINE_MATRIX: Final = np.asarray(
    ((1.0, 0.015, 0.0), (0.0, 1.0, 0.01), (0.02, 0.0, 0.99)),
    dtype=np.float64,
)
AFFINE_MATRIX.setflags(write=False)
AFFINE_TRANSLATION_M: Final = np.asarray((0.04, -0.002, 0.003), dtype=np.float64)
AFFINE_TRANSLATION_M.setflags(write=False)
AFFINE_SIGMA_SCALE: Final = 1.01
STREAMWISE_APPEND_VECTOR_M: Final = np.asarray(
    (PANEL_STEP_M, 0.0, 0.0), dtype=np.float64
)
STREAMWISE_APPEND_VECTOR_M.setflags(write=False)


def _strict_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _first_panel() -> sheet.SheetPanel:
    return sheet.make_panel(
        (0.0, 0.0, 0.0),
        (0.0, SPAN_M, 0.0),
        (PANEL_STEP_M, 0.0, 0.0),
        (PANEL_STEP_M, SPAN_M, 0.0),
        DELTA_GAMMA_M2_PER_S,
        release_index=1,
    )


def _start() -> sheet.AppendResult:
    return sheet.start_incremental_sheet(
        _first_panel(),
        SMOOTHING_RADIUS_M,
        TARGET_SPACING_M,
        particle_cap=PARTICLE_CAP,
    )


def _field_and_jacobian(value: object) -> tuple[np.ndarray, np.ndarray]:
    result = direct_gaussian_erf_velocity_jacobian(
        value.positions,  # type: ignore[attr-defined]
        value.gamma,  # type: ignore[attr-defined]
        value.sigma,  # type: ignore[attr-defined]
        target_positions=FIXED_PROBES_GP1_M,
    )
    return result.velocity, result.jacobian


def _particle_contract(value: object) -> dict[str, Any]:
    positions = value.positions  # type: ignore[attr-defined]
    gamma = value.gamma  # type: ignore[attr-defined]
    sigma = value.sigma  # type: ignore[attr-defined]
    particle_ids = value.particle_ids  # type: ignore[attr-defined]
    count = int(positions.shape[0])
    finite = bool(
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(gamma))
        and np.all(np.isfinite(sigma))
    )
    shape_closed = bool(
        positions.shape == (count, 3)
        and gamma.shape == positions.shape
        and sigma.shape == (count,)
    )
    readonly = bool(
        not positions.flags.writeable
        and not gamma.flags.writeable
        and not sigma.flags.writeable
    )
    unique_ids = len(set(particle_ids)) == count
    passed = bool(
        finite
        and shape_closed
        and readonly
        and unique_ids
        and np.all(sigma > 0.0)
        and count <= PARTICLE_CAP
    )
    return {
        "particle_count": count,
        "finite": finite,
        "shape_closed": shape_closed,
        "readonly": readonly,
        "positive_sigma": bool(np.all(sigma > 0.0)),
        "unique_particle_ids": unique_ids,
        "particle_cap": PARTICLE_CAP,
        "particle_cap_passed": count <= PARTICLE_CAP,
        "passed": passed,
    }


def _prefix_bitwise_equal(
    parent: sheet.IncrementalSheetState,
    child: sheet.IncrementalSheetState,
) -> bool:
    count = parent.positions.shape[0]
    return bool(
        child.positions[:count].tobytes() == parent.positions.tobytes()
        and child.gamma[:count].tobytes() == parent.gamma.tobytes()
        and child.sigma[:count].tobytes() == parent.sigma.tobytes()
        and child.particle_ids[:count] == parent.particle_ids
        and child.lineage[:count] == parent.lineage
    )


def _independent_clone_residuals(
    parent: sheet.IncrementalSheetState,
    child: sheet.IncrementalSheetState,
    current_circulation: float,
) -> dict[str, Any]:
    new_pairs = child.clone_pairs[len(parent.clone_pairs) :]
    cloned_old_indices = tuple(old_index for old_index, _ in new_pairs)
    cloned_new_indices = tuple(clone_index for _, clone_index in new_pairs)
    expected_clone_indices = tuple(
        range(
            parent.positions.shape[0],
            parent.positions.shape[0] + len(parent.downstream_particle_indices),
        )
    )
    old_boundary_set_exact = cloned_old_indices == parent.downstream_particle_indices
    appended_clone_set_exact = cloned_new_indices == expected_clone_indices
    expected_scale = -current_circulation / parent.panels[-1].circulation_m2_s
    position_residual = 0.0
    sigma_residual = 0.0
    gamma_residual = 0.0
    for old_index, clone_index in new_pairs:
        position_residual = max(
            position_residual,
            _max_abs(child.positions[clone_index] - child.positions[old_index]),
        )
        sigma_residual = max(
            sigma_residual,
            abs(float(child.sigma[clone_index] - child.sigma[old_index])),
        )
        gamma_residual = max(
            gamma_residual,
            _max_abs(
                child.gamma[clone_index] - expected_scale * child.gamma[old_index]
            ),
        )
    return {
        "new_clone_pair_count": len(new_pairs),
        "old_boundary_set_exact": old_boundary_set_exact,
        "appended_clone_set_exact": appended_clone_set_exact,
        "expected_clone_gamma_scale": expected_scale,
        "max_clone_position_abs": position_residual,
        "max_clone_sigma_abs": sigma_residual,
        "max_clone_gamma_relation_abs": gamma_residual,
        "passed": bool(
            len(new_pairs) == len(parent.downstream_particle_indices)
            and old_boundary_set_exact
            and appended_clone_set_exact
            and position_residual <= CLONE_TOLERANCE
            and sigma_residual <= CLONE_TOLERANCE
            and gamma_residual <= CLONE_TOLERANCE
        ),
    }


def _zero_transport_rows() -> list[dict[str, Any]]:
    result = _start()
    state = result.state
    rows: list[dict[str, Any]] = []
    for release_count in RELEASE_COUNTS:
        if release_count > 1:
            parent = state
            result = sheet.append_live_basis_panel(
                parent,
                (release_count * PANEL_STEP_M, 0.0, 0.0),
                (release_count * PANEL_STEP_M, SPAN_M, 0.0),
                release_count * DELTA_GAMMA_M2_PER_S,
                particle_cap=PARTICLE_CAP,
            )
            state = result.state
            prefix_independent = _prefix_bitwise_equal(parent, state)
            clones = _independent_clone_residuals(
                parent,
                state,
                release_count * DELTA_GAMMA_M2_PER_S,
            )
        else:
            prefix_independent = True
            clones = {
                "new_clone_pair_count": 0,
                "old_boundary_set_exact": True,
                "appended_clone_set_exact": True,
                "expected_clone_gamma_scale": None,
                "max_clone_position_abs": 0.0,
                "max_clone_sigma_abs": 0.0,
                "max_clone_gamma_relation_abs": 0.0,
                "passed": True,
            }

        direct = sheet.direct_connected_redeposit(
            state.panels,
            SMOOTHING_RADIUS_M,
            TARGET_SPACING_M,
        )
        incremental_field, incremental_jacobian = _field_and_jacobian(state)
        direct_field, direct_jacobian = _field_and_jacobian(direct)
        incremental_impulse = sheet.particle_impulse(state)
        direct_impulse = sheet.particle_impulse(direct)
        analytic_impulse = np.asarray(
            (
                0.0,
                0.0,
                -SPAN_M
                * PANEL_STEP_M
                * DELTA_GAMMA_M2_PER_S
                * release_count
                * (release_count + 1)
                / 2.0,
            ),
            dtype=np.float64,
        )
        field_residual = _max_abs(incremental_field - direct_field)
        jacobian_residual = _max_abs(incremental_jacobian - direct_jacobian)
        impulse_residual = _max_abs(incremental_impulse - direct_impulse)
        analytic_impulse_residual = _max_abs(direct_impulse - analytic_impulse)
        incremental_contract = _particle_contract(state)
        comparator_contract = _particle_contract(direct)
        append_gate = bool(
            result.diagnostics.prefix_bitwise_unchanged
            and prefix_independent
            and result.diagnostics.max_clone_position_abs <= CLONE_TOLERANCE
            and result.diagnostics.max_clone_sigma_abs <= CLONE_TOLERANCE
            and result.diagnostics.max_clone_gamma_relation_abs <= CLONE_TOLERANCE
            and clones["passed"]
        )
        parity_gate = bool(
            field_residual <= FIELD_IMPULSE_TOLERANCE
            and jacobian_residual <= JACOBIAN_TOLERANCE
            and impulse_residual <= FIELD_IMPULSE_TOLERANCE
            and analytic_impulse_residual <= FIELD_IMPULSE_TOLERANCE
        )
        passed = bool(
            append_gate
            and parity_gate
            and incremental_contract["passed"]
            and comparator_contract["passed"]
        )
        rows.append(
            {
                "transport": "zero",
                "release_count": release_count,
                "incremental_particle_count": int(state.positions.shape[0]),
                "direct_connected_particle_count": int(direct.positions.shape[0]),
                "clone_pair_count": len(state.clone_pairs),
                "live_downstream_particle_count": len(
                    state.downstream_particle_indices
                ),
                "append_diagnostics": {
                    "prefix_particle_count": (result.diagnostics.prefix_particle_count),
                    "appended_particle_count": (
                        result.diagnostics.appended_particle_count
                    ),
                    "cloned_particle_count": result.diagnostics.cloned_particle_count,
                    "excluded_fresh_upstream_count": (
                        result.diagnostics.excluded_fresh_upstream_count
                    ),
                    "reported_prefix_bitwise_unchanged": (
                        result.diagnostics.prefix_bitwise_unchanged
                    ),
                    "independent_prefix_bitwise_unchanged": prefix_independent,
                    "reported_max_clone_position_abs": (
                        result.diagnostics.max_clone_position_abs
                    ),
                    "reported_max_clone_sigma_abs": (
                        result.diagnostics.max_clone_sigma_abs
                    ),
                    "reported_max_clone_gamma_relation_abs": (
                        result.diagnostics.max_clone_gamma_relation_abs
                    ),
                    "independent_clone_check": clones,
                },
                "incremental_probe_velocity_gp1_m_per_s": (incremental_field.tolist()),
                "direct_probe_velocity_gp1_m_per_s": direct_field.tolist(),
                "incremental_probe_jacobian_gp1_per_s": (incremental_jacobian.tolist()),
                "direct_probe_jacobian_gp1_per_s": direct_jacobian.tolist(),
                "incremental_impulse_gp1_m4_per_s": incremental_impulse.tolist(),
                "direct_impulse_gp1_m4_per_s": direct_impulse.tolist(),
                "analytic_impulse_gp1_m4_per_s": analytic_impulse.tolist(),
                "max_field_abs_residual": field_residual,
                "max_jacobian_abs_residual": jacobian_residual,
                "max_impulse_abs_residual": impulse_residual,
                "max_analytic_impulse_abs_residual": analytic_impulse_residual,
                "incremental_particle_contract": incremental_contract,
                "direct_particle_contract": comparator_contract,
                "append_gate_passed": append_gate,
                "parity_gate_passed": parity_gate,
                "passed": passed,
            }
        )
    return rows


def _affine_transport_relation(
    parent: sheet.IncrementalSheetState,
    transported: sheet.IncrementalSheetState,
) -> dict[str, Any]:
    position_residual = _max_abs(
        transported.positions
        - (parent.positions @ AFFINE_MATRIX.T + AFFINE_TRANSLATION_M)
    )
    gamma_residual = _max_abs(transported.gamma - parent.gamma @ AFFINE_MATRIX.T)
    sigma_residual = _max_abs(transported.sigma - parent.sigma * AFFINE_SIGMA_SCALE)
    identity_preserved = bool(
        transported.particle_ids == parent.particle_ids
        and transported.lineage == parent.lineage
        and transported.downstream_particle_indices
        == parent.downstream_particle_indices
        and transported.clone_pairs == parent.clone_pairs
    )
    return {
        "max_position_relation_abs": position_residual,
        "max_gamma_relation_abs": gamma_residual,
        "max_sigma_relation_abs": sigma_residual,
        "particle_identity_and_indices_preserved": identity_preserved,
        "passed": bool(
            position_residual == 0.0
            and gamma_residual == 0.0
            and sigma_residual == 0.0
            and identity_preserved
        ),
    }


def _affine_rows() -> list[dict[str, Any]]:
    result = _start()
    state = result.state
    rows: list[dict[str, Any]] = []
    for release_count in RELEASE_COUNTS:
        if release_count > 1:
            pre_transport = state
            transported = sheet.affine_transport_state(
                pre_transport,
                AFFINE_MATRIX,
                AFFINE_TRANSLATION_M,
                AFFINE_SIGMA_SCALE,
            )
            transport_relation = _affine_transport_relation(pre_transport, transported)
            upstream_left = np.asarray(
                transported.panels[-1].downstream_left, dtype=np.float64
            )
            upstream_right = np.asarray(
                transported.panels[-1].downstream_right, dtype=np.float64
            )
            current_circulation = release_count * DELTA_GAMMA_M2_PER_S
            result = sheet.append_live_basis_panel(
                transported,
                upstream_left + STREAMWISE_APPEND_VECTOR_M,
                upstream_right + STREAMWISE_APPEND_VECTOR_M,
                current_circulation,
                particle_cap=PARTICLE_CAP,
            )
            state = result.state
            prefix_independent = _prefix_bitwise_equal(transported, state)
            clones = _independent_clone_residuals(
                transported, state, current_circulation
            )
        else:
            transport_relation = {
                "max_position_relation_abs": 0.0,
                "max_gamma_relation_abs": 0.0,
                "max_sigma_relation_abs": 0.0,
                "particle_identity_and_indices_preserved": True,
                "passed": True,
            }
            prefix_independent = True
            clones = {
                "new_clone_pair_count": 0,
                "old_boundary_set_exact": True,
                "appended_clone_set_exact": True,
                "expected_clone_gamma_scale": None,
                "max_clone_position_abs": 0.0,
                "max_clone_sigma_abs": 0.0,
                "max_clone_gamma_relation_abs": 0.0,
                "passed": True,
            }

        collapsed = sheet.collapse_live_basis_pairs(state)
        fresh = sheet.fresh_geometry_redeposit(state)
        live_field, live_jacobian = _field_and_jacobian(state)
        collapsed_field, collapsed_jacobian = _field_and_jacobian(collapsed)
        fresh_field, fresh_jacobian = _field_and_jacobian(fresh)
        live_impulse = sheet.particle_impulse(state)
        collapsed_impulse = sheet.particle_impulse(collapsed)
        fresh_impulse = sheet.particle_impulse(fresh)

        field_residual = _max_abs(live_field - collapsed_field)
        jacobian_residual = _max_abs(live_jacobian - collapsed_jacobian)
        impulse_residual = _max_abs(live_impulse - collapsed_impulse)
        live_field_norm = float(np.linalg.norm(live_field))
        live_jacobian_norm = float(np.linalg.norm(live_jacobian))
        fresh_field_relative = float(
            np.linalg.norm(live_field - fresh_field) / live_field_norm
        )
        fresh_jacobian_relative = float(
            np.linalg.norm(live_jacobian - fresh_jacobian) / live_jacobian_norm
        )
        fresh_impulse_abs = _max_abs(live_impulse - fresh_impulse)

        live_contract = _particle_contract(state)
        collapsed_contract = _particle_contract(collapsed)
        fresh_contract = _particle_contract(fresh)
        collapse_count_closed = collapsed.positions.shape[0] == (
            state.positions.shape[0] - len(state.clone_pairs)
        )
        append_gate = bool(
            result.diagnostics.prefix_bitwise_unchanged
            and prefix_independent
            and result.diagnostics.max_clone_position_abs <= CLONE_TOLERANCE
            and result.diagnostics.max_clone_sigma_abs <= CLONE_TOLERANCE
            and result.diagnostics.max_clone_gamma_relation_abs <= CLONE_TOLERANCE
            and clones["passed"]
        )
        parity_gate = bool(
            field_residual <= FIELD_IMPULSE_TOLERANCE
            and jacobian_residual <= JACOBIAN_TOLERANCE
            and impulse_residual <= FIELD_IMPULSE_TOLERANCE
        )
        fresh_required = release_count >= 2
        fresh_gate = bool(
            not fresh_required
            or fresh_field_relative > FRESH_REDEPOSITION_MIN_RELATIVE_DIFFERENCE
        )
        passed = bool(
            transport_relation["passed"]
            and append_gate
            and parity_gate
            and fresh_gate
            and collapse_count_closed
            and live_contract["passed"]
            and collapsed_contract["passed"]
            and fresh_contract["passed"]
        )
        gross_gamma_l1 = float(np.sum(np.linalg.norm(state.gamma, axis=1)))
        collapsed_gamma_l1 = float(np.sum(np.linalg.norm(collapsed.gamma, axis=1)))
        rows.append(
            {
                "transport": "affine_live_material",
                "release_count": release_count,
                "affine_transport_count": release_count - 1,
                "incremental_particle_count": int(state.positions.shape[0]),
                "collapsed_live_particle_count": int(collapsed.positions.shape[0]),
                "fresh_geometry_particle_count": int(fresh.positions.shape[0]),
                "clone_pair_count": len(state.clone_pairs),
                "live_downstream_particle_count": len(
                    state.downstream_particle_indices
                ),
                "transport_relation": transport_relation,
                "append_diagnostics": {
                    "prefix_particle_count": (result.diagnostics.prefix_particle_count),
                    "appended_particle_count": (
                        result.diagnostics.appended_particle_count
                    ),
                    "cloned_particle_count": result.diagnostics.cloned_particle_count,
                    "excluded_fresh_upstream_count": (
                        result.diagnostics.excluded_fresh_upstream_count
                    ),
                    "reported_prefix_bitwise_unchanged": (
                        result.diagnostics.prefix_bitwise_unchanged
                    ),
                    "independent_prefix_bitwise_unchanged": prefix_independent,
                    "reported_max_clone_position_abs": (
                        result.diagnostics.max_clone_position_abs
                    ),
                    "reported_max_clone_sigma_abs": (
                        result.diagnostics.max_clone_sigma_abs
                    ),
                    "reported_max_clone_gamma_relation_abs": (
                        result.diagnostics.max_clone_gamma_relation_abs
                    ),
                    "independent_clone_check": clones,
                },
                "live_probe_velocity_gp1_m_per_s": live_field.tolist(),
                "collapsed_probe_velocity_gp1_m_per_s": (collapsed_field.tolist()),
                "fresh_probe_velocity_gp1_m_per_s": fresh_field.tolist(),
                "live_probe_jacobian_gp1_per_s": live_jacobian.tolist(),
                "collapsed_probe_jacobian_gp1_per_s": (collapsed_jacobian.tolist()),
                "fresh_probe_jacobian_gp1_per_s": fresh_jacobian.tolist(),
                "live_impulse_gp1_m4_per_s": live_impulse.tolist(),
                "collapsed_impulse_gp1_m4_per_s": collapsed_impulse.tolist(),
                "fresh_impulse_gp1_m4_per_s": fresh_impulse.tolist(),
                "max_live_collapse_field_abs_residual": field_residual,
                "max_live_collapse_jacobian_abs_residual": jacobian_residual,
                "max_live_collapse_impulse_abs_residual": impulse_residual,
                "fresh_field_relative_difference": fresh_field_relative,
                "fresh_jacobian_relative_difference": fresh_jacobian_relative,
                "max_fresh_impulse_abs_difference": fresh_impulse_abs,
                "fresh_negative_control_required": fresh_required,
                "fresh_negative_control_passed": fresh_gate,
                "gross_gamma_l1": gross_gamma_l1,
                "collapsed_gamma_l1": collapsed_gamma_l1,
                "gross_to_collapsed_gamma_l1_ratio": (
                    gross_gamma_l1 / collapsed_gamma_l1
                ),
                "collapse_particle_count_closed": bool(collapse_count_closed),
                "live_particle_contract": live_contract,
                "collapsed_particle_contract": collapsed_contract,
                "fresh_particle_contract": fresh_contract,
                "append_gate_passed": append_gate,
                "parity_gate_passed": parity_gate,
                "passed": passed,
            }
        )
    return rows


def _rollback_probe() -> dict[str, Any]:
    state = _start().state
    positions = state.positions.tobytes()
    gamma = state.gamma.tobytes()
    sigma = state.sigma.tobytes()
    particle_ids = state.particle_ids
    lineage = state.lineage
    panels = state.panels
    downstream = state.downstream_particle_indices
    clone_pairs = state.clone_pairs
    exception_type: str | None = None
    exception_mentions_cap = False
    try:
        sheet.append_live_basis_panel(
            state,
            (2.0 * PANEL_STEP_M, 0.0, 0.0),
            (2.0 * PANEL_STEP_M, SPAN_M, 0.0),
            2.0 * DELTA_GAMMA_M2_PER_S,
            particle_cap=state.positions.shape[0],
        )
    except RuntimeError as error:
        exception_type = type(error).__name__
        exception_mentions_cap = "cap" in str(error).lower()
    unchanged = bool(
        state.positions.tobytes() == positions
        and state.gamma.tobytes() == gamma
        and state.sigma.tobytes() == sigma
        and state.particle_ids == particle_ids
        and state.lineage == lineage
        and state.panels is panels
        and state.downstream_particle_indices is downstream
        and state.clone_pairs is clone_pairs
    )
    return {
        "forced_particle_cap": int(state.positions.shape[0]),
        "exception_type": exception_type,
        "exception_mentions_cap": exception_mentions_cap,
        "parent_bitwise_unchanged": unchanged,
        "passed": bool(
            exception_type == "RuntimeError" and exception_mentions_cap and unchanged
        ),
    }


def _sheet_module_path() -> Path:
    return Path(__file__).with_name("fluxv_v5h8_incremental_sheet.py")


def run_gate() -> dict[str, Any]:
    zero_rows = _zero_transport_rows()
    affine_rows = _affine_rows()
    rollback = _rollback_probe()
    sheet_sha256 = _file_sha256(_sheet_module_path())
    module_frozen = sheet_sha256 == EXPECTED_SHEET_MODULE_SHA256

    zero_passed = all(bool(row["passed"]) for row in zero_rows)
    affine_passed = all(bool(row["passed"]) for row in affine_rows)
    fresh_passed = all(
        bool(row["fresh_negative_control_passed"])
        for row in affine_rows
        if bool(row["fresh_negative_control_required"])
    )
    max_zero_field = max(float(row["max_field_abs_residual"]) for row in zero_rows)
    max_zero_jacobian = max(
        float(row["max_jacobian_abs_residual"]) for row in zero_rows
    )
    max_zero_impulse = max(float(row["max_impulse_abs_residual"]) for row in zero_rows)
    max_affine_field = max(
        float(row["max_live_collapse_field_abs_residual"]) for row in affine_rows
    )
    max_affine_jacobian = max(
        float(row["max_live_collapse_jacobian_abs_residual"]) for row in affine_rows
    )
    max_affine_impulse = max(
        float(row["max_live_collapse_impulse_abs_residual"]) for row in affine_rows
    )
    fresh_differences = [
        float(row["fresh_field_relative_difference"])
        for row in affine_rows
        if bool(row["fresh_negative_control_required"])
    ]
    max_particle_count = max(
        [int(row["incremental_particle_count"]) for row in zero_rows]
        + [int(row["incremental_particle_count"]) for row in affine_rows]
    )
    passed = bool(
        module_frozen
        and zero_passed
        and affine_passed
        and fresh_passed
        and rollback["passed"]
        and max_particle_count <= PARTICLE_CAP
    )
    return {
        "schema_id": SCHEMA_ID,
        "run_id": RUN_ID,
        "status": (
            "go_v5h8_bounded_affine_live_basis_mechanics_only"
            if passed
            else "stop_v5h8_incremental_sheet_oracle_failed"
        ),
        "passed": passed,
        "scope": {
            "evaluation": "manufactured_simulation_only",
            "observation_access": "none",
            "target_case_branch": "none",
            "forward_flight_package_init_executed": False,
            "ptera_solver_call_count": 0,
            "load_call_count": 0,
            "feedback_call_count": 0,
            "claim_limit": (
                "bounded inherited-material-basis algebra under the frozen "
                "affine family only; no generic rVPM, non-affine, production "
                "solver, stability, load, or aerodynamic-accuracy claim"
            ),
        },
        "contract": {
            "release_counts": list(RELEASE_COUNTS),
            "span_m": SPAN_M,
            "panel_step_m": PANEL_STEP_M,
            "delta_gamma_m2_per_s": DELTA_GAMMA_M2_PER_S,
            "smoothing_radius_m": SMOOTHING_RADIUS_M,
            "target_spacing_m": TARGET_SPACING_M,
            "fixed_probes_gp1_m": FIXED_PROBES_GP1_M.tolist(),
            "affine_matrix": AFFINE_MATRIX.tolist(),
            "affine_translation_m": AFFINE_TRANSLATION_M.tolist(),
            "affine_sigma_scale": AFFINE_SIGMA_SCALE,
            "field_impulse_tolerance": FIELD_IMPULSE_TOLERANCE,
            "jacobian_tolerance": JACOBIAN_TOLERANCE,
            "clone_tolerance": CLONE_TOLERANCE,
            "fresh_redeposit_min_relative_field_difference": (
                FRESH_REDEPOSITION_MIN_RELATIVE_DIFFERENCE
            ),
            "particle_cap": PARTICLE_CAP,
            "expected_sheet_module_sha256": EXPECTED_SHEET_MODULE_SHA256,
            "actual_sheet_module_sha256": sheet_sha256,
        },
        "zero_transport_rows": zero_rows,
        "affine_transport_rows": affine_rows,
        "rollback_probe": rollback,
        "diagnostics": {
            "max_zero_field_abs_residual": max_zero_field,
            "max_zero_jacobian_abs_residual": max_zero_jacobian,
            "max_zero_impulse_abs_residual": max_zero_impulse,
            "max_affine_live_collapse_field_abs_residual": max_affine_field,
            "max_affine_live_collapse_jacobian_abs_residual": (max_affine_jacobian),
            "max_affine_live_collapse_impulse_abs_residual": max_affine_impulse,
            "affine_fresh_field_relative_differences_releases_2_to_4": (
                fresh_differences
            ),
            "minimum_affine_fresh_field_relative_difference": min(fresh_differences),
            "maximum_incremental_particle_count": max_particle_count,
        },
        "gates": {
            "sheet_module_frozen_sha256_passed": module_frozen,
            "all_zero_transport_rows_passed": zero_passed,
            "all_affine_live_basis_rows_passed": affine_passed,
            "fresh_geometry_negative_control_passed": fresh_passed,
            "particle_cap_passed": max_particle_count <= PARTICLE_CAP,
            "cap_failure_rollback_passed": bool(rollback["passed"]),
            "target_access_count": 0,
            "ptera_solver_call_count": 0,
            "load_call_count": 0,
        },
        "production_decision": {
            "promotion": "blocked",
            "reason": (
                "the gate tests a bounded affine clone/collapse identity; "
                "non-affine support incompatibility and long-time cancellation "
                "conditioning remain explicit STOP gates"
            ),
            "required_next_owner_decision": (
                "audit an explicit conservative boundary update or remesh owner "
                "for general transported sheets"
            ),
        },
    }


def _source_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        root / relative
        for relative in (
            "platform/forward_flight_benchmarks/fluxv_v5h8_incremental_sheet.py",
            "platform/forward_flight_benchmarks/run_fluxv_v5h8_incremental_sheet_oracle.py",
            "platform/tests/test_fluxv_v5h8_incremental_sheet.py",
            "platform/tests/test_run_fluxv_v5h8_incremental_sheet_oracle.py",
            "src/fluxvortex/rvpm_edge_bridge.py",
            "src/fluxvortex/rvpm_reference.py",
        )
    )


def write_artifact(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_dir}")
    output_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    summary = run_gate()
    (output_dir / "summary.json").write_text(_strict_json(summary), encoding="utf-8")
    source_manifest = {
        "schema_id": "source-sha256-v1",
        "frozen_sheet_module_sha256": EXPECTED_SHEET_MODULE_SHA256,
        "files": {
            str(path.relative_to(root)): _file_sha256(path)
            for path in _source_paths(root)
        },
    }
    (output_dir / "source_manifest.json").write_text(
        _strict_json(source_manifest), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# FluxV v5h8 incremental-sheet oracle\n\n"
        "Manufactured, observation-free mechanics evidence only. The isolated "
        "runner does not execute the forward-flight package initializer and "
        "contains no Ptera solve, target data, load, feedback, or aerodynamic-"
        "accuracy claim. A passing result does not promote the bounded affine "
        "clone/collapse oracle into a production sheet owner.\n",
        encoding="utf-8",
    )
    payloads = ("README.md", "source_manifest.json", "summary.json")
    result_manifest = {name: _file_sha256(output_dir / name) for name in payloads}
    (output_dir / "result_manifest.json").write_text(
        _strict_json({"schema_id": "result-sha256-v1", "files": result_manifest}),
        encoding="utf-8",
    )
    checksum_files = (*payloads, "result_manifest.json")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{_file_sha256(output_dir / name)}  {name}" for name in checksum_files
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    summary = write_artifact(arguments.output_dir)
    print(_strict_json(summary), end="")
    return 0 if bool(summary["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
