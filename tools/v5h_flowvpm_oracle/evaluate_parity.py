#!/usr/bin/env python3
"""Evaluate the frozen FLOWVPM Julia oracle against the Python v5h backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fluxvortex.rvpm_reference import (
    direct_gaussian_erf_velocity_jacobian,
    unpack_julia_column_major,
)
from fluxvortex.rvpm_transport import (
    corrected_pedrizzetti,
    lsrk3_step_direct,
    make_particle_state,
)

JULIA_J_NAMES = (
    "j11",
    "j21",
    "j31",
    "j12",
    "j22",
    "j32",
    "j13",
    "j23",
    "j33",
)
EXPECTED_ORACLE_SCHEMA = "flowvpm_oracle_v2"
EXPECTED_RK_A = np.asarray([0.0, -5.0 / 9.0, -153.0 / 128.0], dtype=np.float64)
EXPECTED_RK_B = np.asarray([1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0], dtype=np.float64)
EXPECTED_STAGE_SCHEMA = "pre=state_only;rhs=UJ_at_pre;post=state_only"


def vectors(group: dict, prefix: str) -> np.ndarray:
    return np.column_stack([group[f"{prefix}_{axis}"] for axis in "xyz"]).astype(
        np.float64
    )


def jacobian(group: dict) -> np.ndarray:
    flat = np.column_stack([group[name] for name in JULIA_J_NAMES]).astype(np.float64)
    return unpack_julia_column_major(flat)


def storage(group: dict) -> np.ndarray:
    return np.column_stack([group[f"m{index:02d}"] for index in range(1, 10)]).astype(
        np.float64
    )


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm((actual - reference).ravel()))
    return numerator if denominator == 0.0 else numerator / denominator


def row_relative_l2_max(actual: np.ndarray, reference: np.ndarray) -> float:
    actual_rows = np.asarray(actual, dtype=np.float64).reshape(actual.shape[0], -1)
    reference_rows = np.asarray(reference, dtype=np.float64).reshape(
        reference.shape[0], -1
    )
    numerator = np.linalg.norm(actual_rows - reference_rows, axis=1)
    denominator = np.linalg.norm(reference_rows, axis=1)
    relative = numerator.copy()
    nonzero = denominator != 0.0
    relative[nonzero] /= denominator[nonzero]
    return float(np.max(relative, initial=0.0))


def count_nonfinite(value: object) -> int:
    if isinstance(value, dict):
        return sum(count_nonfinite(child) for child in value.values())
    if isinstance(value, list):
        return sum(count_nonfinite(child) for child in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(not np.isfinite(float(value)))
    return 0


def rk_config_checks(
    config: dict,
    *,
    expected_uinf_contract: str,
    timevarying: bool,
) -> dict[str, bool]:
    checks = {
        "steps_eq_2": np.array_equal(
            np.asarray(config["steps"], dtype=np.int64), np.asarray([2], dtype=np.int64)
        ),
        "dt_finite_positive": bool(
            len(config["dt"]) == 1
            and np.isfinite(float(config["dt"][0]))
            and float(config["dt"][0]) > 0.0
        ),
        "rk_a_exact": np.array_equal(
            np.asarray(config["rk_a"], dtype=np.float64), EXPECTED_RK_A
        ),
        "rk_b_exact": np.array_equal(
            np.asarray(config["rk_b"], dtype=np.float64), EXPECTED_RK_B
        ),
        "formulation_f_eq_0": np.array_equal(
            np.asarray(config["formulation_f"], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
        ),
        "formulation_g_eq_0p2": np.array_equal(
            np.asarray(config["formulation_g"], dtype=np.float64),
            np.asarray([0.2], dtype=np.float64),
        ),
        "transposed_eq_1": np.array_equal(
            np.asarray(config["transposed"], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        ),
        "sfs_disabled": np.array_equal(
            np.asarray(config["sfs_enabled"], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        ),
        "viscosity_disabled": np.array_equal(
            np.asarray(config["viscosity_enabled"], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        ),
        "relaxation_disabled": np.array_equal(
            np.asarray(config["relaxation_enabled"], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        ),
        "stage_schema_exact": config["stage_schema"] == EXPECTED_STAGE_SCHEMA,
        "uinf_evaluation_contract_exact": (
            config["uinf_evaluation_contract"] == expected_uinf_contract
        ),
    }
    if timevarying:
        base = np.asarray(
            [config[f"uinf_base_{axis}"][0] for axis in "xyz"], dtype=np.float64
        )
        slope = np.asarray(
            [config[f"uinf_slope_{axis}"][0] for axis in "xyz"], dtype=np.float64
        )
        checks.update(
            {
                "uinf_model_exact": config["uinf_model"] == "affine_in_field_time",
                "uinf_base_finite": bool(np.all(np.isfinite(base))),
                "uinf_slope_finite_nonzero": bool(
                    np.all(np.isfinite(slope)) and np.any(slope != 0.0)
                ),
            }
        )
    else:
        freestream = np.asarray(
            [config[f"uinf_{axis}"][0] for axis in "xyz"], dtype=np.float64
        )
        checks["constant_uinf_finite"] = bool(np.all(np.isfinite(freestream)))
    return {name: bool(value) for name, value in checks.items()}


def evaluate_uj_fixture(fixture: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    inputs = fixture["input"]
    reference = fixture["output"]
    actual = direct_gaussian_erf_velocity_jacobian(
        vectors(inputs, "x"),
        vectors(inputs, "gamma"),
        np.asarray(inputs["sigma"], dtype=np.float64),
    )
    reference_velocity = vectors(reference, "u")
    reference_jacobian = jacobian(reference)
    roles = np.asarray(inputs["role_code"], dtype=np.int64)
    probe_mask = roles == 0
    if not np.any(probe_mask):
        raise ValueError("U/J fixture must contain at least one zero-strength probe")
    metrics = {
        "particle_count": int(roles.size),
        "probe_count": int(np.count_nonzero(probe_mask)),
        "velocity_relative_l2": relative_l2(actual.velocity, reference_velocity),
        "jacobian_relative_l2": relative_l2(actual.jacobian, reference_jacobian),
        "probe_velocity_relative_l2": relative_l2(
            actual.velocity[probe_mask], reference_velocity[probe_mask]
        ),
        "probe_jacobian_relative_l2": relative_l2(
            actual.jacobian[probe_mask], reference_jacobian[probe_mask]
        ),
        "probe_velocity_row_relative_l2_max": row_relative_l2_max(
            actual.velocity[probe_mask], reference_velocity[probe_mask]
        ),
        "probe_jacobian_row_relative_l2_max": row_relative_l2_max(
            actual.jacobian[probe_mask], reference_jacobian[probe_mask]
        ),
    }
    return metrics, actual.velocity, actual.jacobian


def evaluate_rk_fixture(
    fixture: dict,
    *,
    timevarying: bool,
) -> tuple[dict, dict[str, bool]]:
    inputs = fixture["input"]
    config = fixture["config"]
    expected_contract = (
        "once_per_step_at_field_time_before_rk_stages"
        if timevarying
        else "constant_vector_once_per_step"
    )
    config_checks = rk_config_checks(
        config,
        expected_uinf_contract=expected_contract,
        timevarying=timevarying,
    )
    state = make_particle_state(
        vectors(inputs, "x"),
        vectors(inputs, "gamma"),
        np.asarray(inputs["sigma"], dtype=np.float64),
    )
    delta_time = float(config["dt"][0])
    if timevarying:
        uinf_base = np.asarray(
            [config[f"uinf_base_{axis}"][0] for axis in "xyz"], dtype=np.float64
        )
        uinf_slope = np.asarray(
            [config[f"uinf_slope_{axis}"][0] for axis in "xyz"], dtype=np.float64
        )
    else:
        constant_uinf = np.asarray(
            [config[f"uinf_{axis}"][0] for axis in "xyz"], dtype=np.float64
        )

    errors = {
        "position_relative_l2_max": 0.0,
        "gamma_relative_l2_max": 0.0,
        "sigma_relative_l2_max": 0.0,
        "storage_relative_l2_max": 0.0,
        "rhs_velocity_relative_l2_max": 0.0,
        "rhs_jacobian_relative_l2_max": 0.0,
        "field_time_before_absolute_error_max": 0.0,
        "field_time_after_absolute_error_max": 0.0,
        "uinf_used_absolute_error_max": 0.0,
    }
    step_counter_exact = True
    for step in range(1, 3):
        step_reference = fixture[f"step_{step:02d}"]
        expected_time_before = (step - 1) * delta_time
        recorded_time_before = float(step_reference["field_time_before"][0])
        errors["field_time_before_absolute_error_max"] = max(
            errors["field_time_before_absolute_error_max"],
            abs(recorded_time_before - expected_time_before),
        )
        freestream = (
            uinf_base + uinf_slope * recorded_time_before
            if timevarying
            else constant_uinf
        )
        recorded_uinf = np.asarray(
            [step_reference[f"uinf_used_{axis}"][0] for axis in "xyz"],
            dtype=np.float64,
        )
        errors["uinf_used_absolute_error_max"] = max(
            errors["uinf_used_absolute_error_max"],
            float(np.max(np.abs(recorded_uinf - freestream))),
        )
        state, stages = lsrk3_step_direct(
            state,
            delta_time,
            freestream_velocity=freestream,
        )
        for stage_index, stage in enumerate(stages, start=1):
            reference = step_reference[f"stage_{stage_index:02d}"]
            for actual_state, reference_name, actual_storage in (
                (stage.pre, "pre", stage.storage_pre),
                (stage.post, "post", stage.storage_post),
            ):
                reference_state = reference[reference_name]
                errors["position_relative_l2_max"] = max(
                    errors["position_relative_l2_max"],
                    relative_l2(actual_state.positions, vectors(reference_state, "x")),
                )
                errors["gamma_relative_l2_max"] = max(
                    errors["gamma_relative_l2_max"],
                    relative_l2(actual_state.gamma, vectors(reference_state, "gamma")),
                )
                errors["sigma_relative_l2_max"] = max(
                    errors["sigma_relative_l2_max"],
                    relative_l2(
                        actual_state.sigma,
                        np.asarray(reference_state["sigma"], dtype=np.float64),
                    ),
                )
                errors["storage_relative_l2_max"] = max(
                    errors["storage_relative_l2_max"],
                    relative_l2(actual_storage, storage(reference_state)),
                )
            reference_rhs = reference["rhs"]
            errors["rhs_velocity_relative_l2_max"] = max(
                errors["rhs_velocity_relative_l2_max"],
                relative_l2(stage.rhs.velocity, vectors(reference_rhs, "u")),
            )
            errors["rhs_jacobian_relative_l2_max"] = max(
                errors["rhs_jacobian_relative_l2_max"],
                relative_l2(stage.rhs.jacobian, jacobian(reference_rhs)),
            )
        errors["field_time_after_absolute_error_max"] = max(
            errors["field_time_after_absolute_error_max"],
            abs(float(step_reference["field_time_after"][0]) - step * delta_time),
        )
        step_counter_exact = step_counter_exact and (
            int(step_reference["field_step_after"][0]) == step
        )
    errors["field_step_counter_exact"] = bool(step_counter_exact)
    return errors, config_checks


def evaluate(payload: dict) -> dict:
    fixtures = payload["fixtures"]
    oracle_schema_version = payload["meta"]["schema_version"]
    schema_checks = {
        "oracle_schema_v2": oracle_schema_version == EXPECTED_ORACLE_SCHEMA,
        "stage_state_schema_exact": (
            payload["meta"]["stage_state_schema"]
            == "pre/state_only;rhs/UJ_only;post/state_only"
        ),
        "rhs_evaluation_state_is_pre": payload["meta"]["rhs_evaluation_state"]
        == "stage_pre",
    }
    uj_metrics, _, _ = evaluate_uj_fixture(fixtures["uj_direct_gauserf"])
    nearfield_metrics, _, _ = evaluate_uj_fixture(
        fixtures["uj_direct_gauserf_nearfield_sweep"]
    )
    nearfield_input = fixtures["uj_direct_gauserf_nearfield_sweep"]["input"]
    nearfield_positions = vectors(nearfield_input, "x")
    nearfield_sigma = np.asarray(nearfield_input["sigma"], dtype=np.float64)
    source_id = int(nearfield_input["source_particle_id"][0])
    probe_ids = np.asarray(nearfield_input["probe_particle_id"], dtype=np.int64)
    source_index = source_id - 1
    probe_indices = probe_ids - 1
    recorded_ratios = np.asarray(
        nearfield_input["r_over_source_sigma"], dtype=np.float64
    )
    geometric_ratios = (
        np.linalg.norm(
            nearfield_positions[probe_indices] - nearfield_positions[source_index],
            axis=1,
        )
        / nearfield_sigma[source_index]
    )
    nearfield_metrics.update(
        {
            "r_over_sigma_min": float(np.min(recorded_ratios)),
            "r_over_sigma_max": float(np.max(recorded_ratios)),
            "geometry_ratio_absolute_error_max": float(
                np.max(np.abs(geometric_ratios - recorded_ratios))
            ),
            "probe_id_contract_exact": bool(
                np.array_equal(
                    probe_ids,
                    np.flatnonzero(
                        np.asarray(nearfield_input["role_code"], dtype=np.int64) == 0
                    )
                    + 1,
                )
                and source_id == 1
            ),
        }
    )

    rk_errors, fixed_config_checks = evaluate_rk_fixture(
        fixtures["rk3_rvpm_direct_gauserf"], timevarying=False
    )
    timevarying_rk_errors, timevarying_config_checks = evaluate_rk_fixture(
        fixtures["rk3_timevarying_uinf_direct_gauserf"], timevarying=True
    )
    config_checks = {
        "schema": schema_checks,
        "fixed_rk": fixed_config_checks,
        "timevarying_uinf_rk": timevarying_config_checks,
    }
    config_contract_pass = all(
        value
        for check_group in config_checks.values()
        for value in check_group.values()
    )

    relaxation_fixture = fixtures["corrected_pedrizzetti"]
    relaxation_relative_l2_max = 0.0
    relaxation_norm_relative_error_max = 0.0
    for case_name in ("case_001", "case_002", "case_003", "case_004"):
        case = relaxation_fixture[case_name]
        before = vectors(case, "gamma_before")
        reference = vectors(case, "gamma_after")
        actual = corrected_pedrizzetti(
            before,
            unpack_julia_column_major(
                np.asarray(case["j_column_major"], dtype=np.float64)[None, :]
            ),
            float(case["alpha"][0]),
        )
        relaxation_relative_l2_max = max(
            relaxation_relative_l2_max,
            relative_l2(actual, reference),
        )
        before_norm = float(np.linalg.norm(before))
        relaxation_norm_relative_error_max = max(
            relaxation_norm_relative_error_max,
            abs(float(np.linalg.norm(actual)) - before_norm) / before_norm,
        )

    nonfinite_count = count_nonfinite(fixtures)
    gates = {
        "config_contract_all_true": config_contract_pass,
        "velocity_relative_l2_le_1e-12": uj_metrics["velocity_relative_l2"] <= 1e-12,
        "jacobian_relative_l2_le_1e-11": uj_metrics["jacobian_relative_l2"] <= 1e-11,
        "probe_velocity_relative_l2_le_1e-12": uj_metrics["probe_velocity_relative_l2"]
        <= 1e-12,
        "probe_jacobian_relative_l2_le_1e-11": uj_metrics["probe_jacobian_relative_l2"]
        <= 1e-11,
        "nearfield_probe_velocity_row_relative_l2_max_le_1e-9": nearfield_metrics[
            "probe_velocity_row_relative_l2_max"
        ]
        <= 1e-9,
        "nearfield_probe_jacobian_row_relative_l2_max_le_1e-9": nearfield_metrics[
            "probe_jacobian_row_relative_l2_max"
        ]
        <= 1e-9,
        "nearfield_geometry_ratio_absolute_error_max_le_1e-14": nearfield_metrics[
            "geometry_ratio_absolute_error_max"
        ]
        <= 1e-14,
        "nearfield_probe_contract_exact": (
            nearfield_metrics["probe_count"] == 9
            and nearfield_metrics["probe_id_contract_exact"]
            and nearfield_metrics["r_over_sigma_min"] == 1e-4
            and nearfield_metrics["r_over_sigma_max"] == 2.0
        ),
        "rk_state_relative_l2_max_le_1e-11": max(
            rk_errors["position_relative_l2_max"],
            rk_errors["gamma_relative_l2_max"],
            rk_errors["sigma_relative_l2_max"],
            rk_errors["storage_relative_l2_max"],
        )
        <= 1e-11,
        "rk_rhs_relative_l2_max_le_1e-11": max(
            rk_errors["rhs_velocity_relative_l2_max"],
            rk_errors["rhs_jacobian_relative_l2_max"],
        )
        <= 1e-11,
        "rk_clock_and_uinf_contract_le_1e-15": max(
            rk_errors["field_time_before_absolute_error_max"],
            rk_errors["field_time_after_absolute_error_max"],
            rk_errors["uinf_used_absolute_error_max"],
        )
        <= 1e-15
        and rk_errors["field_step_counter_exact"],
        "timevarying_uinf_rk_state_relative_l2_max_le_1e-11": max(
            timevarying_rk_errors["position_relative_l2_max"],
            timevarying_rk_errors["gamma_relative_l2_max"],
            timevarying_rk_errors["sigma_relative_l2_max"],
            timevarying_rk_errors["storage_relative_l2_max"],
        )
        <= 1e-11,
        "timevarying_uinf_rk_rhs_relative_l2_max_le_1e-11": max(
            timevarying_rk_errors["rhs_velocity_relative_l2_max"],
            timevarying_rk_errors["rhs_jacobian_relative_l2_max"],
        )
        <= 1e-11,
        "timevarying_uinf_clock_and_contract_le_1e-15": max(
            timevarying_rk_errors["field_time_before_absolute_error_max"],
            timevarying_rk_errors["field_time_after_absolute_error_max"],
            timevarying_rk_errors["uinf_used_absolute_error_max"],
        )
        <= 1e-15
        and timevarying_rk_errors["field_step_counter_exact"],
        "relaxation_relative_l2_max_le_1e-12": relaxation_relative_l2_max <= 1e-12,
        "relaxation_norm_relative_error_max_le_1e-14": (
            relaxation_norm_relative_error_max <= 1e-14
        ),
        "nonfinite_count_eq_0": nonfinite_count == 0,
        "clip_count_eq_0_by_construction": True,
    }
    return {
        "schema_version": "fluxv_v5h_r0_r1_metrics_v2",
        "oracle_schema_version": oracle_schema_version,
        "config_contract": config_checks,
        "config_contract_pass": config_contract_pass,
        "uj": uj_metrics,
        "nearfield_uj": nearfield_metrics,
        "velocity_relative_l2": uj_metrics["velocity_relative_l2"],
        "jacobian_relative_l2": uj_metrics["jacobian_relative_l2"],
        "rk": rk_errors,
        "rk_timevarying_uinf": timevarying_rk_errors,
        "relaxation_gamma_relative_l2_max": relaxation_relative_l2_max,
        "relaxation_norm_relative_error_max": relaxation_norm_relative_error_max,
        "nonfinite_count": nonfinite_count,
        "clip_count": 0,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    payload = json.loads(arguments.oracle.read_text(encoding="utf-8"))
    metrics = evaluate(payload)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
