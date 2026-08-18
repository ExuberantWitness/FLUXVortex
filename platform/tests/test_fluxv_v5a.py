from __future__ import annotations

import inspect

import numpy as np
import pytest

from forward_flight_benchmarks.fluxv_v5a import (
    DEFAULT_V5A_PARAMETERS,
    FluxVV5AParameters,
    apply_fluxv_v5a_ledger,
    assemble_strip_force_ledger,
    convective_high_pass,
    convective_increment,
    equilibrium_section_residual,
    periodic_convergence_diagnostic,
    project_ldvm_pair_components,
    resolve_normal_suction_to_lift_drag,
)


def _pair(components: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return {
        "delta": {
            "CNc": components[..., 0],
            "CNnc": components[..., 1],
            "CNnonl": components[..., 2],
            "CSf": components[..., 3],
        }
    }


def test_default_is_the_preregistered_single_convective_time() -> None:
    assert DEFAULT_V5A_PARAMETERS.lambda_tau == 1.0
    manifest = DEFAULT_V5A_PARAMETERS.manifest()
    assert manifest["observation_fit"] == "none"
    assert "m_N,m_S" in manifest["state_contract"]
    with pytest.raises(ValueError):
        FluxVV5AParameters(lambda_tau=-0.01)


def test_equilibrium_is_a_direct_section_difference_without_projection_or_qs() -> None:
    result = equilibrium_section_residual(
        normal_equilibrium_coefficient=np.array([0.9, 0.5]),
        normal_attached_coefficient=np.array([0.8, 0.7]),
        drag_equilibrium_coefficient=np.array([0.14, 0.11]),
        drag_baseline_coefficient=0.04,
    )
    np.testing.assert_allclose(result["coefficient"], [[0.1, 0.1], [-0.2, 0.07]])
    assert result["spatial_mapping"] == "direct_section_to_strip_no_ldvm_projection"
    assert result["units"] == "dimensionless_coefficient_before_qS"
    assert (
        "aspect_ratio" not in inspect.signature(equilibrium_section_residual).parameters
    )


def test_project_then_filter_equals_filter_then_constant_component_projection() -> None:
    time = np.arange(7.0)
    components = np.stack(
        (
            0.1 + 0.03 * time,
            -0.04 + 0.01 * time,
            0.02 * np.sin(time),
            -0.08 + 0.015 * time,
        ),
        axis=-1,
    )
    chi = np.linspace(0.01, 0.2, time.size)

    projected_raw = project_ldvm_pair_components(_pair(components), aspect_ratio=4.0)
    projected_filter = convective_high_pass(
        projected_raw["coefficient_ns"], delta_chi=chi, lambda_tau=1.0
    )

    component_filter = convective_high_pass(components, delta_chi=chi, lambda_tau=1.0)
    projected_transient = project_ldvm_pair_components(
        _pair(component_filter["transient_coefficient"]), aspect_ratio=4.0
    )
    projected_state = project_ldvm_pair_components(
        _pair(component_filter["low_pass_state_after"]), aspect_ratio=4.0
    )

    np.testing.assert_allclose(
        projected_filter["transient_coefficient"],
        projected_transient["coefficient_ns"],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        projected_filter["low_pass_state_after"],
        projected_state["coefficient_ns"],
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_incidence_rotation_occurs_after_filter_and_is_time_local() -> None:
    raw_ns = np.array([[1.0, 0.2], [1.0, 0.2], [1.0, 0.2]])
    alpha = np.deg2rad(np.array([0.0, 30.0, -20.0]))
    filtered = convective_high_pass(raw_ns, delta_chi=0.1, lambda_tau=1.0)
    resolved = resolve_normal_suction_to_lift_drag(
        filtered["transient_coefficient"], alpha
    )
    normal = filtered["transient_coefficient"][:, 0]
    suction = filtered["transient_coefficient"][:, 1]
    np.testing.assert_allclose(
        resolved["delta_CL"], normal * np.cos(alpha) + suction * np.sin(alpha)
    )
    np.testing.assert_allclose(
        resolved["delta_CD"], normal * np.sin(alpha) - suction * np.cos(alpha)
    )
    assert resolved["rotation_semantics"].endswith("after high-pass")


def test_convective_increment_accepts_array_kinematics_and_applies_floor() -> None:
    increment = convective_increment(
        relative_speed_m_s=np.array([[2.0, 0.0], [4.0, -1.0]]),
        delta_time_s=np.full((2, 2), 0.05),
        chord_m=np.array([[0.5, 0.25], [1.0, 0.5]]),
        reference_speed_m_s=10.0,
    )
    np.testing.assert_allclose(
        increment,
        [[0.2, 2.0e-6], [0.2, 0.1]],
    )


def test_high_pass_exact_limits_and_constant_discrepancy_decay() -> None:
    raw = np.tile(np.array([0.8, -0.3]), (80, 1))
    zero = convective_high_pass(np.zeros_like(raw), delta_chi=0.1)
    np.testing.assert_array_equal(zero["transient_coefficient"], np.zeros_like(raw))
    np.testing.assert_array_equal(zero["final_state"], np.zeros(2))

    instantaneous = convective_high_pass(raw, delta_chi=0.1, lambda_tau=0.0)
    np.testing.assert_array_equal(
        instantaneous["transient_coefficient"], np.zeros_like(raw)
    )
    np.testing.assert_array_equal(instantaneous["final_state"], raw[-1])

    frozen = convective_high_pass(raw, delta_chi=0.1, lambda_tau=np.inf)
    np.testing.assert_array_equal(frozen["transient_coefficient"], raw)
    np.testing.assert_array_equal(frozen["final_state"], np.zeros(2))

    finite = convective_high_pass(raw, delta_chi=0.2, lambda_tau=1.0)
    assert np.max(np.abs(finite["transient_coefficient"][-1])) < 1.0e-6
    np.testing.assert_allclose(finite["final_state"], raw[-1], atol=1.0e-6)


def test_strip_force_ledger_multiplies_local_qs_exactly_once_and_closes() -> None:
    baseline = np.zeros((2, 2, 2))
    equilibrium = np.array(
        [
            [[0.1, 0.02], [0.2, 0.03]],
            [[0.3, 0.04], [0.4, 0.05]],
        ]
    )
    transient = 0.5 * equilibrium
    pressure = np.array([[10.0, 20.0], [30.0, 40.0]])
    area = np.array([[0.5, 0.25], [0.2, 0.1]])
    ledger = assemble_strip_force_ledger(
        baseline,
        dynamic_pressure_pa=pressure,
        strip_area_m2=area,
        equilibrium_residual_coefficient=equilibrium,
        transient_residual_coefficient=transient,
    )
    expected = (pressure * area)[..., None] * (equilibrium + transient)
    np.testing.assert_allclose(ledger["total_force_n"], expected)
    np.testing.assert_array_equal(ledger["ledger_residual_n"], 0.0)
    assert ledger["ledger_relative_residual"] == 0.0
    assert ledger["qS_multiplication_count"] == 1

    doubled_pressure = assemble_strip_force_ledger(
        baseline,
        dynamic_pressure_pa=2.0 * pressure,
        strip_area_m2=area,
        equilibrium_residual_coefficient=equilibrium,
        transient_residual_coefficient=transient,
    )
    np.testing.assert_allclose(doubled_pressure["total_force_n"], 2.0 * expected)


def test_module_off_and_attached_limits_are_bitwise_uvlm() -> None:
    baseline = np.array([[1.0, -0.0], [-2.5, 3.25]], dtype=np.float64)
    nonzero = np.full_like(baseline, 0.4)
    off = assemble_strip_force_ledger(
        baseline,
        dynamic_pressure_pa=10.0,
        strip_area_m2=0.2,
        equilibrium_residual_coefficient=nonzero,
        transient_residual_coefficient=nonzero,
        mode="off",
    )
    np.testing.assert_array_equal(
        off["total_force_n"].view(np.uint64), baseline.view(np.uint64)
    )

    attached = assemble_strip_force_ledger(
        baseline,
        dynamic_pressure_pa=10.0,
        strip_area_m2=0.2,
        equilibrium_residual_coefficient=np.zeros_like(baseline),
        transient_residual_coefficient=np.zeros_like(baseline),
        mode="full",
    )
    np.testing.assert_array_equal(
        attached["total_force_n"].view(np.uint64), baseline.view(np.uint64)
    )


def test_periodic_convergence_requires_state_and_load_mean() -> None:
    cycle = np.array([[0.1, -0.2], [0.2, -0.1], [0.3, 0.0], [0.2, 0.1]])
    state = np.vstack((cycle, cycle, cycle))
    loads = np.vstack((cycle, cycle, cycle))
    report = periodic_convergence_diagnostic(
        state,
        loads,
        steps_per_cycle=4,
        required_consecutive_passes=2,
    )
    assert report["passed"]
    assert report["final_pass_streak"] == 2

    loads[-4:] += 0.1
    failed = periodic_convergence_diagnostic(state, loads, steps_per_cycle=4)
    assert not failed["passed"]
    assert failed["transitions"][-1]["load_mean_relative_change"] > 1.0e-4


def test_projected_integrated_compatibility_adapter_has_fixed_auditable_keys() -> None:
    baseline = {"CL": np.array([0.2, 0.3]), "CD": np.array([0.1, 0.2])}
    equilibrium = {"CL": np.array([0.01, 0.02]), "CD": np.array([0.03, 0.04])}
    raw = {"CL": np.array([0.4, 0.2]), "CD": np.array([-0.1, 0.3])}
    result = apply_fluxv_v5a_ledger(
        baseline,
        equilibrium,
        raw,
        delta_chi=np.array([0.1, 0.2]),
    )
    required = {
        "CL",
        "CD",
        "equilibrium_CL",
        "equilibrium_CD",
        "raw_ldvm_CL",
        "raw_ldvm_CD",
        "state_CL",
        "state_CD",
        "transient_CL",
        "transient_CD",
        "ledger_residual_CL",
        "ledger_residual_CD",
    }
    assert required <= result.keys()
    assert result["compatibility_scope"] == "projected_integrated_proxy"
    assert result["canonical_strip_gate"] == "blocked"
    np.testing.assert_array_equal(result["ledger_residual_CL"], 0.0)
    np.testing.assert_array_equal(result["ledger_residual_CD"], 0.0)


def test_public_core_has_no_case_or_observation_dispatch() -> None:
    source = inspect.getsource(
        __import__("forward_flight_benchmarks.fluxv_v5a", fromlist=["fluxv_v5a"])
    )
    assert "case_id" not in source
    assert "observation_residual" not in source
