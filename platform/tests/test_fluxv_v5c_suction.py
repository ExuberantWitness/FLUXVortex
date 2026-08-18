from __future__ import annotations

import numpy as np
import pytest

from forward_flight_benchmarks.fluxv_v5c_suction import (
    RateSensitiveSuctionParameters,
    RateSensitiveSuctionState,
    project_axial_suction_loss_to_wind_axes,
    ramesh_axial_suction_coefficient,
    run_rate_sensitive_suction_history,
    step_rate_sensitive_suction,
)


def test_source_frozen_parameter_and_ramesh_suction_cap() -> None:
    parameters = RateSensitiveSuctionParameters()
    assert parameters.state_pole_per_convective_time == 0.5
    assert parameters.manifest()["delta_tau_definition"] == (
        "(|V_rel at 0.75c, perpendicular to local span|/c_local)*dt"
    )
    assert parameters.manifest()["observation_fit"] == "none"
    a0 = np.array([0.05, 0.2, -0.4])
    critical = np.array([0.1, 0.1, 0.2])
    expected = 2.0 * np.pi * np.array([0.05, 0.1, 0.2]) ** 2
    np.testing.assert_allclose(ramesh_axial_suction_coefficient(a0, critical), expected)
    np.testing.assert_allclose(
        ramesh_axial_suction_coefficient(a0, 0.1),
        2.0 * np.pi * np.minimum(np.abs(a0), 0.1) ** 2,
    )
    with pytest.raises(ValueError, match="source-frozen"):
        RateSensitiveSuctionParameters(state_pole_per_convective_time=0.3)


def test_disabled_step_is_bitwise_state_and_load_identity() -> None:
    state = RateSensitiveSuctionState.zeros(2)
    base = np.array([0.1, 0.2])
    after, record = step_rate_sensitive_suction(
        state,
        a0_pre=np.array([0.4, 0.5]),
        lesp_critical=np.array([0.1, 0.1]),
        delta_tau=np.array([0.02, 0.02]),
        base_suction_coefficient=base,
        enabled=False,
    )
    assert after is state
    np.testing.assert_array_equal(record["target_suction_coefficient"], base)
    np.testing.assert_array_equal(
        record["delta_suction_coefficient"], np.zeros_like(base)
    )
    assert record["state_updated"] is False
    assert record["diagnostic_status"] == "not_evaluated_disabled"


def test_disabled_step_does_not_evaluate_ill_scaled_flux_ratio() -> None:
    state = RateSensitiveSuctionState.zeros(1)
    after, record = step_rate_sensitive_suction(
        state,
        a0_pre=np.finfo(float).max,
        lesp_critical=np.finfo(float).tiny,
        delta_tau=0.1,
        base_suction_coefficient=0.2,
        enabled=False,
    )
    assert after is state
    np.testing.assert_array_equal(record["j"], [0.0])
    np.testing.assert_array_equal(record["delta_suction_coefficient"], [0.0])


def test_attached_and_constant_supercritical_limits_are_zero_loss() -> None:
    steps = 20
    attached = run_rate_sensitive_suction_history(
        a0_pre=np.full((steps, 1), 0.05),
        lesp_critical=np.full((steps, 1), 0.1),
        delta_tau=np.full((steps, 1), 0.02),
        base_suction_coefficient=np.full((steps, 1), 0.2),
    )
    np.testing.assert_array_equal(
        attached["delta_suction_coefficient"], np.zeros((steps, 1))
    )

    constant = run_rate_sensitive_suction_history(
        a0_pre=np.full((steps, 1), 0.2),
        lesp_critical=np.full((steps, 1), 0.1),
        delta_tau=np.full((steps, 1), 0.02),
        base_suction_coefficient=np.full((steps, 1), 0.2),
    )
    np.testing.assert_array_equal(
        constant["delta_suction_coefficient"], np.zeros((steps, 1))
    )


def test_rate_excitation_is_causal_bounded_and_axial_only() -> None:
    a0 = np.concatenate([np.linspace(0.05, 0.35, 24), np.linspace(0.35, 0.05, 24)])[
        :, None
    ]
    common = {
        "lesp_critical": np.full_like(a0, 0.1),
        "delta_tau": np.full_like(a0, 0.025),
        "base_suction_coefficient": np.full_like(a0, 0.2),
    }
    first = run_rate_sensitive_suction_history(a0_pre=a0, **common)
    altered = a0.copy()
    altered[30:] *= 1.8
    second = run_rate_sensitive_suction_history(a0_pre=altered, **common)
    np.testing.assert_array_equal(
        first["delta_suction_coefficient"][:30],
        second["delta_suction_coefficient"][:30],
    )
    assert np.max(first["loss_fraction"]) > 0.0
    assert np.all((first["loss_fraction"] >= 0.0) & (first["loss_fraction"] <= 1.0))
    assert np.all(first["target_suction_coefficient"] >= 0.0)
    assert np.all(first["target_suction_coefficient"] <= 0.2)
    np.testing.assert_array_equal(first["delta_normal_coefficient"], np.zeros_like(a0))


def test_causal_rate_does_not_rectify_a_bdf_step_overshoot() -> None:
    # J=[0,4,4,4].  The only physical discrete rate event is the first jump;
    # the plateau must not receive a second excitation from derivative overshoot.
    a0 = np.sqrt(np.array([0.0, 4.0, 4.0, 4.0]))[:, None]
    result = run_rate_sensitive_suction_history(
        a0_pre=a0,
        lesp_critical=np.ones_like(a0),
        delta_tau=np.full_like(a0, 0.1),
        base_suction_coefficient=np.full_like(a0, 0.2),
    )
    np.testing.assert_allclose(result["j_rate"][:, 0], [0.0, 40.0, 0.0, 0.0])
    assert result["chi_equilibrium"][1, 0] > 0.0
    np.testing.assert_array_equal(result["chi_equilibrium"][2:, 0], np.zeros(2))


def test_hot_state_decays_analytically_when_flux_rate_becomes_zero() -> None:
    state = RateSensitiveSuctionState(
        chi=np.array([0.8]),
        previous_j=np.array([4.0]),
        previous_previous_j=np.array([4.0]),
        previous_delta_tau=np.array([0.1]),
        step_count=2,
    )
    after, record = step_rate_sensitive_suction(
        state,
        a0_pre=0.2,
        lesp_critical=0.1,
        delta_tau=0.1,
        base_suction_coefficient=0.2,
    )
    expected_chi = 0.8 * np.exp(-0.5 * 0.1)
    np.testing.assert_allclose(after.chi, [expected_chi], rtol=0.0, atol=1.0e-15)
    np.testing.assert_array_equal(record["j_rate"], [0.0])
    np.testing.assert_array_equal(record["chi_equilibrium"], [0.0])
    np.testing.assert_allclose(record["loss_fraction"], [0.75 * expected_chi])
    assert 0.0 < record["loss_fraction"][0] < 0.75 * 0.8


def test_axial_projection_adds_drag_without_normal_force() -> None:
    result = project_axial_suction_loss_to_wind_axes(
        np.array([-0.1, -0.2]), np.deg2rad(np.array([0.0, 30.0]))
    )
    assert np.all(result["delta_CD"] >= 0.0)
    np.testing.assert_array_equal(result["delta_CN"], np.zeros(2))
    np.testing.assert_allclose(
        result["delta_CL"], result["delta_CS"] * np.sin(np.deg2rad([0.0, 30.0]))
    )
    with pytest.raises(ValueError, match="only reduce"):
        project_axial_suction_loss_to_wind_axes(0.1, 0.0)


def test_shape_and_threshold_fail_closed() -> None:
    state = RateSensitiveSuctionState.zeros(2)
    with pytest.raises(ValueError, match="strip topology"):
        step_rate_sensitive_suction(
            state,
            a0_pre=np.ones(2),
            lesp_critical=np.ones(3),
            delta_tau=np.ones(2),
            base_suction_coefficient=np.ones(2),
        )
    with pytest.raises(ValueError, match="must be positive"):
        ramesh_axial_suction_coefficient(np.ones(2), np.array([0.1, 0.0]))


def test_nonfinite_state_and_ill_scaled_flux_fail_closed() -> None:
    state = RateSensitiveSuctionState.zeros(1)
    bad_state = RateSensitiveSuctionState(
        chi=np.array([np.nan]),
        previous_j=state.previous_j,
        previous_previous_j=state.previous_previous_j,
        previous_delta_tau=state.previous_delta_tau,
        step_count=1,
    )
    with pytest.raises(ValueError, match="state chi must be finite"):
        step_rate_sensitive_suction(
            bad_state,
            a0_pre=0.2,
            lesp_critical=0.1,
            delta_tau=0.1,
            base_suction_coefficient=0.2,
        )
    with pytest.raises(ValueError, match="flux ratio is not finite"):
        step_rate_sensitive_suction(
            state,
            a0_pre=np.finfo(float).max,
            lesp_critical=np.finfo(float).tiny,
            delta_tau=0.1,
            base_suction_coefficient=0.2,
        )
