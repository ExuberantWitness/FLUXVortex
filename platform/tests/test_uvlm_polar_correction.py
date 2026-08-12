from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.cases import YANG_2025
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement
from forward_flight_benchmarks.uvlm_polar_correction import (
    FullAnglePolarParameters,
    add_constant_profile_drag,
    finite_wing_lift_slope,
    full_angle_polar_residual_coefficients,
    movement_polar_residual,
)


def test_finite_wing_slope_and_source_frozen_defaults() -> None:
    parameters = FullAnglePolarParameters()
    assert np.isclose(parameters.section_lift_slope_per_rad, 2.0 * np.pi)
    assert parameters.drag_coefficient_at_90_deg == 1.20
    assert parameters.section_velocity_reference_fraction_chord == 0.25
    expected = 2.0 * np.pi / (1.0 + 2.0 / 3.0)
    assert np.isclose(finite_wing_lift_slope(3.0), expected)
    assert parameters.manifest()["observation_fit"] == "none"


def test_full_angle_residual_has_required_symmetry_and_limits() -> None:
    alpha = np.deg2rad(np.array([-90.0, -20.0, 0.0, 20.0, 90.0]))
    delta_cl, delta_cd = full_angle_polar_residual_coefficients(alpha, 3.0)
    np.testing.assert_allclose(delta_cl, -delta_cl[::-1], atol=1.0e-14)
    np.testing.assert_allclose(delta_cd, delta_cd[::-1], atol=1.0e-14)
    assert delta_cl[2] == 0.0
    assert delta_cd[2] == 0.0
    assert np.isclose(delta_cd[0], 1.20)


def test_yang_movement_residual_is_finite_and_case_agnostic() -> None:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    result = movement_polar_residual(
        movement,
        source_cycle_step_range=(20, 39),
        period_s=YANG_2025.period_s,
        freestream_m_s=YANG_2025.freestream_m_s,
        rho_kg_m3=YANG_2025.rho_kg_m3,
        aspect_ratio=YANG_2025.aspect_ratio,
        output_samples=20,
    )
    assert result["delta_lift_n"].shape == (20,)
    assert result["delta_drag_n"].shape == (20,)
    assert np.all(np.isfinite(result["delta_force_g_n"]))
    assert result["strip_count"] == 4
    assert result["parameters"]["observation_fit"] == "none"
    assert "case_id" not in result
    assert "paper" not in result["parameters"]
    assert result["mean_delta_drag_n"] > 0.0


def test_section_velocity_reference_is_executed_not_manifest_only() -> None:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    common = {
        "movement": movement,
        "source_cycle_step_range": (20, 39),
        "period_s": YANG_2025.period_s,
        "freestream_m_s": YANG_2025.freestream_m_s,
        "rho_kg_m3": YANG_2025.rho_kg_m3,
        "aspect_ratio": YANG_2025.aspect_ratio,
        "output_samples": 20,
    }
    quarter = movement_polar_residual(
        **common,
        parameters=FullAnglePolarParameters(
            section_velocity_reference_fraction_chord=0.25
        ),
    )
    trailing = movement_polar_residual(
        **common,
        parameters=FullAnglePolarParameters(
            section_velocity_reference_fraction_chord=0.75
        ),
    )
    assert not np.allclose(
        quarter["relative_speed_m_s"], trailing["relative_speed_m_s"]
    )


def test_zero_incidence_static_panel_has_zero_residual() -> None:
    delta_cl, delta_cd = full_angle_polar_residual_coefficients(0.0, 2.0)
    assert float(delta_cl) == 0.0
    assert float(delta_cd) == 0.0


def test_constant_profile_drag_zero_reduces_exactly_and_positive_adds_drag() -> None:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    residual = movement_polar_residual(
        movement,
        source_cycle_step_range=(20, 39),
        period_s=YANG_2025.period_s,
        freestream_m_s=YANG_2025.freestream_m_s,
        rho_kg_m3=YANG_2025.rho_kg_m3,
        aspect_ratio=YANG_2025.aspect_ratio,
        output_samples=20,
    )
    phase = np.asarray(residual["phase"])
    baseline = {
        "phase": phase,
        "lift_n": np.linspace(-1.0, 1.0, phase.size),
        "drag_n": np.linspace(0.1, 0.2, phase.size),
        "mean_lift_n": 0.0,
        "mean_drag_n": 0.15,
    }
    common = {
        "rho_kg_m3": YANG_2025.rho_kg_m3,
        "freestream_m_s": YANG_2025.freestream_m_s,
        "area_m2": YANG_2025.area_m2,
    }
    zero = add_constant_profile_drag(baseline, residual, coefficient=0.0, **common)
    np.testing.assert_array_equal(zero["lift_n"], baseline["lift_n"])
    np.testing.assert_array_equal(zero["drag_n"], baseline["drag_n"])
    assert zero["mean_lift_n"] == baseline["mean_lift_n"]
    assert zero["mean_drag_n"] == baseline["mean_drag_n"]

    positive = add_constant_profile_drag(
        baseline, residual, coefficient=0.057, **common
    )
    assert np.mean(positive["drag_n"] - baseline["drag_n"]) > 0.0
    np.testing.assert_allclose(positive["thrust_n"], -positive["drag_n"])
    np.testing.assert_allclose(
        positive["CD"],
        positive["drag_n"]
        / (
            0.5
            * YANG_2025.rho_kg_m3
            * YANG_2025.freestream_m_s**2
            * YANG_2025.area_m2
        ),
    )
