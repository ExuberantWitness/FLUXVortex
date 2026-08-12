from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.cases import (
    IZRAELEVITZ_2017_FIG11,
    YANG_2025,
)
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_fig11_movement,
    build_yang2025_movement,
)
from forward_flight_benchmarks.ullt_attached import (
    OneStateULLTParameters,
    _periodic_derivative,
    blend_attached_with_uvlm_polar,
    lifting_surface_correction_gain,
    movement_one_state_ullt,
    smooth_separation_fraction,
)


def test_published_parameters_and_equation_42_gain() -> None:
    parameters = OneStateULLTParameters()
    assert parameters.lift_indicial_amplitude == -0.5
    assert parameters.lift_indicial_decay == -0.25
    assert parameters.circulation_indicial_amplitude == -0.8
    assert parameters.circulation_indicial_decay == -0.25
    assert parameters.lift_initial_value == 0.5
    assert np.isclose(parameters.circulation_initial_value, 0.2)
    assert np.isclose(parameters.circulation_to_lift_state_gain, 1.6)
    assert parameters.lifting_surface_correction_k == 13.5
    assert parameters.flat_plate_added_mass_factor == 0.85
    assert parameters.manifest()["observation_fit"] == "none"

    numerator = 1.0 / (2.0 * np.pi) + 1.0 / (3.0 * np.pi)
    expected = numerator / (numerator + 13.5 * np.pi / (180.0 * 3.0**2))
    assert np.isclose(lifting_surface_correction_gain(3.0), expected)


def test_periodic_derivative_handles_a_duplicated_phase_zero_endpoint() -> None:
    steps = 40
    delta_time = 1.0 / steps
    phase = np.arange(3 * steps + 1, dtype=float) / steps
    values = np.sin(2.0 * np.pi * phase)
    derivative = _periodic_derivative(values, delta_time, steps)
    expected = 2.0 * np.pi * np.cos(2.0 * np.pi * phase)
    np.testing.assert_allclose(derivative, expected, atol=0.027, rtol=0.0)
    assert abs(derivative[-2] - expected[-2]) < 0.027
    assert abs(derivative[-1] - expected[-1]) < 0.027


def test_selected_last_cycle_is_invariant_to_an_appended_cycle() -> None:
    case = YANG_2025
    short, _ = build_yang2025_movement(
        15.0, settings=(2, 4, 20, 2, 2)
    )
    extended, _ = build_yang2025_movement(
        15.0, settings=(2, 4, 20, 3, 2)
    )
    common = dict(
        source_cycle_step_range=(20, 39),
        period_s=case.period_s,
        freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3,
        aspect_ratio=case.aspect_ratio,
        area_m2=case.area_m2,
        output_samples=20,
    )
    short_result = movement_one_state_ullt(short, **common)
    extended_result = movement_one_state_ullt(extended, **common)
    np.testing.assert_allclose(
        short_result["force_g_n"],
        extended_result["force_g_n"],
        atol=1.0e-12,
        rtol=0.0,
    )


def test_fig11_one_state_prototype_is_finite_symmetric_and_case_agnostic() -> None:
    movement, _ = build_izraelevitz_fig11_movement("smoke")
    case = IZRAELEVITZ_2017_FIG11
    result = movement_one_state_ullt(
        movement,
        source_cycle_step_range=(24, 47),
        period_s=case.period_s,
        freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3,
        aspect_ratio=case.aspect_ratio,
        area_m2=case.area_m2,
        output_samples=24,
    )
    assert result["CL"].shape == (24,)
    assert result["circulation_m2_s"].shape == (24, 14)
    assert np.all(np.isfinite(result["force_g_n"]))
    assert np.all(np.isfinite(result["added_mass_force_g_n"]))
    assert abs(result["mean_CL"]) < 1.0e-3
    assert result["mean_CT"] > 0.0
    assert result["parameters"]["observation_fit"] == "none"
    assert "case_id" not in result
    assert "paper" not in result["parameters"]


def test_same_ullt_api_executes_yang_without_model_branch() -> None:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    case = YANG_2025
    result = movement_one_state_ullt(
        movement,
        source_cycle_step_range=(20, 39),
        period_s=case.period_s,
        freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3,
        aspect_ratio=case.aspect_ratio,
        area_m2=case.area_m2,
        output_samples=20,
    )
    assert result["strip_count"] == 4
    assert np.all(np.isfinite(result["CL"] + result["CD"]))
    assert result["mean_lift_n"] > 0.0


def test_load_level_hybrid_endpoints_and_validation() -> None:
    phase = np.arange(4, dtype=float) / 4.0
    ullt = {
        "phase": phase,
        "lift_n": np.array([1.0, 2.0, 3.0, 4.0]),
        "drag_n": np.array([0.1, 0.2, 0.3, 0.4]),
    }
    uvlm = {
        "phase": phase,
        "lift_n": np.array([5.0, 6.0, 7.0, 8.0]),
        "thrust_n": -np.array([0.5, 0.6, 0.7, 0.8]),
    }
    separation = np.array([0.0, 1.0, 0.25, 0.75])
    result = blend_attached_with_uvlm_polar(ullt, uvlm, separation)
    assert result["lift_n"][0] == ullt["lift_n"][0]
    assert result["lift_n"][1] == uvlm["lift_n"][1]
    assert np.isclose(result["lift_n"][2], 4.0)
    assert np.isclose(result["drag_n"][3], 0.7)


def test_shared_separation_gate_is_local_symmetric_smoothstep() -> None:
    alpha = np.deg2rad(np.array([-20.0, -15.0, -10.0, 0.0, 10.0, 15.0, 20.0]))
    fraction = smooth_separation_fraction(
        alpha,
        attached_limit_deg=10.0,
        fully_separated_deg=20.0,
    )
    np.testing.assert_allclose(fraction, fraction[::-1])
    np.testing.assert_allclose(fraction, [1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0])
