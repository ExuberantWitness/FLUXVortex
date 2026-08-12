from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.yang_plev import (
    IMPLEMENTATION_CONVENTIONS,
    MODEL_SCOPE_LIMITATIONS,
    YANG_2025_PARAMETERS,
    YangPLEVSolver,
    adaptive_wake_coefficient,
    adaptive_wake_split,
    eddy_viscosity_coefficient,
    panel_edge_geometry,
    plev_circulation,
    plev_strength_coefficient,
    rectangular_wing_vertices,
    vortex_core_radius,
    vortex_segment_velocity,
)


def _flow_at_alpha(speed: float, alpha_deg: float) -> np.ndarray:
    alpha = np.deg2rad(alpha_deg)
    return speed * np.array([np.cos(alpha), 0.0, -np.sin(alpha)])


def test_table_1_parameters_are_exact() -> None:
    parameters = YANG_2025_PARAMETERS
    assert parameters.density_kg_m3 == 1.23
    assert parameters.kinematic_viscosity_m2_s == 1.47e-5
    assert parameters.squire_parameter == 0.001
    assert parameters.initial_core_radius_fraction_mean_chord == 1.0e-5
    assert parameters.steps_per_flapping_cycle == 100
    assert parameters.separation_angle_deg == 5.0
    assert parameters.aoa_range_coefficient == 5.0
    assert parameters.attached_plev_coefficient == 0.4
    assert parameters.separated_plev_coefficient == -0.8
    assert parameters.plev_core_radius_coefficient == 0.05
    assert parameters.chordwise_panels == 8
    assert parameters.spanwise_panels == 12
    assert np.isclose(parameters.initial_core_radius_m(0.130), 1.3e-6)


def test_equations_8_and_9_core_growth() -> None:
    gamma = -0.021
    nu = 1.47e-5
    age = 0.037
    initial = 1.3e-6
    expected_delta = 1.0 + 0.001 * abs(gamma) / nu
    assert np.isclose(eddy_viscosity_coefficient(gamma, nu), expected_delta)
    expected_radius = np.sqrt(initial**2 + 4.0 * 1.25643 * nu * expected_delta * age)
    assert np.isclose(
        vortex_core_radius(initial, age, gamma, nu),
        expected_radius,
    )
    assert np.isclose(vortex_core_radius(initial, 0.0, gamma, nu), initial)


def test_equation_10_segment_velocity_has_standard_vector_direction() -> None:
    point = np.array([1.0, 0.0, 0.0])
    start = np.array([0.0, -0.5, 0.0])
    end = np.array([0.0, 0.5, 0.0])
    gamma = 0.73
    core = 0.04

    r1 = start - point
    r2 = end - point
    r0 = r1 - r2
    cross = np.cross(r1, r2)
    expected = (
        gamma
        / (4.0 * np.pi)
        * cross
        / (np.dot(cross, cross) + (core * np.linalg.norm(r0)) ** 2)
        * (np.dot(r0, r1) / np.linalg.norm(r1) - np.dot(r0, r2) / np.linalg.norm(r2))
    )
    actual = vortex_segment_velocity(point, start, end, gamma, core)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)
    assert actual[2] < 0.0
    np.testing.assert_allclose(
        vortex_segment_velocity(point, start, end, -gamma, core),
        -actual,
    )


def test_equations_11_and_12_piecewise_landmarks_and_symmetry() -> None:
    angles_deg = np.array([0.0, 5.0, 15.0, 25.0, 90.0])
    expected = np.array([0.0, 0.0, 0.4, 0.0, -0.8])
    actual = plev_strength_coefficient(np.deg2rad(angles_deg))
    np.testing.assert_allclose(actual, expected, atol=2.0e-15)
    np.testing.assert_allclose(
        plev_strength_coefficient(-np.deg2rad(angles_deg)),
        expected,
        atol=2.0e-15,
    )
    assert np.isclose(plev_circulation(np.deg2rad(15.0), 0.25), 0.1)


def test_equations_13_and_14_aws_split_conserves_circulation() -> None:
    aligned = adaptive_wake_coefficient([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    perpendicular = adaptive_wake_coefficient([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    assert np.isclose(aligned, 1.0)
    assert np.isclose(perpendicular, 0.0)
    retained, wake = adaptive_wake_split(0.37, 0.28)
    assert np.isclose(retained + wake, 0.37)
    assert np.isclose(wake, 0.28 * 0.37)


def test_panel_edge_ring_and_center_collocation_geometry() -> None:
    vertices = rectangular_wing_vertices(
        0.130,
        0.250,
        chordwise_panels=2,
        spanwise_panels=5,
    )
    geometry = panel_edge_geometry(vertices)
    assert geometry.corners_m.shape == (2, 5, 4, 3)
    np.testing.assert_allclose(
        geometry.corners_m[0, 0],
        [
            [0.0, 0.0, 0.0],
            [0.065, 0.0, 0.0],
            [0.065, 0.05, 0.0],
            [0.0, 0.05, 0.0],
        ],
    )
    np.testing.assert_allclose(geometry.collocation_m[0, 0], [0.0325, 0.025, 0.0])
    np.testing.assert_allclose(
        geometry.normals,
        np.broadcast_to([0.0, 0.0, 1.0], geometry.normals.shape),
    )
    np.testing.assert_allclose(geometry.areas_m2, 0.065 * 0.05)


def test_zero_plev_exactly_reduces_to_bound_panel_edge_uvlm() -> None:
    vertices = rectangular_wing_vertices(
        0.130,
        0.250,
        chordwise_panels=2,
        spanwise_panels=4,
    )
    flow = _flow_at_alpha(5.5, 3.0)
    solver_with_switch = YangPLEVSolver(
        delta_time_s=0.004,
        mean_chord_m=0.130,
        freestream_velocity_m_s=flow,
    )
    solver_without = YangPLEVSolver(
        delta_time_s=0.004,
        mean_chord_m=0.130,
        freestream_velocity_m_s=flow,
    )
    below_separation = solver_with_switch.solve_step(vertices, enable_plev=True)
    disabled = solver_without.solve_step(vertices, enable_plev=False)

    np.testing.assert_array_equal(below_separation.k_plev, 0.0)
    np.testing.assert_array_equal(below_separation.plev_circulation_m2_s, 0.0)
    np.testing.assert_allclose(
        below_separation.coupled_aic,
        below_separation.bound_aic,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        below_separation.bound_circulation_m2_s,
        disabled.bound_circulation_m2_s,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert below_separation.max_abs_normal_residual_m_s < 1.0e-11


def test_plev_is_simultaneously_coupled_into_no_penetration_solve() -> None:
    vertices = rectangular_wing_vertices(
        0.130,
        0.250,
        chordwise_panels=2,
        spanwise_panels=4,
    )
    flow = _flow_at_alpha(5.5, 15.0)
    enabled_solver = YangPLEVSolver(
        delta_time_s=0.004,
        mean_chord_m=0.130,
        freestream_velocity_m_s=flow,
    )
    disabled_solver = YangPLEVSolver(
        delta_time_s=0.004,
        mean_chord_m=0.130,
        freestream_velocity_m_s=flow,
    )
    enabled = enabled_solver.solve_step(vertices)
    disabled = disabled_solver.solve_step(vertices, enable_plev=False)

    np.testing.assert_allclose(enabled.k_plev, 0.4, atol=2.0e-15)
    np.testing.assert_allclose(
        enabled.plev_circulation_m2_s,
        0.4 * enabled.bound_circulation_m2_s[0],
    )
    assert (
        np.max(np.abs(enabled.bound_circulation_m2_s - disabled.bound_circulation_m2_s))
        > 1.0e-5
    )
    assert enabled.max_abs_normal_residual_m_s < 1.0e-10
    assert np.any(np.abs(enabled.coupled_aic - enabled.bound_aic) > 1.0e-12)
    assert len(enabled.implementation_conventions) == len(IMPLEMENTATION_CONVENTIONS)
    assert len(enabled.model_scope_limitations) == len(MODEL_SCOPE_LIMITATIONS)


def test_short_rigid_history_carries_previous_circulation() -> None:
    vertices = rectangular_wing_vertices(
        0.130,
        0.250,
        chordwise_panels=2,
        spanwise_panels=3,
    )
    flows = [_flow_at_alpha(5.5, angle) for angle in (0.0, 10.0, 30.0)]
    solver = YangPLEVSolver(
        delta_time_s=0.004,
        mean_chord_m=0.130,
    )
    history = solver.solve_history(
        [vertices, vertices, vertices],
        undisturbed_velocity_history_m_s=flows,
    )
    assert tuple(item.step_index for item in history) == (0, 1, 2)
    np.testing.assert_array_equal(history[0].previous_bound_circulation_m2_s, 0.0)
    np.testing.assert_allclose(
        history[1].previous_bound_circulation_m2_s,
        history[0].bound_circulation_m2_s,
    )
    np.testing.assert_allclose(
        history[2].previous_bound_circulation_m2_s,
        history[1].bound_circulation_m2_s,
    )
    assert all(item.max_abs_normal_residual_m_s < 1.0e-10 for item in history)
