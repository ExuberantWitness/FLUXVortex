from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from forward_flight_benchmarks.fluxv_v5f_native_load import (
    LEG_NAMES,
    calculate_native_surface_release_load_correction,
)


def _panel_index(chord: int, span: int, span_count: int) -> int:
    return chord * span_count + span


def _broadcast_strip(values: np.ndarray, chord_count: int) -> np.ndarray:
    return np.tile(np.asarray(values), chord_count)


def _synthetic_inputs(*, chord_count: int = 2, span_count: int = 3) -> dict[str, Any]:
    panel_count = chord_count * span_count
    chordwise = np.repeat(np.arange(chord_count), span_count)
    spanwise = np.tile(np.arange(span_count), chord_count)
    topology = {
        "airplane_index": np.zeros(panel_count, dtype=int),
        "wing_index": np.zeros(panel_count, dtype=int),
        "chordwise_index": chordwise,
        "spanwise_index": spanwise,
    }
    surface_by_span = np.linspace(0.8, 1.7, span_count)
    wake_by_span = np.linspace(0.2, 0.5, span_count)
    gamma_current_by_span = np.linspace(0.35, 0.85, span_count)
    gamma_previous_by_span = np.linspace(0.10, 0.40, span_count)

    panel = np.arange(panel_count, dtype=float)
    panel_area = 0.7 + 0.1 * panel
    panel_normal = np.column_stack((0.03 * panel, -0.02 * panel, np.ones(panel_count)))
    panel_normal /= np.linalg.norm(panel_normal, axis=1)[:, None]
    collocation = np.column_stack(
        (0.2 + 0.1 * chordwise, -0.8 + 0.4 * spanwise, 0.03 * panel)
    )

    # Ring directions match Ptera's documented ordering: right back->front,
    # front right->left, left front->back, and back left->right.
    leg_vectors = {
        "right": np.tile([-0.35, 0.0, 0.02], (panel_count, 1)),
        "front": np.tile([0.0, -0.40, -0.01], (panel_count, 1)),
        "left": np.tile([0.35, 0.0, -0.02], (panel_count, 1)),
        "back": np.tile([0.0, 0.40, 0.01], (panel_count, 1)),
    }
    leg_velocities = {
        "right": np.column_stack(
            (5.0 + 0.1 * panel, 0.2 + 0.01 * panel, 0.3 - 0.02 * panel)
        ),
        "front": np.column_stack(
            (4.7 + 0.1 * panel, -0.1 + 0.02 * panel, 0.4 - 0.01 * panel)
        ),
        "left": np.column_stack(
            (5.2 + 0.1 * panel, 0.15 - 0.01 * panel, 0.2 + 0.03 * panel)
        ),
        "back": np.column_stack(
            (4.9 + 0.1 * panel, -0.2 + 0.01 * panel, 0.35 + 0.01 * panel)
        ),
    }
    leg_centers = {
        "right": collocation + np.array([0.0, 0.2, 0.0]),
        "front": collocation + np.array([-0.15, 0.0, 0.0]),
        "left": collocation + np.array([0.0, -0.2, 0.0]),
        "back": collocation + np.array([0.15, 0.0, 0.0]),
    }
    return {
        "rho_kg_m3": 1.225,
        "delta_time_s": 0.04,
        "current_step": 3,
        "topology": topology,
        "surface_release_current_m2_s": _broadcast_strip(surface_by_span, chord_count),
        "wake_release_previous_m2_s": _broadcast_strip(wake_by_span, chord_count),
        "gamma_lev_current_m2_s": _broadcast_strip(gamma_current_by_span, chord_count),
        "gamma_lev_previous_m2_s": _broadcast_strip(
            gamma_previous_by_span, chord_count
        ),
        "active_current": np.ones(panel_count, dtype=bool),
        "panel_area_m2": panel_area,
        "panel_normal_gp1": panel_normal,
        "panel_collocation_gp1_cgp1_m": collocation,
        "leg_vectors_gp1_m": leg_vectors,
        "leg_velocities_gp1_m_s": leg_velocities,
        "leg_centers_gp1_cgp1_m": leg_centers,
    }


def _expected_strength_deltas(
    *,
    surface_release: np.ndarray,
    wake_release_previous: np.ndarray,
    current_step: int,
    chord_count: int,
    span_count: int,
) -> dict[str, np.ndarray]:
    strengths = {
        leg: np.zeros(chord_count * span_count, dtype=float) for leg in LEG_NAMES
    }
    effective_wake = (
        np.zeros_like(wake_release_previous)
        if current_step == 0
        else wake_release_previous
    )
    for chord in range(chord_count):
        for span in range(span_count):
            index = _panel_index(chord, span, span_count)
            release = surface_release[index]
            if span == span_count - 1:
                strengths["right"][index] = release
            else:
                to_right = _panel_index(chord, span + 1, span_count)
                strengths["right"][index] = (release - surface_release[to_right]) / 2.0
            if span == 0:
                strengths["left"][index] = release
            else:
                to_left = _panel_index(chord, span - 1, span_count)
                strengths["left"][index] = (release - surface_release[to_left]) / 2.0
            if chord == chord_count - 1:
                strengths["back"][index] = release - effective_wake[index]
    return strengths


def _assert_expected_strengths_and_kj(
    inputs: dict[str, Any], result: dict[str, Any], *, chord_count: int, span_count: int
) -> None:
    expected = _expected_strength_deltas(
        surface_release=inputs["surface_release_current_m2_s"],
        wake_release_previous=inputs["wake_release_previous_m2_s"],
        current_step=inputs["current_step"],
        chord_count=chord_count,
        span_count=span_count,
    )
    expected_kj = np.zeros((chord_count * span_count, 3))
    for leg in LEG_NAMES:
        np.testing.assert_array_equal(
            result["leg_ledger"][leg]["effective_strength_delta_m2_s"],
            expected[leg],
        )
        expected_leg_force = (
            inputs["rho_kg_m3"]
            * expected[leg][:, None]
            * np.cross(
                inputs["leg_velocities_gp1_m_s"][leg],
                inputs["leg_vectors_gp1_m"][leg],
            )
        )
        np.testing.assert_allclose(
            result["leg_ledger"][leg]["delta_force_gp1_n"],
            expected_leg_force,
            atol=2.0e-15,
            rtol=0.0,
        )
        expected_kj += expected_leg_force
    np.testing.assert_allclose(
        result["delta_kutta_joukowski_force_gp1_n"],
        expected_kj,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_activation_uses_surface_edges_and_active_lev_rate_separately() -> None:
    chord_count = 2
    span_count = 3
    inputs = _synthetic_inputs(chord_count=chord_count, span_count=span_count)
    inputs["current_step"] = 1
    inputs["wake_release_previous_m2_s"][:] = 0.0
    inputs["gamma_lev_previous_m2_s"][:] = 0.0
    result = calculate_native_surface_release_load_correction(**inputs)

    _assert_expected_strengths_and_kj(
        inputs, result, chord_count=chord_count, span_count=span_count
    )
    expected_eq17 = -(
        inputs["rho_kg_m3"]
        * inputs["panel_area_m2"][:, None]
        * inputs["panel_normal_gp1"]
        * inputs["gamma_lev_current_m2_s"][:, None]
        / inputs["delta_time_s"]
    )
    np.testing.assert_allclose(result["delta_eq17_force_gp1_n"], expected_eq17)
    np.testing.assert_array_equal(
        result["leg_ledger"]["front"]["effective_strength_delta_m2_s"], 0.0
    )


def test_continuation_uses_current_surface_minus_previous_wake_at_te() -> None:
    chord_count = 2
    span_count = 3
    inputs = _synthetic_inputs(chord_count=chord_count, span_count=span_count)
    result = calculate_native_surface_release_load_correction(**inputs)
    _assert_expected_strengths_and_kj(
        inputs, result, chord_count=chord_count, span_count=span_count
    )

    te_indices = np.arange((chord_count - 1) * span_count, chord_count * span_count)
    expected_te = (
        inputs["surface_release_current_m2_s"][te_indices]
        - inputs["wake_release_previous_m2_s"][te_indices]
    )
    np.testing.assert_array_equal(
        result["leg_ledger"]["back"]["effective_strength_delta_m2_s"][te_indices],
        expected_te,
    )
    expected_rate = (
        inputs["gamma_lev_current_m2_s"] - inputs["gamma_lev_previous_m2_s"]
    ) / inputs["delta_time_s"]
    np.testing.assert_allclose(
        result["delta_eq17_force_gp1_n"],
        -(
            inputs["rho_kg_m3"]
            * inputs["panel_area_m2"][:, None]
            * inputs["panel_normal_gp1"]
            * expected_rate[:, None]
        ),
    )


def test_deactivation_keeps_te_wake_cancellation_but_suppresses_eq17() -> None:
    chord_count = 2
    span_count = 3
    inputs = _synthetic_inputs(chord_count=chord_count, span_count=span_count)
    inputs["surface_release_current_m2_s"][:] = 0.0
    inputs["gamma_lev_current_m2_s"][:] = 0.0
    inputs["active_current"][:] = False
    result = calculate_native_surface_release_load_correction(**inputs)

    _assert_expected_strengths_and_kj(
        inputs, result, chord_count=chord_count, span_count=span_count
    )
    te_indices = np.arange((chord_count - 1) * span_count, chord_count * span_count)
    np.testing.assert_array_equal(
        result["leg_ledger"]["back"]["effective_strength_delta_m2_s"][te_indices],
        -inputs["wake_release_previous_m2_s"][te_indices],
    )
    np.testing.assert_array_equal(result["delta_eq17_force_gp1_n"], 0.0)
    np.testing.assert_array_equal(result["delta_eq17_moment_gp1_cgp1_nm"], 0.0)


def test_inactive_state_ignores_stale_lev_derivative_and_is_exact_zero() -> None:
    inputs = _synthetic_inputs()
    inputs["surface_release_current_m2_s"][:] = 0.0
    inputs["wake_release_previous_m2_s"][:] = 0.0
    inputs["gamma_lev_current_m2_s"][:] = 9.0
    inputs["gamma_lev_previous_m2_s"][:] = -4.0
    inputs["active_current"] = False
    result = calculate_native_surface_release_load_correction(**inputs)
    for name in (
        "delta_kutta_joukowski_force_gp1_n",
        "delta_eq17_force_gp1_n",
        "delta_total_force_gp1_n",
        "delta_kutta_joukowski_moment_gp1_cgp1_nm",
        "delta_eq17_moment_gp1_cgp1_nm",
        "delta_total_moment_gp1_cgp1_nm",
    ):
        np.testing.assert_array_equal(result[name], 0.0)
        assert not np.any(np.signbit(result[name]))


def test_step_zero_forces_previous_wake_release_to_zero() -> None:
    chord_count = 2
    span_count = 3
    inputs = _synthetic_inputs(chord_count=chord_count, span_count=span_count)
    inputs["current_step"] = 0
    inputs["wake_release_previous_m2_s"][:] = 123.0
    inputs["active_current"] = False
    result = calculate_native_surface_release_load_correction(**inputs)
    _assert_expected_strengths_and_kj(
        inputs, result, chord_count=chord_count, span_count=span_count
    )

    te_indices = np.arange((chord_count - 1) * span_count, chord_count * span_count)
    np.testing.assert_array_equal(
        result["leg_ledger"]["back"]["effective_strength_delta_m2_s"][te_indices],
        inputs["surface_release_current_m2_s"][te_indices],
    )
    np.testing.assert_array_equal(result["delta_eq17_force_gp1_n"], 0.0)


def test_front_kj_delta_is_always_bitwise_zero() -> None:
    inputs = _synthetic_inputs(chord_count=3, span_count=4)
    result = calculate_native_surface_release_load_correction(**inputs)
    front = result["leg_ledger"]["front"]
    for name in (
        "effective_strength_delta_m2_s",
        "delta_force_gp1_n",
        "delta_moment_gp1_cgp1_nm",
    ):
        np.testing.assert_array_equal(front[name], 0.0)
        assert not np.any(np.signbit(front[name]))


def test_span_release_has_boundary_values_and_shared_edge_differences() -> None:
    chord_count = 3
    span_count = 4
    inputs = _synthetic_inputs(chord_count=chord_count, span_count=span_count)
    surface_by_span = np.array([1.0, 1.0, 2.0, 4.0])
    inputs["surface_release_current_m2_s"] = _broadcast_strip(
        surface_by_span, chord_count
    )
    result = calculate_native_surface_release_load_correction(**inputs)
    ledger = result["leg_ledger"]

    for chord in range(chord_count):
        indices = np.arange(chord * span_count, (chord + 1) * span_count)
        np.testing.assert_array_equal(
            ledger["right"]["effective_strength_delta_m2_s"][indices],
            [0.0, -0.5, -1.0, 4.0],
        )
        np.testing.assert_array_equal(
            ledger["left"]["effective_strength_delta_m2_s"][indices],
            [1.0, 0.0, 0.5, 1.0],
        )


def test_eq17_is_gated_stripwise_by_active_current() -> None:
    chord_count = 2
    span_count = 3
    inputs = _synthetic_inputs(chord_count=chord_count, span_count=span_count)
    active_by_span = np.array([True, False, True])
    inputs["active_current"] = _broadcast_strip(active_by_span, chord_count)
    result = calculate_native_surface_release_load_correction(**inputs)
    rate = (
        inputs["gamma_lev_current_m2_s"] - inputs["gamma_lev_previous_m2_s"]
    ) / inputs["delta_time_s"]
    rate[~inputs["active_current"]] = 0.0
    expected = -(
        inputs["rho_kg_m3"]
        * inputs["panel_area_m2"][:, None]
        * inputs["panel_normal_gp1"]
        * rate[:, None]
    )
    np.testing.assert_allclose(result["delta_eq17_force_gp1_n"], expected)
    np.testing.assert_array_equal(
        result["delta_eq17_force_gp1_n"][~inputs["active_current"]], 0.0
    )


def test_all_zero_circulation_state_returns_bitwise_positive_zero_ledger() -> None:
    inputs = _synthetic_inputs()
    for name in (
        "surface_release_current_m2_s",
        "wake_release_previous_m2_s",
        "gamma_lev_current_m2_s",
        "gamma_lev_previous_m2_s",
    ):
        inputs[name][:] = 0.0
    inputs["active_current"] = True
    result = calculate_native_surface_release_load_correction(**inputs)
    arrays = [
        result["delta_kutta_joukowski_force_gp1_n"],
        result["delta_eq17_force_gp1_n"],
        result["delta_total_force_gp1_n"],
        result["delta_kutta_joukowski_moment_gp1_cgp1_nm"],
        result["delta_eq17_moment_gp1_cgp1_nm"],
        result["delta_total_moment_gp1_cgp1_nm"],
    ]
    for leg in LEG_NAMES:
        arrays.extend(
            (
                result["leg_ledger"][leg]["effective_strength_delta_m2_s"],
                result["leg_ledger"][leg]["delta_force_gp1_n"],
                result["leg_ledger"][leg]["delta_moment_gp1_cgp1_nm"],
            )
        )
    for array in arrays:
        np.testing.assert_array_equal(array, 0.0)
        assert not np.any(np.signbit(array))


def test_force_and_moment_ledgers_close_per_panel_and_in_total() -> None:
    inputs = _synthetic_inputs()
    result = calculate_native_surface_release_load_correction(**inputs)
    leg_force_sum = sum(
        (result["leg_ledger"][leg]["delta_force_gp1_n"] for leg in LEG_NAMES),
        np.zeros_like(result["delta_kutta_joukowski_force_gp1_n"]),
    )
    leg_moment_sum = sum(
        (result["leg_ledger"][leg]["delta_moment_gp1_cgp1_nm"] for leg in LEG_NAMES),
        np.zeros_like(result["delta_kutta_joukowski_moment_gp1_cgp1_nm"]),
    )
    np.testing.assert_allclose(
        result["delta_kutta_joukowski_force_gp1_n"], leg_force_sum
    )
    np.testing.assert_allclose(
        result["delta_kutta_joukowski_moment_gp1_cgp1_nm"], leg_moment_sum
    )
    np.testing.assert_allclose(
        result["delta_eq17_moment_gp1_cgp1_nm"],
        np.cross(
            inputs["panel_collocation_gp1_cgp1_m"],
            result["delta_eq17_force_gp1_n"],
        ),
    )
    np.testing.assert_allclose(
        result["delta_total_force_gp1_n"],
        result["delta_kutta_joukowski_force_gp1_n"] + result["delta_eq17_force_gp1_n"],
    )
    np.testing.assert_allclose(
        result["delta_total_moment_gp1_cgp1_nm"],
        result["delta_kutta_joukowski_moment_gp1_cgp1_nm"]
        + result["delta_eq17_moment_gp1_cgp1_nm"],
    )
    np.testing.assert_allclose(
        result["delta_total_force_total_gp1_n"],
        np.sum(result["delta_total_force_gp1_n"], axis=0),
    )
    np.testing.assert_allclose(
        result["delta_total_moment_total_gp1_cgp1_nm"],
        np.sum(result["delta_total_moment_gp1_cgp1_nm"], axis=0),
    )
    assert max(result["closure"].values()) < 1.0e-13


def test_eq17_sign_is_ptera_unsteady_sign_when_active() -> None:
    inputs = _synthetic_inputs(chord_count=1, span_count=1)
    inputs["rho_kg_m3"] = 2.0
    inputs["delta_time_s"] = 0.5
    inputs["gamma_lev_current_m2_s"][:] = 3.0
    inputs["gamma_lev_previous_m2_s"][:] = 1.0
    inputs["active_current"] = True
    inputs["panel_area_m2"][:] = 4.0
    inputs["panel_normal_gp1"][:] = [0.0, 0.0, 1.0]
    result = calculate_native_surface_release_load_correction(**inputs)
    np.testing.assert_array_equal(result["delta_eq17_force_gp1_n"], [[0.0, 0.0, -32.0]])


@pytest.mark.parametrize(
    ("path", "bad_value", "error"),
    [
        (
            ("surface_release_current_m2_s",),
            np.array([np.nan] * 6),
            "finite",
        ),
        (("wake_release_previous_m2_s",), np.ones((6, 1)), "shape"),
        (("gamma_lev_current_m2_s",), np.full(6, np.inf), "finite"),
        (("gamma_lev_previous_m2_s",), np.ones(5), "shape"),
        (("panel_area_m2",), np.ones((6, 1)), "shape"),
        (
            ("leg_velocities_gp1_m_s", "front"),
            np.full((6, 3), np.inf),
            "finite",
        ),
        (("panel_normal_gp1",), np.ones((5, 3)), "shape"),
    ],
)
def test_nonfinite_and_shape_errors_fail_closed(
    path: tuple[str, ...], bad_value: np.ndarray, error: str
) -> None:
    inputs = _synthetic_inputs()
    target: dict[str, Any] = inputs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    with pytest.raises(ValueError, match=error):
        calculate_native_surface_release_load_correction(**inputs)


def test_step_zero_still_validates_supplied_wake_audit_array() -> None:
    inputs = _synthetic_inputs()
    inputs["current_step"] = 0
    inputs["wake_release_previous_m2_s"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calculate_native_surface_release_load_correction(**inputs)


def test_topology_state_broadcast_and_activation_errors_fail_closed() -> None:
    inputs = _synthetic_inputs()
    inconsistent = deepcopy(inputs)
    inconsistent["surface_release_current_m2_s"][3] += 0.01
    with pytest.raises(ValueError, match="constant over each strip chord"):
        calculate_native_surface_release_load_correction(**inconsistent)

    inconsistent_active = deepcopy(inputs)
    inconsistent_active["active_current"][3] = False
    with pytest.raises(ValueError, match="active_current must be constant"):
        calculate_native_surface_release_load_correction(**inconsistent_active)

    integer_active = deepcopy(inputs)
    integer_active["active_current"] = np.ones(6, dtype=int)
    with pytest.raises(TypeError, match="booleans"):
        calculate_native_surface_release_load_correction(**integer_active)

    noninteger = deepcopy(inputs)
    noninteger["topology"]["spanwise_index"] = noninteger["topology"][
        "spanwise_index"
    ].astype(float)
    with pytest.raises(TypeError, match="integers"):
        calculate_native_surface_release_load_correction(**noninteger)

    incomplete = deepcopy(inputs)
    incomplete["topology"]["spanwise_index"][1] = 9
    with pytest.raises(ValueError, match="complete rectangular grid"):
        calculate_native_surface_release_load_correction(**incomplete)


@pytest.mark.parametrize("bad_step", [-1, 0.5, True])
def test_current_step_errors_fail_closed(bad_step: Any) -> None:
    inputs = _synthetic_inputs()
    inputs["current_step"] = bad_step
    expected_error = ValueError if bad_step == -1 else TypeError
    with pytest.raises(expected_error, match="current_step"):
        calculate_native_surface_release_load_correction(**inputs)


def test_leg_mappings_require_exact_ptera_leg_names() -> None:
    inputs = _synthetic_inputs()
    del inputs["leg_vectors_gp1_m"]["back"]
    with pytest.raises(ValueError, match="exactly the keys"):
        calculate_native_surface_release_load_correction(**inputs)
