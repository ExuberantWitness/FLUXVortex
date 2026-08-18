from __future__ import annotations

import numpy as np
import pytest

from forward_flight_benchmarks.fluxv_v5e_line_item import (
    CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE,
    KinematicAddedMassProvenance,
    ULLTUVLMLineItemParameters,
    ullt_to_uvlm_line_item_shadow,
)


def _inputs(*, time_count: int = 4, strip_count: int = 2) -> dict[str, object]:
    lift_directions = np.zeros((strip_count, 3), dtype=float)
    lift_directions[:, 2] = 1.0
    f_kj = np.zeros((time_count, strip_count, 3), dtype=float)
    f_kj[:, :, 2] = np.linspace(1.0, 2.0, time_count)[:, None]
    baseline = f_kj.copy()
    baseline[:, :, 0] = 0.25
    baseline[:, :, 2] += 0.4
    added_mass = np.zeros_like(f_kj)
    added_mass[:, :, 0] = -0.05
    added_mass[:, :, 2] = 0.1
    return {
        "baseline_total_force_history": baseline,
        "f_kj_history": f_kj,
        "strip_lift_direction": lift_directions,
        "chord": np.linspace(0.8, 1.2, strip_count),
        "strip_width": np.linspace(0.2, 0.4, strip_count),
        "v_perp": np.full((time_count, strip_count), 3.0),
        "density": 1.2,
        "delta_time": 0.01,
        "kinematic_added_mass_force_history": added_mass,
        "kinematic_added_mass_provenance": (CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE),
    }


class _MustNotBeEvaluated:
    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("disabled execution evaluated an enabled-only input")


def test_disabled_returns_bitwise_exact_baseline_without_new_input_evaluation() -> None:
    baseline = np.array(
        [[[1.0, -0.0, 3.0]], [[np.nextafter(2.0, 3.0), 5.0, -7.0]]],
        dtype=np.float64,
    )
    bomb = _MustNotBeEvaluated()
    result = ullt_to_uvlm_line_item_shadow(
        baseline,
        f_kj_history=bomb,
        strip_lift_direction=bomb,
        chord=bomb,
        strip_width=bomb,
        v_perp=bomb,
        density=bomb,
        delta_time=bomb,
        kinematic_added_mass_force_history=bomb,
        kinematic_added_mass_provenance=bomb,
        initial_state=bomb,
        enabled=False,
    )
    np.testing.assert_array_equal(result["new_force_history"], baseline)
    assert result["new_force_history"].dtype == baseline.dtype
    assert result["new_force_history"].tobytes() == baseline.tobytes()
    assert result["components"]["f_kj_history"] is None
    assert result["state"]["final_state"] is None
    assert result["diagnostics"]["status"] == ("not_evaluated_disabled_exact_baseline")
    assert result["ledger"]["module_off_max_abs_residual"] == 0.0


def test_one_strip_hand_calculation_and_complete_old_fd_gamma_replacement() -> None:
    rho = 2.0
    speed = 4.0
    width = 0.5
    chord = 1.25
    dt = 0.1
    lift_direction = np.array([[0.0, 0.0, 1.0]])
    f_kj = np.array([[[1.0, -2.0, 8.0]]])
    old_fd_gamma = np.array([[[3.0, 5.0, 7.0]]])
    baseline = f_kj + old_fd_gamma
    added_mass = np.array([[[-0.25, 0.5, 0.75]]])

    result = ullt_to_uvlm_line_item_shadow(
        baseline,
        f_kj_history=f_kj,
        strip_lift_direction=lift_direction,
        chord=chord,
        strip_width=width,
        v_perp=speed,
        density=rho,
        delta_time=dt,
        kinematic_added_mass_force_history=added_mass,
        kinematic_added_mass_provenance=(CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE),
    )

    gamma = 8.0 / (rho * speed * width)
    step = 2.0 * speed * dt / chord
    y_gamma = 2.0 * gamma / (chord * 2.0 * np.pi)
    decay = np.exp(-1.25 * step)
    state = 0.5 * (1.0 - decay) * y_gamma
    y_phi = 2.5 * y_gamma - 3.0 * state
    delta_scalar = 0.5 * rho * chord * 2.0 * np.pi * speed * (y_phi - y_gamma) * width
    expected_delta = delta_scalar * lift_direction[None, :, :]
    expected_new = f_kj + expected_delta + added_mass

    np.testing.assert_allclose(result["state"]["gamma_eq_history"], [[gamma]])
    np.testing.assert_allclose(result["state"]["delta_t_tilde_history"], [[step]])
    np.testing.assert_allclose(result["state"]["y_gamma_history"], [[y_gamma]])
    np.testing.assert_allclose(result["state"]["x_history"], [[state]])
    np.testing.assert_allclose(result["state"]["y_phi_history"], [[y_phi]])
    np.testing.assert_allclose(
        result["components"]["removed_old_fd_gamma_force_history"], old_fd_gamma
    )
    np.testing.assert_allclose(
        result["components"]["delta_phi_gamma_force_history"], expected_delta
    )
    np.testing.assert_allclose(result["new_force_history"], expected_new)
    assert result["ledger"]["old_fd_gamma_replaced_completely"] is True
    assert result["ledger"]["new_force_closure_max_abs_residual"] == 0.0
    assert result["ledger"]["replacement_identity_max_abs_residual"] < 2.0e-15


def test_high_incidence_uses_explicit_kj_lift_direction_not_surface_normal() -> None:
    lift_direction = np.array([0.6, 0.0, 0.8])
    surface_normal = np.array([0.0, 0.0, 1.0])
    tangent = np.array([0.8, 0.0, -0.6])
    assert np.dot(lift_direction, surface_normal) < 0.9
    assert abs(np.dot(lift_direction, tangent)) < 1.0e-15

    lift_component = 10.0
    f_kj_vector = lift_component * lift_direction + 4.0 * tangent
    f_kj = f_kj_vector[None, None, :]
    baseline = f_kj + np.array([[[0.2, -0.3, 0.4]]])
    result = ullt_to_uvlm_line_item_shadow(
        baseline,
        f_kj_history=f_kj,
        strip_lift_direction=lift_direction[None, :],
        chord=1.3,
        strip_width=0.4,
        v_perp=2.5,
        density=1.1,
        delta_time=0.02,
        kinematic_added_mass_force_history=np.zeros_like(f_kj),
        kinematic_added_mass_provenance=(CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE),
    )

    expected_gamma = lift_component / (1.1 * 2.5 * 0.4)
    normal_based_gamma = np.dot(f_kj_vector, surface_normal) / (1.1 * 2.5 * 0.4)
    np.testing.assert_allclose(result["state"]["gamma_eq_history"], [[expected_gamma]])
    assert not np.isclose(expected_gamma, normal_based_gamma)

    delta = result["components"]["delta_phi_gamma_force_history"][0, 0]
    parallel_delta = np.dot(delta, lift_direction) * lift_direction
    np.testing.assert_allclose(delta, parallel_delta, rtol=0.0, atol=2.0e-15)
    assert result["ledger"]["kj_lift_direction_identity_max_abs_residual"] < 2e-15
    assert result["ledger"]["delta_lift_direction_max_abs_residual"] < 2e-15


def test_first_sample_is_post_update_under_right_end_zoh() -> None:
    initial_state = 0.3
    y_gamma = 1.4
    chord = 1.0
    speed = 2.0
    dt = 0.04
    gamma = y_gamma * chord * np.pi
    f_kj = np.array([[[0.0, 0.0, gamma]]])
    result = ullt_to_uvlm_line_item_shadow(
        f_kj,
        f_kj_history=f_kj,
        strip_lift_direction=np.array([[0.0, 0.0, 1.0]]),
        chord=chord,
        strip_width=1.0,
        v_perp=speed,
        density=1.0 / speed,
        delta_time=dt,
        kinematic_added_mass_force_history=np.zeros_like(f_kj),
        kinematic_added_mass_provenance=(CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE),
        initial_state=initial_state,
    )
    step = 2.0 * speed * dt / chord
    decay = np.exp(-1.25 * step)
    expected_post_update = decay * initial_state + 0.5 * (1.0 - decay) * y_gamma
    np.testing.assert_allclose(result["state"]["x_history"][0, 0], expected_post_update)
    assert result["state"]["x_history"][0, 0] != initial_state
    assert result["diagnostics"]["time_discretization"] == ("right_end_zero_order_hold")
    assert result["parameters"]["time_discretization"].startswith("right-end ZOH")


def test_right_end_zoh_dt_refinement_converges_for_linear_input() -> None:
    final_time = 1.0

    def final_state(sample_count: int) -> float:
        dt = final_time / sample_count
        right_end_times = (np.arange(sample_count, dtype=float) + 1.0) * dt
        # With c=V=rho=ds=1, y_Gamma=Gamma/pi.  This sets y_Gamma(t)=t.
        f_kj = np.zeros((sample_count, 1, 3))
        f_kj[:, 0, 2] = np.pi * right_end_times
        result = ullt_to_uvlm_line_item_shadow(
            f_kj,
            f_kj_history=f_kj,
            strip_lift_direction=np.array([[0.0, 0.0, 1.0]]),
            chord=1.0,
            strip_width=1.0,
            v_perp=1.0,
            density=1.0,
            delta_time=dt,
            kinematic_added_mass_force_history=np.zeros_like(f_kj),
            kinematic_added_mass_provenance=(CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE),
        )
        return float(result["state"]["final_state"][0])

    physical_decay = 1.25 * 2.0
    exact = 0.5 * (
        final_time - (1.0 - np.exp(-physical_decay * final_time)) / physical_decay
    )
    coarse_error = abs(final_state(20) - exact)
    fine_error = abs(final_state(40) - exact)
    assert fine_error < 0.51 * coarse_error


def test_constant_analytic_equilibrium_has_zero_phi_gamma_mismatch() -> None:
    time_count = 6
    gamma = 2.0 * np.pi
    f_kj = np.zeros((time_count, 1, 3))
    f_kj[:, 0, 2] = gamma
    baseline = f_kj + np.array([0.3, -0.2, 0.4])[None, None, :]
    result = ullt_to_uvlm_line_item_shadow(
        baseline,
        f_kj_history=f_kj,
        strip_lift_direction=np.array([[0.0, 0.0, 1.0]]),
        chord=1.0,
        strip_width=1.0,
        v_perp=1.0,
        density=1.0,
        delta_time=0.03,
        kinematic_added_mass_force_history=np.zeros_like(f_kj),
        kinematic_added_mass_provenance=(CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE),
        # y_Gamma=2, hence the exact constant-input state is x=1.
        initial_state=1.0,
    )
    np.testing.assert_allclose(result["state"]["y_gamma_history"], 2.0)
    np.testing.assert_allclose(result["state"]["x_history"], 1.0, atol=2.0e-16)
    np.testing.assert_allclose(result["state"]["y_phi_history"], 2.0)
    np.testing.assert_allclose(
        result["components"]["delta_phi_gamma_force_history"],
        0.0,
        atol=2.0e-14,
    )
    # The arbitrary old dGamma force is gone rather than surviving at steady state.
    np.testing.assert_allclose(result["new_force_history"], f_kj, atol=2.0e-14)


def test_final_state_supports_explicit_causal_warmup_chaining() -> None:
    common = _inputs(time_count=5, strip_count=1)
    common["f_kj_history"] = np.full((5, 1, 3), [0.0, 0.0, 2.0])
    common["baseline_total_force_history"] = np.asarray(common["f_kj_history"]).copy()
    common["chord"] = 1.0
    common["strip_width"] = 1.0
    common["v_perp"] = 1.0
    common["density"] = 1.0
    common["delta_time"] = 0.08
    common["kinematic_added_mass_force_history"] = np.zeros((5, 1, 3))

    first = ullt_to_uvlm_line_item_shadow(**common)
    second = ullt_to_uvlm_line_item_shadow(
        **common, initial_state=first["state"]["final_state"]
    )

    y_gamma = 2.0 / np.pi
    step = 0.16
    expected_after_ten = 0.5 * y_gamma * (1.0 - np.exp(-1.25 * step * 10))
    np.testing.assert_allclose(
        second["state"]["final_state"], [expected_after_ten], rtol=2.0e-15
    )
    first_mismatch = np.linalg.norm(
        first["components"]["delta_phi_gamma_force_history"][-1]
    )
    second_mismatch = np.linalg.norm(
        second["components"]["delta_phi_gamma_force_history"][-1]
    )
    assert second_mismatch < first_mismatch


def test_multistrip_broadcast_and_force_ledgers_close() -> None:
    inputs = _inputs(time_count=7, strip_count=3)
    result = ullt_to_uvlm_line_item_shadow(**inputs)
    components = result["components"]
    reconstructed = (
        components["f_kj_history"]
        + components["delta_phi_gamma_force_history"]
        + components["kinematic_added_mass_force_history"]
    )
    np.testing.assert_array_equal(result["new_force_history"], reconstructed)
    np.testing.assert_allclose(
        result["span_summed_force_history"],
        np.sum(result["new_force_history"], axis=1),
    )
    baseline = np.asarray(inputs["baseline_total_force_history"])
    np.testing.assert_allclose(
        components["removed_old_fd_gamma_force_history"],
        baseline - np.asarray(inputs["f_kj_history"]),
    )
    assert result["ledger"]["new_force_closure_max_abs_residual"] == 0.0
    assert result["ledger"]["replacement_identity_max_abs_residual"] < 2.0e-15


def test_prefix_is_bitwise_unchanged_by_future_force_mutation() -> None:
    rng = np.random.default_rng(20260814)
    inputs = _inputs(time_count=20, strip_count=2)
    f_kj = np.asarray(inputs["f_kj_history"]).copy()
    f_kj[:, :, 2] = rng.uniform(0.5, 2.0, size=(20, 2))
    inputs["f_kj_history"] = f_kj
    inputs["baseline_total_force_history"] = f_kj + 0.2
    first = ullt_to_uvlm_line_item_shadow(**inputs)

    prefix = 11
    changed_inputs = dict(inputs)
    changed_f_kj = f_kj.copy()
    changed_f_kj[prefix:] = rng.uniform(-100.0, 100.0, size=(9, 2, 3))
    changed_inputs["f_kj_history"] = changed_f_kj
    changed_baseline = np.asarray(inputs["baseline_total_force_history"]).copy()
    changed_baseline[prefix:] = changed_f_kj[prefix:] + 17.0
    changed_inputs["baseline_total_force_history"] = changed_baseline
    changed = ullt_to_uvlm_line_item_shadow(**changed_inputs)

    for key in ("gamma_eq_history", "y_gamma_history", "x_history", "y_phi_history"):
        np.testing.assert_array_equal(
            first["state"][key][:prefix], changed["state"][key][:prefix]
        )
    np.testing.assert_array_equal(
        first["new_force_history"][:prefix],
        changed["new_force_history"][:prefix],
    )


def test_manifest_excludes_other_force_models_and_declares_full_replacement() -> None:
    result = ullt_to_uvlm_line_item_shadow(**_inputs(time_count=2, strip_count=1))
    manifest = result["parameters"]
    assert manifest["section_lift_slope_per_rad"] == 2.0 * np.pi
    assert manifest["state_decay_per_t_tilde"] == 1.25
    assert manifest["state_equilibrium_gain"] == 0.5
    assert "old F_dGamma is removed completely" in manifest["force_rule"]
    assert manifest["excluded_terms"] == (
        "LDVM/LEV, polar, separation, and owner selectors"
    )
    assert manifest["observation_access"] == "none"
    assert manifest["case_or_paper_branch"] == "none"
    assert manifest["gamma_rule"] == "Gamma_eq=(F_KJ dot e_L)/(rho*V_perp*ds)"
    assert manifest["kinematic_added_mass_provenance_tag"] == (
        CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE.value
    )
    provenance = result["provenance"]["kinematic_added_mass"]
    assert provenance == {
        "tag": CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE.value,
        "source": "Izraelevitz et al. (2017), Eqs. (35)-(39)",
        "aspect_ratio_rule": (
            "K_AM=0.85 at AR=3 and 0.95 at AR=6; linear interpolation "
            "on [3,6] with endpoint clamp"
        ),
        "exclusive_replacement": True,
        "force_ownership": (
            "F_AM,kin is supplied once and replaces, rather than augments, "
            "the removed UVLM F_dGamma line item"
        ),
        "canonical_tag_accepted": True,
    }
    assert result["diagnostics"]["state_lookahead_samples"] == 0
    assert result["diagnostics"]["added_mass_causality"] == (
        "caller-provenance; not established by core"
    )
    assert result["diagnostics"]["overall_causality"] == (
        "not_established_by_line_item_core"
    )


@pytest.mark.parametrize(
    "bad_provenance",
    [
        None,
        "unknown-added-mass-source",
        CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE.value,
        {"tag": CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE.value},
    ],
)
def test_missing_raw_or_unknown_added_mass_provenance_is_rejected(
    bad_provenance: object,
) -> None:
    inputs = _inputs(time_count=2, strip_count=1)
    inputs["kinematic_added_mass_provenance"] = bad_provenance
    with pytest.raises(ValueError, match="frozen KinematicAddedMassProvenance"):
        ullt_to_uvlm_line_item_shadow(**inputs)


def test_added_mass_provenance_keyword_is_required() -> None:
    inputs = _inputs(time_count=2, strip_count=1)
    del inputs["kinematic_added_mass_provenance"]
    with pytest.raises(TypeError, match="kinematic_added_mass_provenance"):
        ullt_to_uvlm_line_item_shadow(**inputs)


def test_only_one_canonical_added_mass_provenance_enum_exists() -> None:
    assert list(KinematicAddedMassProvenance) == [
        CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE
    ]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"f_kj_history": np.ones((2, 1, 2))}, "topology"),
        ({"f_kj_history": np.full((2, 1, 3), np.nan)}, "finite"),
        (
            {"strip_lift_direction": np.array([[0.0, 0.0, 2.0]])},
            "unit length",
        ),
        (
            {"strip_lift_direction": np.array([[0.0, np.nan, 1.0]])},
            "finite",
        ),
        ({"chord": 0.0}, "finite and positive"),
        ({"strip_width": -1.0}, "finite and positive"),
        ({"v_perp": np.inf}, "finite and positive"),
        ({"density": np.array([1.0])}, "scalar"),
        ({"density": 0.0}, "finite and positive"),
        ({"delta_time": np.nan}, "finite and positive"),
        ({"initial_state": [0.0, 1.0]}, "initial_state"),
        (
            {"kinematic_added_mass_force_history": np.ones((1, 1, 3))},
            "topology",
        ),
    ],
)
def test_invalid_enabled_inputs_fail_closed(
    replacement: dict[str, object], message: str
) -> None:
    inputs = _inputs(time_count=2, strip_count=1)
    inputs.update(replacement)
    with pytest.raises(ValueError, match=message):
        ullt_to_uvlm_line_item_shadow(**inputs)


def test_invalid_baseline_and_enabled_flag_fail_closed_even_when_disabled() -> None:
    bomb = _MustNotBeEvaluated()
    with pytest.raises(ValueError, match="finite"):
        ullt_to_uvlm_line_item_shadow(
            np.full((1, 1, 3), np.nan),
            f_kj_history=bomb,
            strip_lift_direction=bomb,
            chord=bomb,
            strip_width=bomb,
            v_perp=bomb,
            density=bomb,
            delta_time=bomb,
            kinematic_added_mass_force_history=bomb,
            kinematic_added_mass_provenance=bomb,
            enabled=False,
        )
    with pytest.raises(ValueError, match="Boolean"):
        ullt_to_uvlm_line_item_shadow(**_inputs(), enabled=1)


def test_frozen_coefficients_reject_custom_model_identity() -> None:
    with pytest.raises(ValueError, match="source-frozen"):
        ULLTUVLMLineItemParameters(state_decay_per_t_tilde=1.249)
