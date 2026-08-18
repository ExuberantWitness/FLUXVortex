from __future__ import annotations

import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h4-transport-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h4-transport-mpl")

import numpy as np
import pytest

import forward_flight_benchmarks.fluxv_v5h4_ptera_rvpm_transport as v5h4
from fluxvortex.rvpm_transport import (
    RK_A,
    RK_B,
    lsrk3_step_direct,
    make_particle_state,
)
from fluxvortex.solver import UVPMHybridSolver
from forward_flight_benchmarks.fluxv_v5h3_native_feedback import (
    NativePteraRVPMFeedbackConfig,
)
from forward_flight_benchmarks.fluxv_v5h4_ptera_rvpm_transport import (
    FORCE_SCORING_STATUS,
    NOMINAL_RELATIVE_EPSILON,
    PREREGISTERED_RELATIVE_EPSILONS,
    FrozenExternalField,
    evaluate_frozen_parent_ptera_field,
    lsrk3_step_with_external_field,
    ptera_parent_state_sha256,
    transport_live_v5h2_cloud_with_frozen_ptera,
)
from forward_flight_benchmarks.v5h2_dyadic_cumulative_cloud_transport import (
    materialize_dyadic_cumulative_particle_state,
)

from test_fluxv_v5h3_native_feedback import _run, _two_feedback_reports


def _small_state():
    return make_particle_state(
        positions=np.asarray([[0.1, -0.2, 0.3], [0.35, 0.1, -0.15]], dtype=np.float64),
        gamma=np.asarray(
            [[0.02, -0.01, 0.03], [-0.015, 0.025, 0.005]], dtype=np.float64
        ),
        sigma=np.asarray([0.08, 0.09], dtype=np.float64),
    )


def _real_case():
    reports = _two_feedback_reports()
    solver = _run(NativePteraRVPMFeedbackConfig(enabled=True), reports)
    return reports, solver


def _assert_state_equal(actual, expected) -> None:
    np.testing.assert_array_equal(actual.positions, expected.positions)
    np.testing.assert_array_equal(actual.gamma, expected.gamma)
    np.testing.assert_array_equal(actual.sigma, expected.sigma)


def test_disabled_is_input_blind_and_bitwise_baseline_exact() -> None:
    state = _small_state()
    freestream = np.asarray([2.0, -0.1, 0.25], dtype=np.float64)

    def explode(_points):
        raise AssertionError("disabled path inspected the external Ptera field")

    expected, expected_stages = lsrk3_step_direct(
        state,
        0.02,
        freestream_velocity=freestream,
    )
    result = lsrk3_step_with_external_field(
        state,
        0.02,
        external_field=explode,
        baseline_freestream_velocity_gp1_m_per_s=freestream,
        enabled=False,
    )
    _assert_state_equal(result.final_state, expected)
    assert len(result.baseline_stages) == len(expected_stages) == 3
    for actual, reference in zip(result.baseline_stages, expected_stages, strict=True):
        _assert_state_equal(actual.pre, reference.pre)
        _assert_state_equal(actual.post, reference.post)
        np.testing.assert_array_equal(actual.rhs.velocity, reference.rhs.velocity)
        np.testing.assert_array_equal(actual.rhs.jacobian, reference.rhs.jacobian)
    assert result.stages == ()
    assert result.ptera_center_call_count == 0
    assert result.ptera_finite_difference_call_count == 0
    assert result.parent_write_count == result.load_write_count == 0


def test_zero_external_field_is_bitwise_self_field_exact_reduction() -> None:
    state = _small_state()

    def zero_field(points):
        return FrozenExternalField(
            velocity=np.zeros_like(points),
            jacobian=np.zeros((points.shape[0], 3, 3), dtype=np.float64),
        )

    expected, expected_stages = lsrk3_step_direct(
        state,
        0.02,
        freestream_velocity=np.zeros(3, dtype=np.float64),
    )
    result = lsrk3_step_with_external_field(
        state,
        0.02,
        external_field=zero_field,
        baseline_freestream_velocity_gp1_m_per_s=np.zeros(3, dtype=np.float64),
    )
    _assert_state_equal(result.final_state, expected)
    assert len(result.stages) == len(expected_stages) == 3
    for actual, reference in zip(result.stages, expected_stages, strict=True):
        _assert_state_equal(actual.pre, reference.pre)
        _assert_state_equal(actual.post, reference.post)
        np.testing.assert_array_equal(actual.total_rhs.velocity, reference.rhs.velocity)
        np.testing.assert_array_equal(actual.total_rhs.jacobian, reference.rhs.jacobian)


def test_parent_only_field_matches_native_ptera_and_three_scale_converges() -> None:
    reports, solver = _real_case()
    state = materialize_dyadic_cumulative_particle_state(reports[1])
    targets = state.positions[:12]
    scale = min(
        float(np.min(state.sigma)),
        min(float(airplane.c_ref) for airplane in solver.current_airplanes),
    )
    before = ptera_parent_state_sha256(solver)
    evaluations = tuple(
        evaluate_frozen_parent_ptera_field(
            solver,
            targets,
            epsilon_m=relative * scale,
        )
        for relative in PREREGISTERED_RELATIVE_EPSILONS
    )
    native = UVPMHybridSolver.calculate_solution_velocity(solver, targets)
    for evaluation in evaluations:
        np.testing.assert_array_equal(evaluation.velocity_gp1_m_per_s, native)
        assert evaluation.center_call_count == 1
        assert evaluation.finite_difference_call_count == 6
        assert np.all(np.isfinite(evaluation.jacobian_per_s))
    coarse_difference = np.linalg.norm(
        evaluations[0].jacobian_per_s - evaluations[1].jacobian_per_s
    )
    fine_difference = np.linalg.norm(
        evaluations[1].jacobian_per_s - evaluations[2].jacobian_per_s
    )
    assert fine_difference > 0.0
    assert coarse_difference / fine_difference >= 12.0
    assert ptera_parent_state_sha256(solver) == before


def test_live_transport_replays_lsrk3_and_keeps_parent_bitwise_unchanged() -> None:
    reports, solver = _real_case()
    initial = materialize_dyadic_cumulative_particle_state(reports[1])
    before = ptera_parent_state_sha256(solver)
    result = transport_live_v5h2_cloud_with_frozen_ptera(reports[1], solver)
    assert result.relative_epsilon == NOMINAL_RELATIVE_EPSILON
    assert result.ptera_step_index == 2
    assert result.dvm_for_source_step_index == 3
    assert result.initial_particle_count == reports[1].total_particle_count
    assert len(result.stages) == 3
    assert result.self_field_call_count == 3
    assert result.ptera_center_call_count == 3
    assert result.ptera_finite_difference_call_count == 18
    assert result.ptera_parent_state_unchanged
    assert result.parent_only_bypass
    assert result.feedback_write_count == 0
    assert result.parent_write_count == 0
    assert result.load_write_count == 0
    assert result.force_scoring_status == FORCE_SCORING_STATUS
    assert result.ptera_parent_state_sha256_before == before
    assert result.ptera_parent_state_sha256_after == before
    assert ptera_parent_state_sha256(solver) == before
    assert np.all(np.isfinite(result.final_state.positions))
    assert np.all(np.isfinite(result.final_state.gamma))
    assert np.all(np.isfinite(result.final_state.sigma))
    assert np.all(result.final_state.sigma > 0.0)
    assert not np.array_equal(result.final_state.positions, initial.positions)

    positions = initial.positions.copy()
    gamma = initial.gamma.copy()
    sigma = initial.sigma.copy()
    position_storage = np.zeros_like(positions)
    gamma_storage = np.zeros_like(gamma)
    sigma_storage = np.zeros_like(sigma)
    for expected_stage, (a_coefficient, b_coefficient) in enumerate(
        zip(RK_A, RK_B, strict=True), start=1
    ):
        stage = result.stages[expected_stage - 1]
        assert stage.stage == expected_stage
        np.testing.assert_array_equal(stage.pre.positions, positions)
        np.testing.assert_array_equal(stage.pre.gamma, gamma)
        np.testing.assert_array_equal(stage.pre.sigma, sigma)
        position_storage = (
            a_coefficient * position_storage + 0.02 * stage.total_rhs.velocity
        )
        gamma_storage = (
            a_coefficient * gamma_storage + 0.02 * stage.total_rhs.gamma_rate
        )
        sigma_storage = (
            a_coefficient * sigma_storage + 0.02 * stage.total_rhs.sigma_rate
        )
        positions = positions + b_coefficient * position_storage
        gamma = gamma + b_coefficient * gamma_storage
        sigma = sigma + b_coefficient * sigma_storage
        np.testing.assert_array_equal(stage.post.positions, positions)
        np.testing.assert_array_equal(stage.post.gamma, gamma)
        np.testing.assert_array_equal(stage.post.sigma, sigma)
    np.testing.assert_array_equal(result.final_state.positions, positions)
    np.testing.assert_array_equal(result.final_state.gamma, gamma)
    np.testing.assert_array_equal(result.final_state.sigma, sigma)


def test_three_scale_transport_family_has_second_order_common_limit() -> None:
    outputs = []
    for relative in PREREGISTERED_RELATIVE_EPSILONS:
        reports, solver = _real_case()
        outputs.append(
            transport_live_v5h2_cloud_with_frozen_ptera(
                reports[1],
                solver,
                relative_epsilon=relative,
            ).final_state
        )
    for field in ("positions", "gamma", "sigma"):
        arrays = [getattr(output, field) for output in outputs]
        coarse_difference = np.linalg.norm(arrays[0] - arrays[1])
        fine_difference = np.linalg.norm(arrays[1] - arrays[2])
        assert fine_difference > 0.0
        assert coarse_difference / fine_difference >= 12.0


def test_wrong_time_layer_rejects_then_correct_report_succeeds() -> None:
    reports, solver = _real_case()
    with pytest.raises(ValueError, match="time layer"):
        transport_live_v5h2_cloud_with_frozen_ptera(reports[0], solver)
    result = transport_live_v5h2_cloud_with_frozen_ptera(reports[1], solver)
    assert result.ptera_parent_state_unchanged


def test_report_replay_is_rejected_without_parent_mutation() -> None:
    reports, solver = _real_case()
    transport_live_v5h2_cloud_with_frozen_ptera(reports[1], solver)
    before = ptera_parent_state_sha256(solver)
    with pytest.raises(ValueError, match="already transported"):
        transport_live_v5h2_cloud_with_frozen_ptera(reports[1], solver)
    assert ptera_parent_state_sha256(solver) == before


def test_runtime_parent_callable_replacement_fails_before_call_and_can_retry(
    monkeypatch,
) -> None:
    reports, solver = _real_case()
    original = v5h4._FROZEN_PARENT_VELOCITY
    calls = 0

    def replacement(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(v5h4, "_FROZEN_PARENT_VELOCITY", replacement)
    with pytest.raises(ValueError, match="dependency callable"):
        transport_live_v5h2_cloud_with_frozen_ptera(reports[1], solver)
    assert calls == 0
    monkeypatch.setattr(v5h4, "_FROZEN_PARENT_VELOCITY", original)
    result = transport_live_v5h2_cloud_with_frozen_ptera(reports[1], solver)
    assert result.ptera_parent_state_unchanged


def test_ptera_private_kernel_replacement_fails_before_call_and_can_retry(
    monkeypatch,
) -> None:
    reports, solver = _real_case()
    state = materialize_dyadic_cumulative_particle_state(reports[1])
    module, attribute, original = v5h4._FROZEN_PTERA_CALLABLES[
        "collapsed_velocities_from_ring_vortices"
    ]
    calls = 0

    def replacement(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, attribute, replacement)
    with pytest.raises(ValueError, match="dependency callable"):
        evaluate_frozen_parent_ptera_field(
            solver,
            state.positions[:3],
            epsilon_m=1.0e-4,
        )
    assert calls == 0
    monkeypatch.setattr(module, attribute, original)
    evaluation = evaluate_frozen_parent_ptera_field(
        solver,
        state.positions[:3],
        epsilon_m=1.0e-4,
    )
    assert evaluation.center_call_count == 1


def test_rvpm_private_helper_replacement_fails_before_call_and_can_retry(
    monkeypatch,
) -> None:
    original = v5h4._reference_module.validate_particle_state
    calls = 0

    def replacement(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        v5h4._reference_module,
        "validate_particle_state",
        replacement,
    )
    with pytest.raises(ValueError, match="dependency callable"):
        lsrk3_step_with_external_field(
            _small_state(),
            0.02,
            external_field=lambda points: FrozenExternalField(
                np.zeros_like(points),
                np.zeros((points.shape[0], 3, 3), dtype=np.float64),
            ),
            baseline_freestream_velocity_gp1_m_per_s=np.zeros(3),
        )
    assert calls == 0
    monkeypatch.setattr(
        v5h4._reference_module,
        "validate_particle_state",
        original,
    )
    result = lsrk3_step_with_external_field(
        _small_state(),
        0.02,
        external_field=lambda points: FrozenExternalField(
            np.zeros_like(points),
            np.zeros((points.shape[0], 3, 3), dtype=np.float64),
        ),
        baseline_freestream_velocity_gp1_m_per_s=np.zeros(3),
    )
    assert len(result.stages) == 3


@pytest.mark.parametrize("value", (2.0**-9, 0.0, np.inf, True))
def test_unregistered_or_invalid_epsilon_rejects(value) -> None:
    reports, solver = _real_case()
    with pytest.raises((TypeError, ValueError)):
        transport_live_v5h2_cloud_with_frozen_ptera(
            reports[1], solver, relative_epsilon=value
        )
