from __future__ import annotations

from dataclasses import fields, replace
import gc
import hashlib
import json
from math import log2, pi, sqrt
import weakref

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import erf

import fluxvortex.rvpm_ir_wrk3_stream as stream_module
from fluxvortex.rvpm_ir_wrk3 import (
    ACTIVE_GAMMA_MAXABS_MIN,
    IRWRK3Field,
    IRWRK3TracerField,
    InvariantReference,
    ir_wrk3_step_with_external_field,
    make_ir_wrk3_field,
    make_ir_wrk3_tracer_field,
    validate_ir_wrk3_result,
)
from fluxvortex.rvpm_ir_wrk3_stream import (
    IRWRK3CompactStageRecord,
    IRWRK3StreamCounters,
    IRWRK3StreamEvidence,
    IRWRK3StreamResult,
    IRWRK3StreamStageRecord,
    IRWRK3StreamStageView,
    IRWRK3StreamStopped,
    ir_wrk3_stream_macro,
    make_ir_wrk3_stream_evidence,
    validate_ir_wrk3_stream_result,
)
from fluxvortex.rvpm_transport import ParticleState, make_particle_state


PARENT_TOKEN = "v5h11-stream-test-frozen-parent-v1"
HORIZON_S = 0.11125
ORDER_LEVELS = (4, 8, 16, 32)
MINIMUM_ORDER = 2.8
TRANSLATION = np.asarray((0.7, -0.03, 0.02), dtype=np.float64)
JACOBIANS = (
    np.diag((1.2, -0.4, -0.8)).astype(np.float64),
    np.asarray(
        ((0.0, -1.1, 0.2), (1.1, 0.0, -0.3), (-0.2, 0.3, 0.0)),
        dtype=np.float64,
    ),
    np.asarray(
        ((0.7, 3.0, 0.0), (0.0, -0.2, 1.5), (0.0, 0.0, -0.5)),
        dtype=np.float64,
    ),
)
B2_HORIZON_S = 0.04
B2_DOP853_RTOL = 2.0e-13
B2_DOP853_ATOL = 2.0e-15
B2_FINE_RELATIVE_L2_MAX = 1.0e-10


def _frozen_copy(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
        contiguous.shape
    )


def _state_bytes(state: ParticleState) -> tuple[bytes, bytes, bytes]:
    return (
        np.ascontiguousarray(state.positions).tobytes(order="C"),
        np.ascontiguousarray(state.gamma).tobytes(order="C"),
        np.ascontiguousarray(state.sigma).tobytes(order="C"),
    )


def _assert_frozen_float64(array: object, shape: tuple[int, ...]) -> None:
    assert type(array) is np.ndarray
    assert array.dtype == np.dtype(np.float64)
    assert array.shape == shape
    assert array.flags.c_contiguous
    assert not array.flags.writeable
    assert np.all(np.isfinite(array))
    ancestor: object = array
    while type(ancestor) is np.ndarray:
        assert ancestor.flags.c_contiguous
        assert not ancestor.flags.writeable
        ancestor = ancestor.base
    assert type(ancestor) is bytes


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(1.0e-300, float(np.linalg.norm(expected.ravel())))
    return float(np.linalg.norm((actual - expected).ravel()) / denominator)


def _scaled_norms(gamma: np.ndarray) -> np.ndarray:
    maximum = np.max(np.abs(gamma), axis=1)
    result = np.zeros(len(gamma), dtype=np.float64)
    active = maximum != 0.0
    if np.any(active):
        scaled = gamma[active] / maximum[active, None]
        result[active] = maximum[active] * np.sqrt(
            np.einsum("ni,ni->n", scaled, scaled)
        )
    return result


def _initial_state() -> ParticleState:
    mutable = make_particle_state(
        (
            (0.1, 0.2, 0.3),
            (-0.2, 0.05, 0.15),
            (0.03, -0.11, 0.07),
            (-0.04, 0.09, 0.21),
        ),
        (
            (0.4, -0.2, 0.3),
            (-0.1, 0.35, 0.2),
            (0.22, 0.08, -0.31),
            (0.0, -0.0, 0.0),
        ),
        (0.085, 0.07, 0.12, 0.095),
    )
    return ParticleState(
        _frozen_copy(mutable.positions),
        _frozen_copy(mutable.gamma),
        _frozen_copy(mutable.sigma),
    )


def _tracers() -> np.ndarray:
    return np.asarray(((-0.12, 0.07, 0.18), (0.16, -0.03, 0.04)), dtype=np.float64)


def _affine_field(jacobian: np.ndarray):
    def evaluate(state: ParticleState) -> IRWRK3Field:
        velocity = state.positions @ jacobian.T + TRANSLATION
        repeated = np.repeat(jacobian[None, :, :], len(state.sigma), axis=0)
        return make_ir_wrk3_field(state, velocity, repeated, parent_token=PARENT_TOKEN)

    return evaluate


def _affine_tracer(jacobian: np.ndarray):
    def evaluate(
        state: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        assert parent_token == PARENT_TOKEN
        velocity = tracer_pre @ jacobian.T + TRANSLATION
        return make_ir_wrk3_tracer_field(
            state, tracer_pre, velocity, parent_token=parent_token
        )

    return evaluate


def _affine_exact(
    initial: ParticleState, jacobian: np.ndarray, horizon: float
) -> ParticleState:
    augmented = np.zeros((4, 4), dtype=np.float64)
    augmented[:3, :3] = jacobian
    augmented[:3, 3] = TRANSLATION
    position_map = expm(horizon * augmented)
    homogeneous = np.column_stack(
        (initial.positions, np.ones(len(initial.sigma), dtype=np.float64))
    )
    positions = (homogeneous @ position_map.T)[:, :3]

    strength_map = expm(horizon * jacobian.T)
    gamma = initial.gamma.copy()
    sigma = initial.sigma.copy()
    for index in range(len(initial.sigma) - 1):
        gamma_0 = initial.gamma[index]
        norm_0 = float(np.linalg.norm(gamma_0))
        rho = float(np.linalg.norm(strength_map @ (gamma_0 / norm_0)))
        gamma[index] = rho ** (-3.0 / 5.0) * (strength_map @ gamma_0)
        sigma[index] = initial.sigma[index] * rho ** (-1.0 / 5.0)
    return make_particle_state(positions, gamma, sigma)


def _assert_stage_matches_core(
    compact: IRWRK3StreamStageRecord,
    core_stage: object,
    reference: InvariantReference,
) -> None:
    assert compact.substep == core_stage.substep
    assert compact.stage == core_stage.stage
    assert compact.a == core_stage.a
    assert compact.b == core_stage.b
    assert compact.source_state_sha256 == core_stage.source_state_sha256
    assert compact.pre_state_sha256 == stream_module._stream_state_sha256(
        core_stage.pre
    )
    assert compact.post_state_sha256 == stream_module._stream_state_sha256(
        core_stage.post
    )
    expected_hashes = {
        "tracer_pre_sha256": core_stage.tracer_pre,
        "tracer_post_sha256": core_stage.tracer_post,
        "velocity_sha256": core_stage.field.velocity,
        "jacobian_sha256": core_stage.field.jacobian,
        "gamma_rate_sha256": core_stage.rhs.gamma_rate,
        "tracer_velocity_sha256": core_stage.tracer_field.velocity,
        "invariant_residual_sha256": core_stage.invariant_log_residual,
        "position_storage_pre_sha256": core_stage.position_storage_pre,
        "gamma_storage_pre_sha256": core_stage.gamma_storage_pre,
        "tracer_storage_pre_sha256": core_stage.tracer_storage_pre,
        "position_storage_post_sha256": core_stage.position_storage_post,
        "gamma_storage_post_sha256": core_stage.gamma_storage_post,
        "tracer_storage_post_sha256": core_stage.tracer_storage_post,
    }
    for name, array in expected_hashes.items():
        assert getattr(compact, name) == stream_module._array_sha256(array), name
    assert compact.invariant_residual_max == float(
        np.max(core_stage.invariant_log_residual, initial=0.0)
    )
    active = ~reference.exact_zero_mask
    if np.any(active):
        norms = _scaled_norms(core_stage.post.gamma)
        slog = np.maximum.reduce(
            (
                np.ones(np.count_nonzero(active)),
                np.abs(np.log(norms[active]) - reference.log_gamma_norm_star[active]),
                2.0
                * np.abs(
                    np.log(core_stage.post.sigma[active])
                    - np.log(reference.sigma_star[active])
                ),
            )
        )
        expected_normalized_max = float(
            np.max(core_stage.invariant_log_residual[active] / slog)
        )
    else:
        expected_normalized_max = 0.0
    assert compact.invariant_residual_over_slog_max == expected_normalized_max
    assert compact.invariant_residual_over_slog_max <= stream_module.INVARIANT_LOG_ATOL
    assert compact.parent_token == core_stage.field.parent_token == PARENT_TOKEN


@pytest.mark.parametrize("substeps", (1, 2, 4))
def test_stream_is_bitwise_equivalent_to_full_core_for_small_n(
    substeps: int,
) -> None:
    initial = _initial_state()
    tracers = _tracers()
    core = ir_wrk3_step_with_external_field(
        initial,
        0.0375,
        _affine_field(JACOBIANS[2]),
        transport_substeps=substeps,
        tracer_positions=tracers,
        tracer_field_evaluator=_affine_tracer(JACOBIANS[2]),
        parent_token=PARENT_TOKEN,
    )
    stream = ir_wrk3_stream_macro(
        initial,
        0.0375,
        _affine_field(JACOBIANS[2]),
        transport_substeps=substeps,
        tracer_positions=tracers,
        tracer_field_evaluator=_affine_tracer(JACOBIANS[2]),
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_result(core) is core
    assert validate_ir_wrk3_stream_result(stream) is stream
    assert _state_bytes(stream.final_state) == _state_bytes(core.final_state)
    assert stream.final_tracer_positions.tobytes(
        order="C"
    ) == core.final_tracer_positions.tobytes(order="C")
    assert (
        stream.invariant_reference_sha256 == core.invariant_reference.reference_sha256
    )
    assert stream.initial_state_sha256 == stream_module._stream_state_sha256(
        core.stages[0].pre
    )
    assert stream.final_state_sha256 == stream_module._stream_state_sha256(
        core.final_state
    )
    assert len(stream.stages) == len(core.stages) == 3 * substeps
    for compact, core_stage in zip(stream.stages, core.stages, strict=True):
        _assert_stage_matches_core(compact, core_stage, core.invariant_reference)

    counters = stream.counters
    assert counters.invariant_reference_freeze_count == 1
    assert counters.substep_count == core.counters.substep_count == substeps
    assert counters.stage_count == core.counters.stage_count == 3 * substeps
    assert counters.physical_field_call_count == core.counters.physical_field_call_count
    assert counters.tracer_field_call_count == core.counters.tracer_field_call_count
    assert (
        counters.stage_pre_reconstruction_count
        == core.counters.stage_pre_reconstruction_count
    )
    assert (
        counters.stage_post_reconstruction_count
        == core.counters.stage_post_reconstruction_count
    )
    assert counters.physical_rhs_call_count == core.counters.physical_rhs_call_count
    assert counters.storage_reset_count == substeps
    assert counters.tracer_storage_reset_count == substeps
    assert counters.sigma_storage_update_count == 0
    assert counters.relaxation_call_count == 0
    assert counters.compact_stage_record_count == 3 * substeps
    assert counters.retained_stage_array_count == 0


def test_macro_reference_freezes_once_and_every_substep_resets_storage() -> None:
    substeps = 4
    initial = _initial_state()
    tracers = _tracers()
    result = ir_wrk3_stream_macro(
        initial,
        0.02,
        _affine_field(JACOBIANS[0]),
        transport_substeps=substeps,
        tracer_positions=tracers,
        tracer_field_evaluator=_affine_tracer(JACOBIANS[0]),
        parent_token=PARENT_TOKEN,
    )
    assert result.counters.invariant_reference_freeze_count == 1
    assert result.counters.storage_reset_count == substeps
    assert result.counters.tracer_storage_reset_count == substeps
    zero_position = stream_module._array_sha256(
        np.zeros_like(initial.positions, dtype=np.float64)
    )
    zero_gamma = stream_module._array_sha256(
        np.zeros_like(initial.gamma, dtype=np.float64)
    )
    zero_tracer = stream_module._array_sha256(np.zeros_like(tracers, dtype=np.float64))
    for substep in range(1, substeps + 1):
        first = result.stages[3 * (substep - 1)]
        assert (first.substep, first.stage) == (substep, 1)
        assert first.position_storage_pre_sha256 == zero_position
        assert first.gamma_storage_pre_sha256 == zero_gamma
        assert first.tracer_storage_pre_sha256 == zero_tracer
    assert result.stages[0].previous_chain_sha256 == stream_module._STAGE_CHAIN_GENESIS
    for previous, current in zip(result.stages[:-1], result.stages[1:]):
        assert current.previous_chain_sha256 == previous.chain_sha256
    assert result.stage_chain_sha256 == result.stages[-1].chain_sha256


def test_zero_only_cloud_records_exact_zero_normalized_invariant_gate() -> None:
    initial = make_particle_state(
        ((0.0, 0.0, 0.0),),
        ((0.0, -0.0, 0.0),),
        (0.1,),
    )

    def zero_field(state: ParticleState) -> IRWRK3Field:
        return make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3), dtype=np.float64),
            parent_token=PARENT_TOKEN,
        )

    result = ir_wrk3_stream_macro(
        initial,
        0.01,
        zero_field,
        transport_substeps=2,
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_stream_result(result) is result
    assert all(
        record.invariant_residual_max == 0.0
        and record.invariant_residual_over_slog_max == 0.0
        for record in result.stages
    )


@pytest.mark.parametrize("jacobian", JACOBIANS, ids=("stretch", "rotation", "shear"))
def test_stream_affine_expm_oracle_is_third_order(jacobian: np.ndarray) -> None:
    initial = _initial_state()
    expected = _affine_exact(initial, jacobian, HORIZON_S)
    errors: dict[str, list[float]] = {"positions": [], "gamma": [], "sigma": []}
    for substeps in ORDER_LEVELS:
        result = ir_wrk3_stream_macro(
            initial,
            HORIZON_S,
            _affine_field(jacobian),
            transport_substeps=substeps,
            parent_token=PARENT_TOKEN,
        )
        validate_ir_wrk3_stream_result(result)
        for name in errors:
            errors[name].append(
                _relative_l2(getattr(result.final_state, name), getattr(expected, name))
            )
    for name, channel_errors in errors.items():
        orders = tuple(
            log2(coarse / fine)
            for coarse, fine in zip(channel_errors[:-1], channel_errors[1:])
        )
        assert channel_errors[-1] <= 1.0e-7, (name, channel_errors)
        assert all(order >= MINIMUM_ORDER for order in orders[-2:]), (
            name,
            channel_errors,
            orders,
        )


def _hand_gaussian_field(
    source_positions: np.ndarray,
    source_gamma: np.ndarray,
    source_sigma: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros((len(targets), 3), dtype=np.float64)
    jacobian = np.zeros((len(targets), 3, 3), dtype=np.float64)
    minus_one_over_four_pi = -1.0 / (4.0 * pi)
    sqrt_two_over_pi = sqrt(2.0 / pi)
    for position, gamma, sigma in zip(
        source_positions, source_gamma, source_sigma, strict=True
    ):
        delta_all = targets - position
        radius_squared_all = np.einsum("ij,ij->i", delta_all, delta_all)
        active = radius_squared_all != 0.0
        if not np.any(active):
            continue
        delta = delta_all[active]
        radius_squared = radius_squared_all[active]
        radius = np.sqrt(radius_squared)
        radius_over_sigma = radius / sigma
        exponential = np.exp(-0.5 * radius_over_sigma**2)
        auxiliary = sqrt_two_over_pi * radius_over_sigma * exponential
        regularization = erf(radius_over_sigma / sqrt(2.0)) - auxiliary
        regularization_derivative = radius_over_sigma * auxiliary
        radius_cubed_inverse = 1.0 / (radius_squared * radius)
        cross = np.cross(delta, gamma)
        kernel_cross_gamma = (
            minus_one_over_four_pi * radius_cubed_inverse[:, None] * cross
        )
        velocity[active] += regularization[:, None] * kernel_cross_gamma
        radial_gradient = (
            regularization_derivative / (sigma * radius)
            - 3.0 * regularization / radius_squared
        )
        kronecker = minus_one_over_four_pi * regularization * radius_cubed_inverse
        contribution = np.empty((len(delta), 3, 3), dtype=np.float64)
        contribution[:, 0, 0] = radial_gradient * kernel_cross_gamma[:, 0] * delta[:, 0]
        contribution[:, 1, 0] = (
            radial_gradient * kernel_cross_gamma[:, 1] * delta[:, 0]
            - kronecker * gamma[2]
        )
        contribution[:, 2, 0] = (
            radial_gradient * kernel_cross_gamma[:, 2] * delta[:, 0]
            + kronecker * gamma[1]
        )
        contribution[:, 0, 1] = (
            radial_gradient * kernel_cross_gamma[:, 0] * delta[:, 1]
            + kronecker * gamma[2]
        )
        contribution[:, 1, 1] = radial_gradient * kernel_cross_gamma[:, 1] * delta[:, 1]
        contribution[:, 2, 1] = (
            radial_gradient * kernel_cross_gamma[:, 2] * delta[:, 1]
            - kronecker * gamma[0]
        )
        contribution[:, 0, 2] = (
            radial_gradient * kernel_cross_gamma[:, 0] * delta[:, 2]
            - kronecker * gamma[1]
        )
        contribution[:, 1, 2] = (
            radial_gradient * kernel_cross_gamma[:, 1] * delta[:, 2]
            + kronecker * gamma[0]
        )
        contribution[:, 2, 2] = radial_gradient * kernel_cross_gamma[:, 2] * delta[:, 2]
        jacobian[active] += contribution
    return velocity, jacobian


def _b2_fixture() -> tuple[ParticleState, np.ndarray, np.ndarray, np.ndarray]:
    initial = make_particle_state(
        ((-0.18, 0.02, 0.03), (0.17, -0.04, 0.08), (0.02, 0.21, -0.06)),
        ((0.025, -0.012, 0.018), (-0.014, 0.022, 0.011), (0.019, 0.008, -0.017)),
        (0.12, 0.11, 0.13),
    )
    tracers = np.asarray(((0.04, 0.03, 0.2), (-0.1, 0.13, -0.02)), dtype=np.float64)
    external_jacobian = np.asarray(
        ((0.2, 0.35, 0.0), (-0.1, -0.05, 0.18), (0.04, 0.0, -0.15)),
        dtype=np.float64,
    )
    external_translation = np.asarray((0.12, -0.03, 0.05), dtype=np.float64)
    return initial, tracers, external_jacobian, external_translation


def _b2_callbacks(external_jacobian: np.ndarray, external_translation: np.ndarray):
    def physical(state: ParticleState) -> IRWRK3Field:
        self_velocity, self_jacobian = _hand_gaussian_field(
            state.positions, state.gamma, state.sigma, state.positions
        )
        velocity = self_velocity + state.positions @ external_jacobian.T
        velocity += external_translation
        jacobian = self_jacobian + external_jacobian[None, :, :]
        return make_ir_wrk3_field(state, velocity, jacobian, parent_token=PARENT_TOKEN)

    def tracer(
        state: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        self_velocity, _ = _hand_gaussian_field(
            state.positions, state.gamma, state.sigma, tracer_pre
        )
        velocity = self_velocity + tracer_pre @ external_jacobian.T
        velocity += external_translation
        return make_ir_wrk3_tracer_field(
            state, tracer_pre, velocity, parent_token=parent_token
        )

    return physical, tracer


def _independent_b2_reference(
    initial: ParticleState,
    tracers: np.ndarray,
    external_jacobian: np.ndarray,
    external_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    particle_count = len(initial.sigma)
    tracer_count = len(tracers)
    gamma_norm_star = _scaled_norms(initial.gamma)
    sigma_star = np.array(initial.sigma, dtype=np.float64, copy=True)
    initial_vector = np.concatenate(
        (initial.positions.ravel(), initial.gamma.ravel(), tracers.ravel())
    )

    def reduced_rhs(_time_s: float, vector: np.ndarray) -> np.ndarray:
        position_end = 3 * particle_count
        gamma_end = 6 * particle_count
        positions = vector[:position_end].reshape((particle_count, 3))
        gamma = vector[position_end:gamma_end].reshape((particle_count, 3))
        tracer_positions = vector[gamma_end:].reshape((tracer_count, 3))
        norms = _scaled_norms(gamma)
        sigma = sigma_star * np.sqrt(gamma_norm_star / norms)
        self_velocity, self_jacobian = _hand_gaussian_field(
            positions, gamma, sigma, positions
        )
        velocity = self_velocity + positions @ external_jacobian.T
        velocity += external_translation
        total_jacobian = self_jacobian + external_jacobian[None, :, :]
        stretching = np.einsum("nji,nj->ni", total_jacobian, gamma)
        chi = np.einsum("ni,ni->n", stretching, gamma) / (norms * norms)
        gamma_rate = stretching - 0.6 * chi[:, None] * gamma
        tracer_velocity, _ = _hand_gaussian_field(
            positions, gamma, sigma, tracer_positions
        )
        tracer_velocity += tracer_positions @ external_jacobian.T
        tracer_velocity += external_translation
        return np.concatenate(
            (velocity.ravel(), gamma_rate.ravel(), tracer_velocity.ravel())
        )

    solution = solve_ivp(
        reduced_rhs,
        (0.0, B2_HORIZON_S),
        initial_vector,
        method="DOP853",
        rtol=B2_DOP853_RTOL,
        atol=B2_DOP853_ATOL,
        t_eval=(B2_HORIZON_S,),
    )
    assert solution.success, solution.message
    final = solution.y[:, -1]
    position_end = 3 * particle_count
    gamma_end = 6 * particle_count
    positions = final[:position_end].reshape((particle_count, 3))
    gamma = final[position_end:gamma_end].reshape((particle_count, 3))
    final_tracers = final[gamma_end:].reshape((tracer_count, 3))
    sigma = sigma_star * np.sqrt(gamma_norm_star / _scaled_norms(gamma))
    return positions, gamma, sigma, final_tracers


def test_stream_hand_gaussian_dop853_oracle_is_third_order() -> None:
    initial, tracers, external_jacobian, external_translation = _b2_fixture()
    expected = _independent_b2_reference(
        initial, tracers, external_jacobian, external_translation
    )
    errors: list[list[float]] = [[], [], [], []]
    for substeps in ORDER_LEVELS:
        physical, tracer = _b2_callbacks(external_jacobian, external_translation)
        result = ir_wrk3_stream_macro(
            initial,
            B2_HORIZON_S,
            physical,
            transport_substeps=substeps,
            tracer_positions=tracers,
            tracer_field_evaluator=tracer,
            parent_token=PARENT_TOKEN,
        )
        actual = (
            result.final_state.positions,
            result.final_state.gamma,
            result.final_state.sigma,
            result.final_tracer_positions,
        )
        for channel, actual_array, expected_array in zip(
            errors, actual, expected, strict=True
        ):
            channel.append(_relative_l2(actual_array, expected_array))
    for channel_errors in errors:
        orders = tuple(
            log2(coarse / fine)
            for coarse, fine in zip(channel_errors[:-1], channel_errors[1:])
        )
        assert channel_errors[-1] <= B2_FINE_RELATIVE_L2_MAX, channel_errors
        assert all(MINIMUM_ORDER <= order <= 3.2 for order in orders[-2:]), (
            channel_errors,
            orders,
        )


def test_observer_receives_exact_readonly_ephemeral_views_and_retains_only_bytes() -> (
    None
):
    initial = _initial_state()
    tracers = _tracers()
    coordinates: list[tuple[int, int]] = []
    weak_arrays: list[weakref.ReferenceType[np.ndarray]] = []

    def observer(view: IRWRK3StreamStageView) -> IRWRK3StreamEvidence:
        assert type(view) is IRWRK3StreamStageView
        coordinates.append((view.substep, view.stage))
        expected_field = make_ir_wrk3_field(
            view.pre,
            view.field.velocity,
            view.field.jacobian,
            parent_token=view.parent_token,
        )
        expected_tracer_field = make_ir_wrk3_tracer_field(
            view.pre,
            view.tracer_pre,
            view.tracer_field.velocity,
            parent_token=view.parent_token,
        )
        assert view.field.source_state_sha256 == expected_field.source_state_sha256
        assert (
            view.tracer_field.source_state_sha256
            == expected_tracer_field.source_state_sha256
        )
        assert (
            view.tracer_field.tracer_state_sha256
            == expected_tracer_field.tracer_state_sha256
        )
        arrays = (
            view.pre.positions,
            view.pre.gamma,
            view.pre.sigma,
            view.field.velocity,
            view.field.jacobian,
            view.rhs.stretching,
            view.rhs.chi,
            view.rhs.gamma_rate,
            view.rhs.sigma_rate_diagnostic,
            view.rhs.chain_rule_relative_residual,
            view.tracer_pre,
            view.tracer_field.velocity,
            view.position_storage_pre,
            view.gamma_storage_pre,
            view.tracer_storage_pre,
        )
        for array in arrays:
            _assert_frozen_float64(array, array.shape)
            weak_arrays.append(weakref.ref(array))
        payload = json.dumps(
            {"stage": view.stage, "substep": view.substep},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return make_ir_wrk3_stream_evidence("v5h11-test-observer-v1", payload)

    result = ir_wrk3_stream_macro(
        initial,
        0.02,
        _affine_field(JACOBIANS[1]),
        transport_substeps=2,
        tracer_positions=tracers,
        tracer_field_evaluator=_affine_tracer(JACOBIANS[1]),
        parent_token=PARENT_TOKEN,
        observer=observer,
    )
    assert coordinates == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    assert result.counters.observer_call_count == 6
    assert result.counters.retained_stage_array_count == 0
    assert result.counters.evidence_byte_count == sum(
        len(record.evidence.payload) for record in result.stages
    )
    assert all(type(record.evidence.payload) is bytes for record in result.stages)
    assert all(
        record.evidence.schema == "v5h11-test-observer-v1" for record in result.stages
    )
    gc.collect()
    assert all(reference() is None for reference in weak_arrays)


def _registry_snapshot() -> tuple[object, ...]:
    snapshot = stream_module._snapshot_live_stream_result_registry()
    assert type(snapshot) is tuple and len(snapshot) == 3
    counter, seal, entries = snapshot
    assert type(entries) is tuple
    return (
        counter,
        seal,
        tuple((key, id(result), digest) for key, result, digest in entries),
    )


def _failure_coordinate(error: BaseException) -> tuple[int, int]:
    for name in ("failed_coordinate", "coordinate"):
        value = getattr(error, name, None)
        if type(value) is tuple and len(value) == 2:
            return value
    substep = getattr(error, "substep", None)
    stage = getattr(error, "stage", None)
    if type(substep) is int and type(stage) is int:
        return substep, stage
    raise AssertionError("stream failure did not expose an exact failed coordinate")


def _assert_failure_journal(
    error: BaseException,
    *,
    phase: str,
    coordinate: tuple[int, int],
    stage_began: bool,
    completed_count: int,
) -> IRWRK3StreamStopped:
    assert type(error) is IRWRK3StreamStopped
    stopped = error
    assert stopped.failure_phase == phase
    assert type(stopped.failure_phase) is str
    assert stopped.stage_began is stage_began
    assert type(stopped.failed_coordinate) is tuple
    assert stopped.failed_coordinate == coordinate
    assert _failure_coordinate(stopped) == coordinate
    assert type(stopped.completed_stages) is tuple
    assert len(stopped.completed_stages) == completed_count
    assert stopped.completed_stage_count == completed_count
    assert all(
        type(record) is IRWRK3CompactStageRecord for record in stopped.completed_stages
    )
    expected_chain = stream_module._STAGE_CHAIN_GENESIS
    for index, record in enumerate(stopped.completed_stages):
        assert (record.substep, record.stage) == (index // 3 + 1, index % 3 + 1)
        assert record.previous_chain_sha256 == expected_chain
        expected_chain = record.chain_sha256
        assert all(
            not isinstance(getattr(record, field.name), np.ndarray)
            for field in fields(record)
        )
        assert type(record.invariant_residual_over_slog_max) is float
        assert np.isfinite(record.invariant_residual_over_slog_max)
        assert (
            0.0
            <= record.invariant_residual_over_slog_max
            <= stream_module.INVARIANT_LOG_ATOL
        )
    assert stopped.completed_stage_chain_sha256 == expected_chain
    assert stopped.completed_prefix_sha256 == expected_chain
    assert stopped.__cause__ is stopped.original_cause
    assert isinstance(stopped.original_cause, BaseException)
    with pytest.raises(AttributeError):
        stopped.completed_stages.append(None)  # type: ignore[attr-defined]
    return stopped


def test_observer_failure_reports_coordinate_does_not_register_and_clean_retry() -> (
    None
):
    initial = _initial_state()
    before_state = _state_bytes(initial)

    def prefix_observer(view: IRWRK3StreamStageView) -> IRWRK3StreamEvidence:
        return make_ir_wrk3_stream_evidence(
            "v5h11-test-prefix-v1", f"{view.substep}:{view.stage}".encode("ascii")
        )

    reference = ir_wrk3_stream_macro(
        initial,
        0.02,
        _affine_field(JACOBIANS[0]),
        transport_substeps=2,
        parent_token=PARENT_TOKEN,
        observer=prefix_observer,
    )
    expected_prefix_sha256 = reference.stages[2].chain_sha256
    before_registry = _registry_snapshot()
    calls = 0

    def failing_observer(view: IRWRK3StreamStageView) -> IRWRK3StreamEvidence:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("injected observer stop")
        return make_ir_wrk3_stream_evidence(
            "v5h11-test-prefix-v1", f"{view.substep}:{view.stage}".encode("ascii")
        )

    with pytest.raises(Exception, match="observer|injected|stage") as caught:
        ir_wrk3_stream_macro(
            initial,
            0.02,
            _affine_field(JACOBIANS[0]),
            transport_substeps=2,
            parent_token=PARENT_TOKEN,
            observer=failing_observer,
        )
    assert calls == 4
    assert _failure_coordinate(caught.value) == (2, 1)
    assert type(getattr(caught.value, "completed_stage_count", None)) is int
    assert caught.value.completed_stage_count == 3
    assert type(caught.value.completed_stage_chain_sha256) is str
    assert caught.value.completed_stage_chain_sha256 == expected_prefix_sha256
    assert caught.value.completed_prefix_sha256 == expected_prefix_sha256
    stopped = _assert_failure_journal(
        caught.value,
        phase="observer",
        coordinate=(2, 1),
        stage_began=True,
        completed_count=3,
    )
    assert tuple(record.chain_sha256 for record in stopped.completed_stages) == tuple(
        record.chain_sha256 for record in reference.stages[:3]
    )
    assert _registry_snapshot() == before_registry
    assert _state_bytes(initial) == before_state

    clean = ir_wrk3_stream_macro(
        initial,
        0.02,
        _affine_field(JACOBIANS[0]),
        transport_substeps=2,
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_stream_result(clean) is clean


def test_physical_field_failure_retains_exact_completed_prefix_and_clean_retry() -> (
    None
):
    initial = _initial_state()
    before_state = _state_bytes(initial)
    before_registry = _registry_snapshot()
    base_field = _affine_field(JACOBIANS[0])
    calls = 0

    def failing_field(state: ParticleState) -> IRWRK3Field:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise LookupError("injected physical field stop")
        return base_field(state)

    with pytest.raises(IRWRK3StreamStopped, match="physical_field") as caught:
        ir_wrk3_stream_macro(
            initial,
            0.02,
            failing_field,
            transport_substeps=2,
            parent_token=PARENT_TOKEN,
        )
    stopped = _assert_failure_journal(
        caught.value,
        phase="physical_field",
        coordinate=(2, 1),
        stage_began=True,
        completed_count=3,
    )
    assert type(stopped.original_cause) is LookupError
    assert calls == 4
    assert _registry_snapshot() == before_registry
    assert _state_bytes(initial) == before_state

    clean = ir_wrk3_stream_macro(
        initial,
        0.02,
        base_field,
        transport_substeps=2,
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_stream_result(clean) is clean


def test_tracer_field_failure_retains_exact_completed_prefix_and_clean_retry() -> None:
    initial = _initial_state()
    tracers = _tracers()
    before_state = _state_bytes(initial)
    before_registry = _registry_snapshot()
    base_tracer = _affine_tracer(JACOBIANS[0])
    calls = 0

    def failing_tracer(
        state: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise ArithmeticError("injected tracer field stop")
        return base_tracer(state, tracer_pre, parent_token)

    with pytest.raises(IRWRK3StreamStopped, match="tracer_field") as caught:
        ir_wrk3_stream_macro(
            initial,
            0.02,
            _affine_field(JACOBIANS[0]),
            transport_substeps=2,
            tracer_positions=tracers,
            tracer_field_evaluator=failing_tracer,
            parent_token=PARENT_TOKEN,
        )
    stopped = _assert_failure_journal(
        caught.value,
        phase="tracer_field",
        coordinate=(2, 2),
        stage_began=True,
        completed_count=4,
    )
    assert type(stopped.original_cause) is ArithmeticError
    assert calls == 5
    assert _registry_snapshot() == before_registry
    assert _state_bytes(initial) == before_state

    clean = ir_wrk3_stream_macro(
        initial,
        0.02,
        _affine_field(JACOBIANS[0]),
        transport_substeps=2,
        tracer_positions=tracers,
        tracer_field_evaluator=base_tracer,
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_stream_result(clean) is clean


def test_rhs_failure_retains_exact_completed_prefix_and_clean_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _initial_state()
    before_state = _state_bytes(initial)
    before_registry = _registry_snapshot()
    original_rhs = stream_module._stream_rhs
    calls = 0

    def failing_rhs(
        gamma: np.ndarray,
        sigma: np.ndarray,
        field: IRWRK3Field,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise FloatingPointError("injected RHS stop")
        return original_rhs(gamma, sigma, field)

    monkeypatch.setattr(stream_module, "_stream_rhs", failing_rhs)
    with pytest.raises(IRWRK3StreamStopped, match="rhs") as caught:
        ir_wrk3_stream_macro(
            initial,
            0.02,
            _affine_field(JACOBIANS[0]),
            transport_substeps=2,
            parent_token=PARENT_TOKEN,
        )
    stopped = _assert_failure_journal(
        caught.value,
        phase="rhs",
        coordinate=(2, 1),
        stage_began=True,
        completed_count=3,
    )
    assert type(stopped.original_cause) is FloatingPointError
    assert calls == 4
    assert _registry_snapshot() == before_registry
    assert _state_bytes(initial) == before_state

    monkeypatch.setattr(stream_module, "_stream_rhs", original_rhs)
    clean = ir_wrk3_stream_macro(
        initial,
        0.02,
        _affine_field(JACOBIANS[0]),
        transport_substeps=2,
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_stream_result(clean) is clean


def test_total_evidence_cap_failure_retains_4096_stage_prefix_and_clean_retry() -> None:
    initial = make_particle_state(
        ((0.0, 0.0, 0.0),),
        ((0.2, -0.1, 0.05),),
        (0.09,),
    )
    before_state = _state_bytes(initial)
    before_registry = _registry_snapshot()
    full_evidence = make_ir_wrk3_stream_evidence(
        "v5h11-full-stage-evidence-v1",
        b"x" * stream_module.MAX_EVIDENCE_BYTES_PER_STAGE,
    )

    def observer(_view: IRWRK3StreamStageView) -> IRWRK3StreamEvidence:
        return full_evidence

    with pytest.raises(IRWRK3StreamStopped, match="evidence_cap") as caught:
        ir_wrk3_stream_macro(
            initial,
            0.01,
            _affine_field(JACOBIANS[0]),
            transport_substeps=1366,
            parent_token=PARENT_TOKEN,
            observer=observer,
        )
    stopped = _assert_failure_journal(
        caught.value,
        phase="evidence_cap",
        coordinate=(1366, 2),
        stage_began=True,
        completed_count=4096,
    )
    assert type(stopped.original_cause) is ValueError
    assert _registry_snapshot() == before_registry
    assert _state_bytes(initial) == before_state

    clean = ir_wrk3_stream_macro(
        initial,
        0.001,
        _affine_field(JACOBIANS[0]),
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_stream_result(clean) is clean


def test_wrong_observer_evidence_fails_closed_and_registry_is_unchanged() -> None:
    before = _registry_snapshot()

    def hostile(_view: IRWRK3StreamStageView) -> IRWRK3StreamEvidence:
        valid = make_ir_wrk3_stream_evidence("v5h11-test-tamper-v1", b"clean")
        return replace(valid, payload_sha256="0" * 64)

    with pytest.raises(ValueError, match="evidence|digest"):
        ir_wrk3_stream_macro(
            _initial_state(),
            0.01,
            _affine_field(JACOBIANS[0]),
            parent_token=PARENT_TOKEN,
            observer=hostile,
        )
    assert _registry_snapshot() == before
    with pytest.raises(ValueError, match="exact IRWRK3StreamEvidence|observer"):
        ir_wrk3_stream_macro(
            _initial_state(),
            0.01,
            _affine_field(JACOBIANS[0]),
            parent_token=PARENT_TOKEN,
            observer=lambda _view: None,  # type: ignore[return-value]
        )


def test_observer_cannot_change_live_registry_and_clean_retry_succeeds() -> None:
    initial = _initial_state()
    before_state = _state_bytes(initial)
    nested_once = False

    def reentrant_observer(
        view: IRWRK3StreamStageView,
    ) -> IRWRK3StreamEvidence:
        nonlocal nested_once
        if not nested_once:
            nested_once = True
            nested = ir_wrk3_stream_macro(
                initial,
                0.001,
                _affine_field(JACOBIANS[0]),
                parent_token=PARENT_TOKEN,
            )
            assert validate_ir_wrk3_stream_result(nested) is nested
        return make_ir_wrk3_stream_evidence(
            "v5h11-registry-reentry-v1",
            f"{view.substep}:{view.stage}".encode("ascii"),
        )

    with pytest.raises(RuntimeError, match="registry.*changed|changed.*registry"):
        ir_wrk3_stream_macro(
            initial,
            0.01,
            _affine_field(JACOBIANS[0]),
            parent_token=PARENT_TOKEN,
            observer=reentrant_observer,
        )
    assert nested_once
    assert _state_bytes(initial) == before_state

    clean = ir_wrk3_stream_macro(
        initial,
        0.01,
        _affine_field(JACOBIANS[0]),
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_stream_result(clean) is clean


@pytest.mark.parametrize("attack", ("source", "parent"))
def test_wrong_physical_source_or_parent_fails_closed(attack: str) -> None:
    initial = _initial_state()
    original = _state_bytes(initial)

    def hostile(state: ParticleState) -> IRWRK3Field:
        source = state
        token = PARENT_TOKEN
        if attack == "source":
            shifted = np.array(state.positions, copy=True)
            shifted[0, 0] = np.nextafter(shifted[0, 0], np.inf)
            source = make_particle_state(shifted, state.gamma, state.sigma)
        else:
            token = "wrong-parent"
        return make_ir_wrk3_field(
            source,
            np.zeros_like(source.positions),
            np.zeros((len(source.sigma), 3, 3)),
            parent_token=token,
        )

    with pytest.raises(ValueError, match="source|stage-pre|parent"):
        ir_wrk3_stream_macro(initial, 0.01, hostile, parent_token=PARENT_TOKEN)
    assert _state_bytes(initial) == original


def test_one_ulp_wrong_tracer_attestation_fails_closed() -> None:
    initial = _initial_state()
    tracers = _tracers()

    def hostile(
        state: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        shifted = np.array(tracer_pre, copy=True)
        shifted[0, 0] = np.nextafter(shifted[0, 0], np.inf)
        return make_ir_wrk3_tracer_field(
            state,
            shifted,
            np.zeros_like(shifted),
            parent_token=parent_token,
        )

    with pytest.raises(ValueError, match="tracer.*state|digest|attestation"):
        ir_wrk3_stream_macro(
            initial,
            0.01,
            _affine_field(JACOBIANS[0]),
            tracer_positions=tracers,
            tracer_field_evaluator=hostile,
            parent_token=PARENT_TOKEN,
        )


def test_callback_in_place_code_drift_fails_and_clean_retry_succeeds() -> None:
    callback_calls = 0

    def evil(_state: object) -> object:
        raise AssertionError("drifted callback must never be called")

    def hostile(state: ParticleState) -> IRWRK3Field:
        nonlocal callback_calls
        callback_calls += 1
        response = make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
            parent_token=PARENT_TOKEN,
        )
        hostile.__code__ = evil.__code__
        return response

    original_code = hostile.__code__
    try:
        with pytest.raises(RuntimeError, match="callback|code|drift"):
            ir_wrk3_stream_macro(
                _initial_state(), 0.01, hostile, parent_token=PARENT_TOKEN
            )
        assert callback_calls == 1
    finally:
        hostile.__code__ = original_code
    clean = ir_wrk3_stream_macro(
        _initial_state(),
        0.01,
        _affine_field(JACOBIANS[0]),
        parent_token=PARENT_TOKEN,
    )
    validate_ir_wrk3_stream_result(clean)


def test_n128_small_cloud_has_384_compact_rows_and_no_retained_stage_arrays() -> None:
    initial = make_particle_state(
        ((0.0, 0.0, 0.0), (0.1, -0.03, 0.02)),
        ((0.2, -0.1, 0.05), (-0.07, 0.15, 0.03)),
        (0.09, 0.08),
    )
    tracers = np.asarray(((0.04, 0.02, -0.01),), dtype=np.float64)
    result = ir_wrk3_stream_macro(
        initial,
        0.01,
        _affine_field(JACOBIANS[1]),
        transport_substeps=128,
        tracer_positions=tracers,
        tracer_field_evaluator=_affine_tracer(JACOBIANS[1]),
        parent_token=PARENT_TOKEN,
    )
    assert len(result.stages) == 384
    assert result.counters.substep_count == 128
    assert result.counters.stage_count == 384
    assert result.counters.compact_stage_record_count == 384
    assert result.counters.retained_stage_array_count == 0
    for record in result.stages:
        assert type(record) is IRWRK3StreamStageRecord
        assert all(
            not isinstance(getattr(record, field.name), np.ndarray)
            for field in fields(record)
        )
        assert (
            len(record.evidence.payload) <= stream_module.MAX_EVIDENCE_BYTES_PER_STAGE
        )
    assert result.counters.evidence_byte_count <= stream_module.MAX_TOTAL_EVIDENCE_BYTES


def test_caps_underflow_and_evidence_limits_fail_before_field_or_materialization() -> (
    None
):
    materializations = 0
    field_calls = 0

    class LazyArrayTrap:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
            nonlocal materializations
            materializations += 1
            raise AssertionError("oversize array must not be materialized")

    def field(state: ParticleState) -> IRWRK3Field:
        nonlocal field_calls
        field_calls += 1
        return make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
            parent_token=PARENT_TOKEN,
        )

    oversize = stream_module.MAX_PARTICLE_COUNT + 1
    forged = ParticleState(
        LazyArrayTrap((oversize, 3)),  # type: ignore[arg-type]
        LazyArrayTrap((oversize, 3)),  # type: ignore[arg-type]
        LazyArrayTrap((oversize,)),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="cap"):
        ir_wrk3_stream_macro(forged, 0.01, field)
    assert materializations == 0
    assert field_calls == 0

    with pytest.raises(ValueError, match="substeps|cap"):
        ir_wrk3_stream_macro(
            _initial_state(),
            0.01,
            field,
            transport_substeps=stream_module.MAX_SUBSTEPS + 1,
        )
    with pytest.raises(ValueError, match="underflow|finite|positive"):
        ir_wrk3_stream_macro(
            _initial_state(),
            float(np.nextafter(0.0, 1.0)),
            field,
            transport_substeps=2,
        )
    assert field_calls == 0
    with pytest.raises(ValueError, match="evidence byte cap"):
        make_ir_wrk3_stream_evidence(
            "oversize",
            b"x" * (stream_module.MAX_EVIDENCE_BYTES_PER_STAGE + 1),
        )


def test_active_near_zero_stops_before_field_but_one_ulp_above_runs() -> None:
    calls = 0

    def field(state: ParticleState) -> IRWRK3Field:
        nonlocal calls
        calls += 1
        return make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
            parent_token=PARENT_TOKEN,
        )

    below = np.nextafter(ACTIVE_GAMMA_MAXABS_MIN, 0.0)
    bad = make_particle_state(((0.0, 0.0, 0.0),), ((below, 0.0, 0.0),), (0.1,))
    with pytest.raises(IRWRK3StreamStopped, match="near-zero") as caught:
        ir_wrk3_stream_macro(bad, 0.01, field, parent_token=PARENT_TOKEN)
    stopped = _assert_failure_journal(
        caught.value,
        phase="invariant_reference",
        coordinate=(1, 1),
        stage_began=False,
        completed_count=0,
    )
    assert type(stopped.original_cause) is ValueError
    assert calls == 0
    above = np.nextafter(ACTIVE_GAMMA_MAXABS_MIN, np.inf)
    good = make_particle_state(((0.0, 0.0, 0.0),), ((above, 0.0, 0.0),), (0.1,))
    result = ir_wrk3_stream_macro(good, 0.01, field, parent_token=PARENT_TOKEN)
    assert calls == 3
    assert result.final_state.gamma[0, 0] == above


def test_result_tree_is_exact_readonly_deterministic_and_validator_fails_closed() -> (
    None
):
    initial = _initial_state()
    original = _state_bytes(initial)
    first = ir_wrk3_stream_macro(
        initial,
        0.0175,
        _affine_field(JACOBIANS[2]),
        transport_substeps=4,
        tracer_positions=_tracers(),
        tracer_field_evaluator=_affine_tracer(JACOBIANS[2]),
        parent_token=PARENT_TOKEN,
    )
    after_first_registry = _registry_snapshot()
    second = ir_wrk3_stream_macro(
        initial,
        0.0175,
        _affine_field(JACOBIANS[2]),
        transport_substeps=4,
        tracer_positions=_tracers(),
        tracer_field_evaluator=_affine_tracer(JACOBIANS[2]),
        parent_token=PARENT_TOKEN,
    )
    assert second is first
    assert _registry_snapshot() == after_first_registry
    assert _state_bytes(initial) == original
    assert type(first) is IRWRK3StreamResult
    assert type(first.counters) is IRWRK3StreamCounters
    assert type(first.stages) is tuple
    assert all(type(record) is IRWRK3StreamStageRecord for record in first.stages)
    assert first.result_sha256 == second.result_sha256
    assert first.stage_chain_sha256 == second.stage_chain_sha256
    assert [record.record_sha256 for record in first.stages] == [
        record.record_sha256 for record in second.stages
    ]
    assert _state_bytes(first.final_state) == _state_bytes(second.final_state)
    _assert_frozen_float64(first.final_state.positions, (4, 3))
    _assert_frozen_float64(first.final_state.gamma, (4, 3))
    _assert_frozen_float64(first.final_state.sigma, (4,))
    _assert_frozen_float64(first.final_tracer_positions, (2, 3))
    assert validate_ir_wrk3_stream_result(first) is first

    bad_evidence = replace(first.stages[0].evidence, evidence_sha256="0" * 64)
    bad_stage = replace(first.stages[0], evidence=bad_evidence)
    with pytest.raises(ValueError, match="evidence|digest|record|chain"):
        validate_ir_wrk3_stream_result(
            replace(first, stages=(bad_stage, *first.stages[1:]))
        )
    with pytest.raises(ValueError, match="live issued|exact live"):
        validate_ir_wrk3_stream_result(replace(first))

    last = first.stages[-1]
    tampered_draft = replace(
        last,
        invariant_residual_over_slog_max=float(
            np.nextafter(stream_module.INVARIANT_LOG_ATOL, np.inf)
        ),
        record_sha256="",
        chain_sha256="",
    )
    tampered_record_sha = stream_module._stage_record_sha256(tampered_draft)
    tampered_chain_sha = hashlib.sha256(
        (
            "fluxv-ir-wrk3-stream-stage-link-v1\0"
            + tampered_draft.previous_chain_sha256
            + tampered_record_sha
        ).encode("ascii")
    ).hexdigest()
    tampered_last = replace(
        tampered_draft,
        record_sha256=tampered_record_sha,
        chain_sha256=tampered_chain_sha,
    )
    tampered_result = replace(
        first,
        stages=(*first.stages[:-1], tampered_last),
        stage_chain_sha256=tampered_chain_sha,
        result_sha256="",
    )
    tampered_result = replace(
        tampered_result,
        result_sha256=stream_module._result_sha256(tampered_result),
    )
    with pytest.raises(ValueError, match="normalized invariant residual exceeds"):
        stream_module._validate_result_tree(tampered_result)
    with pytest.raises(ValueError, match="normalized invariant residual exceeds"):
        validate_ir_wrk3_stream_result(tampered_result)

    assert not hasattr(stream_module, "_LIVE_STREAM_RESULT_REGISTRY")
    assert not hasattr(stream_module, "_LIVE_STREAM_RESULT_LOCK")
    assert not any(
        "role" in name or "issuance" in name
        for name in (ir_wrk3_stream_macro.__kwdefaults__ or ())
    )
    assert ir_wrk3_stream_macro.__closure__ is None
    assert not any(
        name == "_STREAM_RESULT_ISSUANCE_ROLE" for name in vars(stream_module)
    )
    forged = replace(first)
    before_registry = _registry_snapshot()
    with pytest.raises(ValueError, match="trusted macro|registration"):
        stream_module._register_result(forged)
    assert _registry_snapshot() == before_registry
    with pytest.raises(ValueError, match="live issued|exact live"):
        validate_ir_wrk3_stream_result(forged)

    modified = replace(
        first,
        delta_time=float(np.nextafter(first.delta_time, np.inf)),
        result_sha256="",
    )
    modified = replace(
        modified,
        result_sha256=stream_module._result_sha256(modified),
    )
    assert stream_module._validate_result_tree(modified) is modified
    with pytest.raises(ValueError, match="trusted macro|registration"):
        stream_module._register_result(modified)
    assert _registry_snapshot() == before_registry
    with pytest.raises(ValueError, match="live issued|exact live"):
        validate_ir_wrk3_stream_result(modified)

    clean = ir_wrk3_stream_macro(
        initial,
        0.018,
        _affine_field(JACOBIANS[2]),
        transport_substeps=4,
        tracer_positions=_tracers(),
        tracer_field_evaluator=_affine_tracer(JACOBIANS[2]),
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_stream_result(clean) is clean
