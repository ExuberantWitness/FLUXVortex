from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from math import log2, pi, sqrt
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import erf

import fluxvortex.rvpm_ir_wrk3 as ir_module
from fluxvortex.rvpm_ir_wrk3 import (
    ACTIVE_GAMMA_MAXABS_MIN,
    FORMULATION_F,
    FORMULATION_G,
    INVARIANT_LOG_ATOL,
    IRWRK3Counters,
    IRWRK3Field,
    IRWRK3StageRecord,
    IRWRK3StepResult,
    IRWRK3TracerField,
    InvariantReference,
    freeze_invariant_reference,
    ir_wrk3_step_with_external_field,
    make_ir_wrk3_field,
    make_ir_wrk3_tracer_field,
    reconstruct_sigma,
    validate_ir_wrk3_result,
)
from fluxvortex.rvpm_transport import ParticleState, make_particle_state


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "fluxv_v5_nextgen_20260814/refine-logs/v5h11_ir_wrk3/EXPERIMENT_PLAN.md"
)
FREEZE = PLAN.with_name("PREREG_FREEZE.json")
DERIVATION = ROOT / "DERIVATION_PACKAGE.md"

PLAN_SHA256 = "66f610963d976cd62ef0baf4aa8bb2c093428860cbbc6269750d99f2cb82bd0c"
FREEZE_SHA256 = "3d8cc957718499848bd47631707aba1cd5711532611ee7b3349ad08e789ef386"
DERIVATION_SHA256 = "430d5280ff96bfd0e29002e7da6b7419eeed2b9d515fcc2a15679fdd875ff1b9"

HORIZON_S = 0.11125
SUBSTEP_LEVELS = (4, 8, 16, 32)
MINIMUM_ORDER = 2.8
FINE_RELATIVE_L2_MAX = 1.0e-7
B2_HORIZON_S = 0.04
B2_FINE_RELATIVE_L2_MAX = 1.0e-10
B2_ORDER_MAX = 3.2
B2_DOP853_RTOL = 2.0e-13
B2_DOP853_ATOL = 2.0e-15
ROUND_MULTIPLIER = 512.0
PARENT_TOKEN = "v5h11-manufactured-frozen-parent-v1"
TRANSLATION = np.asarray((0.7, -0.03, 0.02), dtype=np.float64)
ACTIVE_POSITIONS = np.asarray(
    ((0.1, 0.2, 0.3), (-0.2, 0.05, 0.15), (0.03, -0.11, 0.07)),
    dtype=np.float64,
)
ACTIVE_GAMMA = np.asarray(
    ((0.4, -0.2, 0.3), (-0.1, 0.35, 0.2), (0.22, 0.08, -0.31)),
    dtype=np.float64,
)
ACTIVE_SIGMA = np.asarray((0.085, 0.07, 0.12), dtype=np.float64)
ZERO_POSITION = np.asarray((-0.04, 0.09, 0.21), dtype=np.float64)
ZERO_GAMMA = np.asarray((0.0, -0.0, 0.0), dtype=np.float64)
ZERO_SIGMA = np.float64(0.095)
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


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(1.0e-300, float(np.linalg.norm(expected.ravel())))
    return float(np.linalg.norm((actual - expected).ravel()) / scale)


def _full_affine_state() -> ParticleState:
    return make_particle_state(
        np.vstack((ACTIVE_POSITIONS, ZERO_POSITION)),
        np.vstack((ACTIVE_GAMMA, ZERO_GAMMA)),
        np.concatenate((ACTIVE_SIGMA, (ZERO_SIGMA,))),
    )


def _state_bytes(state: ParticleState) -> tuple[bytes, bytes, bytes]:
    return (
        state.positions.tobytes(order="C"),
        state.gamma.tobytes(order="C"),
        state.sigma.tobytes(order="C"),
    )


def _affine_field(
    jacobian: np.ndarray,
    *,
    calls: list[tuple[bytes, bytes, bytes]] | None = None,
    parent_token: str = PARENT_TOKEN,
) -> Callable[[ParticleState], IRWRK3Field]:
    def evaluate(state: ParticleState) -> IRWRK3Field:
        if calls is not None:
            calls.append(_state_bytes(state))
        velocity = state.positions @ jacobian.T + TRANSLATION
        repeated_jacobian = np.repeat(jacobian[None, :, :], len(state.sigma), axis=0)
        return make_ir_wrk3_field(
            state,
            velocity,
            repeated_jacobian,
            parent_token=parent_token,
        )

    return evaluate


def _affine_exact(
    initial: ParticleState, jacobian: np.ndarray, horizon: float
) -> ParticleState:
    augmented = np.zeros((4, 4), dtype=np.float64)
    augmented[:3, :3] = jacobian
    augmented[:3, 3] = TRANSLATION
    position_map = expm(horizon * augmented)
    positions = np.column_stack((initial.positions, np.ones(len(initial.sigma))))
    positions = (positions @ position_map.T)[:, :3]

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


def _assert_readonly_float64(array: np.ndarray, shape: tuple[int, ...]) -> None:
    assert type(array) is np.ndarray
    assert array.dtype == np.dtype(np.float64)
    assert array.shape == shape
    assert array.flags.c_contiguous
    assert not array.flags.writeable
    assert np.all(np.isfinite(array))


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


def _assert_stage_roundoff(
    stage: IRWRK3StageRecord, reference: InvariantReference
) -> None:
    active = ~reference.exact_zero_mask
    independent_invariant = np.zeros(len(stage.post.sigma), dtype=np.float64)
    if np.any(active):
        norm = _scaled_norms(stage.post.gamma)[active]
        log_norm_ratio = np.log(norm / reference.gamma_norm_star[active])
        twice_log_sigma_ratio = 2.0 * np.log(
            stage.post.sigma[active] / reference.sigma_star[active]
        )
        independent_invariant[active] = np.abs(log_norm_ratio + twice_log_sigma_ratio)
        scale = np.maximum.reduce(
            (
                np.ones(np.count_nonzero(active)),
                np.abs(log_norm_ratio),
                np.abs(twice_log_sigma_ratio),
            )
        )
        assert np.all(
            independent_invariant[active]
            <= ROUND_MULTIPLIER * np.finfo(np.float64).eps * scale
        )
    np.testing.assert_allclose(
        stage.invariant_log_residual,
        independent_invariant,
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )

    norms = _scaled_norms(stage.pre.gamma)
    independent_chain = np.zeros(len(norms), dtype=np.float64)
    if np.any(active):
        gamma_projection = (
            np.einsum("ni,ni->n", stage.pre.gamma[active], stage.rhs.gamma_rate[active])
            / norms[active] ** 2
        )
        sigma_log_rate = (
            stage.rhs.sigma_rate_diagnostic[active] / stage.pre.sigma[active]
        )
        numerator = np.abs(sigma_log_rate + 0.5 * gamma_projection)
        denominator = np.maximum.reduce(
            (
                np.ones(np.count_nonzero(active)),
                np.abs(sigma_log_rate),
                0.5 * np.abs(gamma_projection),
            )
        )
        independent_chain[active] = numerator / denominator
    np.testing.assert_allclose(
        stage.rhs.chain_rule_relative_residual,
        independent_chain,
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )
    assert np.max(independent_chain, initial=0.0) <= INVARIANT_LOG_ATOL


def test_preregistered_b0_inputs_and_governance_hashes_are_exact() -> None:
    assert _file_sha256(FREEZE) == FREEZE_SHA256
    assert _file_sha256(PLAN) == PLAN_SHA256
    assert _file_sha256(DERIVATION) == DERIVATION_SHA256
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen["not_a_v5h10_amendment"] is True
    assert frozen["paper_observation_access"] == "sealed"
    assert frozen["paper_scoring_allowed"] is False
    assert frozen["method"]["name"] == "IR-WRK3"
    assert frozen["method"]["formulation_f"] == FORMULATION_F == 0.0
    assert frozen["method"]["formulation_g"] == FORMULATION_G == 0.2
    assert frozen["method"]["invariant_reference_freeze_count_per_macro_layer"] == 1
    assert (
        frozen["method"]["invariant_reference_rebase_per_inner_substep_allowed"]
        is False
    )
    assert tuple(frozen["manufactured_gate"]["substep_levels"]) == SUBSTEP_LEVELS
    assert frozen["manufactured_gate"]["horizon_s"] == HORIZON_S
    assert frozen["manufactured_gate"]["minimum_observed_order"] == MINIMUM_ORDER
    assert frozen["manufactured_gate"]["finest_relative_l2_max"] == FINE_RELATIVE_L2_MAX
    fixtures = frozen["manufactured_fixtures"]
    np.testing.assert_array_equal(fixtures["translation"], TRANSLATION)
    np.testing.assert_array_equal(fixtures["active_positions"], ACTIVE_POSITIONS)
    np.testing.assert_array_equal(fixtures["active_gamma"], ACTIVE_GAMMA)
    np.testing.assert_array_equal(fixtures["active_sigma"], ACTIVE_SIGMA)
    np.testing.assert_array_equal(fixtures["exact_zero_position"], ZERO_POSITION)
    np.testing.assert_array_equal(fixtures["exact_zero_gamma"], ZERO_GAMMA)
    assert fixtures["exact_zero_sigma"] == ZERO_SIGMA
    for actual, expected in zip(fixtures["jacobians"], JACOBIANS, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("jacobian", JACOBIANS, ids=("stretch", "rotation", "shear"))
def test_affine_expm_oracle_is_third_order_and_preserves_invariant(
    jacobian: np.ndarray,
) -> None:
    initial = _full_affine_state()
    expected = _affine_exact(initial, jacobian, HORIZON_S)
    errors: dict[str, list[float]] = {"positions": [], "gamma": [], "sigma": []}

    for substeps in SUBSTEP_LEVELS:
        calls: list[tuple[bytes, bytes, bytes]] = []
        result = ir_wrk3_step_with_external_field(
            initial,
            HORIZON_S,
            _affine_field(jacobian, calls=calls),
            transport_substeps=substeps,
            parent_token=PARENT_TOKEN,
        )
        assert validate_ir_wrk3_result(result) is result
        assert len(calls) == 3 * substeps
        assert result.counters.invariant_reference_freeze_count == 1
        assert result.counters.stage_count == 3 * substeps
        assert result.counters.physical_field_call_count == 3 * substeps
        assert result.counters.stage_pre_reconstruction_count == 3 * substeps
        assert result.counters.stage_post_reconstruction_count == 3 * substeps
        assert result.counters.physical_rhs_call_count == 3 * substeps
        assert result.counters.sigma_storage_update_count == 0
        assert result.counters.relaxation_call_count == 0
        assert result.final_state.gamma[-1].tobytes() == ZERO_GAMMA.tobytes()
        assert result.final_state.sigma[-1].tobytes() == ZERO_SIGMA.tobytes()
        for stage in result.stages:
            _assert_stage_roundoff(stage, result.invariant_reference)
        for name in errors:
            errors[name].append(
                _relative_l2(getattr(result.final_state, name), getattr(expected, name))
            )

    for name, channel_errors in errors.items():
        assert channel_errors[-1] <= FINE_RELATIVE_L2_MAX, (name, channel_errors)
        orders = tuple(
            log2(coarse / fine)
            for coarse, fine in zip(channel_errors[:-1], channel_errors[1:])
        )
        assert orders[-2] >= MINIMUM_ORDER, (name, channel_errors, orders)
        assert orders[-1] >= MINIMUM_ORDER, (name, channel_errors, orders)


def test_exact_zero_signed_bits_survive_while_position_advects() -> None:
    initial = make_particle_state([ZERO_POSITION], [ZERO_GAMMA], [ZERO_SIGMA])
    result = ir_wrk3_step_with_external_field(
        initial,
        HORIZON_S,
        _affine_field(JACOBIANS[2]),
        transport_substeps=4,
        parent_token=PARENT_TOKEN,
    )
    assert result.final_state.gamma.tobytes() == initial.gamma.tobytes()
    assert result.final_state.sigma.tobytes() == initial.sigma.tobytes()
    assert not np.array_equal(result.final_state.positions, initial.positions)
    assert result.invariant_reference.exact_zero_mask.tolist() == [True]


@pytest.mark.parametrize(
    "magnitude",
    (
        np.nextafter(ACTIVE_GAMMA_MAXABS_MIN, 0.0),
        ACTIVE_GAMMA_MAXABS_MIN,
        np.nextafter(0.0, 1.0),
    ),
    ids=("below", "equal", "subnormal"),
)
def test_active_at_or_below_threshold_stops_before_first_field_call(
    magnitude: float,
) -> None:
    state = make_particle_state([[0.0, 0.0, 0.0]], [[magnitude, 0.0, 0.0]], [0.1])
    calls = 0

    def explode(_state: ParticleState) -> IRWRK3Field:
        nonlocal calls
        calls += 1
        raise AssertionError("near-zero input reached the field")

    with pytest.raises(ValueError, match="near-zero threshold"):
        ir_wrk3_step_with_external_field(state, 0.01, explode)
    assert calls == 0


def test_one_ulp_above_threshold_uses_scaled_norm_and_runs() -> None:
    magnitude = np.nextafter(ACTIVE_GAMMA_MAXABS_MIN, np.inf)
    state = make_particle_state([[0.0, 0.0, 0.0]], [[magnitude, 0.0, 0.0]], [0.1])
    calls: list[tuple[bytes, bytes, bytes]] = []
    result = ir_wrk3_step_with_external_field(
        state,
        0.01,
        _affine_field(np.zeros((3, 3)), calls=calls),
        parent_token=PARENT_TOKEN,
    )
    assert len(calls) == 3
    assert result.final_state.gamma[0, 0] == magnitude
    assert abs(result.final_state.sigma[0] / 0.1 - 1.0) <= INVARIANT_LOG_ATOL


def test_reconstruction_fails_before_float64_overflow_or_underflow() -> None:
    max_state = make_particle_state(
        [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [np.finfo(np.float64).max]
    )
    max_reference = freeze_invariant_reference(max_state)
    with pytest.raises(FloatingPointError, match="outside finite positive"):
        reconstruct_sigma([[0.25, 0.0, 0.0]], max_reference)

    min_positive = np.nextafter(0.0, 1.0)
    min_state = make_particle_state(
        [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [min_positive]
    )
    min_reference = freeze_invariant_reference(min_state)
    with pytest.raises(FloatingPointError, match="outside finite positive"):
        reconstruct_sigma([[4.0, 0.0, 0.0]], min_reference)


def test_reconstruction_rejects_zero_active_classification_change() -> None:
    active = make_particle_state([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [0.1])
    active_reference = freeze_invariant_reference(active)
    with pytest.raises(ValueError, match="classification changed"):
        reconstruct_sigma([[0.0, -0.0, 0.0]], active_reference)

    zero = make_particle_state([[0.0, 0.0, 0.0]], [[0.0, -0.0, 0.0]], [0.1])
    zero_reference = freeze_invariant_reference(zero)
    with pytest.raises(ValueError, match="classification changed"):
        reconstruct_sigma([[1.0, 0.0, 0.0]], zero_reference)


@pytest.mark.parametrize(
    ("formulation_f", "formulation_g"),
    ((0.1, 0.16), (False, 0.2), ("0.0", 0.2)),
    ids=("same-c-forbidden", "bool", "nonreal"),
)
def test_wrong_or_nonexact_formulation_fails_before_field(
    formulation_f: object, formulation_g: object
) -> None:
    calls = 0

    def explode(_state: ParticleState) -> IRWRK3Field:
        nonlocal calls
        calls += 1
        raise AssertionError("invalid formulation reached the field")

    with pytest.raises(ValueError, match="formulation|defined only"):
        ir_wrk3_step_with_external_field(
            _full_affine_state(),
            0.01,
            explode,
            formulation_f=formulation_f,  # type: ignore[arg-type]
            formulation_g=formulation_g,  # type: ignore[arg-type]
        )
    assert calls == 0


def test_macro_freeze_storage_resets_and_negative_trace_validation() -> None:
    result = ir_wrk3_step_with_external_field(
        _full_affine_state(),
        HORIZON_S,
        _affine_field(JACOBIANS[0]),
        transport_substeps=4,
        parent_token=PARENT_TOKEN,
    )
    assert result.operator_trace.count("invariant_reference_frozen") == 1
    assert sum(item.endswith("storage_reset") for item in result.operator_trace) == 4
    assert result.counters.invariant_reference_freeze_count == 1
    assert result.counters.substep_count == 4
    assert result.counters.sigma_storage_update_count == 0
    assert result.counters.relaxation_call_count == 0
    for substep in range(1, 5):
        first = result.stages[3 * (substep - 1)]
        assert first.substep == substep and first.stage == 1
        np.testing.assert_array_equal(
            first.position_storage_pre, np.zeros_like(first.position_storage_pre)
        )
        np.testing.assert_array_equal(
            first.gamma_storage_pre, np.zeros_like(first.gamma_storage_pre)
        )

    for field, value, message in (
        ("invariant_reference_freeze_count", 2, "frozen exactly once"),
        ("sigma_storage_update_count", 1, "forbidden sigma storage"),
        ("relaxation_call_count", 1, "forbidden sigma storage or relaxation"),
    ):
        forged_counters = replace(result.counters, **{field: value})
        forged = replace(result, counters=forged_counters)
        with pytest.raises(ValueError, match=message):
            validate_ir_wrk3_result(forged)


def test_output_types_readonly_deterministic_and_input_unmodified() -> None:
    initial = _full_affine_state()
    original = _state_bytes(initial)
    first = ir_wrk3_step_with_external_field(
        initial,
        HORIZON_S,
        _affine_field(JACOBIANS[1]),
        transport_substeps=8,
        parent_token=PARENT_TOKEN,
    )
    second = ir_wrk3_step_with_external_field(
        initial,
        HORIZON_S,
        _affine_field(JACOBIANS[1]),
        transport_substeps=8,
        parent_token=PARENT_TOKEN,
    )
    assert _state_bytes(initial) == original
    assert type(first) is IRWRK3StepResult
    assert type(first.invariant_reference) is InvariantReference
    assert type(first.counters) is IRWRK3Counters
    assert all(type(stage) is IRWRK3StageRecord for stage in first.stages)
    assert first.result_sha256 == second.result_sha256
    assert first.stage_chain_sha256 == second.stage_chain_sha256
    assert first.operator_trace == second.operator_trace
    assert _state_bytes(first.final_state) == _state_bytes(second.final_state)
    _assert_readonly_float64(first.final_state.positions, (4, 3))
    _assert_readonly_float64(first.final_state.gamma, (4, 3))
    _assert_readonly_float64(first.final_state.sigma, (4,))
    _assert_readonly_float64(first.invariant_reference.gamma_norm_star, (4,))
    _assert_readonly_float64(first.invariant_reference.sigma_star, (4,))
    assert not first.invariant_reference.exact_zero_mask.flags.writeable
    for stage in first.stages:
        for array in (
            stage.pre.positions,
            stage.pre.gamma,
            stage.pre.sigma,
            stage.post.positions,
            stage.post.gamma,
            stage.post.sigma,
            stage.field.velocity,
            stage.field.jacobian,
            stage.rhs.gamma_rate,
            stage.invariant_log_residual,
        ):
            assert array.flags.c_contiguous and not array.flags.writeable


def test_stale_field_binding_and_wrong_parent_fail_closed() -> None:
    initial = _full_affine_state()
    stale = make_ir_wrk3_field(
        initial,
        initial.positions @ JACOBIANS[0].T + TRANSLATION,
        np.repeat(JACOBIANS[0][None, :, :], 4, axis=0),
        parent_token=PARENT_TOKEN,
    )

    with pytest.raises(ValueError, match="stage-pre state"):
        ir_wrk3_step_with_external_field(
            initial,
            HORIZON_S,
            lambda _state: stale,
            transport_substeps=2,
            parent_token=PARENT_TOKEN,
        )

    def wrong_parent(state: ParticleState) -> IRWRK3Field:
        return make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
            parent_token="wrong-parent",
        )

    with pytest.raises(ValueError, match="parent token"):
        ir_wrk3_step_with_external_field(
            initial, HORIZON_S, wrong_parent, parent_token=PARENT_TOKEN
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


def _independent_b2_dop853_reference(
    initial: ParticleState,
    tracers: np.ndarray,
    external_jacobian: np.ndarray,
    external_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    particle_count = len(initial.sigma)
    tracer_count = len(tracers)
    gamma_norm_star = _scaled_norms(initial.gamma)
    sigma_star = np.array(initial.sigma, dtype=np.float64, copy=True)
    exact_zero = gamma_norm_star == 0.0
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
        if np.any((norms == 0.0) != exact_zero):
            raise FloatingPointError(
                "independent reference changed zero classification"
            )
        sigma = sigma_star.copy()
        active = ~exact_zero
        sigma[active] *= np.sqrt(gamma_norm_star[active] / norms[active])

        self_velocity, self_jacobian = _hand_gaussian_field(
            positions, gamma, sigma, positions
        )
        velocity = self_velocity + positions @ external_jacobian.T
        velocity += external_translation
        total_jacobian = self_jacobian + external_jacobian[None, :, :]
        stretching = np.einsum("nji,nj->ni", total_jacobian, gamma)
        chi = np.zeros(particle_count, dtype=np.float64)
        if np.any(active):
            chi[active] = (
                np.einsum("ni,ni->n", stretching[active], gamma[active])
                / norms[active] ** 2
            )
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
    assert solution.t.tolist() == [B2_HORIZON_S]
    final = solution.y[:, -1]
    position_end = 3 * particle_count
    gamma_end = 6 * particle_count
    positions = final[:position_end].reshape((particle_count, 3))
    gamma = final[position_end:gamma_end].reshape((particle_count, 3))
    final_tracers = final[gamma_end:].reshape((tracer_count, 3))
    final_norms = _scaled_norms(gamma)
    sigma = sigma_star.copy()
    active = ~exact_zero
    sigma[active] *= np.sqrt(gamma_norm_star[active] / final_norms[active])
    return positions, gamma, sigma, final_tracers, solution.nfev


def test_b2_dop853_oracle_is_third_order_for_physical_and_tracer_endpoints() -> None:
    initial, tracers, external_jacobian, external_translation = _b2_fixture()
    (
        expected_positions,
        expected_gamma,
        expected_sigma,
        expected_tracers,
        nfev,
    ) = _independent_b2_dop853_reference(
        initial, tracers, external_jacobian, external_translation
    )
    assert nfev > 0
    errors: dict[str, list[float]] = {
        "positions": [],
        "gamma": [],
        "sigma": [],
        "tracers": [],
    }

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
        assert parent_token == PARENT_TOKEN
        self_velocity, _ = _hand_gaussian_field(
            state.positions, state.gamma, state.sigma, tracer_pre
        )
        velocity = self_velocity + tracer_pre @ external_jacobian.T
        velocity += external_translation
        return make_ir_wrk3_tracer_field(
            state, tracer_pre, velocity, parent_token=parent_token
        )

    for substeps in SUBSTEP_LEVELS:
        result = ir_wrk3_step_with_external_field(
            initial,
            B2_HORIZON_S,
            physical,
            transport_substeps=substeps,
            tracer_positions=tracers,
            tracer_field_evaluator=tracer,
            parent_token=PARENT_TOKEN,
        )
        errors["positions"].append(
            _relative_l2(result.final_state.positions, expected_positions)
        )
        errors["gamma"].append(_relative_l2(result.final_state.gamma, expected_gamma))
        errors["sigma"].append(_relative_l2(result.final_state.sigma, expected_sigma))
        errors["tracers"].append(
            _relative_l2(result.final_tracer_positions, expected_tracers)
        )

    for name, channel_errors in errors.items():
        orders = tuple(
            log2(coarse / fine)
            for coarse, fine in zip(channel_errors[:-1], channel_errors[1:])
        )
        assert all(MINIMUM_ORDER <= order <= B2_ORDER_MAX for order in orders[-2:]), (
            name,
            channel_errors,
            orders,
        )
        assert channel_errors[-1] <= B2_FINE_RELATIVE_L2_MAX, (
            name,
            channel_errors,
        )


def test_minimal_b2_tracer_uses_same_stage_source_and_storage_reset() -> None:
    initial, tracers, external_jacobian, external_translation = _b2_fixture()
    events: list[tuple[str, tuple[bytes, bytes, bytes]]] = []

    def physical(state: ParticleState) -> IRWRK3Field:
        key = _state_bytes(state)
        events.append(("physical", key))
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
        key = _state_bytes(state)
        assert events[-1] == ("physical", key)
        assert parent_token == PARENT_TOKEN
        events.append(("tracer", key))
        self_velocity, _ = _hand_gaussian_field(
            state.positions, state.gamma, state.sigma, tracer_pre
        )
        velocity = self_velocity + tracer_pre @ external_jacobian.T
        velocity += external_translation
        return make_ir_wrk3_tracer_field(
            state, tracer_pre, velocity, parent_token=parent_token
        )

    substeps = 4
    result = ir_wrk3_step_with_external_field(
        initial,
        B2_HORIZON_S,
        physical,
        transport_substeps=substeps,
        tracer_positions=tracers,
        tracer_field_evaluator=tracer,
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_result(result) is result
    assert [kind for kind, _ in events] == ["physical", "tracer"] * (3 * substeps)
    assert result.counters.physical_field_call_count == 3 * substeps
    assert result.counters.tracer_field_call_count == 3 * substeps
    assert result.counters.tracer_storage_reset_count == substeps
    assert result.counters.invariant_reference_freeze_count == 1
    _assert_readonly_float64(result.final_tracer_positions, tracers.shape)
    assert not np.array_equal(result.final_tracer_positions, tracers)
    for substep in range(1, substeps + 1):
        first = result.stages[3 * (substep - 1)]
        np.testing.assert_array_equal(
            first.tracer_storage_pre, np.zeros_like(first.tracer_storage_pre)
        )
        assert first.field.source_state_sha256 == first.source_state_sha256
        assert first.field.parent_token == result.parent_token == PARENT_TOKEN
        assert first.tracer_pre.flags.writeable is False
        assert first.tracer_velocity.flags.writeable is False
        assert first.tracer_post.flags.writeable is False


def _simple_result_with_tracer() -> IRWRK3StepResult:
    initial, tracers, external_jacobian, external_translation = _b2_fixture()

    def physical(state: ParticleState) -> IRWRK3Field:
        return make_ir_wrk3_field(
            state,
            state.positions @ external_jacobian.T + external_translation,
            np.repeat(external_jacobian[None, :, :], len(state.sigma), axis=0),
            parent_token=PARENT_TOKEN,
        )

    def tracer(
        state: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        velocity = tracer_pre @ external_jacobian.T + external_translation
        return make_ir_wrk3_tracer_field(
            state, tracer_pre, velocity, parent_token=parent_token
        )

    return ir_wrk3_step_with_external_field(
        initial,
        0.01,
        physical,
        tracer_positions=tracers,
        tracer_field_evaluator=tracer,
        parent_token=PARENT_TOKEN,
    )


def test_validator_recomputes_stage_and_reference_content_digests() -> None:
    result = _simple_result_with_tracer()
    with pytest.raises(ValueError, match="stages.*tuple"):
        validate_ir_wrk3_result(replace(result, stages=list(result.stages)))

    second = replace(result.stages[1], a=0.0)
    forged_stages = (result.stages[0], second, result.stages[2])
    with pytest.raises(ValueError, match="coefficient|stage record|stage chain"):
        validate_ir_wrk3_result(replace(result, stages=forged_stages))

    bad_norm = np.frombuffer(
        np.full_like(result.invariant_reference.gamma_norm_star, 999.0).tobytes(),
        dtype=np.float64,
    )
    bad_reference = replace(
        result.invariant_reference,
        gamma_norm_star=bad_norm,
    )
    with pytest.raises(ValueError, match="reference|digest|ledger"):
        validate_ir_wrk3_result(replace(result, invariant_reference=bad_reference))

    mutable_velocity = result.stages[0].field.velocity.copy()
    mutable_field = replace(result.stages[0].field, velocity=mutable_velocity)
    mutable_stage = replace(result.stages[0], field=mutable_field)
    with pytest.raises(ValueError, match="readonly|C-contiguous"):
        validate_ir_wrk3_result(
            replace(result, stages=(mutable_stage, *result.stages[1:]))
        )

    mutable_base = result.stages[0].field.velocity.copy()
    readonly_alias = mutable_base.view()
    readonly_alias.flags.writeable = False
    aliased_field = replace(result.stages[0].field, velocity=readonly_alias)
    aliased_stage = replace(result.stages[0], field=aliased_field)
    with pytest.raises(ValueError, match="immutable|bytes|base chain"):
        validate_ir_wrk3_result(
            replace(result, stages=(aliased_stage, *result.stages[1:]))
        )
    mutable_base[0, 0] = 777.0


def test_validator_replays_rhs_storage_and_post_against_private_reseal() -> None:
    result = _simple_result_with_tracer()

    def reseal(
        forged_stage: IRWRK3StageRecord,
        *,
        final_state: ParticleState = result.final_state,
    ) -> IRWRK3StepResult:
        raw_stage = replace(forged_stage, trace_sha256="")
        sealed_stage = replace(
            raw_stage,
            trace_sha256=ir_module._stage_record_sha256(raw_stage),
        )
        stages = (*result.stages[:-1], sealed_stage)
        chain = sha256(
            (
                "fluxv-ir-wrk3-stage-chain-v1\0"
                + "".join(stage.trace_sha256 for stage in stages)
            ).encode("ascii")
        ).hexdigest()
        payload = {
            "domain": "fluxv-ir-wrk3-step-result-v1",
            "final_state": ir_module._state_sha256(final_state),
            "final_tracer": ir_module._array_sha256(result.final_tracer_positions),
            "reference": result.invariant_reference.reference_sha256,
            "stage_chain": chain,
            "delta_time": result.delta_time.hex(),
            "parent": result.parent_token,
            "counters": [
                getattr(result.counters, name)
                for name in result.counters.__dataclass_fields__
            ],
            "trace": list(result.operator_trace),
        }
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        return replace(
            result,
            final_state=final_state,
            stages=stages,
            stage_chain_sha256=chain,
            result_sha256=digest,
        )

    last = result.stages[-1]
    forged_rate = np.frombuffer(
        np.full_like(last.rhs.gamma_rate, 999.0).tobytes(), dtype=np.float64
    ).reshape(last.rhs.gamma_rate.shape)
    with pytest.raises(ValueError, match="RHS gamma_rate"):
        validate_ir_wrk3_result(
            reseal(replace(last, rhs=replace(last.rhs, gamma_rate=forged_rate)))
        )

    forged_storage = np.frombuffer(
        np.full_like(last.position_storage_post, 777.0).tobytes(),
        dtype=np.float64,
    ).reshape(last.position_storage_post.shape)
    with pytest.raises(ValueError, match="position storage post"):
        validate_ir_wrk3_result(
            reseal(replace(last, position_storage_post=forged_storage))
        )

    forged_positions = np.array(last.post.positions, copy=True)
    forged_positions[0, 0] += 0.25
    forged_post = ir_module._frozen_state(
        forged_positions, last.post.gamma, last.post.sigma
    )
    forged_last = replace(
        last,
        post=forged_post,
        post_state_sha256=ir_module._state_sha256(forged_post),
    )
    with pytest.raises(ValueError, match="stage post positions"):
        validate_ir_wrk3_result(reseal(forged_last, final_state=forged_post))

    with pytest.raises(ValueError, match="exact live issued"):
        validate_ir_wrk3_result(replace(result))


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("FORMULATION_F", 0.1),
        ("ACTIVE_GAMMA_MAXABS_MIN", 0.0),
        ("INVARIANT_LOG_ATOL", np.inf),
        ("MAX_SUBSTEPS", 1_000_000),
        ("MAX_RETAINED_STAGE_ROWS", 1_000_001),
    ),
)
def test_frozen_contract_global_drift_fails_before_field_and_clean_retry(
    monkeypatch: pytest.MonkeyPatch, name: str, value: object
) -> None:
    calls = 0

    def field(state: ParticleState) -> IRWRK3Field:
        nonlocal calls
        calls += 1
        return make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
        )

    original = getattr(ir_module, name)
    monkeypatch.setattr(ir_module, name, value)
    with pytest.raises(RuntimeError, match="contract global drift"):
        ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, field)
    assert calls == 0
    monkeypatch.setattr(ir_module, name, original)
    clean = ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, field)
    assert calls == 3
    validate_ir_wrk3_result(clean)


def test_numpy_runtime_drift_fails_before_field_and_restores_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def field(state: ParticleState) -> IRWRK3Field:
        nonlocal calls
        calls += 1
        return make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
        )

    original = np.array_equal
    monkeypatch.setattr(np, "array_equal", lambda *_args, **_kwargs: True)
    with pytest.raises(RuntimeError, match="NumPy runtime binding drift"):
        ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, field)
    assert calls == 0
    monkeypatch.setattr(np, "array_equal", original)
    validate_ir_wrk3_result(
        ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, field)
    )


def test_callback_core_rhs_drift_fails_before_rhs_and_clean_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_calls = 0
    evil_rhs_calls = 0
    original_rhs = ir_module._rhs

    def evil_rhs(*_args: object, **_kwargs: object) -> object:
        nonlocal evil_rhs_calls
        evil_rhs_calls += 1
        raise AssertionError("drifted RHS must never execute")

    def hostile_field(state: ParticleState) -> IRWRK3Field:
        nonlocal callback_calls
        callback_calls += 1
        response = make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.broadcast_to(np.diag((0.2, -0.1, -0.1)), (len(state.sigma), 3, 3)),
        )
        monkeypatch.setattr(ir_module, "_rhs", evil_rhs)
        return response

    with pytest.raises(RuntimeError, match="core runtime binding drift.*_rhs"):
        ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, hostile_field)
    assert callback_calls == 1
    assert evil_rhs_calls == 0

    monkeypatch.setattr(ir_module, "_rhs", original_rhs)
    clean = ir_wrk3_step_with_external_field(
        _full_affine_state(),
        0.01,
        _affine_field(np.diag((0.2, -0.1, -0.1))),
        parent_token=PARENT_TOKEN,
    )
    validate_ir_wrk3_result(clean)


def test_callback_in_place_rhs_code_drift_fails_before_rhs_and_clean_retry() -> None:
    callback_calls = 0
    rhs_function = ir_module._rhs
    original_code = rhs_function.__code__

    def evil_rhs(_gamma: object, _sigma: object, _field: object) -> object:
        raise AssertionError("drifted RHS code must never execute")

    def hostile_field(state: ParticleState) -> IRWRK3Field:
        nonlocal callback_calls
        callback_calls += 1
        response = make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
        )
        rhs_function.__code__ = evil_rhs.__code__
        return response

    try:
        with pytest.raises(RuntimeError, match="function code drift.*_rhs"):
            ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, hostile_field)
        assert callback_calls == 1
    finally:
        rhs_function.__code__ = original_code

    clean = ir_wrk3_step_with_external_field(
        _full_affine_state(),
        0.01,
        _affine_field(np.diag((0.2, -0.1, -0.1))),
        parent_token=PARENT_TOKEN,
    )
    validate_ir_wrk3_result(clean)


def test_callback_dataclass_constructor_drift_fails_before_canonical_copy() -> None:
    callback_calls = 0
    evil_constructor_calls = 0
    original_init = IRWRK3Field.__init__

    def evil_init(
        self: IRWRK3Field,
        velocity: np.ndarray,
        jacobian: np.ndarray,
        source_state_sha256: str,
        parent_token: str,
    ) -> None:
        del self, velocity, jacobian, source_state_sha256, parent_token
        nonlocal evil_constructor_calls
        evil_constructor_calls += 1
        raise AssertionError("drifted dataclass constructor must never execute")

    def hostile_field(state: ParticleState) -> IRWRK3Field:
        nonlocal callback_calls
        callback_calls += 1
        response = make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.broadcast_to(np.eye(3), (len(state.sigma), 3, 3)),
        )
        IRWRK3Field.__init__ = evil_init  # type: ignore[method-assign]
        return response

    try:
        with pytest.raises(RuntimeError, match="class member drift.*__init__"):
            ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, hostile_field)
        assert callback_calls == 1
        assert evil_constructor_calls == 0
    finally:
        IRWRK3Field.__init__ = original_init  # type: ignore[method-assign]

    clean = ir_wrk3_step_with_external_field(
        _full_affine_state(),
        0.01,
        _affine_field(np.diag((0.2, -0.1, -0.1))),
        parent_token=PARENT_TOKEN,
    )
    validate_ir_wrk3_result(clean)


def test_callback_standard_library_binding_drift_fails_before_digest_use() -> None:
    callback_calls = 0
    original_sha256 = ir_module.hashlib.sha256

    def hostile_field(state: ParticleState) -> IRWRK3Field:
        nonlocal callback_calls
        callback_calls += 1
        response = make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
        )
        ir_module.hashlib.sha256 = lambda *_args, **_kwargs: original_sha256()  # type: ignore[assignment]
        return response

    try:
        with pytest.raises(
            RuntimeError, match="standard-library runtime binding drift"
        ):
            ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, hostile_field)
        assert callback_calls == 1
    finally:
        ir_module.hashlib.sha256 = original_sha256

    clean = ir_wrk3_step_with_external_field(
        _full_affine_state(),
        0.01,
        _affine_field(np.diag((0.2, -0.1, -0.1))),
        parent_token=PARENT_TOKEN,
    )
    validate_ir_wrk3_result(clean)


def test_callback_live_registry_mutation_is_detected_and_retry_is_clean() -> None:
    issued = ir_wrk3_step_with_external_field(
        _full_affine_state(),
        0.01,
        _affine_field(np.diag((0.2, -0.1, -0.1))),
        parent_token=PARENT_TOKEN,
    )
    forged = replace(issued)
    registry = next(
        cell.cell_contents
        for cell in ir_module._attest_live_result.__closure__ or ()
        if type(cell.cell_contents) is dict
    )
    register_cells = dict(
        zip(
            ir_module._register_live_result.__code__.co_freevars,
            ir_module._register_live_result.__closure__ or (),
            strict=True,
        )
    )
    guard_cells = dict(
        zip(
            ir_module._assert_live_result_registry_unchanged.__code__.co_freevars,
            ir_module._assert_live_result_registry_unchanged.__closure__ or (),
            strict=True,
        )
    )
    compute_seal = register_cells["compute_seal"].cell_contents
    seal_cell = register_cells["seal"]
    snapshot_cell = guard_cells["snapshot"]
    original_snapshot = snapshot_cell.cell_contents
    pre_callback_snapshot = original_snapshot()

    def hostile_field(state: ParticleState) -> IRWRK3Field:
        response = make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
        )
        registry[id(forged)] = (forged, forged.result_sha256)
        seal_cell.cell_contents = compute_seal()

        def forged_snapshot() -> tuple[object, ...]:
            return pre_callback_snapshot

        snapshot_cell.cell_contents = forged_snapshot
        return response

    try:
        with pytest.raises(RuntimeError, match="closure function drift.*snapshot"):
            ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, hostile_field)
    finally:
        snapshot_cell.cell_contents = original_snapshot
        registry.pop(id(forged), None)
        seal_cell.cell_contents = compute_seal()

    validate_ir_wrk3_result(issued)
    with pytest.raises(ValueError, match="exact live issued"):
        validate_ir_wrk3_result(forged)
    clean = ir_wrk3_step_with_external_field(
        _full_affine_state(),
        0.01,
        _affine_field(np.diag((0.2, -0.1, -0.1))),
        parent_token=PARENT_TOKEN,
    )
    validate_ir_wrk3_result(clean)


def test_public_factories_preflight_caps_before_materialization() -> None:
    materializations = 0

    class LazyArrayTrap:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
            nonlocal materializations
            materializations += 1
            raise AssertionError("array materialization must not occur")

    state = _full_affine_state()
    reference = freeze_invariant_reference(state)
    oversize = 1_000_001
    with pytest.raises(ValueError, match="cap exceeded before array materialization"):
        make_ir_wrk3_field(
            state,
            LazyArrayTrap((oversize, 3)),
            np.zeros((len(state.sigma), 3, 3)),
        )
    with pytest.raises(ValueError, match="cap exceeded before array materialization"):
        make_ir_wrk3_tracer_field(
            state,
            LazyArrayTrap((oversize, 3)),
            LazyArrayTrap((oversize, 3)),
        )
    with pytest.raises(ValueError, match="cap exceeded before array materialization"):
        reconstruct_sigma(LazyArrayTrap((oversize, 3)), reference)
    assert materializations == 0


def test_validator_rejects_tracer_state_discontinuity_between_stages() -> None:
    state = _full_affine_state()
    tracers = np.asarray(((0.2, -0.1, 0.05), (0.4, 0.3, -0.2)), dtype=np.float64)

    def tracer_field(
        pre: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        return make_ir_wrk3_tracer_field(
            pre,
            tracer_pre,
            np.zeros_like(tracer_pre),
            parent_token=parent_token,
        )

    result = ir_wrk3_step_with_external_field(
        state,
        0.01,
        _affine_field(np.zeros((3, 3))),
        tracer_positions=tracers,
        tracer_field_evaluator=tracer_field,
        parent_token=PARENT_TOKEN,
    )
    second = result.stages[1]
    changed = np.array(second.tracer_pre, copy=True)
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    changed = np.frombuffer(changed.tobytes(order="C"), dtype=np.float64).reshape(
        changed.shape
    )
    forged_tracer_field = make_ir_wrk3_tracer_field(
        second.pre,
        changed,
        second.tracer_field.velocity,
        parent_token=PARENT_TOKEN,
    )
    stages = list(result.stages)
    stages[1] = replace(
        second,
        tracer_pre=changed,
        tracer_field=forged_tracer_field,
    )
    with pytest.raises(ValueError, match="tracer state continuity"):
        validate_ir_wrk3_result(replace(result, stages=tuple(stages)))


def test_caps_and_substep_underflow_fail_before_materialization_or_field() -> None:
    materializations = 0
    field_calls = 0

    class LazyArrayTrap:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
            nonlocal materializations
            materializations += 1
            raise AssertionError("array materialization must not occur")

    def field(state: ParticleState) -> IRWRK3Field:
        nonlocal field_calls
        field_calls += 1
        return make_ir_wrk3_field(
            state,
            np.zeros_like(state.positions),
            np.zeros((len(state.sigma), 3, 3)),
        )

    oversize = 1_000_001
    forged = ParticleState(
        LazyArrayTrap((oversize, 3)),
        LazyArrayTrap((oversize, 3)),
        LazyArrayTrap((oversize,)),
    )
    with pytest.raises(ValueError, match="cap exceeded before array materialization"):
        ir_wrk3_step_with_external_field(forged, 0.01, field)
    assert materializations == 0
    assert field_calls == 0

    with pytest.raises(ValueError, match="cap exceeded before array materialization"):
        ir_wrk3_step_with_external_field(
            _full_affine_state(),
            0.01,
            field,
            tracer_positions=LazyArrayTrap((oversize, 3)),
            tracer_field_evaluator=lambda *_args: None,  # type: ignore[arg-type]
        )
    assert materializations == 0
    assert field_calls == 0

    with pytest.raises(ValueError, match="retained stage-evidence row cap"):
        ir_wrk3_step_with_external_field(
            make_particle_state(
                np.zeros((100, 3)),
                np.tile((1.0, 0.0, 0.0), (100, 1)),
                np.full(100, 0.085),
            ),
            0.01,
            field,
            transport_substeps=4096,
        )
    assert field_calls == 0

    with pytest.raises(ValueError, match="underflowed"):
        ir_wrk3_step_with_external_field(
            _full_affine_state(),
            float(np.nextafter(0.0, 1.0)),
            field,
            transport_substeps=2,
        )
    assert field_calls == 0

    clean = ir_wrk3_step_with_external_field(_full_affine_state(), 0.01, field)
    assert field_calls == 3
    validate_ir_wrk3_result(clean)


@pytest.mark.parametrize("attack", ("source", "tracer", "parent"))
def test_tracer_field_attestation_rejects_wrong_stage_inputs(
    attack: str,
) -> None:
    initial, tracers, external_jacobian, external_translation = _b2_fixture()

    def physical(state: ParticleState) -> IRWRK3Field:
        return make_ir_wrk3_field(
            state,
            state.positions @ external_jacobian.T + external_translation,
            np.repeat(external_jacobian[None, :, :], len(state.sigma), axis=0),
            parent_token=PARENT_TOKEN,
        )

    def hostile(
        state: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        source = state
        locations = tracer_pre
        token = parent_token
        if attack == "source":
            source = make_particle_state(
                state.positions + 1.0e-6, state.gamma, state.sigma
            )
        elif attack == "tracer":
            locations = tracer_pre.copy()
            locations[0, 0] = np.nextafter(locations[0, 0], np.inf)
        else:
            token = "wrong-parent"
        return make_ir_wrk3_tracer_field(
            source,
            locations,
            np.zeros_like(locations),
            parent_token=token,
        )

    parent_bytes = _state_bytes(initial)
    with pytest.raises(ValueError, match="tracer field"):
        ir_wrk3_step_with_external_field(
            initial,
            0.01,
            physical,
            tracer_positions=tracers,
            tracer_field_evaluator=hostile,
            parent_token=PARENT_TOKEN,
        )
    assert _state_bytes(initial) == parent_bytes
