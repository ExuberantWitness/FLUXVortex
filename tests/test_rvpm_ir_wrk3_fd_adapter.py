from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Callable

import numpy as np
import pytest

import fluxvortex.rvpm_ir_wrk3_fd_adapter as fd_module
from fluxvortex.rvpm_ir_wrk3 import (
    IRWRK3Field,
    IRWRK3TracerField,
    ir_wrk3_step_with_external_field,
    make_ir_wrk3_field,
    make_ir_wrk3_tracer_field,
    validate_ir_wrk3_result,
)
from fluxvortex.rvpm_ir_wrk3_fd_adapter import (
    CallableProvenance,
    FDCallLedger,
    FDEvaluationRecord,
    FDPhysicalResult,
    FDTracerResult,
    FDVelocityCall,
    FrozenParentCenteredFDAdapter,
    FrozenParentVelocity,
    make_frozen_parent_velocity,
    validate_fd_call_ledger,
)
from fluxvortex.rvpm_transport import ParticleState, make_particle_state


PARENT_TOKEN = "v5h11-b2b-mock-frozen-parent-v1"
PARENT_STATE_SHA256 = sha256(b"v5h11-b2b-parent-state-v1").hexdigest()
EPSILON = 2.0**-10
SUBSTEPS = 2
DELTA_TIME = 0.0125

PARENT_JACOBIAN = np.asarray(
    (
        (0.7, -0.3, 0.2),
        (0.1, -0.5, 0.4),
        (-0.2, 0.6, -0.2),
    ),
    dtype=np.float64,
)
PARENT_TRANSLATION = np.asarray((0.35, -0.12, 0.08), dtype=np.float64)
DIRECT_JACOBIAN = np.asarray(
    (
        (-0.15, 0.05, 0.02),
        (0.03, 0.11, -0.04),
        (0.07, -0.02, 0.04),
    ),
    dtype=np.float64,
)
DIRECT_TRANSLATION = np.asarray((-0.03, 0.02, 0.01), dtype=np.float64)


def _initial_state() -> ParticleState:
    mutable = make_particle_state(
        np.asarray(
            ((0.08, -0.03, 0.11), (-0.14, 0.09, 0.04), (0.03, 0.17, -0.06)),
            dtype=np.float64,
        ),
        np.asarray(
            ((0.31, -0.16, 0.22), (-0.12, 0.28, 0.09), (0.18, 0.07, -0.25)),
            dtype=np.float64,
        ),
        np.asarray((0.075, 0.09, 0.082), dtype=np.float64),
    )
    return ParticleState(
        _frozen_copy(mutable.positions),
        _frozen_copy(mutable.gamma),
        _frozen_copy(mutable.sigma),
    )


def _tracers() -> np.ndarray:
    return np.asarray(
        ((-0.21, 0.04, 0.13), (0.05, -0.18, 0.07), (0.16, 0.03, -0.09)),
        dtype=np.float64,
    )


def _array_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes(order="C")


def _frozen_copy(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
        contiguous.shape
    )


def _state_bytes(state: ParticleState) -> tuple[bytes, bytes, bytes]:
    return (
        _array_bytes(state.positions),
        _array_bytes(state.gamma),
        _array_bytes(state.sigma),
    )


def _affine_parent_velocity(targets: np.ndarray) -> np.ndarray:
    return targets @ PARENT_JACOBIAN.T + PARENT_TRANSLATION


def _assert_frozen_float64(array: np.ndarray, shape: tuple[int, ...]) -> None:
    assert type(array) is np.ndarray
    assert array.dtype == np.dtype(np.float64)
    assert array.shape == shape
    assert array.flags.c_contiguous
    assert not array.flags.writeable
    ancestor: object = array
    while type(ancestor) is np.ndarray:
        assert not ancestor.flags.writeable
        assert ancestor.flags.c_contiguous
        ancestor = ancestor.base
    assert type(ancestor) is bytes


def _assert_empty_ledger(ledger: object) -> None:
    assert getattr(ledger, "physical_evaluation_count") == 0
    assert getattr(ledger, "tracer_evaluation_count") == 0
    assert getattr(ledger, "center_call_count") == 0
    assert getattr(ledger, "offset_call_count") == 0
    assert getattr(ledger, "evaluator_call_count") == 0
    assert getattr(ledger, "calls") == ()
    assert getattr(ledger, "evaluations") == ()


class _MockFrozenParent:
    def __init__(self, events: list[tuple[str, bytes]] | None = None) -> None:
        self.events = events if events is not None else []
        self.evaluator_calls = 0
        self.getter_calls = 0
        self.parent_state_hash = PARENT_STATE_SHA256
        self.response_attack: str | None = None
        self.nonfinite_once = False

    def parent_state_sha256(self) -> str:
        self.getter_calls += 1
        return self.parent_state_hash

    def evaluate(self, targets: np.ndarray) -> FrozenParentVelocity:
        _assert_frozen_float64(targets, targets.shape)
        self.evaluator_calls += 1
        self.events.append(("parent", _array_bytes(targets)))
        velocity = _affine_parent_velocity(targets)
        if self.nonfinite_once:
            self.nonfinite_once = False
            velocity = velocity.copy()
            velocity[0, 0] = np.inf
        response = make_frozen_parent_velocity(
            targets,
            velocity,
            parent_token=PARENT_TOKEN,
            parent_state_sha256=PARENT_STATE_SHA256,
        )
        attack = self.response_attack
        self.response_attack = None
        if attack == "target":
            response = replace(response, target_sha256="0" * 64)
        elif attack == "velocity":
            response = replace(response, velocity_sha256="1" * 64)
        elif attack == "token":
            response = replace(response, parent_token="wrong-parent-token")
        elif attack == "parent":
            response = replace(response, parent_state_sha256="2" * 64)
        return response


def _adapter(
    parent: _MockFrozenParent,
    *,
    epsilon: float = EPSILON,
    parent_token: str = PARENT_TOKEN,
    max_target_count: int = 1_000_000,
) -> FrozenParentCenteredFDAdapter:
    return FrozenParentCenteredFDAdapter(
        parent.evaluate,
        parent.parent_state_sha256,
        epsilon=epsilon,
        parent_token=parent_token,
        max_target_count=max_target_count,
    )


def _combined_callbacks(
    adapter: FrozenParentCenteredFDAdapter,
    events: list[tuple[str, bytes]],
    direct_counts: dict[str, int],
) -> tuple[
    Callable[[ParticleState], IRWRK3Field],
    Callable[[ParticleState, np.ndarray, str], IRWRK3TracerField],
]:
    last_physical_source: list[tuple[bytes, bytes, bytes] | None] = [None]

    def physical(state: ParticleState) -> IRWRK3Field:
        direct_counts["physical"] += 1
        events.append(("physical_direct", _array_bytes(state.positions)))
        source = _state_bytes(state)
        last_physical_source[0] = source
        direct_velocity = state.positions @ DIRECT_JACOBIAN.T + DIRECT_TRANSLATION
        direct_jacobian = np.repeat(
            DIRECT_JACOBIAN[None, :, :], len(state.sigma), axis=0
        )
        external = adapter.evaluate_physical(state)
        return make_ir_wrk3_field(
            state,
            direct_velocity + external.field.velocity,
            direct_jacobian + external.field.jacobian,
            parent_token=PARENT_TOKEN,
        )

    def tracer(
        state: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        assert parent_token == PARENT_TOKEN
        assert last_physical_source[0] == _state_bytes(state)
        direct_counts["tracer"] += 1
        events.append(("tracer_direct", _array_bytes(tracer_pre)))
        direct_velocity = tracer_pre @ DIRECT_JACOBIAN.T + DIRECT_TRANSLATION
        external = adapter.evaluate_tracer(state, tracer_pre, parent_token)
        return make_ir_wrk3_tracer_field(
            state,
            tracer_pre,
            direct_velocity + external.field.velocity,
            parent_token=parent_token,
        )

    return physical, tracer


def test_centered_fd_affine_jacobian_is_exact_and_tracer_is_center_only() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    state = _initial_state()
    tracers = _tracers()

    physical = adapter.evaluate_physical(state)
    np.testing.assert_array_equal(
        physical.field.velocity, _affine_parent_velocity(state.positions)
    )
    np.testing.assert_allclose(
        physical.field.jacobian,
        np.repeat(PARENT_JACOBIAN[None, :, :], len(state.sigma), axis=0),
        rtol=0.0,
        atol=128.0 * np.finfo(np.float64).eps,
    )
    tracer = adapter.evaluate_tracer(state, tracers, PARENT_TOKEN)
    np.testing.assert_array_equal(
        tracer.field.velocity, _affine_parent_velocity(tracers)
    )

    ledger = adapter.snapshot()
    assert type(physical) is FDPhysicalResult
    assert type(tracer) is FDTracerResult
    assert type(ledger) is FDCallLedger
    assert all(type(call) is FDVelocityCall for call in ledger.calls)
    assert all(
        type(evaluation) is FDEvaluationRecord for evaluation in ledger.evaluations
    )
    assert type(ledger.velocity_evaluator_provenance) is CallableProvenance
    assert type(ledger.parent_hash_getter_provenance) is CallableProvenance
    assert ledger.physical_evaluation_count == 1
    assert ledger.tracer_evaluation_count == 1
    assert ledger.center_call_count == 2
    assert ledger.offset_call_count == 6
    assert ledger.evaluator_call_count == 8
    assert len(ledger.calls) == 8
    assert len(ledger.evaluations) == 2
    assert adapter.ledger.trace_sha256 == ledger.trace_sha256
    assert [call.sample_kind for call in ledger.calls] == [
        "center",
        "offset",
        "offset",
        "offset",
        "offset",
        "offset",
        "offset",
        "center",
    ]
    assert [(call.axis, call.sign) for call in ledger.calls] == [
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 1),
        (2, -1),
        (2, 1),
        (-1, 0),
    ]
    assert physical.evaluation.source_state_sha256 == physical.field.source_state_sha256
    assert physical.evaluation.target_sha256 == ledger.calls[0].target_sha256
    assert physical.evaluation.jacobian_sha256 is not None
    assert tracer.evaluation.source_state_sha256 == tracer.field.source_state_sha256
    assert tracer.evaluation.target_sha256 == tracer.field.tracer_state_sha256
    for evaluation in ledger.evaluations:
        assert evaluation.parent_token == PARENT_TOKEN
        assert evaluation.parent_state_sha256 == PARENT_STATE_SHA256
        assert evaluation.epsilon == EPSILON

    expected_targets = [state.positions]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            target = state.positions.copy()
            target[:, axis] += sign * EPSILON
            expected_targets.append(target)
    expected_targets.append(tracers)
    assert [payload for _, payload in parent.events] == [
        _array_bytes(targets) for targets in expected_targets
    ]


def test_fixed_substeps_core_composition_has_exact_b2b_call_ledger() -> None:
    events: list[tuple[str, bytes]] = []
    parent = _MockFrozenParent(events)
    adapter = _adapter(parent)
    direct_counts = {"physical": 0, "tracer": 0}
    physical, tracer = _combined_callbacks(adapter, events, direct_counts)
    initial = _initial_state()
    tracers = _tracers()
    before = _state_bytes(initial)

    result = ir_wrk3_step_with_external_field(
        initial,
        DELTA_TIME,
        physical,
        transport_substeps=SUBSTEPS,
        tracer_positions=tracers,
        tracer_field_evaluator=tracer,
        parent_token=PARENT_TOKEN,
    )
    assert validate_ir_wrk3_result(result) is result
    assert _state_bytes(initial) == before

    stage_count = 3 * SUBSTEPS
    ledger = adapter.snapshot()
    assert direct_counts == {"physical": stage_count, "tracer": stage_count}
    assert sum(direct_counts.values()) == 6 * SUBSTEPS
    assert ledger.physical_evaluation_count == stage_count
    assert ledger.tracer_evaluation_count == stage_count
    assert ledger.center_call_count == 6 * SUBSTEPS
    assert ledger.offset_call_count == 18 * SUBSTEPS
    assert ledger.evaluator_call_count == 24 * SUBSTEPS
    assert result.counters.physical_field_call_count == stage_count
    assert result.counters.tracer_field_call_count == stage_count

    event_kinds = [kind for kind, _ in events]
    expected_stage = ["physical_direct", *("parent",) * 7, "tracer_direct", "parent"]
    assert event_kinds == expected_stage * stage_count


def test_adapter_outputs_and_ledger_are_exact_readonly_deterministic_trees() -> None:
    first_parent = _MockFrozenParent()
    second_parent = _MockFrozenParent()
    first = _adapter(first_parent)
    second = _adapter(second_parent)
    state = _initial_state()
    tracers = _tracers()

    first_physical = first.evaluate_physical(state)
    first_tracer = first.evaluate_tracer(state, tracers, PARENT_TOKEN)
    second_physical = second.evaluate_physical(state)
    second_tracer = second.evaluate_tracer(state, tracers, PARENT_TOKEN)

    _assert_frozen_float64(first_physical.field.velocity, state.positions.shape)
    _assert_frozen_float64(first_physical.field.jacobian, (len(state.sigma), 3, 3))
    _assert_frozen_float64(first_tracer.field.velocity, tracers.shape)
    first_ledger = first.snapshot()
    second_ledger = second.snapshot()
    assert type(first_ledger) is type(second_ledger)
    assert first_ledger.trace_sha256 == second_ledger.trace_sha256
    assert first_ledger.physical_evaluation_count == 1
    assert first_ledger.tracer_evaluation_count == 1
    assert first_ledger.center_call_count == second_ledger.center_call_count == 2
    assert first_ledger.offset_call_count == second_ledger.offset_call_count == 6
    assert [call.call_sha256 for call in first_ledger.calls] == [
        call.call_sha256 for call in second_ledger.calls
    ]
    assert [record.evaluation_sha256 for record in first_ledger.evaluations] == [
        record.evaluation_sha256 for record in second_ledger.evaluations
    ]
    np.testing.assert_array_equal(
        first_physical.field.velocity, second_physical.field.velocity
    )
    np.testing.assert_array_equal(
        first_physical.field.jacobian, second_physical.field.jacobian
    )
    np.testing.assert_array_equal(
        first_tracer.field.velocity, second_tracer.field.velocity
    )


@pytest.mark.parametrize("attack", ("target", "velocity", "token", "parent"))
def test_attestation_tamper_rolls_back_and_allows_clean_retry(attack: str) -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    state = _initial_state()
    empty = adapter.snapshot()
    parent.response_attack = attack

    with pytest.raises(
        ValueError, match="target|velocity|token|parent|attestation|hash"
    ):
        adapter.evaluate_physical(state)
    _assert_empty_ledger(adapter.snapshot())
    assert adapter.snapshot().trace_sha256 == empty.trace_sha256

    clean = adapter.evaluate_physical(state)
    np.testing.assert_array_equal(
        clean.field.velocity, _affine_parent_velocity(state.positions)
    )
    ledger = adapter.snapshot()
    assert ledger.physical_evaluation_count == 1
    assert ledger.center_call_count == 1
    assert ledger.offset_call_count == 6


def test_nonfinite_parent_response_rolls_back_and_allows_clean_retry() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    empty = adapter.snapshot()
    parent.nonfinite_once = True

    with pytest.raises((ValueError, FloatingPointError), match="finite"):
        adapter.evaluate_physical(_initial_state())
    _assert_empty_ledger(adapter.snapshot())
    assert adapter.snapshot().trace_sha256 == empty.trace_sha256
    clean = adapter.evaluate_physical(_initial_state())
    assert np.all(np.isfinite(clean.field.velocity))
    assert adapter.snapshot().evaluator_call_count == 7


def test_mutable_particle_state_is_canonicalized_without_input_mutation() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    frozen = _initial_state()
    mutable = make_particle_state(
        frozen.positions.copy(), frozen.gamma.copy(), frozen.sigma.copy()
    )
    before = _state_bytes(mutable)
    assert mutable.positions.flags.writeable

    result = adapter.evaluate_physical(mutable)
    assert _state_bytes(mutable) == before
    assert mutable.positions.flags.writeable
    _assert_frozen_float64(result.field.velocity, mutable.positions.shape)
    _assert_frozen_float64(result.field.jacobian, (len(mutable.sigma), 3, 3))


def test_parent_state_hash_drift_is_zero_evaluator_call_and_clean_retry() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    empty = adapter.snapshot()
    parent.parent_state_hash = "3" * 64

    with pytest.raises(RuntimeError, match="parent state hash drift"):
        adapter.evaluate_physical(_initial_state())
    assert parent.evaluator_calls == 0
    parent.parent_state_hash = PARENT_STATE_SHA256
    _assert_empty_ledger(adapter.snapshot())
    assert adapter.snapshot().trace_sha256 == empty.trace_sha256
    adapter.evaluate_physical(_initial_state())
    assert adapter.snapshot().evaluator_call_count == 7


def test_velocity_callable_code_drift_is_zero_call_and_clean_retry() -> None:
    def evaluate(targets: np.ndarray) -> FrozenParentVelocity:
        return make_frozen_parent_velocity(
            targets,
            _affine_parent_velocity(targets),
            parent_token=PARENT_TOKEN,
            parent_state_sha256=PARENT_STATE_SHA256,
        )

    def replacement(targets: np.ndarray) -> FrozenParentVelocity:
        raise AssertionError(targets)

    def parent_hash() -> str:
        return PARENT_STATE_SHA256

    adapter = FrozenParentCenteredFDAdapter(
        evaluate,
        parent_hash,
        epsilon=EPSILON,
        parent_token=PARENT_TOKEN,
    )
    original_code = evaluate.__code__
    evaluate.__code__ = replacement.__code__
    try:
        with pytest.raises(RuntimeError, match="callable code|callable.*drift"):
            adapter.evaluate_physical(_initial_state())
    finally:
        evaluate.__code__ = original_code
    _assert_empty_ledger(adapter.snapshot())
    clean = adapter.evaluate_physical(_initial_state())
    assert clean.field.velocity.shape == _initial_state().positions.shape


def test_callable_provenance_is_stable_after_adaptive_specialization() -> None:
    def evaluate(targets: np.ndarray) -> FrozenParentVelocity:
        return make_frozen_parent_velocity(
            targets,
            _affine_parent_velocity(targets),
            parent_token=PARENT_TOKEN,
            parent_state_sha256=PARENT_STATE_SHA256,
        )

    def parent_hash() -> str:
        return PARENT_STATE_SHA256

    first = FrozenParentCenteredFDAdapter(
        evaluate,
        parent_hash,
        epsilon=EPSILON,
        parent_token=PARENT_TOKEN,
    )
    first.evaluate_physical(_initial_state())
    for _ in range(64):
        evaluate(_initial_state().positions)

    second = FrozenParentCenteredFDAdapter(
        evaluate,
        parent_hash,
        epsilon=EPSILON,
        parent_token=PARENT_TOKEN,
    )
    second.evaluate_physical(_initial_state())
    first_ledger = first.snapshot()
    second_ledger = second.snapshot()
    assert (
        first_ledger.velocity_evaluator_provenance
        == second_ledger.velocity_evaluator_provenance
    )
    assert [call.call_sha256 for call in first_ledger.calls] == [
        call.call_sha256 for call in second_ledger.calls
    ]


@pytest.mark.parametrize(
    "epsilon",
    (0.0, -EPSILON, np.inf, np.nan, True),
)
def test_invalid_or_unregistered_epsilon_is_zero_evaluator_call(
    epsilon: object,
) -> None:
    parent = _MockFrozenParent()
    with pytest.raises((TypeError, ValueError), match="epsilon"):
        _adapter(parent, epsilon=epsilon)  # type: ignore[arg-type]
    assert parent.evaluator_calls == 0


def test_wrong_tracer_parent_token_is_rejected_before_evaluator_call() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    empty = adapter.snapshot()
    with pytest.raises(ValueError, match="parent token"):
        adapter.evaluate_tracer(_initial_state(), _tracers(), "wrong-parent")
    assert parent.evaluator_calls == 0
    _assert_empty_ledger(adapter.snapshot())
    assert adapter.snapshot().trace_sha256 == empty.trace_sha256


def test_target_cap_is_checked_before_evaluator_call_and_clean_retry() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent, max_target_count=2)
    empty = adapter.snapshot()
    with pytest.raises(ValueError, match="target.*cap|cap.*target"):
        adapter.evaluate_physical(_initial_state())
    assert parent.evaluator_calls == 0
    _assert_empty_ledger(adapter.snapshot())
    assert adapter.snapshot().trace_sha256 == empty.trace_sha256

    original = _initial_state()
    clean_state = ParticleState(
        _frozen_copy(original.positions[:2]),
        _frozen_copy(original.gamma[:2]),
        _frozen_copy(original.sigma[:2]),
    )
    clean_targets = _tracers()[:2]
    clean = adapter.evaluate_tracer(clean_state, clean_targets, PARENT_TOKEN)
    assert clean.field.velocity.shape == (2, 3)
    assert adapter.snapshot().evaluator_call_count == 1


def test_numpy_callable_drift_fails_before_evaluator_and_restores_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    state = _initial_state()
    original = fd_module.np.asarray
    monkeypatch.setattr(
        fd_module.np,
        "asarray",
        lambda value, *args, **kwargs: original(value, *args, **kwargs),
    )
    with pytest.raises(RuntimeError, match="NumPy|callable|binding|drift"):
        adapter.evaluate_physical(state)
    assert parent.evaluator_calls == 0
    monkeypatch.setattr(fd_module.np, "asarray", original)
    _assert_empty_ledger(adapter.snapshot())
    clean = adapter.evaluate_physical(state)
    assert clean.field.velocity.shape == state.positions.shape
    assert adapter.snapshot().evaluator_call_count == 7


def test_post_callback_verifier_rebind_is_rejected_before_commit() -> None:
    calls = [0]
    attack = [True]

    def parent_hash() -> str:
        return PARENT_STATE_SHA256

    def evaluate(targets: np.ndarray) -> FrozenParentVelocity:
        calls[0] += 1
        if attack[0]:
            fd_module._assert_callable_guard = lambda guard: None
        return make_frozen_parent_velocity(
            targets,
            _affine_parent_velocity(targets),
            parent_token=PARENT_TOKEN,
            parent_state_sha256=PARENT_STATE_SHA256,
        )

    original_verifier = fd_module._assert_callable_guard
    adapter = FrozenParentCenteredFDAdapter(
        evaluate,
        parent_hash,
        epsilon=EPSILON,
        parent_token=PARENT_TOKEN,
    )
    try:
        with pytest.raises(RuntimeError, match="verifier|binding|drift"):
            adapter.evaluate_physical(_initial_state())
    finally:
        fd_module._assert_callable_guard = original_verifier
        attack[0] = False
    assert calls == [1]
    _assert_empty_ledger(adapter.snapshot())

    clean = adapter.evaluate_physical(_initial_state())
    assert clean.field.velocity.shape == _initial_state().positions.shape
    assert adapter.snapshot().evaluator_call_count == 7


def test_global_ndarray_content_drift_is_zero_call_and_restores_cleanly() -> None:
    calls = [0]

    def parent_hash() -> str:
        return PARENT_STATE_SHA256

    def evaluate(targets: np.ndarray) -> FrozenParentVelocity:
        calls[0] += 1
        velocity = targets @ PARENT_JACOBIAN.T + PARENT_TRANSLATION
        return make_frozen_parent_velocity(
            targets,
            velocity,
            parent_token=PARENT_TOKEN,
            parent_state_sha256=PARENT_STATE_SHA256,
        )

    adapter = FrozenParentCenteredFDAdapter(
        evaluate,
        parent_hash,
        epsilon=EPSILON,
        parent_token=PARENT_TOKEN,
    )
    original_matrix = PARENT_JACOBIAN.copy()
    PARENT_JACOBIAN[:] = 2.0 * np.eye(3)
    try:
        with pytest.raises(RuntimeError, match="dependency content drift"):
            adapter.evaluate_physical(_initial_state())
    finally:
        PARENT_JACOBIAN[:] = original_matrix
    assert calls == [0]
    _assert_empty_ledger(adapter.snapshot())

    clean = adapter.evaluate_physical(_initial_state())
    np.testing.assert_allclose(
        clean.field.jacobian,
        np.repeat(original_matrix[None, :, :], 3, axis=0),
        rtol=0.0,
        atol=128.0 * np.finfo(np.float64).eps,
    )


def test_kwdefault_content_drift_is_zero_call_and_restores_cleanly() -> None:
    calls = [0]

    def parent_hash() -> str:
        return PARENT_STATE_SHA256

    def evaluate(targets: np.ndarray, *, scale: float = 1.0) -> FrozenParentVelocity:
        calls[0] += 1
        return make_frozen_parent_velocity(
            targets,
            scale * _affine_parent_velocity(targets),
            parent_token=PARENT_TOKEN,
            parent_state_sha256=PARENT_STATE_SHA256,
        )

    adapter = FrozenParentCenteredFDAdapter(
        evaluate,
        parent_hash,
        epsilon=EPSILON,
        parent_token=PARENT_TOKEN,
    )
    assert evaluate.__kwdefaults__ is not None
    evaluate.__kwdefaults__["scale"] = 2.0
    try:
        with pytest.raises(RuntimeError, match="dependency content drift"):
            adapter.evaluate_physical(_initial_state())
    finally:
        evaluate.__kwdefaults__["scale"] = 1.0
    assert calls == [0]
    _assert_empty_ledger(adapter.snapshot())
    adapter.evaluate_physical(_initial_state())
    assert calls == [7]


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    (
        ("_epsilon", EPSILON * 2.0),
        ("_parent_token", "mutated-parent-token"),
        ("_parent_state_sha256", "4" * 64),
        ("_max_target_count", 1),
    ),
)
def test_adapter_config_drift_is_zero_call_and_restores_cleanly(
    attribute: str,
    replacement: object,
) -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    original = getattr(adapter, attribute)
    setattr(adapter, attribute, replacement)
    try:
        with pytest.raises(RuntimeError, match="sealed scalar config drift"):
            adapter.evaluate_physical(_initial_state())
    finally:
        setattr(adapter, attribute, original)
    assert parent.evaluator_calls == 0
    _assert_empty_ledger(adapter.snapshot())
    adapter.evaluate_physical(_initial_state())
    assert parent.evaluator_calls == 7


def test_adapter_evaluator_and_guard_identity_drift_are_sealed() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)

    original_evaluator = adapter._velocity_evaluator
    adapter._velocity_evaluator = lambda targets: parent.evaluate(targets)
    with pytest.raises(RuntimeError, match="sealed identity config drift"):
        adapter.evaluate_physical(_initial_state())
    adapter._velocity_evaluator = original_evaluator

    original_guard = adapter._velocity_guard
    adapter._velocity_guard = replace(original_guard, root=lambda targets: targets)
    with pytest.raises(RuntimeError, match="sealed identity config drift"):
        adapter.evaluate_physical(_initial_state())
    adapter._velocity_guard = original_guard

    assert parent.evaluator_calls == 0
    _assert_empty_ledger(adapter.snapshot())
    adapter.evaluate_physical(_initial_state())
    assert parent.evaluator_calls == 7


def test_tracer_cap_precedes_lazy_materialization_and_allows_retry() -> None:
    class LazyTargets:
        shape = (1_000_001, 3)

        def __init__(self) -> None:
            self.materialization_count = 0

        def __array__(self, dtype: object = None) -> np.ndarray:
            self.materialization_count += 1
            raise AssertionError(dtype)

    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    lazy = LazyTargets()
    with pytest.raises(ValueError, match="cap exceeded"):
        adapter.evaluate_tracer(_initial_state(), lazy, PARENT_TOKEN)
    assert lazy.materialization_count == 0
    assert parent.evaluator_calls == 0
    _assert_empty_ledger(adapter.snapshot())

    clean = adapter.evaluate_tracer(_initial_state(), _tracers(), PARENT_TOKEN)
    assert clean.field.velocity.shape == _tracers().shape
    assert parent.evaluator_calls == 1


def test_issued_ledger_validator_recomputes_tree_and_rejects_copies() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    adapter.evaluate_physical(_initial_state())
    adapter.evaluate_tracer(_initial_state(), _tracers(), PARENT_TOKEN)
    ledger = adapter.snapshot()
    assert validate_fd_call_ledger(ledger) is ledger

    with pytest.raises(ValueError, match="counter recomputation"):
        validate_fd_call_ledger(replace(ledger, physical_evaluation_count=999))
    with pytest.raises(ValueError, match="attestation|digest"):
        validate_fd_call_ledger(
            replace(
                ledger,
                calls=(
                    replace(ledger.calls[0], target_sha256="0" * 64),
                    *ledger.calls[1:],
                ),
            )
        )
    with pytest.raises(ValueError, match="evaluation|trace|digest"):
        validate_fd_call_ledger(
            replace(
                ledger,
                evaluations=(
                    replace(ledger.evaluations[0], target_count=999),
                    *ledger.evaluations[1:],
                ),
            )
        )
    with pytest.raises(ValueError, match="exact tuples"):
        validate_fd_call_ledger(replace(ledger, calls=list(ledger.calls)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact live issued"):
        validate_fd_call_ledger(replace(ledger))


def test_live_ledger_registry_has_no_module_dict_and_detects_direct_tamper() -> None:
    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    ledger = adapter.snapshot()
    assert not hasattr(fd_module, "_LIVE_FD_LEDGER_REGISTRY")
    registry = next(
        cell.cell_contents
        for cell in fd_module._attest_fd_call_ledger.__closure__ or ()
        if type(cell.cell_contents) is dict
    )
    copied = replace(ledger)
    registry[id(copied)] = (copied, copied.ledger_sha256)
    try:
        with pytest.raises(RuntimeError, match="registry integrity drift"):
            validate_fd_call_ledger(copied)
    finally:
        del registry[id(copied)]
    assert validate_fd_call_ledger(ledger) is ledger


def test_callback_cannot_publish_a_ledger_snapshot_during_evaluation() -> None:
    attack = [True]
    adapter_holder: list[FrozenParentCenteredFDAdapter] = []

    def parent_hash() -> str:
        return PARENT_STATE_SHA256

    def evaluate(targets: np.ndarray) -> FrozenParentVelocity:
        if attack[0]:
            adapter_holder[0].snapshot()
        return make_frozen_parent_velocity(
            targets,
            _affine_parent_velocity(targets),
            parent_token=PARENT_TOKEN,
            parent_state_sha256=PARENT_STATE_SHA256,
        )

    adapter = FrozenParentCenteredFDAdapter(
        evaluate,
        parent_hash,
        epsilon=EPSILON,
        parent_token=PARENT_TOKEN,
    )
    adapter_holder.append(adapter)
    with pytest.raises(RuntimeError, match="registry changed during callback"):
        adapter.evaluate_physical(_initial_state())
    attack[0] = False
    _assert_empty_ledger(adapter.snapshot())
    adapter.evaluate_physical(_initial_state())
    assert adapter.snapshot().evaluator_call_count == 7


@pytest.mark.parametrize(
    ("branch", "expected_message"),
    (
        ("evaluations", "evaluation evidence cap exceeded"),
        ("calls", "velocity-call evidence cap exceeded"),
    ),
)
def test_cumulative_evidence_caps_stop_before_evaluator_without_ledger_change(
    branch: str,
    expected_message: str,
) -> None:
    """Use controlled internal lengths; do not spend 12k real evaluations."""

    parent = _MockFrozenParent()
    adapter = _adapter(parent)
    sentinel = object()
    if branch == "evaluations":
        adapter._evaluations = [sentinel] * fd_module.MAX_FD_EVALUATION_COUNT  # type: ignore[list-item]
    else:
        adapter._calls = [sentinel] * (  # type: ignore[list-item]
            fd_module.MAX_FD_VELOCITY_CALL_COUNT - 6
        )
    calls_before = tuple(adapter._calls)
    evaluations_before = tuple(adapter._evaluations)
    trace_before = adapter._trace_sha256
    evaluator_calls_before = parent.evaluator_calls

    with pytest.raises(RuntimeError, match=expected_message):
        adapter.evaluate_physical(_initial_state())

    assert parent.evaluator_calls == evaluator_calls_before
    assert tuple(adapter._calls) == calls_before
    assert tuple(adapter._evaluations) == evaluations_before
    assert adapter._trace_sha256 == trace_before

    clean_parent = _MockFrozenParent()
    clean_adapter = _adapter(clean_parent)
    clean_adapter.evaluate_physical(_initial_state())
    clean_ledger = clean_adapter.snapshot()
    assert clean_parent.evaluator_calls == 7
    assert clean_ledger.physical_evaluation_count == 1
    assert clean_ledger.evaluator_call_count == 7
