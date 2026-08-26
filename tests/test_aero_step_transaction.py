"""Transaction contract for PC trials over separated LEV, TEV and free wake."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fluxvortex.warp_fsi.aero_step_transaction import (
    AeroStepState,
    AeroTransactionError,
    AeroTransactionViolation,
    AeroTrialPayload,
    begin_aero_step_transaction,
)


def _state(step: int = 0, shift: float = 0.0, parent_token: str = "genesis"):
    return AeroStepState(
        step_index=step,
        parent_token=parent_token,
        bound_circulation=np.array([0.2 + shift, -0.1], dtype=np.float64),
        lev_particles=np.array(
            [[0.1 + shift, 0.2, 0.3, 0.0, 1.0, 0.0, 0.02, 0.4]],
            dtype=np.float64,
        ),
        tev_particles=np.array(
            [[0.7, 0.2 + shift, 0.1, 0.0, -0.4, 0.0, 0.03, 0.2]],
            dtype=np.float64,
        ),
        wake_geometry=np.array(
            [
                [
                    [0.8 + shift, -0.5, 0.0],
                    [0.8 + shift, 0.5, 0.0],
                    [1.0 + shift, 0.5, 0.0],
                    [1.0 + shift, -0.5, 0.0],
                ]
            ],
            dtype=np.float64,
        ),
        wake_circulation=np.array([0.15 + shift], dtype=np.float64),
    )


class _FakeAeroEngine:
    def __init__(self) -> None:
        self.state = _state()
        self.evaluation_calls = 0
        self.commit_calls = 0
        self.restore_calls = 0
        self.fail_trial = False
        self.mutate_then_fail = False

    def current_state(self) -> AeroStepState:
        return self.state

    def restore(self, snapshot: AeroStepState) -> None:
        self.restore_calls += 1
        self.state = snapshot

    def evaluate(
        self, snapshot: AeroStepState, q_trial: np.ndarray, dq_trial: np.ndarray
    ) -> AeroTrialPayload:
        self.evaluation_calls += 1
        if self.mutate_then_fail:
            self.state = _state(step=9, shift=8.0, parent_token="hostile")
            raise RuntimeError("hostile trial mutation")
        if self.fail_trial:
            raise RuntimeError("synthetic trial failure")
        shift = 0.01 * float(np.sum(q_trial) + np.sum(dq_trial))
        next_state = _state(
            step=snapshot.step_index + 1,
            shift=shift,
            parent_token=snapshot.state_sha256,
        )
        return AeroTrialPayload(
            source_state_sha256=snapshot.state_sha256,
            generalized_force=np.array([1.0 + shift, -2.0, 0.5], dtype=np.float64),
            proposed_state=next_state,
            residual_norm=abs(shift),
        )

    def commit(self, parent: AeroStepState, proposed: AeroStepState) -> None:
        if self.state.state_sha256 != parent.state_sha256:
            raise RuntimeError("stale parent")
        self.commit_calls += 1
        self.state = proposed


def _begin(engine: _FakeAeroEngine):
    return begin_aero_step_transaction(
        current_state=engine.current_state,
        evaluate_trial=engine.evaluate,
        commit_state=engine.commit,
        restore_state=engine.restore,
    )


def _trial(scale: float = 1.0):
    return (
        np.array([0.1, -0.2, 0.3], dtype=np.float64) * scale,
        np.array([0.04, 0.02, -0.01], dtype=np.float64) * scale,
    )


def test_aero_state_owns_readonly_detached_lev_tev_and_wake_arrays() -> None:
    source = np.ones((1, 8), dtype=np.float64)
    state = AeroStepState(
        step_index=0,
        parent_token="genesis",
        bound_circulation=np.ones(2, dtype=np.float64),
        lev_particles=source,
        tev_particles=source,
        wake_geometry=np.ones((1, 4, 3), dtype=np.float64),
        wake_circulation=np.ones(1, dtype=np.float64),
    )
    source[0, 0] = 99.0
    assert state.lev_particles[0, 0] == 1.0
    assert state.lev_particles.flags.writeable is False
    assert state.tev_particles.flags.writeable is False
    assert state.wake_geometry.flags.writeable is False
    assert len(state.state_sha256) == 64


def test_repeated_pc_trials_do_not_advance_live_state_and_only_latest_commits() -> None:
    engine = _FakeAeroEngine()
    parent_sha = engine.state.state_sha256
    tx = _begin(engine)
    proposal_1 = tx.evaluate(*_trial(1.0))
    assert engine.state.state_sha256 == parent_sha
    proposal_2 = tx.evaluate(*_trial(2.0))
    assert engine.state.state_sha256 == parent_sha
    with pytest.raises(AeroTransactionError, match="latest issued proposal"):
        tx.commit(proposal_1)
    committed = tx.commit(proposal_2)
    assert committed is proposal_2.proposed_state
    assert engine.commit_calls == 1
    assert engine.state.state_sha256 == proposal_2.proposed_state.state_sha256
    assert tx.status == "committed"
    with pytest.raises(AeroTransactionError, match="closed"):
        tx.commit(proposal_2)


def test_abort_preserves_parent_and_prevents_later_commit() -> None:
    engine = _FakeAeroEngine()
    parent = engine.state
    tx = _begin(engine)
    proposal = tx.evaluate(*_trial())
    tx.abort()
    assert tx.status == "aborted"
    assert engine.state.state_sha256 == parent.state_sha256
    assert engine.commit_calls == 0
    with pytest.raises(AeroTransactionError, match="closed"):
        tx.commit(proposal)


def test_trial_exception_with_hostile_mutation_restores_parent_and_clean_retry() -> (
    None
):
    engine = _FakeAeroEngine()
    parent = engine.state
    engine.mutate_then_fail = True
    tx = _begin(engine)
    with pytest.raises(
        AeroTransactionViolation, match="mutated live aerodynamic state"
    ):
        tx.evaluate(*_trial())
    assert tx.status == "failed"
    assert engine.state.state_sha256 == parent.state_sha256
    assert engine.restore_calls == 1
    engine.mutate_then_fail = False
    retry = _begin(engine)
    proposal = retry.evaluate(*_trial())
    retry.commit(proposal)
    fresh_engine = _FakeAeroEngine()
    fresh = _begin(fresh_engine)
    fresh_proposal = fresh.evaluate(*_trial())
    fresh.commit(fresh_proposal)
    assert engine.state.state_sha256 == fresh_engine.state.state_sha256
    assert proposal.proposal_sha256 == fresh_proposal.proposal_sha256


def test_foreign_or_resealed_proposal_cannot_be_committed() -> None:
    engine_a = _FakeAeroEngine()
    engine_b = _FakeAeroEngine()
    tx_a = _begin(engine_a)
    tx_b = _begin(engine_b)
    proposal_a = tx_a.evaluate(*_trial())
    proposal_b = tx_b.evaluate(*_trial())
    with pytest.raises(AeroTransactionError, match="latest issued proposal"):
        tx_a.commit(proposal_b)
    forged = replace(proposal_a)
    with pytest.raises(AeroTransactionError, match="latest issued proposal"):
        tx_a.commit(forged)
    tx_a.commit(proposal_a)


def test_external_parent_drift_before_commit_is_restored_and_rejected() -> None:
    engine = _FakeAeroEngine()
    parent = engine.state
    tx = _begin(engine)
    proposal = tx.evaluate(*_trial())
    engine.state = _state(step=4, shift=2.0, parent_token="external-drift")
    with pytest.raises(AeroTransactionViolation, match="parent state drift"):
        tx.commit(proposal)
    assert tx.status == "failed"
    assert engine.state.state_sha256 == parent.state_sha256
    assert engine.commit_calls == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("bound_circulation", np.ones(2, dtype=np.float32)),
        ("lev_particles", np.ones((1, 7), dtype=np.float64)),
        ("tev_particles", np.full((1, 8), np.nan, dtype=np.float64)),
        ("wake_geometry", np.ones((1, 3, 3), dtype=np.float64)),
    ],
)
def test_aero_state_rejects_wrong_dtype_shape_and_nonfinite_inputs(
    field: str, value: np.ndarray
) -> None:
    kwargs = {
        "step_index": 0,
        "parent_token": "genesis",
        "bound_circulation": np.ones(2, dtype=np.float64),
        "lev_particles": np.ones((1, 8), dtype=np.float64),
        "tev_particles": np.ones((1, 8), dtype=np.float64),
        "wake_geometry": np.ones((1, 4, 3), dtype=np.float64),
        "wake_circulation": np.ones(1, dtype=np.float64),
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        AeroStepState(**kwargs)
