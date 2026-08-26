"""Fail-closed aerodynamic state transaction for strong FSI trial solves.

Predictor--corrector evaluations must see one immutable pre-step state.  They
may propose new bound circulation, separated-LEV particles, newborn-TEV
particles and free-wake geometry, but only the latest issued proposal can be
committed, exactly once.  This module freezes the control-plane semantics used
by the later CUDA adapter; its NumPy state is a small deterministic oracle, not
a production host numerical path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable

import numpy as np

_FLOAT64 = np.dtype(np.float64)
_STATE_DOMAIN = "flux-v5m-aero-step-state-v1"
_PROPOSAL_DOMAIN = "flux-v5m-aero-step-proposal-v1"


class AeroTransactionError(RuntimeError):
    """Base exception for invalid transaction use."""


class AeroTransactionViolation(AeroTransactionError):
    """Raised when a trial or external actor mutates the live parent state."""


def _exact_nonnegative_int(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact int")
    return value


def _token(name: str, value: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty exact str")
    return value


def _frozen_array(
    name: str, value: np.ndarray, *, ndim: int, trailing_shape: tuple[int, ...] = ()
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != _FLOAT64:
        raise TypeError(f"{name} must use float64")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if trailing_shape and value.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(f"{name} must end with shape {trailing_shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


def _array_sha256(value: np.ndarray) -> str:
    header = json.dumps(
        {"dtype": "float64", "shape": list(value.shape)},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def _canonical_sha256(domain: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {"domain": domain, **payload},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AeroStepState:
    """Detached state of bound, separated-LEV, TEV and free-wake owners.

    Particle rows have the fixed oracle layout
    ``[x,y,z,gamma_x,gamma_y,gamma_z,core_radius,age]``.  Free-wake rows retain
    four ring corners plus one circulation scalar.
    """

    step_index: int
    parent_token: str
    bound_circulation: np.ndarray
    lev_particles: np.ndarray
    tev_particles: np.ndarray
    wake_geometry: np.ndarray
    wake_circulation: np.ndarray
    state_sha256: str

    def __init__(
        self,
        *,
        step_index: int,
        parent_token: str,
        bound_circulation: np.ndarray,
        lev_particles: np.ndarray,
        tev_particles: np.ndarray,
        wake_geometry: np.ndarray,
        wake_circulation: np.ndarray,
    ) -> None:
        step = _exact_nonnegative_int("step_index", step_index)
        parent = _token("parent_token", parent_token)
        bound = _frozen_array("bound_circulation", bound_circulation, ndim=1)
        lev = _frozen_array("lev_particles", lev_particles, ndim=2, trailing_shape=(8,))
        tev = _frozen_array("tev_particles", tev_particles, ndim=2, trailing_shape=(8,))
        wake = _frozen_array(
            "wake_geometry", wake_geometry, ndim=3, trailing_shape=(4, 3)
        )
        wake_gamma = _frozen_array("wake_circulation", wake_circulation, ndim=1)
        if wake.shape[0] != wake_gamma.shape[0]:
            raise ValueError(
                "wake_geometry and wake_circulation must contain the same row count"
            )
        digest = _canonical_sha256(
            _STATE_DOMAIN,
            {
                "step_index": step,
                "parent_token": parent,
                "bound_circulation_sha256": _array_sha256(bound),
                "lev_particles_sha256": _array_sha256(lev),
                "tev_particles_sha256": _array_sha256(tev),
                "wake_geometry_sha256": _array_sha256(wake),
                "wake_circulation_sha256": _array_sha256(wake_gamma),
            },
        )
        object.__setattr__(self, "step_index", step)
        object.__setattr__(self, "parent_token", parent)
        object.__setattr__(self, "bound_circulation", bound)
        object.__setattr__(self, "lev_particles", lev)
        object.__setattr__(self, "tev_particles", tev)
        object.__setattr__(self, "wake_geometry", wake)
        object.__setattr__(self, "wake_circulation", wake_gamma)
        object.__setattr__(self, "state_sha256", digest)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AeroTrialPayload:
    """Canonical output returned by one non-mutating aerodynamic trial."""

    source_state_sha256: str
    generalized_force: np.ndarray
    proposed_state: AeroStepState
    residual_norm: float
    payload_sha256: str

    def __init__(
        self,
        *,
        source_state_sha256: str,
        generalized_force: np.ndarray,
        proposed_state: AeroStepState,
        residual_norm: float,
    ) -> None:
        if not _is_sha256(source_state_sha256):
            raise ValueError("source_state_sha256 must be a lowercase SHA-256")
        force = _frozen_array("generalized_force", generalized_force, ndim=1)
        if force.size == 0:
            raise ValueError("generalized_force must be non-empty")
        if type(proposed_state) is not AeroStepState:
            raise TypeError("proposed_state must be an exact AeroStepState")
        if isinstance(residual_norm, bool) or not isinstance(
            residual_norm, (int, float, np.integer, np.floating)
        ):
            raise TypeError("residual_norm must be a real scalar")
        residual = float(residual_norm)
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError("residual_norm must be finite and non-negative")
        payload_sha = _canonical_sha256(
            "flux-v5m-aero-trial-payload-v1",
            {
                "source_state_sha256": source_state_sha256,
                "generalized_force_sha256": _array_sha256(force),
                "proposed_state_sha256": proposed_state.state_sha256,
                "residual_norm_hex": residual.hex(),
            },
        )
        object.__setattr__(self, "source_state_sha256", source_state_sha256)
        object.__setattr__(self, "generalized_force", force)
        object.__setattr__(self, "proposed_state", proposed_state)
        object.__setattr__(self, "residual_norm", residual)
        object.__setattr__(self, "payload_sha256", payload_sha)


@dataclass(frozen=True, slots=True, eq=False)
class AeroStepProposal:
    """Issued trial proposal; only exact live identity may be committed."""

    evaluation_index: int
    parent_state_sha256: str
    trial_input_sha256: str
    generalized_force: np.ndarray
    proposed_state: AeroStepState
    residual_norm: float
    proposal_sha256: str


CurrentState = Callable[[], AeroStepState]
EvaluateTrial = Callable[[AeroStepState, np.ndarray, np.ndarray], AeroTrialPayload]
CommitState = Callable[[AeroStepState, AeroStepState], None]
RestoreState = Callable[[AeroStepState], None]


def _trial_array(name: str, value: np.ndarray) -> np.ndarray:
    result = _frozen_array(name, value, ndim=1)
    if result.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return result


class AeroStepTransaction:
    """One pre-step snapshot with many trial evaluations and at most one commit."""

    __slots__ = (
        "_commit_state",
        "_current_state",
        "_evaluate_trial",
        "_evaluation_count",
        "_issued",
        "_restore_state",
        "_snapshot",
        "_status",
    )

    def __init__(
        self,
        *,
        current_state: CurrentState,
        evaluate_trial: EvaluateTrial,
        commit_state: CommitState,
        restore_state: RestoreState,
    ) -> None:
        callbacks = {
            "current_state": current_state,
            "evaluate_trial": evaluate_trial,
            "commit_state": commit_state,
            "restore_state": restore_state,
        }
        for name, callback in callbacks.items():
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        snapshot = current_state()
        if type(snapshot) is not AeroStepState:
            raise TypeError("current_state must return an exact AeroStepState")
        self._current_state = current_state
        self._evaluate_trial = evaluate_trial
        self._commit_state = commit_state
        self._restore_state = restore_state
        self._snapshot = snapshot
        self._evaluation_count = 0
        self._issued: AeroStepProposal | None = None
        self._status = "open"

    @property
    def snapshot(self) -> AeroStepState:
        return self._snapshot

    @property
    def status(self) -> str:
        return self._status

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    def _ensure_open(self) -> None:
        if self._status != "open":
            raise AeroTransactionError(
                f"transaction is closed with status {self._status}"
            )

    def _live(self) -> AeroStepState:
        live = self._current_state()
        if type(live) is not AeroStepState:
            raise AeroTransactionViolation(
                "current_state stopped returning an exact AeroStepState"
            )
        return live

    def _restore_parent(self) -> None:
        self._restore_state(self._snapshot)
        restored = self._live()
        if restored.state_sha256 != self._snapshot.state_sha256:
            raise AeroTransactionViolation(
                "restore_state did not restore the parent state"
            )

    def _fail_on_parent_drift(self, phase: str) -> None:
        if self._live().state_sha256 == self._snapshot.state_sha256:
            return
        self._restore_parent()
        self._status = "failed"
        raise AeroTransactionViolation(f"parent state drift {phase}")

    def evaluate(self, q_trial: np.ndarray, dq_trial: np.ndarray) -> AeroStepProposal:
        self._ensure_open()
        q = _trial_array("q_trial", q_trial)
        dq = _trial_array("dq_trial", dq_trial)
        if q.shape != dq.shape:
            raise ValueError("q_trial and dq_trial must have identical shapes")
        self._fail_on_parent_drift("before trial evaluation")
        try:
            payload = self._evaluate_trial(self._snapshot, q, dq)
        except Exception as error:
            if self._live().state_sha256 != self._snapshot.state_sha256:
                self._restore_parent()
                self._status = "failed"
                raise AeroTransactionViolation(
                    "trial mutated live aerodynamic state before failing"
                ) from error
            self._status = "failed"
            raise

        if self._live().state_sha256 != self._snapshot.state_sha256:
            self._restore_parent()
            self._status = "failed"
            raise AeroTransactionViolation(
                "trial mutated live aerodynamic state before returning"
            )
        if type(payload) is not AeroTrialPayload:
            self._status = "failed"
            raise TypeError("evaluate_trial must return an exact AeroTrialPayload")
        if payload.source_state_sha256 != self._snapshot.state_sha256:
            self._status = "failed"
            raise AeroTransactionViolation("trial payload is bound to a foreign parent")
        if payload.generalized_force.shape != q.shape:
            self._status = "failed"
            raise ValueError(
                "generalized_force shape differs from the structural state"
            )
        if payload.proposed_state.step_index != self._snapshot.step_index + 1:
            self._status = "failed"
            raise AeroTransactionViolation("proposed state has the wrong step index")
        if payload.proposed_state.parent_token != self._snapshot.state_sha256:
            self._status = "failed"
            raise AeroTransactionViolation("proposed state has the wrong parent token")

        self._evaluation_count += 1
        trial_sha = _canonical_sha256(
            "flux-v5m-aero-trial-input-v1",
            {
                "q_sha256": _array_sha256(q),
                "dq_sha256": _array_sha256(dq),
            },
        )
        proposal_sha = _canonical_sha256(
            _PROPOSAL_DOMAIN,
            {
                "evaluation_index": self._evaluation_count,
                "parent_state_sha256": self._snapshot.state_sha256,
                "trial_input_sha256": trial_sha,
                "payload_sha256": payload.payload_sha256,
            },
        )
        proposal = AeroStepProposal(
            evaluation_index=self._evaluation_count,
            parent_state_sha256=self._snapshot.state_sha256,
            trial_input_sha256=trial_sha,
            generalized_force=payload.generalized_force,
            proposed_state=payload.proposed_state,
            residual_norm=payload.residual_norm,
            proposal_sha256=proposal_sha,
        )
        self._issued = proposal
        return proposal

    def commit(self, proposal: AeroStepProposal) -> AeroStepState:
        self._ensure_open()
        if type(proposal) is not AeroStepProposal or proposal is not self._issued:
            raise AeroTransactionError("commit requires the latest issued proposal")
        self._fail_on_parent_drift("before commit")
        try:
            self._commit_state(self._snapshot, proposal.proposed_state)
        except Exception:
            self._restore_parent()
            self._status = "failed"
            raise
        live = self._live()
        if live.state_sha256 != proposal.proposed_state.state_sha256:
            self._restore_parent()
            self._status = "failed"
            raise AeroTransactionViolation(
                "commit_state did not install the issued proposed state"
            )
        self._status = "committed"
        self._issued = None
        return proposal.proposed_state

    def abort(self) -> None:
        self._ensure_open()
        if self._live().state_sha256 != self._snapshot.state_sha256:
            self._restore_parent()
            self._status = "failed"
            raise AeroTransactionViolation("live state drifted before abort")
        self._status = "aborted"
        self._issued = None


def begin_aero_step_transaction(
    *,
    current_state: CurrentState,
    evaluate_trial: EvaluateTrial,
    commit_state: CommitState,
    restore_state: RestoreState,
) -> AeroStepTransaction:
    """Begin one exact pre-step transaction."""

    return AeroStepTransaction(
        current_state=current_state,
        evaluate_trial=evaluate_trial,
        commit_state=commit_state,
        restore_state=restore_state,
    )


__all__ = [
    "AeroStepProposal",
    "AeroStepState",
    "AeroStepTransaction",
    "AeroTransactionError",
    "AeroTransactionViolation",
    "AeroTrialPayload",
    "begin_aero_step_transaction",
]
