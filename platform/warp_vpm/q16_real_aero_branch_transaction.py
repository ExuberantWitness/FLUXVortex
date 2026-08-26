"""Branch-and-commit ownership for the real CUDA LEV/TEV/free-wake solver.

Each predictor/corrector evaluation receives a deep CUDA/Ptera branch of one
unchanged parent solver.  Discarded branches never become live.  Committing the
latest exact proposal advances the owner pointer once, which avoids attempting
to rewind a partially mutated Ptera object graph in place.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pickle
from typing import Any, Callable

import numpy as np
import torch
import warp as wp

from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.warp_fsi.q16_mandatory_aero_mode import (
    Q16MandatoryAeroModeGuard,
    require_q16_mandatory_aero_mode,
)


class Q16AeroBranchTransactionError(RuntimeError):
    """Invalid lifecycle use of a Q16 aerodynamic branch transaction."""


class Q16AeroBranchTransactionViolation(Q16AeroBranchTransactionError):
    """A live parent or issued proposal drifted outside its owner."""


TrialEvaluator = Callable[
    [CudaJointLEVTEVSolver, wp.array, wp.array],
    wp.array,
]
BranchFactory = Callable[[CudaJointLEVTEVSolver], CudaJointLEVTEVSolver]


def _solver_sha256(solver: CudaJointLEVTEVSolver) -> str:
    torch.cuda.synchronize(solver.cuda_device)
    return hashlib.sha256(pickle.dumps(solver, protocol=5)).hexdigest()


def _clone_solver(solver: CudaJointLEVTEVSolver) -> CudaJointLEVTEVSolver:
    """Clone through Ptera's pickle state; generic deepcopy loses panel vertices."""

    torch.cuda.synchronize(solver.cuda_device)
    clone = pickle.loads(pickle.dumps(solver, protocol=5))
    if type(clone) is not CudaJointLEVTEVSolver:
        raise Q16AeroBranchTransactionViolation(
            "serialized aerodynamic branch changed exact solver type"
        )
    return clone


def _warp_array_sha256(name: str, value: Any) -> str:
    if not isinstance(value, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if not value.device.is_cuda:
        raise ValueError(f"{name} must reside on CUDA")
    if value.dtype != wp.float64:
        raise TypeError(f"{name} must use Warp float64")
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have non-empty shape (batch, dof)")
    host = value.numpy()
    if not bool(np.isfinite(host).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    header = json.dumps(
        {"dtype": "float64", "shape": list(value.shape)},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + host.tobytes(order="C")).hexdigest()


def _proposal_sha256(
    evaluation_index: int,
    parent_state_sha256: str,
    proposed_state_sha256: str,
    generalized_force_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "domain": "flux-v5m-q16-real-aero-branch-proposal-v1",
            "evaluation_index": evaluation_index,
            "generalized_force_sha256": generalized_force_sha256,
            "parent_state_sha256": parent_state_sha256,
            "proposed_state_sha256": proposed_state_sha256,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class Q16CudaAeroBranchProposal:
    evaluation_index: int
    parent_state_sha256: str
    proposed_state_sha256: str
    generalized_force: wp.array
    generalized_force_sha256: str
    proposed_solver: CudaJointLEVTEVSolver
    proposal_sha256: str


@dataclass(frozen=True, slots=True, eq=False)
class _Q16PreparedAeroCommit:
    proposal: Q16CudaAeroBranchProposal


class Q16CudaAeroSolverOwner:
    """Single mutable pointer to the only committed real aerodynamic solver."""

    __slots__ = ("_current_solver", "_generation")

    def __init__(self, solver: CudaJointLEVTEVSolver) -> None:
        if type(solver) is not CudaJointLEVTEVSolver:
            raise TypeError("solver must be an exact CudaJointLEVTEVSolver")
        require_q16_mandatory_aero_mode(solver)
        self._current_solver = solver
        self._generation = 0

    @property
    def current_solver(self) -> CudaJointLEVTEVSolver:
        return self._current_solver

    @property
    def generation(self) -> int:
        return self._generation

    def begin(
        self,
        evaluate_trial: TrialEvaluator,
        *,
        branch_factory: BranchFactory = _clone_solver,
    ) -> Q16CudaAeroBranchTransaction:
        return Q16CudaAeroBranchTransaction(
            self,
            evaluate_trial,
            branch_factory=branch_factory,
        )

    def _restore_failed(
        self,
        transaction_parent: CudaJointLEVTEVSolver,
        parent_snapshot: CudaJointLEVTEVSolver,
    ) -> None:
        if self._current_solver is transaction_parent:
            self._current_solver = parent_snapshot
            self._generation += 1

    def _commit(
        self,
        parent: CudaJointLEVTEVSolver,
        generation: int,
        proposed: CudaJointLEVTEVSolver,
    ) -> None:
        if self._current_solver is not parent or self._generation != generation:
            raise Q16AeroBranchTransactionViolation(
                "aerodynamic owner generation drift"
            )
        require_q16_mandatory_aero_mode(proposed)
        self._commit_prechecked(parent, generation, proposed)

    def _commit_prechecked(
        self,
        parent: CudaJointLEVTEVSolver,
        generation: int,
        proposed: CudaJointLEVTEVSolver,
    ) -> None:
        """Publish after the caller completed every fallible validation."""

        if self._current_solver is not parent or self._generation != generation:
            raise Q16AeroBranchTransactionViolation(
                "aerodynamic owner generation drift"
            )
        self._current_solver = proposed
        self._generation += 1


class Q16CudaAeroBranchTransaction:
    """Many real-solver branches from one parent and at most one owner advance."""

    __slots__ = (
        "_evaluation_count",
        "_evaluate_trial",
        "_branch_factory",
        "_generation",
        "_issued",
        "_mode_guard",
        "_owner",
        "_parent",
        "_parent_sha256",
        "_parent_snapshot",
        "_prepared",
        "_status",
    )

    def __init__(
        self,
        owner: Q16CudaAeroSolverOwner,
        evaluate_trial: TrialEvaluator,
        *,
        branch_factory: BranchFactory = _clone_solver,
    ) -> None:
        if type(owner) is not Q16CudaAeroSolverOwner:
            raise TypeError("owner must be an exact Q16CudaAeroSolverOwner")
        if not callable(evaluate_trial):
            raise TypeError("evaluate_trial must be callable")
        if not callable(branch_factory):
            raise TypeError("branch_factory must be callable")
        parent = owner.current_solver
        guard = require_q16_mandatory_aero_mode(parent)
        parent_sha = _solver_sha256(parent)
        snapshot = branch_factory(parent)
        if type(snapshot) is not CudaJointLEVTEVSolver or snapshot is parent:
            raise Q16AeroBranchTransactionViolation(
                "branch_factory did not issue a detached exact solver"
            )
        guard.verify(parent)
        if _solver_sha256(parent) != parent_sha:
            raise Q16AeroBranchTransactionViolation(
                "live parent drifted while opening transaction"
            )
        require_q16_mandatory_aero_mode(snapshot)
        self._owner = owner
        self._evaluate_trial = evaluate_trial
        self._branch_factory = branch_factory
        self._parent = parent
        self._parent_snapshot = snapshot
        self._parent_sha256 = parent_sha
        self._mode_guard: Q16MandatoryAeroModeGuard = guard
        self._generation = owner.generation
        self._evaluation_count = 0
        self._issued: Q16CudaAeroBranchProposal | None = None
        self._prepared: _Q16PreparedAeroCommit | None = None
        self._status = "open"

    @property
    def status(self) -> str:
        return self._status

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @property
    def parent_state_sha256(self) -> str:
        return self._parent_sha256

    def _ensure_open(self) -> None:
        if self._status != "open":
            raise Q16AeroBranchTransactionError(
                f"transaction is closed with status {self._status}"
            )

    def _parent_drifted(self) -> bool:
        if (
            self._owner.current_solver is not self._parent
            or self._owner.generation != self._generation
        ):
            return True
        try:
            self._mode_guard.verify(self._parent)
            return _solver_sha256(self._parent) != self._parent_sha256
        except Exception:
            return True

    def _reject_parent_drift(self, phase: str) -> None:
        if not self._parent_drifted():
            return
        self._owner._restore_failed(self._parent, self._parent_snapshot)
        self._status = "failed"
        self._issued = None
        raise Q16AeroBranchTransactionViolation(f"live parent drift {phase}")

    def evaluate(self, q_trial: Any, dq_trial: Any) -> Q16CudaAeroBranchProposal:
        self._ensure_open()
        self._prepared = None
        q_sha = _warp_array_sha256("q_trial", q_trial)
        dq_sha = _warp_array_sha256("dq_trial", dq_trial)
        if q_trial.shape != dq_trial.shape:
            raise ValueError("q_trial and dq_trial shapes differ")
        if q_trial.device.alias != dq_trial.device.alias:
            raise ValueError("q_trial and dq_trial devices differ")
        self._reject_parent_drift("before trial")
        branch = self._branch_factory(self._parent)
        if type(branch) is not CudaJointLEVTEVSolver or branch is self._parent:
            self._status = "failed"
            raise Q16AeroBranchTransactionViolation(
                "branch_factory did not issue a detached exact solver"
            )
        require_q16_mandatory_aero_mode(branch)
        branch_q = wp.clone(q_trial)
        branch_dq = wp.clone(dq_trial)
        try:
            force = self._evaluate_trial(branch, branch_q, branch_dq)
        except Exception as error:
            if self._parent_drifted():
                self._owner._restore_failed(self._parent, self._parent_snapshot)
                self._status = "failed"
                raise Q16AeroBranchTransactionViolation(
                    "live parent drift during failed trial"
                ) from error
            self._status = "failed"
            raise
        self._reject_parent_drift("during trial")
        if (
            _warp_array_sha256("q_trial", q_trial) != q_sha
            or _warp_array_sha256("dq_trial", dq_trial) != dq_sha
        ):
            self._status = "failed"
            raise Q16AeroBranchTransactionViolation(
                "structural trial input drift during aerodynamic evaluation"
            )
        require_q16_mandatory_aero_mode(branch)
        force_sha = _warp_array_sha256("generalized_force", force)
        if force.shape != q_trial.shape or force.device.alias != q_trial.device.alias:
            self._status = "failed"
            raise ValueError("generalized_force shape/device differs from q_trial")
        detached_force = wp.clone(force)
        detached_force_sha = _warp_array_sha256("generalized_force", detached_force)
        if detached_force_sha != force_sha:
            self._status = "failed"
            raise Q16AeroBranchTransactionViolation(
                "generalized_force changed while being detached"
            )
        proposed_sha = _solver_sha256(branch)
        self._evaluation_count += 1
        proposal_sha = _proposal_sha256(
            self._evaluation_count,
            self._parent_sha256,
            proposed_sha,
            detached_force_sha,
        )
        proposal = Q16CudaAeroBranchProposal(
            evaluation_index=self._evaluation_count,
            parent_state_sha256=self._parent_sha256,
            proposed_state_sha256=proposed_sha,
            generalized_force=detached_force,
            generalized_force_sha256=detached_force_sha,
            proposed_solver=branch,
            proposal_sha256=proposal_sha,
        )
        self._issued = proposal
        return proposal

    def _prepare_commit(
        self, proposal: Q16CudaAeroBranchProposal
    ) -> _Q16PreparedAeroCommit:
        self._ensure_open()
        if (
            type(proposal) is not Q16CudaAeroBranchProposal
            or proposal is not self._issued
        ):
            raise Q16AeroBranchTransactionError(
                "commit requires the latest issued proposal"
            )
        self._reject_parent_drift("before commit")
        require_q16_mandatory_aero_mode(proposal.proposed_solver)
        if _solver_sha256(proposal.proposed_solver) != proposal.proposed_state_sha256:
            self._status = "failed"
            raise Q16AeroBranchTransactionViolation("issued proposal drift")
        if (
            _warp_array_sha256("generalized_force", proposal.generalized_force)
            != proposal.generalized_force_sha256
        ):
            self._status = "failed"
            raise Q16AeroBranchTransactionViolation("issued proposal force drift")
        prepared = _Q16PreparedAeroCommit(proposal=proposal)
        self._prepared = prepared
        return prepared

    def _commit_prepared_prechecked(
        self, prepared: _Q16PreparedAeroCommit
    ) -> CudaJointLEVTEVSolver:
        """Publish one already-validated proposal without new scientific work."""

        if (
            type(prepared) is not _Q16PreparedAeroCommit
            or prepared is not self._prepared
        ):
            raise Q16AeroBranchTransactionError(
                "commit requires this transaction's prepared proposal"
            )
        proposal = prepared.proposal
        self._owner._commit_prechecked(
            self._parent, self._generation, proposal.proposed_solver
        )
        self._status = "committed"
        self._issued = None
        self._prepared = None
        return proposal.proposed_solver

    def commit(self, proposal: Q16CudaAeroBranchProposal) -> CudaJointLEVTEVSolver:
        return self._commit_prepared_prechecked(self._prepare_commit(proposal))

    def abort(self) -> None:
        self._ensure_open()
        self._reject_parent_drift("before abort")
        self._status = "aborted"
        self._issued = None
        self._prepared = None


__all__ = [
    "Q16AeroBranchTransactionError",
    "Q16AeroBranchTransactionViolation",
    "Q16CudaAeroBranchProposal",
    "Q16CudaAeroBranchTransaction",
    "Q16CudaAeroSolverOwner",
]
