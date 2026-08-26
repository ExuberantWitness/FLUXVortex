"""PartitionedFreeFlightFSI: strong coupling for free-flight flapping.

Unknowns per step: body pose/velocity + Q16 states + aero state.
Same fixed-point iteration as the fixed-frame FSI, but the body SE(3)
state participates in the residual.

U6 skeleton (plan §10): every iteration is a full trial — surface
frames from the current dynamic endpoint GUESS, one aero proposal from
the committed aero parent, ONE ``J^T f`` projection into a
``GeneralizedLoadPacket``, one dynamics propose from the committed
dynamic parent — followed by a kinematic fixed-point residual on the
endpoint.  On convergence the accepted trial is formal-replayed (fresh
aero proposal from the same committed parent at the converged endpoint)
and committed exactly once per subsystem per step.

The aero stepper, composite dynamics and surface kinematics are
duck-typed and mock-injectable; the production multi-surface V5M
adapter is the next phase.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch

from ..dynamics.load_packet import (
    GeneralizedLoadPacket,
    project_surface_loads,
)
from ..state.world import WorldDynamicState
from .partitioned import TransactionCounters


@dataclass(frozen=True)
class FreeFlightStepResult:
    """Per-step coupling report (plan §10.2 true-count semantics)."""
    iterations: int
    residual: float
    work_error: float
    surface_power: float
    generalized_power: float


def _as_world_state(state: Any) -> Any:
    """Normalize a dynamics proposal into a WorldDynamicState-like object."""
    if hasattr(state, "body_states") or hasattr(state, "elastic_states"):
        return state
    if isinstance(state, dict):
        return WorldDynamicState(
            elastic_states=dict(state.get("elastic_states") or {}),
            body_states=dict(state.get("body_states") or {}),
            joint_states=dict(state.get("joint_states") or {}),
        )
    return state


def _iter_tensor_fields(state: Any):
    """Yield (key, tensor) for every body/elastic generalized coordinate.

    Joint states are prescribed (pure functions of time) and therefore
    excluded from the fixed-point residual: they cannot drift between
    iterations by construction.
    """
    for body_id, body in (getattr(state, "body_states", None) or {}).items():
        for name in ("position_I", "quaternion_IB",
                     "linear_velocity_I", "angular_velocity_B"):
            tensor = getattr(body, name, None)
            if isinstance(tensor, torch.Tensor):
                yield f"body:{body_id}:{name}", tensor
    for elastic_id, elastic in (getattr(state, "elastic_states", None) or {}).items():
        for name in ("q", "qd"):
            tensor = getattr(elastic, name, None)
            if isinstance(tensor, torch.Tensor):
                yield f"elastic:{elastic_id}:{name}", tensor


def _blend(old: Any, new: Any, alpha: float) -> Any:
    """Relaxed combination ``old + alpha (new - old)`` for state trees.

    Tensors blend element-wise; dataclasses/dicts/tuples recurse; leaf
    values that cannot blend (strings, warp arrays, foreign objects)
    keep the guess value.  Constant tensor fields (mass, inertia) are
    identical on both sides, so blending them is a no-op.
    """
    if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor) \
            and old.shape == new.shape and old.device == new.device:
        return old + alpha * (new - old)
    if dataclasses.is_dataclass(old) and not isinstance(old, type) \
            and type(old) is type(new):
        changes = {
            f.name: _blend(getattr(old, f.name), getattr(new, f.name), alpha)
            for f in dataclasses.fields(old)
        }
        return dataclasses.replace(old, **changes)
    if isinstance(old, dict) and isinstance(new, dict):
        return {key: _blend(old[key], new.get(key, old[key]), alpha) for key in old}
    if isinstance(old, (tuple, list)) and isinstance(new, (tuple, list)) \
            and len(old) == len(new):
        blended = [_blend(o, n, alpha) for o, n in zip(old, new)]
        return tuple(blended) if isinstance(old, tuple) else blended
    return old


class PartitionedFreeFlightFSI:
    """Free-flight strong coupling — composes U2 (V5M state) + U4 (multi-surface)
    + U5 (SE3/joints/Q16) under one global transaction.

    Parameters
    ----------
    aero_stepper:
        ``propose(aero_state, frames, dt) -> proposal`` where the proposal
        carries ``surface_loads`` (tuple of :class:`SurfaceLoad`); on
        convergence ``commit(owner, proposal)`` installs it.
    composite_dynamics:
        :class:`CompositeBodyJointQ16Dynamics`-like; ``propose(committed,
        loads, dt, time)`` always advances from the COMMITTED dynamic
        parent (plan §10.2 trial rule), never from the relaxed guess.
    surface_kinematics:
        ``evaluate(dynamic_state, time) -> tuple[SurfaceFrame, ...]``
        evaluated at the current endpoint guess.
    """

    def __init__(self, aero_stepper, composite_dynamics, surface_kinematics,
                 coupling_tolerance: float = 5e-7,
                 max_iterations: int = 20,
                 relaxation: float = 0.7,
                 work_tolerance: float | None = 1e-8,
                 formal_replay: bool = True,
                 joint_torque_extractor=None,
                 elastic_force_extractor=None):
        self._aero = aero_stepper
        self._dynamics = composite_dynamics
        self._kinematics = surface_kinematics
        self.tolerance = coupling_tolerance
        self.max_iterations = max_iterations
        self.relaxation = relaxation
        self.work_tolerance = work_tolerance
        self.formal_replay = formal_replay
        self._joint_torque_extractor = joint_torque_extractor
        self._elastic_force_extractor = elastic_force_extractor
        self.counters = TransactionCounters()

    # ------------------------------------------------------------------
    # Load projection: ONE J^T f shared by body/joint/elastic (plan §7.3).
    # ------------------------------------------------------------------
    def _project_loads(self, proposal: Any, guess_state: Any) -> GeneralizedLoadPacket:
        surface_loads = tuple(getattr(proposal, "surface_loads", None) or ())
        if self._elastic_force_extractor is not None:
            elastic_forces = self._elastic_force_extractor(proposal)
        else:
            elastic_forces = getattr(proposal, "elastic_forces", None) or {}
        if self._joint_torque_extractor is not None:
            joint_torques = self._joint_torque_extractor(proposal)
        else:
            joint_torques = getattr(proposal, "joint_torques", None) or {}
        return project_surface_loads(
            surface_loads, guess_state,
            joint_torques=joint_torques, elastic_forces=elastic_forces)

    # ------------------------------------------------------------------
    # Fixed-point residual over the dynamic endpoint guess.
    # ------------------------------------------------------------------
    @staticmethod
    def _residual(guess: Any, candidate: Any) -> float:
        old_fields = dict(_iter_tensor_fields(guess))
        new_fields = dict(_iter_tensor_fields(candidate))
        worst = 0.0
        for key, new_tensor in new_fields.items():
            old_tensor = old_fields.get(key)
            if not isinstance(old_tensor, torch.Tensor) or old_tensor.shape != new_tensor.shape:
                continue
            delta = float(torch.linalg.vector_norm(new_tensor - old_tensor).item())
            if delta == 0.0:
                continue
            scale = max(
                float(torch.linalg.vector_norm(old_tensor).item()),
                float(torch.linalg.vector_norm(new_tensor).item()))
            worst = max(worst, delta / scale)
        return worst

    def _check_work_gate(self, packet: GeneralizedLoadPacket) -> None:
        if self.work_tolerance is None:
            return
        if not packet.audit_passes(tolerance=self.work_tolerance):
            raise RuntimeError(
                "work-conjugacy gate failed: |f.v - Q.qd|/max = "
                f"{packet.relative_work_error:.3e} > {self.work_tolerance:.1e} "
                f"(surface power {packet.surface_power:.6e}, generalized "
                f"power {packet.generalized_power:.6e})")

    # ------------------------------------------------------------------
    # One outer step.
    # ------------------------------------------------------------------
    def advance(self, owner, dt: float, time: float) -> FreeFlightStepResult:
        """One free-flight outer step with fixed-point coupling."""
        committed = owner.dynamic_state  # every trial starts from this parent
        guess = committed

        for iteration in range(1, self.max_iterations + 1):
            # 1. Surface frames from the current dynamic endpoint guess.
            frames = self._kinematics.evaluate(guess, time + dt)
            # 2. Aero propose at these frames, from the committed aero parent.
            proposal = self._aero.propose(owner.aero_state, frames, dt)
            self.counters.aero_proposal_count += 1
            # 3. ONE projection of the surface loads into the packet.
            loads = self._project_loads(proposal, guess)
            self._check_work_gate(loads)
            # 4. Dynamics propose — again from the committed dynamic parent.
            candidate = _as_world_state(
                self._dynamics.propose(committed, loads, dt, time))
            self.counters.dynamic_proposal_count += 1
            # 5. Kinematic fixed-point residual on the endpoint guess.
            residual = self._residual(guess, candidate)
            if residual <= self.tolerance:
                # 6. Formal replay from the same committed parents at the
                #    converged endpoint, then commit exactly once.
                if self.formal_replay:
                    replay_frames = self._kinematics.evaluate(candidate, time + dt)
                    proposal = self._aero.propose(owner.aero_state, replay_frames, dt)
                    self.counters.aero_proposal_count += 1
                    self.counters.formal_replay_count += 1
                self._aero.commit(owner, proposal)
                self.counters.commit_count += 1
                owner.dynamic_state = candidate
                owner.previous_load = loads
                owner.generation += 1
                return FreeFlightStepResult(
                    iterations=iteration,
                    residual=residual,
                    work_error=loads.relative_work_error,
                    surface_power=loads.surface_power,
                    generalized_power=loads.generalized_power,
                )
            # Relax the endpoint guess and retry (trial discarded).
            self.counters.discarded_trial_count += 1
            guess = _blend(guess, candidate, self.relaxation)

        raise RuntimeError(
            f"free-flight coupling did not converge in {self.max_iterations} "
            f"iterations (residual {residual:.3e} > {self.tolerance:.1e}; "
            f"counters {dataclasses.asdict(self.counters)})")
