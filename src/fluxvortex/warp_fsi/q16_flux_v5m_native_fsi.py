"""Strong predictor/corrector coupling of Q16 and native FLUX-V5M CUDA."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import torch
import warp as wp

from . import config
from .q16_flux_v5m_native import (
    NativeV5MProposal,
    Q16NativeV5MOwner,
    Q16NativeV5MSolver,
)
from .q16_flux_v5m_author_loads import (
    Q16NativeAddedMassAction,
    Q16NativeAuthorEndpointLoad,
)
from .q16_structural_solver import Q16CudaNewmarkStepper, Q16StructuralStepResult


NATIVE_Q16_FSI_CONTRACT = "q16-flux-v5m-native-strong-pc-cuda-v1"
ProgressCallback = Callable[[dict[str, Any]], None]


def _clone(value: wp.array) -> wp.array:
    if not isinstance(value, wp.array) or not value.device.is_cuda:
        raise ValueError("native Q16 FSI state must be a CUDA Warp array")
    return wp.clone(value)


def _combine(left: wp.array, right: wp.array, right_factor: float = 1.0) -> wp.array:
    if left.shape != right.shape or left.device.alias != right.device.alias:
        raise ValueError("native Q16 FSI force operands differ")
    result = wp.to_torch(left) + float(right_factor) * wp.to_torch(right)
    if result.device.type != "cuda" or result.dtype is not torch.float64:
        raise RuntimeError("native Q16 FSI force combination left CUDA float64")
    return wp.clone(wp.from_torch(result, dtype=config.DTYPE, requires_grad=False))


def _interpolate(left: wp.array, right: wp.array, beta: float) -> wp.array:
    result = wp.to_torch(left) + float(beta) * (wp.to_torch(right) - wp.to_torch(left))
    return wp.clone(wp.from_torch(result, dtype=config.DTYPE, requires_grad=False))


def _relative_error(
    state: wp.array,
    velocity: wp.array,
    reference_state: wp.array,
    reference_velocity: wp.array,
    delta_time: float,
) -> float:
    q, v = wp.to_torch(state), wp.to_torch(velocity)
    qr, vr = wp.to_torch(reference_state), wp.to_torch(reference_velocity)
    state_error = torch.linalg.vector_norm(q - qr)
    velocity_error = float(delta_time) * torch.linalg.vector_norm(v - vr)
    scale = torch.maximum(
        torch.ones((), device=q.device, dtype=torch.float64),
        torch.maximum(torch.linalg.vector_norm(q), torch.linalg.vector_norm(qr)),
    )
    value = torch.maximum(state_error, velocity_error) / scale
    return float(value.item())


class _Aitken:
    def __init__(self, initial: float) -> None:
        if not 0.0 < initial <= 1.0:
            raise ValueError("Aitken initial factor must lie in (0,1]")
        self.factor = initial
        self.previous_q: torch.Tensor | None = None
        self.previous_v: torch.Tensor | None = None
        self.history: list[float] = []

    def advance(
        self,
        current_q: wp.array,
        current_v: wp.array,
        target_q: wp.array,
        target_v: wp.array,
        dt: float,
    ) -> tuple[wp.array, wp.array]:
        q = wp.to_torch(current_q)
        v = wp.to_torch(current_v)
        rq = wp.to_torch(target_q) - q
        rv = float(dt) * (wp.to_torch(target_v) - v)
        factor = self.factor
        if self.previous_q is not None and self.previous_v is not None:
            dq = rq - self.previous_q
            dv = rv - self.previous_v
            denominator = torch.sum(dq * dq) + torch.sum(dv * dv)
            if bool((denominator > 0.0).item()):
                numerator = torch.sum(self.previous_q * dq) + torch.sum(
                    self.previous_v * dv
                )
                candidate = torch.clamp(-factor * numerator / denominator, 0.05, 1.0)
                if bool(torch.isfinite(candidate).item()):
                    factor = float(candidate.item())
        self.previous_q = rq.detach().clone()
        self.previous_v = rv.detach().clone()
        self.factor = factor
        self.history.append(factor)
        relaxed_q = q + factor * rq
        relaxed_v = v + (factor / float(dt)) * rv
        return (
            wp.clone(wp.from_torch(relaxed_q, dtype=config.DTYPE, requires_grad=False)),
            wp.clone(wp.from_torch(relaxed_v, dtype=config.DTYPE, requires_grad=False)),
        )


@dataclass(slots=True)
class Q16NativeV5MFSIOwner:
    aerodynamic: Q16NativeV5MOwner
    state: wp.array
    velocity: wp.array
    acceleration: wp.array
    aerodynamic_force: wp.array
    aerodynamic_load: Q16NativeAuthorEndpointLoad
    previous_aerodynamic_load: Q16NativeAuthorEndpointLoad | None = None
    generation: int = 0

    @classmethod
    def initialize(
        cls,
        aerodynamic_solver: Q16NativeV5MSolver,
        state: wp.array,
        velocity: wp.array,
        acceleration: wp.array,
    ) -> "Q16NativeV5MFSIOwner":
        if not (state.shape == velocity.shape == acceleration.shape):
            raise ValueError("native Q16 FSI initial structural shapes differ")
        aero_state = aerodynamic_solver.initialize(state, velocity)
        anchor_load = aerodynamic_solver.author_anchor_load(state, velocity)
        return cls(
            aerodynamic=Q16NativeV5MOwner(aero_state),
            state=_clone(state),
            velocity=_clone(velocity),
            acceleration=_clone(acceleration),
            aerodynamic_force=_clone(anchor_load.constant_generalized_force),
            aerodynamic_load=anchor_load,
            previous_aerodynamic_load=None,
        )


@dataclass(frozen=True, slots=True)
class Q16NativeStructuralCheckpoint:
    substep: int
    state: wp.array
    velocity: wp.array
    acceleration: wp.array


@dataclass(frozen=True, slots=True)
class Q16NativeV5MFSIStepResult:
    structural: Q16StructuralStepResult
    aerodynamic: NativeV5MProposal
    coupling_iterations: int
    aerodynamic_evaluations: int
    residual: float
    residual_history: tuple[float, ...]
    relaxation_history: tuple[float, ...]
    owner_generation: int
    checkpoint: Q16NativeStructuralCheckpoint | None


class Q16NativeV5MFSIStepper:
    """One outer aerodynamic step with immutable-branch fixed-point trials."""

    def __init__(
        self,
        structural_solver: Q16CudaNewmarkStepper,
        aerodynamic_solver: Q16NativeV5MSolver,
        *,
        coupling_tolerance: float = 5.0e-7,
        max_coupling_iterations: int = 20,
        relaxation: float = 0.7,
        persistent_relaxation: bool = False,
    ) -> None:
        if type(structural_solver) is not Q16CudaNewmarkStepper:
            raise TypeError("structural_solver must be the production Q16 CUDA stepper")
        if type(aerodynamic_solver) is not Q16NativeV5MSolver:
            raise TypeError("aerodynamic_solver must be the native V5M solver")
        if structural_solver.device != aerodynamic_solver.surface.quarter_transfer.device:
            raise ValueError("Q16 and native V5M use different CUDA devices")
        if structural_solver.dof_count != aerodynamic_solver.surface.mesh.dof_count:
            raise ValueError("Q16 and native V5M structural DOF owners differ")
        if not 0.0 < coupling_tolerance < 1.0:
            raise ValueError("coupling_tolerance must lie in (0,1)")
        if type(max_coupling_iterations) is not int or max_coupling_iterations < 1:
            raise ValueError("max_coupling_iterations must be positive")
        self.structural_solver = structural_solver
        self.aerodynamic_solver = aerodynamic_solver
        self.coupling_tolerance = float(coupling_tolerance)
        self.max_coupling_iterations = max_coupling_iterations
        self.relaxation = float(relaxation)
        # Carry the learned Aitken factor across outer steps: without it the
        # relaxer relearns the factor from the fixed seed every step and the
        # fixed point pays one extra iteration before the residual collapses.
        # The convergence criterion is unchanged; only the trial path differs.
        self.persistent_relaxation = bool(persistent_relaxation)
        self._learned_relaxation: float | None = None

    def _integrate_structure(
        self,
        owner: Q16NativeV5MFSIOwner,
        aerodynamic_load_end: Q16NativeAuthorEndpointLoad,
        prescribed_forces: tuple[wp.array | None, ...],
        delta_time: float,
        *,
        load_betas: tuple[float, ...],
        checkpoint_substep: int | None,
        coupling_iteration: int,
        formal_replay: bool,
        progress_callback: ProgressCallback | None,
    ) -> tuple[Q16StructuralStepResult, Q16NativeStructuralCheckpoint | None]:
        count = len(prescribed_forces)
        if count < 1:
            raise ValueError("native Q16 FSI requires structural substeps")
        substep_dt = float(delta_time) / count
        state, velocity, acceleration = owner.state, owner.velocity, owner.acceleration
        result: Q16StructuralStepResult | None = None
        totals = {
            "newton": 0,
            "cg": 0,
            "gmres": 0,
            "direct": 0,
            "refresh": 0,
            "fallback": 0,
        }
        residual_max = 0.0
        checkpoint: Q16NativeStructuralCheckpoint | None = None
        replay_started = time.perf_counter()
        previous_load = owner.previous_aerodynamic_load
        # Hoist the substep-constant pieces out of the loop: re-differencing
        # the 71 MB Mf1 matrix on every substep is pure launch overhead.
        anchor_am_matrix = owner.aerodynamic_load.added_mass.generalized_matrix
        if previous_load is not None:
            am_slope = (
                anchor_am_matrix
                - previous_load.added_mass.generalized_matrix
            )
        else:
            endpoint_am_matrix = (
                aerodynamic_load_end.added_mass.generalized_matrix
            )
        for index, prescribed in enumerate(prescribed_forces):
            beta = load_betas[index]
            # Author's forward extrapolation (solve_structure.m [0]-[3]):
            #   Qf(t) = Qf_a + (t - t_fluid)/dt_wake × (Qf - Qf_old)
            # i.e. the constant load starts AT the committed anchor and
            # extrapolates with the committed slope (anchor - previous), not
            # interpolating toward the trial endpoint.  When no previous load
            # exists (first outer step) fall back to endpoint interpolation.
            anchor_constant = wp.to_torch(
                owner.aerodynamic_load.constant_generalized_force
            )
            if previous_load is not None:
                previous_constant = wp.to_torch(
                    previous_load.constant_generalized_force
                )
                constant_t = anchor_constant + beta * (
                    anchor_constant - previous_constant
                )
            else:
                endpoint_constant = wp.to_torch(
                    aerodynamic_load_end.constant_generalized_force
                )
                constant_t = anchor_constant + beta * (
                    endpoint_constant - anchor_constant
                )
            constant = wp.from_torch(
                constant_t, dtype=config.DTYPE, requires_grad=False
            )
            anchor_velocity_force = owner.aerodynamic_load.velocity_force(velocity)
            endpoint_velocity_force = aerodynamic_load_end.velocity_force(velocity)
            velocity_force = _interpolate(
                anchor_velocity_force, endpoint_velocity_force, beta
            )
            aerodynamic = _combine(constant, velocity_force)
            total = aerodynamic if prescribed is None else _combine(aerodynamic, prescribed)
            if previous_load is not None:
                # Extrapolate the added-mass matrix with the same committed
                # slope: Mf1(t) = Mf1_a + beta × (Mf1_a - Mf1_old)
                added_mass = Q16NativeAddedMassAction(
                    anchor_am_matrix + beta * am_slope
                )
            else:
                added_mass = Q16NativeAddedMassAction(
                    torch.lerp(anchor_am_matrix, endpoint_am_matrix, beta)
                )
            predictor = self.structural_solver.step(
                state,
                velocity,
                acceleration,
                total,
                delta_time=substep_dt,
                acceleration_load_action=added_mass,
            )
            average_velocity_t = 0.5 * (
                wp.to_torch(velocity) + wp.to_torch(predictor.velocity)
            )
            average_velocity = wp.from_torch(
                average_velocity_t, dtype=config.DTYPE, requires_grad=False
            )
            anchor_velocity_force = owner.aerodynamic_load.velocity_force(
                average_velocity
            )
            endpoint_velocity_force = aerodynamic_load_end.velocity_force(
                average_velocity
            )
            velocity_force = _interpolate(
                anchor_velocity_force, endpoint_velocity_force, beta
            )
            aerodynamic = _combine(constant, velocity_force)
            total = aerodynamic if prescribed is None else _combine(aerodynamic, prescribed)
            result = self.structural_solver.step(
                state,
                velocity,
                acceleration,
                total,
                delta_time=substep_dt,
                acceleration_load_action=added_mass,
            )
            for subresult in (predictor, result):
                totals["newton"] += subresult.newton_iteration_count
                totals["cg"] += subresult.cg_iteration_count
                totals["gmres"] += subresult.gmres_iteration_count
                totals["direct"] += subresult.direct_solve_count
                totals["refresh"] += subresult.live_tangent_refresh_count
                totals["fallback"] += subresult.indefinite_fallback_count
                residual_max = max(residual_max, subresult.relative_residual_max)
            state, velocity, acceleration = result.state, result.velocity, result.acceleration
            if checkpoint_substep == index + 1:
                checkpoint = Q16NativeStructuralCheckpoint(
                    substep=index + 1,
                    state=_clone(state),
                    velocity=_clone(velocity),
                    acceleration=_clone(acceleration),
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "structural_substep",
                        "coupling_iteration": coupling_iteration,
                        "formal_replay": formal_replay,
                        "substep": index + 1,
                        "substep_count": count,
                        "elapsed_seconds": time.perf_counter() - replay_started,
                        "newton_iterations": totals["newton"],
                        "cg_iterations": totals["cg"],
                        "direct_solves": totals["direct"],
                        "relative_residual_max": residual_max,
                    }
                )
        if result is None:
            raise AssertionError("native Q16 FSI structural schedule was empty")
        structural = Q16StructuralStepResult(
            state=result.state,
            velocity=result.velocity,
            acceleration=result.acceleration,
            reaction=result.reaction,
            delta_time=float(delta_time),
            newton_iteration_count=totals["newton"],
            cg_iteration_count=totals["cg"],
            gmres_iteration_count=totals["gmres"],
            direct_solve_count=totals["direct"],
            live_tangent_refresh_count=totals["refresh"],
            indefinite_fallback_count=totals["fallback"],
            relative_residual_max=residual_max,
        )
        return structural, checkpoint

    def advance(
        self,
        owner: Q16NativeV5MFSIOwner,
        *,
        delta_time: float,
        prescribed_forces: tuple[wp.array | None, ...],
        load_betas: tuple[float, ...] | None = None,
        checkpoint_substep: int | None = None,
        author_startup: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Q16NativeV5MFSIStepResult:
        if type(owner) is not Q16NativeV5MFSIOwner:
            raise TypeError("owner must be the native Q16/V5M owner")
        if not math.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("delta_time must be finite and positive")
        if type(prescribed_forces) is not tuple or not prescribed_forces:
            raise TypeError("prescribed_forces must be a non-empty exact tuple")
        aerodynamic_dt = self.aerodynamic_solver.settings.aerodynamic_dt
        if delta_time != aerodynamic_dt:
            if not (
                author_startup
                and len(prescribed_forces) == 1
                and 0.0 < delta_time < aerodynamic_dt
            ):
                raise ValueError("FSI and native V5M aerodynamic clocks differ")
        elif author_startup:
            raise ValueError("author startup must be shorter than one aerodynamic step")
        count = len(prescribed_forces)
        if load_betas is None:
            betas = tuple(index / count for index in range(count))
        else:
            if type(load_betas) is not tuple or len(load_betas) != count:
                raise ValueError("load_betas must match the structural schedule")
            betas = tuple(float(value) for value in load_betas)
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in betas):
                raise ValueError("load_betas must be finite and lie in [0,1]")
            if any(right < left for left, right in zip(betas[:-1], betas[1:])):
                raise ValueError("load_betas must be nondecreasing")
        if checkpoint_substep is not None and (
            type(checkpoint_substep) is not int
            or not 1 <= checkpoint_substep <= count
        ):
            raise ValueError("checkpoint_substep is outside the structural schedule")
        generation = owner.generation
        parent_digest = owner.aerodynamic.state.digest()
        committed_q = _clone(owner.state)
        committed_v = _clone(owner.velocity)
        committed_a = _clone(owner.acceleration)
        if self.structural_solver.nonsymmetric_solver == "reference_dense":
            refresh_started = time.perf_counter()
            refreshed = self.structural_solver.refresh_reference_tangent(committed_q)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "reference_tangent",
                        "refreshed": refreshed,
                        "refresh_count": (
                            self.structural_solver.reference_tangent_cache_refresh_count
                        ),
                        "elapsed_seconds": time.perf_counter() - refresh_started,
                    }
                )
        q_guess, v_guess = self.structural_solver.predict_kinematics(
            committed_q, committed_v, committed_a, delta_time=delta_time
        )
        residual_history: list[float] = []
        relaxer = _Aitken(
            self._learned_relaxation
            if self.persistent_relaxation and self._learned_relaxation is not None
            else self.relaxation
        )
        evaluations = 0
        for iteration in range(1, self.max_coupling_iterations + 1):
            proposal_started = time.perf_counter()
            proposal = owner.aerodynamic.propose(
                self.aerodynamic_solver, q_guess, v_guess
            )
            evaluations += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "aerodynamic_proposal",
                        "coupling_iteration": iteration,
                        "formal_replay": False,
                        "aerodynamic_evaluations": evaluations,
                        "elapsed_seconds": time.perf_counter() - proposal_started,
                    }
                )
            structural, _ = self._integrate_structure(
                owner,
                proposal.author_load,
                prescribed_forces,
                delta_time,
                load_betas=betas,
                checkpoint_substep=checkpoint_substep,
                coupling_iteration=iteration,
                formal_replay=False,
                progress_callback=progress_callback,
            )
            residual = _relative_error(
                structural.state,
                structural.velocity,
                q_guess,
                v_guess,
                delta_time,
            )
            residual_history.append(residual)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "coupling_residual",
                        "coupling_iteration": iteration,
                        "formal_replay": False,
                        "residual": residual,
                        "tolerance": self.coupling_tolerance,
                    }
                )
            if residual <= self.coupling_tolerance:
                proposal_started = time.perf_counter()
                formal_proposal = owner.aerodynamic.propose(
                    self.aerodynamic_solver, structural.state, structural.velocity
                )
                evaluations += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "aerodynamic_proposal",
                            "coupling_iteration": iteration,
                            "formal_replay": True,
                            "aerodynamic_evaluations": evaluations,
                            "elapsed_seconds": time.perf_counter() - proposal_started,
                        }
                    )
                formal_structural, formal_checkpoint = self._integrate_structure(
                    owner,
                    formal_proposal.author_load,
                    prescribed_forces,
                    delta_time,
                    load_betas=betas,
                    checkpoint_substep=checkpoint_substep,
                    coupling_iteration=iteration,
                    formal_replay=True,
                    progress_callback=progress_callback,
                )
                formal_residual = _relative_error(
                    formal_structural.state,
                    formal_structural.velocity,
                    structural.state,
                    structural.velocity,
                    delta_time,
                )
                residual_history.append(formal_residual)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "coupling_residual",
                            "coupling_iteration": iteration,
                            "formal_replay": True,
                            "residual": formal_residual,
                            "tolerance": self.coupling_tolerance,
                        }
                    )
                if formal_residual <= self.coupling_tolerance:
                    if owner.generation != generation:
                        raise RuntimeError("native Q16 FSI owner generation drift")
                    if owner.aerodynamic.state.digest() != parent_digest:
                        raise RuntimeError("native V5M committed state drift")
                    prepared_q = _clone(formal_structural.state)
                    prepared_v = _clone(formal_structural.velocity)
                    prepared_a = _clone(formal_structural.acceleration)
                    prepared_force = _clone(formal_proposal.generalized_force)
                    owner.aerodynamic.commit(formal_proposal)
                    owner.state = prepared_q
                    owner.velocity = prepared_v
                    owner.acceleration = prepared_a
                    owner.aerodynamic_force = prepared_force
                    owner.previous_aerodynamic_load = owner.aerodynamic_load
                    owner.aerodynamic_load = formal_proposal.author_load
                    owner.generation += 1
                    if self.persistent_relaxation:
                        self._learned_relaxation = relaxer.factor
                    return Q16NativeV5MFSIStepResult(
                        structural=formal_structural,
                        aerodynamic=formal_proposal,
                        coupling_iterations=iteration,
                        aerodynamic_evaluations=evaluations,
                        residual=formal_residual,
                        residual_history=tuple(residual_history),
                        relaxation_history=tuple(relaxer.history),
                        owner_generation=owner.generation,
                        checkpoint=formal_checkpoint,
                    )
                q_guess, v_guess = relaxer.advance(
                    structural.state,
                    structural.velocity,
                    formal_structural.state,
                    formal_structural.velocity,
                    delta_time,
                )
            else:
                q_guess, v_guess = relaxer.advance(
                    q_guess,
                    v_guess,
                    structural.state,
                    structural.velocity,
                    delta_time,
                )
        raise RuntimeError(
            "native Q16/V5M strong coupling did not converge; "
            f"residual_history={residual_history!r}"
        )


__all__ = [
    "NATIVE_Q16_FSI_CONTRACT",
    "Q16NativeStructuralCheckpoint",
    "Q16NativeV5MFSIOwner",
    "Q16NativeV5MFSIStepResult",
    "Q16NativeV5MFSIStepper",
]
