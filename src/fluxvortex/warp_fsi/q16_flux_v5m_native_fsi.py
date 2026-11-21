"""Strong predictor/corrector coupling of Q16 and native FLUX-V5M CUDA."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
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

# DIAGNOSTIC ONLY (D1 degeneracy probe, 2026-11-21): uniformly scale the added-mass
# operator Mf1 = rho P A^-1 N by a constant.  This exists to ask a purely numerical
# question -- how much of the membrane-wing load error is attributable to the Mf1
# double count, holding everything else fixed -- by finite-differencing the map
# (Mf1 scale, E) -> (Cn, zmax/c).  It is NOT a calibration knob: reading it from the
# environment means every run states its value in its command line, and 1.0 returns
# the matrix object unchanged so the production path is bitwise untouched.
_MF1_SCALE = float(os.environ.get("FLUXV_MF1_SCALE", "1.0"))
if not math.isfinite(_MF1_SCALE) or _MF1_SCALE <= 0.0:
    raise ValueError(f"FLUXV_MF1_SCALE must be finite and positive, got {_MF1_SCALE}")


def _scale_added_mass(matrix: torch.Tensor) -> torch.Tensor:
    """Apply :data:`_MF1_SCALE`, returning the SAME object when it is 1.0.

    Returning the same object matters beyond tidiness: the M1b memo and the M1 LU
    cache key on ``id()``/``_version`` of this matrix, and an identity copy would
    silently double the 71 MB matrix traffic per outer step.
    """

    if _MF1_SCALE == 1.0:
        return matrix
    return matrix * _MF1_SCALE


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


class _IQNILS:
    """Interface quasi-Newton with inverse least squares (Degroote 2009).

    Fixed point r(d) = d - S(d) = 0; secant pairs (W=d-diff, V=r-diff)
    across the current step plus a reuse window; update
    d <- d - W (V^T V)^+ V^T r_k.  Falls back to relaxed-Aitken steps when
    the pair history is rank-deficient (QR2-style epsilon filter).
    """

    def __init__(self, dt: float, initial: float, *, max_columns: int = 12, epsilon: float = 1e-2) -> None:
        self.dt = float(dt)
        self.initial = float(initial)
        self.max_columns = max_columns
        self.epsilon = float(epsilon)
        self._w: list[torch.Tensor] = []
        self._v: list[torch.Tensor] = []
        self._prev_d: torch.Tensor | None = None
        self._prev_r: torch.Tensor | None = None
        self.history: list[float] = []

    def _residual(self, d: torch.Tensor, target_q: torch.Tensor, tv_scaled: torch.Tensor) -> torch.Tensor:
        # target concatenation in the same metric as _Aitken (v scaled by dt)
        return d - torch.cat((target_q, tv_scaled))

    def advance(self, current_q, current_v, target_q, target_v, dt: float):
        del dt
        q = wp.to_torch(current_q)
        v = wp.to_torch(current_v)
        tq = wp.to_torch(target_q)
        tv_scaled = self.dt * wp.to_torch(target_v)
        d = torch.cat((q, v * self.dt))
        r = self._residual(d, tq, tv_scaled)
        if self._prev_d is not None:
            dw = d - self._prev_d
            dr = r - self._prev_r
            if float(torch.linalg.vector_norm(dr).item()) > 0.0:
                self._w.append(dw)
                self._v.append(dr)
                if len(self._w) > self.max_columns:
                    self._w.pop(0)
                    self._v.pop(0)
        self._prev_d = d.detach().clone()
        self._prev_r = r.detach().clone()
        updated = None
        if len(self._v) >= 2:
            v_mat = torch.stack(self._v, dim=1)
            w_mat = torch.stack(self._w, dim=1)
            # QR2-style filter: drop columns whose residual-difference norm
            # falls below epsilon * the largest.
            norms = torch.linalg.vector_norm(v_mat, dim=0)
            keep = norms > self.epsilon * float(norms.max().item())
            if bool(keep.any().item()) and int(keep.sum().item()) == keep.numel():
                gram = v_mat.T @ v_mat
                rhs = v_mat.T @ r
                ridge = 1.0e-8 * float(torch.diagonal(gram).abs().max().item())
                gram = gram + ridge * torch.eye(
                    gram.shape[0], device=gram.device, dtype=torch.float64
                )
                try:
                    alpha = torch.linalg.solve(gram, rhs)
                    delta = -(w_mat @ alpha)
                    nd = float(torch.linalg.vector_norm(delta).item())
                    nr = float(torch.linalg.vector_norm(r).item())
                    if math.isfinite(nd) and nd <= 1.2 * max(nr, 1.0e-300):
                        updated = d + delta
                        self.history.append(1.0)
                except Exception:
                    updated = None
        if updated is None:
            updated = d + r
            self.history.append(1.0)
        out_q = updated[: q.shape[0]]
        out_v = updated[q.shape[0] :] / self.dt
        return (
            wp.clone(wp.from_torch(out_q, dtype=config.DTYPE, requires_grad=False)),
            wp.clone(wp.from_torch(out_v, dtype=config.DTYPE, requires_grad=False)),
        )


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
        coupling_accelerator: str = "aitken",
        bound_rate_author_weak: bool = False,
        three_slot_mf1_cancel: bool = False,
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
        if coupling_accelerator not in {"aitken", "iqn_ils"}:
            raise ValueError("coupling_accelerator must be 'aitken' or 'iqn_ils'")
        self.coupling_accelerator = coupling_accelerator
        # Stage 2 (2026-11-21, default OFF): reproduce the AUTHOR'S WEAK scheme
        # exactly.  Under wake_history_mode="bound_rate" the constant slot already
        # carries dp_add = rho (Gamma_n - Gamma_{n-1})/dt, i.e. the full one-step
        # unsteady circulation derivative, whose acceleration content is exactly
        # what Mf1/Mf2_1 represent; the author's weak scheme therefore sets
        # Mf1 = Mf2_1 = 0 (calc_fluid_force_weak.m:96-131).  This stepper has no
        # mode branch and adds both unconditionally, so on the bound_rate path the
        # acceleration term is applied twice.  Recorded as unresolved in-tree:
        # diagnostics/README.md:27-33, RESEARCH_BRIEF.md M3, REPAIR...:70 (U3),
        # P1_SHARED_EDGE...:171-175.  Default False keeps the production path
        # bit-identical; the flag only exists so the two contracts can be measured
        # against external ground truth before anyone changes a default.
        if type(bound_rate_author_weak) is not bool:
            raise TypeError("bound_rate_author_weak must be an exact bool")
        self.bound_rate_author_weak = bool(bound_rate_author_weak)
        if self.bound_rate_author_weak and (
            aerodynamic_solver.settings.wake_history_mode != "bound_rate"
        ):
            raise ValueError(
                "bound_rate_author_weak only applies to wake_history_mode='bound_rate'"
            )
        self._zero_added_mass: Q16NativeAddedMassAction | None = None
        self._added_mass_memo: tuple | None = None
        #: S1 (2026-11-22): (id(anchor), id(previous)) -> the committed added-mass
        #: slope.  Memoized so the M1b action memo and the LU cache see stable ids.
        self._am_slope_memo: tuple | None = None
        # Stage 2b (2026-11-21, default OFF): the THREE-SLOT form.  Under
        # bound_rate the constant slot carries dp_add = rho dGamma/dt, whose
        # acceleration content IS Mf1 qddot; the stepper then also adds Mf1
        # implicitly.  Rather than deleting Mf1 (which removes the stabilising
        # term from the effective tangent and goes unstable -- measured: 0.17%
        # step-1 error growing 1.34 -> 6.62 -> 10.52% and then a Newton line
        # search failure), SUBTRACT Mf1 qddot from the load and KEEP Mf1 in the
        # tangent.  At convergence the two cancel in the load while Newton keeps
        # the added mass, which is the author's strong-scheme structure and the
        # external audit's "three-slot" prescription.
        if type(three_slot_mf1_cancel) is not bool:
            raise TypeError("three_slot_mf1_cancel must be an exact bool")
        self.three_slot_mf1_cancel = bool(three_slot_mf1_cancel)
        if self.three_slot_mf1_cancel and self.bound_rate_author_weak:
            raise ValueError(
                "three_slot_mf1_cancel and bound_rate_author_weak are different "
                "contracts and must not be combined"
            )
        if self.three_slot_mf1_cancel and (
            aerodynamic_solver.settings.wake_history_mode != "bound_rate"
        ):
            raise ValueError(
                "three_slot_mf1_cancel only applies to wake_history_mode='bound_rate'"
            )

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
        rhs_observer: list | None = None,
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
        anchor_am_matrix = _scale_added_mass(
            owner.aerodynamic_load.added_mass.generalized_matrix
        )
        if previous_load is not None:
            previous_am_matrix = _scale_added_mass(
                previous_load.added_mass.generalized_matrix
            )
            # S1 (2026-11-22): the slope must be a STABLE OBJECT, not a fresh tensor
            # per call.  `_integrate_structure` runs once per coupling iteration plus
            # once for the formal replay, all with the same committed anchor and
            # previous loads; when the slope was re-differenced every call its id()
            # changed, which invalidated the M1b added-mass memo (keyed on id(slope)),
            # which rebuilt ten actions with ten new data_ptrs, which made the LU
            # cache miss on every pass.  Measured consequence: 29.00 factorizations
            # per step (= 2.9 passes x 10 betas) instead of 10.
            slope_key = (id(anchor_am_matrix), id(previous_am_matrix))
            memo_slope = self._am_slope_memo
            if memo_slope is not None and memo_slope[0] == slope_key:
                am_slope = memo_slope[1]
            else:
                am_slope = anchor_am_matrix - previous_am_matrix
                # Hold both operands so their ids cannot be recycled while memoized.
                self._am_slope_memo = (
                    slope_key,
                    am_slope,
                    anchor_am_matrix,
                    previous_am_matrix,
                )
        else:
            endpoint_am_matrix = _scale_added_mass(
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
            if self.three_slot_mf1_cancel:
                # M4 (2026-11-21): subtract ONE Mf1 qddot at the OUTER-STEP
                # ENDPOINTS, before the beta interpolation/extrapolation.  The
                # previous version subtracted after interpolation and used the
                # substep-initial acceleration (not beta-scaled on the first step),
                # so the subtracted term carried substep-lag noise and was
                # over-subtracted on substep 1.
                #
                # Deviation from the auditor's formula, stated rather than hidden:
                # the owner retains only previous_aerodynamic_load (no
                # previous_acceleration), so the ANCHOR acceleration is used for
                # every endpoint rather than each endpoint's own acceleration.
                acceleration_anchor = wp.to_torch(acceleration)[0]
                anchor_constant = anchor_constant - (
                    anchor_am_matrix @ acceleration_anchor
                ).unsqueeze(0)
            if previous_load is not None:
                previous_constant = wp.to_torch(
                    previous_load.constant_generalized_force
                )
                if self.three_slot_mf1_cancel:
                    previous_constant = previous_constant - (
                        _scale_added_mass(
                            previous_load.added_mass.generalized_matrix
                        )
                        @ acceleration_anchor
                    ).unsqueeze(0)
                constant_t = anchor_constant + beta * (
                    anchor_constant - previous_constant
                )
            else:
                endpoint_constant = wp.to_torch(
                    aerodynamic_load_end.constant_generalized_force
                )
                if self.three_slot_mf1_cancel:
                    endpoint_constant = endpoint_constant - (
                        _scale_added_mass(
                            aerodynamic_load_end.added_mass.generalized_matrix
                        )
                        @ acceleration_anchor
                    ).unsqueeze(0)
                constant_t = anchor_constant + beta * (
                    endpoint_constant - anchor_constant
                )
            constant = wp.from_torch(
                constant_t, dtype=config.DTYPE, requires_grad=False
            )
            include_mf21 = not self.bound_rate_author_weak
            anchor_velocity_force = owner.aerodynamic_load.velocity_force(
                velocity, include_mf21=include_mf21
            )
            endpoint_velocity_force = aerodynamic_load_end.velocity_force(
                velocity, include_mf21=include_mf21
            )
            velocity_force = _interpolate(
                anchor_velocity_force, endpoint_velocity_force, beta
            )
            aerodynamic = _combine(constant, velocity_force)
            total = aerodynamic if prescribed is None else _combine(aerodynamic, prescribed)
            # M1b (2026-11-21): memoize the added-mass action per load fraction beta
            # so every call at the same beta shares ONE matrix object.  The LU cache in
            # the structural solver keys on that matrix.  MEASURED (S1, 2026-11-22):
            # with this memo but the solver's OLD single-slot cache, the 30-step A16
            # window still factorized 29.00 times per step (870/30) although only 10
            # load fractions exist -- the single slot was evicted on every wrap of the
            # beta loop.  With the per-beta bucket it is 10.
            memo = self._added_mass_memo
            # Every matrix the action is built from must be in the key: on the FIRST
            # outer step the action is torch.lerp(anchor, endpoint, beta) and
            # endpoint_am_matrix changes per coupling iteration, so omitting it
            # reused a stale action (caught by the bitwise check, 3 steps).
            used_slope = am_slope if previous_load is not None else None
            used_endpoint = endpoint_am_matrix if previous_load is None else None
            anchor_pair = (
                id(anchor_am_matrix),
                id(used_slope) if used_slope is not None else None,
                id(used_endpoint) if used_endpoint is not None else None,
            )
            if memo is None or memo[0] != anchor_pair:
                # Hold the matrices themselves so their ids cannot be recycled while
                # the memo is live (the same hazard that broke M1's first version).
                memo = (
                    anchor_pair,
                    anchor_am_matrix,
                    used_slope,
                    used_endpoint,
                    {},
                )
                self._added_mass_memo = memo
            by_beta = memo[4]
            added_mass = by_beta.get(beta)
            if added_mass is None:
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
                by_beta[beta] = added_mass
            if self.bound_rate_author_weak:
                # Author weak scheme: Mf1 == 0.  Reuse one zero matrix instead of
                # allocating 71 MB per substep.
                if self._zero_added_mass is None:
                    self._zero_added_mass = Q16NativeAddedMassAction(
                        torch.zeros_like(added_mass.generalized_matrix)
                    )
                added_mass = self._zero_added_mass
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
                average_velocity, include_mf21=not self.bound_rate_author_weak
            )
            endpoint_velocity_force = aerodynamic_load_end.velocity_force(
                average_velocity, include_mf21=not self.bound_rate_author_weak
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
            if rhs_observer is not None:
                # E0 production-RHS observer: READ-ONLY audit of the
                # corrector pass.  Records the actual decomposition the
                # structure consumed, the accepted-increment algorithmic
                # work, and the predictor-lag diagnostic (fixed-point
                # diagnostic only, never added back into totals).
                q_before = wp.to_torch(state)[0]
                dq = wp.to_torch(result.state)[0] - q_before
                constant_t = wp.to_torch(constant)[0]
                velocity_t = wp.to_torch(velocity_force)[0]
                aero_t = wp.to_torch(aerodynamic)[0]
                acc_t = wp.to_torch(result.acceleration)[0]
                v_avg_acc_t = 0.5 * (wp.to_torch(velocity)[0] + wp.to_torch(result.velocity)[0])
                v_avg_acc = wp.from_torch(
                    v_avg_acc_t.unsqueeze(0).contiguous(),
                    dtype=config.DTYPE,
                    requires_grad=False,
                )
                vf_acc_anchor = owner.aerodynamic_load.velocity_force(
                    v_avg_acc, include_mf21=not self.bound_rate_author_weak
                )
                vf_acc_end = aerodynamic_load_end.velocity_force(
                    v_avg_acc, include_mf21=not self.bound_rate_author_weak
                )
                vf_acc_t = wp.to_torch(
                    _interpolate(vf_acc_anchor, vf_acc_end, beta)
                )[0]
                mf1_action_t = added_mass.generalized_matrix @ acc_t
                w_nonacc = float(torch.dot(aero_t, dq).item())
                w_acc = float(torch.dot(mf1_action_t, dq).item())
                rhs_observer.append(
                    {
                        "coupling_iteration": int(coupling_iteration),
                        "formal_replay": bool(formal_replay),
                        "substep": index + 1,
                        "constant_norm": float(constant_t.norm().item()),
                        "velocity_norm": float(velocity_t.norm().item()),
                        "mf1_action_norm": float(mf1_action_t.norm().item()),
                        "total_aero_norm": float(aero_t.norm().item()),
                        "dq_norm": float(dq.norm().item()),
                        # P0-b: the structure consumed total = nonacc + acc;
                        # all three work components recorded separately.
                        "w_nonaccelerative": w_nonacc,
                        "w_accelerative": w_acc,
                        "w_algorithmic": w_nonacc + w_acc,
                        "dw_predictor_lag": float(
                            torch.dot(velocity_t - vf_acc_t, dq).item()
                        ),
                        "identity": (
                            "formal_committed"
                            if formal_replay
                            else "nonformal_causal"
                        ),
                    }
                )
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
        rhs_observer: list | None = None,
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
        parent_fingerprint = owner.aerodynamic.state.mutation_fingerprint()
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
        relaxer = (
            _IQNILS(delta_time, self.relaxation)
            if self.coupling_accelerator == "iqn_ils"
            else _Aitken(
                self._learned_relaxation
                if self.persistent_relaxation and self._learned_relaxation is not None
                else self.relaxation
            )
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
                rhs_observer=rhs_observer,
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
                    rhs_observer=rhs_observer,
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
                    if (
                        owner.aerodynamic.state.mutation_fingerprint()
                        != parent_fingerprint
                    ):
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
                    if self.persistent_relaxation and hasattr(relaxer, "factor"):
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
