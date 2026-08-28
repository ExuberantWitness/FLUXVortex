"""GPU-only matrix-free Newmark--Newton--CG trial for projected Q16 shells."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import torch
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16MITC16EASMesh
from fluxvortex.q16_boundary_constraints import Q16BoundaryConstraints

from . import config
from .kernels_q16_constraints import Q16CudaBoundaryConstraints
from .kernels_q16_mesh import (
    Q16CudaMITC16EASMeshOperator,
    _require_global_state,
)
from .kernels_q16_ans_eas import (
    PROJECTED_B_SLOT_COUNT,
    Q16_EAS_QUADRATURE_POINT_COUNT,
)

DTYPE = config.DTYPE
AccelerationLoadAction = Callable[[wp.array], wp.array]


@wp.kernel
def _q16_newmark_predict_kernel(
    state: wp.array(dtype=DTYPE, ndim=2),
    velocity: wp.array(dtype=DTYPE, ndim=2),
    acceleration: wp.array(dtype=DTYPE, ndim=2),
    delta_time: DTYPE,
    beta: DTYPE,
    gamma: DTYPE,
    state_predictor: wp.array(dtype=DTYPE, ndim=2),
    velocity_predictor: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    state_predictor[batch, dof] = (
        state[batch, dof]
        + delta_time * velocity[batch, dof]
        + delta_time * delta_time * (DTYPE(0.5) - beta) * acceleration[batch, dof]
    )
    velocity_predictor[batch, dof] = (
        velocity[batch, dof]
        + delta_time * (DTYPE(1.0) - gamma) * acceleration[batch, dof]
    )


@wp.kernel
def _q16_newmark_kinematics_kernel(
    state: wp.array(dtype=DTYPE, ndim=2),
    state_predictor: wp.array(dtype=DTYPE, ndim=2),
    velocity_predictor: wp.array(dtype=DTYPE, ndim=2),
    inverse_beta_dt2: DTYPE,
    gamma_dt: DTYPE,
    acceleration: wp.array(dtype=DTYPE, ndim=2),
    velocity: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    value = inverse_beta_dt2 * (state[batch, dof] - state_predictor[batch, dof])
    acceleration[batch, dof] = value
    velocity[batch, dof] = velocity_predictor[batch, dof] + gamma_dt * value


@wp.kernel
def _q16_residual_kernel(
    mass_action: wp.array(dtype=DTYPE, ndim=2),
    mass_velocity: wp.array(dtype=DTYPE, ndim=2),
    mass_damping_coefficient: DTYPE,
    stiffness_damping_force: wp.array(dtype=DTYPE, ndim=2),
    internal_force: wp.array(dtype=DTYPE, ndim=2),
    reference_internal_force: wp.array(dtype=DTYPE, ndim=2),
    external_force: wp.array(dtype=DTYPE, ndim=2),
    residual: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    residual[batch, dof] = (
        mass_action[batch, dof]
        + mass_damping_coefficient * mass_velocity[batch, dof]
        + stiffness_damping_force[batch, dof]
        + internal_force[batch, dof]
        - reference_internal_force[0, dof]
        - external_force[batch, dof]
    )


@wp.kernel
def _q16_effective_action_kernel(
    mass_action: wp.array(dtype=DTYPE, ndim=2),
    tangent_action: wp.array(dtype=DTYPE, ndim=2),
    effective_mass_coefficient: DTYPE,
    result: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    result[batch, dof] = (
        effective_mass_coefficient * mass_action[batch, dof]
        + tangent_action[batch, dof]
    )


@wp.kernel
def _q16_effective_diagonal_kernel(
    mass_diagonal: wp.array(dtype=DTYPE, ndim=2),
    material_diagonal: wp.array(dtype=DTYPE, ndim=2),
    effective_mass_coefficient: DTYPE,
    effective_diagonal: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    effective_diagonal[batch, dof] = (
        effective_mass_coefficient * mass_diagonal[batch, dof]
        + material_diagonal[batch, dof]
    )


@wp.kernel
def _q16_invert_jacobi_kernel(
    diagonal: wp.array(dtype=DTYPE, ndim=2),
    inverse: wp.array(dtype=DTYPE, ndim=2),
    failure: wp.array(dtype=wp.int32, ndim=1),
):
    batch, dof = wp.tid()
    value = diagonal[batch, dof]
    if not wp.isfinite(value) or value <= DTYPE(0.0):
        inverse[batch, dof] = DTYPE(0.0)
        wp.atomic_max(failure, 0, 1)
    else:
        reciprocal = DTYPE(1.0) / value
        if not wp.isfinite(reciprocal) or reciprocal <= DTYPE(0.0):
            inverse[batch, dof] = DTYPE(0.0)
            wp.atomic_max(failure, 0, 1)
        else:
            inverse[batch, dof] = reciprocal


@wp.kernel
def _q16_apply_jacobi_kernel(
    inverse: wp.array(dtype=DTYPE, ndim=2),
    residual: wp.array(dtype=DTYPE, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=1),
    result: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    if active[batch] == 1:
        result[batch, dof] = inverse[batch, dof] * residual[batch, dof]
    else:
        result[batch, dof] = DTYPE(0.0)


@wp.kernel
def _q16_masked_negative_kernel(
    value: wp.array(dtype=DTYPE, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=1),
    result: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    if active[batch] == 1:
        result[batch, dof] = -value[batch, dof]
    else:
        result[batch, dof] = DTYPE(0.0)


@wp.kernel
def _q16_dot_kernel(
    left: wp.array(dtype=DTYPE, ndim=2),
    right: wp.array(dtype=DTYPE, ndim=2),
    dof_count: int,
    result: wp.array(dtype=DTYPE, ndim=1),
):
    batch = wp.tid()
    value = DTYPE(0.0)
    for dof in range(dof_count):
        value = value + left[batch, dof] * right[batch, dof]
    result[batch] = value


@wp.kernel
def _q16_cg_initialize_active_kernel(
    rhs_norm_squared: wp.array(dtype=DTYPE, ndim=1),
    active: wp.array(dtype=wp.int32, ndim=1),
):
    batch = wp.tid()
    if rhs_norm_squared[batch] > DTYPE(0.0):
        active[batch] = 1
    else:
        active[batch] = 0


@wp.kernel
def _q16_cg_alpha_kernel(
    residual_preconditioned: wp.array(dtype=DTYPE, ndim=1),
    direction_action: wp.array(dtype=DTYPE, ndim=1),
    active: wp.array(dtype=wp.int32, ndim=1),
    alpha: wp.array(dtype=DTYPE, ndim=1),
    failure: wp.array(dtype=wp.int32, ndim=1),
):
    batch = wp.tid()
    denominator = direction_action[batch]
    numerator = residual_preconditioned[batch]
    if active[batch] == 0:
        alpha[batch] = DTYPE(0.0)
    elif (
        not wp.isfinite(denominator)
        or not wp.isfinite(numerator)
        or denominator <= DTYPE(0.0)
        or numerator < DTYPE(0.0)
    ):
        alpha[batch] = DTYPE(0.0)
        wp.atomic_max(failure, 0, 1)
    else:
        alpha[batch] = numerator / denominator


@wp.kernel
def _q16_cg_update_kernel(
    solution: wp.array(dtype=DTYPE, ndim=2),
    residual: wp.array(dtype=DTYPE, ndim=2),
    direction: wp.array(dtype=DTYPE, ndim=2),
    action: wp.array(dtype=DTYPE, ndim=2),
    alpha: wp.array(dtype=DTYPE, ndim=1),
    active: wp.array(dtype=wp.int32, ndim=1),
):
    batch, dof = wp.tid()
    if active[batch] == 1:
        coefficient = alpha[batch]
        solution[batch, dof] = (
            solution[batch, dof] + coefficient * direction[batch, dof]
        )
        residual[batch, dof] = residual[batch, dof] - coefficient * action[batch, dof]


@wp.kernel
def _q16_cg_mark_converged_kernel(
    residual_norm_squared: wp.array(dtype=DTYPE, ndim=1),
    rhs_norm_squared: wp.array(dtype=DTYPE, ndim=1),
    tolerance_squared: DTYPE,
    active: wp.array(dtype=wp.int32, ndim=1),
    failure: wp.array(dtype=wp.int32, ndim=1),
):
    batch = wp.tid()
    residual_value = residual_norm_squared[batch]
    rhs_value = rhs_norm_squared[batch]
    if not wp.isfinite(residual_value) or residual_value < DTYPE(0.0):
        wp.atomic_max(failure, 0, 1)
    elif residual_value <= tolerance_squared * rhs_value:
        active[batch] = 0


@wp.kernel
def _q16_cg_direction_kernel(
    direction: wp.array(dtype=DTYPE, ndim=2),
    preconditioned_residual: wp.array(dtype=DTYPE, ndim=2),
    residual_preconditioned: wp.array(dtype=DTYPE, ndim=1),
    previous_residual_preconditioned: wp.array(dtype=DTYPE, ndim=1),
    active: wp.array(dtype=wp.int32, ndim=1),
    failure: wp.array(dtype=wp.int32, ndim=1),
):
    batch, dof = wp.tid()
    denominator = previous_residual_preconditioned[batch]
    numerator = residual_preconditioned[batch]
    if active[batch] == 0:
        direction[batch, dof] = DTYPE(0.0)
    elif denominator <= DTYPE(0.0) or not wp.isfinite(denominator):
        direction[batch, dof] = DTYPE(0.0)
        wp.atomic_max(failure, 0, 1)
    else:
        direction[batch, dof] = (
            preconditioned_residual[batch, dof]
            + (numerator / denominator) * direction[batch, dof]
        )


@wp.kernel
def _q16_state_trial_kernel(
    state: wp.array(dtype=DTYPE, ndim=2),
    increment: wp.array(dtype=DTYPE, ndim=2),
    factor: DTYPE,
    active: wp.array(dtype=wp.int32, ndim=1),
    trial: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    if active[batch] == 1:
        trial[batch, dof] = state[batch, dof] + factor * increment[batch, dof]
    else:
        trial[batch, dof] = state[batch, dof]


@wp.kernel
def _q16_predictor_trial_kernel(
    state: wp.array(dtype=DTYPE, ndim=2),
    predictor: wp.array(dtype=DTYPE, ndim=2),
    factor: DTYPE,
    trial: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    trial[batch, dof] = state[batch, dof] + factor * (
        predictor[batch, dof] - state[batch, dof]
    )


class Q16StructuralStepStopped(RuntimeError):
    """A structural trial failed before publishing a result."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        newton_iteration_count: int,
        cg_iteration_count: int,
        relative_residual_max: float,
        gmres_iteration_count: int = 0,
        newton_residual_history: tuple[float, ...] = (),
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.newton_iteration_count = newton_iteration_count
        self.cg_iteration_count = cg_iteration_count
        self.gmres_iteration_count = gmres_iteration_count
        self.relative_residual_max = relative_residual_max
        self.newton_residual_history = newton_residual_history


@dataclass(frozen=True, slots=True, eq=False)
class Q16StructuralStepResult:
    state: wp.array
    velocity: wp.array
    acceleration: wp.array
    reaction: wp.array
    delta_time: float
    newton_iteration_count: int
    cg_iteration_count: int
    gmres_iteration_count: int
    direct_solve_count: int
    live_tangent_refresh_count: int
    indefinite_fallback_count: int
    relative_residual_max: float


@dataclass(frozen=True, slots=True, eq=False)
class Q16StructuralWorkBalance:
    """CUDA endpoint-work identity for one accepted Newmark step.

    The internal and external terms are endpoint trapezoidal work, not an
    assertion that the nonlinear strain energy was integrated exactly along
    the step.  The start external force is reconstructed from the committed
    start-state equilibrium; this makes the balance an audit of the accepted
    endpoint equations, Newmark kinematics and load transfer.
    """

    kinetic_energy_start: float
    kinetic_energy_end: float
    kinetic_energy_change: float
    internal_trapezoidal_work: float
    damping_trapezoidal_work: float
    external_trapezoidal_work: float
    balance_residual: float
    relative_balance_residual: float
    state_increment_norm: float
    deformation_norm_end: float
    velocity_norm_end: float
    acceleration_norm_end: float


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class Q16CudaNewmarkStepper:
    """Batched CUDA Newmark trial with matrix-free Newton--CG corrections."""

    __slots__ = (
        "_boundary_operator",
        "_free_dof_indices_torch",
        "_mass_matrix_torch",
        "_operator",
        "_reference_internal_force",
        "_reference_state",
        "_reference_tangent_anchor_torch",
        "_reference_tangent_matrix_torch",
        "beta",
        "cg_check_every",
        "cg_tolerance",
        "device",
        "dof_count",
        "gamma",
        "mass_damping_coefficient",
        "max_cg_iterations",
        "max_newton_iterations",
        "newton_tolerance",
        "nonsymmetric_solver",
        "preconditioner",
        "reference_dense_refresh_after",
        "reference_tangent_cache_refresh_count",
        "reference_tangent_refresh_rtol",
        "stiffness_damping_coefficient",
    )

    def __init__(
        self,
        model: Q16MITC16EASMesh,
        boundary: Q16BoundaryConstraints,
        *,
        device: str,
        newton_tolerance: float,
        max_newton_iterations: int,
        cg_tolerance: float,
        max_cg_iterations: int,
        cg_check_every: int,
        preconditioner: str = "material_jacobi",
        nonsymmetric_solver: str = "direct",
        reference_dense_refresh_after: int = 32,
        mass_damping_coefficient: float = 0.0,
        # Accelerator-cadence knob (0.0 = the legacy always-refresh-when-the-
        # anchor-moved behavior).  The reference tangent is a quasi-Newton
        # accelerator only — the live nonlinear residual owns acceptance — so
        # re-assembly is skipped while the committed state has drifted less
        # than this relative amount from the cached anchor.  Newton converges
        # to the same frozen tolerance either way; only the iteration count
        # changes.  Callers must label non-zero values in their manifests.
        reference_tangent_refresh_rtol: float = 0.0,
        stiffness_damping_coefficient: float = 0.0,
    ) -> None:
        if type(model) is not Q16MITC16EASMesh:
            raise TypeError("model must be an exact Q16MITC16EASMesh")
        if type(boundary) is not Q16BoundaryConstraints:
            raise TypeError("boundary must be an exact Q16BoundaryConstraints")
        if boundary.mesh is not model.mesh:
            raise ValueError("boundary and structural model must share one mesh owner")
        self.beta = 0.25
        self.gamma = 0.5
        self.newton_tolerance = _positive_float("newton_tolerance", newton_tolerance)
        if self.newton_tolerance >= 1.0:
            raise ValueError("newton_tolerance must be less than one")
        self.max_newton_iterations = _positive_int(
            "max_newton_iterations", max_newton_iterations
        )
        self.cg_tolerance = _positive_float("cg_tolerance", cg_tolerance)
        if self.cg_tolerance >= 1.0:
            raise ValueError("cg_tolerance must be less than one")
        self.max_cg_iterations = _positive_int("max_cg_iterations", max_cg_iterations)
        self.cg_check_every = _positive_int("cg_check_every", cg_check_every)
        if type(preconditioner) is not str or preconditioner not in {
            "material_jacobi",
            "none",
        }:
            raise ValueError("preconditioner must be 'material_jacobi' or 'none'")
        self.preconditioner = preconditioner
        if type(nonsymmetric_solver) is not str or nonsymmetric_solver not in {
            "direct",
            "gmres",
            "reference_dense",
        }:
            raise ValueError(
                "nonsymmetric_solver must be 'direct', 'gmres', or "
                "'reference_dense'"
            )
        self.nonsymmetric_solver = nonsymmetric_solver
        self.reference_dense_refresh_after = _positive_int(
            "reference_dense_refresh_after", reference_dense_refresh_after
        )
        if (
            not math.isfinite(reference_tangent_refresh_rtol)
            or reference_tangent_refresh_rtol < 0.0
        ):
            raise ValueError(
                "reference_tangent_refresh_rtol must be finite and >= 0"
            )
        self.reference_tangent_refresh_rtol = float(
            reference_tangent_refresh_rtol
        )
        self.mass_damping_coefficient = _nonnegative_float(
            "mass_damping_coefficient", mass_damping_coefficient
        )
        # Kelvin-Voigt stiffness-proportional damping theta*K_ref (the
        # author's MATLAB theta_a*(Qd_eps + Qd_k) form).  The damping force
        # and the Newmark-effective contribution reuse the cached reference
        # tangent, so this requires the reference-dense linear path.
        self.stiffness_damping_coefficient = _nonnegative_float(
            "stiffness_damping_coefficient", stiffness_damping_coefficient
        )
        if self.stiffness_damping_coefficient > 0.0 and (
            self.nonsymmetric_solver != "reference_dense"
        ):
            raise ValueError(
                "stiffness_damping_coefficient requires the reference-dense solver"
            )
        self._operator = Q16CudaMITC16EASMeshOperator(model, device=device)
        self.device = self._operator.device
        self.dof_count = model.mesh.dof_count
        self._boundary_operator = Q16CudaBoundaryConstraints(
            boundary, device=self._operator.device
        )
        constrained = wp.to_torch(self._boundary_operator._constrained_mask).bool()
        self._free_dof_indices_torch = torch.nonzero(
            ~constrained, as_tuple=False
        ).flatten()
        # Keep the fixed consistent mass operator as a dense CUDA matrix for
        # the UVLM added-mass preconditioner.  Build it in bounded batches so
        # a later 5x3 Q16 case does not materialize a dof_count-sized element
        # work batch.  This is a preconditioner only; the accepted residual is
        # still evaluated with the production matrix-free operator.
        mass_matrix = torch.empty(
            (self.dof_count, self.dof_count),
            device=self._operator.device,
            dtype=torch.float64,
        )
        mass_batch_size = min(64, self.dof_count)
        for start in range(0, self.dof_count, mass_batch_size):
            stop = min(start + mass_batch_size, self.dof_count)
            directions = torch.zeros(
                (stop - start, self.dof_count),
                device=self._operator.device,
                dtype=torch.float64,
            )
            rows = torch.arange(stop - start, device=self._operator.device)
            directions[rows, rows + start] = 1.0
            action = wp.to_torch(
                self._operator._mass_action_prechecked(
                    wp.from_torch(directions, dtype=DTYPE, requires_grad=False)
                )
            )
            mass_matrix[:, start:stop] = action.transpose(0, 1)
        if not bool(torch.isfinite(mass_matrix).all().item()):
            raise FloatingPointError("Q16 dense CUDA mass preconditioner is invalid")
        self._mass_matrix_torch = mass_matrix
        reference = wp.array(
            np.ascontiguousarray(model.mesh.reference_state[None, :]),
            dtype=DTYPE,
            device=self._operator.device,
        )
        # StVK stress is exactly zero in the reference configuration.  The
        # projected ANS/EAS quadrature leaves a stiffness-scaled roundoff
        # remainder at high E; freeze and subtract that discrete reference
        # remainder so the implemented residual preserves the constitutive
        # stress-free state instead of treating roundoff as a physical load.
        self._reference_internal_force = self._operator._internal_force_prechecked(
            reference
        )
        self._reference_state = wp.clone(reference)
        self._reference_tangent_anchor_torch = None
        self._reference_tangent_matrix_torch = None
        self.reference_tangent_cache_refresh_count = 0
        if self.nonsymmetric_solver == "reference_dense":
            self.refresh_reference_tangent(reference)

    def refresh_reference_tangent(self, state: wp.array) -> bool:
        """Refresh the CUDA quasi-Newton cache at one committed outer state.

        The cache is a convergence accelerator only.  It is refreshed before
        an outer FSI transaction and remains immutable throughout all trial
        branches; the live nonlinear residual still owns acceptance.
        """

        _require_global_state(
            "reference tangent anchor",
            state,
            device=self.device,
            dof_count=self.dof_count,
        )
        if state.shape[0] != 1:
            raise ValueError("reference tangent cache requires one FSI sample")
        state_t = wp.to_torch(state)
        if self._reference_tangent_anchor_torch is not None:
            anchor = self._reference_tangent_anchor_torch
            if bool(torch.equal(state_t, anchor)):
                return False
            if self.reference_tangent_refresh_rtol > 0.0:
                # Accelerator cadence: skip the (expensive) dense re-assembly
                # while the committed state has drifted less than the labeled
                # relative tolerance from the cached anchor.  Newton still
                # converges to the same frozen tolerance against the live
                # nonlinear residual; only the iteration count can change.
                drift = float(
                    (state_t - anchor).abs().max().item()
                )
                scale = max(
                    float(anchor.abs().max().item()),
                    1.0e-30,
                )
                if drift <= self.reference_tangent_refresh_rtol * scale:
                    return False
        linearization = self._operator._linearization_prechecked(state)
        tangent_matrix = torch.empty(
            (self.dof_count, self.dof_count),
            device=self.device,
            dtype=torch.float64,
        )
        tangent_batch_size = min(64, self.dof_count)
        # Per-batch EAS projected-B workspace cap.  Column entries are
        # batch-independent, so the cached tangent is bit-identical for any
        # batch size; only the launch/conversion overhead changes.  The
        # historical 256 MB cap forced ~10-direction batches (298 warp
        # launch+sync rounds per refresh on the 5x10 membrane, ~4 s/step of
        # pure overhead); 2 GB keeps the worst transient ~1.6 GB on the
        # formal meshes and cuts the rounds by ~6x.  Reduce only if warp-pool
        # fragmentation reappears on much larger meshes.
        workspace_per_direction = (
            self._operator.element_count
            * Q16_EAS_QUADRATURE_POINT_COUNT
            * PROJECTED_B_SLOT_COUNT
            * 8
        )
        if workspace_per_direction > 0:
            budget_directions = max(1, 2_147_483_648 // workspace_per_direction)
            tangent_batch_size = min(tangent_batch_size, budget_directions)
        for start in range(0, self.dof_count, tangent_batch_size):
            stop = min(start + tangent_batch_size, self.dof_count)
            count = stop - start
            directions = torch.zeros(
                (count, self.dof_count),
                device=self.device,
                dtype=torch.float64,
            )
            rows = torch.arange(count, device=self.device)
            directions[rows, rows + start] = 1.0
            repeated_linearization = tuple(
                wp.from_torch(
                    wp.to_torch(value)
                    .expand((count,) + tuple(value.shape[1:]))
                    .contiguous(),
                    dtype=DTYPE,
                    requires_grad=False,
                )
                for value in linearization
            )
            action = wp.to_torch(
                self._operator._tangent_action_linearized_prechecked(
                    wp.from_torch(
                        directions,
                        dtype=DTYPE,
                        requires_grad=False,
                    ),
                    repeated_linearization,
                )
            )
            tangent_matrix[:, start:stop] = action.transpose(0, 1)
        if not bool(torch.isfinite(tangent_matrix).all().item()):
            raise FloatingPointError("Q16 reference tangent cache is non-finite")
        self._reference_tangent_matrix_torch = tangent_matrix
        self._reference_tangent_anchor_torch = state_t.detach().clone()
        self.reference_tangent_cache_refresh_count += 1
        return True

    def _dot(self, left: wp.array, right: wp.array) -> wp.array:
        result = wp.zeros(left.shape[0], dtype=DTYPE, device=self.device)
        wp.launch(
            _q16_dot_kernel,
            dim=left.shape[0],
            inputs=[left, right, self.dof_count],
            outputs=[result],
            device=self.device,
        )
        return result

    def _norms(self, value: wp.array) -> tuple[float, ...]:
        squared = self._dot(value, value)
        wp.synchronize_device(self.device)
        raw = squared.numpy()
        result: list[float] = []
        for item in raw:
            scalar = float(item)
            if not math.isfinite(scalar) or scalar < 0.0:
                raise FloatingPointError("CUDA structural norm became non-finite")
            result.append(math.sqrt(scalar))
        return tuple(result)

    def _kinematics(
        self,
        state: wp.array,
        state_predictor: wp.array,
        velocity_predictor: wp.array,
        inverse_beta_dt2: float,
        gamma_dt: float,
    ) -> tuple[wp.array, wp.array]:
        acceleration = wp.zeros_like(state)
        velocity = wp.zeros_like(state)
        wp.launch(
            _q16_newmark_kinematics_kernel,
            dim=state.shape,
            inputs=[
                state,
                state_predictor,
                velocity_predictor,
                DTYPE(inverse_beta_dt2),
                DTYPE(gamma_dt),
            ],
            outputs=[acceleration, velocity],
            device=self.device,
        )
        return acceleration, velocity

    def _residual(
        self,
        linearization,
        acceleration: wp.array,
        velocity: wp.array,
        external_force: wp.array,
        acceleration_load_action: AccelerationLoadAction | None = None,
    ) -> tuple[wp.array, wp.array]:
        internal = self._operator._internal_force_linearized_prechecked(linearization)
        inertia = self._operator._mass_action_prechecked(acceleration)
        mass_velocity = self._operator._mass_action_prechecked(velocity)
        effective_external = external_force
        if acceleration_load_action is not None:
            acceleration_load = acceleration_load_action(acceleration)
            _require_global_state(
                "acceleration-dependent external force",
                acceleration_load,
                device=self.device,
                dof_count=self.dof_count,
            )
            external_t = wp.to_torch(external_force) + wp.to_torch(acceleration_load)
            effective_external = wp.from_torch(
                external_t, dtype=DTYPE, requires_grad=False
            )
        total = wp.zeros_like(acceleration)
        if self.stiffness_damping_coefficient > 0.0:
            tangent = self._reference_tangent_matrix_torch
            if tangent is None:
                raise Q16StructuralStepStopped(
                    "Q16 stiffness damping needs the reference tangent cache",
                    phase="residual",
                    newton_iteration_count=0,
                    cg_iteration_count=0,
                    gmres_iteration_count=0,
                    relative_residual_max=math.inf,
                )
            damping_force = self.stiffness_damping_coefficient * (
                wp.to_torch(velocity) @ tangent.transpose(0, 1)
            )
            stiffness_damping_force = wp.from_torch(
                damping_force, dtype=DTYPE, requires_grad=False
            )
        else:
            stiffness_damping_force = wp.zeros_like(acceleration)
        wp.launch(
            _q16_residual_kernel,
            dim=acceleration.shape,
            inputs=[
                inertia,
                mass_velocity,
                DTYPE(self.mass_damping_coefficient),
                stiffness_damping_force,
                internal,
                self._reference_internal_force,
                effective_external,
            ],
            outputs=[total],
            device=self.device,
        )
        return total, self._boundary_operator._project_free_prechecked(total)

    def _effective_mass_coefficient(self, inverse_beta_dt2: float) -> float:
        velocity_derivative = self.gamma * math.sqrt(inverse_beta_dt2 / self.beta)
        result = inverse_beta_dt2 + (
            self.mass_damping_coefficient * velocity_derivative
        )
        if not math.isfinite(result) or result <= 0.0:
            raise FloatingPointError("Q16 damped effective mass coefficient is invalid")
        return result

    def _effective_stiffness_damping_factor(self, inverse_beta_dt2: float) -> float:
        """Newmark factor on K_ref from Kelvin-Voigt damping: theta*gamma/(beta*dt)."""

        return self.stiffness_damping_coefficient * self.gamma * math.sqrt(
            inverse_beta_dt2 / self.beta
        )

    def _effective_action(
        self,
        linearization,
        direction: wp.array,
        inverse_beta_dt2: float,
        acceleration_load_action: AccelerationLoadAction | None = None,
    ) -> wp.array:
        admissible = self._boundary_operator._project_free_prechecked(direction)
        mass = self._operator._mass_action_prechecked(admissible)
        tangent = self._operator._tangent_action_linearized_prechecked(
            admissible, linearization
        )
        total = wp.zeros_like(direction)
        wp.launch(
            _q16_effective_action_kernel,
            dim=direction.shape,
            inputs=[
                mass,
                tangent,
                DTYPE(self._effective_mass_coefficient(inverse_beta_dt2)),
            ],
            outputs=[total],
            device=self.device,
        )
        if acceleration_load_action is not None:
            acceleration_load = acceleration_load_action(admissible)
            _require_global_state(
                "acceleration-load tangent action",
                acceleration_load,
                device=self.device,
                dof_count=self.dof_count,
            )
            total_t = wp.to_torch(total) - inverse_beta_dt2 * wp.to_torch(
                acceleration_load
            )
            total = wp.from_torch(total_t, dtype=DTYPE, requires_grad=False)
        if self.stiffness_damping_coefficient > 0.0:
            tangent = self._reference_tangent_matrix_torch
            if tangent is None:
                raise Q16StructuralStepStopped(
                    "Q16 stiffness damping needs the reference tangent cache",
                    phase="linear_solve",
                    newton_iteration_count=0,
                    cg_iteration_count=0,
                    gmres_iteration_count=0,
                    relative_residual_max=math.inf,
                )
            total_t = wp.to_torch(total) + self._effective_stiffness_damping_factor(
                inverse_beta_dt2
            ) * (wp.to_torch(admissible) @ tangent.transpose(0, 1))
            total = wp.from_torch(total_t, dtype=DTYPE, requires_grad=False)
        return self._boundary_operator._project_free_prechecked(total)

    def _jacobi_inverse(
        self, linearization, batch: int, inverse_beta_dt2: float
    ) -> wp.array:
        mass = self._operator._mass_diagonal_prechecked(batch)
        material = self._operator._material_diagonal_linearized_prechecked(
            linearization
        )
        diagonal = wp.zeros_like(mass)
        wp.launch(
            _q16_effective_diagonal_kernel,
            dim=diagonal.shape,
            inputs=[
                mass,
                material,
                DTYPE(self._effective_mass_coefficient(inverse_beta_dt2)),
            ],
            outputs=[diagonal],
            device=self.device,
        )
        inverse = wp.zeros_like(diagonal)
        failure = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            _q16_invert_jacobi_kernel,
            dim=diagonal.shape,
            inputs=[diagonal],
            outputs=[inverse, failure],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if int(failure.numpy()[0]) != 0:
            raise Q16StructuralStepStopped(
                "CUDA structural Jacobi diagonal is non-positive or non-finite",
                phase="preconditioner",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=math.inf,
            )
        return inverse

    def _apply_preconditioner(
        self,
        inverse: wp.array | None,
        residual: wp.array,
        active: wp.array,
    ) -> wp.array:
        if inverse is None:
            return wp.clone(residual)
        result = wp.zeros_like(residual)
        wp.launch(
            _q16_apply_jacobi_kernel,
            dim=residual.shape,
            inputs=[inverse, residual, active],
            outputs=[result],
            device=self.device,
        )
        return self._boundary_operator._project_free_prechecked(result)

    def _acceleration_load_matrix(
        self, acceleration_load_action: AccelerationLoadAction | None
    ) -> torch.Tensor | None:
        """Return an optional explicit CUDA load Jacobian for preconditioning."""

        if acceleration_load_action is None:
            return None
        matrix = getattr(acceleration_load_action, "generalized_matrix", None)
        if matrix is None:
            return None
        if (
            type(matrix) is not torch.Tensor
            or matrix.device.type != "cuda"
            or matrix.dtype is not torch.float64
            or tuple(matrix.shape) != (self.dof_count, self.dof_count)
        ):
            raise RuntimeError(
                "acceleration-load generalized_matrix must be a square CUDA "
                "float64 Q16 operator"
            )
        if not bool(torch.isfinite(matrix).all().item()):
            raise FloatingPointError(
                "acceleration-load generalized_matrix is non-finite"
            )
        return matrix

    def _cg_solve(
        self,
        linearization,
        rhs: wp.array,
        inverse_beta_dt2: float,
        acceleration_load_action: AccelerationLoadAction | None = None,
    ) -> tuple[wp.array, int]:
        batch = rhs.shape[0]
        solution = wp.zeros_like(rhs)
        residual = wp.clone(rhs)
        residual_norm_squared = self._dot(residual, residual)
        rhs_norm_squared = wp.clone(residual_norm_squared)
        active = wp.zeros(batch, dtype=wp.int32, device=self.device)
        failure = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            _q16_cg_initialize_active_kernel,
            dim=batch,
            inputs=[rhs_norm_squared],
            outputs=[active],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if not bool(active.numpy().any()):
            return solution, 0

        inverse = None
        if self.preconditioner == "material_jacobi":
            inverse = self._jacobi_inverse(linearization, batch, inverse_beta_dt2)
        preconditioned = self._apply_preconditioner(inverse, residual, active)
        direction = wp.clone(preconditioned)
        residual_preconditioned = self._dot(residual, preconditioned)
        alpha = wp.zeros(batch, dtype=DTYPE, device=self.device)
        tolerance_squared = self.cg_tolerance * self.cg_tolerance
        for iteration in range(1, self.max_cg_iterations + 1):
            action = self._effective_action(
                linearization,
                direction,
                inverse_beta_dt2,
                acceleration_load_action,
            )
            direction_action = self._dot(direction, action)
            wp.launch(
                _q16_cg_alpha_kernel,
                dim=batch,
                inputs=[
                    residual_preconditioned,
                    direction_action,
                    active,
                ],
                outputs=[alpha, failure],
                device=self.device,
            )
            wp.launch(
                _q16_cg_update_kernel,
                dim=rhs.shape,
                inputs=[solution, residual, direction, action, alpha, active],
                device=self.device,
            )
            next_norm_squared = self._dot(residual, residual)
            wp.launch(
                _q16_cg_mark_converged_kernel,
                dim=batch,
                inputs=[
                    next_norm_squared,
                    rhs_norm_squared,
                    DTYPE(tolerance_squared),
                ],
                outputs=[active, failure],
                device=self.device,
            )
            if iteration % self.cg_check_every == 0 or iteration == 1:
                wp.synchronize_device(self.device)
                if int(failure.numpy()[0]) != 0:
                    raise Q16StructuralStepStopped(
                        "CUDA structural CG encountered non-positive curvature",
                        phase="linear_solve",
                        newton_iteration_count=0,
                        cg_iteration_count=iteration,
                        relative_residual_max=math.inf,
                    )
                if not bool(active.numpy().any()):
                    return solution, iteration
            next_preconditioned = self._apply_preconditioner(inverse, residual, active)
            next_residual_preconditioned = self._dot(residual, next_preconditioned)
            wp.launch(
                _q16_cg_direction_kernel,
                dim=rhs.shape,
                inputs=[
                    direction,
                    next_preconditioned,
                    next_residual_preconditioned,
                    residual_preconditioned,
                    active,
                ],
                outputs=[failure],
                device=self.device,
            )
            wp.copy(residual_norm_squared, next_norm_squared)
            wp.copy(residual_preconditioned, next_residual_preconditioned)

        wp.synchronize_device(self.device)
        rhs_values = rhs_norm_squared.numpy()
        residual_values = residual_norm_squared.numpy()
        relative = max(
            math.sqrt(float(r) / max(float(b), 1.0e-300))
            for r, b in zip(residual_values, rhs_values, strict=True)
        )
        raise Q16StructuralStepStopped(
            "CUDA structural CG did not converge",
            phase="linear_solve",
            newton_iteration_count=0,
            cg_iteration_count=self.max_cg_iterations,
            relative_residual_max=relative,
        )

    def _gmres_solve(
        self,
        linearization,
        rhs: wp.array,
        inverse_beta_dt2: float,
        acceleration_load_action: AccelerationLoadAction | None = None,
    ) -> tuple[wp.array, int]:
        """Left-preconditioned restarted CUDA GMRES for an indefinite tangent."""

        if rhs.shape[0] != 1:
            raise Q16StructuralStepStopped(
                "CUDA structural GMRES fallback requires one active FSI sample",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                gmres_iteration_count=0,
                relative_residual_max=math.inf,
            )
        original_right = wp.to_torch(rhs)
        if (
            original_right.device.type != "cuda"
            or original_right.dtype is not torch.float64
        ):
            raise RuntimeError("Q16 structural GMRES left CUDA float64")
        right_norm = torch.linalg.vector_norm(original_right)
        right_norm_scalar = float(right_norm)
        if not math.isfinite(right_norm_scalar):
            raise Q16StructuralStepStopped(
                "CUDA structural GMRES right-hand side is non-finite",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                gmres_iteration_count=0,
                relative_residual_max=math.inf,
            )
        if right_norm_scalar == 0.0:
            return wp.zeros_like(rhs), 0

        active = wp.ones(1, dtype=wp.int32, device=self.device)
        inverse = None
        added_mass_matrix = self._acceleration_load_matrix(
            acceleration_load_action
        )
        dense_lu: torch.Tensor | None = None
        dense_pivots: torch.Tensor | None = None
        if added_mass_matrix is not None and self.preconditioner == "material_jacobi":
            material_diagonal = wp.to_torch(
                self._operator._material_diagonal_linearized_prechecked(
                    linearization
                )
            )[0]
            effective = (
                self._effective_mass_coefficient(inverse_beta_dt2)
                * self._mass_matrix_torch
                - inverse_beta_dt2 * added_mass_matrix
            ).clone()
            effective.diagonal().add_(material_diagonal)
            free = self._free_dof_indices_torch
            dense_lu, dense_pivots, info = torch.linalg.lu_factor_ex(
                effective[free][:, free]
            )
            if int(info.item()) != 0:
                raise Q16StructuralStepStopped(
                    "CUDA Q16 mass-minus-added-mass preconditioner is singular",
                    phase="preconditioner",
                    newton_iteration_count=0,
                    cg_iteration_count=0,
                    gmres_iteration_count=0,
                    relative_residual_max=math.inf,
                )
        elif self.preconditioner == "material_jacobi":
            inverse = self._jacobi_inverse(linearization, 1, inverse_beta_dt2)

        def apply_left_preconditioner(value: wp.array) -> torch.Tensor:
            if dense_lu is None or dense_pivots is None:
                return wp.to_torch(
                    self._apply_preconditioner(inverse, value, active)
                )
            value_t = wp.to_torch(value)
            result = torch.zeros_like(value_t)
            free = self._free_dof_indices_torch
            result[:, free] = torch.linalg.lu_solve(
                dense_lu,
                dense_pivots,
                value_t[:, free].transpose(0, 1),
            ).transpose(0, 1)
            return result

        right = apply_left_preconditioner(rhs)
        preconditioned_right_norm = torch.linalg.vector_norm(right)
        if (
            not bool(torch.isfinite(preconditioned_right_norm).item())
            or float(preconditioned_right_norm) == 0.0
        ):
            raise Q16StructuralStepStopped(
                "CUDA structural GMRES preconditioned right-hand side is invalid",
                phase="preconditioner",
                newton_iteration_count=0,
                cg_iteration_count=0,
                gmres_iteration_count=0,
                relative_residual_max=math.inf,
            )

        solution = torch.zeros_like(original_right)
        # A one-element Q16 state has 96 dofs.  Keeping its complete Krylov
        # space avoids discarding the indefinite spectral direction at an
        # arbitrary 64-vector restart while retaining a bounded window for
        # larger macro meshes.
        restart = min(128, self.dof_count, self.max_cg_iterations)
        total_iterations = 0
        tolerance = self.cg_tolerance * preconditioned_right_norm
        exact_tolerance = self.cg_tolerance * right_norm
        while total_iterations < self.max_cg_iterations:
            solution_wp = wp.from_torch(solution, dtype=DTYPE, requires_grad=False)
            action_wp = self._effective_action(
                linearization,
                solution_wp,
                inverse_beta_dt2,
                acceleration_load_action,
            )
            original_residual = original_right - wp.to_torch(action_wp)
            residual_wp = wp.from_torch(
                original_residual, dtype=DTYPE, requires_grad=False
            )
            residual = apply_left_preconditioner(residual_wp)
            residual_norm = torch.linalg.vector_norm(residual)
            exact_residual_norm = torch.linalg.vector_norm(original_residual)
            if bool((exact_residual_norm <= exact_tolerance).item()):
                return (
                    wp.clone(wp.from_torch(solution, dtype=DTYPE, requires_grad=False)),
                    total_iterations,
                )
            if not bool(torch.isfinite(residual_norm).item()) or not bool(
                torch.isfinite(exact_residual_norm).item()
            ):
                break

            inner_limit = min(restart, self.max_cg_iterations - total_iterations)
            basis = [residual / residual_norm]
            hessenberg = torch.zeros(
                (inner_limit + 1, inner_limit),
                dtype=torch.float64,
                device=right.device,
            )
            target = torch.zeros(
                inner_limit + 1, dtype=torch.float64, device=right.device
            )
            target[0] = residual_norm
            candidate = solution
            for column in range(inner_limit):
                direction = wp.from_torch(
                    basis[column], dtype=DTYPE, requires_grad=False
                )
                vector_wp = self._effective_action(
                    linearization,
                    direction,
                    inverse_beta_dt2,
                    acceleration_load_action,
                )
                vector = apply_left_preconditioner(vector_wp)
                for row in range(column + 1):
                    coefficient = torch.sum(basis[row] * vector)
                    hessenberg[row, column] = coefficient
                    vector = vector - coefficient * basis[row]
                next_norm = torch.linalg.vector_norm(vector)
                hessenberg[column + 1, column] = next_norm
                if not bool(torch.isfinite(next_norm).item()):
                    break
                if float(next_norm) > 0.0 and column + 1 < inner_limit:
                    basis.append(vector / next_norm)

                active_hessenberg = hessenberg[: column + 2, : column + 1]
                active_target = target[: column + 2]
                try:
                    coefficients = torch.linalg.lstsq(
                        active_hessenberg,
                        active_target,
                        driver="gels",
                    ).solution
                except RuntimeError as error:
                    raise Q16StructuralStepStopped(
                        "CUDA structural GMRES least-squares solve failed",
                        phase="linear_solve",
                        newton_iteration_count=0,
                        cg_iteration_count=0,
                        gmres_iteration_count=total_iterations,
                        relative_residual_max=math.inf,
                    ) from error
                basis_matrix = torch.cat(basis[: column + 1], dim=0)
                candidate = solution + torch.matmul(
                    coefficients, basis_matrix
                ).unsqueeze(0)
                least_squares_residual = torch.linalg.vector_norm(
                    active_target - torch.matmul(active_hessenberg, coefficients)
                )
                total_iterations += 1
                if bool((least_squares_residual <= tolerance).item()):
                    candidate_wp = wp.from_torch(
                        candidate, dtype=DTYPE, requires_grad=False
                    )
                    exact_residual = original_right - wp.to_torch(
                        self._effective_action(
                            linearization,
                            candidate_wp,
                            inverse_beta_dt2,
                            acceleration_load_action,
                        )
                    )
                    if bool(
                        (
                            torch.linalg.vector_norm(exact_residual) <= exact_tolerance
                        ).item()
                    ):
                        return wp.clone(candidate_wp), total_iterations
                    # Loss of Arnoldi orthogonality can make the small least-
                    # squares residual optimistic.  Restart from the candidate
                    # only after checking the real matrix-free CUDA residual.
                    solution = candidate
                    break
                if float(next_norm) == 0.0:
                    solution = candidate
                    break
            else:
                solution = candidate
                continue
            solution = candidate

        solution_wp = wp.from_torch(solution, dtype=DTYPE, requires_grad=False)
        final_residual = original_right - wp.to_torch(
            self._effective_action(
                linearization,
                solution_wp,
                inverse_beta_dt2,
                acceleration_load_action,
            )
        )
        relative = float(torch.linalg.vector_norm(final_residual) / right_norm)
        raise Q16StructuralStepStopped(
            "CUDA structural GMRES did not converge",
            phase="linear_solve",
            newton_iteration_count=0,
            cg_iteration_count=0,
            gmres_iteration_count=total_iterations,
            relative_residual_max=relative,
        )

    def _assemble_dense_nonsymmetric_effective_free(
        self,
        linearization,
        inverse_beta_dt2: float,
        acceleration_load_action: AccelerationLoadAction,
    ) -> torch.Tensor:
        """Assemble one live Q16 effective tangent on free CUDA DOFs."""

        self._acceleration_load_matrix(acceleration_load_action)
        free = self._free_dof_indices_torch
        free_count = int(free.shape[0])
        effective_free = torch.empty(
            (free_count, free_count),
            device=self.device,
            dtype=torch.float64,
        )
        direction_batch_size = min(64, free_count)
        for start in range(0, free_count, direction_batch_size):
            stop = min(start + direction_batch_size, free_count)
            count = stop - start
            directions = torch.zeros(
                (count, self.dof_count),
                device=self.device,
                dtype=torch.float64,
            )
            rows = torch.arange(count, device=self.device)
            directions[rows, free[start:stop]] = 1.0
            repeated_linearization = tuple(
                wp.from_torch(
                    wp.to_torch(value).expand(
                        (count,) + tuple(value.shape[1:])
                    ).contiguous(),
                    dtype=DTYPE,
                    requires_grad=False,
                )
                for value in linearization
            )
            action = wp.to_torch(
                self._effective_action(
                    repeated_linearization,
                    wp.from_torch(
                        directions, dtype=DTYPE, requires_grad=False
                    ),
                    inverse_beta_dt2,
                    acceleration_load_action,
                )
            )
            effective_free[:, start:stop] = action[:, free].transpose(0, 1)
        if not bool(torch.isfinite(effective_free).all().item()):
            raise Q16StructuralStepStopped(
                "CUDA dense Q16 effective tangent assembly is non-finite",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=math.inf,
            )
        return effective_free

    def _solve_dense_effective_free(
        self,
        effective_free: torch.Tensor,
        rhs: wp.array,
    ) -> wp.array:
        """Solve one already assembled free-DOF effective matrix on CUDA."""

        if rhs.shape[0] != 1:
            raise Q16StructuralStepStopped(
                "CUDA dense Q16 solve requires one active FSI sample",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=math.inf,
            )
        free = self._free_dof_indices_torch
        right = wp.to_torch(rhs)[0, free]
        solution_free, info = torch.linalg.solve_ex(effective_free, right)
        if int(info.item()) != 0 or not bool(torch.isfinite(solution_free).all().item()):
            raise Q16StructuralStepStopped(
                "CUDA dense Q16 effective tangent solve failed",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=math.inf,
            )
        relative = torch.linalg.vector_norm(
            effective_free @ solution_free - right
        ) / torch.linalg.vector_norm(right)
        if not bool(torch.isfinite(relative).item()) or float(relative) > max(
            1.0e-11, 10.0 * self.cg_tolerance
        ):
            raise Q16StructuralStepStopped(
                "CUDA dense Q16 effective tangent residual failed",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=float(relative),
            )
        solution = torch.zeros_like(wp.to_torch(rhs))
        solution[0, free] = solution_free
        return wp.from_torch(solution, dtype=DTYPE, requires_grad=False)

    def _dense_nonsymmetric_solve(
        self,
        linearization,
        rhs: wp.array,
        inverse_beta_dt2: float,
        acceleration_load_action: AccelerationLoadAction,
    ) -> wp.array:
        """Assemble and solve one exact Q16 effective tangent on CUDA.

        UVLM distributed pressure makes the acceleration Jacobian generally
        nonsymmetric.  At the Q16 case-reproduction sizes, batching 64 basis
        directions gives the GPU substantially more parallel work than a long
        sequence of single-vector GMRES actions and avoids Krylov stagnation.
        """

        effective_free = self._assemble_dense_nonsymmetric_effective_free(
            linearization,
            inverse_beta_dt2,
            acceleration_load_action,
        )
        return self._solve_dense_effective_free(effective_free, rhs)

    def _reference_dense_nonsymmetric_solve(
        self,
        rhs: wp.array,
        inverse_beta_dt2: float,
        acceleration_load_action: AccelerationLoadAction,
    ) -> wp.array:
        """Solve with a cached reference structural tangent on CUDA.

        This is a quasi-Newton linear solve, not a residual approximation:
        every accepted nonlinear iterate is still checked by ``_residual``
        with the live Q16 constitutive linearization.
        """

        if rhs.shape[0] != 1:
            raise Q16StructuralStepStopped(
                "CUDA reference-dense Q16 solve requires one active FSI sample",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=math.inf,
            )
        tangent = self._reference_tangent_matrix_torch
        added_mass = self._acceleration_load_matrix(acceleration_load_action)
        if tangent is None or added_mass is None:
            raise Q16StructuralStepStopped(
                "CUDA reference-dense Q16 operators are unavailable",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=math.inf,
            )
        effective = (
            tangent * (
                1.0 + self._effective_stiffness_damping_factor(inverse_beta_dt2)
            )
            + self._effective_mass_coefficient(inverse_beta_dt2)
            * self._mass_matrix_torch
            - inverse_beta_dt2 * added_mass
        )
        free = self._free_dof_indices_torch
        effective_free = effective[free][:, free]
        right = wp.to_torch(rhs)[0, free]
        solution_free, info = torch.linalg.solve_ex(effective_free, right)
        if int(info.item()) != 0 or not bool(torch.isfinite(solution_free).all().item()):
            raise Q16StructuralStepStopped(
                "CUDA reference-dense Q16 effective solve failed",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=math.inf,
            )
        relative = torch.linalg.vector_norm(
            effective_free @ solution_free - right
        ) / torch.linalg.vector_norm(right)
        if not bool(torch.isfinite(relative).item()) or float(relative) > max(
            1.0e-11, 10.0 * self.cg_tolerance
        ):
            raise Q16StructuralStepStopped(
                "CUDA reference-dense Q16 linear residual failed",
                phase="linear_solve",
                newton_iteration_count=0,
                cg_iteration_count=0,
                relative_residual_max=float(relative),
            )
        solution = torch.zeros_like(wp.to_torch(rhs))
        solution[0, free] = solution_free
        return wp.from_torch(solution, dtype=DTYPE, requires_grad=False)

    def predict_kinematics(
        self,
        state: Any,
        velocity: Any,
        acceleration: Any,
        *,
        delta_time: float,
    ) -> tuple[wp.array, wp.array]:
        """Return the constrained CUDA Newmark displacement/velocity predictor."""

        dt = _positive_float("delta_time", delta_time)
        beta_dt2 = self.beta * dt * dt
        if not math.isfinite(beta_dt2) or beta_dt2 <= 0.0:
            raise ValueError("delta_time produces an invalid Newmark denominator")
        self._boundary_operator.require_kinematics(state, velocity, acceleration)
        state_predictor = wp.zeros_like(state)
        velocity_predictor = wp.zeros_like(state)
        wp.launch(
            _q16_newmark_predict_kernel,
            dim=state.shape,
            inputs=[
                state,
                velocity,
                acceleration,
                DTYPE(dt),
                DTYPE(self.beta),
                DTYPE(self.gamma),
            ],
            outputs=[state_predictor, velocity_predictor],
            device=self.device,
        )
        return (
            self._boundary_operator._enforce_state_prechecked(state_predictor),
            self._boundary_operator._project_free_prechecked(velocity_predictor),
        )

    def audit_step_work(
        self,
        state_start: Any,
        velocity_start: Any,
        acceleration_start: Any,
        state_end: Any,
        velocity_end: Any,
        acceleration_end: Any,
        external_force_end: Any,
    ) -> Q16StructuralWorkBalance:
        """Evaluate the accepted endpoint work identity on CUDA float64."""

        self._boundary_operator.require_kinematics(
            state_start, velocity_start, acceleration_start
        )
        self._boundary_operator.require_kinematics(
            state_end, velocity_end, acceleration_end
        )
        _require_global_state(
            "external force end",
            external_force_end,
            device=self.device,
            dof_count=self.dof_count,
        )
        if not (
            state_start.shape
            == velocity_start.shape
            == acceleration_start.shape
            == state_end.shape
            == velocity_end.shape
            == acceleration_end.shape
            == external_force_end.shape
            == (1, self.dof_count)
        ):
            raise ValueError("Q16 work audit requires one common structural batch")

        mass_velocity_start = self._operator._mass_action_prechecked(velocity_start)
        mass_velocity_end = self._operator._mass_action_prechecked(velocity_end)
        inertia_start = self._operator._mass_action_prechecked(acceleration_start)
        internal_start = self._operator._internal_force_prechecked(state_start)
        internal_end = self._operator._internal_force_prechecked(state_end)

        q0 = wp.to_torch(state_start)
        q1 = wp.to_torch(state_end)
        v0 = wp.to_torch(velocity_start)
        v1 = wp.to_torch(velocity_end)
        a1 = wp.to_torch(acceleration_end)
        mv0 = wp.to_torch(mass_velocity_start)
        mv1 = wp.to_torch(mass_velocity_end)
        ma0 = wp.to_torch(inertia_start)
        f0 = wp.to_torch(internal_start) - wp.to_torch(self._reference_internal_force)
        f1 = wp.to_torch(internal_end) - wp.to_torch(self._reference_internal_force)
        external_end = wp.to_torch(external_force_end)
        reference = wp.to_torch(self._reference_state)
        values = (q0, q1, v0, v1, a1, mv0, mv1, ma0, f0, f1, external_end)
        if any(
            value.device.type != "cuda" or value.dtype is not torch.float64
            for value in values
        ):
            raise RuntimeError("Q16 work audit left CUDA float64")

        delta_state = q1 - q0
        kinetic_start = 0.5 * torch.sum(v0 * mv0)
        kinetic_end = 0.5 * torch.sum(v1 * mv1)
        kinetic_change = kinetic_end - kinetic_start
        internal_work = torch.sum(0.5 * (f0 + f1) * delta_state)
        damping_start = self.mass_damping_coefficient * mv0
        damping_end = self.mass_damping_coefficient * mv1
        damping_work = torch.sum(0.5 * (damping_start + damping_end) * delta_state)
        reconstructed_external_start = ma0 + damping_start + f0
        external_work = torch.sum(
            0.5 * (reconstructed_external_start + external_end) * delta_state
        )
        balance = kinetic_change + internal_work + damping_work - external_work
        scale = torch.maximum(
            torch.ones((), dtype=torch.float64, device=q0.device),
            torch.maximum(
                torch.abs(kinetic_change),
                torch.maximum(
                    torch.abs(internal_work),
                    torch.maximum(torch.abs(damping_work), torch.abs(external_work)),
                ),
            ),
        )
        relative = torch.abs(balance) / scale
        diagnostics = torch.stack(
            (
                kinetic_start,
                kinetic_end,
                kinetic_change,
                internal_work,
                damping_work,
                external_work,
                balance,
                relative,
                torch.linalg.vector_norm(delta_state),
                torch.linalg.vector_norm(q1 - reference),
                torch.linalg.vector_norm(v1),
                torch.linalg.vector_norm(a1),
            )
        )
        if not bool(torch.isfinite(diagnostics).all().item()):
            raise FloatingPointError("Q16 CUDA work audit became non-finite")
        scalars = tuple(float(value) for value in diagnostics.tolist())
        if (
            scalars[0] < 0.0
            or scalars[1] < 0.0
            or scalars[4] < 0.0
            or any(value < 0.0 for value in scalars[7:])
        ):
            raise FloatingPointError("Q16 CUDA work audit sign invariant failed")
        return Q16StructuralWorkBalance(*scalars)

    def step(
        self,
        state: Any,
        velocity: Any,
        acceleration: Any,
        external_force: Any,
        *,
        delta_time: float,
        acceleration_load_action: AccelerationLoadAction | None = None,
    ) -> Q16StructuralStepResult:
        dt = _positive_float("delta_time", delta_time)
        beta_dt2 = self.beta * dt * dt
        if not math.isfinite(beta_dt2) or beta_dt2 <= 0.0:
            raise ValueError("delta_time produces an invalid Newmark denominator")
        inverse_beta_dt2 = 1.0 / beta_dt2
        gamma_dt = self.gamma * dt
        if not math.isfinite(inverse_beta_dt2):
            raise ValueError("delta_time produces a non-finite Newmark coefficient")
        _require_global_state(
            "external force",
            external_force,
            device=self.device,
            dof_count=self.dof_count,
        )
        if not (
            state.shape == velocity.shape == acceleration.shape == external_force.shape
        ):
            raise ValueError("structural step input batch shapes differ")
        if acceleration_load_action is not None and not callable(
            acceleration_load_action
        ):
            raise TypeError("acceleration_load_action must be callable")

        state_predictor, velocity_predictor = self.predict_kinematics(
            state,
            velocity,
            acceleration,
            delta_time=dt,
        )
        # Newton corrections must not mutate the frozen Newmark predictor.
        # Otherwise the first correction silently changes the reference used
        # to reconstruct acceleration and velocity on every later iteration.
        current = wp.clone(state_predictor)
        free_external = self._boundary_operator._project_free_prechecked(external_force)
        external_norms = self._norms(free_external)
        scales = tuple(max(value, 1.0) for value in external_norms)
        total_cg_iterations = 0
        total_gmres_iterations = 0
        total_direct_solves = 0
        live_tangent_refresh_count = 0
        indefinite_fallback_count = 0
        relative_residual_max = math.inf
        newton_residual_history: list[float] = []
        globalize_newton = False
        refreshed_effective_free: torch.Tensor | None = None

        for newton_iteration in range(self.max_newton_iterations + 1):
            current_acceleration, current_velocity = self._kinematics(
                current,
                state_predictor,
                velocity_predictor,
                inverse_beta_dt2,
                gamma_dt,
            )
            try:
                linearization = self._operator._linearization_prechecked(current)
            except FloatingPointError as error:
                if (
                    newton_iteration != 0
                    or str(error)
                    != "Q16 current geometry is orientation reversing or singular"
                ):
                    raise
                # The Newmark predictor defines the exact acceleration and
                # velocity reconstruction, but it is only an initial Newton
                # guess.  Find the largest admissible point on the CUDA line
                # from the committed state to that predictor; do not change the
                # frozen predictor used by the Newmark equations.
                predictor_factor = 0.5
                predictor_linearization = None
                for _ in range(17):
                    predictor_trial = wp.zeros_like(state_predictor)
                    wp.launch(
                        _q16_predictor_trial_kernel,
                        dim=state_predictor.shape,
                        inputs=[
                            state,
                            state_predictor,
                            DTYPE(predictor_factor),
                        ],
                        outputs=[predictor_trial],
                        device=self.device,
                    )
                    predictor_trial = self._boundary_operator._enforce_state_prechecked(
                        predictor_trial
                    )
                    try:
                        predictor_linearization = (
                            self._operator._linearization_prechecked(predictor_trial)
                        )
                    except FloatingPointError as predictor_error:
                        if (
                            str(predictor_error)
                            != "Q16 current geometry is orientation reversing or singular"
                        ):
                            raise
                        predictor_factor *= 0.5
                        continue
                    current = predictor_trial
                    break
                if predictor_linearization is None:
                    raise Q16StructuralStepStopped(
                        "CUDA structural predictor backtracking did not find an "
                        "admissible Q16 state",
                        phase="predictor_geometry",
                        newton_iteration_count=0,
                        cg_iteration_count=0,
                        gmres_iteration_count=0,
                        relative_residual_max=math.inf,
                    ) from error
                globalize_newton = True
                current_acceleration, current_velocity = self._kinematics(
                    current,
                    state_predictor,
                    velocity_predictor,
                    inverse_beta_dt2,
                    gamma_dt,
                )
                linearization = predictor_linearization
            total_residual, free_residual = self._residual(
                linearization,
                current_acceleration,
                current_velocity,
                external_force,
                acceleration_load_action,
            )
            residual_norms = self._norms(free_residual)
            relatives = tuple(
                residual / scale
                for residual, scale in zip(residual_norms, scales, strict=True)
            )
            relative_residual_max = max(relatives)
            newton_residual_history.append(relative_residual_max)
            if relative_residual_max <= self.newton_tolerance:
                reaction = self._boundary_operator._extract_reaction_prechecked(
                    total_residual
                )
                for name, value in (
                    ("result state", current),
                    ("result velocity", current_velocity),
                    ("result acceleration", current_acceleration),
                    ("result reaction", reaction),
                ):
                    _require_global_state(
                        name,
                        value,
                        device=self.device,
                        dof_count=self.dof_count,
                    )
                return Q16StructuralStepResult(
                    state=current,
                    velocity=current_velocity,
                    acceleration=current_acceleration,
                    reaction=reaction,
                    delta_time=dt,
                    newton_iteration_count=newton_iteration,
                    cg_iteration_count=total_cg_iterations,
                    gmres_iteration_count=total_gmres_iterations,
                    direct_solve_count=total_direct_solves,
                    live_tangent_refresh_count=live_tangent_refresh_count,
                    indefinite_fallback_count=indefinite_fallback_count,
                    relative_residual_max=relative_residual_max,
                )
            if newton_iteration == self.max_newton_iterations:
                raise Q16StructuralStepStopped(
                    "CUDA structural Newton iteration did not converge; residual "
                    f"history={newton_residual_history!r}",
                    phase="newton_convergence",
                    newton_iteration_count=newton_iteration,
                    cg_iteration_count=total_cg_iterations,
                    gmres_iteration_count=total_gmres_iterations,
                    relative_residual_max=relative_residual_max,
                    newton_residual_history=tuple(newton_residual_history),
                )

            active_values = np.asarray(
                [int(value > self.newton_tolerance) for value in relatives],
                dtype=np.int32,
            )
            active = wp.array(active_values, dtype=wp.int32, device=self.device)
            rhs = wp.zeros_like(free_residual)
            wp.launch(
                _q16_masked_negative_kernel,
                dim=free_residual.shape,
                inputs=[free_residual, active],
                outputs=[rhs],
                device=self.device,
            )
            explicit_acceleration_matrix = self._acceleration_load_matrix(
                acceleration_load_action
            )
            if explicit_acceleration_matrix is not None:
                # The distributed UVLM pressure operator is generally
                # nonsymmetric.  CG is mathematically inapplicable even when
                # its first curvature samples happen to be positive.  Small
                # verification meshes retain the exact live-tangent GPU
                # solve.  Larger CASE meshes may explicitly use the cached
                # reference-tangent quasi-Newton solve: its correction is
                # approximate, but every accepted iterate remains gated by
                # the live Q16 nonlinear residual below.  Restarted CUDA
                # GMRES remains an explicit diagnostic option.
                if self.nonsymmetric_solver == "gmres":
                    try:
                        increment, gmres_iterations = self._gmres_solve(
                            linearization,
                            rhs,
                            inverse_beta_dt2,
                            acceleration_load_action,
                        )
                    except Q16StructuralStepStopped as error:
                        raise Q16StructuralStepStopped(
                            str(error),
                            phase=error.phase,
                            newton_iteration_count=newton_iteration,
                            cg_iteration_count=total_cg_iterations,
                            gmres_iteration_count=(
                                total_gmres_iterations
                                + error.gmres_iteration_count
                            ),
                            relative_residual_max=relative_residual_max,
                        ) from error
                    total_gmres_iterations += gmres_iterations
                elif self.nonsymmetric_solver == "reference_dense":
                    try:
                        residual_growth = (
                            len(newton_residual_history) >= 3
                            and newton_residual_history[-1]
                            > 1.05 * newton_residual_history[-2]
                            and newton_residual_history[-2]
                            > 1.05 * newton_residual_history[-3]
                        )
                        use_live_tangent = (
                            refreshed_effective_free is not None
                            or newton_iteration >= self.reference_dense_refresh_after
                            or residual_growth
                        )
                        if use_live_tangent:
                            if refreshed_effective_free is None:
                                refreshed_effective_free = (
                                    self._assemble_dense_nonsymmetric_effective_free(
                                        linearization,
                                        inverse_beta_dt2,
                                        acceleration_load_action,
                                    )
                                )
                                live_tangent_refresh_count += 1
                                # Once the cached correction has demonstrably
                                # moved away from the live nonlinear residual,
                                # every recovery step must pass Armijo-style
                                # residual decrease instead of being accepted
                                # solely because the geometry is admissible.
                                globalize_newton = True
                            increment = self._solve_dense_effective_free(
                                refreshed_effective_free,
                                rhs,
                            )
                        else:
                            increment = self._reference_dense_nonsymmetric_solve(
                                rhs,
                                inverse_beta_dt2,
                                acceleration_load_action,
                            )
                    except Q16StructuralStepStopped as error:
                        raise Q16StructuralStepStopped(
                            str(error),
                            phase=error.phase,
                            newton_iteration_count=newton_iteration,
                            cg_iteration_count=total_cg_iterations,
                            gmres_iteration_count=(
                                total_gmres_iterations
                                + error.gmres_iteration_count
                            ),
                            relative_residual_max=relative_residual_max,
                        ) from error
                    total_direct_solves += 1
                else:
                    try:
                        increment = self._dense_nonsymmetric_solve(
                            linearization,
                            rhs,
                            inverse_beta_dt2,
                            acceleration_load_action,
                        )
                    except Q16StructuralStepStopped as error:
                        raise Q16StructuralStepStopped(
                            str(error),
                            phase=error.phase,
                            newton_iteration_count=newton_iteration,
                            cg_iteration_count=total_cg_iterations,
                            gmres_iteration_count=(
                                total_gmres_iterations
                                + error.gmres_iteration_count
                            ),
                            relative_residual_max=relative_residual_max,
                        ) from error
                    total_direct_solves += 1
            else:
                try:
                    increment, cg_iterations = self._cg_solve(
                        linearization,
                        rhs,
                        inverse_beta_dt2,
                        acceleration_load_action,
                    )
                except Q16StructuralStepStopped as error:
                    if (
                        error.phase == "linear_solve"
                        and "non-positive curvature" in str(error)
                        and rhs.shape[0] == 1
                    ):
                        total_cg_iterations += error.cg_iteration_count
                        indefinite_fallback_count += 1
                        try:
                            increment, gmres_iterations = self._gmres_solve(
                                linearization,
                                rhs,
                                inverse_beta_dt2,
                                acceleration_load_action,
                            )
                        except Q16StructuralStepStopped as gmres_error:
                            raise Q16StructuralStepStopped(
                                str(gmres_error),
                                phase=gmres_error.phase,
                                newton_iteration_count=newton_iteration,
                                cg_iteration_count=total_cg_iterations,
                                gmres_iteration_count=(
                                    total_gmres_iterations
                                    + gmres_error.gmres_iteration_count
                                ),
                                relative_residual_max=relative_residual_max,
                            ) from gmres_error
                        total_gmres_iterations += gmres_iterations
                    else:
                        raise Q16StructuralStepStopped(
                            str(error),
                            phase=error.phase,
                            newton_iteration_count=newton_iteration,
                            cg_iteration_count=(
                                total_cg_iterations + error.cg_iteration_count
                            ),
                            gmres_iteration_count=(
                                total_gmres_iterations
                                + error.gmres_iteration_count
                            ),
                            relative_residual_max=relative_residual_max,
                        ) from error
                else:
                    total_cg_iterations += cg_iterations
            accepted_trial = None
            factor = 1.0
            for _ in range(17):
                trial = wp.zeros_like(current)
                wp.launch(
                    _q16_state_trial_kernel,
                    dim=current.shape,
                    inputs=[current, increment, DTYPE(factor), active],
                    outputs=[trial],
                    device=self.device,
                )
                trial = self._boundary_operator._enforce_state_prechecked(trial)
                try:
                    trial_linearization = self._operator._linearization_prechecked(
                        trial
                    )
                except FloatingPointError as error:
                    if (
                        str(error)
                        != "Q16 current geometry is orientation reversing or singular"
                    ):
                        raise
                    globalize_newton = True
                    factor *= 0.5
                    continue
                if not globalize_newton:
                    accepted_trial = trial
                    break
                trial_acceleration, trial_velocity = self._kinematics(
                    trial,
                    state_predictor,
                    velocity_predictor,
                    inverse_beta_dt2,
                    gamma_dt,
                )
                _, trial_residual = self._residual(
                    trial_linearization,
                    trial_acceleration,
                    trial_velocity,
                    external_force,
                    acceleration_load_action,
                )
                trial_relatives = tuple(
                    residual / scale
                    for residual, scale in zip(
                        self._norms(trial_residual), scales, strict=True
                    )
                )
                if (
                    max(trial_relatives)
                    <= (1.0 - 1.0e-4 * factor) * relative_residual_max
                ):
                    accepted_trial = trial
                    break
                factor *= 0.5
            if accepted_trial is None:
                raise Q16StructuralStepStopped(
                    "CUDA structural Newton line search did not find an "
                    "admissible residual-decreasing state; residual "
                    f"history={newton_residual_history!r}",
                    phase="line_search",
                    newton_iteration_count=newton_iteration,
                    cg_iteration_count=total_cg_iterations,
                    gmres_iteration_count=total_gmres_iterations,
                    relative_residual_max=relative_residual_max,
                    newton_residual_history=tuple(newton_residual_history),
                )
            current = accepted_trial

        raise AssertionError("unreachable Q16 Newton loop exit")


__all__ = [
    "Q16CudaNewmarkStepper",
    "Q16StructuralStepResult",
    "Q16StructuralStepStopped",
    "Q16StructuralWorkBalance",
]
