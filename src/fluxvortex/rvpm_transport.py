"""Isolated FLOWVPM-compatible reformulated VPM transport primitives.

The implementation is a direct Float64 reference backend for FluxV v5h.  It
does not alter or call the legacy ``VortexParticleField`` and deliberately has
no limiter, clipping, SFS, viscosity, FMM, or force coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rvpm_reference import (
    DirectField,
    direct_gaussian_erf_velocity_jacobian,
    validate_particle_state,
)

FloatArray = NDArray[np.float64]
RK_A = (0.0, -5.0 / 9.0, -153.0 / 128.0)
RK_B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)


def _real_float_array(name: str, value: ArrayLike) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf":
        raise ValueError(f"{name} must use a real numeric dtype")
    array = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class ParticleState:
    positions: FloatArray
    gamma: FloatArray
    sigma: FloatArray


@dataclass(frozen=True)
class RVPMStepRHS:
    velocity: FloatArray
    jacobian: FloatArray
    stretching: FloatArray
    z_rate: FloatArray
    gamma_rate: FloatArray
    sigma_rate: FloatArray


@dataclass(frozen=True)
class LSRKStageRecord:
    stage: int
    a: float
    b: float
    pre: ParticleState
    rhs: RVPMStepRHS
    post: ParticleState
    storage_pre: FloatArray
    storage_post: FloatArray


def make_particle_state(
    positions: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
) -> ParticleState:
    """Validate and copy a particle state."""

    position_array, gamma_array, sigma_array = validate_particle_state(
        positions,
        gamma,
        sigma,
    )
    return ParticleState(
        positions=position_array.copy(),
        gamma=gamma_array.copy(),
        sigma=sigma_array.copy(),
    )


def _storage_matrix(
    position_storage: FloatArray,
    gamma_storage: FloatArray,
    sigma_storage: FloatArray,
) -> FloatArray:
    storage = np.zeros((position_storage.shape[0], 9), dtype=np.float64)
    storage[:, 0:3] = position_storage
    storage[:, 3:6] = gamma_storage
    storage[:, 7] = sigma_storage
    return storage


def reformulated_vpm_rhs(
    gamma: ArrayLike,
    sigma: ArrayLike,
    jacobian: ArrayLike,
    *,
    formulation_f: float = 0.0,
    formulation_g: float = 0.2,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(S, Z, dGamma_dt, dSigma_dt)`` for inviscid no-SFS rVPM.

    FLOWVPM's ``transposed=true`` convention is ``S = J.T @ Gamma`` for a
    Jacobian stored as ``J[velocity_component, coordinate_component]``.
    """

    gamma_array = _real_float_array("gamma", gamma)
    sigma_array = _real_float_array("sigma", sigma)
    jacobian_array = _real_float_array("jacobian", jacobian)
    if gamma_array.ndim != 2 or gamma_array.shape[1:] != (3,):
        raise ValueError("gamma must have shape (n, 3)")
    if sigma_array.shape != (gamma_array.shape[0],):
        raise ValueError("sigma must have shape (n,)")
    if jacobian_array.shape != (gamma_array.shape[0], 3, 3):
        raise ValueError("jacobian must have shape (n, 3, 3)")
    if np.any(sigma_array <= 0.0):
        raise ValueError("sigma must be strictly positive")
    if isinstance(formulation_f, (bool, np.bool_)) or isinstance(
        formulation_g, (bool, np.bool_)
    ):
        raise ValueError("formulation parameters must be real numbers, not booleans")
    if not np.isfinite(formulation_f) or not np.isfinite(formulation_g):
        raise ValueError("formulation parameters must be finite")
    denominator = 1.0 + 3.0 * formulation_f
    if denominator == 0.0:
        raise ValueError("1 + 3*f must be nonzero")

    stretching = np.einsum("nji,nj->ni", jacobian_array, gamma_array)
    gamma_norm_squared = np.einsum("ni,ni->n", gamma_array, gamma_array)
    z_rate = np.zeros(gamma_array.shape[0], dtype=np.float64)
    nonzero = gamma_norm_squared > 0.0
    z_rate[nonzero] = (
        (formulation_f + formulation_g)
        / denominator
        * np.einsum(
            "ni,ni->n",
            stretching[nonzero],
            gamma_array[nonzero],
        )
        / gamma_norm_squared[nonzero]
    )
    gamma_rate = stretching - 3.0 * z_rate[:, None] * gamma_array
    sigma_rate = -sigma_array * z_rate
    if not (
        np.all(np.isfinite(stretching))
        and np.all(np.isfinite(z_rate))
        and np.all(np.isfinite(gamma_rate))
        and np.all(np.isfinite(sigma_rate))
    ):
        raise FloatingPointError("reformulated VPM RHS produced non-finite values")
    return stretching, z_rate, gamma_rate, sigma_rate


def lsrk3_step_direct(
    state: ParticleState,
    delta_time: float,
    *,
    freestream_velocity: ArrayLike = (0.0, 0.0, 0.0),
) -> tuple[ParticleState, tuple[LSRKStageRecord, ...]]:
    """Advance one FLOWVPM low-storage RK3 step with direct U/J evaluation."""

    if isinstance(delta_time, (bool, np.bool_)):
        raise ValueError("delta_time must be a real number, not a boolean")
    if not np.isfinite(delta_time) or delta_time <= 0.0:
        raise ValueError("delta_time must be finite and positive")
    validated = make_particle_state(state.positions, state.gamma, state.sigma)
    positions = validated.positions
    gamma = validated.gamma
    sigma = validated.sigma
    freestream = _real_float_array("freestream_velocity", freestream_velocity)
    if freestream.shape != (3,):
        raise ValueError("freestream_velocity must be a finite length-3 vector")

    position_storage = np.zeros_like(positions)
    gamma_storage = np.zeros_like(gamma)
    sigma_storage = np.zeros_like(sigma)
    records: list[LSRKStageRecord] = []

    for stage, (a_coefficient, b_coefficient) in enumerate(
        zip(RK_A, RK_B, strict=True),
        start=1,
    ):
        pre = make_particle_state(positions, gamma, sigma)
        storage_pre = _storage_matrix(
            position_storage,
            gamma_storage,
            sigma_storage,
        )
        field: DirectField = direct_gaussian_erf_velocity_jacobian(
            positions,
            gamma,
            sigma,
        )
        stretching, z_rate, gamma_rate, sigma_rate = reformulated_vpm_rhs(
            gamma,
            sigma,
            field.jacobian,
        )
        rhs = RVPMStepRHS(
            velocity=field.velocity.copy(),
            jacobian=field.jacobian.copy(),
            stretching=stretching.copy(),
            z_rate=z_rate.copy(),
            gamma_rate=gamma_rate.copy(),
            sigma_rate=sigma_rate.copy(),
        )

        position_storage = a_coefficient * position_storage + delta_time * (
            field.velocity + freestream[None, :]
        )
        gamma_storage = a_coefficient * gamma_storage + delta_time * gamma_rate
        sigma_storage = a_coefficient * sigma_storage + delta_time * sigma_rate
        positions = positions + b_coefficient * position_storage
        gamma = gamma + b_coefficient * gamma_storage
        sigma = sigma + b_coefficient * sigma_storage
        if not (
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(gamma))
            and np.all(np.isfinite(sigma))
        ):
            raise FloatingPointError(f"non-finite state after RK stage {stage}")
        if np.any(sigma <= 0.0):
            raise FloatingPointError(f"non-positive sigma after RK stage {stage}")

        post = make_particle_state(positions, gamma, sigma)
        storage_post = _storage_matrix(
            position_storage,
            gamma_storage,
            sigma_storage,
        )
        records.append(
            LSRKStageRecord(
                stage=stage,
                a=a_coefficient,
                b=b_coefficient,
                pre=pre,
                rhs=rhs,
                post=post,
                storage_pre=storage_pre,
                storage_post=storage_post,
            )
        )

    return make_particle_state(positions, gamma, sigma), tuple(records)


def corrected_pedrizzetti(
    gamma: ArrayLike,
    jacobian: ArrayLike,
    alpha: float,
) -> FloatArray:
    """Apply source-faithful corrected Pedrizzetti direction relaxation.

    The normalization factor is computed from the old ``gamma``.  Zero local
    vorticity is an exact no-op.  A zero-strength particle with nonzero local
    vorticity is undefined upstream and fails closed here.
    """

    gamma_array = _real_float_array("gamma", gamma)
    jacobian_array = _real_float_array("jacobian", jacobian)
    scalar_input = gamma_array.ndim == 1
    if scalar_input:
        gamma_array = gamma_array[None, :]
        jacobian_array = jacobian_array[None, :, :]
    if gamma_array.ndim != 2 or gamma_array.shape[1:] != (3,):
        raise ValueError("gamma must have shape (n, 3) or (3,)")
    if jacobian_array.shape != (gamma_array.shape[0], 3, 3):
        raise ValueError("jacobian must have shape (n, 3, 3) or (3, 3)")
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be a real number, not a boolean")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    vorticity = np.column_stack(
        (
            jacobian_array[:, 2, 1] - jacobian_array[:, 1, 2],
            jacobian_array[:, 0, 2] - jacobian_array[:, 2, 0],
            jacobian_array[:, 1, 0] - jacobian_array[:, 0, 1],
        )
    )
    gamma_norm = np.linalg.norm(gamma_array, axis=1)
    vorticity_norm = np.linalg.norm(vorticity, axis=1)
    invalid = (gamma_norm == 0.0) & (vorticity_norm != 0.0)
    if np.any(invalid):
        raise ValueError("zero gamma with nonzero local vorticity is undefined")

    result = gamma_array.copy()
    active = vorticity_norm != 0.0
    if np.any(active):
        old_gamma = gamma_array[active]
        old_norm = gamma_norm[active]
        omega = vorticity[active]
        omega_norm = vorticity_norm[active]
        cosine = np.einsum("ni,ni->n", old_gamma, omega) / (old_norm * omega_norm)
        b_squared = 1.0 - 2.0 * (1.0 - alpha) * alpha * (1.0 - cosine)
        if np.any(~np.isfinite(b_squared)) or np.any(b_squared <= 0.0):
            raise FloatingPointError("corrected Pedrizzetti normalization is singular")
        mixed = (1.0 - alpha) * old_gamma + alpha * old_norm[
            :, None
        ] * omega / omega_norm[:, None]
        result[active] = mixed / np.sqrt(b_squared)[:, None]

    if not np.all(np.isfinite(result)):
        raise FloatingPointError("corrected Pedrizzetti produced non-finite gamma")
    if scalar_input:
        return result[0]
    return result


__all__ = [
    "LSRKStageRecord",
    "ParticleState",
    "RK_A",
    "RK_B",
    "RVPMStepRHS",
    "corrected_pedrizzetti",
    "lsrk3_step_direct",
    "make_particle_state",
    "reformulated_vpm_rhs",
]
