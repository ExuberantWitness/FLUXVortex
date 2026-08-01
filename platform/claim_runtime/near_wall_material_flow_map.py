"""No-load flow-map integrator for near-wall fluid material surfaces.

This is the trajectory part of the N2.6c1b separation oracle.  It advances
fluid material labels under a supplied smooth velocity field as required by
Eq. (2.3) and Algorithm 1 of Santhosh et al. (JFM 969, A25, 2023).

The module is intentionally separate from DDE wake-sheet advection: a
near-wall fluid material surface is a diagnostic set of fluid particles, not
a self-induced vortex sheet and not a structure mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class NearWallFlowMapError(ValueError):
    """Invalid material positions, integration interval, or velocity field."""


VelocityField = Callable[[np.ndarray, float], np.ndarray]


def _positions(name: str, value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim < 2 or result.shape[-1] != 3:
        raise NearWallFlowMapError(
            f"{name} must have shape (...,3), got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise NearWallFlowMapError(f"{name} contains non-finite values")
    return result.copy()


def _sample_velocity(
    velocity_field: VelocityField,
    position: np.ndarray,
    time: float,
) -> np.ndarray:
    try:
        value = velocity_field(position.copy(), float(time))
    except Exception as error:
        raise NearWallFlowMapError(
            f"velocity_field failed at t={time:.17g}"
        ) from error
    result = np.asarray(value, dtype=float)
    if result.shape != position.shape:
        raise NearWallFlowMapError(
            "velocity_field must return the material-position shape "
            f"{position.shape}, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise NearWallFlowMapError(
            f"velocity_field returned non-finite values at t={time:.17g}"
        )
    return result


@dataclass(frozen=True)
class NearWallMaterialFlowMap:
    """Discrete trajectory history retaining the initial material topology."""

    time: np.ndarray
    position: np.ndarray
    step_size: float
    step_count: int
    velocity_evaluation_count: int
    initial_shape: tuple[int, ...]

    @property
    def initial_position(self) -> np.ndarray:
        return self.position[0].copy()

    @property
    def final_position(self) -> np.ndarray:
        return self.position[-1].copy()


def integrate_near_wall_material_flow_map(
    initial_position,
    *,
    initial_time: float,
    final_time: float,
    maximum_step: float,
    velocity_field: VelocityField,
) -> NearWallMaterialFlowMap:
    """Integrate ``dx/dt=f(x,t)`` with classical fixed-step RK4.

    The step is shortened uniformly so that the last stored state lands
    exactly on ``final_time``.  Every leading array index is a material label
    and is carried unchanged throughout the history.
    """
    initial = _positions("initial_position", initial_position)
    t0 = float(initial_time)
    t1 = float(final_time)
    step_limit = float(maximum_step)
    if (
        not np.isfinite(t0)
        or not np.isfinite(t1)
        or t1 <= t0
    ):
        raise NearWallFlowMapError(
            "final_time must be finite and greater than initial_time"
        )
    if not np.isfinite(step_limit) or step_limit <= 0.0:
        raise NearWallFlowMapError(
            "maximum_step must be finite and positive"
        )
    if not callable(velocity_field):
        raise NearWallFlowMapError("velocity_field must be callable")

    step_count = int(np.ceil((t1-t0)/step_limit))
    dt = (t1-t0)/step_count
    time = np.linspace(t0, t1, step_count+1)
    history = np.empty((step_count+1,)+initial.shape, dtype=float)
    history[0] = initial
    state = initial
    for index in range(step_count):
        current_time = float(time[index])
        k1 = _sample_velocity(velocity_field, state, current_time)
        k2 = _sample_velocity(
            velocity_field,
            state+0.5*dt*k1,
            current_time+0.5*dt,
        )
        k3 = _sample_velocity(
            velocity_field,
            state+0.5*dt*k2,
            current_time+0.5*dt,
        )
        k4 = _sample_velocity(
            velocity_field,
            state+dt*k3,
            current_time+dt,
        )
        state = state+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
        if not np.all(np.isfinite(state)):
            raise NearWallFlowMapError(
                f"material trajectory became non-finite at step {index+1}"
            )
        history[index+1] = state

    return NearWallMaterialFlowMap(
        time=time,
        position=history,
        step_size=float(dt),
        step_count=step_count,
        velocity_evaluation_count=4*step_count,
        initial_shape=initial.shape,
    )
