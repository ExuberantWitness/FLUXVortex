"""Unified potential-jump pressure and one-time panel-force assembly.

All vortical states influence one pressure ledger. They may provide local
induced velocity or a physically identified potential-jump-rate channel, but
they may not add a separate total force:

    delta_p = rho * (V_t . grad_s(mu) + sum_k d(mu_k)/dt)
    f_panel = delta_p * area * normal

The sign follows the existing FLUXV/N1 convention. No pressure cap, stall
blend, empirical normal force, or target normalization exists here.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


class UnifiedPressureError(ValueError):
    """Invalid panel field or duplicated/inconsistent pressure channel."""


def _finite(
    name: str,
    value,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise UnifiedPressureError(
            f"{name} must have ndim={ndim}, got shape {array.shape}"
        )
    if shape is not None and array.shape != shape:
        raise UnifiedPressureError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise UnifiedPressureError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class PressureLedgerReport:
    pressure_residual: float
    force_residual: float
    passed: bool


@dataclass(frozen=True)
class UnifiedPanelPressure:
    total_pressure: np.ndarray
    pressure_channels: Mapping[str, np.ndarray]
    total_force: np.ndarray
    force_channels: Mapping[str, np.ndarray]
    area: np.ndarray
    normal: np.ndarray

    def ledger_report(self, *, tolerance: float = 1.0e-12) -> PressureLedgerReport:
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise UnifiedPressureError(
                "tolerance must be finite and non-negative"
            )
        pressure_sum = np.sum(
            np.stack(tuple(self.pressure_channels.values())),
            axis=0,
        )
        force_sum = np.sum(
            np.stack(tuple(self.force_channels.values())),
            axis=0,
        )
        pressure_residual = float(
            np.max(np.abs(pressure_sum - self.total_pressure), initial=0.0)
        )
        force_residual = float(
            np.max(np.abs(force_sum - self.total_force), initial=0.0)
        )
        return PressureLedgerReport(
            pressure_residual=pressure_residual,
            force_residual=force_residual,
            passed=max(pressure_residual, force_residual) <= tolerance,
        )


def unified_panel_pressure(
    *,
    density: float,
    local_velocity,
    surface_gradient,
    potential_rate_channels: Mapping[str, np.ndarray],
    area,
    normal,
) -> UnifiedPanelPressure:
    """Assemble named pressure channels and convert to force exactly once."""
    if not np.isfinite(density) or density <= 0.0:
        raise UnifiedPressureError(
            f"density must be positive and finite, got {density}"
        )
    velocity = _finite("local_velocity", local_velocity, ndim=2)
    gradient = _finite(
        "surface_gradient",
        surface_gradient,
        shape=velocity.shape,
    )
    if velocity.shape[1] != 3:
        raise UnifiedPressureError(
            f"local_velocity must have shape (n,3), got {velocity.shape}"
        )
    panel_area = _finite("area", area, shape=(len(velocity),))
    panel_normal = _finite(
        "normal",
        normal,
        shape=velocity.shape,
    )
    if np.any(panel_area <= 0.0):
        raise UnifiedPressureError("panel area must be positive")
    normal_norm = np.linalg.norm(panel_normal, axis=1)
    if np.any(normal_norm <= 0.0):
        raise UnifiedPressureError("panel normal contains a zero vector")
    if np.max(np.abs(normal_norm - 1.0), initial=0.0) > 1.0e-10:
        raise UnifiedPressureError("panel normal must be unit length")
    if not potential_rate_channels:
        raise UnifiedPressureError(
            "at least one identified potential-rate channel is required"
        )

    channels: dict[str, np.ndarray] = {
        "surface_advection": density * np.einsum(
            "ij,ij->i",
            velocity,
            gradient,
        )
    }
    for raw_name, raw_value in potential_rate_channels.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise UnifiedPressureError(
                "pressure channel names must be non-empty strings"
            )
        name = raw_name.strip()
        if name == "surface_advection" or name in channels:
            raise UnifiedPressureError(
                f"duplicate/reserved pressure channel {name!r}"
            )
        channels[name] = density * _finite(
            f"potential_rate_channels[{name!r}]",
            raw_value,
            shape=(len(velocity),),
        )

    total_pressure = np.sum(np.stack(tuple(channels.values())), axis=0)
    force_channels = {
        name: pressure[:, None] * panel_area[:, None] * panel_normal
        for name, pressure in channels.items()
    }
    total_force = (
        total_pressure[:, None] * panel_area[:, None] * panel_normal
    )
    result = UnifiedPanelPressure(
        total_pressure=total_pressure,
        pressure_channels=MappingProxyType(
            {name: value.copy() for name, value in channels.items()}
        ),
        total_force=total_force,
        force_channels=MappingProxyType(
            {name: value.copy() for name, value in force_channels.items()}
        ),
        area=panel_area.copy(),
        normal=panel_normal.copy(),
    )
    if not result.ledger_report().passed:
        raise UnifiedPressureError("pressure/force ledger failed to close")
    return result


def structured_uvlm_surface_gradient(
    potential_jump,
    *,
    chord_tangent,
    span_tangent,
    chord_step,
    span_step,
    nc: int,
    ns: int,
) -> np.ndarray:
    """Reproduce the frozen N1 structured-grid ``grad_s(Gamma)`` stencil.

    This adapter exists only for equivalence tests and migration. Higher-order
    surface states should supply their own consistent surface gradient.
    """
    if nc <= 0 or ns <= 0:
        raise UnifiedPressureError("nc and ns must be positive")
    count = nc * ns
    jump = _finite("potential_jump", potential_jump, shape=(count,))
    tc = _finite("chord_tangent", chord_tangent, shape=(count, 3))
    ts = _finite("span_tangent", span_tangent, shape=(count, 3))
    dx = _finite("chord_step", chord_step, shape=(count,))
    dy = _finite("span_step", span_step, shape=(count,))
    if np.any(dx <= 0.0) or np.any(dy <= 0.0):
        raise UnifiedPressureError("surface step lengths must be positive")
    tc_norm = np.linalg.norm(tc, axis=1)
    ts_norm = np.linalg.norm(ts, axis=1)
    if (
        np.max(np.abs(tc_norm - 1.0), initial=0.0) > 1.0e-10
        or np.max(np.abs(ts_norm - 1.0), initial=0.0) > 1.0e-10
    ):
        raise UnifiedPressureError("surface tangents must be unit length")

    gamma = jump.reshape(nc, ns)
    dx_grid = dx.reshape(nc, ns)
    dy_grid = dy.reshape(nc, ns)
    dgamma_dx = np.empty((nc, ns))
    dgamma_dx[0] = gamma[0] / dx_grid[0]
    if nc > 1:
        dgamma_dx[1:] = (gamma[1:] - gamma[:-1]) / dx_grid[1:]
    dgamma_dy = np.zeros((nc, ns))
    if ns > 1:
        dgamma_dy[:, 0] = gamma[:, 0] / dy_grid[:, 0]
        dgamma_dy[:, -1] = -gamma[:, -1] / dy_grid[:, -1]
        dgamma_dy[:, 1:-1] = (
            (gamma[:, 2:] - gamma[:, :-2])
            / (2.0 * dy_grid[:, 1:-1])
        )
    return (
        dgamma_dx.reshape(-1, 1) * tc
        + dgamma_dy.reshape(-1, 1) * ts
    )

