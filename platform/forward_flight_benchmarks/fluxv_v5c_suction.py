"""FluxV v5c1 rate-sensitive leading-edge suction-loss state.

The model is deliberately narrower than LDVM/SVDVM.  It uses the normalized
SVDVM leading-edge flux ratio to drive one causal, bounded state and applies
that state only to an explicitly supplied axial-suction coefficient.  It does
not infer suction from total UVLM force and does not own normal force, added
mass, profile drag, circulation, or wake evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SOURCE_FROZEN_STATE_POLE = 0.5


@dataclass(frozen=True)
class RateSensitiveSuctionParameters:
    """Source-frozen state parameters for the v5c1-RSLS candidate."""

    # Izraelevitz writes the one-state decay as
    # exp[2 * b * (U / c) * dt], with b=-0.25.  This module defines
    # delta_tau=(U / c) * dt, so the positive decay magnitude is 2*|b|=0.5.
    state_pole_per_convective_time: float = SOURCE_FROZEN_STATE_POLE

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.state_pole_per_convective_time)
            or self.state_pole_per_convective_time <= 0.0
        ):
            raise ValueError("state pole must be finite and positive")
        if self.state_pole_per_convective_time != SOURCE_FROZEN_STATE_POLE:
            raise ValueError(
                "v5c1 canonical parameter is source-frozen at 2*abs(b)=0.5; "
                "custom poles require a different exploratory model identity"
            )

    def manifest(self) -> dict[str, float | str]:
        return {
            "state_pole_per_convective_time": self.state_pole_per_convective_time,
            "state_pole_source": (
                "Izraelevitz 2017 one-state ULLT b=-0.25 mapped as 2*abs(b)=0.5 "
                "because delta_tau=(U/c)*dt; used only as a state time scale, "
                "not as a ULLT force"
            ),
            "delta_tau_definition": (
                "(|V_rel at 0.75c, perpendicular to local span|/c_local)*dt"
            ),
            "delta_tau_source_mapping": (
                "Izraelevitz t_tilde=(2/c)*integral(v_perp dt); this module uses "
                "Delta tau=integral(v_perp/c dt) and pole 2*abs(b)"
            ),
            "near_zero_convective_speed_policy": (
                "fail closed; adapters may not apply an undocumented positive floor"
            ),
            "flux_ratio_source": (
                "Martinez-Carmena SVDVM Gamma_dot_LE proportional to U^2*A0^2/rLE, "
                "normalized by the same-section critical flux"
            ),
            "rate_excitation_scale": 1.0,
            "rate_excitation_formula": (
                "E=G*abs(dJ/dtau), chi_equilibrium=E/(1+E); the unit scale is "
                "part of the frozen exploratory model identity"
            ),
            "force_owner": "explicit axial suction only",
            "observation_fit": "none",
        }


DEFAULT_SUCTION_PARAMETERS = RateSensitiveSuctionParameters()


@dataclass(frozen=True)
class RateSensitiveSuctionState:
    """Per-strip causal state and prior normalized flux-ratio bookkeeping."""

    chi: np.ndarray
    previous_j: np.ndarray
    previous_previous_j: np.ndarray
    previous_delta_tau: np.ndarray
    step_count: int = 0

    @classmethod
    def zeros(cls, strip_count: int) -> "RateSensitiveSuctionState":
        if strip_count < 1:
            raise ValueError("strip count must be positive")
        zeros = np.zeros(strip_count, dtype=float)
        return cls(
            chi=zeros.copy(),
            previous_j=zeros.copy(),
            previous_previous_j=zeros.copy(),
            previous_delta_tau=np.ones(strip_count, dtype=float),
            step_count=0,
        )


def _strip_array(value: np.ndarray | float, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or one-dimensional array")
    return array


def _same_shape(reference: np.ndarray, value: np.ndarray, name: str) -> None:
    if value.shape != reference.shape:
        raise ValueError(f"{name} does not match strip topology")


def _on_strip_topology(
    value: np.ndarray | float,
    reference: np.ndarray,
    name: str,
) -> np.ndarray:
    array = _strip_array(value, name)
    if array.size == 1 and reference.size != 1:
        array = np.full(reference.shape, float(array[0]), dtype=float)
    _same_shape(reference, array, name)
    return array


def ramesh_axial_suction_coefficient(
    a0_pre: np.ndarray | float,
    lesp_critical: np.ndarray | float,
) -> np.ndarray:
    """Return the capped Ramesh/Hirato axial suction ``CS=2*pi*A0^2``."""

    a0 = _strip_array(a0_pre, "A0")
    critical = _on_strip_topology(lesp_critical, a0, "LESP critical")
    if np.any(critical <= 0.0):
        raise ValueError("LESP critical must be positive")
    capped = np.minimum(np.abs(a0), critical)
    coefficient = 2.0 * np.pi * capped**2
    if not np.all(np.isfinite(coefficient)):
        raise ValueError("axial suction coefficient is not finite")
    return coefficient


def _backward_rate(
    current_j: np.ndarray,
    delta_tau: np.ndarray,
    state: RateSensitiveSuctionState,
) -> np.ndarray:
    if state.step_count == 0:
        return np.zeros_like(current_j)
    # A first-order causal difference is intentional.  A BDF2 derivative
    # overshoots immediately after a step in J; taking its absolute value then
    # rectifies that numerical overshoot into a second, non-physical excitation.
    rate = (current_j - state.previous_j) / delta_tau
    if not np.all(np.isfinite(rate)):
        raise ValueError("normalized flux-rate is not finite")
    return rate


def step_rate_sensitive_suction(
    state: RateSensitiveSuctionState,
    *,
    a0_pre: np.ndarray | float,
    lesp_critical: np.ndarray | float,
    delta_tau: np.ndarray | float,
    base_suction_coefficient: np.ndarray | float,
    enabled: bool = True,
    parameters: RateSensitiveSuctionParameters = DEFAULT_SUCTION_PARAMETERS,
) -> tuple[RateSensitiveSuctionState, dict[str, Any]]:
    """Advance one causal v5c1-RSLS step on every spanwise strip."""

    a0 = _strip_array(a0_pre, "A0")
    critical = _on_strip_topology(lesp_critical, a0, "LESP critical")
    convective_step = _on_strip_topology(delta_tau, a0, "convective step")
    base_cs = _on_strip_topology(
        base_suction_coefficient, a0, "base suction coefficient"
    )
    for value, name in (
        (critical, "LESP critical"),
        (convective_step, "convective step"),
        (base_cs, "base suction coefficient"),
        (np.asarray(state.chi, dtype=float), "state chi"),
        (np.asarray(state.previous_j, dtype=float), "previous J"),
        (np.asarray(state.previous_previous_j, dtype=float), "previous-previous J"),
        (np.asarray(state.previous_delta_tau, dtype=float), "previous convective step"),
    ):
        _same_shape(a0, value, name)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    if not isinstance(state.step_count, (int, np.integer)) or state.step_count < 0:
        raise ValueError("state step count must be a non-negative integer")
    if np.any(critical <= 0.0) or np.any(convective_step <= 0.0):
        raise ValueError("LESP critical and convective step must be positive")
    if np.any(base_cs < 0.0):
        raise ValueError("base suction coefficient cannot be negative")
    if np.any(state.chi < 0.0) or np.any(state.chi > 1.0):
        raise ValueError("incoming suction-loss state lies outside [0, 1]")

    if not enabled:
        zeros = np.zeros_like(base_cs)
        return state, {
            "enabled": False,
            "j": zeros.copy(),
            "j_rate": zeros.copy(),
            "supercritical_gate": zeros.copy(),
            "chi_equilibrium": zeros.copy(),
            "chi_before": np.asarray(state.chi).copy(),
            "chi_after": np.asarray(state.chi).copy(),
            "loss_fraction": zeros.copy(),
            "base_suction_coefficient": base_cs.copy(),
            "target_suction_coefficient": base_cs.copy(),
            "delta_suction_coefficient": zeros.copy(),
            "delta_normal_coefficient": zeros.copy(),
            "state_updated": False,
            "diagnostic_status": "not_evaluated_disabled",
        }

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = np.abs(a0) / critical
        j = ratio * ratio
    if not np.all(np.isfinite(j)):
        raise ValueError("normalized leading-edge flux ratio is not finite")

    gate = np.zeros_like(j)
    supercritical = j > 1.0
    gate[supercritical] = 1.0 - 1.0 / j[supercritical]
    j_rate = np.abs(_backward_rate(j, convective_step, state))
    excitation = gate * j_rate
    chi_equilibrium = excitation / (1.0 + excitation)
    decay = np.exp(-parameters.state_pole_per_convective_time * convective_step)
    chi_before = np.asarray(state.chi, dtype=float)
    chi_after = chi_equilibrium + (chi_before - chi_equilibrium) * decay
    chi_after = np.clip(chi_after, 0.0, 1.0)
    loss = np.clip(gate * chi_after, 0.0, 1.0)
    target_cs = (1.0 - loss) * base_cs
    delta_cs = target_cs - base_cs
    for value, name in (
        (gate, "supercritical gate"),
        (j_rate, "normalized flux-rate"),
        (chi_equilibrium, "equilibrium state"),
        (chi_after, "updated state"),
        (loss, "loss fraction"),
        (target_cs, "target suction coefficient"),
        (delta_cs, "suction-coefficient discrepancy"),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} is not finite")
    next_state = RateSensitiveSuctionState(
        chi=chi_after.copy(),
        previous_j=j.copy(),
        previous_previous_j=np.asarray(state.previous_j, dtype=float).copy(),
        previous_delta_tau=convective_step.copy(),
        step_count=state.step_count + 1,
    )
    return next_state, {
        "enabled": True,
        "j": j,
        "j_rate": j_rate,
        "supercritical_gate": gate,
        "chi_equilibrium": chi_equilibrium,
        "chi_before": chi_before.copy(),
        "chi_after": chi_after,
        "loss_fraction": loss,
        "base_suction_coefficient": base_cs.copy(),
        "target_suction_coefficient": target_cs,
        "delta_suction_coefficient": delta_cs,
        "delta_normal_coefficient": np.zeros_like(delta_cs),
        "state_updated": True,
    }


def project_axial_suction_loss_to_wind_axes(
    delta_suction_coefficient: np.ndarray | float,
    alpha_rad: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Project an axial suction reduction into lift/drag coefficients.

    ``delta_suction_coefficient`` is non-positive for a loss.  The projection
    follows Ramesh's ``CL=CN*cos(alpha)+CS*sin(alpha)`` and
    ``CD=CN*sin(alpha)-CS*cos(alpha)`` with ``delta_CN=0``.
    """

    delta_cs = np.asarray(delta_suction_coefficient, dtype=float)
    alpha = np.asarray(alpha_rad, dtype=float)
    delta_cs, alpha = np.broadcast_arrays(delta_cs, alpha)
    if not np.all(np.isfinite(delta_cs)) or not np.all(np.isfinite(alpha)):
        raise ValueError("suction loss and incidence must be finite")
    if np.any(delta_cs > 1.0e-15):
        raise ValueError("v5c1 may only reduce axial suction")
    return {
        "delta_CL": delta_cs * np.sin(alpha),
        "delta_CD": -delta_cs * np.cos(alpha),
        "delta_CN": np.zeros_like(delta_cs),
        "delta_CS": delta_cs.copy(),
    }


def run_rate_sensitive_suction_history(
    *,
    a0_pre: np.ndarray,
    lesp_critical: np.ndarray,
    delta_tau: np.ndarray,
    base_suction_coefficient: np.ndarray,
    enabled: bool = True,
    initial_state: RateSensitiveSuctionState | None = None,
    parameters: RateSensitiveSuctionParameters = DEFAULT_SUCTION_PARAMETERS,
) -> dict[str, Any]:
    """Run a time-by-strip history without looking ahead or wrapping phase."""

    a0 = np.asarray(a0_pre, dtype=float)
    critical = np.asarray(lesp_critical, dtype=float)
    convective_step = np.asarray(delta_tau, dtype=float)
    base_cs = np.asarray(base_suction_coefficient, dtype=float)
    if a0.ndim != 2:
        raise ValueError("A0 history must have shape (time, strip)")
    for value, name in (
        (critical, "LESP critical"),
        (convective_step, "convective step"),
        (base_cs, "base suction coefficient"),
    ):
        if value.shape != a0.shape:
            raise ValueError(f"{name} history must match A0 history")
    state = initial_state or RateSensitiveSuctionState.zeros(a0.shape[1])
    records: list[dict[str, Any]] = []
    for index in range(a0.shape[0]):
        state, record = step_rate_sensitive_suction(
            state,
            a0_pre=a0[index],
            lesp_critical=critical[index],
            delta_tau=convective_step[index],
            base_suction_coefficient=base_cs[index],
            enabled=enabled,
            parameters=parameters,
        )
        records.append(record)
    keys = (
        "j",
        "j_rate",
        "supercritical_gate",
        "chi_equilibrium",
        "chi_before",
        "chi_after",
        "loss_fraction",
        "base_suction_coefficient",
        "target_suction_coefficient",
        "delta_suction_coefficient",
        "delta_normal_coefficient",
    )
    return {
        key: np.stack([np.asarray(record[key]) for record in records]) for key in keys
    } | {
        "final_state": state,
        "enabled": bool(enabled),
        "parameters": parameters.manifest(),
        "causal_semantics": "forward time traversal only; no periodic roll or future phase",
    }
