"""CUDA periodic-waveform oracle for sealed real Q16 FSI trajectories.

The oracle deliberately evaluates signed three-component source force and the
span-tip Q16 centroid displacement.  Norm-only histories cannot distinguish a
phase error from a periodic response and are therefore not accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from q16_real_fsi_trajectory import (
    Q16RealFSITrajectoryResult,
    validate_q16_real_fsi_trajectory,
)

_AUDIT_DOMAIN = "flux-v5m-q16-periodic-fsi-audit-v1"


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _relative_waveform_l2(previous: torch.Tensor, current: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(current - previous)
    denominator = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(previous),
            torch.linalg.vector_norm(current),
        ),
        torch.tensor(
            torch.finfo(torch.float64).tiny,
            dtype=torch.float64,
            device=previous.device,
        ),
    )
    value = numerator / denominator
    if not bool(torch.isfinite(value).item()):
        raise FloatingPointError("periodic waveform error became non-finite")
    return float(value.item())


@dataclass(frozen=True, slots=True)
class Q16PeriodicCycleComparison:
    previous_cycle: int
    current_cycle: int
    force_waveform_relative_l2: float
    span_tip_waveform_relative_l2: float


@dataclass(frozen=True, slots=True)
class Q16PeriodicFSIAudit:
    trajectory_result_sha256: str
    period: float
    steps_per_cycle: int
    cycle_count: int
    input_repeat_max_abs_error: float
    force_tolerance: float
    span_tip_tolerance: float
    comparisons: tuple[Q16PeriodicCycleComparison, ...]
    multi_cycle_integration_pass: bool
    observed_periodic_steady_pass: bool
    audit_sha256: str


def _audit_sha256(audit: Q16PeriodicFSIAudit) -> str:
    payload = {
        "trajectory_result_sha256": audit.trajectory_result_sha256,
        "period_hex": audit.period.hex(),
        "steps_per_cycle": audit.steps_per_cycle,
        "cycle_count": audit.cycle_count,
        "input_repeat_max_abs_error_hex": audit.input_repeat_max_abs_error.hex(),
        "force_tolerance_hex": audit.force_tolerance.hex(),
        "span_tip_tolerance_hex": audit.span_tip_tolerance.hex(),
        "comparisons": [
            {
                "previous_cycle": item.previous_cycle,
                "current_cycle": item.current_cycle,
                "force_waveform_relative_l2_hex": (
                    item.force_waveform_relative_l2.hex()
                ),
                "span_tip_waveform_relative_l2_hex": (
                    item.span_tip_waveform_relative_l2.hex()
                ),
            }
            for item in audit.comparisons
        ],
        "multi_cycle_integration_pass": audit.multi_cycle_integration_pass,
        "observed_periodic_steady_pass": audit.observed_periodic_steady_pass,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(_AUDIT_DOMAIN.encode("ascii") + b"\0" + encoded).hexdigest()


def audit_q16_periodic_fsi(
    trajectory: Q16RealFSITrajectoryResult,
    *,
    period: float,
    steps_per_cycle: int,
    input_repeat_tolerance: float,
    force_tolerance: float,
    span_tip_tolerance: float,
    device: str = "cuda:0",
) -> Q16PeriodicFSIAudit:
    """Evaluate phase-aligned adjacent-cycle waveforms on CUDA float64.

    ``multi_cycle_integration_pass`` means the sealed trajectory contains at
    least three complete cycles and its input repeats at the requested phases.
    ``observed_periodic_steady_pass`` additionally requires the frozen force,
    displacement and monotone cycle-to-cycle criteria.  It is intentionally an
    observable-level statement, not proof that the complete wake state repeats.
    """

    result = validate_q16_real_fsi_trajectory(trajectory)
    period_value = _positive_float("period", period)
    phase_count = _positive_int("steps_per_cycle", steps_per_cycle)
    input_tolerance = _positive_float(
        "input_repeat_tolerance", input_repeat_tolerance
    )
    force_limit = _positive_float("force_tolerance", force_tolerance)
    tip_limit = _positive_float("span_tip_tolerance", span_tip_tolerance)
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Q16 periodic FSI oracle requires CUDA; no CPU fallback")
    if result.completed_step_count % phase_count != 0:
        raise ValueError("trajectory does not contain an integer number of cycles")
    cycle_count = result.completed_step_count // phase_count
    if cycle_count < 3:
        raise ValueError("periodic FSI audit requires at least three complete cycles")
    period_error = abs(result.delta_time * phase_count - period_value)
    period_scale = max(1.0, period_value, result.delta_time * phase_count)
    if period_error > 64.0 * np.finfo(np.float64).eps * period_scale:
        raise ValueError("period and trajectory time discretization are inconsistent")

    inflow = torch.tensor(
        [record.operating_point_velocity for record in result.records],
        dtype=torch.float64,
        device=selected,
    ).reshape(cycle_count, phase_count)
    force = torch.tensor(
        [
            (
                record.aerodynamic_force_x_w,
                record.aerodynamic_force_y_w,
                record.aerodynamic_force_z_w,
            )
            for record in result.records
        ],
        dtype=torch.float64,
        device=selected,
    ).reshape(cycle_count, phase_count, 3)
    tip = torch.tensor(
        [
            (
                record.span_tip_centroid_displacement_x_w,
                record.span_tip_centroid_displacement_y_w,
                record.span_tip_centroid_displacement_z_w,
            )
            for record in result.records
        ],
        dtype=torch.float64,
        device=selected,
    ).reshape(cycle_count, phase_count, 3)
    if not all(bool(torch.isfinite(value).all().item()) for value in (inflow, force, tip)):
        raise FloatingPointError("periodic FSI history contains a non-finite value")

    input_repeat = torch.max(torch.abs(inflow[1:] - inflow[0:1]))
    input_repeat_error = float(input_repeat.item())
    comparisons = tuple(
        Q16PeriodicCycleComparison(
            previous_cycle=cycle,
            current_cycle=cycle + 1,
            force_waveform_relative_l2=_relative_waveform_l2(
                force[cycle - 1], force[cycle]
            ),
            span_tip_waveform_relative_l2=_relative_waveform_l2(
                tip[cycle - 1], tip[cycle]
            ),
        )
        for cycle in range(1, cycle_count)
    )
    integration_pass = input_repeat_error <= input_tolerance
    final = comparisons[-1]
    previous = comparisons[-2]
    trend_slack = 1.0e-12
    periodic_pass = bool(
        integration_pass
        and final.force_waveform_relative_l2 <= force_limit
        and final.span_tip_waveform_relative_l2 <= tip_limit
        and final.force_waveform_relative_l2
        <= previous.force_waveform_relative_l2 + trend_slack
        and final.span_tip_waveform_relative_l2
        <= previous.span_tip_waveform_relative_l2 + trend_slack
    )
    audit = Q16PeriodicFSIAudit(
        trajectory_result_sha256=result.result_sha256,
        period=period_value,
        steps_per_cycle=phase_count,
        cycle_count=cycle_count,
        input_repeat_max_abs_error=input_repeat_error,
        force_tolerance=force_limit,
        span_tip_tolerance=tip_limit,
        comparisons=comparisons,
        multi_cycle_integration_pass=integration_pass,
        observed_periodic_steady_pass=periodic_pass,
        audit_sha256="0" * 64,
    )
    return Q16PeriodicFSIAudit(
        trajectory_result_sha256=audit.trajectory_result_sha256,
        period=audit.period,
        steps_per_cycle=audit.steps_per_cycle,
        cycle_count=audit.cycle_count,
        input_repeat_max_abs_error=audit.input_repeat_max_abs_error,
        force_tolerance=audit.force_tolerance,
        span_tip_tolerance=audit.span_tip_tolerance,
        comparisons=audit.comparisons,
        multi_cycle_integration_pass=audit.multi_cycle_integration_pass,
        observed_periodic_steady_pass=audit.observed_periodic_steady_pass,
        audit_sha256=_audit_sha256(audit),
    )


__all__ = [
    "Q16PeriodicCycleComparison",
    "Q16PeriodicFSIAudit",
    "audit_q16_periodic_fsi",
]
