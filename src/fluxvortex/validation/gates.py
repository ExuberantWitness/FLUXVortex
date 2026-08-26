"""Acceptance gates using corrected observers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class GateResult:
    gate_name: str
    passed: bool
    value: float | None = None
    target: float | None = None
    tolerance: float | None = None
    detail: str = ""

def camber_gate(mean_map_max_over_c: float, target: float, tolerance: float) -> GateResult:
    return GateResult(
        gate_name="mean_camber_max",
        value=mean_map_max_over_c,
        target=target,
        tolerance=tolerance,
        passed=abs(mean_map_max_over_c - target) <= tolerance,
    )

def cn_gate(mean_cn: float, band: tuple[float, float], rel_tol: float) -> GateResult:
    lo, hi = band
    passed = 0.9 * lo <= mean_cn <= 1.1 * hi
    return GateResult(
        gate_name="mean_cn",
        value=mean_cn,
        target=(lo + hi) / 2,
        tolerance=rel_tol,
        passed=passed,
        detail=f"band [{lo},{hi}] with 10% rel tol",
    )

def no_sign_crossing_gate(crossings: int) -> GateResult:
    return GateResult(
        gate_name="no_sign_crossing",
        value=float(crossings),
        target=0.0,
        passed=crossings == 0,
    )
