"""Lazy bridge to the frozen Rojratsirikul platform adapter.

``src/fluxvortex`` must not import the platform tree at module import time
(the dependency direction is platform -> src).  The unified case schema
projects onto the frozen adapter dataclass only at call time; formal runs
always set ``PYTHONPATH=src:platform:platform/warp_vpm`` so the projection
resolves.  If the platform tree is missing the bridge fails fast instead of
silently falling back to a second geometry implementation.
"""
from __future__ import annotations

from typing import Any


def platform_roj_module() -> Any:
    try:
        from forward_flight_benchmarks import rojratsirikul2011_q16 as module
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError(
            "the frozen Rojratsirikul platform adapter is not importable; "
            "formal runs require PYTHONPATH=src:platform:platform/warp_vpm"
        ) from error
    return module


def platform_case_cls() -> Any:
    """The frozen ``Rojratsirikul2011MembraneCase`` dataclass."""

    return platform_roj_module().Rojratsirikul2011MembraneCase


def make_platform_case(
    case_id: str,
    angle_deg: float,
    *,
    zmax_over_c: float,
    cn_low: float,
    cn_high: float,
    strouhal: float | None = None,
    chordwise_peaks: int | None = None,
    spanwise_peaks: int | None = None,
    purpose: str = "unified-case projection",
    **shared: float,
) -> Any:
    """Project the unified frozen constants onto the adapter dataclass."""

    module = platform_roj_module()
    return module.Rojratsirikul2011MembraneCase(
        case_id=case_id,
        angle_deg=angle_deg,
        purpose=purpose,
        digitized_approx_zmax_over_c=zmax_over_c,
        digitized_approx_cn_low=cn_low,
        digitized_approx_cn_high=cn_high,
        digitized_approx_strouhal=strouhal,
        digitized_approx_chordwise_peak_count=chordwise_peaks,
        digitized_approx_spanwise_peak_count=spanwise_peaks,
        **shared,
    )


__all__ = ["make_platform_case", "platform_case_cls", "platform_roj_module"]
