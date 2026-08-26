"""Meng 2025 flapping wing case for the unified framework.

U4 (plan §11.1/§14): two rigid wings (left/right) on ONE fixed body with
prescribed flap+pitch joints, solved as a single multi-surface V5M system so
left/right mutual induction is part of the same AIC (plan §7.1: one body,
multiple surfaces — the 6-DOF owner is the aircraft body, never per-wing).

The numeric wind-tunnel condition below is the frozen Figure-16 nominal from
the source-audited platform adapter (``meng2025_case.py``); full flap/pitch
motion wiring lands with the U4-F adapter.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MengCaseConfig:
    case_id: str = "MENG-2025"
    # Two wings (left/right), one body, prescribed flap+pitch joints
    n_surfaces: int = 2
    surface_ids: tuple[str, ...] = ("left_wing", "right_wing")
    body_id: str = "body_0"

    # Nominal Figure-16 wind-tunnel condition (frozen platform adapter).
    half_span_m: float = 0.800
    root_chord_m: float = 0.287
    flap_amplitude_peak_to_peak_deg: float = 45.0
    freestream_m_s: float = 8.0
    installation_aoa_deg: float = 5.0
    frequency_hz: float = 2.0
    rho_kg_m3: float = 1.225
    nu_m2_s: float = 1.50e-5

    # Flapping/pitch parameters loaded from the existing adapter when wired


MENG_PRIMARY = MengCaseConfig()
