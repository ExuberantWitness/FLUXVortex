"""Baik 2011 W1-W4 water-tunnel cases: 2D LDVM backend, unified framework.

U3 (plan §11.2): the quasi-2D water-tunnel cases keep the existing 2D CUDA
LDVM production backend; only the case definition, unified transactions,
provenance and result layer migrate onto the unified framework.  The full
kinematic/physical parameters remain owned by the frozen adapter
``platform/forward_flight_benchmarks/baik2012.py`` — this config carries only
the unified case identity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaikCaseConfig:
    case_id: str  # "W1", "W2", "W3", "W4"
    # W1: pure heave; W2-W4: non-sinusoidal heave with varying amplitudes
    # Full parameters come from the existing baik2012.py adapter
    description: str = ""


BAIK_CASES = {
    "W1": BaikCaseConfig(case_id="W1", description="pure heave"),
    "W2": BaikCaseConfig(case_id="W2", description="non-sinusoidal heave variant 2"),
    "W3": BaikCaseConfig(case_id="W3", description="non-sinusoidal heave variant 3"),
    "W4": BaikCaseConfig(case_id="W4", description="non-sinusoidal heave variant 4"),
}
