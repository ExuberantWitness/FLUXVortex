from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.cases import YANG_2025
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement
from forward_flight_benchmarks.run_fluxv_v5d1_source_clock import (
    OUTPUT_SAMPLES,
    OWNER_REFERENCE_FRACTION_CHORD,
    _gate,
    _source_owner,
)


def test_source_owner_uses_local_half_chord_clock_and_is_periodic() -> None:
    movement, _ = build_yang2025_movement(
        15.0,
        settings=(2, 3, 32, 2, 2),
    )
    result = _source_owner(
        movement,
        period_s=YANG_2025.period_s,
        freestream_m_s=YANG_2025.freestream_m_s,
        rho_kg_m3=YANG_2025.rho_kg_m3,
        aspect_ratio=YANG_2025.aspect_ratio,
        chord_m=YANG_2025.chord_m,
    )
    expected = (
        2.0
        * np.asarray(result["relative_speed_m_s"])
        * (YANG_2025.period_s / OUTPUT_SAMPLES)
        / YANG_2025.chord_m
    )
    np.testing.assert_array_equal(result["delta_t_tilde"], expected)
    assert result["persistence"].shape == (OUTPUT_SAMPLES,)
    assert result["strip_persistence"].shape == (OUTPUT_SAMPLES, 3)
    assert result["disabled_max_abs"] == 0.0
    assert result["periodic_state_max_abs"] <= 1.0e-8
    assert np.all(np.asarray(result["persistence"]) >= 0.0)
    assert np.all(np.asarray(result["persistence"]) <= 1.0)
    assert OWNER_REFERENCE_FRACTION_CHORD == 0.75


def test_gate_relations_include_owner_lower_bound() -> None:
    assert _gate("lower", 0.0, ">=", 0.0)["numeric_pass"]
    assert not _gate("lower", -1.0e-12, ">=", 0.0)["numeric_pass"]
    assert _gate("upper", 1.0, "<=", 1.0)["numeric_pass"]
    assert _gate("identity", False, "==", False)["numeric_pass"]
