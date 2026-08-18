from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.cases import YANG_2025
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement
from forward_flight_benchmarks.run_fluxv_v5d2_region_owner import (
    OUTPUT_SAMPLES,
    YANG_ALPHA_SEP_RAD,
    _blend_regions,
    _gate,
    _region_from_movement,
)


def test_integrated_branch_blend_uses_exclusive_region_weights() -> None:
    attached = np.full(OUTPUT_SAMPLES, 1.0)
    lev = np.full(OUTPUT_SAMPLES, 3.0)
    polar = np.full(OUTPUT_SAMPLES, 7.0)
    region = {
        "weights": {
            "wA": np.full(OUTPUT_SAMPLES, 0.25),
            "wL": np.full(OUTPUT_SAMPLES, 0.50),
            "wP": np.full(OUTPUT_SAMPLES, 0.25),
        }
    }
    result = _blend_regions(attached, lev, polar, region)
    np.testing.assert_array_equal(result, np.full(OUTPUT_SAMPLES, 3.5))


def test_region_adapter_uses_075c_geometry_and_closes_weights() -> None:
    movement, _ = build_yang2025_movement(
        15.0,
        settings=(2, 3, 32, 2, 2),
    )
    result = _region_from_movement(
        movement,
        period_s=YANG_2025.period_s,
        freestream_m_s=YANG_2025.freestream_m_s,
        rho_kg_m3=YANG_2025.rho_kg_m3,
        aspect_ratio=YANG_2025.aspect_ratio,
        chord_m=YANG_2025.chord_m,
        alpha_sep_rad=YANG_ALPHA_SEP_RAD,
    )
    weight_sum = sum(np.asarray(value) for value in result["weights"].values())
    np.testing.assert_allclose(weight_sum, 1.0, rtol=0.0, atol=1.0e-15)
    assert result["disabled_max_abs"] == 0.0
    assert result["weight_sum_max_abs"] <= 1.0e-15


def test_count_and_canonical_gates_are_exact() -> None:
    assert _gate("count", 22, 22, exact=True)["numeric_pass"]
    assert not _gate("count", 21, 22, exact=True)["numeric_pass"]
    assert not _gate("canonical", False, True, exact=True)["numeric_pass"]
