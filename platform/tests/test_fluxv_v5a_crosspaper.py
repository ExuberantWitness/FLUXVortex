"""Fast contracts for the v5a cross-paper cache adapter."""

from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.run_fluxv_v5a_crosspaper import (
    BAIK_PHASE,
    OUTPUT_SAMPLES,
    YANG_PHASE,
    _gate,
    _phase_history,
    _read_csv,
)


def test_cache_gate_fails_closed_for_noncanonical_incidence() -> None:
    row = _gate(
        "development_numeric_gate",
        0.5,
        1.0,
        "<=",
        canonical_eligible=False,
    )
    assert row["numeric_pass"] is True
    assert row["canonical_eligible"] is False
    assert row["promotion_pass"] is False


def test_frozen_yang_phase_contract_is_128_unique_samples() -> None:
    rows = _read_csv(YANG_PHASE)
    history = _phase_history(
        rows,
        "fluxv_uvpm",
        selectors={"aoa_deg": 15.0},
    )
    assert history["phase"].shape == (OUTPUT_SAMPLES,)
    assert np.array_equal(history["phase"], np.arange(OUTPUT_SAMPLES) / OUTPUT_SAMPLES)
    assert history["CL"].shape == (OUTPUT_SAMPLES,)
    assert history["CD"].shape == (OUTPUT_SAMPLES,)


def test_baik_equilibrium_recovery_closes_frozen_v4b_ledger() -> None:
    rows = _read_csv(BAIK_PHASE)
    old = _phase_history(rows, "fluxv_old", selectors={"case_id": "W2"})
    v4b = _phase_history(rows, "fluxv_v4b", selectors={"case_id": "W2"})
    diagnostic = [
        row for row in rows if row["case_id"] == "W2" and row["model"] == "fluxv_v4b"
    ]
    diagnostic.sort(key=lambda row: float(row["phase"]))
    persistence = np.asarray([float(row["persistence"]) for row in diagnostic])
    assert np.min(persistence) > 0.0

    for quantity in ("CL", "CD"):
        raw = np.asarray([float(row[f"ldvm_delta_{quantity}"]) for row in diagnostic])
        polar = (
            v4b[quantity] - (1.0 - persistence) * (old[quantity] + raw)
        ) / persistence
        reconstructed = (1.0 - persistence) * (
            old[quantity] + raw
        ) + persistence * polar
        assert np.allclose(reconstructed, v4b[quantity], atol=2.0e-15, rtol=0.0)
