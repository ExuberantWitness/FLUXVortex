"""Fast scope and schema checks for the bounded v5b smoke runner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from forward_flight_benchmarks.run_fluxv_v5b_smoke import (
    _gate,
    _rectangular_half_wing,
    _step_row,
)


def test_synthetic_rectangular_wing_geometry_is_finite() -> None:
    corners, velocity = _rectangular_half_wing(2, 3, alpha_deg=15.0)
    assert corners.shape == (3, 4, 3)
    assert velocity.shape == corners.shape
    assert np.all(np.isfinite(corners))
    assert np.count_nonzero(corners[..., 2]) > 0
    assert np.array_equal(velocity, np.zeros_like(velocity))


def test_shadow_step_schema_explicitly_has_no_force() -> None:
    report = SimpleNamespace(
        status="no_force_diagnostic_only",
        force_coupling="not_implemented",
        active=np.array([True, False]),
        new_sheet=np.array([True, False]),
        eq9_max_abs_residual=0.0,
        kelvin_max_abs_residual=1.0e-15,
        lesp_max_abs_residual=2.0e-15,
        material_gamma_max_abs_change=0.0,
        material_gamma_missing_ids=0,
        new_tev_gamma=np.array([0.1, -0.1]),
        new_lev_gamma=np.array([0.02, 0.0]),
        birth_gamma_max_abs=0.02,
    )
    row = _step_row("G2", "synthetic", 0, report)
    assert row["status"] == "no_force_diagnostic_only"
    assert row["force_coupling"] == "not_implemented"
    assert row["active_strip_count"] == 1
    assert row["new_sheet_count"] == 1
    assert row["has_force_attribute"] is False
    assert row["has_pressure_attribute"] is False


def test_gate_rows_can_pass_without_becoming_force_evidence() -> None:
    row = _gate(
        "g2_kelvin",
        0.0,
        "<=",
        1.0e-12,
        True,
        level="G2",
        evidence_role="circulation_ledger",
    )
    assert row["passed"] is True
    assert row["force_evidence"] is False
