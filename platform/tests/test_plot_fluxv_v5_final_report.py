"""Fail-closed schema tests for the final FluxV v5 report plotter."""

from __future__ import annotations

import copy

import pytest

from forward_flight_benchmarks.plot_fluxv_v5_final_report import (
    _validate_force_gate,
)


def _force_gate_summary() -> dict[str, object]:
    return {
        "status": "no_go_before_crosspaper",
        "crosspaper_performance_status": "blocked_not_scored",
        "promotion_passed": False,
        "paper_results": None,
        "gates": [
            {"gate": "G1_current_FluxV_no_LEV_exact_reduction"},
            {"gate": "G4_single_surface_pressure_force_owner"},
            {"gate": "G5_smooth_birth_limit"},
            {"gate": "G6_Ramesh_high_AR_force_parity"},
        ],
    }


def test_final_plot_accepts_only_blocked_force_gate_schema() -> None:
    _validate_force_gate(_force_gate_summary())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "promotion_pass"),
        ("crosspaper_performance_status", "eligible"),
        ("promotion_passed", True),
        ("paper_results", {"yang2025": {"lift_mae": 0.0}}),
    ],
)
def test_final_plot_rejects_promoted_or_scored_payload(key: str, value: object) -> None:
    summary = copy.deepcopy(_force_gate_summary())
    summary[key] = value
    with pytest.raises(ValueError, match="final report expects force-gate"):
        _validate_force_gate(summary)


def test_final_plot_rejects_incomplete_gate_set() -> None:
    summary = _force_gate_summary()
    summary["gates"] = summary["gates"][:-1]
    with pytest.raises(ValueError, match="frozen G1/G4/G5/G6 set"):
        _validate_force_gate(summary)
