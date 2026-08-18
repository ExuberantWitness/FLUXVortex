from __future__ import annotations

import pytest

from forward_flight_benchmarks.run_fluxv_v5f_refinement import (
    REGISTERED_CORE_RADIUS_RATIOS,
    STEPS_PER_CYCLE,
    evaluate_m5_refinement,
)


def _rows(*, fine_growth: dict[float, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for steps in STEPS_PER_CYCLE:
        for core in REGISTERED_CORE_RADIUS_RATIOS:
            q = 1.0
            if steps == 160:
                q = float(fine_growth[float(core)])
            rows.append(
                {
                    "steps_per_cycle": steps,
                    "core_ratio": float(core),
                    "max_abs_q_m2_s": q,
                    "all_material_geometry_finite": True,
                    "max_no_penetration_residual": 1.0e-14,
                    "max_lesp_constraint_residual": 1.0e-14,
                    "max_eq9_residual_m2_s": 0.0,
                }
            )
    return rows


def test_refinement_gate_passes_bounded_consistent_family() -> None:
    gate = evaluate_m5_refinement(_rows(fine_growth={0.10: 0.9, 0.25: 1.1, 0.49: 1.2}))
    assert gate["algebra_and_finite_pass"] is True
    assert gate["no_one_over_dt_growth_pass"] is True
    assert gate["three_core_direction_consistent"] is True
    assert gate["m5_refinement_pass"] is True
    assert gate["paper_scoring_status"] == "eligible_not_run"


def test_refinement_gate_stops_when_every_core_reaches_one_over_dt_growth() -> None:
    gate = evaluate_m5_refinement(_rows(fine_growth={0.10: 2.1, 0.25: 8.0, 0.49: 4.5}))
    assert gate["algebra_and_finite_pass"] is True
    assert gate["no_one_over_dt_growth_pass"] is False
    assert gate["three_core_direction_consistent"] is True
    assert gate["m5_refinement_pass"] is False
    assert gate["paper_scoring_status"] == "blocked_not_run"


def test_refinement_gate_reports_core_family_disagreement() -> None:
    gate = evaluate_m5_refinement(_rows(fine_growth={0.10: 1.2, 0.25: 2.2, 0.49: 1.5}))
    assert gate["three_core_direction_consistent"] is False
    assert gate["m5_refinement_pass"] is False


def test_refinement_gate_rejects_incomplete_matrix() -> None:
    with pytest.raises(ValueError, match="frozen M5 matrix"):
        evaluate_m5_refinement(_rows(fine_growth={0.10: 1, 0.25: 1, 0.49: 1})[:-1])
