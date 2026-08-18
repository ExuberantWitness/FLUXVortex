"""Tests for the fail-closed FluxV v5 joint-report plotter."""

from __future__ import annotations

import copy

import pytest

from forward_flight_benchmarks.plot_fluxv_v5_joint_report import (
    _extract_mechanical_gates,
    _extract_v5a_comparison,
    _validate_no_force_v5b,
)


def _metric_row(
    benchmark: str,
    case_id: str,
    model: str,
    quantity: str,
    view: str,
    *,
    mae: float,
    rmse: float,
    count: int = 6,
) -> dict[str, str]:
    return {
        "benchmark": benchmark,
        "case_id": case_id,
        "model": model,
        "quantity": quantity,
        "view": view,
        "observation_count": str(count),
        "mae": str(mae),
        "rmse": str(rmse),
    }


def _minimal_metric_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for quantity in ("lift_gf", "drag_gf"):
        rows.extend(
            [
                _metric_row(
                    "yang2025",
                    "six_aoa",
                    "fluxv_v4b",
                    quantity,
                    "cycle_mean",
                    mae=2.0,
                    rmse=3.0,
                ),
                _metric_row(
                    "yang2025",
                    "six_aoa",
                    "fluxv_v5a",
                    quantity,
                    "cycle_mean",
                    mae=1.0,
                    rmse=1.5,
                ),
            ]
        )
    rows.extend(
        [
            _metric_row(
                "izraelevitz2017_fig14",
                "all14",
                "fluxv_v4b",
                "CT",
                "cycle_mean",
                mae=0.1,
                rmse=0.2,
                count=14,
            ),
            _metric_row(
                "izraelevitz2017_fig14",
                "all14",
                "fluxv_v5a",
                "CT",
                "cycle_mean",
                mae=0.2,
                rmse=0.6,
                count=14,
            ),
        ]
    )
    for case_id in ("W1", "W2", "W3", "W4"):
        for quantity in ("CL", "CD"):
            rows.extend(
                [
                    _metric_row(
                        "baik2012",
                        case_id,
                        "fluxv_v4b",
                        quantity,
                        "filtered_1hz",
                        mae=0.5,
                        rmse=1.0,
                        count=400,
                    ),
                    _metric_row(
                        "baik2012",
                        case_id,
                        "fluxv_v5a",
                        quantity,
                        "filtered_1hz",
                        mae=0.75,
                        rmse=1.25,
                        count=400,
                    ),
                ]
            )
    return rows


def _no_force_summary() -> dict[str, object]:
    return {
        "force_coupling": "not_implemented",
        "crosspaper_performance_status": "blocked_not_scored",
        "evidence_role": "topology_and_conservation_only_no_force",
        "gate_counts": {
            "by_level": {
                "G0": {"passed": 1, "total": 1},
                "G1": {"passed": 7, "total": 7},
                "G2": {"passed": 10, "total": 10},
            }
        },
    }


def test_v5a_plot_data_uses_frozen_baseline_rows() -> None:
    comparison = _extract_v5a_comparison(_minimal_metric_rows())
    assert [row["label"] for row in comparison] == [
        "Yang lift",
        "Yang drag",
        "Figure 14 thrust",
        "Baik CL",
        "Baik CD",
    ]
    assert [row["v5a_over_v4b"] for row in comparison] == [
        0.5,
        0.5,
        pytest.approx(3.0),
        1.25,
        1.25,
    ]


def test_no_force_contract_exposes_only_mechanical_gate_counts() -> None:
    rows = _extract_mechanical_gates(_no_force_summary())
    assert [(row["label"], row["passed"], row["total"]) for row in rows] == [
        ("G0", 1, 1),
        ("G1", 7, 7),
        ("G2", 10, 10),
    ]
    assert all(row["force_evidence"] is False for row in rows)
    assert all(
        row["crosspaper_performance_status"] == "blocked_not_scored" for row in rows
    )


def test_no_force_contract_rejects_crosspaper_accuracy_payload() -> None:
    summary = copy.deepcopy(_no_force_summary())
    summary["headline"] = {"yang": {"lift_mae": 0.0}}
    with pytest.raises(ValueError, match="no-force summary containing"):
        _validate_no_force_v5b(summary)
