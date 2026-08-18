"""Regression tests for the frozen all-condition comparison products."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from forward_flight_benchmarks.fluxv_v5_all_conditions import (
    build_aggregate_table,
    build_coverage,
    build_curves,
    build_metrics,
)


def _metric_lookup(metrics, paper, scope, view, model, observable):
    selected = [
        row
        for row in metrics
        if row["paper"] == paper
        and row["scope"] == scope
        and row["view"] == view
        and row["model_id"] == model
        and row["observable"] == observable
    ]
    assert len(selected) == 1
    return selected[0]


def test_frozen_condition_counts_and_duplicate_endpoints() -> None:
    curves = build_curves()
    yang_experiment = [
        row
        for row in curves
        if row["paper"] == "yang2025" and row["model_id"] == "experiment"
    ]
    assert len(yang_experiment) == 12
    assert {row["case_id"] for row in yang_experiment} == {
        "aoa_0",
        "aoa_5",
        "aoa_10",
        "aoa_15",
        "aoa_20",
        "aoa_25",
    }

    fig14_experiment = [
        row
        for row in curves
        if row["paper"] == "izraelevitz2017_fig14" and row["model_id"] == "experiment"
    ]
    assert len(fig14_experiment) == 14
    assert len({row["case_id"] for row in fig14_experiment}) == 12

    for view in ("filtered_1hz", "raw_numeric_diagnostic"):
        for case_id in ("W1", "W2", "W3", "W4"):
            for observable in ("CL", "CD"):
                selected = [
                    row
                    for row in curves
                    if row["paper"] == "baik2012"
                    and row["view"] == view
                    and row["case_id"] == case_id
                    and row["observable"] == observable
                    and row["model_id"] == "experiment"
                ]
                assert len(selected) == 400
                phases = np.asarray([float(row["x_value"]) for row in selected])
                assert phases[0] == 0.0
                assert phases[-1] == 0.9975
                assert len(np.unique(phases)) == 400


def test_headline_metrics_recompute_from_long_table() -> None:
    metrics = build_metrics(build_curves())
    expected = {
        ("yang2025", "all_6_aoa", "cycle_mean", "fluxv_v4b", "lift"): 5.239584000138488,
        ("yang2025", "all_6_aoa", "cycle_mean", "fluxv_v5a", "drag"): 2.37371576544448,
        (
            "izraelevitz2017_fig14",
            "all_14_markers",
            "cycle_mean",
            "fluxv_v4b",
            "CT",
        ): 0.02594916701432961,
        (
            "izraelevitz2017_fig14",
            "unique_12_conditions",
            "cycle_mean",
            "fluxv_v5a",
            "CT",
        ): 0.08930454168688041,
        (
            "baik2012",
            "macro_4_cases",
            "filtered_1hz",
            "fluxv_v4b",
            "CL",
        ): 0.6575418666678028,
        (
            "baik2012",
            "macro_4_cases",
            "filtered_1hz",
            "fluxv_v5a",
            "CD",
        ): 0.40440906359275886,
    }
    for key, rmse in expected.items():
        row = _metric_lookup(metrics, *key)
        np.testing.assert_allclose(float(row["rmse"]), rmse, rtol=0.0, atol=2.0e-14)


def test_v5a_identity_and_v5b_fail_closed_coverage() -> None:
    curves = build_curves()
    by_key = defaultdict(dict)
    for row in curves:
        if row["paper"] == "yang2025":
            by_key[(row["case_id"], row["observable"])][row["model_id"]] = float(
                row["value"]
            )
    for values in by_key.values():
        np.testing.assert_allclose(
            values["fluxv_v5a"], values["fluxv_v1_v2"], rtol=0.0, atol=7.0e-4
        )
    assert not any(row["model_id"] == "fluxv_v5b" for row in curves)
    coverage = {row["model_id"]: row for row in build_coverage()}
    assert coverage["fluxv_v5b"]["status"] == "blocked_before_crosspaper_scoring"
    assert coverage["fluxv_v5b"]["yang2025"] == "not_scored"


def test_aggregate_table_does_not_mix_units_or_claim_v5a_promotion() -> None:
    table = build_aggregate_table(build_metrics(build_curves()))
    assert len(table) == 6
    assert {row["units"] for row in table} == {"gf", "1"}
    by_name = {row["metric"]: row for row in table}
    assert by_name["Yang lift"]["conclusion"] == "v5a_improves_v4b"
    assert by_name["Figure 14 CT (14 markers)"]["conclusion"] == "v5a_regresses_vs_v4b"
    assert by_name["Baik CL (filtered macro)"]["conclusion"] == "v5a_regresses_vs_v4b"
