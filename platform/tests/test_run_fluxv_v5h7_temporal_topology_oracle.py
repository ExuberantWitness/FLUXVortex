"""Tests for the FluxV v5h7 manufactured temporal-topology oracle."""

from __future__ import annotations

import ast
import json

import numpy as np
import pytest

from forward_flight_benchmarks import run_fluxv_v5h7_temporal_topology_oracle as gate


@pytest.mark.parametrize("delta_time_s", gate.DELTA_TIME_S)
def test_independent_graph_has_no_shared_temporal_identity(
    delta_time_s: float,
) -> None:
    graph = gate._independent_graph(delta_time_s)
    assert all(len(edge.incidences) == 1 for edge in graph.edges)
    row = gate._run_row("independent", delta_time_s)
    assert row.two_incidence_edge_count == 0
    assert row.mechanics_passed


@pytest.mark.parametrize("delta_time_s", gate.DELTA_TIME_S)
def test_connected_graph_reconstructs_every_temporal_increment_once(
    delta_time_s: float,
) -> None:
    graph = gate._connected_graph(delta_time_s)
    shared = tuple(edge for edge in graph.edges if len(edge.incidences) == 2)
    expected_count = gate._interval_count(delta_time_s) - 1
    expected_gamma = gate.GAMMA_RATE_M2_PER_S2 * delta_time_s
    assert len(shared) == expected_count
    assert all(edge.circulation == pytest.approx(expected_gamma) for edge in shared)
    row = gate._run_row("connected", delta_time_s)
    assert row.max_shared_edge_gamma_residual <= 1.0e-14
    assert row.impulse_gp1_m4_per_s == pytest.approx(
        row.analytic_impulse_gp1_m4_per_s, abs=1.0e-14
    )
    assert row.mechanics_passed


def test_full_oracle_confirms_vanishing_closed_rings_and_connected_limit() -> None:
    summary = gate.run_gate()
    assert summary["passed"] is True
    assert summary["status"] == "go_temporal_connected_topology_oracle_mechanics_only"
    assert len(summary["rows"]) == 8
    assert summary["gates"] == {
        "all_row_mechanics_passed": True,
        "independent_closed_ring_pathology_confirmed": True,
        "connected_cumulative_sheet_limit_passed": True,
        "target_access_count": 0,
        "ptera_solver_call_count": 0,
        "load_call_count": 0,
    }
    diagnostics = summary["diagnostics"]
    assert all(
        gate.HALVING_LOWER <= value <= gate.HALVING_UPPER
        for value in (
            *diagnostics["independent_impulse_halving_ratios"],
            *diagnostics["independent_probe_halving_ratios"],
        )
    )
    assert diagnostics["connected_impulse_error_ratios"] == pytest.approx(
        [2.0, 2.0, 2.0], rel=1.0e-12
    )
    assert all(
        gate.CONNECTED_RATIO_LOWER <= value <= gate.CONNECTED_RATIO_UPPER
        for value in diagnostics["connected_probe_difference_ratios"]
    )
    assert (
        diagnostics["connected_probe_richardson_relative_difference"]
        <= gate.RICHARDSON_RELATIVE_LIMIT
    )
    assert (
        diagnostics["connected_probe_extrapolated_norm"]
        >= gate.MIN_EXTRAPOLATED_FIELD_NORM
    )


def test_oracle_is_content_deterministic() -> None:
    assert gate.run_gate() == gate.run_gate()


@pytest.mark.parametrize("bad", (0.0, -0.01, np.nan, np.inf, True, 0.03))
def test_invalid_time_steps_fail_closed(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        gate._interval_count(bad)  # type: ignore[arg-type]


def test_artifact_is_strict_and_hash_closed(tmp_path) -> None:
    output = tmp_path / "v5h7"
    summary = gate.write_artifact(output)
    assert summary["passed"]
    parsed = json.loads(
        (output / "summary.json").read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed == summary
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert gate._file_sha256(output / name) == expected


def test_runner_has_no_ptera_or_target_import() -> None:
    source = gate.Path(gate.__file__).read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert all("ptera" not in name.lower() for name in imports)
    assert all("baik" not in name.lower() for name in imports)
    assert all("yang" not in name.lower() for name in imports)
