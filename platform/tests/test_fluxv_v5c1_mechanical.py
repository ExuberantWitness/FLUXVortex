from __future__ import annotations

from pathlib import Path

from forward_flight_benchmarks.run_fluxv_v5c1_mechanical import run


def test_mechanical_runner_is_observation_free_and_all_gates_pass(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v5c1-mechanical"
    summary = run(output)
    assert summary["status"] == "v5c1_mechanical_gates_passed"
    assert summary["promotion_status"] == "mechanical_only_not_crosspaper_scored"
    assert all(summary["gates"].values())
    assert summary["parameter_selection_data"] == []
    assert (output / "synthetic_rate_history.csv").exists()
    assert (output / "numerical_refinement.csv").exists()
    assert (output / "run_manifest.json").exists()
