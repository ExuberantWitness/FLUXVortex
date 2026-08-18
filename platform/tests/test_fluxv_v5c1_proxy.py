from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from forward_flight_benchmarks.run_fluxv_v5c1_proxy import (
    BAIK_PHASE,
    V5C0_PHASE,
    _periodic_state_proxy,
    _phase_history,
    _read_csv,
    _assert_source_frozen_parameters,
    _write_csv,
)


def test_periodic_proxy_is_bounded_axial_only_and_disabled_exact() -> None:
    steps = 64
    phase = np.arange(steps) / steps
    a0_pre = 0.12 + 0.10 * np.sin(2.0 * np.pi * phase)
    separated_cs = 2.0 * np.pi * np.minimum(np.abs(a0_pre), 0.1) ** 2
    result = _periodic_state_proxy(
        a0_pre=a0_pre,
        separated_cs=separated_cs,
        delta_tau=np.full(steps, 0.05),
        alpha_rad=np.deg2rad(15.0 * np.sin(2.0 * np.pi * phase)),
        lesp_critical=0.1,
        aspect_ratio=3.0,
    )
    assert result["disabled_max_abs"] == 0.0
    assert result["cycle_state_error"] <= 1.0e-4
    assert np.min(result["loss_fraction"]) >= 0.0
    assert np.max(result["loss_fraction"]) <= 1.0
    assert np.max(result["delta_suction_coefficient"]) <= 1.0e-15
    np.testing.assert_array_equal(result["delta_CN"], np.zeros(steps))


def test_proxy_parameter_manifest_is_source_frozen_and_observation_free() -> None:
    manifest = _assert_source_frozen_parameters()
    assert manifest["state_pole_per_convective_time"] == 0.5
    assert manifest["observation_fit"] == "none"
    assert "0.75c" in manifest["delta_tau_definition"]
    assert "c_local" in manifest["delta_tau_definition"]
    assert manifest["rate_excitation_scale"] == 1.0


def test_audited_v5c0_loader_has_all_twelve_corrected_histories() -> None:
    rows = _read_csv(V5C0_PHASE)
    conditions = sorted(
        {
            (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
            for row in rows
            if row["model"] == "v5c0_corrected_v4b_075c"
        }
    )
    assert len(conditions) == 12
    for theta, psi in conditions:
        history = _phase_history(
            rows,
            "v5c0_corrected_v4b_075c",
            theta_max_deg=theta,
            phase_offset_deg=psi,
        )
        np.testing.assert_array_equal(history["phase"], np.arange(128) / 128.0)
        assert history["CT"].shape == (128,)
        assert history["persistence"].shape == (128,)


def test_phase_loader_accepts_string_case_selector() -> None:
    history = _phase_history(_read_csv(BAIK_PHASE), "fluxv_v4b", case_id="W1")
    np.testing.assert_array_equal(history["phase"], np.arange(128) / 128.0)
    assert history["CL"].shape == (128,)


def test_csv_writer_refuses_empty_and_preserves_canonical_false(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="empty"):
        _write_csv(tmp_path / "empty.csv", [])
    path = tmp_path / "proxy.csv"
    _write_csv(
        path,
        [
            {
                "benchmark": "yang2025",
                "case_id": "aoa_0",
                "canonical_eligible": "false",
            }
        ],
    )
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["canonical_eligible"] == "false"


def test_frozen_v5c0_summary_is_noncanonical_input_not_parameter_fit() -> None:
    summary_path = V5C0_PHASE.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["parameter_selection_data"] == []
    assert summary["gates"]["profile_drag_application_count"] == 1
    assert summary["gates"]["target_reference_is_source_pivot"] is True


def test_proxy_exports_previous_cycle_state_for_independent_gate_replay() -> None:
    steps = 64
    phase = np.arange(steps) / steps
    result = _periodic_state_proxy(
        a0_pre=0.14 + 0.12 * np.sin(2.0 * np.pi * phase),
        separated_cs=np.full(steps, 0.05),
        delta_tau=np.full(steps, 0.05),
        alpha_rad=np.zeros(steps),
        lesp_critical=0.1,
        aspect_ratio=3.0,
    )
    np.testing.assert_allclose(
        result["chi_cycle_difference"],
        result["chi_after"] - result["chi_previous_cycle"],
        rtol=0.0,
        atol=0.0,
    )
    assert np.max(np.abs(result["chi_cycle_difference"])) <= 1.0e-4
