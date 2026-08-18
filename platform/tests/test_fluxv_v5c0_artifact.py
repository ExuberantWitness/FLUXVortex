from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "fluxv_v5c_nextgen_20260814/runs"
    / "20260814_fluxv_v5c0_reference_audited"
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v5c0_artifact_closes_branch_ledger_and_hashes() -> None:
    means = _rows(RUN / "fig14_v5c0_mean_thrust.csv")
    phases = _rows(RUN / "fig14_v5c0_phase_histories.csv")
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(means) == 12
    assert len(phases) == 12 * 2 * 128
    assert summary["gates"]["legacy_replay_max_abs"] == 0.0
    assert summary["gates"]["profile_drag_application_count"] == 1
    assert (
        summary["corrected_v5c0_metrics"]["all_14_markers"]["rmse"]
        < summary["legacy_v4b_metrics"]["all_14_markers"]["rmse"]
    )

    for path, expected in manifest["result_hashes"].items():
        assert _sha(RUN / path) == expected
    for relative, expected in manifest["source_hashes"].items():
        assert _sha(ROOT / relative) == expected

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in phases:
        key = (row["model"], row["theta_max_deg"], row["phase_offset_deg"])
        groups.setdefault(key, []).append(row)
    assert len(groups) == 24
    for (model, _, _), rows in groups.items():
        rows.sort(key=lambda row: float(row["phase"]))
        phase = np.asarray([float(row["phase"]) for row in rows])
        np.testing.assert_array_equal(phase, np.arange(128) / 128.0)
        p = np.asarray([float(row["persistence"]) for row in rows])
        ldvm = np.asarray([float(row["ldvm_delta_CT"]) for row in rows])
        if model == "v4b_legacy_025c":
            old = np.asarray([float(row["owned_old_CT"]) for row in rows])
            polar = np.asarray([float(row["owned_polar_CT"]) for row in rows])
        else:
            old = np.asarray([float(row["target_old_CT"]) for row in rows])
            polar = np.asarray([float(row["target_polar_CT"]) for row in rows])
        expected = (1.0 - p) * (old + ldvm) + p * polar
        np.testing.assert_allclose(
            np.asarray([float(row["CT"]) for row in rows]),
            expected,
            atol=2.0e-15,
            rtol=0.0,
        )
