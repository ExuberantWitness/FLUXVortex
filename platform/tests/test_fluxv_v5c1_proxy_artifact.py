from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions/fluxv_v5c_nextgen_20260814"
    / "runs/20260814_fluxv_v5c1_proxy_all22_pole05_rate1_reproducible"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reproducible_proxy_artifact_is_hash_closed_and_stopped() -> None:
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))

    assert summary["status"] == "stopped_proxy_crosspaper_gate_failure"
    assert summary["promotion_status"] == "blocked_noncanonical_cache_proxy"
    assert summary["canonical_eligible"] is False
    assert summary["condition_count"] == 22
    assert summary["phase_row_count"] == 2816
    assert summary["state_row_count"] == 11264
    assert summary["paper_gate_pass"] == {
        "yang2025": False,
        "izraelevitz2017_fig14": False,
        "baik2012": False,
    }

    for relative, expected in manifest["source_hashes"].items():
        assert _sha256(REPO_ROOT / relative) == expected
    for name, expected in manifest["result_hashes"].items():
        assert _sha256(RUN / name) == expected


def test_reproducible_proxy_exports_replayable_periodic_state_evidence() -> None:
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    with (RUN / "state_histories.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 11264
    assert {row["canonical_eligible"] for row in rows} == {"false"}
    measured = max(abs(float(row["chi_cycle_difference"])) for row in rows)
    assert measured == pytest.approx(
        summary["mechanical_metrics"]["state_cycle_max_abs"], abs=1.0e-18
    )
