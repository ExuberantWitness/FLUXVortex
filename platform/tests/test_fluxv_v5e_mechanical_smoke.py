from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from forward_flight_benchmarks.run_fluxv_v5e_mechanical_smoke import (
    CLOSURE_TOLERANCE_N,
    DEFAULT_OUTPUT,
    STATE_PERIODIC_TOLERANCE,
    _rotation_history,
    _vectors_to_wind,
    run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_default_output_is_the_frozen_fresh_smoke_directory() -> None:
    assert DEFAULT_OUTPUT.name == "20260814_fluxv_v5e_mechanical_smoke"
    assert DEFAULT_OUTPUT.parent.name == "runs"


def test_wind_rotation_helper_rotates_free_vectors_without_translation() -> None:
    transforms = np.array(
        [
            [
                [0.0, -1.0, 0.0, 17.0],
                [1.0, 0.0, 0.0, -23.0],
                [0.0, 0.0, 1.0, 31.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    rotations = _rotation_history(transforms)
    values = np.array([[[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]])
    np.testing.assert_array_equal(
        _vectors_to_wind(rotations, values),
        [[[-2.0, 1.0, 3.0], [-5.0, -4.0, -6.0]]],
    )
    bad = transforms.copy()
    bad[0, 0, 0] = 0.1
    with pytest.raises(FloatingPointError, match="orthonormal"):
        _rotation_history(bad)


def test_actual_three_case_smoke_closes_and_writes_hashed_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fresh-v5e-smoke"
    summary = run(output)

    assert summary["status"] == "v5e_mechanical_smoke_gates_passed"
    assert summary["promotion_status"] == "mechanical_only_not_experimentally_scored"
    assert summary["observation_files_read"] == []
    assert summary["experimental_scoring_performed"] is False
    assert [case["benchmark"] for case in summary["cases"]] == [
        "yang2025",
        "izraelevitz2017_fig14",
        "baik2012",
    ]
    assert [case["condition"] for case in summary["cases"]] == [
        "aoa15_smoke",
        "theta15_psi60_smoke",
        "W2_smoke",
    ]
    assert [case["samples_per_cycle"] for case in summary["cases"]] == [20, 64, 32]
    assert [case["metrics"]["added_mass_factor"] for case in summary["cases"]] == [
        0.85,
        0.85,
        0.95,
    ]
    for case in summary["cases"]:
        assert case["status"] == "mechanical_gates_passed"
        assert all(case["gates"].values())
        assert case["metrics"]["panel_closure_max_abs_n"] < CLOSURE_TOLERANCE_N
        assert case["metrics"]["strip_closure_max_abs_n"] < CLOSURE_TOLERANCE_N
        assert case["metrics"]["airplane_closure_max_abs_n"] < CLOSURE_TOLERANCE_N
        assert case["metrics"]["disabled_max_abs_n"] == 0.0
        assert case["metrics"]["state_periodic_max_abs"] < STATE_PERIODIC_TOLERANCE
        assert case["line_item_provenance"]["kinematic_added_mass"][
            "exclusive_replacement"
        ]

    force_path = output / "airplane_force_history.csv"
    strip_path = output / "strip_state_history.csv"
    summary_path = output / "summary.json"
    manifest_path = output / "run_manifest.json"
    force_rows = _read_csv(force_path)
    strip_rows = _read_csv(strip_path)
    assert len(force_rows) == 20 + 64 + 32
    assert len(strip_rows) == sum(
        case["samples_per_cycle"] * case["strip_count"] for case in summary["cases"]
    )

    for row in force_rows:
        for axis in "xyz":
            old = float(row[f"old_f{axis}_w_n"])
            new = float(row[f"lineitem_f{axis}_w_n"])
            delta = float(row[f"delta_f{axis}_w_n"])
            f_kj = float(row[f"f_kj_f{axis}_w_n"])
            old_fd_gamma = float(row[f"old_fd_gamma_f{axis}_w_n"])
            delta_phi_gamma = float(row[f"delta_phi_gamma_f{axis}_w_n"])
            added_mass = float(row[f"kinematic_added_mass_f{axis}_w_n"])
            assert new - old == pytest.approx(delta, abs=2.0e-12)
            assert old == pytest.approx(f_kj + old_fd_gamma, abs=2.0e-12)
            assert new == pytest.approx(
                f_kj + delta_phi_gamma + added_mass, abs=2.0e-12
            )
        assert float(row["old_lift_n"]) == pytest.approx(
            -float(row["old_fz_w_n"]), abs=2.0e-12
        )
        assert float(row["old_drag_n"]) == pytest.approx(
            -float(row["old_fx_w_n"]), abs=2.0e-12
        )

    for row in strip_rows:
        lift_direction = np.array(
            [
                float(row["lift_direction_x_gp1"]),
                float(row["lift_direction_y_gp1"]),
                float(row["lift_direction_z_gp1"]),
            ]
        )
        normal = np.array(
            [
                float(row["normal_x_gp1"]),
                float(row["normal_y_gp1"]),
                float(row["normal_z_gp1"]),
            ]
        )
        assert np.linalg.norm(lift_direction) == pytest.approx(1.0, abs=2.0e-12)
        projection = float(row["lift_direction_surface_normal_projection"])
        assert projection == pytest.approx(
            float(np.dot(lift_direction, normal)), abs=2.0e-12
        )
        angle = float(row["lift_direction_surface_normal_angle_deg"])
        assert angle == pytest.approx(
            float(np.rad2deg(np.arccos(np.clip(projection, -1.0, 1.0)))),
            abs=2.0e-10,
        )

    stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert stored_summary == summary
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["observation_files"] == []
    assert manifest["experimental_scoring"] is False
    for relative, expected in manifest["source_hashes"].items():
        assert _sha256(Path(__file__).resolve().parents[2] / relative) == expected
    for name, expected in manifest["result_hashes"].items():
        assert _sha256(output / name) == expected

    with pytest.raises(FileExistsError, match="non-empty"):
        run(output)
