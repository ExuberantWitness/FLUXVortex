"""Controlled LDVM time/wake sensitivity using the frozen W2 full UVLM base."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .baik2012 import (
    BAIK_2012_CASES,
    apply_declared_v4b_transfer,
    build_baik_movement,
    sharp_fourier_lowpass,
)
from .run_provenance import (
    baik_transfer_dependency_paths,
    collect_run_provenance,
    collect_source_hashes,
    prepare_output_directory,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4"
SOURCE_CSV = DOC_ROOT / "source_data/baik2012_w1_w4_corrected_total_cl_cd.csv"
BASE_RUN = DOC_ROOT / "runs/20260813_baik2012_w1_w4_full"
DEFAULT_OUTPUT = DOC_ROOT / "sensitivity/20260813_w2_ldvm_controlled"


def _read_experiment() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case"] == "W2"][:-1]
    return (
        np.asarray([float(row["phase_t_over_T"]) for row in rows]),
        np.asarray([float(row["cl"]) for row in rows]),
        np.asarray([float(row["cd"]) for row in rows]),
    )


def _read_baseline(base_run: Path) -> tuple[dict[str, Any], list[int]]:
    histories_path = base_run / "model_phase_histories.csv"
    summary_path = base_run / "summary.json"
    with histories_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["case_id"] == "W2" and row["model"] == "fluxv_old"
        ]
    if len(rows) != 128:
        raise ValueError("frozen W2 baseline must have 128 phase samples")
    case = BAIK_2012_CASES["W2"]
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    phase = np.asarray([float(row["phase"]) for row in rows])
    cl = np.asarray([float(row["CL"]) for row in rows])
    cd = np.asarray([float(row["CD"]) for row in rows])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_range = summary["case_manifests"]["W2"]["source_cycle_step_range"]
    return (
        {
            "phase": phase,
            "CL": cl,
            "CD": cd,
            "CT": -cd,
            "lift_n": q_area * cl,
            "drag_n": q_area * cd,
            "thrust_n": -q_area * cd,
            "mean_lift_n": float(q_area * np.mean(cl)),
            "mean_drag_n": float(q_area * np.mean(cd)),
            "mean_thrust_n": float(-q_area * np.mean(cd)),
            "mean_CL": float(np.mean(cl)),
            "mean_CD": float(np.mean(cd)),
            "mean_CT": float(-np.mean(cd)),
            "source_cycle_step_range": source_range,
        },
        source_range,
    )


def _metrics(observed: np.ndarray, history: np.ndarray) -> dict[str, float]:
    error = history - observed
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, default=BASE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="explicitly permit replacing same-name files in a non-empty output directory",
    )
    args = parser.parse_args()
    base_run = args.base_run.resolve()
    output = args.output_dir.resolve()
    provenance = collect_run_provenance(
        REPO_ROOT,
        vars(args)
        | {
            "resolved_base_run": base_run,
            "resolved_output_dir": output,
        },
    )
    output = prepare_output_directory(output, allow_existing=args.allow_existing_output)

    case = BAIK_2012_CASES["W2"]
    baseline, source_range = _read_baseline(base_run)
    movement, movement_metadata = build_baik_movement(
        case, "full", settings=(4, 8, 128, 3)
    )
    if list(source_range) != [256, 383]:
        raise ValueError("unexpected source-cycle range for frozen full base")
    experiment_phase, experiment_cl, experiment_cd = _read_experiment()
    configurations = (
        # Time-step changes preserve a half-cycle retained material wake.
        ("time_256", 256, 128),
        ("reference", 512, 256),
        ("time_1024", 1024, 512),
        # Wake-only changes hold the 512-step time/core discretization fixed.
        ("wake_quarter_cycle", 512, 128),
        ("wake_three_quarter_cycle", 512, 384),
    )
    rows: list[dict[str, Any]] = []
    ldvm_settings: list[dict[str, Any]] = []
    for name, steps, wake_steps in configurations:
        print(f"W2 {name}: LDVM steps={steps}, wake={wake_steps}", flush=True)
        result = apply_declared_v4b_transfer(
            case,
            baseline,
            movement,
            output_samples=128,
            ldvm_steps_per_cycle=steps,
            ldvm_max_wake_steps=wake_steps,
            lesp_critical=0.11,
        )
        ldvm_settings.append(
            {
                "configuration": name,
                "integration_steps_per_cycle": result["ldvm_steps_per_cycle"],
                "integration_cycles": 3,
                "delta_time_convective": result["ldvm_delta_time_convective"],
                "section": result["ldvm_settings"],
            }
        )
        for quantity, observed in (("CL", experiment_cl), ("CD", experiment_cd)):
            filtered = sharp_fourier_lowpass(
                np.asarray(result[quantity], dtype=float), maximum_harmonic=3
            )
            prediction = np.interp(
                experiment_phase,
                np.asarray(result["phase"], dtype=float),
                filtered,
                period=1.0,
            )
            rows.append(
                {
                    "configuration": name,
                    "ldvm_steps_per_cycle": steps,
                    "max_wake_steps": wake_steps,
                    "retained_wake_cycles": wake_steps / steps,
                    "core_radius_chord": 0.02,
                    "quantity": quantity,
                    **_metrics(observed, prediction),
                    "mean_coefficient": float(np.mean(result[quantity])),
                    "mean_shedding_fraction": float(np.mean(result["ldvm_shedding"])),
                }
            )
        print(f"W2 {name}: complete", flush=True)

    result_path = output / "w2_ldvm_controlled_sensitivity.csv"
    with result_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    direct_sources = (
        Path(__file__).resolve(),
        Path(__file__).with_name("run_provenance.py").resolve(),
        *baik_transfer_dependency_paths(REPO_ROOT),
        SOURCE_CSV,
        base_run / "model_phase_histories.csv",
        base_run / "summary.json",
    )
    manifest = {
        "run_id": output.name,
        "case": case.manifest(),
        "base_run": str(base_run.relative_to(REPO_ROOT)),
        "base_source_cycle_step_range": source_range,
        "movement": movement_metadata,
        "time_refinement_contract": (
            "256/128, 512/256, 1024/512 preserve a half-cycle retained wake; "
            "the configured vortex-core radius remains fixed at rc/c=0.02"
        ),
        "wake_refinement_contract": (
            "512 steps/cycle fixed while max-wake varies 128/256/384"
        ),
        "ldvm_settings": ldvm_settings,
        "provenance": {
            **provenance,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "source_hashes": collect_source_hashes(REPO_ROOT, direct_sources),
        "result_hashes": {result_path.name: sha256_file(result_path)},
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
