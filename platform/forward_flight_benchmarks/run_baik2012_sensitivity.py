"""One-factor numerical sensitivities for the Baik W2 transfer benchmark."""

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
    run_baik_old_fluxv,
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
DEFAULT_OUTPUT = DOC_ROOT / "sensitivity/20260813_w2_one_factor"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _experiment() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case"] == "W2"]
    phase = np.asarray([float(row["phase_t_over_T"]) for row in rows[:-1]])
    cl = np.asarray([float(row["cl"]) for row in rows[:-1]])
    cd = np.asarray([float(row["cd"]) for row in rows[:-1]])
    return phase, cl, cd


def _scores(
    phase: np.ndarray,
    observed: np.ndarray,
    history: dict[str, Any],
    quantity: str,
) -> tuple[float, float, float]:
    filtered = sharp_fourier_lowpass(
        np.asarray(history[quantity], dtype=float), maximum_harmonic=3
    )
    predicted = np.interp(
        phase,
        np.asarray(history["phase"], dtype=float),
        filtered,
        period=1.0,
    )
    error = predicted - observed
    return (
        float(np.sqrt(np.mean(error**2))),
        float(np.mean(np.abs(error))),
        float(np.mean(error)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="explicitly permit replacing same-name files in a non-empty output directory",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    provenance = collect_run_provenance(
        REPO_ROOT,
        vars(args) | {"resolved_output_dir": output},
    )
    output = prepare_output_directory(output, allow_existing=args.allow_existing_output)

    case = BAIK_2012_CASES["W2"]
    phase, experiment_cl, experiment_cd = _experiment()
    uv_lm_configs = {
        "time_64": (4, 8, 64, 3),
        "reference": (4, 8, 128, 3),
        "span_12": (4, 12, 128, 3),
        "cycles_4": (4, 8, 128, 4),
    }
    rows: list[dict[str, Any]] = []
    ldvm_settings: list[dict[str, Any]] = []
    reference_old = None
    reference_movement = None

    for name, settings in uv_lm_configs.items():
        print(f"W2 UVLM configuration {name}: {settings}", flush=True)
        old, movement, metadata = run_baik_old_fluxv(
            case,
            "full",
            output_samples=128,
            settings=settings,
        )
        v4b = apply_declared_v4b_transfer(
            case,
            old,
            movement,
            output_samples=128,
            ldvm_steps_per_cycle=512,
            ldvm_max_wake_steps=256,
            lesp_critical=0.11,
        )
        ldvm_settings.append(
            {
                "family": "uvlm_single_factor",
                "configuration": name,
                "integration_steps_per_cycle": v4b["ldvm_steps_per_cycle"],
                "integration_cycles": 3,
                "delta_time_convective": v4b["ldvm_delta_time_convective"],
                "section": v4b["ldvm_settings"],
            }
        )
        for model, history in (("fluxv_old", old), ("fluxv_v4b", v4b)):
            for quantity, observed in (("CL", experiment_cl), ("CD", experiment_cd)):
                rmse, mae, bias = _scores(phase, observed, history, quantity)
                rows.append(
                    {
                        "family": "uvlm_single_factor",
                        "configuration": name,
                        "settings": "x".join(str(value) for value in settings),
                        "model": model,
                        "quantity": quantity,
                        "rmse": rmse,
                        "mae": mae,
                        "bias": bias,
                        "mean_coefficient": float(np.mean(history[quantity])),
                        "runtime_s": float(old["runtime_s"])
                        if model == "fluxv_old"
                        else "",
                    }
                )
        if name == "reference":
            reference_old = old
            reference_movement = movement
        del movement
        print(f"W2 UVLM configuration {name}: complete", flush=True)

    if reference_old is None or reference_movement is None:
        raise RuntimeError("reference UVLM configuration was not run")
    for name, steps, wake in (
        ("ldvm_time_256", 256, 256),
        ("reference", 512, 256),
        ("wake_128", 512, 128),
        ("wake_384", 512, 384),
    ):
        print(f"W2 LDVM configuration {name}: steps={steps}, wake={wake}", flush=True)
        v4b = apply_declared_v4b_transfer(
            case,
            reference_old,
            reference_movement,
            output_samples=128,
            ldvm_steps_per_cycle=steps,
            ldvm_max_wake_steps=wake,
            lesp_critical=0.11,
        )
        ldvm_settings.append(
            {
                "family": "ldvm_single_factor",
                "configuration": name,
                "integration_steps_per_cycle": v4b["ldvm_steps_per_cycle"],
                "integration_cycles": 3,
                "delta_time_convective": v4b["ldvm_delta_time_convective"],
                "section": v4b["ldvm_settings"],
            }
        )
        for quantity, observed in (("CL", experiment_cl), ("CD", experiment_cd)):
            rmse, mae, bias = _scores(phase, observed, v4b, quantity)
            rows.append(
                {
                    "family": "ldvm_single_factor",
                    "configuration": name,
                    "settings": f"steps={steps};wake={wake}",
                    "model": "fluxv_v4b",
                    "quantity": quantity,
                    "rmse": rmse,
                    "mae": mae,
                    "bias": bias,
                    "mean_coefficient": float(np.mean(v4b[quantity])),
                    "runtime_s": "",
                }
            )
        print(f"W2 LDVM configuration {name}: complete", flush=True)

    result_path = output / "w2_one_factor_sensitivity.csv"
    _write_csv(result_path, rows)
    manifest = {
        "run_id": output.name,
        "case": case.manifest(),
        "scope": (
            "one factor at a time around (4 chord, 8 span, 128 steps/cycle, "
            "3 cycles, LDVM 512 steps/cycle, 256 wake steps)"
        ),
        "warning": (
            "LDVM time refinement holds the configured core radius fixed at rc/c=0.02, "
            "but the original ldvm_time_256 row retains twice the physical wake duration; "
            "use the controlled sensitivity run for time-refinement claims"
        ),
        "ldvm_settings": ldvm_settings,
        "provenance": {
            **provenance,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "source_hashes": collect_source_hashes(
            REPO_ROOT,
            (
                Path(__file__).resolve(),
                Path(__file__).with_name("run_provenance.py").resolve(),
                *baik_transfer_dependency_paths(REPO_ROOT),
                SOURCE_CSV,
            ),
        ),
        "result_hashes": {result_path.name: sha256_file(result_path)},
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
