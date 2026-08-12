"""Run existing local models on the Yang 2025 rigid-wing six-case matrix."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cases import YANG_2025
from .ptera_adapter import MODEL_SEMANTICS, build_yang2025_movement, run_model
from .yang2025_robofalcon import (
    MODEL_KEY as ROBOFALCON_TRANSFER_KEY,
    MODEL_SEMANTICS as ROBOFALCON_TRANSFER_SEMANTICS,
    run_yang2025_robofalcon_transfer,
)


GF_TO_N = 0.00980665
AOA_DEG = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
LOCAL_MODELS = (
    "fluxv_uvpm",
    "ptera_prescribed_wake_uvlm",
    "ptera_free_wake_uvlm",
    ROBOFALCON_TRANSFER_KEY,
)
MODEL_LABELS = {
    "wind_tunnel_test": "Wind-tunnel test",
    "authors_proposed_modified_uvlm": "Authors' proposed modified UVLM",
    "fluxv_uvpm": "FluxV current load channel",
    "ptera_prescribed_wake_uvlm": "Ptera prescribed-wake control",
    "ptera_free_wake_uvlm": "Ptera free-wake UVLM",
    ROBOFALCON_TRANSFER_KEY: "RoboFalcon2 S3-S4 coefficient transfer",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reference_rows(reference_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with reference_csv.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            aoa = float(source["aoa_deg"])
            uncertainty = float(source["digitization_uncertainty_gf"])
            for model, prefix in (
                ("wind_tunnel_test", "test"),
                ("authors_proposed_modified_uvlm", "proposed"),
            ):
                lift_gf = float(source[f"{prefix}_lift_gf"])
                thrust_gf = float(source[f"{prefix}_thrust_gf"])
                rows.append(
                    {
                        "aoa_deg": aoa,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "status": "reference",
                        "mean_lift_n": lift_gf * GF_TO_N,
                        "mean_drag_n": -thrust_gf * GF_TO_N,
                        "mean_thrust_n": thrust_gf * GF_TO_N,
                        "mean_lift_gf": lift_gf,
                        "mean_drag_gf": -thrust_gf,
                        "mean_thrust_gf": thrust_gf,
                        "mean_CL": "",
                        "mean_CD": "",
                        "mean_CT": "",
                        "runtime_s": "",
                        "digitization_uncertainty_gf": uncertainty,
                        "data_role": "ground_truth" if prefix == "test" else "author_simulation",
                    }
                )
    return rows


def _local_mean_row(aoa: float, model: str, result: dict[str, Any]) -> dict[str, Any]:
    mean_drag_n = -float(result["mean_thrust_n"])
    return {
        "aoa_deg": aoa,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "status": "ok",
        "mean_lift_n": float(result["mean_lift_n"]),
        "mean_drag_n": mean_drag_n,
        "mean_thrust_n": float(result["mean_thrust_n"]),
        "mean_lift_gf": float(result["mean_lift_n"]) / GF_TO_N,
        "mean_drag_gf": mean_drag_n / GF_TO_N,
        "mean_thrust_gf": float(result["mean_thrust_n"]) / GF_TO_N,
        "mean_CL": float(result["mean_CL"]),
        "mean_CD": -float(result["mean_CT"]),
        "mean_CT": float(result["mean_CT"]),
        "runtime_s": float(result["runtime_s"]),
        "digitization_uncertainty_gf": "",
        "data_role": "local_prediction",
    }


def _failed_mean_row(aoa: float, model: str, error: str) -> dict[str, Any]:
    return {
        "aoa_deg": aoa,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "status": "failed",
        "mean_lift_n": "",
        "mean_drag_n": "",
        "mean_thrust_n": "",
        "mean_lift_gf": "",
        "mean_drag_gf": "",
        "mean_thrust_gf": "",
        "mean_CL": "",
        "mean_CD": "",
        "mean_CT": "",
        "runtime_s": "",
        "digitization_uncertainty_gf": "",
        "data_role": "local_prediction",
        "error": error,
    }


def _write_mean_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "aoa_deg",
        "model",
        "model_label",
        "status",
        "mean_lift_n",
        "mean_drag_n",
        "mean_thrust_n",
        "mean_lift_gf",
        "mean_drag_gf",
        "mean_thrust_gf",
        "mean_CL",
        "mean_CD",
        "mean_CT",
        "runtime_s",
        "digitization_uncertainty_gf",
        "data_role",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_phase_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    fields = [
        "aoa_deg",
        "model",
        "model_label",
        "phase",
        "lift_n",
        "drag_n",
        "thrust_n",
        "CL",
        "CD",
        "CT",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            result = cell["result"]
            drag_n = -np.asarray(result["thrust_n"], dtype=float)
            cd = -np.asarray(result["CT"], dtype=float)
            arrays = zip(
                result["phase"],
                result["lift_n"],
                drag_n,
                result["thrust_n"],
                result["CL"],
                cd,
                result["CT"],
            )
            for phase, lift, drag, thrust, cl, cdrag, ct in arrays:
                writer.writerow(
                    {
                        "aoa_deg": cell["aoa_deg"],
                        "model": cell["model"],
                        "model_label": MODEL_LABELS[cell["model"]],
                        "phase": phase,
                        "lift_n": lift,
                        "drag_n": drag,
                        "thrust_n": thrust,
                        "CL": cl,
                        "CD": cdrag,
                        "CT": ct,
                    }
                )


def run_matrix(
    run_dir: Path,
    quality: str,
    models: tuple[str, ...],
    angles: tuple[float, ...],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[2]
    reference_csv = (
        repository_root
        / "docs/forward_flight_large_pitch/reproductions/plev2025/source_data"
        / "yang2025_fig11_rigid_digitized.csv"
    )
    semantics = dict(MODEL_SEMANTICS)
    semantics[ROBOFALCON_TRANSFER_KEY] = ROBOFALCON_TRANSFER_SEMANTICS
    manifest: dict[str, Any] = {
        "run_id": run_dir.name,
        "created_local": datetime.now().astimezone().isoformat(),
        "quality": quality,
        "case": YANG_2025.manifest(),
        "angles_deg": list(angles),
        "models": list(models),
        "model_labels": {model: MODEL_LABELS[model] for model in models},
        "model_semantics": {model: semantics[model] for model in models},
        "reference_csv": str(reference_csv.resolve()),
        "force_contract": {
            "ptera_lift": "-Fz_W",
            "ptera_thrust": "+Fx_W",
            "plotted_drag": "D=-T=-Fx_W",
            "gf_to_n": GF_TO_N,
        },
        "comparability_limits": [
            "Local models use nominal four-bar kinematics reconstructed from rounded links; the authors used an unpublished LDS history.",
            "FluxV UVPM and Ptera prescribed-wake share the same load channel and are not independent predictions.",
            "RoboFalcon2 is a coefficient transfer outside its native geometry, Reynolds number, and calibration speed range.",
            "The local Yang PLEV formula core is excluded because wake/load integration is not implemented.",
        ],
        "status": "running",
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)

    mean_rows = _reference_rows(reference_csv)
    cells: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for aoa in angles:
        for model in models:
            print(f"RUN aoa={aoa:g} model={model} quality={quality}", flush=True)
            try:
                if model == ROBOFALCON_TRANSFER_KEY:
                    result = run_yang2025_robofalcon_transfer(
                        aoa, quality=quality, output_samples=128
                    )
                    movement_metadata = result.get("metadata", {})
                else:
                    movement, movement_metadata = build_yang2025_movement(
                        aoa, quality=quality
                    )
                    result = run_model(
                        movement,
                        model,
                        period_s=YANG_2025.period_s,
                        rho=YANG_2025.rho_kg_m3,
                        speed=YANG_2025.freestream_m_s,
                        area=YANG_2025.area_m2,
                        output_samples=128,
                    )
                result["movement_metadata"] = movement_metadata
                result["drag_n"] = -np.asarray(result["thrust_n"], dtype=float)
                result["CD"] = -np.asarray(result["CT"], dtype=float)
                result["mean_drag_n"] = -float(result["mean_thrust_n"])
                result["mean_CD"] = -float(result["mean_CT"])
                cell = {"aoa_deg": aoa, "model": model, "result": result}
                cells.append(cell)
                mean_rows.append(_local_mean_row(aoa, model, result))
                print(
                    f"OK aoa={aoa:g} model={model} "
                    f"L={result['mean_lift_n'] / GF_TO_N:.3f} gf "
                    f"D={result['mean_drag_n'] / GF_TO_N:.3f} gf "
                    f"runtime={result['runtime_s']:.2f}s",
                    flush=True,
                )
            except Exception as exc:  # preserve missing cells instead of substituting
                failure = {
                    "aoa_deg": aoa,
                    "model": model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                mean_rows.append(_failed_mean_row(aoa, model, failure["error"]))
                print(f"FAILED {failure['error']}", flush=True)
            _atomic_json(
                run_dir / "results.json",
                {"cells": cells, "failures": failures, "complete": False},
            )
            _write_mean_csv(run_dir / "mean_characteristics.csv", mean_rows)

    _write_phase_csv(run_dir / "phase_histories.csv", cells)
    _atomic_json(
        run_dir / "results.json",
        {"cells": cells, "failures": failures, "complete": True},
    )
    manifest["status"] = "complete_with_failures" if failures else "complete"
    manifest["successful_cells"] = len(cells)
    manifest["failed_cells"] = len(failures)
    _atomic_json(run_dir / "run_manifest.json", manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", choices=("smoke", "full"), default="full")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=LOCAL_MODELS,
        default=list(LOCAL_MODELS),
    )
    parser.add_argument("--angles", nargs="+", type=float, default=list(AOA_DEG))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            repository_root
            / "docs/forward_flight_large_pitch/reproductions/plev2025/runs"
            / f"{stamp}_{args.quality}"
        )
    else:
        run_dir = args.run_dir
    run_matrix(
        run_dir.resolve(),
        args.quality,
        tuple(args.models),
        tuple(args.angles),
    )
    print(f"RUN_DIR={run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
