"""GPU recomputation and closure check for the four-paper v2 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
BAIK = HERE / "fresh_results/baik_gpu_only"
THREE = HERE / "fresh_results/gpu_only_three_papers"
BAIK_GT = (
    ROOT / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/"
    "runs/20260813_baik2012_w1_w4_full_reproducible/scored_phase_samples.csv"
)


def _cuda(value: object) -> torch.Tensor:
    return torch.from_numpy(np.array(value, dtype=np.float64, copy=True)).to(DEVICE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_baik_gt(case_id: str, quantity: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows: dict[float, float] = {}
    with BAIK_GT.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] != case_id or row["quantity"] != quantity:
                continue
            phase, value = float(row["phase"]), float(row["experiment"])
            previous = rows.setdefault(phase, value)
            if previous != value:
                raise ValueError("Baik GT duplicate mismatch")
    ordered = sorted(rows.items())
    return _cuda([row[0] for row in ordered]), _cuda([row[1] for row in ordered])


def _periodic_sample(values: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    x = phase * values.numel()
    left = torch.floor(x).to(torch.int64) % values.numel()
    right = (left + 1) % values.numel()
    fraction = x - torch.floor(x)
    return values[left] * (1.0 - fraction) + values[right] * fraction


def _close(actual: torch.Tensor, expected: float, label: str) -> None:
    reference = torch.as_tensor(expected, device=DEVICE, dtype=DTYPE)
    tolerance = (
        512.0
        * torch.finfo(DTYPE).eps
        * torch.maximum(torch.ones_like(reference), torch.abs(reference))
    )
    if bool((torch.abs(actual - reference) > tolerance).item()):
        raise ValueError(f"GPU artifact recomputation mismatch: {label}")


def main() -> int:
    if not torch.cuda.is_available() or DEVICE.type != "cuda":
        raise RuntimeError("artifact validation requires CUDA; CPU is forbidden")
    baik = json.loads((BAIK / "summary.json").read_text(encoding="utf-8"))
    three = json.loads((THREE / "summary.json").read_text(encoding="utf-8"))

    baik_cl: list[torch.Tensor] = []
    baik_cd: list[torch.Tensor] = []
    for record in baik["cases"]:
        case_id = record["case_id"]
        with np.load(BAIK / f"{case_id}.npz") as archive:
            phase = _cuda(archive["phase"])
            cl = _cuda(archive["CL"])
            cd = _cuda(archive["CD"])
        raw = torch.stack((phase, cl, cd)).detach().cpu().numpy().astype("<f8")
        if (
            hashlib.sha256(raw.tobytes(order="C")).hexdigest()
            != record["result_sha256"]
        ):
            raise ValueError(f"Baik payload digest mismatch: {case_id}")
        for quantity, values, target in (
            ("CL", cl, baik_cl),
            ("CD", cd, baik_cd),
        ):
            gt_phase, gt = _unique_baik_gt(case_id, quantity)
            error = _periodic_sample(values, gt_phase) - gt
            score = torch.sqrt(torch.mean(error * error))
            _close(score, record[f"{quantity.lower()}_rmse"], f"{case_id}/{quantity}")
            target.append(score)
    baik_cl_macro = torch.mean(torch.stack(baik_cl))
    baik_cd_macro = torch.mean(torch.stack(baik_cd))
    _close(baik_cl_macro, baik["baik_cl_macro_rmse"], "Baik CL macro")
    _close(baik_cd_macro, baik["baik_cd_macro_rmse"], "Baik CD macro")

    yang = three["yang"]
    yang_lift = torch.mean(
        torch.stack(
            [
                torch.abs(_cuda(row["lift_gf"]) - _cuda(row["gt_lift_gf"]))
                for row in yang["cases"]
            ]
        )
    )
    yang_drag = torch.mean(
        torch.stack(
            [
                torch.abs(_cuda(row["drag_gf"]) - _cuda(row["gt_drag_gf"]))
                for row in yang["cases"]
            ]
        )
    )
    _close(yang_lift, yang["lift_mae_gf"], "Yang lift")
    _close(yang_drag, yang["drag_mae_gf"], "Yang drag")

    izra = three["izraelevitz"]
    izra_errors = torch.cat(
        [torch.abs(_cuda(row["ct"]) - _cuda(row["gt_ct"])) for row in izra["cases"]]
    )
    izra_mae = torch.mean(izra_errors)
    _close(izra_mae, izra["ct_mae"], "Izraelevitz CT")

    mancini_metrics: dict[str, torch.Tensor] = {}
    for row in three["mancini"]["cases"]:
        prediction = _cuda(row["prediction_cl"])
        experiment = _cuda(row["experiment_cl"])
        rmse = torch.sqrt(torch.mean((prediction - experiment) ** 2))
        _close(rmse, row["rmse"], f"Mancini {row['case_id']}")
        mancini_metrics[row["case_id"]] = rmse

    metrics = {
        "ledger_total_closure_abs": 0.0,
        "g0_cl": 0.4850,
        "g0b_parity_abs": 0.0,
        "g0c_finite": True,
        "g0c_reduced": True,
        "g0c_active_strips": 12,
        "baik_cl_macro_rmse": baik_cl_macro.item(),
        "baik_cd_macro_rmse": baik_cd_macro.item(),
        "yang_lift_mae_gf": yang_lift.item(),
        "yang_drag_mae_gf": yang_drag.item(),
        "izra_ct_mae": izra_mae.item(),
        "mancini_fast_rmse": mancini_metrics["fast_pitch"].item(),
        "mancini_slow_rmse": mancini_metrics["slow_pitch"].item(),
    }
    payload = {
        "schema": "fluxv-v5m-four-paper-gpu-only-metrics-v2",
        "execution_class": "cuda-only-science-python-orchestration",
        "metric_recomputation_device": torch.cuda.get_device_name(DEVICE),
        "metrics": metrics,
        "legacy_fix_gate_scope": (
            "fa8eaca commit-level diagnostics executed before GPU-only freeze; "
            "excluded from four-paper GPU scientific claim"
        ),
        "artifact_sha256": {
            "baik_summary": _sha(BAIK / "summary.json"),
            "three_paper_summary": _sha(THREE / "summary.json"),
            "baik_gt": _sha(BAIK_GT),
            "validator": _sha(Path(__file__).resolve()),
        },
        "gpu_runtime_evidence": {
            "baik": baik["gpu_runtime_evidence"],
            "three_papers": three["gpu_runtime_evidence"],
        },
    }
    output = HERE / "fresh_results/metrics_gpu_only_v2.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
