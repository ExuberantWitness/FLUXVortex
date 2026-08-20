"""Fresh Baik W1--W4 validation on the strict CUDA LDVM backend."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "platform"),
    str(ROOT / "platform/warp_vpm"),
]

from forward_flight_benchmarks.baik2012 import BAIK_2012_CASES  # noqa: E402
from ldvm_torch_gpu import LDVM2DCuda  # noqa: E402
from gpu_runtime_monitor import GpuRuntimeMonitor  # noqa: E402


DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
SPC = 512
CYCLES = 3
N_OUT = 128
OUT = Path(__file__).resolve().parent / "fresh_results" / "baik_gpu_only"
GT = (
    ROOT / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/"
    "runs/20260813_baik2012_w1_w4_full_reproducible/scored_phase_samples.csv"
)


def _filter(values: torch.Tensor, maximum_harmonic: int) -> torch.Tensor:
    spectrum = torch.fft.rfft(values)
    spectrum[maximum_harmonic + 1 :] = 0.0
    return torch.fft.irfft(spectrum, n=values.numel())


def _periodic_sample(values: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    x = phase * values.numel()
    left = torch.floor(x).to(torch.int64) % values.numel()
    right = (left + 1) % values.numel()
    frac = x - torch.floor(x)
    return values[left] * (1.0 - frac) + values[right] * frac


def _load_gt(case_id: str, quantity: str) -> tuple[torch.Tensor, torch.Tensor]:
    unique_rows: dict[float, float] = {}
    with GT.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] == case_id and row["quantity"] == quantity:
                phase = float(row["phase"])
                experiment = float(row["experiment"])
                previous = unique_rows.setdefault(phase, experiment)
                if previous != experiment:
                    raise ValueError("Baik GT duplicates disagree at one phase")
    rows = sorted(unique_rows.items())
    return (
        torch.tensor([row[0] for row in rows], device=DEVICE, dtype=DTYPE),
        torch.tensor([row[1] for row in rows], device=DEVICE, dtype=DTYPE),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case_id: str) -> dict[str, object]:
    case = BAIK_2012_CASES[case_id]
    phase = torch.arange(SPC, device=DEVICE, dtype=DTYPE) / SPC
    angle = 2.0 * math.pi * phase
    pitch_amplitude = math.radians(case.implemented_pitch_amplitude_deg)
    peak_plunge = math.radians(case.peak_plunge_induced_alpha_deg)
    alpha = math.radians(case.mean_alpha_deg) - pitch_amplitude * torch.sin(angle)
    heave = -torch.tan(peak_plunge * torch.sin(angle))
    dt_star = case.freestream_m_s * case.period_s / case.chord_m / SPC
    alpha_rate = (torch.roll(alpha, -1) - torch.roll(alpha, 1)) / (2.0 * dt_star)

    solver = LDVM2DCuda(
        U=1.0,
        c=1.0,
        ndiv=32,
        naterm=14,
        dt=dt_star,
        rho=1.0,
        lesp_crit=0.19,
        pivot_xc=case.pivot_fraction_chord,
        core_rc=0.02,
        max_wake=256,
        device=str(DEVICE),
    )
    cl: list[torch.Tensor] = []
    cd: list[torch.Tensor] = []
    start = time.perf_counter()
    for step in range(CYCLES * SPC):
        i = step % SPC
        result = solver.step(alpha[i], alpha_rate[i], heave[i])
        cl.append(result["CLf"])
        cd.append(result["CDf"])
    torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - start
    cl_cycle = torch.stack(cl[-SPC:])
    cd_cycle = torch.stack(cd[-SPC:])
    # 512 -> 128 is an exact phase-aligned decimation.
    cl_128 = _filter(
        cl_cycle[:: SPC // N_OUT].clone(), case.experimental_filter_harmonic
    )
    cd_128 = _filter(
        cd_cycle[:: SPC // N_OUT].clone(), case.experimental_filter_harmonic
    )

    scores: dict[str, float] = {}
    for quantity, values in (("CL", cl_128), ("CD", cd_128)):
        gt_phase, gt_value = _load_gt(case_id, quantity)
        error = _periodic_sample(values, gt_phase) - gt_value
        scores[quantity] = float(torch.sqrt(torch.mean(error * error)).item())
    torch.cuda.synchronize(DEVICE)

    phase_128 = torch.arange(N_OUT, device=DEVICE, dtype=DTYPE) / N_OUT
    arrays = torch.stack((phase_128, cl_128, cd_128)).detach().cpu().numpy()
    raw = arrays.astype("<f8", copy=False).tobytes(order="C")
    np.savez(
        OUT / f"{case_id}.npz",
        phase=arrays[0],
        CL=arrays[1],
        CD=arrays[2],
    )
    return {
        "case_id": case_id,
        "cl_rmse": scores["CL"],
        "cd_rmse": scores["CD"],
        "elapsed_s": elapsed,
        "numerical_device": solver.numerical_device,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    if not torch.cuda.is_available() or DEVICE.type != "cuda":
        raise RuntimeError("Baik GPU-only runner requires CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    case_ids = sys.argv[1:] or ["W1", "W2", "W3", "W4"]
    records = []
    with GpuRuntimeMonitor() as monitor:
        for case_id in case_ids:
            record = run_case(case_id)
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        cl_macro = torch.mean(
            torch.as_tensor(
                [record["cl_rmse"] for record in records],
                device=DEVICE,
                dtype=DTYPE,
            )
        )
        cd_macro = torch.mean(
            torch.as_tensor(
                [record["cd_rmse"] for record in records],
                device=DEVICE,
                dtype=DTYPE,
            )
        )
        torch.cuda.synchronize(DEVICE)
    gpu_evidence = asdict(monitor.evidence())
    summary = {
        "schema": "fluxv-v5m-baik-019-gpu-only-v2",
        "execution_class": "cuda-only-numerical-python-orchestration",
        "base_commit": "fa8eaca9bcaa4b963ecf41683bf77d3c9e3df169",
        "source_sha256": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "ldvm_torch_gpu": _file_sha256(
                ROOT / "platform/warp_vpm/ldvm_torch_gpu.py"
            ),
            "gpu_runtime_monitor": _file_sha256(
                ROOT / "platform/warp_vpm/gpu_runtime_monitor.py"
            ),
            "ground_truth": _file_sha256(GT),
        },
        "cases": records,
        "baik_cl_macro_rmse": float(cl_macro.item()),
        "baik_cd_macro_rmse": float(cd_macro.item()),
        "gpu_device": gpu_evidence["gpu_device"],
        "cuda_kernel_path": "torch-cuda-ldvm-fixed-buffer-branchless-lev-v2",
        "gpu_utilization_observed": gpu_evidence["gpu_utilization_observed"],
        "gpu_memory_peak_mib": gpu_evidence["gpu_memory_peak_mib"],
        "gpu_runtime_evidence": gpu_evidence,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
