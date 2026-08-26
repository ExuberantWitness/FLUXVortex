"""Fresh Baik W1--W4 reproduction with mandatory CUDA LDVM physics.

The scientific data path is CUDA float64.  Python is limited to orchestration,
ground-truth parsing, provenance, and serialization.  A result is qualified
only when separated LEV shedding, the TEV history, and wake convection are all
active and the frozen pre-migration GPU accuracy is not degraded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "platform"), str(ROOT / "platform/warp_vpm")]

from forward_flight_benchmarks.baik2012 import BAIK_2012_CASES  # noqa: E402
from gpu_runtime_monitor import GpuRuntimeMonitor  # noqa: E402
from ldvm_torch_gpu import LDVM2DCuda  # noqa: E402


DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
STEPS_PER_CYCLE = 512
CYCLES = 3
OUTPUT_SAMPLES = 128
FROZEN_CL_MACRO_RMSE = 0.42156276782081215
FROZEN_CD_MACRO_RMSE = 0.2897097942292014
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/baselines/fluxv_v5m_four_case_20260822/results/baik"
)
GROUND_TRUTH = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/"
    "runs/20260813_baik2012_w1_w4_full_reproducible/scored_phase_samples.csv"
)


def _filter(values: torch.Tensor, maximum_harmonic: int) -> torch.Tensor:
    spectrum = torch.fft.rfft(values)
    spectrum[maximum_harmonic + 1 :] = 0.0
    return torch.fft.irfft(spectrum, n=values.numel())


def _periodic_sample(values: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    coordinate = phase * values.numel()
    left = torch.floor(coordinate).to(torch.int64) % values.numel()
    right = (left + 1) % values.numel()
    fraction = coordinate - torch.floor(coordinate)
    return values[left] * (1.0 - fraction) + values[right] * fraction


def _load_ground_truth(case_id: str, quantity: str) -> tuple[torch.Tensor, torch.Tensor]:
    unique_rows: dict[float, float] = {}
    with GROUND_TRUTH.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] == case_id and row["quantity"] == quantity:
                phase = float(row["phase"])
                experiment = float(row["experiment"])
                previous = unique_rows.setdefault(phase, experiment)
                if previous != experiment:
                    raise ValueError("Baik ground-truth duplicates disagree at one phase")
    rows = sorted(unique_rows.items())
    if not rows:
        raise ValueError(f"no ground truth for {case_id}/{quantity}")
    return (
        torch.tensor([row[0] for row in rows], device=DEVICE, dtype=DTYPE),
        torch.tensor([row[1] for row in rows], device=DEVICE, dtype=DTYPE),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def run_case(case_id: str, output: Path) -> dict[str, object]:
    case = BAIK_2012_CASES[case_id]
    phase = torch.arange(STEPS_PER_CYCLE, device=DEVICE, dtype=DTYPE) / STEPS_PER_CYCLE
    angle = 2.0 * math.pi * phase
    pitch_amplitude = math.radians(case.implemented_pitch_amplitude_deg)
    peak_plunge = math.radians(case.peak_plunge_induced_alpha_deg)
    alpha = math.radians(case.mean_alpha_deg) - pitch_amplitude * torch.sin(angle)
    heave = -torch.tan(peak_plunge * torch.sin(angle))
    dt_star = case.freestream_m_s * case.period_s / case.chord_m / STEPS_PER_CYCLE
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
    lev_shed_count = torch.zeros((), device=DEVICE, dtype=torch.int64)
    total_steps = CYCLES * STEPS_PER_CYCLE
    start = time.perf_counter()
    for step in range(total_steps):
        index = step % STEPS_PER_CYCLE
        result = solver.step(alpha[index], alpha_rate[index], heave[index])
        cl.append(result["CLf"])
        cd.append(result["CDf"])
        lev_shed_count.add_(result["shed_lev"].to(torch.int64))
    torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - start

    cl_cycle = torch.stack(cl[-STEPS_PER_CYCLE:])
    cd_cycle = torch.stack(cd[-STEPS_PER_CYCLE:])
    stride = STEPS_PER_CYCLE // OUTPUT_SAMPLES
    cl_output = _filter(cl_cycle[::stride].clone(), case.experimental_filter_harmonic)
    cd_output = _filter(cd_cycle[::stride].clone(), case.experimental_filter_harmonic)
    scores: dict[str, float] = {}
    for quantity, values in (("CL", cl_output), ("CD", cd_output)):
        gt_phase, gt_value = _load_ground_truth(case_id, quantity)
        error = _periodic_sample(values, gt_phase) - gt_value
        scores[quantity] = float(torch.sqrt(torch.mean(error * error)).item())

    witness = torch.stack(
        (
            lev_shed_count.to(dtype=DTYPE),
            torch.as_tensor(solver.nt, device=DEVICE, dtype=DTYPE),
            torch.as_tensor(
                solver.wake_convection_count, device=DEVICE, dtype=DTYPE
            ),
        )
    )
    torch.cuda.synchronize(DEVICE)
    witness_host = witness.detach().cpu().numpy()
    lev_count, tev_count, convection_steps = (int(value) for value in witness_host)
    physics_gate = bool(
        solver.numerical_device.startswith("cuda")
        and lev_count > 0
        and tev_count > 0
        and convection_steps == total_steps
    )

    phase_output = torch.arange(OUTPUT_SAMPLES, device=DEVICE, dtype=DTYPE) / OUTPUT_SAMPLES
    arrays = torch.stack((phase_output, cl_output, cd_output)).detach().cpu().numpy()
    raw = arrays.astype("<f8", copy=False).tobytes(order="C")
    np.savez(output / f"{case_id}.npz", phase=arrays[0], CL=arrays[1], CD=arrays[2])
    return {
        "case_id": case_id,
        "cl_rmse": scores["CL"],
        "cd_rmse": scores["CD"],
        "elapsed_s": elapsed,
        "numerical_device": solver.numerical_device,
        "lev_shed_count": lev_count,
        "tev_history_count": tev_count,
        "wake_convection_steps": convection_steps,
        "physics_gate": physics_gate,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_ids", nargs="*")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() or DEVICE.type != "cuda":
        raise RuntimeError("Baik mandatory runner requires CUDA")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_ids = args.case_ids or ["W1", "W2", "W3", "W4"]
    invalid = sorted(set(case_ids) - set(BAIK_2012_CASES))
    if invalid:
        parser.error(f"unknown Baik case(s): {', '.join(invalid)}")

    records: list[dict[str, object]] = []
    with GpuRuntimeMonitor() as monitor:
        for case_id in case_ids:
            record = run_case(case_id, output)
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        cl_macro = torch.mean(
            torch.tensor([row["cl_rmse"] for row in records], device=DEVICE, dtype=DTYPE)
        )
        cd_macro = torch.mean(
            torch.tensor([row["cd_rmse"] for row in records], device=DEVICE, dtype=DTYPE)
        )
        torch.cuda.synchronize(DEVICE)

    cl_value = float(cl_macro.item())
    cd_value = float(cd_macro.item())
    complete_matrix = set(case_ids) == {"W1", "W2", "W3", "W4"}
    accuracy_gate = bool(
        complete_matrix
        and cl_value <= FROZEN_CL_MACRO_RMSE * (1.0 + 1.0e-12)
        and cd_value <= FROZEN_CD_MACRO_RMSE * (1.0 + 1.0e-12)
    )
    gpu_evidence = asdict(monitor.evidence())
    passed = bool(
        accuracy_gate
        and all(bool(row["physics_gate"]) for row in records)
        and gpu_evidence["gpu_utilization_observed"]
    )
    runner = Path(__file__).resolve()
    summary = {
        "schema": "fluxv-v5m-baik-mandatory-reproduction-v1",
        "status": "PASS" if passed else "FAIL",
        "execution_class": "cuda-only-numerical-python-orchestration",
        "git_head": _git_head(),
        "cases": records,
        "baik_cl_macro_rmse": cl_value,
        "baik_cd_macro_rmse": cd_value,
        "frozen_reference": {
            "cl_macro_rmse": FROZEN_CL_MACRO_RMSE,
            "cd_macro_rmse": FROZEN_CD_MACRO_RMSE,
        },
        "accuracy_gate": accuracy_gate,
        "mandatory_physics": "separated-LEV+TEV-history+free-wake",
        "gpu_runtime_evidence": gpu_evidence,
        "source_sha256": {
            "runner": _sha256(runner),
            "ldvm_torch_gpu": _sha256(ROOT / "platform/warp_vpm/ldvm_torch_gpu.py"),
            "gpu_runtime_monitor": _sha256(ROOT / "platform/warp_vpm/gpu_runtime_monitor.py"),
            "ground_truth": _sha256(GROUND_TRUTH),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
