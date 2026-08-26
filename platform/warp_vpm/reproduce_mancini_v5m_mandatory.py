"""Reproduce Mancini 2017 Figure 4.13(b) with mandatory FLUX-V5M.

The paper geometry, Eldredge pitch history, scoring interval and digitized
observations come from the frozen benchmark adapter. The aerodynamic owner is
the sole V5M production entry: CUDA float64, separated LEV, joint TEV and free
wake. No second LDVM separation increment is added because that would double
own separated circulation already carried by the live three-dimensional wake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pterasoftware as ps
import torch

from bing_joint_ptera import JointConfig
from flux_v5m_gpu import run_flux_v5m_ptera
from forward_flight_benchmarks.mancini2017 import (
    MANCINI_2017_CASES,
    build_mancini_movement,
    load_mancini_fig4_13b_experiment,
)


DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/baselines/fluxv_v5m_four_case_20260822/results/mancini"
)
FROZEN_REFERENCE_RMSE_CL = {
    "fast_pitch": 1.2553197460217713,
    "slow_pitch": 0.29509477610137613,
}


def _cuda(value: object) -> torch.Tensor:
    array = np.array(value, dtype=np.float64, copy=True)
    return torch.from_numpy(array).to(device=DEVICE, dtype=DTYPE)


def _linear_interp(
    target: torch.Tensor,
    coordinates: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if coordinates.ndim != 1 or values.shape != coordinates.shape:
        raise ValueError("interpolation coordinates and values must be aligned 1-D")
    if coordinates.numel() < 2 or not bool(
        torch.all(coordinates[1:] > coordinates[:-1]).item()
    ):
        raise ValueError("interpolation coordinates must be strictly increasing")
    right = torch.searchsorted(coordinates, target, right=True)
    right = torch.clamp(right, 1, coordinates.numel() - 1)
    left = right - 1
    weight = (target - coordinates[left]) / (
        coordinates[right] - coordinates[left]
    )
    return values[left] + weight * (values[right] - values[left])


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    if value.ndim != 0 or not bool(torch.isfinite(value).item()):
        raise FloatingPointError(f"{name} is not a finite scalar")
    return float(value.item())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(
    case_id: str,
    *,
    quality: str,
    output_root: Path,
    show_progress: bool,
    separated_source: str = "dvm_node_ribbon",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Mancini V5M reproduction requires CUDA; CPU is forbidden")
    case = MANCINI_2017_CASES[case_id]
    movement, case_manifest = build_mancini_movement(case, quality)
    problem = ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )
    config = JointConfig(
        enable_lev=True,
        joint_tev=True,
        lev_start_step=0,
        lesp_crit=0.11,
        load_mode="bing",
        particle_capacity=100_000,
        separated_source=separated_source,
        dvm_max_wake=movement.num_steps + 1,
        dvm_pivot_fraction_chord=float(case.pivot_fraction_chord),
    )
    torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    solver = run_flux_v5m_ptera(
        problem,
        config,
        device=str(DEVICE),
        prescribed_wake=False,
        calculate_streamlines=False,
        show_progress=show_progress,
    )
    torch.cuda.synchronize(DEVICE)
    elapsed_seconds = time.perf_counter() - started

    force_w = _cuda(
        [steady.airplanes[0].forces_W for steady in solver.steady_problems]
    )
    if force_w.shape != (movement.num_steps, 3) or not bool(
        torch.isfinite(force_w).all().item()
    ):
        raise FloatingPointError("Mancini V5M returned an invalid force history")
    time_star = (
        torch.arange(movement.num_steps, device=DEVICE, dtype=DTYPE)
        * float(movement.delta_time)
        / float(case.convective_time_s)
        - float(case.warmup_chords)
    )
    valid = (time_star >= -1.0e-10) & (
        time_star <= float(case.observation_chords) + 1.0e-10
    )
    if int(torch.count_nonzero(valid).item()) < 5:
        raise RuntimeError("Mancini scoring window is incomplete")
    target = torch.linspace(
        0.0,
        float(case.observation_chords),
        501,
        device=DEVICE,
        dtype=DTYPE,
    )
    q_area = torch.prod(
        _cuda(
            [
                0.5,
                case.rho_kg_m3,
                case.freestream_m_s,
                case.freestream_m_s,
                case.area_m2,
            ]
        )
    )
    prediction_cl = _linear_interp(
        target,
        time_star[valid],
        (-force_w[:, 2] / q_area)[valid],
    )
    impulse_force_w = torch.zeros_like(force_w)
    impulse_count = len(solver.impulse_force)
    if impulse_count:
        impulse_force_w[-impulse_count:] = _cuda(solver.impulse_force)
    non_impulse_force_w = force_w - impulse_force_w
    impulse_cl = _linear_interp(
        target,
        time_star[valid],
        (-impulse_force_w[:, 2] / q_area)[valid],
    )
    non_impulse_cl = _linear_interp(
        target,
        time_star[valid],
        (-non_impulse_force_w[:, 2] / q_area)[valid],
    )
    experiment = load_mancini_fig4_13b_experiment()
    experiment_time = _cuda(experiment["t_star"])
    experiment_cl = _linear_interp(
        target,
        experiment_time,
        _cuda(experiment[f"CL_{case_id}"]),
    )
    error = prediction_cl - experiment_cl
    rmse = torch.sqrt(torch.mean(error * error))
    mae = torch.mean(torch.abs(error))
    bias = torch.mean(error)
    centered_prediction = prediction_cl - torch.mean(prediction_cl)
    centered_experiment = experiment_cl - torch.mean(experiment_cl)
    correlation = torch.sum(centered_prediction * centered_experiment) / torch.sqrt(
        torch.sum(centered_prediction * centered_prediction)
        * torch.sum(centered_experiment * centered_experiment)
    )
    prediction_peak_index = torch.argmax(prediction_cl)
    experiment_peak_index = torch.argmax(experiment_cl)

    if solver.jcfg.enable_lev is not True or solver.jcfg.joint_tev is not True:
        raise RuntimeError("Mancini runner lost mandatory LEV/TEV configuration")
    if solver._prescribed_wake is not False:
        raise RuntimeError("Mancini runner lost mandatory free wake")
    if solver.lev_pf.device.type != "cuda" or solver.lev_pf.n <= 0:
        raise RuntimeError("Mancini runner did not retain CUDA LEV particles")
    if solver._tev_solved is None or not solver._tev_solved.is_cuda:
        raise RuntimeError("Mancini runner has no CUDA joint-TEV solution")
    if solver.cuda_counters["wake_convection"] != movement.num_steps - 1:
        raise RuntimeError("Mancini free-wake convection count drifted")
    if separated_source == "dvm_node_ribbon":
        source_bank = getattr(solver, "dvm_source_bank", None)
        if (
            source_bank is None
            or source_bank.device.type != "cuda"
            or source_bank.it != movement.num_steps
            or solver.cuda_counters["dvm_source_steps"] != movement.num_steps
            or solver.cuda_counters["dvm_ribbon_shed"] <= 0
        ):
            raise RuntimeError("Mancini DVM node-ribbon source did not fully advance")
        if (
            solver.cuda_counters["impulse"] != 0
            or solver.impulse_force
            or not all(
                row.get("load_owner") == "ptera_kj_plus_dgamma"
                for row in solver.diag
            )
            or bool(
                torch.any(solver._q16_unresolved_impulse_force_w != 0.0).item()
            )
        ):
            raise RuntimeError("Mancini DVM mode lost its unique surface-load owner")

    output_root.mkdir(parents=True, exist_ok=True)
    result_tag = separated_source
    if separated_source == "dvm_node_ribbon":
        result_tag += "_surface_only_lesp_coupled_state_split"
    array_path = output_root / f"{case_id}_{quality}_{result_tag}.npz"
    np.savez(
        array_path,
        t_star=target.detach().cpu().numpy(),
        prediction_cl=prediction_cl.detach().cpu().numpy(),
        non_impulse_cl=non_impulse_cl.detach().cpu().numpy(),
        impulse_cl=impulse_cl.detach().cpu().numpy(),
        experiment_cl=experiment_cl.detach().cpu().numpy(),
        source_t_star=time_star.detach().cpu().numpy(),
        source_total_cl=(-force_w[:, 2] / q_area).detach().cpu().numpy(),
        source_non_impulse_cl=(-non_impulse_force_w[:, 2] / q_area)
        .detach()
        .cpu()
        .numpy(),
        source_impulse_cl=(-impulse_force_w[:, 2] / q_area)
        .detach()
        .cpu()
        .numpy(),
    )
    result: dict[str, Any] = {
        "schema": "flux-v5m-mancini2017-mandatory-mode-v5",
        "paper": (
            "Mancini 2017, Experimental Investigation into Unsteady Force "
            "Transients on Rapidly Maneuvering Wings, Figure 4.13(b)"
        ),
        "case_id": case_id,
        "quality": quality,
        "model_contract": (
            "CUDA-float64 + separated-LEV + joint-TEV + free-wake + "
            f"separated_source={separated_source} + active Ptera LESP rows; "
            "no attached/prescribed fallback "
            "and no post-hoc LDVM load delta"
        ),
        "separated_source": separated_source,
        "load_owner": (
            "ptera_kj_plus_dgamma"
            if separated_source == "dvm_node_ribbon"
            else "ptera_plus_legacy_vortex_impulse"
        ),
        "case_manifest": case_manifest,
        "digitization_sha256": experiment["digitization_sha256"],
        "source_step_count": movement.num_steps,
        "scored_sample_count": int(target.numel()),
        "rmse_cl": _finite_scalar(rmse, "RMSE"),
        "mae_cl": _finite_scalar(mae, "MAE"),
        "bias_cl": _finite_scalar(bias, "bias"),
        "correlation_cl": _finite_scalar(correlation, "correlation"),
        "max_abs_non_impulse_cl": _finite_scalar(
            torch.max(torch.abs(non_impulse_cl)), "non-impulse CL maximum"
        ),
        "max_abs_impulse_cl": _finite_scalar(
            torch.max(torch.abs(impulse_cl)), "impulse CL maximum"
        ),
        "prediction_peak_cl": _finite_scalar(
            prediction_cl[prediction_peak_index], "prediction peak"
        ),
        "prediction_peak_t_star": _finite_scalar(
            target[prediction_peak_index], "prediction peak time"
        ),
        "experiment_peak_cl": _finite_scalar(
            experiment_cl[experiment_peak_index], "experiment peak"
        ),
        "experiment_peak_t_star": _finite_scalar(
            target[experiment_peak_index], "experiment peak time"
        ),
        "elapsed_seconds": elapsed_seconds,
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "gpu_peak_memory_mib": None,
        "cuda_numerical_contract": solver.cuda_numerical_contract,
        "cuda_counters": dict(solver.cuda_counters),
        "solver_step_diagnostics": list(solver.diag),
        "lev_particle_count": int(solver.lev_pf.n),
        "dvm_source_step_count": int(
            getattr(getattr(solver, "dvm_source_bank", None), "it", 0)
        ),
        "dvm_ptera_state_split_step_count": sum(
            row.get("dvm_ptera_pin_strips", 0) > row["lev_strips"]
            for row in solver.diag
        ),
        "dvm_ptera_lesp_pin_max_abs": max(
            row.get("dvm_ptera_lesp_pin_max_abs", 0.0) for row in solver.diag
        ),
        "dvm_ptera_retained_neumann_max_abs": max(
            row.get("dvm_ptera_retained_neumann_max_abs", 0.0)
            for row in solver.diag
        ),
        "joint_tev_solved": True,
        "free_wake": True,
        "result_array": str(array_path),
        "result_array_sha256": _sha256(array_path),
    }
    result["qualification_gate"] = {
        "metric": "rmse_cl",
        "rule": "mandatory V5M must not regress the frozen GPU reference",
        "limit": FROZEN_REFERENCE_RMSE_CL[case_id],
        "passed": result["rmse_cl"] <= FROZEN_REFERENCE_RMSE_CL[case_id],
    }
    result_path = output_root / f"{case_id}_{quality}_{result_tag}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result["result_json"] = str(result_path)
    result["result_json_sha256"] = _sha256(result_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=tuple(MANCINI_2017_CASES),
        default="fast_pitch",
    )
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--separated-source",
        choices=("dvm_node_ribbon", "hirato_ring"),
        default="dvm_node_ribbon",
    )
    args = parser.parse_args()
    result = run_case(
        args.case,
        quality=args.quality,
        output_root=args.output,
        show_progress=args.show_progress,
        separated_source=args.separated_source,
    )
    console_result = {
        key: value
        for key, value in result.items()
        if key != "solver_step_diagnostics"
    }
    print(json.dumps(console_result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["qualification_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
