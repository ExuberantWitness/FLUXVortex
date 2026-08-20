"""Strict CUDA validation for Yang 2025, Izraelevitz 2017 and Mancini 2017."""
# ruff: noqa: E402
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
import pterasoftware
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "platform"),
    str(ROOT / "platform/warp_vpm"),
]

from bing_drag_ledger import LedgerConfig
from bing_gpu_corrections import (
    movement_polar_residual_cuda,
    project_ldvm_delta_to_finite_wing_cuda,
    run_ldvm_separation_pair_cuda,
    run_ledger_cuda,
)
from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaAttachedJointLEVTEVSolver
from gpu_runtime_monitor import GpuRuntimeMonitor
from forward_flight_benchmarks.cases import IzraelevitzSchererCase, Yang2025RigidCase
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
from forward_flight_benchmarks.mancini2017 import (
    MANCINI_2017_CASES,
    load_mancini_fig4_13b_experiment,
    mancini_periodic_pitch_spacing,
)
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_scherer_movement,
    build_yang2025_movement,
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
SPC = 128
N_OUT = 128
G = 9.81
OUT = Path(__file__).resolve().parent / "fresh_results/gpu_only_three_papers"


def _require_cuda() -> None:
    if not torch.cuda.is_available() or DEVICE.type != "cuda":
        raise RuntimeError("this validation is CUDA-only; CPU fallback is forbidden")
    torch.empty(1, dtype=DTYPE, device=DEVICE)


def _tensor(value) -> torch.Tensor:
    return torch.from_numpy(np.array(value, dtype=np.float64, copy=True)).to(DEVICE)


def _linear_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    right = torch.searchsorted(xp, x, right=True)
    right = torch.clamp(right, 1, xp.numel() - 1)
    left = right - 1
    weight = (x - xp[left]) / (xp[right] - xp[left])
    return fp[left] + weight * (fp[right] - fp[left])


def _run_chassis(movement):
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False
    )
    solver = CudaAttachedJointLEVTEVSolver(
        problem, JointConfig(enable_lev=False), device=str(DEVICE)
    )
    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    force = _tensor(
        [problem.airplanes[0].forces_W for problem in solver.steady_problems]
    )
    if not bool(torch.isfinite(force).all().item()):
        raise FloatingPointError("CUDA chassis produced non-finite loads")
    expected = len(solver.steady_problems)
    required = {
        "aic": expected,
        "wake": expected,
        "solve": expected,
        "loads": expected,
        "ledger": expected,
        "wake_convection": expected - 1,
    }
    if any(solver.cuda_counters[key] != value for key, value in required.items()):
        raise RuntimeError(f"CUDA chassis counter drift: {solver.cuda_counters}")
    return solver, force


def _lowpass_last_cycle(force: torch.Tensor, q_area: torch.Tensor, period: float):
    dt = period / SPC

    def lowpass(value: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(value)
        frequency = torch.fft.rfftfreq(value.numel(), d=dt, device=DEVICE)
        return torch.fft.irfft(
            torch.where(frequency <= 1.0, spectrum, 0.0), n=value.numel()
        )

    cl = lowpass(-force[:, 2] / q_area)[-SPC:]
    cd = lowpass(-force[:, 0] / q_area)[-SPC:]
    phase = torch.arange(SPC, dtype=DTYPE, device=DEVICE) / SPC
    return phase, cl, cd


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha(payload: dict) -> str:
    data = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_yang() -> dict:
    case = Yang2025RigidCase()
    q_area = torch.prod(
        _tensor(
            [
                0.5,
                case.rho_kg_m3,
                case.freestream_m_s,
                case.freestream_m_s,
                case.area_m2,
            ]
        )
    )
    cd0 = (2.0 * 1.328 / torch.sqrt(_tensor(case.reynolds))).item()
    observations = {}
    path = (
        ROOT
        / "docs/forward_flight_large_pitch/reproductions/plev2025/source_data/yang2025_fig11_rigid_digitized.csv"
    )
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            observations[float(row["aoa_deg"])] = (
                float(row["test_lift_gf"]),
                -float(row["test_thrust_gf"]),
            )
    lift_errors: list[torch.Tensor] = []
    drag_errors: list[torch.Tensor] = []
    cases = []
    for angle, (gt_lift, gt_drag) in sorted(observations.items()):
        movement = build_yang2025_movement(angle, "full", settings=(8, 12, SPC, 4, 4))
        if isinstance(movement, tuple):
            movement = movement[0]
        solver, force = _run_chassis(movement)
        phase, cl, cd = _lowpass_last_cycle(force, q_area, case.period_s)
        residual = movement_polar_residual_cuda(
            movement,
            source_cycle_step_range=[force.shape[0] - SPC, force.shape[0] - 1],
            period_s=case.period_s,
            freestream_m_s=case.freestream_m_s,
            rho_kg_m3=case.rho_kg_m3,
            aspect_ratio=case.aspect_ratio,
            output_samples=N_OUT,
            device=str(DEVICE),
        )
        ledger = run_ledger_cuda(
            solver.ledger,
            LedgerConfig(
                lesp_crit=0.0872,
                aspect_ratio=case.aspect_ratio,
                rho=case.rho_kg_m3,
                cd0=cd0,
                enable_t1=False,
            ),
            SPC,
            device=str(DEVICE),
        )
        lift_gf = torch.mean(cl * q_area + residual["delta_lift_n"]) / G * 1000.0
        drag_gf = (
            (torch.mean(cd * q_area + residual["delta_drag_n"]) + ledger["mean_t3_N"])
            / G
            * 1000.0
        )
        lift_errors.append(torch.abs(lift_gf - gt_lift))
        drag_errors.append(torch.abs(drag_gf - gt_drag))
        cases.append(
            {
                "aoa_deg": angle,
                "lift_gf": lift_gf,
                "drag_gf": drag_gf,
                "gt_lift_gf": gt_lift,
                "gt_drag_gf": gt_drag,
                "t3_n": ledger["mean_t3_N"],
                "cuda_counters": solver.cuda_counters,
                "phase": phase,
                "cl": cl,
                "cd": cd,
                "phase_sha256": hashlib.sha256(
                    phase.detach().cpu().numpy().tobytes()
                ).hexdigest(),
            }
        )
        print(
            f"Yang AoA {angle:g}: lift={lift_gf.item():.4f} gf drag={drag_gf.item():.4f} gf",
            flush=True,
        )
    return {
        "paper": "Yang et al. 2025 rigid-wing Figure 11",
        "lift_mae_gf": torch.mean(torch.stack(lift_errors)),
        "drag_mae_gf": torch.mean(torch.stack(drag_errors)),
        "v4b_lift_mae_gf": 4.55,
        "v4b_drag_mae_gf": 2.64,
        "cases": cases,
    }


def run_izraelevitz() -> dict:
    case = IzraelevitzSchererCase()
    span = torch.prod(_tensor([case.aspect_ratio, case.chord_m]))
    q_area = torch.prod(
        torch.cat(
            (
                _tensor(
                    [
                        0.5,
                        case.rho_kg_m3,
                        case.freestream_m_s,
                        case.freestream_m_s,
                        case.chord_m,
                    ]
                ),
                span.reshape(1),
            )
        )
    )
    pi_t = _tensor(math.pi)
    omega_star_t = pi_t * case.strouhal / case.heave_to_chord
    period_star_t = 2.0 * pi_t / omega_star_t
    omega_star = float(omega_star_t.item())
    period_star = float(period_star_t.item())
    threshold = LESPThreshold(
        value=float(torch.sin(torch.deg2rad(_tensor(0.90 / 0.065))).item()),
        section_family=case.section_name,
        reynolds=case.freestream_m_s * case.chord_m / case.nu_m2_s,
        source="Scherer static CLa=0.065/deg CLmax=0.90",
    )
    rows = []
    path = (
        ROOT
        / "docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/izraelevitz2017_fig14_digitized.csv"
    )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["data_role"] == "experimental_observation"
        ]
    conditions = sorted(
        {(float(row["theta_max_deg"]), float(row["phase_offset_deg"])) for row in rows}
    )
    errors: list[torch.Tensor] = []
    cases = []
    for theta_max, phase_offset in conditions:
        movement = build_izraelevitz_scherer_movement(
            theta_max, phase_offset, "full", settings=(8, 12, SPC, 4)
        )
        if isinstance(movement, tuple):
            movement = movement[0]
        solver, force = _run_chassis(movement)
        ct_raw = torch.mean(force[-SPC:, 0]) / q_area
        phase = torch.arange(4 * SPC, dtype=DTYPE, device=DEVICE) * (
            2.0 * math.pi / SPC
        )
        theta = torch.deg2rad(_tensor(theta_max))
        offset = torch.deg2rad(_tensor(phase_offset))
        alpha = theta * torch.cos(phase + offset)
        alpha_rate = -theta * omega_star * torch.sin(phase + offset)
        heave_rate = -case.heave_to_chord * omega_star * torch.sin(phase)
        pair = run_ldvm_separation_pair_cuda(
            alpha_rad=alpha,
            alpha_rate_per_convective_time=alpha_rate,
            heave_rate_over_u=heave_rate,
            delta_time_convective=period_star / SPC,
            pivot_fraction_chord=case.pivot_fraction_chord,
            threshold=threshold,
            settings=LDVMSectionSettings(ndiv=50, naterm=24, max_wake_steps=4 * SPC),
            device=str(DEVICE),
        )
        selected = slice(3 * SPC, 4 * SPC)
        projection = project_ldvm_delta_to_finite_wing_cuda(
            pair["delta"]["CNc"][selected],
            pair["delta"]["CNnc"][selected],
            pair["delta"]["CNnonl"][selected],
            pair["delta"]["CSf"][selected],
            alpha[selected],
            aspect_ratio=case.aspect_ratio,
        )
        delta_ct = -torch.mean(projection["delta_CD"])
        ct = ct_raw - 0.057 + delta_ct
        matched = [
            float(row["ct"])
            for row in rows
            if float(row["theta_max_deg"]) == theta_max
            and float(row["phase_offset_deg"]) == phase_offset
        ]
        errors.extend(torch.abs(ct - value) for value in matched)
        cases.append(
            {
                "theta_max_deg": theta_max,
                "phase_offset_deg": phase_offset,
                "ct": ct,
                "ct_raw": ct_raw,
                "delta_ct": delta_ct,
                "gt_ct": matched,
                "cuda_counters": solver.cuda_counters,
            }
        )
        print(f"Izra {theta_max:g}/{phase_offset:g}: CT={ct.item():+.5f}", flush=True)
    return {
        "paper": "Izraelevitz and Scherer 2017 Figure 14",
        "ct_mae": torch.mean(torch.stack(errors)),
        "v4b_ct_mae": 0.0198,
        "cases": cases,
    }


def _build_mancini_movement(case, nc=8, ns=12, spc=128):
    airfoil = pterasoftware.geometry.airfoil.Airfoil(name="naca0012")
    root = pterasoftware.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=case.chord_m,
        num_spanwise_panels=ns,
        spanwise_spacing="cosine",
        control_surface_symmetry_type="symmetric",
    )
    tip = pterasoftware.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=case.chord_m,
        Lp_Wcsp_Lpp=(0.0, case.semispan_m, 0.0),
        num_spanwise_panels=None,
        spanwise_spacing=None,
        control_surface_symmetry_type="symmetric",
    )
    wing = pterasoftware.geometry.wing.Wing(
        name="Mancini CUDA chassis",
        wing_cross_sections=[root, tip],
        symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=nc,
        chordwise_spacing="cosine",
    )
    airplane = pterasoftware.geometry.airplane.Airplane(
        wings=[wing],
        name="Mancini CUDA chassis",
        s_ref=case.area_m2,
        c_ref=case.chord_m,
        b_ref=case.span_m,
    )
    sections = [
        pterasoftware.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in (root, tip)
    ]
    wing_movement = pterasoftware.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=sections,
        ampAngles_Gs_to_Wn_ixyz=(0.0, case.maximum_pitch_deg, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, case.waveform_period_s, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=(
            "sine",
            lambda phase: mancini_periodic_pitch_spacing(phase, case),
            "sine",
        ),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        rotationPointOffset_Gs_Ler=(0.0, 0.0, 0.0),
    )
    airplane_movement = pterasoftware.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wing_movement]
    )
    operating_point = pterasoftware.operating_point.OperatingPoint(
        rho=case.rho_kg_m3,
        vCg__E=case.freestream_m_s,
        alpha=0.0,
        beta=0.0,
        nu=case.nu_m2_s,
    )
    op_movement = (
        pterasoftware.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point
        )
    )
    dt = case.convective_time_s / spc
    steps = int(math.ceil((case.warmup_chords + case.observation_chords) * spc)) + 1
    return pterasoftware.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=op_movement,
        delta_time=dt,
        num_steps=steps,
        max_wake_rows=int(math.ceil(6.0 * spc)),
    )


def _eldredge_cuda(
    time: torch.Tensor, start: float, end: float, smoothing: float
) -> torch.Tensor:
    duration = end - start
    x1, x2 = smoothing * (time - start), smoothing * (time - end)
    first = torch.logaddexp(x1, -x1) - math.log(2.0)
    second = torch.logaddexp(x2, -x2) - math.log(2.0)
    return (first - second + smoothing * duration) / (2.0 * smoothing * duration)


def _gradient_cuda(value: torch.Tensor, spacing: float) -> torch.Tensor:
    out = torch.empty_like(value)
    out[1:-1] = (value[2:] - value[:-2]) / (2.0 * spacing)
    out[0] = (-3.0 * value[0] + 4.0 * value[1] - value[2]) / (2.0 * spacing)
    out[-1] = (3.0 * value[-1] - 4.0 * value[-2] + value[-3]) / (2.0 * spacing)
    return out


def run_mancini() -> dict:
    experiment = load_mancini_fig4_13b_experiment()
    target = torch.linspace(0.0, 5.0, 501, dtype=DTYPE, device=DEVICE)
    gt_time = torch.linspace(0.0, 13.0, 1301, dtype=DTYPE, device=DEVICE)
    cases = []
    for case_id in ("fast_pitch", "slow_pitch"):
        case = MANCINI_2017_CASES[case_id]
        movement = _build_mancini_movement(case)
        solver, force = _run_chassis(movement)
        q_area = torch.prod(
            _tensor(
                [
                    0.5,
                    case.rho_kg_m3,
                    case.freestream_m_s,
                    case.freestream_m_s,
                    case.area_m2,
                ]
            )
        )
        time_all = (
            torch.arange(force.shape[0], dtype=DTYPE, device=DEVICE)
            * movement.delta_time
            / case.convective_time_s
            - case.warmup_chords
        )
        valid = (time_all >= -1.0e-10) & (time_all <= case.observation_chords + 1.0e-10)
        cl_bare = _linear_interp(
            target, time_all[valid], (-force[:, 2] / q_area)[valid]
        )
        dtc = 1.0 / 96.0
        total = case.warmup_chords + case.observation_chords
        count = int(math.floor(total / dtc + 0.5)) + 1
        shifted = (
            torch.arange(count, dtype=DTYPE, device=DEVICE) * dtc - case.warmup_chords
        )
        step = _eldredge_cuda(
            shifted, 0.0, case.acceleration_distance_chords, case.eldredge_smoothing
        )
        alpha = (
            torch.deg2rad(_tensor(case.maximum_pitch_deg - case.initial_pitch_deg))
            * step
        )
        alpha_rate = _gradient_cuda(alpha, dtc)
        threshold = LESPThreshold(
            value=0.11,
            section_family="5% rounded plate",
            reynolds=case.reynolds,
            source="frozen V4B Lcrit=0.11",
            source_role="published transfer hypothesis",
        )
        pair = run_ldvm_separation_pair_cuda(
            alpha_rad=alpha,
            alpha_rate_per_convective_time=alpha_rate,
            heave_rate_over_u=torch.zeros_like(alpha),
            delta_time_convective=dtc,
            pivot_fraction_chord=case.pivot_fraction_chord,
            threshold=threshold,
            settings=LDVMSectionSettings(
                ndiv=50,
                naterm=24,
                core_radius_time_step_ratio=1.3,
                max_wake_steps=count,
            ),
            device=str(DEVICE),
        )
        selected = (shifted >= -1.0e-10) & (
            shifted <= case.observation_chords + 1.0e-10
        )
        sampled = {
            key: _linear_interp(target, shifted[selected], value[selected])
            for key, value in pair["delta"].items()
        }
        sampled_alpha = _linear_interp(target, shifted[selected], alpha[selected])
        projection = project_ldvm_delta_to_finite_wing_cuda(
            sampled["CNc"],
            sampled["CNnc"],
            sampled["CNnonl"],
            sampled["CSf"],
            sampled_alpha,
            aspect_ratio=case.aspect_ratio,
        )
        cl = cl_bare + projection["delta_CL"]
        gt_full = _tensor(experiment[f"CL_{case_id}"])
        gt = _linear_interp(target, gt_time, gt_full)
        rmse = torch.sqrt(torch.mean((cl - gt) ** 2))
        bare_rmse = torch.sqrt(torch.mean((cl_bare - gt) ** 2))
        cases.append(
            {
                "case_id": case_id,
                "rmse": rmse,
                "bare_rmse": bare_rmse,
                "v4b_rmse": 1.2184 if case_id == "fast_pitch" else 0.2908,
                "cuda_counters": solver.cuda_counters,
                "samples": int(target.numel()),
                "time_convective": target,
                "prediction_cl": cl,
                "bare_prediction_cl": cl_bare,
                "experiment_cl": gt,
            }
        )
        print(
            f"Mancini {case_id}: RMSE={rmse.item():.6f} bare={bare_rmse.item():.6f}",
            flush=True,
        )
    return {"paper": "Mancini 2017 Figure 4.13b", "cases": cases}


def main() -> int:
    _require_cuda()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with GpuRuntimeMonitor() as monitor:
        yang = run_yang()
        izraelevitz = run_izraelevitz()
        mancini = run_mancini()
        torch.cuda.synchronize()
    gpu_evidence = asdict(monitor.evidence())
    gt_paths = {
        "yang": ROOT
        / "docs/forward_flight_large_pitch/reproductions/plev2025/source_data/yang2025_fig11_rigid_digitized.csv",
        "izraelevitz": ROOT
        / "docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/izraelevitz2017_fig14_digitized.csv",
        "mancini": ROOT
        / "docs/forward_flight_large_pitch/reproductions/mancini2017_v4b_20260820/source_data/mancini2017_fig4_13b_pitch_lift_digitized.csv",
    }
    mancini_by_id = {row["case_id"]: row for row in mancini["cases"]}
    metrics = {
        "yang_lift_mae_gf": yang["lift_mae_gf"],
        "yang_drag_mae_gf": yang["drag_mae_gf"],
        "izra_ct_mae": izraelevitz["ct_mae"],
        "mancini_fast_rmse": mancini_by_id["fast_pitch"]["rmse"],
        "mancini_slow_rmse": mancini_by_id["slow_pitch"]["rmse"],
    }
    result = {
        "schema": "fluxv-v5m-four-paper-gpu-only-v2",
        "execution_class": "cuda-only-numerical-python-orchestration",
        "base_commit": "fa8eaca9bcaa4b963ecf41683bf77d3c9e3df169",
        "source_sha256": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "ptera_cuda_backend": _file_sha256(
                ROOT / "platform/warp_vpm/bing_joint_ptera_gpu.py"
            ),
            "cuda_corrections": _file_sha256(
                ROOT / "platform/warp_vpm/bing_gpu_corrections.py"
            ),
            "ldvm_torch_gpu": _file_sha256(
                ROOT / "platform/warp_vpm/ldvm_torch_gpu.py"
            ),
            "gpu_runtime_monitor": _file_sha256(
                ROOT / "platform/warp_vpm/gpu_runtime_monitor.py"
            ),
        },
        "ground_truth_sha256": {
            name: _file_sha256(path) for name, path in gt_paths.items()
        },
        "gpu_device": gpu_evidence["gpu_device"],
        "cuda_kernel_path": (
            "torch-cuda-aic-wake-convection-solve-loads-ldvm-polar-ledger-v2"
        ),
        "gpu_utilization_observed": gpu_evidence["gpu_utilization_observed"],
        "gpu_memory_peak_mib": gpu_evidence["gpu_memory_peak_mib"],
        "gpu_runtime_evidence": gpu_evidence,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "metrics": metrics,
        "yang": yang,
        "izraelevitz": izraelevitz,
        "mancini": mancini,
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    result["result_sha256"] = _sha(result)
    payload = json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False)
    (OUT / "summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
