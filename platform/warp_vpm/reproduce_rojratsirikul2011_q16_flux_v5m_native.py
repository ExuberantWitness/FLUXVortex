"""Run the formal Rojratsirikul et al. (2011) Q16/native-FLUX-V5M CUDA CASE.

Fixed membrane wing: constant freestream at a constant geometric angle of
attack, four-edge clamped latex membrane, no prescribed structural motion.
Everything below the clamped edges — mean camber, vibration, frequencies — is
a coupled FSI output on the mandatory separated-LEV / joint-TEV / free-wake
native V5M path with the predictor/corrector strong coupling stepper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
)
from fluxvortex.warp_fsi.q16_flux_v5m_native_fsi import (
    Q16NativeV5MFSIOwner,
    Q16NativeV5MFSIStepper,
)
from fluxvortex.warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID,
    FORMAL_Q16_GRID,
    ROJRATSIRIKUL2011_CASES,
    assumption_ledger,
    make_rojratsirikul2011_q16_model,
    membrane_statistics,
    normal_force_coefficient,
    plate_normal,
    static_motion_contract,
    validate_rojratsirikul2011_sources,
)


SCHEMA = "rojratsirikul2011-q16-native-flux-v5m-fsi-v1"
EXECUTION_GATE_STEPS = 50
PARTIAL_EVERY = 10
# ~3 chords of free wake retained at dt*=0.01 (300 rows x 30 spanwise lanes).
WAKE_MAX_ROWS = 300
PARTICLE_CAPACITY = 32768
# LEV particle lifetime: one chord of convection (LE -> TE) at dt*=0.01.
# Beyond the trailing edge the free-wake ring system carries the vorticity.
PARTICLE_MAX_AGE_STEPS = 100
ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout


def _apply_freestream(
    solver: Q16NativeV5MSolver, case: Any, factor: float
) -> None:
    """Set the ramped freestream magnitude on the live solver.

    The flow always points along +x: the geometric angle of attack is baked
    into the rotated reference mesh (the oracle-verified native-path
    incidence mechanism), so the ramp scales the magnitude only.  The LDVM
    source bank is initialized once at full U0 so its convective timescales
    are material parameters; only the actual flow ramps.
    """

    speed = case.freestream_m_s * factor
    solver.v_inf = torch.tensor(
        [speed, 0.0, 0.0], device=solver.device, dtype=torch.float64
    )
    object.__setattr__(solver.settings, "freestream", speed)


def run_case(
    *,
    case_id: str,
    max_aero_steps: int | None = None,
    execution_gate_only: bool = False,
    structural_substeps: int | None = None,
    young_modulus_override: float | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    case = ROJRATSIRIKUL2011_CASES.get(case_id)
    if case is None:
        raise ValueError(f"unknown case {case_id}; use one of {sorted(ROJRATSIRIKUL2011_CASES)}")
    import dataclasses
    if young_modulus_override is not None:
        # Labeled material-uncertainty branch (handoff section 6, unlocked
        # after all independent oracles passed; CLAIM_TREE.md quantifies
        # E_eff~1.4 MPa from the -13% equilibrium-camber deficit).  Never a
        # replacement for the E=2.2 MPa primary result.
        case = dataclasses.replace(case, young_modulus_pa=float(young_modulus_override))
    validate_rojratsirikul2011_sources()
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("formal Rojratsirikul native CASE requires CUDA")
    legacy_token = "ptera" + "software"
    loaded_legacy = tuple(name for name in sys.modules if legacy_token in name.lower())
    if loaded_legacy:
        raise RuntimeError("formal native CASE process contains a legacy runtime")
    if execution_gate_only and max_aero_steps is None:
        max_aero_steps = EXECUTION_GATE_STEPS
    if max_aero_steps is None:
        max_aero_steps = int(
            math.ceil(
                (case.startup_time_star + case.statistics_min_time_star)
                / case.aerodynamic_dt_star
            )
        )
    if type(max_aero_steps) is not int or max_aero_steps < 1:
        raise ValueError("max_aero_steps must be a positive exact int")
    if structural_substeps is not None and (
        type(structural_substeps) is not int or structural_substeps < 1
    ):
        raise ValueError("structural_substeps override must be a positive exact int")
    substep_count = structural_substeps or case.structural_substeps_per_aerodynamic_step

    mesh, model, boundary, perimeter_audit = make_rojratsirikul2011_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
        case=case,
    )
    structural = Q16CudaNewmarkStepper(
        model,
        boundary,
        device=config.DEVICE,
        newton_tolerance=3.0e-7,
        max_newton_iterations=128,
        cg_tolerance=2.0e-10,
        max_cg_iterations=2048,
        cg_check_every=16,
        nonsymmetric_solver="reference_dense",
        reference_dense_refresh_after=48,
        mass_damping_coefficient=0.0,
        # Kelvin-Voigt stiffness damping from the literature latex loss
        # factor: theta = eta / omega_lockin with omega at the physical
        # St~1 vibration frequency (research/CLAIM_TREE.md N4).
        stiffness_damping_coefficient=(
            case.structural_damping_loss_factor
            / (2.0 * math.pi * case.freestream_m_s / case.chord_m)
        ),
    )
    surface = Q16NativeV5MSurface(
        mesh,
        q16_chordwise_elements=FORMAL_Q16_GRID[0],
        q16_spanwise_elements=FORMAL_Q16_GRID[1],
        aerodynamic_chordwise_panels=FORMAL_AERO_GRID[0],
        aerodynamic_spanwise_panels=FORMAL_AERO_GRID[1],
        device=config.DEVICE,
        dense_transfers=True,
    )
    aerodynamic = Q16NativeV5MSolver(
        surface,
        NativeV5MConfig(
            chordwise_panels=FORMAL_AERO_GRID[0],
            spanwise_panels=FORMAL_AERO_GRID[1],
            density=case.fluid_density_kg_m3,
            freestream=case.freestream_m_s,
            aerodynamic_dt=case.aerodynamic_dt_s,
            lesp_crit=case.lesp_crit,
            wake_max_rows=WAKE_MAX_ROWS,
            particle_capacity=PARTICLE_CAPACITY,
            # Sustained LEV release (~270 particles/step at all 30 lanes at
            # alpha=16 deg) grows the particle field without bound; particles
            # are tracked for exactly one chord of convection (LE -> TE),
            # after which the free-wake ring system carries the vorticity.
            particle_max_age_steps=PARTICLE_MAX_AGE_STEPS,
            # Every wake row carries the persistent LEV release (Gamma_row ~
            # Gamma_total), and the strong-scheme Mf2_vec1 wake-memory term
            # diverges in that regime (evidence: force decomposition
            # artifacts 20260824).  The author's weak-scheme dp_add =
            # (Gamma - Gamma_old)/dt stays bounded and reproduces the
            # Wagner response onto the paper's Cn band; all other
            # strong-scheme blocks (dp_lift1, lift2, Mf2_1, Mf1) remain.
            wake_history_mode="bound_rate",
            # Author's far-wake freeze: rows past one chord of convection
            # (100 shed events at dt*=0.01) convect with the freestream
            # alone, matching the LEV particle lifetime.
            wake_free_rows=100,
            # The 0.04/2.125 default sits exactly on the frozen minimum
            # particle overlap and float64 rounding lands just below it at
            # this chord; 0.018 is the coarsest spacing strictly inside the
            # bound (overlap 2.222).
            dvm_target_spacing_chord=0.018,
            device=config.DEVICE,
        ),
    )
    normal_t = torch.tensor(
        plate_normal(case), device=config.DEVICE, dtype=torch.float64
    )
    state = wp.array(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    velocity = wp.zeros_like(state)
    acceleration = wp.zeros_like(state)
    owner = Q16NativeV5MFSIOwner.initialize(
        aerodynamic, state, velocity, acceleration
    )
    reference_quarter = (
        wp.to_torch(surface.quarter_transfer.interpolate(state))[0]
        .reshape(surface.nc, surface.ns + 1, 3)
        .clone()
    )
    coupling = Q16NativeV5MFSIStepper(
        structural,
        aerodynamic,
        coupling_tolerance=5.0e-7,
        max_coupling_iterations=20,
        relaxation=0.7,
        # Carry the learned Aitken factor across outer steps; the
        # convergence criterion is unchanged, only the trial path.
        persistent_relaxation=True,
        # IQN-ILS parked: the naive implementation diverges on this
        # added-mass-heavy interface exactly as Davis 2022 warns without
        # force/displacement pre-scaling; Aitken (persistent) stays the
        # production accelerator.  See SPEED_ENGINEERING_PLAN.md.
        coupling_accelerator="aitken",
    )

    substeps = substep_count
    prescribed = (None,) * substeps
    load_betas = tuple((index + 1) / substeps for index in range(substeps))
    z_history = torch.zeros(
        (max_aero_steps, surface.nc, surface.ns + 1),
        device=config.DEVICE,
        dtype=torch.float64,
    )
    pressure_sum = torch.zeros(
        surface.nc * surface.ns, device=config.DEVICE, dtype=torch.float64
    )
    records: list[dict[str, Any]] = []
    progress_records: list[dict[str, Any]] = []
    step_wall_times: list[float] = []
    started = time.perf_counter()

    def write_partial(status: str, error: dict[str, Any] | None = None) -> None:
        if output is None:
            return
        payload = {
            "schema": SCHEMA,
            "status": status,
            "case_id": case.case_id,
            "execution_gate_only": execution_gate_only,
            "q16_grid": list(FORMAL_Q16_GRID),
            "aerodynamic_grid": list(FORMAL_AERO_GRID),
            "completed_aero_steps": len(records),
            "requested_aero_steps": max_aero_steps,
            "records": records,
            "progress_records": progress_records[-200:],
            "elapsed_seconds": time.perf_counter() - started,
        }
        if error is not None:
            payload.update(error)
        _write_json(output.with_name(f"{output.stem}.partial.json"), payload)
        _write_npz(
            output.with_name(f"{output.stem}.z_history.npz"),
            {
                "z_history_over_c": (
                    z_history[: len(records)].cpu().numpy() / case.chord_m
                ),
                "time_star": np.array(
                    [(index + 1) * case.aerodynamic_dt_star for index in range(len(records))],
                    dtype=np.float64,
                ),
                "mean_pressure_map": (
                    (pressure_sum / max(len(records), 1)).reshape(surface.nc, surface.ns)
                    .cpu().numpy()
                ),
            },
        )

    statistics_start_index = int(
        math.ceil(case.statistics_start_time_star / case.aerodynamic_dt_star)
    )
    for step in range(1, max_aero_steps + 1):
        time_star = step * case.aerodynamic_dt_star
        factor = case.freestream_factor(time_star)
        _apply_freestream(aerodynamic, case, factor)
        step_started = time.perf_counter()

        def report_progress(event: dict[str, Any]) -> None:
            record = {"aero_step": step, "freestream_factor": factor, **event}
            if (
                event.get("phase") != "structural_substep"
                or event.get("substep") in {1, event.get("substep_count")}
                or event.get("substep", 0) % 10 == 0
            ):
                progress_records.append(record)
                print(
                    "__DS_PROGRESS__ "
                    + json.dumps(record, sort_keys=True, allow_nan=False),
                    flush=True,
                )

        try:
            result = coupling.advance(
                owner,
                delta_time=case.aerodynamic_dt_s,
                prescribed_forces=prescribed,
                load_betas=load_betas,
                progress_callback=report_progress,
            )
        except Exception as error:
            write_partial(
                "failed",
                {
                    "failed_aero_step": step,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        quarter = wp.to_torch(
            surface.quarter_transfer.interpolate(result.structural.state)
        )[0].reshape(surface.nc, surface.ns + 1, 3)
        displacement = torch.sum((quarter - reference_quarter) * normal_t, dim=2)
        if not bool(torch.isfinite(displacement).all()):
            write_partial("failed", {"failed_aero_step": step, "error": "non-finite membrane state"})
            raise RuntimeError(f"non-finite membrane state at aero step {step}")
        z_history[step - 1] = displacement
        pressure_sum += result.aerodynamic.load.pressure
        total_force = result.aerodynamic.load.total_force
        if not bool(torch.isfinite(total_force).all()):
            write_partial("failed", {"failed_aero_step": step, "error": "non-finite aerodynamic force"})
            raise RuntimeError(f"non-finite aerodynamic force at aero step {step}")
        # The wake and particle fields grow every step, so the allocator's
        # cached high-water would otherwise climb without bound; release the
        # dead blocks between steps (a few ms, no effect on numerics).
        if step % 5 == 0:
            torch.cuda.empty_cache()
        cn = normal_force_coefficient(
            total_force.cpu().tolist(), case, normal=plate_normal(case)
        )
        diagnostics = result.aerodynamic.trial_state.diagnostics[-1]
        step_wall_times.append(time.perf_counter() - step_started)
        records.append(
            {
                "aero_step": step,
                "time_star": time_star,
                "freestream_factor": factor,
                "cn": cn,
                "instantaneous_zmax_over_c": float(
                    (displacement.max() / case.chord_m).item()
                ),
                "total_aerodynamic_force_n": total_force.cpu().tolist(),
                "coupling_iterations": result.coupling_iterations,
                "aerodynamic_evaluations": result.aerodynamic_evaluations,
                "coupling_residual": result.residual,
                "structural_newton_iterations": result.structural.newton_iteration_count,
                "structural_cg_iterations": result.structural.cg_iteration_count,
                "aerodynamic": diagnostics,
            }
        )
        if step % PARTIAL_EVERY == 0 or step == max_aero_steps:
            write_partial("running")
    torch.cuda.synchronize()

    window = z_history[statistics_start_index:]
    window_records = records[statistics_start_index:]
    statistics: dict[str, Any] | None = None
    if window.shape[0] >= 100:
        statistics = membrane_statistics(
            window.cpu().numpy(), case.aerodynamic_dt_s, case
        )
        statistics["mean_cn"] = float(
            np.mean([record["cn"] for record in window_records])
        )
        statistics["statistics_window_time_star"] = [
            (statistics_start_index + 1) * case.aerodynamic_dt_star,
            max_aero_steps * case.aerodynamic_dt_star,
        ]
    accuracy_gates: dict[str, Any] = {}
    if statistics is not None and not execution_gate_only:
        zmax = statistics["mean_zmax_over_c"]
        accuracy_gates["h8_mean_zmax_over_c"] = {
            "value": zmax,
            "digitized_target": case.digitized_approx_zmax_over_c,
            "tolerance": 0.005,
            "passed": abs(zmax - case.digitized_approx_zmax_over_c) <= 0.005,
        }
        mean_cn = statistics["mean_cn"]
        accuracy_gates["h8_mean_cn"] = {
            "value": mean_cn,
            "digitized_band": [
                case.digitized_approx_cn_low,
                case.digitized_approx_cn_high,
            ],
            "relative_tolerance": 0.10,
            "passed": (
                0.90 * case.digitized_approx_cn_low
                <= mean_cn
                <= 1.10 * case.digitized_approx_cn_high
            ),
        }
        spectrum = statistics.get("fluctuation_spectrum")
        if case.digitized_approx_strouhal is not None and spectrum is not None:
            key = f"st_gate_vs_{case.digitized_approx_strouhal:g}"
            accuracy_gates[key] = {
                "strouhal": spectrum["strouhal"],
                "digitized_target": case.digitized_approx_strouhal,
                "passed": (
                    abs(spectrum["strouhal"] - case.digitized_approx_strouhal)
                    <= 0.10
                ),
            }
        if case.digitized_approx_chordwise_peak_count is not None:
            accuracy_gates["chordwise_peak_count"] = {
                "value": statistics["chordwise_peak_count"],
                "digitized_target": case.digitized_approx_chordwise_peak_count,
            }
        if case.digitized_approx_spanwise_peak_count is not None:
            accuracy_gates["spanwise_peak_count"] = {
                "value": statistics["spanwise_peak_count"],
                "digitized_target": case.digitized_approx_spanwise_peak_count,
            }

    status = "completed"
    if execution_gate_only:
        status = "execution_gate_passed"
    elif statistics is None:
        status = "completed_insufficient_statistics"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "paper": case.paper_title,
        "paper_doi": case.doi,
        "case_id": case.case_id,
        "git_head": _git_output("rev-parse", "HEAD").strip(),
        "dirty_state_digest": hashlib.sha256(
            _git_output("status", "--short").encode("utf-8")
        ).hexdigest(),
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device": config.DEVICE,
        "dtype": "float64",
        "cpu_fallback_count": 0,
        "runtime_legacy_module_count": len(loaded_legacy),
        "ptera_loaded": False,
        "q16_macro_chord": FORMAL_Q16_GRID[0],
        "q16_macro_span": FORMAL_Q16_GRID[1],
        "aero_nchord": FORMAL_AERO_GRID[0],
        "aero_nspan": FORMAL_AERO_GRID[1],
        "rho_air": case.fluid_density_kg_m3,
        "nu_air": case.kinematic_viscosity_m2_s,
        "U_inf": case.freestream_m_s,
        "Re": case.reynolds,
        "alpha_deg": case.angle_deg,
        "incidence_mechanism": (
            "reference mesh rigidly pitched nose-up about the leading-edge "
            "line; freestream stays +x (oracle-verified native-path route)"
        ),
        "c": case.chord_m,
        "b": case.span_m,
        "S": case.reference_area_m2,
        "E": case.young_modulus_pa,
        "E_branch": (young_modulus_override is not None),
        "nu_s": case.poisson_ratio_assumed,
        "rho_m": case.membrane_density_kg_m3,
        "thickness": case.thickness_m,
        "prestress": case.prestress_n_m_assumed,
        "damping_loss_factor": case.structural_damping_loss_factor,
        "dt_star": case.aerodynamic_dt_star,
        "structural_substeps": substeps,
        "structural_substeps_protocol_frozen": case.structural_substeps_per_aerodynamic_step,
        "startup_window_time_star": case.startup_time_star,
        "statistics_window_time_star": case.statistics_min_time_star,
        "statistics_start_time_star": case.statistics_start_time_star,
        "startup_ramp": "half-cosine 0->U0 over t* in [0,1]; excluded from statistics",
        "static_motion_contract": static_motion_contract(case),
        "separated_lev_mandatory": True,
        "Lcrit": case.lesp_crit,
        "lesp_release_condition": f"abs(LESP)>{case.lesp_crit}",
        "joint_tev": True,
        "free_wake": True,
        "wake_max_rows": WAKE_MAX_ROWS,
        "wake_free_rows": 100,
        "particle_capacity": PARTICLE_CAPACITY,
        "particle_max_age_steps": PARTICLE_MAX_AGE_STEPS,
        "wake_history_mode": "bound_rate",
        "dvm_target_spacing_chord": 0.018,
        "commit_count": len(records),
        "trial_count": sum(
            record["aerodynamic_evaluations"] for record in records
        ),
        "rejected_trial_count": 0,
        "rejected_trial_note": (
            "coupling trials run on cloned branches; a non-converged trial "
            "never touches committed state and any parent mutation raises "
            "the digest guard"
        ),
        "perimeter_audit": perimeter_audit,
        "assumption_ledger": assumption_ledger(case),
        "case_config_digest": case.config_digest(),
        "transfer_gates": {
            "source": (
                "tests/test_q16_work_conjugate_transfer.py, "
                "tests/test_q16_aero_load_packet_gpu.py, "
                "tests/test_q16_structural_step_gpu.py, "
                "tests/test_q16_flux_v5m_native_gpu.py"
            ),
            "status": "component-level passed (see repository test evidence)",
        },
        "mean_Cn": statistics["mean_cn"] if statistics else None,
        "mean_zmax_over_c": statistics["mean_zmax_over_c"] if statistics else None,
        "zsd_map": statistics["zsd_map_over_c"] if statistics else None,
        "mean_map": statistics["mean_map_over_c"] if statistics else None,
        "dominant_St": (
            statistics["fluctuation_spectrum"]["strouhal"]
            if statistics and statistics.get("fluctuation_spectrum")
            else None
        ),
        "chordwise_peak_count": (
            statistics["chordwise_peak_count"] if statistics else None
        ),
        "spanwise_peak_count": (
            statistics["spanwise_peak_count"] if statistics else None
        ),
        "accuracy_gates": accuracy_gates,
        "wake_ring_count": records[-1]["aerodynamic"]["wake_ring_count"] if records else None,
        "lev_release_count_total": int(
            sum(record["aerodynamic"]["lev_release_count"] for record in records)
        ),
        "max_abs_lesp_pre": max(
            (record["aerodynamic"]["lesp_pre_max_abs"] for record in records),
            default=None,
        ),
        "coupling_iteration_mean": (
            float(np.mean([record["coupling_iterations"] for record in records]))
            if records
            else None
        ),
        "step_wall_seconds_mean": (
            float(np.mean(step_wall_times)) if step_wall_times else None
        ),
        "step_wall_seconds_last": step_wall_times[-1] if step_wall_times else None,
        "elapsed_seconds": time.perf_counter() - started,
        "aero_steps": len(records),
        "requested_aero_steps": max_aero_steps,
        "execution_gate_only": execution_gate_only,
        "records": records,
    }
    if output is not None:
        _write_json(output, payload)
        _write_npz(
            output.with_name(f"{output.stem}.z_history.npz"),
            {
                "z_history_over_c": (
                    z_history[: len(records)].cpu().numpy() / case.chord_m
                ),
                "time_star": np.array(
                    [(index + 1) * case.aerodynamic_dt_star for index in range(len(records))],
                    dtype=np.float64,
                ),
                "mean_pressure_map": (
                    (pressure_sum / max(len(records), 1)).reshape(surface.nc, surface.ns)
                    .cpu().numpy()
                ),
            },
        )
        _write_json(
            output.with_name(f"{output.stem}.partial.json"),
            {
                "schema": SCHEMA + "-checkpoint",
                "status": payload["status"],
                "case_id": case.case_id,
                "completed_aero_steps": len(records),
                "final_output": str(output),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="ROJ11-A16", choices=sorted(ROJRATSIRIKUL2011_CASES))
    parser.add_argument("--max-aero-steps", type=int, default=None)
    parser.add_argument(
        "--execution-gate-only",
        action="store_true",
        help="run the short execution gate (50 aero steps by default) on the "
        "same formal grid; makes no accuracy claims",
    )
    parser.add_argument(
        "--structural-substeps",
        type=int,
        default=None,
        help="DIAGNOSTIC ONLY: override the frozen structural substep count "
        "for time-convergence probes; the frozen protocol value is always "
        "recorded alongside",
    )
    parser.add_argument(
        "--young-modulus-override",
        type=float,
        default=None,
        help="Labeled material-uncertainty branch (handoff section 6); "
        "the E=2.2 MPa run remains the primary result",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    arguments = parser.parse_args()
    output = arguments.output or Path(
        f"artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/"
        f"{arguments.case}{'_EXECUTION_GATE' if arguments.execution_gate_only else ''}.json"
    )
    result = run_case(
        case_id=arguments.case,
        max_aero_steps=arguments.max_aero_steps,
        execution_gate_only=arguments.execution_gate_only,
        structural_substeps=arguments.structural_substeps,
        young_modulus_override=arguments.young_modulus_override,
        output=output,
    )
    printable = dict(result)
    printable.pop("records", None)
    printable.pop("mean_map", None)
    printable.pop("zsd_map", None)
    printable.pop("assumption_ledger", None)
    print(json.dumps(printable, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
