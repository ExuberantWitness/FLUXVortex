"""Resumable rigid-wing queue for Rojratsirikul 2011 Figures 9/12/13/15.

HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829 §8 stage D, driven by the
SAME frozen native V5M machinery as the membrane CaseRunner runs (aero grid
15x30, lesp_crit 0.11, ``bound_rate`` wake history, wake rows 300/100,
particle cap 32768 / age 100, dt* = 0.01, CUDA float64).  Every case is a
fixed-pitch rigid rectangular plate (AR = 2, c = 0.0688 m, b = 0.1375 m).

Each (U, alpha) case yields:
  * Figure 9 rigid Cn: mean of F . n_hat / qS over t* in [10, T*];
  * Figure 12/13/15 wake shedding: WakeProbeObserver spectra over the same
    committed states (12 inertial probes at 1c/2c downstream), St and
    St*sin(alpha) with exact closure;
  * retention evidence (particle/wake counts, age-cull totals) so spectral
    jumps from retention can be audited.

Queue order (earlier = higher priority; the checkpoint file makes the queue
resumable and partial completion is still scorable):
  1. U = 10, alpha = 15   (Figure 12 anchor, Re = 48,700, t* = 40)
  2. U = 5,  alpha = 9..25 odd + curve ends  (Re = 24,300, t* = 30)
  3. U = 7.5, alpha = 9..25 odd              (Re = 36,500, t* = 30)
  4. U = 10, remaining curve points          (Re = 48,700, t* = 30)

Outputs (one baseline directory, never overwritten):
  model_observables.csv   -- P2 schema, one row per finished case
  cases/<run_id>.json     -- full force/St/probe evidence per case
  checkpoint.json         -- completed case ids (resume marker)
  run.log
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import time
from pathlib import Path

import torch

from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidNativeV5MSolver,
    RigidV5MSurface,
)
from fluxvortex.warp_fsi.wake_probe_observer import (
    WakeProbeObserver,
    build_downstream_probe_grid,
    summarize_probes,
)

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("roj_rigid_queue")

# ── Frozen Roj shared constants (identical to the membrane runner) ────────
CHORD_M = 0.0688
SPAN_M = 0.1375
DENSITY = 1.208
DT_STAR = 0.01
LESP_CRIT = 0.11
AERO_GRID = (15, 30)
WAKE_MAX_ROWS = 300
WAKE_FREE_ROWS = 100
PARTICLE_CAPACITY = 32768
PARTICLE_MAX_AGE_STEPS = 100
DVM_TARGET_SPACING_CHORD = 0.018
AREA_M2 = CHORD_M * SPAN_M

# (U, alpha, t*_total); Figure 12 anchor first, then Re=24,300 / 36,500
# curves, then the Re=48,700 remainder.
QUEUE: tuple[tuple[float, float, float], ...] = (
    (10.0, 15.0, 40.0),
    (5.0, 0.0, 30.0),
    (5.0, 5.0, 30.0),
    (5.0, 9.0, 30.0),
    (5.0, 11.0, 30.0),
    (5.0, 13.0, 30.0),
    (5.0, 15.0, 30.0),
    (5.0, 17.0, 30.0),
    (5.0, 19.0, 30.0),
    (5.0, 21.0, 30.0),
    (5.0, 23.0, 30.0),
    (5.0, 25.0, 30.0),
    (5.0, 27.0, 30.0),
    (5.0, 30.0, 30.0),
    (7.5, 9.0, 30.0),
    (7.5, 11.0, 30.0),
    (7.5, 13.0, 30.0),
    (7.5, 15.0, 30.0),
    (7.5, 17.0, 30.0),
    (7.5, 19.0, 30.0),
    (7.5, 21.0, 30.0),
    (7.5, 23.0, 30.0),
    (7.5, 25.0, 30.0),
    (10.0, 0.0, 30.0),
    (10.0, 5.0, 30.0),
    (10.0, 9.0, 30.0),
    (10.0, 11.0, 30.0),
    (10.0, 13.0, 30.0),
    (10.0, 17.0, 30.0),
    (10.0, 19.0, 30.0),
    (10.0, 21.0, 30.0),
    (10.0, 23.0, 30.0),
    (10.0, 25.0, 30.0),
    (10.0, 30.0, 30.0),
)
CN_WINDOW_STAR = (10.0, None)  # (start, end); None = run end
PSD_WINDOW_STAR = (10.0, None)

OBSERVABLES_FIELDS = [
    "run_id",
    "case_id",
    "branch",
    "U_m_s",
    "Re",
    "alpha_deg",
    "zmax_over_c_mean_map",
    "Cn_mean",
    "St",
    "St_modified",
    "stationary",
    "statistics_start_tstar",
    "statistics_end_tstar",
    "n_samples",
    "gpu_name",
    "git_commit",
    "source_status",
    "failure_reason",
]


def fixed_pitch_law(alpha_deg: float, device: str):
    theta = math.radians(alpha_deg)
    half = 0.5 * theta
    dtype = torch.float64
    pos = torch.zeros(3, device=device, dtype=dtype)
    quat = torch.tensor(
        [math.cos(half), 0.0, math.sin(half), 0.0], device=device, dtype=dtype
    )
    zero = torch.zeros(3, device=device, dtype=dtype)

    def law(t: float):
        return pos, quat, zero, zero.clone()

    return law


def reynolds(U: float) -> int:
    return int(round(U * CHORD_M / 1.4142e-5))


def run_case(
    U: float,
    alpha_deg: float,
    t_star_total: float,
    *,
    device: str,
    log: logging.Logger,
) -> dict:
    nc, ns = AERO_GRID
    aero_dt = DT_STAR * CHORD_M / U
    total_steps = int(round(t_star_total / DT_STAR))
    q_area = 0.5 * DENSITY * U**2 * AREA_M2
    run_id = f"ROJR-RIGID-U{str(U).replace('.', 'p')}-A{int(round(alpha_deg * 10)):03d}"
    reference = build_izra_reference_grid(
        chordwise_panels=nc,
        spanwise_panels=ns,
        chord_m=CHORD_M,
        span_m=SPAN_M,
        pivot_fraction_chord=0.25,
        device=device,
    )
    kin = PrescribedRigidSurfaceKinematics(
        reference,
        fixed_pitch_law(alpha_deg, device),
        surface_id="roj_rigid_wing",
        body_id="body_0",
    )
    surface = RigidV5MSurface(chordwise_panels=nc, spanwise_panels=ns, device=device)
    settings = NativeV5MConfig(
        chordwise_panels=nc,
        spanwise_panels=ns,
        density=DENSITY,
        freestream=U,
        aerodynamic_dt=aero_dt,
        lesp_crit=LESP_CRIT,
        wake_max_rows=WAKE_MAX_ROWS,
        particle_capacity=PARTICLE_CAPACITY,
        particle_max_age_steps=PARTICLE_MAX_AGE_STEPS,
        wake_history_mode="bound_rate",
        wake_free_rows=WAKE_FREE_ROWS,
        dvm_target_spacing_chord=DVM_TARGET_SPACING_CHORD,
        device=device,
    )
    solver = RigidNativeV5MSolver(surface, settings)
    stepper = RigidV5MStepper(solver, device=device)
    owner = stepper.initialize((kin.evaluate(0.0),))
    geometry = solver.surface.evaluate(None, None)
    probes = build_downstream_probe_grid(
        trailing_edge_world=geometry.trailing_edge,
        chord_m=CHORD_M,
        span_m=SPAN_M,
        freestream_direction=solver.v_inf,
        device=device,
    )
    observer = WakeProbeObserver(probes, device=device)

    alpha = math.radians(alpha_deg)
    n_hat = torch.tensor(
        [math.sin(alpha), 0.0, math.cos(alpha)], device=device, dtype=torch.float64
    )
    cn_history: list[float] = []
    particle_counts: list[int] = []
    wake_rows_list: list[int] = []
    lev_release_total = 0
    kelvin_max = 0.0
    started = time.perf_counter()
    log.info("%s start: U=%.1f alpha=%.1f steps=%d dt*=%.2f", run_id, U, alpha_deg, total_steps, DT_STAR)
    try:
        for step in range(1, total_steps + 1):
            frame = kin.evaluate(step * aero_dt)
            proposal = stepper.propose(owner, (frame,), aero_dt)
            total_force = proposal.load.total_force
            if not bool(torch.isfinite(total_force).all().item()):
                raise FloatingPointError(f"{run_id} step {step}: non-finite force")
            cn_history.append(float(total_force.dot(n_hat).item()) / q_area)
            stepper.commit(owner, proposal)
            observer.sample(solver, owner.state)
            diag = owner.state.diagnostics[-1]
            particle_counts.append(int(diag["particle_count"]))
            wake_rows_list.append(int(diag["wake_ring_count"]))
            lev_release_total += int(diag["lev_release_count"])
            kelvin_max = max(kelvin_max, float(diag["kelvin_max_abs"]))
            if step % 500 == 0:
                log.info(
                    "%s step %d/%d Cn(inst)=%.4f particles=%d wake_rows=%d (%.0fs)",
                    run_id,
                    step,
                    total_steps,
                    cn_history[-1],
                    particle_counts[-1],
                    wake_rows_list[-1],
                    time.perf_counter() - started,
                )
    except Exception as error:  # noqa: BLE001 -- failures stay in the table
        elapsed = time.perf_counter() - started
        log.exception("%s FAILED after %d steps (%.0fs): %s", run_id, len(cn_history), elapsed, error)
        return {
            "run_id": run_id,
            "U_m_s": U,
            "Re": reynolds(U),
            "alpha_deg": alpha_deg,
            "source_status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
            "last_complete_step": len(cn_history),
            "seconds": elapsed,
        }

    elapsed = time.perf_counter() - started
    i0 = int(round(CN_WINDOW_STAR[0] / DT_STAR))
    cn_mean = sum(cn_history[i0:]) / len(cn_history[i0:])
    cn_check = sum(cn_history[: i0]) / i0
    summary = summarize_probes(
        observer,
        chord_m=CHORD_M,
        freestream_m_s=U,
        delta_time_s=aero_dt,
        alpha_deg=alpha_deg,
        start_index=i0,
    )
    has_shedding = math.isfinite(summary.st_mean)
    if has_shedding:
        closure = abs(summary.st_modified - summary.st_mean * math.sin(alpha))
        if closure > 1e-12:
            raise AssertionError(f"{run_id}: St* closure violated ({closure:.2e})")
    # Retention-induced spectral-jump audit: culls happen at fixed ages, so
    # a steady late-time particle plateau means no unresolved retention shock.
    plateau = particle_counts[-500:]
    plateau_spread = (max(plateau) - min(plateau)) / max(1, max(plateau))
    result = {
        "run_id": run_id,
        "U_m_s": U,
        "Re": reynolds(U),
        "alpha_deg": alpha_deg,
        "Cn_mean": cn_mean,
        "Cn_mean_startup_check": cn_check,
        "St": summary.st_mean if has_shedding else None,
        "St_modified": summary.st_modified if has_shedding else None,
        "St_spread_across_probes": summary.st_spread if has_shedding else None,
        "st_per_probe": list(summary.st_per_probe),
        "dominant_psd_st": list(summary.dominant_psd_st),
        "dominant_psd": list(summary.dominant_psd),
        "peak_frequency_hz": summary.peak_frequency_hz,
        "stationary": plateau_spread < 0.2,
        "statistics_start_tstar": CN_WINDOW_STAR[0],
        "statistics_end_tstar": round(len(cn_history) * DT_STAR, 2),
        "n_samples": len(cn_history) - i0,
        "particle_count_final": particle_counts[-1],
        "particle_plateau_relative_spread": plateau_spread,
        "wake_rows_final": wake_rows_list[-1],
        "lev_release_total": lev_release_total,
        "kelvin_max_abs": kelvin_max,
        "steps": total_steps,
        "seconds": elapsed,
        "seconds_per_step": elapsed / total_steps,
        "cn_history_every10": [round(v, 6) for v in cn_history[::10]],
        "source_status": "success",
        "failure_reason": "",
    }
    log.info(
        "%s done: Cn=%.4f St=%s St*=%s stationary=%s (%.0fs)",
        run_id,
        cn_mean,
        f"{summary.st_mean:.4f}" if has_shedding else "no-peak",
        f"{summary.st_modified:.4f}" if has_shedding else "-",
        result["stationary"],
        elapsed,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out",
        default=str(
            ROOT
            / "artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current"
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N cases (0=all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("the rigid queue is GPU-only; CUDA is required")
    out_dir = Path(args.out)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.json"
    completed: set[str] = set()
    if checkpoint_path.is_file():
        completed = set(json.loads(checkpoint_path.read_text())["completed"])
        LOG.info("resuming: %d cases already complete", len(completed))
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        git_commit = "unknown"
    gpu_name = torch.cuda.get_device_name(torch.device(args.device))

    observables_path = out_dir / "model_observables.csv"
    write_header = not observables_path.is_file()
    done = 0
    for U, alpha, t_star in QUEUE:
        run_id = f"ROJR-RIGID-U{str(U).replace('.', 'p')}-A{int(round(alpha * 10)):03d}"
        if run_id in completed:
            continue
        if args.limit and done >= args.limit:
            LOG.info("limit %d reached; stopping", args.limit)
            break
        result = run_case(U, alpha, t_star, device=args.device, log=LOG)
        (cases_dir / f"{run_id}.json").write_text(json.dumps(result, indent=2))
        row = {
            "run_id": run_id,
            "case_id": run_id,
            # One dual-purpose rigid branch: every case scores Figure 9 Cn
            # AND carries the Figure 12/13/15 probe St columns.
            "branch": "rigid_aero",
            "U_m_s": U,
            "Re": reynolds(U),
            "alpha_deg": alpha,
            "zmax_over_c_mean_map": "",
            "Cn_mean": result.get("Cn_mean", ""),
            "St": result.get("St", ""),
            "St_modified": result.get("St_modified", ""),
            "stationary": result.get("stationary", ""),
            "statistics_start_tstar": CN_WINDOW_STAR[0],
            "statistics_end_tstar": result.get("statistics_end_tstar", ""),
            "n_samples": result.get("n_samples", result.get("last_complete_step", "")),
            "gpu_name": gpu_name,
            "git_commit": git_commit,
            "source_status": result.get("source_status", "failed"),
            "failure_reason": result.get("failure_reason", ""),
        }
        with observables_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OBSERVABLES_FIELDS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)
        if result.get("source_status") == "success":
            completed.add(run_id)
            checkpoint_path.write_text(
                json.dumps({"completed": sorted(completed)}, indent=2)
            )
        else:
            LOG.warning("%s recorded as failure (kept in table)", run_id)
        done += 1
    LOG.info("queue pass finished: %d run, %d complete total", done, len(completed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
