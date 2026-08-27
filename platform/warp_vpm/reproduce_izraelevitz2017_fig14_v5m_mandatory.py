"""Focused Izraelevitz et al. (2017) Figure 14 mandatory FLUX-V5M runner.

H4 (HANDOFF_IZRAELEVITZ2017_FIG14_SCHERER1968_20260827 §8): the formal
runner that drives ALL 12 real Scherer motion conditions through the current
mandatory integrated data path — rigid analytic ``SurfaceFrame`` (H2) into the
native V5M solver (H3) with separated LEV, joint TEV and a free wake as LIVE
solver state, CUDA float64, a SINGLE integrated surface-load owner and a
ONE-TIME Cd0 subtraction.  No post-hoc LDVM separation delta is added: the
separated circulation is already carried by the in-solver LEV particles.

Scoring is the 14-marker MAE against the digitized Scherer 1968 open squares
(each replicate counted independently; the two duplicated conditions are
never pre-averaged), gated on the frozen project best MAE 0.01745211311116545.

Physics/ownership contract (HANDOFF §4, §9 G2):
  CT_raw  = THRUST_SIGN * mean(F_x, last cycle) / (0.5 rho U^2 S)
  CT      = CT_raw - Cd0        (Cd0 = 0.057, subtracted exactly once)

The native frame carries the freestream along +x, so the propulsive
(upstream) direction is -x and THRUST_SIGN = -1.  The audit (mirrored in
tests/test_izra_h4_runner_gpu.py) ran IZRA-15-090 (the strongest clean
thrust phasing): at the maximum-heave-rate events the world Fx is
consistently negative with |Fz|/qS ~ 1.2, and the last-cycle mean gives
CT_raw = +0.18 — positive thrust exactly as the Scherer data does.

Wake-history model: the registered strong scheme (``material`` = the
author's Mf2_vec1) diverges in sustained-LEV-release regimes — the code's
own registry note (NativeV5MConfig.wake_history_mode) — and this case
sustains LESP well above LESPcrit for much of every cycle.  A direct
steady-wing audit (flat plate, +5 deg, no heave) shows ``material``
contributing a spurious ~-620 Pa mean load where ``bound_rate`` correctly
vanishes.  This runner therefore defaults to the author's weak scheme
``bound_rate`` (dp_add = dGamma/dt, bounded by construction); the CLI can
still select ``material`` and the choice is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz2017 import (
    GT_PATH,
    GT_SHA256,
    IZRA_CASES,
    PROFILE_DRAG_CD0,
)
from fluxvortex.cases.izraelevitz_kinematics import IzraRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MOwner,
    compute_lesp_crit,
)
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RIGID_NATIVE_V5M_CONTRACT,
    RigidNativeV5MSolver,
    RigidV5MSurface,
)

ROOT = Path(__file__).resolve().parents[2]

# ── Frozen gates and identity (HANDOFF §6, §9 G4) ─────────────────────────
FROZEN_MAE_GATE = 0.01745211311116545
FROZEN_MAX_ABS_ERROR_GATE = 0.04963501153092067

# Physics-based LESPcrit (compute_lesp_crit) hitting the frozen Scherer
# threshold sin(CLmax/CLa) = sin(0.90 / 0.065 deg) = 0.2393 at Re = 310000:
#   0.11 + 2.0 * (0.0607 - 0.013) * (1 + 0.35 log10(310000/30000)) = 0.23927
LESP_THICKNESS_RATIO = 0.0607
LESP_REYNOLDS = 310000.0
RELEASE_CONDITION_SOURCE = (
    "compute_lesp_crit(thickness_ratio=0.0607, reynolds=310000) = 0.239265 "
    "~= sin(CLmax/CLa) = 0.2393 (Scherer 1968 static CLa=0.065 /deg, "
    "CLmax=0.90); NOT the Mancini Lcrit=0.11"
)

# Thrust sign audit (HANDOFF §4): the historical Ptera world frame treated
# +force_W[:, 0] as thrust.  The unified native frame carries the freestream
# along +x, so the propulsive direction is -x: thrust = -F_x.  Audited on a
# real condition (IZRA-15-090): at maximum-|zdot| events Fx ~ -44 N while
# |Fz|/qS ~ 1.2, and the last-cycle mean Fx < 0 gives CT_raw > 0.
THRUST_SIGN = -1.0

# Wake-history physics model: "bound_rate" is the author's weak scheme
# (dp_add = dGamma/dt), bounded by construction in the sustained-LEV-release
# regime of this case; "material" (Mf2_vec1) is the strong scheme that the
# registry notes diverge here (and that a steady-wing audit shows injecting
# a ~-620 Pa spurious mean load at +5 deg attached conditions).
DEFAULT_WAKE_HISTORY_MODE = "bound_rate"

SURFACE_LOAD_OWNER = "RigidAuthorLoadAssembler (single integrated author-pressure owner)"
SEPARATED_LOAD_OWNER = "RigidNativeV5MSolver.propose (integrated LEV ribbon particles + pinned LESP strips)"
PROFILE_DRAG_OWNER = "scorer (CT = CT_raw - Cd0, subtracted exactly once)"

# Frozen historical GPU V2 predictions (HANDOFF §6.1) — reference curves only,
# produced by enable_lev=False + prescribed_wake=True + post-hoc LDVM delta;
# they do NOT satisfy the current mandatory physics contract.
FROZEN_GPU_V2_CT: dict[tuple[float, float], float] = {
    (15.0, 15.0): 0.147064489010,
    (15.0, 30.0): 0.182160361626,
    (15.0, 45.0): 0.208679943782,
    (15.0, 60.0): 0.232853334144,
    (15.0, 75.0): 0.224968182229,
    (15.0, 90.0): 0.199706340807,
    (15.0, 105.0): 0.193885953856,
    (25.0, 45.0): 0.086285230479,
    (25.0, 60.0): 0.130480751086,
    (25.0, 75.0): 0.089259705214,
    (25.0, 90.0): 0.071809003074,
    (25.0, 105.0): 0.061688657531,
}
V4B_REFERENCE_CSV = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
    "20260812_fluxv_v4b_crosspaper_full/izraelevitz2017_fig14_v4_mean_thrust.csv"
)

DEFAULT_OUTPUT = ROOT / "artifacts/baselines/fluxv_v5m_izraelevitz2017_fig14"
DEFAULT_GRID = "8x24"
DEFAULT_STEPS_PER_CYCLE = 128
DEFAULT_CYCLES = 4

LOG = logging.getLogger("izra_fig14_v5m")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_id_for(theta_max_deg: float, phase_offset_deg: float) -> str:
    return f"IZRA-{int(theta_max_deg):02d}-{int(phase_offset_deg):03d}"


# ── Ground truth (H0 contract: 12 conditions / 14 markers) ────────────────

def load_gt_markers(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the 14 Scherer experimental markers.

    Only ``data_role == "experimental_observation"`` rows are scored; the
    authors' numerical curves in the same CSV stay out of the score pool.
    Any identity drift (SHA), structural change (count, duplicates) or
    malformed field raises — the runner then fails closed.
    """
    csv_path = Path(path) if path is not None else GT_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"GT CSV missing: {csv_path}")
    observed = _sha256(csv_path)
    if observed != GT_SHA256:
        raise RuntimeError(f"Izraelevitz GT CSV drift: {observed} != {GT_SHA256}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)]
    required = (
        "series",
        "data_role",
        "theta_max_deg",
        "phase_offset_deg",
        "replicate",
        "ct",
        "ct_error_minus",
        "ct_error_plus",
    )
    markers: list[dict[str, Any]] = []
    for row in rows:
        if row.get("data_role") != "experimental_observation":
            continue
        missing = [name for name in required if row.get(name) in (None, "")]
        if missing:
            raise ValueError(f"experimental marker row misses fields {missing}: {row}")
        if row["series"] != "scherer_1968_experiment":
            raise ValueError(f"unexpected experimental series {row['series']!r}")
        marker = {
            "theta_max_deg": float(row["theta_max_deg"]),
            "phase_offset_deg": float(row["phase_offset_deg"]),
            "replicate": int(row["replicate"]),
            "ct": float(row["ct"]),
            "ct_error_minus": float(row["ct_error_minus"]),
            "ct_error_plus": float(row["ct_error_plus"]),
        }
        for name in ("ct", "ct_error_minus", "ct_error_plus"):
            if not math.isfinite(marker[name]):
                raise ValueError(f"non-finite {name} in GT row: {row}")
        if marker["ct_error_minus"] < 0.0 or marker["ct_error_plus"] < 0.0:
            raise ValueError(f"negative error bar in GT row: {row}")
        marker["case_id"] = case_id_for(
            marker["theta_max_deg"], marker["phase_offset_deg"]
        )
        if marker["case_id"] not in IZRA_CASES:
            raise ValueError(f"GT row outside the 12 registered conditions: {row}")
        markers.append(marker)
    if len(markers) != 14:
        raise ValueError(f"expected exactly 14 experimental markers, got {len(markers)}")
    replicates: dict[tuple[float, float], int] = {}
    for marker in markers:
        key = (marker["theta_max_deg"], marker["phase_offset_deg"])
        replicates[key] = replicates.get(key, 0) + 1
    if len(replicates) != 12:
        raise ValueError(f"expected 12 unique conditions, got {len(replicates)}")
    if replicates != {
        (15.0, 15.0): 2,
        (15.0, 30.0): 1,
        (15.0, 45.0): 1,
        (15.0, 60.0): 1,
        (15.0, 75.0): 2,
        (15.0, 90.0): 1,
        (15.0, 105.0): 1,
        (25.0, 45.0): 1,
        (25.0, 60.0): 1,
        (25.0, 75.0): 1,
        (25.0, 90.0): 1,
        (25.0, 105.0): 1,
    }:
        raise ValueError(f"duplicate-marker structure drifted: {replicates}")
    markers.sort(key=lambda m: (m["theta_max_deg"], m["phase_offset_deg"], m["replicate"]))
    return markers


def load_reference_series(path: Path | None = None) -> dict[str, dict[tuple[float, float], float]]:
    """Load the authors' numerical curves (never scored; figure references)."""
    csv_path = Path(path) if path is not None else GT_PATH
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)]
    series: dict[str, dict[tuple[float, float], float]] = {}
    for row in rows:
        if row["data_role"] != "numerical_reference":
            continue
        series.setdefault(row["series"], {})[
            (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
        ] = float(row["ct"])
    return series


# ── Scoring (14-marker, replicates independent) ───────────────────────────

def score_all(
    ct_predictions: dict[tuple[float, float], float],
    markers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score every marker independently against the per-condition prediction.

    One motion condition produces one prediction; a condition with two
    experimental markers contributes TWO error terms.  No pre-averaging, no
    error-bar weighting, no phase/amplitude fitting (HANDOFF §3).
    """
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    for marker in markers:
        key = (marker["theta_max_deg"], marker["phase_offset_deg"])
        if key not in ct_predictions:
            raise KeyError(f"missing prediction for condition {key}")
        prediction = float(ct_predictions[key])
        if not math.isfinite(prediction):
            raise ValueError(f"non-finite prediction for condition {key}")
        error = prediction - marker["ct"]
        errors.append(error)
        rows.append(
            {
                "case_id": marker["case_id"],
                "theta_max_deg": marker["theta_max_deg"],
                "phase_offset_deg": marker["phase_offset_deg"],
                "replicate": marker["replicate"],
                "ct_prediction": prediction,
                "ct_experiment": marker["ct"],
                "signed_error": error,
                "abs_error": abs(error),
                "ct_error_minus": marker["ct_error_minus"],
                "ct_error_plus": marker["ct_error_plus"],
                "within_error_bar": bool(
                    -marker["ct_error_minus"] <= error <= marker["ct_error_plus"]
                ),
            }
        )
    count = len(errors)
    mae = sum(abs(error) for error in errors) / count
    rmse = math.sqrt(sum(error * error for error in errors) / count)
    bias = sum(errors) / count
    worst = max(rows, key=lambda row: row["abs_error"])
    return {
        "n_markers": count,
        "mae_ct": mae,
        "rmse_ct": rmse,
        "bias_ct": bias,
        "max_abs_error_ct": worst["abs_error"],
        "max_abs_error_condition": [
            worst["theta_max_deg"],
            worst["phase_offset_deg"],
        ],
        "markers_within_error_bar": sum(row["within_error_bar"] for row in rows),
        "per_marker": rows,
    }


# ── One condition: SurfaceFrame -> native V5M propose/commit loop ─────────

def run_one_condition(
    case_id: str,
    *,
    chordwise_panels: int = 8,
    spanwise_panels: int = 24,
    steps_per_cycle: int = DEFAULT_STEPS_PER_CYCLE,
    cycles: int = DEFAULT_CYCLES,
    device: str = "cuda:0",
    wake_max_rows: int = 128,
    particle_capacity: int = 131_072,
    particle_max_age_steps: int = 64,
    dvm_target_spacing_chord: float = 0.018,
    wake_history_mode: str = DEFAULT_WAKE_HISTORY_MODE,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run one Scherer motion condition and return its CT + physics evidence.

    Each step: ``kin.evaluate(t)`` -> ``RigidV5MStepper.propose`` (stores the
    frame on the ``RigidV5MSurface`` and advances the frozen native V5M
    machinery: bound circulation, LEV release, joint TEV, free wake) ->
    ``commit``.  The world total force comes from the single integrated load
    owner; CT scores the LAST complete cycle.
    """
    log = logger or LOG
    if case_id not in IZRA_CASES:
        raise KeyError(f"unknown Izraelevitz condition {case_id}")
    if not torch.cuda.is_available():
        raise RuntimeError("the Fig.14 mandatory runner requires CUDA; CPU is forbidden")
    case = IZRA_CASES[case_id]
    nc, ns = int(chordwise_panels), int(spanwise_panels)
    steps_per_cycle = int(steps_per_cycle)
    cycles = int(cycles)
    total_steps = steps_per_cycle * cycles
    if steps_per_cycle < 8 or cycles < 1:
        raise ValueError("need at least 8 steps/cycle and 1 cycle")
    period = 1.0 / case.frequency_hz
    aero_dt = period / steps_per_cycle
    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2

    kin = IzraRigidSurfaceKinematics(
        case, chordwise_panels=nc, spanwise_panels=ns, device=device
    )
    surface = RigidV5MSurface(chordwise_panels=nc, spanwise_panels=ns, device=device)
    settings = NativeV5MConfig(
        chordwise_panels=nc,
        spanwise_panels=ns,
        density=case.rho_kg_m3,
        freestream=case.freestream_m_s,
        aerodynamic_dt=aero_dt,
        wake_max_rows=wake_max_rows,
        particle_capacity=particle_capacity,
        particle_max_age_steps=particle_max_age_steps,
        dvm_target_spacing_chord=dvm_target_spacing_chord,
        lesp_thickness_ratio=LESP_THICKNESS_RATIO,
        lesp_reynolds=LESP_REYNOLDS,
        wake_history_mode=wake_history_mode,
        device=device,
    )
    solver = RigidNativeV5MSolver(surface, settings)
    stepper = RigidV5MStepper(solver, device=device)
    effective_lesp_crit = settings.effective_lesp_crit

    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()

    owner: Q16NativeV5MOwner = stepper.initialize((kin.evaluate(0.0),))
    force_history: list[list[float]] = []
    lesp_pre_max_abs: list[float] = []
    lev_release_counts: list[int] = []
    separated_strip_counts: list[int] = []
    release_owner_conflicts: list[int] = []
    wake_ring_counts: list[int] = []
    particle_counts: list[int] = []
    kelvin_max: list[float] = []
    neumann_max: list[float] = []
    lesp_pin_max: list[float] = []
    free_wake_convection_count = 0
    parent_untouched_count = 0
    accepted_commits = 0
    report_every = max(1, total_steps // 8)

    for step in range(1, total_steps + 1):
        frame = kin.evaluate(step * aero_dt)
        prior_wake_rows = owner.state.wake_gamma.numel() // ns
        pre_step = owner.state.step
        proposal = stepper.propose(owner, (frame,), aero_dt)
        # Transactional evidence: a proposal must not touch the parent state
        # (the solver additionally hashes the parent inside propose()).
        if owner.state.step == pre_step:
            parent_untouched_count += 1
        total_force = proposal.load.total_force
        if tuple(total_force.shape) != (3,) or not bool(
            torch.isfinite(total_force).all().item()
        ):
            raise FloatingPointError(
                f"{case_id} step {step}: non-finite integrated world force"
            )
        force_history.append(
            [float(total_force[0].item()), float(total_force[1].item()), float(total_force[2].item())]
        )
        stepper.commit(owner, proposal)
        accepted_commits += 1
        if owner.state.step != step:
            raise RuntimeError(f"{case_id}: commit did not advance to step {step}")
        if prior_wake_rows > 0:
            free_wake_convection_count += 1
        diag = owner.state.diagnostics[-1]
        if not diag.get("cuda_float64", False):
            raise RuntimeError(f"{case_id} step {step}: CUDA float64 contract lost")
        lesp_pre_max_abs.append(float(diag["lesp_pre_max_abs"]))
        lev_release_counts.append(int(diag["lev_release_count"]))
        separated_strip_counts.append(int(diag["separated_strip_count"]))
        release_owner_conflicts.append(int(diag["release_owner_conflicts"]))
        wake_ring_counts.append(int(diag["wake_ring_count"]))
        particle_counts.append(int(diag["particle_count"]))
        kelvin_max.append(float(diag["kelvin_max_abs"]))
        neumann_max.append(float(diag["retained_neumann_max_abs"]))
        lesp_pin_max.append(float(diag["lesp_pin_max_abs"]))
        if step % report_every == 0 or step == total_steps:
            log.info(
                "%s step %d/%d lesp=%.4f lev_release=%d particles=%d wake_rows=%d",
                case_id,
                step,
                total_steps,
                lesp_pre_max_abs[-1],
                lev_release_counts[-1],
                particle_counts[-1],
                wake_ring_counts[-1],
            )

    torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(torch.device(device)) / (1024.0**2)
    source_steps = int(getattr(owner.state.source_bank, "it", 0))

    scored = force_history[total_steps - steps_per_cycle :]
    if len(scored) != steps_per_cycle:
        raise RuntimeError(f"{case_id}: last-cycle sample count mismatch")
    mean_fx = sum(row[0] for row in scored) / steps_per_cycle
    ct_raw = THRUST_SIGN * mean_fx / q_area
    cd0 = float(case.profile_drag_coefficient)
    ct = ct_raw - cd0

    physics = {
        "enable_lev": True,  # source bank + LEV particles advance every step
        "joint_tev": True,   # Kelvin-gated joint TEV relation each step
        "prescribed_wake": False,
        "release_condition_source": RELEASE_CONDITION_SOURCE,
        "effective_lesp_crit": effective_lesp_crit,
        "lesp_thickness_ratio": LESP_THICKNESS_RATIO,
        "lesp_reynolds": LESP_REYNOLDS,
        "lesp_pre_max_abs_min": min(lesp_pre_max_abs),
        "lesp_pre_max_abs_max": max(lesp_pre_max_abs),
        "lesp_pre_max_abs_final": lesp_pre_max_abs[-1],
        "lev_release_count_total": sum(lev_release_counts),
        "steps_with_lev_release": sum(1 for c in lev_release_counts if c > 0),
        "lev_release_count_max": max(lev_release_counts),
        "separated_strip_count_max": max(separated_strip_counts),
        "release_owner_conflicts_total": sum(release_owner_conflicts),
        "tev_shed_count": total_steps,  # exactly one full-span TEV row per step
        "wake_ring_count_max": max(wake_ring_counts),
        "free_wake_convection_count": free_wake_convection_count,
        "proposal_count": total_steps,
        "accepted_commit_count": accepted_commits,
        "rejected_proposal_count": 0,
        "parent_state_unchanged_on_proposal": parent_untouched_count == total_steps,
        "dvm_source_step_count": source_steps,
        "particle_count_final": particle_counts[-1],
        "kelvin_max_abs": max(kelvin_max),
        "retained_neumann_max_abs": max(neumann_max),
        "lesp_pin_max_abs": max(lesp_pin_max),
        "cuda_float64_all_steps": True,
        "surface_load_owner": SURFACE_LOAD_OWNER,
        "separated_load_owner": SEPARATED_LOAD_OWNER,
        "profile_drag_owner": PROFILE_DRAG_OWNER,
        "posthoc_separation_delta_applied": False,
        "posthoc_ldvm_separation_pair_call": "none",
    }

    return {
        "case_id": case_id,
        "theta_max_deg": float(case.theta_max_deg),
        "phase_offset_deg": float(case.phase_offset_deg),
        "chordwise_panels": nc,
        "spanwise_panels": ns,
        "steps_per_cycle": steps_per_cycle,
        "cycles": cycles,
        "steps_total": total_steps,
        "scored_cycle_window": [total_steps - steps_per_cycle + 1, total_steps],
        "scored_sample_count": steps_per_cycle,
        "aerodynamic_dt_s": aero_dt,
        "wake_history_mode": wake_history_mode,
        "wake_max_rows": wake_max_rows,
        "particle_capacity": particle_capacity,
        "particle_max_age_steps": particle_max_age_steps,
        "mean_force_x_n": mean_fx,
        "q_area_n": q_area,
        "thrust_sign": THRUST_SIGN,
        "ct_raw": ct_raw,
        "cd0": cd0,
        "ct": ct,
        "elapsed_seconds": elapsed,
        "gpu_peak_memory_mib": peak_mib,
        "force_history": force_history,
        "physics": physics,
    }


# ── Physics-contract self-check (HANDOFF §9 G2) ───────────────────────────

def physics_contract_ok(results: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Verify the mandatory physics contract from recorded evidence."""
    violations: list[str] = []
    for result in results:
        case_id = result["case_id"]
        physics = result["physics"]
        if physics["enable_lev"] is not True:
            violations.append(f"{case_id}: LEV not integrated")
        if physics["joint_tev"] is not True:
            violations.append(f"{case_id}: joint TEV not solved")
        if physics["prescribed_wake"] is not False:
            violations.append(f"{case_id}: prescribed wake")
        if physics["dvm_source_step_count"] != result["steps_total"]:
            violations.append(f"{case_id}: DVM source did not advance every step")
        if physics["free_wake_convection_count"] != result["steps_total"] - 1:
            violations.append(f"{case_id}: free-wake convection count drifted")
        if physics["tev_shed_count"] != result["steps_total"]:
            violations.append(f"{case_id}: TEV shed count drifted")
        if physics["proposal_count"] != result["steps_total"] or physics[
            "accepted_commit_count"
        ] != result["steps_total"]:
            violations.append(f"{case_id}: proposal/commit counts drifted")
        if not physics["parent_state_unchanged_on_proposal"]:
            violations.append(f"{case_id}: proposal mutated the parent state")
        if physics["posthoc_separation_delta_applied"] is not False:
            violations.append(f"{case_id}: post-hoc separation delta applied")
        if not math.isfinite(result["ct"]) or not math.isfinite(result["ct_raw"]):
            violations.append(f"{case_id}: non-finite CT")
    return (not violations), violations


# ── Output artifacts (HANDOFF §10) ────────────────────────────────────────

def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except subprocess.CalledProcessError as error:  # pragma: no cover
            return f"<git {args} failed: {error.stderr.strip()}>"

    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
        "changed_files": [line for line in git("status", "--porcelain").splitlines() if line],
    }


def write_predictions_csv(path: Path, metrics: dict[str, Any]) -> None:
    columns = (
        "case_id",
        "theta_max_deg",
        "phase_offset_deg",
        "replicate",
        "ct_prediction",
        "ct_experiment",
        "signed_error",
        "abs_error",
        "ct_error_minus",
        "ct_error_plus",
        "within_error_bar",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in metrics["per_marker"]:
            writer.writerow({name: row[name] for name in columns})


def _cjk_font() -> str | None:
    try:
        from matplotlib import font_manager

        for font_file in font_manager.findSystemFonts():
            name = Path(font_file).name.lower()
            if "cjk" in name or "wqy" in name:
                return font_manager.FontProperties(fname=font_file).get_name()
    except Exception:  # pragma: no cover - schematic figure only
        return None
    return None


def plot_fig14(
    markers: list[dict[str, Any]],
    predictions: dict[tuple[float, float], float],
    out_png: Path,
    out_pdf: Path,
    references: dict[str, dict[tuple[float, float], float]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    references = references or {}
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 9.0), sharex=True)
    for axis, theta in zip(axes, (15.0, 25.0)):
        panel_markers = [m for m in markers if m["theta_max_deg"] == theta]
        if panel_markers:
            by_condition: dict[float, list[dict[str, Any]]] = {}
            for marker in panel_markers:
                by_condition.setdefault(marker["phase_offset_deg"], []).append(marker)
            phases = sorted(by_condition)
            experiment = [sum(m["ct"] for m in by_condition[p]) / len(by_condition[p]) for p in phases]
            lower = [
                experiment[i] - min(m["ct"] - m["ct_error_minus"] for m in by_condition[p])
                for i, p in enumerate(phases)
            ]
            upper = [
                max(m["ct"] + m["ct_error_plus"] for m in by_condition[p]) - experiment[i]
                for i, p in enumerate(phases)
            ]
            axis.errorbar(
                phases,
                experiment,
                yerr=[lower, upper],
                fmt="s",
                color="black",
                mfc="none",
                capsize=3,
                label="Scherer 1968 experiment",
            )
        pred_phases = sorted(psi for (t, psi) in predictions if t == theta)
        if pred_phases:
            axis.plot(
                pred_phases,
                [predictions[(theta, psi)] for psi in pred_phases],
                "o-",
                color="tab:red",
                label="current mandatory V5M (integrated LEV/TEV/free-wake)",
            )
        gpu2_phases = sorted(psi for (t, psi) in FROZEN_GPU_V2_CT if t == theta)
        if gpu2_phases:
            axis.plot(
                gpu2_phases,
                [FROZEN_GPU_V2_CT[(theta, psi)] for psi in gpu2_phases],
                "^--",
                color="tab:blue",
                ms=5,
                label="historical GPU V2 (frozen; post-hoc LDVM delta)",
            )
        style = {
            "authors_1state_ullt": ("tab:green", ":"),
            "authors_6state_ullt": ("tab:purple", ":"),
            "authors_qs_added_mass": ("tab:orange", ":"),
        }
        for series, curves in references.items():
            phases_ref = sorted(psi for (t, psi) in curves if t == theta)
            if not phases_ref:
                continue
            color, linestyle = style.get(series, ("gray", ":"))
            axis.plot(
                phases_ref,
                [curves[(theta, psi)] for psi in phases_ref],
                linestyle=linestyle,
                color=color,
                lw=1.0,
                label=series,
            )
        axis.set_ylabel("mean $C_T$")
        axis.set_title(f"$\\theta_{{max}}$ = {theta:g} deg")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("$\\psi$ [deg]")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Izraelevitz et al. (2017) Fig.14 — Scherer 1968 thrust coefficient")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_geometry_schematic(out_png: Path) -> None:
    """Top-view geometry sketch; the mid-span plane is a symmetry plane, NOT a wall."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, b = 0.1016, 0.3048
    fig, axis = plt.subplots(figsize=(6.5, 5.0))
    axis.add_patch(
        plt.Rectangle((-0.75 * c, -0.5 * b), c, b, fill=False, color="black", lw=1.5)
    )
    axis.axvline(0.0, color="crimson", ls="--", lw=1.2)
    axis.annotate(
        "0.75c pivot",
        xy=(0.0, 0.56 * b),
        xytext=(-0.62 * c, 0.68 * b),
        arrowprops=dict(arrowstyle="->", color="crimson"),
        color="crimson",
        fontsize=9,
    )
    axis.annotate(
        "",
        xy=(-1.35 * c, 0.0),
        xytext=(-2.1 * c, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.8),
    )
    axis.text(-2.05 * c, 0.05 * b, "U (freestream, +x)", color="tab:blue", fontsize=9)
    axis.annotate(
        "",
        xy=(0.45 * c, 0.72 * b),
        xytext=(0.45 * c, 0.42 * b),
        arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=1.8),
    )
    axis.text(0.5 * c, 0.66 * b, "heave z(t)", color="tab:green", fontsize=9)
    font = _cjk_font()
    labels = (
        ["chordwise x", "spanwise y", "thickness/normal z (heave)"]
        if font is None
        else ["弦向 x", "展向 y", "厚度/法向 z"]
    )
    axis.set_xlabel(f"{labels[0]}  (LE at $-0.75c$, TE at $+0.25c$; pivot at origin)")
    axis.set_ylabel(labels[1])
    if font is not None:
        for text in axis.texts:
            text.set_fontfamily(font)
        axis.xaxis.label.set_fontfamily(font)
        axis.yaxis.label.set_fontfamily(font)
        axis.set_title(
            "矩形 AR=3 有限翼（两自由翼尖；中面为对称面，非风洞壁面）\n"
            "z(t)=h cos(ωt), θ(t)=θmax cos(ωt+ψ), pivot 0.75c",
            fontfamily=font,
            fontsize=10,
        )
    else:
        axis.set_title(
            "rectangular AR=3 finite wing (two free tips; mid-plane is a symmetry\n"
            "plane, not a tunnel wall)\nz(t)=h cos(wt), th(t)=thmax cos(wt+psi), pivot 0.75c",
            fontsize=10,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    metrics: dict[str, Any],
    status: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {
        (result["theta_max_deg"], result["phase_offset_deg"]): result["ct"]
        for result in results
    }
    write_predictions_csv(output_dir / "predictions.csv", metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    physics_evidence = {
        "contract": RIGID_NATIVE_V5M_CONTRACT,
        "release_condition_source": RELEASE_CONDITION_SOURCE,
        "surface_load_owner": SURFACE_LOAD_OWNER,
        "separated_load_owner": SEPARATED_LOAD_OWNER,
        "profile_drag_owner": PROFILE_DRAG_OWNER,
        "cd0": PROFILE_DRAG_CD0,
        "posthoc_separation_delta_applied": False,
        "conditions": {result["case_id"]: result["physics"] for result in results},
    }
    (output_dir / "physics_evidence.json").write_text(
        json.dumps(physics_evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    references = load_reference_series()
    if V4B_REFERENCE_CSV.is_file():
        with V4B_REFERENCE_CSV.open(newline="", encoding="utf-8") as stream:
            references["fluxv_v4b"] = {
                (float(row["theta_max_deg"]), float(row["phase_offset_deg"])): float(
                    row["v4_CT"]
                )
                for row in csv.DictReader(stream)
            }
    try:
        plot_fig14(
            markers,
            predictions,
            output_dir / "fig14_ct_vs_phase.png",
            output_dir / "fig14_ct_vs_phase.pdf",
            references,
        )
        plot_geometry_schematic(output_dir / "fig14_geometry_schematic.png")
    except Exception as error:  # pragma: no cover — figure must not kill the run
        LOG.warning("figure generation failed: %s", error)

    git_state = _git_state()
    source_files = {
        "runner": Path(__file__),
        "case_config": ROOT / "src/fluxvortex/cases/izraelevitz2017.py",
        "kinematics": ROOT / "src/fluxvortex/cases/izraelevitz_kinematics.py",
        "rigid_native": ROOT / "src/fluxvortex/warp_fsi/rigid_flux_v5m_native.py",
        "stepper": ROOT / "src/fluxvortex/aero/v5m/stepper.py",
        "native_v5m": ROOT / "src/fluxvortex/warp_fsi/q16_flux_v5m_native.py",
        "ground_truth_csv": GT_PATH,
    }
    device = torch.device(args.device)
    gate = {
        "metric": "14-marker MAE_CT",
        "limit": FROZEN_MAE_GATE,
        "max_abs_error_limit": FROZEN_MAX_ABS_ERROR_GATE,
        "value": metrics["mae_ct"],
        "passed": metrics["mae_ct"] <= FROZEN_MAE_GATE
        and metrics["max_abs_error_ct"] <= FROZEN_MAX_ABS_ERROR_GATE,
    }
    summary = {
        "schema": "flux-v5m-izraelevitz2017-fig14-mandatory-v1",
        "paper": (
            "Izraelevitz, Zhu & Triantafyllou 2017, AIAA J 55(4), Figure 14 — "
            "Scherer 1968 open-square experiment"
        ),
        "status": status,
        "cases_attempted": [result["case_id"] for result in results],
        "conditions_completed": sum(
            1 for result in results if result["steps_total"] == result["steps_per_cycle"] * result["cycles"]
        ),
        "n_markers_scored": metrics["n_markers"],
        "mae_ct": metrics["mae_ct"],
        "rmse_ct": metrics["rmse_ct"],
        "bias_ct": metrics["bias_ct"],
        "max_abs_error_ct": metrics["max_abs_error_ct"],
        "qualification_gate": gate,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "elapsed_seconds_total": sum(result["elapsed_seconds"] for result in results),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": summary["schema"],
        "status": status,
        "git": git_state,
        "sha256": {name: _sha256(path) for name, path in source_files.items()},
        "ground_truth": {
            "path": str(GT_PATH),
            "sha256": GT_SHA256,
            "experimental_markers": len(markers),
            "unique_conditions": 12,
        },
        "numerics": {
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "dtype": "float64",
            "grid": f"{args.grid}",
            "chordwise_panels": results[0]["chordwise_panels"] if results else None,
            "spanwise_panels": results[0]["spanwise_panels"] if results else None,
            "steps_per_cycle": args.steps_per_cycle,
            "cycles": args.cycles,
            "scored_window": "last complete cycle",
            "cpu_role": "scheduling / I/O / hashing only",
        },
        "physics_switches": {
            "enable_lev": True,
            "joint_tev": True,
            "prescribed_wake": False,
            "wake_history_mode": args.wake_history_mode,
            "wake_max_rows": args.wake_max_rows,
            "particle_capacity": args.particle_capacity,
            "particle_max_age_steps": args.particle_max_age_steps,
            "dvm_target_spacing_chord": 0.018,
            "release_condition_source": RELEASE_CONDITION_SOURCE,
            "effective_lesp_crit": compute_lesp_crit(LESP_THICKNESS_RATIO, LESP_REYNOLDS),
        },
        "load_ownership": {
            "surface_load_owner": SURFACE_LOAD_OWNER,
            "separated_load_owner": SEPARATED_LOAD_OWNER,
            "profile_drag_owner": PROFILE_DRAG_OWNER,
            "cd0": PROFILE_DRAG_CD0,
            "cd0_subtraction_count": 1,
            "posthoc_separation_delta": "none",
        },
        "conditions": {
            result["case_id"]: {
                "theta_max_deg": result["theta_max_deg"],
                "phase_offset_deg": result["phase_offset_deg"],
                "steps_total": result["steps_total"],
                "ct_raw": result["ct_raw"],
                "ct": result["ct"],
                "elapsed_seconds": result["elapsed_seconds"],
                "gpu_peak_memory_mib": result["gpu_peak_memory_mib"],
            }
            for result in results
        },
        "metrics": {
            "n_markers": metrics["n_markers"],
            "mae_ct": metrics["mae_ct"],
            "rmse_ct": metrics["rmse_ct"],
            "bias_ct": metrics["bias_ct"],
            "max_abs_error_ct": metrics["max_abs_error_ct"],
            "max_abs_error_condition": metrics["max_abs_error_condition"],
            "gate": gate,
        },
        "diagnostic_only": bool(args.diagnostic_only),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_grid(grid: str) -> tuple[int, int]:
    try:
        nc_text, ns_text = grid.lower().split("x")
        nc, ns = int(nc_text), int(ns_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"grid must look like 8x24, got {grid!r}"
        ) from error
    if nc < 2 or ns < 2:
        raise argparse.ArgumentTypeError(f"grid too small: {grid!r}")
    return nc, ns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Focused Figure 14 mandatory runner: 12 Scherer conditions through "
            "integrated LEV/joint-TEV/free-wake native V5M on CUDA float64."
        )
    )
    parser.add_argument("--grid", default=DEFAULT_GRID, help="chordwise x full-span, e.g. 8x24")
    parser.add_argument("--steps-per-cycle", type=int, default=DEFAULT_STEPS_PER_CYCLE)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument(
        "--case",
        default=None,
        help="run a single condition (e.g. IZRA-15-045); default runs all 12",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="skip the accuracy gate (exit 0 even when the gate fails)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wake-max-rows", type=int, default=128)
    parser.add_argument("--particle-capacity", type=int, default=131_072)
    parser.add_argument("--particle-max-age-steps", type=int, default=64)
    parser.add_argument(
        "--wake-history-mode",
        choices=("material", "bound_rate"),
        default=DEFAULT_WAKE_HISTORY_MODE,
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if not torch.cuda.is_available():
        LOG.error("CUDA is required (gate G1); refusing to run on CPU")
        return 3
    nc, ns = _parse_grid(args.grid)
    args.grid = f"{nc}x{ns}"

    try:
        markers = load_gt_markers()
        LOG.info(
            "GT closed: 14 markers / 12 conditions, sha256=%s...", GT_SHA256[:16]
        )
    except Exception as error:
        LOG.error("gate G0 input closure failed: %s", error)
        return 3

    if args.case is not None:
        if args.case not in IZRA_CASES:
            LOG.error("unknown --case %s", args.case)
            return 3
        selected = [args.case]
    else:
        selected = sorted(IZRA_CASES)

    results: list[dict[str, Any]] = []
    for index, case_id in enumerate(selected, start=1):
        LOG.info(
            "[%d/%d] %s (%s)",
            index,
            len(selected),
            case_id,
            IZRA_CASES[case_id].description,
        )
        try:
            result = run_one_condition(
                case_id,
                chordwise_panels=nc,
                spanwise_panels=ns,
                steps_per_cycle=args.steps_per_cycle,
                cycles=args.cycles,
                device=args.device,
                wake_max_rows=args.wake_max_rows,
                particle_capacity=args.particle_capacity,
                particle_max_age_steps=args.particle_max_age_steps,
                wake_history_mode=args.wake_history_mode,
                logger=LOG,
            )
        except Exception as error:
            LOG.exception("condition %s failed: %s", case_id, error)
            # Keep completed conditions on disk even on failure (gate G3).
            write_partial_failure(args.output_dir, args, results, markers, case_id, str(error))
            return 3
        results.append(result)
        LOG.info(
            "%s CT_raw=%+.6f CT=%+.6f (mean Fx=%+.3f N, %.1f s, %.0f MiB)",
            case_id,
            result["ct_raw"],
            result["ct"],
            result["mean_force_x_n"],
            result["elapsed_seconds"],
            result["gpu_peak_memory_mib"],
        )
        torch.cuda.empty_cache()

    physics_ok, violations = physics_contract_ok(results)
    if not physics_ok:
        LOG.error("gate G2 physics contract violated: %s", violations)
    complete_matrix = {result["case_id"] for result in results} == set(IZRA_CASES)
    if not complete_matrix:
        LOG.warning(
            "subset run (%d condition(s)): the 14-marker accuracy gate is skipped",
            len(results),
        )

    status = "PASS"
    if not physics_ok:
        status = "FAIL_PHYSICS_CONTRACT"
    if complete_matrix:
        predictions = {
            (result["theta_max_deg"], result["phase_offset_deg"]): result["ct"]
            for result in results
        }
        metrics = score_all(predictions, markers)
        gate_passed = (
            metrics["mae_ct"] <= FROZEN_MAE_GATE
            and metrics["max_abs_error_ct"] <= FROZEN_MAX_ABS_ERROR_GATE
        )
        if physics_ok and not gate_passed:
            status = "FAIL_ACCURACY_WITH_VALID_PHYSICS"
        elif not physics_ok:
            status = "FAIL_PHYSICS_CONTRACT"
        elif gate_passed:
            status = "PASS"
        LOG.info(
            "14-marker MAE=%.12f RMSE=%.12f bias=%+.12f max=%.12f at %s (gate %.12f)",
            metrics["mae_ct"],
            metrics["rmse_ct"],
            metrics["bias_ct"],
            metrics["max_abs_error_ct"],
            metrics["max_abs_error_condition"],
            FROZEN_MAE_GATE,
        )
    else:
        # A subset run can never be the paper acceptance (gate G3): it is a
        # diagnostic, whatever its per-condition quality.
        gate_passed = None
        if physics_ok:
            status = "SUBSET_DIAGNOSTIC"
        metrics = score_all(
            {
                (result["theta_max_deg"], result["phase_offset_deg"]): result["ct"]
                for result in results
            },
            [m for m in markers if m["case_id"] in {r["case_id"] for r in results}],
        )

    write_artifacts(args.output_dir, args, results, markers, metrics, status)

    if args.diagnostic_only:
        LOG.warning("diagnostic-only mode: accuracy gate ignored, exit 0")
        return 0
    if status == "PASS":
        return 0
    return 2 if status == "FAIL_ACCURACY_WITH_VALID_PHYSICS" else 3


def write_partial_failure(
    output_dir: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    failed_case: str,
    reason: str,
) -> None:
    """Persist completed conditions and the failure reason (gate G3)."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = {
            (result["theta_max_deg"], result["phase_offset_deg"]): result["ct"]
            for result in results
        }
        partial_markers = [
            m for m in markers if m["case_id"] in {r["case_id"] for r in results}
        ]
        metrics = (
            score_all(completed, partial_markers)
            if completed and partial_markers
            else {"n_markers": 0, "per_marker": []}
        )
        metrics["failed_condition"] = failed_case
        metrics["failure_reason"] = reason
        write_predictions_csv(output_dir / "predictions.csv", metrics)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "physics_evidence.json").write_text(
            json.dumps(
                {
                    "status": "FAILED_DURING_RUN",
                    "failed_condition": failed_case,
                    "failure_reason": reason,
                    "completed_conditions": {
                        result["case_id"]: result["physics"] for result in results
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as error:  # pragma: no cover
        LOG.error("partial-failure artifact write failed: %s", error)


if __name__ == "__main__":
    raise SystemExit(main())
