"""A4 re-test after the TE sheet-connectivity correction.

Runs the frozen Roj rigid case (15x30 V5M, AR=2 flat plate, U=5 m/s) at
A0/A5/A15/A19 for a short window and records, per angle:

* the TE geometry invariants that the pre-fix code violated
  (row-to-row join gap, shared-edge directed cosine, newborn polygon area);
* the load and release diagnostics (last-step Cn, last-50-step mean Cn);
* the joint-path gate diagnostics (deposited-flow Neumann, newborn LEV
  circulation, Kelvin identity).

Every row is a STARTUP-SEGMENT sample: t* = steps*0.01, so 150 steps is
t* = 1.5.  These numbers must not be read as steady-state accuracy.

Usage:
  PYTHONPATH=src:platform:platform/warp_vpm python3 \
    artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/p1_te_retest.py \
    --out <result.json> [--steps 150] [--angles 0,5,15,19] [--legacy]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidNativeV5MSolver,
    RigidV5MSurface,
)

DEVICE = "cuda:0"
CHORD_M = 0.0688
SPAN_M = 0.1375
FREESTREAM = 5.0
DT_STAR = 0.01
NC, NS = 15, 30
DT = DT_STAR * CHORD_M / FREESTREAM
Q_AREA = 0.5 * 1.208 * FREESTREAM**2 * CHORD_M * SPAN_M
ANALYTIC_NEWBORN_AREA = FREESTREAM * DT * SPAN_M / NS

# Digitized Figure 9 rigid U=5 m/s anchors (exact CSV lookup values).
EXPERIMENT_CN = {0.0: 0.02, 5.0: 0.3413, 15.0: 0.70, 19.0: 0.80}


def fixed_pitch_law(alpha_deg: float):
    theta = math.radians(alpha_deg)
    half = 0.5 * theta
    quat = torch.tensor(
        [math.cos(half), 0.0, math.sin(half), 0.0], device=DEVICE, dtype=torch.float64
    )
    zero = torch.zeros(3, device=DEVICE, dtype=torch.float64)

    def law(_t: float):
        return zero, quat, zero, zero.clone()

    return law


def geometry_invariants(wake_rings: torch.Tensor, te_ring: torch.Tensor) -> dict:
    chain = wake_rings.reshape(-1, NS, 4, 3)
    if chain.shape[0] < 2:
        return {}
    back = chain[:-1, :, [3, 2]]
    front = chain[1:, :, :2]
    new_edge = chain[0, :, 3] - chain[0, :, 2]
    old_edge = chain[1, :, 1] - chain[1, :, 0]
    cosine = (new_edge * old_edge).sum(-1) / (
        new_edge.norm(dim=-1) * old_edge.norm(dim=-1)
    )
    area = 0.5 * torch.cross(chain[0], chain[0].roll(-1, dims=1), dim=-1).sum(1)
    front_to_te = (chain[1, :, :2] - te_ring).norm(dim=-1).max()
    return {
        "join_gap_max_m": float((back - front).norm(dim=-1).max().item()),
        "cosine_min": float(cosine.min().item()),
        "cosine_max": float(cosine.max().item()),
        "newborn_area_max_m2": float(area.norm(dim=-1).max().item()),
        "front_to_te_max_m": float(front_to_te.item()),
    }


def run(
    alpha: float,
    steps: int,
    joint: bool,
    particle_age: int = 100,
    particle_capacity: int = 32768,
    flux: bool = False,
    window: int = 0,
    te_leg: bool = False,
    le_open: bool = False,
    le_open_drop_tev: bool = False,
    sigma_chord: float | None = None,
    spacing_chord: float | None = None,
    ramp_steps: int = 0,
    stretching: bool = False,
    core_spreading: bool = False,
    stretching_rvpm: bool = False,
) -> dict:
    reference = build_izra_reference_grid(
        chordwise_panels=NC,
        spanwise_panels=NS,
        chord_m=CHORD_M,
        span_m=SPAN_M,
        pivot_fraction_chord=0.25,
        device=DEVICE,
    )
    kin = PrescribedRigidSurfaceKinematics(
        reference, fixed_pitch_law(alpha), surface_id="w", body_id="b"
    )
    surface = RigidV5MSurface(chordwise_panels=NC, spanwise_panels=NS, device=DEVICE)
    settings = NativeV5MConfig(
        chordwise_panels=NC,
        spanwise_panels=NS,
        density=1.208,
        freestream=FREESTREAM,
        aerodynamic_dt=DT,
        lesp_crit=0.11,
        wake_max_rows=300,
        particle_capacity=particle_capacity,
        particle_max_age_steps=particle_age,
        wake_history_mode="bound_rate",
        wake_free_rows=100,
        dvm_target_spacing_chord=0.018 if spacing_chord is None else spacing_chord,
        **({} if sigma_chord is None else {"dvm_smoothing_radius_chord": sigma_chord}),
        device=DEVICE,
        joint_separation_solve=joint,
        flux_consistent_shedding=flux,
        le_constraint_window_steps=window,
        te_leg_candidate_diagnostic=te_leg,
        le_open_deposit=le_open,
        le_open_deposit_drop_tev_compensation=le_open_drop_tev,
        particle_stretching=stretching,
        particle_core_spreading=core_spreading,
        particle_stretching_rvpm=stretching_rvpm,
    )
    solver = RigidNativeV5MSolver(surface, settings)
    stepper = RigidV5MStepper(solver, device=DEVICE)
    owner = stepper.initialize((kin.evaluate(0.0),))
    normal = torch.tensor(
        [math.sin(math.radians(alpha)), 0.0, math.cos(math.radians(alpha))],
        device=DEVICE,
        dtype=torch.float64,
    )

    cn_series: list[float] = []
    newborn_series: list[float] = []
    particle_series: list[int] = []
    te_leg_delta_cn_series: list[float] = []
    cull_count_series: list[int] = []
    cull_circulation_series: list[float] = []
    lev_total = 0
    particles = 0
    deposited = 0.0
    newborn = 0.0
    kelvin = 0.0
    geom: dict = {}
    worst_gap = 0.0
    worst_cosine = -2.0
    ramp_factor_series: list[float] = []
    for step in range(1, steps + 1):
        # FROZEN Roj startup protocol: half-cosine 0 -> U0 over t* in [0,1]
        # (rojratsirikul2011_q16.py:174-182, tagged "numerical_protocol").  The
        # production runner applies it every step via apply_freestream, which sets
        # BOTH solver.v_inf and settings.freestream; the deposit reads the latter.
        # ramp_steps=0 keeps the historical impulsive start.
        if ramp_steps > 0:
            time_star = step * DT_STAR
            startup = ramp_steps * DT_STAR
            factor = (
                1.0
                if time_star >= startup
                else 0.5 * (1.0 - math.cos(math.pi * time_star / startup))
            )
        else:
            factor = 1.0
        ramp_factor_series.append(factor)
        speed = FREESTREAM * factor
        solver.v_inf = torch.tensor(
            [speed, 0.0, 0.0], device=DEVICE, dtype=torch.float64
        )
        object.__setattr__(solver.settings, "freestream", speed)
        proposal = stepper.propose(owner, (kin.evaluate(step * DT),), DT)
        # ramp_steps == 0 must reproduce the historical normalisation bit for bit,
        # so use the module constant outright instead of re-deriving it.
        if ramp_steps > 0:
            q_live = max(
                0.5 * 1.208 * speed * speed * CHORD_M * SPAN_M, 1.0e-300
            )
        else:
            q_live = Q_AREA
        cn_series.append(float(torch.dot(proposal.load.total_force, normal).item()) / q_live)
        stepper.commit(owner, proposal)
        diag = owner.state.diagnostics[-1]
        lev_total += int(diag["lev_release_count"])
        particles = int(diag["particle_count"])
        deposited = max(deposited, float(diag.get("full_surface_neumann_max_abs", 0.0)))
        step_newborn = abs(float(diag.get("newborn_lev_circulation_max_abs", 0.0)))
        newborn = max(newborn, step_newborn)
        newborn_series.append(step_newborn)
        particle_series.append(particles)
        candidate = diag.get("te_leg_candidate") or {}
        delta = candidate.get("te_leg_candidate_delta_n")
        if delta:
            q_area = 0.5 * 1.208 * FREESTREAM**2 * CHORD_M * SPAN_M
            normal = torch.tensor(
                [math.sin(math.radians(alpha)), 0.0, math.cos(math.radians(alpha))],
                device=DEVICE,
                dtype=torch.float64,
            )
            te_leg_delta_cn_series.append(
                float(torch.dot(torch.tensor(delta, device=DEVICE, dtype=torch.float64), normal).item() / q_area)
            )
        if step % 25 == 0 or step <= 3:
            # Progress trace: without it a failure deep in a 300-step leg cannot be
            # localised (the first Stage 2 run died with only a traceback and no
            # step number).  Goes to stderr so the JSON payload is unaffected.
            print(
                f"[a={alpha:g}] step {step}/{steps} cn={cn_series[-1]:+.4f} "
                f"n_p={particles} newborn={step_newborn:.3e}",
                file=sys.stderr,
                flush=True,
            )
        cull_count_series.append(int(diag.get("particle_cull_count", 0)))
        cull_circulation_series.append(
            float(diag.get("particle_cull_circulation", 0.0))
        )
        kelvin = max(kelvin, float(diag.get("kelvin_identity_residual", diag.get("kelvin_max_abs", 0.0))))
        if step >= 2:
            geometry = solver.surface.evaluate(None, None)
            te_ring = geometry.rings.reshape(NC, NS, 4, 3)[-1, :, [3, 2]]
            invariants = geometry_invariants(owner.state.wake_rings, te_ring)
            if invariants:
                geom = invariants
                worst_gap = max(worst_gap, invariants["join_gap_max_m"])
                worst_cosine = max(worst_cosine, invariants["cosine_max"])

    tail = cn_series[-50:]
    return {
        "alpha_deg": alpha,
        "joint": joint,
        "steps": steps,
        "t_star": steps * DT_STAR,
        "cn_last": cn_series[-1],
        "cn_tail50_mean": sum(tail) / len(tail),
        "cn_tail50_min": min(tail),
        "cn_tail50_max": max(tail),
        "experiment_cn": EXPERIMENT_CN.get(alpha),
        "lev_release_total": lev_total,
        "particles": particles,
        "deposited_neumann_max_m_s": deposited,
        "newborn_lev_max": newborn,
        "kelvin_max": kelvin,
        "worst_join_gap_max_m": worst_gap,
        "worst_cosine_max": worst_cosine,
        "final_geometry": geom,
        # Full series so a divergence can be located to a step instead of
        # being discovered as a single terminal number.
        "cn_series": cn_series,
        "newborn_series": newborn_series,
        "particle_series": particle_series,
        "te_leg_delta_cn_series": te_leg_delta_cn_series,
        "cull_count_series": cull_count_series,
        "cull_circulation_series": cull_circulation_series,
        "ramp_factor_series": ramp_factor_series,
        "newborn_area_over_analytic": (
            geom.get("newborn_area_max_m2", float("nan")) / ANALYTIC_NEWBORN_AREA
            if geom
            else float("nan")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--angles", default="0,5,15,19")
    parser.add_argument("--legacy", action="store_true", help="run with joint flag OFF")
    parser.add_argument(
        "--te-leg",
        action="store_true",
        help="L2-3: record the two TE-leg strength candidates per step",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="E1-WINDOW: LE-closure RHS window in steps (0 = off)",
    )
    parser.add_argument(
        "--flux",
        action="store_true",
        help="E1-FLUX: prescribe the shed circulation as 0.5*U_s^2*dt",
    )
    parser.add_argument(
        "--particle-age",
        type=int,
        default=100,
        help="particle_max_age_steps (retention horizon; frozen Roj value 100)",
    )
    parser.add_argument(
        "--particle-capacity",
        type=int,
        default=32768,
        help="particle_capacity (raise together with --particle-age)",
    )
    parser.add_argument(
        "--le-open",
        action="store_true",
        help="Route A: deposit the newborn LE sheet as an OPEN spanwise filament "
        "(net vorticity -gamma_lev) instead of a closed zero-circulation ribbon",
    )
    parser.add_argument(
        "--le-open-no-tev",
        action="store_true",
        help="Route A sub-variant: also remove the +gamma_lev compensation from the "
        "TE wake row (requires --le-open)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="EXPERIMENT ONLY (route A): dvm_smoothing_radius_chord. A MONOPOLE core "
        "must satisfy sigma <= x_c0/4 = 0.015c so the collocation sits on the 1/r "
        "branch (f(4)=0.2499 vs 0.25); the frozen dipole value 0.04c is 2.7x too "
        "large for that role. Keep smoothing/spacing = 2.125.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="EXPERIMENT ONLY (route A): dvm_target_spacing_chord; must stay sigma/2.125",
    )
    parser.add_argument(
        "--stretching",
        action="store_true",
        help="Stage 2: 3-D vortex stretching d(gamma)/dt = J . gamma on the particles",
    )
    parser.add_argument(
        "--stretching-rvpm",
        action="store_true",
        help="use the frozen rVPM stretching form dgamma/dt = S - 3 Z gamma "
        "(research_rvpm_arch.md V4) instead of the bare dgamma/dt = S",
    )
    parser.add_argument(
        "--core-spreading",
        action="store_true",
        help="Stage 2: also evolve the core radius d(sigma)/dt = -sigma Z (needs --stretching)",
    )
    parser.add_argument(
        "--ramp-steps",
        type=int,
        default=0,
        help="Frozen Roj startup ramp: half-cosine 0->U0 over the first N steps "
        "(t* = N*0.01). The project value is t*=1 => 100 steps, tagged "
        "numerical_protocol and excluded from statistics. 0 keeps the historical "
        "impulsive start (the +43..+73 first-step cn artefact).",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")
    le_open = bool(args.le_open or args.le_open_no_tev)
    le_open_drop_tev = bool(args.le_open_no_tev)
    angles = [float(a) for a in args.angles.split(",") if a.strip()]
    results = [
        run(
            a,
            args.steps,
            joint=not args.legacy,
            particle_age=args.particle_age,
            particle_capacity=args.particle_capacity,
            flux=args.flux,
            window=args.window,
            te_leg=args.te_leg,
            le_open=le_open,
            le_open_drop_tev=le_open_drop_tev,
            sigma_chord=args.sigma,
            spacing_chord=args.spacing,
            ramp_steps=int(args.ramp_steps),
            stretching=bool(
                args.stretching or args.core_spreading or args.stretching_rvpm
            ),
            core_spreading=bool(args.core_spreading),
            stretching_rvpm=bool(args.stretching_rvpm),
        )
        for a in angles
    ]
    payload = {
        "note": (
            "startup-segment samples (t* = steps*0.01); NOT steady-state accuracy"
        ),
        "particle_max_age_steps": int(args.particle_age),
        "particle_capacity": int(args.particle_capacity),
        "le_open_deposit": le_open,
        "le_open_deposit_drop_tev_compensation": le_open_drop_tev,
        "dvm_smoothing_radius_chord": args.sigma,
        "dvm_target_spacing_chord": args.spacing,
        "ramp_steps": int(args.ramp_steps),
        "particle_stretching": bool(args.stretching or args.core_spreading),
        "particle_core_spreading": bool(args.core_spreading),
        "particle_stretching_rvpm": bool(args.stretching_rvpm),
        "ramp_protocol": "frozen half-cosine 0->U0 over t* in [0,1]; 0 = impulsive start",
        "analytic_newborn_area_m2": ANALYTIC_NEWBORN_AREA,
        "streamwise_segment_m": FREESTREAM * DT,
        "results": results,
        "ramp_factor_series": results[0].get("ramp_factor_series"),
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
