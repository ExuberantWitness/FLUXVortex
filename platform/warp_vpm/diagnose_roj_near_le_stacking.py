"""Which LEV cohorts dominate the induced velocity at the leading edge?

A15 (rigid AR=2, U=5, joint path) diverges late even with ZERO particle culls
(age=200 probe), so the cause is not retention.  The remaining hypothesis from
the review is geometric: consecutive Gaussian ribbons are anchored at the
physical LE with a frontier displacement of only 0.005-0.0066c, so consecutive
sheets nearly coincide and their mutual induction at the LE can grow without
bound while the bound LE circulation is pinned at the critical LESP.

This diagnostic decomposes the particle-induced normal velocity at the LE
collocation rows by BIRTH-STEP COHORT (newest, 2nd newest, ..., and the bulk),
on a live trajectory.  It changes nothing in the solver.

Interpretation guide:
  * if the newest 1-3 cohorts dominate and their share grows, the instability
    is near-LE stacking -> the fix is buffer geometry / core placement, not a
    new constraint coefficient;
  * if the whole population grows together, the feedback is in the solve.

Usage:
  PYTHONPATH=src:platform:platform/warp_vpm python3 \
    platform/warp_vpm/diagnose_roj_near_le_stacking.py \
    --alpha 15 --steps 650 --out <new>.json
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidNativeV5MSolver,
    RigidV5MSurface,
)
from pfield_torch_gpu import CudaParticleField

DEVICE = "cuda:0"
CHORD_M = 0.0688
SPAN_M = 0.1375
FREESTREAM = 5.0
DT_STAR = 0.01
NC, NS = 15, 30
DT = DT_STAR * CHORD_M / FREESTREAM

COHORTS = (
    ("newest", 1),
    ("second", 2),
    ("third", 3),
    ("4-5", 5),
    ("6-10", 10),
    ("older", None),
)


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


def _subset_field(field, mask: torch.Tensor) -> CudaParticleField:
    count = int(torch.count_nonzero(mask).item())
    scratch = CudaParticleField(field.capacity, device=field.device)
    if count:
        scratch.pos[:count].copy_(field.pos[: field.n][mask])
        scratch.gamma[:count].copy_(field.gamma[: field.n][mask])
        scratch.sigma[:count].copy_(field.sigma[: field.n][mask])
        scratch.circul[:count].copy_(field.circul[: field.n][mask])
        scratch.n = count
    return scratch


def cohort_profile(solver, state) -> dict:
    """Decompose the particle normal velocity at the LE rows by cohort."""
    geometry = solver.surface.evaluate(None, None)
    field = state.particle_field
    if not field.n:
        return {"particles": 0}
    targets = geometry.collocation[:NS]
    normals = geometry.normals[:NS]
    live_birth = field.birth_step[: field.n]
    newest = int(live_birth.max().item())
    rows = []
    covered = torch.zeros_like(live_birth, dtype=torch.bool)
    for name, age in COHORTS:
        if age is None:
            mask = ~covered
        elif name == "4-5":
            mask = (live_birth <= newest - 3) & (live_birth >= newest - 4)
        elif name == "6-10":
            mask = (live_birth <= newest - 5) & (live_birth >= newest - 9)
        else:
            mask = live_birth == newest - (age - 1)
        covered = covered | mask
        count = int(torch.count_nonzero(mask).item())
        if count == 0:
            rows.append({"cohort": name, "particles": 0})
            continue
        scratch = _subset_field(field, mask)
        velocity = scratch.velocity_at_cuda(targets)
        normal_velocity = torch.sum(velocity * normals, dim=1)
        distance = torch.linalg.vector_norm(
            field.pos[: field.n][mask]
            - geometry.leading_edge[:NS].mean(dim=0)[None, :],
            dim=1,
        )
        rows.append(
            {
                "cohort": name,
                "particles": count,
                "le_normal_velocity_max_abs": float(
                    torch.max(torch.abs(normal_velocity)).item()
                ),
                "le_normal_velocity_rms": float(
                    torch.sqrt(torch.mean(normal_velocity**2)).item()
                ),
                "mean_le_distance_over_chord": float(
                    torch.mean(distance).item() / CHORD_M
                ),
            }
        )
    total = torch.sum(
        field.velocity_at_cuda(targets) * normals, dim=1
    )
    return {
        "particles": int(field.n),
        "newest_birth_step": newest,
        "total_le_normal_velocity_rms": float(
            torch.sqrt(torch.mean(total**2)).item()
        ),
        "cohorts": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--particle-age", type=int, default=200)
    parser.add_argument("--particle-capacity", type=int, default=131072)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")

    reference = build_izra_reference_grid(
        chordwise_panels=NC,
        spanwise_panels=NS,
        chord_m=CHORD_M,
        span_m=SPAN_M,
        pivot_fraction_chord=0.25,
        device=DEVICE,
    )
    kin = PrescribedRigidSurfaceKinematics(
        reference, fixed_pitch_law(args.alpha), surface_id="w", body_id="b"
    )
    settings = NativeV5MConfig(
        chordwise_panels=NC,
        spanwise_panels=NS,
        density=1.208,
        freestream=FREESTREAM,
        aerodynamic_dt=DT,
        lesp_crit=0.11,
        wake_max_rows=300,
        particle_capacity=args.particle_capacity,
        particle_max_age_steps=args.particle_age,
        wake_history_mode="bound_rate",
        wake_free_rows=100,
        dvm_target_spacing_chord=0.018,
        device=DEVICE,
        joint_separation_solve=True,
    )
    solver = RigidNativeV5MSolver(
        RigidV5MSurface(chordwise_panels=NC, spanwise_panels=NS, device=DEVICE),
        settings,
    )
    stepper = RigidV5MStepper(solver, device=DEVICE)
    owner = stepper.initialize((kin.evaluate(0.0),))

    probes = []
    for step in range(1, args.steps + 1):
        proposal = stepper.propose(owner, (kin.evaluate(step * DT),), DT)
        stepper.commit(owner, proposal)
        if step % args.probe_every == 0 or step == args.steps:
            profile = cohort_profile(solver, owner.state)
            profile["step"] = step
            profile["newborn_max_abs"] = float(
                owner.state.diagnostics[-1].get(
                    "newborn_lev_circulation_max_abs", 0.0
                )
            )
            probes.append(profile)
            print(
                f"step {step}: particles={profile.get('particles')} "
                f"le_rms={profile.get('total_le_normal_velocity_rms', 0):.3e} "
                f"newborn={profile['newborn_max_abs']:.3e}"
            )

    payload = {"alpha_deg": args.alpha, "steps": args.steps, "probes": probes}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
