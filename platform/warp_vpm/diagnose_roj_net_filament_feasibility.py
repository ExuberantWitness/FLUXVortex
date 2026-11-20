"""Is the net-LE-filament constraint well posed in the CURRENT geometry?

Step 3 (E1-BUFFER-NETFILAMENT) presupposes that the bound first-row filament and
the shed leading-edge sheet share one material edge, so that the VoFFLE net
filament

    Gamma_LE = Gamma_bound,first - lambda_first

is a meaningful quantity to constrain to s*kappa_crit (Bird et al. 2021,
eqs 14/17-19).  The project review already measured that the bound first-row
filament sits 0.0166667 c BEHIND the physical LE while the Gaussian band is
anchored AT the physical LE, i.e. they are not the same material line.

This diagnostic quantifies the consequence before any code is restructured:
per step it reports

  Gamma_b   = gamma_bound[:ns]                 (bound LE-row circulation, m^2/s)
  lambda    = lev_strip_last_release           (newest shed sheet row, m^2/s)
  net       = Gamma_b - lambda                 (the net filament, m^2/s)
  target    = -sign(lesp_pre) * crit * scale   (the critical value, m^2/s)

and the ratio net/target.  Interpretation:

  * ratio ~= 1  -> the bound-only pin already satisfies the net relation, so a
                   net-filament rewrite is a no-op and the defect is elsewhere;
  * ratio ~= 0  -> the shed sheet cancels the whole bound filament, i.e. the
                   net relation is badly violated and the representation (not
                   the coefficient) has to change first;
  * anything else -> the mismatch is quantified per angle, and it tells us how
                   much of the runaway is representational.

No solver change; read-only sampling of the committed/proposed state.

Usage:
  PYTHONPATH=src:platform:platform/warp_vpm python3 \
    platform/warp_vpm/diagnose_roj_net_filament_feasibility.py \
    --alpha 15 --steps 300 --out <new>.json
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

DEVICE = "cuda:0"
CHORD_M = 0.0688
SPAN_M = 0.1375
FREESTREAM = 5.0
DT_STAR = 0.01
NC, NS = 15, 30
DT = DT_STAR * CHORD_M / FREESTREAM
LESP_FACTOR = 1.13


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--stride", type=int, default=10)
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
        particle_capacity=32768,
        particle_max_age_steps=100,
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

    rows = []
    for step in range(1, args.steps + 1):
        proposal = stepper.propose(owner, (kin.evaluate(step * DT),), DT)
        state = proposal.trial_state
        trial_grid_old = proposal.trial_state.gamma_bound.reshape(NC, NS)
        geometry = solver.surface.evaluate(None, None)

        chord = 0.5 * (
            torch.linalg.vector_norm(
                geometry.trailing_edge[:-1] - geometry.leading_edge[:-1], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.trailing_edge[1:] - geometry.leading_edge[1:], dim=1
            )
        )
        first_dx = 0.5 * (
            torch.linalg.vector_norm(
                geometry.rings[:NS, 3] - geometry.rings[:NS, 0], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.rings[:NS, 2] - geometry.rings[:NS, 1], dim=1
            )
        )
        theta = torch.acos(torch.clamp(1.0 - 2.0 * first_dx / chord, -1.0, 1.0))
        scale = chord * FREESTREAM * (theta + torch.sin(theta)) / LESP_FACTOR

        gamma_b = state.gamma_bound[:NS]
        lam = state.lev_strip_last_release
        # sign of the critical target is the sign of the pre-solve LESP; the
        # diagnostics record the pre-solve LESP magnitude only, so use the sign
        # of the bound LE circulation as its discrete proxy (same convention as
        # the pin target, which is -sign(lesp)*crit*scale while lesp = -G/scale).
        sign = torch.sign(gamma_b)
        target = -sign * settings.effective_lesp_crit * scale
        net = gamma_b - lam
        # Kelvin-implied shed amount: the step's decrease of each strip's TOTAL
        # bound circulation.  trial.gamma_bound still holds the committed value
        # here, so this is a genuine per-step increment (same convention the
        # bound_rate TEV path uses).
        grid_new = state.gamma_bound.reshape(NC, NS)
        grid_old = trial_grid_old
        lambda_kelvin = -(grid_new.sum(dim=0) - grid_old.sum(dim=0))
        kelvin_ratio = torch.median(
            lam.abs() / torch.clamp(lambda_kelvin.abs(), min=1e-30)
        )
        ratio = net / torch.where(
            target.abs() > 0, target, torch.full_like(target, float("nan"))
        )

        diag = state.diagnostics[-1]
        if step % args.stride == 0 or step <= 3:
            rows.append(
                {
                    "step": step,
                    "separated_strips": int(diag["separated_strip_count"]),
                    "gamma_b_max_abs": float(gamma_b.abs().max().item()),
                    "lambda_max_abs": float(lam.abs().max().item()),
                    "net_max_abs": float(net.abs().max().item()),
                    "target_max_abs": float(target.abs().max().item()),
                    "net_over_target_median": float(
                        torch.nanmedian(ratio[torch.isfinite(ratio)]).item()
                    )
                    if bool(torch.isfinite(ratio).any().item())
                    else float("nan"),
                    "sum_gamma_bound_median": float(
                        torch.median(grid_new.sum(dim=0)).item()
                    ),
                    "lambda_kelvin_median": float(
                        torch.median(lambda_kelvin).item()
                    ),
                    "lambda_solved_over_kelvin_median": float(kelvin_ratio.item()),
                    "lambda_over_gamma_median": float(
                        torch.median(
                            (lam.abs() / torch.clamp(gamma_b.abs(), min=1e-30))
                        ).item()
                    ),
                }
            )
        stepper.commit(owner, proposal)

    payload = {"alpha_deg": args.alpha, "steps": args.steps, "rows": rows}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1)
    for row in rows[:3] + rows[-3:]:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
