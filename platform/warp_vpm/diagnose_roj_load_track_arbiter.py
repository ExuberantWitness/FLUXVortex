"""⑬ arbiter: which load track is right?  Decided WITHIN a case where track B is externally validated.

The two tracks are:
  B (resolved/KJ): the rigid native solver's `load.total_force` = sum of the KJ
     filament + unsteady point forces.  This is the definition behind the validated
     results: rigid AR=2 alpha=5 -> Cn 0.3425 vs experiment 0.3413 (+0.35%), and
     Izraelevitz Fig.14 CT MAE 0.0409.
  A (author quadrature): `author_load.constant_pressure * areas * normals`, which the
     Q16 membrane path still feeds to the structure (and which over-predicts membrane
     Cn by +34%).

Cross-case comparison (rigid uses B, membrane uses A) is suggestive but confounded by
the case difference.  This probe removes the confound: on the SAME alpha=5 rigid state
it evaluates BOTH definitions and compares each against the experimental 0.3413.

If B lands on the experiment and A does not, track A carries a systematic bias on a
case where B is externally validated -- which is the cleanest arbitration available
without new external ground truth.

Run:
  PYTHONPATH=src:platform:platform/warp_vpm python3 \
    platform/warp_vpm/diagnose_roj_load_track_arbiter.py --steps 300 --ramp-steps 100
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
from fluxvortex.warp_fsi.rigid_flux_v5m_native import RigidNativeV5MSolver, RigidV5MSurface

CHORD, SPAN, U, RHO = 0.0688, 0.1375, 5.0, 1.208
NC, NS = 15, 30
DT_STAR = 0.01
EXPERIMENT_CN = 0.3413


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ramp-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")

    dt = DT_STAR * CHORD / U
    reference = build_izra_reference_grid(
        chordwise_panels=NC, spanwise_panels=NS, chord_m=CHORD, span_m=SPAN,
        pivot_fraction_chord=0.25, device=args.device,
    )
    theta = math.radians(args.alpha) * 0.5
    attitude = torch.tensor(
        [math.cos(theta), 0.0, math.sin(theta), 0.0], device=args.device, dtype=torch.float64
    )
    zero = torch.zeros(3, device=args.device, dtype=torch.float64)

    def law(_t: float):
        return zero, attitude, zero, zero.clone()

    kin = PrescribedRigidSurfaceKinematics(reference, law, surface_id="w", body_id="b")
    settings = NativeV5MConfig(
        chordwise_panels=NC, spanwise_panels=NS, density=RHO, freestream=U,
        aerodynamic_dt=dt, lesp_crit=0.11, wake_max_rows=300, particle_capacity=32768,
        particle_max_age_steps=100, wake_history_mode="bound_rate", wake_free_rows=100,
        dvm_target_spacing_chord=0.018, device=args.device, joint_separation_solve=True,
    )
    solver = RigidNativeV5MSolver(
        RigidV5MSurface(chordwise_panels=NC, spanwise_panels=NS, device=args.device),
        settings,
    )
    stepper = RigidV5MStepper(solver, device=args.device)
    owner = stepper.initialize((kin.evaluate(0.0),))
    normal = torch.tensor(
        [math.sin(math.radians(args.alpha)), 0.0, math.cos(math.radians(args.alpha))],
        device=args.device, dtype=torch.float64,
    )

    cn_b: list[float] = []
    cn_a: list[float] = []
    for step in range(1, args.steps + 1):
        if args.ramp_steps > 0:
            t_star = step * DT_STAR
            startup = args.ramp_steps * DT_STAR
            factor = (
                1.0 if t_star >= startup
                else 0.5 * (1.0 - math.cos(math.pi * t_star / startup))
            )
        else:
            factor = 1.0
        speed = U * factor
        solver.v_inf = torch.tensor(
            [speed, 0.0, 0.0], device=args.device, dtype=torch.float64
        )
        object.__setattr__(solver.settings, "freestream", speed)
        proposal = stepper.propose(owner, (kin.evaluate(step * dt),), dt)
        geometry = solver.surface.evaluate(None, None)
        pressure = proposal.author_load.constant_pressure
        quadrature_force = torch.sum(
            pressure[:, None] * geometry.areas[:, None] * geometry.normals, dim=0
        )
        q_live = max(0.5 * RHO * speed * speed * CHORD * SPAN, 1.0e-300)
        cn_b.append(float(torch.dot(proposal.load.total_force, normal).item()) / q_live)
        cn_a.append(float(torch.dot(quadrature_force, normal).item()) / q_live)
        stepper.commit(owner, proposal)
        if step % 100 == 0:
            print(
                f"step {step}: cn_B(KJ)={cn_b[-1]:+.4f}  cn_A(quadrature)={cn_a[-1]:+.4f}",
                file=sys.stderr, flush=True,
            )

    lo = args.ramp_steps if args.ramp_steps < args.steps else args.steps // 2
    if args.steps - lo < 1:
        print("post-ramp window is empty; increase --steps")
        return 2
    b = sum(cn_b[lo:]) / len(cn_b[lo:])
    a = sum(cn_a[lo:]) / len(cn_a[lo:])
    print(f"alpha={args.alpha:g}  post-ramp window steps {lo+1}-{args.steps}")
    print(f"  track B (resolved/KJ)   mean Cn = {b:+.4f}   vs experiment {EXPERIMENT_CN:.4f}  ({100*(b-EXPERIMENT_CN)/EXPERIMENT_CN:+.1f}%)")
    print(f"  track A (quadrature)    mean Cn = {a:+.4f}   vs experiment {EXPERIMENT_CN:.4f}  ({100*(a-EXPERIMENT_CN)/EXPERIMENT_CN:+.1f}%)")
    print(f"  A/B = {a/b:.4f}")
    if abs(b - EXPERIMENT_CN) < abs(a - EXPERIMENT_CN):
        print("  => track B is closer on a case where B is externally validated: A carries a systematic bias")
    else:
        print("  => track A is closer here; the cross-case inference is NOT supported")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"alpha": args.alpha, "cn_B": cn_b, "cn_A": cn_a,
                       "mean_B": b, "mean_A": a, "experiment": EXPERIMENT_CN},
                      handle, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
