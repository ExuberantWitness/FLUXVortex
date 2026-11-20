"""L2-1: is the shed circulation flux-consistent?

Literature ruling (see the two literature reviews, 2026-11-21): heavy blob
overlap (0.13-0.17 sigma) is NOT an instability cause; the consistency
requirement is that the circulation carried per unit sheet length equals the
physical vorticity flux at the edge.  For a sharp edge the Kaden/Didden slug
model gives the circulation shed per step per unit span

    dGamma_flux = 0.5 * U_s^2 * dt

with U_s the slip speed at separation.  The implied sheet strength is
gamma_phys = dGamma_flux / ds, where ds is the streamwise extent of the
deposited sheet.

Definitions used here (all measurable from the solver state, no solver change):
  * Gamma_ribbon = lev_strip_last_release[strip]  (m^2/s, what we actually deposit)
  * U_s          = |Gamma_bound_LE| / dx_LE       (m/s): for a vortex sheet the
                   tangential velocity jump across the bound sheet IS its
                   strength, so the bound first-row sheet strength is the
                   natural discrete slip at the leading edge.
  * ds           = mean streamwise offset of the newest cohort's particles from
                   the leading edge (m) -- the deposited sheet's extent.
  * gamma_recon  = Gamma_ribbon / ds

Outputs the ratio Gamma_ribbon / dGamma_flux and gamma_recon / (0.5 U_s) per
step, plus a dt-halved pass to expose dt-dependence (the runaway fingerprint:
if the shed circulation is prescribed independently of dt, gamma_recon ~ 1/dt).

Usage:
  PYTHONPATH=src:platform:platform/warp_vpm python3 \
    platform/warp_vpm/diagnose_roj_flux_consistency.py --alpha 15 --steps 650 --out <new>.json
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
NC, NS = 15, 30
DT_STAR_NOMINAL = 0.01


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


def run(alpha: float, steps: int, dt_star: float) -> dict:
    dt = dt_star * CHORD_M / FREESTREAM
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
    settings = NativeV5MConfig(
        chordwise_panels=NC,
        spanwise_panels=NS,
        density=1.208,
        freestream=FREESTREAM,
        aerodynamic_dt=dt,
        lesp_crit=0.11,
        wake_max_rows=300,
        particle_capacity=32768,  # frozen Roj values: measure production config
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
    for step in range(1, steps + 1):
        proposal = stepper.propose(owner, (kin.evaluate(step * dt),), dt)
        state = proposal.trial_state
        geometry = solver.surface.evaluate(None, None)

        # first-cell chord (m): dx_LE per strip
        dx_le = 0.5 * (
            torch.linalg.vector_norm(
                geometry.rings[:NS, 3] - geometry.rings[:NS, 0], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.rings[:NS, 2] - geometry.rings[:NS, 1], dim=1
            )
        )
        gamma_le = state.gamma_bound[:NS]
        u_s = torch.abs(gamma_le) / dx_le  # m/s, sheet-strength = slip

        gamma_ribbon = state.lev_strip_last_release
        # streamwise extent of the newest deposited cohort
        field = state.particle_field
        ds = torch.zeros((), device=DEVICE, dtype=torch.float64)
        if field.n:
            newest = int(field.birth_step[: field.n].max().item())
            fresh = field.birth_step[: field.n] == newest
            fresh &= field.source_strip[: field.n] >= 0
            if bool(fresh.any().item()):
                # chordwise extent only: measure from the particle's own LE
                # node so the span offset cannot enter the metric.
                strip_ids = torch.clamp(
                    field.source_strip[: field.n][fresh], 0, NS - 1
                )
                offset = field.pos[: field.n][fresh] - geometry.leading_edge[strip_ids]
                chord_dir = torch.tensor(
                    [math.cos(math.radians(alpha)), 0.0, math.sin(math.radians(alpha))],
                    device=DEVICE,
                    dtype=torch.float64,
                )
                ds = torch.mean(torch.abs(offset @ chord_dir))

        dt_flux = 0.5 * u_s**2 * dt  # m^2/s per strip
        strip = int(torch.argmax(torch.abs(gamma_ribbon)).item())
        with torch.no_grad():
            rows.append(
                {
                    "step": step,
                    "strip": strip,
                    "gamma_ribbon": float(gamma_ribbon[strip].item()),
                    "gamma_flux": float(dt_flux[strip].item()),
                    "ratio_ribbon_over_flux": float(
                        (gamma_ribbon[strip] / dt_flux[strip]).item()
                    )
                    if abs(float(dt_flux[strip].item())) > 0
                    else float("nan"),
                    "u_s": float(u_s[strip].item()),
                    "ds_m": float(ds.item()),
                    "gamma_recon": float((gamma_ribbon[strip] / ds).item())
                    if float(ds.item()) > 0
                    else float("nan"),
                    "gamma_phys_sheet": 0.5 * float(u_s[strip].item()),
                    "newborn_max_abs": float(
                        state.diagnostics[-1].get(
                            "newborn_lev_circulation_max_abs", 0.0
                        )
                    ),
                }
            )
        stepper.commit(owner, proposal)

    return {
        "alpha_deg": alpha,
        "steps": steps,
        "dt_star": dt_star,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument(
        "--t-star",
        type=float,
        default=3.5,
        help="physical window: each dt* pass runs t*/dt* steps",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dt-stars",
        default="0.01,0.005",
        help="comma separated dt* values; the second exposes dt-dependence",
    )
    parser.add_argument("--step-stride-in-output", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")
    dt_stars = [float(v) for v in args.dt_stars.split(",") if v.strip()]

    results = []
    for dt_star in dt_stars:
        # identical physical window for every dt* (the whole point)
        steps = int(round(args.t_star / dt_star))
        print(f"--- dt*={dt_star}: {steps} steps ---")
        result = run(args.alpha, steps, dt_star)
        rows = result["rows"]
        kept = rows[:: args.step_stride_in_output]
        ratios = [
            r["ratio_ribbon_over_flux"]
            for r in rows[200:]
            if r["ratio_ribbon_over_flux"] == r["ratio_ribbon_over_flux"]
        ]
        summary = {
            "dt_star": dt_star,
            "steps": steps,
            "ratio_median_after_step200": float(
                torch.tensor(ratios).median().item()
            )
            if ratios
            else float("nan"),
            "ratio_max": float(max(ratios)) if ratios else float("nan"),
            "ds_over_Us_dt_median": float(
                torch.tensor(
                    [
                        r["ds_m"] / max(r["u_s"] * dt_star * CHORD_M / FREESTREAM, 1e-30)
                        for r in rows[200:]
                    ]
                ).median().item()
            )
            if rows
            else float("nan"),
        }
        results.append({"summary": summary, "rows": kept})
        print(json.dumps(summary, indent=1))

    with open(args.out, "w") as handle:
        json.dump({"alpha_deg": args.alpha, "runs": results}, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
