"""Cross-time total-vorticity ledger for the native V5M system.

The solver's own ``kelvin`` diagnostic is the algebraic identity
``gamma_tev - Gamma_TE - gamma_lev == 0`` and carries no information.  This probe
measures the invariant that actually matters:

    Omega(t) = sum_bound Gamma_i A_i n_i  +  sum_wake Gamma_k A_k n_k  +  sum_p gamma_p

For a closed vortex ring of circulation ``Gamma`` spanning a planar area ``A``
with unit normal ``n``, ``integral(omega) dV = Gamma A n``; for a vortex-segment
particle, ``gamma_p = circul * segment`` is exactly its contribution.  Total
vorticity is conserved for a potential-flow system (the bound sheet is the only
place vorticity can be exchanged with the wake), so any per-step drift is a
bookkeeping defect -- and its rate identifies *which* term is mis-billed.

Reporting the drift *per unit gamma_lev* is the decision quantity: a drift of
``+gamma_lev`` per step means the newly shed amount is accounted twice (once in
the closed LE ribbon, once as ``+gamma_lev`` added to the TE wake row).

Run:
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 \
        python3 platform/warp_vpm/diagnose_roj_circulation_ledger.py --alpha 15 --steps 12
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

CHORD = 0.0688
SPAN = 0.1375
FREESTREAM = 5.0
NC, NS = 15, 30
RHO = 1.208


def ring_vorticity(rings: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """``sum_k Gamma_k A_k n_k`` for (n,4,3) rings; row-major over (row, span)."""

    if rings.shape[0] == 0:
        return torch.zeros(3, device=rings.device, dtype=torch.float64)
    corners = rings.reshape(-1, 4, 3)
    cross_sum = torch.cross(corners, torch.roll(corners, -1, dims=1), dim=2).sum(dim=1)
    area_normal = 0.5 * cross_sum
    return torch.sum(area_normal * gamma.reshape(-1, 1), dim=0)


def ledger(solver, state) -> dict[str, torch.Tensor]:
    """Total vorticity split by carrier."""

    geometry = solver.surface.evaluate(None, None)
    omega_bound = ring_vorticity(geometry.rings, state.gamma_bound)
    omega_wake = ring_vorticity(state.wake_rings, state.wake_gamma)
    field = state.particle_field
    omega_particles = (
        field.gammas_cuda.sum(dim=0)
        if field.n
        else torch.zeros(3, device=state.gamma_bound.device, dtype=torch.float64)
    )
    return {
        "bound": omega_bound,
        "wake": omega_wake,
        "particles": omega_particles,
        "total": omega_bound + omega_wake + omega_particles,
    }


def build(alpha_deg: float, dt_star: float, device: str):
    dt = dt_star * CHORD / FREESTREAM
    reference = build_izra_reference_grid(
        chordwise_panels=NC,
        spanwise_panels=NS,
        chord_m=CHORD,
        span_m=SPAN,
        pivot_fraction_chord=0.25,
        device=device,
    )
    theta = math.radians(alpha_deg) * 0.5
    attitude = torch.tensor(
        [math.cos(theta), 0.0, math.sin(theta), 0.0], device=device, dtype=torch.float64
    )
    zero = torch.zeros(3, device=device, dtype=torch.float64)

    def law(_t: float):
        return zero, attitude, zero, zero.clone()

    kinematics = PrescribedRigidSurfaceKinematics(
        reference, law, surface_id="w", body_id="b"
    )
    settings = NativeV5MConfig(
        chordwise_panels=NC,
        spanwise_panels=NS,
        density=RHO,
        freestream=FREESTREAM,
        aerodynamic_dt=dt,
        lesp_crit=0.11,
        wake_max_rows=300,
        particle_capacity=32768,
        particle_max_age_steps=100,
        wake_history_mode="bound_rate",
        wake_free_rows=100,
        dvm_target_spacing_chord=0.018,
        device=device,
        joint_separation_solve=True,
    )
    solver = RigidNativeV5MSolver(
        RigidV5MSurface(chordwise_panels=NC, spanwise_panels=NS, device=device), settings
    )
    stepper = RigidV5MStepper(solver, device=device)
    owner = stepper.initialize((kinematics.evaluate(0.0),))
    return solver, stepper, owner, kinematics, dt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--dt-star", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    solver, stepper, owner, kinematics, dt = build(args.alpha, args.dt_star, args.device)

    initial = ledger(solver, owner.state)
    scale = float(torch.linalg.vector_norm(initial["bound"]).item())
    print(f"alpha = {args.alpha:g} deg, dt* = {args.dt_star:g}, steps = {args.steps}")
    print(f"|Omega_bound(0)| = {scale:.6e} m^3/s (drift scale)")
    print()
    print(" step  lam_max      |Omega_bound|   |Omega_wake|    |Omega_part|   "
          "|Omega_tot|      d|Omega|    d|Omega|/lam  n_p    n_wake")
    print(" " + "-" * 118)

    rows = []
    previous_total = initial["total"]
    for step in range(1, args.steps + 1):
        proposal = stepper.propose(owner, (kinematics.evaluate(step * dt),), dt)
        diagnostics = proposal.trial_state.diagnostics[-1]
        lam = diagnostics.get("newborn_lev_circulation_max_abs", float("nan"))
        stepper.commit(owner, proposal)
        current = ledger(solver, owner.state)
        delta = current["total"] - previous_total
        delta_mag = float(torch.linalg.vector_norm(delta).item())
        lam_value = float(lam)
        ratio = delta_mag / lam_value if lam_value > 0.0 else float("nan")
        magnitude = float(torch.linalg.vector_norm(current["total"]).item())
        print(f" {step:4d}  {lam_value:10.4e}  "
              f"{float(torch.linalg.vector_norm(current['bound']).item()):14.6e}  "
              f"{float(torch.linalg.vector_norm(current['wake']).item()):13.6e}  "
              f"{float(torch.linalg.vector_norm(current['particles']).item()):13.6e}  "
              f"{magnitude:13.6e}  {delta_mag:11.4e}  {ratio:+11.4f}  "
              f"{owner.state.particle_field.n:5d}  {owner.state.wake_rings.shape[0] // NS:5d}")
        rows.append(
            {
                "step": step,
                "lambda_max": lam_value,
                "omega_bound_norm": float(torch.linalg.vector_norm(current["bound"]).item()),
                "omega_wake_norm": float(torch.linalg.vector_norm(current["wake"]).item()),
                "omega_particles_norm": float(
                    torch.linalg.vector_norm(current["particles"]).item()
                ),
                "omega_total_norm": magnitude,
                "omega_total": [float(v) for v in current["total"].tolist()],
                "delta_norm": delta_mag,
                "delta_over_lambda": ratio,
                "particle_count": owner.state.particle_field.n,
                "wake_row_count": owner.state.wake_rings.shape[0] // NS,
            }
        )
        previous_total = current["total"]

    cumulative = [
        row for row in rows if row["lambda_max"] == row["lambda_max"] and row["lambda_max"] > 0
    ]
    if cumulative:
        mean_ratio = sum(row["delta_over_lambda"] for row in cumulative) / len(cumulative)
        print()
        print(f"mean |dOmega| / lambda_max over releasing steps = {mean_ratio:+.4f}")
        print("  ~0  => the shed amount is conserved once;  ~1 => booked twice")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"alpha_deg": args.alpha, "dt_star": args.dt_star, "rows": rows}, handle, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
