"""Rebuild and re-solve the bound system on the state the production solver actually solved.

The question
------------
On the rigid plate at alpha = 5 deg the production solver and the steady straight-wake
lattice (diagnose_ring_lattice_steady_cpu.py) agree at eps = 1e-9 and part below 1e-11
(production 0.1525 against 0.2194 at 1e-12).  The plan author's reading
(FLUX_V5M_PLAN_AUTHOR_REPLY_AND_V3_20261125.md, section 1 question 3) is that the
circulation is solved against a wake that has already been convected but does not yet
contain the new trailing-edge row, so the trailing edge carries an uncancelled bound leg
with a 1*U*dt gap behind it, and that the legacy insertion leaves rows 0 and 1 overlapping.

This script checks that reading on the production state itself, off line:

  1. residual of the PRODUCTION circulation in the system rebuilt from the solve-time wake
     (convected, new row not yet inserted).  Rounding level means the solve is exactly the
     system the plan describes.
  2. the same system with the nascent trailing-edge row added and tied to the
     trailing-edge unknowns -- the P0-B topology -- solved on the SAME convected wake.
     This is what P0-B should give on this state, before any production code is touched.
  3. a straight connected wake of the same row count for reference (the steady lattice).

Usage
-----
    CUDA_VISIBLE_DEVICES="" FLUXV_GPU_ONLY=0 PFIELD_DEVICE=cpu PYTHONPATH=src \\
        python3 platform/warp_vpm/diagnose_production_state_resolve_cpu.py \\
        --dump eps_sweep/state_rigid_a5_eps1e-12.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CORE_FRACTION = 1.0e-6          # the production value for these evaluations


def velocity(points: torch.Tensor, rings: torch.Tensor, reference_length: float, chunk: int) -> torch.Tensor:
    """(points, rings, 3) velocity of unit-strength rings, a chunk of rings at a time."""

    from fluxvortex.warp_fsi import kernels_ring_velocity as production

    out = torch.empty((points.shape[0], rings.shape[0], 3), dtype=torch.float64)
    for start in range(0, rings.shape[0], chunk):
        stop = min(start + chunk, rings.shape[0])
        # the kernel returns the four legs already summed
        out[:, start:stop] = production.native_ring_velocity_expanded(
            points, rings[start:stop], core_fraction=CORE_FRACTION, reference_length=reference_length
        )
    return out


def normal_coefficient(gamma: torch.Tensor, nc: int, ns: int, chord: float, speed: float, span: float) -> float:
    """Kutta-Joukowski coefficient on the trailing-edge row: 2 * sum(Gamma) * ds / (U S)."""

    return float(2.0 * gamma[(nc - 1) * ns :].sum() * (span / ns) / (speed * chord * span))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--chunk", type=int, default=400)
    arguments = parser.parse_args(argv)

    from fluxvortex.warp_fsi import kernels_ring_velocity as production

    data = np.load(arguments.dump)
    nc, ns = int(data["nc"]), int(data["ns"])
    chord, speed, dt = float(data["chord"]), float(data["speed"]), float(data["dt"])
    bound = torch.tensor(data["bound_rings"], dtype=torch.float64)
    collocation = torch.tensor(data["collocation"], dtype=torch.float64)
    normals = torch.tensor(data["normals"], dtype=torch.float64)
    gamma_production = torch.tensor(data["gamma_bound"], dtype=torch.float64)
    wake = torch.tensor(data["wake_rings"], dtype=torch.float64)          # solve-time
    wake_gamma = torch.tensor(data["wake_gamma"], dtype=torch.float64)
    v_inf = torch.tensor(data["v_inf"], dtype=torch.float64)
    dumped_eps = float(data["eps"])
    rows = int(wake.shape[0] // ns)
    reference_length = 1.0 / nc
    span = float(torch.linalg.vector_norm(bound[ns - 1, 2] - bound[0, 3]).item())

    kj = lambda g: normal_coefficient(g, nc, ns, chord, speed, span)
    print(f"dump: alpha {float(data['alpha_deg']):g} deg, {int(data['steps'])} steps, eps {dumped_eps:g}; "
          f"{nc} x {ns} bound rings; solve-time wake {rows} rows (speed {float(v_inf.norm()):.3f} m/s)")
    print(f"production circulation: Cn by Kutta-Joukowski {kj(gamma_production):+.5f}; "
          f"trailing-edge row sum {float(gamma_production[(nc - 1) * ns:].sum()):+.5f} m^2/s")
    row0 = wake[:ns]
    front, back = row0[:, 0], row0[:, 2]
    rear = bound.reshape(nc, ns, 4, 3)[-1, :, 2]
    unit = speed * dt
    gap_x = float((front[:, 0].mean() - rear[:, 0].mean()) / unit)
    print(f"solve-time row 0: front is {gap_x:+.3f} U*dt behind the trailing edge; "
          f"mean normal offset of its front corners {float(front[:, 2].mean() - rear[:, 2].mean()) / unit:+.3f} U*dt")

    # 1. the system the production solve saw: bound rings + the solve-time wake
    bound_velocity = velocity(collocation, bound, reference_length, arguments.chunk)
    matrix = torch.sum(bound_velocity * normals[:, None, :], dim=2)
    wake_velocity = torch.sum(velocity(collocation, wake, reference_length, arguments.chunk) * wake_gamma[:, None], dim=1)
    rhs = -torch.sum((v_inf[None, :] + wake_velocity) * normals, dim=1)
    residual = float((matrix @ gamma_production - rhs).abs().max() / rhs.abs().max())
    gamma_resolved = torch.linalg.solve(matrix, rhs)

    # 2. the P0-B topology on the same convected wake: the nascent row [trailing edge, row 0's
    #    upstream corners] with its strength tied to the trailing-edge unknowns.  The convected
    #    rows keep their stored strength and stay in the right-hand side, exactly as the solver
    #    has them.  (An earlier version of this script solved this system against the
    #    freestream alone, i.e. with no wake rows at all, and printed 0.079 for the 1e-12
    #    state; those numbers were withdrawn.)
    convected_front = torch.stack((row0[:, 0], row0[:, 1]), dim=1)       # (ns, 2, 3)
    rear_right, rear_left = bound.reshape(nc, ns, 4, 3)[-1, :, 2], bound.reshape(nc, ns, 4, 3)[-1, :, 3]
    nascent = torch.stack((rear_left, rear_right, convected_front[:, 1], convected_front[:, 0]), dim=1)
    nascent_unit = velocity(collocation, nascent, reference_length, arguments.chunk)        # (P, ns, 3)
    coupled = matrix.clone()
    coupled[:, (nc - 1) * ns :] += torch.sum(nascent_unit * normals[:, None, :], dim=2)
    gamma_p0b = torch.linalg.solve(coupled, rhs)
    residual_p0b = float((coupled @ gamma_production - rhs).abs().max() / rhs.abs().max())
    kelvin_gap = float((torch.sum(wake_gamma[:ns]) - torch.sum(gamma_production[(nc - 1) * ns :])).abs())

    # 3. straight connected wake of the same row count along the freestream, every row tied
    #    to the trailing-edge unknowns: the steady lattice, for reference
    rhs_free = -torch.sum(v_inf[None, :] * normals, dim=1)
    step = torch.tensor([unit, 0.0, 0.0], dtype=torch.float64)
    row_index = torch.arange(rows, dtype=torch.float64).reshape(rows, 1, 1, 1)
    c0 = torch.stack((rear_left, rear_right), dim=1)[None] + row_index * step    # (rows, ns, 2, 3)
    c1 = torch.stack((rear_left, rear_right), dim=1)[None] + (row_index + 1.0) * step
    straight = torch.stack((c0[..., 0, :], c0[..., 1, :], c1[..., 1, :], c1[..., 0, :]), dim=2)
    straight_unit = velocity(collocation, straight.reshape(-1, 4, 3), reference_length, arguments.chunk)
    straight_coupled = matrix.clone()
    straight_coupled[:, (nc - 1) * ns :] += torch.sum(
        straight_unit.reshape(collocation.shape[0], rows, ns, 3) * normals[:, None, None, :], dim=(1, 3)
    )
    gamma_straight = torch.linalg.solve(straight_coupled, rhs_free)

    print()
    print(f"{'system':74s} {'Cn':>9}  {'residual of production Gamma':>12}")
    print(f"{'production (the circulation the solver returned)':74s} {kj(gamma_production):+9.5f}")
    print(f"{'convected wake, NO nascent row (the legacy order)':74s} {kj(gamma_resolved):+9.5f}  {residual:12.1e}")
    print(f"{'convected wake + nascent row tied to the TE unknowns (P0-B)':74s} {kj(gamma_p0b):+9.5f}  {residual_p0b:12.1e}")
    print(f"{'straight connected wake, every row tied (steady lattice)':74s} {kj(gamma_straight):+9.5f}")
    print(f"\nnewest wake row strength minus the trailing-edge row (should be ~0 in a steady state): {kelvin_gap:.2e} m^2/s")
    print("reading: whichever row carries a residual at rounding level is the system the production\n"
          "circulation solves -- the first under the legacy order, the second under FLUXV_TE_WAKE=v3.\n"
          "The difference between the first two rows is what closing the trailing-edge gap with an\n"
          "implicit nascent row is worth ON THIS wake shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
