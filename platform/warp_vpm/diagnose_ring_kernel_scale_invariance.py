"""Is the ring-velocity kernel scale invariant?  Pure CPU, no solver, no simulation.

The question
------------
`native_ring_velocity_expanded` -- the function `native_aic` calls to build the bound
influence matrix -- evaluates the straight-filament Biot-Savart law with one
regularisation (kernels_ring_velocity.py):

    base = (a x b) / (|a x b|^2 + 1e-9) * [(a - b) . (a_hat - b_hat)] / (4 pi)

`|a x b|` has units of m^2, so the frozen `1e-9` has units of m^4 and the exact law is
recovered only where |a x b|^2 >> 1e-9.  For a point at perpendicular distance h from a
filament of length L, |a x b| = L h.  On the Rojratsirikul et al. (2011) wing the chord
is 68.8 mm: a panel is 4.6 mm, so inside the lattice L h is about (4.6 mm)^2 = 2e-5 m^2
and its square 4e-10 -- BELOW the constant.  Biot-Savart is invariant under a uniform
scaling of the geometry (velocity scales as circulation / length); a kernel with a
fixed-length regularisation is not, and how far it departs is the size of the effect.

What it does
------------
Takes one ring of the rigid-plate lattice (15 x 30 panels on a 68.8 mm x 137.5 mm
planform, uniform panels, ring spanning one panel as the solver lays them out) and
evaluates, on the CPU, the production function and the unregularised textbook law at

  * the ring's own centre and the panel collocation point (the self terms of the AIC),
  * the centres of the neighbours one and two panels away in each direction,
  * a point four panels away and the centre of the farthest ring,

then repeats everything on the same ring scaled by 10 and 100.  Reported: the damping
factor kernel/exact of the velocity component normal to the ring's plane, and the
|a x b|^2 of the closest filament.

A factor of 1.00 everywhere means the regularisation is inert at this scale and the
hypothesis is dead.  Anything below 1 on the self and nearest-neighbour rows says the
lattice's near field is being suppressed by a constant that was frozen at another scale.

Usage
-----
    python3 platform/warp_vpm/diagnose_ring_kernel_scale_invariance.py --out /tmp/ring_scale.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CHORD_M, SPAN_M = 0.0688, 0.1375      # Rojratsirikul et al. (2011), AR = 2
NC, NS = 15, 30                       # the production aero grid
CORE_FRACTION = 1.0e-6                # what native_aic passes


def textbook_biot_savart(points: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """Straight-filament Biot-Savart with no regularisation at all.

    v = 1/(4 pi) * sum_segments (a x b)/|a x b|^2 * [(a - b) . (a_hat - b_hat)]
    for unit circulation; a and b run from the filament's ends to the field point.
    """

    starts = ring
    ends = torch.roll(ring, shifts=-1, dims=0)
    total = torch.zeros((points.shape[0], 3), dtype=torch.float64)
    smallest_cross_sq = torch.full((points.shape[0], 1), float("inf"), dtype=torch.float64)
    for start, end in zip(starts, ends):
        a = points - start[None, :]
        b = points - end[None, :]
        cross = torch.linalg.cross(a, b, dim=1)
        cross_sq = torch.sum(cross * cross, dim=1, keepdim=True)
        unit_difference = (
            a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
            - b / torch.linalg.vector_norm(b, dim=1, keepdim=True)
        )
        scalar = torch.sum((a - b) * unit_difference, dim=1, keepdim=True)
        total = total + cross / cross_sq * scalar / (4.0 * math.pi)
        smallest_cross_sq = torch.minimum(smallest_cross_sq, cross_sq)
    return total, smallest_cross_sq[:, 0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--mid-chord-row", type=int, default=7,
                        help="1-based chordwise row of the test ring")
    arguments = parser.parse_args(argv)
    torch.manual_seed(0)

    from fluxvortex.warp_fsi import kernels_ring_velocity as production

    dx, ds = CHORD_M / NC, SPAN_M / NS
    row, strip = arguments.mid_chord_row - 1, NS // 2
    # The solver's layout: corners front-left, front-right, back-right, back-left, the
    # bound sheet spanning [0.25 dx, c + 0.25 dx]; the collocation point is the panel
    # centre (rojratsirikul2011_q16 / rigid_flux_v5m_native conventions).
    x_front = 0.25 * dx + row * dx
    y_left = -0.5 * SPAN_M + strip * ds
    ring = torch.tensor(
        [
            [x_front, y_left, 0.0],
            [x_front + dx, y_left, 0.0],
            [x_front + dx, y_left + ds, 0.0],
            [x_front, y_left + ds, 0.0],
        ],
        dtype=torch.float64,
    )
    centre = ring.mean(dim=0)
    collocation = torch.tensor([(row + 0.5) * dx, y_left + 0.5 * ds, 0.0], dtype=torch.float64)
    targets = {
        "ring centre (self)": centre,
        "panel collocation point (self)": collocation,
        "neighbour +1 panel chordwise": centre + torch.tensor([dx, 0.0, 0.0]),
        "neighbour -1 panel chordwise": centre - torch.tensor([dx, 0.0, 0.0]),
        "neighbour +1 strip": centre + torch.tensor([0.0, ds, 0.0]),
        "neighbour +2 strips": centre + torch.tensor([0.0, 2.0 * ds, 0.0]),
        "neighbour +4 strips": centre + torch.tensor([0.0, 4.0 * ds, 0.0]),
        "far corner ring (7 panels, 15 strips)": centre
        + torch.tensor([-3.0 * dx, 7.0 * ds, 0.0]),
    }

    print(f"panel {dx * 1e3:.3f} mm x {ds * 1e3:.3f} mm; frozen eps = {production._EPS_V:g} m^4")
    rows: list[dict[str, object]] = []
    for scale in (1.0, 10.0, 100.0):
        points = torch.stack(list(targets.values())) * scale
        rings = (ring * scale)[None, :, :]
        kernel = production.native_ring_velocity_expanded(
            points, rings, core_fraction=CORE_FRACTION, reference_length=(1.0 / NC) * scale
        )[:, 0, :]
        exact, smallest = textbook_biot_savart(points, rings[0])
        print(f"\n=== geometry scaled x{scale:g}: chord {CHORD_M * scale:.4g} m, panel {dx * scale * 1e3:.4g} mm ===")
        print(f"{'target':38s} {'|a x b|^2 min':>14} {'kernel v_z':>13} {'exact v_z':>13} {'kernel/exact':>13}")
        for index, (name, _) in enumerate(targets.items()):
            ratio = float(kernel[index, 2] / exact[index, 2])
            rows.append({
                "scale": scale,
                "target": name,
                "min_cross_sq_m4": float(smallest[index]),
                "kernel_vz": float(kernel[index, 2]),
                "exact_vz": float(exact[index, 2]),
                "kernel_over_exact": ratio,
            })
            print(f"{name:38s} {float(smallest[index]):14.4e} {float(kernel[index, 2]):13.6e} "
                  f"{float(exact[index, 2]):13.6e} {ratio:13.4f}")

    self_rows = [r for r in rows if r["target"].startswith("panel collocation")]
    print("\nself term (panel collocation point), kernel/exact by scale: "
          + "  ".join(f"x{r['scale']:g}: {r['kernel_over_exact']:.4f}" for r in self_rows))
    if arguments.out:
        Path(arguments.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
