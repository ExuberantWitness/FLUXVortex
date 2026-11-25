"""Steady vortex lattice of the rigid plate with the PRODUCTION ring kernel, on the CPU.

Purpose
-------
`diagnose_ring_kernel_scale_invariance.py` showed that the frozen regularisation of the
ring kernel, `|a x b|^2 -> |a x b|^2 + 1e-9` (units m^4), is not inert at the
Rojratsirikul scale (chord 68.8 mm, panels 4.6 mm): the self term comes out at 7% of the
Biot-Savart value and the nearest-neighbour terms change sign.  That is a statement
about kernel entries.  This script asks what it does to the SOLUTION.

What it solves
--------------
The classical steady vortex lattice: the rigid plate flat in z = 0, freestream at angle
alpha, 15 x 30 bound rings laid out as the solver lays them (corners front-left,
front-right, back-right, back-left; the sheet spanning [0.25 dx, c + 0.25 dx]; the
collocation point at the ring centre, i.e. on the three-quarter-chord line, which is
where the solver puts it), plus a straight wake of `--wake-rows` rows of
length U dt downstream of the trailing edge.  Each wake row's rings carry the
trailing-edge-row unknown of their strip -- the steady form of the unsteady shedding the
solver performs (`gamma_tev` = last bound row).  No-penetration at the 450 collocation
points closes the system.  No leading-edge pin, no particles, no separation model.

Every influence entry is evaluated by `native_ring_velocity_expanded`, i.e. the function
`native_aic` calls.  The regularisation constant is overridden on the imported module
object IN THIS PROCESS ONLY; no file is edited.

Reported for each value of the constant: the Kutta-Joukowski coefficient
2 * sum(Gamma_TE) * ds / (U S) and the spanwise shape of the trailing-edge row (the
section circulation of this lattice), against the elliptic thin-wing value.

Reading it
----------
If the coefficient at 1e-9 reproduces the production rigid-plate result and falls to
the thin-wing value as the constant goes to zero, the excess normal force is that
constant.  If it does not move, the constant is not the cause and the proxy says so.

Usage
-----
    CUDA_VISIBLE_DEVICES="" FLUXV_GPU_ONLY=0 PFIELD_DEVICE=cpu PYTHONPATH=src \\
        python3 platform/warp_vpm/diagnose_ring_lattice_steady_cpu.py --out /tmp/lattice.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CHORD_M, SPAN_M = 0.0688, 0.1375      # Rojratsirikul et al. (2011), AR = 2
NC, NS = 15, 30                       # the production aero grid
SPEED = 5.0
CORE_FRACTION = 1.0e-6                # what native_aic passes


def _influence_z(
    points: torch.Tensor, rings: torch.Tensor, chunk: int, nc: int
) -> torch.Tensor:
    """Normal (z) velocity at `points` from unit-strength `rings`, one chunk at a time."""

    out = torch.empty((points.shape[0], rings.shape[0]), dtype=torch.float64)
    for start in range(0, rings.shape[0], chunk):
        block = rings[start : start + chunk]
        out[:, start : start + block.shape[0]] = production.native_ring_velocity_expanded(
            points, block, core_fraction=CORE_FRACTION, reference_length=1.0 / nc
        )[:, :, 2]
    return out


def build(
    chord: float, span: float, nc: int, ns: int, wake_rows: int, dt_star: float,
    wake_tilt_deg: float = 0.0, wake_layout: str = "connected",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bound rings, straight wake rows, collocation points.

    `wake_tilt_deg` inclines the whole wake about the trailing edge (negative = the wake
    leaves downwards).  A real wake leaves the trailing edge along the local flow, i.e.
    a few degrees down, and rolls up at the tips; this is the cheapest way to measure how
    much the solution cares.

    `wake_layout` names where the rows sit relative to the sheet's aft edge, all straight:
      connected  row k spans [k, k+1] U*dt; the first row's front leg is ON the aft edge
                 and cancels the bound trailing-edge leg (the Katz-Plotkin implicit wake)
      gap1       every row one step further downstream ([k+1, k+2]); the aft edge is left
                 with an uncancelled bound leg -- what the production solver has at the
                 moment it solves, because it convects the wake first and inserts the new
                 row afterwards
      overlap    rows [0, 1] and [0, 2], then [k, k+1]; the production solver's legacy
                 post-insertion topology, in which the previous newest row is re-riveted
                 to the aft edge and overlaps the new one
    """
    dx, ds = chord / nc, span / ns
    row_dt = dt_star * chord                        # U*dt: one wake row per step
    x_edges = 0.25 * dx + nc * dx                   # trailing edge of the bound sheet
    y_edges = torch.linspace(-0.5 * span, 0.5 * span, ns + 1, dtype=torch.float64)

    bound, collocation = [], []
    for j in range(nc):
        x_front = 0.25 * dx + j * dx
        for i in range(ns):
            y0, y1 = float(y_edges[i]), float(y_edges[i + 1])
            bound.append([[x_front, y0, 0.0], [x_front + dx, y0, 0.0],
                          [x_front + dx, y1, 0.0], [x_front, y1, 0.0]])
            # The solver's collocation points sit on the three-quarter-chord lines
            # (q16_flux_v5m_author_loads.py: "(value + 0.75) / nc"), which is the ring's
            # centre -- the Katz-Plotkin 1/4 - 3/4 rule.  An earlier version used the
            # panel centre and was 40% low with no regularisation at all.
            collocation.append([x_front + 0.5 * dx, 0.5 * (y0 + y1), 0.0])
    tilt = math.radians(wake_tilt_deg)
    along, upward = math.cos(tilt), math.sin(tilt)

    def downstream(distance: float, y: float) -> list[float]:
        return [x_edges + distance * along, y, distance * upward]

    if wake_layout == "connected":
        row_spans = [(k * row_dt, (k + 1) * row_dt) for k in range(wake_rows)]
    elif wake_layout == "gap1":
        row_spans = [((k + 1) * row_dt, (k + 2) * row_dt) for k in range(wake_rows)]
    elif wake_layout == "overlap":
        row_spans = [(0.0, row_dt), (0.0, 2.0 * row_dt)] + [
            (k * row_dt, (k + 1) * row_dt) for k in range(2, wake_rows)
        ]
    elif wake_layout == "legacy_solve":
        # the same topology one convection step later, i.e. what the legacy path presents
        # to the linear solve: a gap [0, 1] at the aft edge and rows 0 / 1 overlapping
        row_spans = [(row_dt, 2.0 * row_dt), (row_dt, 3.0 * row_dt)] + [
            (k * row_dt, (k + 1) * row_dt) for k in range(3, wake_rows + 1)
        ]
    else:
        raise ValueError(f"unknown wake layout {wake_layout!r}")
    wake = []
    for s0, s1 in row_spans:
        for i in range(ns):
            y0, y1 = float(y_edges[i]), float(y_edges[i + 1])
            # the solver's wake ring: (rear right, rear left, downstream left, downstream right)
            wake.append([downstream(s0, y1), downstream(s0, y0),
                         downstream(s1, y0), downstream(s1, y1)])
    return (
        torch.tensor(bound, dtype=torch.float64),
        torch.tensor(wake, dtype=torch.float64),
        torch.tensor(collocation, dtype=torch.float64),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="5,10,16")
    parser.add_argument("--wake-rows", type=int, default=300,
                        help="the solver keeps wake_max_rows = 300 rows (= 3 chords)")
    parser.add_argument("--eps", default="1e-9,1e-10,1e-11,1e-12,1e-30")
    parser.add_argument("--chunk", type=int, default=400)
    parser.add_argument("--chord", type=float, default=CHORD_M)
    parser.add_argument("--span", type=float, default=SPAN_M)
    parser.add_argument("--nc", type=int, default=NC)
    parser.add_argument("--ns", type=int, default=NS)
    parser.add_argument("--dt-star", type=float, default=0.01,
                        help="time step in chords of travel; it sets the wake row length U*dt, "
                             "i.e. the length of the wake's streamwise filaments")
    parser.add_argument("--wake-layout", choices=["connected", "gap1", "overlap", "legacy_solve"], default="connected")
    parser.add_argument("--wake-tilt-deg", type=float, default=0.0,
                        help="incline the straight wake about the trailing edge (negative = down)")
    parser.add_argument("--quiet", action="store_true",
                        help="one line per value of the constant, no table header")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args(argv)

    global production
    from fluxvortex.warp_fsi import kernels_ring_velocity as production

    frozen = production._EPS_V
    alphas = [float(v) for v in arguments.alphas.split(",")]
    epsilons = [float(v) for v in arguments.eps.split(",")]
    chord, span, nc, ns = arguments.chord, arguments.span, arguments.nc, arguments.ns
    dx, ds = chord / nc, span / ns
    aspect_ratio = span / chord

    print(f"chord {chord} m, span {span} m, {nc} x {ns} panels of {dx * 1e3:.3f} mm; "
          f"wake tilt {arguments.wake_tilt_deg:+g} deg; "
          f"wake {arguments.wake_rows} rows of {0.01 * arguments.chord * 1e3:.4f} mm; "
          f"frozen constant in the file: {frozen:g}")
    results: list[dict[str, object]] = []
    for alpha_deg in alphas:
        alpha = math.radians(alpha_deg)
        bound, wake, collocation = build(
            chord, span, nc, ns, arguments.wake_rows, arguments.dt_star,
            arguments.wake_tilt_deg, arguments.wake_layout,
        )
        rhs = -SPEED * math.sin(alpha) * torch.ones(collocation.shape[0], dtype=torch.float64)
        # Prandtl's lifting line assumes the chordwise loading and the downwash to be
        # uniform over the chord, which fails below AR ~ 4: at AR = 2 it is about 25% high.
        # The low-aspect-ratio reference is Helmbold's lifting-surface formula,
        #   CL_alpha = 2 pi AR / (2 + sqrt(AR^2 + 4)).
        # Cn is referred to Helmbold; the lifting-line value is printed only because notes
        # A18-A20 used it as the yardstick.
        lifting_line = 2.0 * math.pi * aspect_ratio / (aspect_ratio + 2.0) * math.sin(alpha)
        theory = (
            2.0 * math.pi * aspect_ratio / (2.0 + math.sqrt(aspect_ratio**2 + 4.0)) * math.sin(alpha)
        )
        print(f"\n=== alpha = {alpha_deg:g} deg   AR = {aspect_ratio:g};  wake layout {arguments.wake_layout};  "
              f"Helmbold lifting surface = {theory:.4f} (CL_alpha {theory / alpha:.3f} /rad);  "
              f"Prandtl lifting line = {lifting_line:.4f} (high at low AR) ===")
        print(f"{'eps (m^4)':>10} {'Cn by K-J':>10} {'Cn/Helmbold':>11} {'tip/centre':>11} "
              f"{'mean/centre':>12} {'outer-20%':>10} {'gamma_TE centre':>16}   {'|G1|/G_TE':>9}")
        for eps in epsilons:
            # The constant is set on the module BEFORE any entry is evaluated, for every
            # value: nothing is carried over from the previous value.
            production._EPS_V = eps
            bound_eps = _influence_z(collocation, bound, arguments.chunk, nc)
            wake_eps = _influence_z(collocation, wake, arguments.chunk, nc).reshape(
                collocation.shape[0], arguments.wake_rows, ns
            ).sum(dim=1)
            system = bound_eps.clone()
            system[:, (nc - 1) * ns :] += wake_eps
            gamma = torch.linalg.solve(system, rhs)
            te_row = gamma[(nc - 1) * ns :].numpy()
            if te_row.mean() < 0.0:                 # ring traversal orientation only
                gamma, te_row = -gamma, -te_row
            coefficient = 2.0 * float(te_row.sum()) * ds / (SPEED * chord * span)
            y = (np.arange(ns) + 0.5) / ns
            outer = (y < 0.1) | (y > 0.9)
            centre = te_row[ns // 2]
            record = {
                "alpha_deg": alpha_deg, "eps_m4": eps, "cn_kutta_joukowski": coefficient,
                "cn_theory_elliptic": theory, "cn_over_theory": coefficient / theory,
                "tip_over_centre": float(te_row[0] / centre),
                "mean_over_centre": float(te_row.mean() / centre),
                "outer_20pct_share": float(te_row[outer].sum() / te_row.sum()),
                "gamma_te_centre_m2_s": float(centre),
                "gamma_row1_mean_m2_s": float(gamma[:ns].mean()),
                "chord_m": chord, "span_m": span, "nc": nc, "ns": ns,
                "te_row": [float(v) for v in te_row],
            }
            results.append(record)
            print(f"{eps:10.0e} {coefficient:10.4f} {coefficient / theory:10.3f} "
                  f"{record['tip_over_centre']:11.3f} {record['mean_over_centre']:12.3f} "
                  f"{record['outer_20pct_share']:10.3f} {centre:16.5f} "
                  f"| Gamma_1/Gamma_TE {abs(record['gamma_row1_mean_m2_s']) / centre:8.3f}")
    production._EPS_V = frozen
    print("\nelliptic reference shape: tip strip / centre 0.256, span mean / centre 0.787, "
          "outer-20% share 0.105")
    if arguments.out:
        Path(arguments.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
