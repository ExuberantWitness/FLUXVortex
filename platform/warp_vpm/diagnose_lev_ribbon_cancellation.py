"""How much of a newborn LEV survives its own closure?  Pure geometry, no time stepping.

The question
------------
`near_le_cohorts_a15_700step.json` measured, on a 700-step run, that the NEWEST
leading-edge cohort induces le_normal_velocity_rms = 0.067 m/s while the whole
particle field induces 1.153 m/s: the newborn LEV reaches the first collocation
row at 5.8% of the field average.  The same file shows the cloud has diffused to
a mean distance of 0.84 chord, so there is no compact vortex core near the
leading edge -- which is why the mid-span mean flow is indistinguishable from
potential flow and no recirculation region appears (research note A13).

`CudaParticleField.add_open_filament_particles` states the mechanism: a
`add_connected_ribbon_particles` deposit closes the sheet into a loop whose
return leg sits one deposit-thickness away, "and hence cancels the leading-edge
sheet to 96%".

Hirato et al. (2019), J. Fluid Mech. 862: "Vortex-sheet representation of
leading-edge vortex shedding from finite wings", does NOT deposit an open
filament: the LEV is a chain of closed rings whose forward-most leg is tied to
A0.  What makes it work is that each ring encloses real area -- the ring is
placed on the circular arc through the leading edge and the two
leading-edge-adjacent surface points, at ONE-THIRD of the distance from the
leading edge to the last LEV ring.  The loop is closed but its two legs are
separated by the sheet's own extent.

So the control parameter is the deposit THICKNESS, and this script measures the
cancellation as a function of it, through the production deposit operators
themselves rather than through a reimplementation.

What it does
------------
Builds a straight 31-node leading-edge line of the real A16 span, deposits
strengths of the real pinned magnitude (-A0*scale = -0.0342 m^2/s per strip),
and evaluates the induced normal velocity at the first collocation row for
  * the open filament (one leg, monopole -- the reference),
  * the closed ribbon at thickness d in {1e-4 .. 0.05} c, along the surface
    normal and along the chord (the two directions that bracket Hirato's arc).

The reported number is  induced_normal / open_filament_normal, i.e. the fraction
of the LEV that survives its own closure.  It decides whether the one-third
rule can be a fix at all: if even a 5%-chord-thick loop cancels most of the
sheet, the closure has to go; if thickness recovers most of it, the fix is
geometric and cheap.

Usage
-----
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 \\
        python3 platform/warp_vpm/diagnose_lev_ribbon_cancellation.py --out /tmp/cancel.json
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
for sub in ("src", "platform", "platform/warp_vpm"):
    sys.path.insert(0, str(REPO_ROOT / sub))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")

    from pfield_torch_gpu import CudaParticleField

    device = torch.device(arguments.device)
    # Rojratsirikul et al. (2011): chord 68.8 mm, span 137.5 mm (AR = 2).  The SURVIVAL
    # RATIOS below are invariant under a uniform scaling of the geometry (sigma and
    # spacing scale with the chord, velocities as 1/length), but the absolute m/s are
    # not -- an earlier run used a 0.25 m chord and so reported velocities 3.63x small.
    chord, span = 0.0688, 0.1375
    nc, ns = 15, 30
    alpha = math.radians(arguments.alpha)
    chord_dir = torch.tensor([math.cos(alpha), 0.0, -math.sin(alpha)], dtype=torch.float64)
    normal_dir = torch.tensor([math.sin(alpha), 0.0, math.cos(alpha)], dtype=torch.float64)

    # Leading-edge line and the first collocation row, in the solver's own frame.
    y = torch.linspace(-0.5 * span, 0.5 * span, ns + 1, dtype=torch.float64, device=device)
    nodes = torch.zeros((ns + 1, 3), dtype=torch.float64, device=device)
    nodes[:, 1] = y
    dx = chord / nc
    first_collocation = (
        nodes[:-1] + nodes[1:]
    ) / 2.0 + 0.5 * dx * chord_dir.to(device)
    normal = normal_dir.to(device).expand(ns, 3)

    # The pinned newborn strength (research note A7b: -A0*scale = -0.0342 m^2/s).
    strength = torch.full((ns,), -0.0342, dtype=torch.float64, device=device)
    spacing = 0.018 * chord                      # DVM_TARGET_SPACING_CHORD of the runner
    #                                                 and of the rigid queue (not the
    #                                                 NativeV5MConfig default)
    radius = 0.04 * chord

    def induced_normal(field) -> np.ndarray:
        if field.n == 0:
            return np.zeros(ns)
        velocity = field.velocity_at_cuda(first_collocation)
        return (velocity * normal).sum(dim=1).detach().cpu().numpy()

    def fresh() -> "CudaParticleField":
        return CudaParticleField(capacity=400000, device=device)

    rows: list[dict[str, object]] = []

    # ---- reference: one leg only (monopole) --------------------------------
    reference = fresh()
    reference.add_open_filament_particles(
        nodes, strength, smoothing_radius=radius, target_spacing=spacing, birth_step=0
    )
    reference_normal = induced_normal(reference)
    reference_rms = float(np.sqrt(np.mean(reference_normal**2)))
    rows.append({
        "arm": "open_filament",
        "thickness_over_chord": 0.0,
        "particles": int(reference.n),
        "normal_rms_m_s": reference_rms,
        "ratio_to_open": 1.0,
        "normal_max_abs_m_s": float(np.max(np.abs(reference_normal))),
    })
    print(f"open filament (monopole reference): {int(reference.n)} particles, "
          f"first-row normal velocity rms {reference_rms:.5f} m/s")

    # ---- closed ribbon at increasing thickness ---------------------------
    thick_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 1.0 / 30.0, 5e-2]
    for direction, direction_name in ((normal_dir, "normal"), (chord_dir, "chord")):
        for thickness in thick_list:
            field = fresh()
            frontier = nodes + thickness * chord * direction.to(device).expand(ns + 1, 3)
            field.add_connected_ribbon_particles(
                nodes, frontier, strength,
                smoothing_radius=radius, target_spacing=spacing, birth_step=0,
                connector_source_strips=torch.clamp(
                    torch.arange(ns + 1, device=device, dtype=torch.int64) - 1,
                    min=0, max=ns - 1,
                ),
            )
            got = induced_normal(field)
            rms = float(np.sqrt(np.mean(got**2)))
            rows.append({
                "arm": f"closed_ribbon_{direction_name}",
                "thickness_over_chord": thickness,
                "particles": int(field.n),
                "normal_rms_m_s": rms,
                "ratio_to_open": rms / max(reference_rms, 1e-30),
                "normal_max_abs_m_s": float(np.max(np.abs(got))),
            })
            print(f"closed ribbon {direction_name:6s} d/c={thickness:<8.4g} "
                  f"{int(field.n):6d} particles   rms {rms:.5f} m/s   "
                  f"survives {100.0 * rms / max(reference_rms, 1e-30):5.1f}%")

    payload = {
        "alpha_deg": arguments.alpha,
        "nc": nc, "ns": ns, "chord_m": chord, "span_m": span,
        "pinned_strength_m2_s": float(strength[0]),
        "target_spacing_chord": spacing / chord,
        "smoothing_radius_chord": radius / chord,
        "rows": rows,
    }
    print()
    best = max((r for r in rows if r["arm"] != "open_filament"), key=lambda r: r["ratio_to_open"])
    print(f"best closed-ribbon arm: {best['arm']} d/c={best['thickness_over_chord']:.4g} "
          f"survives {100.0 * best['ratio_to_open']:.1f}% of the monopole")
    if arguments.out:
        Path(arguments.out).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
