#!/usr/bin/env python3
"""Where is the leading-edge sheet, and does its induction raise or lower the leading-edge loading?

Reads a snapshot of the research runner (CPU only, no solver) and answers the one question the
H case left open (note A25, section 4): the attached suction parameter grows from 0.27 to 1.0 as
the sheet grows, which means the sheet's induction at its own leading-edge row acts like a
stronger freestream instead of a weaker one.  A sheet that detaches must do the opposite — that is
what makes the suction parameter fall below the critical value and the shedding stop.

Reported per strip and in aggregate:

  * the sheet's distance from the chord plane, along the surface normal of its strip: positive is
    the suction side.  A sheet on the pressure side, or one that stays inside the boundary layer
    of the leading edge, cannot lower the loading;
  * the normal velocity the sheet induces at the leading-edge collocation row, next to the bound
    sheet's and the freestream's, so the signs can be compared directly;
  * the ratio of the sheet's induction to the freestream's normal component, per strip.

    python3 platform/warp_vpm/diagnose_lev_sheet_geometry.py --snapshot DIR/snapshot_00204.npz
"""

from __future__ import annotations

import argparse

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--strips", default=None, help="comma-separated strip indices (default: a spread)")
    arguments = parser.parse_args(argv)
    data = np.load(arguments.snapshot)
    ns = int(data["spanwise_panels"])
    nc = int(data["chordwise_panels"])
    if "lev_rings" not in data:
        print("this snapshot carries no leading-edge chain")
        return 1
    rings = data["lev_rings"]              # (rows, ns, 4, 3)
    gamma = data["lev_gamma"]              # (rows, ns)
    count = data["lev_count"]              # (ns,)
    collocation = data["collocation"].reshape(nc, ns, 3)
    normals = data["normals"].reshape(nc, ns, 3)
    bound_rings = data["bound_rings"]
    bound_gamma = data["gamma_load"]
    leading = data["leading_edge"]
    trailing = data["trailing_edge"]
    speed = float(data["speed_m_s"])
    axes = data["pose_axes"]
    pseudovortex = data.get("pseudovortex_rings")
    pseudovortex_gamma = data.get("pseudovortex_gamma")
    lesp = data["lesp_strip"]

    # the leading-edge collocation row: where the suction parameter is measured
    le_points = collocation[0]
    le_normals = normals[0]
    # the freestream in the body frame of the snapshot: the runner stores the pose, not v_inf
    freestream_body = axes.T @ np.array([speed, 0.0, 0.0])
    freestream_normal = float(np.mean(freestream_body * le_normals))     # (ns,)

    # the chord plane of every strip, in the body frame: leading edge, spanwise, chord
    strip_chord = 0.5 * ((trailing[:-1] + trailing[1:]) - (leading[:-1] + leading[1:]))
    chord_length = np.linalg.norm(strip_chord, axis=1, keepdims=True)
    chord_hat = strip_chord / chord_length
    strip_leading = 0.5 * (leading[:-1] + leading[1:])

    def ring_induced(points: np.ndarray, rings: np.ndarray, strengths: np.ndarray) -> np.ndarray:
        """Velocity at `points` of every ring, summed with its strength — the production kernel."""

        import torch
        from fluxvortex.warp_fsi import kernels_ring_velocity as kernel

        if rings.shape[0] == 0:
            return np.zeros_like(points)
        unit = kernel.native_ring_velocity_expanded(
            torch.as_tensor(points, dtype=torch.float64),
            torch.as_tensor(rings, dtype=torch.float64),
            core_fraction=0.0,
            reference_length=0.0,
        )
        strength = torch.as_tensor(strengths, dtype=torch.float64)
        total = torch.sum(unit * strength[None, :, None], dim=1)
        return total.numpy()

    strips = ([int(value) for value in arguments.strips.split(",")] if arguments.strips
              else sorted(set([0, ns // 4, ns // 2, 3 * ns // 4, ns - 1])))

    print(f"{arguments.snapshot}")
    print(f"  chord {float(data['chord_m']):g} m, span {float(data['speed_m_s']):g} m/s, grid {nc} x {ns}, "
          f"step {int(data['step'])}, t* {float(data['time']) * speed / float(data['chord_m']):.3f}")
    rows_of = np.arange(gamma.shape[0])[:, None]
    live = rows_of < count[None, :]                    # (rows, ns): the rows that carry a ring
    sheet_circulation = float(gamma[live].sum())
    print(f"  strips shedding: {int((count > 0).sum())} of {ns}; "
          f"sheet circulation {sheet_circulation:+.4f} m^2/s "
          f"({sheet_circulation / (speed * float(data['chord_m'])):+.2f} U c)")
    print(f"  freestream normal component at the leading-edge row {freestream_normal:+.4f} m/s "
          f"(the sign that a larger attachment would have to overcome)")
    print()
    header = ("strip  rows  nearest/m  farthest/m  above?  Vn_sheet  Vn_pseudo  Vn_totLEV  "
              "Vn_bound  Vn_inf  totLEV/inf  sheet/inf  lesp")
    print(header)
    for strip in strips:
        rows = int(count[strip])
        if rows == 0:
            print(f"{strip:5d} {rows:5d}  -")
            continue
        # distance of every ring centre from the chord plane of its strip, along +n
        plane_point = strip_leading[strip]
        centres = rings[:rows, strip].mean(axis=1)                       # (rows, 3)
        offset = centres - plane_point[None, :]
        height = offset @ le_normals[strip]                              # signed distance along +n
        along = offset @ chord_hat[strip]
        point = le_points[strip][None, :]
        v_sheet = ring_induced(point, rings[:rows, strip], gamma[:rows, strip])[0]
        v_bound = ring_induced(point, bound_rings, bound_gamma)[0]
        if pseudovortex is not None:
            v_pseudo = ring_induced(point, pseudovortex[strip][None], pseudovortex_gamma[strip][None])[0]
        else:
            v_pseudo = np.zeros(3)
        normal = le_normals[strip]
        vn_sheet = float(v_sheet @ normal)
        vn_bound = float(v_bound @ normal)
        vn_pseudo = float(v_pseudo @ normal)
        vn_tot = vn_sheet + vn_pseudo
        vn_inf = float(freestream_body @ normal)
        fraction = vn_sheet / vn_inf if abs(vn_inf) > 1e-12 else float("nan")
        fraction_tot = vn_tot / vn_inf if abs(vn_inf) > 1e-12 else float("nan")
        print(f"{strip:5d} {rows:5d}  {height.max():.4f}  {height.min():+.4f}  "
              f"{'suction' if np.median(height) > 0 else 'PRESSURE':8s} "
              f"{vn_sheet:+8.4f} {vn_pseudo:+10.4f} {vn_tot:+10.4f} {vn_bound:+9.4f} {vn_inf:+7.4f} "
              f"{fraction_tot:+10.2f} {fraction:+9.2f} {lesp[strip]:+7.3f}")
        print(f"       ring height along +n (rows 0..{min(rows, 8) - 1}): "
              + " ".join(f"{value:+.4f}" for value in height[:min(rows, 8)]))
        print(f"       ring distance along the chord  : "
              + " ".join(f"{value:+.4f}" for value in along[:min(rows, 8)]))
    # aggregate over the shedding strips
    shedding = np.nonzero(count > 0)[0]
    if shedding.size:
        heights = []
        fractions_bare = []
        fractions_tot = []
        for strip in shedding:
            rows = int(count[strip])
            centres = rings[:rows, strip].mean(axis=1)
            heights.append((centres - strip_leading[strip][None, :]) @ le_normals[strip])
            vn_sheet = float(ring_induced(le_points[strip][None, :], rings[:rows, strip],
                                          gamma[:rows, strip])[0] @ le_normals[strip])
            vn_pseudo = (
                float(ring_induced(le_points[strip][None, :], pseudovortex[strip][None],
                                   pseudovortex_gamma[strip][None])[0] @ le_normals[strip])
                if pseudovortex is not None else 0.0
            )
            vn_inf = float(freestream_body @ le_normals[strip])
            if abs(vn_inf) > 1e-12:
                fractions_bare.append(vn_sheet / vn_inf)
                fractions_tot.append((vn_sheet + vn_pseudo) / vn_inf)
        heights = np.concatenate(heights)
        print()
        print(f"  over the {shedding.size} shedding strips: ring height along +n "
              f"median {np.median(heights):+.4f} m, {100.0 * np.mean(heights > 0):.0f} % above chord; "
              f"PHYSICAL (sheet+pseudo)/inf: median {np.median(fractions_tot):+.2f}, "
              f"{100.0 * np.mean(np.array(fractions_tot) < 0):.0f} % relieving (<0); "
              f"[bare sheet/inf without pseudo: median {np.median(fractions_bare):+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
