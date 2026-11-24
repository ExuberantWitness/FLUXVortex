"""Visual cross-check of the MID-SPAN TIME-AVERAGED FLOW against Rojratsirikul et al. (2011) Fig. 7(b).

Why an image and not a scalar
-----------------------------
A scalar (zmax/c, Cn) cannot tell whether the model has the right FLOW PHENOMENON or only
the right number.  The paper's Figure 7 is the time-averaged velocity magnitude with
streamlines in the mid-span plane.  For the MEMBRANE wing at U_inf = 5 m/s:

    "the flow on the membrane wing is separated but the shear layer remains close to the
     surface" ... "the closed streamline pattern (recirculation region) is seen over the
     wing surface, while it is located further downstream for the lower freestream
     velocity, and even further away from the trailing edge for the rigid wing."

In the figure itself the RIGID plate at alpha = 16 deg (column a) has a large deep-blue
recirculation zone covering the chord; the MEMBRANE at the same angle (column b) does not:
its streamlines follow the cambered surface and a high-speed band (V/U_inf ~ 1.2-1.5) lies
over the upper surface.

What this does
--------------
Steps the case once and samples the velocity on a mid-span grid after every COMMITTED outer
step, through the runner's own `progress_callback`.  The velocity uses exactly the solver's
induction (freestream + bound rings + wake rings + particles -- the four terms of
`_external_velocity`).  It is averaged over the last `--average-last` steps and drawn in the
paper's layout: flow from RIGHT to LEFT, V/U_inf on the paper's 0..1.5 scale.

Besides the picture it reports the two features the paper's text names, counted on the grid
rather than eyeballed: streamwise velocity opposing the freestream above the wing, and a
slow core.

Usage:
    python3 platform/warp_vpm/visualize_roj_midspan_flow.py --case ROJ11-A16 --steps 200 \\
        --out artifacts/experiment/.../A16_midspan_flow.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ROJ11-A16")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--average-last", type=int, default=100,
                        help="average the flow over the last N committed steps")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-npz", default=None,
                        help="also store the sampled field, so the figure can be redrawn "
                             "without re-running the simulation (default: <out>.field.npz)")
    parser.add_argument("--from-npz", default=None,
                        help="skip the simulation and redraw from a saved field")
    arguments = parser.parse_args(argv)
    if arguments.from_npz:
        return draw_from_npz(arguments.from_npz, arguments.out)
    if not arguments.save_npz:
        arguments.save_npz = str(Path(arguments.out).with_suffix(".field.npz"))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")

    from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
    from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

    spec = ROJRATSIRIKUL2011_UNIFIED_CASES[arguments.case]
    chord, span, speed = float(spec.chord_m), float(spec.span_m), float(spec.freestream_m_s)
    runner = RojratsirikulCaseRunner(spec, device=arguments.device)
    runner.build()
    solver = runner.aerodynamic
    nc, ns = int(solver.settings.chordwise_panels), int(solver.settings.spanwise_panels)

    # Sampling grid: REGULAR in chord axes (needed by streamplot), mapped back to solver
    # axes for the induction evaluation.  Chord = (cos a, 0, -sin a), normal = (sin a, 0, cos a).
    _alpha = np.radians(float(spec.angle_deg))
    _cx = np.linspace(-0.25 * chord, 1.30 * chord, arguments.nx)
    _cz = np.linspace(-0.30 * chord, 0.60 * chord, arguments.nz)
    _gcx, _gcz = np.meshgrid(_cx, _cz)
    _sx = _gcx * np.cos(_alpha) + _gcz * np.sin(_alpha)
    _sz = -_gcx * np.sin(_alpha) + _gcz * np.cos(_alpha)
    points = torch.tensor(
        np.stack([_sx.ravel(), np.full(_sx.size, 0.5 * span), _sz.ravel()], axis=1),
        dtype=torch.float64, device=arguments.device,
    )

    start_average = max(arguments.steps - arguments.average_last + 1, 1)
    velocity_sum = torch.zeros_like(points)
    sections: list[np.ndarray] = []
    time_stars: list[float] = []
    arm = {
        "FLUXV_LE_PIN": os.environ.get("FLUXV_LE_PIN", "legacy"),
        "FLUXV_JOINT_SEPARATION": os.environ.get("FLUXV_JOINT_SEPARATION", "0"),
    }

    dt_star = float(spec.aerodynamic_dt_star)
    last_sampled = {"step": -1}

    def sample() -> None:
        state = runner.owner.aerodynamic.state
        geometry = solver.surface.evaluate(runner.owner.state, runner.owner.velocity)
        with torch.no_grad():
            velocity = (
                solver.v_inf[None, :].expand(points.shape[0], 3)
                + solver._ring_velocity(points, geometry.rings, state.gamma_bound, rough=False)
                + solver._ring_velocity(points, state.wake_rings, state.wake_gamma, rough=False)
            )
            if state.particle_field.n:
                velocity = velocity + state.particle_field.velocity_at_cuda(points)
        velocity_sum.add_(velocity.detach())
        sections.append(
            geometry.rings.reshape(nc, ns, 4, 3)[:, ns // 2, :, :].reshape(-1, 3).cpu().numpy()
        )

    def on_event(event: dict) -> None:
        # The callback fires for EVERY coupling phase of a step (aerodynamic_proposal,
        # coupling_residual, structural_substep, reference_tangent), several times per
        # outer step.  Sampling on each would weight the steps unevenly, and sampling on an
        # early one would read a trial state.  So: remember the step, and sample it once at
        # the START of the next step's first event -- by then `owner` holds the state that
        # was COMMITTED for that step.
        step = int(event["aero_step"])
        if last_sampled["step"] != step:
            previous = step - 1
            last_sampled["step"] = step
            if previous >= start_average:
                sample()
                time_stars.append(previous * dt_star)

    payload = runner.run(max_aero_steps=arguments.steps, progress_callback=on_event)
    if arguments.steps >= start_average and last_sampled["step"] != arguments.steps + 1:
        sample()
        time_stars.append(arguments.steps * dt_star)
    counted = len(sections)
    if counted == 0:
        raise SystemExit("no steps were averaged; check --steps / --average-last")

    mean_velocity = (velocity_sum / counted).cpu().numpy().reshape(arguments.nz, arguments.nx, 3)
    section = np.mean(sections, axis=0)
    magnitude = np.linalg.norm(mean_velocity, axis=2) / speed

    # The picture is drawn in CHORD-LINE axes, as the paper draws it.  Solver axes:
    # freestream along +x; the mesh is rotated about its leading edge with the trailing
    # edge going DOWN (rojratsirikul2011_q16.py rotate_q16_mesh_about_leading_edge), so
    #     chord  = (cos a, 0, -sin a)        upper-surface normal = (sin a, 0, cos a)
    # matplotlib's streamplot needs a REGULAR grid, so the flow is sampled on a regular grid
    # in chord axes from the outset: sample points are built by mapping that grid back to
    # solver axes, and the sampled velocity is rotated into chord components.
    alpha = np.radians(float(spec.angle_deg))
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    def to_solver(cx: np.ndarray, cz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cx * cos_a + cz * sin_a, -cx * sin_a + cz * cos_a

    cx_line = np.linspace(-0.25 * chord, 1.30 * chord, arguments.nx)      # chordwise, LE -> TE
    cz_line = np.linspace(-0.30 * chord, 0.60 * chord, arguments.nz)      # normal to the chord
    grid_cx, grid_cz = np.meshgrid(cx_line, cz_line)
    solver_x, solver_z = to_solver(grid_cx, grid_cz)
    section_cx = section[:, 0] * cos_a - section[:, 2] * sin_a
    section_cz = section[:, 0] * sin_a + section[:, 2] * cos_a
    u_chord = mean_velocity[:, :, 0] * cos_a - mean_velocity[:, :, 2] * sin_a      # + toward the TE
    w_chord = mean_velocity[:, :, 0] * sin_a + mean_velocity[:, :, 2] * cos_a      # normal to the chord

    order = np.argsort(section_cx)
    over_wing = (grid_cx > 0.0) & (grid_cx < chord) & (grid_cz > 0.0) & (grid_cz < 0.6 * chord)

    # Sign convention, stated once because an earlier version got it wrong and reported the
    # whole freestream as "reversed": u_chord > 0 points toward the TRAILING edge, which is
    # the direction the freestream travels.  Flow opposing the freestream is u_chord < 0.
    # In chord axes the paper's frame is x to the right = toward the TE, but the paper
    # draws the flow from right to left, so the picture is flipped for display only.
    reversed_flow = (u_chord < 0.0) & over_wing
    slow_core = (magnitude < 0.3) & over_wing
    # the high-speed band the paper shows over the upper surface
    band = over_wing & (grid_cz > 0.0) & (grid_cz < 0.35 * chord)
    band_peak = float(magnitude[band].max()) if band.any() else float("nan")

    if arguments.save_npz:
        Path(arguments.save_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            arguments.save_npz,
            u_chord=u_chord, w_chord=w_chord, magnitude=magnitude,
            cx_line=cx_line, cz_line=cz_line,
            section_cx=section_cx, section_cz=section_cz,
            counted=np.int64(counted), time_stars=np.array(time_stars),
            chord=np.float64(chord), speed=np.float64(speed),
            arm_json=np.array(__import__("json").dumps(arm)),
            case=np.array(arguments.case), steps=np.int64(arguments.steps),
            start_average=np.int64(start_average), angle_deg=np.float64(spec.angle_deg),
        )
        print(f"saved field to {arguments.save_npz}")
    return draw_from_npz(arguments.save_npz, arguments.out)


def _figure_column(arm: dict) -> str:
    """Which column of the paper's Figure 7 this run corresponds to: (a) rigid, (b) membrane U=5."""

    return "a" if "rigid" in str(arm.get("wing", "")).lower() else "b"


def draw_from_npz(npz_path: str, out: str) -> int:
    """Draw the mid-span figure from a saved field.  Pure plotting: no simulation."""

    data = np.load(npz_path, allow_pickle=False)
    u_chord, w_chord, magnitude = data["u_chord"], data["w_chord"], data["magnitude"]
    cx_line, cz_line, chord = data["cx_line"], data["cz_line"], float(data["chord"])
    section_cx, section_cz = data["section_cx"], data["section_cz"]
    counted, time_stars = int(data["counted"]), data["time_stars"]
    speed, start_average, steps = float(data["speed"]), int(data["start_average"]), int(data["steps"])
    case, alpha_deg = str(data["case"]), float(data["angle_deg"])
    arm = __import__("json").loads(str(data["arm_json"]))

    grid_cx, grid_cz = np.meshgrid(cx_line, cz_line)
    over_wing = (grid_cx > 0.0) & (grid_cx < chord) & (grid_cz > 0.0) & (grid_cz < 0.6 * chord)
    reversed_flow = (u_chord < 0.0) & over_wing
    slow_core = (magnitude < 0.3) & over_wing
    band = over_wing & (grid_cz > 0.0) & (grid_cz < 0.35 * chord)
    band_peak = float(magnitude[band].max()) if band.any() else float("nan")
    order = np.argsort(section_cx)

    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    image = axis.pcolormesh((-grid_cx / chord)[:, ::-1], grid_cz / chord, magnitude[:, ::-1],
                            cmap="jet", vmin=0.0, vmax=1.5, shading="gouraud")
    axis.streamplot((-cx_line / chord)[::-1], cz_line / chord,
                    (-u_chord)[:, ::-1], w_chord[:, ::-1],
                    color="k", density=1.6, linewidth=0.55, arrowsize=0.7)
    axis.plot(-section_cx[order] / chord, section_cz[order] / chord, color="k", linewidth=2.2)
    axis.set_xlim(-cx_line[-1] / chord, -cx_line[0] / chord)
    axis.set_ylim(cz_line[0] / chord, cz_line[-1] / chord)
    axis.set_aspect("equal")
    axis.set_xlabel("x/c from the leading edge, increasing to the right   (flow from right to left, as in the paper)")
    axis.set_ylabel("z/c")
    axis.set_title(
        f"{case}, alpha={alpha_deg:g} deg, U={speed:g} m/s: time-averaged |V|/U_inf "
        f"and streamlines, mid-span plane\n"
        f"model arm {arm}; averaged over steps {start_average}-{steps} "
        f"(t* {time_stars[0]:.2f}..{time_stars[-1]:.2f})  --  compare Rojratsirikul 2011 Fig. 7({_figure_column(arm)})",
        fontsize=9,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046)
    colorbar.set_label("V/U_inf")
    figure.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    plt.close(figure)

    total = int(over_wing.sum())
    print(f"case {case}  arm {arm}")
    print(f"averaged over {counted} committed steps, t* {time_stars[0]:.2f} .. {time_stars[-1]:.2f}")
    print(f"max |V|/U_inf over the grid            : {magnitude.max():.3f}")
    print(f"peak |V|/U_inf in the band 0..0.35c above the chord : {band_peak:.3f}"
          + ("     paper Fig. 7(a), rigid alpha=16: a deep LOW-speed zone (< 0.3) covers the chord"
             if _figure_column(arm) == "a" else
             "     paper Fig. 7(b), membrane alpha=16: high-speed band ~1.2-1.5"))
    print(f"above-wing cells with flow AGAINST the freestream : {int(reversed_flow.sum())} of {total}"
          f"  ({100.0 * reversed_flow.sum() / max(total, 1):.1f}%)")
    column = _figure_column(arm)
    if column == "a":
        print("    paper Fig. 7(a), RIGID alpha=16: a large closed recirculation zone over the chord, deep blue")
    else:
        print("    paper Fig. 7(b), MEMBRANE alpha=16: streamlines follow the surface, no large closed recirculation")
    print(f"above-wing cells with |V|/U_inf < 0.3              : {int(slow_core.sum())} of {total}"
          f"  ({100.0 * slow_core.sum() / max(total, 1):.1f}%)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
