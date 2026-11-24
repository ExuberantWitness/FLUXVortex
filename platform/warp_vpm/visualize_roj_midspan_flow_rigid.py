"""Mid-span time-averaged flow for the RIGID FLAT PLATE, against Rojratsirikul et al. (2011) Fig. 7(a).

This is the discriminating half of the Figure 7 comparison.  At alpha = 16 deg the paper's
left column -- the RIGID plate at U_inf = 5 m/s -- shows a large deep-blue zone over the
whole chord with a CLOSED recirculation region, while the membrane at the same angle
(column b) shows none: "the closed streamline pattern (recirculation region) is seen over
the wing surface, while it is ... even further away from the trailing edge for the rigid
wing".

The membrane run came out fully attached (no cell with flow opposing the freestream, no
cell slower than 0.3 U_inf).  That is consistent with the paper for the membrane, but it
only means something if the SAME method can produce a recirculation at all.  If the rigid
plate also comes out attached, then "attached" is a property of the method, not a
prediction about the membrane, and the agreement on the membrane side is empty.

What is kept identical to the membrane run, so the two columns differ only in rigidity:
  * aero grid (15 x 30), lesp_crit, wake/particle settings  -- the frozen rigid queue
  * the half-cosine startup ramp over t* in [0,1] (--ramp-steps 100)
  * the sampling grid, the four-term induction and the averaging window
  * the LE pin arm (FLUXV_LE_PIN), read from the environment like every other run

The rigid wing uses the in-tree RigidNativeV5MSolver, which is a DIFFERENT class from the
membrane solver but shares the aerodynamic core.  That is a stated difference of this
comparison, not an accident.

Usage:
    python3 platform/warp_vpm/visualize_roj_midspan_flow_rigid.py --alpha 16 --steps 200 \\
        --out artifacts/experiment/.../RIGID_A16_midspan_flow.png
"""

from __future__ import annotations

import argparse
import math
import os
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
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ramp-steps", type=int, default=100,
                        help="half-cosine startup ramp over this many steps (the membrane "
                             "run ramps over t* in [0,1], i.e. 100 steps at dt*=0.01)")
    parser.add_argument("--average-last", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-npz", default=None)
    arguments = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")

    import queue_roj_rigid_fig9_12_13_15 as frozen
    from diagnose_roj_rigid_loads import build
    from visualize_roj_midspan_flow import draw_from_npz

    device = arguments.device
    speed, chord, span = 5.0, frozen.CHORD_M, frozen.SPAN_M
    nc, ns = frozen.AERO_GRID
    solver, stepper, owner, kin = build(arguments.alpha, observed=False)
    solver.v_inf = solver.v_inf.to(device)
    dt = float(solver.settings.aerodynamic_dt)

    # Sampling grid, identical to the membrane script: regular in chord axes, mapped back
    # to solver axes.  The rigid plate is pitched about its quarter chord by the prescribed
    # kinematics; measured on the built geometry at alpha=16 its mean chord direction is
    # (0.96126, 0, -0.27564) = (cos a, 0, -sin a) and its mean normal is (sin a, 0, cos a),
    # i.e. the trailing edge is DOWN, exactly the convention of the membrane mesh.
    alpha = math.radians(arguments.alpha)
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    cx_line = np.linspace(-0.25 * chord, 1.30 * chord, arguments.nx)
    cz_line = np.linspace(-0.30 * chord, 0.60 * chord, arguments.nz)
    grid_cx, grid_cz = np.meshgrid(cx_line, cz_line)
    solver_x = grid_cx * cos_a + grid_cz * sin_a
    solver_z = -grid_cx * sin_a + grid_cz * cos_a
    points = torch.tensor(
        np.stack([solver_x.ravel(), np.full(solver_x.size, 0.5 * span), solver_z.ravel()], axis=1),
        dtype=torch.float64, device=device,
    )

    def ramp_factor(step: int) -> float:
        if arguments.ramp_steps <= 0 or step >= arguments.ramp_steps:
            return 1.0
        return 0.5 * (1.0 - math.cos(math.pi * step / arguments.ramp_steps))

    start_average = max(arguments.steps - arguments.average_last + 1, 1)
    velocity_sum = torch.zeros_like(points)
    sections: list[np.ndarray] = []
    time_stars: list[float] = []
    base_v_inf = solver.v_inf.clone()
    cn_series: list[float] = []

    for step in range(1, arguments.steps + 1):
        factor = ramp_factor(step)
        solver.v_inf = base_v_inf * factor
        try:
            object.__setattr__(solver.settings, "freestream", speed * factor)
        except AttributeError:
            pass
        proposal = stepper.propose(owner, (kin.evaluate(step * dt),), dt)
        geometry = solver.surface.evaluate(None, None)
        if step >= start_average:
            state = proposal.trial_state
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
            time_stars.append(step * frozen.DT_STAR)
        normal = geometry.normals[0]
        normal = normal / normal.norm()
        cn_series.append(float(
            (proposal.load.total_force @ normal / (0.5 * frozen.DENSITY * speed * speed * frozen.AREA_M2)).item()
        ))
        stepper.commit(owner, proposal)

    counted = len(sections)
    mean_velocity = (velocity_sum / counted).cpu().numpy().reshape(arguments.nz, arguments.nx, 3)
    section = np.mean(sections, axis=0)
    magnitude = np.linalg.norm(mean_velocity, axis=2) / speed
    section_cx = section[:, 0] * cos_a - section[:, 2] * sin_a
    section_cz = section[:, 0] * sin_a + section[:, 2] * cos_a
    u_chord = mean_velocity[:, :, 0] * cos_a - mean_velocity[:, :, 2] * sin_a
    w_chord = mean_velocity[:, :, 0] * sin_a + mean_velocity[:, :, 2] * cos_a

    arm = {"FLUXV_LE_PIN": os.environ.get("FLUXV_LE_PIN", "legacy"),
           "wing": "rigid flat plate", "alpha_deg": arguments.alpha}
    save_path = arguments.save_npz or str(Path(arguments.out).with_suffix(".field.npz"))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        u_chord=u_chord, w_chord=w_chord, magnitude=magnitude,
        cx_line=cx_line, cz_line=cz_line, section_cx=section_cx, section_cz=section_cz,
        counted=np.int64(counted), time_stars=np.array(time_stars),
        chord=np.float64(chord), speed=np.float64(speed),
        arm_json=np.array(__import__("json").dumps(arm)),
        case=np.array(f"RIGID-A{arguments.alpha:g}"), steps=np.int64(arguments.steps),
        start_average=np.int64(start_average), angle_deg=np.float64(arguments.alpha),
        cn_series=np.array(cn_series),
    )
    print(f"saved field to {save_path}")
    print(f"cn over the averaged window : mean {np.mean(cn_series[start_average-1:]):+.4f}   "
          f"max|cn| {np.max(np.abs(cn_series)):.4f}   (paper Fig. 9, rigid U=5, alpha=16: ~0.80)")
    print(f"separated strips at the last step: "
          f"{int(proposal.trial_state.separated_previous.sum()) if proposal.trial_state.separated_previous is not None else 'n/a'} of {ns}")
    return draw_from_npz(save_path, arguments.out)


if __name__ == "__main__":
    raise SystemExit(main())
