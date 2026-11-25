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
    parser.add_argument("--sample-stride", type=int, default=1,
                        help="sample every Nth step inside the averaging window.  A time "
                             "average over a long window does not need every step, and the "
                             "sampling is what costs: each sample evaluates four induction "
                             "terms at nx*nz points against the whole particle field.")
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
    force_sum = torch.zeros(3, device=device, dtype=torch.float64)
    te_row_sum = 0.0

    for step in range(1, arguments.steps + 1):
        factor = ramp_factor(step)
        solver.v_inf = base_v_inf * factor
        try:
            object.__setattr__(solver.settings, "freestream", speed * factor)
        except AttributeError:
            pass
        proposal = stepper.propose(owner, (kin.evaluate(step * dt),), dt)
        geometry = solver.surface.evaluate(None, None)
        if step >= start_average and (step - start_average) % arguments.sample_stride == 0:
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
        force = proposal.load.total_force
        cn_series.append(float(
            (force @ normal / (0.5 * frozen.DENSITY * speed * speed * frozen.AREA_M2)).item()
        ))
        if step >= start_average:
            force_sum.add_(force.detach().to(torch.float64))
            # Circulation of the trailing-edge row, summed over the strips.  In this
            # ring lattice each chordwise row is a closed ring, so a loop around a
            # section crosses the front and back legs of every row and they cancel
            # except at the two ends: the SECTION circulation is the trailing-edge
            # row's strength, and the sum over all 15 rows is 10x it and means
            # nothing physical (note A18).
            te_row_sum += float(
                proposal.trial_state.gamma_bound.reshape(nc, ns)[-1].sum().item()
            )
        stepper.commit(owner, proposal)

    counted = len(sections)
    # Three force coefficients, so the number compared with the paper is never a
    # convention in disguise.  The incidence is baked into the mesh with the trailing
    # edge DOWN, so the wing normal is (sin a, 0, cos a) and the paper's Cn is
    # Fx sin a + Fz cos a = L cos a + D sin a (the streamwise term ADDS).
    q_s = 0.5 * frozen.DENSITY * speed * speed * frozen.AREA_M2
    window_steps = max(arguments.steps - start_average + 1, 1)
    mean_force = (force_sum / window_steps).detach().cpu().numpy()
    te_row_mean = te_row_sum / window_steps
    # Kutta-Joukowski on the lattice's own section circulation: L = rho U sum(Gamma_i ds_i).
    kj_coefficient = 2.0 * te_row_mean * (span / ns) / (speed * chord * span)
    lift_coefficient = float(mean_force[2]) / q_s
    streamwise_coefficient = float(mean_force[0]) / q_s
    cn_formula = (float(mean_force[0]) * sin_a + float(mean_force[2]) * cos_a) / q_s
    aspect_ratio = span / chord
    cn_theory = 2.0 * math.pi * aspect_ratio / (aspect_ratio + 2.0) * math.sin(alpha)
    try:
        from fluxvortex.cases.rojratsirikul2011_observations import figure9_value

        paper_cn: float | None = figure9_value(
            "rigid_flat_plate", float(speed), float(arguments.alpha)
        ).cn
    except Exception as error:  # the lookup is exact-match; a sweep angle may be absent
        paper_cn = None
        print(f"(no digitised rigid Fig. 9 row at U={speed:g}, alpha={arguments.alpha:g}: {error})")
    mean_velocity = (velocity_sum / counted).cpu().numpy().reshape(arguments.nz, arguments.nx, 3)
    section = np.mean(sections, axis=0)
    magnitude = np.linalg.norm(mean_velocity, axis=2) / speed
    section_cx = section[:, 0] * cos_a - section[:, 2] * sin_a
    section_cz = section[:, 0] * sin_a + section[:, 2] * cos_a
    u_chord = mean_velocity[:, :, 0] * cos_a - mean_velocity[:, :, 2] * sin_a
    w_chord = mean_velocity[:, :, 0] * sin_a + mean_velocity[:, :, 2] * cos_a

    arm = {"FLUXV_LE_PIN": os.environ.get("FLUXV_LE_PIN", "legacy"),
           "FLUXV_LE_SHEET_EXTENT": os.environ.get("FLUXV_LE_SHEET_EXTENT", "off"),
           "FLUXV_LESP_MASK_OFF": os.environ.get("FLUXV_LESP_MASK_OFF", ""),
           "wing": "rigid flat plate", "alpha_deg": arguments.alpha}
    save_path = arguments.save_npz or str(Path(arguments.out).with_suffix(".field.npz"))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # Spanwise loading of the bound sheet at the last step, as the SIGNED sum of each
    # strip's row strengths.  On the membrane (notes A9 and A18) the outermost strip
    # carries 85% of the centre strip's circulation where an elliptic loading has 25%,
    # i.e. the loading is nearly flat and the trailing vorticity sits in the outermost
    # strip.  That weakens the finite-span downwash: a uniform horseshoe has the lift
    # slope 2*pi*AR/(AR+1) against the ellipse's 2*pi*AR/(AR+2).  Printed here so the
    # rigid sweep says whether the rigid plate is loaded the same way.
    strip_circulation = (
        proposal.trial_state.gamma_bound.detach().reshape(nc, ns).sum(dim=0).cpu().numpy()
    )
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
        lift_coefficient=np.float64(lift_coefficient),
        streamwise_coefficient=np.float64(streamwise_coefficient),
        cn_formula=np.float64(cn_formula), paper_cn=np.float64(np.nan if paper_cn is None else paper_cn),
        cn_theory_elliptic=np.float64(cn_theory), particle_count=np.int64(
            proposal.trial_state.particle_field.n),
        strip_circulation=strip_circulation,
        te_row_last=proposal.trial_state.gamma_bound.detach().reshape(nc, ns)[-1].cpu().numpy(),
        te_row_window_mean_sum=np.float64(te_row_mean),
        kj_coefficient=np.float64(kj_coefficient),
    )
    print(f"saved field to {save_path}")
    window_cn = cn_series[start_average - 1:]
    cn_mean = float(np.mean(window_cn))
    print(f"Cn along the WING NORMAL, every step of the window : mean {cn_mean:+.4f}   "
          f"min {np.min(window_cn):+.4f}   max {np.max(window_cn):+.4f}   "
          f"max|cn| over the whole run {np.max(np.abs(cn_series)):.4f}")
    print(f"  same window, force components /(q S)      : lift Fz {lift_coefficient:+.4f}   "
          f"streamwise Fx {streamwise_coefficient:+.4f}")
    print(f"  Cn by Fx sin a + Fz cos a                 : {cn_formula:+.4f}   "
          f"(must agree with the wing-normal mean)")
    if paper_cn is None:
        print(f"  paper Fig. 9, rigid plate U={speed:g}, alpha={arguments.alpha:g}        : no digitised row")
    else:
        print(f"  paper Fig. 9, rigid plate U={speed:g}, alpha={arguments.alpha:g}        : {paper_cn:.4f}"
              f"   -> model / experiment {100.0 * (cn_mean / paper_cn - 1.0):+.1f}%")
    print(f"  attached thin wing, elliptic, AR={aspect_ratio:g}  : {cn_theory:.4f}"
          f"   -> model / theory {100.0 * (cn_mean / cn_theory - 1.0):+.1f}%")
    print(f"  particles at the last step                 : {int(proposal.trial_state.particle_field.n)}"
          f" of capacity {int(proposal.trial_state.particle_field.capacity)}")
    centre = strip_circulation[ns // 2]
    negative_strips = int(np.count_nonzero(strip_circulation * np.sign(centre) < 0.0))
    y_strip = (np.arange(ns) + 0.5) / ns
    outer = (y_strip < 0.1) | (y_strip > 0.9)
    elliptic = np.sqrt(np.clip(1.0 - (2.0 * y_strip - 1.0) ** 2, 0.0, None))
    print(f"bound circulation per strip at the last step, as a fraction of the centre strip "
          f"({negative_strips} strip(s) of opposite sign):")
    print("  tip -> centre -> tip: " + " ".join(f"{v / centre:.2f}" for v in strip_circulation))
    print(f"  tip strips / centre : {strip_circulation[0] / centre:.3f} and {strip_circulation[-1] / centre:.3f}"
          f"   (elliptic {elliptic[0]:.3f})")
    print(f"  span mean / centre  : {np.mean(strip_circulation) / centre:.3f}   (elliptic {elliptic.mean():.3f})")
    print(f"  outer-20% share     : {strip_circulation[outer].sum() / strip_circulation.sum():.3f}"
          f"   (elliptic {elliptic[outer].sum() / elliptic.sum():.3f})")
    # The same for the TRAILING-EDGE row, which is the section circulation of this ring
    # lattice (the row sums above are a staircase of closed rings and are not).
    te_row = proposal.trial_state.gamma_bound.detach().reshape(nc, ns)[-1].cpu().numpy()
    centre_te = te_row[ns // 2]
    gamma_2d = math.pi * chord * speed * math.sin(alpha)
    elliptic_centre = 4.0 / math.pi * 0.5 * cn_theory * speed * chord
    print(f"Kutta-Joukowski check on the window mean : 2*sum(Gamma_TE)*ds/(U S) = {kj_coefficient:+.4f}"
          f"   against lift {lift_coefficient:+.4f} and Cn {cn_mean:+.4f}")
    print(f"trailing-edge row Gamma(y) at the last step, m^2/s, as a fraction of the centre strip:")
    print("  tip -> centre -> tip: " + " ".join(f"{v / centre_te:.2f}" for v in te_row))
    print(f"  centre strip        : {centre_te:.5f}   2-D thin airfoil pi*c*U*sin(a) {gamma_2d:.5f}"
          f" -> {centre_te / gamma_2d:.3f} of 2-D;   elliptic AR={aspect_ratio:g} centre {elliptic_centre:.5f}"
          f" -> {centre_te / elliptic_centre:.3f}")
    print(f"  tip strips / centre : {te_row[0] / centre_te:.3f} and {te_row[-1] / centre_te:.3f}"
          f"   (elliptic {elliptic[0]:.3f})")
    print(f"  span mean / centre  : {np.mean(te_row) / centre_te:.3f}   (elliptic {elliptic.mean():.3f})")
    print(f"  outer-20% share     : {te_row[outer].sum() / te_row.sum():.3f}"
          f"   (elliptic {elliptic[outer].sum() / elliptic.sum():.3f})")
    # The separation criterion divides the LEADING-EDGE ROW's ring strength by
    # c*U*(theta+sin theta)/1.13 and compares with lesp_crit.  Thin-airfoil theory puts
    # that quantity (A0) near alpha in radians for an attached flat plate, i.e. 0.087 at
    # 5 deg.  Printed with the pin switched off (FLUXV_LESP_MASK_OFF=1) it says what the
    # lattice's own attached solution gives for it.
    row1 = proposal.trial_state.gamma_bound.detach().reshape(nc, ns)[0].cpu().numpy()
    rings = geometry.rings.reshape(nc, ns, 4, 3)
    first_dx = float(
        (
            0.5
            * (
                torch.linalg.vector_norm(rings[0, :, 3] - rings[0, :, 0], dim=1)
                + torch.linalg.vector_norm(rings[0, :, 2] - rings[0, :, 1], dim=1)
            )
        ).mean().item()
    )
    theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * first_dx / chord)))
    lesp_scale = chord * speed * (theta + math.sin(theta)) / 1.13
    print(f"leading-edge row at the last step     : mean |Gamma_1| {np.abs(row1).mean():.5f} m^2/s,"
          f" signs {sorted(set(np.sign(row1).tolist()))}")
    print(f"  as the criterion sees it, |Gamma_1| / (c U (theta+sin theta)/1.13) = "
          f"{np.abs(row1).mean() / lesp_scale:.4f}   (lesp_crit {frozen.LESP_CRIT};"
          f" thin-airfoil A0 for an attached plate at this alpha = {alpha:.4f})")
    print(f"separated strips at the last step: "
          f"{int(proposal.trial_state.separated_previous.sum()) if proposal.trial_state.separated_previous is not None else 'n/a'} of {ns}")
    return draw_from_npz(save_path, arguments.out)


if __name__ == "__main__":
    raise SystemExit(main())
