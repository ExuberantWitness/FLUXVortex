"""S2.0: does the deposited LE ribbon reproduce a MONOPOLE at the first collocation row?

The deposit is a closed ribbon whose two spanwise sheets carry ``+gamma`` (on the
physical leading edge) and ``-gamma`` (on the convected frontier), closed by
streamwise connectors.  Its net circulation is zero, so at a target far compared
with the LE->frontier gap it behaves as a dipole whose normal induction scales
like ``gamma * ds * f'(r)`` with ``f(u) = (1-exp(-u^2/2))/u``.

The physical object it is supposed to emulate is a MONOPOLE: a single spanwise
sheet of circulation ``gamma`` sitting on the leading edge.  Therefore

    w_dip / w_mono  =  1 - f(r - ds) / f(r)

should hold, with ``r`` the LE->collocation distance and ``ds`` the frontier
displacement.  ``f`` rises to a peak at ``u_peak = 1.5852 sigma``, so this ratio
is positive when ``r < r_peak`` and negative beyond it.  This probe measures the
ratio directly (no model change, no production flag) and sweeps ``sigma`` to
locate the sign flip, which is the decision quantity for route B.

Run:
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 \
        python3 platform/warp_vpm/diagnose_roj_le_kernel_sign.py --alpha 15
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch

from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import RigidNativeV5MSolver, RigidV5MSurface
from pfield_torch_gpu import CudaParticleField

CHORD = 0.0688
SPAN = 0.1375
FREESTREAM = 5.0
NC, NS = 15, 30
RHO = 1.208

#: sigma/c values to sweep; spacing is always sigma/2.125 so the frozen overlap
#: gate is satisfied exactly (route B's grouped constraint).
SIGMA_SWEEP = (0.040, 0.036, 0.034, 0.032, 0.030, 0.028, 0.024, 0.020)
OVERLAP = 2.125
U_PEAK = 1.585201  # bisection root of exp(-u^2/2)(u^2+1) = 1


def filament_profile(u: float) -> float:
    """Gaussian-filament induced velocity magnitude, in units of gamma/(2 pi sigma)."""
    if u <= 0.0:
        return 0.0
    return (1.0 - math.exp(-0.5 * u * u)) / u


def build_case(alpha_deg: float, dt_star: float, device: str, steps: int):
    """Run ``steps`` production proposals and capture the last ribbon deposit inputs."""

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
    from fluxvortex.aero.v5m.stepper import RigidV5MStepper

    solver = RigidNativeV5MSolver(
        RigidV5MSurface(chordwise_panels=NC, spanwise_panels=NS, device=device), settings
    )
    stepper = RigidV5MStepper(solver, device=device)
    owner = stepper.initialize((kinematics.evaluate(0.0),))

    captured: list[dict] = []
    original = CudaParticleField.add_connected_ribbon_particles

    def spy(self, anchor_nodes, frontier_nodes, strengths, **kwargs):
        captured.append(
            {
                "anchor": anchor_nodes.clone(),
                "frontier": frontier_nodes.clone(),
                "strengths": strengths.clone(),
                "smoothing_radius": float(kwargs["smoothing_radius"]),
                "target_spacing": float(kwargs["target_spacing"]),
            }
        )
        return original(self, anchor_nodes, frontier_nodes, strengths, **kwargs)

    CudaParticleField.add_connected_ribbon_particles = spy
    try:
        for step in range(1, steps + 1):
            proposal = stepper.propose(owner, (kinematics.evaluate(step * dt),), dt)
            stepper.commit(owner, proposal)
    finally:
        CudaParticleField.add_connected_ribbon_particles = original

    if not captured:
        raise RuntimeError("no ribbon deposit captured")
    geometry = solver.surface.evaluate(None, None)
    cbar = float(torch.mean(owner.state.reference_cell_chord).item())
    return geometry, cbar, captured[-1], float(dt)


def subdivide(starts, ends, circs, sigma_value: float, spacing_value: float, device: str):
    """Mirror CudaParticleField.add_connected_ribbon_particles' edge subdivision."""

    edge_vector = ends - starts
    edge_length = torch.linalg.vector_norm(edge_vector, dim=1)
    active = (torch.abs(circs) >= 1.0e-14) & (edge_length >= 1.0e-12)
    starts = starts[active]
    edge_vector = edge_vector[active]
    edge_length = edge_length[active]
    circs = circs[active]
    if starts.shape[0] == 0:
        return None
    spacing = torch.as_tensor(spacing_value, device=device, dtype=torch.float64)
    counts = torch.clamp(torch.ceil(edge_length / spacing), min=1.0).to(torch.int64)
    total = int(counts.sum().item())
    edge_index = torch.repeat_interleave(
        torch.arange(starts.shape[0], device=device, dtype=torch.int64), counts
    )
    offsets = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(total, device=device, dtype=torch.int64) - torch.repeat_interleave(
        offsets, counts
    )
    per_edge = counts[edge_index]
    fraction = (local.to(torch.float64) + 0.5) / per_edge.to(torch.float64)
    segment = edge_vector[edge_index] / per_edge[:, None]
    positions = starts[edge_index] + fraction[:, None] * edge_vector[edge_index]
    circulation = circs[edge_index]
    gamma = circulation[:, None] * segment
    return positions, gamma, circulation


def normal_velocity(chunks, sigma_value, spacing_value, targets, normals, device):
    """Normal induction at ``targets`` from the union of ``chunks`` edge groups."""

    field = CudaParticleField(65536, device=device)
    for chunk in chunks:
        if chunk is None:
            continue
        positions, gamma, circulation = chunk
        count = positions.shape[0]
        field.add_particles(
            positions,
            gamma,
            torch.full((count,), sigma_value, device=device, dtype=torch.float64),
            circul=circulation,
            birth_step=0,
        )
    if field.n == 0:
        return torch.zeros(targets.shape[0], device=device, dtype=torch.float64), 0
    velocity = field.velocity_at_cuda(targets)
    return torch.sum(velocity * normals, dim=1), field.n


def edge_groups(anchor, frontier, strengths, device):
    """The three edge groups of the closed ribbon, in kernel order."""

    spanwise_le = (anchor[:-1], anchor[1:], strengths)
    spanwise_frontier = (frontier[:-1], frontier[1:], -strengths)
    connector_circulation = torch.cat(
        (-strengths[:1], strengths[:-1] - strengths[1:], strengths[-1:])
    )
    connectors = (anchor, frontier, connector_circulation)
    return spanwise_le, spanwise_frontier, connectors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--dt-star", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    geometry, cbar, deposit, dt = build_case(
        args.alpha, args.dt_star, args.device, args.steps
    )
    anchor = deposit["anchor"]
    frontier = deposit["frontier"]
    strengths = deposit["strengths"]

    targets = geometry.collocation[:NS].to(torch.float64).contiguous()
    normals = geometry.normals[:NS].to(torch.float64).contiguous()
    displacement = float(
        torch.linalg.vector_norm(frontier - anchor, dim=1).mean().item()
    )
    x_c0 = float(torch.linalg.vector_norm(targets[NS // 2] - anchor[NS // 2]).item())
    gamma_max = float(torch.max(torch.abs(strengths)).item())

    print(f"alpha = {args.alpha:g} deg, dt* = {args.dt_star:g} (dt = {dt:.6e} s)")
    print(f"cbar = {cbar:.6e} m, x_c0 = {x_c0 / cbar:.4f} c, "
          f"frontier displacement ds = {displacement / cbar:.4f} c")
    print(f"max |gamma_lev| = {gamma_max:.4e} m^2/s "
          f"(production sigma = {deposit['smoothing_radius'] / cbar:.4f} c, "
          f"spacing = {deposit['target_spacing'] / cbar:.4f} c)")
    print()
    print("  sigma/c   r_peak/c  r/x_c0 |  w_mono      w_dip       w_dip/w_mono "
          "|  pred 1-f(r-ds)/f(r) |  w_anti     w_conn    | np")
    print("  " + "-" * 118)

    record: dict[str, object] = {
        "alpha_deg": args.alpha,
        "dt_star": args.dt_star,
        "dt": dt,
        "cbar": cbar,
        "x_c0_over_c": x_c0 / cbar,
        "frontier_displacement_over_c": displacement / cbar,
        "gamma_max": gamma_max,
        "production_sigma_over_c": deposit["smoothing_radius"] / cbar,
        "production_spacing_over_c": deposit["target_spacing"] / cbar,
        "u_peak": U_PEAK,
        "sweep": [],
    }

    groups = edge_groups(anchor, frontier, strengths, args.device)
    for sigma_rel in SIGMA_SWEEP:
        sigma_value = sigma_rel * cbar
        spacing_value = sigma_rel / OVERLAP * cbar
        n_le = subdivide(*groups[0], sigma_value, spacing_value, args.device)
        n_fr = subdivide(*groups[1], sigma_value, spacing_value, args.device)
        n_cn = subdivide(*groups[2], sigma_value, spacing_value, args.device)

        w_mono, _ = normal_velocity([n_le], sigma_value, spacing_value, targets, normals, args.device)
        w_anti, _ = normal_velocity([n_fr], sigma_value, spacing_value, targets, normals, args.device)
        w_conn, _ = normal_velocity([n_cn], sigma_value, spacing_value, targets, normals, args.device)
        w_dip, np_dip = normal_velocity(
            [n_le, n_fr, n_cn], sigma_value, spacing_value, targets, normals, args.device
        )

        mid = NS // 2
        mono_mid = float(w_mono[mid].item())
        dip_mid = float(w_dip[mid].item())
        ratio = dip_mid / mono_mid if mono_mid != 0.0 else float("nan")
        r = x_c0 / sigma_value
        ds = displacement / sigma_value
        predicted = 1.0 - filament_profile(r - ds) / filament_profile(r)
        r_peak = U_PEAK * sigma_rel

        print(f"  {sigma_rel:7.4f}  {r_peak:8.4f}  {x_c0 / (r_peak * cbar):7.3f} | "
              f"{mono_mid:+.4e}  {dip_mid:+.4e}  {ratio:+9.4f} | {predicted:+19.4f} | "
              f"{float(w_anti[mid].item()):+.3e} {float(w_conn[mid].item()):+.3e} | {np_dip:6d}")

        record["sweep"].append(
            {
                "sigma_over_c": sigma_rel,
                "spacing_over_c": sigma_rel / OVERLAP,
                "r_peak_over_c": r_peak,
                "x_c0_over_r_peak": float(x_c0 / (r_peak * cbar)),
                "w_mono_mid": mono_mid,
                "w_anti_mid": float(w_anti[mid].item()),
                "w_conn_mid": float(w_conn[mid].item()),
                "w_dip_mid": dip_mid,
                "ratio_dip_over_mono_mid": ratio,
                "ratio_predicted": predicted,
                "particle_count": np_dip,
                "w_dip_max_abs": float(torch.max(torch.abs(w_dip)).item()),
                "w_mono_max_abs": float(torch.max(torch.abs(w_mono)).item()),
            }
        )

    # A ratio sign change across the sweep is the route-B precondition.
    ratios = [entry["ratio_dip_over_mono_mid"] for entry in record["sweep"]]
    signs = {1 if value > 0.0 else -1 for value in ratios if value == value}
    record["ratio_sign_changes"] = len(signs) > 1
    print()
    print(f"ratio sign set over sweep: {sorted(signs)} -> "
          f"sign flip = {record['ratio_sign_changes']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
