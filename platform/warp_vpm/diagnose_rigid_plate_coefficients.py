"""Force coefficients of the rigid flat plate on the PRODUCTION path, with no flow sampling.

Why a second script
-------------------
`visualize_roj_midspan_flow_rigid.py` samples the mid-span field, which costs about
22 GiB and half the run time.  Confirming a force coefficient needs none of that.  This
script steps the same production solver (`diagnose_roj_rigid_loads.build`, the frozen
rigid queue, the same half-cosine startup ramp) and reports only what the comparison
with the experiment needs:

  * Cn along the wing normal (the paper's convention), lift and streamwise components,
  * the Kutta-Joukowski check on the lattice's own section circulation (the
    trailing-edge row, note A18) on the SAME window,
  * the spanwise shape of that row,
  * the digitised rigid-plate value from Figure 9 and the attached elliptic thin-wing
    value.

It also proves which regularisation constant each kernel path is using.  The ring
kernel's denominator constant (kernels_ring_velocity.py, `|a x b|^2 + eps`) has two
live copies on this path -- the torch function behind the influence matrix and the
fused Warp kernel behind every other ring-velocity evaluation -- and at this wing's
scale it is not inert (note A19).  `FLUXV_RING_EPS_V` overrides both; at startup the
script evaluates ONE panel-sized ring at its centre with each path and with the
unregularised textbook law, so the printed ratios show the constant actually in force.

Usage
-----
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \\
        python3 platform/warp_vpm/diagnose_rigid_plate_coefficients.py --alpha 5 \\
        --steps 1500 --out /tmp/rigid_a5.json
Stage the run length: 3 steps, then 150, then the full length.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "platform", "platform/warp_vpm"):
    sys.path.insert(0, str(REPO_ROOT / sub))


def _textbook_self(ring: torch.Tensor, point: torch.Tensor) -> float:
    """Normal velocity at `point` from a unit-strength flat ring, no regularisation."""

    total = 0.0
    for index in range(4):
        start, end = ring[index], ring[(index + 1) % 4]
        a, b = point - start, point - end
        cross = torch.linalg.cross(a, b)
        unit_difference = a / a.norm() - b / b.norm()
        total += float((cross[2] / float(cross.dot(cross))) * float(((a - b) * unit_difference).sum()))
    return total / (4.0 * math.pi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--average-last", type=int, default=1200)
    parser.add_argument("--ramp-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dump-state", default=None,
                        help="save the final state (bound rings, collocation points, normals, "
                             "circulations, wake rings and strengths, freestream) as an .npz so "
                             "the steady system can be rebuilt and re-solved off line")
    parser.add_argument("--dt-star", type=float, default=None,
                        help="gate G-dt only (plan V3): step with this dt* and keep everything "
                             "measured in chords the same (wake 3 chords, free rows and particle "
                             "age 1 chord); pass --steps scaled to the same end time.  The "
                             "production value stays 0.01")
    parser.add_argument("--scale", type=float, default=None,
                        help="gate G-Scale only (plan V3): scale chord, span and U*dt together by "
                             "this factor, dt* unchanged, so the run is the same problem in other "
                             "units; everything else, the freestream included, stays.  Compare the "
                             "coefficients and the chord-normalised state with the scale-1 run")
    arguments = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")
    started = time.time()

    import queue_roj_rigid_fig9_12_13_15 as frozen
    from diagnose_roj_rigid_loads import build
    from fluxvortex.warp_fsi import kernels_ring_velocity as ring_kernel

    if arguments.scale is not None:
        # Gate G-Scale: build() takes the chord and the span from these module constants, and
        # the step from dt* x chord, so one factor carries all three.  This process only.
        if not (math.isfinite(arguments.scale) and arguments.scale > 0.0):
            raise SystemExit("--scale must be positive and finite")
        frozen.CHORD_M = frozen.CHORD_M * arguments.scale
        frozen.SPAN_M = frozen.SPAN_M * arguments.scale
        print(f"gate G-Scale: x{arguments.scale:g}: chord {frozen.CHORD_M:g} m, span {frozen.SPAN_M:g} m, "
              f"dt {frozen.DT_STAR * frozen.CHORD_M / 5.0:g} s")
    if arguments.dt_star is not None:
        # Gate G-dt: build() reads these module constants when it is called, so the
        # override lives in this process only.  Lengths stay what they are at dt* = 0.01.
        if not (math.isfinite(arguments.dt_star) and arguments.dt_star > 0.0):
            raise SystemExit("--dt-star must be positive and finite")
        frozen.DT_STAR = arguments.dt_star
        frozen.WAKE_MAX_ROWS = int(round(3.0 / arguments.dt_star))
        frozen.WAKE_FREE_ROWS = int(round(1.0 / arguments.dt_star))
        frozen.PARTICLE_MAX_AGE_STEPS = int(round(1.0 / arguments.dt_star))
        print(f"gate G-dt: dt* {arguments.dt_star:g}, wake {frozen.WAKE_MAX_ROWS} rows, "
              f"free rows {frozen.WAKE_FREE_ROWS}, particle age {frozen.PARTICLE_MAX_AGE_STEPS} steps")

    device = torch.device(arguments.device)
    speed, chord, span = 5.0, frozen.CHORD_M, frozen.SPAN_M
    nc, ns = frozen.AERO_GRID
    dx, ds = chord / nc, span / ns

    # ---- which constant is each kernel path using? -------------------------
    ring = torch.tensor(
        [[0.0, 0.0, 0.0], [dx, 0.0, 0.0], [dx, ds, 0.0], [0.0, ds, 0.0]],
        dtype=torch.float64, device=device,
    )[None]
    centre = torch.tensor([[0.5 * dx, 0.5 * ds, 0.0]], dtype=torch.float64, device=device)
    torch_path = ring_kernel.native_ring_velocity_expanded(
        centre, ring, core_fraction=1.0e-6, reference_length=ring_kernel.mean_chordwise_panel(ring),
    )[0, 0, 2]
    fused_path = ring_kernel.native_ring_velocity_fused(
        centre, ring, torch.ones(1, dtype=torch.float64, device=device),
        core_fraction=1.0e-6, reference_length=ring_kernel.mean_chordwise_panel(ring),
    )[0, 2]
    exact = _textbook_self(ring[0].cpu(), centre[0].cpu())
    kernel_check = {
        "eps_torch_path": ring_kernel._EPS_V,
        "eps_fused_kernel": ring_kernel._C_EPS_V,
        "self_torch_over_exact": float(torch_path.item()) / exact,
        "self_fused_over_exact": float(fused_path.item()) / exact,
    }
    print(f"ring kernel constant: torch path {ring_kernel._EPS_V:g}, fused kernel "
          f"{ring_kernel._C_EPS_V:g}; one panel-sized ring at its centre, over the "
          f"unregularised law: torch {kernel_check['self_torch_over_exact']:.4f}, "
          f"fused {kernel_check['self_fused_over_exact']:.4f}")
    kernel_check["kernel"] = "v3" if ring_kernel._KERNEL_V3 else "legacy"
    if ring_kernel._KERNEL_V3:
        import fluxvortex.warp_fsi.q16_flux_v5m_native as _native

        print(f"ring kernel: v3 (exact law, collinear guard {ring_kernel._COLLINEAR_REL:g}); "
              f"trailing-edge wake: {'v3 (implicit nascent row)' if _native._TE_WAKE_V3 else 'legacy'}; "
              f"wake core {_native._WAKE_CORE_PANELS:g} chordwise panels")
        kernel_check.update({
            "te_wake": "v3" if _native._TE_WAKE_V3 else "legacy",
            "wake_core_panels": _native._WAKE_CORE_PANELS,
            "collinear_rel": ring_kernel._COLLINEAR_REL,
        })
    if abs(kernel_check["self_fused_over_exact"] - kernel_check["self_torch_over_exact"]) > 1e-6:
        raise RuntimeError(
            "the fused kernel and the torch path are using DIFFERENT constants; "
            "FLUXV_RING_EPS_V did not reach the compiled kernel"
        )

    # ---- the run -----------------------------------------------------------
    solver, stepper, owner, kin = build(arguments.alpha, observed=False)
    solver.v_inf = solver.v_inf.to(device)
    dt = float(solver.settings.aerodynamic_dt)
    alpha = math.radians(arguments.alpha)
    q_s = 0.5 * frozen.DENSITY * speed * speed * chord * span

    # For --dump-state: the state AS SEEN BY THE SOLVE.  propose() convects the wake
    # first, builds the influence matrix, solves, and only afterwards inserts the new
    # trailing-edge row.  The trial state after propose() therefore holds the wake AFTER
    # insertion, which is not the wake the circulation was solved against.  The influence
    # matrix is built at exactly the solve moment, so wrapping that call captures the
    # right snapshot.  (An earlier version dumped the post-insertion state; the plan
    # author caught it, FLUX_V5M_PLAN_AUTHOR_REPLY_AND_V3_20261125.md section 0.2.)
    solve_time: dict[str, object] = {"calls": 0}
    if arguments.dump_state:
        import fluxvortex.warp_fsi.q16_flux_v5m_native as native_module

        original_native_aic = native_module.native_aic

        def capturing_native_aic(geometry, **keywords):
            caller = sys._getframe(1)
            trial = caller.f_locals.get("trial") if caller is not None else None
            if trial is not None:
                solve_time["calls"] = int(solve_time["calls"]) + 1
                solve_time["bound_rings"] = geometry.rings.detach().clone()
                solve_time["collocation"] = geometry.collocation.detach().clone()
                solve_time["normals"] = geometry.normals.detach().clone()
                solve_time["wake_rings"] = trial.wake_rings.detach().clone()
                solve_time["wake_gamma"] = trial.wake_gamma.detach().clone()
            return original_native_aic(geometry, **keywords)

        native_module.native_aic = capturing_native_aic

    start_average = max(arguments.steps - arguments.average_last + 1, 1)
    base_v_inf = solver.v_inf.clone()
    cn_series: list[float] = []
    particle_series: list[int] = []
    force_sum = torch.zeros(3, device=device, dtype=torch.float64)
    te_row_sum = 0.0

    for step in range(1, arguments.steps + 1):
        factor = 1.0 if step >= arguments.ramp_steps else 0.5 * (
            1.0 - math.cos(math.pi * step / arguments.ramp_steps)
        )
        solver.v_inf = base_v_inf * factor
        object.__setattr__(solver.settings, "freestream", speed * factor)
        proposal = stepper.propose(owner, (kin.evaluate(step * dt),), dt)
        geometry = solver.surface.evaluate(None, None)
        normal = geometry.normals[0]
        force = proposal.load.total_force
        cn_series.append(float((force @ (normal / normal.norm()) / q_s).item()))
        particle_series.append(int(proposal.trial_state.particle_field.n))
        if step >= start_average:
            force_sum.add_(force.detach().to(torch.float64))
            te_row_sum += float(
                proposal.trial_state.gamma_bound.reshape(nc, ns)[-1].sum().item()
            )
        stepper.commit(owner, proposal)

    window = cn_series[start_average - 1:]
    window_steps = max(len(window), 1)
    mean_force = (force_sum / window_steps).cpu().numpy()
    cn_mean = float(np.mean(window))
    lift = float(mean_force[2]) / q_s
    streamwise = float(mean_force[0]) / q_s
    cn_formula = (streamwise * math.sin(alpha) + lift * math.cos(alpha))
    kj = 2.0 * (te_row_sum / window_steps) * ds / (speed * chord * span)

    # ---- shape of the wake right behind the trailing edge (diagnostic only) ----
    # Why: with the regularisation removed, the first production point (alpha = 5 deg,
    # eps = 1e-12) came out at Cn 0.1525 against 0.2194 for the steady straight-wake
    # lattice of diagnose_ring_lattice_steady_cpu.py, although the two agree at 1e-9.
    # The kernel is the same in both; what the steady proxy does not have is a wake
    # that moves.  Recorded so every later sweep point says whether the wake rows
    # nearest the edge are where a convected sheet should put them (about 0.01 chord
    # apart, all downstream of the edge) or are bunched, stretched or folded back.
    # Wrapped: a diagnostic must never be the reason a run fails.
    wake_shape: dict | None = None
    try:
        wake_rows = proposal.trial_state.wake_rings.reshape(-1, ns, 4, 3)
        chord_dir = geometry.trailing_edge - geometry.leading_edge
        chord_dir = chord_dir / torch.linalg.vector_norm(chord_dir, dim=1, keepdim=True)
        unit_normal = geometry.normals[0] / geometry.normals[0].norm()
        front_edge = 0.5 * (wake_rows[:, :, 0, :] + wake_rows[:, :, 1, :])      # (rows, ns, 3)
        edge_mid = 0.5 * (geometry.trailing_edge[:-1, :] + geometry.trailing_edge[1:, :])
        offset = front_edge - edge_mid[None, :, :]
        along = torch.sum(offset * 0.5 * (chord_dir[:-1] + chord_dir[1:])[None, :, :], dim=2) / chord
        normal_offset = torch.sum(offset * unit_normal[None, None, :], dim=2) / chord
        spacing = along[1:, :] - along[:-1, :]
        shown = min(int(along.shape[0]), 6)
        wake_shape = {
            "rows": int(along.shape[0]),
            "first_rows_chord_aft_of_edge": [float(along[k].mean()) for k in range(shown)],
            "first_rows_normal_offset": [float(normal_offset[k].mean()) for k in range(shown)],
            "spacing_first20_mean": float(spacing[:20].mean()),
            "spacing_first20_min": float(spacing[:20].min()),
            "spacing_first20_max": float(spacing[:20].max()),
            "nodes_upstream_of_edge": int((along < -1.0e-6).sum()),
            "max_abs_normal_offset_all_rows": float(normal_offset.abs().max()),
            "nominal_spacing": float(speed * dt / chord),
        }
        # Circulation carried by the wake.  In a steady state every row holds the
        # trailing-edge row's circulation of its strip at shedding time, so the newest
        # and the oldest row should both equal the current trailing-edge sum and the
        # total should be rows x that sum.  A mismatch locates the difference between
        # this solver and the steady straight-wake lattice in the wake's STRENGTH
        # rather than only in its shape.
        wake_gamma = proposal.trial_state.wake_gamma.reshape(-1, ns)
        gamma_now = proposal.trial_state.gamma_bound.reshape(nc, ns)
        edge_sum = float(gamma_now[-1].sum())
        wake_shape.update({
            "newest_row_circulation_sum": float(wake_gamma[0].sum()),
            "oldest_row_circulation_sum": float(wake_gamma[-1].sum()),
            "wake_total_circulation": float(wake_gamma.sum()),
            "trailing_edge_row_circulation_sum_now": edge_sum,
            "wake_total_if_steady": edge_sum * float(wake_gamma.shape[0]),
        })
    except Exception as error:  # pragma: no cover - diagnostics must not break a run
        wake_shape = {"error": repr(error)}

    if arguments.dump_state:
        # Two snapshots of the last step's wake: the one the circulation was solved
        # against (captured at the influence-matrix call: convected, new row not yet
        # inserted) and the one left after insertion.  diagnose_production_state_
        # resolve_cpu.py rebuilds the linear system from the first.
        try:
            state = proposal.trial_state
            np.savez(
                arguments.dump_state,
                bound_rings=solve_time["bound_rings"].cpu().numpy(),
                collocation=solve_time["collocation"].cpu().numpy(),
                normals=solve_time["normals"].cpu().numpy(),
                wake_rings=solve_time["wake_rings"].cpu().numpy(),
                wake_gamma=solve_time["wake_gamma"].cpu().numpy(),
                gamma_bound=state.gamma_bound.detach().cpu().numpy(),
                wake_rings_after_insertion=state.wake_rings.detach().cpu().numpy(),
                wake_gamma_after_insertion=state.wake_gamma.detach().cpu().numpy(),
                v_inf=solver.v_inf.detach().cpu().numpy(),
                nc=np.int64(nc), ns=np.int64(ns), chord=np.float64(chord),
                density=np.float64(frozen.DENSITY), speed=np.float64(speed),
                eps=np.float64(ring_kernel._EPS_V), alpha_deg=np.float64(arguments.alpha),
                steps=np.int64(arguments.steps), solve_calls=np.int64(solve_time["calls"]),
                dt=np.float64(dt),
            )
            print(f"final state written to {arguments.dump_state} "
                  f"(solve-time wake captured on {int(solve_time['calls'])} influence-matrix calls)")
        except Exception as error:  # pragma: no cover
            print(f"state dump failed (run result unaffected): {error!r}")

    gamma = proposal.trial_state.gamma_bound.detach().reshape(nc, ns).cpu().numpy()
    te_row, row1 = gamma[-1], gamma[0]
    centre_te = te_row[ns // 2]
    y = (np.arange(ns) + 0.5) / ns
    outer = (y < 0.1) | (y > 0.9)
    elliptic = np.sqrt(np.clip(1.0 - (2.0 * y - 1.0) ** 2, 0.0, None))
    aspect_ratio = span / chord
    theory = 2.0 * math.pi * aspect_ratio / (aspect_ratio + 2.0) * math.sin(alpha)
    # Helmbold's low-aspect-ratio slope, CL_alpha = 2 pi AR / (2 + sqrt(AR^2 + 4)): the
    # line the steady lattice is checked against (note A20).  The expression just above,
    # 2 pi AR / (AR + 2), is a quarter higher at AR = 2.
    theory_helmbold = (
        2.0 * math.pi * aspect_ratio / (2.0 + math.sqrt(aspect_ratio**2 + 4.0)) * math.sin(alpha)
    )
    try:
        from fluxvortex.cases.rojratsirikul2011_observations import figure9_value

        paper = figure9_value("rigid_flat_plate", float(speed), float(arguments.alpha)).cn
    except Exception as error:  # exact-match lookup
        paper = None
        print(f"(no digitised rigid Figure 9 row at alpha={arguments.alpha:g}: {error})")
    gamma_2d = math.pi * chord * speed * math.sin(alpha)

    result = {
        "alpha_deg": arguments.alpha, "steps": arguments.steps, "window": [start_average, arguments.steps],
        "environment": {
            "FLUXV_RING_EPS_V": __import__("os").environ.get("FLUXV_RING_EPS_V", ""),
            "FLUXV_RING_KERNEL": __import__("os").environ.get("FLUXV_RING_KERNEL", "legacy"),
            "FLUXV_TE_WAKE": __import__("os").environ.get("FLUXV_TE_WAKE", "legacy"),
            "FLUXV_WAKE_CORE_PANELS": __import__("os").environ.get("FLUXV_WAKE_CORE_PANELS", "0.25"),
            "FLUXV_LE_PIN": __import__("os").environ.get("FLUXV_LE_PIN", "legacy"),
            "FLUXV_LESP_MASK_OFF": __import__("os").environ.get("FLUXV_LESP_MASK_OFF", ""),
            "FLUXV_PACKET_CLOSURE_SCALE": __import__("os").environ.get(
                "FLUXV_PACKET_CLOSURE_SCALE", "net"
            ),
        },
        "kernel_check": kernel_check,
        "cn_normal": {"mean": cn_mean, "min": float(np.min(window)), "max": float(np.max(window))},
        "cn_by_formula": cn_formula, "lift": lift, "streamwise": streamwise,
        "max_abs_cn_whole_run": float(np.max(np.abs(cn_series))),
        "paper_cn_rigid": paper, "theory_elliptic": theory,
        "theory_helmbold": theory_helmbold,
        "model_over_paper": (cn_mean / paper - 1.0) if paper else None,
        "model_over_theory": cn_mean / theory - 1.0,
        "model_over_helmbold": cn_mean / theory_helmbold - 1.0,
        "kutta_joukowski_check": kj,
        "trailing_edge_row": {
            "centre_m2_s": float(centre_te), "centre_over_2d": float(centre_te / gamma_2d),
            "tip_over_centre": float(te_row[0] / centre_te),
            "mean_over_centre": float(te_row.mean() / centre_te),
            "outer_20pct_share": float(te_row[outer].sum() / te_row.sum()),
            "values": [float(v) for v in te_row],
        },
        "elliptic_reference": {"tip_over_centre": float(elliptic[0]),
                               "mean_over_centre": float(elliptic.mean()),
                               "outer_20pct_share": float(elliptic[outer].sum() / elliptic.sum())},
        "leading_edge_row": {"mean_abs_m2_s": float(np.abs(row1).mean()),
                             "signs": sorted(set(np.sign(row1).tolist()))},
        "particles_last": particle_series[-1], "particles_max": int(max(particle_series)),
        "cn_series_every_100": cn_series[::100],
        "wake_shape": wake_shape,
        "elapsed_seconds": time.time() - started,
    }
    r = result["trailing_edge_row"]
    print(f"\n=== rigid plate alpha={arguments.alpha:g} deg, {arguments.steps} steps, "
          f"eps {ring_kernel._EPS_V:g}, {result['elapsed_seconds']:.0f} s ===")
    print(f"Cn along the wing normal, window mean : {cn_mean:+.4f}   "
          f"(min {result['cn_normal']['min']:+.4f}, max {result['cn_normal']['max']:+.4f}; "
          f"max|cn| whole run {result['max_abs_cn_whole_run']:.4f})")
    print(f"  lift {lift:+.4f}  streamwise {streamwise:+.4f}  "
          f"Cn by Fx sin a + Fz cos a {cn_formula:+.4f}  Kutta-Joukowski {kj:+.4f}")
    if paper:
        print(f"  paper Fig. 9, rigid U=5 : {paper:.4f}  -> model / experiment "
              f"{100.0 * (cn_mean / paper - 1.0):+.1f}%")
    print(f"  attached thin wing, elliptic AR={aspect_ratio:g} : {theory:.4f}  -> model / theory "
          f"{100.0 * (cn_mean / theory - 1.0):+.1f}%")
    print(f"  attached thin wing, Helmbold AR={aspect_ratio:g} : {theory_helmbold:.4f}  -> model / theory "
          f"{100.0 * (cn_mean / theory_helmbold - 1.0):+.1f}%")
    print(f"trailing-edge row: centre {r['centre_m2_s']:.5f} m^2/s = {r['centre_over_2d']:.3f} of 2-D; "
          f"tip/centre {r['tip_over_centre']:.3f} (elliptic {elliptic[0]:.3f}); "
          f"mean/centre {r['mean_over_centre']:.3f} (elliptic {elliptic.mean():.3f}); "
          f"outer-20% {r['outer_20pct_share']:.3f} (elliptic {result['elliptic_reference']['outer_20pct_share']:.3f})")
    print(f"leading-edge row: mean |Gamma_1| {result['leading_edge_row']['mean_abs_m2_s']:.5f} m^2/s, "
          f"signs {result['leading_edge_row']['signs']};  particles at the last step "
          f"{result['particles_last']} (max {result['particles_max']})")
    if wake_shape and "error" not in wake_shape:
        print(f"wake at the last step: {wake_shape['rows']} rows; first rows, chord aft of the edge: "
              + " ".join(f"{v:.4f}" for v in wake_shape["first_rows_chord_aft_of_edge"])
              + "; normal offset: " + " ".join(f"{v:+.4f}" for v in wake_shape["first_rows_normal_offset"])
              + f"\n  row spacing over the first 20 rows (nominal {wake_shape['nominal_spacing']:.4f}): "
              f"mean {wake_shape['spacing_first20_mean']:.4f}, min {wake_shape['spacing_first20_min']:.4f}, "
              f"max {wake_shape['spacing_first20_max']:.4f};  nodes upstream of the edge: "
              f"{wake_shape['nodes_upstream_of_edge']};  largest normal offset over all rows: "
              f"{wake_shape['max_abs_normal_offset_all_rows']:.3f} chord")
        if "newest_row_circulation_sum" in wake_shape:
            print(f"  wake circulation: newest row {wake_shape['newest_row_circulation_sum']:+.5f}, "
                  f"oldest row {wake_shape['oldest_row_circulation_sum']:+.5f}, total "
                  f"{wake_shape['wake_total_circulation']:+.4f};  trailing-edge row now "
                  f"{wake_shape['trailing_edge_row_circulation_sum_now']:+.5f}, so a steady wake would total "
                  f"{wake_shape['wake_total_if_steady']:+.4f}")
    else:
        print(f"wake shape diagnostic unavailable: {wake_shape}")
    if arguments.out:
        Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
