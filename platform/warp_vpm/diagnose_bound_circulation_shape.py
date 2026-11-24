"""Shape of the BOUND CIRCULATION on the committed state: chordwise and spanwise.

Why this exists
---------------
A literature cross-check (research note A9) found two shapes that do not match what the
literature reports for a low-aspect-ratio wing at high incidence:

  chordwise  -- the span-averaged steady pressure is a SAWTOOTH (-2.9 / +12.0 / -3.0 / ...)
                where surface-pressure measurements on thin flat plates show a flat
                plateau associated with separation, with no leading-edge suction spike;
  spanwise   -- the circulation does NOT fall to zero at the tips (outermost strip 1.49x its
                neighbour; outer 20% of span holds 0.161 of the circulation, elliptic 0.105).

Both trace back to the leading-edge row: on the legacy pin it carries the OPPOSITE sign to
the rest of the bound circulation, so dx_gamma[0] = Gamma[0]/dx is negative and
dx_gamma[1] = (Gamma[1]-Gamma[0])/dx is inflated.

This script measures the shape for one arm so the arms can be compared on the mechanism.
It is NOT the selection criterion -- that is the error against the digitised experiment on
the converged window.

How it captures
---------------
`sys.setprofile` reads the locals of `propose()` at its return, the same technique as
`diagnose_roj_joint_history.py`, so the solver is not modified.  Only the last propose of
the LAST step is kept: that is the committed state.

Usage
-----
    FLUXV_LE_PIN=signed python3 platform/warp_vpm/diagnose_bound_circulation_shape.py \
        --case ROJ11-A16 --steps 150 --out /tmp/shape_signed.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


def _capture_last_propose(runner, steps: int) -> dict:
    """Run `steps` and return the locals of the final propose()."""

    captured: dict[str, object] = {}

    def grab(frame, event, arg):
        if (
            event == "return"
            and frame.f_code.co_name == "propose"
            and frame.f_code.co_filename.endswith("q16_flux_v5m_native.py")
        ):
            local = frame.f_locals
            if "gamma" in local and "scale" in local:
                for key in (
                    "gamma", "gamma_pre", "scale", "geometry", "surface_separated",
                    "source_result", "external_flow",
                ):
                    if key in local:
                        value = local[key]
                        captured[key] = (
                            value.detach().clone()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
        return grab

    sys.setprofile(grab)
    try:
        payload = runner.run(max_aero_steps=steps)
    finally:
        sys.setprofile(None)
    return {"payload": payload, **captured}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ROJ11-A16")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")

    from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
    from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

    spec = ROJRATSIRIKUL2011_UNIFIED_CASES[arguments.case]
    runner = RojratsirikulCaseRunner(spec, device=arguments.device)
    runner.build()
    captured = _capture_last_propose(runner, arguments.steps)
    payload = captured["payload"]

    settings = runner.aerodynamic.settings
    nc, ns = int(settings.chordwise_panels), int(settings.spanwise_panels)
    chord, span = float(spec.chord_m), float(spec.span_m)
    density, speed = float(spec.fluid_density_kg_m3), float(spec.freestream_m_s)
    dynamic_pressure = 0.5 * density * speed * speed
    normal = torch.tensor(
        [-math.sin(math.radians(spec.angle_deg)), 0.0, math.cos(math.radians(spec.angle_deg))],
        dtype=torch.float64, device=arguments.device,
    )

    gamma = captured["gamma"].reshape(nc, ns)
    gamma_pre = captured["gamma_pre"][:ns]
    geometry = captured["geometry"]
    scale = captured["scale"]
    areas = geometry.areas.reshape(nc, ns)
    normals = geometry.normals.reshape(nc, ns, 3)
    collocation = geometry.collocation.reshape(nc, ns, 3)
    separated = captured["surface_separated"]

    # ---- leading-edge sign -------------------------------------------------
    sign_attached = torch.sign(gamma_pre)
    sign_pinned = torch.sign(gamma[0])
    separated_count = int(separated.sum())
    opposite = int((separated & (sign_attached * sign_pinned < 0)).sum())

    # ---- chordwise ---------------------------------------------------------
    rings = geometry.rings.reshape(nc, ns, 4, 3)
    tau_x = 0.5 * (rings[:, :, 2] + rings[:, :, 3] - rings[:, :, 0] - rings[:, :, 1])
    tau_x = tau_x / torch.linalg.vector_norm(tau_x, dim=2, keepdim=True)
    dx = torch.linalg.vector_norm(
        0.5 * (rings[:, :, 2] + rings[:, :, 3] - rings[:, :, 0] - rings[:, :, 1]), dim=2
    )
    # the solver's own leading-edge convention (matches the author's MATLAB verbatim)
    dx_gamma = torch.empty_like(gamma)
    dx_gamma[0] = gamma[0] / dx[0]
    dx_gamma[1:] = (gamma[1:] - gamma[:-1]) / dx[1:]
    external = captured["external_flow"].reshape(nc, ns, 3)
    steady = density * torch.sum(external * tau_x * dx_gamma[:, :, None], dim=2)

    def normal_coefficient(pressure: torch.Tensor) -> float:
        force = torch.sum(
            pressure[:, :, None] * areas[:, :, None] * normals, dim=(0, 1)
        )
        return float((force * normal).sum()) / (dynamic_pressure * chord * span)

    steady_mean = steady.mean(dim=1)
    # sawtooth = how many times the SIGN of the span-averaged steady pressure flips
    flips = int((torch.sign(steady_mean[1:]) * torch.sign(steady_mean[:-1]) < 0).sum())
    # peakiness = largest |pressure| divided by the median |pressure| over the interior rows
    interior = steady_mean[2:-1].abs()
    peakiness = float(steady_mean.abs().max() / interior.median().clamp(min=1e-30))

    # ---- spanwise ----------------------------------------------------------
    strip_circulation = gamma.sum(dim=0)
    magnitude = float(strip_circulation.abs().max())
    y_over_b = (collocation[0, :, 1] / span).cpu()
    elliptic = torch.sqrt(
        torch.clamp(1.0 - (2.0 * y_over_b - 1.0) ** 2, min=0.0)
    )
    outer = (y_over_b < 0.1) | (y_over_b > 0.9)
    share = float(strip_circulation[outer].sum() / strip_circulation.sum())
    elliptic_share = float(elliptic[outer].sum() / elliptic.sum())
    tip_over_neighbour = (
        float(strip_circulation[0].abs() / strip_circulation[1].abs())
        if float(strip_circulation[1].abs()) > 0 else float("nan")
    )

    result = {
        "case": arguments.case,
        "steps": arguments.steps,
        "arm": {
            "FLUXV_LE_PIN": os.environ.get("FLUXV_LE_PIN", "legacy"),
            "FLUXV_JOINT_SEPARATION": os.environ.get("FLUXV_JOINT_SEPARATION", "0"),
        },
        "leading_edge": {
            "sign_attached_gamma_pre": sorted(set(sign_attached.tolist())),
            "sign_pinned_gamma_le": sorted(set(sign_pinned.tolist())),
            "separated_strips": separated_count,
            "separated_strips_with_opposite_sign": opposite,
            "gamma_le_mean": float(gamma[0].mean()),
            "scale_mean": float(scale.mean()),
        },
        "chordwise": {
            "x_over_c": [float(v) for v in (collocation[:, 0, 0] / chord).cpu()],
            "gamma_span_mean": [float(v) for v in gamma.mean(dim=1).cpu()],
            "steady_pressure_over_q_span_mean": [float(v) for v in (steady_mean / dynamic_pressure).cpu()],
            "sign_flips_of_span_averaged_pressure": flips,
            "peak_over_interior_median": peakiness,
            "steady_normal_coefficient": normal_coefficient(steady),
        },
        "spanwise": {
            "y_over_b": [float(v) for v in y_over_b],
            "strip_circulation": [float(v) for v in strip_circulation.cpu()],
            "tip_over_neighbour": tip_over_neighbour,
            "outer_20pct_share": share,
            "outer_20pct_share_elliptic": elliptic_share,
        },
        "cn_series_last": float(payload["records"][-1]["cn"]),
        "max_abs_cn": max(abs(float(r["cn"])) for r in payload["records"]),
    }

    print(f"\n=== bound circulation shape: {arguments.case}, {arguments.steps} steps, "
          f"arm {result['arm']} ===")
    edge = result["leading_edge"]
    print(f"leading edge : sign(attached) {edge['sign_attached_gamma_pre']}  "
          f"sign(pinned) {edge['sign_pinned_gamma_le']}  separated {edge['separated_strips']}  "
          f"opposite-sign {edge['separated_strips_with_opposite_sign']}")
    chordwise = result["chordwise"]
    print(f"chordwise    : pressure sign flips {chordwise['sign_flips_of_span_averaged_pressure']}"
          f"   peak/interior-median {chordwise['peak_over_interior_median']:.2f}"
          f"   steady Fn/qS {chordwise['steady_normal_coefficient']:+.5f}")
    print("               " + "  ".join(
        f"{v:+.2f}" for v in chordwise["steady_pressure_over_q_span_mean"]))
    spanwise = result["spanwise"]
    print(f"spanwise     : tip/neighbour {spanwise['tip_over_neighbour']:.3f}"
          f"   outer-20% share {spanwise['outer_20pct_share']:.3f}"
          f" (elliptic {spanwise['outer_20pct_share_elliptic']:.3f})")
    print(f"run          : max|cn| {result['max_abs_cn']:.4f}   last cn {result['cn_series_last']:+.5f}")

    if arguments.out:
        Path(arguments.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
