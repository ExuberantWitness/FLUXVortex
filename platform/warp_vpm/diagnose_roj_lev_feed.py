"""What the leading edge actually feeds into the LEV, step by step.

`particle_circulation_sum` is exactly zero for every full run (A16, 2100 steps:
1.7e-21 -> -1.1e-16) and that number carries NO information: every deposit in
this solver -- LEV sheet and TEV ribbon alike -- is a closed loop, and a closed
loop has zero net vector moment by construction.  So the question "is the
leading edge feeding the LEV at all, and with what strength?" cannot be answered
from the stored payloads.

`propose()` keeps it in two locals: `gamma_lev` (the per-strip circulation this
step deposits, m^2/s) and the accumulated `trial.lev_strip_released_total`.
This script reads them via `sys.setprofile` at the return of
`_deposit_dvm_ribbon`, so the solver is not modified, and prints the feeding
law against the bound leading-edge circulation.

Why it matters (research note A14)
----------------------------------
Hirato et al. (2019) feed the LEV sheet with a strength that is CONSTANT along
the arc and equal to the forward-most bound leg Gamma_1, tied to A0/LESP_crit --
the sheet keeps being fed for as long as the LESP stays above critical.  If our
`gamma_lev` is instead an increment (Gamma_b^n - Gamma_b^{n-1}) it tends to zero
as the state settles, so fixing the sheet GEOMETRY would change nothing: there
would be no vorticity to give an arc to.

Usage
-----
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 \\
        python3 platform/warp_vpm/diagnose_roj_lev_feed.py --case ROJ11-A16 \\
        --steps 40 --out /tmp/lev_feed.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ROJ11-A16")
    parser.add_argument("--steps", type=int, default=40)
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

    rows: list[dict[str, object]] = []
    sheet: list[dict[str, object]] = []

    def grab_sheet(frame, event, arg):
        """Aft edge of the newborn sheet, in chord fractions.

        On the production arm (`FLUXV_LE_SHEET_EXTENT=off`) this is the depth of the
        deposit: 0.0054 chord = 0.08 panel on A16, which is the sliver whose two legs
        cancel (note A14).  With `FLUXV_LE_SHEET_EXTENT=chord` the chordwise reach is
        replaced by a fixed number of panels, and this is the number that proves the
        flag lands there rather than where a units slip put it.  `reference_node_chord`
        is the FULL chord, hence the ratio is directly in chord fractions.
        """

        if (
            event == "return"
            and frame.f_code.co_name == "_lev_sheet_frontier"
            and isinstance(arg, torch.Tensor)
        ):
            local = frame.f_locals
            geometry = local.get("geometry")
            trial = local.get("trial")
            node_x = local.get("node_x")
            node_normal = local.get("node_normal")
            if geometry is not None and trial is not None and node_x is not None:
                offset = arg - geometry.leading_edge
                chord_ref = trial.reference_node_chord
                sheet.append({
                    "step": int(trial.step),
                    "chordwise_over_chord": float(
                        (torch.sum(offset * node_x, dim=1) / chord_ref).max().item()
                    ),
                    "normal_over_chord": float(
                        (torch.sum(offset * node_normal, dim=1) / chord_ref).max().item()
                    ),
                    "aft_edge_over_chord": float(
                        (torch.linalg.vector_norm(offset, dim=1) / chord_ref).max().item()
                    ),
                })

    def grab(frame, event, arg):
        grab_sheet(frame, event, arg)
        if (
            event == "return"
            and frame.f_code.co_name == "_deposit_dvm_ribbon"
            and frame.f_code.co_filename.endswith("q16_flux_v5m_native.py")
        ):
            local = frame.f_locals
            gamma_lev = local.get("gamma_lev")
            trial = local.get("trial")
            result = local.get("result")
            if gamma_lev is None or trial is None:
                return grab
            released = result.get("shed_lev") if isinstance(result, dict) else None
            total = getattr(trial, "lev_strip_released_total", None)
            rows.append({
                "step": int(trial.step),
                "strips": int(gamma_lev.numel()),
                "lev_sum": float(gamma_lev.sum().item()),
                "lev_abs_sum": float(gamma_lev.abs().sum().item()),
                "lev_max_abs": float(gamma_lev.abs().max().item()),
                "lev_signs": int(torch.sign(gamma_lev).sum().item()),
                "cells_released": int(released.sum().item()) if released is not None else None,
                "released_total_abs": float(total.abs().sum().item()) if total is not None else None,
            })
        return grab

    sys.setprofile(grab)
    try:
        payload = runner.run(max_aero_steps=arguments.steps)
    finally:
        sys.setprofile(None)

    geometry = runner.aerodynamic.surface.evaluate(
        runner.owner.state, runner.owner.velocity
    )
    settings = runner.aerodynamic.settings
    nc, ns = int(settings.chordwise_panels), int(settings.spanwise_panels)
    gamma_bound = runner.owner.aerodynamic.state.gamma_bound.reshape(nc, ns)
    pin_value = float(gamma_bound[0].abs().mean().item())

    print(f"\n=== LEV feeding law: {arguments.case}, {arguments.steps} steps, "
          f"ns={ns}, nc={nc} ===")
    print(f"bound leading-edge row mean |Gamma| = {pin_value:.5f} m^2/s "
          f"(this is what Hirato's Gamma_1 feeds the sheet with)")
    print(f"{'step':>5} {'released':>8} {'sum':>11} {'sum|.|':>11} {'max|.|':>10} "
          f"{'sign':>5} {'sum/pin':>8}")
    for index, row in enumerate(rows):
        if index not in (0, 1, 2, 3, 4, 9, 19, 29, len(rows) - 1):
            continue
        ratio = row["lev_sum"] / pin_value if pin_value else float("nan")
        print(f"{row['step']:>5} {str(row['cells_released']):>8} {row['lev_sum']:>+11.5f} "
              f"{row['lev_abs_sum']:>11.5f} {row['lev_max_abs']:>10.5f} "
              f"{row['lev_signs']:>+5} {ratio:>+8.3f}")

    if rows:
        tail = rows[len(rows) // 2:]
        tail_sum = sum(r["lev_sum"] for r in tail) / max(len(tail), 1)
        tail_abs = sum(r["lev_abs_sum"] for r in tail) / max(len(tail), 1)
        print(f"\ntail mean (second half): sum {tail_sum:+.5f}  sum|.| {tail_abs:.5f}  "
              f"sum/pin {tail_sum / pin_value if pin_value else float('nan'):+.4f}")
        print("  Hirato feeds with Gamma_1 (~= the pin value, ratio +-1); a ratio near 0 "
              "means the sheet is fed by an INCREMENT that decays as the state settles.")

    print(f"\n=== newborn sheet placement (aft edge, fraction of chord) ===")
    print(f"{'step':>5} {'chordwise':>10} {'normal':>9} {'|offset|':>9}")
    seen: set[int] = set()
    for row in sheet:
        if row["step"] in seen:
            continue
        seen.add(row["step"])
        print(f"{row['step']:>5} {row['chordwise_over_chord']:>10.5f} "
              f"{row['normal_over_chord']:>9.5f} {row['aft_edge_over_chord']:>9.5f}")
    if sheet:
        tail = sheet[-min(len(sheet), 30):]
        print(f"  mean aft-edge chordwise extent: "
              f"{sum(r['chordwise_over_chord'] for r in tail) / len(tail):.5f} c "
              f"(= {sum(r['chordwise_over_chord'] for r in tail) / len(tail) * 15:.2f} panels at nc=15)")

    payload_fields = {
        "case": arguments.case, "steps": arguments.steps,
        "pin_value_mean_abs": pin_value, "rows": rows, "sheet": sheet,
    }
    if arguments.out:
        Path(arguments.out).write_text(
            json.dumps(payload_fields, indent=1) + "\n", encoding="utf-8"
        )
        print(f"wrote {arguments.out}")
    del geometry
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
