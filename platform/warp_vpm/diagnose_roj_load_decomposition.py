"""⑬ precondition: is the 0.28-0.52 load discrepancy a QUADRATURE ERROR or a DECOMPOSITION MISMATCH?

The recorded `projection_quadrature_discrepancy` compares the FULL instantaneous 5P
resolved packet against the legacy **constant-only** generalized force.  Production
additionally applies `velocity_force(qdot)` and the Mf1 added-mass action inside the
structural operator, so the comparison is not like-for-like.  Whether ⑬ (a
decomposition-preserving migration) is feasible at all depends on the relative size of
those missing components:

  * if (|velocity| + |mf1 action|) / |constant| is of the SAME order as the recorded
    discrepancy, the discrepancy is mostly the unsteady terms being absent from the
    comparison -- i.e. a decomposition mismatch, and migrating all three slots 1:1 is
    the correct fix;
  * if they are far smaller, the discrepancy is a genuine quadrature/pressure error and
    migrating the constant term alone would not close it.

Read-only: runs the case through the unified runner and prints the
`resolved_load_evidence["shadow_5p_to_q16"]` ledger that the runner already records.

Usage:
  PYTHONPATH=src:platform:platform/warp_vpm python3 \
    platform/warp_vpm/diagnose_roj_load_decomposition.py --case ROJ11-A16 --steps 3
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ROJ11-A16")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")

    spec = ROJRATSIRIKUL2011_UNIFIED_CASES[args.case]
    runner = RojratsirikulCaseRunner(spec, device=args.device)
    runner.build()
    payload = runner.run(max_aero_steps=args.steps)
    ledger = (payload.get("resolved_load_evidence") or {}).get("shadow_5p_to_q16") or {}
    if not ledger:
        print("no shadow evidence recorded (cadence/last-step sampling missed)")
        return 1

    norms = ledger.get("legacy_decomposition_norms") or {}
    constant = (norms.get("legacy_constant_norm") or {}).get("mean")
    velocity = (norms.get("legacy_velocity_norm") or {}).get("mean")
    mf1 = (norms.get("legacy_mf1_action_norm") or {}).get("mean")
    discrepancy = (
        ledger.get("fiveP_instantaneous_vs_legacy_constant_only_relative") or {}
    ).get("mean")

    print(f"case={args.case} steps={args.steps} shadow samples={ledger.get('samples_time_star')}")
    print(f"  legacy |constant|        = {constant}")
    print(f"  legacy |velocity_force|  = {velocity}")
    print(f"  legacy |Mf1 action|      = {mf1}")
    print(f"  recorded 5P-vs-constant  = {discrepancy}")
    if constant and velocity is not None and mf1 is not None:
        missing = (velocity + mf1) / constant
        print(f"  (|velocity|+|Mf1|)/|constant| = {missing:.4f}")
        if discrepancy:
            print(
                "  => the unsteady terms are "
                + (
                    "of the same order as the discrepancy: it is a DECOMPOSITION "
                    "mismatch, migrate all three slots"
                    if missing > 0.3 * discrepancy
                    else "much smaller than the discrepancy: it is a QUADRATURE error, "
                    "migrating the constant slot alone will not close it"
                )
            )
    three_term = ledger.get("packet_full_vs_legacy_three_term_relative")
    if three_term is not None:
        value = three_term if isinstance(three_term, (int, float)) else three_term.get("mean")
        print(f"  |packet - (const+vel+Mf1)| / |packet| = {value}")
        print("   => " + ("the two tracks AGREE (residual << the recorded discrepancy): the "
                         "0.28-0.52 discrepancy is a comparison artefact, migration unnecessary"
                         if value is not None and value < 0.05 else
                         "the tracks still differ: a decomposition-preserving migration is needed"))
    print(f"  transfer force closure  = {ledger.get('transfer_force_max_abs_error')}")
    print(f"  work conjugacy          = {ledger.get('work_pointwise_relative_error')}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(ledger, handle, indent=1, default=str)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
