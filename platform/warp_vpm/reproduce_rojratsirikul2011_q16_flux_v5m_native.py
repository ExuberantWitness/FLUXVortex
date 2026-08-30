"""Thin CLI over the unified Rojratsirikul 2011 CaseRunner.

HANDOFF_UNIFIED_FRAMEWORK §9 P1: this module only parses arguments, calls
``RojratsirikulCaseRunner`` (the unified composition of Q16SurfaceFrameAdapter
/ V5M3DStepper / Q16DynamicsAdapter / PartitionedStrongFSI / WorldOwner /
GlobalTransaction), writes artifacts and propagates ``ResultStatus.exit_code``.
The legacy inline production loop was retired with it; the frozen numerics it
drove are reached through the adapters, never duplicated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fluxvortex.cases.rojratsirikul2011 import (
    ROJRATSIRIKUL2011_SENSITIVITY_BRANCHES,
    ROJRATSIRIKUL2011_UNIFIED_CASES,
)
from fluxvortex.runtime.case_runner import (
    EXECUTION_GATE_STEPS,
    RojratsirikulCaseRunner,
    apply_freestream as _apply_freestream,  # re-exported for the frozen tests
)

UNIFIED_CASES = {
    **ROJRATSIRIKUL2011_UNIFIED_CASES,
    **ROJRATSIRIKUL2011_SENSITIVITY_BRANCHES,
}
DEFAULT_OUTPUT_DIR = (
    "artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current"
)


def run_case(
    *,
    case_id: str,
    max_aero_steps: int | None = None,
    execution_gate_only: bool = False,
    structural_substeps: int | None = None,
    young_modulus_override: float | None = None,
    damping_loss_factor: float | None = None,
    output: Path | None = None,
) -> dict:
    """Delegate to the unified CaseRunner (compatibility entry point)."""

    spec = UNIFIED_CASES.get(case_id)
    if spec is None:
        raise ValueError(
            f"unknown case {case_id}; use one of {sorted(UNIFIED_CASES)}"
        )
    if young_modulus_override is not None:
        # Labeled material-uncertainty branch (handoff §3.3); the E=2.2 MPa
        # registry entry stays the primary result.
        spec = spec.with_material_branch(
            young_modulus_pa=float(young_modulus_override),
            branch=(
                "post_hoc_calibrated_sensitivity"
                f"_E{young_modulus_override:g}"
            ),
            case_id=f"{case_id}-E{young_modulus_override:g}",
        )
    runner = RojratsirikulCaseRunner(
        spec,
        structural_substeps=structural_substeps,
        damping_loss_factor=damping_loss_factor,
    )
    runner.build()
    return runner.run(
        max_aero_steps=max_aero_steps,
        execution_gate_only=execution_gate_only,
        output=output,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="ROJ11-A16",
        choices=sorted(UNIFIED_CASES),
    )
    parser.add_argument("--max-aero-steps", type=int, default=None)
    parser.add_argument(
        "--execution-gate-only",
        action="store_true",
        help=f"run the short execution gate ({EXECUTION_GATE_STEPS} aero "
        "steps by default) on the same formal grid; no accuracy claims",
    )
    parser.add_argument(
        "--structural-substeps",
        type=int,
        default=None,
        help="DIAGNOSTIC ONLY: override the frozen structural substep count; "
        "the frozen protocol value is always recorded alongside",
    )
    parser.add_argument(
        "--young-modulus-override",
        type=float,
        default=None,
        help="labeled material-uncertainty branch (handoff §3.3); the "
        "E=2.2 MPa run remains the primary result",
    )
    parser.add_argument(
        "--damping-loss-factor",
        type=float,
        default=None,
        help="labeled damping branch (handoff §3.3); default is the frozen "
        "literature eta=0.1, eta=0 gives the undamped structural baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    arguments = parser.parse_args()
    output = arguments.output or Path(
        f"{DEFAULT_OUTPUT_DIR}/"
        f"{arguments.case}"
        f"{'_EXECUTION_GATE' if arguments.execution_gate_only else '_FULL'}.json"
    )
    result = run_case(
        case_id=arguments.case,
        max_aero_steps=arguments.max_aero_steps,
        execution_gate_only=arguments.execution_gate_only,
        structural_substeps=arguments.structural_substeps,
        young_modulus_override=arguments.young_modulus_override,
        damping_loss_factor=arguments.damping_loss_factor,
        output=output,
    )
    # M0-1 (MODIFICATION_PLAN_ROJ_ACCURACY_PERFORMANCE_20260830): the
    # solver-execution status owns the CLI exit code; post-processing /
    # artifact failures are reported with a SEPARATE non-zero code so a
    # successful solve is never misread as a failed one.
    try:
        printable = dict(result)
        for key in (
            "records",
            "mean_map",
            "zsd_map",
            "assumption_ledger",
            "field_roles",
        ):
            printable.pop(key, None)
        window = printable.get("window_selection")
        if isinstance(window, dict):  # None on degenerate short slices
            printable["window_selection"] = {
                key: value
                for key, value in window.items()
                if key != "candidates"
            }
        else:
            printable["window_selection"] = window  # None serialized as null
        print(json.dumps(printable, indent=2, sort_keys=True, allow_nan=False))
    except Exception as error:  # noqa: BLE001 -- artifact layer only
        print(
            json.dumps(
                {
                    "artifact_status": "failed",
                    "solver_status": result.get("result_status", {}),
                    "artifact_error": f"{type(error).__name__}: {error}",
                },
                indent=2,
            )
        )
        # Solver completed -> keep its own verdict distinct (exit 3 = the
        # solve itself is fine but the artifact/report layer failed).
        solver_exit = int(result.get("result_status", {}).get("exit_code", 1))
        return 3 if solver_exit == 0 else solver_exit
    return int(result["result_status"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
