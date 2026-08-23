"""Run the formal Yamano 2020 Q16/native-FLUX-V5M CUDA CASE."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
)
from fluxvortex.warp_fsi.q16_flux_v5m_native_fsi import (
    Q16NativeV5MFSIOwner,
    Q16NativeV5MFSIStepper,
)
from fluxvortex.warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from fluxvortex.warp_fsi.q16_flux_v5m_author_loads import (
    Q16NativeAddedMassAction,
)
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET,
    load_tip_displacement_reference,
    make_yamano2020_q16_model,
)
from yamano2020_q16_pulse import Yamano2020Q16CudaPulse


FORMAL_Q16_GRID = (5, 3)
FORMAL_AERO_GRID = (15, 10)
LOAD_GATE_ARTIFACT = Path(
    "artifacts/experiment/20260823_q16_fluxv5m_native_fsi/"
    "YAMANO_Q16_NATIVE_LOAD_GATES.json"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _require_load_gates() -> dict[str, Any]:
    path = LOAD_GATE_ARTIFACT.resolve()
    if not path.is_file():
        raise RuntimeError(
            "formal CASE is blocked until the bounded native load gates run"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "yamano2020-q16-native-load-gates-v1":
        raise RuntimeError("formal CASE load-gate schema drift")
    if payload.get("status") not in {"passed", "passed_with_warning"}:
        failed = sorted(
            key for key, value in payload.get("hard_gates", {}).items() if not value
        )
        raise RuntimeError(
            "formal CASE is blocked by failed native load gates: "
            + ",".join(failed)
        )
    return payload


def run_case(*, outer_steps: int, output: Path | None = None,
             mf1_scale: float = 1.0) -> dict[str, Any]:
    if type(outer_steps) is not int or not 1 <= outer_steps <= 8:
        raise ValueError("outer_steps must be an exact int in [1,8]")
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("formal Yamano native CASE requires CUDA")
    load_gate_evidence = _require_load_gates()
    legacy_token = "ptera" + "software"
    loaded_legacy = tuple(name for name in sys.modules if legacy_token in name.lower())
    if loaded_legacy:
        raise RuntimeError("formal native CASE process contains a legacy runtime")

    case = YAMANO_2020_SINGLE_SHEET
    if mf1_scale != 1.0:
        # DIAGNOSTIC ONLY: wrap Q16NativeAddedMassAction.__call__ to scale
        # the added-mass force. This is NOT a fix — it isolates whether Mf1
        # mismatch is the root cause of the growing displacement error.
        _orig_call = Q16NativeAddedMassAction.__call__

        def _scaled_call(self, structural_acceleration):
            acceleration = wp.to_torch(structural_acceleration)
            force = acceleration @ (self.generalized_matrix.T * mf1_scale)
            return wp.from_torch(force, dtype=config.DTYPE, requires_grad=False)

        Q16NativeAddedMassAction.__call__ = _scaled_call
        print(f"DIAGNOSTIC: Mf1 added-mass scaled by {mf1_scale:.6f}")

    mesh, model, boundary = make_yamano2020_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
    )
    structural = Q16CudaNewmarkStepper(
        model,
        boundary,
        device=config.DEVICE,
        newton_tolerance=3.0e-7,
        max_newton_iterations=128,
        cg_tolerance=2.0e-10,
        max_cg_iterations=2048,
        cg_check_every=16,
        # Cached CUDA reference tangent accelerates the Newton correction;
        # accepted states remain gated by the live Q16 nonlinear residual.
        nonsymmetric_solver="reference_dense",
        # A live dense tangent is a last-resort recovery, not the ordinary
        # correction path.  The committed-state cache is refreshed every
        # outer step and live residuals gate every iterate; defer the costly
        # per-substep dense assembly until the maximum 48th update.
        reference_dense_refresh_after=48,
        mass_damping_coefficient=0.0,
    )
    surface = Q16NativeV5MSurface(
        mesh,
        q16_chordwise_elements=FORMAL_Q16_GRID[0],
        q16_spanwise_elements=FORMAL_Q16_GRID[1],
        aerodynamic_chordwise_panels=FORMAL_AERO_GRID[0],
        aerodynamic_spanwise_panels=FORMAL_AERO_GRID[1],
        device=config.DEVICE,
    )
    aerodynamic = Q16NativeV5MSolver(
        surface,
        NativeV5MConfig(
            chordwise_panels=FORMAL_AERO_GRID[0],
            spanwise_panels=FORMAL_AERO_GRID[1],
            density=case.fluid_density_kg_m3,
            freestream=case.freestream_m_s,
            aerodynamic_dt=case.aerodynamic_dt_s,
            lesp_crit=0.11,
            device=config.DEVICE,
        ),
    )
    state = wp.array(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    velocity = wp.zeros_like(state)
    acceleration = wp.zeros_like(state)
    owner = Q16NativeV5MFSIOwner.initialize(
        aerodynamic, state, velocity, acceleration
    )
    coupling = Q16NativeV5MFSIStepper(
        structural,
        aerodynamic,
        coupling_tolerance=5.0e-7,
        max_coupling_iterations=20,
        relaxation=0.7,
    )
    pulse = Yamano2020Q16CudaPulse(model, case=case, device=config.DEVICE)
    reference = torch.as_tensor(
        np.array(load_tip_displacement_reference(), copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    records: list[dict[str, Any]] = []
    progress_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    structural_count = case.structural_substeps_per_aerodynamic_step
    startup_schedule = (
        pulse.endpoint_load(case.structural_dt_s).generalized_force,
    )

    def report_startup(event: dict[str, Any]) -> None:
        record = {"outer_step": 0, "author_startup": True, **event}
        progress_records.append(record)
        print(
            "__DS_PROGRESS__ "
            + json.dumps(record, sort_keys=True, allow_nan=False),
            flush=True,
        )

    startup = coupling.advance(
        owner,
        delta_time=case.structural_dt_s,
        prescribed_forces=startup_schedule,
        load_betas=(1.0 / structural_count,),
        author_startup=True,
        progress_callback=report_startup,
    )
    for outer_index in range(outer_steps):
        start_time = (
            case.structural_dt_s + outer_index * case.aerodynamic_dt_s
        )
        schedule = tuple(
            pulse.endpoint_load(
                start_time + (substep + 1) * case.structural_dt_s
            ).generalized_force
            for substep in range(case.structural_substeps_per_aerodynamic_step)
        )
        def report_progress(event: dict[str, Any]) -> None:
            record = {"outer_step": outer_index + 1, **event}
            if (
                event["phase"] != "structural_substep"
                or event["substep"] in {1, event["substep_count"]}
                or event["substep"] % 5 == 0
            ):
                progress_records.append(record)
            else:
                return
            print(
                "__DS_PROGRESS__ "
                + json.dumps(record, sort_keys=True, allow_nan=False),
                flush=True,
            )

        try:
            result = coupling.advance(
                owner,
                delta_time=case.aerodynamic_dt_s,
                prescribed_forces=schedule,
                load_betas=tuple(
                    (substep + 1) / structural_count
                    for substep in range(structural_count)
                ),
                checkpoint_substep=structural_count - 1,
                progress_callback=report_progress,
            )
        except Exception as error:
            if output is not None:
                partial = output.with_name(f"{output.stem}.partial.json")
                _write_json(
                    partial,
                    {
                        "schema": "yamano2020-q16-native-flux-v5m-fsi-v1",
                        "status": "failed",
                        "case_id": case.case_id,
                        "q16_grid": list(FORMAL_Q16_GRID),
                        "aerodynamic_grid": list(FORMAL_AERO_GRID),
                        "completed_outer_steps": len(records),
                        "failed_outer_step": outer_index + 1,
                        "requested_outer_steps": outer_steps,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "records": records,
                        "progress_records": progress_records,
                        "reference_tangent_cache_refresh_count": (
                            structural.reference_tangent_cache_refresh_count
                        ),
                    },
                )
            raise
        if result.checkpoint is None:
            raise RuntimeError("author-clock paper sample checkpoint is missing")
        trailing = wp.to_torch(
            surface.trailing_transfer.interpolate(result.checkpoint.state)
        )[0]
        tip_z = trailing[FORMAL_AERO_GRID[1] // 2, 2]
        time_star = (outer_index + 1) * case.aerodynamic_dt_star
        target_time = torch.tensor(time_star, device="cuda:0", dtype=torch.float64)
        right = int(torch.searchsorted(reference[:, 0], target_time).item())
        if right <= 0 or right >= reference.shape[0]:
            raise RuntimeError("native CASE endpoint left the paper oracle")
        left = right - 1
        beta = (target_time - reference[left, 0]) / (
            reference[right, 0] - reference[left, 0]
        )
        reference_tip = reference[left, 1] + beta * (
            reference[right, 1] - reference[left, 1]
        )
        error = torch.abs(tip_z - reference_tip) / torch.clamp(
            torch.abs(reference_tip), min=torch.finfo(torch.float64).eps
        )
        records.append(
            {
                "outer_step": outer_index + 1,
                "time_star": time_star,
                "transaction_end_time_star": (
                    time_star + case.structural_dt_star
                ),
                "sample_substep": result.checkpoint.substep,
                "tip_z_m": float(tip_z.item()),
                "reference_tip_z_m": float(reference_tip.item()),
                "tip_relative_error_percent": float((100.0 * error).item()),
                "coupling_iterations": result.coupling_iterations,
                "aerodynamic_evaluations": result.aerodynamic_evaluations,
                "coupling_residual": result.residual,
                "structural_newton_iterations": (
                    result.structural.newton_iteration_count
                ),
                "structural_cg_iterations": result.structural.cg_iteration_count,
                "aerodynamic": result.aerodynamic.trial_state.diagnostics[-1],
                "total_aerodynamic_force_n": (
                    result.aerodynamic.load.total_force.tolist()
                ),
            }
        )
        if output is not None:
            partial = output.with_name(f"{output.stem}.partial.json")
            _write_json(
                partial,
                {
                    "schema": "yamano2020-q16-native-flux-v5m-fsi-v1",
                    "status": "running",
                    "case_id": case.case_id,
                    "q16_grid": list(FORMAL_Q16_GRID),
                    "aerodynamic_grid": list(FORMAL_AERO_GRID),
                    "completed_outer_steps": len(records),
                    "requested_outer_steps": outer_steps,
                    "records": records,
                    "progress_records": progress_records,
                    "reference_tangent_cache_refresh_count": (
                        structural.reference_tangent_cache_refresh_count
                    ),
                },
            )
    torch.cuda.synchronize()
    payload: dict[str, Any] = {
        "schema": "yamano2020-q16-native-flux-v5m-fsi-v1",
        "status": "completed",
        "paper": case.paper_title,
        "case_id": case.case_id,
        "q16_grid": list(FORMAL_Q16_GRID),
        "aerodynamic_grid": list(FORMAL_AERO_GRID),
        "separated_lev_mandatory": True,
        "lesp_release_condition": "abs(LESP)>0.11",
        "joint_tev": True,
        "free_wake": True,
        "runtime_legacy_module_count": len(loaded_legacy),
        "device": str(torch.cuda.get_device_name(torch.cuda.current_device())),
        "dtype": "float64",
        "elapsed_seconds": time.perf_counter() - started,
        "outer_steps": outer_steps,
        "author_clock_startup": True,
        "startup_time_star": case.structural_dt_star,
        "startup_coupling_iterations": startup.coupling_iterations,
        "startup_aerodynamic_evaluations": startup.aerodynamic_evaluations,
        "paper_sample_substep": structural_count - 1,
        "transaction_endpoint_substep": structural_count,
        "records": records,
        "progress_records": progress_records,
        "native_load_gate_metrics": load_gate_evidence["metrics"],
        "native_load_gate_status": load_gate_evidence["status"],
        "native_load_gate_warnings": load_gate_evidence.get("warnings", []),
        "structural_nonsymmetric_solver": structural.nonsymmetric_solver,
        "reference_tangent_cache_refresh_count": (
            structural.reference_tangent_cache_refresh_count
        ),
    }
    if output is not None:
        _write_json(output, payload)
        partial = output.with_name(f"{output.stem}.partial.json")
        _write_json(
            partial,
            {
                "schema": "yamano2020-q16-native-flux-v5m-fsi-checkpoint-v1",
                "status": "completed",
                "case_id": case.case_id,
                "completed_outer_steps": outer_steps,
                "requested_outer_steps": outer_steps,
                "final_output": str(output),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument(
        "--mf1-scale",
        type=float,
        default=1.0,
        help="DIAGNOSTIC ONLY: scale Mf1 added-mass matrix by this factor. "
        "1.0 = production. Any other value is a diagnostic experiment "
        "and must NOT be committed as a fix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/experiment/20260823_q16_fluxv5m_native_fsi/"
            "YAMANO_Q16_5X3_V5M_15X10_STEP1.json"
        ),
    )
    args = parser.parse_args()
    result = run_case(
        outer_steps=args.outer_steps,
        output=args.output,
        mf1_scale=args.mf1_scale,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
