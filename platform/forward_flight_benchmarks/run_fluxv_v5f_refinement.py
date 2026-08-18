"""Run the observation-free FluxV v5f M5 refinement gate.

The run fixes one Yang-2025 movement and advances it to the same physical
half-cycle with four temporal resolutions and all three preregistered Hirato
Eq. 25 core ratios.  It never opens an experimental load file and never scores
a paper metric.  Its only purpose is to decide whether the native material-LEV
time marcher is sufficiently bounded to permit later paper evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import pterasoftware as ps

from forward_flight_benchmarks.cases import YANG_2025
from forward_flight_benchmarks.fluxv_v5f_native_solver import (
    REGISTERED_CORE_RADIUS_RATIOS,
    NativeMaterialLEVTimeMarchConfig,
    NativeMaterialLEVTimeMarchSolver,
    make_fluxv_v5f_native_time_march_solver,
)
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement
from forward_flight_benchmarks.run_provenance import (
    collect_run_provenance,
    collect_source_hashes,
    prepare_output_directory,
    sha256_file,
)


REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DOC_ROOT: Final = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "fluxv_v5c_nextgen_20260814"
)
DEFAULT_OUTPUT: Final = DOC_ROOT / "runs/20260814_fluxv_v5f_m5_refinement"
STEPS_PER_CYCLE: Final = (20, 40, 80, 160)
CHORDWISE_PANELS: Final = 2
SPANWISE_PANELS: Final = 4
LESP_CRITICAL: Final = float(np.sin(np.deg2rad(5.0)))
HORIZON_CYCLES: Final = 0.5
NO_PENETRATION_TOLERANCE: Final = 1.0e-10
LESP_TOLERANCE: Final = 1.0e-10
ONE_OVER_DT_GROWTH_LIMIT: Final = 2.0


class _StopAtPhysicalHorizon(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty refinement CSV")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _source_paths() -> tuple[Path, ...]:
    benchmark = REPO_ROOT / "platform/forward_flight_benchmarks"
    return (
        Path(__file__).resolve(),
        benchmark / "fluxv_v5f_native_solver.py",
        benchmark / "fluxv_v5f_native_lev.py",
        benchmark / "fluxv_v5f_native_load.py",
        benchmark / "fluxv_v5f_live_load_adapter.py",
        benchmark / "fluxv_v5f_material_state.py",
        benchmark / "ptera_adapter.py",
        benchmark / "cases.py",
        benchmark / "run_provenance.py",
        REPO_ROOT / "platform/claim_runtime/hirato_equations.py",
        REPO_ROOT / "platform/claim_runtime/hirato_shadow.py",
        REPO_ROOT / "src/fluxvortex/solver.py",
        REPO_ROOT / "src/fluxvortex/particles.py",
        REPO_ROOT / "src/fluxvortex/kernel.py",
        REPO_ROOT / "pyproject.toml",
    )


def _run_configuration(
    *,
    steps_per_cycle: int,
    core_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    movement, movement_manifest = build_yang2025_movement(
        15.0,
        "smoke",
        settings=(
            CHORDWISE_PANELS,
            SPANWISE_PANELS,
            steps_per_cycle,
            1,
            1,
        ),
    )
    problem = ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )
    solver = make_fluxv_v5f_native_time_march_solver(
        problem,
        config=NativeMaterialLEVTimeMarchConfig(
            enabled=True,
            lesp_critical=LESP_CRITICAL,
            core_radius_ratio=core_ratio,
        ),
        max_particles=100_000,
        stretch=False,
        free_wake=False,
    )
    if not isinstance(solver, NativeMaterialLEVTimeMarchSolver):
        raise TypeError("finite v5f refinement did not construct its native solver")

    horizon_steps = int(round(HORIZON_CYCLES * steps_per_cycle))
    original_wake_hook = solver._calculate_wake_wing_influences

    def stop_at_horizon() -> None:
        if int(solver._current_step) >= horizon_steps:
            raise _StopAtPhysicalHorizon
        original_wake_hook()

    solver._calculate_wake_wing_influences = stop_at_horizon  # type: ignore[method-assign]
    try:
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    except _StopAtPhysicalHorizon:
        pass
    if len(solver.v5f_step_reports) != horizon_steps:
        raise RuntimeError("v5f refinement did not reach the frozen physical horizon")
    if solver.v5f_poisoned:
        raise RuntimeError("v5f refinement solver became poisoned")

    uc = YANG_2025.freestream_m_s * YANG_2025.chord_m
    per_step: list[dict[str, Any]] = []
    for report in solver.v5f_step_reports:
        q_abs_max = float(np.max(np.abs(report.q_by_strip_m2_s), initial=0.0))
        condition = float(np.linalg.cond(report.augmented_solution.augmented_matrix))
        per_step.append(
            {
                "steps_per_cycle": steps_per_cycle,
                "delta_time_s": float(solver.delta_time),
                "core_ratio": core_ratio,
                "step": int(report.step),
                "phase_t_over_T": float(report.step / steps_per_cycle),
                "active_strip_count": int(np.count_nonzero(report.active)),
                "material_ring_count": int(report.material_snapshot.gamma_m2_s.size),
                "max_abs_q_m2_s": q_abs_max,
                "q_l2_m2_s": float(np.linalg.norm(report.q_by_strip_m2_s)),
                "max_abs_q_over_Uc": float(q_abs_max / uc),
                "augmented_condition_number": condition,
                "no_penetration_max_abs": (
                    report.augmented_solution.no_penetration_max_abs
                ),
                "lesp_constraint_max_abs": (
                    report.augmented_solution.lesp_constraint_max_abs
                ),
                "eq9_max_abs_m2_s": float(report.eq9_max_abs_m2_s),
                "material_geometry_finite": bool(
                    np.all(np.isfinite(report.material_snapshot.ptera_rings_gp1_m))
                ),
                "force_scoring_status": report.force_scoring_status,
                "canonical_eligible": False,
            }
        )

    snapshot = solver.material_lev_state.snapshot()
    q_max = max(float(row["max_abs_q_m2_s"]) for row in per_step)
    aggregate = {
        "steps_per_cycle": steps_per_cycle,
        "delta_time_s": float(solver.delta_time),
        "core_ratio": core_ratio,
        "horizon_steps": horizon_steps,
        "horizon_cycles": HORIZON_CYCLES,
        "active_step_count": sum(
            int(int(row["active_strip_count"]) > 0) for row in per_step
        ),
        "final_material_ring_count": int(snapshot.gamma_m2_s.size),
        "max_abs_q_m2_s": q_max,
        "max_abs_q_over_Uc": float(q_max / uc),
        "max_augmented_condition_number": max(
            float(row["augmented_condition_number"]) for row in per_step
        ),
        "max_no_penetration_residual": max(
            float(row["no_penetration_max_abs"]) for row in per_step
        ),
        "max_lesp_constraint_residual": max(
            float(row["lesp_constraint_max_abs"]) for row in per_step
        ),
        "max_eq9_residual_m2_s": max(
            float(row["eq9_max_abs_m2_s"]) for row in per_step
        ),
        "all_material_geometry_finite": all(
            bool(row["material_geometry_finite"]) for row in per_step
        ),
        "movement_manifest": movement_manifest,
        "canonical_eligible": False,
    }
    return per_step, aggregate


def evaluate_m5_refinement(
    aggregate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the preregistered no-``1/dt`` and core-family stop rules."""

    lookup = {
        (int(row["steps_per_cycle"]), float(row["core_ratio"])): row
        for row in aggregate_rows
    }
    expected = {
        (steps, float(core))
        for steps in STEPS_PER_CYCLE
        for core in REGISTERED_CORE_RADIUS_RATIOS
    }
    if set(lookup) != expected:
        raise ValueError("aggregate rows do not cover the frozen M5 matrix")

    core_gates: list[dict[str, Any]] = []
    for core in REGISTERED_CORE_RADIUS_RATIOS:
        coarse = float(lookup[(80, float(core))]["max_abs_q_m2_s"])
        fine = float(lookup[(160, float(core))]["max_abs_q_m2_s"])
        if coarse <= 0.0 or not (math.isfinite(coarse) and math.isfinite(fine)):
            raise ValueError("fine-level q maxima must be positive and finite")
        growth = fine / coarse
        gate = growth < ONE_OVER_DT_GROWTH_LIMIT
        core_gates.append(
            {
                "core_ratio": float(core),
                "max_q_80_m2_s": coarse,
                "max_q_160_m2_s": fine,
                "fine_growth_factor": growth,
                "half_dt_one_over_dt_exclusion_pass": gate,
            }
        )

    algebra_pass = all(
        bool(row["all_material_geometry_finite"])
        and float(row["max_no_penetration_residual"]) <= NO_PENETRATION_TOLERANCE
        and float(row["max_lesp_constraint_residual"]) <= LESP_TOLERANCE
        and float(row["max_eq9_residual_m2_s"]) <= 1.0e-12
        for row in aggregate_rows
    )
    no_one_over_dt_pass = all(
        bool(row["half_dt_one_over_dt_exclusion_pass"]) for row in core_gates
    )
    core_family_consistent = (
        len({bool(row["half_dt_one_over_dt_exclusion_pass"]) for row in core_gates})
        == 1
    )
    m5_pass = algebra_pass and no_one_over_dt_pass and core_family_consistent
    return {
        "algebra_and_finite_pass": algebra_pass,
        "core_gates": core_gates,
        "no_one_over_dt_growth_pass": no_one_over_dt_pass,
        "three_core_direction_consistent": core_family_consistent,
        "m5_refinement_pass": m5_pass,
        "stop_reason": (
            None
            if m5_pass
            else "fine-level material circulation grows at least as fast as 1/dt"
        ),
        "paper_scoring_status": "eligible_not_run" if m5_pass else "blocked_not_run",
        "canonical_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    provenance = collect_run_provenance(REPO_ROOT, vars(args))
    # Preserve the interpreter identity without embedding a workstation path
    # in a potentially public result manifest.  The exact Python version and
    # package versions remain recorded under ``environment``.
    provenance["invocation"]["executable"] = Path(
        provenance["invocation"]["executable"]
    ).name
    recorded_argv = list(provenance["invocation"]["argv"])
    if recorded_argv:
        invoked_path = Path(recorded_argv[0]).resolve()
        try:
            recorded_argv[0] = invoked_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            recorded_argv[0] = invoked_path.name
    provenance["invocation"]["argv"] = recorded_argv
    output = prepare_output_directory(
        args.output_dir,
        allow_existing=args.allow_existing_output,
    )

    started = time.perf_counter()
    step_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for steps_per_cycle in STEPS_PER_CYCLE:
        for core_ratio in REGISTERED_CORE_RADIUS_RATIOS:
            steps, aggregate = _run_configuration(
                steps_per_cycle=steps_per_cycle,
                core_ratio=float(core_ratio),
            )
            step_rows.extend(steps)
            aggregate_rows.append(aggregate)

    gate = evaluate_m5_refinement(aggregate_rows)
    per_step_path = output / "per_step_refinement.csv"
    aggregate_path = output / "aggregate_refinement.csv"
    summary_path = output / "summary.json"
    manifest_path = output / "run_manifest.json"
    _write_csv(per_step_path, step_rows)
    _write_csv(aggregate_path, aggregate_rows)

    summary = {
        "run_id": output.name,
        "status": (
            "mechanical_m5_pass_unscored"
            if gate["m5_refinement_pass"]
            else "stopped_m5_refinement_failure"
        ),
        "scope": "observation-free Yang movement mechanical refinement",
        "frozen_matrix": {
            "aoa_deg": 15.0,
            "lesp_critical": LESP_CRITICAL,
            "grid_chord_span": [CHORDWISE_PANELS, SPANWISE_PANELS],
            "steps_per_cycle": list(STEPS_PER_CYCLE),
            "core_radius_ratios": list(REGISTERED_CORE_RADIUS_RATIOS),
            "horizon_cycles": HORIZON_CYCLES,
            "cycles_built": 1,
            "wake_cycles": 1,
            "core_scale": (
                "ratio*min(global_min_span_panel_width," "U_inf*LESPcrit*dt/sqrt(2))"
            ),
            "experimental_load_files_read": [],
            "parameter_fit_to_target": False,
        },
        "gate": gate,
        "aggregate": aggregate_rows,
        "runtime_s": time.perf_counter() - started,
        "claim_limit": (
            "Mechanical refinement only; no paper accuracy or generalized "
            "aerodynamic-performance claim is permitted."
        ),
    }
    _write_json(summary_path, summary)

    manifest = {
        "run_id": output.name,
        "provenance": provenance,
        "source_hashes": collect_source_hashes(REPO_ROOT, _source_paths()),
        "result_hashes": {
            per_step_path.name: sha256_file(per_step_path),
            aggregate_path.name: sha256_file(aggregate_path),
            summary_path.name: sha256_file(summary_path),
        },
        "paper_scoring_status": gate["paper_scoring_status"],
        "canonical_eligible": False,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
