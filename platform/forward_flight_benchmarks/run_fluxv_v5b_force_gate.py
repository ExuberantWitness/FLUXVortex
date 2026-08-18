"""Run the pre-paper force-promotion gates for FluxV v5b.

This runner deliberately stops before Yang/Figure-14/Baik scoring.  It tests
whether the chronological shared-wake pressure path is mechanically closed,
whether a smooth LEV birth has a regular ``dt -> 0`` limit, and—most
importantly—whether the pristine no-LEV path is the same load model as the
current FluxV/UVLM baseline.  A failed equivalence gate is a scientific
NO-GO, not a runner error, and is written as a durable result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .cases import YANG_2025
from .fluxv_v5b_sequence import (
    FluxVV5BSequenceConfig,
    run_fluxv_v5b_force_sequence,
)
from .fluxv_v5b_ptera_adapter import (
    build_crosspaper_smoke_input,
    coefficients_from_gp1_force,
)
from .fluxv_v5b_shared_wake import (
    FluxVV5BSharedWakeConfig,
    FluxVV5BSharedWakeCore,
    birth_limit_diagnostic,
)
from .ptera_adapter import build_yang2025_movement, run_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814"
)
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5b_force_gate_frozen"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_current_fluxv_reduction_gate() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare no-LEV v5b against the actual current FluxV load channel."""

    settings = (2, 4, 20, 2, 2)
    adapted = build_crosspaper_smoke_input("yang2025")
    history = adapted.history
    corners = history.corners_history
    velocity = history.corner_velocity_history
    config = FluxVV5BSequenceConfig(
        shared_wake=FluxVV5BSharedWakeConfig(
            nc=corners.shape[1] - 1,
            ns=corners.shape[2] - 1,
            u_infinity=tuple(history.u_infinity_gp1_m_s),
            dt=history.delta_time_s,
            lesp_crit=10.0,
            core_radius=0.01 * YANG_2025.chord_m,
            mirror_symmetry=history.hirato_mirror_symmetry,
        ),
        density=history.rho_kg_m3,
    )
    sequence = run_fluxv_v5b_force_sequence(
        config,
        corners,
        velocity,
        phase=np.arange(len(corners), dtype=float) / len(corners),
    )
    current_movement, _ = build_yang2025_movement(
        15.0,
        "smoke",
        settings=settings,
    )
    current = run_model(
        current_movement,
        "fluxv_uvpm",
        period_s=YANG_2025.period_s,
        rho=YANG_2025.rho_kg_m3,
        speed=YANG_2025.freestream_m_s,
        area=YANG_2025.area_m2,
        output_samples=settings[2],
    )
    coefficients = coefficients_from_gp1_force(
        sequence.force_history_n,
        history,
    )
    scored = history.final_cycle_indices
    v5b_cl = np.asarray(coefficients["CL"])[scored]
    v5b_cd = np.asarray(coefficients["CD"])[scored]
    old_cl = np.asarray(current["CL"], dtype=float)
    old_cd = -np.asarray(current["CT"], dtype=float)
    if v5b_cl.shape != old_cl.shape or v5b_cd.shape != old_cd.shape:
        raise ValueError("current FluxV and v5b final cycles are not aligned")
    rows = [
        {
            "phase": index / len(v5b_cl),
            "current_fluxv_CL": old_cl[index],
            "v5b_no_lev_CL": v5b_cl[index],
            "current_fluxv_CD": old_cd[index],
            "v5b_no_lev_CD": v5b_cd[index],
        }
        for index in range(len(v5b_cl))
    ]
    mean_cl_old = float(np.mean(old_cl))
    mean_cd_old = float(np.mean(old_cd))
    mean_cl_v5b = float(np.mean(v5b_cl))
    mean_cd_v5b = float(np.mean(v5b_cd))
    return {
        "case": "Yang2025_AoA15_no_LEV_reduction_probe",
        "settings": list(settings),
        "movement_metadata": adapted.manifest,
        "current_fluxv_mean_CL": mean_cl_old,
        "v5b_no_lev_mean_CL": mean_cl_v5b,
        "current_fluxv_mean_CD": mean_cd_old,
        "v5b_no_lev_mean_CD": mean_cd_v5b,
        "mean_CL_abs_difference": abs(mean_cl_v5b - mean_cl_old),
        "mean_CD_abs_difference": abs(mean_cd_v5b - mean_cd_old),
        "phase_CL_max_abs_difference": float(np.max(np.abs(v5b_cl - old_cl))),
        "phase_CD_max_abs_difference": float(np.max(np.abs(v5b_cd - old_cd))),
        "v5b_active_lev_steps": int(
            sum(np.any(item.pre_convection_report.active) for item in sequence.steps)
        ),
        "sequence_guards_passed": sequence.guards.passed,
        "exact_reduction_tolerance": 1.0e-12,
    }, rows


def _smooth_ramp_geometry_velocity(
    time_s: float,
    *,
    duration_s: float = 0.2,
    amplitude_deg: float = 20.0,
    nc: int = 2,
    ns: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Quintic start-up with analytic zero-endpoint angular velocity."""

    coordinate = float(np.clip(time_s / duration_s, 0.0, 1.0))
    spacing = 10.0 * coordinate**3 - 15.0 * coordinate**4 + 6.0 * coordinate**5
    if 0.0 < coordinate < 1.0:
        spacing_rate = (
            30.0 * coordinate**2 - 60.0 * coordinate**3 + 30.0 * coordinate**4
        ) / duration_s
    else:
        spacing_rate = 0.0
    angle = np.deg2rad(amplitude_deg * spacing)
    angle_rate = np.deg2rad(amplitude_deg * spacing_rate)
    x = np.linspace(0.0, 1.0, nc + 1)
    y = np.linspace(0.0, 1.0, ns + 1)
    relative = x - 0.25
    corners = np.zeros((nc + 1, ns + 1, 3))
    velocity = np.zeros_like(corners)
    corners[..., 0] = (0.25 + relative * np.cos(angle))[:, None]
    corners[..., 1] = y[None, :]
    corners[..., 2] = (relative * np.sin(angle))[:, None]
    velocity[..., 0] = (-relative * np.sin(angle) * angle_rate)[:, None]
    velocity[..., 2] = (relative * np.cos(angle) * angle_rate)[:, None]
    return corners, velocity


def _run_birth_limit_gate() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure the first smooth threshold crossing on a nested dt family."""

    levels = np.asarray([0.01, 0.005, 0.0025, 0.00125])
    rows: list[dict[str, Any]] = []
    gamma: list[float] = []
    for dt in levels:
        core = FluxVV5BSharedWakeCore(
            FluxVV5BSharedWakeConfig(
                nc=2,
                ns=2,
                u_infinity=(2.0, 0.0, 0.0),
                dt=float(dt),
                lesp_crit=0.05,
                core_radius=0.01,
                mirror_symmetry=False,
            )
        )
        found: dict[str, Any] | None = None
        found_step = -1
        for step in range(int(np.ceil(0.25 / dt)) + 1):
            corners, velocity = _smooth_ramp_geometry_velocity(step * dt)
            report = core.step(corners, velocity, step=step)
            if report["new_lev_count"]:
                found = report
                found_step = step
                break
        if found is None:
            raise RuntimeError(f"smooth onset did not occur at dt={dt}")
        strength = float(np.max(np.abs(found["new_lev_gamma"])))
        displacement_ratio = float(
            np.max(np.abs(found["lev_birth_displacement_over_Udt"]), initial=0.0)
        )
        gamma.append(strength)
        rows.append(
            {
                "dt_s": dt,
                "first_active_step": found_step,
                "first_active_time_s": found_step * dt,
                "max_abs_A0_event": float(np.max(np.abs(found["a0_event"]))),
                "birth_gamma_max_abs_m2_s": strength,
                "birth_displacement_over_Udt_max_abs": displacement_ratio,
                "lesp_residual_max_abs": found["lesp_max_abs_residual"],
            }
        )
    diagnostic = birth_limit_diagnostic(levels, gamma)
    local_orders = np.asarray(diagnostic["local_orders"], dtype=float)
    exponent = float(diagnostic["slope_p"])
    bounded_velocity = (
        max(row["birth_displacement_over_Udt_max_abs"] for row in rows) <= 1.0
    )
    passed = bool(
        0.8 <= exponent <= 1.25 and np.all(local_orders > 0.0) and bounded_velocity
    )
    return {
        "diagnostic": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in diagnostic.items()
        },
        "frozen_gate": {
            "slope_interval": [0.8, 1.25],
            "all_local_orders_positive": True,
            "max_birth_displacement_over_Udt": 1.0,
        },
        "passed": passed,
        "note": (
            "This smooth quintic crossing replaces the invalid discontinuous "
            "pitched-at-t0 diagnostic; it was introduced after that diagnostic "
            "and is therefore a development, not blind, gate."
        ),
    }, rows


def _run_active_force_ledger_gate() -> dict[str, Any]:
    corners, velocity = _smooth_ramp_geometry_velocity(0.1)
    result = run_fluxv_v5b_force_sequence(
        FluxVV5BSequenceConfig(
            shared_wake=FluxVV5BSharedWakeConfig(
                nc=2,
                ns=2,
                u_infinity=(2.0, 0.0, 0.0),
                dt=0.01,
                lesp_crit=0.05,
                core_radius=0.01,
            ),
            density=1.225,
        ),
        np.repeat(corners[None], 3, axis=0),
        np.repeat(velocity[None], 3, axis=0),
    )
    return {
        "passed": result.guards.passed,
        "force_ledger_count": result.guards.force_ledger_count,
        "max_eq9_residual": result.guards.max_eq9_residual,
        "max_lesp_residual": result.guards.max_lesp_residual,
        "max_convection_residual": result.guards.max_convection_residual,
        "max_material_gamma_change": result.guards.max_material_gamma_change,
        "max_velocity_ledger_residual": result.guards.max_velocity_ledger_residual,
        "max_pressure_ledger_residual": result.guards.max_pressure_ledger_residual,
        "max_force_channel_residual": result.guards.max_force_channel_residual,
        "unique_force_owner": result.guards.unique_force_owner,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory exists and is non-empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)

    reduction, reduction_rows = _run_current_fluxv_reduction_gate()
    birth, birth_rows = _run_birth_limit_gate()
    force = _run_active_force_ledger_gate()
    reduction_passed = bool(
        reduction["v5b_active_lev_steps"] == 0
        and reduction["phase_CL_max_abs_difference"] <= 1.0e-12
        and reduction["phase_CD_max_abs_difference"] <= 1.0e-12
    )
    gates = [
        {
            "gate": "G1_current_FluxV_no_LEV_exact_reduction",
            "passed": reduction_passed,
            "measured": max(
                reduction["phase_CL_max_abs_difference"],
                reduction["phase_CD_max_abs_difference"],
            ),
            "relation": "<=",
            "threshold": 1.0e-12,
        },
        {
            "gate": "G4_single_surface_pressure_force_owner",
            "passed": bool(force["passed"]),
            "measured": max(
                force["max_velocity_ledger_residual"],
                force["max_pressure_ledger_residual"],
                force["max_force_channel_residual"],
            ),
            "relation": "<=",
            "threshold": 1.0e-12,
        },
        {
            "gate": "G5_smooth_birth_limit",
            "passed": bool(birth["passed"]),
            "measured": birth["diagnostic"]["slope_p"],
            "relation": "in",
            "threshold": "[0.8,1.25] and every local order > 0",
        },
        {
            "gate": "G6_Ramesh_high_AR_force_parity",
            "passed": False,
            "measured": "not_run",
            "relation": "required_before paper scoring",
            "threshold": "pass",
        },
    ]
    promotion = all(bool(row["passed"]) for row in gates)

    reduction_path = output / "no_lev_current_fluxv_comparison.csv"
    birth_path = output / "smooth_birth_refinement.csv"
    gates_path = output / "gate_results.csv"
    _write_csv(reduction_path, reduction_rows)
    _write_csv(birth_path, birth_rows)
    _write_csv(gates_path, gates)

    sources = [
        Path(__file__).resolve(),
        REPO_ROOT / "platform/forward_flight_benchmarks/fluxv_v5b_shared_wake.py",
        REPO_ROOT / "platform/forward_flight_benchmarks/fluxv_v5b_force.py",
        REPO_ROOT / "platform/forward_flight_benchmarks/fluxv_v5b_sequence.py",
        REPO_ROOT / "platform/forward_flight_benchmarks/fluxv_v5b_ptera_adapter.py",
        REPO_ROOT / "platform/forward_flight_benchmarks/ptera_adapter.py",
        REPO_ROOT / "src/fluxvortex/solver.py",
    ]
    results = [reduction_path, birth_path, gates_path]
    summary = {
        "run_id": output.name,
        "status": "promotion_pass" if promotion else "no_go_before_crosspaper",
        "crosspaper_performance_status": "eligible"
        if promotion
        else "blocked_not_scored",
        "force_coupling": "single_surface_pressure_ledger_implemented",
        "current_fluxv_exact_reduction": reduction,
        "active_force_ledger": force,
        "smooth_birth_limit": birth,
        "gates": gates,
        "promotion_passed": promotion,
        "stop_reason": (
            None
            if promotion
            else "G1 current-FluxV no-LEV equivalence and/or prerequisite G6 failed"
        ),
        "paper_results": None,
        "prohibited_claims": [
            "no v5b Yang/Figure-14/Baik accuracy claim",
            "no replacement of v4b by this standalone N1 force path",
            "no use of internal ledger closure as experimental validation",
        ],
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "argv": sys.argv,
        },
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sources
        },
        "result_hashes": {path.name: _sha256(path) for path in results},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "crosspaper_performance_status": summary[
                    "crosspaper_performance_status"
                ],
                "gates": gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
