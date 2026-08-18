"""Run bounded no-force smoke gates for the FluxV v5b shared-wake core.

This runner intentionally has no Yang, Figure-14, or Baik scoring path.  The
current v5b core is a Hirato live-wake conservation/topology diagnostic and
does not couple pressure or aerodynamic force.  Therefore the cross-paper
performance status is always ``blocked_not_scored`` in this stage, even when
all G0--G2 smoke gates pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ROOT = REPO_ROOT / "platform"
DOC_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814"
)
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5b_no_force_smoke"
RAMESH_TEST = PLATFORM_ROOT / "tests/test_ramesh_ldvm_reference.py"
RAMESH_CORE = PLATFORM_ROOT / "ldvm_fourier.py"
HIRATO_LIVE = PLATFORM_ROOT / "claim_runtime/hirato_live_shadow.py"
HIRATO_EQUATIONS = PLATFORM_ROOT / "claim_runtime/hirato_equations.py"
V5B_CORE = PLATFORM_ROOT / "forward_flight_benchmarks/fluxv_v5b_shared_wake.py"


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


def _field(report: Any, name: str, default: Any = None) -> Any:
    if isinstance(report, Mapping):
        return report.get(name, default)
    return getattr(report, name, default)


def _as_float(value: Any, *, empty: float = 0.0) -> float:
    if value is None:
        return empty
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return empty
    return float(np.max(np.abs(array)))


def _as_count(value: Any) -> int:
    if value is None:
        return 0
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return int(np.count_nonzero(array))
    return int(array.size)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _gate(
    gate_id: str,
    measured: float | int | bool | str,
    relation: str,
    threshold: float | int | bool | str,
    passed: bool,
    *,
    level: str,
    evidence_role: str,
    note: str = "",
) -> dict[str, Any]:
    return {
        "gate_level": level,
        "gate_id": gate_id,
        "measured": measured,
        "relation": relation,
        "threshold": threshold,
        "passed": bool(passed),
        "evidence_role": evidence_role,
        "force_evidence": False,
        "note": note,
    }


def _rectangular_half_wing(
    nc: int,
    ns: int,
    *,
    alpha_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nc + 1)
    y = np.linspace(0.0, 1.0, ns + 1)
    corners = np.zeros((nc + 1, ns + 1, 3))
    corners[..., 0] = x[:, None]
    corners[..., 1] = y[None, :]
    if alpha_deg:
        angle = np.deg2rad(alpha_deg)
        relative_x = corners[..., 0] - 0.25
        corners[..., 0] = 0.25 + relative_x * np.cos(angle)
        corners[..., 2] = relative_x * np.sin(angle)
    return corners, np.zeros_like(corners)


def _run_g0(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the unchanged Ramesh partial-parity suite as a subprocess."""

    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-v5b")
    environment.setdefault("MPLCONFIGDIR", "/tmp/mpl-v5b")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_ramesh_ldvm_reference.py",
        "-q",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PLATFORM_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    runtime = time.monotonic() - started
    log_path = output / "g0_ramesh_pytest.log"
    log_path.write_text(
        "command: " + " ".join(command) + "\n\n" + completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    passed = completed.returncode == 0
    gates = [
        _gate(
            "g0_ramesh_reference_pytest",
            completed.returncode,
            "==",
            0,
            passed,
            level="G0",
            evidence_role="frozen_2d_partial_parity_guard",
            note="existing three-test Ramesh suite; not finite-wing force evidence",
        )
    ]
    return gates, {
        "command": command,
        "returncode": completed.returncode,
        "runtime_s": runtime,
        "log": log_path.name,
        "passed": passed,
    }


def _step_row(level: str, case_id: str, step: int, report: Any) -> dict[str, Any]:
    active = _field(report, "active", _field(report, "active_mask"))
    new_sheet = _field(report, "new_sheet", _field(report, "new_sheet_mask"))
    return {
        "gate_level": level,
        "case_id": case_id,
        "step": step,
        "status": _field(report, "status", "no_force_diagnostic_only"),
        "force_coupling": _field(report, "force_coupling", "not_implemented"),
        "active_strip_count": int(np.count_nonzero(active))
        if active is not None
        else int(_field(report, "active_count", 0)),
        "new_sheet_count": int(np.count_nonzero(new_sheet))
        if new_sheet is not None
        else int(_field(report, "new_sheet_count", 0)),
        "eq9_max_abs_residual": float(_field(report, "eq9_max_abs_residual", 0.0)),
        "kelvin_max_abs_residual": float(
            _field(report, "kelvin_max_abs_residual", 0.0)
        ),
        "lesp_max_abs_residual": float(_field(report, "lesp_max_abs_residual", 0.0)),
        "convection_ledger_max_abs_residual": float(
            _field(report, "convection_ledger_max_abs_residual", 0.0)
        ),
        "material_gamma_max_abs_change": float(
            _field(report, "material_gamma_max_abs_change", 0.0)
        ),
        "material_gamma_missing_ids": _as_count(
            _field(report, "material_gamma_missing_ids", [])
        ),
        "new_tev_gamma_max_abs": _as_float(_field(report, "new_tev_gamma")),
        "new_lev_gamma_max_abs": _as_float(_field(report, "new_lev_gamma")),
        "birth_gamma_max_abs": float(_field(report, "birth_gamma_max_abs", 0.0)),
        "has_force_attribute": bool(
            _field(report, "force", None) is not None
            or _field(report, "force_n", None) is not None
        ),
        "has_pressure_attribute": bool(_field(report, "pressure", None) is not None),
    }


def _run_g1() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .fluxv_v5b_shared_wake import (
        FluxVV5BSharedWakeConfig,
        FluxVV5BSharedWakeCore,
        dispatch_v5b_or_parent,
    )

    calls = {"count": 0}
    sentinel = {"identity": object(), "loads": np.array([1.0, -2.0])}

    def parent_callback() -> dict[str, Any]:
        calls["count"] += 1
        return sentinel

    dispatched = dispatch_v5b_or_parent(
        False,
        parent_callback,
    )
    identity_pass = dispatched is sentinel and calls["count"] == 1

    config = FluxVV5BSharedWakeConfig(
        nc=2,
        ns=2,
        u_infinity=(2.0, 0.0, 0.0),
        dt=0.01,
        lesp_crit=10.0,
        core_radius=0.01,
        mirror_symmetry=False,
    )
    core = FluxVV5BSharedWakeCore(config)
    corners, velocity = _rectangular_half_wing(2, 2)
    reports = [core.step(corners, velocity, step=step) for step in range(3)]
    rows = [
        _step_row("G1", "flat_no_lev", index, report)
        for index, report in enumerate(reports)
    ]
    max_active = max(row["active_strip_count"] for row in rows)
    max_lev_birth = max(row["new_lev_gamma_max_abs"] for row in rows)
    max_eq9 = max(row["eq9_max_abs_residual"] for row in rows)
    max_kelvin = max(row["kelvin_max_abs_residual"] for row in rows)
    max_convection = max(row["convection_ledger_max_abs_residual"] for row in rows)
    no_force = all(
        row["force_coupling"] == "not_implemented"
        and not row["has_force_attribute"]
        and not row["has_pressure_attribute"]
        for row in rows
    )
    gates = [
        _gate(
            "g1_disabled_dispatch_identity",
            identity_pass,
            "is",
            True,
            identity_pass,
            level="G1",
            evidence_role="module_off_exact_identity",
        ),
        _gate(
            "g1_no_lev_active_count",
            max_active,
            "==",
            0,
            max_active == 0,
            level="G1",
            evidence_role="synthetic_no_lev_topology",
        ),
        _gate(
            "g1_no_lev_birth_gamma",
            max_lev_birth,
            "==",
            0.0,
            max_lev_birth == 0.0,
            level="G1",
            evidence_role="synthetic_no_lev_topology",
        ),
        _gate(
            "g1_eq9_closure",
            max_eq9,
            "<=",
            1.0e-14,
            max_eq9 <= 1.0e-14,
            level="G1",
            evidence_role="Hirato_Eq9_identity",
        ),
        _gate(
            "g1_kelvin_closure",
            max_kelvin,
            "<=",
            1.0e-12,
            max_kelvin <= 1.0e-12,
            level="G1",
            evidence_role="shared_wake_circulation_ledger",
        ),
        _gate(
            "g1_convection_ledger_closure",
            max_convection,
            "<=",
            1.0e-14,
            max_convection <= 1.0e-14,
            level="G1",
            evidence_role="material_velocity_ledger",
        ),
        _gate(
            "g1_no_force_output",
            no_force,
            "==",
            True,
            no_force,
            level="G1",
            evidence_role="scope_guard",
            note="passing this gate blocks rather than enables paper accuracy scoring",
        ),
    ]
    return (
        gates,
        rows,
        {
            "dispatcher_returned_identical_parent_object": identity_pass,
            "parent_callback_count": calls["count"],
            "force_coupling": "not_implemented" if no_force else "unexpected_output",
        },
    )


def _run_g2() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .fluxv_v5b_shared_wake import (
        FluxVV5BSharedWakeConfig,
        FluxVV5BSharedWakeCore,
        birth_limit_diagnostic,
    )

    config = FluxVV5BSharedWakeConfig(
        nc=2,
        ns=2,
        u_infinity=(2.0, 0.0, 0.0),
        dt=0.01,
        lesp_crit=0.05,
        core_radius=0.01,
        mirror_symmetry=False,
    )
    core = FluxVV5BSharedWakeCore(config)
    corners, velocity = _rectangular_half_wing(2, 2, alpha_deg=15.0)
    reports = [core.step(corners, velocity, step=step) for step in range(2)]
    rows = [
        _step_row("G2", "pitched15_live", index, report)
        for index, report in enumerate(reports)
    ]
    active = max(row["active_strip_count"] for row in rows)
    newborn_first = rows[0]["new_sheet_count"]
    newborn_second = rows[1]["new_sheet_count"]
    eq9 = max(row["eq9_max_abs_residual"] for row in rows)
    kelvin = max(row["kelvin_max_abs_residual"] for row in rows)
    lesp = max(row["lesp_max_abs_residual"] for row in rows)
    material = max(row["material_gamma_max_abs_change"] for row in rows)
    missing = max(row["material_gamma_missing_ids"] for row in rows)
    finite_birth = all(np.isfinite(row["birth_gamma_max_abs"]) for row in rows)
    refinement_dt = np.asarray([0.02, 0.01, 0.005, 0.0025])
    refinement_birth = []
    for delta_time in refinement_dt:
        refined_config = FluxVV5BSharedWakeConfig(
            nc=2,
            ns=2,
            u_infinity=(2.0, 0.0, 0.0),
            dt=float(delta_time),
            lesp_crit=0.05,
            core_radius=0.01,
            mirror_symmetry=False,
        )
        refined = FluxVV5BSharedWakeCore(refined_config).step(
            corners,
            velocity,
            step=0,
        )
        refinement_birth.append(float(refined["birth_gamma_max_abs"]))
    birth = birth_limit_diagnostic(refinement_dt, refinement_birth)
    birth_passed = bool(birth["tends_to_zero"])
    birth_metric = float(birth["slope_p"])
    gates = [
        _gate(
            "g2_live_active_strip",
            active,
            ">=",
            1,
            active >= 1,
            level="G2",
            evidence_role="live_LESP_event",
        ),
        _gate(
            "g2_first_step_new_sheet",
            newborn_first,
            ">=",
            1,
            newborn_first >= 1,
            level="G2",
            evidence_role="LEV_birth_topology",
        ),
        _gate(
            "g2_second_step_no_rebirth",
            newborn_second,
            "==",
            0,
            newborn_second == 0,
            level="G2",
            evidence_role="material_sheet_identity",
        ),
        _gate(
            "g2_eq9_closure",
            eq9,
            "<=",
            1.0e-14,
            eq9 <= 1.0e-14,
            level="G2",
            evidence_role="Hirato_Eq9_identity",
        ),
        _gate(
            "g2_kelvin_closure",
            kelvin,
            "<=",
            1.0e-12,
            kelvin <= 1.0e-12,
            level="G2",
            evidence_role="shared_wake_circulation_ledger",
        ),
        _gate(
            "g2_active_lesp_closure",
            lesp,
            "<=",
            1.0e-12,
            lesp <= 1.0e-12,
            level="G2",
            evidence_role="LESP_constraint",
        ),
        _gate(
            "g2_material_gamma_invariance",
            material,
            "<=",
            1.0e-14,
            material <= 1.0e-14,
            level="G2",
            evidence_role="material_vortex_identity",
        ),
        _gate(
            "g2_material_ids_complete",
            missing,
            "==",
            0,
            missing == 0,
            level="G2",
            evidence_role="material_vortex_identity",
        ),
        _gate(
            "g2_birth_strength_finite",
            finite_birth,
            "==",
            True,
            finite_birth,
            level="G2",
            evidence_role="birth_finiteness",
        ),
        _gate(
            "g2_birth_limit",
            birth_metric,
            ">",
            0.0,
            birth_passed,
            level="G2",
            evidence_role="dt_to_zero_birth_limit",
            note=(
                "core-defined positive-order diagnostic only; a near-zero fitted "
                "order is weak asymptotic evidence and cannot authorize force"
            ),
        ),
    ]
    return (
        gates,
        rows,
        {
            "force_coupling": "not_implemented",
            "birth_limit": _jsonable(birth),
            "birth_limit_interpretation": (
                "topology-only positive-order diagnostic; fitted order may be near "
                "zero and supplies no pressure/force evidence"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.allow_existing_output:
        parser.error("output directory exists and is non-empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)

    g0_gates, g0 = _run_g0(output)
    g1_gates, g1_rows, g1 = _run_g1()
    g2_gates, g2_rows, g2 = _run_g2()
    gates = g0_gates + g1_gates + g2_gates
    shadow_rows = g1_rows + g2_rows
    all_passed = all(bool(row["passed"]) for row in gates)
    force_coupling = "not_implemented"
    gate_path = output / "gate_results.csv"
    shadow_path = output / "shadow_steps.csv"
    _write_csv(gate_path, gates)
    _write_csv(shadow_path, shadow_rows)

    direct_sources = (
        Path(__file__).resolve(),
        V5B_CORE,
        RAMESH_TEST,
        RAMESH_CORE,
        HIRATO_LIVE,
        HIRATO_EQUATIONS,
        PLATFORM_ROOT / "tests/test_fluxv_v5b_smoke.py",
        PLATFORM_ROOT / "tests/test_fluxv_v5b_shared_wake.py",
        DOC_ROOT / "V5B_SMOKE_GATE_CONTRACT.md",
    )
    result_paths = (
        output / "g0_ramesh_pytest.log",
        gate_path,
        shadow_path,
    )
    summary = {
        "run_id": output.name,
        "status": (
            "smoke_gates_passed_no_force"
            if all_passed
            else "smoke_gate_failure_no_force"
        ),
        "crosspaper_performance_status": "blocked_not_scored",
        "blocked_reason": (
            "shared-wake core has no pressure/force coupling; shadow output cannot "
            "be compared with Yang, Figure 14, or Baik loads"
        ),
        "force_coupling": force_coupling,
        "all_g0_g2_passed": all_passed,
        "gate_counts": {
            "passed": sum(bool(row["passed"]) for row in gates),
            "total": len(gates),
            "by_level": {
                level: {
                    "passed": sum(
                        bool(row["passed"])
                        for row in gates
                        if row["gate_level"] == level
                    ),
                    "total": sum(row["gate_level"] == level for row in gates),
                }
                for level in ("G0", "G1", "G2")
            },
        },
        "g0": g0,
        "g1": g1,
        "g2": g2,
        "evidence_role": "topology_and_conservation_only_no_force",
        "prohibited_claims": [
            "no Yang/Figure-14/Baik accuracy result",
            "no lift/drag/thrust improvement claim",
            "no equivalence between shadow circulation and aerodynamic force",
        ],
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "argv": sys.argv,
        },
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in direct_sources
        },
        "result_hashes": {path.name: _sha256(path) for path in result_paths},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "crosspaper_performance_status": "blocked_not_scored",
                "gate_counts": summary["gate_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
