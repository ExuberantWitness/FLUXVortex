"""Run observation-free mechanical gates for the FluxV v5c1-RSLS state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from forward_flight_benchmarks.fluxv_v5c_suction import (
    DEFAULT_SUCTION_PARAMETERS,
    RateSensitiveSuctionState,
    project_axial_suction_loss_to_wind_axes,
    run_rate_sensitive_suction_history,
    step_rate_sensitive_suction,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "fluxv_v5c_nextgen_20260814"
)
DEFAULT_OUTPUT = (
    DOC_ROOT / "runs/20260814_fluxv_v5c1_mechanical_pole05_rate1_reproducible"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_history(
    steps_per_cycle: int,
    *,
    cycles: int = 12,
    convective_period: float = 8.0,
    enabled: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    phase = np.arange(steps_per_cycle, dtype=float) / steps_per_cycle
    a0_cycle = 0.14 + 0.12 * np.sin(2.0 * np.pi * phase)
    a0 = np.tile(a0_cycle, cycles)[:, None]
    critical = np.full_like(a0, 0.1)
    delta_tau = np.full_like(a0, convective_period / steps_per_cycle)
    base_cs = 2.0 * np.pi * np.minimum(np.abs(a0), critical) ** 2
    result = run_rate_sensitive_suction_history(
        a0_pre=a0,
        lesp_critical=critical,
        delta_tau=delta_tau,
        base_suction_coefficient=base_cs,
        enabled=enabled,
    )
    return phase, result


def run(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    histories: dict[int, tuple[np.ndarray, dict[str, Any]]] = {
        steps: _synthetic_history(steps) for steps in (32, 64, 128, 256)
    }
    phase128, primary = histories[128]
    last = slice(-128, None)
    previous = slice(-256, -128)
    alpha = np.deg2rad(20.0 * np.sin(2.0 * np.pi * phase128))
    projection = project_axial_suction_loss_to_wind_axes(
        primary["delta_suction_coefficient"][last, 0], alpha
    )

    # Attached and constant-supercritical histories are exact no-loss limits.
    attached_a0 = np.full((128, 1), 0.05)
    constant_a0 = np.full((128, 1), 0.2)
    common = {
        "lesp_critical": np.full((128, 1), 0.1),
        "delta_tau": np.full((128, 1), 0.05),
        "base_suction_coefficient": np.full((128, 1), 0.2),
    }
    attached = run_rate_sensitive_suction_history(a0_pre=attached_a0, **common)
    constant = run_rate_sensitive_suction_history(a0_pre=constant_a0, **common)
    _, disabled = _synthetic_history(128, enabled=False)
    hot_state = RateSensitiveSuctionState(
        chi=np.array([0.8]),
        previous_j=np.array([4.0]),
        previous_previous_j=np.array([4.0]),
        previous_delta_tau=np.array([0.1]),
        step_count=2,
    )
    hot_after, hot_record = step_rate_sensitive_suction(
        hot_state,
        a0_pre=0.2,
        lesp_critical=0.1,
        delta_tau=0.1,
        base_suction_coefficient=0.2,
    )
    hot_expected = 0.8 * np.exp(
        -DEFAULT_SUCTION_PARAMETERS.state_pole_per_convective_time * 0.1
    )

    # Future-only perturbation proves prefix causality without cyclic wrapping.
    a0 = np.tile(0.14 + 0.12 * np.sin(2.0 * np.pi * phase128), 3)[:, None]
    altered = a0.copy()
    altered[300:] *= 1.8
    causal_common = {
        "lesp_critical": np.full_like(a0, 0.1),
        "delta_tau": np.full_like(a0, 8.0 / 128.0),
        "base_suction_coefficient": 2.0 * np.pi * np.minimum(np.abs(a0), 0.1) ** 2,
    }
    original = run_rate_sensitive_suction_history(a0_pre=a0, **causal_common)
    altered_common = dict(causal_common)
    altered_common["base_suction_coefficient"] = (
        2.0 * np.pi * np.minimum(np.abs(altered), 0.1) ** 2
    )
    changed = run_rate_sensitive_suction_history(a0_pre=altered, **altered_common)

    refinement_rows: list[dict[str, Any]] = []
    refinement_errors: list[float] = []
    for coarse, fine in ((32, 64), (64, 128), (128, 256)):
        phase_c, result_c = histories[coarse]
        phase_f, result_f = histories[fine]
        loss_c = result_c["loss_fraction"][-coarse:, 0]
        loss_f = result_f["loss_fraction"][-fine:, 0]
        interpolated = np.interp(phase_f, phase_c, loss_c, period=1.0)
        relative_l2 = float(
            np.linalg.norm(loss_f - interpolated) / max(np.linalg.norm(loss_f), 1.0e-15)
        )
        refinement_errors.append(relative_l2)
        refinement_rows.append(
            {
                "coarse_steps_per_cycle": coarse,
                "fine_steps_per_cycle": fine,
                "relative_l2_loss_fraction": relative_l2,
                "fine_mean_loss_fraction": float(np.mean(loss_f)),
                "fine_max_loss_fraction": float(np.max(loss_f)),
            }
        )

    history_rows: list[dict[str, Any]] = []
    for index, phase in enumerate(phase128):
        source_index = -128 + index
        history_rows.append(
            {
                "phase": phase,
                "a0_pre": 0.14 + 0.12 * np.sin(2.0 * np.pi * phase),
                "j": primary["j"][source_index, 0],
                "j_rate": primary["j_rate"][source_index, 0],
                "supercritical_gate": primary["supercritical_gate"][source_index, 0],
                "chi": primary["chi_after"][source_index, 0],
                "loss_fraction": primary["loss_fraction"][source_index, 0],
                "base_CS": primary["base_suction_coefficient"][source_index, 0],
                "target_CS": primary["target_suction_coefficient"][source_index, 0],
                "delta_CS": primary["delta_suction_coefficient"][source_index, 0],
                "delta_CL": projection["delta_CL"][index],
                "delta_CD": projection["delta_CD"][index],
                "delta_CN": projection["delta_CN"][index],
            }
        )

    metrics = {
        "disabled_max_abs": float(
            np.max(np.abs(disabled["delta_suction_coefficient"]))
        ),
        "attached_max_abs": float(
            np.max(np.abs(attached["delta_suction_coefficient"]))
        ),
        "constant_zero_rate_max_abs": float(
            np.max(np.abs(constant["delta_suction_coefficient"]))
        ),
        "hot_zero_rate_decay_error": float(abs(hot_after.chi[0] - hot_expected)),
        "hot_zero_rate_equilibrium_max_abs": float(
            np.max(np.abs(hot_record["chi_equilibrium"]))
        ),
        "hot_zero_rate_loss_fraction": float(hot_record["loss_fraction"][0]),
        "state_min": float(np.min(primary["chi_after"])),
        "state_max": float(np.max(primary["chi_after"])),
        "loss_min": float(np.min(primary["loss_fraction"])),
        "loss_max": float(np.max(primary["loss_fraction"])),
        "target_suction_min": float(np.min(primary["target_suction_coefficient"])),
        "target_minus_base_max": float(
            np.max(
                primary["target_suction_coefficient"]
                - primary["base_suction_coefficient"]
            )
        ),
        "delta_suction_max": float(np.max(primary["delta_suction_coefficient"])),
        "delta_normal_max_abs": float(
            np.max(np.abs(primary["delta_normal_coefficient"]))
        ),
        "minimum_drag_increment": float(np.min(projection["delta_CD"])),
        "cycle_state_max_abs": float(
            np.max(
                np.abs(
                    primary["chi_after"][last, 0] - primary["chi_after"][previous, 0]
                )
            )
        ),
        "causal_prefix_max_abs": float(
            np.max(
                np.abs(
                    original["delta_suction_coefficient"][:300]
                    - changed["delta_suction_coefficient"][:300]
                )
            )
        ),
        "refinement_128_to_256_relative_l2": refinement_errors[-1],
    }
    gates = {
        "disabled_identity": metrics["disabled_max_abs"] <= 1.0e-12,
        "attached_identity": metrics["attached_max_abs"] <= 1.0e-12,
        "constant_zero_rate_identity": (
            metrics["constant_zero_rate_max_abs"] <= 1.0e-12
        ),
        "hot_zero_rate_analytic_decay": (
            metrics["hot_zero_rate_decay_error"] <= 1.0e-12
            and metrics["hot_zero_rate_equilibrium_max_abs"] <= 1.0e-12
            and metrics["hot_zero_rate_loss_fraction"] > 0.0
        ),
        "state_bounded": metrics["state_min"] >= 0.0 and metrics["state_max"] <= 1.0,
        "loss_bounded": metrics["loss_min"] >= 0.0 and metrics["loss_max"] <= 1.0,
        "target_suction_nonnegative": metrics["target_suction_min"] >= 0.0,
        "normal_owner_unchanged": metrics["delta_normal_max_abs"] <= 1.0e-12,
        "axial_suction_only_sign": (
            metrics["delta_suction_max"] <= 1.0e-12
            and metrics["target_minus_base_max"] <= 1.0e-12
            and metrics["minimum_drag_increment"] >= -1.0e-12
        ),
        "cycle_state_converged": metrics["cycle_state_max_abs"] <= 1.0e-4,
        "causal_prefix_identity": metrics["causal_prefix_max_abs"] <= 1.0e-12,
        "refinement_pass": metrics["refinement_128_to_256_relative_l2"] <= 0.05,
    }
    if not all(gates.values()):
        failed = [key for key, passed in gates.items() if not passed]
        raise RuntimeError(f"v5c1 mechanical gate failure: {failed}")

    history_path = output / "synthetic_rate_history.csv"
    refinement_path = output / "numerical_refinement.csv"
    _write_csv(history_path, history_rows)
    _write_csv(refinement_path, refinement_rows)
    summary = {
        "run_id": output.name,
        "status": "v5c1_mechanical_gates_passed",
        "promotion_status": "mechanical_only_not_crosspaper_scored",
        "metrics": metrics,
        "gates": gates,
        "parameters": DEFAULT_SUCTION_PARAMETERS.manifest(),
        "parameter_selection_data": [],
        "synthetic_history": {
            "a0_law": "0.14 + 0.12*sin(2*pi*phase)",
            "lesp_critical": 0.1,
            "convective_time_per_cycle": 8.0,
            "total_cycles": 12,
            "discarded_warmup_cycles": 11,
            "scored_cycles": 1,
        },
        "limitations": [
            "These are analytic mechanical gates, not experimental accuracy evidence.",
            "The sign gate proves only that axial suction is reduced and produces no negative drag over the tested +/-20 degree incidence; it is not an independent aerodynamic-work balance.",
            "Canonical paper scoring requires a same-time-layer strip A0 and axial-suction ledger export.",
        ],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": output.name,
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("fluxv_v5c_suction.py").resolve(),
            )
        },
        "result_hashes": {
            path.name: _sha256(path)
            for path in (history_path, refinement_path, summary_path)
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
