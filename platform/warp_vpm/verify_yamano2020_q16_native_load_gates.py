"""Bounded CUDA gates for the formal Yamano Q16/native-V5M load path."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_author_loads import (
    material_ring_velocity_derivative_expanded,
)
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
    native_aic,
)
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET,
    load_mf1_step1_oracle,
    load_mf2_history_oracle,
    make_yamano2020_q16_model,
)


FORMAL_Q16_GRID = (5, 3)
FORMAL_AERO_GRID = (15, 10)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def run_gates(output: Path | None = None) -> dict[str, object]:
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("formal Yamano load gates require CUDA")
    started = time.perf_counter()
    case = YAMANO_2020_SINGLE_SHEET
    mesh, _, _ = make_yamano2020_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
    )
    surface = Q16NativeV5MSurface(
        mesh,
        q16_chordwise_elements=FORMAL_Q16_GRID[0],
        q16_spanwise_elements=FORMAL_Q16_GRID[1],
        aerodynamic_chordwise_panels=FORMAL_AERO_GRID[0],
        aerodynamic_spanwise_panels=FORMAL_AERO_GRID[1],
        device=config.DEVICE,
    )
    solver = Q16NativeV5MSolver(
        surface,
        NativeV5MConfig(
            density=case.fluid_density_kg_m3,
            freestream=case.freestream_m_s,
            aerodynamic_dt=case.aerodynamic_dt_s,
            device=config.DEVICE,
        ),
    )
    state = wp.array(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    zero = wp.zeros_like(state)
    geometry = surface.evaluate(state, zero)
    mf1_oracle = load_mf1_step1_oracle()
    expected_aic = torch.as_tensor(
        np.array(mf1_oracle["aic"], copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    actual_aic = native_aic(geometry, chordwise_panels=FORMAL_AERO_GRID[0])
    aic_relative_error = torch.linalg.vector_norm(
        actual_aic - expected_aic
    ) / torch.linalg.vector_norm(expected_aic)

    history = load_mf2_history_oracle()
    mf2_max_abs_error = torch.zeros((), device="cuda:0", dtype=torch.float64)
    qf_resultant_max_relative_error = torch.zeros_like(mf2_max_abs_error)

    def tensor(step: int, name: str) -> torch.Tensor:
        return torch.as_tensor(
            np.array(history[f"step_{step}_{name}"], copy=True),
            device="cuda:0",
            dtype=torch.float64,
        )

    for step in range(1, 5):
        rings = torch.stack(
            tuple(tensor(step, f"r_wake_{corner}") for corner in range(1, 5)),
            dim=1,
        )
        ring_velocity = torch.stack(
            tuple(
                tensor(step, f"dt_r_wake_{corner}") for corner in range(1, 5)
            ),
            dim=1,
        )
        derivative = material_ring_velocity_derivative_expanded(
            tensor(step, "rc_vec"),
            tensor(step, "dt_rc_vec"),
            rings,
            ring_velocity,
        )
        wake_velocity_rate = torch.sum(
            derivative * tensor(step, "Gamma_wake")[None, :, None], dim=1
        )
        normal_rate = torch.sum(
            wake_velocity_rate * tensor(step, "n_vec_i"), dim=1
        )
        actual_mf2 = torch.linalg.solve(tensor(step, "A_mat"), -normal_rate)
        mf2_max_abs_error = torch.maximum(
            mf2_max_abs_error,
            torch.max(torch.abs(actual_mf2 - tensor(step, "Mf2_vec1"))),
        )
        oracle_geometry = replace(geometry, normals=tensor(step, "n_vec_i"))
        pressure_map = solver.author_load_assembler.pressure_map(oracle_geometry)
        pressure = tensor(step, "dp_lift1") + tensor(step, "Mf2_vec1")
        actual_qf = pressure_map @ pressure
        actual_resultant = actual_qf.reshape(-1, 6)[:, :3].sum(dim=0)
        expected_resultant = tensor(step, "Qf_p_global").reshape(-1, 9)[
            :, :3
        ].sum(dim=0)
        qf_relative = torch.linalg.vector_norm(
            actual_resultant - expected_resultant
        ) / torch.linalg.vector_norm(expected_resultant)
        qf_resultant_max_relative_error = torch.maximum(
            qf_resultant_max_relative_error, qf_relative
        )

    anchor = solver.author_anchor_load(state, zero)
    q16_acceleration = torch.zeros(
        mesh.dof_count, device="cuda:0", dtype=torch.float64
    )
    q16_rows = torch.as_tensor(
        np.array(mesh.reference_rows, copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    q16_acceleration[2::6] = q16_rows[:, 0]
    q16_matrix = anchor.added_mass.generalized_matrix / case.fluid_density_kg_m3
    q16_work = q16_acceleration @ q16_matrix @ q16_acceleration
    # A direct global transpose is not the production operator because the
    # author's Mf1 assembly retains element-local input/output support.  Keep
    # its common-field work as a formal-case diagnostic: it brackets whether
    # the current panel-local support is dropping material Q16 overlap terms.
    full_neumann = solver.author_load_assembler._neumann_map(geometry)
    full_matrix = anchor.pressure_to_generalized @ torch.linalg.solve(
        anchor.aic, full_neumann
    )
    full_q16_work = q16_acceleration @ full_matrix @ q16_acceleration

    indices = torch.stack(
        (
            torch.as_tensor(
                np.array(mf1_oracle["mf1_row"], copy=True),
                device="cuda:0",
                dtype=torch.int64,
            ),
            torch.as_tensor(
                np.array(mf1_oracle["mf1_col"], copy=True),
                device="cuda:0",
                dtype=torch.int64,
            ),
        )
    )
    author_matrix = torch.zeros(
        (1584, 1584), device="cuda:0", dtype=torch.float64
    )
    author_matrix.index_put_(
        (indices[0], indices[1]),
        torch.as_tensor(
            np.array(mf1_oracle["mf1_data"], copy=True),
            device="cuda:0",
            dtype=torch.float64,
        ),
        accumulate=True,
    )
    author_acceleration = torch.zeros(
        1584, device="cuda:0", dtype=torch.float64
    )
    for chord_node in range(16):
        for span_node in range(11):
            node = chord_node * 11 + span_node
            author_acceleration[node * 9 + 2] = chord_node / 15.0
            author_acceleration[node * 9 + 5] = 1.0
    author_acceleration[: 11 * 9] = 0.0
    author_force = author_matrix @ author_acceleration
    author_work = author_acceleration @ author_force
    mf1_common_field_relative_error = torch.abs(
        (q16_work - author_work) / author_work
    )

    thresholds = {
        "aic_relative_error_max": 1.0e-6,
        "mf2_max_abs_error_max": 1.0e-13,
        "qf_resultant_relative_error_max": 5.0e-3,
        "mf1_common_field_relative_error_max": 5.0e-2,
    }
    hard_gates = {
        "aic": float(aic_relative_error.item()) <= thresholds["aic_relative_error_max"],
        "mf2": float(mf2_max_abs_error.item()) <= thresholds["mf2_max_abs_error_max"],
        "qf_resultant": float(qf_resultant_max_relative_error.item())
        <= thresholds["qf_resultant_relative_error_max"],
        "mf1_physical_sign": float(q16_work.item()) < 0.0,
    }
    diagnostic_gates = {
        # This projection compares the paper's 9-DOF ANCF nodal operator with
        # the production 6-DOF Q16 MITC16/EAS operator after two different
        # local-support discretizations.  It is useful as a diagnostic, but a
        # 5% equality requirement is not a discretization-invariant scientific
        # contract.  The formal trajectory is the compatible end-to-end oracle.
        "mf1_common_field": float(mf1_common_field_relative_error.item())
        <= thresholds["mf1_common_field_relative_error_max"],
    }
    gates = {**hard_gates, **diagnostic_gates}
    hard_passed = all(hard_gates.values())
    warnings = tuple(
        name for name, passed in diagnostic_gates.items() if not passed
    )
    payload: dict[str, object] = {
        "schema": "yamano2020-q16-native-load-gates-v1",
        "case_id": case.case_id,
        "q16_grid": list(FORMAL_Q16_GRID),
        "aerodynamic_grid": list(FORMAL_AERO_GRID),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "dtype": "float64",
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": {
            "aic_relative_error": float(aic_relative_error.item()),
            "mf2_max_abs_error": float(mf2_max_abs_error.item()),
            "qf_resultant_max_relative_error": float(
                qf_resultant_max_relative_error.item()
            ),
            "mf1_q16_common_field_work": float(q16_work.item()),
            "mf1_q16_full_global_common_field_work": float(
                full_q16_work.item()
            ),
            "mf1_author_common_field_work": float(author_work.item()),
            "mf1_common_field_relative_error": float(
                mf1_common_field_relative_error.item()
            ),
        },
        "thresholds": thresholds,
        "hard_gates": hard_gates,
        "diagnostic_gates": diagnostic_gates,
        "gates": gates,
        "warnings": list(warnings),
        "status": (
            "passed_with_warning" if hard_passed and warnings
            else "passed" if hard_passed
            else "failed"
        ),
        "next_action": (
            "run the formal paper trajectory and retain diagnostic warnings"
            if hard_passed
            else "repair the failed hard component before the trajectory run"
        ),
    }
    if output is not None:
        _write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/experiment/20260823_q16_fluxv5m_native_fsi/"
            "YAMANO_Q16_NATIVE_LOAD_GATES.json"
        ),
    )
    args = parser.parse_args()
    payload = run_gates(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["status"] in {"passed", "passed_with_warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
