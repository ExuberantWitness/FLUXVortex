"""CUDA-only modal audit of the Yamano step-one Mf1 added-mass operator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import (
    Q16CudaAerodynamicLoadPacket,
)
from fluxvortex.warp_fsi.q16_lev_impulse_transfer import (
    Q16CudaLEVImpulseStripLoad,
)
from forward_flight_benchmarks.yamano2020_q16 import (
    MF1_STEP1_ORACLE_SHA256,
    MF1_STEP1_SOURCE_FIXTURE_SHA256,
    YAMANO_2020_SINGLE_SHEET,
    load_mf1_step1_oracle,
)
from q16_real_fsi_coupling import _Q16CudaFrozenAddedMassLoad
from reproduce_yamano2020_q16_fsi import _build


def _cuda_sparse(oracle: dict[str, np.ndarray], prefix: str) -> torch.Tensor:
    shape = tuple(int(value) for value in oracle["matrix_shape"].tolist())
    indices = torch.stack(
        (
            torch.as_tensor(
                np.array(oracle[f"{prefix}_row"], dtype=np.int64, copy=True),
                device="cuda:0",
                dtype=torch.int64,
            ),
            torch.as_tensor(
                np.array(oracle[f"{prefix}_col"], dtype=np.int64, copy=True),
                device="cuda:0",
                dtype=torch.int64,
            ),
        )
    )
    values = torch.as_tensor(
        np.array(oracle[f"{prefix}_data"], dtype=np.float64, copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    result = torch.sparse_coo_tensor(
        indices,
        values,
        size=shape,
        device="cuda:0",
        dtype=torch.float64,
    ).coalesce()
    if result.device.type != "cuda" or result.dtype is not torch.float64:
        raise RuntimeError("Yamano sparse oracle left CUDA float64")
    return result


def _reference_modal_operators(
    oracle: dict[str, np.ndarray],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    mass = _cuda_sparse(oracle, "mass")
    mf1 = _cuda_sparse(oracle, "mf1")
    modes = torch.as_tensor(
        np.array(oracle["phi_dq"], dtype=np.float64, copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    modal_mass = modes.T @ torch.sparse.mm(mass, modes)
    scales = torch.sqrt(torch.diagonal(modal_mass))
    modes = modes / scales.unsqueeze(0)
    modal_mass = modes.T @ torch.sparse.mm(mass, modes)
    modal_mf1 = modes.T @ torch.sparse.mm(mf1, modes)
    neumann = torch.as_tensor(
        np.array(oracle["neumann"], dtype=np.float64, copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    gamma_rate = torch.as_tensor(
        np.array(oracle["gamma_rate"], dtype=np.float64, copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    modal_neumann = neumann @ modes
    modal_gamma_rate = gamma_rate @ modes
    pressure_to_generalized = torch.sparse.mm(mf1, torch.linalg.pinv(gamma_rate))
    modal_pressure_test = modes.T @ pressure_to_generalized
    pulse = torch.as_tensor(
        np.array(oracle["pulse_q"], dtype=np.float64, copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    modal_pulse = modes.T @ pulse
    orthogonality = torch.max(
        torch.abs(modal_mass - torch.eye(5, device="cuda:0", dtype=torch.float64))
    )
    return (
        modal_mf1,
        modal_pulse,
        modes,
        modal_neumann,
        modal_gamma_rate,
        modal_pressure_test,
        float(orthogonality.item()),
    )


def _q16_modal_basis(
    owner: object,
    stepper: object,
    *,
    mode_count: int = 5,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    structural = stepper.structural_stepper
    operator = structural._operator
    constrained = wp.to_torch(structural._boundary_operator._constrained_mask).bool()
    free = torch.nonzero(~constrained, as_tuple=False).flatten()
    free_count = int(free.numel())
    reference = wp.to_torch(owner.state)[0]
    stiffness = torch.empty(
        (free_count, free_count), device="cuda:0", dtype=torch.float64
    )
    mass = torch.empty_like(stiffness)
    for start in range(0, free_count, batch_size):
        stop = min(start + batch_size, free_count)
        batch_free = free[start:stop]
        count = stop - start
        state = reference.repeat(count, 1)
        directions = torch.zeros(
            (count, operator.dof_count), device="cuda:0", dtype=torch.float64
        )
        directions[torch.arange(count, device="cuda:0"), batch_free] = 1.0
        tangent = wp.to_torch(
            operator.tangent_action(
                wp.from_torch(state, dtype=config.DTYPE),
                wp.from_torch(directions, dtype=config.DTYPE),
            )
        )
        inertia = wp.to_torch(
            operator.mass_action(wp.from_torch(directions, dtype=config.DTYPE))
        )
        stiffness[:, start:stop] = tangent[:, free].T
        mass[:, start:stop] = inertia[:, free].T
    stiffness = 0.5 * (stiffness + stiffness.T)
    mass = 0.5 * (mass + mass.T)
    cholesky = torch.linalg.cholesky(mass)
    left = torch.linalg.solve_triangular(cholesky, stiffness, upper=False)
    standard = torch.linalg.solve_triangular(cholesky, left.T, upper=False).T
    standard = 0.5 * (standard + standard.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(standard)
    if not bool(torch.all(eigenvalues[:mode_count] > 0.0).item()):
        raise RuntimeError("Q16 clamped modal pencil is not positive")
    free_modes = torch.linalg.solve_triangular(
        cholesky.T, eigenvectors[:, :mode_count], upper=True
    )
    modes = torch.zeros(
        (operator.dof_count, mode_count), device="cuda:0", dtype=torch.float64
    )
    modes[free] = free_modes
    modal_mass = free_modes.T @ mass @ free_modes
    orthogonality = torch.max(
        torch.abs(
            modal_mass - torch.eye(mode_count, device="cuda:0", dtype=torch.float64)
        )
    )
    return modes, mass, free, eigenvalues[:mode_count], float(orthogonality.item())


def run_modal_projection_audit(
    q16_chordwise_element_count: int,
    q16_spanwise_element_count: int,
    *,
    aerodynamic_chordwise_panel_count: int = 15,
    aerodynamic_spanwise_panel_count: int = 10,
) -> dict[str, object]:
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("Yamano Mf1 modal audit requires CUDA")
    oracle = load_mf1_step1_oracle()
    started = time.perf_counter()
    (
        reference_mf1,
        reference_pulse,
        _,
        reference_neumann,
        reference_gamma_rate,
        reference_pressure_test,
        reference_orthogonality,
    ) = _reference_modal_operators(oracle)
    owner, stepper, pulse = _build(
        outer_step_count=1,
        q16_chordwise_element_count=q16_chordwise_element_count,
        q16_spanwise_element_count=q16_spanwise_element_count,
        aerodynamic_chordwise_panel_count=aerodynamic_chordwise_panel_count,
        aerodynamic_spanwise_panel_count=aerodynamic_spanwise_panel_count,
        coordinate_frame="author",
    )
    modes, _, _, eigenvalues, q16_orthogonality = _q16_modal_basis(owner, stepper)
    solver = owner.aero_owner.current_solver
    packet = Q16CudaAerodynamicLoadPacket.from_solver(solver)
    lev = Q16CudaLEVImpulseStripLoad.from_solver(solver)
    complete = stepper.complete_transfer.map(packet, lev, owner.state)
    added = _Q16CudaFrozenAddedMassLoad.from_solver(
        solver, complete, stepper.complete_transfer.resolved_transfer
    )
    q16_mf1 = modes.T @ added.generalized_matrix @ modes
    q16_neumann = added.neumann_map @ modes
    q16_gamma_rate = added.gamma_rate_map @ modes
    q16_pressure_test = modes.T @ (
        -YAMANO_2020_SINGLE_SHEET.fluid_density_kg_m3 * added.pressure_to_generalized
    )
    case = YAMANO_2020_SINGLE_SHEET
    pulse_peak_time_s = 0.1 * case.length_m / case.freestream_m_s
    pulse_force = wp.to_torch(pulse.endpoint_load(pulse_peak_time_s).generalized_force)[
        0
    ]
    # q_in_norm reaches 0.5 at t*=0.1; divide it out to compare the spatial
    # generalized load vector stored by the MATLAB fixture.
    q16_pulse = modes.T @ (pulse_force / 0.5)
    alignment = torch.where(
        reference_pulse * q16_pulse < 0.0,
        -torch.ones_like(q16_pulse),
        torch.ones_like(q16_pulse),
    )
    q16_mf1 = alignment.unsqueeze(1) * q16_mf1 * alignment.unsqueeze(0)
    q16_pulse = alignment * q16_pulse
    q16_neumann = q16_neumann * alignment.unsqueeze(0)
    q16_gamma_rate = q16_gamma_rate * alignment.unsqueeze(0)
    q16_pressure_test = alignment.unsqueeze(1) * q16_pressure_test
    reference_diagonal = torch.diagonal(reference_mf1)
    q16_diagonal = torch.diagonal(q16_mf1)
    diagonal_relative_error = torch.abs(
        q16_diagonal - reference_diagonal
    ) / torch.clamp(torch.abs(reference_diagonal), min=1.0e-14)
    reference_neumann_norm = torch.linalg.vector_norm(reference_neumann, dim=0)
    q16_neumann_norm = torch.linalg.vector_norm(q16_neumann, dim=0)
    reference_gamma_rate_norm = torch.linalg.vector_norm(reference_gamma_rate, dim=0)
    q16_gamma_rate_norm = torch.linalg.vector_norm(q16_gamma_rate, dim=0)
    neumann_norm_ratio = q16_neumann_norm / reference_neumann_norm
    gamma_rate_norm_ratio = q16_gamma_rate_norm / reference_gamma_rate_norm
    reference_pressure_test_norm = torch.linalg.vector_norm(
        reference_pressure_test, dim=1
    )
    q16_pressure_test_norm = torch.linalg.vector_norm(q16_pressure_test, dim=1)
    pressure_test_norm_ratio = q16_pressure_test_norm / reference_pressure_test_norm
    pressure_test_scale = torch.sum(
        q16_pressure_test * reference_pressure_test, dim=1
    ) / torch.sum(reference_pressure_test.square(), dim=1)
    pressure_test_cosine = torch.sum(
        q16_pressure_test * reference_pressure_test, dim=1
    ) / (q16_pressure_test_norm * reference_pressure_test_norm)
    neumann_cosine = torch.sum(reference_neumann * q16_neumann, dim=0) / (
        reference_neumann_norm * q16_neumann_norm
    )
    gamma_rate_cosine = torch.sum(reference_gamma_rate * q16_gamma_rate, dim=0) / (
        reference_gamma_rate_norm * q16_gamma_rate_norm
    )
    reference_factorization_residual = torch.max(
        torch.abs(reference_pressure_test @ reference_gamma_rate - reference_mf1)
    ) / torch.clamp(torch.max(torch.abs(reference_mf1)), min=1.0)
    q16_factorization_residual = torch.max(
        torch.abs(q16_pressure_test @ q16_gamma_rate - q16_mf1)
    ) / torch.clamp(torch.max(torch.abs(q16_mf1)), min=1.0)

    aic_metrics: dict[str, object] | None = None
    reference_aic = torch.as_tensor(
        np.array(oracle["aic"], dtype=np.float64, copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    current_aic = solver._cuda_aic
    if tuple(current_aic.shape) == tuple(reference_aic.shape):
        reference_singular = torch.linalg.svdvals(reference_aic)
        current_singular = torch.linalg.svdvals(current_aic)
        reference_frobenius = torch.linalg.vector_norm(reference_aic)
        current_frobenius = torch.linalg.vector_norm(current_aic)
        aic_metrics = {
            "reference_frobenius_norm": float(reference_frobenius.item()),
            "q16_ptera_frobenius_norm": float(current_frobenius.item()),
            "q16_to_reference_frobenius_ratio": float(
                (current_frobenius / reference_frobenius).item()
            ),
            "reference_singular_value_min": float(reference_singular[-1].item()),
            "reference_singular_value_max": float(reference_singular[0].item()),
            "q16_ptera_singular_value_min": float(current_singular[-1].item()),
            "q16_ptera_singular_value_max": float(current_singular[0].item()),
        }
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in (
            reference_mf1,
            q16_mf1,
            reference_pulse,
            q16_pulse,
            diagonal_relative_error,
            reference_neumann,
            q16_neumann,
            reference_gamma_rate,
            q16_gamma_rate,
            reference_pressure_test,
            q16_pressure_test,
        )
    )
    if not finite:
        raise FloatingPointError("Yamano Mf1 modal audit is non-finite")
    torch.cuda.synchronize()
    return {
        "schema": "yamano2020-q16-mf1-modal-audit-v1",
        "case_id": case.case_id,
        "paper_doi": case.doi,
        "q16_mesh": [
            q16_chordwise_element_count,
            q16_spanwise_element_count,
        ],
        "aerodynamic_panels": [
            aerodynamic_chordwise_panel_count,
            aerodynamic_spanwise_panel_count,
        ],
        "oracle_sha256": MF1_STEP1_ORACLE_SHA256,
        "source_fixture_sha256": MF1_STEP1_SOURCE_FIXTURE_SHA256,
        "reference_modal_mf1": reference_mf1.tolist(),
        "q16_modal_mf1": q16_mf1.tolist(),
        "reference_modal_mf1_diagonal": reference_diagonal.tolist(),
        "q16_modal_mf1_diagonal": q16_diagonal.tolist(),
        "modal_mf1_diagonal_relative_error": diagonal_relative_error.tolist(),
        "modal_mf1_diagonal_rms_relative_error": float(
            torch.sqrt(torch.mean(diagonal_relative_error.square())).item()
        ),
        "modal_mf1_diagonal_max_relative_error": float(
            torch.max(diagonal_relative_error).item()
        ),
        "reference_unit_pulse_modal_force": reference_pulse.tolist(),
        "q16_unit_pulse_modal_force": q16_pulse.tolist(),
        "reference_modal_neumann_l2": reference_neumann_norm.tolist(),
        "q16_modal_neumann_l2": q16_neumann_norm.tolist(),
        "q16_to_reference_modal_neumann_l2_ratio": neumann_norm_ratio.tolist(),
        "q16_to_reference_modal_neumann_cosine": neumann_cosine.tolist(),
        "reference_modal_gamma_rate_l2": reference_gamma_rate_norm.tolist(),
        "q16_modal_gamma_rate_l2": q16_gamma_rate_norm.tolist(),
        "q16_to_reference_modal_gamma_rate_l2_ratio": (gamma_rate_norm_ratio.tolist()),
        "q16_to_reference_modal_gamma_rate_cosine": gamma_rate_cosine.tolist(),
        "reference_modal_pressure_test_l2": reference_pressure_test_norm.tolist(),
        "q16_modal_pressure_test_l2": q16_pressure_test_norm.tolist(),
        "q16_to_reference_modal_pressure_test_l2_ratio": (
            pressure_test_norm_ratio.tolist()
        ),
        "q16_to_reference_modal_pressure_test_scale": pressure_test_scale.tolist(),
        "q16_to_reference_modal_pressure_test_cosine": pressure_test_cosine.tolist(),
        "reference_mf1_factorization_relative_residual": float(
            reference_factorization_residual.item()
        ),
        "q16_mf1_factorization_relative_residual": float(
            q16_factorization_residual.item()
        ),
        "aic_spectral_metrics": aic_metrics,
        "q16_omega_star": (
            torch.sqrt(eigenvalues) * case.length_m / case.freestream_m_s
        ).tolist(),
        "reference_modal_mass_orthogonality_max_abs": reference_orthogonality,
        "q16_modal_mass_orthogonality_max_abs": q16_orthogonality,
        "cuda_scientific_path": True,
        "cpu_numerical_fallback": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _pair(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("value must be AxB")
    try:
        result = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair values must be integers") from error
    if len(result) != 2 or any(value <= 0 for value in result):
        raise argparse.ArgumentTypeError("pair values must be positive")
    return result[0], result[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q16-mesh", type=_pair, default=(2, 2))
    parser.add_argument("--aero-panels", type=_pair, default=(15, 10))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_modal_projection_audit(
        args.q16_mesh[0],
        args.q16_mesh[1],
        aerodynamic_chordwise_panel_count=args.aero_panels[0],
        aerodynamic_spanwise_panel_count=args.aero_panels[1],
    )
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
        return 0
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
