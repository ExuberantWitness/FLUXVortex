"""CUDA-only Q16 modal entry gate for Yamano et al. (2020)."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_mesh import Q16CudaMITC16EASMeshOperator
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET,
    load_nondimensional_natural_frequencies,
    make_yamano2020_q16_model,
    validate_yamano2020_sources,
)


def _lowest_clamped_modes(
    eigenvalues: torch.Tensor,
    *,
    mode_count: int,
) -> torch.Tensor:
    if eigenvalues.device.type != "cuda" or eigenvalues.dtype is not torch.float64:
        raise RuntimeError("Yamano eigenvalues left CUDA float64")
    if eigenvalues.numel() < mode_count:
        raise RuntimeError("Yamano Q16 pencil has too few modes")
    modes = eigenvalues[:mode_count]
    # This case clamps the complete leading-edge Q16 state, so it has no rigid
    # modes.  A global relative cutoff is invalid: the through-thickness modes
    # make the pencil span O(h^-2), and such a cutoff deletes the physical
    # low-frequency bending modes when h/L=1e-3.
    if not bool(torch.all(modes > 0.0).item()):
        raise RuntimeError("Yamano leading-edge-clamped Q16 pencil is not positive")
    return modes


def run_modal_case(
    chordwise_element_count: int,
    spanwise_element_count: int,
    *,
    device: str = "cuda:0",
    assembly_batch_size: int = 64,
) -> dict[str, object]:
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Yamano Q16 modal reproduction requires CUDA")
    if not wp.is_cuda_available():
        raise RuntimeError("Yamano Q16 modal reproduction requires Warp CUDA")
    if type(assembly_batch_size) is not int or assembly_batch_size <= 0:
        raise ValueError("assembly_batch_size must be a positive exact int")
    validate_yamano2020_sources()
    case = YAMANO_2020_SINGLE_SHEET
    reference = torch.as_tensor(
        np.array(
            load_nondimensional_natural_frequencies(),
            dtype=np.float64,
            copy=True,
        ),
        device=selected,
        dtype=torch.float64,
    )
    mesh, model, constraints = make_yamano2020_q16_model(
        chordwise_element_count=chordwise_element_count,
        spanwise_element_count=spanwise_element_count,
        case=case,
    )
    free = torch.as_tensor(
        np.array(constraints.free_dofs, dtype=np.int64, copy=True),
        device=selected,
        dtype=torch.int64,
    )
    free_count = int(free.numel())
    reference_state = torch.as_tensor(
        np.array(mesh.reference_state, dtype=np.float64, copy=True),
        device=selected,
        dtype=torch.float64,
    )

    operator = Q16CudaMITC16EASMeshOperator(model, device=device)
    torch.cuda.reset_peak_memory_stats(selected)
    torch.cuda.synchronize(selected)
    started = time.perf_counter()
    stiffness = torch.empty(
        (free_count, free_count), device=selected, dtype=torch.float64
    )
    mass = torch.empty_like(stiffness)
    largest_batch = min(assembly_batch_size, free_count)
    for start in range(0, free_count, assembly_batch_size):
        stop = min(start + assembly_batch_size, free_count)
        batch_free = free[start:stop]
        batch_count = stop - start
        state = reference_state.repeat(batch_count, 1)
        directions = torch.zeros(
            (batch_count, mesh.dof_count),
            device=selected,
            dtype=torch.float64,
        )
        directions[torch.arange(batch_count, device=selected), batch_free] = 1.0
        tangent_actions = wp.to_torch(
            operator.tangent_action(
                wp.from_torch(state, dtype=config.DTYPE),
                wp.from_torch(directions, dtype=config.DTYPE),
            )
        )
        mass_actions = wp.to_torch(
            operator.mass_action(wp.from_torch(directions, dtype=config.DTYPE))
        )
        stiffness[:, start:stop] = tangent_actions[:, free].T
        mass[:, start:stop] = mass_actions[:, free].T
    stiffness_scale = torch.clamp(torch.max(torch.abs(stiffness)), min=1.0)
    mass_scale = torch.clamp(torch.max(torch.abs(mass)), min=1.0)
    stiffness_symmetry = torch.max(torch.abs(stiffness - stiffness.T)) / stiffness_scale
    mass_symmetry = torch.max(torch.abs(mass - mass.T)) / mass_scale
    stiffness = 0.5 * (stiffness + stiffness.T)
    mass = 0.5 * (mass + mass.T)
    cholesky, info = torch.linalg.cholesky_ex(mass)
    if int(torch.max(info).item()) != 0:
        raise RuntimeError("Yamano Q16 constrained mass matrix is not SPD")
    left_solved = torch.linalg.solve_triangular(
        cholesky, stiffness, upper=False
    )
    standard = torch.linalg.solve_triangular(
        cholesky, left_solved.T, upper=False
    ).T
    standard = 0.5 * (standard + standard.T)
    eigenvalues = torch.linalg.eigvalsh(standard)
    modes = _lowest_clamped_modes(
        eigenvalues, mode_count=int(reference.numel())
    )
    omega_star = (
        torch.sqrt(modes) * case.length_m / case.freestream_m_s
    )
    relative_error = torch.abs(omega_star - reference) / reference
    torch.cuda.synchronize(selected)
    elapsed = time.perf_counter() - started
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (stiffness, mass, eigenvalues, omega_star, relative_error)
    ):
        raise FloatingPointError("Yamano Q16 modal result is non-finite")
    return {
        "case_id": case.case_id,
        "paper_doi": case.doi,
        "q16_chordwise_element_count": chordwise_element_count,
        "q16_spanwise_element_count": spanwise_element_count,
        "q16_element_count": mesh.element_count,
        "q16_node_count": mesh.node_count,
        "q16_total_dof_count": mesh.dof_count,
        "q16_free_dof_count": constraints.free_dof_count,
        "gpu_assembly_batch_size": largest_batch,
        "gpu_assembly_batch_count": (
            free_count + assembly_batch_size - 1
        )
        // assembly_batch_size,
        "leading_edge_node_count": int(constraints.constrained_nodes.size),
        "young_modulus_pa": case.young_modulus_pa,
        "solid_density_kg_m3": case.solid_density_kg_m3,
        "reference_omega_star": reference.tolist(),
        "q16_omega_star": omega_star.tolist(),
        "relative_errors": relative_error.tolist(),
        "rms_relative_error": float(
            torch.sqrt(torch.mean(relative_error * relative_error)).item()
        ),
        "max_relative_error": float(torch.max(relative_error).item()),
        "stiffness_symmetry_relative": float(stiffness_symmetry.item()),
        "mass_symmetry_relative": float(mass_symmetry.item()),
        "minimum_mass_cholesky_diagonal": float(
            torch.min(torch.diagonal(cholesky)).item()
        ),
        "generalized_eigenvalue_minimum": float(torch.min(eigenvalues).item()),
        "generalized_eigenvalue_maximum": float(torch.max(eigenvalues).item()),
        "clamped_mode_selection_threshold": 0.0,
        "lowest_generalized_eigenvalues": eigenvalues[:16].tolist(),
        "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(
            selected
        ),
        "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(selected),
        "elapsed_seconds": elapsed,
        "cuda_scientific_path": True,
        "cpu_numerical_fallback": False,
    }


def _mesh_spec(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mesh must be CHORDxSPAN")
    try:
        chord, span = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("mesh counts must be integers") from error
    if chord <= 0 or span <= 0:
        raise argparse.ArgumentTypeError("mesh counts must be positive")
    return chord, span


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meshes",
        nargs="+",
        type=_mesh_spec,
        default=[(1, 1), (2, 2), (5, 3)],
    )
    args = parser.parse_args()
    results = [run_modal_case(chord, span) for chord, span in args.meshes]
    payload = {
        "schema": "yamano2020-q16-modal-case-entry-v1",
        "status": "PASS",
        "results": results,
    }
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
