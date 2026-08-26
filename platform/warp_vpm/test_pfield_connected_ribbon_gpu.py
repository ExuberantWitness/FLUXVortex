"""Mechanical gates for node-owned connected ribbon deposition on CUDA."""
from __future__ import annotations

import pytest
import torch

from pfield_torch_gpu import CudaParticleField


def _nodes() -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda:0")
    anchor = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        device=device,
        dtype=torch.float64,
    )
    frontier = anchor + torch.tensor(
        [0.2, 0.0, 0.0], device=device, dtype=torch.float64
    )
    return anchor, frontier


def test_uniform_cells_cancel_the_shared_chordwise_edge() -> None:
    anchor, frontier = _nodes()
    field = CudaParticleField(capacity=128, device="cuda:0")
    count = field.add_connected_ribbon_particles(
        anchor,
        frontier,
        torch.tensor([2.0, 2.0], device="cuda:0", dtype=torch.float64),
        smoothing_radius=0.25,
        target_spacing=0.1,
        birth_step=3,
    )
    diagnostics = field.last_connected_ribbon_diagnostics
    assert diagnostics is not None
    assert count == 44
    assert diagnostics["retained_edge_count"] == 6
    assert diagnostics["seam_count"] == 0
    assert diagnostics["global_vector_closure"] <= 2.0e-14
    assert bool(torch.all(field.sigmas_cuda == 0.25).item())
    assert bool(torch.all(field.birth_step[: field.n] == 3).item())


def test_nonuniform_cells_retain_one_owned_interior_edge() -> None:
    anchor, frontier = _nodes()
    field = CudaParticleField(capacity=128, device="cuda:0")
    count = field.add_connected_ribbon_particles(
        anchor,
        frontier,
        torch.tensor([2.0, 1.0], device="cuda:0", dtype=torch.float64),
        smoothing_radius=0.25,
        target_spacing=0.1,
        birth_step=1,
    )
    diagnostics = field.last_connected_ribbon_diagnostics
    assert diagnostics is not None
    assert count == 46
    assert diagnostics["retained_edge_count"] == 7
    assert diagnostics["global_vector_closure"] <= 2.0e-14
    assert int(torch.count_nonzero(field.source_strips_cuda == -1).item()) == 6


def test_connected_ribbon_rejects_host_data_and_insufficient_overlap() -> None:
    anchor, frontier = _nodes()
    field = CudaParticleField(capacity=128, device="cuda:0")
    with pytest.raises(ValueError, match="implicit upload"):
        field.add_connected_ribbon_particles(
            anchor.cpu(),
            frontier,
            torch.ones(2, device="cuda:0", dtype=torch.float64),
            smoothing_radius=0.25,
            target_spacing=0.1,
            birth_step=0,
        )
    with pytest.raises(ValueError, match="minimum overlap"):
        field.add_connected_ribbon_particles(
            anchor,
            frontier,
            torch.ones(2, device="cuda:0", dtype=torch.float64),
            smoothing_radius=0.2,
            target_spacing=0.1,
            birth_step=0,
        )


def test_production_connector_owners_are_retained_on_cuda() -> None:
    anchor, frontier = _nodes()
    field = CudaParticleField(capacity=128, device="cuda:0")
    owners = torch.tensor([0, 0, 1], device="cuda:0", dtype=torch.int64)
    field.add_connected_ribbon_particles(
        anchor,
        frontier,
        torch.tensor([2.0, 1.0], device="cuda:0", dtype=torch.float64),
        smoothing_radius=0.25,
        target_spacing=0.1,
        birth_step=2,
        connector_source_strips=owners,
    )
    assert bool(torch.all((field.source_strips_cuda >= 0)).item())
    assert bool(torch.all((field.source_strips_cuda < 2)).item())
    assert field.last_connected_ribbon_diagnostics is not None
    assert field.last_connected_ribbon_diagnostics[
        "connectors_have_strip_owner"
    ] is True
