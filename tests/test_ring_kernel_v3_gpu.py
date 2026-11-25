"""Ring kernel, fused Warp kernel against the torch path and against an independent law.  GPU.

Under FLUXV_RING_KERNEL=v3 the fused kernel `_ring_bs_partials_v3` must give the same
velocities as the torch path of the same module -- the influence matrix is built from the
torch path and everything else from the fused one, so a difference here would make the
solve inconsistent with the wake it was solved against -- and both must agree with a
per-leg Biot-Savart written out here, with and without the core radius.

Under the default flag the same comparison guards the legacy pair.

Usage
-----
    PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 PYTHONPATH=src:platform:platform/warp_vpm \\
        python3 -m pytest tests/test_ring_kernel_v3_gpu.py -q
"""

from __future__ import annotations

import importlib
import math
import os

import pytest
import torch

CHORD, SPAN = 0.0688, 0.1375
NC, NS = 15, 30
DEVICE = "cuda:0"


@pytest.fixture
def kernel_v3():
    saved = os.environ.get("FLUXV_RING_KERNEL")
    os.environ["FLUXV_RING_KERNEL"] = "v3"
    try:
        yield importlib.reload(_kernel_module())
    finally:
        if saved is None:
            os.environ.pop("FLUXV_RING_KERNEL", None)
        else:
            os.environ["FLUXV_RING_KERNEL"] = saved
        importlib.reload(_kernel_module())


@pytest.fixture
def kernel_legacy():
    saved = os.environ.get("FLUXV_RING_KERNEL")
    os.environ.pop("FLUXV_RING_KERNEL", None)
    try:
        yield importlib.reload(_kernel_module())
    finally:
        if saved is not None:
            os.environ["FLUXV_RING_KERNEL"] = saved
        importlib.reload(_kernel_module())


def _kernel_module():
    import fluxvortex.warp_fsi.kernels_ring_velocity as kernel

    return kernel


def ring_lattice(count: int, size: float) -> torch.Tensor:
    """`count` closed rings of edge ~size with random shapes, so no two legs coincide."""

    generator = torch.Generator().manual_seed(3)
    corners = torch.rand((count, 4, 3), generator=generator, dtype=torch.float64)
    centres = torch.rand((count, 1, 3), generator=generator, dtype=torch.float64) * 40.0 * size
    return corners * size + centres


def reference(points: torch.Tensor, rings: torch.Tensor, gamma: torch.Tensor, core_radius: float | None) -> torch.Tensor:
    """Per-leg Biot-Savart summed with the gamma of its ring; zero on a leg's line."""

    total = torch.zeros_like(points)
    for leg in range(4):
        start = rings[:, leg]
        end = rings[:, (leg + 1) % 4]
        a = points[:, None, :] - start[None, :, :]
        b = points[:, None, :] - end[None, :, :]
        edge = a - b
        cross = torch.linalg.cross(a, b, dim=2)
        cross_sq = torch.sum(cross * cross, dim=2)
        edge_sq = torch.sum(edge * edge, dim=2)
        unit_difference = (
            a / torch.linalg.vector_norm(a, dim=2, keepdim=True)
            - b / torch.linalg.vector_norm(b, dim=2, keepdim=True)
        )
        scalar = torch.sum(edge * unit_difference, dim=2)
        # The same dimensionless collinear test the v3 kernel uses.  A point exactly on
        # a leg's line does not always give a x b = 0: away from the axes it comes out as
        # rounding noise, and the raw law then returns a noise-dominated value of order
        # |e| / (4 pi |a||b|) -- the unregularised law is simply undefined there.
        on_line = cross_sq <= 1.0e-24 * edge_sq * edge_sq
        factor = torch.where(
            ~on_line, scalar / (4.0 * math.pi * torch.clamp(cross_sq, min=1e-300)),
            torch.zeros_like(scalar),
        )
        if core_radius is not None:
            h_sq = cross_sq / torch.clamp(edge_sq, min=1e-300)
            factor = factor * h_sq / torch.sqrt(h_sq * h_sq + core_radius**4)
        total = total + torch.sum(cross * (factor * gamma)[:, :, None], dim=1)
    return total


def _case(size: float, count: int, points_count: int):
    generator = torch.Generator().manual_seed(5)
    rings = ring_lattice(count, size).to(DEVICE)
    gamma = torch.randn(count, generator=generator, dtype=torch.float64).to(DEVICE) * 0.03
    points = (torch.rand((points_count, 3), generator=generator, dtype=torch.float64) * 60.0 - 10.0) * size
    return rings, gamma, points.to(DEVICE)


def _on_line(rings: torch.Tensor) -> torch.Tensor:
    """A point on the line of the first leg, beyond its end (not on a coordinate axis)."""

    leg_vector = rings[0, 1] - rings[0, 0]
    return rings[0, 0] + (0.37 + math.sqrt(2.0)) * leg_vector


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only")
def test_v3_fused_matches_torch_and_the_law(kernel_v3):
    size = CHORD / NC
    rings, gamma, points = _case(size, 37, 900)
    points = torch.cat((points, _on_line(rings)[None, :]), dim=0)
    fused = kernel_v3.native_ring_velocity_fused(points, rings, gamma, core_fraction=0.0, reference_length=0.0)
    torch_path = torch.sum(
        kernel_v3.native_ring_velocity_expanded(points, rings, core_fraction=0.0, reference_length=0.0)
        * gamma[None, :, None], dim=1
    )
    assert torch.isfinite(fused).all()
    assert float((fused - torch_path).abs().max() / fused.abs().max()) < 1.0e-12
    assert float((fused - reference(points, rings, gamma, None)).abs().max() / fused.abs().max()) < 1.0e-12
    # the on-line point: the leg it lies on contributes exactly zero, the others stand
    row = fused[-1]
    others = torch.cat((rings[1:], rings[:1]), dim=0)        # rotate so the leg is a different one
    assert torch.isfinite(row).all()
    assert float(row.abs().max()) < 1.0e3 * float(gamma.abs().max()) / size
    del others


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only")
def test_v3_core_radius_on_the_fused_kernel(kernel_v3):
    size = CHORD / NC
    rings, gamma, points = _case(size, 37, 900)
    core = 0.25 * size
    fused = kernel_v3.native_ring_velocity_fused(
        points, rings, gamma, core_fraction=0.0, reference_length=0.0, core_radius=core
    )
    assert torch.isfinite(fused).all()
    assert float((fused - reference(points, rings, gamma, core)).abs().max() / fused.abs().max()) < 1.0e-12
    # far from every filament the core is not felt
    far = points + 4000.0 * size          # every ring lies within 45 sizes of the origin
    plain = kernel_v3.native_ring_velocity_fused(far, rings, gamma, core_fraction=0.0, reference_length=0.0)
    cored = kernel_v3.native_ring_velocity_fused(
        far, rings, gamma, core_fraction=0.0, reference_length=0.0, core_radius=core
    )
    assert float((plain - cored).abs().max() / plain.abs().max()) < 1.0e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only")
def test_v3_scale_gate_g_scale(kernel_v3):
    """Plan V3 gate G-Scale: velocity x length is invariant under x1 / x10 / x100 to 1e-10.

    One square ring evaluated at its centre, at the panel size of the Rojratsirikul wing
    and at ten and a hundred times that, on the fused kernel and on the torch path.
    """

    products = {"fused": [], "torch": []}
    for scale in (1.0, 10.0, 100.0):
        size = CHORD / NC * scale
        ring = torch.tensor(
            [[[0.0, 0.0, 0.0], [size, 0.0, 0.0], [size, size, 0.0], [0.0, size, 0.0]]],
            dtype=torch.float64, device=DEVICE,
        )
        centre = torch.tensor([[0.5 * size, 0.5 * size, 0.0]], dtype=torch.float64, device=DEVICE)
        unit = torch.ones(1, dtype=torch.float64, device=DEVICE)
        fused = kernel_v3.native_ring_velocity_fused(
            centre, ring, unit, core_fraction=0.0, reference_length=0.0
        )
        torch_path = kernel_v3.native_ring_velocity_expanded(
            centre, ring, core_fraction=0.0, reference_length=0.0
        )[0, 0]
        products["fused"].append(float(fused[0, 2].item()) * size)
        products["torch"].append(float(torch_path[2].item()) * size)
    for name, values in products.items():
        spread = max(values) - min(values)
        assert spread <= 1.0e-10 * abs(values[0]), (name, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only")
def test_legacy_pair_unchanged(kernel_legacy):
    assert kernel_legacy._KERNEL_V3 is False
    size = CHORD / NC
    rings, gamma, points = _case(size, 37, 900)
    fused = kernel_legacy.native_ring_velocity_fused(
        points, rings, gamma, core_fraction=1.0e-6, reference_length=1.0 / NC
    )
    torch_path = torch.sum(
        kernel_legacy.native_ring_velocity_expanded(
            points, rings, core_fraction=1.0e-6, reference_length=1.0 / NC
        )
        * gamma[None, :, None], dim=1
    )
    assert float((fused - torch_path).abs().max() / fused.abs().max()) < 1.0e-12
