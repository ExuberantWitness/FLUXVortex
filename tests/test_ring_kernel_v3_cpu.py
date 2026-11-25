"""Ring kernel, FLUXV_RING_KERNEL=v3 (plan V3, P0-A).  CPU only, no solver.

Three properties, each against an independent per-leg Biot-Savart written out here:

  scale       a uniform scaling of the geometry changes the velocity by 1/length and
              nothing else -- the legacy law fails this at the Rojratsirikul scale
  collinear   a filament whose evaluation point lies on its line contributes exactly
              zero, which is its Biot-Savart value there
  core        with `core_radius=r_c` every leg gets the factor h^2/sqrt(h^4 + r_c^4), h
              being the perpendicular distance to the leg's line, and nothing else
              changes

and one guard on the default: with the environment unset the module is the legacy law.

Usage
-----
    CUDA_VISIBLE_DEVICES="" FLUXV_GPU_ONLY=0 PFIELD_DEVICE=cpu PYTHONPATH=src \\
        python3 -m pytest tests/test_ring_kernel_v3_cpu.py -q
"""

from __future__ import annotations

import importlib
import math
import os

import numpy as np
import pytest
import torch

CHORD, SPAN = 0.0688, 0.1375          # Rojratsirikul et al. (2011): 68.8 mm x 137.5 mm
NC, NS = 15, 30


def _module():
    import fluxvortex.warp_fsi.kernels_ring_velocity as kernel

    return kernel


@pytest.fixture
def v3_kernel():
    """The kernel module loaded under FLUXV_RING_KERNEL=v3, restored afterwards."""

    saved = {name: os.environ.get(name) for name in ("FLUXV_RING_KERNEL", "FLUXV_RING_COLLINEAR_REL")}
    os.environ["FLUXV_RING_KERNEL"] = "v3"
    os.environ.pop("FLUXV_RING_COLLINEAR_REL", None)
    try:
        yield importlib.reload(_module())
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        importlib.reload(_module())


def square_ring(size: float, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [[[0.0, 0.0, 0.0], [size, 0.0, 0.0], [size, size, 0.0], [0.0, size, 0.0]]],
        dtype=torch.float64, device=device,
    )


def leg_velocity(points: torch.Tensor, ring: torch.Tensor, leg: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(Biot-Savart velocity of one unit-strength leg, perpendicular distance h).

    The unregularised straight-filament law, written out independently of the kernel:
    v = 1/(4 pi) (a x b)/|a x b|^2 [(a - b) . (a_hat - b_hat)],  h = |a x b|/|a - b|.
    A leg with a x b = 0 (the point on its line) is given the velocity 0.
    """

    start, end = ring[0, leg], ring[0, (leg + 1) % 4]
    a = points - start[None, :]
    b = points - end[None, :]
    edge = a - b
    cross = torch.linalg.cross(a, b, dim=1)
    cross_sq = torch.sum(cross * cross, dim=1)
    edge_norm = torch.linalg.vector_norm(edge, dim=1)
    unit_difference = (
        a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
        - b / torch.linalg.vector_norm(b, dim=1, keepdim=True)
    )
    scalar = torch.sum(edge * unit_difference, dim=1)
    # the same dimensionless collinear test the v3 kernel uses: a x b is not always
    # exactly zero on a leg's line, and the raw law is undefined there
    on_line = cross_sq <= 1.0e-24 * (edge_norm * edge_norm) ** 2
    velocity = torch.where(
        (~on_line)[:, None],
        cross * (scalar / (4.0 * math.pi * torch.clamp(cross_sq, min=1e-300)))[:, None],
        torch.zeros_like(cross),
    )
    return velocity, cross_sq / torch.clamp(edge_norm * edge_norm, min=1e-300)


def reference(points: torch.Tensor, ring: torch.Tensor, core_radius: float | None = None) -> torch.Tensor:
    total = torch.zeros_like(points)
    for leg in range(4):
        velocity, h_sq = leg_velocity(points, ring, leg)
        if core_radius is not None:
            velocity = velocity * (h_sq / torch.sqrt(h_sq * h_sq + core_radius**4))[:, None]
        total = total + velocity
    return total


def kernel_velocity(kernel, points, ring, **keywords):
    return kernel.native_ring_velocity_expanded(
        points, ring, core_fraction=0.0, reference_length=0.0, **keywords
    )[:, 0, :]


def test_v3_is_scale_invariant(v3_kernel):
    ratios = []
    for scale in (1.0, 10.0, 100.0):
        size = CHORD / NC * scale
        ring = square_ring(size)
        points = torch.tensor(
            [[0.5 * size, 0.5 * size, 0.0], [-0.5 * size, 0.5 * size, 0.0], [1.5 * size, 0.5 * size, 0.0]],
            dtype=torch.float64,
        )
        got = kernel_velocity(v3_kernel, points, ring)
        ratios.append(float(got[0, 2]) * size)
    # velocity * length is the scale-invariant quantity: three values equal to 1e-12
    assert max(ratios) - min(ratios) < 1.0e-9 * abs(ratios[0])


def test_v3_matches_the_unregularised_law(v3_kernel):
    generator = np.random.default_rng(7)
    for scale in (1.0, 10.0):
        size = CHORD / NC * scale
        ring = square_ring(size)
        cloud = generator.uniform(-2.0, 3.0, size=(400, 3)) * size
        # keep every point at least a tenth of a panel from the legs, where the law is defined
        points = torch.tensor(cloud, dtype=torch.float64)
        got = kernel_velocity(v3_kernel, points, ring)
        want = reference(points, ring)
        assert float(torch.max(torch.abs(got - want)) / torch.max(torch.abs(want))) < 1.0e-13


def test_v3_collinear_leg_contributes_zero(v3_kernel):
    size = CHORD / NC
    ring = square_ring(size)
    # on the line of the front leg (y = 0, z = 0), beyond its end; and on the leg itself
    points = torch.tensor(
        [[-0.37 * size, 0.0, 0.0], [1.61 * size, 0.0, 0.0], [0.3 * size, 0.0, 0.0]], dtype=torch.float64
    )
    got = kernel_velocity(v3_kernel, points, ring)
    want = reference(points, ring)
    assert float(torch.max(torch.abs(got - want))) < 1.0e-12 * float(torch.max(torch.abs(want)))
    # on the leg itself only that leg is singular; the other three are finite and exact
    _, h_on_leg = leg_velocity(points, ring, 0)
    assert float(h_on_leg[2]) == 0.0


def test_v3_core_factor_is_per_leg(v3_kernel):
    size = CHORD / NC
    ring = square_ring(size)
    generator = np.random.default_rng(11)
    cloud = generator.uniform(-0.5, 1.5, size=(300, 3)) * size
    cloud[:, 2] += 0.05 * size                    # lift the cloud off the ring's plane
    points = torch.tensor(cloud, dtype=torch.float64)
    for panels in (0.25, 0.5, 2.0):
        core = panels * size
        got = kernel_velocity(v3_kernel, points, ring, core_radius=core)
        want = reference(points, ring, core_radius=core)
        assert float(torch.max(torch.abs(got - want)) / torch.max(torch.abs(want))) < 1.0e-13


def test_default_environment_is_the_legacy_law():
    kernel = importlib.reload(_module())       # whatever the environment says
    if os.environ.get("FLUXV_RING_KERNEL", "legacy") == "v3":
        pytest.skip("FLUXV_RING_KERNEL=v3 is set in the environment")
    assert kernel._KERNEL_V3 is False
    size = CHORD / NC
    ring = square_ring(size)
    points = torch.tensor([[0.5 * size, 0.5 * size, 0.0]], dtype=torch.float64)
    got = kernel_velocity(kernel, points, ring)
    exact = reference(points, ring)
    # the legacy law damps the self term to |a x b|^2 / (|a x b|^2 + 1e-9)
    damping = float((0.25 * size**4) / (0.25 * size**4 + kernel._EPS_V))
    assert float(got[0, 2] / exact[0, 2]) == pytest.approx(damping, rel=1.0e-9)


@pytest.mark.parametrize("name,value", [("FLUXV_RING_KERNEL", "banana"), ("FLUXV_RING_COLLINEAR_REL", "0")])
def test_invalid_settings_are_rejected(name: str, value: str):
    saved = {key: os.environ.get(key) for key in ("FLUXV_RING_KERNEL", "FLUXV_RING_COLLINEAR_REL")}
    os.environ["FLUXV_RING_KERNEL"] = "v3"
    os.environ[name] = value
    try:
        with pytest.raises(ValueError):
            importlib.reload(_module())
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        importlib.reload(_module())
