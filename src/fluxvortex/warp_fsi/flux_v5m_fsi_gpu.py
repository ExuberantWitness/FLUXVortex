"""Fail-closed FLUX-V5M production facade for the Warp FSI data plane."""
from __future__ import annotations

import sys
from typing import Any

import warp as wp

from . import config

FSI_GPU_CONTRACT = "flux-v5m-warp-fsi-cuda-no-science-host-fallback-v1"
_FORBIDDEN_REFERENCE_MODULES = (
    "fluxvortex.warp_fsi.ml_fluid",
    "fluxvortex.warp_fsi.ml_chain",
)


def assert_v5m_fsi_runtime() -> str:
    """Attest the strict CUDA runtime before importing numerical kernels."""

    if not config.GPU_ONLY:
        raise RuntimeError("FLUX-V5M FSI requires FLUXV_GPU_ONLY=1")
    if config.dtype_name() != "float64" or config.DTYPE != wp.float64:
        raise RuntimeError("FLUX-V5M FSI requires the frozen float64 precision")
    device = wp.get_device(config.DEVICE)
    if not device.is_cuda:
        raise RuntimeError("FLUX-V5M FSI numerical device must be CUDA")
    loaded = tuple(name for name in _FORBIDDEN_REFERENCE_MODULES if name in sys.modules)
    if loaded:
        raise RuntimeError(
            "CPU MATLAB-parity reference modules are forbidden in V5M production: "
            + ", ".join(loaded)
        )
    return device.alias


def _require_cuda_warp_array(name: str, value: Any, device: str) -> None:
    if not isinstance(value, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if value.device.alias != device or not value.device.is_cuda:
        raise ValueError(f"{name} must reside on {device}")
    if value.dtype != config.DTYPE:
        raise TypeError(f"{name} dtype differs from the frozen V5M dtype")


def make_v5m_gpu_fluid_solver(
    solver: Any,
    *,
    scgeom: Any = None,
    wake: bool = False,
    wake_max_rows: int = 64,
) -> Any:
    """Construct the authorised Warp CUDA fluid solver."""

    device = assert_v5m_fsi_runtime()
    from .coupled import GpuFluidSolve

    return GpuFluidSolve(
        solver,
        scgeom=scgeom,
        device=device,
        wake=wake,
        wake_max_rows=wake_max_rows,
    )


def run_v5m_gpu_coupled_trajectory(
    constants: Any,
    fluid_solver: Any,
    q0: Any,
    dq0: Any,
    pulse_shape: Any,
    profile: Any,
    delta_time: float,
    num_steps: int,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Run the supported FSI trajectory with a CUDA-only numerical data plane."""

    device = assert_v5m_fsi_runtime()
    _require_cuda_warp_array("q0", q0, device)
    _require_cuda_warp_array("dq0", dq0, device)
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("num_steps must be a positive exact int")
    if (
        not hasattr(fluid_solver, "device")
        or wp.get_device(fluid_solver.device).alias != device
    ):
        raise ValueError("fluid solver is not bound to the V5M CUDA device")
    from .coupled import gpu_coupled_trajectory

    result = gpu_coupled_trajectory(
        constants,
        fluid_solver,
        q0,
        dq0,
        pulse_shape,
        profile,
        delta_time,
        num_steps,
        device=device,
        **kwargs,
    )
    _require_cuda_warp_array("final q", result[0], device)
    _require_cuda_warp_array("final dq", result[1], device)
    assert_v5m_fsi_runtime()
    return result


__all__ = [
    "FSI_GPU_CONTRACT",
    "assert_v5m_fsi_runtime",
    "make_v5m_gpu_fluid_solver",
    "run_v5m_gpu_coupled_trajectory",
]
