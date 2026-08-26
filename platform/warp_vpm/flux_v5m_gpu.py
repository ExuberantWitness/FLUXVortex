"""Single strict-CUDA production entry point for FLUX-V5M.

The legacy modules in this directory remain useful numerical references, but
they are not production entry points: several of them intentionally retain
NumPy implementations.  This module is the fail-closed boundary for V5M.

CPU work is limited to Ptera object construction, Python control flow and
serialization.  A capability is advertised as ``production_cuda`` only after
its complete scientific data path has an executable CUDA contract test.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaAttachedJointLEVTEVSolver

GPU_NUMERICAL_CONTRACT = "flux-v5m-science-data-plane-cuda-float64-v1"
PRODUCTION_STATUS = "production_cuda"
FAIL_CLOSED_STATUS = "fail_closed_pending_cuda_port"
OUT_OF_SCOPE_STATUS = "outside_v5m_scope_fail_closed"


@dataclass(frozen=True, slots=True)
class FluxV5MCapability:
    """Immutable statement of one production capability."""

    name: str
    status: str
    evidence: tuple[str, ...]
    restriction: str


_CAPABILITIES = (
    FluxV5MCapability(
        name="ptera_attached_single_wing_prescribed_wake",
        status=FAIL_CLOSED_STATUS,
        evidence=("test_flux_v5m_gpu_entry.py",),
        restriction=(
            "legacy diagnostic backend only; V5M production requires separated "
            "LEV, joint TEV and free wake"
        ),
    ),
    FluxV5MCapability(
        name="ldvm_and_finite_wing_corrections",
        status=PRODUCTION_STATUS,
        evidence=("test_gpu_only_corrections.py",),
        restriction="all dynamic histories are CUDA float64 on one device",
    ),
    FluxV5MCapability(
        name="ptera_active_lev",
        status=PRODUCTION_STATUS,
        evidence=("test_ptera_gpu_active_lev.py",),
        restriction="one airplane, one wing, joint TEV and free wake mandatory",
    ),
    FluxV5MCapability(
        name="ptera_joint_lev_tev",
        status=PRODUCTION_STATUS,
        evidence=("test_ptera_gpu_active_lev.py",),
        restriction="one airplane, one wing, separated LEV and free wake mandatory",
    ),
    FluxV5MCapability(
        name="ptera_free_wake",
        status=PRODUCTION_STATUS,
        evidence=("test_ptera_gpu_only_backend.py",),
        restriction=(
            "one airplane, one wing, no image surface; separated LEV and joint "
            "TEV mandatory"
        ),
    ),
    FluxV5MCapability(
        name="ptera_multi_airplane_multi_wing_or_image",
        status=OUT_OF_SCOPE_STATUS,
        evidence=(),
        restriction="generic Ptera topology, not a V5M production mode",
    ),
    FluxV5MCapability(
        name="warp_fsi_end_to_end",
        status=PRODUCTION_STATUS,
        evidence=(
            "test_flux_v5m_fsi_gpu_contract.py",
            "warp_fsi.validate:STRUCT_CG,COUPLING,TRAJ,NEWMARK_AM",
        ),
        restriction="strict facade; ml_fluid/ml_chain are CPU reference modules",
    ),
)


def capability_matrix() -> tuple[FluxV5MCapability, ...]:
    """Return the exact immutable V5M capability matrix."""

    return _CAPABILITIES


def capability_status() -> Mapping[str, str]:
    """Return a read-only name-to-status view for launch tooling."""

    return MappingProxyType({item.name: item.status for item in _CAPABILITIES})


def require_cuda_device(device: str | torch.device) -> torch.device:
    """Resolve and exercise a CUDA device without a CPU fallback."""

    if not torch.cuda.is_available():
        raise RuntimeError("FLUX-V5M requires CUDA; CPU fallback is forbidden")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("FLUX-V5M production device must be CUDA")
    torch.empty(1, device=resolved, dtype=torch.float64)
    return resolved


def require_cuda_float64_tensors(
    named_tensors: Mapping[str, torch.Tensor] | Sequence[tuple[str, torch.Tensor]],
    *,
    device: str | torch.device | None = None,
    finite: bool = True,
) -> torch.device:
    """Reject host, mixed-device, wrong-dtype and non-finite science tensors.

    This function never calls ``Tensor.to`` and therefore cannot hide a host
    numerical path by silently copying it to CUDA.
    """

    items = tuple(
        named_tensors.items() if isinstance(named_tensors, Mapping) else named_tensors
    )
    if not items:
        raise ValueError("at least one science tensor is required")
    expected = require_cuda_device(device or items[0][1].device)
    for name, value in items:
        if type(value) is not torch.Tensor:
            raise TypeError(f"{name} must be an exact torch.Tensor")
        if value.device.type != "cuda":
            raise ValueError(f"{name} must be CUDA; implicit host upload is forbidden")
        if value.device != expected:
            raise ValueError(f"{name} is on {value.device}, expected {expected}")
        if value.dtype is not torch.float64:
            raise TypeError(f"{name} must use torch.float64")
        if finite and not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains non-finite values")
    return expected


def make_flux_v5m_ptera_solver(
    unsteady_problem: Any,
    config: JointConfig | None = None,
    *,
    device: str = "cuda:0",
) -> CudaAttachedJointLEVTEVSolver:
    """Construct the currently authorised strict-CUDA Ptera solver.

    Attached, active LEV and joint LEV/TEV use the same strict CUDA chassis.
    Calling the CPU reference is never an alternative.
    """

    selected = config or JointConfig(enable_lev=True, joint_tev=True)
    if type(selected.enable_lev) is not bool or not selected.enable_lev:
        raise ValueError("FLUX-V5M production requires separated LEV")
    if type(selected.joint_tev) is not bool or not selected.joint_tev:
        raise ValueError("FLUX-V5M production requires joint TEV")
    require_cuda_device(device)
    return CudaAttachedJointLEVTEVSolver(
        unsteady_problem,
        selected,
        device=str(require_cuda_device(device)),
    )


def run_flux_v5m_ptera(
    unsteady_problem: Any,
    config: JointConfig | None = None,
    *,
    device: str = "cuda:0",
    prescribed_wake: bool = False,
    calculate_streamlines: bool = False,
    show_progress: bool = False,
) -> CudaAttachedJointLEVTEVSolver:
    """Run the mandatory separated-LEV, joint-TEV and free-wake mode."""

    if type(prescribed_wake) is not bool:
        raise TypeError("prescribed_wake must be an exact bool")
    if prescribed_wake:
        raise ValueError("FLUX-V5M production requires a free wake")

    solver = make_flux_v5m_ptera_solver(
        unsteady_problem,
        config,
        device=device,
    )
    solver.run(
        prescribed_wake=prescribed_wake,
        calculate_streamlines=calculate_streamlines,
        show_progress=show_progress,
    )
    return solver


__all__ = [
    "FAIL_CLOSED_STATUS",
    "GPU_NUMERICAL_CONTRACT",
    "OUT_OF_SCOPE_STATUS",
    "PRODUCTION_STATUS",
    "FluxV5MCapability",
    "capability_matrix",
    "capability_status",
    "make_flux_v5m_ptera_solver",
    "require_cuda_device",
    "require_cuda_float64_tensors",
    "run_flux_v5m_ptera",
]
