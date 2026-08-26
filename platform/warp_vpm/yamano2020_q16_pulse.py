"""CUDA-only prescribed pulse owner for the Yamano 2020 Q16 case."""

from __future__ import annotations

import math

import torch
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16MITC16EASMesh
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_mesh import Q16CudaMITC16EASMeshOperator
from fluxvortex.warp_fsi.q16_prescribed_endpoint_load import (
    Q16CudaPrescribedEndpointLoad,
)
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET,
    Yamano2020SingleSheetCase,
)


class Yamano2020Q16CudaPulse:
    """Consistent volume-to-generalized half-sine pulse on CUDA float64."""

    __slots__ = ("_unit_force", "case", "device", "dof_count")

    def __init__(
        self,
        model: Q16MITC16EASMesh,
        *,
        case: Yamano2020SingleSheetCase = YAMANO_2020_SINGLE_SHEET,
        device: str = "cuda:0",
    ) -> None:
        if type(model) is not Q16MITC16EASMesh:
            raise TypeError("model must be an exact Q16MITC16EASMesh")
        if type(case) is not Yamano2020SingleSheetCase:
            raise TypeError("case must be an exact Yamano2020SingleSheetCase")
        selected = torch.device(device)
        if selected.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Yamano pulse construction requires CUDA")
        if model.density != case.solid_density_kg_m3:
            raise ValueError("Q16 density differs from the frozen Yamano case")
        acceleration = torch.zeros(
            (1, model.mesh.dof_count), device=selected, dtype=torch.float64
        )
        # q_in_norm=1 corresponds to rho_f*U^2/h volume force.  A uniform
        # acceleration has zero director acceleration and is integrated by the
        # same consistent Q16 mass operator used by the structural solve.
        acceleration[:, 2::6] = (
            case.fluid_density_kg_m3
            * case.freestream_m_s**2
            / (case.thickness_m * case.solid_density_kg_m3)
        )
        operator = Q16CudaMITC16EASMeshOperator(model, device=device)
        unit_force = operator.mass_action(
            wp.from_torch(acceleration, dtype=config.DTYPE, requires_grad=False)
        )
        unit_force_t = wp.to_torch(unit_force)
        if (
            unit_force_t.device.type != "cuda"
            or unit_force_t.dtype is not torch.float64
            or not bool(torch.isfinite(unit_force_t).all().item())
        ):
            raise RuntimeError("Yamano pulse generalized force left CUDA float64")
        self._unit_force = wp.clone(unit_force)
        self.case = case
        self.device = selected
        self.dof_count = model.mesh.dof_count

    def endpoint_load(self, endpoint_time_s: float) -> Q16CudaPrescribedEndpointLoad:
        scale = self.case.pulse_nondimensional_at_time_s(endpoint_time_s)
        scaled = wp.to_torch(self._unit_force) * scale
        return Q16CudaPrescribedEndpointLoad.from_force(
            wp.from_torch(scaled, dtype=config.DTYPE, requires_grad=False),
            source_id="yamano2020-half-sine-volume-pulse",
            endpoint_time_s=float(endpoint_time_s),
        )

    def total_force_z_n(self, endpoint_time_s: float) -> float:
        load = self.endpoint_load(endpoint_time_s)
        force = wp.to_torch(load.generalized_force)
        total = torch.sum(force[:, 2::6])
        value = float(total.item())
        if not math.isfinite(value):
            raise FloatingPointError("Yamano pulse total force became non-finite")
        return value


__all__ = ["Yamano2020Q16CudaPulse"]
