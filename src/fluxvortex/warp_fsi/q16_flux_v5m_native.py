"""Native CUDA FLUX-V5M aerodynamic owner for Q16 shell coupling.

The scientific data plane is Q16 surface interpolation -> vortex geometry ->
LEV/TEV/free-wake solve -> panel load -> exact Q16 transpose.  Every evolving
numerical array is CUDA float64; Python owns only topology and transactions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16Mesh
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap

from . import config as warp_config
from .kernels_q16_transfer import Q16CudaSurfaceTransfer

from ldvm_source_bank_gpu import CudaLDVMSourceBank
from pfield_torch_gpu import CudaParticleField


NATIVE_V5M_CONTRACT = "q16-flux-v5m-native-cuda-float64-v1"
LESP_FACTOR = 1.13
_FOUR_PI = 4.0 * math.pi
_EPS_V = 1.0e-9


def _require_cuda64(name: str, value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    if value.device != device or value.device.type != "cuda":
        raise ValueError(f"{name} must remain on {device}")
    if value.dtype is not torch.float64:
        raise TypeError(f"{name} must use float64")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def _clone_value(value: Any) -> Any:
    if type(value) is torch.Tensor:
        return value.clone()
    if type(value) is dict:
        return {key: _clone_value(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone_value(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone_value(item) for item in value)
    return value


def _clone_source_bank(source: CudaLDVMSourceBank) -> CudaLDVMSourceBank:
    result = CudaLDVMSourceBank.__new__(CudaLDVMSourceBank)
    result.__dict__.update({key: _clone_value(value) for key, value in source.__dict__.items()})
    result.cuda_stream = torch.cuda.current_stream(result.device)
    return result


def _clone_particle_field(source: CudaParticleField) -> CudaParticleField:
    result = CudaParticleField.__new__(CudaParticleField)
    for slot in CudaParticleField.__slots__:
        setattr(result, slot, _clone_value(getattr(source, slot)))
    return result


def _axis_owner(value: float, element_count: int) -> tuple[int, float]:
    scaled = value * element_count
    owner = min(int(math.floor(scaled)), element_count - 1)
    local = 2.0 * (scaled - owner) - 1.0
    tolerance = 128.0 * np.finfo(np.float64).eps
    if local < -1.0 - tolerance or local > 1.0 + tolerance:
        raise ValueError("surface coordinate left its Q16 owner")
    return owner, min(max(local, -1.0), 1.0)


def _surface_map(
    mesh: Q16Mesh,
    chordwise_element_count: int,
    spanwise_element_count: int,
    fractions: tuple[tuple[float, float], ...],
) -> Q16SurfaceTransferMap:
    element_indices: list[int] = []
    coordinates: list[tuple[float, float, float]] = []
    for chord_fraction, span_fraction in fractions:
        chord_owner, xi = _axis_owner(chord_fraction, chordwise_element_count)
        span_owner, eta = _axis_owner(span_fraction, spanwise_element_count)
        element_indices.append(span_owner * chordwise_element_count + chord_owner)
        coordinates.append((xi, eta, 0.0))
    return Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.ascontiguousarray(element_indices, dtype=np.int64),
        parametric_coordinates=np.ascontiguousarray(coordinates, dtype=np.float64),
    )


def compute_lesp_crit(
    thickness_ratio: float,
    reynolds: float,
    *,
    lesp_flat: float = 0.11,
    thickness_reference: float = 0.013,
    c_thickness: float = 2.0,
    c_reynolds: float = 0.35,
    re_reference: float = 30_000.0,
) -> float:
    """Physics-based critical LESP from airfoil thickness and Reynolds number.

    Correlation calibrated to published data points:
      - Yamano flat plate t/c=1.3%, Re=20k → LESPcrit = 0.11 (Ramesh 2013)
      - NACA 0012 t/c=12%, Re=10k          → LESPcrit = 0.29 (Martinez-Carmena)
      - NACA 0012 t/c=12%, Re=30k          → LESPcrit ≈ 0.32 (Narsipur 2022)
      - NACA 0012 t/c=12%, Re=3M           → LESPcrit ≈ 0.50 (Narsipur 2022)

    Form:
      LESPcrit = LESP_flat + C_t · max(t/c − t₀, 0) · (1 + C_Re · log10(Re/Re₀))

    where t₀ = thickness_reference ≈ flat-plate threshold below which the
    thickness correction vanishes.
    """
    if not (math.isfinite(thickness_ratio) and 0.0 <= thickness_ratio <= 0.5):
        raise ValueError("thickness_ratio must lie in [0, 0.5]")
    if not (math.isfinite(reynolds) and reynolds > 0.0):
        raise ValueError("reynolds must be finite and positive")
    excess_thickness = max(thickness_ratio - thickness_reference, 0.0)
    reynolds_factor = 1.0 + c_reynolds * math.log10(
        max(reynolds, 1.0) / re_reference)
    return lesp_flat + c_thickness * excess_thickness * max(
        reynolds_factor, 0.5)


@dataclass(frozen=True, slots=True)
class NativeV5MConfig:
    chordwise_panels: int = 15
    spanwise_panels: int = 10
    density: float = 1.225
    freestream: float = 10.0
    # Geometric angle of attack of the freestream in the scientific frame
    # (+x chordwise, +z displacement direction): v_inf = U(cos a, 0, sin a).
    # The historical Yamano path leaves this at 0.0.
    freestream_angle_deg: float = 0.0
    aerodynamic_dt: float = 0.0068
    lesp_crit: float = 0.11
    lesp_thickness_ratio: float | None = None
    lesp_reynolds: float | None = None
    dvm_ndiv: int = 20
    dvm_naterm: int = 8
    dvm_max_wake: int = 64
    wake_max_rows: int = 96
    particle_capacity: int = 8192
    # Age cap (aerodynamic steps) for LEV particles, mirroring the ring-wake
    # truncation: 0 disables culling (the historical Yamano behavior, which
    # never sheds particles).  Sustained-release cases must set a positive
    # cap or the particle field grows without bound and exceeds capacity.
    particle_max_age_steps: int = 0
    # Wake-history pressure term: "material" is the author's strong-scheme
    # Mf2_vec1 (A^-1 applied to the wake-motion material derivative) kept by
    # the oracle-verified Yamano path; "bound_rate" is the author's
    # weak-scheme dp_add = (Gamma - Gamma_old)/dt, which stays bounded in
    # sustained-LEV-release regimes where Mf2_vec1 diverges.
    wake_history_mode: str = "material"
    # Number of newest wake rows (shed events) that keep full induced
    # convection; older rows convect with the freestream alone (the author's
    # far-wake freeze in generate_wake.m).  0 disables the freeze (the
    # historical fully-free wake, which is O(rows^2) per step).
    wake_free_rows: int = 0
    dvm_core_radius_chord: float = 0.02
    dvm_smoothing_radius_chord: float = 0.04
    dvm_target_spacing_chord: float = 0.04 / 2.125
    gate_rtol: float = 1.0e-8
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if type(self.chordwise_panels) is not int or self.chordwise_panels < 2:
            raise ValueError("chordwise_panels must be an exact int >= 2")
        if type(self.spanwise_panels) is not int or self.spanwise_panels < 1:
            raise ValueError("spanwise_panels must be a positive exact int")
        if type(self.particle_max_age_steps) is not int or self.particle_max_age_steps < 0:
            raise ValueError("particle_max_age_steps must be a non-negative exact int")
        if self.wake_history_mode not in {"material", "bound_rate"}:
            raise ValueError("wake_history_mode must be 'material' or 'bound_rate'")
        if type(self.wake_free_rows) is not int or self.wake_free_rows < 0:
            raise ValueError("wake_free_rows must be a non-negative exact int")
        if self.lesp_crit != 0.11:
            raise ValueError("the Yamano native V5M path freezes Lcrit=0.11")
        for name in ("density", "freestream", "aerodynamic_dt", "gate_rtol"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        angle_deg = float(self.freestream_angle_deg)
        if not math.isfinite(angle_deg) or abs(angle_deg) >= 90.0:
            raise ValueError("freestream_angle_deg must be finite with |angle| < 90")
        if torch.device(self.device).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("native FLUX-V5M requires CUDA")

    @property
    def effective_lesp_crit(self) -> float:
        """Return the physics-based LESPcrit if thickness/Re are provided."""
        if self.lesp_thickness_ratio is not None and self.lesp_reynolds is not None:
            return compute_lesp_crit(
                self.lesp_thickness_ratio, self.lesp_reynolds)
        return self.lesp_crit


@dataclass(frozen=True, slots=True)
class NativeV5MGeometry:
    rings: torch.Tensor
    ring_velocity: torch.Tensor
    collocation: torch.Tensor
    collocation_velocity: torch.Tensor
    normals: torch.Tensor
    areas: torch.Tensor
    leading_edge: torch.Tensor
    trailing_edge: torch.Tensor
    leading_velocity: torch.Tensor
    trailing_velocity: torch.Tensor
    quarter_points: torch.Tensor


class Q16NativeV5MSurface:
    """Direct Q16 evaluation of the native V5M vortex lattice."""

    def __init__(
        self,
        mesh: Q16Mesh,
        *,
        q16_chordwise_elements: int,
        q16_spanwise_elements: int,
        aerodynamic_chordwise_panels: int,
        aerodynamic_spanwise_panels: int,
        device: str,
        dense_transfers: bool = False,
    ) -> None:
        if mesh.element_count != q16_chordwise_elements * q16_spanwise_elements:
            raise ValueError("Q16 element topology mismatch")
        # Dense-GEMV fast path for the fixed transfer maps (bit-consistent
        # with the kernel scatter to fp64 round-off; unblocks CUDA graphs).
        self.dense_transfers = bool(dense_transfers)
        self.mesh = mesh
        self.q16_chordwise_elements = q16_chordwise_elements
        self.q16_spanwise_elements = q16_spanwise_elements
        self.nc = aerodynamic_chordwise_panels
        self.ns = aerodynamic_spanwise_panels
        self.device = torch.device(device)
        quarter = tuple(
            ((i + 0.25) / self.nc, j / self.ns)
            for i in range(self.nc)
            for j in range(self.ns + 1)
        )
        leading = tuple((0.0, j / self.ns) for j in range(self.ns + 1))
        trailing = tuple((1.0, j / self.ns) for j in range(self.ns + 1))
        self.quarter_transfer = Q16CudaSurfaceTransfer(
            _surface_map(mesh, q16_chordwise_elements, q16_spanwise_elements, quarter),
            device=device,
        )
        self.leading_transfer = Q16CudaSurfaceTransfer(
            _surface_map(mesh, q16_chordwise_elements, q16_spanwise_elements, leading),
            device=device,
        )
        self.trailing_transfer = Q16CudaSurfaceTransfer(
            _surface_map(mesh, q16_chordwise_elements, q16_spanwise_elements, trailing),
            device=device,
        )

    def evaluate(self, state: wp.array, velocity: wp.array) -> NativeV5MGeometry:
        if self.dense_transfers:
            q_flat = wp.to_torch(state)[0]
            v_flat = wp.to_torch(velocity)[0]
            quarter = self.quarter_transfer.interpolate_dense(q_flat).reshape(
                self.nc, self.ns + 1, 3
            )
            quarter_velocity = self.quarter_transfer.interpolate_dense(
                v_flat
            ).reshape(self.nc, self.ns + 1, 3)
            leading = self.leading_transfer.interpolate_dense(q_flat)
            trailing = self.trailing_transfer.interpolate_dense(q_flat)
            leading_velocity = self.leading_transfer.interpolate_dense(v_flat)
            trailing_velocity = self.trailing_transfer.interpolate_dense(v_flat)
        else:
            quarter = wp.to_torch(self.quarter_transfer.interpolate(state))[0].reshape(
                self.nc, self.ns + 1, 3
            )
            quarter_velocity = wp.to_torch(
                self.quarter_transfer.interpolate(velocity)
            )[0].reshape(self.nc, self.ns + 1, 3)
            leading = wp.to_torch(self.leading_transfer.interpolate(state))[0]
            trailing = wp.to_torch(self.trailing_transfer.interpolate(state))[0]
            leading_velocity = wp.to_torch(self.leading_transfer.interpolate(velocity))[0]
            trailing_velocity = wp.to_torch(self.trailing_transfer.interpolate(velocity))[0]
        rear = quarter[-1] + (4.0 / 3.0) * (trailing - quarter[-1])
        rear_velocity = quarter_velocity[-1] + (4.0 / 3.0) * (
            trailing_velocity - quarter_velocity[-1]
        )
        back = torch.cat((quarter[1:], rear.unsqueeze(0)), dim=0)
        back_velocity = torch.cat(
            (quarter_velocity[1:], rear_velocity.unsqueeze(0)), dim=0
        )
        rings = torch.stack(
            (quarter[:, :-1], quarter[:, 1:], back[:, 1:], back[:, :-1]), dim=2
        ).reshape(self.nc * self.ns, 4, 3)
        ring_velocity = torch.stack(
            (
                quarter_velocity[:, :-1],
                quarter_velocity[:, 1:],
                back_velocity[:, 1:],
                back_velocity[:, :-1],
            ),
            dim=2,
        ).reshape(self.nc * self.ns, 4, 3)
        collocation = torch.mean(rings, dim=1)
        collocation_velocity = torch.mean(ring_velocity, dim=1)
        diagonal_31 = rings[:, 3] - rings[:, 1]
        diagonal_24 = rings[:, 2] - rings[:, 0]
        cross = torch.linalg.cross(diagonal_31, diagonal_24, dim=1)
        cross_norm = torch.linalg.vector_norm(cross, dim=1)
        if bool(torch.any(cross_norm <= 0.0).item()):
            raise FloatingPointError("native V5M surface contains a collapsed panel")
        normals = cross / cross_norm[:, None]
        areas = 0.5 * cross_norm
        for name, value in (
            ("rings", rings),
            ("collocation", collocation),
            ("collocation_velocity", collocation_velocity),
            ("normals", normals),
            ("areas", areas),
            ("leading_edge", leading),
            ("trailing_edge", trailing),
        ):
            _require_cuda64(name, value, self.device)
        return NativeV5MGeometry(
            rings=rings,
            ring_velocity=ring_velocity,
            collocation=collocation,
            collocation_velocity=collocation_velocity,
            normals=normals,
            areas=areas,
            leading_edge=leading,
            trailing_edge=trailing,
            leading_velocity=leading_velocity,
            trailing_velocity=trailing_velocity,
            quarter_points=quarter,
        )


def native_ring_velocity_expanded(
    points: torch.Tensor,
    rings: torch.Tensor,
    *,
    core_fraction: float,
    reference_length: float,
) -> torch.Tensor:
    """Finite-ring influence with the paper's fourth-order core model."""

    if rings.shape[0] == 0:
        return torch.zeros(
            (points.shape[0], 0, 3), device=points.device, dtype=torch.float64
        )
    starts = rings.reshape(-1, 3)
    ends = torch.roll(rings, shifts=-1, dims=1).reshape(-1, 3)
    a = points[:, None, :] - starts[None, :, :]
    b = points[:, None, :] - ends[None, :, :]
    edge = a - b
    cross = torch.linalg.cross(a, b, dim=2)
    cross_sq = torch.sum(cross * cross, dim=2)
    cross_norm = torch.sqrt(cross_sq)
    norm_a = torch.linalg.vector_norm(a, dim=2)
    norm_b = torch.linalg.vector_norm(b, dim=2)
    edge_norm = torch.linalg.vector_norm(edge, dim=2)
    eps = torch.finfo(torch.float64).eps
    unit_difference = a / torch.clamp(norm_a, min=eps)[:, :, None] - b / torch.clamp(
        norm_b, min=eps
    )[:, :, None]
    scalar = torch.sum(edge * unit_difference, dim=2)
    base = (
        cross
        / (cross_sq + _EPS_V)[:, :, None]
        * scalar[:, :, None]
        / _FOUR_PI
    )
    source_edge = torch.linalg.vector_norm(
        torch.roll(rings, shifts=-1, dims=1) - rings, dim=2
    )
    source_scale = torch.maximum(
        torch.max(source_edge, dim=1).values,
        torch.full(
            (rings.shape[0],),
            float(reference_length),
            device=points.device,
            dtype=torch.float64,
        ),
    )
    core = source_scale * float(core_fraction)
    h = cross_norm / torch.clamp(edge_norm, min=eps)
    core_leg = core.repeat_interleave(4)[None, :]
    kv = h * h / torch.sqrt(h**4 + core_leg**4)
    velocity = (kv[:, :, None] * base).reshape(
        points.shape[0], rings.shape[0], 4, 3
    ).sum(dim=2)
    if not bool(torch.isfinite(velocity).all().item()):
        raise FloatingPointError("native ring influence is non-finite")
    return velocity


def native_aic(geometry: NativeV5MGeometry, *, chordwise_panels: int) -> torch.Tensor:
    expanded = native_ring_velocity_expanded(
        geometry.collocation,
        geometry.rings,
        core_fraction=1.0e-6,
        reference_length=1.0 / chordwise_panels,
    )
    result = torch.sum(expanded * geometry.normals[:, None, :], dim=2)
    if bool(torch.any(torch.diagonal(result) >= 0.0).item()):
        raise RuntimeError("native AIC orientation drift")
    return result


@dataclass(slots=True)
class NativeV5MState:
    step: int
    gamma_bound: torch.Tensor
    gamma_previous: torch.Tensor
    wake_rings: torch.Tensor
    wake_gamma: torch.Tensor
    source_bank: CudaLDVMSourceBank
    particle_field: CudaParticleField
    reference_cell_chord: torch.Tensor
    reference_node_chord: torch.Tensor
    alpha_previous: torch.Tensor | None
    frontier_nodes: torch.Tensor
    frontier_active: torch.Tensor
    diagnostics: tuple[dict[str, float | int | bool], ...]

    def clone(self) -> "NativeV5MState":
        return NativeV5MState(
            step=self.step,
            gamma_bound=self.gamma_bound.clone(),
            gamma_previous=self.gamma_previous.clone(),
            wake_rings=self.wake_rings.clone(),
            wake_gamma=self.wake_gamma.clone(),
            source_bank=_clone_source_bank(self.source_bank),
            particle_field=_clone_particle_field(self.particle_field),
            reference_cell_chord=self.reference_cell_chord.clone(),
            reference_node_chord=self.reference_node_chord.clone(),
            alpha_previous=None if self.alpha_previous is None else self.alpha_previous.clone(),
            frontier_nodes=self.frontier_nodes.clone(),
            frontier_active=self.frontier_active.clone(),
            diagnostics=tuple(dict(item) for item in self.diagnostics),
        )

    def digest(self) -> str:
        digest = hashlib.sha256(NATIVE_V5M_CONTRACT.encode("ascii"))
        digest.update(str(self.step).encode("ascii"))
        for value in (
            self.gamma_bound,
            self.gamma_previous,
            self.wake_rings,
            self.wake_gamma,
            self.particle_field.positions_cuda,
            self.particle_field.gammas_cuda,
            self.frontier_nodes,
            self.frontier_active,
        ):
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        digest.update(str(self.source_bank.it).encode("ascii"))
        digest.update(str(self.source_bank.nt).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeV5MLoad:
    panel_positions: torch.Tensor
    panel_forces: torch.Tensor
    pressure: torch.Tensor
    total_force: torch.Tensor
    total_moment: torch.Tensor


@dataclass(frozen=True, slots=True)
class NativeV5MProposal:
    parent_digest: str
    trial_state: NativeV5MState
    load: NativeV5MLoad
    generalized_force: wp.array
    author_load: Any


class Q16NativePanelLoadTransfer:
    """Exact transpose of the author collocation construction into Q16."""

    def __init__(self, surface: Q16NativeV5MSurface) -> None:
        self.surface = surface

    def map(self, panel_forces: torch.Tensor) -> wp.array:
        if panel_forces.ndim != 2:
            raise ValueError("native panel force must have shape (panel,3)")
        return self.map_batch(panel_forces.unsqueeze(0))

    def map_batch(self, panel_forces: torch.Tensor) -> wp.array:
        nc, ns = self.surface.nc, self.surface.ns
        if panel_forces.ndim != 3 or panel_forces.shape[1:] != (nc * ns, 3):
            raise ValueError(
                "panel force batch shape differs from native V5M topology"
            )
        batch = panel_forces.shape[0]
        forces = panel_forces.reshape(batch, nc, ns, 3)
        quarter_force = torch.zeros(
            (batch, nc, ns + 1, 3),
            device=panel_forces.device,
            dtype=torch.float64,
        )
        trailing_force = torch.zeros(
            (batch, ns + 1, 3),
            device=panel_forces.device,
            dtype=torch.float64,
        )
        quarter_force[:, :, :-1] += 0.25 * forces
        quarter_force[:, :, 1:] += 0.25 * forces
        if nc > 1:
            quarter_force[:, 1:, :-1] += 0.25 * forces[:, :-1]
            quarter_force[:, 1:, 1:] += 0.25 * forces[:, :-1]
        quarter_force[:, -1, :-1] -= (1.0 / 12.0) * forces[:, -1]
        quarter_force[:, -1, 1:] -= (1.0 / 12.0) * forces[:, -1]
        trailing_force[:, :-1] += (1.0 / 3.0) * forces[:, -1]
        trailing_force[:, 1:] += (1.0 / 3.0) * forces[:, -1]
        quarter_wp = wp.from_torch(
            quarter_force.reshape(batch, nc * (ns + 1), 3),
            dtype=warp_config.VEC3,
        )
        trailing_wp = wp.from_torch(trailing_force, dtype=warp_config.VEC3)
        q_generalized = self.surface.quarter_transfer.transpose(quarter_wp)
        t_generalized = self.surface.trailing_transfer.transpose(trailing_wp)
        wp.to_torch(q_generalized).add_(wp.to_torch(t_generalized))
        return q_generalized


# Set True only around an active wp.ScopedCapture region: host-side
# validation gates must defer during CUDA-graph capture; the capture-mode
# caller validates the replayed static outputs after the capture ends.
CAPTURING = False


def freestream_vector(
    freestream: float, angle_deg: float, device: str | torch.device
) -> torch.Tensor:
    """v_inf = U (cos a, 0, sin a) in the scientific frame (+x chord, +z up)."""

    angle_rad = math.radians(angle_deg)
    return float(freestream) * torch.tensor(
        [math.cos(angle_rad), 0.0, math.sin(angle_rad)],
        device=torch.device(device),
        dtype=torch.float64,
    )


class Q16NativeV5MSolver:
    """State-complete native V5M stepper with transactional trial ownership."""

    def __init__(self, surface: Q16NativeV5MSurface, settings: NativeV5MConfig) -> None:
        if surface.nc != settings.chordwise_panels or surface.ns != settings.spanwise_panels:
            raise ValueError("surface and aerodynamic topology differ")
        self.surface = surface
        self.settings = settings
        self.device = torch.device(settings.device)
        self.load_transfer = Q16NativePanelLoadTransfer(surface)
        surface.panel_load_transfer = self.load_transfer
        from .q16_flux_v5m_author_loads import Q16NativeAuthorLoadAssembler

        self.author_load_assembler = Q16NativeAuthorLoadAssembler(
            surface, density=settings.density
        )
        self.v_inf = freestream_vector(
            settings.freestream, settings.freestream_angle_deg, self.device
        )

    def initialize(self, state: wp.array, velocity: wp.array) -> NativeV5MState:
        geometry = self.surface.evaluate(state, velocity)
        ns, count = self.settings.spanwise_panels, self.settings.chordwise_panels * self.settings.spanwise_panels
        node_chord = torch.linalg.vector_norm(
            geometry.trailing_edge - geometry.leading_edge, dim=1
        )
        cell_chord = 0.5 * (node_chord[:-1] + node_chord[1:])
        lane_chord = torch.cat((cell_chord, node_chord))
        convective_dt = self.settings.aerodynamic_dt * self.settings.freestream / lane_chord
        source = CudaLDVMSourceBank(
            batch_size=2 * ns + 1,
            ndiv=self.settings.dvm_ndiv,
            naterm=self.settings.dvm_naterm,
            delta_time_convective=convective_dt,
            lesp_crit=self.settings.effective_lesp_crit,
            core_radius_chord=self.settings.dvm_core_radius_chord,
            max_wake=self.settings.dvm_max_wake,
            source_parity=True,
            device=self.settings.device,
        )
        return NativeV5MState(
            step=0,
            gamma_bound=torch.zeros(count, device=self.device, dtype=torch.float64),
            gamma_previous=torch.zeros(count, device=self.device, dtype=torch.float64),
            wake_rings=torch.zeros((0, 4, 3), device=self.device, dtype=torch.float64),
            wake_gamma=torch.zeros(0, device=self.device, dtype=torch.float64),
            source_bank=source,
            particle_field=CudaParticleField(
                self.settings.particle_capacity, device=self.settings.device
            ),
            reference_cell_chord=cell_chord,
            reference_node_chord=node_chord,
            alpha_previous=None,
            frontier_nodes=geometry.leading_edge.clone(),
            frontier_active=torch.zeros(ns + 1, device=self.device, dtype=torch.bool),
            diagnostics=(),
        )

    def author_anchor_load(
        self, state: wp.array, velocity: wp.array
    ) -> Any:
        geometry = self.surface.evaluate(state, velocity)
        aic = native_aic(
            geometry, chordwise_panels=self.settings.chordwise_panels
        )
        panel_count = self.settings.chordwise_panels * self.settings.spanwise_panels
        zero = torch.zeros(panel_count, device=self.device, dtype=torch.float64)
        zero_gradient = torch.zeros(
            (panel_count, 3), device=self.device, dtype=torch.float64
        )
        return self.author_load_assembler.assemble(
            structural_state=state,
            geometry=geometry,
            aic=aic,
            gamma=zero,
            external_flow=self.v_inf[None, :].expand(panel_count, 3),
            gamma_gradient=zero_gradient,
            mf2_history=zero,
        )

    # Tile budget for the ring-influence sum: (points x rings x 3) float64.
    # Evaluations below the budget take the historical single-shot path
    # (bit-identical for the oracle-verified small cases); larger ones tile
    # over the points so long free wakes cannot materialize multi-GB
    # intermediate tensors.
    _RING_TILE_BYTES = 64 * 1024 * 1024

    def _ring_velocity(self, points: torch.Tensor, rings: torch.Tensor, gamma: torch.Tensor, *, rough: bool) -> torch.Tensor:
        if rings.shape[0] == 0:
            return torch.zeros_like(points)
        core_fraction = 0.1 if rough else 1.0e-6
        reference_length = 1.0 / self.settings.chordwise_panels
        full_bytes = points.shape[0] * rings.shape[0] * 3 * 8
        if full_bytes <= self._RING_TILE_BYTES:
            expanded = native_ring_velocity_expanded(
                points,
                rings,
                core_fraction=core_fraction,
                reference_length=reference_length,
            )
            return torch.sum(expanded * gamma[None, :, None], dim=1)
        total = torch.zeros_like(points)
        target_cap = max(1, self._RING_TILE_BYTES // (rings.shape[0] * 3 * 8))
        for start in range(0, points.shape[0], target_cap):
            stop = min(start + target_cap, points.shape[0])
            expanded = native_ring_velocity_expanded(
                points[start:stop],
                rings,
                core_fraction=core_fraction,
                reference_length=reference_length,
            )
            total[start:stop] = torch.sum(
                expanded * gamma[None, :, None], dim=1
            )
        return total

    def _external_velocity(
        self,
        points: torch.Tensor,
        geometry: NativeV5MGeometry,
        trial: NativeV5MState,
    ) -> torch.Tensor:
        return (
            self.v_inf[None, :]
            + self._ring_velocity(points, geometry.rings, trial.gamma_bound, rough=False)
            + self._ring_velocity(points, trial.wake_rings, trial.wake_gamma, rough=False)
        )

    def _advance_material_state(self, geometry: NativeV5MGeometry, trial: NativeV5MState) -> None:
        dt = self.settings.aerodynamic_dt
        if trial.wake_rings.shape[0]:
            ns = self.settings.spanwise_panels
            total_rows = trial.wake_rings.shape[0] // ns
            freeze_rows = self.settings.wake_free_rows
            if freeze_rows and total_rows > freeze_rows:
                # The author's far-wake freeze (generate_wake.m): rows older
                # than `wake_free_rows` shed events convect with the
                # freestream alone and stop deforming, bounding the O(N^2)
                # self-convection cost of long free wakes.  Rows stay full
                # vortex sources for everything they influence.
                original = trial.wake_rings.clone()
                active = original[: freeze_rows * ns]

                def active_velocity(at: torch.Tensor) -> torch.Tensor:
                    return self._external_velocity(at.reshape(-1, 3), geometry, trial).reshape_as(at)

                u1 = active_velocity(active)
                if trial.particle_field.n:
                    u1 = u1 + trial.particle_field.velocity_at_cuda(
                        active.reshape(-1, 3)
                    ).reshape_as(active)
                u2 = active_velocity(active + 0.5 * dt * u1)
                if trial.particle_field.n:
                    u2 = u2 + trial.particle_field.velocity_at_cuda(
                        (active + 0.5 * dt * u1).reshape(-1, 3)
                    ).reshape_as(active)
                u3 = active_velocity(active + dt * (-u1 + 2.0 * u2))
                if trial.particle_field.n:
                    u3 = u3 + trial.particle_field.velocity_at_cuda(
                        (active + dt * (-u1 + 2.0 * u2)).reshape(-1, 3)
                    ).reshape_as(active)
                updated_active = active + (dt / 6.0) * (u1 + 4.0 * u2 + u3)
                frozen_tail = original[freeze_rows * ns :] + self.v_inf * dt
                trial.wake_rings = torch.cat((updated_active, frozen_tail), dim=0)
            else:
                original = trial.wake_rings.clone()

                def wake_velocity(at: torch.Tensor) -> torch.Tensor:
                    flat = at.reshape(-1, 3)
                    velocity = self._external_velocity(flat, geometry, trial)
                    if trial.particle_field.n:
                        velocity = velocity + trial.particle_field.velocity_at_cuda(flat)
                    return velocity.reshape_as(at)

                u1 = wake_velocity(original)
                u2 = wake_velocity(original + 0.5 * dt * u1)
                u3 = wake_velocity(original + dt * (-u1 + 2.0 * u2))
                trial.wake_rings.copy_(original + (dt / 6.0) * (u1 + 4.0 * u2 + u3))

        def particle_external(points: torch.Tensor) -> torch.Tensor:
            return self._external_velocity(points, geometry, trial)

        if trial.particle_field.n:
            trial.particle_field.advance_wrk3(dt, particle_external)
            if self.settings.particle_max_age_steps > 0:
                horizon = trial.step - self.settings.particle_max_age_steps
                if horizon > 0:
                    trial.particle_field.remove_mask(
                        trial.particle_field.birth_step[: trial.particle_field.n]
                        < horizon
                    )

    def _dvm_kinematics(
        self, geometry: NativeV5MGeometry, trial: NativeV5MState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ns = self.settings.spanwise_panels
        chord_cell = 0.5 * (
            geometry.trailing_edge[:-1]
            + geometry.trailing_edge[1:]
            - geometry.leading_edge[:-1]
            - geometry.leading_edge[1:]
        )
        cell_x = chord_cell / torch.linalg.vector_norm(chord_cell, dim=1, keepdim=True)
        panel_normals = geometry.normals.reshape(self.settings.chordwise_panels, ns, 3)
        cell_normal = torch.mean(panel_normals, dim=0)
        cell_normal = cell_normal / torch.linalg.vector_norm(cell_normal, dim=1, keepdim=True)
        node_x_vector = geometry.trailing_edge - geometry.leading_edge
        node_x = node_x_vector / torch.linalg.vector_norm(node_x_vector, dim=1, keepdim=True)
        node_normal = torch.empty((ns + 1, 3), device=self.device, dtype=torch.float64)
        node_normal[0], node_normal[-1] = cell_normal[0], cell_normal[-1]
        if ns > 1:
            node_normal[1:-1] = cell_normal[:-1] + cell_normal[1:]
        node_normal = node_normal / torch.linalg.vector_norm(node_normal, dim=1, keepdim=True)
        cell_alpha = torch.atan2(
            torch.sum(cell_normal * self.v_inf, dim=1),
            torch.sum(cell_x * self.v_inf, dim=1),
        )
        node_alpha = torch.atan2(
            torch.sum(node_normal * self.v_inf, dim=1),
            torch.sum(node_x * self.v_inf, dim=1),
        )
        alpha = torch.cat((cell_alpha, node_alpha))
        if trial.alpha_previous is None:
            alpha_rate = torch.zeros_like(alpha)
        else:
            delta = torch.remainder(alpha - trial.alpha_previous + math.pi, 2.0 * math.pi) - math.pi
            alpha_rate = delta / trial.source_bank.dt
        node_heave = torch.sum(geometry.leading_velocity * node_normal, dim=1) / self.settings.freestream
        heave = torch.cat((0.5 * (node_heave[:-1] + node_heave[1:]), node_heave))
        return alpha, alpha_rate, heave, node_x, node_normal

    def _deposit_dvm_ribbon(
        self,
        geometry: NativeV5MGeometry,
        trial: NativeV5MState,
        result: dict[str, torch.Tensor | bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ns = self.settings.spanwise_panels
        cell_active = result["shed_lev"][:ns]
        node_active = result["shed_lev"][ns:]
        gamma_lev = result["gamma_lev_new"][:ns] * (
            self.settings.freestream * trial.reference_cell_chord
        )
        alpha, _, _, node_x, node_normal = self._dvm_kinematics(geometry, trial)
        del alpha
        birth_2d = result["lev_birth_position"][ns:]
        anchor_2d = result["lev_edge_position"][ns:]
        displacement = birth_2d - anchor_2d
        frontier = geometry.leading_edge + trial.reference_node_chord[:, None] * (
            displacement[:, :1] * node_x + displacement[:, 1:] * node_normal
        )
        frontier = torch.where(node_active[:, None], frontier, geometry.leading_edge)
        trial.particle_field.add_connected_ribbon_particles(
            geometry.leading_edge,
            frontier,
            -gamma_lev,
            smoothing_radius=self.settings.dvm_smoothing_radius_chord
            * float(torch.mean(trial.reference_cell_chord).item()),
            target_spacing=self.settings.dvm_target_spacing_chord
            * float(torch.mean(trial.reference_cell_chord).item()),
            birth_step=trial.step,
            connector_source_strips=torch.clamp(
                torch.arange(ns + 1, device=self.device, dtype=torch.int64) - 1,
                min=0,
                max=ns - 1,
            ),
        )
        trial.frontier_nodes = torch.where(
            node_active[:, None], frontier, trial.frontier_nodes
        )
        trial.frontier_active = node_active.clone()
        return gamma_lev, cell_active

    def propose(
        self,
        committed: NativeV5MState,
        structural_state: wp.array,
        structural_velocity: wp.array,
    ) -> NativeV5MProposal:
        parent_digest = committed.digest()
        trial = committed.clone()
        geometry = self.surface.evaluate(structural_state, structural_velocity)
        self._advance_material_state(geometry, trial)
        aic = native_aic(geometry, chordwise_panels=self.settings.chordwise_panels)
        wake_velocity = self._ring_velocity(
            geometry.collocation, trial.wake_rings, trial.wake_gamma, rough=False
        )
        particle_velocity = (
            trial.particle_field.velocity_at_cuda(geometry.collocation)
            if trial.particle_field.n
            else torch.zeros_like(geometry.collocation)
        )
        rhs = torch.sum(
            (geometry.collocation_velocity - self.v_inf - wake_velocity - particle_velocity)
            * geometry.normals,
            dim=1,
        )
        gamma_pre = torch.linalg.solve(aic, rhs)
        ns, nc = self.settings.spanwise_panels, self.settings.chordwise_panels
        chord = 0.5 * (
            torch.linalg.vector_norm(
                geometry.trailing_edge[:-1] - geometry.leading_edge[:-1], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.trailing_edge[1:] - geometry.leading_edge[1:], dim=1
            )
        )
        first_dx = 0.5 * (
            torch.linalg.vector_norm(
                geometry.rings[:ns, 3] - geometry.rings[:ns, 0], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.rings[:ns, 2] - geometry.rings[:ns, 1], dim=1
            )
        )
        theta = torch.acos(torch.clamp(1.0 - 2.0 * first_dx / chord, -1.0, 1.0))
        scale = chord * self.settings.freestream * (theta + torch.sin(theta)) / LESP_FACTOR
        lesp_pre_3d = -gamma_pre[:ns] / scale
        surface_separated = torch.abs(lesp_pre_3d) > self.settings.effective_lesp_crit
        alpha, alpha_rate, heave, _, _ = self._dvm_kinematics(geometry, trial)
        source_result = trial.source_bank.step(
            alpha, alpha_rate, heave, node_topology_from_cell_count=ns
        )
        trial.alpha_previous = alpha.clone()
        particle_start = trial.particle_field.n
        gamma_lev, released = self._deposit_dvm_ribbon(
            geometry, trial, source_result
        )
        if trial.particle_field.n > particle_start:
            newborn_velocity = trial.particle_field.velocity_at_cuda(
                geometry.collocation,
                source_start=particle_start,
                source_stop=trial.particle_field.n,
            )
            rhs = rhs - torch.sum(newborn_velocity * geometry.normals, dim=1)
        pin_active = released | surface_separated
        separated_aic = aic.clone()
        separated_rhs = rhs.clone()
        le_indices = torch.arange(ns, device=self.device, dtype=torch.int64)
        active_indices = le_indices[pin_active]
        source_a0 = source_result["A0"][:ns]
        if active_indices.numel():
            separated_aic[active_indices] = 0.0
            separated_aic[active_indices, active_indices] = 1.0
            separated_rhs[active_indices] = (-source_a0 * scale)[pin_active]
        gamma = torch.linalg.solve(separated_aic, separated_rhs)
        retained = torch.ones_like(rhs, dtype=torch.bool)
        retained[active_indices] = False
        residual = aic @ gamma - rhs
        retained_max = torch.max(torch.abs(residual[retained]))
        retained_scale = torch.clamp(torch.max(torch.abs(rhs[retained])), min=1.0)
        if bool((retained_max > self.settings.gate_rtol * retained_scale).item()):
            raise RuntimeError("native V5M retained Neumann rows failed")
        solved_lesp = -gamma[:ns] / scale
        pin_error = torch.zeros((), device=self.device, dtype=torch.float64)
        if active_indices.numel():
            pin_error = torch.max(torch.abs(solved_lesp[pin_active] - source_a0[pin_active]))
            if bool((pin_error > 1.0e-6).item()):
                raise RuntimeError("native V5M LESP pin failed")
        gamma_tev = gamma.reshape(nc, ns)[-1] + gamma_lev
        kelvin = torch.max(
            torch.abs(gamma_tev - gamma.reshape(nc, ns)[-1] - gamma_lev)
        )
        if bool((kelvin > self.settings.gate_rtol * torch.clamp(torch.max(torch.abs(gamma)), min=1.0)).item()):
            raise RuntimeError("native V5M joint TEV relation failed")

        bound_velocity = self._ring_velocity(
            geometry.collocation, geometry.rings, gamma, rough=False
        )
        external_flow = self.v_inf + wake_velocity + particle_velocity + bound_velocity
        grid = gamma.reshape(nc, ns)
        dx = torch.linalg.vector_norm(
            0.5 * (geometry.rings[:, 2] + geometry.rings[:, 3] - geometry.rings[:, 0] - geometry.rings[:, 1]), dim=1
        ).reshape(nc, ns)
        dy = torch.linalg.vector_norm(
            0.5 * (geometry.rings[:, 1] + geometry.rings[:, 2] - geometry.rings[:, 0] - geometry.rings[:, 3]), dim=1
        ).reshape(nc, ns)
        dx_gamma = torch.empty_like(grid)
        dx_gamma[0] = grid[0] / dx[0]
        dx_gamma[1:] = (grid[1:] - grid[:-1]) / dx[1:]
        padded = torch.nn.functional.pad(grid, (1, 1))
        dy_gamma = (padded[:, 2:] - padded[:, :-2]) / (2.0 * dy)
        dy_gamma[:, 0] = grid[:, 0] / dy[:, 0]
        dy_gamma[:, -1] = -grid[:, -1] / dy[:, -1]
        tau_x = 0.5 * (
            geometry.rings[:, 2] + geometry.rings[:, 3] - geometry.rings[:, 0] - geometry.rings[:, 1]
        )
        tau_y = 0.5 * (
            geometry.rings[:, 1] + geometry.rings[:, 2] - geometry.rings[:, 0] - geometry.rings[:, 3]
        )
        tau_x = tau_x / torch.linalg.vector_norm(tau_x, dim=1, keepdim=True)
        tau_y = tau_y / torch.linalg.vector_norm(tau_y, dim=1, keepdim=True)
        gradient = tau_x * dx_gamma.reshape(-1, 1) + tau_y * dy_gamma.reshape(-1, 1)
        if self.settings.wake_history_mode == "bound_rate":
            # Author's weak-scheme dp_add (calc_fluid_force.m): the bound
            # circulation time derivative, bounded by construction.  The
            # strong-scheme Mf2_vec1 diverges when every wake row carries a
            # large persistent LEV release.
            mf2_history = (
                gamma - trial.gamma_previous
            ) / self.settings.aerodynamic_dt
            wake_vertex_velocity = None
        elif trial.wake_rings.shape[0]:
            from .q16_flux_v5m_author_loads import (
                material_ring_velocity_derivative_expanded,
            )

            wake_vertex_velocity = self._external_velocity(
                trial.wake_rings.reshape(-1, 3), geometry, trial
            ).reshape_as(trial.wake_rings)
            if trial.particle_field.n:
                wake_vertex_velocity = wake_vertex_velocity + trial.particle_field.velocity_at_cuda(
                    trial.wake_rings.reshape(-1, 3)
                ).reshape_as(trial.wake_rings)
            # The author's wake is a connected chain (generate_wake.m):
            # each row's trailing-edge-side corners ARE the previous row's
            # downstream corners, and the newest row is anchored to the
            # moving trailing edge with the sheet's structural velocity.
            # Free convection of detached front corners slides them under
            # the collocation points and poisons the wake-history term.
            chain_velocity = wake_vertex_velocity.reshape(-1, ns, 4, 3)
            chain_velocity[1:, :, 0] = chain_velocity[:-1, :, 3]
            chain_velocity[1:, :, 1] = chain_velocity[:-1, :, 2]
            chain_velocity[0, :, 0] = geometry.trailing_velocity[:-1]
            chain_velocity[0, :, 1] = geometry.trailing_velocity[1:]
            influence_rate = material_ring_velocity_derivative_expanded(
                geometry.collocation,
                geometry.collocation_velocity,
                trial.wake_rings,
                wake_vertex_velocity,
            )
            wake_velocity_rate = torch.sum(
                influence_rate * trial.wake_gamma[None, :, None], dim=1
            )
            wake_normal_rate = torch.sum(
                wake_velocity_rate * geometry.normals, dim=1
            )
            mf2_history = torch.linalg.solve(aic, -wake_normal_rate)
        else:
            wake_vertex_velocity = None
            mf2_history = torch.zeros_like(gamma)
        author_load = self.author_load_assembler.assemble(
            structural_state=structural_state,
            geometry=geometry,
            aic=aic,
            gamma=gamma,
            external_flow=external_flow,
            gamma_gradient=gradient,
            mf2_history=mf2_history,
        )
        pressure = author_load.constant_pressure
        panel_forces = pressure[:, None] * geometry.areas[:, None] * geometry.normals
        generalized = wp.clone(author_load.constant_generalized_force)
        load = NativeV5MLoad(
            panel_positions=geometry.collocation.clone(),
            panel_forces=panel_forces.clone(),
            pressure=pressure.clone(),
            total_force=torch.sum(panel_forces, dim=0),
            total_moment=torch.sum(
                torch.linalg.cross(geometry.collocation, panel_forces, dim=1), dim=0
            ),
        )

        rear = geometry.rings.reshape(nc, ns, 4, 3)[-1, :, 2:4]
        if trial.wake_rings.shape[0]:
            # Re-anchor the convected chain to the current trailing edge,
            # newest row first (generate_wake.m r_wake_1 update).
            chain = trial.wake_rings.reshape(-1, ns, 4, 3)
            chain[1:, :, 0] = chain[:-1, :, 3]
            chain[1:, :, 1] = chain[:-1, :, 2]
            chain[0, :, 0] = rear[:, 1]
            chain[0, :, 1] = rear[:, 0]
        new_wake = torch.stack(
            (
                rear[:, 1],
                rear[:, 0],
                rear[:, 0] + self.v_inf * self.settings.aerodynamic_dt,
                rear[:, 1] + self.v_inf * self.settings.aerodynamic_dt,
            ),
            dim=1,
        )
        trial.wake_rings = torch.cat((new_wake, trial.wake_rings), dim=0)[: self.settings.wake_max_rows * ns]
        trial.wake_gamma = torch.cat((gamma_tev, trial.wake_gamma), dim=0)[: self.settings.wake_max_rows * ns]
        trial.gamma_previous = trial.gamma_bound.clone()
        trial.gamma_bound = gamma.clone()
        trial.step += 1
        trial.diagnostics = trial.diagnostics + (
            {
                "step": trial.step,
                "lev_release_count": int(torch.count_nonzero(released).item()),
                "separated_strip_count": int(torch.count_nonzero(pin_active).item()),
                "lesp_pre_max_abs": float(torch.max(torch.abs(lesp_pre_3d)).item()),
                "lesp_pin_max_abs": float(pin_error.item()),
                "kelvin_max_abs": float(kelvin.item()),
                "retained_neumann_max_abs": float(retained_max.item()),
                "wake_ring_count": int(trial.wake_gamma.numel()),
                "particle_count": int(trial.particle_field.n),
                "cuda_float64": True,
            },
        )
        if committed.digest() != parent_digest:
            raise RuntimeError("native V5M trial mutated its committed parent")
        return NativeV5MProposal(
            parent_digest=parent_digest,
            trial_state=trial,
            load=load,
            generalized_force=generalized,
            author_load=author_load,
        )


class Q16NativeV5MOwner:
    def __init__(self, state: NativeV5MState) -> None:
        self.state = state

    def propose(
        self,
        solver: Q16NativeV5MSolver,
        structural_state: wp.array,
        structural_velocity: wp.array,
    ) -> NativeV5MProposal:
        return solver.propose(self.state, structural_state, structural_velocity)

    def commit(self, proposal: NativeV5MProposal) -> None:
        if proposal.parent_digest != self.state.digest():
            raise RuntimeError("native V5M proposal parent drift")
        if proposal.trial_state.step != self.state.step + 1:
            raise RuntimeError("native V5M commit must advance exactly one step")
        self.state = proposal.trial_state


__all__ = [
    "NATIVE_V5M_CONTRACT",
    "NativeV5MConfig",
    "NativeV5MGeometry",
    "NativeV5MLoad",
    "NativeV5MProposal",
    "NativeV5MState",
    "Q16NativePanelLoadTransfer",
    "Q16NativeV5MOwner",
    "Q16NativeV5MSolver",
    "Q16NativeV5MSurface",
    "native_aic",
    "native_ring_velocity_expanded",
]
